from model import ROMPv1
import cv2
import numpy as np
import os
import sys
import os.path as osp
import torch
from torch import nn
import argparse

from post_parser import SMPL_parser, body_mesh_projection2image, parsing_outputs
from utils import img_preprocess, create_OneEuroFilter, euclidean_distance, check_filter_state, \
    time_cost, download_model, determine_device, ResultSaver, WebcamVideoStream, convert_cam_to_3d_trans, \
    wait_func, collect_frame_path, progress_bar, get_tracked_ids, smooth_results, convert_tensor2numpy, save_video_results
from post_parser import CenterMap
from vis_human import setup_renderer, rendering_romp_bev_results


def create_romp_settings():
    """Create ROMP settings equivalent to: python main.py -m webcam --show"""
    class Args:
        def __init__(self):
            # Equivalent to -m webcam
            self.mode = 'webcam'
            # Equivalent to --show
            self.show = True

            # Check if running in Docker container
            if os.path.exists('./romp_models'):
                # Docker container paths
                self.smpl_path = "./romp_models/SMPL_NEUTRAL.pth"
                self.model_path = "./romp_models/ROMP.pkl"
                self.model_onnx_path = "./romp_models/ROMP.onnx"
            else:
                # Local development paths
                self.smpl_path = "./romp_models/SMPL_NEUTRAL.pth"
                self.model_path = "./romp_models/ROMP.pkl"
                self.model_onnx_path = "./romp_models/ROMP.onnx"

            # Default values from main.py
            self.save_path = osp.join(osp.expanduser("~"), 'ROMP_results')
            self.GPU = 0
            self.onnx = False
            self.temporal_optimize = False
            self.center_thresh = 0.25
            self.show_largest = False
            self.smooth_coeff = 3.
            self.calc_smpl = True  # Will be set to True because show=True
            self.render_mesh = True  # Will be set to True because show=True
            self.renderer = 'pyrender'
            self.show_items = 'mesh'
            self.save_video = False
            self.frame_rate = 24
            self.root_align = False
            self.webcam_id = 0

    args = Args()

    # Apply the same logic as in romp_settings function
    if not torch.cuda.is_available():
        args.GPU = -1
        args.temporal_optimize = False
    if args.show:
        args.render_mesh = True
    if args.render_mesh or args.show_largest:
        args.calc_smpl = True

    # Check if model files exist
    if not os.path.exists(args.smpl_path):
        if os.path.exists(args.smpl_path.replace('SMPL_NEUTRAL.pth', 'smpl_packed_info.pth')):
            args.smpl_path = args.smpl_path.replace(
                'SMPL_NEUTRAL.pth', 'smpl_packed_info.pth')
        print('please prepare SMPL model files following instructions at https://github.com/Arthur151/ROMP/blob/master/simple_romp/README.md#installation')

    if not os.path.exists(args.model_path):
        romp_url = 'https://github.com/Arthur151/ROMP/releases/download/V2.0/ROMP.pkl'
        download_model(romp_url, args.model_path, 'ROMP')

    if not os.path.exists(args.model_onnx_path) and args.onnx:
        romp_onnx_url = 'https://github.com/Arthur151/ROMP/releases/download/V2.0/ROMP.onnx'
        download_model(romp_onnx_url, args.model_onnx_path, 'ROMP')

    return args


class ROMP(nn.Module):
    def __init__(self, romp_settings):
        super(ROMP, self).__init__()
        self.settings = romp_settings
        self.tdevice = determine_device(self.settings.GPU)
        self._build_model_()
        self._initilization_()

    def _build_model_(self):
        if not self.settings.onnx:
            model = ROMPv1().eval()
            model.load_state_dict(torch.load(
                self.settings.model_path, map_location=self.tdevice))
            model = model.to(self.tdevice)
            self.model = nn.DataParallel(model)
        else:
            try:
                import onnxruntime
            except ImportError:
                print(
                    'To use onnx model, we need to install the onnxruntime python package. Please install it by youself if failed!')
                if not torch.cuda.is_available():
                    os.system('pip install onnxruntime')
                else:
                    os.system('pip install onnxruntime-gpu')
                import onnxruntime
            print('creating onnx model')
            self.ort_session = onnxruntime.InferenceSession(self.settings.model_onnx_path,
                                                            providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
            print('created!')

    def _initilization_(self):
        self.centermap_parser = CenterMap(
            conf_thresh=self.settings.center_thresh)

        if self.settings.calc_smpl:
            self.smpl_parser = SMPL_parser(
                self.settings.smpl_path).to(self.tdevice)

        if self.settings.temporal_optimize:
            self._initialize_optimization_tools_()

        if self.settings.render_mesh:
            self.visualize_items = self.settings.show_items.split(',')
            self.renderer = setup_renderer(name=self.settings.renderer)

    def single_image_forward(self, image):
        input_image, image_pad_info = img_preprocess(image)
        if self.settings.onnx:
            center_maps, params_maps = self.ort_session.run(
                None, {'image': input_image.numpy().astype(np.float32)})
            center_maps, params_maps = torch.from_numpy(center_maps).to(
                self.tdevice), torch.from_numpy(params_maps).to(self.tdevice)
        else:
            center_maps, params_maps = self.model(input_image.to(self.tdevice))
        params_maps[:, 0] = torch.pow(1.1, params_maps[:, 0])
        parsed_results = parsing_outputs(
            center_maps, params_maps, self.centermap_parser)
        return parsed_results, image_pad_info

    def _initialize_optimization_tools_(self):
        self.OE_filters = {}
        if not self.settings.show_largest:
            try:
                from norfair import Tracker
            except ImportError:
                print(
                    'To perform temporal optimization, installing norfair for tracking.')
                os.system('pip install norfair')
                from norfair import Tracker
            self.tracker = Tracker(
                distance_function=euclidean_distance, distance_threshold=200)  # 120
            self.tracker_initialized = False

    def temporal_optimization(self, outputs, signal_ID):
        check_filter_state(self.OE_filters, signal_ID,
                           self.settings.show_largest, self.settings.smooth_coeff)
        if self.settings.show_largest:
            max_id = torch.argmax(outputs['cam'][:, 0])
            outputs['smpl_thetas'], outputs['smpl_betas'], outputs['cam'] = \
                smooth_results(self.OE_filters[signal_ID],
                               outputs['smpl_thetas'][max_id], outputs['smpl_betas'][max_id], outputs['cam'][max_id])
            outputs['smpl_thetas'], outputs['smpl_betas'], outputs['cam'] = outputs['smpl_thetas'].unsqueeze(
                0), outputs['smpl_betas'].unsqueeze(0), outputs['cam'].unsqueeze(0)
        else:
            pred_cams = outputs['cam']
            from norfair import Detection
            detections = [Detection(points=cam[[2, 1]]*512)
                          for cam in pred_cams.cpu().numpy()]
            if not self.tracker_initialized:
                for _ in range(8):
                    tracked_objects = self.tracker.update(
                        detections=detections)
            tracked_objects = self.tracker.update(detections=detections)
            if len(tracked_objects) == 0:
                return outputs
            tracked_ids = get_tracked_ids(detections, tracked_objects)
            for ind, tid in enumerate(tracked_ids):
                if tid not in self.OE_filters[signal_ID]:
                    self.OE_filters[signal_ID][tid] = create_OneEuroFilter(
                        self.settings.smooth_coeff)

                outputs['smpl_thetas'][ind], outputs['smpl_betas'][ind], outputs['cam'][ind] = \
                    smooth_results(self.OE_filters[signal_ID][tid],
                                   outputs['smpl_thetas'][ind], outputs['smpl_betas'][ind], outputs['cam'][ind])

            outputs['track_ids'] = np.array(tracked_ids).astype(np.int32)
        return outputs

    def forward(self, image, signal_ID=0, **kwargs):
        outputs, image_pad_info = self.single_image_forward(image)
        if outputs is None:
            return None
        if self.settings.temporal_optimize:
            outputs = self.temporal_optimization(outputs, signal_ID)
        outputs['cam_trans'] = convert_cam_to_3d_trans(outputs['cam'])
        if self.settings.calc_smpl:
            outputs = self.smpl_parser(
                outputs, root_align=self.settings.root_align)
            outputs.update(body_mesh_projection2image(
                outputs['joints'], outputs['cam'], vertices=outputs['verts'], input2org_offsets=image_pad_info))
        if self.settings.render_mesh:
            rendering_cfgs = {'mesh_color': 'identity',
                              'items': self.visualize_items, 'renderer': self.settings.renderer}
            outputs = rendering_romp_bev_results(
                self.renderer, outputs, image, rendering_cfgs)
        if self.settings.show:
            cv2.imshow('rendered', outputs['rendered_image'])
            wait_func(self.settings.mode)
        return convert_tensor2numpy(outputs)


def main():
    """Main function that replicates: python main.py -m webcam --show"""
    # Create settings equivalent to -m webcam --show
    args = create_romp_settings()

    # Initialize ROMP model
    romp = ROMP(args)
    print(f"Using device: {romp.tdevice}")

    # Webcam mode (equivalent to args.mode == 'webcam')
    cap = WebcamVideoStream(args.webcam_id)
    cap.start()
    while True:
        frame = cap.read()
        _ = romp(frame)
    cap.stop()


if __name__ == '__main__':
    main()
