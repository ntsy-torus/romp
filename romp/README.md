# ROMP Docker API

This Docker setup provides a REST API for the ROMP (Regression of Multiple 3D People) model, enabling easy integration with UX/UI applications.

## Features

• **3D Human Pose Estimation** - Detect and estimate 3D poses from images/videos
• **REST API** - Easy integration with any frontend framework
• **GPU Support** - Optimized for NVIDIA GPUs
• **Real-time Processing** - Fast inference for webcam and video streams
• **Multiple Output Formats** - JSON data, rendered images, 3D meshes
• **Web Interface** - Optional web UI for testing (port 8080)

## Quick Start

### Prerequisites

• Docker and Docker Compose installed
• NVIDIA Docker runtime (for GPU support)
• At least 4GB GPU memory recommended

### 1. Setup Model Files

Place your ROMP model files in the `romp_models/` directory:

```
romp_models/
├── ROMP.pkl
├── ROMP.onnx
├── SMPL_NEUTRAL.pth
└── SMPLA_NEUTRAL.pth
```

### 2. Build and Run

```bash
# Build the Docker image
docker-compose build

# Start the services
docker-compose up -d

# Check if the API is running
curl http://localhost:5000/health
```

### 3. Test the API

```bash
# Test with a sample image
curl -X POST http://localhost:5000/process_image \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."}'
```

## API Endpoints

### Health Check
- **GET** `/health` - Check if the service is running and model status

### Image Processing
- **POST** `/process_image` - Process a single image
- **POST** `/get_rendered_image` - Get image with 3D mesh overlay

### Video Processing
- **POST** `/process_video` - Process multiple video frames

### Model Information
- **GET** `/model_info` - Get model configuration details

## API Usage Examples

### Process Single Image

```javascript
const processImage = async (imageFile) => {
  const base64 = await fileToBase64(imageFile);
  
  const response = await fetch('http://localhost:5000/process_image', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: base64
    })
  });
  
  const result = await response.json();
  return result;
};
```

### Process Video Frames

```javascript
const processVideo = async (videoFrames) => {
  const frames = await Promise.all(
    videoFrames.map(frame => fileToBase64(frame))
  );
  
  const response = await fetch('http://localhost:5000/process_video', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      frames: frames
    })
  });
  
  const result = await response.json();
  return result;
};
```

### Get Rendered Image

```javascript
const getRenderedImage = async (imageFile) => {
  const base64 = await fileToBase64(imageFile);
  
  const response = await fetch('http://localhost:5000/get_rendered_image', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image: base64
    })
  });
  
  const result = await response.json();
  return result.rendered_image; // Base64 encoded image
};
```

## Response Format

### Successful Image Processing

```json
{
  "success": true,
  "num_people": 2,
  "poses": [...],           // SMPL pose parameters
  "cameras": [...],         // Camera parameters
  "joints": [...],          // 3D joint positions
  "vertices": [...],        // 3D mesh vertices
  "projections_2d": [...]   // 2D joint projections
}
```

### Health Check Response

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Error Response

```json
{
  "error": "Error message describing what went wrong"
}
```

## Configuration

### Environment Variables

- `NVIDIA_VISIBLE_DEVICES` - GPU device selection
- `CUDA_VISIBLE_DEVICES` - CUDA device selection

### Model Settings

The model can be configured by modifying the `create_romp_settings()` function in `app.py`:

- `center_thresh` - Detection confidence threshold (default: 0.25)
- `temporal_optimize` - Enable temporal smoothing (default: False)
- `smooth_coeff` - Smoothing coefficient (default: 3.0)
- `show_largest` - Show only the largest person (default: False)
- `calc_smpl` - Calculate SMPL parameters (default: True)
- `render_mesh` - Enable mesh rendering (default: True)

## Services

The Docker Compose setup includes:

- **romp-api** (port 5000) - Main API server
- **romp-web** (port 8080) - Optional web interface for testing

## Troubleshooting

### Common Issues

• **GPU not detected** - Ensure NVIDIA Docker runtime is installed
• **Model files missing** - Check that all model files are in `romp_models/`
• **Out of memory** - Reduce batch size or use CPU mode
• **Slow performance** - Enable ONNX mode for faster inference

### Logs

```bash
# View container logs
docker-compose logs -f romp-api

# View specific service logs
docker logs romp-api
```

### Performance Optimization

• Use ONNX model for faster inference
• Enable temporal optimization for video streams
• Adjust confidence threshold based on use case
• Use GPU with sufficient memory (4GB+ recommended)

## Integration Examples

### React.js Integration

```jsx
import React, { useState } from 'react';

const PoseDetector = () => {
  const [result, setResult] = useState(null);
  
  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    const base64 = await fileToBase64(file);
    
    try {
      const response = await fetch('http://localhost:5000/process_image', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64 })
      });
      
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    }
  };
  
  return (
    <div>
      <input type="file" onChange={handleImageUpload} />
      {result && <div>Found {result.num_people} people</div>}
    </div>
  );
};
```

### Python Client

```python
import requests
import base64
from PIL import Image
import io

def process_image(image_path):
    # Load and encode image
    with open(image_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # Send request
    response = requests.post(
        'http://localhost:5000/process_image',
        json={'image': f'data:image/jpeg;base64,{image_data}'}
    )
    
    return response.json()

# Usage
result = process_image('path/to/image.jpg')
print(f"Found {result['num_people']} people")
```

## License

This project uses the ROMP model. Please refer to the original ROMP repository for licensing terms.

## Support

For issues related to:
- ROMP model: Check the original ROMP repository
- Docker setup: Check Docker logs and configuration
- API integration: Review the API documentation and examples
