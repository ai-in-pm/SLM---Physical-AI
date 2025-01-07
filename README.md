# SLM Physical AI

A lightweight multimodal AI system inspired by NVIDIA's Physical AI Model, optimized for small language models (SLMs). This system enables efficient processing of visual and textual inputs to generate robotic actions, designed specifically for edge deployment.

## Architecture

The system follows a token-based processing pipeline with the following components:

### 1. Input Processing
- **Vision Processing**: 
  - EfficientNet-B0 and YOLO for object detection
  - Real-time video processing and object tracking
  - LiDAR and depth camera integration
- **Speech Processing**:
  - Speech-to-Text using Whisper
  - Text-to-Speech synthesis
  - Multilingual support
- **Text Processing**: 
  - Multiple language model support (local and API-based)
  - Cross-language translation
- **Sensor Integration**:
  - LiDAR point cloud processing
  - Depth camera integration
  - IMU and additional sensor support
  - Sensor fusion and calibration

### 2. World Foundation Model
- Processes combined image and text tokens
- Maintains contextual understanding of the environment
- Uses transformer-based architecture for efficient processing
- Optimized for edge deployment with mixed-precision support

### 3. Action Generation
- Generates discrete action tokens for robotic control
- Cross-attention mechanism for action selection
- Supports various robotic tasks through flexible action space
- ROS integration for robot control

### 4. Collaborative Agent Network
- **Planner Agent**: Task planning and resource allocation
- **Executor Agent**: Physical action execution
- **Monitor Agent**: System state and performance monitoring
- **Analyzer Agent**: Data analysis and optimization
- Asynchronous message passing between agents

### 5. XAI and Monitoring
- Model decision explanations
- Attention visualization
- Resource monitoring (CPU, GPU, Memory)
- Performance tracking
- System health assessment

### 6. Safety and Error Handling
- Robust error detection and recovery
- Safety constraints for physical actions
- Continuous system health monitoring
- Graceful degradation under resource constraints

## Framework Integration

### Speech Processing
```python
from models.speech_processor import SpeechProcessor

# Initialize speech processor
processor = SpeechProcessor(
    whisper_model="openai/whisper-small",
    tts_model="tts_models/en/ljspeech/tacotron2-DDC"
)

# Speech to text
transcription = processor.transcribe_audio("audio.wav")

# Text to speech
audio = processor.synthesize_speech(
    "Hello, I am a robot",
    output_path="output.wav"
)

# Multilingual support
translated = processor.translate_speech(
    "audio.wav",
    source_lang="en",
    target_lang="es"
)
```

### Sensor Integration
```python
from models.sensor_processor import SensorProcessor

# Initialize sensor processor
sensor = SensorProcessor(
    use_lidar=True,
    use_depth=True,
    use_imu=True
)

# Process LiDAR data
lidar_results = sensor.process_lidar(point_cloud)

# Process depth camera
depth_results = sensor.process_depth(depth_image)

# Track objects
tracking = sensor.track_objects(frame)

# Generate spatial map
spatial_map = sensor.get_spatial_map(sensor_data)
```

### XAI and Monitoring
```python
from models.xai_monitor import XAIMonitor

# Initialize monitor
monitor = XAIMonitor(model)

# Get decision explanation
explanation = monitor.explain_decision(input_data)

# Monitor resources
resources = monitor.monitor_resources()

# Visualize attention
monitor.visualize_attention(
    attention_weights,
    save_path="attention.png"
)

# Generate report
report = monitor.generate_report()
```

### Collaborative Agents
```python
from models.agent_network import CollaborativeSystem

# Initialize system
system = CollaborativeSystem()

# Execute task
await system.execute_task({
    "type": "pick_and_place",
    "object": "cube",
    "target": "table"
})

# Monitor execution
status = await system.network.route_message(
    AgentMessage(
        sender="user",
        receiver="system_monitor",
        content={"type": "status_request"}
    )
)
```

## Project Structure
```
SLM-Physical-AI/
├── models/
│   ├── vision_processor.py     # Vision and object detection
│   ├── speech_processor.py     # Speech processing (STT/TTS)
│   ├── sensor_processor.py     # Sensor integration
│   ├── text_processor.py       # Text processing with multiple LLMs
│   ├── language_model_manager.py  # Multi-model management
│   ├── agent_network.py        # Collaborative agent system
│   ├── xai_monitor.py         # Explainability and monitoring
│   ├── token_processor.py     # Token generation and processing
│   ├── fusion.py             # World Foundation Model
│   └── physical_ai.py        # Main model architecture
├── training/
│   ├── data_loader.py        # Multimodal data handling
│   └── trainer.py            # Training pipeline
├── deployment/
│   ├── optimizer.py          # Model optimization for edge
│   └── inference.py          # Inference engine
├── safety/
│   ├── constraints.py        # Safety constraints
│   └── error_handler.py      # Error handling
├── ros/
│   ├── scripts/             # ROS node implementations
│   ├── launch/              # ROS launch files
│   └── config/              # ROS configuration files
├── ar/
│   ├── interface.py         # AR interface
│   └── visualization.py     # AR visualization
└── simulation/
    └── environment.py        # PyBullet simulation with ROS bridge
```

## Features

- **Multimodal Processing**:
  - Vision (Image, Video, LiDAR)
  - Speech (STT/TTS)
  - Text (Multiple LLMs)
  - Sensor data

- **Collaborative Intelligence**:
  - Multi-agent system
  - Task planning and execution
  - Resource monitoring
  - Data analysis

- **Explainable AI**:
  - Decision explanations
  - Attention visualization
  - Performance tracking
  - System health monitoring

- **Safety and Reliability**:
  - Error handling
  - Safety constraints
  - Resource optimization
  - Graceful degradation

- **Edge Optimization**:
  - Mixed-precision training
  - Model quantization
  - TensorRT integration
  - Resource monitoring

- **Extended Interfaces**:
  - AR visualization
  - Natural language interface
  - Cross-language support
  - IoT connectivity

## Performance Optimization

The system includes several optimizations:
- Mixed-precision training (FP16)
- Model quantization (INT8)
- TensorRT integration
- Efficient token-based processing
- Lightweight vision backbone
- Resource-aware scheduling
- Energy consumption monitoring
- Adaptive compute allocation

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
