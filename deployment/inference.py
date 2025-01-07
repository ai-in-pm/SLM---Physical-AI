import torch
import tensorrt as trt
import numpy as np
from PIL import Image
from typing import Dict, Union, List
import pycuda.driver as cuda
import pycuda.autoinit

class InferenceEngine:
    def __init__(
        self,
        engine_path: str,
        max_batch_size: int = 1,
        device: str = 'cuda'
    ):
        """
        TensorRT inference engine for edge deployment
        Args:
            engine_path: Path to TensorRT engine file
            max_batch_size: Maximum batch size for inference
            device: Device to run inference on
        """
        self.max_batch_size = max_batch_size
        self.device = device
        
        # Load TensorRT engine
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        
        with open(engine_path, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.buffers = {}
        self.output_shapes = {}
        
        for binding in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(binding)
            binding_shape = self.engine.get_binding_shape(binding)
            binding_dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate device memory
            size = np.dtype(binding_dtype).itemsize
            for s in binding_shape:
                size *= abs(s)  # Use abs as dynamic dimensions are negative
                
            device_mem = cuda.mem_alloc(size)
            self.buffers[binding_name] = device_mem
            
            if not self.engine.binding_is_input(binding):
                self.output_shapes[binding_name] = binding_shape
    
    def _preprocess_image(
        self,
        image: Union[str, Image.Image, np.ndarray],
        target_size: tuple = (224, 224)
    ) -> np.ndarray:
        """Preprocess image for inference"""
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Resize and normalize
        image = image.resize(target_size, Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        
        # Normalize using ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Add batch dimension and transpose to NCHW
        image = np.transpose(image, (2, 0, 1))[np.newaxis, ...]
        return image
    
    def _preprocess_text(
        self,
        text: str,
        tokenizer,
        max_length: int = 128
    ) -> Dict[str, np.ndarray]:
        """Preprocess text for inference"""
        encoded = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='np'
        )
        return {
            'input_ids': encoded['input_ids'].astype(np.int32),
            'attention_mask': encoded['attention_mask'].astype(np.int32)
        }
    
    def infer(
        self,
        inputs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Run inference
        Args:
            inputs: Dictionary of input name to numpy array
        Returns:
            Dictionary of output name to numpy array
        """
        # Prepare input buffers
        for name, array in inputs.items():
            cuda.memcpy_htod(self.buffers[name], array)
        
        # Run inference
        self.context.execute_v2(list(self.buffers.values()))
        
        # Get outputs
        outputs = {}
        for name, shape in self.output_shapes.items():
            # Allocate host memory for output
            dtype = trt.nptype(self.engine.get_binding_dtype(
                self.engine.get_binding_index(name)
            ))
            output = np.empty(shape, dtype=dtype)
            
            # Copy output from device to host
            cuda.memcpy_dtoh(output, self.buffers[name])
            outputs[name] = output
            
        return outputs
    
    def process_single_input(
        self,
        image: Union[str, Image.Image, np.ndarray],
        text: str,
        tokenizer
    ) -> Dict[str, np.ndarray]:
        """
        Process a single input for inference
        Args:
            image: Input image
            text: Input text
            tokenizer: Text tokenizer
        Returns:
            Model outputs
        """
        # Preprocess inputs
        image_tensor = self._preprocess_image(image)
        text_tensors = self._preprocess_text(text, tokenizer)
        
        # Combine inputs
        inputs = {
            'image': image_tensor,
            **text_tensors
        }
        
        # Run inference
        return self.infer(inputs)
    
    def __del__(self):
        """Cleanup CUDA memory"""
        for buffer in self.buffers.values():
            buffer.free()
