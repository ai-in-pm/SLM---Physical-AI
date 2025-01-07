import torch
import torch.nn as nn
import onnx
import tensorrt as trt
import numpy as np
from typing import Dict, List, Optional

class ModelOptimizer:
    def __init__(
        self,
        model: nn.Module,
        input_shapes: Dict[str, List[int]],
        precision: str = 'fp16',
        workspace_size: int = 1 << 30  # 1GB
    ):
        """
        Optimize model for edge deployment
        Args:
            model: PyTorch model to optimize
            input_shapes: Dictionary of input names to shapes
            precision: 'fp32', 'fp16', or 'int8'
            workspace_size: TensorRT workspace size in bytes
        """
        self.model = model
        self.input_shapes = input_shapes
        self.precision = precision
        self.workspace_size = workspace_size
        
    def quantize_weights(self, model: nn.Module) -> nn.Module:
        """Quantize model weights to int8"""
        def _quantize_layer(layer):
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                # Quantize weights
                scale = layer.weight.abs().max() / 127.
                layer.weight.data = torch.round(layer.weight.data / scale) * scale
                
                # Quantize bias if present
                if layer.bias is not None:
                    scale = layer.bias.abs().max() / 127.
                    layer.bias.data = torch.round(layer.bias.data / scale) * scale
            
        model.apply(_quantize_layer)
        return model
    
    def optimize_graph(self, model: nn.Module) -> nn.Module:
        """Optimize model graph structure"""
        # Fuse batch norm layers
        torch.quantization.fuse_modules(
            model,
            [['conv', 'bn', 'relu'], ['linear', 'relu']],
            inplace=True
        )
        return model
    
    def export_onnx(
        self,
        model: nn.Module,
        onnx_path: str,
        input_names: List[str],
        output_names: List[str],
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None
    ):
        """Export model to ONNX format"""
        # Create dummy inputs
        dummy_inputs = {}
        for name, shape in self.input_shapes.items():
            dummy_inputs[name] = torch.randn(*shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            tuple(dummy_inputs.values()),
            onnx_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=12,
            do_constant_folding=True
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
    def convert_to_tensorrt(
        self,
        onnx_path: str,
        engine_path: str
    ):
        """Convert ONNX model to TensorRT engine"""
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        
        # Parse ONNX
        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise ValueError('Failed to parse ONNX file')
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = self.workspace_size
        
        # Set precision
        if self.precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
            
        # Build engine
        engine = builder.build_engine(network, config)
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
            
    def optimize(
        self,
        onnx_path: str,
        engine_path: str,
        input_names: List[str],
        output_names: List[str]
    ):
        """Full optimization pipeline"""
        # Step 1: Quantize weights if using INT8
        if self.precision == 'int8':
            self.model = self.quantize_weights(self.model)
        
        # Step 2: Optimize graph
        self.model = self.optimize_graph(self.model)
        
        # Step 3: Export to ONNX
        self.export_onnx(
            self.model,
            onnx_path,
            input_names,
            output_names
        )
        
        # Step 4: Convert to TensorRT
        self.convert_to_tensorrt(onnx_path, engine_path)
        
        return engine_path
