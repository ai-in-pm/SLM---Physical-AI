import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
import psutil
import GPUtil
from captum.attr import IntegratedGradients, LayerGradCam
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

class XAIMonitor:
    """
    Provides explainability features and resource monitoring for the Physical AI system
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize explainability tools
        self.integrated_gradients = IntegratedGradients(self.model)
        self.grad_cam = LayerGradCam(self.model, self.model.fusion)
        
        # Resource monitoring
        self.resource_history = {
            "cpu": [],
            "memory": [],
            "gpu": [],
            "temperature": []
        }
        
        # Performance metrics
        self.metrics = {
            "latency": [],
            "accuracy": [],
            "energy": []
        }
    
    def explain_decision(
        self,
        input_data: Dict[str, torch.Tensor],
        target_class: Optional[int] = None
    ) -> Dict[str, Union[np.ndarray, str]]:
        """
        Generate explanations for model decisions
        Args:
            input_data: Dictionary of input tensors
            target_class: Optional target class for attribution
        Returns:
            Dictionary containing various explanations
        """
        try:
            explanations = {}
            
            # Integrated Gradients attribution
            attributions = self.integrated_gradients.attribute(
                input_data["image"],
                target=target_class,
                n_steps=50
            )
            explanations["feature_importance"] = attributions.cpu().numpy()
            
            # GradCAM visualization
            cam = self.grad_cam.attribute(
                input_data["image"],
                target=target_class
            )
            explanations["attention_map"] = cam.cpu().numpy()
            
            # Generate text explanation
            explanations["text_explanation"] = self._generate_text_explanation(
                attributions,
                cam,
                target_class
            )
            
            return explanations
            
        except Exception as e:
            self.logger.error(f"Error generating explanations: {str(e)}")
            return {"error": str(e)}
    
    def _generate_text_explanation(
        self,
        attributions: torch.Tensor,
        cam: torch.Tensor,
        target_class: Optional[int]
    ) -> str:
        """
        Generate natural language explanation for model decision
        """
        try:
            # Analyze feature importance
            important_features = torch.topk(
                attributions.abs().mean(dim=(2, 3)),
                k=5
            )
            
            # Analyze attention regions
            attention_regions = torch.topk(
                cam.view(cam.size(0), -1),
                k=5
            )
            
            # Generate explanation
            explanation = (
                f"The model's decision was primarily based on:\n"
                f"1. Strong activation in regions: {attention_regions.indices.tolist()}\n"
                f"2. Important features: {important_features.indices.tolist()}\n"
            )
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating text explanation: {str(e)}")
            return f"Error: {str(e)}"
    
    def monitor_resources(self) -> Dict[str, float]:
        """
        Monitor system resource usage
        Returns:
            Dictionary of current resource usage
        """
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent()
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # GPU usage
            gpu_stats = []
            if torch.cuda.is_available():
                for gpu in GPUtil.getGPUs():
                    gpu_stats.append({
                        "id": gpu.id,
                        "load": gpu.load * 100,
                        "memory_used": gpu.memoryUsed,
                        "temperature": gpu.temperature
                    })
            
            # Update history
            self.resource_history["cpu"].append(cpu_percent)
            self.resource_history["memory"].append(memory_percent)
            if gpu_stats:
                self.resource_history["gpu"].append(gpu_stats[0]["load"])
                self.resource_history["temperature"].append(gpu_stats[0]["temperature"])
            
            return {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "gpu_stats": gpu_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error monitoring resources: {str(e)}")
            return {"error": str(e)}
    
    def track_performance(
        self,
        latency: float,
        accuracy: float,
        energy: Optional[float] = None
    ):
        """
        Track system performance metrics
        """
        try:
            self.metrics["latency"].append(latency)
            self.metrics["accuracy"].append(accuracy)
            if energy is not None:
                self.metrics["energy"].append(energy)
            
        except Exception as e:
            self.logger.error(f"Error tracking performance: {str(e)}")
    
    def visualize_attention(
        self,
        attention_weights: torch.Tensor,
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Visualize attention weights
        Args:
            attention_weights: Attention weight tensor
            save_path: Optional path to save visualization
        Returns:
            Matplotlib figure if save_path is None
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(
                attention_weights.cpu().numpy(),
                ax=ax,
                cmap="viridis"
            )
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
                return None
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error visualizing attention: {str(e)}")
            return None
    
    def visualize_feature_space(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Visualize feature space using t-SNE
        Args:
            features: Feature tensor
            labels: Optional label tensor
            save_path: Optional path to save visualization
        Returns:
            Matplotlib figure if save_path is None
        """
        try:
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            features_2d = tsne.fit_transform(features.cpu().numpy())
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 10))
            if labels is not None:
                scatter = ax.scatter(
                    features_2d[:, 0],
                    features_2d[:, 1],
                    c=labels.cpu().numpy(),
                    cmap="tab10"
                )
                plt.colorbar(scatter)
            else:
                ax.scatter(features_2d[:, 0], features_2d[:, 1])
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
                return None
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error visualizing feature space: {str(e)}")
            return None
    
    def generate_report(
        self,
        save_path: Optional[str] = None
    ) -> Dict[str, Union[Dict, str]]:
        """
        Generate comprehensive system report
        Args:
            save_path: Optional path to save report
        Returns:
            Dictionary containing report data
        """
        try:
            report = {
                "resource_usage": {
                    "cpu_average": np.mean(self.resource_history["cpu"]),
                    "memory_average": np.mean(self.resource_history["memory"]),
                    "gpu_average": np.mean(self.resource_history["gpu"]) if self.resource_history["gpu"] else None
                },
                "performance_metrics": {
                    "average_latency": np.mean(self.metrics["latency"]),
                    "average_accuracy": np.mean(self.metrics["accuracy"]),
                    "average_energy": np.mean(self.metrics["energy"]) if self.metrics["energy"] else None
                },
                "system_health": self._assess_system_health()
            }
            
            if save_path:
                import json
                with open(save_path, "w") as f:
                    json.dump(report, f, indent=4)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            return {"error": str(e)}
    
    def _assess_system_health(self) -> Dict[str, str]:
        """
        Assess overall system health based on metrics
        Returns:
            Dictionary containing health status for different components
        """
        try:
            health = {}
            
            # CPU health
            cpu_avg = np.mean(self.resource_history["cpu"])
            health["cpu"] = (
                "critical" if cpu_avg > 90 else
                "warning" if cpu_avg > 70 else
                "healthy"
            )
            
            # Memory health
            mem_avg = np.mean(self.resource_history["memory"])
            health["memory"] = (
                "critical" if mem_avg > 90 else
                "warning" if mem_avg > 70 else
                "healthy"
            )
            
            # GPU health
            if self.resource_history["gpu"]:
                gpu_avg = np.mean(self.resource_history["gpu"])
                health["gpu"] = (
                    "critical" if gpu_avg > 90 else
                    "warning" if gpu_avg > 70 else
                    "healthy"
                )
            
            return health
            
        except Exception as e:
            self.logger.error(f"Error assessing system health: {str(e)}")
            return {"error": str(e)}
