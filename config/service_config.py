import os
from dataclasses import dataclass
from typing import Optional, Union, List


@dataclass
class ServiceConfig:
    """服务配置"""
    # 服务设置
    host: str = "0.0.0.0"
    port: int = 8010
    workers: int = 1

    # 模型设置
    model_path: str = "/home/zhangjiang/bge-m3-model"
    devices: Optional[Union[str, List[str]]] = "cuda:0"

    # 日志配置
    log_level: str = "INFO"
    log_file: str = "/home/zhangjiang/logs/embeddings/embeddings_service.log"

    def __post_init__(self):
        """初始化后处理"""
        # 自动检测设备
        try:
            import torch
            if not torch.cuda.is_available():
                self.device = "cpu"
                self.use_fp16 = False
        except ImportError:
            self.device = "cpu"
            self.use_fp16 = False

        # 从环境变量加载配置
        self._load_from_env()

    def _load_from_env(self):
        """从环境变量加载配置"""
        if os.getenv("EMBEDDINGS_HOST"):
            self.host = os.getenv("EMBEDDINGS_HOST")
        if os.getenv("EMBEDDINGS_PORT"):
            self.port = int(os.getenv("EMBEDDINGS_PORT"))
        if os.getenv("EMBEDDINGS_MODEL_PATH"):
            self.model_path = os.getenv("EMBEDDINGS_MODEL_PATH")
        if os.getenv("EMBEDDINGS_DEVICES"):
            self.devices = os.getenv("EMBEDDINGS_DEVICES")

# 全局配置实例
config = ServiceConfig()