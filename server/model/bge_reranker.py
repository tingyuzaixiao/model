import time
import numpy as np
from typing import List, Tuple, Optional, Dict, Any

import torch
from sentence_transformers import CrossEncoder

from config.service_config import config
from config.logging_config import logger


class BGEReranker:
    """BGE重排序模型封装"""

    def __init__(self):
        self.logger = logger.getChild('model')
        self.model = None
        self.model_name = config.model_name
        self.device = config.device
        self.is_loaded = False
        self.load_time = 0
        self.total_queries = 0
        self.total_processing_time = 0

    def load_model(self) -> bool:
        """加载模型"""
        try:
            self.logger.info(f"开始加载模型: {self.model_name}")
            start_time = time.time()

            # 加载模型
            self.model = CrossEncoder(
                self.model_name,
                max_length=config.max_length,
                device=self.device
            )

            # FP16优化
            if config.use_fp16:
                if self.device == "cuda" and torch.cuda.is_available():
                    self.model.model.half()
                    self.logger.info("已启用FP16精度优化")
                elif self.device == "cpu":
                    self.logger.warning("FP16不支持CPU，已禁用")
                    config.use_fp16 = False  # 禁用后续 FP16

            self.load_time = time.time() - start_time
            self.is_loaded = True

            self.logger.info(f"模型加载成功: {self.model_name}")
            self.logger.info(f"加载耗时: {self.load_time:.2f}秒")
            self.logger.info(f"模型设备: {self.device}, 最大长度: {config.max_length}")

            # 预热模型
            self._warmup()

            return True

        except Exception as e:
            self.logger.error(f"模型加载失败: {str(e)}", exc_info=True)
            self.is_loaded = False
            return False

    def _warmup(self):
        """预热模型"""
        self.logger.info("开始模型预热...")
        warmup_queries = [
            ("军队体检视力标准", "视力检查要求双眼裸眼视力不低于4.8"),
            ("体检肝功能指标", "肝功能检查包括ALT、AST等指标")
        ]

        try:
            for i in range(5):  # 预热3轮
                self.model.predict(warmup_queries, batch_size=2, show_progress_bar=False)
            self.logger.info("模型预热完成")
        except Exception as e:
            self.logger.warning(f"模型预热失败: {str(e)}")

    # def _warmup_2(self):
    #     self.logger.info("开始模型预热 (使用真实业务样本)...")
    #     # 从配置加载真实样本（或使用历史数据）
    #     warmup_queries = config.warmup_queries  # 从配置加载 10-20 个真实样本
    #
    #     # 批量预热（用实际 batch_size）
    #     batch_size = min(len(warmup_queries), 8)  # 避免过大 batch
    #     try:
    #         for _ in range(5):  # 增加预热轮次
    #             self.model.predict(warmup_queries, batch_size=batch_size, show_progress_bar=False)
    #         self.logger.info("模型预热完成 (共 %d 轮)", 5)
    #     except Exception as e:
    #         self.logger.warning(f"模型预热失败: {str(e)}")

    def rerank(self,
               query: str,
               documents: List[str],
               top_k: Optional[int] = None,
               batch_size: Optional[int] = None) -> Tuple[List[str], List[float], Dict[str, Any]]:
        """执行重排序"""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("模型未加载")

        if not documents:
            return [], [], {"processing_time": 0, "documents_processed": 0}

        if top_k is None:
            top_k = len(documents)
        if batch_size is None:
            batch_size = config.batch_size

        self.logger.debug(f"重排序请求: 查询长度={len(query)}, 文档数量={len(documents)}, top_k={top_k}")

        start_time = time.time()
        self.total_queries += 1

        try:
            # 因为显存够，这里没有分批处理（避免显存溢出）
            # 每对(query, doc)的总长度不能超过max_length
            pairs = [(query, doc) for doc in documents]
            # 执行推理
            scores = self.model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            # 转换为numpy数组（如果返回的是tensor）
            # if hasattr(scores, 'cpu'):
            #     scores = scores.cpu().numpy()
            # scores = np.array(scores)

            # 排序和选择top_k
            scored_docs = list(zip(documents, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)

            ranked_documents = [doc for doc, score in scored_docs[:top_k]]
            ranked_scores = [float(score) for doc, score in scored_docs[:top_k]]

            processing_time = time.time() - start_time
            self.total_processing_time += processing_time

            # 记录性能指标
            metrics = {
                "processing_time": processing_time,
                "documents_processed": len(documents),
                "batch_size": batch_size,
                "average_score": float(np.mean(scores)) if len(scores) > 0 else 0.0,
                "max_score": float(np.max(scores)) if len(scores) > 0 else 0.0,
                "min_score": float(np.min(scores)) if len(scores) > 0 else 0.0
            }

            self.logger.info(
                f"重排序完成: {len(documents)}文档 -> {top_k}结果, "
                f"耗时{processing_time:.3f}秒, 平均分{metrics['average_score']:.4f}"
            )

            return ranked_documents, ranked_scores, metrics

        except Exception as e:
            self.logger.error(f"重排序处理失败: {str(e)}", exc_info=True)
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        avg_time = 0
        if self.total_queries > 0:
            avg_time = self.total_processing_time / self.total_queries

        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "device": self.device,
            "load_time": self.load_time,
            "total_queries": self.total_queries,
            "average_processing_time": avg_time,
            "max_length": config.max_length
        }


# 全局模型实例
reranker_model = BGEReranker()