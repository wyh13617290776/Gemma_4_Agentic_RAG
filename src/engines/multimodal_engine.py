import cv2
import numpy as np
import asyncio
import logging
import gc

logger = logging.getLogger("AgenticRAG")


class MultimodalEngine:
    """
    视觉多模态引擎
    采用"用后即焚"策略：仅在处理图片的这几秒钟内占用内存，结束后强制销毁。
    """
    def __init__(self):
        logger.info("多模态引擎调度器就绪..")

    async def process_files(self, uploaded_files: list) -> str:
        """
        全模态信息提取主入口：异步处理上传的文件列表，提取纯文本特征
        """
        if not uploaded_files:
            return ""
        # 将 CPU 密集型的 OCR 任务扔进后台线程池，防止死锁主事件循环
        return await asyncio.to_thread(self._sync_extract_ocr, uploaded_files)

    def _sync_extract_ocr(self, uploaded_files: list) -> str:
        media_context = ""
        ocr_instance = None

        try:
            logger.info("👁️ 正在向系统申请内存启动 RapidOCR 引擎...")
            from rapidocr_onnxruntime import RapidOCR
            ocr_instance = RapidOCR()

            for file in uploaded_files:
                file_ext = file.name.split('.')[-1].lower()

                if file_ext in ["png", "jpg", "jpeg", "bmp", "webp"]:
                    try:
                        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                        result, _ = ocr_instance(img)

                        if result:
                            texts = [item[1] for item in result]
                            extracted_text = "\n".join(texts)
                            media_context += f"【附件图像 ({file.name}) 的极速 OCR 提取内容】:\n{extracted_text}\n\n"
                        else:
                            media_context += f"【附件图像 ({file.name})】: 未提取到有效文字内容。\n\n"

                        # 重置文件指针
                        file.seek(0)
                    except Exception as e:
                        logger.error(f"RapidOCR 图像处理异常 ({file.name}): {e}")

        finally:
            # 核心防御：强制销毁实例并回收内存
            if ocr_instance is not None:
                del ocr_instance
                gc.collect()
                logger.info("♻️ RapidOCR视觉任务结束，实例已销毁，内存已完全释放。")

        return media_context.strip()
