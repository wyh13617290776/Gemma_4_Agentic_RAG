import cv2
import numpy as np
import asyncio
from rapidocr_onnxruntime import RapidOCR # 或者根据你安装的版本导入

class MultimodalEngine:
    def __init__(self):
        print("🚀 初始化 RapidOCR 视觉引擎...")
        # 初始化 RapidOCR，可根据你的机器配置调整参数
        self.ocr = RapidOCR()

    async def process_files(self, uploaded_files: list) -> str:
        """
        全模态脱水主入口：异步处理上传的文件列表，提取纯文本特征
        """
        if not uploaded_files:
            return ""

        # 将 CPU 密集型的 OCR 任务扔进后台线程池，防止死锁 Streamlit 主事件循环
        return await asyncio.to_thread(self._sync_extract_ocr, uploaded_files)

    def _sync_extract_ocr(self, uploaded_files: list) -> str:
        media_context = ""
        for file in uploaded_files:
            file_ext = file.name.split('.')[-1].lower()

            # 目前仅拦截图像进行 OCR 脱水
            if file_ext in ["png", "jpg", "jpeg", "bmp", "webp"]:
                try:
                    # 1. 转换 Streamlit UploadedFile 为 OpenCV 矩阵格式
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                    # 2. 启动装甲识别
                    result, _ = self.ocr(img)

                    # 3. 组装结果
                    if result:
                        # result 结构为: [([[pt1, pt2, pt3, pt4]], '文本', 置信度), ...]
                        texts = [item[1] for item in result]
                        extracted_text = "\n".join(texts)
                        media_context += f"【附件图像 ({file.name}) 的极速 OCR 提取内容】:\n{extracted_text}\n\n"
                    else:
                        media_context += f"【附件图像 ({file.name})】: 未提取到有效文字内容。\n\n"

                    # 架构师防呆补丁：重置文件指针！
                    # 因为 read() 已经把指针读到了末尾，如果后续你还要把图片转 Base64 喂给大模型，必须 seek(0)！
                    file.seek(0)
                    
                except Exception as e:
                    print(f"⚠️ RapidOCR 图像物理脱水失败 ({file.name}): {e}")

        return media_context.strip()