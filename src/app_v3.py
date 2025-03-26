import streamlit as st
import base64
import requests
import time
import json
import logger
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
from dataclasses import dataclass
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os
import pathlib

import logging
import sys

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger
    
logger = get_logger(__name__)
load_dotenv()

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL")


@dataclass
class BillTranslation:
    bill_number: str
    departure_port: str
    arrival_port: str
    shipper_name: str
    shipper_address: str
    consignee_name: str
    consignee_address: str
    notify_party_name: str
    notify_party_address: str
    quantity: str
    container_count: str
    goods_names: List[str]
    shipping_date: str
    vessel_name: str
    gross_weight: str


class BillInfo(BaseModel):
    bill_number: str = Field(description="提单号")
    departure_port: str = Field(description="发运港口")
    arrival_port: str = Field(description="收货港口")
    shipper_name: str = Field(description="发货人公司名称")
    shipper_address: str = Field(description="发货人地址")
    consignee_name: str = Field(description="收货人公司名称")
    consignee_address: str = Field(description="收货人地址")
    notify_party_name: str = Field(description="通知人公司名称")
    notify_party_address: str = Field(description="通知人地址")
    quantity: str = Field(description="货物数量")
    container_count: str = Field(description="独立箱数")
    goods_names: List[str] = Field(description="货物品名列表")
    shipping_date: str = Field(description="发运时间")
    vessel_name: str = Field(description="船名")
    gross_weight: str = Field(description="毛重")


class BillExtractor:
    def __init__(self, api_key: str, model_type: str = "openai", base_url: str = None):
        self.parser = PydanticOutputParser(pydantic_object=BillInfo)

        if model_type == "openai":
            self.llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo",
                openai_api_key=api_key,
                streaming=True,
            )
        elif model_type == "deepseek":
            self.llm = ChatOpenAI(
                temperature=0,
                model_name="deepseek-v3",
                openai_api_key=api_key,
                openai_api_base=base_url,
                streaming=True,
            )

        template = """
        对文本进行规范化处理，具体要求如下：
        - 处理文本中的空格问题，包括单词间空格、多余空格
        - 合理安排换行格式，避免单词中间换行
        - 确保英文单词的完整性，正确处理单词边界
        - 对连续大写字母的公司名称进行正确分词
        - 保持专有名词原有的大小写格式

        然后从处理后的文本中提取以下字段，不要添加文本中没有的信息：
        - 提单号
        - 发运港口
        - 收货港口
        - 发货人
        - 收货人
        - 通知人
        - 货物数量
        - 独立箱数
        - 货物品名
        - 发运时间

        文本内容：
        {text}

        {format_instructions}
        """

        self.prompt = PromptTemplate(
            template=template,
            input_variables=["text"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    def extract(self, text: str, callback=None, text_callback=None) -> BillInfo:
        if callback:
            callback("正在处理文本...")

        _input = self.prompt.format(text=text)

        if callback:
            callback("正在使用AI提取信息...")

        output = ""
        extract_info_shown = False  # 新增标志位
        for chunk in self.llm.stream(_input):
            if callback and not extract_info_shown:
                callback("正在提取信息...")
                extract_info_shown = True
            if text_callback:
                output += chunk.content
                text_callback(output)

        if callback:
            callback("正在解析提取结果...")

        bill_info = self.parser.parse(output)

        if callback:
            callback("信息提取完成！")

        return bill_info


def process_file(uploaded_file):
    try:
        logger.info(f"开始处理文件: {uploaded_file.name}")
        start_time = time.time()
        if uploaded_file.type.startswith("image"):
            image_data = base64.b64encode(uploaded_file.getvalue()).decode("ascii")
            logger.info("处理图片文件")
        else:
            pdf_data = uploaded_file.getvalue()
            image_data = base64.b64encode(pdf_data).decode("ascii")
            logger.info("处理PDF文件")

        payload = {
            "file": image_data,
            "fileType": (1 if uploaded_file.type.startswith("image") else 0),
            "device": "gpu:0",
            "use_doc_orientation_classify": False,  # 文档方向分类模块
            "use_doc_unwarping": False,  # 文档扭曲矫正模块
            "use_textline_orientation": False,  # 文本行方向分类模块
            "use_seal_recognition": False,  # 印章识别
            "use_formula_recognition": False,  # 公式识别
        }

        logger.info("调用OCR服务")
        response = requests.post(API_URL, json=payload)
        if response.status_code != 200:
            logger.error(f"OCR服务调用失败: {response.status_code}")
            raise Exception(f"OCR服务调用失败: {response.status_code}")

        result = response.json()["result"]
        markdown_texts = []

        for i, res in enumerate(result["layoutParsingResults"]):
            markdown_texts.append(res["markdown"]["text"])
            # 移除频繁的print语句
            # print(res["prunedResult"])
            md_dir = pathlib.Path(f"markdown_{i}")
            md_dir.mkdir(exist_ok=True)
            (md_dir / "doc.md").write_text(res["markdown"]["text"])
            for img_path, img in res["markdown"]["images"].items():
                img_path = md_dir / img_path
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img_path.write_bytes(base64.b64decode(img))
            logger.debug(f"Markdown document saved at {md_dir / 'doc.md'}")
            for img_name, img in res["outputImages"].items():
                img_path = f"{img_name}_{i}.jpg"
                with open(img_path, "wb") as f:
                    f.write(base64.b64decode(img))
                logger.debug(f"Output image saved at {img_path}")

        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        logger.info(f"文件处理完成，耗时: {processing_time}秒")
        return "\n".join(markdown_texts), processing_time
    except requests.exceptions.RequestException as e:
        logger.error(f"网络请求失败: {str(e)}")
        st.error(f"网络请求失败: {str(e)}")
        return None, None
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {str(e)}")
        st.error(f"JSON解析失败: {str(e)}")
        return None, None
    except Exception as e:
        logger.error(f"文件处理失败: {str(e)}")
        st.error(f"文件处理失败: {str(e)}")
        return None, None


def extract_bill_info(extractor: BillExtractor, text: str):
    logger.info("开始提取提单信息")
    start_time = time.time()

    result_container = st.container()
    status_container = st.empty()
    result_container.subheader("提单信息")

    text_area = st.empty()
    current_text = ""

    def update_status(message):
        status_container.info(message)
        logger.info(message)

    def update_text(text):
        nonlocal current_text
        if text.strip() == "```json":
            return
        elif text.strip() == "```":
            return
        if not hasattr(update_text, "_last_logged"):
            update_text._last_logged = ""
        if text != update_text._last_logged:
            # 先清除旧的内容
            text_area.empty()
            current_text = text
            # 显示更新后的完整内容
            text_area.markdown(current_text)
            # logger.info(f'更新显示文本: {text[:100]}...')
            update_text._last_logged = text

    result = extractor.extract(text, callback=update_status, text_callback=update_text)
    end_time = time.time()
    extraction_time = round(end_time - start_time, 2)

    logger.info(f"提单信息提取完成，耗时: {extraction_time}秒")
    logger.info(f"提取结果: {json.dumps(vars(result), ensure_ascii=False, indent=2)}")

    with result_container:
        col1, col2 = st.columns(2)

        with col1:
            st.text_input("提单号", result.bill_number, disabled=True)
            st.text_input("发运港口", result.departure_port, disabled=True)
            st.text_input("收货港口", result.arrival_port, disabled=True)
            st.text_input("发货人公司名称", result.shipper_name, disabled=True)
            st.text_input("发货人地址", result.shipper_address, disabled=True)
            st.text_input("收货人公司名称", result.consignee_name, disabled=True)
            st.text_input("收货人地址", result.consignee_address, disabled=True)

        with col2:
            st.text_input("通知人公司名称", result.notify_party_name, disabled=True)
            st.text_input("通知人地址", result.notify_party_address, disabled=True)
            st.text_input("货物数量", result.quantity, disabled=True)
            st.text_input("独立箱数", result.container_count, disabled=True)
            st.text_input("货物品名", ", ".join(result.goods_names), disabled=True)
            st.text_input("发运时间", result.shipping_date, disabled=True)
            st.text_input("船名", result.vessel_name, disabled=True)
            st.text_input("毛重", result.gross_weight, disabled=True)

    return result, extraction_time


def display_processing_times(ocr_time, extraction_time):
    time_container = st.empty()
    time_container.markdown("### 处理时间统计")

    with time_container.container():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("OCR处理耗时", f"{ocr_time}秒")
        with col2:
            st.metric("结构化信息提取耗时", f"{extraction_time}秒")


def init_session_state():
    if "file_processed" not in st.session_state:
        st.session_state.analysis_text = ""
        st.session_state.markdown_result = None
        st.session_state.bill_info = None
        st.session_state.ocr_time = None
        st.session_state.extraction_time = None
        st.session_state.file_processed = False


def handle_file_preview(uploaded_file):
    if uploaded_file.type.startswith("image"):
        st.image(uploaded_file, use_column_width=True)
    else:
        pdf_data = uploaded_file.getvalue()
        pdf_base64 = base64.b64encode(pdf_data).decode("utf-8")
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{pdf_base64}" width="100%" height="600px" type="application/pdf"></iframe>',
            unsafe_allow_html=True,
        )


def handle_processed_file(extractor, status_container):
    if st.session_state.bill_info:
        st.session_state.bill_info, st.session_state.extraction_time = (
            extract_bill_info(extractor, st.session_state.markdown_result)
        )
        display_processing_times(
            st.session_state.ocr_time, st.session_state.extraction_time
        )


def handle_new_file(extractor, status_container, uploaded_file):
    handle_file_preview(uploaded_file)

    with status_container:
        with st.spinner("正在处理文件..."):
            st.session_state.markdown_result, st.session_state.ocr_time = process_file(
                uploaded_file
            )
            st.success("文件处理完成！")

    if extractor:
        with status_container:
            with st.spinner("正在使用AI提取结构化信息..."):
                st.session_state.bill_info, st.session_state.extraction_time = (
                    extract_bill_info(extractor, st.session_state.markdown_result)
                )
                st.success("结构化信息提取完成！")

        display_processing_times(
            st.session_state.ocr_time, st.session_state.extraction_time
        )
        st.session_state.file_processed = True


def handle_uploaded_file(uploaded_file, extractor, status_container):
    try:
        logger.info(f"用户上传文件: {uploaded_file.name}")
        if st.session_state.file_processed:
            handle_processed_file(extractor, status_container)
        else:
            handle_new_file(extractor, status_container, uploaded_file)
    except Exception as e:
        logger.error(f"文件处理失败: {str(e)}")
        st.error(f"文件处理失败: {str(e)}")


def main():
    logger.info("启动应用")
    init_session_state()
    st.set_page_config(page_title="提单PDF信息识别与提取", layout="wide")
    st.title("提单PDF信息识别与提取")

    if not st.session_state.file_processed:
        st.session_state.analysis_text = ""

    status_container = st.container()
    extractor = BillExtractor(API_KEY, model_type="deepseek", base_url=BASE_URL)

    for key in [
        "markdown_result",
        "bill_info",
        "ocr_time",
        "extraction_time",
    ]:
        if key not in st.session_state:
            st.session_state[key] = None

    uploaded_file = st.file_uploader("上传文件", type=["pdf", "png", "jpg", "jpeg"])
    if uploaded_file is not None:
        handle_uploaded_file(uploaded_file, extractor, status_container)
    else:
        st.warning("请上传文件以开始处理")


if __name__ == "__main__":
    main()
