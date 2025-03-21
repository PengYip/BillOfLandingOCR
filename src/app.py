import streamlit as st
from pathlib import Path
import base64
import cv2
import numpy as np
from paddlex import create_pipeline
from bill_extractor import BillExtractor
from image_preprocessor import ImagePreprocessor
from pdf_processor import PDFProcessor

st.set_page_config(page_title="OCR PDF处理器", layout="wide")

st.title("PDF OCR处理器")

@st.cache_resource
def get_pipeline():
    return create_pipeline(pipeline="PP-StructureV3", device="gpu")

def process_pdf(pipeline, uploaded_file, output_path, remove_watermark=False, watermark_params=None, enhance_params=None):
    # 确保temp目录在工作空间内
    temp_path = Path("temp").resolve()
    temp_path.mkdir(parents=True, exist_ok=True)
    
    # 生成安全的临时文件名
    pdf_path = temp_path / uploaded_file.name
    
    try:
        # 保存上传的文件
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            raise FileNotFoundError(f"临时文件创建失败或为空：{pdf_path}")
        
        # 如果需要去水印，先进行预处理
        if remove_watermark:
            pdf_processor = PDFProcessor()
            processed_pdf_path = temp_path / f"processed_{uploaded_file.name}"
            pdf_processor.process_pdf(
                str(pdf_path),
                str(processed_pdf_path),
                remove_watermark_params=watermark_params,
                enhance_params=enhance_params
            )
            pdf_path = processed_pdf_path
        
        # 处理PDF文件
        output = pipeline.predict(
            input=str(pdf_path),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
        
        markdown_list = []
        markdown_images = []
        
        for res in output:
            md_info = res.markdown
            markdown_list.append(md_info)
            markdown_images.append(md_info.get("markdown_images", {}))
        
        markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)
        
        # 确保输出目录存在
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存Markdown文件
        mkd_file_path = output_path / f"{pdf_path.stem}.md"
        with open(mkd_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_texts)
        
        # 保存图片文件
        for item in markdown_images:
            if item:
                for path, image in item.items():
                    file_path = output_path / path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(file_path)
        
        return markdown_texts, pdf_path
    
    except Exception as e:
        # 清理临时文件
        try:
            if pdf_path.exists():
                pdf_path.unlink()
        except Exception as cleanup_error:
            print(f"清理临时文件失败：{str(cleanup_error)}")
        
        if isinstance(e, FileNotFoundError):
            raise Exception(f"临时文件创建失败：{str(e)}")
        elif isinstance(e, PermissionError):
            raise Exception(f"无权限访问临时文件：{str(e)}")
        else:
            raise Exception(f"PDF处理失败：{str(e)}")

def extract_bill_info(extractor, markdown_result):
    bill_info = extractor.extract(markdown_result)
    
    basic_info = {
        "字段": ["提单号", "发运港口", "收货港口", "发运时间"],
        "值": [bill_info.bill_number, bill_info.departure_port, bill_info.arrival_port, bill_info.shipping_date]
    }
    st.write("基本信息：")
    st.dataframe(basic_info, hide_index=True)
    
    party_info = {
        "字段": ["发货人", "收货人", "通知人"],
        "值": [bill_info.shipper, bill_info.consignee, bill_info.notify_party]
    }
    st.write("相关方信息：")
    st.dataframe(party_info, hide_index=True)
    
    goods_info = {
        "字段": ["货物品名", "货物数量", "独立箱数"],
        "值": [bill_info.goods_name, bill_info.quantity, bill_info.container_count]
    }
    st.write("货物信息：")
    st.dataframe(goods_info, hide_index=True)
    
    csv_data = "提单号,发运港口,收货港口,发货人,收货人,通知人,货物品名,货物数量,独立箱数,发运时间\n"
    csv_data += f"{bill_info.bill_number},{bill_info.departure_port},{bill_info.arrival_port},{bill_info.shipper},{bill_info.consignee},{bill_info.notify_party},{bill_info.goods_name},{bill_info.quantity},{bill_info.container_count},{bill_info.shipping_date}"
    
    st.download_button(
        label="导出CSV文件",
        data=csv_data.encode('utf-8-sig'),
        file_name="提单信息.csv",
        mime="text/csv",
        help="点击下载提单信息的CSV文件"
    )
    
    return bill_info

def translate_bill_info(extractor, bill_info):
    translated_info = extractor.translate(bill_info)
    
    basic_info_translated = {
        "字段": ["提单号", "发运港口", "收货港口", "发运时间"],
        "值": [translated_info.bill_number, translated_info.departure_port, 
              translated_info.arrival_port, translated_info.shipping_date]
    }
    st.write("基本信息：")
    st.dataframe(basic_info_translated, hide_index=True)
    
    party_info_translated = {
        "字段": ["发货人", "收货人", "通知人"],
        "值": [translated_info.shipper, translated_info.consignee, 
              translated_info.notify_party]
    }
    st.write("相关方信息：")
    st.dataframe(party_info_translated, hide_index=True)
    
    goods_info_translated = {
        "字段": ["货物品名", "货物数量", "独立箱数"],
        "值": [translated_info.goods_name, translated_info.quantity, 
              translated_info.container_count]
    }
    st.write("货物信息：")
    st.dataframe(goods_info_translated, hide_index=True)

def display_pdf_and_markdown(pdf_path, markdown_result):
    st.subheader("PDF原文")
    with open(pdf_path, "rb") as pdf_file:
        base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)
    
    st.subheader("预览效果")
    with st.container():
        st.markdown(
            f"<div style='width: 100%; word-wrap: break-word; overflow-wrap: break-word;'>{markdown_result}</div>",
            unsafe_allow_html=True
        )
    
    with st.expander("查看Markdown源码", expanded=False):
        st.text_area("Markdown内容", markdown_result, height=400)

def preview_watermark_removal():
    st.title("水印去除预览")
    
    # 上传图片
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # 读取图片
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 添加去水印参数控制
        st.sidebar.markdown("### 去水印参数")
        brightness_threshold = st.sidebar.slider("亮度阈值", 0, 255, 200)
        saturation_threshold = st.sidebar.slider("饱和度阈值", 0, 255, 30)
        contrast_alpha = st.sidebar.slider("对比度增强系数", 0.1, 3.0, 1.3)
        contrast_beta = st.sidebar.slider("亮度调整值", -100, 100, 0)
        
        # 添加图像增强参数控制
        st.sidebar.markdown("### 图像增强参数")
        brightness = st.sidebar.slider("亮度", 0.1, 3.0, 1.0)
        contrast = st.sidebar.slider("对比度", 0.1, 3.0, 1.0)
        sharpness = st.sidebar.slider("锐化程度", 0.1, 3.0, 1.0)
        
        # 设置参数
        watermark_params = {
            'brightness_threshold': brightness_threshold,
            'saturation_threshold': saturation_threshold,
            'contrast_alpha': contrast_alpha,
            'contrast_beta': contrast_beta
        }
        
        enhance_params = {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness
        }
        
        # 处理图像
        preprocessor = ImagePreprocessor()
        processed_image = preprocessor.preprocess_image(
            image,
            remove_watermark_params=watermark_params,
            enhance_params=enhance_params
        )
        
        # 显示原图和处理后的图像
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 原图")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        with col2:
            st.markdown("### 处理后的图像")
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

def main():
    # 创建状态容器
    status_container = st.container()
    with status_container:
        pdf_status = st.empty()
        extract_status = st.empty()
        translate_status = st.empty()
    
    pipeline = get_pipeline()
    
    api_key = st.sidebar.text_input("API Key", value="sk-uHRn1euhCC8sYQBPN9hZA1AJvGFDVX6hhoqs50SLwXk8QPOv", type="password")
    base_url = st.sidebar.text_input("DeepSeek API URL", value="https://api.lkeap.cloud.tencent.com/v1", type="password")
    extractor = BillExtractor(api_key, model_type="deepseek", base_url=base_url) if api_key and base_url else None
    
    # 添加去水印选项
    st.sidebar.markdown("### 图像预处理选项")
    remove_watermark = st.sidebar.checkbox("去除水印")
    
    watermark_params = None
    enhance_params = None
    
    if remove_watermark:
        st.sidebar.markdown("#### 去水印参数")
        brightness_threshold = st.sidebar.slider("亮度阈值", 0, 255, 200)
        saturation_threshold = st.sidebar.slider("饱和度阈值", 0, 255, 30)
        contrast_alpha = st.sidebar.slider("对比度系数", 0.1, 3.0, 1.3)
        contrast_beta = st.sidebar.slider("亮度调整", -100, 100, 0)
        
        watermark_params = {
            'brightness_threshold': brightness_threshold,
            'saturation_threshold': saturation_threshold,
            'contrast_alpha': contrast_alpha,
            'contrast_beta': contrast_beta
        }
        
        st.sidebar.markdown("#### 图像增强参数")
        brightness = st.sidebar.slider("亮度", 0.1, 3.0, 1.0)
        contrast = st.sidebar.slider("对比度", 0.1, 3.0, 1.0)
        sharpness = st.sidebar.slider("锐化", 0.1, 3.0, 1.0)
        
        enhance_params = {
            'brightness': brightness,
            'contrast': contrast,
            'sharpness': sharpness
        }
    
    uploaded_file = st.file_uploader("上传PDF文件", type=["pdf"])
    
    if uploaded_file is not None:
        try:
            # 处理PDF文件
            with status_container:
                with st.spinner("正在处理PDF文件..."):
                    pdf_status.info("正在解析PDF文件，请稍候...")
                    markdown_result, pdf_path = process_pdf(
                        pipeline,
                        uploaded_file,
                        "output",
                        remove_watermark=remove_watermark,
                        watermark_params=watermark_params,
                        enhance_params=enhance_params
                    )
                    pdf_status.success("PDF处理完成！")
            
            # 提取结构化信息
            if extractor:
                with status_container:
                    with st.spinner("正在使用AI提取结构化信息..."):
                        extract_status.info("正在使用AI提取结构化信息，请稍候...")
                        st.subheader("提单信息")
                        bill_info = extract_bill_info(extractor, markdown_result)
                        extract_status.success("结构化信息提取完成！")
                
                # 翻译功能
                if st.button("一键翻译"):
                    with status_container:
                        with st.spinner("正在翻译..."):
                            translate_status.info("正在使用AI翻译，请稍候...")
                            st.subheader("翻译结果")
                            translate_bill_info(extractor, bill_info)
                            translate_status.success("翻译完成！")
            else:
                st.info("请输入OpenAI API Key以启用提单信息提取功能")
            
            # 显示PDF和Markdown
            display_pdf_and_markdown(pdf_path, markdown_result)
            
            # 在显示完成后清理临时文件
            try:
                if pdf_path.exists():
                    pdf_path.unlink()
            except Exception as cleanup_error:
                print(f"清理临时文件失败：{str(cleanup_error)}")
            
        except Exception as e:
            st.error(f"处理过程中发生错误：{str(e)}")

if __name__ == "__main__":
    main()