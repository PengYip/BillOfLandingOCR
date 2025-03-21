import streamlit as st
from pathlib import Path
import base64
from paddlex import create_pipeline
from bill_extractor import BillExtractor

st.set_page_config(page_title="OCR PDF处理器", layout="wide")

st.title("PDF OCR处理器")

@st.cache_resource
def get_pipeline():
    return create_pipeline(pipeline="PP-StructureV3", device="gpu")

def process_pdf(pipeline, uploaded_file, output_path):
    # 保存上传的文件
    temp_path = Path("temp")
    temp_path.mkdir(exist_ok=True)
    pdf_path = temp_path / uploaded_file.name
    
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    try:
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
        
        # 保存结果
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        mkd_file_path = output_path / f"{pdf_path.stem}.md"
        with open(mkd_file_path, "w", encoding="utf-8") as f:
            f.write(markdown_texts)
        
        for item in markdown_images:
            if item:
                for path, image in item.items():
                    file_path = output_path / path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(file_path)
        
        return markdown_texts, pdf_path
    except Exception as e:
        # 发生错误时清理临时文件
        if pdf_path.exists():
            pdf_path.unlink()
        raise e

def main():
    # 初始化pipeline
    pipeline = get_pipeline()
    
    # 初始化提单信息提取器
    api_key = st.sidebar.text_input("API Key", value="sk-uHRn1euhCC8sYQBPN9hZA1AJvGFDVX6hhoqs50SLwXk8QPOv", type="password")
    base_url = st.sidebar.text_input("DeepSeek API URL", value="https://api.lkeap.cloud.tencent.com/v1", type="password")
    extractor = BillExtractor(api_key, model_type="deepseek", base_url=base_url) if api_key and base_url else None
    
    # 文件上传
    uploaded_file = st.file_uploader("上传PDF文件", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner("正在处理PDF文件..."):
            try:
                markdown_result, pdf_path = process_pdf(pipeline, uploaded_file, "output")
                
                # 显示处理结果
                st.success("PDF处理完成！")
                
                # 提取结构化信息
                if extractor:
                    try:
                        bill_info = extractor.extract(markdown_result)
                        st.subheader("提单信息")
                        
                        # 基本信息表格
                        basic_info = {
                            "字段": ["提单号", "发运港口", "收货港口", "发运时间"],
                            "值": [bill_info.bill_number, bill_info.departure_port, bill_info.arrival_port, bill_info.shipping_date]
                        }
                        st.write("基本信息：")
                        st.dataframe(basic_info, hide_index=True)
                        
                        # 相关方信息表格
                        party_info = {
                            "字段": ["发货人", "收货人", "通知人"],
                            "值": [bill_info.shipper, bill_info.consignee, bill_info.notify_party]
                        }
                        st.write("相关方信息：")
                        st.dataframe(party_info, hide_index=True)
                        
                        # 货物信息表格
                        goods_info = {
                            "字段": ["货物品名", "货物数量", "独立箱数"],
                            "值": [bill_info.goods_name, bill_info.quantity, bill_info.container_count]
                        }
                        st.write("货物信息：")
                        st.dataframe(goods_info, hide_index=True)
                        
                        # 添加导出按钮
                        csv_data = "提单号,发运港口,收货港口,发货人,收货人,通知人,货物品名,货物数量,独立箱数,发运时间\n"
                        csv_data += f"{bill_info.bill_number},{bill_info.departure_port},{bill_info.arrival_port},{bill_info.shipper},{bill_info.consignee},{bill_info.notify_party},{bill_info.goods_name},{bill_info.quantity},{bill_info.container_count},{bill_info.shipping_date}"
                        
                        # 使用Streamlit的下载功能
                        st.download_button(
                            label="导出CSV文件",
                            data=csv_data.encode('utf-8-sig'),
                            file_name="提单信息.csv",
                            mime="text/csv",
                            help="点击下载提单信息的CSV文件"
                        )
                        
                        # 添加翻译按钮
                        if st.button("一键翻译"):
                            with st.spinner("正在翻译..."):
                                translated_info = extractor.translate(bill_info)
                                
                                st.subheader("翻译结果")
                                
                                # 基本信息表格（翻译结果）
                                basic_info_translated = {
                                    "字段": ["提单号", "发运港口", "收货港口", "发运时间"],
                                    "值": [translated_info.bill_number, translated_info.departure_port, 
                                          translated_info.arrival_port, translated_info.shipping_date]
                                }
                                st.write("基本信息：")
                                st.dataframe(basic_info_translated, hide_index=True)
                                
                                # 相关方信息表格（翻译结果）
                                party_info_translated = {
                                    "字段": ["发货人", "收货人", "通知人"],
                                    "值": [translated_info.shipper, translated_info.consignee, 
                                          translated_info.notify_party]
                                }
                                st.write("相关方信息：")
                                st.dataframe(party_info_translated, hide_index=True)
                                
                                # 货物信息表格（翻译结果）
                                goods_info_translated = {
                                    "字段": ["货物品名", "货物数量", "独立箱数"],
                                    "值": [translated_info.goods_name, translated_info.quantity, 
                                          translated_info.container_count]
                                }
                                st.write("货物信息：")
                                st.dataframe(goods_info_translated, hide_index=True)
                    except Exception as e:
                        st.warning(f"提取结构化信息时发生错误：{str(e)}")
                else:
                    st.info("请输入OpenAI API Key以启用提单信息提取功能")
                
                # 创建两行布局
                st.subheader("PDF原文")
                # 使用base64编码显示PDF文件
                with open(pdf_path, "rb") as pdf_file:
                    base64_pdf = base64.b64encode(pdf_file.read()).decode('utf-8')
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)
                
                # 显示Markdown预览效果
                st.subheader("预览效果")
                # 使用容器控制总宽度，并设置CSS样式使文本自动换行
                with st.container():
                    st.markdown(
                        f"<div style='width: 100%; word-wrap: break-word; overflow-wrap: break-word;'>{markdown_result}</div>",
                        unsafe_allow_html=True
                    )
                
                # 在底部显示Markdown源码
                with st.expander("查看Markdown源码", expanded=False):
                    st.text_area("Markdown内容", markdown_result, height=400)
                
                # 清理临时文件
                if pdf_path.exists():
                    pdf_path.unlink()
                
            except Exception as e:
                st.error(f"处理PDF时发生错误：{str(e)}")

if __name__ == "__main__":
    main()