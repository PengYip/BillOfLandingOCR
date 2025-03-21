from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import BaseLanguageModel
from dataclasses import dataclass

@dataclass
class BillTranslation:
    bill_number: str
    departure_port: str
    arrival_port: str
    shipper: str
    consignee: str
    notify_party: str
    quantity: str
    container_count: str
    goods_name: str
    shipping_date: str

class BillInfo(BaseModel):
    bill_number: str = Field(description="提单号")
    departure_port: str = Field(description="发运港口")
    arrival_port: str = Field(description="收货港口")
    shipper: str = Field(description="发货人")
    consignee: str = Field(description="收货人")
    notify_party: str = Field(description="通知人")
    quantity: str = Field(description="货物数量")
    container_count: str = Field(description="独立箱数")
    goods_name: str = Field(description="货物品名")
    shipping_date: str = Field(description="发运时间")

class BillExtractor:
    def __init__(self, api_key: str, model_type: str = "openai", base_url: Optional[str] = None):
        self.parser = PydanticOutputParser(pydantic_object=BillInfo)
        
        if model_type == "openai":
            self.llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=api_key)
        elif model_type == "deepseek":
            from langchain.chat_models import ChatOpenAI
            self.llm = ChatOpenAI(
                temperature=0,
                model_name="deepseek-v3",
                openai_api_key=api_key,
                openai_api_base=base_url
            )
        
        template = """
        从以下提单文本中提取关键信息。请仔细分析文本内容，提取以下字段：
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
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    def extract(self, text: str) -> BillInfo:
        """从文本中提取结构化的提单信息"""
        _input = self.prompt.format(text=text)
        output = self.llm.predict(_input)
        return self.parser.parse(output)
    
    def translate(self, bill_info: BillInfo) -> BillTranslation:
        """将提单信息翻译成中文"""
        translate_prompt = """
        请将以下提单信息翻译成中文，保持专业性和准确性：
        
        提单号: {bill_number}
        发运港口: {departure_port}
        收货港口: {arrival_port}
        发货人: {shipper}
        收货人: {consignee}
        通知人: {notify_party}
        货物数量: {quantity}
        独立箱数: {container_count}
        货物品名: {goods_name}
        发运时间: {shipping_date}
        """
        
        # 格式化提示词
        _input = translate_prompt.format(
            bill_number=bill_info.bill_number,
            departure_port=bill_info.departure_port,
            arrival_port=bill_info.arrival_port,
            shipper=bill_info.shipper,
            consignee=bill_info.consignee,
            notify_party=bill_info.notify_party,
            quantity=bill_info.quantity,
            container_count=bill_info.container_count,
            goods_name=bill_info.goods_name,
            shipping_date=bill_info.shipping_date
        )
        
        # 调用LLM进行翻译
        output = self.llm.predict(_input)
        
        # 解析翻译结果
        lines = output.strip().split('\n')
        translated_info = {}
        
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                if '提单号' in key:
                    translated_info['bill_number'] = value
                elif '发运港口' in key:
                    translated_info['departure_port'] = value
                elif '收货港口' in key:
                    translated_info['arrival_port'] = value
                elif '发货人' in key:
                    translated_info['shipper'] = value
                elif '收货人' in key:
                    translated_info['consignee'] = value
                elif '通知人' in key:
                    translated_info['notify_party'] = value
                elif '货物数量' in key:
                    translated_info['quantity'] = value
                elif '独立箱数' in key:
                    translated_info['container_count'] = value
                elif '货物品名' in key:
                    translated_info['goods_name'] = value
                elif '发运时间' in key:
                    translated_info['shipping_date'] = value
        
        return BillTranslation(
            bill_number=translated_info.get('bill_number', ''),
            departure_port=translated_info.get('departure_port', ''),
            arrival_port=translated_info.get('arrival_port', ''),
            shipper=translated_info.get('shipper', ''),
            consignee=translated_info.get('consignee', ''),
            notify_party=translated_info.get('notify_party', ''),
            quantity=translated_info.get('quantity', ''),
            container_count=translated_info.get('container_count', ''),
            goods_name=translated_info.get('goods_name', ''),
            shipping_date=translated_info.get('shipping_date', '')
        )