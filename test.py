# coding=utf-8
import sys
import io
import os
# 设置标准输出编码为UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 打印启动信息
print("Start Amazing RAG~")

# 导入所需库
from langchain.chains import RetrievalQA  # 用于构建问答链
from langchain_community.llms import Ollama  # 使用Ollama语言模型
from langchain_community.document_loaders import PDFPlumberLoader  # PDF文档加载器
from langchain_community.vectorstores import Chroma  # 向量存储
from langchain_community.embeddings import OllamaEmbeddings  # 嵌入模型

 
# os.environ["DASHSCOPE_API_KEY"] = "你的通义千问 API key"
 
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "lsxxxxx"
os.environ["LANGCHAIN_PROJECT"] = "RAG_test"


# 加载PDF文档并分割
loader = PDFPlumberLoader("TriageCviz.pdf")
documents = loader.load_and_split()

# 初始化嵌入模型
embeddings = OllamaEmbeddings(model="deepseek-r1:7b")

# 创建向量存储
vectorstore = Chroma.from_documents(documents, embeddings)

# 初始化语言模型
llm = Ollama(model="deepseek-r1:7b")

# 构建问答链
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectorstore.as_retriever())

# 定义查询问题
query = "状态机信息在哪里查？"

# 执行查询并获取响应
response = qa_chain.run(query)

# 打印查询结果
print(response)
