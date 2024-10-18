import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Print NEO4J_URI to verify if the .env is loaded correctly
print(os.getenv('NEO4J_URI'))

# Initialize the language model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)

# Initialize Neo4jGraph with correct URI
graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
    enhanced_schema=True,
)
qery =  """ 
load csv  with  headers from  'https://raw.githubusercontent.com/huynhanh48/Duan/master/FAQ.csv'
as row 
create (user:userFAQ {question:row.Question,answer: row.Answer})
"""
graph.query(qery)
graph.refresh_schema()
from langchain_core.prompts import ChatPromptTemplate

# Tạo prompt để tạo câu hỏi liên quan
translation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Bạn sẽ là trợ lý ảo về ngân hàng hỗ trợ người dùng phát sinh ra query dễ hiểu cho AI hiểu\
            tìm một câu hỏi liên quan đến query trên , kết hợp cả câu góc và query bạn hãy tạo ra query mới dễ hiểu nhất cho  mô hình chat bot có thể trả lời\
            example : Dịch vụ Ngân hàng số VCB Digibank là gì?\
            query new : dịch vụ số VCB Digibank và và một số thông tin về VCB Digibank?\
            không hiển thị thông tin  thừa thải chỉ hiện thị câu trả lời của bạn khi tạo ra câu query mới",
        ),
        ("human", "{input}"),
    ]
)


# Kết hợp prompt với mô hình ngôn ngữ
chain = translation_prompt | llm 

# Chạy truy vấn để nhận 3 câu hỏi liên quan
response = chain.invoke({"input": 'Dịch vụ Ngân hàng số VCB Digibank là gì?'}) 

perfomencechain = response.content
print(perfomencechain)
# Enable dangerous requests and create the GraphCypherQAChain
chain = GraphCypherQAChain.from_llm(
    graph=graph, 
    llm=llm, 
    verbose=True, 
    allow_dangerous_requests=True  # Acknowledge dangerous requests
)

# Run a query to the chain
response = chain.invoke({"query": perfomencechain})
print(response)