"""Main entrypoint for the app."""
import logging
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

app = FastAPI()
templates = Jinja2Templates(directory="templates")
docsearch: Optional[Chroma] = None

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)  # 设置全局日志level，不设置默认WARN

# save log to file
file_handler = logging.FileHandler('logs/app.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt='%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(formatter)

# print to screen
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


@app.on_event("startup")
async def startup_event():
    print("startup begin")
    # 加载文件夹中的所有txt类型的文件
    loader = DirectoryLoader('../doc', glob='**/*.epub')
    # 将数据转成 document 对象，每个文件会作为一个 document
    documents = loader.load()

    # 初始化加载器
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    # 切割加载的 document
    split_docs = text_splitter.split_documents(documents)

    # 初始化 openai 的 embeddings 对象
    embeddings = OpenAIEmbeddings()
    # 将 document 通过 openai 的 embeddings 对象计算 embedding 向量信息并临时存入 Chroma 向量数据库，用于后续匹配查询
    global docsearch
    docsearch = Chroma.from_documents(split_docs, embeddings)

    print("startup end")


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(docsearch, question_handler, stream_handler)
    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            logger.info("question:" + question)
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            chat_history.append((question, result["answer"]))

            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn
    logger.info("main start")
    uvicorn.run(app, host="0.0.0.0", port=9001)
