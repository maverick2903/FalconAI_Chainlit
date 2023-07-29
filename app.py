import os
import chainlit as cl
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")

model_id = 'tiiuae/falcon-7b-instruct'

falcon_llm = HuggingFaceHub(huggingfacehub_api_token=api_key,
                            repo_id=model_id,
                            model_kwargs={"temperature":0.8,"max_new_tokens":2000})

template = """
You are an AI model which gives helpful and clever answers to user queries.
{question}
"""

@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template=template, input_variables=["question"])
    falcon_chain = LLMChain(llm=falcon_llm, prompt=prompt, verbose=True)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", falcon_chain)


@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    # Send the response
    await cl.Message(content=res["text"]).send()
