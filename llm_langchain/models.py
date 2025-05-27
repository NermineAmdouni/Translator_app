from langchain_openai import AzureChatOpenAI 
from langchain.prompts import PromptTemplate
from .config import OPENAI_API_KEY, OPENAI_API_ENDPOINT, MODEL_NAME, OPENAI_API_VERSION

def get_llm():
    return AzureChatOpenAI(
        api_key=OPENAI_API_KEY,
        azure_endpoint=OPENAI_API_ENDPOINT, 
        deployment_name=MODEL_NAME,          
        api_version=OPENAI_API_VERSION
    )

def create_chain(prompt_template, llm=None):
    if llm is None:
        llm = get_llm()
    
    if isinstance(prompt_template, str):
        prompt = PromptTemplate.from_template(prompt_template)
    else:
        prompt = prompt_template
        
    return prompt | llm