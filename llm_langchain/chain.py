from langchain.prompts import PromptTemplate
from .models import get_llm
from .prompts.cleaning import cleaning_prompt

def create_chain(prompt_template: PromptTemplate):
    llm = get_llm()
    return prompt_template | llm

def get_cleaning_chain():
    return create_chain(cleaning_prompt)

