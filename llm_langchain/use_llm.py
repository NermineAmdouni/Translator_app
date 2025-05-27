from .chain import get_cleaning_chain


def clean_text(text: str) -> str:
    return get_cleaning_chain().invoke({"text": text}).content







