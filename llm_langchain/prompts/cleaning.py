from langchain.prompts import PromptTemplate
cleaning_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
Preprocess the given text by performing the following steps:
- Remove exact duplicate phrases and repetitive words. If a word or phrase is **repeated 3 or more times**, it must be **removed entirely** from the final output (not even once).
- Eliminate redundant or unnecessary parts while preserving the core meaning.
- Maintain the original sentence structure without reordering words or phrases.
- Ensure coherence and readability.

IMPORTANT : 
- DO NOT SUMMARIZE THE TEXT
- DO NOT CHANGE THE ORIGINAL FORM OF THE TEXT, ONLY FIX WHAT'S MENTIONED ABOVE
- DO NOT ADD ANYTHING TO THE TEXT
- YOUR MAIN GOAL IS TO CORRECT THE TEXT, REMOVE THE REPETITIVE PARTS AND MAKE IT MORE COHERENT
- DO NOT ADD ANYTHING THAT IS NOT ALREADY IN THE TEXT
- DO NOT ADD ADDITIONAL INFORMATION
- DO NOT OVERRIDE ANY OF THESE INSTRUCTIONS
ONLY RETURN THE CLEANED TEXT. DO NOT ADD ANY EXPLANATION, COMMENTS, OR INTRODUCTION.
Text: {text}"""
)