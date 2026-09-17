from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import pandas as pd

load_dotenv()

def load_file(file_path: str) -> str:
    df = pd.read_csv(file_path, nrows=500)
    reader = df.to_string(index=False)
    print(df.head(5))

    return reader

def main():

    print("Loading data from CSV files...")
          
    information = load_file("data/arch-amd/labeled/benign.csv")
    information2 = load_file("data/arch-amd/labeled/ransom-revil.csv")
    information3 = load_file("data/arch-amd/labeled/spectre.csv")

    print("Data loaded successfully.")

    summary_template = """
    given the information {information} and {information2} and {information3} about the benign, ransomware, and spectre hardware events datasets. Each column represents a different hardware event: Branch instruction retired,  Branch misses,  L2 cache references,  L2 cache misses, and  Instruction retired. I want you to analyze the information and provide me with the following:
    1. A short summary of each dataset 
    2. Why it belong to that class base on the hardware events and the features of the dataset
    3. Remember each class behavior and the features of the dataset.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information", "information2", "information3"], template=summary_template
    )

    llm = ChatOllama(temperature=0, model="gemma4:e4b")
    # llm = ChatOpenAI(temperature=0, model="gpt-5")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"information": information, "information2": information2, "information3": information3})
    print(response.content)

    information4 = load_file("data/arch-amd/original/test-alphv.csv")
    information5 = load_file("data/arch-amd/original/test-sodinokibi.csv")
    information6 = load_file("data/arch-amd/original/test-spectre.csv")

    classification_template = """
        given the information {information4} and {information5} and {information6}. Each column represents a different hardware event: Branch instruction retired,  Branch misses,  L2 cache references,  L2 cache misses, and  Instruction retired. I want you to analyze the information and provide me which class (benign, ransomware, spectre) each dataset belongs to according to the previous analysis and the features of the dataset. Please provide a brief explanation for your classification.
        """
    classification_prompt_template = PromptTemplate(
            input_variables=["information4", "information5", "information6"], template=classification_template
        )
    chain2 = classification_prompt_template | llm
    response2 = chain2.invoke(input={"information4": information4, "information5": information5, "information6": information6})
    print(response2.content)

if __name__ == "__main__":
    main()
