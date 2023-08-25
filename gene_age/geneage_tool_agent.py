from dotenv import load_dotenv
_ = load_dotenv("test.env", override=True)

from langchain.agents import create_csv_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.agents import tool
import thefuzz.fuzz as fuzz
import polars as pl
from polars.internals import DataFrame
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

def write_data(frame: DataFrame) -> str:
    res = ""
    i = 0
    for col in frame.get_columns():
        res += col.name + " ; "
    res = res[:-3] + "\n"
    for line in frame.rows():
        for cel in line:
            if cel is None:
                res += "unknown ; "
            else:
                res += str(cel) + " ; "
        res = res[:-3] + "\n"
        if i > 8:
            break
        i += 1

    return res[:-1]


@tool
def gene_information(input_text: str) -> str:
    """You should use this tool to get information about animal genes.
    It returns csv table where rows are separated with new line characters and columns with ";" symbol.
    This table contains information about different genes whose names are similar, but only some of them are what you need.
    The first row is the header of this table. Longevity Influence is the main column that describes the overall gene influence on longevity.
    The method column describes the intervention type that was done in the experiment. Overexpression is the only type of intervention that
    makes a direct correlation with gene longevity influence. In other words, if Method is Overexpression and Lifespan Effect is increase
    then it is pro longevity influence. If Method is Overexpression and Lifespan Effect is decrease
    then it is anti longevity influence. All other types of Method block gene expression and has invers correlation with longevity influence.
    For example if Method is Deletion and Lifespan Effect is increase then it is anti longevity influence. If Method is Knockout and
    Lifespan Effect is decrease then it is pro longevity influence.
    Input should be the string with the gene name. It could be Entrez Gene ID, Gene Symbol, Gene Name,
    Unigene ID, Ensembl ID. If there is no gene name in the table, say its is unknown gene."""
    search_text = (" "+input_text+" ").lower()

    def levenshtein_dist(struct: dict) -> int:
        return max(fuzz.partial_ratio(" "+str(struct["Entrez Gene ID"])+" ".lower(), search_text),
                   fuzz.partial_ratio(" "+str(struct["Gene Symbol"])+" ".lower(), search_text),
                   fuzz.partial_ratio(" "+str(struct["Gene Name"])+" ".lower(), search_text),
                   fuzz.partial_ratio(" "+str(struct["Unigene ID"])+" ".lower(), search_text),
                   fuzz.partial_ratio(" "+str(struct["Ensembl ID"])+" ".lower(), search_text))

    frame = pl.read_csv("data/genage_models.csv", sep=";", infer_schema_length=0)
    frame = frame.select([pl.struct(["Entrez Gene ID", "Gene Symbol", "Gene Name", "Unigene ID", "Ensembl ID"]).apply(levenshtein_dist).alias("dist"),
                          "Entrez Gene ID", "Gene Symbol", "Gene Name", "Unigene ID", "Ensembl ID", "Lifespan Effect", "Phenotype Description", "Longevity Influence", "Method"])
    frame = frame.sort(by="dist", reverse=True).select(pl.exclude("dist"))

    return write_data(frame)

# print(gene_information("181727"))
# print("\n\n")
# print(gene_information("aak-2"))
# print("\n\n")
# print(gene_information("AMP-Activated Kinase"))
# print("\n\n")
# print(gene_information("Cel.17479"))
# print("\n\n")
# print(gene_information("T01C8.1"))
# print(gene_information("APOE"))
# print(gene_information("apoe"))

# model="gpt-3.5-turbo",
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

agent = initialize_agent(
    [gene_information],
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True)

print(agent("What is effect of gene 14-3-3epsilon on longevity?"))