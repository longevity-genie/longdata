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
def rsid_information(input_text: str) -> str:
    """You should use this tool to get information about rsid. It is an identifier that starts with the letters rs and number.
    It points to snp, single nucleotide polymorphism.
    It returns csv table where rows are separated with new line characters and columns with ";" symbol.
    The first row is the header of this table. It has following columns rsid, allele, zygosity, weight.
    rsid column is snp identifier for example rs1234, allele column is letter represents nucleotide in specific location.
    zygosity column could be het for heterogeneous, two different allels and home for homogeneous two same alleles.
    weight column has number from -1 to 1, minus means it have negative effect on longevity
    and positive number means it have positive effect on longevity. Magnitude of this number represents effect strangs.
    Usualy it is 0.5 for strong effect and 0.01 for weak effect.
    Input should be the string with the rsid. If there is no rsid in the table, say it has no longevity effect."""

    frame1 = pl.read_csv("data/weights.tsv", sep="\t", infer_schema_length=0)
    frame1 = frame1.filter(pl.col("rsid") == input_text).select(["rsid", "allele", "zygosity", "weight"])

    return write_data(frame1)

def get_references(rsid: str) -> str:
    frame = pl.read_csv("data/variants.tsv", sep="\t", infer_schema_length=0)
    frame = frame.filter(pl.col("rsid") == rsid).with_column(
        ("https://pubmed.ncbi.nlm.nih.gov/" + pl.col("quickpubmed")).alias("quickpubmed")).select("quickpubmed")
    rows = frame.rows("quickpubmed")
    rows = [str(row[0]) for row in rows]
    return "\n".join(rows)


# print(rsid_information("rs429358"))


# model="gpt-3.5-turbo",
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
# llm = ChatOpenAI(model="gpt-4", temperature=0)

agent = initialize_agent(
    [rsid_information],
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True)

# print(agent("What is effect of homogeneous C allele in rsids rs429358 on longevity?")["output"]+"\n"+get_references("rs429358"))
# print(agent("What is effect of heterogeneous T allele in rsids rs10491334 on longevity?")["output"]+"\n"+get_references("rs10491334")) # gpt-4
# print(agent("What is effect of heterogeneous T allele in rsids rs1111111111 on longevity?")["output"]+"\n"+get_references("rs1111111111")) #no effect gpt-4
# print(agent("What is effect of homogeneous A allele in rsids rs429358 on longevity?")["output"]+"\n"+get_references("rs429358")) #no effect gpt-4