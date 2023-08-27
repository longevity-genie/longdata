from langchain.agents import AgentType, initialize_agent
from langchain.agents import tool
import polars as pl
from langchain.schema.language_model import BaseLanguageModel
from typing import Any

from util import write_data

def get_references(rsid: str) -> str:
    frame = pl.read_csv("data/variants.tsv", separator="\t", infer_schema_length=0)
    frame = frame.filter(pl.col("rsid") == rsid).with_column(
        ("https://pubmed.ncbi.nlm.nih.gov/" + pl.col("quickpubmed")).alias("quickpubmed")).select("quickpubmed")
    rows = frame.rows("quickpubmed")
    rows = [str(row[0]) for row in rows]
    return "\n".join(rows)


longevitymap_template = """You are a very smart biology professor. \
You are great at answering questions about human single nucleotide polymorphisms (snp) and rsid in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""


def get_longevitymap_agent_info(llm: BaseLanguageModel, table_path:str, verbose: bool = False, **kwargs: Any,) -> dict:

    @tool
    def rsid_information(input_text: str) -> str:
        """You should use this tool to get information about rsid. It is an identifier that starts with the letters rs and number.
        It points to snp, single nucleotide polymorphism.
        It returns csv table where rows are separated with new line characters and columns with ";" symbol.
        The first row is the header of this table. It has the following columns rsid, allele, zygosity, weight.
        rsid column is snp identifier for example rs1234, allele column is letter represents nucleotide in specific location.
        zygosity column could be het for heterozygosity, two different alleles and hom for homozygosity, two same alleles.
        If zygosity column is het and allele column is C, for example, it means that weight applies to all combinations of allels.
        Such as CA, CG, CT, AC, GC, TC but not CC because it is homozygosity. Also, there is different notation C/A is the same as CA,
        G/G is same as GG, T/A is same as TA. weight column has number from -1 to 1, minus means it has negative effect on longevity
        and positive number means it have positive effect on longevity. Magnitude of this number represents effect strength.
        Usually, it is 0.5 for strong effect and 0.01 for weak effect.
        Input should be the string with the rsid. If there is no rsid in the table, say it has no longevity effect."""

        frame1 = pl.read_csv(table_path, separator="\t", infer_schema_length=0)
        frame1 = frame1.filter(pl.col("rsid") == input_text).select(["rsid", "allele", "zygosity", "weight"])

        return write_data(frame1)

    agent = initialize_agent(
        [rsid_information],
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
        **kwargs,)

    return {
        "name": "longevitymap",
        "description": "Good for answering questions about rsid",
        "prompt_template": longevitymap_template,
        "chain": agent,
    }