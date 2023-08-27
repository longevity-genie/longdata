from langchain.agents import AgentType, initialize_agent
from langchain.agents import tool
import thefuzz.fuzz as fuzz
import polars as pl
from langchain.schema.language_model import BaseLanguageModel
from util import write_data
from typing import Any

geneage_template = """You are a very smart biology professor. \
You are great at answering questions about longevity genes in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""

def get_geneage_agent_info(llm: BaseLanguageModel, table_path:str, verbose: bool = False, **kwargs: Any,) -> dict:
    @tool
    def gene_information(input_text: str) -> str:
        """You should use this tool to get information about animal genes.
        It returns csv table where rows are separated with new line characters and columns with ";" symbol.
        This table contains information about different genes whose names are similar, but only some of them are what you need.
        The first row is the header of this table. Longevity Influence is the main column that describes the overall gene influence on longevity.
        The method column describes the intervention type that was done in the experiment. Overexpression is the only type of intervention that
        makes a direct correlation with gene longevity influence. In other words, if Method is Overexpression and Lifespan Effect is increase
        then it is pro longevity influence. If Method is Overexpression and Lifespan Effect is decrease
        then it is anti longevity influence. All other types of Method block gene expression and has inverse correlation with longevity influence.
        For example if Method is Deletion and Lifespan Effect is increase then it is anti longevity influence. If Method is Knockout and
        Lifespan Effect is decrease then it is pro longevity influence.
        Input should be the string with the gene name. It could be Entrez Gene ID, Gene Symbol, Gene Name,
        Unigene ID, Ensembl ID. If there is no gene name in the table, say its is unknown gene."""
        search_text = (" " + input_text + " ").lower()

        def levenshtein_dist(struct: dict) -> int:
            return max(fuzz.partial_ratio(" " + str(struct["Entrez Gene ID"]) + " ".lower(), search_text),
                       fuzz.partial_ratio(" " + str(struct["Gene Symbol"]) + " ".lower(), search_text),
                       fuzz.partial_ratio(" " + str(struct["Gene Name"]) + " ".lower(), search_text),
                       fuzz.partial_ratio(" " + str(struct["Unigene ID"]) + " ".lower(), search_text),
                       fuzz.partial_ratio(" " + str(struct["Ensembl ID"]) + " ".lower(), search_text))

        frame = pl.read_csv(table_path, separator=";", infer_schema_length=0)
        frame = frame.select([pl.struct(
            ["Entrez Gene ID", "Gene Symbol", "Gene Name", "Unigene ID", "Ensembl ID"]).apply(levenshtein_dist).alias(
            "dist"),
                              "Entrez Gene ID", "Gene Symbol", "Gene Name", "Unigene ID", "Ensembl ID",
                              "Lifespan Effect", "Phenotype Description", "Longevity Influence", "Method"])
        frame = frame.sort(by="dist", descending=True).select(pl.exclude("dist"))

        return write_data(frame)

    agent = initialize_agent(
        [gene_information],
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=verbose,
        **kwargs,)

    return {
        "name": "geneage",
        "description": "Good for answering questions about longevity genes",
        "prompt_template": geneage_template,
        "chain": agent,
    }
