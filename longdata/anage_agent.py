from typing import Any

from langchain.agents import AgentType, initialize_agent
from langchain.agents import tool
import thefuzz.fuzz as fuzz
import polars as pl
from langchain.schema.language_model import BaseLanguageModel

from util import write_data

def get_column(col_name):
    fields = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus",
              "Species", "Female maturity (days)", "Male maturity (days)",
              "Gestation/Incubation (days)", "Weaning (days)", "Litter/Clutch size",
              "Litters/Clutches per year", "Inter-litter/Interbirth interval",
              "Birth weight (g)", "Weaning weight (g)", "Adult weight (g)",
              "Growth rate (1/days)", "Maximum longevity (yrs)", "IMR (per yr)",
              "MRDT (yrs)", "Metabolic rate (W)", "Body mass (g)", "Temperature (K)"]

    arry = sorted([[fuzz.partial_ratio(col_name, f), f] for f in fields], reverse=True)
    return arry[0][1]


anage_template = """You are a very smart biology professor. \
You are great at answering questions about animals biology in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""

def get_anage_agent_info(llm: BaseLanguageModel, table_path:str, verbose: bool = False, **kwargs: Any,) -> dict:

    @tool
    def animal_information(input_text: str) -> str:
        """You should use this tool for getting information about animals. It returns csv table where rows separated with new line character
        and columns with ";" symbol. This table contains information about different animals which names are similar but only one animal is what you need.
        First row is header of this table with fields "Science name", "Common name" and field from input text.
        Input should be string separated with ";" first part it is animal name, common name or latin name (Genus Species). Second part is column of interest.
        For example input_text = "dog;body mass" will query table with information about animals which name contain dog and with additional field "Body mass (g)".
        You could query this function with such fields: Kingdom, Phylum, Class, Order, Family, Genus, Species, Female maturity, Male maturity, Gestation/Incubation, Weaning, Litter/Clutch size, Litters/Clutches, Inter-litter/Interbirth interval, Birth weight, Weaning weight, Adult weight, Growth rate, Maximum longevity, Infant mortality rate IMR, Mortality Rate Doubling Time MRDT, Metabolic rate, Body mass, Temperature.
        If there is no animal name in the table say its is unknown animal."""

        parts = input_text.split(";")
        column_name = get_column(parts[1].strip())
        search_text = " " + parts[0].strip() + " "

        def levenshtein_dist(struct: dict) -> int:
            return max(fuzz.partial_ratio(struct["Science name"], search_text),
                       fuzz.partial_ratio(struct["Common name"], search_text))

        frame = pl.read_csv(table_path, infer_schema_length=0)
        frame = frame.with_columns((pl.col("Genus") + " " + pl.col("Species")).alias("Science name"))
        frame = frame.select(
            [pl.struct(["Science name", "Common name"]).apply(levenshtein_dist).alias("dist"), "Science name",
             "Common name", column_name])
        frame = frame.sort(by="dist", descending=True).select(["Science name", "Common name", column_name])

        return write_data(frame)

    @tool
    def animals_min_max_information(input_text: str) -> str:
        """You should use this tool for getting minimum and maximum value of animal features among all the animals.
        You should use this tool only if you do NOT know animal name.
        It returns table with Science name, Common name and field you specify in input field,
        value in this field is result of operation you specify in input.
        Input should be string with two parts separated with ';' sign.
        First part is column name and second is operation to be done on data in this column.
        Operation should be 'min' for minimum and 'max' for maximum operations.
        For example input_text = 'Temperature;min' will return table with information about animal with the smallest temperature.
        Fields that could be query is following: Female maturity, Male maturity, Gestation/Incubation, Weaning, Litter/Clutch size, Litters/Clutches, Inter-litter/Interbirth interval, Birth weight, Weaning weight, Adult weight, Growth rate, Maximum longevity, Infant mortality rate IMR, Mortality Rate Doubling Time MRDT, Metabolic rate, Body mass, Temperature."""
        parts = input_text.split(";")
        column_name = get_column(parts[0].strip())
        command = parts[1].strip()

        frame = pl.read_csv(table_path, infer_schema_length=0)
        frame = frame.with_columns((pl.col("Genus") + " " + pl.col("Species")).alias("Science name"))
        frame = frame.filter(~pl.col(column_name).is_null())
        frame = frame.with_columns([pl.col(column_name).cast(pl.Float32)])
        if command.lower() == "max":
            return frame.filter(pl.max(column_name) == pl.col(column_name)).select(
                ["Science name", "Common name", column_name])
        if command.lower() == "min":
            return frame.filter(pl.min(column_name) == pl.col(column_name)).select(
                ["Science name", "Common name", column_name])

        raise ValueError(
            f"Error wrong operation ({command}) specified in input to animal_min_max_avg_information() tool.")

    agent = initialize_agent(
        [animal_information, animals_min_max_information],
        llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose = verbose,
        **kwargs,)

    return {
        "name": "anage",
        "description": "Good for answering questions about animals biology",
        "prompt_template": anage_template,
        "chain": agent,
    }
