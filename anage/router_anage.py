from dotenv import load_dotenv
_ = load_dotenv("test.env", override=True)

from langchain.chains.router import MultiPromptChain
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.base import MultiRouteChain
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.agents import tool
import thefuzz.fuzz as fuzz
import polars as pl
from polars.internals import DataFrame

from typing import List
class CustomRouterChain(MultiRouteChain):
    @property
    def output_keys(self) -> List[str]:
        return ["output"]


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
                res += cel + " ; "
        res = res[:-3] + "\n"
        if i > 8:
            break
        i += 1

    return res[:-1]

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

@tool
def animal_information(input_text: str) -> str:
    """You should use this tool for getting information about animals. It returns csv table where rows separated with new line character
    and columns with ";" symbol. This table containe information about different animals which names are similar but only one animal is what you need.
    First row is header of this table with fields "Science name", "Common name" and field from input text.
    Input should be string separated with ";" first part it is animal name, common name or latin name (Genus Species). Second part is column of interest.
    For example input_text = "dog;body mass" will query table with information about animals which name contain dog and with additional field "Body mass (g)".
    You could query this function with such fields: Kingdom, Phylum, Class, Order, Family, Genus, Species, Female maturity, Male maturity, Gestation/Incubation, Weaning, Litter/Clutch size, Litters/Clutches, Inter-litter/Interbirth interval, Birth weight, Weaning weight, Adult weight, Growth rate, Maximum longevity, Infant mortality rate IMR, Mortality Rate Doubling Time MRDT, Metabolic rate, Body mass, Temperature.
    If there is no animal name in the table say its is unknown animal."""

    parts = input_text.split(";")
    column_name = get_column(parts[1].strip())
    search_text = " "+parts[0].strip()+" "

    def levenshtein_dist(struct: dict) -> int:
        return max(fuzz.partial_ratio(struct["Science name"], search_text), fuzz.partial_ratio(struct["Common name"], search_text))

    frame = pl.read_csv("anage_data.csv", infer_schema_length=0)
    frame = frame.with_columns((pl.col("Genus")+" "+pl.col("Species")).alias("Science name"))
    frame = frame.select([pl.struct(["Science name", "Common name"]).apply(levenshtein_dist).alias("dist"), "Science name", "Common name", column_name])
    frame = frame.sort(by="dist", reverse=True).select(["Science name", "Common name", column_name])

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
    For example input_text = 'Temperature;min' will return table with information about animal with smallest temperature.
    Fields that could be query is following: Female maturity, Male maturity, Gestation/Incubation, Weaning, Litter/Clutch size, Litters/Clutches, Inter-litter/Interbirth interval, Birth weight, Weaning weight, Adult weight, Growth rate, Maximum longevity, Infant mortality rate IMR, Mortality Rate Doubling Time MRDT, Metabolic rate, Body mass, Temperature."""
    parts = input_text.split(";")
    column_name = get_column(parts[0].strip())
    command = parts[1].strip()

    frame = pl.read_csv("anage_data.csv", infer_schema_length=0)
    frame = frame.with_columns((pl.col("Genus") + " " + pl.col("Species")).alias("Science name"))
    frame = frame.filter(~pl.col(column_name).is_null())
    frame = frame.with_column(pl.col(column_name).cast(pl.Float32))
    if command.lower() == "max":
        return frame.filter(pl.max(column_name) == pl.col(column_name)).select(["Science name", "Common name", column_name])
    if command.lower() == "min":
        return frame.filter(pl.min(column_name) == pl.col(column_name)).select(["Science name", "Common name", column_name])

    raise ValueError(f"Error wrong operation ({command}) specified in input to animal_min_max_avg_information() tool.")


# print(animals_min_max_information("Maximum longevity (yrs);min"))
# print(animal_information("Anaxyrus terrestris;adult weight"))
# model="gpt-3.5-turbo",
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

anage_agent = initialize_agent(
    [animal_information, animals_min_max_information],
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose = True)

anage_template = """You are a very smart biology professor. \
You are great at answering questions about animals biology in a concise and easy to understand manner. \
When you don't know the answer to a question you admit that you don't know.

Here is a question:
{input}"""

prompt_infos = [
    {
        "name": "anage",
        "description": "Good for answering questions about animals biology",
        "prompt_template": anage_template,
    }
]

# llm = OpenAI()

destination_chains = {}
for p_info in prompt_infos:
    name = p_info["name"]
    prompt_template = p_info["prompt_template"]
    prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
    # chain = LLMChain(llm=llm, prompt=prompt)
    destination_chains[name] = anage_agent
default_chain = ConversationChain(llm=llm, output_key="output")

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser(),
)
router_chain = LLMRouterChain.from_llm(llm, router_prompt)
# CustomRouterChain(
chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=default_chain,
    verbose=True,
)

with open("questions.tsv") as f:
    i = 0
    skeep = 6
    limit = 7
    for line in f:
        if i == limit:
            break
        i += 1
        if i <= skeep:
            continue
        parts = line.split("\t")
        try:
            print(chain(parts[0], True), "expected: ", parts[1])
        except Exception as e:
            print(e)

# print(chain.run("What is black body radiation?"))

# print(chain.run("What is the first prime number greater than 40 such that one plus the prime number is divisible by 3"))