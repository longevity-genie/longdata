from dotenv import load_dotenv, find_dotenv
key_path = find_dotenv()
_ = load_dotenv(key_path, override=True)

from langchain.chat_models import ChatOpenAI
from agent_router_chain import AgentRouterChain
from anage.anage_agent import get_anage_agent_info
from geneage.geneage_agent import get_geneage_agent_info
from longevitymap.longevitymap_agent import get_longevitymap_agent_info


def main():
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    router = AgentRouterChain.from_prompts(llm, [
        get_anage_agent_info(llm, "data/anage_data.csv", True),
        get_geneage_agent_info(llm, "data/genage_models.csv", True),
        get_longevitymap_agent_info(llm, "data/longevitymap_weights.tsv", True)])

    with open("data/questions_short.tsv") as f:
        lines = f.readlines()
        for line in lines:
            question, answer = line.split("\t")
            print(router(question), "expected:", answer)


if __name__ == "__main__":
    main()