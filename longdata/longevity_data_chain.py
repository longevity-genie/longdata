from typing import Any

from langchain.schema.language_model import BaseLanguageModel

from longdata import agent_router_chain
from longdata.agent_router_chain import AgentRouterChain
from pathlib import Path

from longdata.anage_agent import get_anage_agent_info
from longdata.geneage_agent import get_geneage_agent_info
from longdata.longevitymap_agent import get_longevitymap_agent_info


class LongevityDataChain(AgentRouterChain):
    @classmethod
    def from_folder(
            cls,
            llm: BaseLanguageModel,
            folder: Path,
            verbose: bool = True,
            **kwargs: Any,
    ) -> AgentRouterChain:
        anage = str(folder / "anage_data.csv")
        genage = str(folder / "genage_models.csv")
        weights = str(folder / "longevitymap_weights.tsv")
        return cls.from_prompts(llm, [
            get_anage_agent_info(llm, anage, verbose, **kwargs,),
            get_geneage_agent_info(llm, genage, verbose, **kwargs,),
            get_longevitymap_agent_info(llm, weights, verbose, **kwargs,)],
            **kwargs,
        )