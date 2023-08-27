from pathlib import Path

from dotenv import load_dotenv, find_dotenv

from longdata.longevity_data_chain import LongevityDataChain

key_path = find_dotenv()
_ = load_dotenv(key_path, override=True)

from langchain.chat_models import ChatOpenAI


def main():
    # model ="gpt-3.5-turbo"
    model = "gpt-4"
    llm = ChatOpenAI(model=model, temperature=0)
    router = LongevityDataChain.from_folder(llm, Path("data"), return_intermediate_steps = True)

    with open("data/questions_short.tsv") as f:
        lines = f.readlines()
        for line in lines:
            question, answer = line.split("\t")
            print(router(question), "expected:", answer)


if __name__ == "__main__":
    main()