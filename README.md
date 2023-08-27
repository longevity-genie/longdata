# Longevity database agents

LongData is a submodule of longevity-genie devoted to LLM agents that query databases

When you start working it is recommended to pip install the local folder

```bash
micromamba create -f environment.yaml
micromamиa activate longdata
```

to develop locally it is recommended to:
```bash
pip install -e .
```

to run the test questions you can do:
```bash
python main.py
```