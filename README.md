# AlphaRename


Install Required library
`torch, transformers, vertexai, datasets`

Run commands the AlphaRename directory

Modidy the `__main__.py` file in each model directory to set up which you want to run(COT, Zero shot, Few Shot, Substitution)

Run Gemini Evaluation
`python -m gemini`

Run Lamma Evaluation
`python -m llama`

Run StarCoder Evaluation
`python -m starcoder`

Run DeepSeek Evaluation
`python -m deepseek`

Generate alphaRename dataset
`python alpha/generate.py`

Generate Substitution dataset
`python alpha/generate_substitution.py`

Generate instruction function application traces
`python alpha/generate_execution.py`

Generate hint list used for generating instruction dataset.
`python alpha/generate_hint.py`
