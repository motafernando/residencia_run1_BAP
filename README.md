# Reprodução e Extensão do WinoBias (Mistral 7B e Llama 3 8B)

Este diretório reúne o código, os manifestos de configuração, os dados preparados e as saídas brutas das rodadas locais do benchmark WinoBias baseadas em *Bias after Prompting* (Sivakumar et al., 2025) e estendidas com o grid de prompts inspirado em Gao e Kreiss (2025). Em cada rodada são avaliadas 3.168 frases nas condições intrínseca, zero-shot e 3-shot, totalizando 9.504 registros por rodada com a semente 2026.

## Organização

- `prepare.py`, `protocol.py`, `parser_v3.py`, `parser_v4.py`, `parser_v5.py` e `run.py`: preparação dos dados, definição das variantes de prompt (`original`, `refined`, `evaluation_task` e `evaluation_gender_bias`), leitura das respostas, pontuação e execução da inferência.
- `recompute_saved_run.py`, `consolidar_resultados_v4.py`, `auditar_comparacao_mistral.py`, `auditar_parser_v5.py`, `audit_unknown_parsing.py` e `audit_parser_v3.py`: scripts de verificação e recálculo offline de métricas sobre as respostas salvas, sem nova inferência.
- `configs/mistral-full-3168.json` e `configs/mistral-grid-*.json`: manifestos das quatro variantes executadas com o modelo `mistralai/Mistral-7B-Instruct-v0.3`.
- `configs/llama-grid-*.json`: manifestos das variantes configuradas para o modelo `meta-llama/Meta-Llama-3-8B-Instruct`.
- `docs/grid-prompts-winobias.md`: especificação estrutural das quatro variantes do grid de prompts (`P0` a `P3`).
- `data/raw/corefBias/`: arquivos originais do WinoBias obtidos do commit `0bce984dd081bbc10b0622f326727a024c607895`, com a licença original.
- `data/prepared/`: inventário (`inventory.json`), lote piloto (`smoke24.json`) e conjunto completo normalizado (`all_records.json`, 3.168 frases).
- `results/`: diretórios das seis rodadas completas executadas (quatro com Mistral 7B e duas com Llama 3 8B), contendo `config.json`, `prompts.json`, `responses.jsonl`, `metrics.json`, `run.json`, `sample.json` e cópias exatas dos scripts utilizados (`*.used.py`).

Os modelos não estão incluídos neste repositório: os checkpoints oficiais (`mistralai/Mistral-7B-Instruct-v0.3`, revisão `0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a`, e `meta-llama/Meta-Llama-3-8B-Instruct`, revisão `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920`) são baixados via `download-model.py` para o diretório local `cache/` (ignorado pelo Git).

## Protocolo e escolhas de implementação

O artigo de referência orienta a comparação entre avaliação intrínseca, zero-shot e 3-shot no WinoBias, define RPA para as frases Type 2 e apresenta O-SB e A-SB como medidas de viés. A seção 3.2 especifica a comparação intrínseca pela probabilidade logarítmica total das candidatas. As Figuras 2 e 5 e o Apêndice A apresentam a estrutura dos prompts, os três exemplos de contexto e a ordenação aleatória das alternativas.

As escolhas abaixo estão registradas no código para permitir reprodução e verificação:

- **Dados:** utilizamos os oito arquivos `dev` e `test` do commit do `corefBias` indicado acima, totalizando 3.168 frases. `prepare.py` extrai o primeiro pronome marcado e as duas primeiras ocupações distintas, preservando as anotações completas em `data/prepared/`.
- **Prompts e variantes (`protocol.py` e `docs/grid-prompts-winobias.md`):** além da configuração `original` (`P0`), o parâmetro `prompt_variant` seleciona as variantes `refined` (`P1`), `evaluation_task` (`P2`) e `evaluation_gender_bias` (`P3`). A ordenação das alternativas é fixada por frase com a semente `2026`. Na geração zero-shot e 3-shot, `run.py` aplica o chat template do respectivo tokenizer.
- **Pontuação intrínseca e geração (`run.py`):** a condição intrínseca soma as log-probabilidades dos tokens de cada candidata sem normalização por comprimento. Para as respostas geradas, utiliza-se decodificação *greedy* (`do_sample=False`), limite de `48` novos tokens e semente `2026`.
- **Leitura das respostas (`parser_v3.py`, `parser_v4.py` e `parser_v5.py`):** preserva-se a separação entre o critério estrito de RPA no texto completo e a leitura pela alternativa escolhida. Os scripts de recálculo offline permitem auditar as saídas sem modificar os arquivos `responses.jsonl` brutos de cada execução.

## Rodadas completas versionadas em `results/`

1. **Mistral 7B — P0 (`original`):** `results/20261021T224558462021Z-inference/`
2. **Mistral 7B — P1 (`refined`):** `results/grid-20261026t211027z/full/refined/20261026T212312065659Z-inference/`
3. **Mistral 7B — P2 (`evaluation_task`):** `results/grid-20261026t211027z/full/evaluation_task/20261027T001137163705Z-inference/`
4. **Mistral 7B — P3 (`evaluation_gender_bias`):** `results/grid-20261026t211027z/full/evaluation_gender_bias/20261027T030929544031Z-inference/`
5. **Llama 3 8B — P0 (`original`):** `results/llama-fullgpu-round-20261027/full/20261027T125049826965Z-inference/`
6. **Llama 3 8B — P1 (`refined`):** `results/llama-refined-fullgpu-20261027/full/20261027T190835747818Z-inference/`

## Executar ou reproduzir

Em PowerShell, na raiz deste diretório:

```powershell
python prepare.py --offline
python download-wheels.py
python download-model.py --manifest configs/mistral-full-3168.json
.\run-docker.ps1 -Manifest configs/mistral-full-3168.json -Build -Preflight
.\run-docker.ps1 -Manifest configs/mistral-full-3168.json -Inference
```

Para executar uma variante específica do grid ou outro modelo, substitua o manifesto em `-Manifest` (por exemplo, `configs/mistral-grid-refined.json` ou `configs/llama-grid-original.json`).

Fonte do estudo base: [Sivakumar et al. (2025)](https://aclanthology.org/2025.findings-emnlp.1008/). Fonte do enquadramento de avaliação: [Gao e Kreiss (2025)](https://aclanthology.org/2025.emnlp-main.342/). Fonte dos dados: [corefBias](https://github.com/uclanlp/corefBias/tree/0bce984dd081bbc10b0622f326727a024c607895/WinoBias/wino).
