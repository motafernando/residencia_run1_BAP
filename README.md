# Rodada 1 do WinoBias com Mistral 7B Instruct v0.3

Este diretório reúne o código, os dados e as saídas da primeira rodada local do núcleo WinoBias de *Bias after Prompting* (Sivakumar et al., 2025). Foram avaliadas 3.168 frases nas condições intrínseca, zero-shot e 3-shot, totalizando 9.504 registros com uma semente. Os resultados são preliminares: o artigo relata cinco sementes nas condições com prompting, e algumas regras necessárias à implementação não estão especificadas integralmente na publicação.

## Organização

- `prepare.py`, `protocol.py` e `run.py`: preparação dos dados, prompts experimentais, pontuação e inferência.
- `recompute_saved_run.py` e `audit_unknown_parsing.py`: auditoria e recálculo offline das métricas, sem nova inferência.
- `configs/mistral-full-3168.json`: configuração efetivamente usada na rodada; os demais manifestos registram o piloto e o perfil genérico.
- `data/raw/corefBias/`: arquivos do WinoBias obtidos do commit `0bce984dd081bbc10b0622f326727a024c607895`, com a licença original.
- `data/prepared/`: inventário e registros normalizados. O inventário aponta 89 frases que requerem revisão por múltiplos pronomes ou terceira ocupação.
- `results/20261021T224558462021Z-inference/`: respostas, prompts enviados, manifesto, métricas e versões anteriores preservadas.
- `results/gpu-monitor-full3168.csv`: amostragem de utilização e memória da GPU durante a rodada.
- `docs/`: registro sintético da execução em formato Word.

O checkpoint candidato foi fixado no repositório oficial `mistralai/Mistral-7B-Instruct-v0.3`, revisão `0d4b76e1efeb5eb6f6b5e757c79870472e04bd3a`, em BF16 e sem quantização. Não há confirmação de que esse SHA seja o usado pelas autoras. O modelo não está incluído neste diretório: seus arquivos são obtidos do Hugging Face pelo script `download-model.py` e ficam no volume local `cache/`.

## Ler os resultados

O arquivo `responses.jsonl` contém as saídas brutas e não foi alterado pelo recálculo. `responses.parsed-v2.jsonl` contém a interpretação derivada. `metrics.generated-original.json` e `metrics.pre-answer-parser-v2.json` preservam etapas anteriores; `metrics.json` contém os agregados usados no registro sintético. `run.json` guarda o ambiente, os hashes e as contagens da inferência. `protocol.used.py` e `run.used.py` são as cópias exatas dos arquivos carregados na execução, conferidas contra os hashes do manifesto. `protocol.py` e `run.py`, na raiz, são versões posteriores à revisão do parser.

O artigo informa 3.160 frases; os oito arquivos do commit indicado contêm 3.168. A rodada conservou todas as frases, inclusive as 89 sinalizadas para revisão. O parser v2 separa a ocupação selecionada na primeira linha da resposta das menções no texto completo usadas para RPA. Essa regra é uma convenção local documentada, pois o artigo não fornece o algoritmo de extração para respostas com explicação.

O texto histórico `interpretacao-rodada1-v2.md` recebeu neste pacote apenas uma correção de arredondamento: a mudança de A-SB Type 2 no 3-shot é de 8,3684% para 8,3651%, isto é, 8,37% nas duas versões com duas casas decimais. Os arquivos de métricas e respostas foram copiados sem alteração.

## Preparar uma nova execução

O ambiente usa uma imagem OCI baseada em PyTorch/CUDA, fixada em `image-lock.json`, mais as versões de `requirements.txt`. O `Dockerfile` instala wheels locais. Antes do build, execute `download-wheels.py` para obtê-los e conferir os hashes registrados em `wheelhouse-manifest.json`. `cache/` e `wheelhouse/` ficam fora do Git.

Em PowerShell, na raiz deste diretório:

```powershell
python prepare.py --offline
python download-wheels.py
python download-model.py --manifest configs/mistral-full-3168.json
.\run-docker.ps1 -Manifest configs/mistral-full-3168.json -Build -Preflight
.\run-docker.ps1 -Manifest configs/mistral-full-3168.json -Inference
```

O download do checkpoint requer acesso ao repositório oficial. O script não grava `HF_TOKEN` nos manifestos. Os comandos de inferência criam um novo diretório em `results/`; não reescrevem a rodada preservada. Para recalcular o parser da rodada arquivada, use `recompute_saved_run.py` sobre uma cópia dos resultados, pois o script grava novos arquivos derivados no diretório informado.

Fonte do estudo: [Sivakumar et al. (2025)](https://aclanthology.org/2025.findings-emnlp.1008/). Fonte dos dados: [corefBias](https://github.com/uclanlp/corefBias/tree/0bce984dd081bbc10b0622f326727a024c607895/WinoBias/wino).
