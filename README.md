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

## O que vem do artigo e o que foi decidido nesta rodada

O artigo orienta a comparação entre avaliação intrínseca, zero-shot e 3-shot no WinoBias, define RPA para as frases Type 2 e apresenta O-SB e A-SB como medidas de viés. Seus exemplos incluem `Unknown` para casos ambíguos Type 1. O estudo relata cinco sementes nas condições com prompting. Esses pontos orientam a rodada, mas não determinam sozinhos todos os passos necessários para produzir os números.

As escolhas abaixo são **operacionalizações nossas**. Estão registradas para permitir auditoria e podem produzir resultados diferentes dos publicados. Não dispomos de uma implementação dos autores que confirme sua equivalência.

- **Dados:** usamos os oito arquivos `dev` e `test` do commit do corefBias indicado acima. Eles somam 3.168 frases, não as 3.160 informadas no artigo. Mantivemos todas, inclusive as 89 sinalizadas para revisão. Para transformar as anotações em entradas do experimento, `prepare.py` escolhe o primeiro pronome marcado e as duas primeiras ocupações distintas; conserva as demais menções para auditoria. O artigo não esclarece quais oito frases foram excluídas nem como tratar todos esses casos.
- **Checkpoint:** a revisão oficial fixada é uma candidata histórica, escolhida para esta execução. A identidade dos pesos, do tokenizer e da revisão usada pelas autoras não está confirmada pelo artigo. BF16 sem quantização e os limites de memória com offload foram escolhas para a máquina local, registradas em `configs/mistral-full-3168.json`.
- **Prompts e formato:** `protocol.py` constrói a pergunta, as alternativas e o prefixo intrínseco a partir de cada frase. Os três exemplos de contexto seguem os exemplos apresentados na Figura 5; a ordem das alternativas é sorteada de modo reprodutível por frase com a semente 2026. Na geração zero-shot e 3-shot, `run.py` aplica o chat template do tokenizer. Não confirmamos que o artigo tenha usado essa mesma renderização, ordem de alternativas ou tokenização.
- **Pontuação intrínseca e geração:** `run.py` soma as log-probabilidades dos tokens de cada ocupação candidata, sem dividir pelo comprimento. Para as respostas geradas, usamos decodificação greedy, limite de 48 tokens e uma semente. Esses detalhes são escolhas implementadas e registradas; não devem ser atribuídos às autoras sem confirmação.
- **Leitura das respostas:** o parser atual em `protocol.py` usa, para O-SB/A-SB, a ocupação nomeada na primeira linha não vazia. `Unknown` explícito ou uma linha sem ocupação única não seleciona ocupação. Uma resposta formada apenas por `A`, `B` ou `C` tem a letra registrada, mas a letra não vira automaticamente seleção para essas métricas. Para o RPA Type 2, exigimos que apenas uma ocupação candidata apareça no texto completo e corresponda ao referente correto. O artigo fornece o critério geral de RPA, mas não o algoritmo de extração para respostas com explicações; essa separação entre escolha e menções é nossa.
- **Cálculo das métricas:** em `protocol.py`, a taxa de seleção de cada ocupação usa como denominador as frases do gênero considerado em que ela aparece como candidata. O A-SB usa as 40 ocupações. Respostas sem ocupação selecionada permanecem nesses denominadores; falhas de execução são contadas à parte. `recompute_saved_run.py` calcula a correlação de Pearson por ocupação. Essas regras de contagem são declarações da implementação, não uma confirmação de todos os detalhes do cálculo das autoras.

O parser foi revisado **depois** da inferência. O recálculo em `recompute_saved_run.py` alterou apenas as leituras derivadas e as métricas: `responses.jsonl` continua com as saídas originais; `metrics.generated-original.json` e `metrics.pre-answer-parser-v2.json` preservam os agregados anteriores. Por isso, a versão de cada métrica precisa acompanhar qualquer comparação com o artigo.

Ainda faltam a revisão manual das 89 frases e de respostas selecionadas, a explicação da diferença entre 3.168 e 3.160, a confirmação dos detalhes do checkpoint e do prompting e as sementes adicionais. As correlações desta pasta são descritivas; não reproduzem os testes de significância publicados. **Esta rodada é uma aproximação documentada do experimento principal, não uma reprodução exata.**

## Ler os resultados

O arquivo `responses.jsonl` contém as saídas brutas e não foi alterado pelo recálculo. `responses.parsed-v2.jsonl` contém a interpretação derivada. `metrics.generated-original.json` e `metrics.pre-answer-parser-v2.json` preservam etapas anteriores; `metrics.json` contém os agregados usados no registro sintético. `run.json` guarda o ambiente, os hashes e as contagens da inferência. `protocol.used.py` e `run.used.py` são as cópias exatas dos arquivos carregados na execução, conferidas contra os hashes do manifesto. `protocol.py` e `run.py`, na raiz, são versões posteriores à revisão do parser.

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
