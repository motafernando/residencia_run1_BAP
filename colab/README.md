# Execução pós-Gate 9 no Google Colab

## Escopo

Este runbook documenta a grade pós-Gate 9, os controles de execução e a
consolidação multi-seed. O notebook oficial é
`colab/run_colab_grid.ipynb`.

## Erros diagnosticados em 15/11/2026

1. O carregamento concluiu, mas os objetos atuais de Llama e Gemma não
   expuseram `hf_device_map`. O registro agora usa esse atributo quando
   existe e, em carga integral, registra o dispositivo efetivo do modelo.
2. A instalação sem lock aceitou versões incompatíveis de
   `huggingface-hub` e `protobuf`. O Colab agora instala versões exatas e
   o preflight confirma as versões efetivas.
3. O Llama 3.3 70B retornou `403 GatedRepoError`. O preflight baixa apenas
   `config.json` de cada modelo e encerra antes da grade quando falta acesso.
4. Células com `!python` não criavam uma barreira confiável entre smoke e
   grade. As etapas agora usam `subprocess.run(..., check=True)`.

## Acesso aos modelos

O secret `HF_TOKEN` precisa pertencer à mesma conta aprovada nas páginas:

- <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct>
- <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct>
- <https://huggingface.co/google/gemma-4-E4B-it>
- <https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512>
- <https://huggingface.co/Qwen/Qwen3-8B>

O token nunca é impresso nem salvo nos artefatos.

## Ordem de execução

1. Confirmar A100 80 GB com `nvidia-smi`.
2. Montar o Drive e carregar `HF_TOKEN`.
3. Clonar ou atualizar o repositório com avanço linear e registrar
   `REPO_COMMIT`.
4. Instalar `colab/requirements.txt`.
5. Executar `colab/preflight.py --block smoke`.
6. Executar o smoke em ordem crescente de custo.
7. Conferir os seis `run.json`, erros, VRAM de pico e formatos de resposta.
8. Executar os Blocos A, B e C em células separadas.
9. Consolidar somente depois de todas as runs previstas estarem completas.

Não use “Executar tudo” antes de o preflight e o smoke terminarem com sucesso.

## Matriz efetiva

- Smoke: 6 runs de 24 itens.
- Bloco A: 30 runs, dois modelos × três prompts × cinco seeds
  (`2026–2030`). A seed 2026 é repetida no Colab para manter as cinco
  repetições no mesmo ambiente.
- Bloco B: 63 runs.
- Bloco C: 9 runs do Llama 3.3 70B INT8.

O smoke é diagnóstico e não integra médias, desvios, intervalos ou taxas de
inversão.

## Persistência e retomada

Cada resposta é acrescentada a `responses.jsonl`, seguida de `flush` e
`fsync`. Estados `running` e `loading_model` podem ser retomados. Uma run
marcada `failed` começa em pasta nova, preservando a anterior como evidência.
Isso evita tentar retomar uma falha de carregamento com código, dependências ou
hashes alterados.

Runs completas repetidas com a mesma combinação de modelo, precisão, prompt e
seed fazem a consolidação falhar. A duplicata deve ser auditada e excluída do
conjunto de entrada de forma explícita.

## Proveniência registrada

Cada run preserva manifesto, amostra, hashes SHA-256 do código e entradas,
commit resolvido do modelo e tokenizer, hashes do checkpoint, versões de
pacotes, Python, CUDA e GPU, mapa de dispositivos, VRAM de pico, tempo, erros e
respostas brutas incrementais.

Modelos com revisão `main` registram o commit efetivamente resolvido. Antes de
uma publicação ou réplica posterior, os manifestos devem ser congelados nesse
commit.

## Consolidação

`analises/consolidar_multiseed.py`:

- ignora automaticamente runs de `smoke24.json`;
- exige uma única run completa por seed e grupo;
- informa média, desvio padrão e IC de 95% de Student entre seeds;
- informa A-SB conjunto e por Type 1/Type 2;
- informa contagens e taxas de `Unknown` e seleção nula;
- calcula inversão por item entre seeds;
- calcula inversão pareada BF16 × INT8 pela mesma seed, item e condição.

Os arquivos principais são `runs_summary.csv`,
`multiseed_summary.csv`, `flip_rates.csv` e, quando há pares de precisão,
`precision_flip_rates.csv`.

## Parser e revisão humana

O consolidado usa o parser v15 sem carregar silenciosamente o registro
contextual congelado do Gate 9. Esse registro é específico aos contextos
auditados anteriormente e não valida semanticamente saídas novas. Respostas
`rules_needs_review`, novos formatos, recusas e conflitos letra-texto devem
ser revisados antes de tratar as métricas como resultados finais.

Para reproduzir tabelas históricas do Gate 9, use `--use-registry` somente
com o registro congelado cuja integridade é verificada pelo SHA-256 definido em
`parser_v15.py`.

## Critérios para liberar a grade completa

- preflight concluído para todos os modelos do smoke;
- seis smokes completos e sem erros;
- nenhum padrão novo de resposta sem revisão;
- VRAM do Llama 70B abaixo do limite físico com margem operacional;
- artefatos presentes no Drive após reconexão;
- commit do repositório e relatório do preflight arquivados.



