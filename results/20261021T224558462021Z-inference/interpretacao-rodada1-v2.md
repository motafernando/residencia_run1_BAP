# Interpretação revisada da primeira rodada WinoBias

Esta leitura usa a rodada de 3.168 registros do Mistral 7B Instruct v0.3, em BF16, sem quantização e com uma semente (2026). O artigo relata cinco sementes para as condições com prompting. Portanto, os resultados abaixo são descritivos e provisórios; não representam a reprodução completa nem incluem testes de significância.

## Regra de leitura adotada

O artigo informa que RPA é calculado nas frases Type 2 e que a resposta de prompting é correta quando apenas o referente está presente no texto gerado. Também define O-SB como a diferença entre as taxas de geração de uma ocupação em contextos com pronome masculino e feminino, e calcula A-SB sobre todas as ocupações. Nos exemplos ambíguos Type 1, o artigo usa Unknown como resposta correta.

O artigo não descreve um algoritmo completo para separar uma resposta curta de sua explicação. Nesta versão, O-SB/A-SB usam a ocupação explicitamente nomeada na primeira linha não vazia; uma resposta explícita Unknown seleciona nenhuma ocupação. Para RPA Type 2, exige-se uma única ocupação candidata no texto completo, correspondente ao referente; uma resposta explícita Unknown não é acerto. As regras são convenções locais e permanecem identificadas como tais.

## Resultados descritivos

| Condição | RPA Type 2 | A-SB Type 1 | A-SB Type 2 |
| --- | ---: | ---: | ---: |
| Intrínseco | 85,04% (1.347/1.584) | 46,35% | 21,76% |
| Zero-shot | 94,13% (1.491/1.584) | 41,09% | 8,74% |
| 3-shot | 94,13% (1.491/1.584) | 40,79% | 8,37% |

Os cálculos separam Type 1 e Type 2 para permitir a leitura por ambiguidade. A-SB não deve ser interpretado como acurácia: Unknown em Type 1 pode ser uma resposta correta à tarefa, mas não seleciona uma ocupação. O denominador permanece com todos os registros elegíveis, conforme a cobertura de A-SB indicada no artigo.

## Efeito da revisão do parser

A versão anterior procurava ocupações e a palavra Unknown em toda a resposta. Isso misturava a opção escolhida com termos usados na explicação. A versão 2 registra separadamente a ocupação selecionada na primeira linha e as menções no texto completo usadas pelo critério estrito de RPA.

O RPA Type 2 não mudou: permanece em 94,13% nas condições zero-shot e 3-shot. O A-SB mudou pouco em relação ao cálculo anterior: no zero-shot, Type 1 passou de 41,15% para 41,09%, sem alteração em Type 2; no 3-shot, Type 1 passou de 41,04% para 40,79%, e Type 2 passou de 8,3684% para 8,3651% (8,37% nas duas versões com duas casas decimais). A classificação de respostas mudou mais do que os agregados, sobretudo por separar respostas explicitamente Unknown de menções a ocupações nas justificativas.

Com o parser v2, foram identificadas 15 respostas Unknown em Type 1 e 2 em Type 2 no zero-shot; no 3-shot, 152 em Type 1 e 23 em Type 2. Type 1 fica fora do RPA. Em Type 2, Unknown não é acerto de resolução.

## Correlação e limites

A correlação de Pearson por ocupação entre os escores intrínsecos e os de prompting, calculada nesta rodada com uma única semente, foi ρ = 0,9791 no zero-shot e ρ = 0,9795 no 3-shot para o conjunto completo. Separando Type 2, os valores foram 0,8779 e 0,8708. São estimativas descritivas sem p-valores; não devem ser comparadas diretamente às correlações agregadas sobre cinco sementes do artigo.

A base disponível contém 3.168 registros, enquanto o artigo informa 3.160. Os 89 registros sinalizados para revisão continuam incluídos no agregado principal; a análise sem esses registros está em `metrics.json`. Esta interpretação não resolve a diferença de contagem, não confirma que a revisão do checkpoint seja a usada pelas autoras e não substitui a inspeção manual das respostas sinalizadas.

## Rastreio

`responses.jsonl` preserva as respostas brutas da inferência. `responses.parsed-v2.jsonl` registra a leitura derivada; `metrics.pre-answer-parser-v2.json` preserva os agregados anteriores; `metrics.json` contém os agregados recalculados. O processamento foi offline, sem nova inferência. O script reexecutável é `recompute_saved_run.py`; a regra está em `protocol.py`.

Fonte primária: Sivakumar et al. (2025), seção 3.2, seção 4.2, Figura 5 e apêndice A: https://aclanthology.org/2025.findings-emnlp.1008.pdf.
