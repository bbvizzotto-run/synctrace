# SyncTrace — Runbook de Campanha Experimental após Liberação SKKU

## Finalidade e gatilho de início

Este runbook inicia em **T0**, definido como o momento em que há: (i) aprovação formal de acesso aos datasets pela SKKU/mantenedores, e (ii) o ambiente local com a **RTX 5070 Ti** configurado. Com a confirmação da indisponibilidade de GPU pela SKKU, a execução será realizada localmente. Nenhum resultado anterior a T0 deve ser reportado como resultado de benchmark. Smoke tests em dados sintéticos comprovam somente a integridade do software.

> **Regra de parada:** não iniciar treinamento oficial se faltar um dos três requisitos, se a licença proibir o uso pretendido, ou se a inspeção de dados não conseguir mapear cada clipe a uma identidade (`speaker_id`).

## Suposições de planejamento

| Recurso | Planejamento recomendado | Alternativa caso seja limitado |
|---|---|---|
| GPUs | 1 GPU NVIDIA RTX 5070 Ti (Local) | Execução sequencial otimizada para alto desempenho |
| Dados | FakeAVCeleb e LipSyncTIMIT extraídos em armazenamento local de leitura | Um dataset aprovado: executar apenas piloto e aguardar o segundo para comparações cross-dataset |
| Código | Commit `fce5045` ou sucessor validado | Executar testes e registrar o commit antes de qualquer job |
| Repetições | Sementes 42, 52 e 62 | Nunca reduzir sementes sem declarar a mudança no manuscrito |

Com quatro GPUs, a campanha deve ocupar aproximadamente **10–14 dias de calendário**, incluindo reexecuções. Com uma GPU, manter a mesma ordem e os mesmos controles, reservando **cerca de 3–4 semanas**. Estas são janelas operacionais, não promessas de duração: a duração de cada job deve ser recalibrada após o piloto da Fase 2.

## Cronograma operacional

| Janela relativa | Fase | Execução | Critério de aprovação | Entregável |
|---|---|---|---|---|
| T0 a T0+4 h | A. Conformidade e ambiente | Executar `scripts/check_local_env.py`, registrar aprovação, licença, GPU (RTX 5070 Ti), driver, CUDA, PyTorch e commit | Licença aceita; diagnóstico OK; testes passam | `run_metadata.json` e log de ambiente |
| T0+4 h a T0+12 h | B. Inventário e splits | Validar mídia/áudio, deduplicar IDs, gerar manifestos e splits 70/15/15 por `speaker_id` para sementes 42/52/62 | Nenhuma identidade atravessa splits; contagens e classes auditadas | CSVs de manifestos + hashes |
| T0+12 h a T0+2 d | C. Piloto | Uma semente por dataset, poucas épocas, com checkpoint e avaliador oficial | Sem NaN, leitura integral, perda reduz, latência registrada | relatório de piloto; orçamento de épocas ajustado |
| Dias 2–6 | D. Modelo completo | SyncTrace completo nas três sementes em cada dataset; selecionar somente por validação | Três checkpoints e avaliações de teste completas por dataset | métricas por semente e predições auditáveis |
| Dias 5–9 | E. Ablações | Executar `fixed_severity`, `no_mamba`, `dense_attention`, `no_sae` mantendo split, seed, preprocessing e orçamento | Cada variante conclui as três sementes ou registra falha reprodutível | matriz de ablações |
| Dias 7–10 | F. Baselines multimodais | Treinar/avaliar comparadores sob a mesma janela de 16 frames, log-mel e hardware | Mesmos manifestos, critérios e temporização | tabela de comparação justa |
| Dias 10–11 | G. Eficiência | Medir parâmetros, FLOPs, memória de pico e latência mediana/p95 após warm-up | Mesma GPU/CPU, versão de runtime, batch e protocolo de timing | `efficiency.json` e logs |
| Dias 11–12 | H. Auditoria | Reexecutar uma semente do modelo completo e uma ablação; calcular média±DP e diferenças pareadas | Reexecução coerente; sem vazamento; artefatos completos | checklist de reprodutibilidade |
| Dias 12–14 | I. Manuscrito | Preencher somente campos com artefatos aprovados; atualizar tabelas, figuras, discussão e limitações | Cada número aponta para manifesto, seed e log | pacote LaTeX e resultados para revisão |

## Matriz de paralelismo

| Grupo de jobs | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|---|---|---|---|---|
| Modelo principal | FakeAVCeleb, seed 42 | FakeAVCeleb, seed 52 | FakeAVCeleb, seed 62 | LipSyncTIMIT, seed 42 |
| Modelo principal (continuação) | LipSyncTIMIT, seed 52 | LipSyncTIMIT, seed 62 | avaliação/eficiência | reserva para reexecução |
| Ablações | `fixed_severity` | `no_mamba` | `dense_attention` | `no_sae` |

Em um único dispositivo, executar cada linha da matriz da esquerda para a direita. Não executar duas variantes simultaneamente no mesmo dispositivo se isso alterar a memória de pico, a frequência de clock ou a latência medida.

## Checkpoints obrigatórios por experimento

Cada job deve armazenar configuração resolvida, commit Git, seed, hash do manifesto, log por época, checkpoint selecionado, predições do teste, métricas por identidade e perfil de eficiência. O nome do diretório deve seguir:

```text
artifacts/official/<dataset>/<variant>/seed_<seed>/<timestamp>_<commit>/
```

Uma falha deve preservar o stderr, configuração e última época concluída; ela não deve ser silenciosamente removida do resumo final.

## Critérios de qualidade antes de preencher o artigo

| Aspecto | Exigência mínima |
|---|---|
| Qualidade de dados | Split por identidade validado e manifestos versionados |
| Estimativa principal | Três sementes por dataset com média±DP |
| Comparação | Mesmos inputs, splits, orçamento e hardware para todos os métodos |
| Ablação | Uma mudança arquitetural/objetiva por linha; sem alterar outros controles |
| Eficiência | Mediana e p95 após warm-up; memória e versões de software registradas |
| Generalização | Pelo menos uma avaliação cross-dataset, quando ambas as bases estiverem liberadas |
| Integridade editorial | Sem resultados de smoke tests ou estimativas de ARM apresentados como métricas de benchmark |

## Contingências

Se apenas LipSyncTIMIT for aprovado, executar Fases A--C e preparar o modelo principal desse dataset, mas aguardar FakeAVCeleb para publicar comparações cross-dataset. Se apenas FakeAVCeleb for aprovado, aplicar a mesma regra. Se ocorrer indisponibilidade de GPU superior a 24 horas, congelar os jobs, manter os checkpoints e reprogramar sem mudar sementes, manifests ou orçamento. Se um experimento falhar por OOM, reduzir o batch size e registrar a alteração antes da retomada; não misturar métricas de configurações distintas em uma mesma média.

## Gatilho para me acionar

Quando a SKKU enviar a aprovação, encaminhe a mensagem ou o link de download e informe o caminho de montagem na GPU. A primeira ação será executar a lista da Fase A e devolver um relatório de prontidão antes de iniciar qualquer treinamento oficial.
