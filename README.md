# VoltLens — Classificação de Dígitos de Hidrômetros com AlexNet, VGG16 e ResNet

**Disciplina:** Visão Computacional — CESAR School, 2026.2  
**Integrantes:**   
· Helder Barros (hcbf@cesar.school)  
· Felipe Salustiano (fsm2@cesar.school)  
· Lucas Mendes (ljm@cesar.school)  
· Lucas Rafael (lrgms@cesar.school)  
· José Kevin (jkss@cesar.school)  

## Fonte dos dados e licença

Dataset: [Yandex.Toloka Water Meters](https://www.kaggle.com/datasets/tapakah68/yandextoloka-water-meters-dataset),
licença **CC BY-NC-ND 4.0**. Este repositório **não** contém imagens do dataset, recortes gerados nem checkpoints.
O uso é restrito a fins acadêmicos e não comerciais.

## Como reproduzir

```bash
python -m venv .venv
.venv\Scripts\activate                       # Linux/macOS: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # GPU NVIDIA (RTX 50xx exige CUDA 12.8)
pip install -r requirements.txt
python -m ipykernel install --user --name vc-p2-hidrometros
# Dados: o notebook procura data.csv em ../vc-20262-p2/data (ou na variável WATERMETERS_DIR).
# Se não encontrar, baixa pela API do Kaggle (kagglehub) -> precisa de ~/.kaggle/kaggle.json.
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 pratica_watermeters.ipynb
python tools/clear_image_outputs.py pratica_watermeters.ipynb   # limpa as saídas que exibem imagens do dataset
```

- Ambiente usado: local, Windows 11, Python 3.12.10, PyTorch 2.11.0+cu128, torchvision 0.26.0, GPU **NVIDIA GeForce RTX 5060 Ti (8 GB)**
- Tempo total de execução: ~15 min (10,5 min de treino dos quatro modelos)
- Split utilizado: `splits.csv` **oficial** fornecido pelo professor (não alterado)
- As células que exibem imagens do dataset têm a tag `dataset-images` e a saída limpa antes do commit.

| Arquivo | Conteúdo |
|---|---|
| `pratica_watermeters.ipynb` | notebook executado (Partes 1–5) |
| `splits.csv` | divisão oficial (cópia sem alteração) |
| `results.csv`, `history_*.csv` | métricas no teste e histórico de treino |
| `figures/` | 4 figuras obrigatórias + 2 da análise exploratória |
| `reports/` | `classification_report` de cada modelo, latência em CPU, orientação de cada faixa, tipos de erro |

## Protocolo de treino

| Modelo | Pesos | Entrada | Batch | LR cabeça | LR backbone | Épocas máx. | Melhor época |
|---|---|---|---|---|---|---|---|
| smallcnn | do zero | 64×48 | 256 | 1e-3 | — | 10 | 10 |
| alexnet | ImageNet (IMAGENET1K_V1) | 128×128 | 128 | 1e-3 | 1e-4 | 8 | 6 |
| vgg16_bn | ImageNet (IMAGENET1K_V1) | 128×128 | 32 | 1e-3 | 1e-5 | 8 | 7 |
| resnet18 | ImageNet (IMAGENET1K_V1) | 128×128 | 128 | 1e-3 | 1e-4 | 8 | 7 |

Comum a todos: `AdamW` (weight decay 0,01) + `OneCycleLR` (passo por *batch*, `max_lr` = LR de cada grupo);
`CrossEntropyLoss(label_smoothing=0.1)`; 1ª época com o backbone congelado (pré-treinados); *early stopping* pelo
F1-macro da validação, paciência 3 (nenhum modelo chegou a parar antes); `seed=42` com cuDNN determinístico;
*augmentation* `RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05))` +
`ColorJitter(brightness=0.3, contrast=0.3)`, sem flip; normalização com média/desvio do ImageNet.

Alterações em relação ao protocolo oficial (e justificativa): **nenhuma alteração de hiperparâmetro.** Decisões de
implementação, iguais para os quatro modelos:

- **Cabeça** = a última camada linear (substituída por `Linear(·, 10)`); **backbone** = todo o resto. Na `smallcnn`
  todos os parâmetros usam LR 1e-3.
- A *augmentation* é aplicada no recorte original 64×48; o redimensionamento bilinear para 128×128 e a normalização
  são feitos na GPU (mais rápido que na CPU no Windows, onde usamos `num_workers=0`).
- `seconds` no histórico = tempo da época de treino, **sem** a validação; `s_per_epoch` = mediana dessa coluna.
- `ms_per_image` = latência na GPU, batch 1, entrada já pré-processada, mediana de 200 execuções após 20 de *warm-up*.

## Resultados (conjunto de teste)

| model | acc_digit | f1_macro | acc_reading | params_M | s_per_epoch | ms_per_image | ckpt_MB | best_epoch |
|---|---|---|---|---|---|---|---|---|
| smallcnn | 0.8008 | 0.7522 | 0.2941 | 0.928 | 5.03 | 0.982 | 3.55 | 10 |
| alexnet | 0.9171 | 0.8968 | 0.6257 | 57.045 | 5.48 | 1.347 | 217.61 | 6 |
| vgg16_bn | 0.9372 | 0.9283 | 0.7112 | 134.31 | 29.19 | 3.325 | 512.41 | 7 |
| resnet18 | 0.9599 | 0.9536 | 0.8449 | 11.182 | 6.94 | 3.02 | 42.73 | 7 |

Latência adicional em **CPU** (batch 1, mesma metodologia; `reports/cpu_latency.csv`): smallcnn 2,42 ms ·
alexnet 9,94 ms · vgg16_bn 40,51 ms · resnet18 9,36 ms.

Figuras (pasta `figures/`):

- Matrizes de confusão normalizadas: `figures/confusion_matrices.png`
- Acurácia por posição: `figures/accuracy_by_position.png`
- Curvas de treino: `figures/training_curves.png`
- Desempenho × custo: `figures/performance_vs_cost.png`
- Extras da análise exploratória: `figures/eda_readings.png`, `figures/eda_digit_distribution.png`

## Respostas às questões

### 1.1 O que contém cada pasta e cada coluna do `data.csv`? Quais pastas servem para treinar um modelo?

O repositório tem três pastas principais e um arquivo CSV:

- `images/`: contém as fotos dos hidrômetros. São as imagens usadas como entrada do modelo.
- `masks/`: contém máscaras binárias que mostram a região do visor em cada foto. Elas podem ser usadas para treinar um modelo que localize ou segmente o visor.
- `collage/`: contém as fotos com as máscaras desenhadas por cima, para facilitar a conferência visual.
- `data.csv`: reúne as informações de cada foto e tem três colunas:
  - `photo_name`: nome do arquivo da imagem.
  - `value`: valor da leitura do hidrômetro, em metros cúbicos, guardado como número decimal.
  - `location`: coordenadas normalizadas do polígono que marca a região do visor, com valores entre 0 e 1.

Para treinar o classificador de dígitos, usamos as imagens de `images/` e as informações de `data.csv`. O valor da leitura serve para montar os rótulos, e a coluna `location` indica onde recortar o visor. As máscaras também podem ajudar a treinar um modelo de segmentação. Já a pasta `collage/` serve principalmente para conferir se as marcações estão corretas.

O conjunto tem 1.244 fotos, mas uma foi descartada porque apresentava quatro casas decimais e não seguia o padrão esperado. Por isso, o pipeline utilizou 1.243 imagens válidas.

### 1.2 O rótulo disponível é por imagem ou por dígito? O que isso implica para a tarefa?

O rótulo está disponível por imagem, porque cada foto tem a leitura completa do hidrômetro, e não uma marcação individual para cada dígito.

Para treinar o classificador, o pipeline transforma a leitura em oito caracteres e divide a faixa do visor em oito recortes iguais. Cada caractere é associado ao recorte da mesma posição. Portanto, o processo presume que cada fatia contém o dígito esperado, mas essa correspondência não foi anotada diretamente.

Isso pode causar problemas quando o recorte fica desalinhado: o modelo pode receber um rótulo errado no treinamento ou ser considerado errado na avaliação mesmo quando reconheceu corretamente o que aparece na imagem. Ao todo, 1.243 imagens geraram 9.944 recortes.

Uma alternativa seria usar um modelo de sequência, como uma CRNN com CTC, que tenta ler a faixa inteira sem precisar separar os dígitos antes.

### 2.1 Qual é a fonte mais confiável para o rótulo de cada imagem? Justifique com evidências.

O nome do arquivo é a fonte mais confiável para criar os rótulos. Os 1.244 registros coincidem com a coluna `value`, e o texto evita possíveis diferenças causadas pela conversão para `float`. Também conferimos manualmente seis imagens para verificar a quantidade de casas decimais. Mesmo assim, consideramos importante conferir o visor nos casos de dúvida.

### 2.2 Qual dígito deve ser a classe majoritária? Por quê?

O dígito `0` é a classe majoritária. Ele representa 35,7% de todos os dígitos e 34,4% dos recortes de teste, ou seja, 515 dos 1.496 recortes.

Isso acontece principalmente porque 95,9% das leituras têm de um a três dígitos na parte inteira, embora o formato usado no dataset reserve cinco posições para essa parte. Assim, várias leituras recebem zeros à esquerda para completar as posições. Além disso, 128 leituras têm apenas uma ou duas casas decimais registradas e recebem zeros à direita para completar os três dígitos decimais esperados.

Esse desequilíbrio pode fazer a acurácia parecer boa mesmo quando o modelo tem dificuldade com outros números. Por isso, também usamos o F1-macro, que dá o mesmo peso ao desempenho de cada uma das dez classes.

### 3.1 Por que a divisão é feita por imagem e não por recorte? Por que todos os grupos usam o mesmo `splits.csv`?

A divisão é feita por imagem porque os oito recortes de uma mesma foto compartilham características como iluminação, câmera, foco, ângulo e possíveis erros de alinhamento.

Se alguns recortes da mesma foto fossem usados no treinamento e outros no teste, o modelo poderia encontrar no teste imagens muito parecidas com as que já viu. Isso seria vazamento de dados (*data leakage*) e poderia deixar os resultados artificialmente altos.

Todos os integrantes usam o mesmo `splits.csv` para que os modelos sejam avaliados com as mesmas imagens e os resultados possam ser comparados de forma justa. A divisão definida pelo professor também evita que cada grupo escolha, sem querer, um conjunto de teste mais fácil. Isso é importante porque o conjunto de teste tem apenas 187 imagens e a margem de incerteza aproximada da `acc_reading` fica entre ±5 e ±7 pontos percentuais. Portanto, resultados de divisões diferentes poderiam variar o suficiente para dificultar a comparação entre os modelos.

A divisão tem 870 imagens para treinamento, 186 para validação e 187 para teste. Os oito recortes de cada imagem ficam sempre no mesmo grupo.

### 3.2 Quantas faixas foram corrigidas? Identifique e quantifique os tipos de recorte errado que o pipeline produz.

Foram corrigidas 120 faixas: 105 giradas 90 graus e 15 giradas 180 graus. Além disso, 34 das faixas verticais também precisaram de rotação de 180 graus, totalizando 49 faixas com esse giro.

No teste, a ResNet18 apresentou 60 erros, divididos em quatro categorias:

- **Outros:** 24 erros.
- **Faixas com quatro ou mais dígitos errados:** 14 erros.
- **Faixas de cabeça para baixo:** 13 erros.
- **Tambores em transição:** 9 erros.

Além disso, o diagnóstico apontou 13 faixas possivelmente invertidas nos conjuntos de treinamento, validação e teste, sendo oito, três e duas, respectivamente. Como cada faixa gera oito dígitos, até 104 recortes podem ser afetados. Não contamos o total de casos de perspectiva extrema ou desalinhamento, mas esses problemas também podem prejudicar a leitura.

### 3.3 Que erro a divisão da faixa em larguras iguais introduz? Como você o corrigiria?

Dividir a faixa em oito recortes de 48 pixels pressupõe que todos os dígitos têm a mesma largura e que a região do visor começa e termina exatamente nos limites dos oito dígitos. Na prática, existem margens, separadores entre a parte inteira e a decimal, diferenças de perspectiva e dígitos que parecem ter larguras diferentes.

Por isso, um pequeno erro no começo da faixa pode aumentar ao longo dos recortes. Um dígito pode acabar dividido entre duas fatias, ou a fatia pode conter parte da moldura. Mesmo assim, o rótulo continua sendo atribuído de acordo com a posição original, gerando ruído nos dados.

A posição 7 teve a menor acurácia em todos os modelos. Na ResNet18, ela foi de 91,4%; na smallcnn, apenas 51,9%. Isso é um sinal de que essa posição é mais difícil, mas não prova que o desalinhamento seja a única causa. Também podem influenciar os tambores em transição e os zeros inferidos no final de algumas leituras.

Algumas formas de melhorar o processo seriam procurar os espaços entre os dígitos usando o perfil de projeção vertical, identificar contornos depois de binarizar a imagem, usar os vértices do polígono como referência para separar os dígitos inteiros dos decimais ou fazer recortes com uma pequena sobreposição. Outra opção seria usar uma CRNN com CTC para ler a faixa toda sem dividi-la em oito partes fixas.

### 4.1 Por que normalizar as imagens com a média e o desvio padrão do ImageNet?

As redes pré-treinadas foram treinadas com imagens normalizadas usando a média e o desvio padrão do ImageNet. Usar a mesma normalização ajuda a manter os dados em uma escala parecida com aquela que as redes já conhecem.

Se a entrada tiver uma distribuição muito diferente, as ativações das camadas também podem mudar, dificultando o aproveitamento dos pesos que o modelo aprendeu no pré-treinamento.

A smallcnn foi treinada do zero e não depende diretamente dessa normalização. Mesmo assim, o mesmo pré-processamento foi aplicado a todos para manter um padrão de comparação.

Isso não elimina todas as diferenças entre os dados: as fotos de hidrômetros são bem diferentes das imagens naturais do ImageNet. Além disso, os recortes de 64 × 48 pixels são redimensionados para 128 × 128, o que estica os dígitos e altera suas proporções.

### 4.2 Compare o número de parâmetros dos quatro modelos. Onde se concentram os parâmetros da AlexNet e da VGG16?

O número de parâmetros indica quantos valores existem nos pesos que o modelo aprende. A comparação ficou assim:

| Modelo | Número de parâmetros | Tamanho do checkpoint |
|---|---:|---:|
| smallcnn | 0,928 milhão | 3,55 MiB |
| AlexNet | 57,045 milhões | 217,61 MiB |
| VGG16-BN | 134,31 milhões | 512,41 MiB |
| ResNet18 | 11,182 milhões | 42,73 MiB |

Na AlexNet, 95,7% dos parâmetros estão nas camadas totalmente conectadas, que somam cerca de 54,58 milhões. Na VGG16-BN, 89% estão nessas camadas, totalizando 119,59 milhões. Só a primeira camada totalmente conectada da VGG16-BN possui aproximadamente 102,8 milhões de parâmetros.

Isso acontece porque essas redes transformam os mapas de características em um vetor e conectam esse vetor a camadas grandes, com milhares de neurônios.

A ResNet18 tem uma estrutura mais econômica: usa *global average pooling* e uma camada final pequena, então a maior parte de seus parâmetros fica nas convoluções. Mesmo a smallcnn, que é bem menor, tem a maioria dos parâmetros na camada totalmente conectada.

### 4.3 Qual a diferença entre treinar só a cabeça e fazer *fine-tuning* completo? Por que congelar o backbone apenas na primeira época?

A cabeça é a camada final que transforma as características extraídas em uma classificação. O *backbone* é o restante da rede, que identifica características da imagem.

Quando treinamos só a cabeça, o backbone fica com os pesos pré-treinados congelados. Isso é mais simples e preserva as características que a rede já aprendeu, mas limita o quanto ela consegue se adaptar aos dígitos dos hidrômetros.

No *fine-tuning* completo, todas as camadas podem ajustar seus pesos para o novo problema. Isso permite uma adaptação maior, mas também exige cuidado para não alterar demais o que a rede aprendeu anteriormente.

O backbone ficou congelado apenas na primeira época para dar à cabeça, que começa com pesos aleatórios, uma chance de aprender a classificar usando as características existentes. Depois, a rede inteira foi liberada para o ajuste.

Da primeira para a segunda época, a acurácia de validação passou de 51,5% para 80,6% na AlexNet, de 52,6% para 82,6% na VGG16-BN e de 36,2% para 86,4% na ResNet18.

Mesmo assim, não dá para afirmar que todo esse ganho veio apenas do *fine-tuning*. A taxa de aprendizado também aumentou nesse período, e nos modelos com BatchNorm (VGG16-BN e ResNet18) as médias móveis dessas camadas continuaram sendo atualizadas enquanto o backbone estava congelado.

### 4.4 Por que não usar *flip* neste problema?

O *flip* pode criar imagens que não correspondem ao que aparece de verdade nos hidrômetros. Um espelhamento horizontal altera a forma de vários dígitos e cria versões que não são comuns em um visor real.

Uma rotação de 180 graus, que equivale a combinar os espelhamentos horizontal e vertical, pode fazer dígitos como 6 e 9 parecerem um com o outro. Se a imagem for transformada, mas o rótulo continuar igual, o modelo aprende uma associação incorreta.

Como a orientação correta faz parte do problema, o pipeline já possui uma etapa para detectar e corrigir faixas invertidas. Usar *flip* poderia desfazer esse trabalho. Por isso, foram usadas outras formas de aumento de dados, como pequenas rotações, translações e mudanças de cor, que simulam variações reais das fotos sem alterar intencionalmente a identidade dos dígitos.

### 5.1 Qual modelo teve o melhor F1-macro? O ranking é o mesmo da acurácia? Por quê?

A ResNet18 teve o melhor F1-macro (0,9536), seguida pela VGG16-BN (0,9283), AlexNet (0,8968) e smallcnn (0,7522). A ordem foi a mesma na acurácia por dígito.

A acurácia pode ser favorecida pelo zero, que é a classe mais frequente. Já o F1-macro dá o mesmo peso para todas as classes. A diferença entre as duas métricas foi de 4,9 pontos percentuais na smallcnn e apenas 0,6 na ResNet18, indicando que a ResNet18 teve um desempenho mais equilibrado entre os dígitos.

### 5.2 Quais pares de classes mais se confundem? Por quê?

Os pares com mais confusões entre os quatro modelos foram:

| Dígito verdadeiro → previsto | Erros |
|---|---:|
| 1 → 0 | 35 |
| 6 → 0 | 23 |
| 8 → 6 | 18 |
| 0 → 1 | 17 |
| 8 → 0 | 17 |
| 4 → 7 | 16 |
| 2 → 0 | 16 |
| 8 → 9 | 16 |

O zero aparece bastante entre os erros, provavelmente porque é a classe mais frequente no dataset. Além disso, alguns dígitos têm formatos parecidos, como 0, 6, 8 e 9. Fotos borradas, recortes desalinhados e tambores em transição também podem dificultar a identificação correta.

### 5.3 Por que `acc_reading` é muito menor que `acc_digit`?

A `acc_digit` mede quantos recortes foram classificados corretamente. Já a `acc_reading` só considera a leitura correta quando os oito dígitos de uma imagem estão certos ao mesmo tempo.

Por isso, mesmo com uma acurácia por dígito alta, basta um erro para que a leitura completa seja considerada errada. Para a ResNet18, se os erros fossem independentes, a estimativa seria:

`0,9599^8 ≈ 0,721`

Ou seja, cerca de 72,1% das leituras completas estariam certas nessa hipótese. Na avaliação real, a ResNet18 conseguiu 84,49% de `acc_reading`, porque os erros não se distribuem igualmente entre todas as imagens. Muitas leituras estão totalmente corretas, enquanto alguns casos mais difíceis concentram vários erros.

Das 187 imagens do teste, a ResNet18 acertou os oito dígitos em 158. Outras 18 tiveram apenas um dígito errado, enquanto 11 concentraram de dois a sete erros.

Isso mostra que, para melhorar a leitura completa, pode valer mais a pena corrigir problemas de orientação e recorte nas imagens difíceis do que tentar ganhar alguns décimos na acurácia por dígito.

### 5.4 Qual modelo você colocaria em um aplicativo de celular? Justifique com números.

Eu escolheria a ResNet18 porque ela apresentou o melhor equilíbrio entre qualidade da leitura, tamanho do modelo e velocidade em CPU.

Ela obteve F1-macro de 0,9536 e acertou 84,49% das leituras completas. Tem cerca de 11,18 milhões de parâmetros e um checkpoint de 42,73 MiB. A latência medida em CPU foi de 9,36 ms por imagem, o que daria aproximadamente 75 ms para processar oito dígitos em sequência.

A AlexNet tem latência parecida, de 9,94 ms por imagem, mas ocupa 217,61 MiB e acerta 62,57% das leituras completas. A VGG16-BN é ainda maior, com 512,41 MiB, e leva 40,51 ms por imagem na CPU.

A smallcnn é a menor e mais rápida, com 3,55 MiB e 2,42 ms por imagem, mas sua `acc_reading` é de apenas 29,41%. Ou seja, ela não reconheceu corretamente cerca de sete em cada dez leituras completas.

Há uma ressalva: os tempos foram medidos em uma CPU de desktop, não em um celular. Além disso, o sistema real precisaria localizar o visor automaticamente, enquanto neste experimento essa região já foi indicada por um polígono anotado.

### 5.5 O que a ResNet traz em relação à VGG16 neste problema?

A ResNet18 apresentou resultados melhores usando menos recursos que a VGG16-BN. Seu F1-macro foi de 0,9536, contra 0,9283, e acertou 84,49% das leituras completas, enquanto a VGG16-BN acertou 71,12%.

Além disso, a ResNet18 tem cerca de 12 vezes menos parâmetros, ocupa 42,73 MiB e leva 9,36 ms por imagem na CPU. A VGG16-BN ocupa 512,41 MiB e leva 40,51 ms.

Uma vantagem da ResNet18 são as conexões residuais, que facilitam o treinamento de redes profundas. Na nossa avaliação, ela foi a melhor opção por combinar maior precisão com menor custo computacional. Porém, as taxas de aprendizado e os tamanhos de lote diferem entre os modelos, como prevê o protocolo, e usamos apenas uma semente, então não podemos atribuir toda a diferença apenas à arquitetura.

### 5.6 A smallcnn foi competitiva com as redes pré-treinadas? O que isso indica?

Não. A smallcnn teve F1-macro de 0,7522 e acertou 29,41% das leituras completas, enquanto a ResNet18 acertou 84,49%. Apesar de ser menor e mais rápida, a smallcnn apresentou resultados bem inferiores.

Isso indica que, mesmo com poucos dados, partir de pesos pré-treinados no ImageNet ajudou bastante no reconhecimento dos dígitos dos hidrômetros. Testar mais épocas na smallcnn também seria interessante, pois seu resultado ainda melhorava no final do treinamento.

## Limitações

- **Erro de dados × erro de modelo:** dos 60 dígitos errados pela ResNet18 no teste, cerca de 27 (45 %) vêm de recortes
  ruins (13 de faixas de cabeça para baixo + 14 de faixas desalinhadas). Outros 9 (15 %) são tambores em transição,
  em que o próprio rótulo é ambíguo. Só cerca de 40 % parecem erros genuínos do modelo (borrão, reflexo, forma).
  Melhorar o pipeline de recorte provavelmente rende mais que trocar de arquitetura.
- **Orientação:** a heurística dos decimais vermelhos falha quando o visor não tem vermelho (363 faixas). O
  classificador de orientação treinado com pseudo-rótulos ainda deixa cerca de 13 faixas invertidas (1 %), quase
  todas verticais.
- **Divisão em larguras iguais** (Questão 3.3): a principal fonte de desalinhamento, sobretudo na posição 7.
- **Rótulos:** a leitura vem de anotação humana; tambores entre dois números têm rótulo ambíguo e o zero final dos
  decimais foi inferido (o float o perde).
- **Avaliação:** o teste tem só 187 imagens (1496 recortes), então diferenças de ~1 p.p. em `acc_digit` estão dentro do
  ruído. Rodamos uma única seed; os tempos dependem da GPU (RTX 5060 Ti) e não incluem o pré-processamento.
- **O que mudaríamos:** segmentar os tambores pelo perfil de projeção, usar o rótulo de orientação por consistência
  entre modelos, treinar a smallcnn por mais épocas e testar um modelo de sequência (CRNN + CTC) sobre a faixa
  inteira.

## Declaração de uso de IA

### Identificação da atividade

| Campo | Valor |
|---|---|
| Disciplina | Visão Computacional — VC001 |
| Data | 13 / 10 / 2026 |
| Alunos(as) e matrícula | Helder Barros — 2025100387 · Felipe Salustiano — 2025100192 · Lucas Mendes — 2025100323 · Lucas Rafael — 2025100517 · José Kevin — 2025100196 |
| Atividade | Prática II — Classificação de Dígitos de Hidrômetros com AlexNet, VGG16 e ResNet |

### Seção A — Uso de LLM / assistente de IA

| Campo | Valor |
|---|---|
| Utilizou IA? | ☒ Sim ☐ Não |
| Modelo(s) usado(s) | Claude Opus 5.5 (`claude-opus-5-5`, Anthropic), via extensão Claude Code para VS Code |
| Prompt(s) utilizado(s) | Ver lista abaixo (prompts exatos, na ordem em que foram enviados) |
| Resposta gerada | Ver descrição abaixo |
| Adaptações realizadas | Ver descrição abaixo |

#### Prompts utilizados (texto exato)

1. `@vc-20262-p2/README.md guie em como executar oq tem nesse readme. Crie uma nova pasta para o projeto dentro da pasta "VC", separada da pasta "vc-20262-p2". utilize os dados de @vc-20262-p2/data/`
2. `oq está faltando?`
3. `n seria interessante se eu mesmo rodar o notebook?`
4. `como eu sei como está o andamento da sua?`
5. `pode me dizer como faço para acompanhar?`
6. (colado do terminal) `while ($true) { Clear-Host; ... }` — `'while' não é reconhecido como um comando interno ou externo, um programa operável ou um arquivo em lotes.`
7. `Quais são os arquivos que eu tenho que colocar no repositório do git?`
8. `a maneira como o projeto está estruturado, é a melhor? o history na raiz do projeto, o notebbok na raiz também. acho q está desorganizado. poderia reorganizar? se reorganizar alguma coisa vai mudar?`
9. `oq preciso fazer agr?`

#### Resposta gerada e contribuições da IA

A IA foi utilizada como ferramenta de apoio durante o desenvolvimento do projeto, auxiliando na compreensão das etapas do trabalho, na identificação de pendências, na organização dos arquivos e na resolução de dúvidas relacionadas à execução e à publicação do projeto.

Entre as contribuições da IA, destacam-se:

- **Orientação sobre o desenvolvimento:** auxílio na interpretação das instruções do README original e na identificação de etapas necessárias para atender aos requisitos do trabalho.
- **Acompanhamento da execução:** esclarecimentos sobre como acompanhar o andamento da execução do notebook e interpretar problemas relacionados ao ambiente de desenvolvimento.
- **Organização dos arquivos:** sugestões sobre a estrutura do projeto, os arquivos que deveriam ser versionados e os cuidados necessários ao reorganizar diretórios.
- **Apoio à documentação:** auxílio na elaboração e organização do README, incluindo orientações para apresentar o protocolo experimental, os resultados e as limitações do projeto.
- **Declaração de uso de IA:** auxílio no preenchimento do modelo fornecido pelo professor, com base nas informações sobre a utilização da ferramenta durante o trabalho.
- **Esclarecimento de dúvidas técnicas:** orientações para compreender erros, avaliar alternativas e decidir quais etapas deveriam ser executadas ou revisadas pelo grupo.

Essas contribuições não significam que a IA tenha desenvolvido integralmente o projeto. O trabalho envolveu atividades realizadas pelo grupo, que permaneceu responsável pelas decisões técnicas, pela revisão do conteúdo e pela entrega final.

#### Adaptações, validação e revisão realizadas pelo grupo

- O grupo manteve a responsabilidade pelo desenvolvimento e pela condução do trabalho, utilizando as sugestões da IA como apoio, e não como substituição das atividades realizadas pelos integrantes.
- A estrutura do projeto foi analisada com auxílio da IA. Após essa discussão, o grupo decidiu manter os entregáveis na raiz, conforme as orientações do roteiro.
- O grupo optou por não executar novamente o notebook, a fim de preservar os resultados já obtidos.
- Os resultados apresentados na entrega foram conferidos com o arquivo `results.csv` versionado. As verificações do roteiro registradas no notebook incluem 1243 imagens válidas, 1 imagem descartada, 9944 recortes e a verificação de que nenhuma imagem aparece em mais de um conjunto de dados.
- As respostas às questões, a documentação e a seção de limitações foram revisadas pelo grupo, que ficou responsável por verificar a coerência das informações e adequá-las ao trabalho realizado.
- O conteúdo produzido com auxílio da IA foi utilizado como referência durante a construção da entrega, o grupo avaliou o conteúdo e o reescreveu quando necessário.

### Seção B — Uso de skill (Claude Code / SKILL.md)

| Campo | Valor |
|---|---|
| Utilizou skill? | ☒ Sim ☐ Não |
| Nome da skill | `dataviz` (skill embutida do Claude Code) |
| Conteúdo do SKILL.md | Anexado em [`docs/ia/SKILL_dataviz.md`](docs/ia/SKILL_dataviz.md). Foi usada só como guia de estilo das figuras (paleta categórica fixa por modelo, linhas finas, legenda e rótulos diretos). |
| Versão / hash | Versão empacotada no Claude Code 2.1.296; carregada em 13/10/2026 (o arquivo não fica salvo em disco, por isso não há hash) |

### Seção C — Uso de SDD (System Design Document)

| Campo | Valor |
|---|---|
| Utilizou SDD? | ☐ Sim ☒ Não |
| Arquivo SDD | — |
| Partes geradas por IA | — |

### Declaração de responsabilidade

Declaramos que:

- As informações prestadas nesta declaração são verdadeiras e completas.
- Compreendemos e somos responsáveis por todo o conteúdo entregue, incluindo as partes geradas com auxílio de IA.
- Estamos cientes de que a omissão do uso de IA configura conduta academicamente desonesta.
