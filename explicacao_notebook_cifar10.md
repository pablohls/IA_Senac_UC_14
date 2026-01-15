# Explicação Detalhada: Fine-tuning ViT no CIFAR-10

Este documento explica detalhadamente cada célula do notebook `cifar10_finetuning.ipynb`, que implementa fine-tuning de um Vision Transformer (ViT) pré-treinado no dataset CIFAR-10.

---

## Célula 1: Instalação de Bibliotecas

```python
!pip install torch torchvision
!pip install transformers datasets
!pip install transformers[torch]
!pip install matplotlib
!pip install scikit-learn
!pip install --upgrade transformers
```

**Objetivo:** Instala todas as dependências necessárias para o projeto.

**Bibliotecas:**
- `torch` e `torchvision`: Framework de deep learning e processamento de imagens
- `transformers` e `datasets`: Acesso a modelos pré-treinados e datasets do Hugging Face
- `matplotlib`: Visualização de imagens e gráficos
- `scikit-learn`: Métricas de avaliação (acurácia, matriz de confusão)

---

## Célula 2: Importações

```python
# PyTorch
import torch
import torchvision
from torchvision.transforms import Normalize, Resize, ToTensor, Compose

# For displaying images
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

# Loading dataset
from datasets import load_dataset

# Transformers
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer

# Matrix operations
import numpy as np

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
```

**Objetivo:** Importa todos os módulos necessários para processamento de imagens, treinamento de modelos e avaliação.

---

## Célula 3: Carregamento do Dataset CIFAR-10

```python
trainds, testds = load_dataset("cifar10", split=["train[:5000]","test[:1000]"])
splits = trainds.train_test_split(test_size=0.1)
trainds = splits['train']
valds = splits['test']
trainds, valds, testds
```

**Objetivo:** Carrega e divide o dataset CIFAR-10 em conjuntos de treino, validação e teste.

**Divisão dos dados:**
- **Treino:** 4.500 imagens (90% de 5.000)
- **Validação:** 500 imagens (10% de 5.000)
- **Teste:** 1.000 imagens

**Por que validação?** Para monitorar o desempenho durante o treinamento e evitar overfitting.

---

## Célula 4: Mapeamento de Labels (itos e stoi)

```python
itos = dict((k,v) for k,v in enumerate(trainds.features['label'].names))
stoi = dict((v,k) for k,v in enumerate(trainds.features['label'].names))
itos
```

### O que é `itos` (int to string)?

Dicionário que mapeia **índices numéricos para nomes de classes**.

**Estrutura:**
```python
{
    0: 'airplane',
    1: 'automobile', 
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}
```

### O que é `stoi` (string to int)?

Dicionário que mapeia **nomes de classes para índices numéricos** (inverso de itos).

**Estrutura:**
```python
{
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}
```

### Por que são necessários?

1. **Modelos trabalham com números:** O modelo precisa de índices (0-9) para processar
2. **Humanos entendem textos:** Quando visualizamos resultados, "airplane" é mais claro que "0"
3. **Conversão bidirecional:**
   - Use `itos` quando o modelo prediz `3` e você quer saber que é "cat"
   - Use `stoi` quando precisa converter "dog" para `5` para treinar

### Exemplo prático:
```python
# Modelo prevê classe 7
predicted_class = 7
print(itos[predicted_class])  # Output: 'horse'

# Converter nome para índice
class_name = 'bird'
class_index = stoi[class_name]  # Output: 2
```

---

## Célula 5: Visualização de uma Imagem

```python
index = 0
img, lab = trainds[index]['img'], itos[trainds[index]['label']]
print(lab)
img
```

**Objetivo:** Exibe uma amostra do dataset com seu rótulo para verificação visual.

**Exemplo de saída:**
```
airplane
[Imagem 32x32 de um avião]
```

---

## Célula 6: Configuração do Processador de Imagens

```python
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name) 

mu, sigma = processor.image_mean, processor.image_std
size = processor.size
```

**Objetivo:** Carrega o processador pré-treinado do ViT e extrai parâmetros de normalização.

### Por que extrair média e desvio padrão?

A extração da média (`mu`) e desvio padrão (`sigma`) é **crítica** para o sucesso do fine-tuning.

#### 1. Consistência com o Pré-treinamento

O modelo ViT foi treinado no ImageNet com normalização específica. Usar valores diferentes causa:
- Predições ruins ou aleatórias
- Convergência lenta ou instável
- Perda do conhecimento pré-treinado

#### 2. Valores padrão do ImageNet

```python
mu = [0.485, 0.456, 0.406]     # média por canal RGB
sigma = [0.229, 0.224, 0.225]  # desvio padrão por canal RGB
```

Estes valores foram calculados sobre milhões de imagens do ImageNet.

#### 3. Fórmula de normalização

Para cada pixel em cada canal (R, G, B):

$$\text{pixel\_normalizado} = \frac{\text{pixel\_original} - \mu}{\sigma}$$

Transforma valores de `[0, 255]` para aproximadamente `[-2, 2]`, centralizando em zero.

#### 4. Por que não usar média/desvio do CIFAR-10?

Se você calculasse valores específicos do CIFAR-10 e os usasse:

```python
# ❌ ERRADO - quebraria o transfer learning
mu_cifar = [0.491, 0.482, 0.446]
sigma_cifar = [0.247, 0.243, 0.261]
```

O modelo esperaria ver dados normalizados de uma forma, mas receberia de outra. É como treinar alguém para ler metros e depois dar medidas em pés!

#### 5. Transfer Learning depende disso

O poder do transfer learning vem de:
- Usar pesos pré-treinados ✓
- Processar dados da **mesma forma** que no treinamento original ✓

**Resumo:** Extrair `mu` e `sigma` do processador mantém compatibilidade com o treinamento original, garantindo que o transfer learning funcione.

---

## Célula 7: Definição de Transformações de Imagem

```python
norm = Normalize(mean=mu, std=sigma)

_transf = Compose([
    Resize(size['height']),
    ToTensor(),
    norm
]) 

def transf(arg):
    arg['pixels'] = [_transf(image.convert('RGB')) for image in arg['img']]
    return arg
```

**Objetivo:** Cria pipeline de pré-processamento para preparar imagens para o modelo.

### Pipeline de transformação:

1. **Resize:** 32×32 → 224×224 (tamanho esperado pelo ViT)
2. **ToTensor:** Converte PIL Image para tensor PyTorch
3. **Normalize:** Aplica normalização com `mu` e `sigma`

### Transformação visual:

```
Imagem Original (CIFAR-10)
├─ Formato: PIL Image
├─ Tamanho: 32×32 pixels
└─ Range: [0, 255]
           ↓
   [Resize to 224×224]
           ↓
   [ToTensor - converte para PyTorch]
           ↓
   [Normalize com mu e sigma]
           ↓
Imagem Processada
├─ Formato: torch.Tensor
├─ Tamanho: 3×224×224
└─ Range: aproximadamente [-2, 2]
```

---

## Célula 8: Aplicação das Transformações

```python
trainds.set_transform(transf)
valds.set_transform(transf)
testds.set_transform(transf)
```

**Objetivo:** Aplica as transformações definidas a todos os datasets (treino, validação e teste).

**Importante:** As transformações são aplicadas **on-the-fly** (em tempo real) quando as imagens são acessadas, economizando memória.

---

## Célula 9: Visualização de Imagem Transformada

```python
idx = 0
ex = trainds[idx]['pixels']
ex = (ex+1)/2  # Ajusta range de [-1,1] para [0,1]
exi = ToPILImage()(ex)
plt.imshow(exi)
plt.show()
```

**Objetivo:** Verifica visualmente se as transformações foram aplicadas corretamente.

**Nota:** `(ex+1)/2` converte o range de `[-1, 1]` para `[0, 1]` porque `plt.imshow()` requer valores nesse intervalo.

---

## Célula 10: Carregamento do Modelo Base

```python
model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name)
print(model.classifier)
```

**Objetivo:** Carrega o modelo ViT pré-treinado com a configuração original (1000 classes do ImageNet).

**Saída esperada:**
```
Linear(in_features=768, out_features=1000, bias=True)
```

O classificador tem 1000 saídas (ImageNet tem 1000 classes).

---

## Célula 11: Adaptação do Modelo para CIFAR-10

```python
model = ViTForImageClassification.from_pretrained(
    model_name, 
    num_labels=10, 
    ignore_mismatched_sizes=True, 
    id2label=itos, 
    label2id=stoi
)
print(model.classifier)
```

**Objetivo:** Ajusta o modelo para a tarefa específica com 10 classes do CIFAR-10.

**Parâmetros importantes:**
- `num_labels=10`: CIFAR-10 tem 10 classes (não 1000)
- `ignore_mismatched_sizes=True`: Permite substituir a camada final
- `id2label=itos`: Mapeamento de índices para nomes
- `label2id=stoi`: Mapeamento de nomes para índices

**Saída esperada:**
```
Linear(in_features=768, out_features=10, bias=True)
```

Agora o classificador tem apenas 10 saídas!

**O que aconteceu:**
- A camada de classificação original (768→1000) foi **substituída**
- Nova camada (768→10) foi **inicializada aleatoriamente**
- Todas as outras camadas mantêm os **pesos pré-treinados**

---

## Célula 12: Configuração de Argumentos de Treinamento

```python
args = TrainingArguments(
    f"test-cifar-10",
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.04,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='logs',
    remove_unused_columns=False,
)
```

**Objetivo:** Define os hiperparâmetros para o treinamento.

### Parâmetros explicados:

| Parâmetro | Valor | Significado |
|-----------|-------|-------------|
| `save_strategy` | "epoch" | Salva checkpoint após cada época |
| `eval_strategy` | "epoch" | Avalia no conjunto de validação após cada época |
| `learning_rate` | 2e-5 | Taxa de aprendizado (0.00002) - baixa para fine-tuning |
| `per_device_train_batch_size` | 10 | 10 imagens por batch durante treino |
| `per_device_eval_batch_size` | 4 | 4 imagens por batch durante avaliação |
| `num_train_epochs` | 3 | Treina por 3 épocas completas |
| `weight_decay` | 0.04 | Regularização L2 para prevenir overfitting |
| `load_best_model_at_end` | True | Carrega o melhor modelo ao final |
| `metric_for_best_model` | "accuracy" | Usa acurácia para determinar melhor modelo |
| `logging_dir` | 'logs' | Diretório para logs do TensorBoard |
| `remove_unused_columns` | False | Mantém todas as colunas do dataset |

**Por que learning rate baixo?** Em fine-tuning, queremos fazer ajustes sutis nos pesos pré-treinados, não reaprender do zero.

---

## Célula 13: Funções Auxiliares (collate_fn e compute_metrics)

```python
def collate_fn(examples):
    pixels = torch.stack([example["pixels"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixels, "labels": labels}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return dict(accuracy=accuracy_score(predictions, labels))
```

### 1. `collate_fn` - Função de Collation (Agrupamento)

**Importância:**
- **Agrupa exemplos individuais em batches:** Organiza múltiplas amostras em tensores
- **Compatibilidade:** Renomeia `"pixels"` para `"pixel_values"` (nome esperado pelo ViT)
- **Otimização:** Permite processar 10 imagens simultaneamente

#### Transformação visual:

```
Entrada: Lista de 10 exemplos
┌─────────────────────────────┐
│ {"pixels": tensor[3,224,224], "label": 3} │
│ {"pixels": tensor[3,224,224], "label": 7} │
│ {"pixels": tensor[3,224,224], "label": 0} │
│ ...                                        │
│ {"pixels": tensor[3,224,224], "label": 2} │
└─────────────────────────────┘
            ↓ collate_fn
Saída: Batch unificado
┌─────────────────────────────────────┐
│ {                                   │
│   "pixel_values": tensor[10,3,224,224], │
│   "labels": tensor[10]              │
│ }                                   │
└─────────────────────────────────────┘
```

**Sem essa função:** O Trainer não saberia como combinar múltiplas amostras corretamente!

### 2. `compute_metrics` - Função de Avaliação

**Importância:**
- **Monitoramento:** Calcula acurácia após cada época
- **Seleção do melhor modelo:** Usa essa métrica para salvar o melhor checkpoint
- **Conversão:** Transforma logits em predições de classe

#### Como funciona:

```python
# Passo 1: Modelo retorna probabilidades/logits para 10 classes
predictions = [
    [0.1, 0.05, 0.8, 0.01, 0.02, 0.01, 0.00, 0.01, 0.00, 0.00],  # classe 2
    [0.05, 0.6, 0.05, 0.15, 0.05, 0.05, 0.02, 0.01, 0.01, 0.01]   # classe 1
]

# Passo 2: argmax encontra índice com maior valor
predictions = [2, 1]

# Passo 3: Compara com labels verdadeiros
labels = [2, 1]  # correto!

# Passo 4: Calcula e retorna acurácia
accuracy = 2/2 = 1.0 (100%)
```

### Por que são Críticas?

**Sem `collate_fn`:**
- ❌ Treinamento falharia ou seria extremamente lento
- ❌ Incompatibilidade de nomes causaria erros

**Sem `compute_metrics`:**
- ❌ Treinamento "às cegas" sem métricas interpretáveis
- ❌ Não conseguiria usar `load_best_model_at_end=True`
- ❌ Só veria a loss, sem saber a acurácia real

---

## Célula 14: Configuração do Trainer

```python
trainer = Trainer(
    model,
    args, 
    train_dataset=trainds,
    eval_dataset=valds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    tokenizer=processor,
)
```

**Objetivo:** Cria o objeto Trainer do Hugging Face que gerencia todo o processo de treinamento.

**Componentes:**
- `model`: O ViT adaptado para CIFAR-10
- `args`: Hiperparâmetros definidos na Célula 12
- `train_dataset`: 4.500 imagens de treino
- `eval_dataset`: 500 imagens de validação
- `data_collator`: Função para criar batches
- `compute_metrics`: Função para calcular acurácia
- `tokenizer`: Processador de imagens (usado para salvar configuração)

---

## Célula 15: Treinamento do Modelo

```python
trainer.train()
```

**Objetivo:** Executa o fine-tuning do modelo nos dados de CIFAR-10.

**O que acontece:**
1. Treina por 3 épocas
2. Após cada época:
   - Salva checkpoint
   - Avalia no conjunto de validação
   - Calcula e exibe acurácia
3. Ao final, carrega o melhor modelo baseado em acurácia

**Saída esperada:**
```
Epoch | Training Loss | Validation Loss | Accuracy
------|---------------|-----------------|----------
  1   |    0.5234     |     0.4123      |  0.8760
  2   |    0.3421     |     0.3234      |  0.9020
  3   |    0.2156     |     0.2987      |  0.9180
```

---

## Célula 16: Avaliação no Conjunto de Teste

```python
outputs = trainer.predict(testds)
print(outputs.metrics)
```

**Objetivo:** Avalia o modelo treinado no conjunto de teste (1.000 imagens não vistas).

**Saída esperada:**
```python
{
    'test_loss': 0.3124,
    'test_accuracy': 0.9100,
    'test_runtime': 12.34,
    'test_samples_per_second': 81.03
}
```

**Interpretação:** O modelo alcançou ~91% de acurácia no conjunto de teste!

---

## Célula 17: Verificação de uma Predição Individual

```python
itos[np.argmax(outputs.predictions[0])], itos[outputs.label_ids[0]]
```

**Objetivo:** Compara a predição do modelo com o label verdadeiro para o primeiro exemplo do teste.

**Exemplo de saída:**
```python
('airplane', 'airplane')  # ✓ Correto!
```

ou

```python
('automobile', 'airplane')  # ✗ Erro
```

---

## Célula 18: Matriz de Confusão

```python
y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

labels = trainds.features['label'].names
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)
```

**Objetivo:** Cria e exibe uma matriz de confusão mostrando o desempenho detalhado por classe.

### O que é uma Matriz de Confusão?

Tabela que mostra:
- **Linhas:** Classes verdadeiras
- **Colunas:** Classes preditas
- **Valores:** Quantidade de predições

**Exemplo simplificado:**

```
                Predito
              cat  dog  bird
    cat   │   85    3    2   │
Real dog   │    4   88    1   │
    bird  │    2    1   89   │
```

**Interpretação:**
- Diagonal principal (85, 88, 89): Predições corretas
- Fora da diagonal: Confusões/erros
- Exemplo: 3 gatos foram confundidos com cachorros

### Por que é útil?

1. Identifica quais classes são confundidas frequentemente
2. Mostra padrões de erro (ex: "cat" vs "dog" é mais difícil que "airplane" vs "ship")
3. Ajuda a entender onde o modelo precisa melhorar

---

## Resumo do Fluxo Completo

```
1. Instalar bibliotecas
         ↓
2. Importar módulos
         ↓
3. Carregar CIFAR-10 (5K treino + 1K teste)
         ↓
4. Dividir em treino/validação/teste
         ↓
5. Criar mapeamentos itos/stoi
         ↓
6. Configurar processador ViT
         ↓
7. Definir transformações (resize + normalize)
         ↓
8. Aplicar transformações aos datasets
         ↓
9. Carregar modelo ViT pré-treinado
         ↓
10. Adaptar para 10 classes do CIFAR-10
         ↓
11. Configurar hiperparâmetros
         ↓
12. Definir funções auxiliares
         ↓
13. Criar Trainer
         ↓
14. Treinar por 3 épocas
         ↓
15. Avaliar no teste
         ↓
16. Visualizar resultados e matriz de confusão
```

---

## Conceitos-Chave

### Transfer Learning
Usar conhecimento de um modelo pré-treinado (ImageNet) para uma nova tarefa (CIFAR-10).

### Fine-tuning
Ajustar levemente os pesos pré-treinados para a nova tarefa, em vez de treinar do zero.

### Vision Transformer (ViT)
Arquitetura que aplica o mecanismo de attention do Transformer para visão computacional.

### Normalização
Transformar pixels para uma distribuição padronizada, facilitando o aprendizado.

### Batch Processing
Processar múltiplas imagens simultaneamente para eficiência computacional.

---

## Resultados Esperados

Com este notebook, você deve obter:
- **Acurácia de treino:** ~95%+
- **Acurácia de validação:** ~91-92%
- **Acurácia de teste:** ~91%
- **Tempo de treinamento:** 5-10 minutos (com GPU)

---

## Possíveis Melhorias

1. **Data Augmentation:** Adicionar rotações, flips, crops aleatórios
2. **Mais épocas:** Treinar por mais tempo
3. **Learning rate scheduling:** Reduzir learning rate gradualmente
4. **Mais dados:** Usar todo o CIFAR-10 (50K treino)
5. **Ensemble:** Combinar múltiplos modelos

---

*Este documento foi gerado para auxiliar no entendimento completo do processo de fine-tuning de Vision Transformers no dataset CIFAR-10.*
