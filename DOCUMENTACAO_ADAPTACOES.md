# Documentação de Adaptações: De CIFAR-10 para Dataset Customizado

## 📋 Visão Geral das Mudanças

Este documento detalha **todas as alterações** feitas no notebook para transformá-lo de um exemplo acadêmico (CIFAR-10) em um pipeline **profissional e realista** de machine learning com dados customizados.

### Princípio Principal
> **Em competições Kaggle reais**: o arquivo `test.csv` **NÃO tem labels**. O modelo faz predições em dados nunca vistos e você submete as predições.

---

## 🔄 Mudança 1: Imports (Célula 2)

### ❌ Problema com Código Original
```python
from datasets import load_dataset  # Hugging Face datasets
```
- Acoplado ao formato HuggingFace
- Menos flexível para dados customizados
- Difícil estender para novos tipos de dados

### ✅ Solução Implementada
```python
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader, random_split
```

### 📚 Por Quê?
1. **pandas**: Manipular CSVs é simples e universal
2. **os**: Navegar sistema de arquivos (padrão em ML real)
3. **Dataset**: Classe base do PyTorch - padrão da indústria
4. **random_split**: Dividir train/val mantendo ordem

### 🎯 Quando Usar
- Dados em pasta + CSV (99% das aplicações reais)
- Projetos independentes de plataformas
- Quando precisa customizar carregamento de dados

---

## 🗂️ Mudança 2: Criar CustomImageDataset (Célula 3)

### ❌ Problema Original
```python
trainds, testds = load_dataset("cifar10", split=["train[:5000]","test[:1000]"])
```
- HuggingFace lida tudo internamente
- Opaco: não sabemos como dados são carregados
- Difícil adaptar para test set sem labels

### ✅ Solução: Classe CustomImageDataset
```python
class CustomImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, has_labels=True):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.has_labels = has_labels
        # ...
    
    def __getitem__(self, idx):
        # Carregar imagem
        # Se tem_labels: retornar label
        # Se não tem_labels: retornar None/skip (CENÁRIO KAGGLE)
```

### 📊 Comparação de Estruturas

| Aspecto | Antigo (CIFAR-10) | Novo (Customizado) |
|---------|-------------------|-------------------|
| Fonte | Hugging Face | CSV + Pasta |
| Train/Val | Dividir HF dataset | random_split |
| Test labels | Tem labels | **SEM labels** ✓ |
| Flexibilidade | Nenhuma | Completa |
| Debugging | Difícil | Simples |

### 🎯 Vantagem Principal
```python
# Você controla tudo:
# 1. Onde estão os dados (caminho)
# 2. Como carregar (transformações)
# 3. O que retornar (pixels, labels, file_id)
# 4. Lidar com dados sem labels (REAL!)
```

---

## 🔢 Mudança 3: Classes Dinâmicas (Célula 4)

### ❌ Código Original
```python
# CIFAR-10 tem exatamente 10 classes, hardcoded:
itos = dict((k,v) for k,v in enumerate(trainds.features['label'].names))
# Resultado: ['airplane', 'automobile', 'bird', ...]
```

### ✅ Código Novo
```python
class_names = [f'class_{i}' for i in full_dataset.classes]
itos = dict(enumerate(class_names))
# Descobre automaticamente do CSV!
```

### ⚙️ Como Funciona
1. **Detecção automática**: `full_dataset.classes` = valores únicos no CSV
2. **Genérico**: Funciona para 2, 10, 100, ou 1000 classes
3. **Rastreável**: Cada classe tem nome único (class_0, class_1, etc)

### 📈 Exemplo Prático
```
train.csv contém:
file_id,class
19661,0
10805,0
...
11377,1
10546,1

full_dataset.classes = [0, 1]
itos = {0: 'class_0', 1: 'class_1'}
stoi = {'class_0': 0, 'class_1': 1}
```

---

## 📸 Mudança 4: Aplicar Transformações (Célula 8)

### ❌ Problema
```python
# HuggingFace tem método set_transform:
trainds.set_transform(transf)
```
- Método específico de HF Dataset
- Não funciona com PyTorch Dataset genérico

### ✅ Solução: TransformDataset Wrapper
```python
class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.transform:
            sample['pixels'] = self.transform(sample['img'])
        return sample

trainds = TransformDataset(trainds, transform=_transf)
```

### 🤔 Por Quê?
1. **Padrão PyTorch**: Transformações são aplicadas em `__getitem__`
2. **Eficiente**: Carrega imagem + transforma sob demanda (lazy loading)
3. **Reutilizável**: Funciona com qualquer Dataset

---

## 🎯 Mudança 5: Dinâmica de Classes no Modelo (Célula 11)

### ❌ Hardcoded (Antigo)
```python
model = ViTForImageClassification.from_pretrained(
    model_name, 
    num_labels=10,  # 🔴 CIFAR-10 específico!
    ...
)
```

### ✅ Dinâmico (Novo)
```python
num_labels = len(full_dataset.classes)  # Automático!
model = ViTForImageClassification.from_pretrained(
    model_name, 
    num_labels=num_labels,  # 🟢 Qualquer número
    ...
)
```

### 🔄 Impacto
- **Antes**: Mudar CIFAR-10 → novo dataset = editar código
- **Depois**: Simplesmente mudar CSV, model se adapta!

---

## 📦 Mudança 6: collate_fn com file_ids (Célula 13)

### ❌ Código Original
```python
def collate_fn(examples):
    pixels = torch.stack([example["pixels"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixels, "labels": labels}
    # ❌ Perde informação de file_id!
```

### ✅ Código Novo
```python
def collate_fn(examples):
    pixels = torch.stack([example["pixels"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    file_ids = [example["file_id"] for example in examples]  # ✅ NOVO
    return {
        "pixel_values": pixels, 
        "labels": labels,
        "file_id": file_ids  # Rastrear para submissão Kaggle
    }
```

### 🎯 Por Quê?
```
Para submissão Kaggle, você precisa de:
  file_id₁ → class_predito₁
  file_id₂ → class_predito₂
  ...

Sem rastrear file_id, você perde a correspondência!
```

---

## 🔮 Mudança 7: Test Predictions SEM Labels (Célula 16)

### ❌ Problema (CIFAR-10)
```python
# CIFAR-10 test tinha labels
outputs = trainer.predict(testds)
print(outputs.metrics)  # Pode calcular accuracy
```

### ✅ Realidade (Kaggle)
```python
# Test set NÃO tem labels!
outputs = trainer.predict(testds)  # Sem labels = sem .metrics

# Mas temos:
predicted_classes = np.argmax(outputs.predictions, axis=1)
# → Estas são as predições para submissão
```

### 🚨 Consequência
```
❌ ERRADO: comparar predições test com labels (não existem!)
✅ CERTO: submeter predições ao Kaggle, eles verificam

Isso muda TUDO a forma de validar!
```

---

## 📝 Mudança 8: Gerar Arquivo de Submissão (Célula 17 - NOVO)

### ⚠️ Este código não existia antes!

```python
# Célula completamente nova para workflow Kaggle

test_file_ids = []
for i in range(len(testds.dataset)):
    sample = testds.dataset[i]
    test_file_ids.append(sample['file_id'])

submission_df = pd.DataFrame({
    'file_id': test_file_ids,
    'class': predicted_classes
})

submission_df.to_csv('submission.csv', index=False)
```

### 🎯 Fluxo Real
```
1. Treinar em train.csv (com labels)
2. Fazer predictions em test.csv (sem labels)
3. Criar CSV: file_id, class_predito
4. Enviar para Kaggle
5. Kaggle compara com labels verdadeiros (que você não tem)
6. Recebe score público
```

---

## 📊 Mudança 9: Análise em Validation Set (Célula 18)

### ❌ Antigo
```python
# Analisava test set:
y_true = outputs.label_ids  # test set labels (CIFAR-10 tinha)
y_pred = outputs.predictions.argmax(1)
# Confusion matrix no test
```

### ✅ Novo
```python
# Analisa validation set (que sempre tem labels):
val_outputs = trainer.predict(valds)  # ← val não test!
y_true = y_true  # labels do val
y_pred = np.argmax(val_outputs.predictions, axis=1)
# Confusion matrix no val
```

### 🔍 Diferença Crítica
| Dataset | Labels? | Uso |
|---------|---------|-----|
| **Train** | ✅ Sim | Treinar modelo |
| **Val** | ✅ Sim | Monitorar progresso, analyzePerformance |
| **Test** | ❌ Não | Fazer predições para submissão |

---

## 🎓 Conceitos Aprofundados

### 1️⃣ Transfer Learning e Normalização

```python
# Por que usar ImageNet mean/std?
mu = [0.485, 0.456, 0.406]  # ImageNet average pixel values
sigma = [0.229, 0.224, 0.225]  # ImageNet std dev

# Modelo foi TREINADO em ImageNet com estes valores
# Usar diferentes valores = distribuição input diferente
# = Pesos pré-treinados perdem efetividade

# Analogia: treinar alguém a ler metros, depois medir em feet
```

### 2️⃣ Data Leakage

```python
# ❌ NUNCA fazer isto:
train_val_combined = load_all_data()
splits = train_test_split(train_val_combined)

# ✅ CORRETO:
train = load_train_data()
val = load_val_data()  # Ou dividir train
test = load_test_data()

# Razão: dados de teste nunca devem influenciar modelo
```

### 3️⃣ Stratified Splits

```python
# random_split mantém proporções?
# NÃO! Para dados desbalanceados, use:
from sklearn.model_selection import train_test_split

indices = np.arange(len(full_dataset))
labels = full_dataset.df['class'].values

train_idx, val_idx = train_test_split(
    indices, 
    test_size=0.1, 
    stratify=labels  # ← Garante mesma distribuição
)
```

---

## 🛠️ Como Estender Este Código

### Cenário 1: Adicionar mais dados
```python
# Antes:
# 1. Tirar data antigo
# 2. Reescrever código

# Depois:
# 1. Colocar novas imagens em /data/train/
# 2. Adicionar linhas em train.csv
# 3. PRONTO! Rodar notebook novamente
```

### Cenário 2: Mudar para multi-label
```python
# train.csv atual:
# file_id,class
# 19661,0

# Multi-label:
# file_id,class
# 19661,"0,1,3"

# Adaptar CustomImageDataset.__getitem__:
labels = [int(x) for x in label_str.split(',')]
return {
    'pixels': pixels,
    'labels': labels,  # Agora lista de labels
    'file_id': file_id
}
```

### Cenário 3: Usar data augmentation
```python
from torchvision.transforms import RandomHorizontalFlip, RandomRotation

_transf = Compose([
    RandomHorizontalFlip(p=0.5),      # Novo!
    RandomRotation(degrees=15),        # Novo!
    Resize(size['height']),
    ToTensor(),
    norm
])
# Automáticamente aplicado durante treinamento
```

---

## 🎯 Comparação Lado-a-Lado

### Fluxo CIFAR-10 (Original)
```
1. load_dataset("cifar10")  ← Automático, predefinido
2. train_test_split()
3. Transformações via set_transform
4. Treinar em train/val
5. Testar em test (COM labels verdadeiros)
6. Confusion matrix = ✓ Valida tudo
```

### Fluxo Customizado (Novo) ⭐
```
1. Ler train.csv (com labels) ← Você controla
2. Ler test.csv (SEM labels) ← Realista!
3. CustomImageDataset + TransformDataset
4. Treinar em train/val
5. Prever em test (SEM labels)
6. Gerar submission.csv para Kaggle
7. Kaggle valida (você nunca vê test labels)
```

---

## 📚 Para Aprofundar

### Tópicos Relacionados
1. **Data Augmentation** - Aumentar dataset sinteticamente
2. **Class Imbalance** - Lidar com classes desbalanceadas
3. **Hyperparameter Tuning** - Otimizar learning_rate, batch_size, etc
4. **Cross-Validation** - Validação mais robusta
5. **Ensemble Methods** - Combinar múltiplos modelos
6. **Interpretability** - Entender decisões do modelo (Grad-CAM, etc)

### Exercícios Propostos
1. **Fácil**: Adicionar 100 novas imagens ao dataset
2. **Médio**: Implementar class weighting para classes desbalanceadas
3. **Difícil**: Implementar k-fold cross-validation
4. **Avançado**: Usar model ensemble (5 modelos) para submissão final

---

## 🔗 Referências

- [PyTorch Dataset Documentation](https://pytorch.org/docs/stable/data.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [Kaggle Competitions Guide](https://www.kaggle.com/docs/competitions)

---

## ✅ Checklist: Estrutura Profissional

- [x] Código sem labels no test set (realista)
- [x] CustomImageDataset genérico e reutilizável
- [x] Rastreamento de file_ids para submissão
- [x] Validation set para análise (não test!)
- [x] Confusion matrix interpretável
- [x] Arquivo de submissão em formato Kaggle
- [x] Documentação inline das mudanças
- [x] Código antigo comentado para comparação
- [x] Extensível para novos dados

---

**Versão**: 1.0  
**Data**: 14/01/2026  
**Status**: ✅ Pronto para produção
