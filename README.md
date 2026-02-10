# Progetto Object Detection

Questo repository contiene l'implementazione e l'analisi di diverse architetture per l'**Object Detection** [cite: 2], sviluppate per il dataset **COCO 2017** e il dataset **xView**

---

## 👥 Contribuenti
* **Bosco Salvatore**
* **Danilo Cioffi**
* **Paolo Altucci**
* **Ernesto Cifuni**

---

## 🎯 Obiettivi del Progetto
L'obiettivo principale è confrontare diverse metodologie di addestramento, dalla creazione di modelli "da zero" (from scratch) all'utilizzo di modelli pre-addestrati e Foundation Models

### 🛠️ Modelli Implementati

#### 1. Implementazione da Scratch (con Babysitting)
* **Faster R-CNN**: Implementato e addestrato su dataset COCO utilizzando l'architettura ResNet-50 FPN.
* **RetinaNet**: Implementato e addestrato su dataset COCO seguendo le specifiche della ricerca originale.

#### 2. Transfer Learning (Fine-Tuning)
* **YOLO v11**: Fine-tuning di un modello pre-addestrato su COCO per il rilevamento di oggetti nel dataset xView.
* **Mask R-CNN**: Fine-tuning dell'architettura ResNet-50 FPN sul dataset xView.

#### 3. Foundation Models
* **Detectron2**: Utilizzo del framework senza necessità di ri-addestramento per scopi di benchmark.

---

## 📊 Metodologia di Valutazione
Tutti i metodi sono stati rigorosamente valutati per confrontarne le performance.

### Funzioni di Loss Utilizzate
* **Classificazione**: Cross-Entropy Loss [cite: 14] e Focal Loss.
* **Localizzazione**: Mean Absolute Error (MAE) [cite: 16] e Mean Squared Error (MSE).

### Metriche di Valutazione
* **IoU** (Intersection over Union).
* **Precision** e **Recall**.
* **mAP** (Mean Average Precision).

---

## 🔍 Explainable AI (XAI)
Per interpretare il funzionamento interno dei modelli, è stata implementata la tecnica di **Activation Maximization** applicata a Faster R-CNN.

* **Visualizzazione**: Generazione di immagini per layer e filtri selezionati.
* **Analisi**: Discussione sulle strutture catturate, come bordi, texture e regioni semantiche.
