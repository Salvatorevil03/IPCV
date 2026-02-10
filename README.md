# Progetto Object Detection - Gruppo D4 [cite: 1]

Questo repository contiene l'implementazione e l'analisi di diverse architetture per l'**Object Detection** [cite: 2], sviluppate per il dataset **COCO 2017** e il dataset **xView** [cite: 3].

---

## 👥 Contribuenti
* **Bosco Salvatore**
* **Danilo Cioffi**
* **Paolo Altucci**
* **Ernesto Cifuni**

---

## 🎯 Obiettivi del Progetto
L'obiettivo principale è confrontare diverse metodologie di addestramento, dalla creazione di modelli "da zero" (from scratch) all'utilizzo di modelli pre-addestrati e Foundation Models [cite: 4, 7, 9].

### 🛠️ Modelli Implementati

#### 1. Implementazione da Scratch (con Babysitting)
* **Faster R-CNN**: Implementato e addestrato su dataset COCO utilizzando l'architettura ResNet-50 FPN [cite: 4, 5].
* **RetinaNet**: Implementato e addestrato su dataset COCO seguendo le specifiche della ricerca originale [cite: 6].

#### 2. Transfer Learning (Fine-Tuning)
* **YOLO v11**: Fine-tuning di un modello pre-addestrato su COCO per il rilevamento di oggetti nel dataset xView [cite: 7].
* **Mask R-CNN**: Fine-tuning dell'architettura ResNet-50 FPN sul dataset xView [cite: 8].

#### 3. Foundation Models
* **Detectron2**: Utilizzo del framework senza necessità di ri-addestramento per scopi di benchmark [cite: 9].

---

## 📊 Metodologia di Valutazione
Tutti i metodi sono stati rigorosamente valutati per confrontarne le performance [cite: 10].

### Funzioni di Loss Utilizzate [cite: 13]
* **Classificazione**: Cross-Entropy Loss [cite: 14] e Focal Loss [cite: 15].
* **Localizzazione**: Mean Absolute Error (MAE) [cite: 16] e Mean Squared Error (MSE) [cite: 17].

### Metriche di Valutazione [cite: 18]
* **IoU** (Intersection over Union) [cite: 19].
* **Precision** e **Recall** [cite: 20].
* **mAP** (Mean Average Precision) [cite: 21].

---

## 🔍 Explainable AI (XAI)
Per interpretare il funzionamento interno dei modelli, è stata implementata la tecnica di **Activation Maximization** applicata a Faster R-CNN [cite: 11].

* **Visualizzazione**: Generazione di immagini per layer e filtri selezionati [cite: 12].
* **Analisi**: Discussione sulle strutture catturate, come bordi, texture e regioni semantiche [cite: 12].
