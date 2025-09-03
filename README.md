# Fine-Tuning DistilBERT for Restaurant Search NER

This project fine-tunes **DistilBERT** for **Named Entity Recognition (NER)** on the **MIT Restaurant Search Dataset**.  
The goal is to extract key entities (such as Dish, Cuisine, Restaurant Name, Price, Hours, Rating, Location, and Amenities) from restaurant-related user queries.

---

## Dataset

We use the **MIT Restaurant Search NER Dataset** from [MIT CSAIL](https://groups.csail.mit.edu/sls/downloads/restaurant/).  
It contains user queries annotated with IOB-style entity tags.

- **Entities**:
  - `Dish`
  - `Cuisine`
  - `Restaurant_Name`
  - `Price`
  - `Hours`
  - `Rating`
  - `Location`
  - `Amenity`
  - `O` (Outside any entity)

- **Data Statistics**:
  - Training set: **6,425 samples**
  - Validation set: **918 samples**
  - Test set: **1,836 samples**
  - Total: **9,179 samples**

---

## Approach

1. **Preprocessing**:  
   - Loaded BIO-formatted dataset (`.bio` files).  
   - Converted into tokenized sequences with aligned NER tags.  

2. **Model**:  
   - Fine-tuned `distilbert-base-uncased` using Hugging Face `Trainer`.  
   - Token classification head added for multi-class NER.  

3. **Evaluation Metrics**:  
   - **Precision, Recall, F1-score, Accuracy** using `seqeval`.  

4. **Training Setup**:  
   - Optimizer: AdamW with learning rate `2e-5`.  
   - Epochs: 3  
   - Batch processing with dynamic padding (`DataCollatorForTokenClassification`).  

---

## Results

### Training & Validation Performance

| Epoch | Training Loss | Validation Loss | Precision | Recall | F1 | Accuracy |
|-------|---------------|-----------------|-----------|--------|----|----------|
| 1     | 0.6372        | 0.3310          | 0.6937    | 0.7541 | 0.7227 | 0.8990 |
| 2     | 0.2547        | 0.3156          | 0.7565    | 0.7787 | 0.7674 | 0.9082 |
| 3     | 0.2063        | 0.3182          | 0.7559    | 0.7830 | 0.7692 | 0.9108 |

### Test Performance

- **Test Loss**: 0.2833  
- **Precision**: 0.7796  
- **Recall**: 0.8066  
- **F1 Score**: 0.7929  
- **Accuracy**: 0.9201  

---

## Example Inference

Query:  
`"which restaurant serves the best shushi in new york?"`

Predicted Entities:
- **Rating**: `best`  
- **Dish**: `shushi`  
- **Location**: `new york`  

---

## Key Takeaways

- **DistilBERT** performs strongly on the MIT Restaurant dataset, achieving **~92% accuracy**.  
- The model generalizes well to unseen queries and correctly identifies restaurant-related entities.  
- Fine-tuning smaller transformer models like DistilBERT makes NER efficient without major accuracy trade-offs.  

---

## Future Work

- Extend to **MIT Movie NER dataset** for cross-domain benchmarking.  
- Experiment with **data augmentation** for rare entity classes.  
- Compare performance with **BERT-base, RoBERTa, and TinyBERT**.  
- Deploy as a **restaurant query chatbot** for real-time entity extraction.  

---

## References

- [MIT Restaurant Corpus](https://groups.csail.mit.edu/sls/downloads/restaurant/)  
- [Hugging Face Transformers](https://huggingface.co/transformers/)  
- [SeqEval](https://github.com/chakki-works/seqeval)  

