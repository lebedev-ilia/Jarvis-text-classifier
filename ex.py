from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-multilingual-uncased")
k = 0
for i in model.parameters():
    if (len(i.shape)) == 2:
        k += i.shape[0] * i.shape[1]
    else:
        k += i.shape[0]
print(k)