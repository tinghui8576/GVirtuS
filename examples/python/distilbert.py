#from transformers import AutoModel, AutoTokenizer
import ctypes

lib = ctypes.CDLL("/usr/local/gvirtus/lib/frontend/libcudart.so")
import torch
#print("Loading DistilBERT model for inference...")
#model = AutoModel.from_pretrained("distilbert-base-uncased").eval().cuda()
#tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

#inputs = tokenizer("Inference test!", return_tensors="pt").to("cuda")
#with torch.no_grad():
#    emb = model(**inputs).last_hidden_state.mean(1)
#print(emb.shape)