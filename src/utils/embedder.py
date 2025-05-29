from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm

class SimpleTextEmbedder:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device
        
        #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_embeddings(self, text_list: list[str], batch_size=64):   
        embeddings = []
        for i in tqdm(range(0, len(text_list), batch_size)):
            # Tokenize sentences
            encoded_input = self.tokenizer(text_list[i: i +batch_size], max_length=512, padding='max_length', truncation=True, return_tensors='pt')
            # Compute token embeddings
            encoded_input = encoded_input.to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                # Perform pooling. In this case, mean pooling.
                embeddings.append(self.mean_pooling(model_output, encoded_input['attention_mask'])) 
        return torch.cat(embeddings).detach().cpu().numpy()