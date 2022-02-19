from torch.utils.data import Dataset
from transformers.data.processors.utils import InputFeatures
import numpy as np
from transformers import AutoTokenizer

class BERTDataloader(Dataset):
    def __init__(self, text, target,model_conf, label_map):
        super(BERTDataloader).__init__()
        self.text = text
        self.target = target
        self.tokenizer = AutoTokenizer.from_pretrained(model_conf['model_name'])
        self.max_len = model_conf['max_len']
        self.label_map = label_map
      
    def __len__(self):
        return len(self.text)

    def one_hot(self,index):
        output = np.zeros(len(self.label_map.classes_) )
        output[index]=1
        return output.tolist()

    def __getitem__(self,item):
        text = str(self.text[item])
    
        tokenizer_output = self.tokenizer(text,truncation=True,padding='max_length',max_length=256)
        input_ids = tokenizer_output['input_ids']
        attention_mask = tokenizer_output['attention_mask']

        label_int = self.label_map.transform([self.target[item]])[0]
        label_onehot = self.one_hot(label_int)
        
        return InputFeatures(input_ids=input_ids ,attention_mask=attention_mask, label=label_onehot )