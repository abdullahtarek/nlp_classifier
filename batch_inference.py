import hydra
from transformers import AutoTokenizer
import pandas as pd
from utils.text_cleaner import Cleaner
from models.bert_model import BertModel
import pickle

@hydra.main(config_path="conf/",config_name="config")
def main(cfg):

    # Read configs
    inference_conf = cfg['inference_conf']
    model_conf = cfg['model_conf']

    # Read mode,tokenizer and label encoder
    tokenizer = AutoTokenizer.from_pretrained(model_conf['model_name'])
    label_encoder =  pickle.load(open(inference_conf['label_map_path'], "rb"))
    num_labels = len(label_encoder.classes_)
    model = BertModel(model_conf,num_labels,inference_flag=True)
    
    # Read input file
    df_test = pd.read_csv(inference_conf['input_file_path'])
    
    # Clean Text
    cleaner = Cleaner()
    df_test['text_cleaned'] = df_test[inference_conf['text_column']].apply(cleaner.clean)
    
    df_test = df_test.sample(20)

    samples = df_test['text_cleaned'].tolist()

    # Predict 
    preds = model.predict(samples, inference_conf,tokenizer,label_encoder)

    # Save results
    output_df = pd.DataFrame({inference_conf['output_column_name']:preds})
    output_df.to_csv(inference_conf['output_file_path'],index=None)




if __name__ == "__main__":
    main()