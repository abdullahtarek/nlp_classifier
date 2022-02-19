from models.bert_model import BertModel
import pandas as pd
from utils.text_cleaner import Cleaner
from sklearn.model_selection import train_test_split
from data_loader.data_loader import BERTDataloader
from sklearn import preprocessing
import hydra
import pickle
import os


@hydra.main(config_path="conf/",config_name="config")
def main(cfg):

    training_conf = cfg['training_conf']
    model_conf = cfg['model_conf']

    print(training_conf)
    print(model_conf)
    
    # Read dataset
    dataset_df = pd.read_csv(training_conf['dataset_path'])
    label_column_name = training_conf['label_column']
    text_column_name = training_conf['text_column']

    # Clean dataset
    cleaner = Cleaner()
    dataset_df['text_cleaned'] = dataset_df[text_column_name].apply(cleaner.clean)

    # Label encoder
    le = preprocessing.LabelEncoder()
    le.fit(dataset_df[label_column_name].tolist())
    num_labels = len(le.classes_)

    ## Save label map
    if not os.path.isdir(training_conf['training_output_dir']):
        os.mkdir(training_conf['training_output_dir'])
    
    label_map_path = os.path.join(training_conf['training_output_dir'],'label_map.pickle')
    with open(label_map_path, 'wb') as handle:
        pickle.dump(le, handle)
    
    # train/validation split
    dataset_df = dataset_df.sample(frac=1,random_state=42).reset_index(drop=True)
    df_train, df_val = train_test_split(dataset_df, test_size=training_conf['test_size'],stratify=dataset_df[label_column_name] ,random_state=42)

    # Dataloaders
    train_dataset = BERTDataloader(df_train["text_cleaned"].to_list(),df_train[training_conf['label_column']].to_list(),model_conf,le)
    val_dataset = BERTDataloader(df_val["text_cleaned"].to_list(),df_val[training_conf['label_column']].to_list(),model_conf,le)
    
    # Training
    bert_model = BertModel(model_conf,num_labels)
    bert_model.train(train_dataset,val_dataset,training_conf)


if __name__ == "__main__":
    main()