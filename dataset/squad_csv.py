import os
import numpy as np
import pandas as pd 
import json
from handle_text import handle_text, handle_text_lower

'''
input_file_path: path to the squad json file.
record_path: path to deepest level in json file
'''

def squad11_csv(input_file_path, record_path = ['data','paragraphs','qas']):
    print("Reading the json file")    
    file = json.loads(open(input_file_path, encoding='utf8').read())

    # parsing different level's in the json file
    m = pd.json_normalize(file, record_path)
    r = pd.json_normalize(file, record_path[:-1])

    # combining it into single dataframe
    m['context'] = np.repeat(r['context'].values, r.qas.str.len())
    main = m[['question','context']].copy()
    main.loc[:,'label'] = 'true'
    main.iloc[:,:-1] = main.iloc[:,:-1].applymap(handle_text_lower)

    print("Shape of the dataframe is {}".format(main.shape))
    print("Done")
    return main

def squad20_csv(input_file_path, record_path = ['data','paragraphs','qas']):
    print("Reading the json file")    
    file = json.loads(open(input_file_path, encoding='utf-8').read())

    # parsing different level's in the json file
    m = pd.json_normalize(file, record_path)
    r = pd.json_normalize(file, record_path[:-1])
    
    # combining it into single dataframe
    m['context'] = np.repeat(r['context'].values, r.qas.str.len())
    main = m[['question','context', 'is_impossible']].copy()
    main.iloc[:,:-1] = main.iloc[:,:-1].applymap(handle_text_lower)

    print("Shape of the dataframe is {}".format(main.shape))
    print("Done")
    return main

def train_zalo_csv(input_file_path):
    print("Reading the json file")    
    file = json.loads(open(input_file_path, encoding='utf8').read())

    # parsing different level's in the json file
    m = pd.json_normalize(file)

    # combining it into single dataframe
    main = m[['question','text', 'label']].copy()
    main.iloc[:,:-1] = main.iloc[:,:-1].applymap(handle_text_lower)

    print("Shape of the dataframe is {}".format(main.shape))
    print("Done")
    return main

def test_zalo_csv(input_file_path, answer_file_path):
    print("Reading the json file")
    file = json.loads(open(input_file_path, encoding='utf8').read())
    ans = json.loads(open(answer_file_path, encoding='utf8').read())

    # parsing different level's in the json file
    m = pd.json_normalize(file)
    r = pd.json_normalize(file, 'paragraphs')
    answer = pd.json_normalize(ans)

    # combining it into single dataframe
    r['question'] = np.repeat(m['question'].values, m.paragraphs.str.len())
    r['__id__'] = np.repeat(m['__id__'].values, m.paragraphs.str.len())
    
    # merge test data with label
    answer = r.merge(answer, how='left')
    # fill label for false answer
    answer = answer.fillna(value={'label': 'False'})
    
    main = answer[['question', 'text', 'label']].copy()
    main.iloc[:,:-1] = main.iloc[:,:-1].applymap(handle_text_lower)
    print("Shape of the dataframe is {}".format(main.shape))
    print("Done")
    return main

if __name__ == "__main__":
    # for filename in os.listdir('./dataset/raw/1.1'):
    #     if filename.endswith(".json"):
    #         print(filename)
    #         filepath = os.path.join('./dataset/raw/1.1', filename)
    #         df = squad11_csv(input_file_path=filepath)
    #         new_filename = os.path.splitext(filename)[0] + '.csv'
    #         new_filepath = os.path.join('./dataset/preprocess/1.1', new_filename)
    #         df.to_csv(new_filepath, header=False, index=False, encoding='utf8', sep='\t')

    # for filename in os.listdir('./dataset/raw/2.0'):
    #     if filename.endswith(".json"):
    #         print(filename)
    #         filepath = os.path.join('./dataset/raw/2.0', filename)
    #         df = squad20_csv(input_file_path=filepath)
    #         new_filename = os.path.splitext(filename)[0] + '.csv'
    #         new_filepath = os.path.join('./dataset/preprocess/2.0', new_filename)
    #         df.to_csv(new_filepath, header=False, index=False, encoding='utf8', sep='\t')

    input_file_path = './dataset/raw/train_ZALO.json'
    print('train_ZALO.json')
    df = train_zalo_csv(input_file_path=input_file_path)
    df.to_csv('./dataset/preprocess/train_ZALO_cased.csv', header=False, index=False, encoding='utf8', sep='\t')

    input_file_path = './dataset/raw/test_ZALO.json'
    answer_file_path = './dataset/raw/test_with_answers.json'
    print('test_ZALO.json')
    df = test_zalo_csv(input_file_path=input_file_path, answer_file_path=answer_file_path)
    df.to_csv('./dataset/preprocess/test_ZALO_cased.csv', header=False, index=False, encoding='utf8', sep='\t')

    print("Done")