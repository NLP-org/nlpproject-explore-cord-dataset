import torch
from transformers import BertTokenizer, BertModel, BertConfig
import os, os.path, shutil
import json
import pickle
from COVIDModel import COVIDModel

# loop through all files to :
# 1. fetch abstract and paper id.
# 2. get embeddings for all windows for the abstract.
# 3. pool them (default = mean).
# 3. store pooled embeddings in a pickle file by name as paper_id.

folder_paths = ["noncomm_use_subset/pdf_json", "custom_license/pdf_json", "comm_use_subset/pdf_json", "biorxiv_medrxiv/pdf_json"]
#folder_paths = ["Cord-dataset"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cm = COVIDModel(model_name='bert-base-uncased', max_length=500, stride=250)

for f in folder_paths:
  #list_of_id_and_embedding = []
  os.makedirs(os.path.join("./bert_embeddings_500_base_uncased/abstract_embedding/",f))
  os.makedirs(os.path.join("./bert_embeddings_500_base_uncased/passage_embedding/",f))
  os.makedirs(os.path.join("./bert_embeddings_500_base_uncased/document_embedding/",f))
  try:
    num_files = os.listdir(f)
    print(len(num_files))
    for _file in num_files:
      try:
        full_filename = os.path.join(f, _file)
        if os.path.exists(full_filename):
          with open(full_filename) as rf:
            data = json.load(rf)
            paper_id = data['paper_id']
            abstract_list = data['abstract']
            abstract = ''
            for i in abstract_list:
              abstract += i['text'] + ' '

            body_list = data['body_text']
            full_text = ''
            for i in body_list:
              full_text += i['text'] + ' '
          pf = os.path.join('./bert_embeddings_500_base_uncased/abstract_embedding/'+f, paper_id + '.pt')
          if not os.path.exists(pf):
            embeddings = cm.getEmbedding(text=full_text)
            torch.save(embeddings, pf)

          # full text
          pf = os.path.join('./bert_embeddings_500_base_uncased/document_embedding/'+f, paper_id + '.pt')
          if not os.path.exists(pf):
            embeddings = cm.getEmbedding(text=full_text)
            torch.save(embeddings, pf)
          
          # passage
          pf = './bert_embeddings_500_base_uncased/passage_embedding/'+f+'/' + paper_id
          if not os.path.exists(pf):
            os.makedirs(pf)
          for x in range(len(body_list)):
            passage = body_list[x]['text']
            pf = os.path.join('./bert_embeddings_500_base_uncased/passage_embedding/'+f+'/' + paper_id, paper_id + '_' + str(x) + '.pt')
            if not os.path.exists(pf):
              embeddings = cm.getEmbedding(text=passage)
              torch.save(embeddings, pf)
        else:
          print(full_filename + ' doesn\'t exist.')
      except Exception as er:
        print(er, _file)
  except Exception as e:
    print(e, f)
