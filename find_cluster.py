from COVIDModel import COVIDModel
import pickle
import os
import numpy as np
import torch

cm = COVIDModel(model_name='bert-base-uncased', max_length=500, stride=250)
#question = "What do we know about virus genetics, origin, and evolution?"
question = "What do we know about vaccines and therapeutics?"
#question = "What has been published about information sharing and inter-sectoral collaboration?"
#question = "What has been published about ethical and social science considerations?"
embeddings = cm.getEmbedding(text=question)
embeddings = embeddings.reshape(768)
embeddings = list(embeddings.data.cpu().numpy())

def cosineSimilarity(vector1, vector2):
  print(type(vector1), type(vector2))
  return np.dot(vector1,vector2)/(np.sqrt(np.dot(vector1,vector1)*np.dot(vector2,vector2)))

def findCluster(input_embedding, cluster_embeddings):
  cosine_similarities = np.array([cosineSimilarity(input_embedding,cluster_embedding) for cluster_embedding in cluster_embeddings])
  return np.argmax(cosine_similarities)

def findDocs(input_embedding, cluster_embeddings):
  cosine_similarities = np.array([cosineSimilarity(input_embedding,cluster_embedding) for cluster_embedding in cluster_embeddings])
  print(np.sort(cosine_similarities))
  return np.argsort(cosine_similarities)

cluster_id_to_doc_id = None

with open("bert_embeddings_500_base_uncased_abstract_cluster_id_to_doc_ids.pkl", "rb") as input_file:
  cluster_id_to_doc_id = pickle.load(input_file)

cluster_id_to_centroid = None
with open("bert_embeddings_500_base_uncased_abstract_cluster_id_to_centroid.pkl", "rb") as input_file:
  cluster_id_to_centroid = pickle.load(input_file)

list_of_centroids = cluster_id_to_centroid.values()
cluster_id = findCluster(embeddings, list_of_centroids)
#print(type(list_of_centroids))
print("Cluster ID:{}".format(cluster_id))

list_of_docs = cluster_id_to_doc_id[cluster_id]
# get embeddings of all docs from abstract embeddings
doc_embeds = []
doc_id_to_embed = {}
directory = "bert_embeddings_500_base_uncased/abstract_embedding"
folders = os.listdir(directory)
for folder in folders:
  path = directory + "/" + folder + "/pdf_json"
  files = os.listdir(path)
  print(path, len(files))
  for f in files:
    if f in list_of_docs:
      embedding = torch.load(path+"/"+f)
      embedding = embedding.reshape(768)
      doc_embeds.append(list(embedding.data.cpu().numpy()))
      doc_id_to_embed[len(doc_embeds)-1] = f
top_10_docs = []

ids = findDocs(embeddings, doc_embeds)

for i in range(10):
  doc_index_sort = ids[len(ids)-1-i]
  top_10_docs.append(doc_id_to_embed[doc_index_sort])
  
print(top_10_docs)
