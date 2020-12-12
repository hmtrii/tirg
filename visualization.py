import pickle
import matplotlib.pyplot as plt
import numpy as np
from main import load_dataset, create_model_and_optimizer
from tqdm import tqdm
from test_retrieval import test
import img_text_composition_models
import torch
from LSH import create_hash_table
import time

def pkl_save(path,obj):
  with open(path, 'wb') as handle:
      pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
def pkl_load(path):
  with open(path, 'rb') as handle:
      return pickle.load(handle)

class Opt:
	def __init__(self):
		self.dataset = "fashion200k"
		self.dataset_path = "./dataset/Fashion200k"
		self.batch_size = 32
		self.embed_dim = 512
		self.hashing = True

def create_model(texts, embed_dim):
	model = img_text_composition_models.TIRG(texts, embed_dim=embed_dim)
	return model

def compute_similary_by_hasing(all_imgs_feature, queries_feature, test_queries):
	hash_table = create_hash_table(all_imgs_feature, 10, 512)
	sims = []
	for i, query in tqdm(enumerate(queries_feature)):
		idx_imgs = hash_table.__getitem__(query)
		spatial_search = np.array([all_imgs_feature[i] for i in idx_imgs])
		if spatial_search.shape[0] != 0:
			dis_value = query.dot(spatial_search.T)
			sim = np.full(all_imgs_feature.shape[0], -10e10)
			for i in idx_imgs:
				sim[i] = dis_value[0]
				dis_value = dis_value[1:]
		else:
			sim = query.dot(all_imgs_feature.T)

		sim[test_queries[i]['source_img_id']] = -10e10
		sims += [sim]

	return np.array(sims)
	
def compute_similary_normal(all_imgs_feature, queries_feature, test_queries):
	sims = queries_feature.dot(all_imgs_feature.T)
	for i, t in enumerate(test_queries):
		try:
			sims[i, t["source_img_id"]] = -10e10  # remove query image
		except:
			pass

	return sims

if __name__ == "__main__":
	tic = time.time()
	opt = Opt()
	# query_text = ["replace white with black"]

	# model = create_model(query_text, opt.embed_dim)
	# model.text_model.embedding_layer = torch.nn.Embedding(5590, 512)
	# checkpoint = torch.load('./models/checkpoint_fashion200k.pth', \
	# 						map_location=torch.device("cpu"))
	# model.load_state_dict(checkpoint['model_state_dict'])
	# model.eval()

	queries_feature = pkl_load("./pkl/all_queries.pkl")
	all_imgs_feature = pkl_load("./pkl/all_imgs.pkl")
	all_captions = pkl_load("./pkl/all_captions.pkl")
	img_ids = pkl_load("./pkl/img_ids.pkl")
	mods = pkl_load("./pkl/mods.pkl")
	# nn_result = pkl_load("./pkl/nn_result.pkl")
	all_target_captions = pkl_load("./pkl/all_target_captions.pkl")

	# trainset, testset = load_dataset(opt)
	# output = test(opt, model, testset)

	# test_queries = testset.get_test_queries()
	

	queries_feature = queries_feature[:5000]
	# feature normalization
	for i in range(queries_feature.shape[0]):
		queries_feature[i, :] /= np.linalg.norm(queries_feature[i, :])

	for i in range(all_imgs_feature.shape[0]):
		all_imgs_feature[i, :] /= np.linalg.norm(all_imgs_feature[i, :])

	# match test queries to target images, get nearest neighbors
	test_queries = pkl_load("./pkl/test_queries.pkl")
	if opt.hashing:
		sims = compute_similary_by_hasing(all_imgs_feature, queries_feature, test_queries)
	else:
		sims = compute_similary_normal(all_imgs_feature, queries_feature, test_queries)
	
	nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]
	# # pkl_save("./pkl/nn_result.pkl", nn_result)
	# compute recalls
	out = []
	nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
	for k in [1, 5, 10, 50, 100]:
		r = 0.0
		for i, nns in enumerate(nn_result):
			if all_target_captions[i] in nns[:k]:
				r += 1
		r /= len(nn_result)
		out += [('recall_top' + str(k) + '_correct_composition', r)]
	tac = time.time()
	print(out)
	print(tac - tic)
