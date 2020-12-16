import torch
import torchvision
import PIL
import img_text_composition_models
import dataset
from main import load_dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
from LSH import create_hash_table
# from pkl import pkl_load, pkl_save
from tools import pkl, opt, create_model

def retriveve_custom_query(query_img_raw, query_text, query_feature, spatial_search, testset, model):
	# Normalize feature
	query_feature[0, :] /= np.linalg.norm(query_feature[0,:])

	# spatial_search = pkl.pkl_load("./pkl/normalized_all_imgs_feature.pkl")

	sims = query_feature.dot(spatial_search.T)
	nn_result = np.argsort(-sims[0, :])[:50]

	c = 5
	r = 3
	fig = plt.figure(figsize=(10, 10))
	# Show query
	fig.add_subplot(r, c, 3)
	plt.imshow(query_img_raw)
	plt.title(query_text[0])
	plt.axis("off")
	# Show output
	k = 10
	for i in range(k):
		img = testset.get_img(int(nn_result[i]), raw_img=True)
		fig.add_subplot(r, c, i+6)
		plt.imshow(img)
		plt.axis('off')
	
	plt.show()

def create_custom_query_feature(model):
	transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ])

	query_img_raw = PIL.Image.open('./dataset/Fashion200k/women/dresses/cocktail_dresses/36784190/36784190_5.jpeg').convert("RGB")
	query_img = transform(query_img_raw)
	query_img = [query_img]
	query_img = torch.stack(query_img).float()
	query_img = torch.autograd.Variable(query_img)
	query_text = ["replace green with pink"]

	# Compute query feature
	query_feature = model.compose_img_text(query_img, query_text).data.cpu().numpy()

	return query_img_raw, query_text, query_feature

def retrieve_random_query_from_testset(testset):
	queries_feature = pkl.pkl_load("./pkl/all_queries.pkl")[0:2000]
	for i in range(queries_feature.shape[0]):
		queries_feature[i, :] /= np.linalg.norm(queries_feature[i, :])

	all_imgs_feature = pkl.pkl_load("./pkl/normalized_all_imgs_feature.pkl")
	sims = queries_feature.dot(all_imgs_feature.T)

	test_queries = testset.get_test_queries()
	for i, t in enumerate(test_queries):
		try:    
			sims[i, t["source_img_id"]] = -10e10  # remove query image
		except:
			pass
	
	nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]

	chosen_idx = np.random.randint(0, 2000)
	print(chosen_idx)
	# Plot query
	img_ids = pkl.pkl_load("./pkl/img_ids.pkl")
	mods = pkl.pkl_load("./pkl/mods.pkl")
	all_target_captions = pkl.pkl_load("./pkl/all_target_captions.pkl")
	true_idx = img_ids[chosen_idx]
	im = testset.get_img(true_idx, raw_img=True)
	plt.imshow(im)
	mod = mods[chosen_idx]
	plt.title(mod)

	# Plot output
	plt.figure(figsize=(20,10))
	plt.axis('off')
	width = 4
	height = 4
	for i in range(width*height):
		plt.subplot(width,height,i+1)
		plt.xticks([])
		plt.yticks([])
		idx = nn_result[chosen_idx][i]
		img = testset.get_img(int(idx), raw_img=True)
		plt.imshow(img)
		
		if idx == all_target_captions[chosen_idx]:	
			plt.title("GT")
	plt.show()
	
if __name__ == "__main__":	
	opt = opt.Opt()
	model = create_model.create_model(opt).eval()
	trainset, testset = load_dataset(opt)
	if opt.retrieve_by_random:
		retrieve_random_query_from_testset(testset)
	else:
		query_img_raw, query_text, query_feature = create_custom_query_feature(model)
		# all_imgs_feature = pkl.pkl_load("./pkl/all_imgs.pkl")
		if opt.hashing:
			all_imgs_feature = pkl.pkl_load("./pkl/all_imgs.pkl")
			hash_table = create_hash_table(all_imgs_feature, 4, 512)
			spatial_search = hash_table.__getitem__(query_feature[0])
			for i in range(spatial_search.shape[0]):
				spatial_search[i, :] /= np.linalg.norm(spatial_search[i, :])
		else:
			spatial_search = pkl.pkl_load("./pkl/normalized_all_imgs_feature.pkl")
		retriveve_custom_query(query_img_raw, query_text, query_feature, spatial_search, testset, model)
	
	

	