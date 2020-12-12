"""Compute feature and save into pkl file"""
import torch
from tqdm import tqdm
import numpy as np
import datasets
from main import load_dataset
from tools import opt, create_model, pkl
import PIL

def save_info_test_queries(opt, testset):
    test_queries = testset.get_test_queries()
    pkl.pkl_save("./pkl/test_queries.pkl", test_queries)
    print("Test queries information are saved at pkl/test_quriese.pkl")
    

def save_test_queries_feature(opt, model):
    test_queries = pkl.pkl_load("./pkl/test_queries.pkl")
    # compute test query features
    imgs = []
    mods = []
    all_queries = []
    for t in tqdm(test_queries):
        imgs += [testset.get_img(t["source_img_id"])]
        mods += [t["mod"]["str"]]
        if len(imgs) >= opt.batch_size or t is test_queries[-1]:
            if "torch" not in str(type(imgs[0])):
                imgs = [torch.from_numpy(d).float() for d in imgs]
            imgs = torch.stack(imgs).float()
            if torch.cuda.is_available():
                imgs = torch.autograd.Variable(imgs).cuda()
            else:
                imgs = torch.autograd.Variable(imgs).cpu()
            mods = [t for t in mods]
            f = model.compose_img_text(imgs, mods).data.cpu().numpy()
            all_queries += [f]
            imgs = []
            mods = []
    all_queries = np.concatenate(all_queries)
    all_target_captions = [t["target_caption"] for t in test_queries]

    # Save image id in test queries and modified string
    img_ids = []
    mods = []
    for t in tqdm(test_queries):
        img_ids += [t["source_img_id"]]
        mods += [t["mod"]["str"]]

    pkl.pkl_save("./pkl/all_queries.pkl", all_queries)
    pkl.pkl_save("./pkl/all_target_captions.pkl", all_target_captions)
    pkl.pkl_save("./pkl/img_ids.pkl", img_ids)
    pkl.pkl_save("./pkl/mods.pkl", mods)
    print("All queries feature are save at pkl/all_queries.pkl")
    print("All target captions are saved ar pkl/all_target_captions.pkl")
    print("All image id in test queries are saved at pkl/img_ids.pkl")
    print("All modified string are saved at pkl/mods.pkl")

def save_all_imgs_feature(opt, testset):
    """Extract image fature in testset and save them"""
    all_imgs = []
    all_captions = []

    # compute all image features
    imgs = []
    all_imgs = []
    for i in tqdm(list(range(len(testset.imgs)))):
        imgs += [testset.get_img(i)]
        if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
            if "torch" not in str(type(imgs[0])):
                imgs = [torch.from_numpy(d).float() for d in imgs]
            imgs = torch.stack(imgs).float()
            if torch.cuda.is_available():
                imgs = torch.autograd.Variable(imgs).cuda()
            else:
                imgs = torch.autograd.Variable(imgs).cpu()

            imgs = model.extract_img_feature(imgs).data.cpu().numpy()
            all_imgs += [imgs]
            imgs = []
    all_imgs = np.concatenate(all_imgs)
    all_captions = [img["captions"][0] for img in testset.imgs]

    pkl.pkl_save("./pkl/all_imgs.pkl", all_imgs)
    pkl.pkl_save("./pkl/all_captions.pkl", all_captions)

def save_normalize_all_imgs_feature():
    all_imgs_feature = pkl.pkl_load("./pkl/all_imgs.pkl")
    for i in tqdm(range(all_imgs_feature.shape[0])):
        all_imgs_feature /= np.linalg.norm(all_imgs_feature[i,:])

    pkl.pkl_save("./pkl/normalized_all_imgs_feature.pkl", all_imgs_feature)


if __name__ == "__main__":
    opt = opt.Opt()
    trainset, testset = load_dataset(opt)
    # model = create_model.create_model(opt)
    # model.eval()
    # print(testset.get_test_queries()[2000])
    # testset.get_img(108, raw_img=True).show()
    # save_info_test_queries(opt, testset)
    # save_test_queries_feature(opt, model)
    # save_all_imgs_feature(opt, testset)
    # save_normalize_all_imgs_feature()


    
    n = 2165
    all_captions = pkl.pkl_load("./pkl/all_captions.pkl")
    img_ids = pkl.pkl_load("./pkl/img_ids.pkl")
    mods = pkl.pkl_load("./pkl/mods.pkl")
    all_target_captions = pkl.pkl_load("./pkl/all_target_captions.pkl")
    test_queries = pkl.pkl_load("./pkl/test_queries.pkl")
    

    print(testset.get_test_queries()[n])
    print(test_queries[n])
    print(img_ids[n])
    print(mods[n])
    print(testset.imgs[img_ids[n]]['file_path'])
    print(all_captions[img_ids[n]])
    img = testset.get_img(img_ids[n], raw_img=True)
    img.show()
    # print(img_ids[:100])
