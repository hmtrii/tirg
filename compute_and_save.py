"""Compute feature and save into pkl file"""
import torch
from tqdm import tqdm
import numpy as np
import datasets
from main import load_dataset
from tools import opt, create_model, pkl
import PIL
import text_model

def save_info_test_queries(opt, testset):
    test_queries = testset.get_test_queries()
    pkl.pkl_save("./pkl/test_queries.pkl", test_queries)
    print("Test queries information are saved at pkl/test_quriese.pkl")
    
def save_test_queries_feature(opt, model, testset):
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

def save_vocab(opt, trainset):
    vocab = text_model.SimpleVocab()
    for t in trainset.get_all_texts():
        vocab.add_text_to_vocab(t)
    pkl.pkl_save("./pkl/vocab.pkl", vocab)


if __name__ == "__main__":
    opt = opt.Opt()
    # model = create_model.create_model(opt)
    trainset, testset = load_dataset(opt)

    trainloader = trainset.get_loader(
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    
    for data in trainloader:
        print(data[0])
        s = trainset.get_img(data[0]['source_img_id'], raw_img=True)
        t =trainset.get_img(data[0]['target_img_id'], raw_img=True)
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 6))
        fig.add_subplot(3, 5, 3)
        plt.imshow(s)
        fig.add_subplot(3, 5, 8)
        plt.imshow(t)
        # img.show()
        plt.show()
        break


    # n = np.random.randint(1000)
    # a = testset.get_test_queries()[n]
    # print(a['source_caption'])
    # print(a['target_caption'])
    # print(a['mod']['str'])
    # source = testset.get_img(a['source_img_id'], raw_img=True)
    # target = testset.get_img(a['target_img_id'], raw_img=True)
    
    






    # all_captions = pkl.pkl_load("./pkl/all_captions.pkl")
    # img_ids = pkl.pkl_load("./pkl/img_ids.pkl")
    # mods = pkl.pkl_load("./pkl/mods.pkl")
    # all_target_captions = pkl.pkl_load("./pkl/all_target_captions.pkl")
    # test_queries = pkl.pkl_load("./pkl/test_queries.pkl")


    # n = np.random.randint(5000)
    # print(test_queries[n])
    # print(img_ids[n])
    # idx = img_ids[n]
    # target_idx = testset.get_test_queries()[n]['target_img_id']
    # s_img = testset.get_img(idx, raw_img=True)
    # t_img = testset.get_img(target_idx, raw_img=True)
    # print(testset.imgs[idx]['file_path'])
    # print(testset.imgs[target_idx]['file_path'])
    # print(all_captions[idx])
    # print(all_target_captions[n])
    # print(mods[n])
    
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(6, 6))
    # fig.add_subplot(3, 5, 3)
    # plt.imshow(s_img)
    # fig.add_subplot(3, 5, 8)
    # plt.imshow(t_img)
    # # img.show()
    # plt.show()



    # f1 = open('./dataset/Bonn_Furniture_Styles_Dataset/test_queries.txt')
    # f2 = open('./dataset/Bonn_Furniture_Styles_Dataset/my_test_queries.txt', 'w')
    # for line in f1.readlines():
    #     path_1, path_2 = line.split()
    #     if path_1.split('/')[1] == path_2.split('/')[1]:
    #         f2.write(line)