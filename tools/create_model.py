import img_text_composition_models
import torch
import text_model
from tools import pkl

def create_model(opt):
    text = ["init query"]
    model = img_text_composition_models.TIRG(text, embed_dim=opt.embed_dim)
    model.text_model.embedding_layer = torch.nn.Embedding(5590, 512)
    vocab = pkl.pkl_load("./pkl/vocab.pkl")
    model.text_model.vocab = vocab

    if torch.cuda.is_available():
        checkpoint = torch.load('./models/checkpoint_fashion200k.pth')
    else:
        checkpoint = torch.load('./models/checkpoint_fashion200k.pth', \
                            map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# opt = {'embed_dim': 512}
# model = create_model(opt)
# mode.text_model