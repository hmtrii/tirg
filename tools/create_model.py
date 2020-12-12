import img_text_composition_models
import torch

def create_model(opt):
    text = ["init query"]

    model = img_text_composition_models.TIRG(text, embed_dim=opt.embed_dim)
    model.text_model.embedding_layer = torch.nn.Embedding(5590, 512)

    if torch.cuda.is_available():
        checkpoint = torch.load('./models/checkpoint_fashion200k.pth')
    else:
        checkpoint = torch.load('./models/checkpoint_fashion200k.pth', \
                            map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['model_state_dict'])
    return model