import torch
import torchvision.transforms as transforms
from PIL import Image
import pickle
from model import EncoderDecoder

def print_examples(model, device, vocab):
    transform = transforms.Compose([
        transforms.Resize(226),                     
        transforms.RandomCrop(224),                 
        transforms.ToTensor(),                               
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])    

    model.eval()
    test_img1 = transform(Image.open("test_examples/dog.jpg").convert("RGB")).unsqueeze(0)
    features = model.encoder(test_img1.to(device))
    caps, alphas = model.decoder.generate_caption(features, vocab=vocab)
    caption = ' '.join(caps)
    print("Example 1 CORRECT: Dog on a beach by the ocean")
    print(
        "Example 1 OUTPUT: "
        + caption
    )
    
    test_img2 = transform(
        Image.open("test_examples/child.jpg").convert("RGB")
    ).unsqueeze(0)
    features = model.encoder(test_img2.to(device))
    caps,alphas = model.decoder.generate_caption(features, vocab=vocab)
    caption = ' '.join(caps)
    print("Example 2 CORRECT: Child holding red frisbee outdoors")
    print(
        "Example 2 OUTPUT: "
        + caption
    )
    
    test_img3 = transform(Image.open("test_examples/bus.png").convert("RGB")).unsqueeze(0)
    features = model.encoder(test_img3.to(device))
    caps,alphas = model.decoder.generate_caption(features, vocab=vocab)
    caption = ' '.join(caps)
    print("Example 3 CORRECT: Bus driving by parked cars")
    print(
        "Example 3 OUTPUT: "
        + caption
    )
    
    test_img4 = transform(
        Image.open("test_examples/boat.png").convert("RGB")
    ).unsqueeze(0)
    features = model.encoder(test_img4.to(device))
    caps,alphas = model.decoder.generate_caption(features, vocab=vocab)
    caption = ' '.join(caps)
    print("Example 4 CORRECT: A small boat in the ocean")
    print(
        "Example 4 OUTPUT: "
        + caption
    )
    
    test_img5 = transform(
        Image.open("test_examples/horse.png").convert("RGB")
    ).unsqueeze(0)
    features = model.encoder(test_img5.to(device))
    caps,alphas = model.decoder.generate_caption(features, vocab=vocab)
    caption = ' '.join(caps)
    print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    print(
        "Example 5 OUTPUT: "
        + caption
    )
    model.train()


def save_model(model, model_pth, num_epochs, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim):
    model_state = {
        'num_epochs':num_epochs,
        'embed_size':embed_size,
        'vocab_size':vocab_size,
        'attention_dim':attention_dim,
        'encoder_dim':encoder_dim,
        'decoder_dim':decoder_dim,
        'state_dict':model.state_dict()
    }

    torch.save(model_state, model_pth)

def load_model(path, vocab_pth):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_state = torch.load(path, map_location=torch.device(device))
    
    embed_size=model_state['embed_size']
    vocab_size = model_state['vocab_size']
    attention_dim=model_state['attention_dim']
    encoder_dim=model_state['encoder_dim']
    decoder_dim=model_state['decoder_dim']
    
    model = EncoderDecoder(
                embed_size=embed_size,
                vocab_size = vocab_size,
                attention_dim=attention_dim,
                encoder_dim=encoder_dim,
                decoder_dim=decoder_dim,
                device=device
            ).to(device)
    model.load_state_dict(model_state["state_dict"])
    model.eval()

    with open(vocab_pth, "rb") as input_file:
        vocab = pickle.load(input_file)

    return model, vocab