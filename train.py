import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from utils import save_model, load_model, print_examples
from get_loader import get_loader, FlickrDataset
from model import EncoderDecoder
import pickle
    
def train():

    #setting the constants
    data_location =  "flickr30k"
    BATCH_SIZE = 64
    NUM_WORKER = 8

    transforms = T.Compose([
        T.Resize(226),                     
        T.RandomCrop(224),                 
        T.ToTensor(),                               
        T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])

    #testing the dataset class
    dataset =  FlickrDataset(
        root_dir = data_location+"/images",
        captions_file = data_location+"/captions.txt",
        transform=transforms
    )

    #writing the dataloader
    data_loader = get_loader(
        dataset= dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        shuffle=True,
    )
    
    #vocab_size
    vocab_size = len(dataset.vocab)

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(dataset.vocab, f)
    
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint = False
    model_pth = 'attention_model_state.pth'
    vocab_pth = "vocab.pkl"
    
    #Hyperparams
    embed_size=300
    vocab_size = len(dataset.vocab)
    attention_dim=256
    encoder_dim=2048
    decoder_dim=512
    learning_rate = 3e-4

    # initialize model, loss etc
    model = EncoderDecoder(
        embed_size=300,
        vocab_size = len(dataset.vocab),
        attention_dim=256,
        encoder_dim=2048,
        decoder_dim=512,
        device=device
    ).to(device)

    vocab = dataset.vocab

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if load_checkpoint:
        model, vocab = load_model(model_pth, vocab_pth)

    model.train()

    num_epochs = 25
    print_every = 1000

    for epoch in range(1,num_epochs+1):   
        for idx, (image, captions) in tqdm(enumerate(data_loader), total=len(dataset)//BATCH_SIZE):
            image,captions = image.to(device),captions.to(device)
    
            # Zero the gradients.
            optimizer.zero_grad()
    
            # Feed forward
            outputs,attentions = model(image, captions)
    
            # Calculate the batch loss.
            targets = captions[:,1:]
            loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))
            
            # Backward pass.
            loss.backward()
    
            # Update the parameters in the optimizer.
            optimizer.step()
    
            if (idx+1)%print_every == 0:
                print("Epoch: {} loss: {:.5f}".format(epoch,loss.item()))
                print_examples(model, device, vocab)
                    
                model.train()
                
        #save the latest model
        save_model(model, model_pth, epoch, embed_size, vocab_size, attention_dim, encoder_dim, decoder_dim)


if __name__ == "__main__":
    train()