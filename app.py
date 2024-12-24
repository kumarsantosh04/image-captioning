import streamlit as st
import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
import pickle
import os
from PIL import Image 
import torch
import torchvision.transforms as T
from model import EncoderDecoder
import gdown

# Set custom web page title
st.set_page_config(page_title="Caption Generator App", page_icon="📷", layout="wide")
MODEL_URL = 'https://drive.google.com/file/d/1gQUwVrO45R4REWFR_EXwhxmEEP9Zd33m/view'
VOCAB_URL = 'https://drive.google.com/file/d/1eJN5ip8bauOakbO9nDdqPS3dc6q5Dtxm/view'

@st.cache_data
def loading_model(path, vocab):
    gdown.download(MODEL_URL, path, quiet=False, fuzzy=True)
    gdown.download(VOCAB_URL, vocab, quiet=False, fuzzy=True)
    
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

    with open(vocab, "rb") as input_file:
        vocab = pickle.load(input_file)

    return model, device, vocab

model, device, vocab = loading_model('attention_model_state.pth', 'vocab.pkl')

upload_area, result_area = st.columns(2)
attention_plots = st.expander(label='Attention Plots:  Model focus areas while captioning')

with upload_area:
    # Streamlit app
    st.title("Image Caption Generator")
    st.markdown(
        "This app will generate a caption for uploaded image using a attention based trained LSTM model [encoder-decoder]."
    )
    
    # Upload image
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Process uploaded image
if uploaded_image is not None:
    # Display loading spinner while processing
    with st.spinner("Generating caption..."):
        print(uploaded_image)

        img = Image.open(uploaded_image).convert("RGB")

        #defining the transform to be applied
        transforms = T.Compose([
            T.Resize((224,224)),                     
            # T.RandomCrop(224),                 
            T.ToTensor(),                               
            T.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        ])
        img_tranformed = transforms(img).unsqueeze(0)

        def get_caps_from(features_tensors):
            #generate the caption
            print("Generating")
            model.eval()
            with torch.no_grad():
                features = model.encoder(features_tensors.to(device))
                caps,alphas = model.decoder.generate_caption(features, vocab=vocab, max_len=50)
                caption = ' '.join(caps)
            
            return caption.replace("<SOS>", "").replace("<EOS>", "").replace("<PAD>", "").replace("<UNK>", "unknown").strip(" "), alphas

        #Show attention
        def plot_attention(img, caption, attention_mask):
            #untransform
            img = img.squeeze(0)
            img[0] = img[0] * 0.229
            img[1] = img[1] * 0.224 
            img[2] = img[2] * 0.225 
            img[0] += 0.485 
            img[1] += 0.456 
            img[2] += 0.406
            
            img = img.numpy().transpose((1, 2, 0))
            temp_image = img
        
            fig = plt.figure(figsize=(15, 15))

            caption = caption.split(" ")
            cap_len = len(caption)
            for l in range(cap_len):
                temp_att = attention_mask[l].reshape(7,7)
                temp_att = scipy.ndimage.zoom(temp_att, 224//7, order=3)
                ax = fig.add_subplot(cap_len//2,cap_len//2, l+1)
                ax.set_title(caption[l])
                img = ax.imshow(temp_image)
                ax.imshow(temp_att, cmap='gray', alpha=0.8, extent=img.get_extent())
                
            plt.tight_layout()
            st.pyplot(fig)

        generated_caption, attn_mask = get_caps_from(img_tranformed)

    with upload_area:
        if st.button("Regenerate ⟳", type="primary"):
            generated_caption, attn_mask = get_caps_from(img_tranformed)
            print(generated_caption)
        st.code(generated_caption, language="markdown")
        
            
    with result_area:
        st.image(uploaded_image, caption=generated_caption)
        
        
    with attention_plots:
        plot_attention(img_tranformed, generated_caption, attn_mask)
        