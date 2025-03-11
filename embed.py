import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer


text_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)


def Encode_Text(text):
    return text_model.encode(text, convert_to_tensor=True)


def Encode_Image(image_path):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
    return image_features.squeeze().cpu().numpy()  

if __name__ == "__main__":
    text = "This is a sample caption for an image."
    image_path = "test_data\667626_18933d713e.jpg"  

    text_embedding = Encode_Text(text)
    image_embedding = Encode_Image(image_path)

    # Display the shape of embeddings
    print("Text Embedding Shape:", text_embedding.shape)
    print("Image Embedding Shape:", image_embedding.shape)
