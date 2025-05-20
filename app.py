from fastai.vision.all import *
import gradio as gr

# Eğitilirken kullanılan custom label_func fonksiyonu
def label_func(x): return x.parent.name

# Modeli yükle
learn = load_learner('food_classifier.pkl')
labels = learn.dls.vocab

# Tahmin fonksiyonu
def predict(img):
    img = PILImage.create(img)
    _, _, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

# Gradio arayüzü
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Upload Food Image"),
    outputs=gr.Label(num_top_classes=5),
    title="🍽️ Food Image Classifier",
    description="Upload a photo of food, and this Fastai-based model will try to guess what it is!"
)

# Uygulama başlat
demo.launch()