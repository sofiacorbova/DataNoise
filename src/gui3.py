import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import torch
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from my_transform import AddGaussianNoise, AddPoissonNoise, AddSaltPepperNoise
import os

model = None
class_names = [f"class {i}" for i in range(37)]  # Alebo načítaj podľa tvojej úlohy
transform_base = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("1200x800")
app.title("DreamTeamPopiciGUInajs")

def get_class_names_from_list(filepath):
    class_id_to_name = {}

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_name = parts[0]
            class_id = int(parts[1]) - 1  # Zniž na indexovanie od 0
            if class_id not in class_id_to_name:
                # Odstráni číselnú časť názvu
                class_name = "_".join(image_name.split("_")[:-1])
                class_id_to_name[class_id] = class_name

    # Výsledok zoradený podľa class_id
    return [name for _, name in sorted(class_id_to_name.items())]

def get_true_label_from_list(filepath, image_filename):
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] in image_filename:
                return int(parts[1]) - 1
    return None  # Ak sa nenašla


def load_model_file():
    global model
    filepath = filedialog.askopenfilename(filetypes=[("Model files", "*.pth")])
    if not filepath:
        return

    checkpoint = torch.load(filepath, map_location='cpu')

    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, len(class_names))
    )

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model_status_label.configure(text=f"Model načítaný: {os.path.basename(filepath)}", text_color="green")


input_img_tk = None
input_img_pil = None  # ← pridaj originál
# nacianie obrazka
def nacitaj_obrazok():
    global input_img_tk, input_img_pil, input_true_label

    filepath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
    if not filepath:
        return
    
    filename = os.path.basename(filepath).split('.')[0]
    input_true_label = get_true_label_from_list("data/oxford-iiit-pet/annotations/list.txt", filename)

    img = Image.open(filepath).convert("RGB")
    input_img_pil = img.copy()  # ← uložíme originál PIL.Image

    resized = img.resize((480, 360), Image.LANCZOS)
    input_img_tk = ImageTk.PhotoImage(resized)
    input_image_label.configure(image=input_img_tk)
    input_image_label.image = input_img_tk


def predict():
    global input_img_pil, model

    if input_img_pil is None or model is None:
        print("❌ Najprv načítaj obrázok a model.")
        return

    img_pil = input_img_pil.resize((224, 224)).convert("RGB")
    img_tensor = transforms.ToTensor()(img_pil)  # rozsah [0, 1]

    # === Pridaj šum podľa sliderov ===
    mean = slider1.get() / 100.0
    std = slider2.get() / 100.0
    lam = slider3.get()
    salt = slider4.get() / 100.0
    pepper = slider5.get() / 100.0
    total_sp = min(1.0, salt + pepper)

    # Pridaj šum iba ak je nenulový
    if std > 0:
        img_tensor = AddGaussianNoise(mean=mean, std=std)(img_tensor)
    if lam > 0:
        img_tensor = AddPoissonNoise(lam=lam)(img_tensor)
    if total_sp > 0:
        img_tensor = AddSaltPepperNoise(amount=total_sp)(img_tensor)

    # Pre model normalizuj (musí byť až po šume!)
    input_tensor = transforms.Normalize([0.5]*3, [0.5]*3)(img_tensor).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1).squeeze()
        top_prob, top_idx = torch.max(probs, 0)

    predicted_label = class_names[top_idx]
    confidence = top_prob.item() * 100

    # Obrázok na výstup – späť do [0, 1] (neaplikuj normalizáciu)
    out_img = transforms.ToPILImage()(img_tensor.clamp(0, 1))
    out_img = out_img.resize((480, 360), Image.LANCZOS)
    output_img_tk = ImageTk.PhotoImage(out_img)
    output_image_label.configure(image=output_img_tk)
    output_image_label.image = output_img_tk

    true_label_name = class_names[input_true_label] if input_true_label is not None else "neznáma"
    prediction_label.configure(text=f"Skutočná: {true_label_name} | Predikcia: {predicted_label} ({confidence:.1f}%)")


top_control_frame = ctk.CTkFrame(app)
top_control_frame.pack(fill="x", padx=20, pady=(20, 5))

load_button = ctk.CTkButton(top_control_frame, text="Load Image", command=nacitaj_obrazok)
load_button.grid(row=0, column=0, padx=10)

load_model_button = ctk.CTkButton(top_control_frame, text="Load Model", command=load_model_file)
load_model_button.grid(row=0, column=1, padx=10)

model_status_label = ctk.CTkLabel(top_control_frame, text="Model nenačítaný", text_color="red", font=ctk.CTkFont(size=14))
model_status_label.grid(row=0, column=2, padx=20)


main_frame = ctk.CTkFrame(app)
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=0)
main_frame.grid_columnconfigure(2, weight=1)
main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_rowconfigure(1, weight=0)
main_frame.grid_rowconfigure(2, weight=0) 

# vystupny obr
input_frame = ctk.CTkFrame(main_frame, fg_color="#333333", width=400, height=400)
input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
input_frame.grid_propagate(False)
input_frame.grid_rowconfigure(1, weight=1)
input_frame.grid_columnconfigure(0, weight=1)

input_label = ctk.CTkLabel(input_frame, text="Input image", font=ctk.CTkFont(size=16))
input_label.grid(row=0, column=0, pady=10)

input_image_label = ctk.CTkLabel(input_frame, text="")
input_image_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

# vystupny obrazok
output_frame = ctk.CTkFrame(main_frame, fg_color="#333333", width=400, height=400)
output_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
output_frame.grid_propagate(False)
output_frame.grid_rowconfigure(1, weight=1)
output_frame.grid_columnconfigure(0, weight=1)

output_label = ctk.CTkLabel(output_frame, text="Output image", font=ctk.CTkFont(size=16))
output_label.grid(row=0, column=0, pady=10)

output_image_label = ctk.CTkLabel(output_frame, text="")
output_image_label.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

slider_block_frame = ctk.CTkFrame(app, width=700, height=400)
slider_block_frame.pack(side="bottom", pady=20, padx=20)  

# grrid pre sliddere
slider_block_frame.grid_columnconfigure(0, weight=1)
slider_block_frame.grid_columnconfigure(1, weight=1)
slider_block_frame.grid_columnconfigure(2, weight=1)

# ukazovatel hosnoty slidera
def update_value(slider, value_label):
    value = round(slider.get(), 1)  
    value_label.configure(text=str(value))

column1_label = ctk.CTkLabel(slider_block_frame, text="Gaussian Noise", font=ctk.CTkFont(size=14))
column1_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
# gausovsky sum
slider1_label = ctk.CTkLabel(slider_block_frame, text="μ", font=ctk.CTkFont(size=12))
slider1_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

slider1 = ctk.CTkSlider(slider_block_frame, from_=0, to=100)
slider1.set(0.0)
slider1.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

slider1_value = ctk.CTkLabel(slider_block_frame, text="0.0", font=ctk.CTkFont(size=12))
slider1_value.grid(row=2, column=1, padx=10, pady=5)

slider1.configure(command=lambda value: update_value(slider1, slider1_value))

slider2_label = ctk.CTkLabel(slider_block_frame, text="σ", font=ctk.CTkFont(size=12))
slider2_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")

slider2 = ctk.CTkSlider(slider_block_frame, from_=0, to=100)
slider2.set(0.0)
slider2.grid(row=4, column=0, padx=10, pady=5, sticky="ew")

slider2_value = ctk.CTkLabel(slider_block_frame, text="0.0", font=ctk.CTkFont(size=12))
slider2_value.grid(row=4, column=1, padx=10, pady=5)

slider2.configure(command=lambda value: update_value(slider2, slider2_value))

# poison sum
column2_label = ctk.CTkLabel(slider_block_frame, text="Poison Noise", font=ctk.CTkFont(size=14))
column2_label.grid(row=0, column=2, padx=10, pady=5, sticky="w")

slider3_label = ctk.CTkLabel(slider_block_frame, text="λ", font=ctk.CTkFont(size=12))
slider3_label.grid(row=1, column=2, padx=10, pady=5, sticky="w")

slider3 = ctk.CTkSlider(slider_block_frame, from_=0, to=100)
slider3.set(0.0)
slider3.grid(row=2, column=2, padx=10, pady=5, sticky="ew")

slider3_value = ctk.CTkLabel(slider_block_frame, text="0.0", font=ctk.CTkFont(size=12))
slider3_value.grid(row=2, column=3, padx=10, pady=5)

slider3.configure(command=lambda value: update_value(slider3, slider3_value))

# salat and pepper sum
column3_label = ctk.CTkLabel(slider_block_frame, text="Salt-and-Pepper Noise", font=ctk.CTkFont(size=14))
column3_label.grid(row=0, column=4, padx=10, pady=5, sticky="w")

slider4_label = ctk.CTkLabel(slider_block_frame, text="salt_prob", font=ctk.CTkFont(size=12))
slider4_label.grid(row=1, column=4, padx=10, pady=5, sticky="w")

slider4 = ctk.CTkSlider(slider_block_frame, from_=0, to=100)
slider4.set(0.0)
slider4.grid(row=2, column=4, padx=10, pady=5, sticky="ew")

slider4_value = ctk.CTkLabel(slider_block_frame, text="0.0", font=ctk.CTkFont(size=12))
slider4_value.grid(row=2, column=5, padx=10, pady=5)

slider4.configure(command=lambda value: update_value(slider4, slider4_value))

slider5_label = ctk.CTkLabel(slider_block_frame, text="pepper_prob", font=ctk.CTkFont(size=12))
slider5_label.grid(row=3, column=4, padx=10, pady=5, sticky="w")

slider5 = ctk.CTkSlider(slider_block_frame, from_=0, to=100)
slider5.set(0.0)
slider5.grid(row=4, column=4, padx=10, pady=5, sticky="ew")

slider5_value = ctk.CTkLabel(slider_block_frame, text="0.0", font=ctk.CTkFont(size=12))
slider5_value.grid(row=4, column=5, padx=10, pady=5)

slider5.configure(command=lambda value: update_value(slider5, slider5_value))

predict_button = ctk.CTkButton(slider_block_frame, text="Predict", command=predict)
predict_button.grid(row=5, column=2, columnspan=2, pady=10, padx=20, sticky="ew")

class_names = get_class_names_from_list("data/oxford-iiit-pet/annotations/list.txt")

prediction_label = ctk.CTkLabel(output_frame, text="", font=ctk.CTkFont(size=16))
prediction_label.grid(row=2, column=0, pady=10)


app.mainloop()
