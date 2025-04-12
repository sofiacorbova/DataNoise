import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("1200x800")
app.title("DreamTeamPopiciGUInajs")

input_img_tk = None 
# nacianie obrazka
def nacitaj_obrazok():
    global input_img_tk

    filepath = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
    if not filepath:
        return

    img = Image.open(filepath)
    img = img.resize((480, 360), Image.LANCZOS)
    input_img_tk = ImageTk.PhotoImage(img)

    input_image_label.configure(image=input_img_tk)

def predict():
    print("Predikcia spustená!")

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

# nacitanie obr tlacitko
load_button = ctk.CTkButton(input_frame, text="Load Image", command=nacitaj_obrazok)
load_button.grid(row=2, column=0, pady=10)

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
slider1.grid(row=2, column=0, padx=10, pady=5, sticky="ew")

slider1_value = ctk.CTkLabel(slider_block_frame, text="0.0", font=ctk.CTkFont(size=12))
slider1_value.grid(row=2, column=1, padx=10, pady=5)

slider1.configure(command=lambda value: update_value(slider1, slider1_value))

slider2_label = ctk.CTkLabel(slider_block_frame, text="σ", font=ctk.CTkFont(size=12))
slider2_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")

slider2 = ctk.CTkSlider(slider_block_frame, from_=0, to=100)
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
slider4.grid(row=2, column=4, padx=10, pady=5, sticky="ew")

slider4_value = ctk.CTkLabel(slider_block_frame, text="0.0", font=ctk.CTkFont(size=12))
slider4_value.grid(row=2, column=5, padx=10, pady=5)

slider4.configure(command=lambda value: update_value(slider4, slider4_value))

slider5_label = ctk.CTkLabel(slider_block_frame, text="pepper_prob", font=ctk.CTkFont(size=12))
slider5_label.grid(row=3, column=4, padx=10, pady=5, sticky="w")

slider5 = ctk.CTkSlider(slider_block_frame, from_=0, to=100)
slider5.grid(row=4, column=4, padx=10, pady=5, sticky="ew")

slider5_value = ctk.CTkLabel(slider_block_frame, text="0.0", font=ctk.CTkFont(size=12))
slider5_value.grid(row=4, column=5, padx=10, pady=5)

slider5.configure(command=lambda value: update_value(slider5, slider5_value))

predict_button = ctk.CTkButton(slider_block_frame, text="Predict", command=predict)
predict_button.grid(row=5, column=2, columnspan=2, pady=10, padx=20, sticky="ew")

app.mainloop()
