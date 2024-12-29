import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Scrollbar, Text, Canvas, Frame
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use("TkAgg")

# Model isimleri ve yolları
model_paths = {
    "Buğday": "C:/Users/emirf/OneDrive/Masaüstü/tezuygulama/bugday_siniflandirici.joblib",
    "Patates": "C:/Users/emirf/OneDrive/Masaüstü/tezuygulama/patates_siniflandirici.joblib",
    "Mısır": "C:/Users/emirf/OneDrive/Masaüstü/tezuygulama/misir_siniflandirici.joblib"
}

# Varsayılan model
current_model_path = model_paths["Buğday"]

# Model yükleme
try:
    model = joblib.load(current_model_path)
except Exception as e:
    messagebox.showerror("Hata", f"Model yüklenemedi: {str(e)}")
    exit()

def goruntu_ozellikleri(goruntu_yolu):
    goruntu = cv2.imread(str(goruntu_yolu))
    if goruntu is None:
        raise ValueError(f"Görüntü okunamadı: {goruntu_yolu}")
    goruntu = cv2.resize(goruntu, (200, 200))
    hsv = cv2.cvtColor(goruntu, cv2.COLOR_BGR2HSV)
    ortalama_hsv = cv2.mean(hsv)
    gri = cv2.cvtColor(goruntu, cv2.COLOR_BGR2GRAY)
    kenarlar = cv2.Canny(gri, 100, 200)
    kenar_yogunlugu = np.sum(kenarlar) / (200 * 200)
    gri = cv2.GaussianBlur(gri, (7, 7), 0)
    glcm = np.zeros((256, 256), dtype=np.uint32)
    for i in range(gri.shape[0]-1):
        for j in range(gri.shape[1]-1):
            i_intensity = gri[i, j]
            j_intensity = gri[i, j+1]
            glcm[i_intensity, j_intensity] += 1
    glcm = glcm / np.sum(glcm)
    contrast = np.sum(np.square(np.arange(256)[:, np.newaxis] - np.arange(256)) * glcm)
    ozellikler = np.array([
        ortalama_hsv[0], ortalama_hsv[1], ortalama_hsv[2],
        kenar_yogunlugu,
        contrast
    ])
    return ozellikler

def modeli_degistir(event):
    global model, current_model_path
    secilen_model = model_menu.get()
    current_model_path = model_paths[secilen_model]
    try:
        model = joblib.load(current_model_path)
        messagebox.showinfo("Başarılı", f"Model değiştirildi: {secilen_model}")
    except Exception as e:
        messagebox.showerror("Hata", f"Model yüklenemedi: {str(e)}")

def goruntu_sec():
    dosya_yolu = filedialog.askopenfilename(
        title="Görüntü Seç",
        filetypes=[("Resim Dosyaları", "*.jpg *.jpeg *.png")]
    )
    if dosya_yolu:
        try:
            ozellikler = goruntu_ozellikleri(dosya_yolu)
            tahmin = model.predict([ozellikler])[0]
            olasiliklar = model.predict_proba([ozellikler])[0]
            durum = "Sağlıklı" if tahmin == 1 else "Sağlıksız"
            guven = olasiliklar[1] if tahmin == 1 else olasiliklar[0]
            sonuc_label.config(text=f"Durum: {durum}\nGüven: {guven:.2f}", 
                               fg="green" if tahmin == 1 else "red")
            saglikli_olasılık = olasiliklar[1]
            sagliksiz_olasılık = olasiliklar[0]
            update_graph([sagliksiz_olasılık, saglikli_olasılık])

            img = Image.open(dosya_yolu)
            img = img.resize((200, 200))
            imgtk = ImageTk.PhotoImage(img)
            image_label.config(image=imgtk)
            image_label.image = imgtk

            # Özellikler bölümü
            ozellik_text.config(state=tk.NORMAL)
            ozellik_text.delete(1.0, tk.END)
            ozellik_text.insert(tk.END, f"Özellikler:\nHSV: {ozellikler[0]:.2f}, {ozellikler[1]:.2f}, {ozellikler[2]:.2f}\nKenar Yoğunluğu: {ozellikler[3]:.4f}\nKontrast: {ozellikler[4]:.4f}\nSağlıklı Oranı: {saglikli_olasılık:.2f}\nSağlıksız Oranı: {sagliksiz_olasılık:.2f}")
            ozellik_text.config(state=tk.DISABLED)
            # Pencereyi boyutlandır
            root.update_idletasks()
            canvas_frame.configure(scrollregion=canvas_frame.bbox("all"))
        except Exception as e:
            messagebox.showerror("Hata", str(e))

def update_graph(data):
    ax.clear()
    labels = ["Sağlıksız", "Sağlıklı"]
    colors = ["red", "green"]
    ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title("Sınıf Olasılık Dağılımı")
    canvas.draw()

root = tk.Tk()
root.title("Bitki Hastalık Sınıflandırma")
root.geometry("600x650")
root.configure(bg="#f5f5f5")

main_frame = Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)
canvas_frame = Canvas(main_frame)
canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
scrollbar = Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas_frame.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
canvas_frame.configure(yscrollcommand=scrollbar.set)
canvas_frame.bind('<Configure>', lambda e: canvas_frame.configure(scrollregion=canvas_frame.bbox("all")))

frame = Frame(canvas_frame)
canvas_frame.create_window((0, 0), window=frame, anchor="nw")

model_menu = ttk.Combobox(frame, values=list(model_paths.keys()), state="readonly", font=("Arial", 14))
model_menu.current(0)
model_menu.grid(row=0, column=0, pady=10, padx=10)
model_menu.bind("<<ComboboxSelected>>", modeli_degistir)

sec_button = tk.Button(frame, text="Görüntü Seç", command=goruntu_sec, bg="#4CAF50", fg="white", font=("Arial", 14))
sec_button.grid(row=1, column=0, pady=10)

sonuc_label = tk.Label(frame, text="Henüz bir görüntü seçilmedi", font=("Arial", 16), bg="#f5f5f5", fg="#333")
sonuc_label.grid(row=2, column=0, pady=10)

image_label = tk.Label(frame, bg="#f5f5f5")
image_label.grid(row=3, column=0, pady=10)

ozellik_text = Text(frame, wrap=tk.WORD, height=10, width=70)
ozellik_text.grid(row=4, column=0, pady=10)
ozellik_text.config(state=tk.DISABLED)

fig = Figure(figsize=(6, 4), dpi=100)
ax = fig.add_subplot(111)
ax.set_title("Sınıf Olasılık Dağılımı")
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().grid(row=5, column=0, pady=20)
canvas.draw()

root.mainloop()
