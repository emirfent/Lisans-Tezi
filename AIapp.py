import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

matplotlib.use("TkAgg")

def goruntu_ozellikleri(goruntu_yolu):
    """
    Görüntüden özellik çıkarır.
    Özellikler: Ort. HSV (H,S,V), Kenar Yoğunluğu, Kontrast
    """
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

try:
    model = joblib.load("patates_siniflandirici.joblib")
except Exception as e:
    messagebox.showerror("Hata", f"Model yüklenemedi: {str(e)}")
    exit()

def hakkinda():
    messagebox.showinfo("Hakkında", 
                        "Bu uygulama, yaprak görüntülerini sağlıklı veya sağlıksız olarak sınıflandırır.\n"
                        "Daha önce eğitilmiş bir SVM modelini kullanır.\n\n"
                        "Yazar: [Sizin İsminiz]")

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

            # Label'ları güncelle
            sonuc_label.config(text=f"Durum: {durum}\nGüven: {guven:.2f}", 
                               fg="green" if tahmin == 1 else "red")

            saglikli_olasılık = olasiliklar[1]
            sagliksiz_olasılık = olasiliklar[0]

            ozellik_text = (
                f"Detaylı Görüntü Özellikleri:\n\n"
                f"- Ortalama HSV:\n"
                f"  H: {ozellikler[0]:.2f}, S: {ozellikler[1]:.2f}, V: {ozellikler[2]:.2f}\n"
                f"- Kenar Yoğunluğu: {ozellikler[3]:.4f} \n"
                f"- Kontrast: {ozellikler[4]:.4f} \n\n"
                f"Tahmin Sonuçları:\n"
                f"- Sağlıklı Olasılık: {saglikli_olasılık:.2f}\n"
                f"- Sağlıksız Olasılık: {sagliksiz_olasılık:.2f}\n"
            )

            if tahmin == 0:  # Sağlıksız durum için öneri ekle
                ozellik_text += (
                    "\nErken Yanıklık Tespit Edildi!\n"
                    "- Yaprağı daha ayrıntılı bir inceleme için bir uzmana gösterin.\n"
                    
                    "\nÇözüm Önerileri:\n"
                    "- Dirençli Patates Çeşitleri Dikimi\n"
                    "- Dönüşümlü Ekim Uygulaması.\n"
                    "- Doğru Sulama ve Havalandırma.\n"
                    "- Fungisit Kullanımı.\n"
                    "- Tarımsal Hijyen ve Artık Yönetimi.\n"
                    "- Yukarıdaki yöntemler kullanılarak bu bitki hastalığının önüne geçebilirsiniz.\n"
                )

            ozellik_textbox.config(state='normal')
            ozellik_textbox.delete(1.0, tk.END)
            ozellik_textbox.insert(tk.END, ozellik_text)
            ozellik_textbox.config(state='disabled')

            img = Image.open(dosya_yolu)
            img = img.resize((400, 400))
            imgtk = ImageTk.PhotoImage(img)
            image_label.config(image=imgtk)
            image_label.image = imgtk

            update_graph([sagliksiz_olasılık, saglikli_olasılık])

        except Exception as e:
            messagebox.showerror("Hata", str(e))

def update_graph(data):
    # data [Sağlıksız_olasılık, Sağlıklı_olasılık]
    ax.clear()
    labels = ["Sağlıksız", "Sağlıklı"]
    colors = ["red", "green"]
    ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title("Sınıf Olasılık Dağılımı")
    canvas.draw()

# Ana pencere
root = tk.Tk()
root.title("Yaprak Sınıflandırma")
root.geometry("650x650")
root.configure(bg="#f5f5f5")

# Kaydırma çubuğu eklemek için çerçeve
main_frame = tk.Frame(root, bg="#f5f5f5")
main_frame.pack(fill=tk.BOTH, expand=1)

canvas_frame = tk.Canvas(main_frame, bg="#f5f5f5")
canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas_frame.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

canvas_frame.configure(yscrollcommand=scrollbar.set)
canvas_frame.bind('<Configure>', lambda e: canvas_frame.configure(scrollregion=canvas_frame.bbox("all")))

content_frame = tk.Frame(canvas_frame, bg="#f5f5f5")
canvas_frame.create_window((0, 0), window=content_frame, anchor="nw")

# Menü çubuğu
menubar = tk.Menu(root)
menubar.add_command(label="Hakkında", command=hakkinda)
root.config(menu=menubar)

# Ana çerçeve
frame = tk.Frame(content_frame, bg="#f5f5f5")
frame.pack(pady=20, padx=20, side=tk.TOP)

# Görüntü seç butonu
sec_button = tk.Button(frame, text="Görüntü Seç", command=goruntu_sec, bg="#4CAF50", fg="white", font=("Arial", 14), padx=20, pady=10)
sec_button.grid(row=0, column=0, padx=20, pady=20, sticky="ew")

# Sonuç etiketi
sonuc_label = tk.Label(frame, text="Henüz bir görüntü seçilmedi", font=("Arial", 16), bg="#f5f5f5", fg="#333")
sonuc_label.grid(row=1, column=0, padx=20, pady=20)

# Özellik çerçevesi
ozellik_frame = tk.LabelFrame(frame, text="Özellikler", font=("Arial", 14), bg="#f5f5f5", fg="#333", padx=10, pady=10)
ozellik_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")

ozellik_textbox = tk.Text(ozellik_frame, width=50, height=15, state='disabled', font=("Arial", 12), bg="#ffffff", fg="#333")
ozellik_textbox.pack()

# Görüntü etiketi
image_label = tk.Label(frame, bg="#f5f5f5")
image_label.grid(row=3, column=0, padx=20, pady=20)

# Grafik çerçevesi
graph_frame = tk.Frame(content_frame, bg="#f5f5f5")
graph_frame.pack(pady=20, padx=20, side=tk.BOTTOM)

fig = Figure(figsize=(6, 4), dpi=100)
ax = fig.add_subplot(111)
ax.set_title("Sınıf Olasılık Dağılımı")
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.draw()
canvas.get_tk_widget().pack()

root.mainloop()
