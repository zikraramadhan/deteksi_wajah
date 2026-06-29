import cv2
import numpy as np
from flask import Flask, render_template, request
from mtcnn import MTCNN
from PIL import Image
import io
import base64

app = Flask(__name__)

# Inisialisasi detektor MTCNN (sekali saat aplikasi berjalan)
detector = MTCNN()

def deteksi_dan_gambar_wajah(image_bytes):
    """
    Fungsi untuk mendeteksi wajah menggunakan MTCNN,
    menggambar bounding box merah di setiap wajah,
    lalu mengembalikan gambar hasil (base64).
    """
    # Baca gambar dari bytes menggunakan PIL, lalu konversi ke RGB
    pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Konversi ke numpy array untuk OpenCV (BGR)
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Deteksi wajah dengan MTCNN
    faces = detector.detect_faces(opencv_image)
    jumlah_wajah = len(faces)

    # Gambar bounding box MERAH untuk setiap wajah
    for face in faces:
        x, y, w, h = face['box']
        x, y = abs(x), abs(y)
        cv2.rectangle(opencv_image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Merah

    # Tidak menambahkan teks jumlah di pojok kiri atas (dihilangkan)

    # Konversi ke base64 untuk dikirim ke template
    _, buffer = cv2.imencode('.jpeg', opencv_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64, jumlah_wajah

@app.route('/')
def index():
    """Halaman utama: form upload gambar."""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Menangani upload gambar, menjalankan deteksi, dan menampilkan hasil."""
    if 'image' not in request.files:
        return render_template('index.html', error='Tidak ada file yang diunggah.')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error='Nama file kosong.')

    # Validasi ekstensi (opsional)
    allowed_extensions = {'jpeg'}
    if '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return render_template('index.html', error='Format file tidak didukung. Gunakan PNG')

    try:
        image_bytes = file.read()
        img_base64, jumlah = deteksi_dan_gambar_wajah(image_bytes)
        return render_template('result.html', 
                               result_image=img_base64, 
                               jumlah_wajah=jumlah)
    except Exception as e:
        return render_template('index.html', error=f'Gagal memproses gambar: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)