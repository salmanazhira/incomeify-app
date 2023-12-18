from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('SalaryModel.h5')

# Mapping dictionary
career_level_mapping = {
 'Supervisor/Coordinator': 4,
 'Employee (non-management & non-supervisor)': 3,
 'Manager/Assistant Manager': 2,
 'Fresh Graduate/Less than 1 year experience': 1,
 'CEO/General Manager/Director/Senior Manager': 0
}

location_mapping = {
 'Banten': 13, 'Jakarta Pusat': 47, 'Surabaya': 148, 'Tangerang': 151,
 'Jakarta Timur': 50, 'Jakarta Barat': 46, 'Jakarta Raya': 48, 'Batam': 16,
 'Jawa Barat': 53, 'Jawa Timur': 55, 'Aceh': 0, 'Jakarta Selatan': 49,
 'Cilegon': 33, 'Sumatera Utara': 147, 'Jakarta Utara': 51, 'Semarang': 132,
 'Jambi': 52, 'Cikarang': 30, 'Bali': 107, 'Palembang': 129,
 'Riau': 114, 'Pasuruan': 142, 'Sulawesi Tengah': 6, 'Bandung': 105,
 'Padang': 18, 'Bekasi': 146, 'Sumatera Selatan': 128,
 'Rangkasbitung': 162, 'Yogyakarta': 28, 'Cianjur': 11,
 'Banjarmasin': 65, 'Kediri': 88, 'Makassar': 131, 'Samarinda': 12,
 'Banjarnegara': 36, 'Cirebon': 116, 'Pekanbaru': 5,
 'Bandar Lampung': 24, 'Bogor': 59, 'Kalimantan Barat': 95, 'Maros': 138,
 'Sleman': 102, 'Nunukan': 39, 'Depok': 97, 'Medan': 64, 'Karawang': 139,
 'Sukabumi': 54, 'Jawa Tengah': 83, 'Lhokseumawe': 38,
 'Denpasar': 136, 'Sidoarjo': 150, 'Tanah Bumbu': 71, 'Klungkung': 156,
 'Tegal': 82, 'Lampung': 149, 'Surakarta': 104, 'Nusa Tenggara Timur': 89,
 'Malang': 144, 'Sulawesi Utara': 153, 'Tanjung Pinang': 93,
 'Manado': 31, 'Cikupa': 19, 'Bengkulu': 140, 'Sulawesi Barat': 145,
 'Sumatera Barat': 79, 'Kuta': 63, 'Kalimantan Utara': 4,
 'Balikpapan': 134, 'Serang': 2, 'Badung': 121, 'Pontianak': 108,
 'Palopo': 29, 'Cibinong': 96, 'Mataram': 103, 'Nusa Tenggara Barat': 101,
 'Muara Enim': 109, 'Palu': 9, 'Banjar': 141, 'Sulawesi Selatan': 100,
 'Mojokerto': 40, 'Dumai': 157, 'Ternate': 126, 'Purwokerto': 60,
 'Kalimantan Selatan': 87, 'Gresik': 152, 'Magelang': 1,
 'Tanjung Balai': 43, 'Ambon': 7, 'Bangka': 58, 'Jepara': 115,
 'Pekalongan': 125, 'Purwakarta': 32, 'Cilacap': 44, 'Halmahera': 133,
 'Seminyak': 110, 'Pandeglang': 14, 'Bantul': 62,
 'Kalimantan Timur': 42, 'Gowa': 20, 'Binjai': 41, 'Gorontalo': 35,
 'Cimahi': 61, 'Kalimantan Tengah': 75, 'Kotawaringin Timur': 123,
 'Prabumulih': 56, 'Jayapura': 37, 'Citeureup': 27, 'Bukittinggi': 122,
 'Poso': 117, 'Pemalang': 112, 'Papua': 70, 'Klaten': 158, 'Timika': 155,
 'Tasikmalaya': 57, 'Jember': 77, 'Kulon Progo': 15,
 'Banyuwangi': 119, 'Penajam Paser Utara': 26, 'Brebes': 86,
 'Madura': 72, 'Kota Banda Aceh': 135, 'Sibolga': 106,
 'Palangkaraya': 127, 'Purworejo': 10, 'Banjarbaru': 69, 'Ketapang': 154,
 'Tarakan': 76, 'Kudus': 66, 'Kendari': 8, 'Bangka Belitung': 78,
 'Kupang': 111, 'Pangkal Pinang': 159, 'Tuban': 90, 'Maluku': 137,
 'Singkawang': 130, 'Salatiga': 91, 'Maluku Utara': 160, 'Ungaran': 34,
 'Cileungsi': 92, 'Mamuju': 124, 'Probolinggo': 99, 'Minahasa': 23,
 'Blitar': 17, 'Batu': 67, 'Kepulauan Riau': 85, 'Madiun': 68,
 'Kepulauan Seribu': 120, 'Ponorogo': 161, 'Wonogiri ': 98, 'Metro': 81,
 'Kutai Timur': 73, 'Kotabaru': 84, 'Lombok': 21, 'Bintan': 80,
 'Kutai Kartanegara': 45, 'Hulu Sungai Tengah': 25, 'Bontang': 22,
 'Bitung': 118, 'Pematangsiantar': 143, 'Sulawesi Tenggara': 74,
 'Kotawaringin Barat': 94, 'Manokwari': 113, 'Kutai Barat': 31,
 'Papua Barat': 119
}

education_level_mapping = {
 'Sarjana (S1)': 4,
 'Magister (S2)': 2,
 'SMU/SMK/STM': 3,
 'D4 (Diploma)': 0,
 'Doktor (S3)': 1
}

employment_type_mapping = {
 'Fulltime': 2,
 'Contract': 0,
 'Part time': 1,
 'Temporary': 3
}

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': True,
        'message': 'API is running!'
    }), 200

@app.route('/predict', methods=['POST'])
def predict_salary():
    try:

        data = request.json
        career_level = career_level_mapping.get(data.get('career_level'))
        location = location_mapping.get(data.get('location'))
        experience_level = data.get('experience_level')
        education_level = education_level_mapping.get(data.get('education_level'))
        employment_type = employment_type_mapping.get(data.get('employment_type'))

        # Convert into array
        input_array = np.array([[career_level, location, experience_level, education_level, employment_type]])

        prediction = model.predict(input_array) * 10000000

        return jsonify({
                'status': True,
                'prediction': int(prediction)
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
