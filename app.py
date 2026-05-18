from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import sqlite3
from datetime import datetime

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'carbon_model.pkl')
DB_PATH = os.path.join(BASE_DIR, 'submissions.db')

# Model yükle
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"❌ Model yüklenemedi: {e}")
    model = None

# Veritabanı otomatik oluştur
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS submissions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        electricity_cost REAL,
        heating_type TEXT,
        gas_cost REAL,
        coal_tons REAL,
        meat_freq TEXT,
        recycle TEXT,
        cargo_monthly INTEGER,
        domestic_flights INTEGER,
        international_flights INTEGER,
        transport_mode TEXT,
        fuel_type TEXT,
        co2_result REAL,
        submitted_at TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            electricity_cost = float(request.form['electricity_cost_tl'])
            heating = request.form['heating_type']
            gas_cost = float(request.form.get('gas_cost_tl', 0) or 0)
            coal_tons = float(request.form.get('coal_tons', 0) or 0)
            meat = request.form['meat_freq']
            recycle = request.form['recycle']
            cargo = int(request.form['cargo_monthly'])
            domestic = int(request.form['domestic_flights']) * 2
            international = int(request.form['international_flights']) * 2
            transport = request.form['transport_mode']
            fuel = request.form['fuel_type']

            enc = {
                'heating_type': {'gas': 0, 'coal': 1},
                'meat_freq': {'none': 0, 'few_times': 1, 'daily': 2},
                'recycle': {'yes': 0, 'no': 1},
                'transport_mode': {'none': 0, 'public': 1, 'private': 2},
                'fuel_type': {'electric': 0, 'hybrid': 1, 'gasoline': 2, 'diesel': 3}
            }
            
            features = np.array([[
                electricity_cost, enc['heating_type'][heating],
                gas_cost if heating == 'gas' else 0,
                coal_tons if heating == 'coal' else 0,
                enc['meat_freq'][meat], enc['recycle'][recycle],
                cargo, domestic, international,
                enc['transport_mode'][transport], enc['fuel_type'][fuel]
            ]])
            
            prediction = model.predict(features)[0]
            co2_kg = max(0, float(prediction))
            
            breakdown_raw = {
                'electricity': electricity_cost * 1.2 * 0.45,
                'heating': gas_cost * 0.21 if heating == 'gas' else coal_tons * 2800,
                'transport': 1200 if transport == 'private' and fuel in ['diesel','gasoline'] else 400,
                'flights': (domestic * 180) + (international * 550)
            }
            breakdown = {k: min(100, (v / 2000) * 100) for k, v in breakdown_raw.items()}
            
            trees = co2_kg / 22
            avg_person = 4500
            vs_avg = ((co2_kg - avg_person) / avg_person) * 100
            
            tips = []
            if electricity_cost > 1500: tips.append("💡 LED ve A++ cihazlarla %30 tasarruf sağlayabilirsiniz.")
            if heating == 'coal': tips.append("🔥 Kömür yerine doğalgaz/ısı pompası ile %60 azaltım mümkün.")
            if meat == 'daily': tips.append("🥩 Haftada 2 gün etsiz beslenme ~200 kg CO2 tasarrufu sağlar.")
            if recycle == 'no': tips.append("♻️ Geri dönüşüm karbon ayak izinizi %12 azaltır.")
            if cargo > 6: tips.append("📦 Kargoları birleştirerek ambalaj emisyonunu düşürün.")
            if (domestic + international) > 4: tips.append("✈️ Uçuşları tren/video konferans ile değiştirin.")
            if transport == 'private' and fuel in ['diesel', 'gasoline']: tips.append("🚗 Toplu taşıma/elektrikli araç ile emisyonu yarıya indirebilirsiniz.")
            
            default_tips = [" Enerji sertifikalı ürünler tercih edin.", "🚲 Kısa mesafede yürüyün/bisiklet kullanın.", " Yerel/mevsimsel gıda tüketin."]
            while len(tips) < 3:
                for t in default_tips:
                    if t not in tips: tips.append(t); break
            
            # DB'ye kaydet
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''INSERT INTO submissions (electricity_cost, heating_type, gas_cost, coal_tons, meat_freq, recycle, cargo_monthly, domestic_flights, international_flights, transport_mode, fuel_type, co2_result, submitted_at) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (electricity_cost, heating, gas_cost, coal_tons, meat, recycle, cargo, domestic, international, transport, fuel, co2_kg, datetime.now().isoformat()))
            conn.commit()
            conn.close()
            
            return render_template('result.html', 
                                 co2_kg=co2_kg, trees=trees, vs_avg=vs_avg, tips=tips[:5],
                                 breakdown=breakdown, breakdown_kg=breakdown_raw,
                                 model_r2=0.9842, model_mae=241.35)
        except Exception as e:
            return render_template('index.html', error=f"⚠️ Hata: {str(e)}")
    return render_template('index.html', error=None)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)