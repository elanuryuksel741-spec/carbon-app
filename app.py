import os
from flask import Flask, render_template, request
import joblib
import numpy as np
import sqlite3
from datetime import datetime

# PostgreSQL için (Render'da kullanılır)
try:
    import psycopg2
    from urllib.parse import urlparse
except ImportError:
    psycopg2 = None

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

# === VERİTABANI BAĞLANTI FONKSİYONU ===
def get_db_connection():
    # Render'da PostgreSQL, localhost'ta SQLite
    if os.environ.get('RENDER') and psycopg2 and os.environ.get('DATABASE_URL'):
        # PostgreSQL bağlantısı (Render)
        result = urlparse(os.environ['DATABASE_URL'])
        conn = psycopg2.connect(
            dbname=result.path[1:],
            user=result.username,
            password=result.password,
            host=result.hostname,
            port=result.port
        )
        # Tabloyu oluştur (ilk çalıştırmada)
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS submissions (
            id SERIAL PRIMARY KEY,
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
            submitted_at TIMESTAMP
        )''')
        conn.commit()
        cur.close()
        return conn, 'postgres'
    else:
        # SQLite bağlantısı (localhost)
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            electricity_cost REAL, heating_type TEXT, gas_cost REAL, coal_tons REAL,
            meat_freq TEXT, recycle TEXT, cargo_monthly INTEGER,
            domestic_flights INTEGER, international_flights INTEGER,
            transport_mode TEXT, fuel_type TEXT, co2_result REAL, submitted_at TEXT
        )''')
        conn.commit()
        return conn, 'sqlite'

# === FORM İŞLEME ===
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
            
            # === VERİTABANINA KAYDET (PostgreSQL veya SQLite) ===
            conn, db_type = get_db_connection()
            cur = conn.cursor()
            
            if db_type == 'postgres':
                cur.execute('''INSERT INTO submissions 
                    (electricity_cost, heating_type, gas_cost, coal_tons, meat_freq, recycle, 
                     cargo_monthly, domestic_flights, international_flights, transport_mode, 
                     fuel_type, co2_result, submitted_at) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)''',
                    (electricity_cost, heating, gas_cost, coal_tons, meat, recycle, 
                     cargo, domestic, international, transport, fuel, co2_kg, datetime.now()))
            else:
                cur.execute('''INSERT INTO submissions 
                    (electricity_cost, heating_type, gas_cost, coal_tons, meat_freq, recycle, 
                     cargo_monthly, domestic_flights, international_flights, transport_mode, 
                     fuel_type, co2_result, submitted_at) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (electricity_cost, heating, gas_cost, coal_tons, meat, recycle, 
                     cargo, domestic, international, transport, fuel, co2_kg, datetime.now().isoformat()))
            
            conn.commit()
            cur.close()
            conn.close()
            
            # Render Logs için çıktı
            print(f"✅ SUBMISSION: CO2={co2_kg:.1f}kg | DB={db_type} | Time={datetime.now().isoformat()}")
            
            return render_template('result.html', 
                                 co2_kg=co2_kg, trees=trees, vs_avg=vs_avg, tips=tips[:5],
                                 breakdown=breakdown, breakdown_kg=breakdown_raw,
                                 model_r2=0.9842, model_mae=241.35)
        except Exception as e:
            print(f"❌ HATA: {str(e)}")
            return render_template('index.html', error=f"⚠️ Hata: {str(e)}")
    return render_template('index.html', error=None)

# === BASİT ADMIN SAYFASI: Şifreli + Tüm Alanlar ===
@app.route('/admin')
def admin_view():
    # Basit şifre koruması: ?pass=SENIN_SIFREN
    if request.args.get('pass') != 'karbon2026':  # Şifreyi buradan değiştir
        return "🔐 Yetkisiz erişim. Doğru şifreyi girin: /admin?pass=XXXX", 401
    
    try:
        conn, db_type = get_db_connection()
        cur = conn.cursor()
        
        # TÜM ALANLARI ÇEK (id hariç sıralama isteğe bağlı)
        if db_type == 'postgres':
            query = """
                SELECT id, electricity_cost, heating_type, gas_cost, coal_tons, 
                       meat_freq, recycle, cargo_monthly, domestic_flights, 
                       international_flights, transport_mode, fuel_type, 
                       co2_result, submitted_at 
                FROM submissions ORDER BY id DESC LIMIT 50
            """
        else:
            query = """
                SELECT id, electricity_cost, heating_type, gas_cost, coal_tons, 
                       meat_freq, recycle, cargo_monthly, domestic_flights, 
                       international_flights, transport_mode, fuel_type, 
                       co2_result, submitted_at 
                FROM submissions ORDER BY id DESC LIMIT 50
            """
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]  # Sütun isimleri
        cur.close()
        conn.close()
        
        # HTML Tablo Oluştur
        html = f"""
        <!DOCTYPE html><html><head><meta charset="UTF-8"><title>🔐 Admin Panel</title>
        <style>
            body{{font-family:'Segoe UI',monospace;padding:20px;background:linear-gradient(135deg,#fdf2f8,#f0f9ff)}}
            .container{{max-width:1200px;margin:0 auto;background:rgba(255,255,255,0.9);border-radius:16px;padding:20px;box-shadow:0 8px 32px rgba(0,0,0,0.1)}}
            h2{{color:#ec4899;text-align:center}}
            table{{border-collapse:collapse;width:100%;font-size:0.85rem}}
            th,td{{border:1px solid #e2e8f0;padding:10px;text-align:left;vertical-align:top}}
            th{{background:linear-gradient(135deg,#ec4899,#06b6d4);color:white;position:sticky;top:0}}
            tr:nth-child(even){{background:#f8faf9}}
            tr:hover{{background:#fef3c7}}
            .scroll{{overflow-x:auto;max-height:70vh}}
            .footer{{text-align:center;margin-top:20px;color:#6b7280}}
            .badge{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:0.75rem;background:#ec4899;color:white}}
        </style></head><body>
        <div class="container">
        <h2>📊 Tüm Kullanıcı Girdileri (DB: {db_type})</h2>
        <p style="text-align:center;color:#6b7280">Son 50 kayıt • Şifreli erişim</p>
        <div class="scroll"><table><thead><tr>
        """
        for col in columns:
            html += f"<th>{col}</th>"
        html += "</tr></thead><tbody>"
        
        for row in rows:
            html += "<tr>"
            for i, val in enumerate(row):
                if columns[i] == 'submitted_at' and val:
                    # Tarihi güzelleştir
                    from datetime import datetime
                    try:
                        if isinstance(val, str):
                            dt = datetime.fromisoformat(val.replace('Z', '+00:00'))
                        else:
                            dt = val
                        html += f"<td>{dt.strftime('%d.%m.%Y %H:%M')}</td>"
                    except:
                        html += f"<td>{val}</td>"
                elif isinstance(val, float):
                    html += f"<td>{val:.2f}</td>"
                else:
                    html += f"<td>{val}</td>"
            html += "</tr>"
        html += """
        </tbody></table></div>
        <p class="footer"><a href="/">← Ana Sayfa</a> • <a href="/admin?pass=karbon2026">🔄 Yenile</a></p>
        </div></body></html>
        """
        return html
    except Exception as e:
        return f"❌ Admin Error: {str(e)}"
    
# === DEBUG ENDPOINT: DB bağlantısını test et ===
@app.route('/debug-db')
def debug_db():
    try:
        conn, db_type = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM submissions")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        return f"✅ DB Connected! Type: {db_type} | Records: {count} | Time: {datetime.now().isoformat()}"
    except Exception as e:
        return f"❌ DB Error: {str(e)}"
    

# Render için PORT ayarı
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)