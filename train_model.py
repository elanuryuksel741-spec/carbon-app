import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

print("📊 Gelişmiş sentetik veri oluşturuluyor (12 özellik + gerçek faktörler)...")
np.random.seed(42)
n = 15000

# === TEMEL ÖZELLİKLER ===
electricity_cost_tl = np.random.uniform(200, 3000, n)  # Aylık elektrik faturası (TL)
heating_type = np.random.choice(['gas', 'coal'], n, p=[0.7, 0.3])
gas_cost_tl = np.random.uniform(300, 2500, n) * (heating_type == 'gas').astype(int)
coal_tons = np.random.uniform(1, 8, n) * (heating_type == 'coal').astype(int)

# === YENİ SORULAR ===
meat_freq = np.random.choice(['none', 'few_times', 'daily'], n, p=[0.15, 0.55, 0.30])
recycle = np.random.choice(['yes', 'no'], n, p=[0.4, 0.6])
cargo_monthly = np.random.poisson(4, n)  # Aylık kargo siparişi
domestic_flights = np.random.poisson(1.5, n)  # Yıllık iç hat uçuş (tek yön)
international_flights = np.random.poisson(0.8, n)  # Yıllık dış hat uçuş (tek yön)
transport_mode = np.random.choice(['public', 'private', 'none'], n, p=[0.4, 0.5, 0.1])
fuel_type = np.random.choice(['diesel', 'gasoline', 'hybrid', 'electric'], n, p=[0.35, 0.4, 0.15, 0.10])

# === CO2 HESAPLAMA FORMÜLLERİ (EPA/IEA/DEFRA 2024) ===
co2 = np.zeros(n)

# 1. Elektrik (TL → kWh → CO2): 1 TL ≈ 1.2 kWh, 1 kWh ≈ 0.45 kg CO2
co2 += (electricity_cost_tl * 1.2) * 0.45

# 2. Isınma
co2 += np.where(heating_type == 'gas', gas_cost_tl * 0.21, 0)  # 1 TL doğalgaz ≈ 0.21 kg CO2
co2 += np.where(heating_type == 'coal', coal_tons * 2800, 0)  # 1 ton kömür ≈ 2800 kg CO2

# 3. Et tüketimi (haftalık → yıllık kg CO2)
meat_map = {'none': 0, 'few_times': 150, 'daily': 650}  # kg CO2/yıl
co2 += pd.Series(meat_freq).map(meat_map).values

# 4. Geri dönüşüm (-%12 emisyon azaltır)
co2 *= np.where(recycle == 'yes', 0.88, 1.0)

# 5. Kargo siparişleri (her sipariş ≈ 2.3 kg CO2)
co2 += cargo_monthly * 2.3 * 12  # yıllık

# 6. Uçuşlar (iç hat: 1 uçuş ≈ 180 kg, dış hat: 1 uçuş ≈ 550 kg CO2)
co2 += domestic_flights * 180
co2 += international_flights * 550

# 7. Araç kullanımı
vehicle_km_yearly = np.random.uniform(5000, 30000, n)
fuel_map = {'diesel': 0.15, 'gasoline': 0.12, 'hybrid': 0.07, 'electric': 0.045}  # kg CO2/km
transport_factor = np.where(transport_mode == 'private', 
                           pd.Series(fuel_type).map(fuel_map).fillna(0.12).values,
                           np.where(transport_mode == 'public', 0.04, 0))
co2 += vehicle_km_yearly * transport_factor

# Gürültü ekle (%5-8 varyasyon)
noise = np.random.normal(0, co2.std() * 0.065, n)
co2 = np.maximum(co2 + noise, 0)

# === VERİ SETİ OLUŞTUR ===
df = pd.DataFrame({
    'electricity_cost_tl': electricity_cost_tl,
    'heating_type': heating_type,
    'gas_cost_tl': gas_cost_tl,
    'coal_tons': coal_tons,
    'meat_freq': meat_freq,
    'recycle': recycle,
    'cargo_monthly': cargo_monthly,
    'domestic_flights': domestic_flights,
    'international_flights': international_flights,
    'transport_mode': transport_mode,
    'fuel_type': fuel_type,
    'co2_kg': co2
})

# === KATEGORİK DEĞİŞKENLERİ SAYISALA ÇEVİR ===
encodings = {
    'heating_type': {'gas': 0, 'coal': 1},
    'meat_freq': {'none': 0, 'few_times': 1, 'daily': 2},
    'recycle': {'yes': 0, 'no': 1},
    'transport_mode': {'none': 0, 'public': 1, 'private': 2},
    'fuel_type': {'electric': 0, 'hybrid': 1, 'gasoline': 2, 'diesel': 3}
}
for col, mapping in encodings.items():
    df[col + '_num'] = df[col].map(mapping)

# === MODEL EĞİTİMİ ===
feature_cols = ['electricity_cost_tl', 'heating_type_num', 'gas_cost_tl', 'coal_tons',
                'meat_freq_num', 'recycle_num', 'cargo_monthly', 'domestic_flights',
                'international_flights', 'transport_mode_num', 'fuel_type_num']
X = df[feature_cols]
y = df['co2_kg']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🌲 Random Forest modeli eğitiliyor (11 özellik)...")
model = RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_leaf=5, 
                              random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# === DOĞRULUK RAPORU ===
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100

print("\n📈 MODEL DOĞRULUK RAPORU:")
print(f"   ✅ R² Score: {r2:.4f} → %{r2*100:.1f} varyans açıklanıyor")
print(f"   📉 MAE: {mae:.2f} kg CO2 (ortalama tahmin hatası)")
print(f"   📐 RMSE: {rmse:.2f} kg CO2 (büyük hatalara duyarlı)")
print(f"   🎯 MAPE: %{mape:.1f} (ortalama yüzde hata)")

# Özellik önem sıralaması
print("\n🔍 En Etkili Faktörler:")
for name, imp in sorted(zip(feature_cols, model.feature_importances_), 
                        key=lambda x: x[1], reverse=True)[:6]:
    print(f"   • {name}: %{imp*100:.1f}")

# Modeli kaydet
joblib.dump(model, "carbon_model.pkl")
size_mb = os.path.getsize("carbon_model.pkl") / (1024*1024)
print(f"\n💾 carbon_model.pkl kaydedildi ({size_mb:.2f} MB) ✅ <50MB Render limiti")