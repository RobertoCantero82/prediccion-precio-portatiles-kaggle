<div align="center">

<img src="laptop_price_predictor_banner.svg" alt="Laptop Price Predictor" width="680"/>

</div>

---

## 📌 Sobre el proyecto

Modelo de regresión para predecir el precio de portátiles (en €) a partir de sus especificaciones técnicas. Desarrollado como parte de una competición interna del bootcamp de Machine Learning de **The Bridge**, terminando **1º en el leaderboard privado** con un MAE de **138.385 €**.

La clave del éxito: apostar por **features causales y numéricas** que reflejan la física real del hardware, en lugar de técnicas de encoding que memorizan el dataset de entrenamiento.

---

## 🏆 Resultado final

| Leaderboard | Posición | MAE |
|:-----------:|:--------:|:---:|
| 🌐 Público  | 2º       | 153.544 € |
| 🔒 Privado  | **1º** 🥇 | **138.385 €** |

> El modelo subió de 2º a 1º en el privado porque generalizaba mejor que la competencia. El overfitting al público fue el principal error de los demás participantes.

---

## 🗂️ Estructura del repositorio

```
prediccion-precio-portatiles-kaggle/
├── 📓 trabajo_v7.ipynb          # Notebook principal (EDA + FE + modelo)
├── 📓 trabajo_inicial.ipynb     # Notebook versión inicial
├── 🤖 v7_XGBoost.pkl            # Modelo final entrenado (pickle)
├── 🤖 modelo_v1.pkl             # Modelo versión inicial (pickle)
├── 📤 submission_v7.csv         # Predicciones finales enviadas a Kaggle
├── 📤 submission1.csv           # Predicciones versión inicial
├── 📊 train.csv                 # Dataset de entrenamiento (ver nota*)
├── 📊 test.csv                  # Dataset de test (ver nota*)
├── 📊 sample_submission.csv     # Formato de entrega Kaggle
├── 🖼️ laptop_price_predictor_banner.svg  # Banner animado del README
├── ⚖️ LICENSE
└── 📖 README.md
```

> *Los datos pertenecen a la competición del bootcamp. Descárgalos desde la plataforma correspondiente.

---

## ⚙️ Pipeline completo

### 1. Análisis exploratorio
- Distribución del precio (histograma + describe)
- Heatmap de correlaciones con variables numéricas originales
- Precio medio por marca y tipo de portátil

**Hallazgo clave:** de las variables originales, solo RAM correlacionaba fuerte con el precio (0.75). El tamaño de pantalla (Inches) era casi irrelevante por sí solo.

---

### 2. Feature Engineering — 30+ variables nuevas

El grueso del trabajo. Todas las variables creadas tienen **sentido causal**, no son transformaciones matemáticas arbitrarias.

| Grupo | Features creadas | Lógica |
|-------|-----------------|--------|
| **CPU** | `Cpu_brand`, `Cpu_tier`, `Cpu_ghz`, `Cpu_gen`, `Cpu_model_num`, `Cpu_suffix_tier` | Un i7-HK vale más que un i5-U. El sufijo del modelo (U/H/HQ/HK) indica el perfil de uso. |
| **GPU** | `Gpu_brand`, `Gpu_is_dedicated`, `Gpu_tier`, `Gpu_model_num` | GPU dedicada vs integrada. Tier numérico (GTX 1080 = 6, integrada = 0). |
| **Pantalla** | `Screen_pixels`, `PPI`, `Is_IPS`, `Is_touchscreen`, `Is_4K`, `Is_retina` | PPI = calidad real. Un 4K en 13" es muy distinto a un 4K en 17". |
| **Almacenamiento** | `Has_SSD`, `Has_HDD`, `Has_hybrid`, `Storage_GB`, `SSD_GB` | SSD sube el precio; HDD lo baja. |
| **Marca/OS** | `Company_tier`, `OS_group`, `is_gaming`, `is_pro`, `is_macbook`, `is_workstation` | Razer/Apple = premium (2), Dell/HP = mid (1), resto = budget (0). |
| **Interacciones** | `Ram_x_GpuTier`, `Cpu_x_Gpu`, `Cpu_ghz_x_tier` | La combinación RAM alta + GPU potente vale más que cada componente por separado. |

---

### 3. Encoding

- **Target Encoding suavizado** (smooth = 10) para `Company`, `TypeName` y `OS_group` — mezcla la media de cada categoría con la media global para evitar overfitting en categorías pequeñas.
- **Label Encoding** simple para `Cpu_brand` y `Gpu_brand` — pocas categorías, el modelo aprende la relación solo.

---

### 4. Modelo — XGBoost

```python
xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=11
)
```

- **Target transformado:** `log1p(precio)` → `expm1(predicción)`. Los precios tienen cola larga (mayoría 500-2000€, algunos 5000+). Con logaritmo el modelo aprende mejor la escala.
- **Validación:** KFold 5 splits, shuffle=True, random_state=42.
- **Entrenamiento final:** sobre todo el dataset de train antes de predecir el test.

---

### 5. Top features más importantes

```
TypeName_te        ████████████████████  (tipo de portátil)
Ram                ██████████████████
Cpu_x_Gpu          █████████████████     (interacción CPU × GPU)
Company_te         ████████████████      (marca → precio medio)
Ram_x_GpuTier      ███████████████
PPI                ██████████████
SSD_GB             █████████████
Cpu_tier           ████████████
Gpu_tier           ███████████
Cpu_ghz_x_tier     ██████████
```

Las tres interacciones creadas aparecen en el top 10. La combinación de specs importa más que cada spec por separado.

---

## 💡 Lecciones aprendidas

**¿Por qué este modelo gana en el privado?**

| Técnica | Riesgo de overfitting | Decisión en v7 |
|---------|----------------------|----------------|
| Target encoding alta cardinalidad (modelo, producto) | 🔴 Alto | ❌ Descartado |
| Target encoding suavizado (marca, tipo, OS) | 🟡 Moderado | ✅ Incluido con smooth=10 |
| Features causales numéricas (tier, PPI, interacciones) | 🟢 Bajo | ✅ Base del modelo |

La v6 tenía mejor CV MAE pero overfiteaba al público. La v7 tenía peor CV pero **generalizó mejor al privado**. El gap entre CV y leaderboard público es la señal de alarma más valiosa en cualquier competición.

---

## 🚀 Cómo usar el modelo

```python
import pickle
import pandas as pd
import numpy as np

# Cargar modelo
with open('v7_XGBoost.pkl', 'rb') as f:
    modelo = pickle.load(f)

# El input debe tener exactamente las mismas features del notebook (feature_cols)
# Ver trabajo_v7.ipynb sección 7 para la lista completa

prediccion_log = modelo.predict(X_test)
prediccion_euros = np.expm1(prediccion_log)
print(f"Precio estimado: {prediccion_euros[0]:.2f} €")
```

---

## 🛠️ Stack

- **Python 3.10+**
- **pandas** — manipulación de datos
- **numpy** — operaciones numéricas
- **XGBoost** — modelo principal
- **scikit-learn** — validación cruzada y encoding
- **matplotlib / seaborn** — visualizaciones

---

## 👤 Autor

**Roberto Cantero** — [@RobertoCantero82](https://github.com/RobertoCantero82)

Bootcamp Machine Learning · The Bridge · 2025

---

<div align="center">

*Construido con feature engineering manual, sentido común de hardware, y muchas iteraciones* 💻

</div>
