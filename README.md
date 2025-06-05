# 🌌 Aurora V7 – Sistema Neuroacústico Modular de LatitudBinaural

**Aurora V7** es un sistema profesional de generación de audio neuroacústico diseñado para crear experiencias auditivas personalizadas y terapéuticas. Integra múltiples motores y módulos para generar archivos de audio WAV con capas de modulación cerebral, texturas emocionales y armónicas de alta calidad.

---

## 🧩 Arquitectura Modular

Aurora V7 se estructura en **tres niveles principales**, incluyendo parches de optimización para algunos motores.

---

## 🧠 1. Aurora Director V7 (El Cerebro)

- **Archivo:** `aurora_director_v7.py`
- **Descripción:** Es el **núcleo de control** del sistema, responsable de:
  - Orquestar la ejecución de los motores y módulos.
  - Controlar la estructura narrativa, intensidad y transiciones.
  - Integrar todas las capas generadas y coordinar la exportación del archivo WAV.

---

## 🎵 2. Motores Principales

### 🚀 NeuroMix Aurora V27 (Motor Principal)
- **Archivo:** `neuromix_aurora_v27.py`
- **Descripción:** Motor principal de generación neuroacústica.
  - Genera capas binaurales, isocrónicas, AM/FM y simulación de neurotransmisores.
  - Base para todos los procesos de audio y generación principal.

### ⚡ Parches de Optimización para NeuroMix
- **neuromix_definitivo_v7.py:** Motor optimizado con mejoras de procesamiento.
  - Incluye algoritmos avanzados de modulación.
  - Integra mejoras de estructura y efectos adicionales.
- **activacion_neuromix_definitivo.py:** Script que activa el motor definitivo y sustituye el motor base cuando se desea potenciar el sistema.
  - Uso recomendado: ejecutar después de inicializar el sistema.

### 🎨 HarmonicEssence V34
- **Archivo:** `harmonicEssence_v34.py`
- **Descripción:** Motor de generación de pads, texturas y efectos armónicos.
  - Incluye efectos etéreos, shimmer y tribal.
  - Compatible con el motor principal NeuroMix V27.

### 🔧 Parche de Optimización para HarmonicEssence
- **Archivo:** `harmonic_essence_optimizations.py`
- **Descripción:** Aumenta la potencia y las texturas de HarmonicEssence.
  - Incluye técnicas de modulación avanzada y mejoras de calidad.

### 🔥 HyperMod V32
- **Archivo:** `hypermod_v32.py`
- **Descripción:** Motor que estructura las fases y controla las transiciones.
  - Coordina la intensidad y modulación temporal.

---

## 🧩 3. Módulos de Soporte

- **emotion_style_profiles.py:** Define perfiles emocionales y presets de estilos.
- **objective_manager.py:** Gestiona objetivos terapéuticos y de estilo.
- **aurora_quality_pipeline.py:** Normaliza y masteriza el audio final.
- **verify_structure.py:** Valida integridad y sincronización de la pista.
- **sync_and_scheduler.py:** Coordina la ejecución de capas y bloques.
- **Carmine_Analyzer.py:** Analiza métricas técnicas y emocionales.
- **field_profiles.py:** Define perfiles de campos acústicos.
- **presets_fases.py:** Configura presets por fase de audio.
- **psychedelic_effects_tables.json:** Contiene efectos psicodélicos y perfiles especiales.
- **harmony_generator.py:** Genera acordes y escalas emocionales.

---

## 🚀 Cómo usar Aurora V7

1️⃣ **Instalar las dependencias:**  
```bash
./reset_aurora.sh
```

2️⃣ **Activar el entorno virtual:**  
```bash
source aurora_env/bin/activate
```

3️⃣ **Iniciar el backend:**  
```bash
python aurora_flask_app.py
```

4️⃣ **Acceder a la GUI en tu navegador:**  
```
http://127.0.0.1:5000
```

5️⃣ **Activar el motor NeuroMix Definitivo (opcional):**  
```bash
cd aurora_system
python activacion_neuromix_definitivo.py
```

---

## 📂 Estructura del Proyecto

```
aurora_v7_backend/
├── aurora_flask_app.py
├── aurora_bridge.py
├── aurora_system/
│   ├── aurora_director_v7.py
│   ├── neuromix_aurora_v27.py
│   ├── neuromix_definitivo_v7.py
│   ├── activacion_neuromix_definitivo.py
│   ├── hypermod_v32.py
│   ├── harmonicEssence_v34.py
│   ├── harmonic_essence_optimizations.py
│   ├── emotion_style_profiles.py
│   ├── objective_manager.py
│   ├── aurora_quality_pipeline.py
│   ├── verify_structure.py
│   ├── sync_and_scheduler.py
│   ├── Carmine_Analyzer.py
│   ├── field_profiles.py
│   ├── presets_fases.py
│   ├── psychedelic_effects_tables.json
│   └── harmony_generator.py
├── reset_aurora.sh
└── requirements.txt
```

---

## 🎯 Estado Actual del Proyecto

✅ Backend y GUI operativos.  
✅ Motores y parches integrados.  
✅ API REST lista para uso profesional.  
✅ Compatible con procesos de producción de audio WAV de alta calidad.

---

**Desarrollado por LatitudBinaural | Junio 2025**
