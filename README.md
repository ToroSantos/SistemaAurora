# 🌌 Aurora V7 – Sistema Neuroacústico Modular de LatitudBinaural

**Aurora V7** es un sistema profesional de generación de audio neuroacústico diseñado para crear experiencias auditivas personalizadas y terapéuticas. Integra varios módulos y motores para generar archivos de audio WAV con capas de modulación cerebral, texturas emocionales y armónicas de alta calidad.

---

## 🧩 Arquitectura Modular

Aurora V7 se estructura en **tres niveles principales**:

---

## 🧠 1. Aurora Director V7 (El Cerebro)

- **Archivo:** `aurora_director_v7.py`
- **Descripción:** Este es el **cerebro del sistema**, responsable de orquestar la ejecución de todos los motores y módulos.
  - Controla la estructura temporal y la narrativa del audio.
  - Coordina las intensidades, fases y presets.
  - Gestiona la integración de capas generadas por los motores.
  - Centraliza la comunicación con el backend (Flask).

---

## 🎵 2. Motores Principales

- **neuromix_aurora_v27.py**
  - Genera la capa neuroacústica principal (ondas binaurales, AM/FM, isocrónicas).
  - Soporta perfiles personalizados de neurotransmisores y efectos.

- **hypermod_v32.py**
  - Controla la estructura, intensidad y transiciones entre fases.
  - Administra el paneo dinámico y las envolventes temporales.

- **harmonicEssence_v34.py**
  - Genera pads armónicos, texturas y efectos de modulación emocional.
  - Integra texturas "breathy", shimmer, tribal y otros estilos musicales.

---

## 🧩 3. Módulos de Soporte

- **emotion_style_profiles.py**
  - Define perfiles emocionales y presets de estilos.
  - Ofrece categorías de texturas emocionales que enriquecen la pista final.

- **objective_manager.py**
  - Gestiona los objetivos terapéuticos y de estilo.
  - Permite seleccionar y enrutar objetivos definidos por el usuario.

- **aurora_quality_pipeline.py**
  - Normaliza, comprime y masteriza el audio final.
  - Controla la calidad y exportación del archivo WAV.

- **verify_structure.py**
  - Valida la integridad de la pista generada.
  - Genera reportes técnicos de estructura, envolventes y sincronización.

- **sync_and_scheduler.py**
  - Coordina la ejecución de capas por bloques.
  - Garantiza la sincronización entre motores y la estructura narrativa.

- **Carmine_Analyzer.py**
  - Analiza métricas de la pista generada (técnicas y emocionales).
  - Provee recomendaciones de mejora.

- **field_profiles.py**
  - Define perfiles de campos acústicos y presets de fase.

- **presets_fases.py**
  - Contiene presets por fase, ajustando duración e intensidad.

- **psychedelic_effects_tables.json**
  - Tabla JSON con perfiles psicodélicos y efectos para enriquecer la pista.

- **harmony_generator.py**
  - Genera acordes y escalas emocionales que refuerzan la textura armónica.

---

## 🚀 Cómo usar Aurora V7

1️⃣ **Instala las dependencias:**
```bash
./reset_aurora.sh
```

2️⃣ **Activa el entorno virtual:**
```bash
source aurora_env/bin/activate
```

3️⃣ **Inicia el backend:**
```bash
python aurora_flask_app.py
```

4️⃣ **Accede a la GUI en tu navegador:**
```
http://127.0.0.1:5000
```

---

## 🎯 Estado Actual del Proyecto

✅ Backend y GUI completamente operativos.  
✅ Motores funcionales generando WAV profesional.  
✅ API REST lista para integraciones avanzadas.  
✅ Estructura modular flexible para futuras expansiones.  
✅ Validación técnica y emocional de la pista generada.

---

## 📂 Estructura del Proyecto

```
aurora_v7_backend/
├── aurora_flask_app.py
├── aurora_bridge.py
├── aurora_system/
│   ├── aurora_director_v7.py
│   ├── neuromix_aurora_v27.py
│   ├── hypermod_v32.py
│   ├── harmonicEssence_v34.py
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

## 🌟 Conclusión

Aurora V7 es una plataforma profesional de generación de audio neuroacústico y emocional, lista para producción y adaptable a diferentes objetivos terapéuticos y experimentales.

---

**Desarrollado por LatitudBinaural | Junio 2025**
