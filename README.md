# 🌌 Aurora V7 – Sistema Neuroacústico Modular Avanzado

**Aurora V7** es el sistema neuroacústico de LatitudBinaural, diseñado para crear experiencias auditivas terapéuticas, emocionales y cognitivas. Integra tecnologías avanzadas de generación de audio, modulación cerebral y texturas emocionales en una plataforma modular altamente personalizable. Esta versión está diseñada para generar audio WAV profesional listo para su uso terapéutico y experimental.

---

## 🧠 ¿Qué es Aurora V7?

Aurora V7 es un **sistema neuroacústico modular** que produce experiencias auditivas personalizadas en función de un objetivo terapéutico y emocional. Su motor central integra varios generadores de capas de audio (neuroacústicas, emocionales y texturales) para producir una pista estereofónica con estructura narrativa.

- **Propósito principal**: Activar estados mentales, emocionales y espirituales específicos, utilizando la modulación de ondas cerebrales y texturas emocionales.
- **Formato de salida**: Archivos `.wav` listos para reproducción en audífonos o altavoces.

---

## 🧩 Arquitectura Modular

Aurora V7 se organiza en tres niveles principales:

### 🌐 1. Aurora Director V7 (El Cerebro)

El **Aurora Director V7** es el núcleo central que coordina todos los motores y módulos. Sus funciones clave incluyen:
- Orquestar la ejecución de cada motor y módulo en función del objetivo elegido.
- Gestionar las fases de la experiencia auditiva (preparación, intención, clímax y resolución).
- Integrar las distintas capas de audio generadas por los motores para crear una pista cohesiva.
- Administrar la comunicación con la GUI y el backend Flask.

**Archivo principal:**  
`aurora_system/aurora_director_v7.py`

---

### ⚙️ 2. Motores Principales

Los **motores** son responsables de generar las capas de audio especializadas. Cada motor está diseñado para aportar una dimensión única a la pista:
- **neuromix_aurora_v27.py**: Genera la capa neuroacústica principal (ondas binaurales, AM/FM, isocrónicas).
- **hypermod_v32.py**: Controla la estructura y dinámica de la pista (fases, intensidad, duración).
- **harmonicEssence_v34.py**: Añade texturas emocionales, pads armónicos y efectos de paneo espacial avanzado.
- **NoiseEngine_v5.py**: Genera ruido texturizado y efectos ambientales.
- **EmotionLayer_v4.py**: Complementa con texturas emocionales adicionales.

Cada motor puede personalizarse según el objetivo terapéutico y está diseñado para integrarse de forma modular con el Aurora Director.

**Ubicación:**  
`aurora_system/`

---

### 🧩 3. Módulos de Soporte

Estos módulos complementan y enriquecen la funcionalidad de Aurora V7. Incluyen:
- **verify_structure.py**: Validación científica de la estructura auditiva y generación de reportes detallados.
- **objective_manager.py**: Enrutamiento inteligente de objetivos terapéuticos y estilos.
- **profiles_and_effects.py**: Consolidación de efectos psicodélicos, perfiles emocionales y presets.
- **aurora_quality_pipeline.py**: Normalización, compresión y mastering del audio final.
- **harmony_generator.py**: Generación de pads armónicos y transiciones suaves.
- **sync_and_scheduler.py**: Coordinación de sincronización y tiempos de cada capa.

**Ubicación:**  
`aurora_system/`

---

## 🔧 Uso del Sistema

1️⃣ **Instalación de dependencias**:  
```bash
./reset_aurora.sh
```

2️⃣ **Activar el entorno virtual**:  
```bash
source aurora_env/bin/activate
```

3️⃣ **Iniciar Aurora V7**:  
```bash
python aurora_flask_app.py
```

4️⃣ **Accede a la GUI**:  
```
http://127.0.0.1:5000
```

Desde ahí, podrás elegir el objetivo, emoción, intensidad y duración de la experiencia auditiva, y generar la pista WAV profesional.

---

## 🎯 Estado Actual del Proyecto

✅ **Backend y GUI** completamente operativos.  
✅ **Aurora Director V7** generando audio real con alta calidad.  
✅ **Motores funcionales** (NeuroMix V27, HyperMod V32, etc.).  
✅ **APIs REST** listas para integraciones avanzadas.  
✅ **GUI profesional** con panel de control intuitivo y reproductor de audio.  
✅ **Sistema de métricas** para seguimiento en tiempo real.  
✅ **Descarga directa** de pistas generadas.

### 📊 Métricas Actuales:
- 🌟 Sistema: 100% operativo
- 🎵 Motores: 5/5 activos
- 📊 Calidad de audio: 93% (excelente)
- ⚡ Generación de pista: ~2.5 segundos promedio
- ✅ Tasa de éxito: 100%

---

## 📂 Estructura General del Proyecto

```
aurora_v7_backend/
├── aurora_flask_app.py           → Backend Flask
├── aurora_bridge.py              → Conexión entre GUI y Aurora Director
├── aurora_system/
│   ├── aurora_director_v7.py     → Cerebro maestro
│   ├── [motores y módulos]       → Núcleo de generación
│   └── verify_structure.py       → Validación y reportes
├── templates/                    → GUI HTML
├── static/                       → CSS, JS y audios
├── logs/                         → Registro del sistema
├── reset_aurora.sh               → Script de reinicio y dependencias
└── requirements.txt              → Lista de dependencias
```

---

## 🧭 Próximos Pasos

- Mejorar la UX de la GUI para móviles.  
- Integrar efectos neuroquímicos avanzados en NeuroMix.  
- Ampliar el análisis emocional con Carmine_Analyzer.  
- Incorporar visualizadores avanzados para las métricas.

---

## 🌟 Conclusión

Aurora V7 es una plataforma profesional y robusta para la generación de experiencias auditivas terapéuticas, científicas y experimentales. Ofrece una interfaz completa, motores avanzados y la posibilidad de personalización por objetivo. Es el núcleo del sistema de LatitudBinaural y está listo para uso profesional. 🎧

---

**Desarrollado por LatitudBinaural | Junio 2025**
