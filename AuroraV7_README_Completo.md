# 📝 Aurora V7 - README Completo

## 🚀 ¿Qué es Aurora V7?

Aurora V7 es un sistema avanzado de generación de audio neuroacústico desarrollado para crear experiencias auditivas profundas, terapéuticas y personalizadas. Combina un backend robusto en Python, múltiples motores especializados y una interfaz gráfica profesional, permitiendo la generación de audio WAV de alta calidad a través de APIs y una GUI intuitiva.

---

## 🧠 ¿Cómo está organizado?

Aurora V7 se divide en **tres grandes bloques**:

---

## 🌟 1. El Cerebro - Aurora Director V7

- **¿Qué es?**
  - Es el **núcleo central** del sistema. Coordina la ejecución de todos los motores y módulos, orquesta la generación del audio y administra la lógica de flujo.
- **¿Qué hace?**
  - Administra la duración de las sesiones.
  - Controla las intensidades y transiciones de cada capa de audio.
  - Supervisa la sincronización de los motores y las capas.
  - Se conecta al backend Flask para responder a las solicitudes de generación de audio.
- **Archivo principal:**
  - `aurora_system/aurora_director_v7.py`

---

## 🎵 2. Los Motores - Núcleo de Sonido

- **¿Qué son?**
  - Son los módulos que **generan las ondas de audio** (binaurales, AM, FM, pads, ruido, neurotransmisores, etc.).
- **¿Qué hacen?**
  - Cada motor tiene una especialidad:
    - **NeuroMix V27:** Capa de audio neuroacústica base.
    - **HyperMod V32:** Control de estructura y modulación temporal.
    - **PsyLayer V6:** Añade efectos psicodélicos o fractales.
    - **NoiseEngine V5:** Genera capas de ruido texturizado.
    - **EmotionLayer V4:** Agrega texturas emocionales al audio.
  - Se comunican con el Aurora Director para integrarse en la mezcla final.
- **Ubicación:**
  - `aurora_system/` (cada motor es un archivo Python).

---

## 🧩 3. Los Módulos de Soporte - Complementos Clave

- **¿Qué son?**
  - Son scripts y librerías que asisten a los motores y al director.
- **¿Qué hacen?**
  - Validan configuraciones (`verify_structure.py`).
  - Generan presets (`presets_emocionales.py`, `presets_fases.py`).
  - Manejan la generación de envolventes y sincronización.
  - Administran la exportación y compresión de archivos.
- **Ubicación:**
  - `aurora_system/` y subcarpetas.

---

## 🛠️ Cómo usar Aurora V7

1️⃣ **Instala dependencias:**
```bash
./reset_aurora.sh
```

2️⃣ **Inicia el backend:**
```bash
source aurora_env/bin/activate
python aurora_flask_app.py
```

3️⃣ **Accede a la GUI en:**
```
http://127.0.0.1:5000
```

4️⃣ **Genera experiencias auditivas** desde el formulario.

---

## 🚀 ¿Listo para usar?

¡Sí! Aurora V7 es completamente funcional y listo para producción. 🌟

---

## 📂 Estructura general

```
aurora_v7_backend/
├── aurora_flask_app.py          -> Backend Flask
├── aurora_bridge.py             -> Comunicación con Aurora Director
├── aurora_system/
│   ├── aurora_director_v7.py    -> Cerebro central
│   ├── [motores y módulos]      -> Núcleo de generación
│   └── verify_structure.py      -> Validación y control
├── templates/                   -> GUI HTML
├── static/                      -> CSS, JS y audios
├── logs/                        -> Registro del sistema
├── reset_aurora.sh              -> Script de reinicio y dependencias
└── requirements.txt             -> Lista de dependencias
```

---

## 🎯 Estado actual

✅ Backend y GUI completamente operativos  
✅ 5 Motores activos generando audio WAV real  
✅ API REST funcionando al 100%  
✅ Generación de experiencias auditivas de alta calidad  
✅ Control avanzado de intensidades y presets  
✅ Listo para uso profesional

---

📅 *Última actualización: Junio 2025*
