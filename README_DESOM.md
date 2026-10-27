# DESOM: Deep Embedded Self-Organizing Maps (MaNGA IFU)

Pipeline modularizado y optimizado para la replicación fiel de los resultados y figuras del artículo científico (*Teimoorinia, Archinuk, Woo et al., The Astronomical Journal* / *arXiv:2112.03425*).

---

## Características Principales

1. **Fidelidad Estricta al Paper y Mejoras de Ingeniería:**
   * Rango de longitud de onda en reposo fijado a **`[3650, 8950] Å`** en $4544$ bins logarítmicos.
   * Filtro de calidad de baxel fijado a **$S/N \ge 10.0$** (Paper §2).
   * Detección y descarte global de outliers de flujo (percentil 0.5% inferior y 0.5% superior protegiendo líneas de emisión).
   * Normalización de huellas (fingerprints) cruzada idéntica a `DESOM2_series.py` (percentil 99.9 por nodo + MinMax por nodo + MinMax por galaxia).
   * Cálculo de métricas originales: **error de cuantización** y **error topográfico** (en espacio latente y original).
2. **Opciones Configurables (CLI y UI Web):**
   * **Unidad de Split (`--split-mode`):**
     * `galaxy`: División por galaxias (70% Train, 15% Val, 15% Test) evitando cualquier fuga de baxels de una misma galaxia entre conjuntos (recomendado).
     * `spectrum`: Partición aleatoria directa por espectros individuales (réplica exacta del paper).
   * **Distancia en el Mapa (`--map-distance`):**
     * `manhattan`: Distancia Manhattan bidimensional exacta $|Δ\text{fila}| + |Δ\text{col}|$ (corregida).
     * `paper`: Réplica exacta del código original $|a - b| // \text{ancho} + |Δ\text{col}|$.
3. **Soporte de Almacenamiento Dual:**
   * **Local:** Almacena datos, modelos y figuras en tu laptop (`./workspace_data/experiments/...`).
   * **Nube (Google Drive / Colab):** Almacena directamente en `/content/drive/MyDrive/...`.
4. **Compatibilidad Universal con GPUs:**
   * `set_memory_growth=True` activado por defecto: permite entrenar sin colapsos de memoria en GPUs dedicadas pequeñas como **GTX 1650 (4 GB VRAM)**, tarjetas modernas **RTX 3060/4060/5060**, aceleración Metal en **Apple Silicon**, o CPU multi-hilo.
5. **Replicación Directa de Figuras del Paper:**
   * **Figura 1:** Muestra de espectros normalizados ($S_1 \dots S_9$) y líneas de emisión características.
   * **Figura 3:** Malla $15 \times 15$ de prototipos espectrales decodificados (DESOM-1) con las posiciones de los espectros de la Fig. 1 marcadas en rojo.
   * **Figura 7:** Huella dactilar individual de una galaxia con conteo numérico de baxels en cada uno de los 225 nodos.
   * **Figura 9:** Mapa de ocupación y distribución de galaxias por nodo en DESOM-2 ($10 \times 10$).
   * **Figura 10:** Malla $10 \times 10$ de prototipos decodificados de diversidad galáctica (DESOM-2).

---

## Estructura del Código

```text
DESOM/
├── desom/
│   ├── config.py           # Configuración central (local vs drive), hardware y parámetros
│   ├── io_utils.py         # Escritura atómica, hashes SHA-256 y seguimiento de estado
│   ├── data/
│   │   ├── catalog.py      # Descarga y selección de galaxias (DRPall DR15)
│   │   ├── cubes.py        # Descarga y lectura de cubos FITS LOGCUBE
│   │   ├── processor.py    # Voronoi binning sin sesgos, covarianzas de ruido y rebinning [3650, 8950] Å
│   │   └── consolidate.py  # Filtros de flujo globales y partición (Etapas 1 y 2)
│   ├── models/
│   │   ├── base.py         # BaseDESOM abstracta con optimización tf.function y soporte de distancias
│   │   ├── som_layer.py    # SOMLayer con topología 2D y enfriamiento exponencial
│   │   ├── desom1.py       # Autoencoder Convolucional 1D + SOM (15x15)
│   │   └── desom2.py       # Autoencoder Convolucional 2D + SOM (10x10)
│   ├── training/
│   │   ├── engine.py       # Motor de entrenamiento unificado con logging topográfico y reanudación
│   │   ├── trainer_desom1.py # Envoltorio de entrenamiento para DESOM-1
│   │   └── trainer_desom2.py # Envoltorio de entrenamiento para DESOM-2
│   ├── analysis/
│   │   ├── fp_normalize.py # Algoritmo de normalización de huellas del paper
│   │   ├── fingerprints.py # Proyección de baxels y generación de huellas 15x15
│   │   ├── metrics.py      # Error de cuantización y error topográfico
│   │   └── evaluation.py   # Evaluación rigurosa en conjunto de prueba (Held-out test set)
│   ├── plots/
│   │   ├── spectral_plots.py   # Figuras 1 y 3 (Espectros y prototipos con orientación unificada)
│   │   ├── fingerprint_plots.py# Figura 7 y cuadrícula de huellas dactilares
│   │   └── galaxy_map_plots.py # Figuras 9 y 10 (Ocupación y prototipos DESOM-2)
│   └── cli/
│       ├── run_stage1.py   # CLI Etapa 1: Descarga y Voronoi binning
│       ├── run_stage2.py   # CLI Etapa 2: Consolidación, auditoría y Fig 1
│       ├── run_stage3.py   # CLI Etapa 3: Entrenamiento DESOM-1, evaluación y Figs 3/7
│       ├── run_stage4.py   # CLI Etapa 4: Entrenamiento DESOM-2, evaluación y Figs 9/10
│       └── run_all.py      # Ejecutor maestro end-to-end
├── tests/
│   └── test_pipeline.py    # Suite de pruebas unitarias automatizadas (pytest)
├── app.py                  # Servidor de interfaz web interactiva
├── Pipeline_Unificado.ipynb# Notebook para Google Colab
├── requirements.txt        # Dependencias del proyecto
└── Paper Original/         # Archivos de referencia del repositorio de los autores
```

---

## Interfaz Gráfica Interactiva (Dashboard)

El proyecto incluye un dashboard web interactivo con tema astronómico (*Dark Mode*), sin dependencias externas pesadas:

```bash
python3 app.py --port 8050
```
Abre tu navegador en `http://localhost:8050` para acceder a:
1. **Demografía del Dataset:** Estadísticas de redshift ($\min, \max, \mu, \text{mediana}, \sigma$), histograma interactivo, gráfico de distribución de splits y tabla de búsqueda de galaxias.
2. **Control de Configuración:** Modifica interactivamente el tamaño de muestra, el S/N mínimo, la unidad de split (`galaxy` vs `spectrum`) y la distancia de la malla SOM (`manhattan` vs `paper`).
3. **Consola y Ejecución en Vivo:** Lanza cualquier etapa o el pipeline completo y sigue los logs de ejecución en tiempo real.
4. **Explorador Visual de Resultados:** Inspecciona las mallas de prototipos de DESOM-1 ($15 \times 15$) y los mapas de galaxias de DESOM-2 ($10 \times 10$).
5. **Galería de Figuras:** Previsualización y descarga directa en alta resolución de todas las figuras del paper generadas.

---

## Instrucciones de Uso por Consola

### 1. Instalación de Dependencias
```bash
pip install -r requirements.txt
```

### 2. Ejecutar Pruebas Automatizadas
Verifica que todos los componentes matemáticos, modelos y persistencia funcionen correctamente:
```bash
pytest -q tests
```

### 3. Ejecución Completa (End-to-End)
Para ejecutar todas las etapas con una muestra de **500 galaxias** en almacenamiento local:
```bash
python3 -m desom.cli.run_all --storage local --sample 500 --split-mode galaxy --map-distance manhattan
```

### 4. Ejecución por Etapas Individuales

#### **Etapa 1: Descarga y Procesamiento de Espectros**
Descarga catálogo DRPall, selecciona las galaxias y genera baxels con Voronoi binning ($S/N \ge 10$):
```bash
python3 -m desom.cli.run_stage1 --storage local --sample 500
```

#### **Etapa 2: Consolidación, Auditoría y Figura 1**
Aplica cortes globales de flujo, particiona datos y genera la **Figura 1**:
```bash
python3 -m desom.cli.run_stage2 --storage local --sample 500 --split-mode galaxy
```
*Salidas:* 
* `workspace_data/experiments/run_500g/figures/figure_1_normalized_spectra.png`
* `desom1_dataset_ready.npz`

#### **Etapa 3: Entrenamiento DESOM-1, Evaluación y Huellas (Figuras 3 y 7)**
Entrena DESOM-1, calcula error topográfico en test set, genera la malla de prototipos (**Figura 3**) y las huellas (**Figura 7**):
```bash
python3 -m desom.cli.run_stage3 --storage local --sample 500 --epochs 65000 --batch-size 128 --map-distance manhattan
```
*Salidas:* 
* `figures/figure_3_desom1_prototypes_15x15.png`
* `figures/figure_7_fingerprint_<plateifu>.png`
* `desom1/desom1_test_metrics.json`
* `galaxy_fingerprints.npz`

#### **Etapa 4: Entrenamiento DESOM-2 y Diversidad de Galaxias (Figuras 9 y 10)**
Entrena DESOM-2 sobre las huellas normalizadas y genera los mapas de ocupación (**Figura 9**) y prototipos decodificados (**Figura 10**):
```bash
python3 -m desom.cli.run_stage4 --storage local --sample 500 --epochs 40000 --map-distance manhattan
```
*Salidas:* 
* `figures/figure_9_desom2_node_counts.png`
* `figures/figure_10_desom2_galaxy_grid_10x10.png`
* `desom2/desom2_test_metrics.json`
