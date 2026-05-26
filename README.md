# Proyecto Final — Programación Paralela y Concurrente D02

**CUCEI · Diego Alejandro Guzmán Paniagua · 19-05-2026**

Implementación y análisis de paralelización con OpenMP sobre dos cargas de trabajo intensas: generación del conjunto de Mandelbrot en 8K y un filtro de convolución Gaussiana separable.

---

## Hardware de Prueba

| Parámetro | Valor |
|-----------|-------|
| **CPU** | Intel Core i5-1334U (Raptor Lake — 13ª Gen, 2023) |
| **Arquitectura de Cores** | Híbrida heterogénea: 2 P-cores + 8 E-cores |
| **Núcleos físicos / lógicos** | 10 / 12 (HT activo en P-cores: 2×2 + 8 = 12) |
| **Caché L1 / L2 / L3** | 48 KiB (P) / 32 KiB (E) · 6.5 MiB · 12 MiB |
| **RAM** | 16 GiB DDR4 |
| **Compilador / OpenMP** | GCC 13.3.0 · OpenMP 4.5 |

---

## Estructura del repositorio

```
├── src/
│   ├── sequential.cpp     # Código base secuencial (generado con IA)
│   ├── parallel_ai.cpp    # Baseline paralela con #pragma omp parallel for (generado con IA)
│   └── optimized.cpp      # Optimizaciones manuales: schedulers, histograma, SIMD, afinidad
├── scripts/
│   ├── benchmark.sh       # Automatiza mediciones (hilos 1-12, schedulers múltiples)
│   └── plot.py            # Genera gráficas con matplotlib
├── results/
│   ├── timings.csv                  # Datos crudos de benchmark
│   ├── exec_time_vs_threads.png     # Gráfica tiempo vs hilos
│   └── speedup_vs_threads.png       # Gráfica speedup vs hilos + curva Amdahl
├── docs/
│   ├── Proyecto_final.pdf           # Enunciado oficial del proyecto
│   └── OpenMP.pdf                   # Material de referencia de la cátedra
├── report.md              # Reporte técnico completo
└── plan.md                # Plan de implementación fase a fase
```

---

## Compilación

```bash
# Secuencial
g++ -O2 -o sequential src/sequential.cpp

# Baseline paralela (IA)
g++ -O2 -fopenmp -o parallel_ai src/parallel_ai.cpp

# Optimizado con vectorización SIMD
g++ -O3 -march=native -fopenmp -fopt-info-vec-optimized -o optimized src/optimized.cpp
```

---

## Ejecución

```bash
# Secuencial
./sequential

# Paralelo — N hilos
OMP_NUM_THREADS=4 ./parallel_ai

# Con scheduler específico
OMP_NUM_THREADS=12 OMP_SCHEDULE="dynamic,1" ./optimized

# Con afinidad de hilos (10 pts extra)
OMP_PROC_BIND=spread OMP_PLACES=cores OMP_NUM_THREADS=2 ./optimized
```

---

## Benchmarks completos

```bash
bash scripts/benchmark.sh   # genera results/timings.csv
python3 scripts/plot.py     # genera los PNG en results/
```

El script evalúa hilos de 1 a 12 con schedulers: `static`, `dynamic:1/4/16/64`, `guided:1/4`.

---

## Fases implementadas

| # | Descripción | Archivo |
|---|-------------|---------|
| 1 | Código secuencial base | `src/sequential.cpp` |
| 2 | Baseline paralela generada por IA | `src/parallel_ai.cpp` |
| 3 | Comparación de schedulers static/dynamic/guided | `src/optimized.cpp` |
| 4 | Histograma: atomic vs reduction · demostración de false sharing | `src/optimized.cpp` |
| 5 | Vectorización SIMD (`#pragma omp simd`) + afinidad de hilos | `src/optimized.cpp` |
| 6 | Scripts de benchmark y reporte técnico | `scripts/` · `report.md` |

---

## Resultados principales

- **Mandelbrot (CPU-bound irregular):** Speedup máximo de **`5.04×`** con 12 hilos y scheduler **`dynamic:1`**. En CPUs con núcleos asimétricos (P-cores + E-cores), los schedulers dinámicos son obligatorios para evitar *Barrier Starvation*.
- **Gaussian Blur (Memory-bound regular):** Speedup máximo de **`2.86×`** con 8 hilos. La saturación del bus de memoria impone una pared física (*memory wall*) independiente del número de hilos.
- **Histograma:** La versión `reduction` tardó **`0.0069 s`**, siendo **101.6× más rápida** que `atomic` (`0.7014 s`), al eliminar por completo la contención en el bus de caché.

---

## Documentación

- [`report.md`](report.md) — Reporte técnico completo.
- [`docs/Proyecto_final.pdf`](docs/Proyecto_final.pdf) — Enunciado oficial del proyecto.
- [`docs/OpenMP.pdf`](docs/OpenMP.pdf) — Material de referencia de la cátedra (A. I. Paredes López).
