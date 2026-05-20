# Plan: Proyecto Final — Programación Paralela y Concurrente D02

## Context

El proyecto requiere implementar y analizar la paralelización con OpenMP de dos cargas de trabajo intensas:
- **Tarea A:** Generación del conjunto de Mandelbrot en resolución 8K (7680×4320).
- **Tarea B:** Filtro de convolución Gaussiana (radio 15, kernel 31×31) sobre la imagen generada.

El objetivo final es producir: (1) un repositorio Git con historial claro, y (2) un reporte técnico en Markdown con gráficas, análisis de speedup, Ley de Amdahl y conclusiones.

**Hardware objetivo (Intel Core i3-6006U):**
- 2 núcleos físicos / 4 lógicos (Hyper-Threading)
- L1d: 32 KiB/núcleo | L2: 256 KiB/núcleo | L3: 3 MiB compartida
- RAM: 15 GiB | GCC 13.3.0 con OpenMP

---

## Estructura de Archivos a Crear

```
Proyecto Final/
├── src/
│   ├── sequential.cpp       # Commit 1: código base secuencial
│   ├── parallel_ai.cpp      # Commit 2: baseline paralela (IA)
│   └── optimized.cpp        # Commits 3-5: optimizaciones manuales
├── scripts/
│   ├── benchmark.sh         # Automatiza mediciones con distintos hilos/schedulers
│   └── plot.py              # Genera gráficas con matplotlib
├── results/                 # CSVs con datos de benchmark (generados al ejecutar)
└── report.md                # Reporte técnico final
```

---

## Fase 1 — Código Secuencial (`src/sequential.cpp`)

**Tarea A — Mandelbrot 8K:**
```
Resolución: 7680 × 4320 (33.18 M píxeles)
max_iter: 1000
Salida: mandelbrot.ppm (PPM binario)
Timing: omp_get_wtime() alrededor de cada sección
```

**Tarea B — Gaussian Blur separable:**
```
Radio: 15 → kernel 1D de 31 elementos
Dos pasadas: horizontal (fila por fila), luego vertical (columna por columna)
Separar en dos loops independientes para facilitar paralelización posterior
Salida: blurred.ppm
```

**Por qué separable:** kernel 31×31 = 961 ops/px vs. 31+31 = 62 ops/px con filtro separable (15.5× menos trabajo).

---

## Fase 2 — Baseline Paralela IA (`src/parallel_ai.cpp`)

Agregar `#pragma omp parallel for` básico (scheduler static por defecto):
- Mandelbrot: paralelizar el loop externo (sobre filas Y)
- Blur horizontal: paralelizar sobre filas
- Blur vertical: paralelizar sobre columnas

**Prompt a documentar (para la sección de Metodología del reporte):**
```
"Tengo este programa secuencial en C++ que genera un fractal de Mandelbrot
en 8K y le aplica un desenfoque Gaussiano separable. Paralelízalo usando
OpenMP agregando únicamente pragmas #pragma omp parallel for con el
scheduler por defecto. Mantén el código mínimamente modificado."
```

---

## Fase 3 — Balanceo de Carga (Schedulers)

En `src/optimized.cpp`, parametrizar el scheduler vía argumento de línea de comandos:
```cpp
// OMP_SCHEDULE=dynamic,16 ./optimized
// o flag --scheduler static|dynamic|guided --chunk N
```

**Schedulers a evaluar para Tarea A (Mandelbrot):**
| Scheduler | Chunk | Razón de eficacia esperada |
|-----------|-------|---------------------------|
| static    | auto  | baseline, desbalanceado (filas del centro son baratas) |
| dynamic   | 1     | mejor balance, más overhead |
| dynamic   | 16    | compromiso overhead/balance |
| guided    | 1     | chunks decrecientes, buen balance con menos overhead |

**Por qué Mandelbrot es irregular:** filas centrales (cardioide) convergen en pocas iteraciones; filas del borde iteran hasta max_iter → desbalance severo con static.

Script `benchmark.sh` ejecuta todas las combinaciones de:
- Hilos: 1, 2, 3, 4, 5, 6, 7, 8 (de 1 hasta 2× núcleos lógicos)
- Schedulers: static, dynamic:1, dynamic:4, dynamic:16, dynamic:64, guided:1, guided:4

---

## Fase 4 — Sincronización y False Sharing

### Histograma de colores (cuantizado a 256 bins de luminancia):

**Versión A — exclusión mutua (`#pragma omp atomic`):**
```cpp
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    #pragma omp atomic
    histogram[pixel_value(i)]++;
}
```
Problema: contención masiva en `atomic` cuando muchos hilos actualizan el mismo bin.

**Versión B — `reduction` con arreglo local:**
```cpp
int local_hist[256] = {0};
#pragma omp parallel for firstprivate(local_hist)
for (int i = 0; i < N; i++) local_hist[pixel_value(i)]++;
// merge manual al final con critical
```

**Demostración de False Sharing:**
- Arreglo compartido `int hist_shared[256]`: los threads actualizan bins adyacentes → misma línea de caché (64 bytes = 16 ints) → invalidación constante.
- Solución: padding a 64 bytes por bin, o usar arreglo local por thread.
- Medir diferencia de tiempo: atomic vs. reduction vs. false-sharing naive.

---

## Fase 5 — SPMD y Afinidad

### Vectorización SIMD del filtro (Tarea B):

Reestructurar el inner loop de convolución para que sea vectorizable:
```cpp
#pragma omp simd reduction(+:sum)
for (int k = -radius; k <= radius; k++) {
    sum += kernel[k + radius] * row[j + k];
}
```

Compilar con flags de verificación:
```bash
g++ -O3 -march=native -fopenmp -fopt-info-vec-optimized -o optimized src/optimized.cpp
```
La salida del compilador debe indicar qué loops fueron vectorizados.

### Afinidad de hilos (10 pts extra):

Medir con y sin afinidad para 2 y 4 hilos:
```bash
# Sin afinidad
OMP_NUM_THREADS=4 ./optimized

# Con afinidad — hilos pegados a núcleos físicos
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=4 ./optimized

# Spread — un hilo por núcleo físico
OMP_PROC_BIND=spread OMP_PLACES=cores OMP_NUM_THREADS=2 ./optimized
```

En i3-6006U con 2 núcleos físicos: `close` agrupa dos hilos lógicos por núcleo → comparten L2 (256 KiB), lo que favorece al filtro de convolución con reutilización de datos de fila.

---

## Fase 6 — Benchmarking y Gráficas

`scripts/benchmark.sh` → genera `results/timings.csv`:
```
threads,task,scheduler,chunk,time_s
1,mandelbrot,static,auto,45.2
2,mandelbrot,static,auto,24.1
...
```

`scripts/plot.py` genera:
1. `results/exec_time_vs_threads.png` — Tiempo ejecución vs hilos
2. `results/speedup_vs_threads.png` — Speedup vs hilos + curva teórica Amdahl

**Ley de Amdahl:** S(n) = 1 / (f_serial + (1-f_serial)/n)
Estimar f_serial (fracción secuencial) ajustando la curva a los datos medidos.

---

## Fase 7 — Reporte Técnico (`report.md`)

Secciones del reporte:
1. **Especificaciones de Hardware** — ya conocidas, llenar directamente
2. **Metodología** — prompts exactos usados, análisis del código IA
3. **Gráfica Tiempo vs Hilos** — embed `results/exec_time_vs_threads.png`
4. **Gráfica Speedup vs Hilos** — embed `results/speedup_vs_threads.png`
5. **Análisis de Rendimiento** — Amdahl, punto de degradación por overhead SO
6. **Conclusiones**

---

## Historial de Commits Git

| # | Commit message | Archivos |
|---|---------------|---------|
| 1 | `Add sequential Mandelbrot 8K + separable Gaussian blur` | `src/sequential.cpp` |
| 2 | `Add AI parallel baseline using OpenMP static scheduler` | `src/parallel_ai.cpp` |
| 3 | `Add scheduler comparison: static/dynamic/guided with chunksizes` | `src/optimized.cpp` (fase schedulers) |
| 4 | `Add histogram: atomic vs reduction, document false sharing` | `src/optimized.cpp` (fase histograma) |
| 5 | `Add SPMD vectorization and thread affinity (OMP_PROC_BIND/OMP_PLACES)` | `src/optimized.cpp` (fase SPMD) |
| 6 | `Add benchmark scripts and final technical report` | `scripts/`, `results/`, `report.md` |

---

## Verificación (end-to-end)

```bash
# 1. Compilar secuencial
g++ -O2 -o sequential src/sequential.cpp && ./sequential
# Verificar: mandelbrot.ppm y blurred.ppm generados

# 2. Compilar paralelo baseline
g++ -O2 -fopenmp -o parallel_ai src/parallel_ai.cpp && OMP_NUM_THREADS=4 ./parallel_ai

# 3. Compilar optimizado con vectorización
g++ -O3 -march=native -fopenmp -fopt-info-vec-optimized -o optimized src/optimized.cpp

# 4. Correr benchmarks completos
bash scripts/benchmark.sh  # genera results/timings.csv

# 5. Generar gráficas
python3 scripts/plot.py    # genera PNG en results/

# 6. Verificar reporte
# Abrir report.md y confirmar que las imágenes de gráficas están referenciadas correctamente
```

---

## Notas Técnicas Importantes

- **PPM binario (P6):** formato más simple para imágenes sin dependencias externas.
- **Filtro separable vs. completo:** el separable es obligatorio a 8K para que el tiempo sea manejable en este hardware.
- **Overflow en histograma:** con 33M píxeles, usar `long long` o `int64_t` en los contadores.
- **Timer:** usar `omp_get_wtime()` (pared) no `clock()` (CPU acumulada multi-hilo).
- **Warmup:** descartar primera corrida para evitar efectos de caché fría en mediciones.
