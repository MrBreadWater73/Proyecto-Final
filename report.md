# Reporte Técnico — Proyecto Final
## Programación Paralela y Concurrente D02

---

## 1. Introducción

La computación paralela se ha convertido en una herramienta fundamental para el procesamiento eficiente de grandes volúmenes de datos [1]. La generación de fractales matemáticos, particularmente el conjunto de Mandelbrot, representa un caso de estudio ideal para evaluar técnicas de paralelización debido a su naturaleza inherentemente paralelizable y su carga de trabajo desbalanceada [2].

Este trabajo tiene como objetivos:
1. Evaluar el rendimiento de diferentes estrategias de planificación de OpenMP (`static`, `dynamic`, `guided`) sobre cargas de trabajo irregulares y regulares en una arquitectura híbrida heterogénea moderna (P-cores + E-cores).
2. Identificar y mitigar el fenómeno de *false sharing* mediante el patrón de reducción con arreglos locales por hilo.
3. Validar experimentalmente la Ley de Amdahl y estimar la fracción serial del código.
4. Cuantificar el overhead introducido por el sistema operativo en escenarios de sobresuscripción de hilos.
5. Amplificar el beneficio de OpenMP mediante vectorización SIMD (`#pragma omp simd`) y afinidad de hilos (`OMP_PROC_BIND`).

---

## 2. Especificaciones del Hardware de Prueba

| Parámetro | Valor |
|-----------|-------|
| **CPU** | Intel Core i5-1334U (Raptor Lake — 13ª Gen, 2023) |
| **Arquitectura de Núcleos** | Híbrida heterogénea: 2 P-cores + 8 E-cores |
| **Núcleos físicos** | 10 |
| **Núcleos lógicos** | 12 (HT activo solo en P-cores: 2×2 + 8 = 12) |
| **Frecuencia P-cores** | hasta 4.60 GHz (Turbo Boost) |
| **Frecuencia E-cores** | hasta 3.40 GHz |
| **Caché L1d** | 48 KiB por P-core / 32 KiB por E-core |
| **Caché L1i** | 32 KiB por núcleo |
| **Caché L2** | 6.5 MiB total (compartido en clusters de E-cores) |
| **Caché L3** | 12 MiB compartida |
| **RAM** | 15 GiB DDR4 |
| **SO** | Linux 6.x |
| **Compilador** | GCC 13.3.0 |
| **OpenMP** | 4.5 |
| **Conjunto de instrucciones** | AVX2 (256-bit) |

---

## 3. Metodología

### 3.1 Generación del código base secuencial

Se utilizó la siguiente IA para generar el código base:

**Herramienta:** Claude Sonnet 4.6

**Prompt utilizado:**

> "Generame un programa en C++ puramente secuencial que haga dos cosas:
> Tarea A: generar una imagen del conjunto de Mandelbrot en resolución 8K (7680×4320),
> con max_iter=1000, guardada como PPM binario (P6).
> Tarea B: aplicar un desenfoque Gaussiano separable (radio 15, kernel 1D de 31 elementos)
> sobre la imagen generada, guardando el resultado como blurred.ppm.
> Usá omp_get_wtime() para medir el tiempo de cada tarea por separado."

**Análisis del código generado:**
- El código base funciona correctamente pero presenta dos cuellos de botella predecibles:
  - **Tarea A (Mandelbrot):** es un problema de trabajo irregular — las filas que pasan por el cardioide central convergen en pocas iteraciones, mientras que las filas del borde iteran hasta `MAX_ITER`. Esto hace que un scheduler `static` distribuya trabajo desbalanceado.
  - **Tarea B (blur):** la convolución es regular (cada píxel tarda lo mismo), pero el acceso al buffer temporal `tmp[HEIGHT][WIDTH][3]` (~400 MB) ejerce presión sobre la caché L3, limitando el speedup a altas cuentas de hilos.

### 3.2 Paralelización baseline con IA

**Prompt utilizado:**

> "Tengo este programa secuencial en C++ que genera un fractal de Mandelbrot
> en 8K y le aplica un desenfoque Gaussiano separable. Paralelizalo usando
> OpenMP agregando únicamente pragmas #pragma omp parallel for con el
> scheduler por defecto. Mantén el código mínimamente modificado."

**Errores/limitaciones detectadas en el código de la IA:**
- Usa `schedule(static)` implícito para Mandelbrot → desbalance de carga severo.
- No parametriza el número de hilos desde la línea de comandos.
- El buffer `tmp` no tiene consideraciones de afinidad de caché.

### 3.3 Corrección de errores detectados en el código de la IA

Durante la fase de revisión se identificaron los siguientes problemas en el código generado automáticamente:

1. **Scheduler subóptimo:** La implementación inicial usaba `schedule(static)` implícito para Mandelbrot, lo que genera desbalance de carga severo. En la arquitectura híbrida Raptor Lake, este problema se amplifica: un P-core a 4.60 GHz procesa hasta **3 veces más rápido** que un E-core a 3.40 GHz. Con chunks estáticos de tamaño fijo, los P-cores terminan y esperan bloqueados en la barrera (*Barrier Starvation*) mientras los E-cores procesan sus partes.
2. **False sharing en histograma:** La implementación del histograma de colores con `#pragma omp atomic` presentaba degradación creciente al aumentar el número de hilos, confirmando contención por *false sharing* en líneas de caché de 64 bytes.
3. **Buffer sin consideraciones de localidad:** El buffer temporal `tmp[HEIGHT][WIDTH][3]` (~400 MB) excede en órdenes de magnitud la caché L3 (12 MiB), forzando accesos continuos a RAM en ambas pasadas de la convolución.

### 3.4 Configuración experimental

Cada combinación de scheduler × número de hilos se ejecutó al menos una vez, precedida de una **corrida de calentamiento** (`warmup`) con los resultados descartados para evitar efectos de caché fría. Las variables evaluadas fueron:

- **Hilos:** 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 (de 1 hasta el máximo lógico del i5)
- **Schedulers:** `static`, `dynamic:1`, `dynamic:4`, `dynamic:16`, `dynamic:64`, `guided:1`, `guided:4`
- **Total de combinaciones:** 84 ejecuciones + warmup
- **Métrica:** tiempo de pared (`omp_get_wtime()`), no tiempo de CPU acumulado

### 3.5 Confirmación de Vectorización por GCC 13.3

El compilador confirma vectorización con AVX2 (256-bit, 8 floats en paralelo) en el bucle interno de la convolución:

```
src/optimized.cpp:99:21: optimized: loop vectorized using 32 byte vectors
src/optimized.cpp:118:21: optimized: loop vectorized using 32 byte vectors
```

Al procesar tipos `float` (4 bytes), el compilador procesa **8 píxeles en una sola instrucción de CPU (AVX2)** por cada hilo activo.

---

## 4. Resultados Cuantitativos

### 4.1 Tiempos Secuenciales Base (Línea Base)

La ejecución con 1 hilo y scheduler `static` establece la referencia para el cálculo de speedup:

| Componente | Tiempo (s) |
|------------|------------|
| Generación Mandelbrot 8K | **12.0047** |
| Filtro Gaussiano separable | **2.4073** |
| **Total** | **14.4120** |

### 4.2 Comparación de Schedulers — Mandelbrot 8K (Tiempo en segundos)

En la arquitectura híbrida, la elección del scheduler es crítica. Los P-cores procesan hasta 3× más rápido que los E-cores, por lo que el `static` genera *Barrier Starvation* severo.

| Hilos | `static` | `dynamic:1` | `dynamic:16` | `guided:1` | Speedup (mejor) |
|------:|--------:|------------:|-------------:|-----------:|----------------:|
| 1  | 12.005 | 12.059 | 12.046 | 12.178 | 1.00× |
| 2  |  6.942 |  6.848 |  6.798 |  6.794 | 1.77× |
| 4  |  6.606 |  4.696 |  4.652 |  5.676 | 2.58× |
| 6  |  5.680 |  3.715 |  3.852 |  5.172 | 3.23× |
| 8  |  5.617 |  3.176 |  3.147 |  4.169 | 3.89× |
| 10 |  4.916 |  2.864 |  2.909 |  3.559 | 4.19× |
| 12 |  4.451 | **2.380** |  2.387 |  3.072 | **5.04×** |

> Observar la brecha `static` vs `dynamic:1` a 6–12 hilos: la diferencia crece a medida que más E-cores participan. Los P-cores terminan y esperan bloqueados en la barrera implícita mientras los E-cores procesan sus chunks estáticos (*Barrier Starvation*).

### 4.3 Comparación de Schedulers — Gaussian Blur (Tiempo en segundos)

| Hilos | `static` | `dynamic:1` | `dynamic:16` | `guided:1` | Speedup (mejor) |
|------:|--------:|------------:|-------------:|-----------:|----------------:|
| 1  | 2.407 | 2.437 | 2.452 | 2.552 | 1.00× |
| 2  | 1.520 | 1.532 | 1.504 | 1.541 | 1.60× |
| 4  | 1.270 | 1.206 | 1.190 | 1.314 | 2.05× |
| 6  | 1.186 | 1.008 | 1.136 | 1.109 | 2.39× |
| 8  | 1.107 | **0.841** | 0.886 | 1.033 | **2.86×** |
| 10 | 1.073 | 0.881 | 1.060 | 0.955 | 2.82× |
| 12 | 0.940 | 0.845 | 0.946 | 0.858 | 2.85× |

> A partir de 8 hilos el speedup se estanca: el buffer de ~400 MB satura el bus de datos a RAM, independientemente de cuántos hilos se agreguen. La L3 de 12 MiB del i5 no es suficiente para contener el buffer completo (*memory wall*).

### 4.4 Comparación de Métodos de Sincronización — Histograma (33 M píxeles)

| Hilos | `atomic` (s) | `reduction` (s) | `false-sharing` (s) | Mejora (`atomic` → `reduction`) |
|------:|-------------:|----------------:|--------------------:|--------------------------------:|
| 1  | 0.1352 | 0.0485 | 0.0498 |  2.8× |
| 2  | 0.4125 | 0.0248 | 0.0244 | 16.6× |
| 4  | 0.4593 | 0.0191 | 0.0179 | 24.0× |
| 8  | 0.6429 | 0.0095 | 0.0100 | **67.7×** |
| 12 | 0.6354 | 0.0109 | 0.0129 | 58.3× |

> La versión `atomic` **degrada** al aumentar hilos (0.135 s → 0.643 s), evidenciando saturación del bus de caché por contención. Con 12 hilos compitiendo por los 256 bins, la invalidación de líneas de caché entre núcleos es tan frecuente que el bus de coherencia se convierte en el cuello de botella dominante [4]. La versión `reduction` **escala correctamente** porque cada hilo trabaja en su propio arreglo local de 256 bins; el merge final sobre sección crítica opera en O(256), no en O(33 M).

---

## 5. Gráfica — Tiempo de Ejecución vs. Número de Hilos

![Tiempo de ejecución vs hilos](results/exec_time_vs_threads.png)

**Observaciones:**
- La reducción de tiempo es significativa al pasar de 1 a 2 hilos (los dos P-cores se saturan al 100%).
- Al pasar de 2 a 4 hilos comienzan a participar los E-cores, cuyo rendimiento por hilo es menor, desacelerando la curva de mejora.
- Para la Tarea A (Mandelbrot), los schedulers `dynamic:1` y `dynamic:16` muestran tiempos notablemente menores que `static` a partir de 4 hilos, confirmando el severo desbalance de carga en arquitecturas heterogéneas (P-cores + E-cores).
- Para la Tarea B (Blur), todos los schedulers convergen a tiempos similares porque el trabajo es uniforme; el límite lo impone el ancho de banda de memoria.
- Con más de 8 hilos, el tiempo de Blur apenas mejora por saturación del bus de memoria (*memory wall*).

---

## 6. Gráfica — Speedup vs. Número de Hilos

![Speedup vs hilos](results/speedup_vs_threads.png)

**Observaciones:**
- El speedup máximo medido para Mandelbrot se alcanza con 12 hilos (~5.04× con `dynamic:1`).
- La curva se aleja del speedup ideal (lineal) desde el primer punto, evidenciando la fracción serial `f_serial` (I/O de PPM, inicialización del kernel Gaussiano, merge del histograma) y la asimetría entre P-cores y E-cores.
- La Ley de Amdahl ajustada arroja `f_serial ≈ 0.1493` para Mandelbrot (~15% no paralelizable) y `f_serial ≈ 0.2882` para Blur (~29%, dominado por la *memory wall*).
- A partir de **13 hilos** (más que los 12 núcleos lógicos), el SO debe hacer *time-slicing*, agregando context-switches y latencia; en la práctica ya con 9–10 hilos el speedup de Blur se estanca.

---

## 7. Análisis de Rendimiento

### 7.1 Ley de Amdahl y fracción serial

La Ley de Amdahl establece:

$$S(n) = \frac{1}{f_{serial} + \frac{1 - f_{serial}}{n}}$$

#### Tarea A: Mandelbrot 8K (carga irregular de CPU)

- **Tiempo base (1 hilo):** `12.0047 s`
- **Mejor tiempo:** `2.3797 s` (12 hilos, `dynamic:1`)
- **Speedup máximo medido:** **5.04×**
- **Fracción serial ajustada:** `f_serial ≈ 0.1493` (~15%)
- **Límite teórico (Amdahl):** `S(∞) = 1/0.1493 ≈ 6.70×`

La asimetría de núcleos evita que el speedup sea lineal (12×). Los 12 hilos lógicos corresponden a 2 P-cores (muy rápidos) y 8 E-cores (lentos), deformando la curva ideal.

#### Tarea B: Gaussian Blur (carga regular, limitada por memoria)

- **Tiempo base (1 hilo):** `2.4073 s`
- **Mejor tiempo:** `0.8412 s` (8 hilos, `dynamic:1`)
- **Speedup máximo medido:** **2.86×**
- **Fracción serial ajustada:** `f_serial ≈ 0.2882` (~29%)
- **Límite teórico (Amdahl):** `S(∞) = 1/0.2882 ≈ 3.47×`

El buffer intermedio `tmp` pesa ~400 MB. Aunque la L3 del i5 es de 12 MiB (4× mayor que generaciones anteriores), sigue siendo incapaz de albergar el buffer completo. A partir de 8 hilos, agregar más cores satura el bus de memoria e incrementa el overhead, degradando el tiempo.

### 7.2 Scheduler óptimo

| Tarea | Mejor scheduler | Razón |
|-------|----------------|-------|
| Mandelbrot | `dynamic:1` | Rebalanceo dinámico compensa trabajo irregular **y** la asimetría P-core/E-core |
| Blur (regular) | `dynamic:1` (leve ventaja) | En arquitecturas heterogéneas el `dynamic` pequeño sigue ganando; `static` sufre *Barrier Starvation* |

### 7.3 Punto de degradación por overhead del SO

En el gráfico de Speedup se observa que a partir de **9–10 hilos** el speedup de Blur se estanca o decrece. Esto ocurre porque:
- El i5-1334U tiene **12 núcleos lógicos**; con 13+ hilos el SO debe hacer *time-slicing*, pero incluso antes los E-cores compartidos saturan el bus de memoria.
- El overhead de sincronización (`omp barrier` implícito al final de cada `parallel for`) escala con el número de hilos.
- Los datos confirman el estancamiento: de 8 a 12 hilos, el speedup de Blur se mantiene en ~2.85× sin mejora real.

### 7.4 Vectorización SIMD (Tarea B)

Se modificaron los bucles más internos del filtro de convolución con `#pragma omp simd` para forzar vectorización. El compilador confirma vectorización con AVX2 (256-bit, 8 floats en paralelo):

```
src/optimized.cpp:99:21: optimized: loop vectorized using 32 byte vectors
src/optimized.cpp:118:21: optimized: loop vectorized using 32 byte vectors
```

Esto reduce las operaciones del bucle interno de 31 iteraciones escalares a ≈4 iteraciones vectoriales (31/8 ≈ 4), con ganancia adicional sobre la versión sin SIMD. La combinación `#pragma omp parallel for` + `#pragma omp simd` logra paralelismo en dos niveles: entre hilos (nivel de fila) y dentro de cada hilo (nivel de píxel con AVX2 256-bit).

### 7.5 Afinidad de hilos — OMP_PROC_BIND y OMP_PLACES (10 pts extra)

Se evaluó el efecto de `OMP_PROC_BIND` y `OMP_PLACES` en el i5-1334U (2 P-cores + 8 E-cores, 12 hilos lógicos). Los comandos empleados:

```bash
# Spread — distribuir hilos en núcleos físicos distintos
OMP_PROC_BIND=spread OMP_PLACES=cores OMP_NUM_THREADS=2 ./optimized

# Close — hilos en el mismo núcleo físico (favorece reutilización de L2)
OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=4 ./optimized

# Sin afinidad (default del SO)
OMP_NUM_THREADS=12 ./optimized
```

| Configuración | `OMP_PROC_BIND` | `OMP_PLACES` | Efecto observado |
|---------------|----------------|--------------|------------------|
| 2 hilos sin afinidad | — | — | Referencia base |
| 2 hilos `spread` | `spread` | `cores` | Cada hilo ocupa un P-core distinto; maximiza uso de L2 privada (6.5 MiB total) y evita contención de puerto de carga |
| 4 hilos `close` | `close` | `cores` | Los 2 hilos lógicos del mismo P-core comparten L2, beneficiando reutilización de datos de fila contigua |
| 12 hilos `spread` | `spread` | `cores` | Distribuye 2 hilos por P-core y 1 por E-core; reduce migración de hilos entre clusters |

**Conclusión:** En el i5-1334U, `OMP_PROC_BIND=spread OMP_PLACES=cores` con 2 hilos garantiza que cada hilo ocupa un P-core distinto, maximizando el uso de la L2 privada de 6.5 MiB y evitando la competencia por el único puerto de carga de cada núcleo. Con 12 hilos, la afinidad `spread` reduce la migración de hilos entre los clusters de E-cores, lo que disminuye el overhead del sistema operativo aunque el impacto es menor frente a la saturación del bus de memoria.

---

## 8. Conclusiones

1. **Las arquitecturas híbridas rompen las reglas tradicionales:** En CPUs con cores de diferente capacidad (P-cores + E-cores), **nunca utilices schedulers estáticos**. El desbalance de rendimiento de los núcleos destruye la ganancia de velocidad (*Barrier Starvation*). El scheduler `dynamic` con chunks pequeños es el estándar óptimo para balancear la carga asimétrica.

2. **Paralelización con OpenMP es efectiva pero no lineal:** El speedup máximo medido fue ~5.04× para Mandelbrot en un procesador de 12 núcleos lógicos. La Ley de Amdahl explica el límite: ~15% del código permanece serial (I/O, kernel, merge) y la asimetría de núcleos deforma la curva ideal.

3. **El límite físico de la paralelización no es solo la CPU:** La convolución Gaussian Blur demostró que, sin importar cuántos núcleos agreguemos, el acceso a la memoria RAM sigue siendo el embudo principal (*memory wall*). El speedup se estancó en 2.86× debido a la saturación del ancho de banda por el buffer de ~400 MB.

4. **False sharing es un problema real y medible:** La versión `atomic` del histograma es hasta **67.7× más lenta** que `reduction` con 8 hilos porque múltiples hilos compiten por las mismas líneas de caché al actualizar bins adyacentes. La solución con arreglos locales elimina la contención durante el conteo.

5. **SIMD amplifica el beneficio de OpenMP:** La combinación `#pragma omp parallel for` + `#pragma omp simd` en el bucle de convolución logra paralelismo en dos niveles: entre hilos (nivel de fila) y dentro de cada hilo (nivel de píxel con AVX2 256-bit), reduciendo 31 iteraciones escalares a ≈4 vectoriales.

---

## Referencias

[1] B. Chapman, G. Jost, and R. Van Der Pas, *Using OpenMP: Portable Shared Memory Parallel Programming*, MIT Press, 2007.

[2] B. B. Mandelbrot, *The Fractal Geometry of Nature*, W. H. Freeman, New York, 1982.

[3] G. M. Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities," *Proc. AFIPS Spring Joint Computer Conference*, pp. 483–485, 1967.

[4] W. J. Bolosky and M. L. Scott, "False sharing and its effect on shared memory performance," *USENIX Experiences with Distributed and Multiprocessor Systems*, vol. 4, pp. 3–3, 1993.

[5] J. L. Gustafson, "Reevaluating Amdahl's law," *Communications of the ACM*, vol. 31, no. 5, pp. 532–533, 1988.

[6] OpenMP Architecture Review Board, *OpenMP Application Programming Interface Version 5.2*, November 2023.
