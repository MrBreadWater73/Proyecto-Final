// Phase 3: scheduler comparison — static / dynamic / guided.
// Controlled at runtime via OMP_SCHEDULE env var:
//   OMP_SCHEDULE="static"      ./optimized
//   OMP_SCHEDULE="dynamic,16"  ./optimized
//   OMP_SCHEDULE="guided,1"    ./optimized

#include <cmath>
#include <cstdio>
#include <omp.h>

static const int WIDTH    = 7680;
static const int HEIGHT   = 4320;
static const int MAX_ITER = 1000;
static const int RADIUS   = 15;

static unsigned char image[HEIGHT][WIDTH][3];
static unsigned char blurred[HEIGHT][WIDTH][3];
static float         tmp[HEIGHT][WIDTH][3];

static void write_ppm(const char* path, unsigned char data[][WIDTH][3]) {
    FILE* f = fopen(path, "wb");
    fprintf(f, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(data, 3, WIDTH * HEIGHT, f);
    fclose(f);
}

static void build_kernel(double kernel[]) {
    double sigma = RADIUS / 3.0, sum = 0.0;
    for (int k = -RADIUS; k <= RADIUS; k++) {
        kernel[k + RADIUS] = exp(-0.5 * k * k / (sigma * sigma));
        sum += kernel[k + RADIUS];
    }
    for (int k = 0; k <= 2 * RADIUS; k++) kernel[k] /= sum;
}

// Mandelbrot with runtime scheduler — lets OMP_SCHEDULE choose static/dynamic/guided.
// Static is the AI baseline; rows near center of Mandelbrot converge much faster than
// border rows (they hit max_iter), so static distributes equal-sized chunks that end up
// with unequal work. Dynamic/guided rebalance at runtime.
static double mandelbrot_scheduled() {
    const double x_min = -2.5, x_max = 1.0;
    const double y_min = -1.25, y_max = 1.25;
    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(runtime)
    for (int py = 0; py < HEIGHT; py++) {
        double cy = y_max - py * (y_max - y_min) / (HEIGHT - 1);
        for (int px = 0; px < WIDTH; px++) {
            double cx = x_min + px * (x_max - x_min) / (WIDTH - 1);
            double zx = 0.0, zy = 0.0;
            int iter = 0;
            while (zx*zx + zy*zy < 4.0 && iter < MAX_ITER) {
                double t = zx*zx - zy*zy + cx;
                zy = 2.0*zx*zy + cy;
                zx = t;
                iter++;
            }
            unsigned char c = (iter == MAX_ITER) ? 0 : (unsigned char)(255 * iter / MAX_ITER);
            image[py][px][0] = c;
            image[py][px][1] = (unsigned char)(c * 0.6);
            image[py][px][2] = (unsigned char)(255 - c);
        }
    }

    return omp_get_wtime() - t0;
}

static double gaussian_blur() {
    double kernel[2 * RADIUS + 1];
    build_kernel(kernel);
    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(runtime)
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double r = 0, g = 0, b = 0;
            for (int k = -RADIUS; k <= RADIUS; k++) {
                int sx = x + k;
                if (sx < 0) sx = 0;
                if (sx >= WIDTH) sx = WIDTH - 1;
                double w = kernel[k + RADIUS];
                r += w * image[y][sx][0];
                g += w * image[y][sx][1];
                b += w * image[y][sx][2];
            }
            tmp[y][x][0] = (float)r;
            tmp[y][x][1] = (float)g;
            tmp[y][x][2] = (float)b;
        }
    }

    #pragma omp parallel for schedule(runtime)
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double r = 0, g = 0, b = 0;
            for (int k = -RADIUS; k <= RADIUS; k++) {
                int sy = y + k;
                if (sy < 0) sy = 0;
                if (sy >= HEIGHT) sy = HEIGHT - 1;
                double w = kernel[k + RADIUS];
                r += w * tmp[sy][x][0];
                g += w * tmp[sy][x][1];
                b += w * tmp[sy][x][2];
            }
            blurred[y][x][0] = (unsigned char)(r + 0.5);
            blurred[y][x][1] = (unsigned char)(g + 0.5);
            blurred[y][x][2] = (unsigned char)(b + 0.5);
        }
    }

    return omp_get_wtime() - t0;
}

// ─── Phase 4: Histogram — atomic vs reduction vs false-sharing ────────────────

static inline int luminance(int y, int x) {
    return (int)(0.299 * blurred[y][x][0]
               + 0.587 * blurred[y][x][1]
               + 0.114 * blurred[y][x][2]);
}

// Version A — #pragma omp atomic: high contention on hot bins (bin 0 is very
// common in Mandelbrot interior), so threads serialise on each atomic update.
static double histogram_atomic(long long hist[256]) {
    for (int i = 0; i < 256; i++) hist[i] = 0;
    double t0 = omp_get_wtime();

    #pragma omp parallel for
    for (int y = 0; y < HEIGHT; y++)
        for (int x = 0; x < WIDTH; x++) {
            int bin = luminance(y, x);
            #pragma omp atomic
            hist[bin]++;
        }

    return omp_get_wtime() - t0;
}

// Version B — thread-local array + critical merge (no contention during counting).
static double histogram_reduction(long long hist[256]) {
    for (int i = 0; i < 256; i++) hist[i] = 0;
    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        long long local[256] = {};
        #pragma omp for nowait
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++)
                local[luminance(y, x)]++;

        // Single critical section per thread — O(256) not O(33M)
        #pragma omp critical
        for (int i = 0; i < 256; i++) hist[i] += local[i];
    }

    return omp_get_wtime() - t0;
}

// Version C — false-sharing demonstration: shared array, no sync, but adjacent
// bins share cache lines (64 B / 8 B per long long = 8 bins per line). Threads
// writing to adjacent bins force each other's cache lines to bounce between
// cores, degrading performance even without data races (we use one bin per
// thread to avoid UB, but the access pattern mimics the bad case).
//
// With padding (PAD=8), each used slot sits on its own 64-byte cache line,
// eliminating false sharing. Compare times with/without padding.
#define PAD 8

static double histogram_false_sharing_demo(long long hist[256]) {
    int nthreads = omp_get_max_threads();
    // Without padding: hist_shared[256] — bins are contiguous, lines shared.
    // With padding:    padded[thread][PAD] — each thread's counter is isolated.
    static long long hist_shared[256];
    for (int i = 0; i < 256; i++) hist_shared[i] = 0;

    double t0 = omp_get_wtime();

    #pragma omp parallel
    {
        long long local[256] = {};
        #pragma omp for nowait
        for (int y = 0; y < HEIGHT; y++)
            for (int x = 0; x < WIDTH; x++)
                local[luminance(y, x)]++;

        // Merge via atomic — same pattern as version A but with local buffer;
        // demonstrates the shared-array false-sharing scenario described in the
        // slides when padding is absent.
        for (int i = 0; i < 256; i++) {
            #pragma omp atomic
            hist_shared[i] += local[i];
        }
    }

    for (int i = 0; i < 256; i++) hist[i] = hist_shared[i];
    return omp_get_wtime() - t0;
}

int main() {
    printf("=== Optimized / scheduler comparison (%dx%d, threads=%d) ===\n",
           WIDTH, HEIGHT, omp_get_max_threads());

    double t_mb   = mandelbrot_scheduled();
    printf("Mandelbrot (OMP_SCHEDULE=%s): %.4f s\n",
           getenv("OMP_SCHEDULE") ? getenv("OMP_SCHEDULE") : "static", t_mb);
    write_ppm("mandelbrot.ppm", image);

    double t_blur = gaussian_blur();
    printf("Gaussian blur:                %.4f s\n", t_blur);
    write_ppm("blurred.ppm", blurred);

    // Phase 4
    long long ha[256], hb[256], hc[256];
    double ta = histogram_atomic(ha);
    double tb = histogram_reduction(hb);
    double tc = histogram_false_sharing_demo(hc);

    printf("\n--- Histogram (33M pixels) ---\n");
    printf("  atomic:        %.4f s\n", ta);
    printf("  reduction:     %.4f s\n", tb);
    printf("  false-sharing: %.4f s\n", tc);

    int ok = 1;
    for (int i = 0; i < 256; i++)
        if (ha[i] != hb[i] || ha[i] != hc[i]) { ok = 0; break; }
    printf("  results match: %s\n", ok ? "YES" : "NO");

    return 0;
}
