#include <cmath>
#include <cstdio>
#include <cstring>
#include <omp.h>

static const int WIDTH    = 7680;
static const int HEIGHT   = 4320;
static const int MAX_ITER = 1000;
static const int RADIUS   = 15;

static unsigned char image[HEIGHT][WIDTH][3];
static unsigned char blurred[HEIGHT][WIDTH][3];

static void write_ppm(const char* path, unsigned char data[][WIDTH][3]) {
    FILE* f = fopen(path, "wb");
    fprintf(f, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
    fwrite(data, 3, WIDTH * HEIGHT, f);
    fclose(f);
}

static void mandelbrot() {
    double t0 = omp_get_wtime();

    const double x_min = -2.5, x_max = 1.0;
    const double y_min = -1.25, y_max = 1.25;

    for (int py = 0; py < HEIGHT; py++) {
        double cy = y_max - py * (y_max - y_min) / (HEIGHT - 1);
        for (int px = 0; px < WIDTH; px++) {
            double cx = x_min + px * (x_max - x_min) / (WIDTH - 1);
            double zx = 0.0, zy = 0.0;
            int iter = 0;
            while (zx*zx + zy*zy < 4.0 && iter < MAX_ITER) {
                double tmp = zx*zx - zy*zy + cx;
                zy = 2.0*zx*zy + cy;
                zx = tmp;
                iter++;
            }
            unsigned char c = (iter == MAX_ITER) ? 0 : (unsigned char)(255 * iter / MAX_ITER);
            image[py][px][0] = c;
            image[py][px][1] = (unsigned char)(c * 0.6);
            image[py][px][2] = (unsigned char)(255 - c);
        }
    }

    double t1 = omp_get_wtime();
    printf("Mandelbrot: %.4f s\n", t1 - t0);
}

static void gaussian_blur() {
    double t0 = omp_get_wtime();

    // Precompute 1D Gaussian kernel (sigma = radius/3)
    double kernel[2 * RADIUS + 1];
    double sigma = RADIUS / 3.0;
    double sum = 0.0;
    for (int k = -RADIUS; k <= RADIUS; k++) {
        kernel[k + RADIUS] = exp(-0.5 * k * k / (sigma * sigma));
        sum += kernel[k + RADIUS];
    }
    for (int k = 0; k <= 2 * RADIUS; k++) kernel[k] /= sum;

    // Temporary buffer for horizontal pass
    static float tmp[HEIGHT][WIDTH][3];

    // Horizontal pass
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

    // Vertical pass
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

    double t1 = omp_get_wtime();
    printf("Gaussian blur: %.4f s\n", t1 - t0);
}

int main() {
    printf("=== Sequential (%dx%d) ===\n", WIDTH, HEIGHT);

    mandelbrot();
    write_ppm("mandelbrot.ppm", image);

    gaussian_blur();
    write_ppm("blurred.ppm", blurred);

    return 0;
}
