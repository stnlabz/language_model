/*
 * STN-Labz native language model — experimental 64-state byte GRU.
 * AI-assisted code, Operator-authorized Milestone 3.
 * No external model, agent integration, networking or background processes.
 *
 * Training uses online truncated BPTT of depth one with state carried over
 * 64-byte segments. This is a deliberately limited experimental optimizer.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define H 64
#define V 256
#define LIMIT (16u * 1024u * 1024u)
#define MAGIC UINT64_C(0x53544e4c4d303031)
#define VERSION 1u
#define WINDOW 64
#define RATE 0.015f

typedef struct {
    float iz[V][H], ir[V][H], in[V][H];
    float uz[H][H], ur[H][H], un[H][H];
    float bz[H], br[H], bn[H];
    float out[H][V], bo[V];
} model;

typedef struct {
    float before[H], z[H], r[H], n[H], h[H], p[V];
    unsigned char input;
} step;

static uint64_t rng = UINT64_C(20261020);
static uint64_t random64(void)
{
    rng ^= rng >> 12;
    rng ^= rng << 25;
    rng ^= rng >> 27;
    return rng * UINT64_C(2685821657736338717);
}
static float uniform(void) { return (float)((random64() >> 40) / 16777216.0 - 0.5); }
static float sigmoid(float x)
{
    if (x < -30.0f) return 0.0f;
    if (x > 30.0f) return 1.0f;
    return 1.0f / (1.0f + expf(-x));
}
static void initialize(model *m)
{
    memset(m, 0, sizeof(*m));
    for (int c = 0; c < V; c++)
        for (int j = 0; j < H; j++) {
            m->iz[c][j] = uniform() * 0.16f;
            m->ir[c][j] = uniform() * 0.16f;
            m->in[c][j] = uniform() * 0.16f;
        }
    for (int i = 0; i < H; i++) {
        for (int j = 0; j < H; j++) {
            m->uz[i][j] = uniform() * 0.12f;
            m->ur[i][j] = uniform() * 0.12f;
            m->un[i][j] = uniform() * 0.12f;
        }
        for (int k = 0; k < V; k++) m->out[i][k] = uniform() * 0.12f;
    }
}
static void forward(const model *m, const float before[H], unsigned char token, step *s)
{
    s->input = token;
    memcpy(s->before, before, sizeof(s->before));
    for (int j = 0; j < H; j++) {
        float az = m->bz[j] + m->iz[token][j];
        float ar = m->br[j] + m->ir[token][j];
        for (int i = 0; i < H; i++) {
            az += before[i] * m->uz[i][j];
            ar += before[i] * m->ur[i][j];
        }
        s->z[j] = sigmoid(az);
        s->r[j] = sigmoid(ar);
    }
    for (int j = 0; j < H; j++) {
        float an = m->bn[j] + m->in[token][j];
        for (int i = 0; i < H; i++) an += (s->r[i] * before[i]) * m->un[i][j];
        s->n[j] = tanhf(an);
        s->h[j] = (1.0f - s->z[j]) * s->n[j] + s->z[j] * before[j];
    }
    float peak = -INFINITY;
    for (int k = 0; k < V; k++) {
        float logit = m->bo[k];
        for (int j = 0; j < H; j++) logit += s->h[j] * m->out[j][k];
        s->p[k] = logit;
        if (logit > peak) peak = logit;
    }
    float sum = 0.0f;
    for (int k = 0; k < V; k++) {
        s->p[k] = expf(s->p[k] - peak);
        sum += s->p[k];
    }
    for (int k = 0; k < V; k++) s->p[k] /= sum;
}
static float update(model *m, step *s, unsigned char target)
{
    float error[V], dh[H] = {0}, dz[H], dn[H], dr[H] = {0};
    float loss = -logf(fmaxf(s->p[target], 1e-20f));
    for (int k = 0; k < V; k++) error[k] = s->p[k] - (k == target ? 1.0f : 0.0f);
    for (int j = 0; j < H; j++)
        for (int k = 0; k < V; k++) dh[j] += m->out[j][k] * error[k];
    for (int j = 0; j < H; j++) {
        dn[j] = dh[j] * (1.0f - s->z[j]) * (1.0f - s->n[j] * s->n[j]);
        dz[j] = dh[j] * (s->before[j] - s->n[j]) * s->z[j] * (1.0f - s->z[j]);
    }
    for (int i = 0; i < H; i++) {
        float d = 0;
        for (int j = 0; j < H; j++) d += m->un[i][j] * dn[j];
        dr[i] = d * s->before[i] * s->r[i] * (1.0f - s->r[i]);
    }
    /* Gradient clipping is componentwise to bound single-step changes. */
    for (int k = 0; k < V; k++) {
        float e = fmaxf(-1.0f, fminf(1.0f, error[k]));
        m->bo[k] -= RATE * e;
        for (int j = 0; j < H; j++) m->out[j][k] -= RATE * e * s->h[j];
    }
    for (int j = 0; j < H; j++) {
        float z = fmaxf(-1.0f, fminf(1.0f, dz[j]));
        float r = fmaxf(-1.0f, fminf(1.0f, dr[j]));
        float n = fmaxf(-1.0f, fminf(1.0f, dn[j]));
        m->iz[s->input][j] -= RATE * z;
        m->ir[s->input][j] -= RATE * r;
        m->in[s->input][j] -= RATE * n;
        m->bz[j] -= RATE * z;
        m->br[j] -= RATE * r;
        m->bn[j] -= RATE * n;
        for (int i = 0; i < H; i++) {
            m->uz[i][j] -= RATE * z * s->before[i];
            m->ur[i][j] -= RATE * r * s->before[i];
            m->un[i][j] -= RATE * n * (s->r[i] * s->before[i]);
        }
    }
    return loss;
}
static unsigned char *read_data(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned char *p = malloc(LIMIT + 1u);
    if (!p) { fclose(f); return NULL; }
    size_t got = fread(p, 1, LIMIT + 1u, f);
    int failed = ferror(f);
    int eof = feof(f);
    fclose(f);
    if (failed || !eof || got < 2 || got > LIMIT) { free(p); return NULL; }
    *n = got;
    return p;
}
static float evaluate(const model *m, const unsigned char *p, size_t n)
{
    float h[H] = {0};
    double loss = 0;
    step s;
    for (size_t t = 0; t + 1 < n; t++) {
        forward(m, h, p[t], &s);
        loss -= log(fmax((double)s.p[p[t+1]], 1e-20));
        memcpy(h, s.h, sizeof(h));
        if ((t + 1) % WINDOW == 0) memset(h, 0, sizeof(h));
    }
    return (float)(loss / (double)(n - 1));
}
static uint64_t checksum(const unsigned char *p, size_t n)
{
    uint64_t h = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= UINT64_C(1099511628211); }
    return h;
}
static int save(const char *path, const model *m)
{
    /* No replacement of existing files; checkpoint is explicitly requested. */
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return 0; }
    f = fopen(path, "wb");
    if (!f) return 0;
    const uint64_t header[4] = {MAGIC, VERSION, sizeof(*m),
        checksum((const unsigned char *)m, sizeof(*m))};
    int ok = fwrite(header, sizeof(header), 1, f) == 1 &&
             fwrite(m, sizeof(*m), 1, f) == 1;
    if (fclose(f)) ok = 0;
    if (!ok) remove(path);
    return ok;
}
static int restore(const char *path, model *m)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint64_t header[4];
    int ok = fread(header, sizeof(header), 1, f) == 1 &&
             header[0] == MAGIC && header[1] == VERSION &&
             header[2] == sizeof(*m);
    if (ok) ok = fread(m, sizeof(*m), 1, f) == 1 && fgetc(f) == EOF &&
                 checksum((const unsigned char *)m, sizeof(*m)) == header[3];
    fclose(f);
    return ok;
}
static void usage(const char *name)
{
    fprintf(stderr, "usage:\n  %s train <train.bin> <validation.bin> <steps:1..100000> <new-checkpoint>\n  %s eval <checkpoint> <validation.bin>\n  %s generate <checkpoint> <ascii-prefix> <length:1..256>\n", name, name, name);
}
static int parse_steps(const char *s, int max)
{
    char *end;
    long n = strtol(s, &end, 10);
    return (s[0] && !*end && n >= 1 && n <= max) ? (int)n : 0;
}
int main(int argc, char **argv)
{
    if (argc < 2) { usage(argv[0]); return 2; }
    model *m = malloc(sizeof(*m));
    if (!m) return 1;
    if (argc == 6 && !strcmp(argv[1], "train")) {
        int steps = parse_steps(argv[4], 100000);
        if (!steps || !strcmp(argv[2], argv[3]) || !strcmp(argv[2], argv[5]) ||
            !strcmp(argv[3], argv[5])) { usage(argv[0]); free(m); return 2; }
        size_t n = 0, v = 0;
        unsigned char *train = read_data(argv[2], &n);
        unsigned char *valid = read_data(argv[3], &v);
        if (!train || !valid) {
            fputs("missing, short, or oversized dataset\n", stderr);
            free(train); free(valid); free(m); return 1;
        }
        initialize(m);
        printf("parameters=%zu initial_validation_loss=%.6f\n",
               sizeof(*m)/sizeof(float), evaluate(m, valid, v));
        float h[H] = {0};
        step s;
        double total = 0;
        for (int t = 0; t < steps; t++) {
            size_t pos = (size_t)t % (n - 1u);
            if (pos == 0 || pos % WINDOW == 0) memset(h, 0, sizeof(h));
            forward(m, h, train[pos], &s);
            total += update(m, &s, train[pos + 1u]);
            memcpy(h, s.h, sizeof(h));
        }
        printf("steps=%d training_loss=%.6f validation_loss=%.6f\n",
               steps, total / steps, evaluate(m, valid, v));
        int ok = save(argv[5], m);
        if (!ok) fputs("cannot create checkpoint (or already exists)\n", stderr);
        free(train); free(valid); free(m);
        return ok ? 0 : 1;
    }
    if (argc == 4 && !strcmp(argv[1], "eval")) {
        size_t n = 0;
        unsigned char *data = read_data(argv[3], &n);
        if (!data || !restore(argv[2], m)) {
            fputs("invalid data or checkpoint\n", stderr);
            free(data); free(m); return 1;
        }
        printf("validation_loss=%.6f\n", evaluate(m, data, n));
        free(data); free(m); return 0;
    }
    if (argc == 5 && !strcmp(argv[1], "generate")) {
        int count = parse_steps(argv[4], 256);
        if (!count || !restore(argv[2], m) || !argv[3][0]) {
            fputs("invalid prefix, output limit, or checkpoint\n", stderr);
            free(m); return 1;
        }
        float h[H] = {0};
        step s;
        const unsigned char *p = (const unsigned char *)argv[3];
        for (size_t i = 0; p[i]; i++) {
            forward(m, h, p[i], &s);
            memcpy(h, s.h, sizeof(h));
        }
        for (int i = 0; i < count; i++) {
            float u = (float)((random64() >> 40) / 16777216.0);
            int next = 255;
            for (int k = 0; k < V; k++) {
                u -= s.p[k];
                if (u <= 0) { next = k; break; }
            }
            /* Escaped bytes make arbitrary generated values safe to inspect. */
            printf("\\x%02x", (unsigned int)next);
            forward(m, h, (unsigned char)next, &s);
            memcpy(h, s.h, sizeof(h));
        }
        putchar('\n');
        free(m); return 0;
    }
    usage(argv[0]); free(m); return 2;
}
