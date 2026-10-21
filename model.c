/*
 * STN-Labz native language model — experimental 64-state byte GRU.
 * AI-assisted code, Operator-authorized Milestone 3.
 * No external model, agent integration, networking or background processes.
 *
 * Training uses gradient accumulation and truncated backpropagation through
 * 64-byte sequences. No cross-sequence gradient is retained.
 */
#include <errno.h>
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
#define STATE_VERSION 2u
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
/* Truncated backpropagation through up to WINDOW successive byte predictions.
 * All gradients are accumulated before weights change, preserving the
 * forward-pass weights used by the entire sequence. */
static float clamp_gradient(float value)
{
    return fmaxf(-1.0f, fminf(1.0f, value));
}
static double update_sequence(model *m, const step *history,
                              const unsigned char *targets, int length)
{
    model *g = calloc(1, sizeof(*g));
    if (!g) return -1.0;
    float future[H] = {0};
    double loss = 0.0;
    for (int t = length - 1; t >= 0; --t) {
        const step *s = &history[t];
        float error[V], dh[H], dz[H], dn[H], dr[H], previous[H];
        loss -= log(fmax((double)s->p[targets[t]], 1e-20));
        for (int k = 0; k < V; ++k) {
            error[k] = s->p[k] - (k == targets[t] ? 1.0f : 0.0f);
            g->bo[k] += error[k];
            for (int j = 0; j < H; ++j)
                g->out[j][k] += s->h[j] * error[k];
        }
        for (int j = 0; j < H; ++j) {
            dh[j] = future[j];
            for (int k = 0; k < V; ++k)
                dh[j] += m->out[j][k] * error[k];
            dn[j] = dh[j] * (1.0f - s->z[j]) *
                    (1.0f - s->n[j] * s->n[j]);
            dz[j] = dh[j] * (s->before[j] - s->n[j]) *
                    s->z[j] * (1.0f - s->z[j]);
        }
        for (int i = 0; i < H; ++i) {
            float candidate = 0.0f;
            for (int j = 0; j < H; ++j)
                candidate += m->un[i][j] * dn[j];
            dr[i] = candidate * s->before[i] *
                    s->r[i] * (1.0f - s->r[i]);
        }
        /* Compute all reset-gate derivatives before the previous-state
         * gradient. This avoids using partially initialized dr values. */
        for (int i = 0; i < H; ++i) {
            previous[i] = dh[i] * s->z[i];
            float candidate = 0.0f;
            for (int j = 0; j < H; ++j) {
                candidate += m->un[i][j] * dn[j];
                previous[i] += m->uz[i][j] * dz[j] +
                               m->ur[i][j] * dr[j];
            }
            previous[i] += candidate * s->r[i];
        }
        for (int j = 0; j < H; ++j) {
            g->iz[s->input][j] += dz[j];
            g->ir[s->input][j] += dr[j];
            g->in[s->input][j] += dn[j];
            g->bz[j] += dz[j];
            g->br[j] += dr[j];
            g->bn[j] += dn[j];
            for (int i = 0; i < H; ++i) {
                g->uz[i][j] += s->before[i] * dz[j];
                g->ur[i][j] += s->before[i] * dr[j];
                g->un[i][j] += (s->r[i] * s->before[i]) * dn[j];
            }
        }
        memcpy(future, previous, sizeof(future));
        /* A fresh record has a fresh hidden state: do not backpropagate
         * across the preceding newline when a block spans two records. */
        if (t > 0 && history[t - 1].input == '\n')
            memset(future, 0, sizeof(future));
    }
    /* Apply a per-batch learning rate to the *mean* 64-byte gradient.
     * Multiplying RATE by 64 previously made updates up to 0.96 per
     * weight per batch; that can erase learned patterns during adaptation.
     * Keep the trained architecture and checkpoint format unchanged. */
    const float rate = RATE;
    float *weights = (float *)m;
    const float *grads = (const float *)g;
    for (size_t i = 0; i < sizeof(*m) / sizeof(float); ++i)
        weights[i] -= rate * clamp_gradient(grads[i] / (float)length);
    free(g);
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
        /* Records are one training statement per line. Do not carry
         * unrelated sentence context across the newline boundary. */
        if (p[t] == '\n' || (t + 1) % WINDOW == 0)
            memset(h, 0, sizeof(h));
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
static int restore_state(const char *path, model *m, uint64_t *position,
                         uint64_t *data_hash, uint64_t *data_length);
static int restore(const char *path, model *m)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint64_t header[4] = {0};
    int ok = fread(header, sizeof(header), 1, f) == 1 &&
             header[0] == MAGIC && header[1] == VERSION &&
             header[2] == sizeof(*m);
    if (ok) ok = fread(m, sizeof(*m), 1, f) == 1 && fgetc(f) == EOF &&
                 checksum((const unsigned char *)m, sizeof(*m)) == header[3];
    fclose(f);
    if (!ok && header[0] == MAGIC && header[1] == STATE_VERSION) {
        uint64_t position, hash, length;
        return restore_state(path, m, &position, &hash, &length);
    }
    return ok;
}
/* Stateful checkpoints include completed positions and training-data identity.
 * Legacy version-1 checkpoints remain readable for evaluation and generation.
 * Stateful training requires 64-position sessions for identical update grouping. */
static int save_state(const char *path, const model *m, uint64_t position,
                      uint64_t data_hash, uint64_t data_length)
{
    FILE *f = fopen(path, "rb");
    if (f) { fclose(f); return 0; }
    f = fopen(path, "wb");
    if (!f) return 0;
    const uint64_t header[7] = {MAGIC, STATE_VERSION, sizeof(*m),
        checksum((const unsigned char *)m, sizeof(*m)), position,
        data_hash, data_length};
    int ok = fwrite(header, sizeof(header), 1, f) == 1 &&
             fwrite(m, sizeof(*m), 1, f) == 1;
    if (fclose(f)) ok = 0;
    if (!ok) remove(path);
    return ok;
}
static int restore_state(const char *path, model *m, uint64_t *position,
                         uint64_t *data_hash, uint64_t *data_length)
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    uint64_t header[7];
    int ok = fread(header, sizeof(header), 1, f) == 1 &&
             header[0] == MAGIC && header[1] == STATE_VERSION &&
             header[2] == sizeof(*m) && header[4] % WINDOW == 0 &&
             header[6] >= 2 && header[6] <= LIMIT;
    if (ok) ok = fread(m, sizeof(*m), 1, f) == 1 && fgetc(f) == EOF &&
                 checksum((const unsigned char *)m, sizeof(*m)) == header[3];
    if (ok) {
        *position = header[4];
        *data_hash = header[5];
        *data_length = header[6];
    }
    fclose(f);
    return ok;
}
static double train_blocks(model *m, const unsigned char *train, size_t n,
                           uint64_t start, int count)
{
    double total = 0.0;
    for (int first = 0; first < count; first += WINDOW) {
        step history[WINDOW];
        unsigned char targets[WINDOW];
        float h[H] = {0};
        for (int k = 0; k < WINDOW; ++k) {
            size_t pos = (size_t)((start + (uint64_t)first +
                                   (uint64_t)k) % (n - 1u));
            if (pos == 0 || (pos > 0 && train[pos - 1u] == '\n'))
                memset(h, 0, sizeof(h));
            forward(m, h, train[pos], &history[k]);
            targets[k] = train[pos + 1u];
            memcpy(h, history[k].h, sizeof(h));
        }
        double loss = update_sequence(m, history, targets, WINDOW);
        if (loss < 0.0) return -1.0;
        total += loss;
    }
    return total;
}
/* UTF-8 decoder state tracks legal next-byte ranges. A complete sequence
 * has remaining==0; restricting leads near the end prevents truncation. */
typedef struct { int remaining, low, high; } utf8_state;
static int utf8_advance(utf8_state *state, int b)
{
    if (state->remaining) {
        if (b < state->low || b > state->high) return 0;
        --state->remaining;
        state->low = 0x80;
        state->high = 0xbf;
        return 1;
    }
    if (b >= 0 && b <= 0x7f) return 1;
    if (b >= 0xc2 && b <= 0xdf) {
        state->remaining = 1;
        state->low = 0x80; state->high = 0xbf;
    } else if (b == 0xe0) {
        state->remaining = 2;
        state->low = 0xa0; state->high = 0xbf;
    } else if (b >= 0xe1 && b <= 0xec) {
        state->remaining = 2;
        state->low = 0x80; state->high = 0xbf;
    } else if (b == 0xed) {
        state->remaining = 2;
        state->low = 0x80; state->high = 0x9f;
    } else if (b >= 0xee && b <= 0xef) {
        state->remaining = 2;
        state->low = 0x80; state->high = 0xbf;
    } else if (b == 0xf0) {
        state->remaining = 3;
        state->low = 0x90; state->high = 0xbf;
    } else if (b >= 0xf1 && b <= 0xf3) {
        state->remaining = 3;
        state->low = 0x80; state->high = 0xbf;
    } else if (b == 0xf4) {
        state->remaining = 3;
        state->low = 0x80; state->high = 0x8f;
    } else return 0;
    return 1;
}
static int utf8_allowed(const utf8_state *state, int b, int bytes_left)
{
    utf8_state next = *state;
    return utf8_advance(&next, b) && next.remaining <= bytes_left;
}
static int parse_seed(const char *value, uint64_t *seed)
{
    char *end;
    if (!value[0] || value[0] == '-') return 0;
    errno = 0;
    unsigned long long result = strtoull(value, &end, 10);
    if (errno || *end) return 0;
    *seed = (uint64_t)result;
    return 1;
}
/* Decode-only controls. This leaves training and old generate unchanged. */
static int parse_float_range(const char *text, float lo, float hi, float *out)
{
    char *end;
    errno = 0;
    float value = strtof(text, &end);
    if (!text[0] || *end || errno || !isfinite(value) ||
        value < lo || value > hi) return 0;
    *out = value;
    return 1;
}
static void usage(const char *name)
{
    fprintf(stderr, "usage:\n  %s train <train.bin> <validation.bin> <steps:1..100000> <new-checkpoint>\n  %s eval <checkpoint> <validation.bin>\n  %s generate <checkpoint> <utf8-prefix> <length:1..256> [seed:uint64] [utf8]\n", name, name, name);
}
static int parse_steps(const char *s, int max)
{
    char *end;
    long n = strtol(s, &end, 10);
    return (s[0] && !*end && n >= 1 && n <= max) ? (int)n : 0;
}

/* Experimental corpus-vocabulary constrained GRU sampling.
 * The GRU still chooses each byte: this only disallows invented spellings.
 * It does NOT establish grammar, factuality, or prompt comprehension. */
#define LEX_MAX_NODES 200000
typedef struct { int child[27]; unsigned char terminal; } lex_node;
static lex_node *lex_nodes;
static int lex_count;
static int lex_code(unsigned char c) {
    if (c >= 'a' && c <= 'z') return (int)(c-'a');
    if (c >= 'A' && c <= 'Z') return (int)(c-'A');
    if (c == '\'') return 26;
    return -1;
}
static void lex_insert(const unsigned char *word, int n) {
    int node=0;
    if(n<2 || n>32)return;
    for(int i=0;i<n;i++){
        int c=lex_code(word[i]);
        if(c<0)return;
        int child=lex_nodes[node].child[c];
        if(!child){
            if(lex_count>=LEX_MAX_NODES)return;
            child=lex_count++;
            lex_nodes[node].child[c]=child;
        }
        node=child;
    }
    lex_nodes[node].terminal=1;
}
static int lex_load(const char *path) {
    size_t n=0;
    unsigned char *data=read_data(path,&n);
    if(!data)return 0;
    lex_nodes=calloc(LEX_MAX_NODES,sizeof(*lex_nodes));
    if(!lex_nodes){free(data);return 0;}
    lex_count=1;
    for(size_t i=0;i<n;){
        while(i<n && lex_code(data[i])<0)i++;
        size_t start=i;
        while(i<n && lex_code(data[i])>=0)i++;
        if(i>start)lex_insert(data+start,(int)(i-start));
    }
    free(data);
    return lex_count>20;
}
static int lex_allowed(int node,int in_word,int c) {
    int k=lex_code((unsigned char)c);
    if(k>=0){
        /* Uppercase is allowed at the start of a word; the trie
         * validates the spelling case-insensitively. */
        if(c>='A'&&c<='Z'&&in_word)return 0;
        return lex_nodes[node].child[k]!=0;
    }
    if(in_word && !lex_nodes[node].terminal)return 0;
    return c==' ' || c=='.' || c==',' || c=='\n' || c=='?' || c=='!';
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
        double total = 0.0;
        for (int first = 0; first < steps;) {
            int count = steps - first;
            if (count > WINDOW) count = WINDOW;
            step history[WINDOW];
            unsigned char targets[WINDOW];
            float h[H] = {0};
            for (int k = 0; k < count; ++k) {
                size_t pos = (size_t)(first + k) % (n - 1u);
                if (pos == 0) memset(h, 0, sizeof(h));
                forward(m, h, train[pos], &history[k]);
                targets[k] = train[pos + 1u];
                memcpy(h, history[k].h, sizeof(h));
            }
            double block_loss = update_sequence(m, history, targets, count);
            if (block_loss < 0.0) {
                fputs("out of memory during training\n", stderr);
                free(train); free(valid); free(m); return 1;
            }
            total += block_loss;
            first += count;
        }
        printf("steps=%d training_loss=%.6f validation_loss=%.6f\n",
               steps, total / steps, evaluate(m, valid, v));
        int ok = save(argv[5], m);
        if (!ok) fputs("cannot create checkpoint (or already exists)\n", stderr);
        free(train); free(valid); free(m);
        return ok ? 0 : 1;
    }
    if (argc == 6 && !strcmp(argv[1], "train-state")) {
        int count = parse_steps(argv[4], 100000);
        if (!count || count % WINDOW != 0 ||
            !strcmp(argv[2], argv[3]) ||
            !strcmp(argv[2], argv[5]) || !strcmp(argv[3], argv[5])) {
            fputs("stateful training requires 64-position aligned steps and distinct paths\n", stderr);
            free(m); return 2;
        }
        size_t n = 0, v = 0;
        unsigned char *train = read_data(argv[2], &n);
        unsigned char *valid = read_data(argv[3], &v);
        if (!train || !valid) {
            fputs("missing, short, or oversized dataset\n", stderr);
            free(train); free(valid); free(m); return 1;
        }
        uint64_t position = 0, data_hash = checksum(train, n);
        initialize(m);
        printf("parameters=%zu initial_validation_loss=%.6f\n",
               sizeof(*m) / sizeof(float), evaluate(m, valid, v));
        double loss = train_blocks(m, train, n, position, count);
        if (loss < 0.0) {
            fputs("out of memory during training\n", stderr);
            free(train); free(valid); free(m); return 1;
        }
        position += (uint64_t)count;
        printf("steps=%d total_positions=%llu training_loss=%.6f validation_loss=%.6f\n",
               count, (unsigned long long)position, loss / count,
               evaluate(m, valid, v));
        int ok = save_state(argv[5], m, position, data_hash, n);
        if (!ok) fputs("cannot create checkpoint (or already exists)\n", stderr);
        free(train); free(valid); free(m);
        return ok ? 0 : 1;
    }
    /* Explicit new-corpus continuation: preserve weights, reset data position.
     * This command NEVER replaces the source checkpoint. */
    if (argc == 7 && !strcmp(argv[1], "adapt")) {
        int count = parse_steps(argv[5], 100000);
        if (!count || count % WINDOW != 0 ||
            !strcmp(argv[2], argv[3]) || !strcmp(argv[2], argv[4]) ||
            !strcmp(argv[2], argv[6]) || !strcmp(argv[3], argv[4]) ||
            !strcmp(argv[3], argv[6]) || !strcmp(argv[4], argv[6])) {
            fputs("invalid adapt paths or step count\n", stderr);
            free(m); return 2;
        }
        size_t n = 0, v = 0;
        unsigned char *train = read_data(argv[3], &n);
        unsigned char *valid = read_data(argv[4], &v);
        if (!train || !valid || !restore(argv[2], m)) {
            fputs("invalid source checkpoint or new dataset\n", stderr);
            free(train); free(valid); free(m); return 1;
        }
        printf("adapt_source=%s initial_validation_loss=%.6f\n",
               argv[2], evaluate(m, valid, v));
        double loss = train_blocks(m, train, n, 0, count);
        if (loss < 0.0) {
            fputs("out of memory during adaptation\n", stderr);
            free(train); free(valid); free(m); return 1;
        }
        printf("steps=%d total_positions=%d training_loss=%.6f validation_loss=%.6f\n",
               count, count, loss / count, evaluate(m, valid, v));
        int ok = save_state(argv[6], m, (uint64_t)count, checksum(train, n), n);
        if (!ok) fputs("cannot create adapted checkpoint\n", stderr);
        free(train); free(valid); free(m);
        return ok ? 0 : 1;
    }
    if (argc == 7 && !strcmp(argv[1], "resume")) {
        int count = parse_steps(argv[5], 100000);
        if (!count || count % WINDOW != 0 ||
            !strcmp(argv[2], argv[3]) || !strcmp(argv[2], argv[4]) ||
            !strcmp(argv[2], argv[6]) || !strcmp(argv[3], argv[4]) ||
            !strcmp(argv[3], argv[6]) || !strcmp(argv[4], argv[6])) {
            fputs("invalid resume paths or step count (multiple of 64 required)\n", stderr);
            free(m); return 2;
        }
        size_t n = 0, v = 0;
        unsigned char *train = read_data(argv[3], &n);
        unsigned char *valid = read_data(argv[4], &v);
        uint64_t position = 0, data_hash = 0, data_length = 0;
        if (!train || !valid ||
            !restore_state(argv[2], m, &position, &data_hash, &data_length) ||
            data_hash != checksum(train, n) || data_length != n ||
            UINT64_MAX - position < (uint64_t)count) {
            fputs("invalid state checkpoint or training data mismatch\n", stderr);
            free(train); free(valid); free(m); return 1;
        }
        printf("resume_from=%llu initial_validation_loss=%.6f\n",
               (unsigned long long)position, evaluate(m, valid, v));
        double loss = train_blocks(m, train, n, position, count);
        if (loss < 0.0) {
            fputs("out of memory during training\n", stderr);
            free(train); free(valid); free(m); return 1;
        }
        position += (uint64_t)count;
        printf("steps=%d total_positions=%llu training_loss=%.6f validation_loss=%.6f\n",
               count, (unsigned long long)position, loss / count,
               evaluate(m, valid, v));
        int ok = save_state(argv[6], m, position, data_hash, n);
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
    /* Diagnostic only: next-byte probabilities after consuming a prefix. */
    if (argc == 4 && !strcmp(argv[1], "probe")) {
        if (!argv[3][0] || !restore(argv[2], m)) {
            fputs("invalid prefix or checkpoint\n", stderr);
            free(m); return 1;
        }
        float h[H] = {0};
        step s;
        const unsigned char *p = (const unsigned char *)argv[3];
        for (size_t i = 0; p[i]; ++i) {
            forward(m, h, p[i], &s);
            memcpy(h, s.h, sizeof(h));
        }
        for (int k = 0; k < V; ++k)
            printf("%s%.9g", k ? "," : "", (double)s.p[k]);
        putchar('\n');
        free(m); return 0;
    }

    if(argc==10 && !strcmp(argv[1],"sample-lex")){
        int count=parse_steps(argv[5],256);
        uint64_t seed=0;
        float temperature=0,penalty=0;
        if(!count || !argv[4][0] || !restore(argv[2],m) ||
           !parse_seed(argv[6],&seed) ||
           !parse_float_range(argv[7],0.25f,2.0f,&temperature) ||
           !parse_float_range(argv[8],0.0f,2.0f,&penalty) ||
           strcmp(argv[9],"utf8") || !lex_load(argv[3])){
            fputs("invalid sample-lex inputs or insufficient corpus vocabulary\n",stderr);
            free(lex_nodes);free(m);return 2;
        }
        rng=seed?seed:UINT64_C(20261020);
        float h[H]={0};
        step st;
        const unsigned char *prefix=(const unsigned char *)argv[4];
        if(!strcmp(argv[4],"record-start")){
            /* This corpus mostly consists of standalone sentences, not
             * question/answer pairs. Start at a record boundary rather
             * than priming with a largely absent QA format. */
            forward(m,h,'\n',&st);
            memcpy(h,st.h,sizeof(h));
        } else {
            for(size_t i=0;prefix[i];i++){
                forward(m,h,prefix[i],&st);
                memcpy(h,st.h,sizeof(h));
            }
        }
        int node=0,in_word=0,recent_count=0;
        unsigned char recent[32]={0},output[257]={0};
        int emitted=0,last_boundary=0;
        for(int i=0;i<count;i++){
            double w[V]={0},mass=0;
            for(int c=0;c<128;c++){
                if(!lex_allowed(node,in_word,c))continue;
                int repeats=0;
                for(int j=0;j<recent_count;j++)if(recent[j]==(unsigned char)c)repeats++;
                double prob=fmax((double)st.p[c],1e-30);
                w[c]=exp(log(prob)/temperature-(double)penalty*repeats);
                mass+=w[c];
            }
            if(!(mass>0) || !isfinite(mass))break;
            double u=((random64()>>40)/16777216.0)*mass;
            int next=-1,fallback=-1;
            for(int c=0;c<128;c++)if(w[c]>0){
                fallback=c;u-=w[c];if(u<=0){next=c;break;}
            }
            if(next<0)next=fallback;
            if(next<0)break;
            int k=lex_code((unsigned char)next);
            if(k>=0){node=lex_nodes[node].child[k];in_word=1;}
            else {node=0;in_word=0;last_boundary=i+1;}
            output[emitted++]=(unsigned char)next;
            recent[i%32]=(unsigned char)next;
            if(recent_count<32)recent_count++;
            forward(m,h,(unsigned char)next,&st);
            memcpy(h,st.h,sizeof(h));
        }
        /* Avoid displaying cut-off invented fragments. */
        output[last_boundary]=0;
        puts((const char *)output);
        free(lex_nodes);free(m);return last_boundary>0?0:1;
    }
    /* Experimental sampler: byte-level temperature and recent-byte penalty.
     * Fixed seed and unchanged checkpoint support controlled comparisons. */
    if (argc == 9 && !strcmp(argv[1], "sample")) {
        int count = parse_steps(argv[4], 256);
        uint64_t seed = 0;
        float temperature = 0.0f, penalty = 0.0f;
        if (!count || !argv[3][0] || !restore(argv[2], m) ||
            !parse_seed(argv[5], &seed) ||
            !parse_float_range(argv[6], 0.25f, 2.0f, &temperature) ||
            !parse_float_range(argv[7], 0.0f, 2.0f, &penalty) ||
            strcmp(argv[8], "utf8")) {
            fputs("invalid sample arguments\n", stderr);
            free(m); return 2;
        }
        rng = seed ? seed : UINT64_C(20261020);
        float h[H] = {0};
        step s;
        utf8_state state = {0, 0, 0};
        const unsigned char *prefix = (const unsigned char *)argv[3];
        for (size_t i = 0; prefix[i]; ++i) {
            if (!utf8_advance(&state, prefix[i])) {
                fputs("invalid UTF-8 prefix\n", stderr); free(m); return 2;
            }
            forward(m, h, prefix[i], &s);
            memcpy(h, s.h, sizeof(h));
        }
        if (state.remaining) {
            fputs("incomplete UTF-8 prefix\n", stderr); free(m); return 2;
        }
        unsigned char recent[32] = {0};
        int recent_count = 0;
        for (int i = 0; i < count; ++i) {
            double weights[V] = {0};
            double mass = 0.0;
            for (int k = 0; k < V; ++k) {
                if (!utf8_allowed(&state, k, count - i - 1)) continue;
                int repeats = 0;
                for (int q = 0; q < recent_count; ++q)
                    if (recent[q] == (unsigned char)k) ++repeats;
                /* Apply penalty to repeated bytes, not words or meaning. */
                double probability = fmax((double)s.p[k], 1e-30);
                weights[k] = exp(log(probability) / temperature -
                                 (double)penalty * repeats);
                mass += weights[k];
            }
            if (!(mass > 0.0) || !isfinite(mass)) {
                fputs("no legal sample candidate\n", stderr);
                free(m); return 1;
            }
            double u = ((random64() >> 40) / 16777216.0) * mass;
            int next = -1, fallback = -1;
            for (int k = 0; k < V; ++k) {
                if (weights[k] <= 0.0) continue;
                fallback = k;
                u -= weights[k];
                if (u <= 0.0) { next = k; break; }
            }
            if (next < 0) next = fallback;
            if (next < 0 || !utf8_advance(&state, next)) {
                fputs("invalid UTF-8 sample state\n", stderr);
                free(m); return 1;
            }
            printf("\\x%02x", (unsigned int)next);
            recent[i % 32] = (unsigned char)next;
            if (recent_count < 32) ++recent_count;
            forward(m, h, (unsigned char)next, &s);
            memcpy(h, s.h, sizeof(h));
        }
        putchar('\n');
        free(m); return 0;
    }
    if (argc >= 5 && argc <= 7 && !strcmp(argv[1], "generate")) {
        int count = parse_steps(argv[4], 256);
        int strict_utf8 = argc == 7 && !strcmp(argv[6], "utf8");
        uint64_t seed = UINT64_C(20261020);
        if (!count || !restore(argv[2], m) || !argv[3][0] ||
            (argc >= 6 && !parse_seed(argv[5], &seed)) ||
            (argc == 7 && !strict_utf8)) {
            fputs("invalid prefix, output limit, seed, mode, or checkpoint\n", stderr);
            free(m); return 1;
        }
        rng = seed ? seed : UINT64_C(20261020);
        float h[H] = {0};
        step s;
        const unsigned char *p = (const unsigned char *)argv[3];
        utf8_state state = {0, 0, 0};
        for (size_t i = 0; p[i]; i++) {
            if (strict_utf8 && !utf8_advance(&state, p[i])) {
                fputs("invalid UTF-8 prefix\n", stderr);
                free(m); return 1;
            }
            forward(m, h, p[i], &s);
            memcpy(h, s.h, sizeof(h));
        }
        if (strict_utf8 && state.remaining) {
            fputs("incomplete UTF-8 prefix\n", stderr);
            free(m); return 1;
        }
        for (int i = 0; i < count; i++) {
            float u = (float)((random64() >> 40) / 16777216.0);
            int next = 255;
            if (strict_utf8) {
                float mass = 0.0f;
                for (int k = 0; k < V; k++)
                    if (utf8_allowed(&state, k, count - i - 1))
                        mass += s.p[k];
                if (!(mass > 0.0f) || !isfinite(mass)) {
                    fputs("no legal UTF-8 next byte\n", stderr);
                    free(m); return 1;
                }
                u *= mass;
                for (int k = 0; k < V; k++) {
                    if (!utf8_allowed(&state, k, count - i - 1)) continue;
                    next = k;
                    u -= s.p[k];
                    if (u <= 0) break;
                }
                /* Last legal candidate if rounding left a positive residual. */
                if (!utf8_allowed(&state, next, count - i - 1)) {
                    fputs("invalid UTF-8 sampling state\n", stderr);
                    free(m); return 1;
                }
                (void)utf8_advance(&state, next);
            } else {
                for (int k = 0; k < V; k++) {
                    u -= s.p[k];
                    if (u <= 0) { next = k; break; }
                }
            }
            printf("\\x%02x", (unsigned int)next);
            forward(m, h, (unsigned char)next, &s);
            memcpy(h, s.h, sizeof(h));
        }
        putchar('\n');
        free(m); return 0;
    }
    usage(argv[0]); free(m); return 2;
}
