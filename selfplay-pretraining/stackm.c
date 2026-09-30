// Bounded stack machine: an alternative universal substrate to the Brainf*ck tape (same interface as bf.c).
// Values are uint32 (wrapping); the stack is a ring of `cap` cells (pushing onto a full stack drops the
// bottom); popping an empty stack yields 0. Unmatched brackets are no-ops, so every string is executable.
// '.' emits the low byte of the top of stack WITHOUT popping. Execution stops at the step budget, the end of
// the program, or the T-th emitted byte. With unbounded integers, * / % suffice to encode a two-counter
// machine (Minsky), so the unbounded idealization is Turing complete, like the paper's BF machine.
//
//   0 1   push constant          + - * / % ^ =  binary ops (a op b; x/0 = x%0 = 0; '=' pushes a==b)
//   d s o x r   dup swap over drop rot(a b c -> b c a)
//   . emit   , push random byte   [ ] loop while top != 0 (peek, not pop)
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t *v; int cap, base, n; } Stack;

static inline void push(Stack *s, uint32_t x) {
    if (s->n < s->cap) { s->v[(s->base + s->n) % s->cap] = x; s->n++; }
    else { s->v[s->base] = x; s->base = (s->base + 1) % s->cap; }
}
static inline uint32_t pop(Stack *s) {
    if (!s->n) return 0;
    s->n--;
    return s->v[(s->base + s->n) % s->cap];
}
static inline uint32_t peek(Stack *s, int i) {
    return i < s->n ? s->v[(s->base + s->n - 1 - i) % s->cap] : 0;
}
static inline uint64_t xs64(uint64_t *st) {
    uint64_t x = *st;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *st = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static void run_one(const uint8_t *src, int n, uint8_t *out, int T, int cap, int64_t max_steps, uint64_t seed,
                    int32_t *out_len, int32_t *max_depth, int32_t *match, int32_t *bstack, uint32_t *mem) {
    int sp = 0;
    for (int i = 0; i < n; i++) match[i] = -1;
    for (int i = 0; i < n; i++) {
        if (src[i] == '[') bstack[sp++] = i;
        else if (src[i] == ']' && sp > 0) { int j = bstack[--sp]; match[i] = j; match[j] = i; }
    }
    Stack s = {mem, cap, 0, 0};
    memset(out, 0, T);
    uint64_t rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    int pc = 0, emitted = 0, depth = 0, maxd = 0;
    int64_t steps = 0;
    uint32_t a, b, c;
    while (pc < n && steps < max_steps && emitted < T) {
        steps++;
        switch (src[pc]) {
            case '0': push(&s, 0); break;
            case '1': push(&s, 1); break;
            case '+': b = pop(&s); a = pop(&s); push(&s, a + b); break;
            case '-': b = pop(&s); a = pop(&s); push(&s, a - b); break;
            case '*': b = pop(&s); a = pop(&s); push(&s, a * b); break;
            case '/': b = pop(&s); a = pop(&s); push(&s, b ? a / b : 0); break;
            case '%': b = pop(&s); a = pop(&s); push(&s, b ? a % b : 0); break;
            case '^': b = pop(&s); a = pop(&s); push(&s, a ^ b); break;
            case '=': b = pop(&s); a = pop(&s); push(&s, a == b); break;
            case 'd': push(&s, peek(&s, 0)); break;
            case 's': b = pop(&s); a = pop(&s); push(&s, b); push(&s, a); break;
            case 'o': push(&s, peek(&s, 1)); break;
            case 'x': pop(&s); break;
            case 'r': c = pop(&s); b = pop(&s); a = pop(&s); push(&s, b); push(&s, c); push(&s, a); break;
            case '.': out[emitted++] = (uint8_t)(peek(&s, 0) & 255); break;
            case ',': push(&s, (uint32_t)(xs64(&rng) >> 56)); break;
            case '[':
                if (match[pc] >= 0) {
                    if (peek(&s, 0)) { depth++; if (depth > maxd) maxd = depth; }
                    else { pc = match[pc] + 1; continue; }
                }
                break;
            case ']':
                if (match[pc] >= 0) {
                    if (peek(&s, 0)) { pc = match[pc] + 1; continue; }
                    depth--;
                }
                break;
            default: break;
        }
        pc++;
    }
    *out_len = emitted;
    *max_depth = maxd;
}

void sm_run_batch(const uint8_t *progs, const int32_t *lens, int n_progs, int stride,
                  uint8_t *out, int32_t *out_len, int32_t *max_depth,
                  int T, int cap, int64_t max_steps, const uint64_t *seeds) {
    int32_t *match = malloc(sizeof(int32_t) * stride);
    int32_t *bstack = malloc(sizeof(int32_t) * stride);
    uint32_t *mem = malloc(sizeof(uint32_t) * cap);
    for (int p = 0; p < n_progs; p++)
        run_one(progs + (size_t)p * stride, lens[p], out + (size_t)p * T, T, cap, max_steps, seeds[p],
                out_len + p, max_depth + p, match, bstack, mem);
    free(match); free(bstack); free(mem);
}
