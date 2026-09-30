// Bounded Brainf*ck machine U(x, w) from "Self-Play Pretraining with Zero Data", appendix E.
// - circular tape of `tape_len` byte cells (all arithmetic mod 256)
// - unmatched brackets are no-ops
// - ',' reads i.i.d. uniform bytes from a random tape (xorshift64* seeded per program)
// - execution stops at the step budget, the end of the program, or the T-th emitted byte
// Also reports the maximum dynamic loop depth (used for MAP-Elites niches, appendix G).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { OP_LEFT, OP_RIGHT, OP_INC, OP_DEC, OP_OPEN, OP_CLOSE, OP_OUT, OP_IN, OP_NOP };

static inline uint64_t xs64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static void run_one(const uint8_t *src, int n, uint8_t *out, int T, int tape_len, int64_t max_steps,
                    uint64_t seed, int32_t *out_len, int32_t *max_depth,
                    uint8_t *ops, int32_t *match, int32_t *stack, uint8_t *tape) {
    // compile
    int sp = 0;
    for (int i = 0; i < n; i++) {
        switch (src[i]) {
            case '<': ops[i] = OP_LEFT; break;
            case '>': ops[i] = OP_RIGHT; break;
            case '+': ops[i] = OP_INC; break;
            case '-': ops[i] = OP_DEC; break;
            case '.': ops[i] = OP_OUT; break;
            case ',': ops[i] = OP_IN; break;
            case '[': ops[i] = OP_OPEN; stack[sp++] = i; break;
            case ']':
                if (sp > 0) { int j = stack[--sp]; ops[i] = OP_CLOSE; match[i] = j; match[j] = i; }
                else ops[i] = OP_NOP;
                break;
            default: ops[i] = OP_NOP;
        }
    }
    while (sp > 0) ops[stack[--sp]] = OP_NOP;  // unmatched '['

    memset(tape, 0, tape_len);
    memset(out, 0, T);
    uint64_t rng = seed ? seed : 0x9E3779B97F4A7C15ULL;
    int pc = 0, head = 0, emitted = 0, depth = 0, maxd = 0;
    int64_t steps = 0;
    while (pc < n && steps < max_steps && emitted < T) {
        steps++;
        switch (ops[pc]) {
            case OP_LEFT: head = head ? head - 1 : tape_len - 1; pc++; break;
            case OP_RIGHT: head = (head + 1 == tape_len) ? 0 : head + 1; pc++; break;
            case OP_INC: tape[head]++; pc++; break;
            case OP_DEC: tape[head]--; pc++; break;
            case OP_OUT: out[emitted++] = tape[head]; pc++; break;
            case OP_IN: tape[head] = (uint8_t)(xs64(&rng) >> 56); pc++; break;
            case OP_OPEN:
                if (tape[head]) { depth++; if (depth > maxd) maxd = depth; pc++; }
                else pc = match[pc] + 1;
                break;
            case OP_CLOSE:
                if (tape[head]) pc = match[pc] + 1;
                else { depth--; pc++; }
                break;
            default: pc++;
        }
    }
    *out_len = emitted;
    *max_depth = maxd;
}

// progs: [n_progs, stride] bytes (already macro-expanded), lens: [n_progs]
// out: [n_progs, T], out_len / max_depth: [n_progs]
void bf_run_batch(const uint8_t *progs, const int32_t *lens, int n_progs, int stride,
                  uint8_t *out, int32_t *out_len, int32_t *max_depth,
                  int T, int tape_len, int64_t max_steps, const uint64_t *seeds) {
    uint8_t *ops = malloc(stride);
    int32_t *match = malloc(sizeof(int32_t) * stride);
    int32_t *stack = malloc(sizeof(int32_t) * stride);
    uint8_t *tape = malloc(tape_len);
    for (int p = 0; p < n_progs; p++) {
        run_one(progs + (size_t)p * stride, lens[p], out + (size_t)p * T, T, tape_len, max_steps,
                seeds[p], out_len + p, max_depth + p, ops, match, stack, tape);
    }
    free(ops); free(match); free(stack); free(tape);
}
