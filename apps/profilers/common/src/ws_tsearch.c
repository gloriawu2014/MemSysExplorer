#define _GNU_SOURCE  /* for tdestroy() */
#include "ws_tsearch.h"

#include <stdlib.h>
#include <search.h>  /* tsearch/tfind/twalk/tdestroy */
#include <string.h>

struct ws_node {
    uintptr_t key;  /* caller-provided canonical key (already aligned/shifted) */
    uint64_t  count;
};

struct ws_ctx {
    void     *root;   /* tsearch root */
    ws_stats_t stats; /* incremental counters */
};

/* --- helpers --- */

static int ws_cmp(const void *pa, const void *pb) {
    const struct ws_node *a = (const struct ws_node *)pa;
    const struct ws_node *b = (const struct ws_node *)pb;
    if (a->key < b->key) return -1;
    if (a->key > b->key) return 1;
    return 0;
}

static void ws_free_node(void *p) { free(p); }

/* --- API --- */

ws_ctx_t *ws_create(void) {
    return (ws_ctx_t *)calloc(1, sizeof(ws_ctx_t));
}

void ws_destroy(ws_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->root) tdestroy(ctx->root, ws_free_node);
    free(ctx);
}

void ws_reset(ws_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->root) {
        tdestroy(ctx->root, ws_free_node);
        ctx->root = NULL;
    }
    ctx->stats = (ws_stats_t){0,0,0};
}

void ws_record(ws_ctx_t *ctx, uintptr_t key) {
    if (!ctx) return;

    ctx->stats.total += 1;

    struct ws_node *n = (struct ws_node *)malloc(sizeof(*n));
    if (!n) return;  /* best-effort */
    n->key = key;    /* already aligned/normalized by the caller */
    n->count = 1;

    void **slot = tsearch(n, &ctx->root, ws_cmp);
    if (!slot) {     /* out of memory inside tsearch */
        free(n);
        return;
    }

    struct ws_node *stored = *(struct ws_node **)slot;
    if (stored == n) {
        /* brand-new key */
        ctx->stats.distinct += 1;
        ctx->stats.singles  += 1;  /* count==1 */
        return;
    }

    /* existing key */
    if (stored->count == 1 && ctx->stats.singles > 0)
        ctx->stats.singles -= 1;   /* 1 -> 2 transition removes a single */
    stored->count += 1;
    free(n);  /* discard candidate */
}

void ws_get_stats(ws_ctx_t *ctx, ws_stats_t *out_stats) {
    if (!ctx || !out_stats) return;
    *out_stats = ctx->stats;
}

