#ifndef WS_TSEARCH_H
#define WS_TSEARCH_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

/* Stats we maintain incrementally */
typedef struct {
    uint64_t singles;   /* #keys seen exactly once */
    uint64_t distinct;  /* #distinct keys */
    uint64_t total;     /* total events recorded */
} ws_stats_t;

/* Opaque context */
typedef struct ws_ctx ws_ctx_t;

/* Lifecycle */
ws_ctx_t *ws_create(void);
void      ws_destroy(ws_ctx_t *ctx);

/* Reset (drop all nodes + zero stats) */
void      ws_reset(ws_ctx_t *ctx);

/* Record one access for an already-canonicalized key.
   NOTE: You must ensure 'key' is already aligned/normalized as you intend
   (e.g., 64B-aligned address, or a line index). We do NOT modify it. */
void      ws_record(ws_ctx_t *ctx, uintptr_t key);

/* Snapshot stats in O(1). No tree walk. */
void      ws_get_stats(ws_ctx_t *ctx, ws_stats_t *out_stats);

#ifdef __cplusplus
}
#endif

#endif /* WS_TSEARCH_H */

