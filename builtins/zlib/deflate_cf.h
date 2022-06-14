/* deflate.h -- internal compression state
 * Copyright (C) 1995-2012 Jean-loup Gailly
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* WARNING: this file should *not* be used by applications. It is
   part of the implementation of the compression library and is
   subject to change. Applications should only use zlib.h.
 */

/* @(#) $Id$ */

#ifndef DEFLATE_H
#define DEFLATE_H

#include "zutil.h"

/* ===========================================================================
 * Internal compression state.
 */

#define LENGTH_CODES 29
/* number of length codes, not counting the special END_BLOCK code */

#define LITERALS  256
/* number of literal bytes 0..255 */

#define L_CODES (LITERALS+1+LENGTH_CODES)
/* number of Literal or Length codes, including the END_BLOCK code */

#define D_CODES   30
/* number of distance codes */

#define BL_CODES  19
/* number of codes used to transfer the bit lengths */

#define HEAP_SIZE (2*L_CODES+1)
/* maximum heap size */

#define MAX_BITS 15
/* All codes must not exceed MAX_BITS bits */

#define Buf_size 64
/* size of bit buffer in bi_buf */

#define INIT_STATE    42
#define EXTRA_STATE   69
#define NAME_STATE    73
#define COMMENT_STATE 91
#define HCRC_STATE   103
#define BUSY_STATE   113
#define FINISH_STATE 666
/* Stream status */


/* Data structure describing a single value and its code string. */
typedef struct ct_data_s {
    union {
        uint16_t  freq;       /* frequency count */
        uint16_t  code;       /* bit string */
    } fc;
    union {
        uint16_t  dad;        /* father node in Huffman tree */
        uint16_t  len;        /* length of bit string */
    } dl;
} ct_data;

#define Freq fc.freq
#define Code fc.code
#define Dad  dl.dad
#define Len  dl.len

typedef struct static_tree_desc_s  static_tree_desc;

typedef struct tree_desc_s {
    ct_data *dyn_tree;           /* the dynamic tree */
    int     max_code;            /* largest code with non zero frequency */
    static_tree_desc *stat_desc; /* the corresponding static tree */
} tree_desc;

typedef uint16_t Pos;
typedef uint32_t IPos;

/* A Pos is an index in the character window. We use short instead of int to
 * save space in the various tables. IPos is used only for parameter passing.
 */

typedef struct internal_state {
    z_streamp  strm;              /* pointer back to this zlib stream */
    int        status;            /* as the name implies */
    uint8_t    *pending_buf;      /* output still pending */
    uint64_t   pending_buf_size;  /* size of pending_buf */
    uint8_t    *pending_out;      /* next pending byte to output to the stream */
    uint32_t   pending;           /* nb of bytes in the pending buffer */
    int        wrap;              /* bit 0 true for zlib, bit 1 true for gzip */
    gz_headerp gzhead;            /* gzip header information to write */
    uint32_t   gzindex;           /* where in extra, name, or comment */
    uint8_t    method;            /* can only be DEFLATED */
    int        last_flush;        /* value of flush param for previous deflate call */

    /* used by deflate.c: */
    uint32_t  w_size;        /* LZ77 window size (32K by default) */
    uint32_t  w_bits;        /* log2(w_size)  (8..16) */
    uint32_t  w_mask;        /* w_size - 1 */

    uint8_t *window;
    /* Sliding window. Input bytes are read into the second half of the window,
     * and move to the first half later to keep a dictionary of at least wSize
     * bytes. With this organization, matches are limited to a distance of
     * wSize-MAX_MATCH bytes, but this ensures that IO is always
     * performed with a length multiple of the block size. Also, it limits
     * the window size to 64K, which is quite useful on MSDOS.
     * To do: use the user input buffer as sliding window.
     */

    uint32_t window_size;
    /* Actual size of window: 2*wSize, except when the user input buffer
     * is directly used as sliding window.
     */

    Pos *prev;
    /* Link to older string with same hash index. To limit the size of this
     * array to 64K, this link is maintained only for the last 32K strings.
     * An index in this array is thus a window index modulo 32K.
     */

    Pos *head; /* Heads of the hash chains or NIL. */

    uint32_t  ins_h;          /* hash index of string to be inserted */
    uint32_t  hash_size;      /* number of elements in hash table */
    uint32_t  hash_bits;      /* log2(hash_size) */
    uint32_t  hash_mask;      /* hash_size-1 */

    uint32_t  hash_shift;
    /* Number of bits by which ins_h must be shifted at each input
     * step. It must be such that after MIN_MATCH steps, the oldest
     * byte no longer takes part in the hash key, that is:
     *   hash_shift * MIN_MATCH >= hash_bits
     */

    long block_start;
    /* Window position at the beginning of the current output block. Gets
     * negative when the window is moved backwards.
     */

    uint32_t match_length;           /* length of best match */
    IPos prev_match;                 /* previous match */
    int match_available;             /* set if previous match exists */
    uint32_t strstart;               /* start of string to insert */
    uint32_t match_start;            /* start of matching string */
    uint32_t lookahead;              /* number of valid bytes ahead in window */

    uint32_t prev_length;
    /* Length of the best match at previous step. Matches not greater than this
     * are discarded. This is used in the lazy match evaluation.
     */

    uint32_t max_chain_length;
    /* To speed up deflation, hash chains are never searched beyond this
     * length.  A higher limit improves compression ratio but degrades the
     * speed.
     */

    uint32_t max_lazy_match;
    /* Attempt to find a better match only when the current match is strictly
     * smaller than this value. This mechanism is used only for compression
     * levels >= 4.
     */
#   define max_insert_length  max_lazy_match
    /* Insert new strings in the hash table only if the match length is not
     * greater than this length. This saves time but degrades compression.
     * max_insert_length is used only for compression levels <= 3.
     */

    int level;    /* compression level (1..9) */
    int strategy; /* favor or force Huffman coding*/

    uint32_t good_match;
    /* Use a faster search when the previous match is longer than this */

    int nice_match; /* Stop searching when current match exceeds this */

    /* used by trees.c: */
    /* Didn't use ct_data typedef below to suppress compiler warning */
    struct ct_data_s dyn_ltree[HEAP_SIZE];   /* literal and length tree */
    struct ct_data_s dyn_dtree[2*D_CODES+1]; /* distance tree */
    struct ct_data_s bl_tree[2*BL_CODES+1];  /* Huffman tree for bit lengths */

    struct tree_desc_s l_desc;               /* desc. for literal tree */
    struct tree_desc_s d_desc;               /* desc. for distance tree */
    struct tree_desc_s bl_desc;              /* desc. for bit length tree */

    uint16_t bl_count[MAX_BITS+1];
    /* number of codes at each bit length for an optimal tree */

    int heap[2*L_CODES+1];      /* heap used to build the Huffman trees */
    int heap_len;               /* number of elements in the heap */
    int heap_max;               /* element of largest frequency */
    /* The sons of heap[n] are heap[2*n] and heap[2*n+1]. heap[0] is not used.
     * The same heap array is used to build all trees.
     */

    uint8_t depth[2*L_CODES+1];
    /* Depth of each subtree used as tie breaker for trees of equal frequency
     */

    uint8_t *l_buf;          /* buffer for literals or lengths */

    uint32_t  lit_bufsize;
    /* Size of match buffer for literals/lengths.  There are 4 reasons for
     * limiting lit_bufsize to 64K:
     *   - frequencies can be kept in 16 bit counters
     *   - if compression is not successful for the first block, all input
     *     data is still in the window so we can still emit a stored block even
     *     when input comes from standard input.  (This can also be done for
     *     all blocks if lit_bufsize is not greater than 32K.)
     *   - if compression is not successful for a file smaller than 64K, we can
     *     even emit a stored file instead of a stored block (saving 5 bytes).
     *     This is applicable only for zip (not gzip or zlib).
     *   - creating new Huffman trees less frequently may not provide fast
     *     adaptation to changes in the input data statistics. (Take for
     *     example a binary file with poorly compressible code followed by
     *     a highly compressible string table.) Smaller buffer sizes give
     *     fast adaptation but have of course the overhead of transmitting
     *     trees more frequently.
     *   - I can't count above 4
     */

    uint32_t last_lit;      /* running index in l_buf */

    uint16_t *d_buf;
    /* Buffer for distances. To simplify the code, d_buf and l_buf have
     * the same number of elements. To use different lengths, an extra flag
     * array would be necessary.
     */

    uint64_t opt_len;        /* bit length of current block with optimal trees */
    uint64_t static_len;     /* bit length of current block with static trees */
    uint32_t matches;       /* number of string matches in current block */
    uint32_t insert;        /* bytes at end of window left to insert */

#ifdef DEBUG
    uint64_t compressed_len; /* total bit length of compressed file mod 2^32 */
    uint64_t bits_sent;      /* bit length of compressed data sent mod 2^32 */
#endif

    uint64_t bi_buf;
    /* Output buffer. bits are inserted starting at the bottom (least
     * significant bits).
     */
    int bi_valid;
    /* Number of valid bits in bi_buf.  All bits above the last valid bit
     * are always zero.
     */

    uint64_t high_water;
    /* High water mark offset in window for initialized bytes -- bytes above
     * this are set to zero in order to avoid memory check warnings when
     * longest match routines access bytes past the input.  This is then
     * updated to the new high water mark.
     */

} deflate_state;

/* Output a byte on the stream.
 * IN assertion: there is enough room in pending_buf.
 */
#define put_byte(s, c) {s->pending_buf[s->pending++] = (c);}

/* ===========================================================================
 * Output a short LSB first on the stream.
 * IN assertion: there is enough room in pendingBuf.
 */

#define put_short(s, w) { \
    s->pending += 2; \
    *(ush*)(&s->pending_buf[s->pending - 2]) = (w) ; \
}

#define MIN_LOOKAHEAD (MAX_MATCH+MIN_MATCH+1)
/* Minimum amount of lookahead, except at the end of the input file.
 * See deflate.c for comments about the MIN_MATCH+1.
 */

#define MAX_DIST(s)  ((s)->w_size-MIN_LOOKAHEAD)
/* In order to simplify the code, particularly on 16 bit machines, match
 * distances are limited to MAX_DIST instead of WSIZE.
 */

#define WIN_INIT MAX_MATCH
/* Number of bytes after end of data in window to initialize in order to avoid
   memory checker errors from longest match routines */

        /* in trees.c */
void ZLIB_INTERNAL _tr_init(deflate_state *s);
int  ZLIB_INTERNAL _tr_tally(deflate_state *s, uint32_t dist, unsigned lc);
void ZLIB_INTERNAL _tr_flush_block(deflate_state *s, uint8_t *buf,
                        uint64_t stored_len, int last);
void ZLIB_INTERNAL _tr_flush_bits(deflate_state *s);
void ZLIB_INTERNAL _tr_align(deflate_state *s);
void ZLIB_INTERNAL _tr_stored_block(deflate_state *s, uint8_t *buf,
                        uint64_t stored_len, int last);

#define d_code(dist) \
   ((dist) < 256 ? _dist_code[dist] : _dist_code[256+((dist)>>7)])
/* Mapping from a distance to a distance code. dist is the distance - 1 and
 * must not have side effects. _dist_code[256] and _dist_code[257] are never
 * used.
 */

extern const uint8_t ZLIB_INTERNAL _length_code[];
extern const uint8_t ZLIB_INTERNAL _dist_code[];

#ifdef _MSC_VER

/* MSC doesn't have __builtin_expect.  Just ignore likely/unlikely and
   hope the compiler optimizes for the best.
*/
#define likely(x)       (x)
#define unlikely(x)     (x)

int __inline __builtin_ctzl(unsigned long mask)
{
    unsigned long index ;

    return _BitScanForward(&index, mask) == 0 ? 32 : ((int)index) ;
}
#else
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)
#endif

#endif /* DEFLATE_H */
