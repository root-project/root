/* @(#)root/zip:$Id$ */
/* Author: */
/*

 Copyright (C) 1990-1993 Mark Adler, Richard B. Wales, Jean-loup Gailly,
 Kai Uwe Rommel and Igor Mandrichenko.
 For conditions of distribution and use, see copyright notice in zlib.h

 */
#include "Bits.h"

/*
 *  trees.c by Jean-loup Gailly
 *
 *  This is a new version of im_ctree.c originally written by Richard B. Wales
 *  for the defunct implosion method.
 *
 *  PURPOSE
 *
 *      Encode various sets of source values using variable-length
 *      binary code trees.
 *
 *  DISCUSSION
 *
 *      The PKZIP "deflation" process uses several Huffman trees. The more
 *      common source values are represented by shorter bit sequences.
 *
 *      Each code tree is stored in the ZIP file in a compressed form
 *      which is itself a Huffman encoding of the lengths of
 *      all the code strings (in ascending order by source values).
 *      The actual code strings are reconstructed from the lengths in
 *      the UNZIP process, as described in the "application note"
 *      (APPNOTE.TXT) distributed as part of PKWARE's PKZIP program.
 *
 *  REFERENCES
 *
 *      Lynch, Thomas J.
 *          Data Compression:  Techniques and Applications, pp. 53-55.
 *          Lifetime Learning Publications, 1985.  ISBN 0-534-03418-7.
 *
 *      Storer, James A.
 *          Data Compression:  Methods and Theory, pp. 49-50.
 *          Computer Science Press, 1988.  ISBN 0-7167-8156-5.
 *
 *      Sedgewick, R.
 *          Algorithms, p290.
 *          Addison-Wesley, 1983. ISBN 0-201-06672-6.
 *
 *  INTERFACE
 *
 *      void ct_init (ush *attr, int *method)
 *          Allocate the match buffer, initialize the various tables and save
 *          the location of the internal file attribute (ascii/binary) and
 *          method (DEFLATE/STORE)
 *
 *      void ct_tally (int dist, int lc);
 *          Save the match info and tally the frequency counts.
 *
 *      long flush_block (char *buf, ulg stored_len, int eof)
 *          Determine the best encoding for the current block: dynamic trees,
 *          static trees or store, and output the encoded block to the zip
 *          file. Returns the total compressed length for the file so far.
 *
 */

#include <ctype.h>
/* #include "zip.h" */
/* #include "ZIP.h" */

/* ===========================================================================
 * Constants
 */

#define MAX_BITS 15
/* All codes must not exceed MAX_BITS bits */

#define MAX_BL_BITS 7
/* Bit length codes must not exceed MAX_BL_BITS bits */

#define LENGTH_CODES 29
/* number of length codes, not counting the special END_BLOCK code */

#define LITERALS  256
/* number of literal bytes 0..255 */

#define END_BLOCK 256
/* end of block literal code */

#define L_CODES (LITERALS+1+LENGTH_CODES)
/* number of Literal or Length codes, including the END_BLOCK code */

#define D_CODES   30
/* number of distance codes */

#define BL_CODES  19
/* number of codes used to transfer the bit lengths */


local int near extra_lbits[LENGTH_CODES] /* extra bits for each length code */
= {0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0};

local int near extra_dbits[D_CODES] /* extra bits for each distance code */
= {0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};

local int near extra_blbits[BL_CODES]/* extra bits for each bit length code */
= {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,7};

#define STORED_BLOCK 0
#define STATIC_TREES 1
#define DYN_TREES    2
/* The three kinds of block type */

#ifndef LIT_BUFSIZE
#  ifdef SMALL_MEM
#    define LIT_BUFSIZE  0x2000
#  else
#  ifdef MEDIUM_MEM
#    define LIT_BUFSIZE  0x4000
#  else
#    define LIT_BUFSIZE  0x8000
#  endif
#  endif
#endif
#define DIST_BUFSIZE  LIT_BUFSIZE
/* Sizes of match buffers for literals/lengths and distances.  There are
 * 4 reasons for limiting LIT_BUFSIZE to 64K:
 *   - frequencies can be kept in 16 bit counters
 *   - if compression is not successful for the first block, all input data is
 *     still in the window so we can still emit a stored block even when input
 *     comes from standard input.  (This can also be done for all blocks if
 *     LIT_BUFSIZE is not greater than 32K.)
 *   - if compression is not successful for a file smaller than 64K, we can
 *     even emit a stored file instead of a stored block (saving 5 bytes).
 *   - creating new Huffman trees less frequently may not provide fast
 *     adaptation to changes in the input data statistics. (Take for
 *     example a binary file with poorly compressible code followed by
 *     a highly compressible string table.) Smaller buffer sizes give
 *     fast adaptation but have of course the overhead of transmitting trees
 *     more frequently.
 *   - I can't count above 4
 * The current code is general and allows DIST_BUFSIZE < LIT_BUFSIZE (to save
 * memory at the expense of compression). Some optimizations would be possible
 * if we rely on DIST_BUFSIZE == LIT_BUFSIZE.
 */

#define REP_3_6      16
/* repeat previous bit length 3-6 times (2 bits of repeat count) */

#define REPZ_3_10    17
/* repeat a zero length 3-10 times  (3 bits of repeat count) */

#define REPZ_11_138  18
/* repeat a zero length 11-138 times  (7 bits of repeat count) */

/* ===========================================================================
 * Local data
 */

/* Data structure describing a single value and its code string. */
typedef struct ct_data {
   union {
      ush  freq;       /* frequency count */
      ush  code;       /* bit string */
   } fc;
   union {
      ush  dad;        /* father node in Huffman tree */
      ush  len;        /* length of bit string */
   } dl;
} ct_data;

#define Freq fc.freq
#define Code fc.code
#define Dad  dl.dad
#define Len  dl.len

#define HEAP_SIZE (2*L_CODES+1)
/* maximum heap size */

uch near bl_order[BL_CODES]
= {16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15};
/* The lengths of the bit length codes are sent in order of decreasing
 * probability, to avoid transmitting the lengths for unused bit length codes.
 */

typedef struct tree_desc {
   ct_data near *dyn_tree;      /* the dynamic tree */
   ct_data near *static_tree;   /* corresponding static tree or NULL */
   int     near *extra_bits;    /* extra bits for each code or NULL */
   int     extra_base;          /* base index for extra_bits */
   int     elems;               /* max number of elements in the tree */
   int     max_length;          /* max bit length for the codes */
   int     max_code;            /* largest code with non zero frequency */
} tree_desc;

struct tree_internal_state {
   ct_data near dyn_ltree[HEAP_SIZE];   /* literal and length tree */
   ct_data near dyn_dtree[2*D_CODES+1]; /* distance tree */

   ct_data near static_ltree[L_CODES+2];
   /* The static literal tree. Since the bit lengths are imposed, there is no
    * need for the L_CODES extra codes used during heap construction. However
    * The codes 286 and 287 are needed to build a canonical tree (see ct_init
    * below).
    */

   ct_data near static_dtree[D_CODES];
   /* The static distance tree. (Actually a trivial tree since all codes use
    * 5 bits.)
    */

   ct_data near bl_tree[2*BL_CODES+1];
   /* Huffman tree for the bit lengths */

   tree_desc near l_desc;

   tree_desc near d_desc;

   tree_desc near bl_desc;


   ush near bl_count[MAX_BITS+1];
   /* number of codes at each bit length for an optimal tree */

   int near heap[2*L_CODES+1]; /* heap used to build the Huffman trees */
   int heap_len;               /* number of elements in the heap */
   int heap_max;               /* element of largest frequency */
   /* The sons of heap[n] are heap[2*n] and heap[2*n+1]. heap[0] is not used.
    * The same heap array is used to build all trees.
    */

   uch near depth[2*L_CODES+1];
   /* Depth of each subtree used as tie breaker for trees of equal frequency */

   uch length_code[MAX_MATCH-MIN_MATCH+1];
   /* length code for each normalized match length (0 == MIN_MATCH) */

   uch dist_code[512];
   /* distance codes. The first 256 values correspond to the distances
    * 3 .. 258, the last 256 values correspond to the top 8 bits of
    * the 15 bit distances.
    */

   int near base_length[LENGTH_CODES];
   /* First normalized length for each code (0 = MIN_MATCH) */

   int near base_dist[D_CODES];
   /* First normalized distance for each code (0 = distance of 1) */

#ifndef DYN_ALLOC
   uch far l_buf[LIT_BUFSIZE];  /* buffer for literals/lengths */
   ush far d_buf[DIST_BUFSIZE]; /* buffer for distances */
#else
   uch far *l_buf;
   ush far *d_buf;
#endif

   uch near flag_buf[(LIT_BUFSIZE/8)];
   /* flag_buf is a bit array distinguishing literals from lengths in
    * l_buf, and thus indicating the presence or absence of a distance.
    */

   unsigned last_lit;    /* running index in l_buf */
   unsigned last_dist;   /* running index in d_buf */
   unsigned last_flags;  /* running index in flag_buf */
   uch flags;            /* current flags not yet saved in flag_buf */
   uch flag_bit;         /* current bit used in flags */
   /* bits are filled in flags starting at bit 0 (least significant).
    * Note: these flags are overkill in the current code since we don't
    * take advantage of DIST_BUFSIZE == LIT_BUFSIZE.
    */

   ulg opt_len;        /* bit length of current block with optimal trees */
   ulg static_len;     /* bit length of current block with static trees */

   ulg compressed_len; /* total bit length of compressed file */

   ulg input_len;      /* total byte length of input file */
   /* input_len is for debugging only since we can get it by other means. */

   ush *R__file_type;        /* pointer to UNKNOWN, BINARY or ASCII */
   int *R__file_method;      /* pointer to DEFLATE or STORE */

};

#include "ThreadLocalStorage.h"

/* ===========================================================================
 * Allocate the per thread ZTree internal state object
 */
TTHREAD_TLS_DECLARE(int,tree_state_isInit);
TTHREAD_TLS_DECLARE(tree_internal_state*,tree_state);

tree_internal_state *R__get_thread_tree_state() {

   TTHREAD_TLS_INIT(int,tree_state_isInit,0);
   TTHREAD_TLS_INIT(tree_internal_state *,tree_state,0);
   if (!TTHREAD_TLS_GET(int,tree_state_isInit)) {
      TTHREAD_TLS_SET(int,tree_state_isInit,1);
      TTHREAD_TLS_SET(tree_internal_state*,tree_state,fcalloc(1,sizeof(tree_internal_state)));
   }
   return TTHREAD_TLS_GET(tree_internal_state*,tree_state);
}

#ifdef DEBUG
/* extern ulg R__bits_sent; */ /* bit length of the compressed data */
/* extern ulg R__isize;     */ /* byte length of input file */
#endif

/* extern long R__block_start;       */ /* window offset of current block */
/* extern unsigned near R__strstart; */ /* window offset of current string */

/* ===========================================================================
 * Local (static) routines in this file.
 */

local void R__init_block     OF((tree_internal_state *t_state));
local void R__pqdownheap     OF((tree_internal_state *t_state, ct_data near *tree, int k));
local void R__gen_bitlen     OF((tree_internal_state *t_state, tree_desc near *desc));
local void R__gen_codes      OF((tree_internal_state *t_state, ct_data near *tree, int max_code));
local void R__build_tree     OF((tree_internal_state *t_state, tree_desc near *desc));
local void R__scan_tree      OF((tree_internal_state *t_state, ct_data near *tree, int max_code));
local void R__send_tree      OF((bits_internal_state *state, tree_internal_state *t_state, ct_data near *tree, int max_code));
local int  R__build_bl_tree  OF((tree_internal_state *t_state));
local void R__send_all_trees OF((bits_internal_state *state, tree_internal_state *t_state, int lcodes, int dcodes, int blcodes));
local void R__compress_block OF((bits_internal_state *state, tree_internal_state *t_state, ct_data near *ltree, ct_data near *dtree));
local void R__set_file_type  OF((tree_internal_state *t_state));


#ifndef DEBUG
#  define send_code(c, tree) R__send_bits(state,tree[c].Code, tree[c].Len)
/* Send a code of the given tree. c and tree must not have side effects */

#else /* DEBUG */
#  define send_code(c, tree) \
{ R__error("\ncd %3d ",(c)); \
R__send_bits(state,tree[c].Code, tree[c].Len); }
#endif

#define d_code(dist) \
((dist) < 256 ? t_state->dist_code[dist] : t_state->dist_code[256+((dist)>>7)])
/* Mapping from a distance to a distance code. dist is the distance - 1 and
 * must not have side effects. dist_code[256] and dist_code[257] are never
 * used.
 */

#define MAX(a,b) (a >= b ? a : b)
/* the arguments must not have side effects */

void R__tree_desc_init(tree_desc *tree_description,
                       ct_data near *dyn_tree,      /* the dynamic tree */
                       ct_data near *static_tree,   /* corresponding static tree or NULL */
                       int     near *extra_bits,    /* extra bits for each code or NULL */
                       int     extra_base,          /* base index for extra_bits */
                       int     elems,               /* max number of elements in the tree */
                       int     max_length,          /* max bit length for the codes */
                       int     max_code            /* largest code with non zero frequency */
)
{
   tree_description->dyn_tree = dyn_tree;
   tree_description->static_tree = static_tree;
   tree_description->extra_bits = extra_bits;
   tree_description->extra_base = extra_base;
   tree_description->elems = elems;
   tree_description->max_length = max_length;
   tree_description->max_code = max_code;
}

/* ===========================================================================
 * Allocate the match buffer, initialize the various tables and save the
 * location of the internal file attribute (ascii/binary) and method
 * (DEFLATE/STORE).
 */
int R__ct_init(tree_internal_state *t_state, ush *attr, int *method)
/* ush  *attr;    pointer to internal file attribute */
/* int  *method;  pointer to compression method */
{
   int n;        /* iterates over tree elements */
   int bits;     /* bit counter */
   int length;   /* length value */
   int code;     /* code value */
   int dist;     /* distance index */

   t_state->R__file_type   = attr;
   t_state->R__file_method = method;
   t_state->compressed_len = t_state->input_len = 0L;

   if (t_state->static_dtree[0].Len != 0) return 0; /* ct_init already called */

   R__tree_desc_init(&t_state->l_desc, t_state->dyn_ltree, t_state->static_ltree, extra_lbits, LITERALS+1, L_CODES, MAX_BITS, 0);
   R__tree_desc_init(&t_state->d_desc, t_state->dyn_dtree, t_state->static_dtree, extra_dbits, 0,          D_CODES, MAX_BITS, 0);
   R__tree_desc_init(&t_state->bl_desc,  t_state->bl_tree,   NULL,       extra_blbits, 0,         BL_CODES, MAX_BL_BITS, 0);
#ifdef DYN_ALLOC
   d_buf = (ush far*) fcalloc(DIST_BUFSIZE, sizeof(ush));
   l_buf = (uch far*) fcalloc(LIT_BUFSIZE/2, 2);
   /* Avoid using the value 64K on 16 bit machines */
   if (l_buf == NULL || d_buf == NULL) {
      R__error("R__ct_init: out of memory");
      return 1;
   }
#endif

   /* Initialize the mapping length (0..255) -> length code (0..28) */
   length = 0;
   for (code = 0; code < LENGTH_CODES-1; code++) {
      t_state->base_length[code] = length;
      for (n = 0; n < (1<<extra_lbits[code]); n++) {
         t_state->length_code[length++] = (uch)code;
      }
   }
   Assert (length == 256, "R__ct_init: length != 256");
   /* Note that the length 255 (match length 258) can be represented
    * in two different ways: code 284 + 5 bits or code 285, so we
    * overwrite length_code[255] to use the best encoding:
    */
   t_state->length_code[length-1] = (uch)code;

   /* Initialize the mapping dist (0..32K) -> dist code (0..29) */
   dist = 0;
   for (code = 0 ; code < 16; code++) {
      t_state->base_dist[code] = dist;
      for (n = 0; n < (1<<extra_dbits[code]); n++) {
         t_state->dist_code[dist++] = (uch)code;
      }
   }
   Assert (dist == 256, "R__ct_init: dist != 256");
   dist >>= 7; /* from now on, all distances are divided by 128 */
   for ( ; code < D_CODES; code++) {
      t_state->base_dist[code] = dist << 7;
      for (n = 0; n < (1<<(extra_dbits[code]-7)); n++) {
         t_state->dist_code[256 + dist++] = (uch)code;
      }
   }
   Assert (dist == 256, "R__ct_init: 256+dist != 512");

   /* Construct the codes of the static literal tree */
   for (bits = 0; bits <= MAX_BITS; bits++) t_state->bl_count[bits] = 0;
   n = 0;
   while (n <= 143) t_state->static_ltree[n++].Len = 8, t_state->bl_count[8]++;
   while (n <= 255) t_state->static_ltree[n++].Len = 9, t_state->bl_count[9]++;
   while (n <= 279) t_state->static_ltree[n++].Len = 7, t_state->bl_count[7]++;
   while (n <= 287) t_state->static_ltree[n++].Len = 8, t_state->bl_count[8]++;
   /* Codes 286 and 287 do not exist, but we must include them in the
    * tree construction to get a canonical Huffman tree (longest code
    * all ones)
    */
   R__gen_codes(t_state,(ct_data near *)t_state->static_ltree, L_CODES+1);

   /* The static distance tree is trivial: */
   for (n = 0; n < D_CODES; n++) {
      t_state->static_dtree[n].Len = 5;
      t_state->static_dtree[n].Code = R__bi_reverse(n, 5);
   }

   /* Initialize the first block of the first file: */
   R__init_block(t_state);
   return 0;
}

/* ===========================================================================
 * Initialize a new block.
 */
local void R__init_block(tree_internal_state *t_state)
{
   int n; /* iterates over tree elements */

   /* Initialize the trees. */
   for (n = 0; n < L_CODES;  n++) t_state->dyn_ltree[n].Freq = 0;
   for (n = 0; n < D_CODES;  n++) t_state->dyn_dtree[n].Freq = 0;
   for (n = 0; n < BL_CODES; n++) t_state->bl_tree[n].Freq = 0;

   t_state->dyn_ltree[END_BLOCK].Freq = 1;
   t_state->opt_len = t_state->static_len = 0L;
   t_state->last_lit = t_state->last_dist = t_state->last_flags = 0;
   t_state->flags = 0; t_state->flag_bit = 1;
}

#define SMALLEST 1
/* Index within the heap array of least frequent node in the Huffman tree */


/* ===========================================================================
 * Remove the smallest element from the heap and recreate the heap with
 * one less element. Updates heap and heap_len.
 */
#define pqremove(tree, top) \
{\
top = t_state->heap[SMALLEST]; \
t_state->heap[SMALLEST] = t_state->heap[t_state->heap_len--]; \
R__pqdownheap(t_state, tree, SMALLEST); \
}

/* ===========================================================================
 * Compares to subtrees, using the tree depth as tie breaker when
 * the subtrees have equal frequency. This minimizes the worst case length.
 */
#define smaller(tree, n, m) \
(tree[n].Freq < tree[m].Freq || \
(tree[n].Freq == tree[m].Freq && t_state->depth[n] <= t_state->depth[m]))

/* ===========================================================================
 * Restore the heap property by moving down the tree starting at node k,
 * exchanging a node with the smallest of its two sons if necessary, stopping
 * when the heap property is re-established (each father smaller than its
 * two sons).
 */
local void R__pqdownheap(tree_internal_state *t_state, ct_data near *tree, int k)
/* ct_data near *tree;   the tree to restore */
/* int k;                node to move down */
{
   int v = t_state->heap[k];
   int j = k << 1;  /* left son of k */
   int htemp;       /* required because of bug in SASC compiler */

   while (j <= t_state->heap_len) {
      /* Set j to the smallest of the two sons: */
      if (j < t_state->heap_len && smaller(tree, t_state->heap[j+1], t_state->heap[j])) j++;

      /* Exit if v is smaller than both sons */
      htemp = t_state->heap[j];
      if (smaller(tree, v, htemp)) break;

      /* Exchange v with the smallest son */
      t_state->heap[k] = htemp;
      k = j;

      /* And continue down the tree, setting j to the left son of k */
      j <<= 1;
   }
   t_state->heap[k] = v;
}

/* ===========================================================================
 * Compute the optimal bit lengths for a tree and update the total bit length
 * for the current block.
 * IN assertion: the fields freq and dad are set, heap[heap_max] and
 *    above are the tree nodes sorted by increasing frequency.
 * OUT assertions: the field len is set to the optimal bit length, the
 *     array bl_count contains the frequencies for each bit length.
 *     The length opt_len is updated; static_len is also updated if stree is
 *     not null.
 */
local void R__gen_bitlen(tree_internal_state *t_state, tree_desc near *desc)
/* tree_desc near *desc;  the tree descriptor */
{
   ct_data near *tree  = desc->dyn_tree;
   int near *extra     = desc->extra_bits;
   int base            = desc->extra_base;
   int max_code        = desc->max_code;
   int max_length      = desc->max_length;
   ct_data near *stree = desc->static_tree;
   int h;              /* heap index */
   int n, m;           /* iterate over the tree elements */
   int bits;           /* bit length */
   int xbits;          /* extra bits */
   ush f;              /* frequency */
   int overflow = 0;   /* number of elements with bit length too large */

   for (bits = 0; bits <= MAX_BITS; bits++) t_state->bl_count[bits] = 0;

   /* In a first pass, compute the optimal bit lengths (which may
    * overflow in the case of the bit length tree).
    */
   tree[t_state->heap[t_state->heap_max]].Len = 0; /* root of the heap */

   for (h = t_state->heap_max+1; h < HEAP_SIZE; h++) {
      n = t_state->heap[h];
      bits = tree[tree[n].Dad].Len + 1;
      if (bits > max_length) bits = max_length, overflow++;
      tree[n].Len = bits;
      /* We overwrite tree[n].Dad which is no longer needed */

      if (n > max_code) continue; /* not a leaf node */

      t_state->bl_count[bits]++;
      xbits = 0;
      if (n >= base) xbits = extra[n-base];
      f = tree[n].Freq;
      t_state->opt_len += (ulg)f * (bits + xbits);
      if (stree) t_state->static_len += (ulg)f * (stree[n].Len + xbits);
   }
   if (overflow == 0) return;

   Trace((stderr,"\nbit length overflow\n"));
   /* This happens for example on obj2 and pic of the Calgary corpus */

   /* Find the first bit length which could increase: */
   do {
      bits = max_length-1;
      while (t_state->bl_count[bits] == 0) bits--;
      t_state->bl_count[bits]--;      /* move one leaf down the tree */
      t_state->bl_count[bits+1] += 2; /* move one overflow item as its brother */
      t_state->bl_count[max_length]--;
      /* The brother of the overflow item also moves one step up,
       * but this does not affect bl_count[max_length]
       */
      overflow -= 2;
   } while (overflow > 0);

   /* Now recompute all bit lengths, scanning in increasing frequency.
    * h is still equal to HEAP_SIZE. (It is simpler to reconstruct all
    * lengths instead of fixing only the wrong ones. This idea is taken
    * from 'ar' written by Haruhiko Okumura.)
    */
   for (bits = max_length; bits != 0; bits--) {
      n = t_state->bl_count[bits];
      while (n != 0) {
         m = t_state->heap[--h];
         if (m > max_code) continue;
         if (tree[m].Len != (unsigned) bits) {
            Trace((stderr,"code %d bits %d->%d\n", m, tree[m].Len, bits));
            t_state->opt_len += ((long)bits-(long)tree[m].Len)*(long)tree[m].Freq;
            tree[m].Len = bits;
         }
         n--;
      }
   }
}

/* ===========================================================================
 * Generate the codes for a given tree and bit counts (which need not be
 * optimal).
 * IN assertion: the array bl_count contains the bit length statistics for
 * the given tree and the field len is set for all tree elements.
 * OUT assertion: the field code is set for all tree elements of non
 *     zero code length.
 */
local void R__gen_codes (tree_internal_state *t_state, ct_data near *tree, int max_code)
/* ct_data near *tree;         the tree to decorate */
/* int max_code;               largest code with non zero frequency */
{
   ush next_code[MAX_BITS+1]; /* next code value for each bit length */
   ush code = 0;              /* running code value */
   int bits;                  /* bit index */
   int n;                     /* code index */

   /* The distribution counts are first used to generate the code values
    * without bit reversal.
    */
   for (bits = 1; bits <= MAX_BITS; bits++) {
      next_code[bits] = code = (code + t_state->bl_count[bits-1]) << 1;
   }
   /* Check that the bit counts in bl_count are consistent. The last code
    * must be all ones.
    */
   Assert (code + bl_count[MAX_BITS]-1 == (1<<MAX_BITS)-1,
           "inconsistent bit counts");
   Tracev((stderr,"\nR__gen_codes: max_code %d ", max_code));

   for (n = 0;  n <= max_code; n++) {
      int len = tree[n].Len;
      if (len == 0) continue;
      /* Now reverse the bits */
      tree[n].Code = R__bi_reverse(next_code[len]++, len);

      Tracec(tree != static_ltree, (stderr,"\nn %3d %c l %2d c %4x (%x) ",
                                    n, (isgraph(n) ? n : ' '), len, tree[n].Code, next_code[len]-1));
   }
}

/* ===========================================================================
 * Construct one Huffman tree and assigns the code bit strings and lengths.
 * Update the total bit length for the current block.
 * IN assertion: the field freq is set for all tree elements.
 * OUT assertions: the fields len and code are set to the optimal bit length
 *     and corresponding code. The length opt_len is updated; static_len is
 *     also updated if stree is not null. The field max_code is set.
 */
local void R__build_tree(tree_internal_state *t_state, tree_desc near *desc)
/* tree_desc near *desc;  the tree descriptor */
{
   ct_data near *tree   = desc->dyn_tree;
   ct_data near *stree  = desc->static_tree;
   int elems            = desc->elems;
   int n, m;          /* iterate over heap elements */
   int max_code = -1; /* largest code with non zero frequency */
   int node = elems;  /* next internal node of the tree */

   /* Construct the initial heap, with least frequent element in
    * heap[SMALLEST]. The sons of heap[n] are heap[2*n] and heap[2*n+1].
    * heap[0] is not used.
    */
   t_state->heap_len = 0, t_state->heap_max = HEAP_SIZE;

   for (n = 0; n < elems; n++) {
      if (tree[n].Freq != 0) {
         t_state->heap[++t_state->heap_len] = max_code = n;
         t_state->depth[n] = 0;
      } else {
         tree[n].Len = 0;
      }
   }

   /* The pkzip format requires that at least one distance code exists,
    * and that at least one bit should be sent even if there is only one
    * possible code. So to avoid special checks later on we force at least
    * two codes of non zero frequency.
    */
   while (t_state->heap_len < 2) {
      int new1 = t_state->heap[++t_state->heap_len] = (max_code < 2 ? ++max_code : 0);
      tree[new1].Freq = 1;
      t_state->depth[new1] = 0;
      t_state->opt_len--; if (stree) t_state->static_len -= stree[new1].Len;
      /* new is 0 or 1 so it does not have extra bits */
   }
   desc->max_code = max_code;

   /* The elements heap[heap_len/2+1 .. heap_len] are leaves of the tree,
    * establish sub-heaps of increasing lengths:
    */
   for (n = t_state->heap_len/2; n >= 1; n--) R__pqdownheap(t_state, tree, n);

   /* Construct the Huffman tree by repeatedly combining the least two
    * frequent nodes.
    */
   do {
      pqremove(tree, n);   /* n = node of least frequency */
      m = t_state->heap[SMALLEST];  /* m = node of next least frequency */

      t_state->heap[--t_state->heap_max] = n; /* keep the nodes sorted by frequency */
      t_state->heap[--t_state->heap_max] = m;

      /* Create a new node father of n and m */
      tree[node].Freq = tree[n].Freq + tree[m].Freq;
      t_state->depth[node] = (uch) (MAX(t_state->depth[n], t_state->depth[m]) + 1);
      tree[n].Dad = tree[m].Dad = node;
#ifdef DUMP_BL_TREE
      if (tree == t_state->bl_tree) {
         fprintf(stderr,"\nnode %d(%d), sons %d(%d) %d(%d)",
                 node, tree[node].Freq, n, tree[n].Freq, m, tree[m].Freq);
      }
#endif
      /* and insert the new node in the heap */
      t_state->heap[SMALLEST] = node++;
      R__pqdownheap(t_state, tree, SMALLEST);

   } while (t_state->heap_len >= 2);

   t_state->heap[--t_state->heap_max] = t_state->heap[SMALLEST];

   /* At this point, the fields freq and dad are set. We can now
    * generate the bit lengths.
    */
   R__gen_bitlen(t_state,(tree_desc near *)desc);

   /* The field len is now set, we can generate the bit codes */
   R__gen_codes (t_state, (ct_data near *)tree, max_code);
}

/* ===========================================================================
 * Scan a literal or distance tree to determine the frequencies of the codes
 * in the bit length tree. Updates opt_len to take into account the repeat
 * counts. (The contribution of the bit length codes will be added later
 * during the construction of bl_tree.)
 */
local void R__scan_tree (tree_internal_state *t_state, ct_data near *tree, int max_code)
/* ct_data near *tree;  the tree to be scanned */
/* int max_code;        and its largest code of non zero frequency */
{
   int n;                     /* iterates over all tree elements */
   int prevlen = -1;          /* last emitted length */
   int curlen;                /* length of current code */
   int nextlen = tree[0].Len; /* length of next code */
   int count = 0;             /* repeat count of the current code */
   int max_count = 7;         /* max repeat count */
   int min_count = 4;         /* min repeat count */

   if (nextlen == 0) max_count = 138, min_count = 3;
   tree[max_code+1].Len = (ush)-1; /* guard */

   for (n = 0; n <= max_code; n++) {
      curlen = nextlen; nextlen = tree[n+1].Len;
      if (++count < max_count && curlen == nextlen) {
         continue;
      } else if (count < min_count) {
         t_state->bl_tree[curlen].Freq += count;
      } else if (curlen != 0) {
         if (curlen != prevlen) t_state->bl_tree[curlen].Freq++;
         t_state->bl_tree[REP_3_6].Freq++;
      } else if (count <= 10) {
         t_state->bl_tree[REPZ_3_10].Freq++;
      } else {
         t_state->bl_tree[REPZ_11_138].Freq++;
      }
      count = 0; prevlen = curlen;
      if (nextlen == 0) {
         max_count = 138, min_count = 3;
      } else if (curlen == nextlen) {
         max_count = 6, min_count = 3;
      } else {
         max_count = 7, min_count = 4;
      }
   }
}

/* ===========================================================================
 * Send a literal or distance tree in compressed form, using the codes in
 * bl_tree.
 */
local void R__send_tree (bits_internal_state *state, tree_internal_state *t_state, ct_data near *tree, int max_code)
/* ct_data near *tree;  the tree to be scanned */
/* int max_code;        and its largest code of non zero frequency */
{
   int n;                     /* iterates over all tree elements */
   int prevlen = -1;          /* last emitted length */
   int curlen;                /* length of current code */
   int nextlen = tree[0].Len; /* length of next code */
   int count = 0;             /* repeat count of the current code */
   int max_count = 7;         /* max repeat count */
   int min_count = 4;         /* min repeat count */

   /* tree[max_code+1].Len = -1; */  /* guard already set */
   if (nextlen == 0) max_count = 138, min_count = 3;

   for (n = 0; n <= max_code; n++) {
      curlen = nextlen; nextlen = tree[n+1].Len;
      if (++count < max_count && curlen == nextlen) {
         continue;
      } else if (count < min_count) {
         do { send_code(curlen, t_state->bl_tree); } while (--count != 0);

      } else if (curlen != 0) {
         if (curlen != prevlen) {
            send_code(curlen, t_state->bl_tree); count--;
         }
         Assert(count >= 3 && count <= 6, " 3_6?");
         send_code(REP_3_6, t_state->bl_tree); R__send_bits(state,count-3, 2);

      } else if (count <= 10) {
         send_code(REPZ_3_10, t_state->bl_tree); R__send_bits(state,count-3, 3);

      } else {
         send_code(REPZ_11_138, t_state->bl_tree); R__send_bits(state,count-11, 7);
      }
      count = 0; prevlen = curlen;
      if (nextlen == 0) {
         max_count = 138, min_count = 3;
      } else if (curlen == nextlen) {
         max_count = 6, min_count = 3;
      } else {
         max_count = 7, min_count = 4;
      }
   }
}

/* ===========================================================================
 * Construct the Huffman tree for the bit lengths and return the index in
 * bl_order of the last bit length code to send.
 */
local int R__build_bl_tree(tree_internal_state *t_state)
{
   int max_blindex;  /* index of last bit length code of non zero freq */

   /* Determine the bit length frequencies for literal and distance trees */
   R__scan_tree(t_state,(ct_data near *)t_state->dyn_ltree, t_state->l_desc.max_code);
   R__scan_tree(t_state,(ct_data near *)t_state->dyn_dtree, t_state->d_desc.max_code);

   /* Build the bit length tree: */
   R__build_tree(t_state,(tree_desc near *)(&t_state->bl_desc));
   /* opt_len now includes the length of the tree representations, except
    * the lengths of the bit lengths codes and the 5+5+4 bits for the counts.
    */

   /* Determine the number of bit length codes to send. The pkzip format
    * requires that at least 4 bit length codes be sent. (appnote.txt says
    * 3 but the actual value used is 4.)
    */
   for (max_blindex = BL_CODES-1; max_blindex >= 3; max_blindex--) {
      if (t_state->bl_tree[bl_order[max_blindex]].Len != 0) break;
   }
   /* Update opt_len to include the bit length tree and counts */
   t_state->opt_len += 3*(max_blindex+1) + 5+5+4;
   Tracev((stderr, "\ndyn trees: dyn %ld, stat %ld", t_state->opt_len, t_state->static_len));

   return max_blindex;
}

/* ===========================================================================
 * Send the header for a block using dynamic Huffman trees: the counts, the
 * lengths of the bit length codes, the literal tree and the distance tree.
 * IN assertion: lcodes >= 257, dcodes >= 1, blcodes >= 4.
 */
local void R__send_all_trees(bits_internal_state *state, tree_internal_state *t_state, int lcodes, int dcodes, int blcodes)
/* int lcodes, dcodes, blcodes;  number of codes for each tree */
{
   int rank;                    /* index in bl_order */

   Assert (lcodes >= 257 && dcodes >= 1 && blcodes >= 4, "not enough codes");
   Assert (lcodes <= L_CODES && dcodes <= D_CODES && blcodes <= BL_CODES,
           "too many codes");
   Tracev((stderr, "\nbl counts: "));
   R__send_bits(state,lcodes-257, 5);
   /* not +255 as stated in appnote.txt 1.93a or -256 in 2.04c */
   R__send_bits(state,dcodes-1,   5);
   R__send_bits(state,blcodes-4,  4); /* not -3 as stated in appnote.txt */
   for (rank = 0; rank < blcodes; rank++) {
      Tracev((stderr, "\nbl code %2d ", bl_order[rank]));
      R__send_bits(state,t_state->bl_tree[bl_order[rank]].Len, 3);
   }
   Tracev((stderr, "\nbl tree: sent %ld", R__bits_sent));

   R__send_tree(state,t_state,(ct_data near *)t_state->dyn_ltree, lcodes-1); /* send the literal tree */
   Tracev((stderr, "\nlit tree: sent %ld", R__bits_sent));

   R__send_tree(state,t_state,(ct_data near *)t_state->dyn_dtree, dcodes-1); /* send the distance tree */
   Tracev((stderr, "\ndist tree: sent %ld", R__bits_sent));
}

/* ===========================================================================
 * Determine the best encoding for the current block: dynamic trees, static
 * trees or store, and output the encoded block to the zip file. This function
 * returns the total compressed length for the file so far.
 */
ulg R__flush_block(bits_internal_state *state, char *buf, ulg stored_len, int eof, int *errorflag)
/* char *buf;         input block, or NULL if too old */
/* ulg stored_len;    length of input block */
/* int eof;           true if this is the last block for a file */
{
   tree_internal_state *t_state = state->t_state;
   ulg opt_lenb, static_lenb; /* opt_len and static_len in bytes */
   int max_blindex;  /* index of last bit length code of non zero freq */

   t_state->flag_buf[t_state->last_flags] = t_state->flags; /* Save the flags for the last 8 items */

   /* Check if the file is ascii or binary */
   if (*t_state->R__file_type == (ush)UNKNOWN) R__set_file_type(t_state);

   /* Construct the literal and distance trees */
   R__build_tree(t_state,(tree_desc near *)(&t_state->l_desc));
   Tracev((stderr, "\nlit data: dyn %ld, stat %ld", t_state->opt_len, t_state->static_len));

   R__build_tree(t_state,(tree_desc near *)(&t_state->d_desc));
   Tracev((stderr, "\ndist data: dyn %ld, stat %ld", opt_len, static_len));
   /* At this point, opt_len and static_len are the total bit lengths of
    * the compressed block data, excluding the tree representations.
    */

   /* Build the bit length tree for the above two trees, and get the index
    * in bl_order of the last bit length code to send.
    */
   max_blindex = R__build_bl_tree(t_state);

   /* Determine the best encoding. Compute first the block length in bytes */
   opt_lenb = (t_state->opt_len+3+7)>>3;
   static_lenb = (t_state->static_len+3+7)>>3;
   t_state->input_len += stored_len; /* for debugging only */

   Trace((stderr, "\nopt %lu(%lu) stat %lu(%lu) stored %lu lit %u dist %u ",
          opt_lenb, opt_len, static_lenb, static_len, stored_len,
          last_lit, last_dist));

   if (static_lenb <= opt_lenb) opt_lenb = static_lenb;

#ifndef PGP /* PGP can't handle stored blocks */
   /* If compression failed and this is the first and last block,
    * and if the zip file can be seeked (to rewrite the local header),
    * the whole file is transformed into a stored file:
    */
#ifdef FORCE_METHOD
   if (gCompressionLevel == 1 && eof && compressed_len == 0L) { /* force stored file */
#else
      if (stored_len <= opt_lenb && eof && t_state->compressed_len == 0L && 0) {
#endif
         /* Since LIT_BUFSIZE <= 2*WSIZE, the input data must be there: */
         if (buf == (char *) NULL) { R__error ("block vanished"); *errorflag = 1; }

         R__copy_block(state, buf, (unsigned)stored_len, 0); /* without header */
         t_state->compressed_len = stored_len << 3;
         *t_state->R__file_method = STORE;
      } else
#endif /* PGP */

#ifdef FORCE_METHOD
         if (gCompressionLevel == 2 && buf != (char*)NULL) { /* force stored block */
#else
            if (stored_len+4 <= opt_lenb && buf != (char*)NULL) {
               /* 4: two words for the lengths */
#endif
               /* The test buf != NULL is only necessary if LIT_BUFSIZE > WSIZE.
                * Otherwise we can't have processed more than WSIZE input bytes since
                * the last block flush, because compression would have been
                * successful. If LIT_BUFSIZE <= WSIZE, it is never too late to
                * transform a block into a stored block.
                */
               R__send_bits(state,(STORED_BLOCK<<1)+eof, 3);  /* send block type */
               t_state->compressed_len = (t_state->compressed_len + 3 + 7) & ~7L;
               t_state->compressed_len += (stored_len + 4) << 3;

               R__copy_block(state, buf, (unsigned)stored_len, 1); /* with header */

#ifdef FORCE_METHOD
            } else if (gCompressionLevel == 3) { /* force static trees */
#else
            } else if (static_lenb == opt_lenb) {
#endif
               R__send_bits(state,(STATIC_TREES<<1)+eof, 3);
               R__compress_block(state, t_state, (ct_data near *)t_state->static_ltree,
                                 (ct_data near *)t_state->static_dtree );
               t_state->compressed_len += 3 + t_state->static_len;
            } else {
               R__send_bits(state,(DYN_TREES<<1)+eof, 3);
               R__send_all_trees(state, t_state, t_state->l_desc.max_code+1, t_state->d_desc.max_code+1, max_blindex+1);
               R__compress_block(state, t_state, (ct_data near *)t_state->dyn_ltree, (ct_data near *)t_state->dyn_dtree);
               t_state->compressed_len += 3 + t_state->opt_len;
            }
            Assert (t_state->compressed_len == t_state->R__bits_sent, "bad compressed size");
            R__init_block(t_state);

            if (eof) {
#if defined(PGP) && !defined(MMAP)
               /* Wipe out sensitive data for pgp */
               /*
                *# ifdef DYN_ALLOC
                *       extern uch *R__window;
                *# else
                *       extern uch R__window[];
                *# endif
                */
               memset(R__window, 0, (unsigned)(2*WSIZE-1)); /* -1 needed if WSIZE=32K */
#else /* !PGP */
               Assert (input_len == R__isize, "bad input size");
#endif
               R__bi_windup(state);
               t_state->compressed_len += 7;  /* align on byte boundary */
            }
            Tracev((stderr,"\ncomprlen %lu(%lu) ", t_state->compressed_len>>3,
                    t_state->compressed_len-7*eof));

            return t_state->compressed_len >> 3;
         }

      /* ===========================================================================
       * Save the match info and tally the frequency counts. Return true if
       * the current block must be flushed.
       */
      int R__ct_tally (bits_internal_state *state, int dist, int lc)
      /* int dist;   distance of matched string */
      /* int lc;     match length-MIN_MATCH or unmatched char (if dist==0) */
      {
         tree_internal_state *t_state = state->t_state;
         t_state->l_buf[t_state->last_lit++] = (uch)lc;
         if (dist == 0) {
            /* lc is the unmatched char */
            t_state->dyn_ltree[lc].Freq++;
         } else {
            /* Here, lc is the match length - MIN_MATCH */
            dist--;             /* dist = match distance - 1 */
            Assert((ush)dist < (ush)MAX_DIST &&
                   (ush)lc <= (ush)(MAX_MATCH-MIN_MATCH) &&
                   (ush)d_code(dist) < (ush)D_CODES,  "R__ct_tally: bad match");

            t_state->dyn_ltree[t_state->length_code[lc]+LITERALS+1].Freq++;
            t_state->dyn_dtree[d_code(dist)].Freq++;

            t_state->d_buf[t_state->last_dist++] = dist;
            t_state->flags |= t_state->flag_bit;
         }
         t_state->flag_bit <<= 1;

         /* Output the flags if they fill a byte: */
         if ((t_state->last_lit & 7) == 0) {
            t_state->flag_buf[t_state->last_flags++] = t_state->flags;
            t_state->flags = 0, t_state->flag_bit = 1;
         }
         /* Try to guess if it is profitable to stop the current block here */
         if (gCompressionLevel > 2 && (t_state->last_lit & 0xfff) == 0) {
            /* Compute an upper bound for the compressed length */
            ulg out_length = (ulg)t_state->last_lit*8L;
            ulg in_length = (ulg)state->R__strstart-state->R__block_start;
            int dcode;
            for (dcode = 0; dcode < D_CODES; dcode++) {
               out_length += (ulg)t_state->dyn_dtree[dcode].Freq*(5L+extra_dbits[dcode]);
            }
            out_length >>= 3;
            Trace((stderr,"\nlast_lit %u, last_dist %u, in %ld, out ~%ld(%ld%%) ",
                   t_state->last_lit, t_state->last_dist, in_length, out_length,
                   100L - out_length*100L/in_length));
            if (t_state->last_dist < t_state->last_lit/2 && out_length < in_length/2) return 1;
         }
         return (t_state->last_lit == LIT_BUFSIZE-1 || t_state->last_dist == DIST_BUFSIZE);
         /* We avoid equality with LIT_BUFSIZE because of wraparound at 64K
          * on 16 bit machines and because stored blocks are restricted to
          * 64K-1 bytes.
          */
      }

      /* ===========================================================================
       * Send the block data compressed using the given Huffman trees
       */
      local void R__compress_block(bits_internal_state *state, tree_internal_state *t_state, ct_data near *ltree, ct_data near *dtree)
      /* ct_data near *ltree;  literal tree */
      /* ct_data near *dtree;  distance tree */
      {
         unsigned dist;      /* distance of matched string */
         int lc;             /* match length or unmatched char (if dist == 0) */
         unsigned lx = 0;    /* running index in l_buf */
         unsigned dx = 0;    /* running index in d_buf */
         unsigned fx = 0;    /* running index in flag_buf */
         uch flag = 0;       /* current flags */
         unsigned code;      /* the code to send */
         int extra;          /* number of extra bits to send */

         if (t_state->last_lit != 0) do {
            if ((lx & 7) == 0) flag = t_state->flag_buf[fx++];
            lc = t_state->l_buf[lx++];
            if ((flag & 1) == 0) {
               send_code(lc, ltree); /* send a literal byte */
               Tracecv(isgraph(lc), (stderr," '%c' ", lc));
            } else {
               /* Here, lc is the match length - MIN_MATCH */
               code = t_state->length_code[lc];
               send_code(code+LITERALS+1, ltree); /* send the length code */
               extra = extra_lbits[code];
               if (extra != 0) {
                  lc -= t_state->base_length[code];
                  R__send_bits(state,lc, extra);        /* send the extra length bits */
               }
               dist = t_state->d_buf[dx++];
               /* Here, dist is the match distance - 1 */
               code = d_code(dist);
               Assert (code < D_CODES, "bad d_code");

               send_code(code, dtree);       /* send the distance code */
               extra = extra_dbits[code];
               if (extra != 0) {
                  dist -= t_state->base_dist[code];
                  R__send_bits(state,dist, extra);   /* send the extra distance bits */
               }
            } /* literal or match pair ? */
            flag >>= 1;
         } while (lx < t_state->last_lit);

         send_code(END_BLOCK, ltree);
      }

      /* ===========================================================================
       * Set the file type to ASCII or BINARY, using a crude approximation:
       * binary if more than 20% of the bytes are <= 6 or >= 128, ascii otherwise.
       * IN assertion: the fields freq of dyn_ltree are set and the total of all
       * frequencies does not exceed 64K (to fit in an int on 16 bit machines).
       */
      local void R__set_file_type(tree_internal_state *t_state)
      {
         int n = 0;
         unsigned ascii_freq = 0;
         unsigned bin_freq = 0;
         while (n < 7)        bin_freq += t_state->dyn_ltree[n++].Freq;
         while (n < 128)    ascii_freq += t_state->dyn_ltree[n++].Freq;
         while (n < LITERALS) bin_freq += t_state->dyn_ltree[n++].Freq;
         *t_state->R__file_type = bin_freq > (ascii_freq >> 2) ? BINARY : ASCII;
#ifndef PGP
#if 0
         if (*t_state->R__file_type == BINARY && translate_eol) {
            warn("-l used on binary file", "");
         }
#endif
#endif
      }
