/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
 * Bits.h is a collection of function definitions for the historic compression
 * algorithm used by ROOT.  This appears to have been a modified ZLIB-like
 * algorithm.
 *
 * As best as we can tell, this has not been used in 20+ years.
 */

#include "ZIP.h"

/* This following used to be declared (as globals) in ZDeflate.h */

/**
 * Size of internal hash table.
 *
 * Previously controlled by various macros which no longer exist.
 *
 */
#define HASH_BITS  15
#define HASH_SIZE (unsigned)(1<<HASH_BITS)

#if defined(BIG_MEM) || defined(MMAP)
typedef unsigned Pos; /* must be at least 32 bits */
#else
typedef ush Pos;
#endif
typedef unsigned IPos;
/* A Pos is an index in the character window. We use short instead of int to
 * save space in the various tables. IPos is used only for parameter passing.
 */

/* end of ZDeflate.h section */


#ifdef __cplusplus
extern "C" {
#endif

struct bits_internal_state {
   unsigned short bi_buf;
/* Output buffer. bits are inserted starting at the bottom (least significant
 * bits).
 */

   int bi_valid;
/* Number of valid bits in bi_buf.  All bits above the last valid bit
 * are always zero.
 */

   char *in_buf, *out_buf;
/* Current input and output buffers. in_buf is used only for in-memory
 * compression.
 */

   unsigned in_offset, out_offset;
/* Current offset in input and output buffers. in_offset is used only for
 * in-memory compression. On 16 bit machiens, the buffer is limited to 64K.
 */

   unsigned in_size, out_size;
/* Size of current input and output buffers */

/* On some platform (MacOS) marking this thread local does not work,
   however in our use this is a constant, so we do not really need to make it
   thread local */
/* __thread */
/* not used?
   int (*R__read_buf) OF((char *buf, unsigned size)) = R__mem_read;
*/
/* Current input function. Set to R__mem_read for in-memory compression */

#ifdef DEBUG
   ulg R__bits_sent;   /* bit length of the compressed data */
#endif

   int error_flag;

   /* The following used to be declared (as globals) in ZDeflate.h */

   /* ===========================================================================
    * Local data used by the "longest match" routines.
    */

#ifndef DYN_ALLOC
   uch    R__window[2L*WSIZE];
   /* Sliding window. Input bytes are read into the second half of the window,
    * and move to the first half later to keep a dictionary of at least WSIZE
    * bytes. With this organization, matches are limited to a distance of
    * WSIZE-MAX_MATCH bytes, but this ensures that IO is always
    * performed with a length multiple of the block size. Also, it limits
    * the window size to 64K, which is quite useful on MSDOS.
    * To do: limit the window size to WSIZE+BSZ if SMALL_MEM (the code would
    * be less efficient since the data would have to be copied WSIZE/BSZ times)
    */
   Pos    R__prev[WSIZE];
   /* Link to older string with same hash index. To limit the size of this
    * array to 64K, this link is maintained only for the last 32K strings.
    * An index in this array is thus a window index modulo 32K.
    */
   Pos    R__head[HASH_SIZE];
   /* Heads of the hash chains or NIL. If your compiler thinks that
    * HASH_SIZE is a dynamic value, recompile with -DDYN_ALLOC.
    */
#else
   uch    * near R__window ; /* = NULL; */
   Pos    * near R__prev   ; /* = NULL; */
   Pos    * near R__head;
#endif
   ulg R__window_size;
   /* window size, 2*WSIZE except for MMAP or BIG_MEM, where it is the
    * input file length plus MIN_LOOKAHEAD.
    */

   long R__block_start;
   /* window position at the beginning of the current output block. Gets
    * negative when the window is moved backwards.
    */

   /* local */ int sliding;
   /* Set to false when the input file is already in memory */

   /* local */ unsigned ins_h;  /* hash index of string to be inserted */

#define H_SHIFT  ((HASH_BITS+MIN_MATCH-1)/MIN_MATCH)
   /* Number of bits by which ins_h and del_h must be shifted at each
    * input step. It must be such that after MIN_MATCH steps, the oldest
    * byte no longer takes part in the hash key, that is:
    *   H_SHIFT * MIN_MATCH >= HASH_BITS
    */

   unsigned int near R__prev_length;
   /* Length of the best match at previous step. Matches not greater than this
    * are discarded. This is used in the lazy match evaluation.
    */

   unsigned near R__strstart;      /* start of string to insert */
   unsigned near R__match_start;   /* start of matching string */
   /* local */ int           eofile;           /* flag set at end of input file */
   /* local */ unsigned      lookahead;        /* number of valid bytes ahead in window */

   unsigned near R__max_chain_length;
   /* To speed up deflation, hash chains are never searched beyond this length.
    * A higher limit improves compression ratio but degrades the speed.
    */

   /* local */ unsigned int max_lazy_match;
   /* Attempt to find a better match only when the current match is strictly
    * smaller than this value. This mechanism is used only for compression
    * levels >= 4.
    */
#define max_insert_length  state->max_lazy_match
   /* Insert new strings in the hash table only if the match length
    * is not greater than this length. This saves time but degrades compression.
    * max_insert_length is used only for compression levels <= 3.
    */

   unsigned near R__good_match;
   /* Use a faster search when the previous match is longer than this */

#ifdef  FULL_SEARCH
# define R__nice_match MAX_MATCH
#else
   int near R__nice_match; /* Stop searching when current match exceeds this */
#endif

   tree_internal_state *t_state;
   /* Pointer to the ZTree internal state, this will be thread local */
};

/* ===========================================================================
 * Initialize the bit string routines.
 */
int R__bi_init (bits_internal_state *state);

/* ===========================================================================
 * Send a value on a given number of bits.
 * IN assertion: length <= 16 and value fits in length bits.
 */
    /* int value;   value to send */
    /* int length;  number of bits */
void R__send_bits(bits_internal_state *state, int value, int length);


/* ===========================================================================
 * Reverse the first len bits of a code, using straightforward code (a faster
 * method would use a table)
 * IN assertion: 1 <= len <= 15
 */
    /* unsigned code;  the value to invert */
    /* int len;        its bit length */
unsigned R__bi_reverse(unsigned code, int len);


/* ===========================================================================
 * Write out any remaining bits in an incomplete byte.
 */
void R__bi_windup(bits_internal_state *state);


/* ===========================================================================
 * Copy a stored block to the zip file, storing first the length and its
 * one's complement if requested.
 */
    /* char far *buf;  the input data */
    /* unsigned len;   its length */
    /* int header;     true if block header must be written */
void R__copy_block(bits_internal_state *state, char far *buf, unsigned len, int header);


/* ===========================================================================
 * In-memory read function. As opposed to file_read(), this function
 * does not perform end-of-line translation, and does not update the
 * crc and input size.
 *    Note that the size of the entire input buffer is an unsigned long,
 * but the size used in R__mem_read() is only an unsigned int. This makes a
 * difference on 16 bit machines. R__mem_read() may be called several
 * times for an in-memory compression.
 */
int  R__mem_read     OF((bits_internal_state *state, char *b,    unsigned bsize));

/* ===========================================================================
 * In-memory compression. This version can be used only if the entire input
 * fits in one memory buffer. The compression is then done in a single
 * call of R__memcompress(). (An extension to allow repeated calls would be
 * possible but is not needed here.)
 * The first two bytes of the compressed output are set to a short with the
 * method used (DEFLATE or STORE). The following four bytes contain the CRC.
 * The values are stored in little-endian order on all machines.
 * This function returns the byte size of the compressed output, including
 * the first six bytes (method and crc).
 */
    /* char *tgt, *src;        target and source buffers */
    /* ulg tgtsize, srcsize;   target and source sizes */
ulg R__memcompress(char *tgt, ulg tgtsize, char *src, ulg srcsize);


/**
 * Decompress a deflated entry.
 */
int R__Inflate(uch** ibufptr, long*  ibufcnt, uch** obufptr, long*  obufcnt);

#ifdef __cplusplus
}  // extern "C"
#endif
