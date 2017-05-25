/* @(#)root/zip:$Id$ */
/* Author: */
/*

 Copyright (C) 1990-1993 Mark Adler, Richard B. Wales, Jean-loup Gailly,
 Kai Uwe Rommel and Igor Mandrichenko.
 For conditions of distribution and use, see copyright notice in zlib.h

*/
/*
 * Changed for ROOT. Functions names have a R__ prepended to differentiate
 * them from function names in later versions of zlib.
 */

#include "zlib.h"
#include "RConfigure.h"
#include "ZipLZMA.h"
#include "ZipLZ4.h"

#include <stdio.h>
#include <assert.h>

/*
 *  bits.c by Jean-loup Gailly and Kai Uwe Rommel.
 *
 *  This is a new version of im_bits.c originally written by Richard B. Wales
 *
 *  PURPOSE
 *
 *      Output variable-length bit strings. Compression can be done
 *      to a file or to memory.
 *
 *  DISCUSSION
 *
 *      The PKZIP "deflate" file format interprets compressed file data
 *      as a sequence of bits.  Multi-bit strings in the file may cross
 *      byte boundaries without restriction.
 *
 *      The first bit of each byte is the low-order bit.
 *
 *      The routines in this file allow a variable-length bit value to
 *      be output right-to-left (useful for literal values). For
 *      left-to-right output (useful for code strings from the tree routines),
 *      the bits must have been reversed first with R__bi_reverse().
 *
 *      For in-memory compression, the compressed bit stream goes directly
 *      into the requested output buffer. The input data is read in blocks
 *      by the R__mem_read() function. The buffer is limited to 64K on 16 bit
 *      machines.
 *
 *  INTERFACE
 *
 *      void R__bi_init (bits_internal_state *state)
 *          Initialize the bit string routines.
 *
 *      void R__send_bits (int value, int length)
 *          Write out a bit string, taking the source bits right to
 *          left.
 *
 *      int R__bi_reverse (int value, int length)
 *          Reverse the bits of a bit string, taking the source bits left to
 *          right and emitting them right to left.
 *
 *      void R__bi_windup (void)
 *          Write out any remaining bits in an incomplete byte.
 *
 *      void R__copy_block(char far *buf, unsigned len, int header)
 *          Copy a stored block to the zip file, storing first the length and
 *          its one's complement if requested.
 *
 *      int R__seekable(void)
 *          Return true if the zip file can be seeked.
 *
 *      ulg R__memcompress (char *tgt, ulg tgtsize, char *src, ulg srcsize);
 *          Compress the source buffer src into the target buffer tgt.
 */

/* #include "zip.h" */
/* #include "ZIP.h" */

/* extern ulg R__window_size; */ /* size of sliding window */


/* ===========================================================================
 *  Prototypes for local functions
 */
/* local int  R__mem_read     OF((char *buf, unsigned size)); */
local void R__flush_outbuf OF((bits_internal_state *state,unsigned w, unsigned size));


/* ===========================================================================
 * Local data used by the "bit string" routines.
 */

#define Buf_size (8 * 2*sizeof(char))
/* Number of bits used within bi_buf. (bi_buf might be implemented on
 * more than 16 bits on some systems.)
 */

/* The following used to be declared (as globals) in ZDeflate.h */

/* Compile with MEDIUM_MEM to reduce the memory requirements or
 * with SMALL_MEM to use as little memory as possible. Use BIG_MEM if the
 * entire input file can be held in memory (not possible on 16 bit systems).
 * Warning: defining these symbols affects HASH_BITS (see below) and thus
 * affects the compression ratio. The compressed output
 * is still correct, and might even be smaller in some cases.
 */

#ifdef SMALL_MEM
#   define HASH_BITS  13  /* Number of bits used to hash strings */
#endif
#ifdef MEDIUM_MEM
#   define HASH_BITS  14
#endif
#ifndef HASH_BITS
#   define HASH_BITS  15
/* For portability to 16 bit machines, do not use values above 15. */
#endif

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

/* Output a 16 bit value to the bit stream, lower (oldest) byte first */
#define PUTSHORT(w) \
{ if (state->out_offset < state->out_size-1) { \
    state->out_buf[state->out_offset++] = (char) ((w) & 0xff); \
    state->out_buf[state->out_offset++] = (char) ((ush)(w) >> 8); \
  } else { \
    R__flush_outbuf(state,(w),2); \
  } \
}

#define PUTBYTE(b) \
{ if (state->out_offset < state->out_size) { \
    state->out_buf[state->out_offset++] = (char) (b); \
  } else { \
    R__flush_outbuf(state,(b),1); \
  } \
}

/* ===========================================================================
   R__ZipMode is used to select the compression algorithm when R__zip is called
   and when R__zipMultipleAlgorithm is called with its last argument set to 0.
   R__ZipMode = 1 : ZLIB compression algorithm is used (default)
   R__ZipMode = 2 : LZMA compression algorithm is used
   R__ZipMode = 4 : LZ4  compression algorithm is used
   R__ZipMode = 0 or 3 : a very old compression algorithm is used
   (the very old algorithm is supported for backward compatibility)
   The LZMA algorithm requires the external XZ package be installed when linking
   is done. LZMA typically has significantly higher compression factors, but takes
   more CPU time and memory resources while compressing.

  The LZ4 algorithm requires the external LZ4 package to be installed when linking
  is done.  LZ4 typically has the worst compression ratios, but much faster decompression
  speeds - sometimes by an order of magnitude.
*/
int R__ZipMode = 1;

/* ===========================================================================
 *  Prototypes for local functions
 */
local int  R__mem_read     OF((bits_internal_state *state, char *b,    unsigned bsize));
local void R__flush_outbuf OF((bits_internal_state *state, unsigned w, unsigned bytes));

/* ===========================================================================
   Function to set the ZipMode
 */
void R__SetZipMode(int mode)
{
   R__ZipMode = mode;
}

/* ===========================================================================
 * Initialize the bit string routines.
 */
int R__bi_init (bits_internal_state *state)
    /* FILE *zipfile;   output zip file, NULL for in-memory compression */
{
    state->bi_buf = 0;
    state->bi_valid = 0;
    state->error_flag = 0;
#ifdef DEBUG
    state->R__bits_sent = 0L;
#endif
    return 0;
}

/* ===========================================================================
 * Send a value on a given number of bits.
 * IN assertion: length <= 16 and value fits in length bits.
 */
void R__send_bits(bits_internal_state *state, int value, int length)
    /* int value;   value to send */
    /* int length;  number of bits */
{
#ifdef DEBUG
    Tracevv((stderr," l %2d v %4x ", length, value));
    Assert(length > 0 && length <= 15, "invalid length");
    state->R__bits_sent += (ulg)length;
#endif
    /* If not enough room in bi_buf, use (valid) bits from bi_buf and
     * (16 - bi_valid) bits from value, leaving (width - (16-bi_valid))
     * unused bits in value.
     */
    if (state->bi_valid > (int)Buf_size - length) {
        state->bi_buf |= (value << state->bi_valid);
        PUTSHORT(state->bi_buf);
        state->bi_buf = (ush)value >> (Buf_size - state->bi_valid);
        state->bi_valid += length - Buf_size;
    } else {
        state->bi_buf |= value << state->bi_valid;
        state->bi_valid += length;
    }
}

/* ===========================================================================
 * Reverse the first len bits of a code, using straightforward code (a faster
 * method would use a table)
 * IN assertion: 1 <= len <= 15
 */
unsigned R__bi_reverse(unsigned code, int len)
    /* unsigned code;  the value to invert */
    /* int len;        its bit length */
{
    register unsigned res = 0;
    do {
        res |= code & 1;
        code >>= 1, res <<= 1;
    } while (--len > 0);
    return res >> 1;
}

/* ===========================================================================
 * Flush the current output buffer.
 */
local void R__flush_outbuf(bits_internal_state *state, unsigned w, unsigned bytes)
    /* unsigned w;      value to flush */
    /* unsigned bytes;  number of bytes to flush (0, 1 or 2) */
{
    R__error("output buffer too small for in-memory compression");
    state->error_flag = 1;

    /* Encrypt and write the output buffer: */
    state->out_offset = 0;
    if (bytes == 2) {
        PUTSHORT(w);
    } else if (bytes == 1) {
        state->out_buf[state->out_offset++] = (char) (w & 0xff);
    }
}

/* ===========================================================================
 * Write out any remaining bits in an incomplete byte.
 */
void R__bi_windup(bits_internal_state *state)
{
    if (state->bi_valid > 8) {
        PUTSHORT(state->bi_buf);
    } else if (state->bi_valid > 0) {
        PUTBYTE(state->bi_buf);
    }
    state->bi_buf = 0;
    state->bi_valid = 0;
#ifdef DEBUG
    state->R__bits_sent = (state->R__bits_sent+7) & ~7;
#endif
}

/* ===========================================================================
 * Copy a stored block to the zip file, storing first the length and its
 * one's complement if requested.
 */
void R__copy_block(bits_internal_state *state, char far *buf, unsigned len, int header)
    /* char far *buf;  the input data */
    /* unsigned len;   its length */
    /* int header;     true if block header must be written */
{
    R__bi_windup(state);              /* align on byte boundary */

    if (header) {
        PUTSHORT((ush)len);
        PUTSHORT((ush)~len);
#ifdef DEBUG
        R__bits_sent += 2*16;
#endif
    }
    if (state->out_offset + len > state->out_size) {
        R__error("output buffer too small for in-memory compression");
        if (verbose) fprintf(stderr, "R__zip: out_offset=%d, len=%d, out_size=%d\n",state->out_offset,len,state->out_size);
        state->error_flag = 1;
    } else {
        memcpy(state->out_buf + state->out_offset, buf, len);
        state->out_offset += len;
    }
#ifdef DEBUG
    state->R__bits_sent += (ulg)len<<3;
#endif
}


/* ===========================================================================
 * Return true if the zip file can be seeked. This is used to check if
 * the local header can be re-rewritten. This function always returns
 * true for in-memory compression.
 * IN assertion: the local header has already been written (ftell() > 0).
 */
int R__seekable()
{
#if 0
    return (zfile == NULL ||
            (fseek(zfile, -1L, SEEK_CUR) == 0 &&
             fseek(zfile,  1L, SEEK_CUR) == 0));
#endif

    return (0);
}

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

ulg R__memcompress(char *tgt, ulg tgtsize, char *src, ulg srcsize)
    /* char *tgt, *src;        target and source buffers */
    /* ulg tgtsize, srcsize;   target and source sizes */
{
    ush att      = (ush)UNKNOWN;
    ush flags    = 0;
    ulg crc      = 0;
    int method   = Z_DEFLATED;
    bits_internal_state state;

    if (tgtsize <= 6L) { R__error("target buffer too small"); /* errorflag = 1; */ }
#if 0
    crc = updcrc((char *)NULL, 0);
    crc = updcrc(src, (extent) srcsize);
#endif
#ifdef DYN_ALLOC
    state.R__window = 0;
    state.R__prev = 0;
#endif

    /* R__read_buf  = R__mem_read; */
    /* assert(R__read_buf == R__mem_read); */
    state.in_buf    = src;
    state.in_size   = (unsigned)srcsize;
    state.in_offset = 0;

    state.out_buf    = tgt;
    state.out_size   = (unsigned)tgtsize;
    state.out_offset = 2 + 4;
    state.R__window_size = 0L;

    R__bi_init(&state);
    state.t_state = R__get_thread_tree_state();
    R__ct_init(state.t_state, &att, &method);
    R__lm_init(&state,(level != 0 ? level : 1), &flags);
    R__Deflate(&state,&(state.error_flag));
    state.R__window_size = 0L; /* was updated by lm_init() */

    /* For portability, force little-endian order on all machines: */
    tgt[0] = (char)(method & 0xff);
    tgt[1] = (char)((method >> 8) & 0xff);
    tgt[2] = (char)(crc & 0xff);
    tgt[3] = (char)((crc >> 8) & 0xff);
    tgt[4] = (char)((crc >> 16) & 0xff);
    tgt[5] = (char)((crc >> 24) & 0xff);

    return (ulg)state.out_offset;
}

/* ===========================================================================
 * In-memory read function. As opposed to file_read(), this function
 * does not perform end-of-line translation, and does not update the
 * crc and input size.
 *    Note that the size of the entire input buffer is an unsigned long,
 * but the size used in R__mem_read() is only an unsigned int. This makes a
 * difference on 16 bit machines. R__mem_read() may be called several
 * times for an in-memory compression.
 */
local int R__mem_read(bits_internal_state *state,char *b, unsigned bsize)
{
    if (state->in_offset < state->in_size) {
        ulg block_size = state->in_size - state->in_offset;
        if (block_size > (ulg)bsize) block_size = (ulg)bsize;
        memcpy(b, state->in_buf + state->in_offset, (unsigned)block_size);
        state->in_offset += (unsigned)block_size;
        return (int)block_size;
    } else {
        return 0; /* end of input */
    }
}

/***********************************************************************
 *                                                                     *
 * Name: R__zip                                      Date:    20.01.95 *
 * Author: E.Chernyaev (IHEP/Protvino)               Revised:          *
 *                                                                     *
 * Function: In memory ZIP compression                                 *
 *           It's a variant of R__memcompress adopted to be issued from*
 *           FORTRAN. Written for DELPHI collaboration (CERN)          *
 *                                                                     *
 * Input: cxlevel - compression level                                  *
 *        srcsize - size of input buffer                               *
 *        src     - input buffer                                       *
 *        tgtsize - size of target buffer                              *
 *                                                                     *
 * Output: tgt - target buffer (compressed)                            *
 *         irep - size of compressed data (0 - if error)               *
 *                                                                     *
 ***********************************************************************/
#define HDRSIZE 9
/* static  __thread int error_flag; */

void R__zipMultipleAlgorithm(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep, int compressionAlgorithm)
     /* int cxlevel;                      compression level */
     /* int  *srcsize, *tgtsize, *irep;   source and target sizes, replay */
     /* char *tgt, *src;                  source and target buffers */
     /* compressionAlgorithm 0 = use global setting */
     /*                      1 = zlib */
     /*                      2 = lzma */
     /*                      3 = old */
{
  int err;
  int method   = Z_DEFLATED;

  if (cxlevel <= 0) {
    *irep = 0;
    return;
  }

  if (compressionAlgorithm == 0) {
    compressionAlgorithm = R__ZipMode;
  }

  // The LZMA compression algorithm from the XZ package
  if (compressionAlgorithm == 2) {
    R__zipLZMA(cxlevel, srcsize, src, tgtsize, tgt, irep);
    return;
  } else if (compressionAlgorithm == 4) {
     R__zipLZ4(cxlevel, srcsize, src, tgtsize, tgt, irep);
     return;
  }

  // The very old algorithm for backward compatibility
  // 0 for selecting with R__ZipMode in a backward compatible way
  // 3 for selecting in other cases
  if (compressionAlgorithm == 3 || compressionAlgorithm == 0) {
    bits_internal_state state;
    ush att      = (ush)UNKNOWN;
    ush flags    = 0;
    if (cxlevel > 9) cxlevel = 9;
    level        = cxlevel;

    *irep        = 0;
    /* error_flag   = 0; */
    if (*tgtsize <= 0) {
       R__error("target buffer too small");
       return;
    }
    if (*srcsize > 0xffffff) {
       R__error("source buffer too big");
       return;
    }

#ifdef DYN_ALLOC
    state.R__window = 0;
    state.R__prev = 0;
#endif

    /* R__read_buf  = R__mem_read; */
    /* assert(R__read_buf == R__mem_read); */
    state.in_buf    = src;
    state.in_size   = (unsigned) (*srcsize);
    state.in_offset = 0;

    state.out_buf     = tgt;
    state.out_size    = (unsigned) (*tgtsize);
    state.out_offset  = HDRSIZE;
    state.R__window_size = 0L;

    if (0 != R__bi_init(&state) ) return;       /* initialize bit routines */
    state.t_state = R__get_thread_tree_state();
    if (0 != R__ct_init(state.t_state,&att, &method)) return; /* initialize tree routines */
    if (0 != R__lm_init(&state,level, &flags)) return; /* initialize compression */
    R__Deflate(&state,&state.error_flag);                  /* compress data */
    if (state.error_flag != 0) return;

    tgt[0] = 'C';               /* Signature 'C'-Chernyaev, 'S'-Smirnov */
    tgt[1] = 'S';
    tgt[2] = (char) method;

    state.out_size  = state.out_offset - HDRSIZE;         /* compressed size */
    tgt[3] = (char)(state.out_size & 0xff);
    tgt[4] = (char)((state.out_size >> 8) & 0xff);
    tgt[5] = (char)((state.out_size >> 16) & 0xff);

    tgt[6] = (char)(state.in_size & 0xff);         /* decompressed size */
    tgt[7] = (char)((state.in_size >> 8) & 0xff);
    tgt[8] = (char)((state.in_size >> 16) & 0xff);

    *irep     = state.out_offset;
    return;

  // 1 is for ZLIB (which is the default), ZLIB is also used for any illegal
  // algorithm setting
  } else {

    z_stream stream;
    //Don't use the globals but want name similar to help see similarities in code
    unsigned l_in_size, l_out_size;
    *irep = 0;

    /* error_flag   = 0; */
    if (*tgtsize <= 0) {
       R__error("target buffer too small");
       return;
    }
    if (*srcsize > 0xffffff) {
       R__error("source buffer too big");
       return;
    }

    stream.next_in   = (Bytef*)src;
    stream.avail_in  = (uInt)(*srcsize);

    stream.next_out  = (Bytef*)(&tgt[HDRSIZE]);
    stream.avail_out = (uInt)(*tgtsize);

    stream.zalloc    = (alloc_func)0;
    stream.zfree     = (free_func)0;
    stream.opaque    = (voidpf)0;

    if (cxlevel > 9) cxlevel = 9;
    err = deflateInit(&stream, cxlevel);
    if (err != Z_OK) {
       printf("error %d in deflateInit (zlib)\n",err);
       return;
    }

    err = deflate(&stream, Z_FINISH);
    if (err != Z_STREAM_END) {
       deflateEnd(&stream);
       /* No need to print an error message. We simply abandon the compression
          the buffer cannot be compressed or compressed buffer would be larger than original buffer
          printf("error %d in deflate (zlib) is not = %d\n",err,Z_STREAM_END);
       */
       return;
    }

    err = deflateEnd(&stream);

    tgt[0] = 'Z';               /* Signature ZLib */
    tgt[1] = 'L';
    tgt[2] = (char) method;

    l_in_size   = (unsigned) (*srcsize);
    l_out_size  = stream.total_out;             /* compressed size */
    tgt[3] = (char)(l_out_size & 0xff);
    tgt[4] = (char)((l_out_size >> 8) & 0xff);
    tgt[5] = (char)((l_out_size >> 16) & 0xff);

    tgt[6] = (char)(l_in_size & 0xff);         /* decompressed size */
    tgt[7] = (char)((l_in_size >> 8) & 0xff);
    tgt[8] = (char)((l_in_size >> 16) & 0xff);

    *irep = stream.total_out + HDRSIZE;
    return;
  }
}

void R__zip(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep)
{
  R__zipMultipleAlgorithm(cxlevel, srcsize, src, tgtsize, tgt, irep, 0);
}

void R__error(char *msg)
{
  if (verbose) fprintf(stderr,"R__zip: %s\n",msg);
  /* error_flag = 1; */
}


