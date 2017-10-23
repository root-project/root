
/**
 *
 * Copyright (C) 1990-1993 Mark Adler, Richard B. Wales, Jean-loup Gailly,
 * Kai Uwe Rommel and Igor Mandrichenko.
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 * Changed for ROOT. Functions names have a R__ prepended to differentiate
 * them from function names in later versions of zlib.
 */

#include "Bits.h"

#include "zlib.h"

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


// Global value of the compression level for the old compression algorithm.
// NOTE: Not thread-safe.
int gCompressionLevel = 6;

/* ===========================================================================
 *  Prototypes for local functions
 */
local void R__flush_outbuf OF((bits_internal_state *state,unsigned w, unsigned size));


/* ===========================================================================
 * Local data used by the "bit string" routines.
 */
/* Number of bits used within bi_buf. (bi_buf might be implemented on
 * more than 16 bits on some systems.)
 */
#define Buf_size (8 * 2*sizeof(char))

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
 *  Prototypes for local functions
 */
local void R__flush_outbuf OF((bits_internal_state *state, unsigned w, unsigned bytes));


/*
 * Simple error-printing.
 *
 * Note that gBitsVerbose defaults to 0 and is a compilation-time change one
 * must do to get an error message back.
 */
static int gBitsVerbose = 0;
void R__error(const char *msg)
{
  if (gBitsVerbose) fprintf(stderr,"R__zip: %s\n",msg);
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
        if (gBitsVerbose) fprintf(stderr, "R__zip: out_offset=%d, len=%d, out_size=%d\n",state->out_offset,len,state->out_size);
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
    R__lm_init(&state,(gCompressionLevel != 0 ? gCompressionLevel : 1), &flags);
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
int R__mem_read(bits_internal_state *state,char *b, unsigned bsize)
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

