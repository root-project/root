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

#ifdef R__HAS_LZMACOMPRESSION
#include "R__LZMA.h"
#endif

#include <stdio.h>

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
 *      void R__bi_init (FILE *zipfile)
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
local int  R__mem_read     OF((char *buf, unsigned size));
local void R__flush_outbuf OF((unsigned w, unsigned size));


/* ===========================================================================
 * Local data used by the "bit string" routines.
 */
local FILE *zfile; /* output zip file */

local unsigned short bi_buf;
/* Output buffer. bits are inserted starting at the bottom (least significant
 * bits).
 */

#define Buf_size (8 * 2*sizeof(char))
/* Number of bits used within bi_buf. (bi_buf might be implemented on
 * more than 16 bits on some systems.)
 */

local int bi_valid;
/* Number of valid bits in bi_buf.  All bits above the last valid bit
 * are always zero.
 */

local char *in_buf, *out_buf;
/* Current input and output buffers. in_buf is used only for in-memory
 * compression.
 */

local unsigned in_offset, out_offset;
/* Current offset in input and output buffers. in_offset is used only for
 * in-memory compression. On 16 bit machiens, the buffer is limited to 64K.
 */

local unsigned in_size, out_size;
/* Size of current input and output buffers */

int (*R__read_buf) OF((char *buf, unsigned size)) = R__mem_read;
/* Current input function. Set to R__mem_read for in-memory compression */

#ifdef DEBUG
ulg R__bits_sent;   /* bit length of the compressed data */
#endif

/* Output a 16 bit value to the bit stream, lower (oldest) byte first */
#define PUTSHORT(w) \
{ if (out_offset < out_size-1) { \
    out_buf[out_offset++] = (char) ((w) & 0xff); \
    out_buf[out_offset++] = (char) ((ush)(w) >> 8); \
  } else { \
    R__flush_outbuf((w),2); \
  } \
}

#define PUTBYTE(b) \
{ if (out_offset < out_size) { \
    out_buf[out_offset++] = (char) (b); \
  } else { \
    R__flush_outbuf((b),1); \
  } \
}


/* ===========================================================================
   R__ZipMode is used to select the compression algorithm when R__zip is called
   and when R__zipMultipleAlgorithm is called with its last argument set to 0.
   R__ZipMode = 1 : ZLIB compression algorithm is used (default)
   R__ZipMode = 2 : LZMA compression algorithm is used
   R__ZipMode = 0 or 3 : a very old compression algorithm is used
   (the very old algorithm is supported for backward compatibility)
   The LZMA algorithm requires the external XZ package be installed when linking
   is done. LZMA typically has significantly higher compression factors, but takes
   more CPU time and memory resources while compressing.
*/
int R__ZipMode = 1;

/* ===========================================================================
 *  Prototypes for local functions
 */
local int  R__mem_read     OF((char *b,    unsigned bsize));
local void R__flush_outbuf OF((unsigned w, unsigned bytes));

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
void R__bi_init (FILE *zipfile)
    /* FILE *zipfile;   output zip file, NULL for in-memory compression */
{
    zfile  = zipfile;
    bi_buf = 0;
    bi_valid = 0;
#ifdef DEBUG
    R__bits_sent = 0L;
#endif
}

/* ===========================================================================
 * Send a value on a given number of bits.
 * IN assertion: length <= 16 and value fits in length bits.
 */
void R__send_bits(int value, int length)
    /* int value;   value to send */
    /* int length;  number of bits */
{
#ifdef DEBUG
    Tracevv((stderr," l %2d v %4x ", length, value));
    Assert(length > 0 && length <= 15, "invalid length");
    R__bits_sent += (ulg)length;
#endif
    /* If not enough room in bi_buf, use (valid) bits from bi_buf and
     * (16 - bi_valid) bits from value, leaving (width - (16-bi_valid))
     * unused bits in value.
     */
    if (bi_valid > (int)Buf_size - length) {
        bi_buf |= (value << bi_valid);
        PUTSHORT(bi_buf);
        bi_buf = (ush)value >> (Buf_size - bi_valid);
        bi_valid += length - Buf_size;
    } else {
        bi_buf |= value << bi_valid;
        bi_valid += length;
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
local void R__flush_outbuf(unsigned w, unsigned bytes)
    /* unsigned w;      value to flush */
    /* unsigned bytes;  number of bytes to flush (0, 1 or 2) */
{
    R__error("output buffer too small for in-memory compression");

    /* Encrypt and write the output buffer: */
    out_offset = 0;
    if (bytes == 2) {
        PUTSHORT(w);
    } else if (bytes == 1) {
        out_buf[out_offset++] = (char) (w & 0xff);
    }
}

/* ===========================================================================
 * Write out any remaining bits in an incomplete byte.
 */
void R__bi_windup()
{
    if (bi_valid > 8) {
        PUTSHORT(bi_buf);
    } else if (bi_valid > 0) {
        PUTBYTE(bi_buf);
    }
    if (zfile != (FILE *) NULL) {
        R__flush_outbuf(0, 0);
    }
    bi_buf = 0;
    bi_valid = 0;
#ifdef DEBUG
    R__bits_sent = (R__bits_sent+7) & ~7;
#endif
}

/* ===========================================================================
 * Copy a stored block to the zip file, storing first the length and its
 * one's complement if requested.
 */
void R__copy_block(char far *buf, unsigned len, int header)
    /* char far *buf;  the input data */
    /* unsigned len;   its length */
    /* int header;     true if block header must be written */
{
    R__bi_windup();              /* align on byte boundary */

    if (header) {
        PUTSHORT((ush)len);
        PUTSHORT((ush)~len);
#ifdef DEBUG
        R__bits_sent += 2*16;
#endif
    }
    if (out_offset + len > out_size) {
        R__error("output buffer too small for in-memory compression");
        if (verbose) fprintf(stderr, "R__zip: out_offset=%d, len=%d, out_size=%d\n",out_offset,len,out_size);
    } else {
        memcpy(out_buf + out_offset, buf, len);
        out_offset += len;
    }
#ifdef DEBUG
    R__bits_sent += (ulg)len<<3;
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

    if (tgtsize <= 6L) R__error("target buffer too small");
#if 0
    crc = updcrc((char *)NULL, 0);
    crc = updcrc(src, (extent) srcsize);
#endif
    R__read_buf  = R__mem_read;
    in_buf    = src;
    in_size   = (unsigned)srcsize;
    in_offset = 0;

    out_buf    = tgt;
    out_size   = (unsigned)tgtsize;
    out_offset = 2 + 4;
    R__window_size = 0L;

    R__bi_init((FILE *)NULL);
    R__ct_init(&att, &method);
    R__lm_init((level != 0 ? level : 1), &flags);
    R__Deflate();
    R__window_size = 0L; /* was updated by lm_init() */

    /* For portability, force little-endian order on all machines: */
    tgt[0] = (char)(method & 0xff);
    tgt[1] = (char)((method >> 8) & 0xff);
    tgt[2] = (char)(crc & 0xff);
    tgt[3] = (char)((crc >> 8) & 0xff);
    tgt[4] = (char)((crc >> 16) & 0xff);
    tgt[5] = (char)((crc >> 24) & 0xff);

    return (ulg)out_offset;
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
local int R__mem_read(char *b, unsigned bsize)
{
    if (in_offset < in_size) {
        ulg block_size = in_size - in_offset;
        if (block_size > (ulg)bsize) block_size = (ulg)bsize;
        memcpy(b, in_buf + in_offset, (unsigned)block_size);
        in_offset += (unsigned)block_size;
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
static  int error_flag;

void R__zipMultipleAlgorithm(int cxlevel, int *srcsize, char *src, int *tgtsize, char *tgt, int *irep, int compressionAlgorithm)
     /* int cxlevel;                      compression level */
     /* int  *srcsize, *tgtsize, *irep;   source and target sizes, replay */
     /* char *tgt, *src;                  source and target buffers */
     /* compressionAlgorithm 0 = use global setting */
     /*                      1 = zlib */
     /*                      2 = lzma */
     /*                      3 = old */
{
  if (cxlevel <= 0) {
    *irep = 0;
    return;
  }

  int err;
  int method   = Z_DEFLATED;

  if (compressionAlgorithm == 0) {
    compressionAlgorithm = R__ZipMode;
  }

  // The LZMA compression algorithm from the XZ package
  if (compressionAlgorithm == 2) {
#ifdef R__HAS_LZMACOMPRESSION
    R__zipLZMA(cxlevel, srcsize, src, tgtsize, tgt, irep);
    return;
#endif
#ifndef R__HAS_LZMACOMPRESSION
    compressionAlgorithm = 1;
    static int warningGiven = 0;
    if (warningGiven == 0) {
      warningGiven = 1;
      fprintf(stderr,"Warning R__zipMultipleAlgorithm:\n"
              "There was a request to compress data using the LZMA\n"
              "compression algorithm. But either the LZMA compression\n"
              "libraries were not evailable when this root installation\n"
              "was configured or LZMA compression was explicitly disabled.\n"
              "ZLIB compression will be used instead. This warning will only\n"
              "be given once per process\n");
    }
    compressionAlgorithm = 1;
#endif
  }

  // The very old algorithm for backward compatibility
  // 0 for selecting with R__ZipMode in a backward compatible way
  // 3 for selecting in other cases
  if (compressionAlgorithm == 3 || compressionAlgorithm == 0) {
    ush att      = (ush)UNKNOWN;
    ush flags    = 0;
    if (cxlevel > 9) cxlevel = 9;
    level        = cxlevel;

    *irep        = 0;
    error_flag   = 0;
    if (*tgtsize <= 0) R__error("target buffer too small");
    if (error_flag != 0) return;
    if (*srcsize > 0xffffff) R__error("source buffer too big");
    if (error_flag != 0) return;

    R__read_buf  = R__mem_read;
    in_buf    = src;
    in_size   = (unsigned) (*srcsize);
    in_offset = 0;

    out_buf     = tgt;
    out_size    = (unsigned) (*tgtsize);
    out_offset  = HDRSIZE;
    R__window_size = 0L;

    R__bi_init((FILE *)NULL);      /* initialize bit routines */
    if (error_flag != 0) return;
    R__ct_init(&att, &method);     /* initialize tree routines */
    if (error_flag != 0) return;
    R__lm_init(level, &flags);     /* initialize compression */
    if (error_flag != 0) return;
    R__Deflate();                  /* compress data */
    if (error_flag != 0) return;

    tgt[0] = 'C';               /* Signature 'C'-Chernyaev, 'S'-Smirnov */
    tgt[1] = 'S';
    tgt[2] = (char) method;

    out_size  = out_offset - HDRSIZE;         /* compressed size */
    tgt[3] = (char)(out_size & 0xff);
    tgt[4] = (char)((out_size >> 8) & 0xff);
    tgt[5] = (char)((out_size >> 16) & 0xff);

    tgt[6] = (char)(in_size & 0xff);         /* decompressed size */
    tgt[7] = (char)((in_size >> 8) & 0xff);
    tgt[8] = (char)((in_size >> 16) & 0xff);

    *irep     = out_offset;
    return;

  // 1 is for ZLIB (which is the default), ZLIB is also used for any illegal
  // algorithm setting
  } else {

    z_stream stream;
    *irep = 0;

    error_flag   = 0;
    if (*tgtsize <= 0) R__error("target buffer too small");
    if (error_flag != 0) return;
    if (*srcsize > 0xffffff) R__error("source buffer too big");
    if (error_flag != 0) return;


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

    in_size   = (unsigned) (*srcsize);
    out_size  = stream.total_out;             /* compressed size */
    tgt[3] = (char)(out_size & 0xff);
    tgt[4] = (char)((out_size >> 8) & 0xff);
    tgt[5] = (char)((out_size >> 16) & 0xff);

    tgt[6] = (char)(in_size & 0xff);         /* decompressed size */
    tgt[7] = (char)((in_size >> 8) & 0xff);
    tgt[8] = (char)((in_size >> 16) & 0xff);

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
  error_flag = 1;
}
