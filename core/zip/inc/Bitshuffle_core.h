/*
 * Bitshuffle - Filter for improving compression of typed binary data.
 *
 * This file is part of Bitshuffle
 * Author: Kiyoshi Masui <kiyo@physics.ubc.ca>
 * Website: http://www.github.com/kiyo-masui/bitshuffle
 * Created: 2014
 *
 * See LICENSE file for details about copyright and rights to use.
 *
 *
 * Header File
 *
 * Worker routines return an int64_t which is the number of bytes processed
 * if positive or an error code if negative.
 *
 * Error codes:
 *      -1    : Failed to allocate memory.
 *      -11   : Missing SSE.
 *      -12   : Missing AVX.
 *      -13   : Missing Arm Neon.
 *      -80   : Input size not a multiple of 8.
 *      -81   : block_size not multiple of 8.
 *      -91   : Decompression error, wrong number of bytes processed.
 *      -1YYY : Error internal to compression routine with error code -YYY.
 */


#ifndef BITSHUFFLE_CORE_H
#define BITSHUFFLE_CORE_H

// We assume GNU g++ defining `__cplusplus` has stdint.h
#if (defined (__STDC_VERSION__) && __STDC_VERSION__ >= 199900L) || defined(__cplusplus)
#include <stdint.h>
#else
  typedef unsigned char       uint8_t;
  typedef unsigned short      uint16_t;
  typedef unsigned int        uint32_t;
  typedef   signed int        int32_t;
  typedef unsigned long long  uint64_t;
  typedef long long           int64_t;
#endif

#include <stdlib.h>


// These are usually set in the setup.py.
#ifndef BSHUF_VERSION_MAJOR
#define BSHUF_VERSION_MAJOR 0
#define BSHUF_VERSION_MINOR 3
#define BSHUF_VERSION_POINT 5
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --- bshuf_using_SSE2 ----
 *
 * Whether routines where compiled with the SSE2 instruction set.
 *
 * Returns
 * -------
 *  1 if using SSE2, 0 otherwise.
 *
 */
int bshuf_using_SSE2(void);


/* ---- bshuf_using_AVX2 ----
 *
 * Whether routines where compiled with the AVX2 instruction set.
 *
 * Returns
 * -------
 *  1 if using AVX2, 0 otherwise.
 *
 */
int bshuf_using_AVX2(void);


/* ---- bshuf_default_block_size ----
 *
 * The default block size as function of element size.
 *
 * This is the block size used by the blocked routines (any routine
 * taking a *block_size* argument) when the block_size is not provided
 * (zero is passed).
 *
 * The results of this routine are guaranteed to be stable such that
 * shuffled/compressed data can always be decompressed.
 *
 * Parameters
 * ----------
 *  elem_size : element size of data to be shuffled/compressed.
 *
 */
size_t bshuf_default_block_size(const size_t elem_size);


/* ---- bshuf_bitshuffle ----
 *
 * Bitshuffle the data.
 *
 * Transpose the bits within elements, in blocks of *block_size*
 * elements.
 *
 * Parameters
 * ----------
 *  in : input buffer, must be of size * elem_size bytes
 *  out : output buffer, must be of size * elem_size bytes
 *  size : number of elements in input
 *  elem_size : element size of typed data
 *  block_size : Do transpose in blocks of this many elements. Pass 0 to
 *  select automatically (recommended).
 *
 * Returns
 * -------
 *  number of bytes processed, negative error-code if failed.
 *
 */
int64_t bshuf_bitshuffle(const void* in, void* out, const size_t size,
        const size_t elem_size, size_t block_size);


/* ---- bshuf_bitunshuffle ----
 *
 * Unshuffle bitshuffled data.
 *
 * Untranspose the bits within elements, in blocks of *block_size*
 * elements.
 *
 * To properly unshuffle bitshuffled data, *size*, *elem_size* and *block_size*
 * must match the parameters used to shuffle the data.
 *
 * Parameters
 * ----------
 *  in : input buffer, must be of size * elem_size bytes
 *  out : output buffer, must be of size * elem_size bytes
 *  size : number of elements in input
 *  elem_size : element size of typed data
 *  block_size : Do transpose in blocks of this many elements. Pass 0 to
 *  select automatically (recommended).
 *
 * Returns
 * -------
 *  number of bytes processed, negative error-code if failed.
 *
 */
int64_t bshuf_bitunshuffle(const void* in, void* out, const size_t size,
        const size_t elem_size, size_t block_size);

#ifdef __cplusplus
} // extern "C"
#endif

#endif  // BITSHUFFLE_CORE_H
