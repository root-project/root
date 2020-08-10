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
 *      -80   : Input size not a multiple of 8.
 *      -81   : block_size not multiple of 8.
 *      -91   : Decompression error, wrong number of bytes processed.
 *      -1YYY : Error internal to compression routine with error code -YYY.
 */


#ifndef BITSHUFFLE_H
#define BITSHUFFLE_H

#include <stdlib.h>
#include "Bitshuffle_core.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- bshuf_compress_lz4_bound ----
 *
 * Bound on size of data compressed with *bshuf_compress_lz4*.
 *
 * Parameters
 * ----------
 *  size : number of elements in input
 *  elem_size : element size of typed data
 *  block_size : Process in blocks of this many elements. Pass 0 to
 *  select automatically (recommended).
 *
 * Returns
 * -------
 *  Bound on compressed data size.
 *
 */
size_t bshuf_compress_lz4_bound(const size_t size,
        const size_t elem_size, size_t block_size);


/* ---- bshuf_compress_lz4 ----
 *
 * Bitshuffled and compress the data using LZ4.
 *
 * Transpose within elements, in blocks of data of *block_size* elements then
 * compress the blocks using LZ4.  In the output buffer, each block is prefixed
 * by a 4 byte integer giving the compressed size of that block.
 *
 * Output buffer must be large enough to hold the compressed data.  This could
 * be in principle substantially larger than the input buffer.  Use the routine
 * *bshuf_compress_lz4_bound* to get an upper limit.
 *
 * Parameters
 * ----------
 *  in : input buffer, must be of size * elem_size bytes
 *  out : output buffer, must be large enough to hold data.
 *  size : number of elements in input
 *  elem_size : element size of typed data
 *  block_size : Process in blocks of this many elements. Pass 0 to
 *  select automatically (recommended).
 *
 * Returns
 * -------
 *  number of bytes used in output buffer, negative error-code if failed.
 *
 */
int64_t bshuf_compress_lz4(const void* in, void* out, const size_t size, const size_t
        elem_size, size_t block_size);


/* ---- bshuf_decompress_lz4 ----
 *
 * Undo compression and bitshuffling.
 *
 * Decompress data then un-bitshuffle it in blocks of *block_size* elements.
 *
 * To properly unshuffle bitshuffled data, *size*, *elem_size* and *block_size*
 * must patch the parameters used to compress the data.
 *
 *
 * Parameters
 * ----------
 *  in : input buffer
 *  out : output buffer, must be of size * elem_size bytes
 *  size : number of elements in input
 *  elem_size : element size of typed data
 *  block_size : Process in blocks of this many elements. Pass 0 to
 *  select automatically (recommended).
 *
 * Returns
 * -------
 *  number of bytes consumed in *input* buffer, negative error-code if failed.
 *
 */
int64_t bshuf_decompress_lz4(const void* in, void* out, const size_t size,
        const size_t elem_size, size_t block_size);

#ifdef __cplusplus
} // extern "C"
#endif

#endif  // BITSHUFFLE_H
