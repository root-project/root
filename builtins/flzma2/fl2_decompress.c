/*
* Copyright (c) 2018, Conor McCarthy
* All rights reserved.
* Parts based on zstd_decompress.c copyright Yann Collet
*
* This source code is licensed under both the BSD-style license (found in the
* LICENSE file in the root directory of this source tree) and the GPLv2 (found
* in the COPYING file in the root directory of this source tree).
* You may select, at your option, one of the above-listed licenses.
*/

#include <string.h>
#include "fast-lzma2.h"
#include "fl2_errors.h"
#include "fl2_internal.h"
#include "mem.h"
#include "util.h"
#include "lzma2_dec.h"
#include "fl2_threading.h"
#include "fl2_pool.h"
#include "atomic.h"
#ifndef NO_XXHASH
#  include "xxhash.h"
#endif


#define LZMA2_PROP_UNINITIALIZED 0xFF


FL2LIB_API unsigned long long FL2LIB_CALL FL2_findDecompressedSize(const void *src, size_t srcSize)
{
    return LZMA2_getUnpackSize(src, srcSize);
}

FL2LIB_API size_t FL2LIB_CALL FL2_getDictSizeFromProp(unsigned char prop)
{
    return LZMA2_getDictSizeFromProp(prop);
}

typedef struct
{
    LZMA2_DCtx* dec;
    const void *src;
    size_t packPos;
    size_t packSize;
    size_t unpackPos;
    size_t unpackSize;
    size_t res;
    LZMA2_finishMode finish;
} FL2_blockDecMt;

struct FL2_DCtx_s
{
    LZMA2_DCtx dec;
#ifndef FL2_SINGLETHREAD
    FL2_blockDecMt *blocks;
    FL2POOL_ctx *factory;
    size_t nbThreads;
#endif
    BYTE lzma2prop;
};

FL2LIB_API size_t FL2LIB_CALL FL2_decompress(void* dst, size_t dstCapacity,
    const void* src, size_t compressedSize)
{
    return FL2_decompressMt(dst, dstCapacity, src, compressedSize, 1);
}

FL2LIB_API size_t FL2LIB_CALL FL2_decompressMt(void* dst, size_t dstCapacity,
    const void* src, size_t compressedSize,
    unsigned nbThreads)
{
    FL2_DCtx* const dctx = FL2_createDCtxMt(nbThreads);
    if(dctx == NULL)
        return FL2_ERROR(memory_allocation);

    size_t const dSize = FL2_decompressDCtx(dctx,
        dst, dstCapacity,
        src, compressedSize);

    FL2_freeDCtx(dctx);

    return dSize;
}

FL2LIB_API FL2_DCtx* FL2LIB_CALL FL2_createDCtx(void)
{
    return FL2_createDCtxMt(1);
}

FL2LIB_API FL2_DCtx *FL2LIB_CALL FL2_createDCtxMt(unsigned nbThreads)
{
    DEBUGLOG(3, "FL2_createDCtx");

    FL2_DCtx* const dctx = malloc(sizeof(FL2_DCtx));

    if (dctx == NULL)
        return NULL;

    LZMA_constructDCtx(&dctx->dec);

    dctx->lzma2prop = LZMA2_PROP_UNINITIALIZED;

    nbThreads = FL2_checkNbThreads(nbThreads);

#ifndef FL2_SINGLETHREAD
    dctx->nbThreads = 1;
    dctx->blocks = NULL;
    dctx->factory = NULL;

    if (nbThreads > 1) {
        dctx->blocks = malloc(nbThreads * sizeof(FL2_blockDecMt));
        dctx->factory = FL2POOL_create(nbThreads - 1);

        if (dctx->blocks == NULL || dctx->factory == NULL) {
            FL2_freeDCtx(dctx);
            return NULL;
        }
        dctx->blocks[0].dec = &dctx->dec;

        for (; dctx->nbThreads < nbThreads; ++dctx->nbThreads) {

            dctx->blocks[dctx->nbThreads].dec = malloc(sizeof(LZMA2_DCtx));

            if (dctx->blocks[dctx->nbThreads].dec == NULL) {
                FL2_freeDCtx(dctx);
                return NULL;
            }
            LZMA_constructDCtx(dctx->blocks[dctx->nbThreads].dec);
        }
    }
#endif

    return dctx;
}

FL2LIB_API size_t FL2LIB_CALL FL2_freeDCtx(FL2_DCtx* dctx)
{
    if (dctx == NULL)
        return FL2_error_no_error;

    DEBUGLOG(3, "FL2_freeDCtx");

    LZMA_destructDCtx(&dctx->dec);

#ifndef FL2_SINGLETHREAD
    if (dctx->blocks != NULL) {
        for (unsigned thread = 1; thread < dctx->nbThreads; ++thread) {
            LZMA_destructDCtx(dctx->blocks[thread].dec);
            free(dctx->blocks[thread].dec);
        }
        free(dctx->blocks);
    }
    FL2POOL_free(dctx->factory);
#endif
    free(dctx);

    return FL2_error_no_error;
}

#ifndef FL2_SINGLETHREAD

FL2LIB_API unsigned FL2LIB_CALL FL2_getDCtxThreadCount(const FL2_DCtx * dctx)
{
    return (unsigned)dctx->nbThreads;
}

/* FL2_decompressCtxBlock() : FL2POOL_function type */
static void FL2_decompressCtxBlock(void* const jobDescription, ptrdiff_t const n)
{
    FL2_blockDecMt* const blocks = (FL2_blockDecMt*)jobDescription;
    size_t srcLen = blocks[n].packSize;

    DEBUGLOG(4, "Thread %u: decoding block of input size %u, output size %u", (unsigned)n, (unsigned)srcLen, (unsigned)blocks[n].unpackSize);

    blocks[n].res = LZMA2_decodeToDic(blocks[n].dec, blocks[n].unpackSize, blocks[n].src, &srcLen, blocks[n].finish);

    /* If no error occurred, store into res the dic_pos value, which is the end of the decompressed data in the buffer */
    if (!FL2_isError(blocks[n].res))
        blocks[n].res = blocks[n].dec->dic_pos;
}

static size_t FL2_decompressCtxBlocksMt(FL2_DCtx* const dctx, const BYTE *const src, BYTE *const dst, size_t const dstCapacity, size_t const nbThreads)
{
    FL2_blockDecMt* const blocks = dctx->blocks;

    /* Initial check for block 0. The others are uncalculated */
    if (dstCapacity < blocks[0].unpackSize)
        return FL2_ERROR(dstSize_tooSmall);

    blocks[0].packPos = 0;
    blocks[0].unpackPos = 0;
    blocks[0].src = src;

    BYTE const prop = dctx->lzma2prop & FL2_LZMA_PROP_MASK;

    for (size_t thread = 1; thread < nbThreads; ++thread) {
        blocks[thread].packPos = blocks[thread - 1].packPos + blocks[thread - 1].packSize;
        blocks[thread].unpackPos = blocks[thread - 1].unpackPos + blocks[thread - 1].unpackSize;
        blocks[thread].src = src + blocks[thread].packPos;
        CHECK_F(LZMA2_initDecoder(blocks[thread].dec, prop, dst + blocks[thread].unpackPos, blocks[thread].unpackSize));
    }
    if (dstCapacity < blocks[nbThreads - 1].unpackPos + blocks[nbThreads - 1].unpackSize)
        return FL2_ERROR(dstSize_tooSmall);

    /* Decompress thread 1..n */
    FL2POOL_addRange(dctx->factory, FL2_decompressCtxBlock, blocks, 1, nbThreads);

    /* Decompress thread 0 */
    CHECK_F(LZMA2_initDecoder(blocks[0].dec, prop, dst + blocks[0].unpackPos, blocks[0].unpackSize));
    FL2_decompressCtxBlock(blocks, 0);

    FL2POOL_waitAll(dctx->factory, 0);

    size_t dSize = 0;
    for (size_t thread = 0; thread < nbThreads; ++thread) {
        if (FL2_isError(blocks[thread].res))
            return blocks[thread].res;
        dSize += blocks[thread].res;
    }
    return dSize;
}

static void FL2_resetMtBlocks(FL2_DCtx* const dctx)
{
    for (size_t thread = 0; thread < dctx->nbThreads; ++thread) {
        dctx->blocks[thread].finish = LZMA_FINISH_ANY;
        dctx->blocks[thread].packSize = 0;
        dctx->blocks[thread].unpackSize = 0;
    }
}

/* Decompress an entire stream stored in memory */
static size_t FL2_decompressDCtxMt(FL2_DCtx* const dctx,
    void* dst, size_t dstCapacity,
    const void* src, size_t *const srcLen)
{
    size_t srcSize = *srcLen;
    *srcLen = 0;

    FL2_resetMtBlocks(dctx);

    size_t unpackSize = 0;
    FL2_blockDecMt* const blocks = dctx->blocks;
    size_t thread = 0;
    size_t pos = 0;
    while (pos < srcSize) {
        LZMA2_chunk inf;
        int type = LZMA2_parseInput(src, pos, srcSize - pos, &inf);

        /* All src data must be in memory so CHUNK_MORE_DATA is an error */
        if (type == CHUNK_ERROR || type == CHUNK_MORE_DATA)
            return FL2_ERROR(corruption_detected);

        /* CHUNK_DICT_RESET is used to signal block completion except for pos 0 */
        if (pos == 0 && type == CHUNK_DICT_RESET)
            type = CHUNK_CONTINUE;

        if (type == CHUNK_DICT_RESET || type == CHUNK_FINAL) {
            if (type == CHUNK_FINAL) {
                /* The finish value will be passed to the decoder */
                blocks[thread].finish = LZMA_FINISH_END;
                /* CHUNK_FINAL means a single 0 byte */
                assert(inf.pack_size == 1);
                ++blocks[thread].packSize;
            }
            /* Move to the next thread. Decoding will begin if all threads are used. */
            ++thread;
        }
        if (type == CHUNK_FINAL || (type == CHUNK_DICT_RESET && thread == dctx->nbThreads)) {
            size_t res = FL2_decompressCtxBlocksMt(dctx, (BYTE*)src, dst, dstCapacity, thread);
            if (FL2_isError(res))
                return res;

            unpackSize += res;
            /* Store the unpack size in decoder 0 where it would be in single thread */
            dctx->dec.dic_pos = unpackSize;
            /* Input used is the end of data consumed by the last thread */
            *srcLen += blocks[thread - 1].packPos + blocks[thread - 1].packSize;

            if (type == CHUNK_FINAL)
                return LZMA_STATUS_FINISHED;

            /* Only excecuted at a dict reset. pos is the location of the reset */
            src = (BYTE*)src + pos;
            srcSize -= pos;
            dst = (BYTE*)dst + res;
            dstCapacity -= res;
            pos = 0;
            thread = 0;

            FL2_resetMtBlocks(dctx);
        }
        else {
            /* Not the end or a dict reset, so add it to the current block */
            blocks[thread].packSize += inf.pack_size;
            blocks[thread].unpackSize += inf.unpack_size;
            pos += inf.pack_size;
        }
    }
    return FL2_ERROR(srcSize_wrong);
}

#endif

FL2LIB_API size_t FL2LIB_CALL FL2_initDCtx(FL2_DCtx * dctx, unsigned char prop)
{
    if((prop & FL2_LZMA_PROP_MASK) > 40)
        return FL2_ERROR(corruption_detected);

    dctx->lzma2prop = prop;
    return FL2_error_no_error;
}

FL2LIB_API size_t FL2LIB_CALL FL2_decompressDCtx(FL2_DCtx* dctx,
    void* dst, size_t dstCapacity,
    const void* src, size_t srcSize)
{
    BYTE prop = dctx->lzma2prop;
    const BYTE *srcBuf = src;

    if (prop == LZMA2_PROP_UNINITIALIZED) {
        prop = *(const BYTE*)src;
        ++srcBuf;
        --srcSize;
    }

#ifndef NO_XXHASH
    BYTE const doHash = prop >> FL2_PROP_HASH_BIT;
#endif
    prop &= FL2_LZMA_PROP_MASK;

    DEBUGLOG(4, "FL2_decompressDCtx : dict prop 0x%X, do hash %u", prop, doHash);

    size_t srcPos = srcSize;

    size_t dicPos = 0;
    size_t res;
#ifndef FL2_SINGLETHREAD
    if (dctx->blocks != NULL) {
        dctx->lzma2prop = prop;
        res = FL2_decompressDCtxMt(dctx, dst, dstCapacity, srcBuf, &srcPos);
    }
    else 
#endif
    {
        CHECK_F(LZMA2_initDecoder(&dctx->dec, prop, dst, dstCapacity));

        dicPos = dctx->dec.dic_pos;

        res = LZMA2_decodeToDic(&dctx->dec, dstCapacity, srcBuf, &srcPos, LZMA_FINISH_END);
    }

    dctx->lzma2prop = LZMA2_PROP_UNINITIALIZED;

    if (FL2_isError(res))
        return res;
    /* All src data must be in memory */
    if (res == LZMA_STATUS_NEEDS_MORE_INPUT)
        return FL2_ERROR(srcSize_wrong);

    dicPos = dctx->dec.dic_pos - dicPos;

#ifndef NO_XXHASH
    if (doHash) {
        XXH32_canonical_t canonical;
        U32 hash;

        DEBUGLOG(4, "Checking hash");

        if (srcSize - srcPos < XXHASH_SIZEOF)
            return FL2_ERROR(srcSize_wrong);

        memcpy(&canonical, srcBuf + srcPos, XXHASH_SIZEOF);
        hash = XXH32_hashFromCanonical(&canonical);
        if (hash != XXH32(dst, dicPos, 0))
            return FL2_ERROR(checksum_wrong);
    }
#endif
    return dicPos;
}

/*===== Streaming decompression functions =====*/

typedef enum
{
    FL2DEC_STAGE_INIT,
    FL2DEC_STAGE_DECOMP,
#ifndef FL2_SINGLETHREAD
    FL2DEC_STAGE_MT_WRITE,
#endif
    FL2DEC_STAGE_HASH,
    FL2DEC_STAGE_FINISHED
} FL2_decStage;

#ifndef FL2_SINGLETHREAD
typedef struct FL2_decInbuf_s FL2_decInbuf;

struct FL2_decInbuf_s
{
    FL2_decInbuf *next;
    size_t length;
    BYTE inBuf[1];
};

typedef struct
{
    FL2_decInbuf *first;
    FL2_decInbuf *last;
    size_t startPos;
    size_t endPos;
    size_t unpackSize;
} FL2_decBlock;

typedef struct
{
    LZMA2_DCtx dec;
    FL2_decBlock inBlock;
    BYTE *outBuf;
    size_t bufSize;
    size_t res;
} FL2_decJob;

typedef struct
{
    FL2POOL_ctx* factory;
    FL2_decInbuf *head;
    FL2_decInbuf *cur;
    size_t curPos;
    size_t numThreads;
    size_t maxThreads;
    size_t srcThread;
    size_t srcPos;
    size_t memTotal;
    size_t memLimit;
    BYTE isFinal;
    BYTE failState;
    BYTE canceled;
    BYTE prop;
#ifndef NO_XXHASH
    XXH32_canonical_t hash;
#endif
    FL2_decJob threads[1];
} FL2_decMt;
#endif

#define LZMA_OVERLAP_SIZE (LZMA_REQUIRED_INPUT_MAX * 2)

struct FL2_DStream_s
{
#ifndef FL2_SINGLETHREAD
    FL2_decMt *decmt;
    FL2POOL_ctx* decompressThread;
#endif
    LZMA2_DCtx dec;
    FL2_outBuffer* asyncOutput;
    FL2_inBuffer* asyncInput;
    size_t asyncRes;
    U64 streamTotal;
    size_t overlapSize;
    FL2_atomic progress;
    unsigned timeout;
#ifndef NO_XXHASH
    XXH32_state_t *xxh;
    XXH32_canonical_t xxhIn;
    size_t xxhPos;
#endif
    FL2_decStage stage;
    BYTE doHash;
    BYTE loopCount;
    BYTE wait;
    BYTE overlap[LZMA_OVERLAP_SIZE];
};

static size_t FL2_decompressInput(FL2_DStream* fds, FL2_outBuffer* output, FL2_inBuffer* input)
{
    if (fds->stage == FL2DEC_STAGE_DECOMP) {
        size_t destSize = output->size - output->pos;
        size_t srcSize = input->size - input->pos;
        size_t const res = LZMA2_decodeToBuf(&fds->dec, (BYTE*)output->dst + output->pos, &destSize, (const BYTE*)input->src + input->pos, &srcSize, LZMA_FINISH_ANY);

        DEBUGLOG(5, "Decoded %u bytes", (U32)destSize);

#ifndef NO_XXHASH
        if (fds->doHash)
            XXH32_update(fds->xxh, (BYTE*)output->dst + output->pos, destSize);
#endif
        FL2_atomic_add(fds->progress, (long)destSize);

        output->pos += destSize;
        input->pos += srcSize;

        if (FL2_isError(res))
            return res;
        if (res == LZMA_STATUS_FINISHED) {
            DEBUGLOG(4, "Found end mark");
            fds->stage = fds->doHash ? FL2DEC_STAGE_HASH : FL2DEC_STAGE_FINISHED;
        }
    }
    return FL2_error_no_error;
}

static size_t FL2_decompressOverlappedInput(FL2_DStream* fds, FL2_outBuffer* output, FL2_inBuffer* input)
{
    if (fds->overlapSize != 0) {
        size_t toRead = MIN(input->size - input->pos, LZMA_OVERLAP_SIZE - fds->overlapSize);
        memcpy(fds->overlap + fds->overlapSize, (BYTE*)input->src + input->pos, toRead);
        FL2_inBuffer temp = { fds->overlap, fds->overlapSize + toRead, 0 };
        CHECK_F(FL2_decompressInput(fds, output, &temp));
        if (temp.pos >= fds->overlapSize) {
            input->pos += temp.pos - fds->overlapSize;
            fds->overlapSize = 0;
        }
        else {
            fds->overlapSize -= temp.pos;
            memmove(fds->overlap, fds->overlap + temp.pos, fds->overlapSize);
        }
    }
    if(input->pos == input->size)
        return FL2_error_no_error;

    if(fds->overlapSize == 0)
        CHECK_F(FL2_decompressInput(fds, output, input));

    size_t toRead = input->size - input->pos;
    /* More input needed if not finished, output not full and input is below minimum.
     * Safe to take all input because stream will be beyond decomp stage if the terminator is present. */
    if (fds->stage == FL2DEC_STAGE_DECOMP && output->pos < output->size && toRead <= LZMA_REQUIRED_INPUT_MAX) {
        toRead = MIN(toRead, LZMA_OVERLAP_SIZE - fds->overlapSize);
        memcpy(fds->overlap + fds->overlapSize, (BYTE*)input->src + input->pos, toRead);
        input->pos += toRead;
        fds->overlapSize += toRead;
    }
    return FL2_error_no_error;
}

#ifndef FL2_SINGLETHREAD

/* Free buffer nodes from node to the end, except keep */
static void LZMA2_freeInbufNodeChain(FL2_decMt *const decmt, FL2_decInbuf *node, FL2_decInbuf *const keep)
{
    while (node) {
        FL2_decInbuf *const next = node->next;
        if (node != keep) {
            decmt->memTotal -= sizeof(FL2_decInbuf) + LZMA2_MT_INPUT_SIZE - 1;
            free(node);
        }
        else {
            node->next = NULL;
        }
        node = next;
    }
}

/* Free all buffer nodes except the head */
static void LZMA2_freeExtraInbufNodes(FL2_decMt *const decmt)
{
    LZMA2_freeInbufNodeChain(decmt, decmt->head->next, NULL);
    decmt->head->next = NULL;
    decmt->head->length = 0;
}

static void FL2_freeOutputBuffers(FL2_decMt *const decmt)
{
    for (size_t thread = 0; thread < decmt->maxThreads; ++thread)
        if(decmt->threads[thread].outBuf != NULL) {
            decmt->memTotal -= decmt->threads[thread].bufSize;
            free(decmt->threads[thread].outBuf);
            decmt->threads[thread].outBuf = NULL;
        }
    decmt->numThreads = 0;
}

static void FL2_lzma2DecMt_cleanup(FL2_decMt *const decmt)
{
    if (decmt) {
        FL2_freeOutputBuffers(decmt);
        LZMA2_freeExtraInbufNodes(decmt);
    }
}

static void FL2_lzma2DecMt_free(FL2_decMt *const decmt)
{
    if (decmt) {
        FL2_freeOutputBuffers(decmt);
        LZMA2_freeInbufNodeChain(decmt, decmt->head, NULL);
        FL2POOL_free(decmt->factory);
        free(decmt);
    }
}

static void FL2_lzma2DecMt_init(FL2_decMt *const decmt)
{
    if (decmt) {
        decmt->cur = NULL;
        decmt->failState = 0;
        decmt->isFinal = 0;
        decmt->canceled = 0;
        decmt->memTotal = 0;
        FL2_freeOutputBuffers(decmt);
        LZMA2_freeExtraInbufNodes(decmt);
        decmt->threads[0].inBlock.first = decmt->head;
        decmt->threads[0].inBlock.last = decmt->head;
        decmt->threads[0].inBlock.startPos = 0;
        decmt->threads[0].inBlock.endPos = 0;
        decmt->threads[0].inBlock.unpackSize = 0;
    }
}

static int FL2_lzma2DecMt_initProp(FL2_decMt *const decmt, BYTE prop)
{
    decmt->prop = prop;
    size_t const dictSize = LZMA2_getDictSizeFromProp(prop);
    /* Minimum memory is for two threads, one dict size per thread plus a minimal amount of
     * compressed data for each. Compression to < 1/6 is uncommon. */
    if (decmt->memLimit < (dictSize + dictSize / 6U) * 2U) {
        DEBUGLOG(3, "Using ST decompression due to dict size %u, memory limit %u", (unsigned)dictSize, (unsigned)decmt->memLimit);
        decmt->failState = 1;
        return 1;
    }
    return 0;
}

static FL2_decInbuf * FL2_createInbufNode(FL2_decMt *const decmt, FL2_decInbuf *const prev)
{
    decmt->memTotal += sizeof(FL2_decInbuf) + LZMA2_MT_INPUT_SIZE - 1;
    if (decmt->memTotal > decmt->memLimit)
        return NULL;

    FL2_decInbuf *const node = malloc(sizeof(FL2_decInbuf) + LZMA2_MT_INPUT_SIZE - 1);
    if (node == NULL)
        return NULL;

    node->next = NULL;
    node->length = 0;
    if (prev) {
        /* Node buffers overlap by LZMA_REQUIRED_INPUT_MAX */
        memcpy(node->inBuf, prev->inBuf + prev->length - LZMA_REQUIRED_INPUT_MAX, LZMA_REQUIRED_INPUT_MAX);
        prev->next = node;
        node->length = LZMA_REQUIRED_INPUT_MAX;
    }
    return node;
}

static FL2_decMt *FL2_lzma2DecMt_create(unsigned maxThreads)
{
    maxThreads += !maxThreads;

    FL2_decMt *const decmt = malloc(sizeof(FL2_decMt) + (maxThreads - 1) * sizeof(FL2_decJob));
    if (decmt == NULL)
        return NULL;

    decmt->memTotal = 0;
    decmt->memLimit = (size_t)1 << 29;
    decmt->maxThreads = 0;

    /* The head always exists and is only freed on deallocation */
    decmt->head = FL2_createInbufNode(decmt, NULL);
    if (decmt->head == NULL) {
        free(decmt);
        return NULL;
    }

    decmt->factory = FL2POOL_create(maxThreads - 1);

    if (maxThreads > 1 && decmt->factory == NULL) {
        FL2_lzma2DecMt_free(decmt);
        return NULL;
    }
    decmt->numThreads = 0;
    decmt->maxThreads = maxThreads;

    for (size_t n = 0; n < maxThreads; ++n) {
        decmt->threads[n].outBuf = NULL;
        LZMA_constructDCtx(&decmt->threads[n].dec);
    }
    FL2_lzma2DecMt_init(decmt);

    return decmt;
}

/* Read chunk headers and advance inBlock->endPos to the next chunk
 * until it points beyond the available data.
 * Add the size of each chunk to inBlock->unpackSize
 */
static LZMA2_parseRes FL2_parseMt(FL2_decBlock* const inBlock)
{
    LZMA2_parseRes res = CHUNK_MORE_DATA;
    FL2_decInbuf *const cur = inBlock->last;
    if (cur == NULL)
        return res;

    int first = inBlock->unpackSize == 0;

    while (inBlock->endPos < cur->length) {
        LZMA2_chunk inf;
        res = LZMA2_parseInput(cur->inBuf, inBlock->endPos, cur->length - inBlock->endPos, &inf);
        if (first && res == CHUNK_DICT_RESET)
            res = CHUNK_CONTINUE;
        if (res != CHUNK_CONTINUE)
            break;

        inBlock->endPos += inf.pack_size;
        inBlock->unpackSize += inf.unpack_size;

        first = 0;
    }
    /* Skip the 1-byte end marker if found */
    inBlock->endPos += (res == CHUNK_FINAL);
    return res;
}

/* Decompress an entire block starting with a dict reset and ending with
 * the last chunk before the next dict reset, or the terminator.
 * The input is a chain of buffers.
 */
static size_t FL2_decompressBlockMt(FL2_DStream* const fds, size_t const thread)
{
    FL2_decMt *const decmt = fds->decmt;
    FL2_decJob *const ti = &decmt->threads[thread];
    LZMA2_DCtx *const dec = &ti->dec;

    DEBUGLOG(4, "Thread %u: decoding block of size %u", (unsigned)thread, (unsigned)ti->bufSize);

    CHECK_F(LZMA2_initDecoder(dec, decmt->prop, ti->outBuf, ti->bufSize));

    /* Input buffer node containing the starting chunk. If thread > 0 this is usually
     * the last input buffer node of the previous thread. */
    FL2_decInbuf *cur = ti->inBlock.first;
    /* Position of the starting chunk. */
    size_t inPos = ti->inBlock.startPos;
    /* Flag to indicate this block ends with the terminator */
    BYTE const last = decmt->isFinal && (thread == decmt->numThreads - 1);

    while (!decmt->canceled) {
        size_t srcSize = cur->length - inPos;
        size_t const dicPos = dec->dic_pos;

        size_t const res = LZMA2_decodeToDic(dec,
            ti->bufSize,
            cur->inBuf + inPos, &srcSize,
            last && cur == ti->inBlock.last ? LZMA_FINISH_END : LZMA_FINISH_ANY);

        CHECK_F(res);

        FL2_atomic_add(fds->progress, (long)(dec->dic_pos - dicPos));

        if (res == LZMA_STATUS_FINISHED)
            DEBUGLOG(4, "Found end mark");

        if (cur == ti->inBlock.last)
            break;

        /* Advance the position and switch to the next input buffer in the chain if necessary */
        inPos += srcSize;
        if (inPos + LZMA_REQUIRED_INPUT_MAX >= cur->length) {
            inPos -= cur->length - LZMA_REQUIRED_INPUT_MAX;
            cur = cur->next;
        }
    }

    if (decmt->canceled)
        return FL2_ERROR(canceled);

    return FL2_error_no_error;
}

/*
 * Write the data from the output buffer of each thread.
 */
static size_t FL2_writeStreamBlocks(FL2_DStream* const fds, FL2_outBuffer* const output)
{
    FL2_decMt *const decmt = fds->decmt;

    for (; decmt->srcThread < fds->decmt->numThreads; ++decmt->srcThread) {
        FL2_decJob *thread = decmt->threads + decmt->srcThread;
        size_t to_write = MIN(thread->bufSize - decmt->srcPos, output->size - output->pos);
        memcpy((BYTE*)output->dst + output->pos, thread->outBuf + decmt->srcPos, to_write);

#ifndef NO_XXHASH
        if (fds->doHash)
            XXH32_update(fds->xxh, (BYTE*)output->dst + output->pos, to_write);
#endif
        decmt->srcPos += to_write;
        output->pos += to_write;

        if (decmt->srcPos < thread->bufSize)
            break;

        decmt->srcPos = 0;
    }
    if (decmt->srcThread < fds->decmt->numThreads)
        return 0;

    FL2_freeOutputBuffers(fds->decmt);
    fds->decmt->numThreads = 0;

    return 1;
}

/* FL2_decompressBlock() : FL2POOL_function type */
static void FL2_decompressBlock(void* const jobDescription, ptrdiff_t const n)
{
    FL2_DStream* const fds = (FL2_DStream*)jobDescription;
    fds->decmt->threads[n].res = FL2_decompressBlockMt(fds, n);
}

static size_t FL2_decompressBlocksMt(FL2_DStream* const fds)
{
    /* Set the threads to work on the blocks */
    FL2_decMt * const decmt = fds->decmt;
    FL2POOL_addRange(decmt->factory, FL2_decompressBlock, fds, 1, decmt->numThreads);

    /* Do block 0 in the main thread */
    decmt->threads[0].res = FL2_decompressBlockMt(fds, 0);
    FL2POOL_waitAll(fds->decmt->factory, 0);

    /* Free all input buffers except the last */
    FL2_decInbuf *const keep = decmt->threads[decmt->numThreads - 1].inBlock.last;
    LZMA2_freeInbufNodeChain(decmt, decmt->head, keep);
    /* The last becomes the new head */
    decmt->head = keep;
    decmt->threads[0].inBlock.first = keep;
    decmt->threads[0].inBlock.last = keep;
    /* Initialize the start and end to the next chunk */
    decmt->threads[0].inBlock.endPos = decmt->threads[decmt->numThreads - 1].inBlock.endPos;
    decmt->threads[0].inBlock.startPos = decmt->threads[0].inBlock.endPos;
    decmt->threads[0].inBlock.unpackSize = 0;

    for (size_t thread = 0; thread < decmt->numThreads; ++thread)
        if (FL2_isError(decmt->threads[thread].res))
            return decmt->threads[thread].res;

    decmt->srcThread = 0;
    decmt->srcPos = 0;

    return FL2_error_no_error;
}

static size_t FL2_handleFinalChunkMt(FL2_decMt *const decmt, size_t res)
{
    FL2_decBlock *inBlock = &decmt->threads[decmt->numThreads].inBlock;

    FL2_decJob * const done = decmt->threads + decmt->numThreads;
    ++decmt->numThreads;

    done->bufSize = done->inBlock.unpackSize;
    decmt->memTotal += done->bufSize;
    if (decmt->memTotal > decmt->memLimit)
        return FL2_ERROR(memory_allocation);

    /* Decompressed data will be stored in outBuf */
    done->outBuf = malloc(done->bufSize);
    if (done->outBuf == NULL)
        return FL2_ERROR(memory_allocation);

    decmt->isFinal = (res == CHUNK_FINAL);

    if (decmt->numThreads == decmt->maxThreads || decmt->isFinal)
        return 1;

    /* Set up the start of the next series of chunks. The first buffer is the last of the those already loaded. */
    inBlock = &decmt->threads[decmt->numThreads].inBlock;
    inBlock->first = done->inBlock.last;
    inBlock->last = inBlock->first;
    inBlock->endPos = done->inBlock.endPos;
    inBlock->startPos = inBlock->endPos;
    inBlock->unpackSize = 0;

    return 0;
}

/* Read input into the buffer chain, adding new nodes when necessary. 
 * The chunks in each buffer are parsed before a new buffer is allocated.
 * No new buffers will be allocated after the terminator is encountered.
 * Returns 1 if the terminator was found or enough work exists for all threads,
 * 0 if input is empty,
 * or FL2_error_corruption_detected, or FL2_error_memory_allocation.
 * The memory limit is enforced by returning FL2_error_memory_allocation.
 */
static size_t FL2_loadInputMt(FL2_decMt *const decmt, FL2_inBuffer* const input)
{
    FL2_decBlock *inBlock = &decmt->threads[decmt->numThreads].inBlock;
    LZMA2_parseRes res = CHUNK_CONTINUE;
    /* Continue while input is available or the parse pos is not beyond the end */
    while (input->pos < input->size || inBlock->endPos < inBlock->last->length) {
        if (inBlock->endPos < inBlock->last->length) {
            res = FL2_parseMt(inBlock);
            if (res == CHUNK_ERROR)
                return FL2_ERROR(corruption_detected);

            if (res == CHUNK_DICT_RESET || res == CHUNK_FINAL) {
                /* We have a complete series of chunks starting from a dict reset and
                 * ending with another reset or the terminator. Set up the thread job. */
                size_t end = FL2_handleFinalChunkMt(decmt, res);

                /* end is nonzero if memory limit hit or ready to decode */
                if (end != 0) {
                    inBlock = &decmt->threads[decmt->numThreads - 1].inBlock;
                    /* rewind input in case data beyond terminator was read. Required for xxhash and container formats */
                    size_t back = MIN(input->pos, inBlock->last->length - inBlock->endPos);
                    input->pos -= back;
                    inBlock->last->length -= back;
                    return end;
                }
                inBlock = &decmt->threads[decmt->numThreads].inBlock;
            }
        }
        if (inBlock->last->length >= LZMA2_MT_INPUT_SIZE && inBlock->endPos + LZMA_REQUIRED_INPUT_MAX >= inBlock->last->length) {
            /* Create a new buffer if endPos is within the overlap region. The function copies the overlap. */
            FL2_decInbuf *const next = FL2_createInbufNode(decmt, inBlock->last);
            if (next == NULL) {
                if (inBlock->endPos < inBlock->last->length) {
                    size_t back = MIN(input->pos, inBlock->last->length - inBlock->endPos);
                    input->pos -= back;
                    inBlock->last->length -= back;
                }
                return FL2_ERROR(memory_allocation);
            }
            inBlock->last = next;
            inBlock->endPos -= LZMA2_MT_INPUT_SIZE - LZMA_REQUIRED_INPUT_MAX;
        }
        /* Read as much input as possible */
        size_t toread = MIN(input->size - input->pos, LZMA2_MT_INPUT_SIZE - inBlock->last->length);
        memcpy(inBlock->last->inBuf + inBlock->last->length, (BYTE*)input->src + input->pos, toread);
        inBlock->last->length += toread;
        input->pos += toread;

        /* Do not continue if we have an incomplete chunk header */
        if (res == CHUNK_MORE_DATA && toread == 0)
            break;
    }
    return 0;
}

/* Handle MT buffer allocation failure.
 * Decompress input from the MT buffer chain
 * until it is possible to switch to the caller's input buffer
 */
static size_t FL2_decompressFailedMt(FL2_DStream* const fds, FL2_outBuffer* const output, FL2_inBuffer* const input)
{
    FL2_decMt *const decmt = fds->decmt;

    if(decmt->head->length == 0)
        return FL2_decompressOverlappedInput(fds, output, input);

    if (!decmt->failState) {
        /* On first call of this function, free any output buffers already allocated,
         * and set up the read position in the input buffer chain. The main thread's decoder needs initialization too. */
        DEBUGLOG(3, "Switching to ST decompression. Memory: %u, limit %u", (unsigned)decmt->memTotal, (unsigned)decmt->memLimit);

        FL2_freeOutputBuffers(decmt);

        decmt->cur = decmt->threads[0].inBlock.first;
        decmt->curPos = decmt->threads[0].inBlock.startPos;

        decmt->failState = 1;

        CHECK_F(LZMA2_initDecoder(&fds->dec, decmt->prop, NULL, 0));
    }
    FL2_decInbuf *const cur = decmt->cur;

    FL2_inBuffer temp;
    temp.src = cur->inBuf;
    temp.pos = decmt->curPos;
    temp.size = cur->length;

    CHECK_F(FL2_decompressInput(fds, output, &temp));

    decmt->curPos = temp.pos;

    if (temp.pos + LZMA_REQUIRED_INPUT_MAX >= temp.size) {
        if (cur->next == NULL) {
            /* The last buffer in the chain */
            fds->overlapSize = temp.size - temp.pos;
            memcpy(fds->overlap, cur->inBuf + temp.pos, fds->overlapSize);
            decmt->cur = NULL;
            LZMA2_freeExtraInbufNodes(decmt);
        }
        else {
            decmt->curPos -= cur->length - LZMA_REQUIRED_INPUT_MAX;
            decmt->cur = cur->next;
        }
    }

    return FL2_error_no_error;
}

static size_t FL2_decompressStreamMt(FL2_DStream* const fds, FL2_outBuffer* const output, FL2_inBuffer* const input)
{
    FL2_decMt *const decmt = fds->decmt;

    /* failState is set if the memory limit was hit or allocation failed */
    if(decmt->failState)
        return FL2_decompressFailedMt(fds, output, input);

    if (fds->stage == FL2DEC_STAGE_DECOMP) {
        /* Allocate and fill the input buffer chain */
        size_t const res = FL2_loadInputMt(decmt, input);

        /* Failover if allocation failed */
        if (FL2_getErrorCode(res) == FL2_error_memory_allocation)
            return FL2_decompressFailedMt(fds, output, input);
        CHECK_F(res);

        /* res > 0 means all threads have input or the terminator was encountered */
        if (res > 0) {
            CHECK_F(FL2_decompressBlocksMt(fds));
            fds->stage = FL2DEC_STAGE_MT_WRITE;
        }
    }
    if (fds->stage == FL2DEC_STAGE_MT_WRITE) {
        if (FL2_writeStreamBlocks(fds, output))
            fds->stage = decmt->isFinal ? (fds->doHash ? FL2DEC_STAGE_HASH : FL2DEC_STAGE_FINISHED)
                : FL2DEC_STAGE_DECOMP;
    }
    return fds->stage != FL2DEC_STAGE_FINISHED;
}

#endif /* FL2_SINGLETHREAD */

FL2LIB_API FL2_DStream* FL2LIB_CALL FL2_createDStream(void)
{
    return FL2_createDStreamMt(1);
}

static void FL2_resetDStream(FL2_DStream *fds)
{
    fds->stage = FL2DEC_STAGE_INIT;
    fds->asyncRes = 0;
    fds->streamTotal = 0;
    fds->overlapSize = 0;
    fds->progress = 0;
#ifndef NO_XXHASH
    fds->xxhPos = 0;
#endif
    fds->loopCount = 0;
    fds->wait = 0;
}

FL2LIB_API FL2_DStream *FL2LIB_CALL FL2_createDStreamMt(unsigned nbThreads)
{
    FL2_DStream* const fds = malloc(sizeof(FL2_DStream));
    DEBUGLOG(3, "FL2_createDStream");

    if (fds != NULL) {
        LZMA_constructDCtx(&fds->dec);

        nbThreads = FL2_checkNbThreads(nbThreads);

        FL2_resetDStream(fds);
        fds->timeout = 0;

#ifndef FL2_SINGLETHREAD
        fds->decompressThread = NULL;
        fds->decmt = (nbThreads > 1) ? FL2_lzma2DecMt_create(nbThreads) : NULL;
#endif

#ifndef NO_XXHASH
        fds->xxh = NULL;
#endif
        fds->doHash = 0;
    }

    return fds;
}

FL2LIB_API size_t FL2LIB_CALL FL2_freeDStream(FL2_DStream* fds)
{
    if (fds != NULL) {
        DEBUGLOG(3, "FL2_freeDStream");
        LZMA_destructDCtx(&fds->dec);
#ifndef FL2_SINGLETHREAD
        FL2POOL_free(fds->decompressThread);
        FL2_lzma2DecMt_free(fds->decmt);
#endif
#ifndef NO_XXHASH
        XXH32_freeState(fds->xxh);
#endif
        free(fds);
    }
    return 0;
}

FL2LIB_API void FL2LIB_CALL FL2_setDStreamMemoryLimitMt(FL2_DStream * fds, size_t limit)
{
#ifndef FL2_SINGLETHREAD
    if (fds->decmt != NULL)
        fds->decmt->memLimit = limit;
#endif
}

FL2LIB_API size_t FL2LIB_CALL FL2_initDStream(FL2_DStream* fds)
{
    DEBUGLOG(4, "FL2_initDStream");

    if (fds->wait)
        return FL2_ERROR(stage_wrong);

    FL2_resetDStream(fds);

#ifndef FL2_SINGLETHREAD
    FL2_lzma2DecMt_init(fds->decmt);
#endif
    return FL2_error_no_error;
}

FL2LIB_API size_t FL2LIB_CALL FL2_setDStreamTimeout(FL2_DStream * fds, unsigned timeout)
{
#ifndef FL2_SINGLETHREAD
    /* decompressThread is only used if a timeout is specified */
    if (timeout != 0) {
        if (fds->decompressThread == NULL) {
            fds->decompressThread = FL2POOL_create(1);
            if (fds->decompressThread == NULL)
                return FL2_ERROR(memory_allocation);
        }
    }
    else if (!fds->wait) {
        /* Only free the thread if decompression not underway */
        FL2POOL_free(fds->decompressThread);
        fds->decompressThread = NULL;
    }
    fds->timeout = timeout;
#endif
    return FL2_error_no_error;
}

FL2LIB_API size_t FL2LIB_CALL FL2_waitDStream(FL2_DStream * fds)
{
#ifndef FL2_SINGLETHREAD
    if (FL2POOL_waitAll(fds->decompressThread, fds->timeout) != 0)
        return FL2_ERROR(timedOut);
#endif
    /* decompressThread writes the result into asyncRes before sleeping */
    return fds->asyncRes;
}

FL2LIB_API void FL2LIB_CALL FL2_cancelDStream(FL2_DStream *fds)
{
#ifndef FL2_SINGLETHREAD
    if (fds->decompressThread != NULL) {
        fds->decmt->canceled = 1;

        FL2POOL_waitAll(fds->decompressThread, 0);

        fds->decmt->canceled = 0;
    }
    FL2_lzma2DecMt_cleanup(fds->decmt);
#endif
}

FL2LIB_API unsigned long long FL2LIB_CALL FL2_getDStreamProgress(const FL2_DStream * fds)
{
    return fds->streamTotal + fds->progress;
}

static size_t FL2_initDStream_prop(FL2_DStream* const fds, BYTE prop)
{
    fds->doHash = prop >> FL2_PROP_HASH_BIT;
    prop &= FL2_LZMA_PROP_MASK;

    /* If MT decoding is enabled and the dict is not too large, decoder init will occur elsewhere */
#ifndef FL2_SINGLETHREAD
    if (fds->decmt == NULL || FL2_lzma2DecMt_initProp(fds->decmt, prop))
#endif
        CHECK_F(LZMA2_initDecoder(&fds->dec, prop, NULL, 0));

#ifndef NO_XXHASH
    if (fds->doHash) {
        if (fds->xxh == NULL) {
            DEBUGLOG(3, "Creating hash state");
            fds->xxh = XXH32_createState();
            if (fds->xxh == NULL)
                return FL2_ERROR(memory_allocation);
        }
        XXH32_reset(fds->xxh, 0);
    }
#endif
    return FL2_error_no_error;
}

FL2LIB_API size_t FL2LIB_CALL FL2_initDStream_withProp(FL2_DStream* fds, unsigned char prop)
{
    CHECK_F(FL2_initDStream(fds));
    CHECK_F(FL2_initDStream_prop(fds, prop));
    fds->stage = FL2DEC_STAGE_DECOMP;
    return FL2_error_no_error;
}

static size_t FL2_decompressStream_blocking(FL2_DStream* fds, FL2_outBuffer* output, FL2_inBuffer* input)
{
#ifndef FL2_SINGLETHREAD
    FL2_decMt *const decmt = fds->decmt;
#endif
    size_t const prevOut = output->pos;
    size_t const prevIn = input->pos;

    if (input->pos < input->size
#ifndef FL2_SINGLETHREAD
        || decmt
#endif
        ) {
        if (fds->stage == FL2DEC_STAGE_INIT) {
            BYTE prop = ((const BYTE*)input->src)[input->pos];
            ++input->pos;
            FL2_initDStream_prop(fds, prop);
            fds->stage = FL2DEC_STAGE_DECOMP;
        }
#ifndef FL2_SINGLETHREAD
        if (decmt) {
            size_t res = FL2_decompressStreamMt(fds, output, input);
            if (FL2_isError(res)) {
                FL2_lzma2DecMt_cleanup(decmt);
                return res;
            }
        }
        else
#endif
        {
            CHECK_F(FL2_decompressOverlappedInput(fds, output, input));
        }
        if (fds->stage == FL2DEC_STAGE_HASH) {
#ifndef NO_XXHASH
#ifndef FL2_SINGLETHREAD
            if (fds->overlapSize != 0) {
                /* Must copy buffered data before using input */
                size_t toRead = MIN(XXHASH_SIZEOF - fds->xxhPos, fds->overlapSize);
                memcpy(fds->xxhIn.digest + fds->xxhPos, fds->overlap, toRead);
                fds->xxhPos += toRead;
                fds->overlapSize = 0;
            }
#endif
            size_t toRead = MIN(XXHASH_SIZEOF - fds->xxhPos, input->size - input->pos);
            memcpy(fds->xxhIn.digest + fds->xxhPos, (BYTE*)input->src + input->pos, toRead);
            input->pos += toRead;
            fds->xxhPos += toRead;
            if (fds->xxhPos == XXHASH_SIZEOF) {
                DEBUGLOG(4, "Checking hash");
                U32 hash = XXH32_hashFromCanonical(&fds->xxhIn);
                if (hash != XXH32_digest(fds->xxh))
                    return FL2_ERROR(checksum_wrong);
                fds->stage = FL2DEC_STAGE_FINISHED;
            }
#else
            fds->stage = FL2DEC_STAGE_FINISHED;
#endif /* NO_XXHASH */
        }
    }
    if (fds->stage != FL2DEC_STAGE_FINISHED && prevOut == output->pos && prevIn == input->pos) {
        /* No progress was made */
        ++fds->loopCount;
        if (fds->loopCount > 2) {
            FL2_cancelDStream(fds);
            return FL2_ERROR(buffer);
        }
    }
    else {
        fds->loopCount = 0;
    }

    if (fds->stage == FL2DEC_STAGE_FINISHED) {
#ifndef FL2_SINGLETHREAD
        FL2_lzma2DecMt_cleanup(decmt);
#endif
        return 0;
    }
    else {
        return 1;
    }
}

/* FL2_decompressStream_async() : FL2POOL_function type */
static void FL2_decompressStream_async(void* const jobDescription, ptrdiff_t const n)
{
    FL2_DStream* const fds = (FL2_DStream*)jobDescription;

    fds->asyncRes = FL2_decompressStream_blocking(fds, fds->asyncOutput, fds->asyncInput);
    fds->wait = 0;

    (void)n;
}

FL2LIB_API size_t FL2LIB_CALL FL2_decompressStream(FL2_DStream* fds, FL2_outBuffer* output, FL2_inBuffer* input)
{
    fds->streamTotal += fds->progress;
    fds->progress = 0;

#ifndef FL2_SINGLETHREAD
    if (fds->decompressThread != NULL) {
        /* Calling FL2_decompressStream() while waiting for decompressThread to fall idle is not allowed */
        if (fds->wait)
            return FL2_ERROR(stage_wrong);

        fds->asyncOutput = output;
        fds->asyncInput = input;
        /* FL2_decompressStream_async will reset fds->wait upon completion */
        fds->wait = 1;

        FL2POOL_add(fds->decompressThread, FL2_decompressStream_async, fds, 0);

        /* Wait for completion or a timeout */
        CHECK_F(FL2_waitDStream(fds));

        /* FL2_decompressStream_async() stores result in asyncRes */
        return fds->asyncRes;
    }
    else
#endif
    {
        return FL2_decompressStream_blocking(fds, output, input);
    }
}

FL2LIB_API size_t FL2LIB_CALL FL2_estimateDCtxSize(unsigned nbThreads)
{
    nbThreads = FL2_checkNbThreads(nbThreads);
    if (nbThreads > 1)
        return nbThreads * (sizeof(FL2_blockDecMt) + sizeof(FL2_DCtx));

    return sizeof(FL2_DCtx);
}

FL2LIB_API size_t FL2LIB_CALL FL2_estimateDStreamSize(size_t dictSize, unsigned nbThreads)
{
    nbThreads = FL2_checkNbThreads(nbThreads);
    if (nbThreads > 1) {
        /* Estimate 50% compression and a block size of 4 * dictSize */
        return nbThreads * sizeof(FL2_DCtx) + (dictSize + dictSize / 2) * 4 * nbThreads;
    }
    return LZMA2_decMemoryUsage(dictSize);
}