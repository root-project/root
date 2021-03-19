/* lzma2_dec.h -- LZMA2 Decoder
2017-04-03 : Igor Pavlov : Public domain
Modified for FL2 by Conor McCarthy */

#ifndef __LZMA_DEC_H
#define __LZMA_DEC_H

#include "mem.h"

#if defined (__cplusplus)
extern "C" {
#endif

/* #define LZMA_DEC_PROB16 */
/* 32-bit probs can increase the speed on some CPUs,
   but memory usage for LZMA2_DCtx::probs will be doubled in that case */

#ifdef LZMA_DEC_PROB16
typedef U16 LZMA2_prob;
#else
typedef U32 LZMA2_prob;
#endif


/* ---------- LZMA Properties ---------- */

typedef struct
{
	BYTE lc;
	BYTE lp;
	BYTE pb;
	BYTE pad_;
	U32 dic_size;
} LZMA2_props;

/* LzmaProps_Decode - decodes properties
Returns:
  SZ_OK
  SZ_ERROR_UNSUPPORTED - Unsupported properties
*/


/* ---------- LZMA Decoder state ---------- */

/* LZMA_REQUIRED_INPUT_MAX = number of required input bytes for worst case.
   Num bits = log2((2^11 / 31) ^ 22) + 26 < 134 + 26 = 160; */

#define LZMA_REQUIRED_INPUT_MAX 20

#define kNumPosBitsMax 4
#define kNumPosStatesMax (1 << kNumPosBitsMax)

#define kLenNumLowBits 3
#define kLenNumLowSymbols (1 << kLenNumLowBits)
#define kLenNumHighBits 8
#define kLenNumHighSymbols (1 << kLenNumHighBits)

#define LenLow 0
#define LenHigh (LenLow + 2 * (kNumPosStatesMax << kLenNumLowBits))
#define kNumLenProbs (LenHigh + kLenNumHighSymbols)

#define LenChoice LenLow
#define LenChoice2 (LenLow + (1 << kLenNumLowBits))

#define kNumStates 12
#define kNumStates2 16
#define kNumLitStates 7

#define kStartPosModelIndex 4
#define kEndPosModelIndex 14
#define kNumFullDistances (1 << (kEndPosModelIndex >> 1))

#define kNumPosSlotBits 6
#define kNumLenToPosStates 4

#define kNumAlignBits 4
#define kAlignTableSize (1 << kNumAlignBits)

#define kMatchMinLen 2
#define kMatchSpecLenStart (kMatchMinLen + kLenNumLowSymbols * 2 + kLenNumHighSymbols)

/* External ASM code needs same CLzmaProb array layout. So don't change it. */

/* (probs_1664) is faster and better for code size at some platforms */
/*
#ifdef MY_CPU_X86_OR_AMD64
*/
#define kStartOffset 1664
#define GET_PROBS p->probs_1664
/*
#define GET_PROBS p->probs + kStartOffset
#else
#define kStartOffset 0
#define GET_PROBS p->probs
#endif
*/

#define SpecPos (-kStartOffset)
#define IsRep0Long (SpecPos + kNumFullDistances)
#define RepLenCoder (IsRep0Long + (kNumStates2 << kNumPosBitsMax))
#define LenCoder (RepLenCoder + kNumLenProbs)
#define IsMatch (LenCoder + kNumLenProbs)
#define Align (IsMatch + (kNumStates2 << kNumPosBitsMax))
#define IsRep (Align + kAlignTableSize)
#define IsRepG0 (IsRep + kNumStates)
#define IsRepG1 (IsRepG0 + kNumStates)
#define IsRepG2 (IsRepG1 + kNumStates)
#define PosSlot (IsRepG2 + kNumStates)
#define Literal (PosSlot + (kNumLenToPosStates << kNumPosSlotBits))
#define NUM_BASE_PROBS (Literal + kStartOffset)

#if Align != 0 && kStartOffset != 0
#error Stop_Compiling_Bad_LZMA_kAlign
#endif

#if NUM_BASE_PROBS != 1984
#error Stop_Compiling_Bad_LZMA_PROBS
#endif


#define kLzmaLitSize 0x300

#define LzmaProps_GetNumProbs(p) (NUM_BASE_PROBS + ((U32)kLzmaLitSize << ((p)->lc + (p)->lp)))


#define CALC_POS_STATE(processed_pos, pb_mask) (((processed_pos) & (pb_mask)) << 4)
#define COMBINED_PS_STATE (pos_state + state)
#define GET_LEN_STATE (pos_state)

#define kLzma2LcLpMax 4U


typedef struct
{
    LZMA2_props prop;
    BYTE *dic;
	size_t dic_pos;
	size_t dic_buf_size;
	const BYTE *buf;
	LZMA2_prob *probs_1664;
	U32 range;
	U32 code;
    U32 processed_pos;
    U32 check_dic_size;
    U32 reps[4];
	unsigned state;
	unsigned state2;
	unsigned remain_len;
    size_t pack_size;
    size_t unpack_size;
    BYTE control;
    BYTE need_init_dic;
	BYTE need_init_state;
	BYTE need_init_state2;
    BYTE need_init_prop;
	BYTE need_flush;
	BYTE ext_dic;
	BYTE pad_;
    LZMA2_prob probs[NUM_BASE_PROBS + ((U32)kLzmaLitSize << kLzma2LcLpMax)];
} LZMA2_DCtx;

void LZMA_constructDCtx(LZMA2_DCtx *p);

typedef enum
{
  LZMA_FINISH_ANY,   /* finish at any point */
  LZMA_FINISH_END    /* block must be finished at the end */
} LZMA2_finishMode;

typedef enum
{
    LZMA_STATUS_NOT_SPECIFIED,     /* use main error code instead */
    LZMA_STATUS_FINISHED,          /* stream was finished */
    LZMA_STATUS_NOT_FINISHED,      /* stream was not finished */
    LZMA_STATUS_NEEDS_MORE_INPUT,  /* you must provide more input bytes */
    LZMA_STATUS_OUTPUT_FULL        /* not finished; output buffer is full */
} LZMA2_status;

void LZMA_destructDCtx(LZMA2_DCtx *const p);

size_t LZMA2_getDictSizeFromProp(BYTE const dict_prop);

#define LZMA2_CONTENTSIZE_ERROR   (size_t)-1

U64 LZMA2_getUnpackSize(const BYTE *const src, size_t const src_len);

size_t LZMA2_decMemoryUsage(size_t const dict_size);

size_t LZMA2_initDecoder(LZMA2_DCtx *const p, BYTE const dict_prop, BYTE *const dic, size_t dic_buf_size);

size_t LZMA2_decodeToDic(LZMA2_DCtx *const p, size_t const dic_limit,
    const BYTE *const src, size_t *const src_len, LZMA2_finishMode const finish_mode);

size_t LZMA2_decodeToBuf(LZMA2_DCtx *const p, BYTE *dest, size_t *const dest_len,
    const BYTE *src, size_t *const src_len, LZMA2_finishMode const finish_mode);

typedef enum
{
    CHUNK_MORE_DATA,
    CHUNK_CONTINUE,
    CHUNK_DICT_RESET,
    CHUNK_FINAL,
    CHUNK_ERROR
} LZMA2_parseRes;

typedef struct
{
    size_t pack_size;
    size_t unpack_size;
} LZMA2_chunk;

#if defined(FL2_DEBUG) && (FL2_DEBUG>=1)
#  define LZMA2_MT_INPUT_SIZE 0x400
#else
#  define LZMA2_MT_INPUT_SIZE 0x40000
#endif

LZMA2_parseRes LZMA2_parseInput(const BYTE* const in_buf, size_t const pos, ptrdiff_t const len, LZMA2_chunk *const inf);

#if defined (__cplusplus)
}
#endif

#endif
