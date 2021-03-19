/* lzma2_dec.c -- LZMA2 Decoder
Based upon LzmaDec.c 2018-02-28 : Igor Pavlov : Public domain
Modified for FL2 by Conor McCarthy */

#include <stdlib.h>
#include "fl2_errors.h"
#include "fl2_internal.h"
#include "lzma2_dec.h"
#include "platform.h"


#ifdef HAVE_SMALL
#  define LZMA_SIZE_OPT
#endif

#define kNumTopBits 24
#define kTopValue ((U32)1 << kNumTopBits)

#define kNumBitModelTotalBits 11
#define kBitModelTotal (1 << kNumBitModelTotalBits)
#define kNumMoveBits 5

#define RC_INIT_SIZE 5

#define NORMALIZE if (range < kTopValue) { range <<= 8; code = (code << 8) | (*buf++); }

#define IF_BIT_0(p) ttt = *(p); NORMALIZE; bound = (range >> kNumBitModelTotalBits) * ttt; if (code < bound)
#define UPDATE_0(p) range = bound; *(p) = (LZMA2_prob)(ttt + ((kBitModelTotal - ttt) >> kNumMoveBits));
#define UPDATE_1(p) range -= bound; code -= bound; *(p) = (LZMA2_prob)(ttt - (ttt >> kNumMoveBits));
#define GET_BIT2(p, i, A0, A1) IF_BIT_0(p) \
    { UPDATE_0(p); i = (i + i); A0; } else \
    { UPDATE_1(p); i = (i + i) + 1; A1; }

#if defined __x86_64__s || defined _M_X64

#define USE_CMOV

#define PREP_BIT(p) ttt = *(p); NORMALIZE; bound = (range >> kNumBitModelTotalBits) * ttt
#define UPDATE_PREP_0 U32 r0 = bound; unsigned p0 = (ttt + ((kBitModelTotal - ttt) >> kNumMoveBits))
#define UPDATE_PREP_1 U32 r1 = range - bound; unsigned p1 = (ttt - (ttt >> kNumMoveBits))
#define UPDATE_COND(p) range=(code < bound) ? r0 : r1; *p = (LZMA2_prob)((code < bound) ? p0 : p1)
#define UPDATE_CODE code = code - ((code < bound) ? 0 : bound)

#define TREE_GET_BIT(probs, i) { LZMA2_prob *pp = (probs)+(i); PREP_BIT(pp); \
    UPDATE_PREP_0; unsigned i0 = (i + i); \
    UPDATE_PREP_1; unsigned i1 = (i + i) + 1; \
    UPDATE_COND(pp); \
    i = (code < bound) ? i0 : i1; \
    UPDATE_CODE; \
}

#define REV_BIT_VAR(probs, i, m) { LZMA2_prob *pp = (probs)+(i); PREP_BIT(pp); \
    UPDATE_PREP_0; U32 i0 = i + m; U32 m2 = m + m; \
    UPDATE_PREP_1; U32 i1 = i + m2; \
    UPDATE_COND(pp); \
    i = (code < bound) ? i0 : i1; \
    m = m2; \
    UPDATE_CODE; \
}
#define REV_BIT_CONST(probs, i, m) { LZMA2_prob *pp = (probs)+(i); PREP_BIT(pp); \
    UPDATE_PREP_0; \
    UPDATE_PREP_1; \
    UPDATE_COND(pp); \
    i += m + (code < bound ? 0 : m); \
    UPDATE_CODE; \
}
#define REV_BIT_LAST(probs, i, m) { LZMA2_prob *pp = (probs)+(i); PREP_BIT(pp); \
    UPDATE_PREP_0; \
    UPDATE_PREP_1; \
    UPDATE_COND(pp); \
    i -= code < bound ? m : 0; \
    UPDATE_CODE; \
}

#define MATCHED_LITER_DEC \
    match_byte += match_byte; \
    bit = offs; \
    offs &= match_byte; \
    prob_lit = prob + (offs + bit + symbol); \
    PREP_BIT(prob_lit); \
    { UPDATE_PREP_0; unsigned i0 = (symbol + symbol); \
    UPDATE_PREP_1; unsigned i1 = (symbol + symbol) + 1; \
    UPDATE_COND(prob_lit); \
    symbol = (code < bound) ? i0 : i1; \
    offs = (code < bound) ? offs ^ bit : offs; \
    UPDATE_CODE; }

#else
#define TREE_GET_BIT(probs, i) { GET_BIT2(probs + i, i, ;, ;); }

#define REV_BIT(p, i, A0, A1) IF_BIT_0(p + i) \
    { UPDATE_0(p + i); A0; } else \
    { UPDATE_1(p + i); A1; }
#define REV_BIT_VAR(  p, i, m) REV_BIT(p, i, i += m; m += m, m += m; i += m; )
#define REV_BIT_CONST(p, i, m) REV_BIT(p, i, i += m;       , i += m * 2; )
#define REV_BIT_LAST( p, i, m) REV_BIT(p, i, i -= m        , ; )

#define MATCHED_LITER_DEC \
    match_byte += match_byte; \
    bit = offs; \
    offs &= match_byte; \
    prob_lit = prob + (offs + bit + symbol); \
    GET_BIT2(prob_lit, symbol, offs ^= bit; , ;)

#endif

#define TREE_DECODE(probs, limit, i) \
    { i = 1; do { TREE_GET_BIT(probs, i); } while (i < limit); i -= limit; }

#ifdef LZMA_SIZE_OPT
#define TREE_6_DECODE(probs, i) TREE_DECODE(probs, (1 << 6), i)
#else
#define TREE_6_DECODE(probs, i) \
    { i = 1; \
    TREE_GET_BIT(probs, i); \
    TREE_GET_BIT(probs, i); \
    TREE_GET_BIT(probs, i); \
    TREE_GET_BIT(probs, i); \
    TREE_GET_BIT(probs, i); \
    TREE_GET_BIT(probs, i); \
    i -= 0x40; }
#endif

#define NORMAL_LITER_DEC TREE_GET_BIT(prob, symbol)

#define NORMALIZE_CHECK if (range < kTopValue) { return 0; }

#define IF_BIT_0_CHECK(p) ttt = *(p); NORMALIZE_CHECK; bound = (range >> kNumBitModelTotalBits) * ttt; if (code < bound)
#define UPDATE_0_CHECK range = bound;
#define UPDATE_1_CHECK range -= bound; code -= bound;
#define GET_BIT2_CHECK(p, i, A0, A1) IF_BIT_0_CHECK(p) \
    { UPDATE_0_CHECK; i = (i + i); A0; } else \
    { UPDATE_1_CHECK; i = (i + i) + 1; A1; }
#define GET_BIT_CHECK(p, i) GET_BIT2_CHECK(p, i, ; , ;)
#define TREE_DECODE_CHECK(probs, limit, i) \
    { i = 1; do { GET_BIT_CHECK(probs + i, i) } while (i < limit); i -= limit; }

#define REV_BIT_CHECK(p, i, m) IF_BIT_0_CHECK(p + i) \
    { UPDATE_0_CHECK; i += m; m += m; } else \
    { UPDATE_1_CHECK; m += m; i += m; }

/*
00000000  -  EOS
00000001 U U  -  Uncompressed Reset Dic
00000010 U U  -  Uncompressed No Reset
100uuuuu U U P P  -  LZMA no reset
101uuuuu U U P P  -  LZMA reset state
110uuuuu U U P P S  -  LZMA reset state + new prop
111uuuuu U U P P S  -  LZMA reset state + new prop + reset dic

u, U - Unpack Size
P - Pack Size
S - Props
*/

#define LZMA2_CONTROL_LZMA (1 << 7)
#define LZMA2_CONTROL_COPY_NO_RESET 2
#define LZMA2_CONTROL_COPY_RESET_DIC 1
#define LZMA2_CONTROL_EOF 0

#define LZMA2_IS_UNCOMPRESSED_STATE(control) ((control & LZMA2_CONTROL_LZMA) == 0)

#define LZMA2_GET_LZMA_MODE(control) ((control >> 5) & 3)
#define LZMA2_IS_THERE_PROP(mode) ((mode) >= 2)

#ifdef SHOW_DEBUG_INFO
#define PRF(x) x
#else
#define PRF(x)
#endif

typedef enum
{
    LZMA2_STATE_CONTROL,
    LZMA2_STATE_DATA,
    LZMA2_STATE_FINISHED,
    LZMA2_STATE_ERROR
} LZMA2_state;

#define LZMA_DIC_MIN (1 << 12)

static BYTE LZMA_tryDummy(const LZMA2_DCtx *const p)
{
    const LZMA2_prob *probs = GET_PROBS;
	unsigned state = p->state;
	U32 range = p->range;
	U32 code = p->code;

    const LZMA2_prob *prob;
    U32 bound;
    unsigned ttt;
    unsigned pos_state = CALC_POS_STATE(p->processed_pos, (1 << p->prop.pb) - 1);

    prob = probs + IsMatch + COMBINED_PS_STATE;
    IF_BIT_0_CHECK(prob)
    {
        UPDATE_0_CHECK

            prob = probs + Literal;
        if (p->check_dic_size != 0 || p->processed_pos != 0)
            prob += ((U32)kLzmaLitSize *
            ((((p->processed_pos) & ((1 << (p->prop.lp)) - 1)) << p->prop.lc) +
                (p->dic[(p->dic_pos == 0 ? p->dic_buf_size : p->dic_pos) - 1] >> (8 - p->prop.lc))));

        if (state < kNumLitStates)
        {
            unsigned symbol = 1;
            do { GET_BIT_CHECK(prob + symbol, symbol) } while (symbol < 0x100);
        }
        else
        {
            unsigned match_byte = p->dic[p->dic_pos - p->reps[0] +
                (p->dic_pos < p->reps[0] ? p->dic_buf_size : 0)];
            unsigned offs = 0x100;
            unsigned symbol = 1;
            do
            {
                unsigned bit;
                const LZMA2_prob *prob_lit;
                match_byte += match_byte;
                bit = offs;
                offs &= match_byte;
                prob_lit = prob + (offs + bit + symbol);
                GET_BIT2_CHECK(prob_lit, symbol, offs ^= bit; , ; )
            } while (symbol < 0x100);
        }
    }
    else
    {
        unsigned len;
        UPDATE_1_CHECK;

        prob = probs + IsRep + state;
        IF_BIT_0_CHECK(prob)
        {
            UPDATE_0_CHECK;
            state = 0;
            prob = probs + LenCoder;
        }
        else
        {
            UPDATE_1_CHECK;
            prob = probs + IsRepG0 + state;
            IF_BIT_0_CHECK(prob)
            {
                UPDATE_0_CHECK;
                prob = probs + IsRep0Long + COMBINED_PS_STATE;
                IF_BIT_0_CHECK(prob)
                {
                    UPDATE_0_CHECK;
                    NORMALIZE_CHECK;
                    return 1;
                }
                else
                {
                    UPDATE_1_CHECK;
                }
            }
            else
            {
                UPDATE_1_CHECK;
                prob = probs + IsRepG1 + state;
                IF_BIT_0_CHECK(prob)
                {
                    UPDATE_0_CHECK;
                }
                else
                {
                    UPDATE_1_CHECK;
                    prob = probs + IsRepG2 + state;
                    IF_BIT_0_CHECK(prob)
                    {
                        UPDATE_0_CHECK;
                    }
                    else
                    {
                        UPDATE_1_CHECK;
                    }
                }
            }
            state = kNumStates;
            prob = probs + RepLenCoder;
        }
        {
            unsigned limit, offset;
            const LZMA2_prob *prob_len = prob + LenChoice;
            IF_BIT_0_CHECK(prob_len)
            {
                UPDATE_0_CHECK;
                prob_len = prob + LenLow + GET_LEN_STATE;
                offset = 0;
                limit = 1 << kLenNumLowBits;
            }
            else
            {
                UPDATE_1_CHECK;
                prob_len = prob + LenChoice2;
                IF_BIT_0_CHECK(prob_len)
                {
                    UPDATE_0_CHECK;
                    prob_len = prob + LenLow + GET_LEN_STATE + (1 << kLenNumLowBits);
                    offset = kLenNumLowSymbols;
                    limit = 1 << kLenNumLowBits;
                }
                else
                {
                  UPDATE_1_CHECK;
                  prob_len = prob + LenHigh;
                  offset = kLenNumLowSymbols * 2;
                  limit = 1 << kLenNumHighBits;
                }
            }
            TREE_DECODE_CHECK(prob_len, limit, len);
            len += offset;
        }

        if (state < 4)
        {
            unsigned pos_slot;
            prob = probs + PosSlot +
                ((len < kNumLenToPosStates - 1 ? len : kNumLenToPosStates - 1) <<
                    kNumPosSlotBits);
            TREE_DECODE_CHECK(prob, 1 << kNumPosSlotBits, pos_slot);
            if (pos_slot >= kStartPosModelIndex)
            {
                unsigned num_direct_bits = ((pos_slot >> 1) - 1);

                if (pos_slot < kEndPosModelIndex)
                {
                    prob = probs + SpecPos + ((2 | (pos_slot & 1)) << num_direct_bits);
                }
                else
                {
                    num_direct_bits -= kNumAlignBits;
                    do
                    {
                        NORMALIZE_CHECK;
                        range >>= 1;
                        code -= range & (((code - range) >> 31) - 1);
                        /* if (code >= range) code -= range; */
                    } while (--num_direct_bits != 0);
                    prob = probs + Align;
                    num_direct_bits = kNumAlignBits;
                }
                {
                    unsigned i = 1;
                    unsigned m = 1;
                    do
                    {
                        REV_BIT_CHECK(prob, i, m);
                    } while (--num_direct_bits != 0);
                }
            }
        }
    }
    NORMALIZE_CHECK;
    return 1;
}

/* First LZMA-symbol is always decoded.
And it decodes new LZMA-symbols while (buf < buf_limit), but "buf" is without last normalization
Out:
  Result:
    0 - OK
    1 - Error
*/

#ifdef LZMA2_DEC_OPT

int LZMA_decodeReal_3(LZMA2_DCtx *p, size_t limit, const BYTE *buf_limit);

#else

static int LZMA_decodeReal_3(LZMA2_DCtx *p, size_t limit, const BYTE *buf_limit)
{
    LZMA2_prob *const probs = GET_PROBS;

    unsigned state = p->state;
    U32 rep0 = p->reps[0], rep1 = p->reps[1], rep2 = p->reps[2], rep3 = p->reps[3];
    unsigned const pb_mask = ((unsigned)1 << (p->prop.pb)) - 1;
    unsigned const lc = p->prop.lc;
    unsigned const lp_mask = ((unsigned)0x100 << p->prop.lp) - ((unsigned)0x100 >> lc);

    BYTE *const dic = p->dic;
    size_t const dic_buf_size = p->dic_buf_size;
    size_t dic_pos = p->dic_pos;

    U32 processed_pos = p->processed_pos;
    U32 const check_dic_size = p->check_dic_size;
    unsigned len = 0;

    const BYTE *buf = p->buf;
    U32 range = p->range;
    U32 code = p->code;

    do
    {
        LZMA2_prob *prob;
        U32 bound;
        unsigned ttt;
        unsigned pos_state = CALC_POS_STATE(processed_pos, pb_mask);

        prob = probs + IsMatch + COMBINED_PS_STATE;
        IF_BIT_0(prob)
        {
            unsigned symbol;
            UPDATE_0(prob);
            prob = probs + Literal;
            if (processed_pos != 0 || check_dic_size != 0)
                prob += (U32)3 * ((((processed_pos << 8) + dic[(dic_pos == 0 ? dic_buf_size : dic_pos) - 1]) & lp_mask) << lc);
            processed_pos++;

            if (state < kNumLitStates)
            {
                state -= (state < 4) ? state : 3;
                symbol = 1;
#ifdef LZMA_SIZE_OPT
                do { NORMAL_LITER_DEC } while (symbol < 0x100);
#else
                NORMAL_LITER_DEC
                    NORMAL_LITER_DEC
                    NORMAL_LITER_DEC
                    NORMAL_LITER_DEC
                    NORMAL_LITER_DEC
                    NORMAL_LITER_DEC
                    NORMAL_LITER_DEC
                    NORMAL_LITER_DEC
#endif
            }
            else
            {
                unsigned match_byte = dic[dic_pos - rep0 + (dic_pos < rep0 ? dic_buf_size : 0)];
                unsigned offs = 0x100;
                state -= (state < 10) ? 3 : 6;
                symbol = 1;
#ifdef LZMA_SIZE_OPT
                do
                {
                    unsigned bit;
                    LZMA2_prob *prob_lit;
                    MATCHED_LITER_DEC
                } while (symbol < 0x100);
#else
                {
                    unsigned bit;
                    LZMA2_prob *prob_lit;
                    MATCHED_LITER_DEC
                    MATCHED_LITER_DEC
                    MATCHED_LITER_DEC
                    MATCHED_LITER_DEC
                    MATCHED_LITER_DEC
                    MATCHED_LITER_DEC
                    MATCHED_LITER_DEC
                    MATCHED_LITER_DEC
                }
#endif
            }

            dic[dic_pos++] = (BYTE)symbol;
            continue;
        }

        UPDATE_1(prob);
        prob = probs + IsRep + state;
        IF_BIT_0(prob)
        {
            UPDATE_0(prob);
            state += kNumStates;
            prob = probs + LenCoder;
        }
        else
        {
            UPDATE_1(prob);
            /*
            that case was checked before with kBadRepCode:
            if (check_dic_size == 0 && processed_pos == 0)
            return 1;
            */
            prob = probs + IsRepG0 + state;
            IF_BIT_0(prob)
            {
                UPDATE_0(prob);
                prob = probs + IsRep0Long + COMBINED_PS_STATE;
                IF_BIT_0(prob)
                {
                      UPDATE_0(prob);
                      dic[dic_pos] = dic[dic_pos - rep0 + (dic_pos < rep0 ? dic_buf_size : 0)];
                      dic_pos++;
                      processed_pos++;
                      state = state < kNumLitStates ? 9 : 11;
                      continue;
                }
                UPDATE_1(prob);
            }
            else
            {
                U32 distance;
                UPDATE_1(prob);
                prob = probs + IsRepG1 + state;
                IF_BIT_0(prob)
                {
                    UPDATE_0(prob);
                    distance = rep1;
                }
                else
                {
                    UPDATE_1(prob);
                    prob = probs + IsRepG2 + state;
#ifdef USE_CMOV
                    PREP_BIT(prob);
                    UPDATE_PREP_0;
                    UPDATE_PREP_1;
                    UPDATE_COND(prob);
                    distance = code < bound ? rep2 : rep3;
                    rep3 = code < bound ? rep3 : rep2;
                    UPDATE_CODE;
#else
                    IF_BIT_0(prob)
                    {
                        UPDATE_0(prob);
                        distance = rep2;
                    }
                    else
                    {
                        UPDATE_1(prob);
                        distance = rep3;
                        rep3 = rep2;
                    }
#endif
                    rep2 = rep1;
                }
                rep1 = rep0;
                rep0 = distance;
            }
            state = state < kNumLitStates ? 8 : 11;
            prob = probs + RepLenCoder;
        }

#ifdef LZMA_SIZE_OPT
        unsigned lim, offset;
        LZMA2_prob *prob_len = prob + LenChoice;
        IF_BIT_0(prob_len)
        {
              UPDATE_0(prob_len);
              prob_len = prob + LenLow + GET_LEN_STATE;
              offset = 0;
              lim = (1 << kLenNumLowBits);
        }
        else
        {
            UPDATE_1(prob_len);
            prob_len = prob + LenChoice2;
            IF_BIT_0(prob_len)
            {
                UPDATE_0(prob_len);
                prob_len = prob + LenLow + GET_LEN_STATE + (1 << kLenNumLowBits);
                offset = kLenNumLowSymbols;
                lim = (1 << kLenNumLowBits);
            }
            else
            {
                UPDATE_1(prob_len);
                prob_len = prob + LenHigh;
                offset = kLenNumLowSymbols * 2;
                lim = (1 << kLenNumHighBits);
            }
        }
        TREE_DECODE(prob_len, lim, len);
        len += offset;
#else
        LZMA2_prob *prob_len = prob + LenChoice;
        IF_BIT_0(prob_len)
        {
              UPDATE_0(prob_len);
              prob_len = prob + LenLow + GET_LEN_STATE;
              len = 1;
              TREE_GET_BIT(prob_len, len);
              TREE_GET_BIT(prob_len, len);
              TREE_GET_BIT(prob_len, len);
              len -= 8;
        }
        else
        {
            UPDATE_1(prob_len);
            prob_len = prob + LenChoice2;
            IF_BIT_0(prob_len)
            {
                UPDATE_0(prob_len);
                prob_len = prob + LenLow + GET_LEN_STATE + (1 << kLenNumLowBits);
                len = 1;
                TREE_GET_BIT(prob_len, len);
                TREE_GET_BIT(prob_len, len);
                TREE_GET_BIT(prob_len, len);
            }
            else
            {
                UPDATE_1(prob_len);
                prob_len = prob + LenHigh;
                TREE_DECODE(prob_len, (1 << kLenNumHighBits), len);
                len += kLenNumLowSymbols * 2;
            }
        }
#endif

        if (state >= kNumStates)
        {
            U32 distance;
            prob = probs + PosSlot +
                ((len < kNumLenToPosStates ? len : kNumLenToPosStates - 1) << kNumPosSlotBits);
            TREE_6_DECODE(prob, distance);
            if (distance >= kStartPosModelIndex)
            {
                unsigned pos_slot = (unsigned)distance;
                unsigned num_direct_bits = (unsigned)(((distance >> 1) - 1));
                distance = (2 | (distance & 1));
                if (pos_slot < kEndPosModelIndex)
                {
                    distance <<= num_direct_bits;
                    prob = probs + SpecPos;
                    {
                        U32 m = 1;
                        distance++;
                        do
                        {
                            REV_BIT_VAR(prob, distance, m);
                        } while (--num_direct_bits);
                        distance -= m;
                    }
                }
                else
                {
                    num_direct_bits -= kNumAlignBits;
                    do
                    {
                        NORMALIZE
                        range >>= 1;

                        U32 t;
                        code -= range;
                        t = (0 - ((U32)code >> 31));
                        distance = (distance << 1) + (t + 1);
                        code += range & t;
                    } while (--num_direct_bits != 0);
                    prob = probs + Align;
                    distance <<= kNumAlignBits;
                    {
                        U32 i = 1;
                        REV_BIT_CONST(prob, i, 1);
                        REV_BIT_CONST(prob, i, 2);
                        REV_BIT_CONST(prob, i, 4);
                        REV_BIT_LAST(prob, i, 8);
                        distance |= i;
                    }
                }
            }

            rep3 = rep2;
            rep2 = rep1;
            rep1 = rep0;
            rep0 = distance + 1;
            if (distance >= (check_dic_size == 0 ? processed_pos : check_dic_size))
            {
                p->dic_pos = dic_pos;
                return 1;
            }
            state = (state < kNumStates + kNumLitStates) ? kNumLitStates : kNumLitStates + 3;
        }

        len += kMatchMinLen;

        size_t rem = limit - dic_pos;
        if (rem == 0)
        {
            p->dic_pos = dic_pos;
            return 1;
        }

        unsigned cur_len = ((rem < len) ? (unsigned)rem : len);
        size_t pos = dic_pos - rep0 + (dic_pos < rep0 ? dic_buf_size : 0);

        processed_pos += cur_len;

        len -= cur_len;
        if (cur_len <= dic_buf_size - pos)
        {
            BYTE *dest = dic + dic_pos;
            ptrdiff_t src = (ptrdiff_t)pos - (ptrdiff_t)dic_pos;
            const BYTE *end = dest + cur_len;
            dic_pos += cur_len;
            do
                *(dest) = (BYTE)*(dest + src);
            while (++dest != end);
        }
        else
        {
            do
            {
                dic[dic_pos++] = dic[pos];
                if (++pos == dic_buf_size)
                    pos = 0;
            } while (--cur_len != 0);
        }
    } while (dic_pos < limit && buf < buf_limit);

    NORMALIZE;

    p->buf = buf;
    p->range = range;
    p->code = code;
    p->remain_len = len;
    p->dic_pos = dic_pos;
    p->processed_pos = processed_pos;
    p->reps[0] = rep0;
    p->reps[1] = rep1;
    p->reps[2] = rep2;
    p->reps[3] = rep3;
    p->state = state;

    return 0;
}

#endif

static void LZMA_writeRem(LZMA2_DCtx *const p, size_t const limit)
{
    if (p->remain_len != 0 && p->remain_len < kMatchSpecLenStart)
    {
        BYTE *const dic = p->dic;
        size_t dic_pos = p->dic_pos;
        size_t const dic_buf_size = p->dic_buf_size;
        unsigned len = p->remain_len;
        size_t const rep0 = p->reps[0];
        size_t const rem = limit - dic_pos;
        if (rem < len)
            len = (unsigned)(rem);

        if (p->check_dic_size == 0 && p->prop.dic_size - p->processed_pos <= len)
            p->check_dic_size = p->prop.dic_size;

        p->processed_pos += len;
        p->remain_len -= len;
        while (len != 0)
        {
            len--;
            dic[dic_pos] = dic[dic_pos - rep0 + (dic_pos < rep0 ? dic_buf_size : 0)];
            dic_pos++;
        }
        p->dic_pos = dic_pos;
    }
}

#define kRange0 0xFFFFFFFF
#define kBound0 ((kRange0 >> kNumBitModelTotalBits) << (kNumBitModelTotalBits - 1))
#define kBadRepCode (kBound0 + (((kRange0 - kBound0) >> kNumBitModelTotalBits) << (kNumBitModelTotalBits - 1)))
#if kBadRepCode != (0xC0000000 - 0x400)
#error Stop_Compiling_Bad_LZMA_Check
#endif

static size_t LZMA_decodeReal2(LZMA2_DCtx *const p, size_t const limit, const BYTE *const buf_limit)
{
    if (p->buf == buf_limit && !LZMA_tryDummy(p))
        return FL2_ERROR(corruption_detected);
    do
    {
        size_t limit2 = limit;
        if (p->check_dic_size == 0)
        {
            U32 const rem = p->prop.dic_size - p->processed_pos;
            if (limit - p->dic_pos > rem)
                limit2 = p->dic_pos + rem;
            if (p->processed_pos == 0)
                if (p->code >= kBadRepCode)
                    return FL2_ERROR(corruption_detected);
        }

        do {
            if (LZMA_decodeReal_3(p, limit2, buf_limit) != 0)
                return FL2_ERROR(corruption_detected);
        } while (p->dic_pos < limit2 && p->buf == buf_limit && LZMA_tryDummy(p));

        if (p->check_dic_size == 0 && p->processed_pos >= p->prop.dic_size)
            p->check_dic_size = p->prop.dic_size;

        LZMA_writeRem(p, limit);
    } while (p->dic_pos < limit && p->buf < buf_limit && p->remain_len < kMatchSpecLenStart);

    if (p->remain_len > kMatchSpecLenStart)
        p->remain_len = kMatchSpecLenStart;

    return FL2_error_no_error;
}


static void LZMA_initDicAndState(LZMA2_DCtx *const p, BYTE const init_dic, BYTE const init_state)
{
    p->need_flush = 1;
    p->remain_len = 0;

    if (init_dic)
    {
        p->processed_pos = 0;
        p->check_dic_size = 0;
        p->need_init_state = 1;
    }
    if (init_state)
        p->need_init_state = 1;
}

static void LZMA_init(LZMA2_DCtx *const p)
{
    p->dic_pos = 0;
    LZMA_initDicAndState(p, 1, 1);
}

static void LZMA_initStateReal(LZMA2_DCtx *const p)
{
    size_t const num_probs = LzmaProps_GetNumProbs(&p->prop);
    LZMA2_prob *probs = p->probs;
    for (size_t i = 0; i < num_probs; i++)
        probs[i] = kBitModelTotal >> 1;
    p->reps[0] = p->reps[1] = p->reps[2] = p->reps[3] = 1;
    p->state = 0;
    p->need_init_state = 0;
}

static size_t LZMA_decodeToDic(LZMA2_DCtx *const p, size_t const dic_limit, const BYTE *src, size_t *const src_len,
    LZMA2_finishMode finish_mode)
{
    size_t in_size = *src_len;
    (*src_len) = 0;
    LZMA_writeRem(p, dic_limit);

    if (p->need_flush)
    {
        if (in_size < RC_INIT_SIZE)
        {
            return LZMA_STATUS_NEEDS_MORE_INPUT;
        }
        if (src[0] != 0)
            return FL2_ERROR(corruption_detected);
        p->code =
            ((U32)src[1] << 24)
            | ((U32)src[2] << 16)
            | ((U32)src[3] << 8)
            | ((U32)src[4]);
        src += RC_INIT_SIZE;
        (*src_len) += RC_INIT_SIZE;
        in_size -= RC_INIT_SIZE;
        p->range = 0xFFFFFFFF;
        p->need_flush = 0;
    }

    while (1) {
        if (p->dic_pos >= dic_limit)
        {
            if (p->remain_len == 0 && p->code == 0) {
                return LZMA_STATUS_FINISHED;
            }
            return LZMA_STATUS_NOT_FINISHED;
        }

        if (p->need_init_state)
            LZMA_initStateReal(p);

        const BYTE *buf_limit;
        if (finish_mode == LZMA_FINISH_END) {
            buf_limit = src + in_size;
        }
        else {
            if (in_size <= LZMA_REQUIRED_INPUT_MAX) {
                return LZMA_STATUS_NEEDS_MORE_INPUT;
            }
            buf_limit = src + in_size - LZMA_REQUIRED_INPUT_MAX;
        }
        p->buf = src;

        CHECK_F(LZMA_decodeReal2(p, dic_limit, buf_limit));

        size_t const processed = (size_t)(p->buf - src);
        (*src_len) += processed;
        src += processed;
        in_size -= processed;
    }
}

void LZMA_constructDCtx(LZMA2_DCtx *p)
{
    p->dic = NULL;
    p->ext_dic = 1;
    p->state2 = LZMA2_STATE_FINISHED;
	p->probs_1664 = p->probs + 1664;
}

static void LZMA_freeDict(LZMA2_DCtx *const p)
{
    if (!p->ext_dic) {
        free(p->dic);
    }
    p->dic = NULL;
}

void LZMA_destructDCtx(LZMA2_DCtx *const p)
{
    LZMA_freeDict(p);
}

size_t LZMA2_getDictSizeFromProp(BYTE const dict_prop)
{
    if (dict_prop > 40)
        return FL2_ERROR(corruption_detected);

    size_t const dict_size = (dict_prop == 40)
        ? (size_t)-1
        : (((size_t)2 | (dict_prop & 1)) << (dict_prop / 2 + 11));
    return dict_size;
}

static size_t LZMA2_dictBufSize(size_t const dict_size)
{
    size_t mask = ((size_t)1 << 12) - 1;
    if (dict_size >= ((size_t)1 << 30)) mask = ((size_t)1 << 22) - 1;
    else if (dict_size >= ((size_t)1 << 22)) mask = ((size_t)1 << 20) - 1;

    size_t dic_buf_size = ((size_t)dict_size + mask) & ~mask;
    if (dic_buf_size < dict_size)
        dic_buf_size = dict_size;

    return dic_buf_size;
}

size_t LZMA2_decMemoryUsage(size_t const dict_size)
{
    return sizeof(LZMA2_DCtx) + LZMA2_dictBufSize(dict_size);
}

size_t LZMA2_initDecoder(LZMA2_DCtx *const p, BYTE const dict_prop, BYTE *const dic, size_t dic_buf_size)
{
    size_t const dict_size = LZMA2_getDictSizeFromProp(dict_prop);
    if (FL2_isError(dict_size))
        return dict_size;

    if (dic == NULL) {
        dic_buf_size = LZMA2_dictBufSize(dict_size);

        if (p->dic == NULL || dic_buf_size != p->dic_buf_size) {
            LZMA_freeDict(p);
            p->dic = malloc(dic_buf_size);
            if (p->dic == NULL)
                return FL2_ERROR(memory_allocation);
            p->ext_dic = 0;
        }
    }
    else {
        LZMA_freeDict(p);
        p->dic = dic;
        p->ext_dic = 1;
    }
    p->dic_buf_size = dic_buf_size;
    p->prop.lc = 3;
    p->prop.lp = 0;
    p->prop.lc = 2;
    p->prop.dic_size = (U32)dict_size;

    p->state2 = LZMA2_STATE_CONTROL;
    p->need_init_dic = 1;
    p->need_init_state2 = 1;
    p->need_init_prop = 1;
    LZMA_init(p);
    return FL2_error_no_error;
}

static void LZMA_updateWithUncompressed(LZMA2_DCtx *const p, const BYTE *const src, size_t const size)
{
    memcpy(p->dic + p->dic_pos, src, size);
    p->dic_pos += size;
    if (p->check_dic_size == 0 && p->prop.dic_size - p->processed_pos <= size)
        p->check_dic_size = p->prop.dic_size;
    p->processed_pos += (U32)size;
}

static unsigned LZMA2_nextChunkInfo(BYTE *const control,
    size_t *const unpack_size, size_t *const pack_size,
    LZMA2_props *const prop,
    const BYTE *const src, ptrdiff_t *const src_len)
{
    ptrdiff_t const len = *src_len;
    *src_len = 0;
    if (len <= 0)
        return LZMA2_STATE_CONTROL;
    *control = *src;
    if (*control == 0) {
        *src_len = 1;
        return LZMA2_STATE_FINISHED;
    }
    if (len < 3)
        return LZMA2_STATE_CONTROL;
    if (LZMA2_IS_UNCOMPRESSED_STATE(*control)) {
        if (*control > 2)
            return LZMA2_STATE_ERROR;
        *src_len = 3;
        *unpack_size = (((size_t)src[1] << 8) | src[2]) + 1;
    }
    else {
        S32 const has_prop = LZMA2_IS_THERE_PROP(LZMA2_GET_LZMA_MODE(*control));
        if (len < 5 + has_prop)
            return LZMA2_STATE_CONTROL;
        *src_len = 5 + has_prop;
        *unpack_size = ((size_t)(*control & 0x1F) << 16) + ((size_t)src[1] << 8) + src[2] + 1;
        *pack_size = ((size_t)src[3] << 8) + src[4] + 1;
        if (has_prop) {
            BYTE b = src[5];
            if (b >= (9 * 5 * 5))
                return LZMA2_STATE_ERROR;
            BYTE const lc = b % 9;
            b /= 9;
            prop->pb = b / 5;
            BYTE const lp = b % 5;
            if (lc + lp > kLzma2LcLpMax)
                return LZMA2_STATE_ERROR;
            prop->lc = (BYTE)lc;
            prop->lp = (BYTE)lp;
        }
    }
    return LZMA2_STATE_DATA;
}

static size_t LZMA2_decodeChunkToDic(LZMA2_DCtx *const p, size_t const dic_limit,
    const BYTE *src, size_t *const src_len, LZMA2_finishMode const finish_mode)
{
    if (p->state2 == LZMA2_STATE_FINISHED)
        return LZMA_STATUS_FINISHED;

    size_t const in_size = *src_len;
    *src_len = 0;

    if (p->state2 == LZMA2_STATE_CONTROL) {
        ptrdiff_t len = in_size - *src_len;
        p->state2 = LZMA2_nextChunkInfo(&p->control, &p->unpack_size, &p->pack_size, &p->prop, src, &len);

        *src_len += len;
        src += len;

        if (p->state2 == LZMA2_STATE_ERROR)
            return FL2_ERROR(corruption_detected);
        else if (p->state2 == LZMA2_STATE_CONTROL)
            return LZMA_STATUS_NEEDS_MORE_INPUT;
        else if (p->state2 == LZMA2_STATE_FINISHED)
            return LZMA_STATUS_FINISHED;

        if (LZMA2_IS_UNCOMPRESSED_STATE(p->control))
        {
            BYTE const init_dic = (p->control == LZMA2_CONTROL_COPY_RESET_DIC);
            if (init_dic)
                p->need_init_prop = p->need_init_state2 = 1;
            else if (p->need_init_dic)
                return FL2_ERROR(corruption_detected);
            p->need_init_dic = 0;
            LZMA_initDicAndState(p, init_dic, 0);
        }
        else {
            unsigned const mode = LZMA2_GET_LZMA_MODE(p->control);
            BYTE const init_dic = (mode == 3);
            BYTE const init_state = (mode != 0);
            if ((!init_dic && p->need_init_dic) || (!init_state && p->need_init_state2))
                return FL2_ERROR(corruption_detected);

            LZMA_initDicAndState(p, init_dic, init_state);
            p->need_init_dic = 0;
            p->need_init_state2 = 0;
        }
    }
    if (p->state2 == LZMA2_STATE_DATA) {
        size_t in_cur = in_size - *src_len;
        size_t const dic_pos = p->dic_pos;
        size_t out_cur = dic_limit - dic_pos;

        if (out_cur == 0)
            return (finish_mode == LZMA_FINISH_ANY) ? LZMA_STATUS_OUTPUT_FULL : FL2_ERROR(dstSize_tooSmall);

        if (out_cur > p->unpack_size)
            out_cur = p->unpack_size;

        if (LZMA2_IS_UNCOMPRESSED_STATE(p->control)) {
            if (in_cur == 0)
                return LZMA_STATUS_NEEDS_MORE_INPUT;

            if (in_cur > out_cur)
                in_cur = out_cur;

            LZMA_updateWithUncompressed(p, src, in_cur);

            src += in_cur;
            *src_len += in_cur;
            p->unpack_size -= in_cur;
        }
        else
        {
            LZMA2_finishMode cur_finish_mode = LZMA_FINISH_END;
            if (in_cur < p->pack_size) {
                if (in_cur < LZMA_REQUIRED_INPUT_MAX)
                    return LZMA_STATUS_NEEDS_MORE_INPUT;
                cur_finish_mode = LZMA_FINISH_ANY;
            }
            else {
                in_cur = p->pack_size;
            }

            size_t res = LZMA_decodeToDic(p, dic_pos + out_cur, src, &in_cur, cur_finish_mode);

            src += in_cur;
            *src_len += in_cur;
            p->pack_size -= in_cur;
            p->unpack_size -= p->dic_pos - dic_pos;

            if (FL2_isError(res))
                return res;

            /* error if decoder not finished but chunk output is complete */
            if (res != LZMA_STATUS_FINISHED && p->unpack_size == 0)
                return FL2_ERROR(corruption_detected);

            /* Error conditions:
               1. need input but chunk is finished
               2. have output space, input not needed, but nothing was written*/
            if (res == LZMA_STATUS_NEEDS_MORE_INPUT)
                return (p->pack_size == 0) ? FL2_ERROR(corruption_detected) : res;
            else if (in_cur == 0 && p->dic_pos == dic_pos)
                return FL2_ERROR(corruption_detected);
        }

        if (p->unpack_size == 0)
            p->state2 = LZMA2_STATE_CONTROL;
    }
    return LZMA_STATUS_NOT_FINISHED;
}

size_t LZMA2_decodeToDic(LZMA2_DCtx *const p, size_t const dic_limit,
    const BYTE *const src, size_t *const src_len, LZMA2_finishMode const finish_mode)
{
    if (p->state2 == LZMA2_STATE_ERROR)
        return FL2_ERROR(corruption_detected);
    
    size_t const in_size = *src_len;
    size_t in_pos = 0;
    size_t res;

    do {
        size_t len = in_size - in_pos;
        res = LZMA2_decodeChunkToDic(p, dic_limit, src + in_pos, &len, finish_mode);
        in_pos += len;
        if (FL2_isError(res)) {
            p->state2 = LZMA2_STATE_ERROR;
            break;
        }
    } while (res != LZMA_STATUS_FINISHED
        && res != LZMA_STATUS_NEEDS_MORE_INPUT
        && res != LZMA_STATUS_OUTPUT_FULL);

    *src_len = in_pos;
    return res;
}

#if 0
size_t LZMA2_decodeToDic(LZMA2_DCtx *const p, size_t const dic_limit,
    const BYTE *src, size_t *const src_len, LZMA2_finishMode const finish_mode)
{
    size_t const in_size = *src_len;
    size_t res = FL2_error_no_error;
    *src_len = 0;

    while (p->state2 != LZMA2_STATE_ERROR)
    {
        if(p->state2 == LZMA2_STATE_CONTROL) {
            ptrdiff_t len = in_size - *src_len;
            p->state2 = LZMA2_nextChunkInfo(&p->control, &p->unpack_size, &p->pack_size, &p->prop, src, &len);
            *src_len += len;
            src += len;
        }

        if (p->state2 == LZMA2_STATE_FINISHED)
            return LZMA_STATUS_FINISHED;

        size_t const dic_pos = p->dic_pos;

        if (dic_pos == dic_limit && finish_mode == LZMA_FINISH_ANY)
            return LZMA_STATUS_NOT_FINISHED;

        if (p->state2 != LZMA2_STATE_DATA && p->state2 != LZMA2_STATE_DATA_CONT)
        {
            if (p->state2 == LZMA2_STATE_CONTROL)
            {
                return LZMA_STATUS_NEEDS_MORE_INPUT;
            }
            break;
        }

        size_t in_cur = in_size - *src_len;
        size_t out_cur = dic_limit - dic_pos;
        LZMA2_finishMode cur_finish_mode = LZMA_FINISH_ANY;

        if (out_cur >= p->unpack_size)
        {
            out_cur = (size_t)p->unpack_size;
        }
        if (in_cur >= p->pack_size)
            cur_finish_mode = LZMA_FINISH_END;

        if (LZMA2_IS_UNCOMPRESSED_STATE(p->control))
        {
            if (in_cur == 0)
            {
                return LZMA_STATUS_NEEDS_MORE_INPUT;
            }

            if (p->state2 == LZMA2_STATE_DATA)
            {
                BYTE const init_dic = (p->control == LZMA2_CONTROL_COPY_RESET_DIC);
                if (init_dic)
                    p->need_init_prop = p->need_init_state2 = 1;
                else if (p->need_init_dic)
                    break;
                p->need_init_dic = 0;
                LZMA_initDicAndState(p, init_dic, 0);
            }

            if (in_cur > out_cur)
                in_cur = out_cur;
            if (in_cur == 0)
                break;

            LZMA_updateWithUncompressed(p, src, in_cur);

            src += in_cur;
            *src_len += in_cur;
            p->unpack_size -= (U32)in_cur;
            p->state2 = (p->unpack_size == 0) ? LZMA2_STATE_CONTROL : LZMA2_STATE_DATA_CONT;
        }
        else
        {
            if (p->state2 == LZMA2_STATE_DATA)
            {
                unsigned const mode = LZMA2_GET_LZMA_MODE(p->control);
                BYTE const init_dic = (mode == 3);
                BYTE const init_state = (mode != 0);
                if ((!init_dic && p->need_init_dic) || (!init_state && p->need_init_state2))
                    break;

                LZMA_initDicAndState(p, init_dic, init_state);
                p->need_init_dic = 0;
                p->need_init_state2 = 0;
                p->state2 = LZMA2_STATE_DATA_CONT;
            }

            if (in_cur > p->pack_size)
                in_cur = (size_t)p->pack_size;

            res = LZMA_decodeToDic(p, dic_pos + out_cur, src, &in_cur, cur_finish_mode);

            src += in_cur;
            *src_len += in_cur;
            p->pack_size -= (U32)in_cur;
            out_cur = p->dic_pos - dic_pos;
            p->unpack_size -= (U32)out_cur;

            if (FL2_isError(res))
                break;

            if (res == LZMA_STATUS_NEEDS_MORE_INPUT)
            {
                if (p->pack_size == 0)
                    break;
                return res;
            }

            if (p->pack_size == 0 && p->unpack_size == 0)
            {
                if (res != LZMA_STATUS_FINISHED)
                    break;
                p->state2 = LZMA2_STATE_CONTROL;
            }
            else if (in_cur == 0 && out_cur == 0)
            {
                break;
            }

        }
    }

    p->state2 = LZMA2_STATE_ERROR;
    if (FL2_isError(res))
        return res;
    return FL2_ERROR(corruption_detected);
}
#endif

size_t LZMA2_decodeToBuf(LZMA2_DCtx *const p, BYTE *dest, size_t *const dest_len, const BYTE *src, size_t *const src_len, LZMA2_finishMode const finish_mode)
{
    size_t out_size = *dest_len, in_size = *src_len;
    *src_len = *dest_len = 0;

    for (;;)
    {
        if (p->dic_pos == p->dic_buf_size)
            p->dic_pos = 0;

        size_t const dic_pos = p->dic_pos;
        LZMA2_finishMode cur_finish_mode = LZMA_FINISH_ANY;
        size_t out_cur = p->dic_buf_size - dic_pos;

        if (out_cur >= out_size) {
            out_cur = out_size;
            cur_finish_mode = finish_mode;
        }

        size_t in_cur = in_size;
        size_t const res = LZMA2_decodeToDic(p, dic_pos + out_cur, src, &in_cur, cur_finish_mode);

        src += in_cur;
        in_size -= in_cur;
        *src_len += in_cur;
        out_cur = p->dic_pos - dic_pos;
        memcpy(dest, p->dic + dic_pos, out_cur);
        dest += out_cur;
        out_size -= out_cur;
        *dest_len += out_cur;
        if (FL2_isError(res) || res == LZMA_STATUS_FINISHED)
            return res;
        if (out_cur == 0 || out_size == 0)
            return FL2_error_no_error;
    }
}

U64 LZMA2_getUnpackSize(const BYTE *const src, size_t const src_len)
{
    U64 unpack_total = 0;
    size_t pos = 1;
    while (pos < src_len) {
        LZMA2_chunk inf;
        int const type = LZMA2_parseInput(src, pos, src_len - pos, &inf);
        if (type == CHUNK_FINAL)
            return unpack_total;
        pos += inf.pack_size;
        if (type == CHUNK_ERROR || type == CHUNK_MORE_DATA)
            break;
        unpack_total += inf.unpack_size;
    }
    return LZMA2_CONTENTSIZE_ERROR;
}

LZMA2_parseRes LZMA2_parseInput(const BYTE* const in_buf, size_t const pos, ptrdiff_t const len, LZMA2_chunk *const inf)
{
    inf->pack_size = 0;
    inf->unpack_size = 0;

    if (len <= 0)
        return CHUNK_ERROR;

    BYTE const control = in_buf[pos];
    if (control == 0) {
        inf->pack_size = 1;
        return CHUNK_FINAL;
    }
    if (len < 3)
        return CHUNK_MORE_DATA;
    if (LZMA2_IS_UNCOMPRESSED_STATE(control)) {
        if (control > 2)
            return CHUNK_ERROR;
        inf->unpack_size = (((U32)in_buf[pos + 1] << 8) | in_buf[pos + 2]) + 1;
        inf->pack_size = 3 + inf->unpack_size;
    }
    else {
        S32 const has_prop = LZMA2_IS_THERE_PROP(LZMA2_GET_LZMA_MODE(control));
        if (len < 5 + has_prop)
            return CHUNK_MORE_DATA;
        inf->unpack_size = ((U32)(control & 0x1F) << 16) + ((U32)in_buf[pos + 1] << 8) + in_buf[pos + 2] + 1;
        inf->pack_size = 5 + has_prop + ((U32)in_buf[pos + 3] << 8) + in_buf[pos + 4] + 1;
        if (LZMA2_GET_LZMA_MODE(control) == 3)
            return CHUNK_DICT_RESET;
    }
    return CHUNK_CONTINUE;
}
