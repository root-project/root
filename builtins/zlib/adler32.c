/* adler32.c -- compute the Adler-32 checksum of a data stream
 * Copyright (C) 1995-2011 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* @(#) $Id$ */


#include "zutil.h"

#if defined (__x86_64__) && defined (__linux__)
#include <xmmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>

#include "cpuid.h"
#endif

static uLong adler32_combine_ OF((uLong adler1, uLong adler2, z_off64_t len2));

#define BASE 65521      /* largest prime smaller than 65536 */
#define NMAX 5552
/* NMAX is the largest n such that 255n(n+1)/2 + (n+1)(BASE-1) <= 2^32-1 */

/* 
 * As we are using _signed_ integer arithmetic for the SSE/AVX2 implementations,
 * we consider the max as 2^31-1
 */
#define NMAX_VEC 5552

#define NMAX_VEC2 5552

#define DO1(buf,i)  {adler += (buf)[i]; sum2 += adler;}
#define DO2(buf,i)  DO1(buf,i); DO1(buf,i+1);
#define DO4(buf,i)  DO2(buf,i); DO2(buf,i+2);
#define DO8(buf,i)  DO4(buf,i); DO4(buf,i+4);
#define DO16(buf)   DO8(buf,0); DO8(buf,8);

/* use NO_DIVIDE if your processor does not do division in hardware --
   try it both ways to see which is faster */
#ifdef NO_DIVIDE
/* note that this assumes BASE is 65521, where 65536 % 65521 == 15
   (thank you to John Reiser for pointing this out) */
#  define CHOP(a) \
    do { \
        unsigned long tmp = a >> 16; \
        a &= 0xffffUL; \
        a += (tmp << 4) - tmp; \
    } while (0)
#  define MOD28(a) \
    do { \
        CHOP(a); \
        if (a >= BASE) a -= BASE; \
    } while (0)
#  define MOD(a) \
    do { \
        CHOP(a); \
        MOD28(a); \
    } while (0)
#  define MOD63(a) \
    do { /* this assumes a is not negative */ \
        z_off64_t tmp = a >> 32; \
        a &= 0xffffffffL; \
        a += (tmp << 8) - (tmp << 5) + tmp; \
        tmp = a >> 16; \
        a &= 0xffffL; \
        a += (tmp << 4) - tmp; \
        tmp = a >> 16; \
        a &= 0xffffL; \
        a += (tmp << 4) - tmp; \
        if (a >= BASE) a -= BASE; \
    } while (0)
#else
#  define MOD(a) a %= BASE
#  define MOD28(a) a %= BASE
#  define MOD63(a) a %= BASE
#endif

/* ========================================================================= */
uLong ZEXPORT adler32_default(uLong adler, const Bytef *buf, uInt len)
{	
    unsigned long sum2;
    unsigned n;

    /* split Adler-32 into component sums */
    sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;

    /* in case user likes doing a byte at a time, keep it fast */
    if (len == 1) {
        adler += buf[0];
        if (adler >= BASE)
            adler -= BASE;
        sum2 += adler;
        if (sum2 >= BASE)
            sum2 -= BASE;
        return adler | (sum2 << 16);
    }

    /* initial Adler-32 value (deferred check for len == 1 speed) */
    if (buf == Z_NULL)
        return 1L;

    /* in case short lengths are provided, keep it somewhat fast */
    if (len < 16) {
        while (len--) {
            adler += *buf++;
            sum2 += adler;
        }
        if (adler >= BASE)
            adler -= BASE;
        MOD28(sum2);            /* only added so many BASE's */
        return adler | (sum2 << 16);
    }

    /* do length NMAX blocks -- requires just one modulo operation */
    while (len >= NMAX) {
        len -= NMAX;
        n = NMAX / 16;          /* NMAX is divisible by 16 */
        do {
            DO16(buf);          /* 16 sums unrolled */
            buf += 16;
        } while (--n);
        MOD(adler);
        MOD(sum2);
    }

    /* do remaining bytes (less than NMAX, still just one modulo) */
    if (len) {                  /* avoid modulos if none remaining */
        while (len >= 16) {
            len -= 16;
            DO16(buf);
            buf += 16;
        }
        while (len--) {
            adler += *buf++;
            sum2 += adler;
        }
        MOD(adler);
        MOD(sum2);
    }

    /* return recombined sums */
    return adler | (sum2 << 16);
}

#if defined (__x86_64__) && defined (__linux__)

#ifdef _MSC_VER

/* MSC doesn't have __builtin_expect.  Just ignore likely/unlikely and
   hope the compiler optimizes for the best.
*/
#define likely(x)       (x)
#define unlikely(x)     (x)

#else

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

#endif

/* ========================================================================= */
 __attribute__ ((target ("sse4.2")))
uLong ZEXPORT adler32_sse42(uLong adler, const Bytef *buf, uInt len)
{
    unsigned long sum2;

    /* split Adler-32 into component sums */
    sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;

    /* in case user likes doing a byte at a time, keep it fast */
    if (unlikely(len == 1)) {
        adler += buf[0];
        if (adler >= BASE)
            adler -= BASE;
        sum2 += adler;
        if (sum2 >= BASE)
            sum2 -= BASE;
        return adler | (sum2 << 16);
    }

    /* initial Adler-32 value (deferred check for len == 1 speed) */
    if (unlikely(buf == Z_NULL))
        return 1L;

    /* in case short lengths are provided, keep it somewhat fast */
    if (unlikely(len < 16)) {
        while (len--) {
            adler += *buf++;
            sum2 += adler;
        }
        if (adler >= BASE)
            adler -= BASE;
        MOD28(sum2);            /* only added so many BASE's */
        return adler | (sum2 << 16);
    }

    uint32_t __attribute__ ((aligned(16))) s1[4], s2[4];
    s1[0] = s1[1] = s1[2] = 0; s1[3] = adler;
    s2[0] = s2[1] = s2[2] = 0; s2[3] = sum2;
    char __attribute__ ((aligned(16))) dot1[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    __m128i dot1v = _mm_load_si128((__m128i*)dot1);
    char __attribute__ ((aligned(16))) dot2[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    __m128i dot2v = _mm_load_si128((__m128i*)dot2);
    short __attribute__ ((aligned(16))) dot3[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    __m128i dot3v = _mm_load_si128((__m128i*)dot3);
    // We will need to multiply by 
    //char __attribute__ ((aligned(16))) shift[4] = {0, 0, 0, 4}; //{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4};
    char __attribute__ ((aligned(16))) shift[16] = {4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    __m128i shiftv = _mm_load_si128((__m128i*)shift);
    while (len >= 16) {
       __m128i vs1 = _mm_load_si128((__m128i*)s1);
       __m128i vs2 = _mm_load_si128((__m128i*)s2);
       __m128i vs1_0 = vs1;
       int k = (len < NMAX_VEC ? (int)len : NMAX_VEC);
       k -= k % 16;
       len -= k;
       while (k >= 16) {
           /*
              vs1 = adler + sum(c[i])
              vs2 = sum2 + 16 vs1 + sum( (16-i+1) c[i] )

              NOTE: 256-bit equivalents are:
                _mm256_maddubs_epi16 <- operates on 32 bytes to 16 shorts
                _mm256_madd_epi16    <- Sums 16 shorts to 8 int32_t.
              We could rewrite the below to use 256-bit instructions instead of 128-bit.
           */
           __m128i vbuf = _mm_loadu_si128((__m128i*)buf);
           buf += 16;
           k -= 16;
           __m128i v_short_sum1 = _mm_maddubs_epi16(vbuf, dot1v); // multiply-add, resulting in 8 shorts.
           __m128i vsum1 = _mm_madd_epi16(v_short_sum1, dot3v);  // sum 8 shorts to 4 int32_t;
           __m128i v_short_sum2 = _mm_maddubs_epi16(vbuf, dot2v);
           vs1 = _mm_add_epi32(vsum1, vs1);
           __m128i vsum2 = _mm_madd_epi16(v_short_sum2, dot3v);
           vs1_0 = _mm_sll_epi32(vs1_0, shiftv);
           vsum2 = _mm_add_epi32(vsum2, vs2);
           vs2   = _mm_add_epi32(vsum2, vs1_0);
           vs1_0 = vs1;
       }
       // At this point, we have partial sums stored in vs1 and vs2.  There are AVX512 instructions that
       // would allow us to sum these quickly (VP4DPWSSD).  For now, just unpack and move on.
       uint32_t __attribute__((aligned(16))) s1_unpack[4];
       uint32_t __attribute__((aligned(16))) s2_unpack[4];
       _mm_store_si128((__m128i*)s1_unpack, vs1);
       _mm_store_si128((__m128i*)s2_unpack, vs2);
       adler = (s1_unpack[0] % BASE) + (s1_unpack[1] % BASE) + (s1_unpack[2] % BASE) + (s1_unpack[3] % BASE);
       MOD(adler);
       s1[3] = adler;
       sum2 = (s2_unpack[0] % BASE) + (s2_unpack[1] % BASE) + (s2_unpack[2] % BASE) + (s2_unpack[3] % BASE);
       MOD(sum2);
       s2[3] = sum2;
    }

    while (len--) {
       adler += *buf++;
       sum2 += adler;
    }
    MOD(adler);
    MOD(sum2);

    /* return recombined sums */
    return adler | (sum2 << 16);
}

/* ========================================================================= */
__attribute__ ((target ("avx2")))
uLong ZEXPORT adler32_avx2(uLong adler, const Bytef *buf, uInt len)
{
    unsigned long sum2;

    /* split Adler-32 into component sums */
    sum2 = (adler >> 16) & 0xffff;
    adler &= 0xffff;

    /* in case user likes doing a byte at a time, keep it fast */
    if (unlikely(len == 1)) {
        adler += buf[0];
        if (adler >= BASE)
            adler -= BASE;
        sum2 += adler;
        if (sum2 >= BASE)
            sum2 -= BASE;
        return adler | (sum2 << 16);
    }

    /* initial Adler-32 value (deferred check for len == 1 speed) */
    if (unlikely(buf == Z_NULL))
        return 1L;

    /* in case short lengths are provided, keep it somewhat fast */
    if (unlikely(len < 32)) {
        while (len--) {
            adler += *buf++;
            sum2 += adler;
        }
        if (adler >= BASE)
            adler -= BASE;
        MOD28(sum2);            /* only added so many BASE's */
        return adler | (sum2 << 16);
    }

    uint32_t __attribute__ ((aligned(32))) s1[8], s2[8];
    memset(s1, '\0', sizeof(uint32_t)*7); s1[7] = adler; // TODO: would a masked load be faster?
    memset(s2, '\0', sizeof(uint32_t)*7); s2[7] = sum2;
    char __attribute__ ((aligned(32))) dot1[32] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    __m256i dot1v = _mm256_load_si256((__m256i*)dot1);
    char __attribute__ ((aligned(32))) dot2[32] = {32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    __m256i dot2v = _mm256_load_si256((__m256i*)dot2);
    short __attribute__ ((aligned(32))) dot3[16] = {1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1};
    __m256i dot3v = _mm256_load_si256((__m256i*)dot3);
    // We will need to multiply by 
    char __attribute__ ((aligned(16))) shift[16] = {5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    __m128i shiftv = _mm_load_si128((__m128i*)shift);
    while (len >= 32) {
       __m256i vs1 = _mm256_load_si256((__m256i*)s1);
       __m256i vs2 = _mm256_load_si256((__m256i*)s2);
       __m256i vs1_0 = vs1;
       int k = (len < NMAX_VEC ? (int)len : NMAX_VEC);
       k -= k % 32;
       len -= k;
       while (k >= 32) {
           /*
              vs1 = adler + sum(c[i])
              vs2 = sum2 + 16 vs1 + sum( (16-i+1) c[i] )
           */
           __m256i vbuf = _mm256_loadu_si256((__m256i*)buf);
           buf += 32;
           k -= 32;
           __m256i v_short_sum1 = _mm256_maddubs_epi16(vbuf, dot1v); // multiply-add, resulting in 8 shorts.
           __m256i vsum1 = _mm256_madd_epi16(v_short_sum1, dot3v);  // sum 8 shorts to 4 int32_t;
           __m256i v_short_sum2 = _mm256_maddubs_epi16(vbuf, dot2v);
           vs1 = _mm256_add_epi32(vsum1, vs1);
           __m256i vsum2 = _mm256_madd_epi16(v_short_sum2, dot3v);
           vs1_0 = _mm256_sll_epi32(vs1_0, shiftv);
           vsum2 = _mm256_add_epi32(vsum2, vs2);
           vs2   = _mm256_add_epi32(vsum2, vs1_0);
           vs1_0 = vs1;
       }
       // At this point, we have partial sums stored in vs1 and vs2.  There are AVX512 instructions that
       // would allow us to sum these quickly (VP4DPWSSD).  For now, just unpack and move on.
       uint32_t __attribute__((aligned(32))) s1_unpack[8];
       uint32_t __attribute__((aligned(32))) s2_unpack[8];
       _mm256_store_si256((__m256i*)s1_unpack, vs1);
       _mm256_store_si256((__m256i*)s2_unpack, vs2);
       adler = (s1_unpack[0] % BASE) + (s1_unpack[1] % BASE) + (s1_unpack[2] % BASE) + (s1_unpack[3] % BASE) + (s1_unpack[4] % BASE) + (s1_unpack[5] % BASE) + (s1_unpack[6] % BASE) + (s1_unpack[7] % BASE);
       MOD(adler);
       s1[7] = adler;
       sum2 = (s2_unpack[0] % BASE) + (s2_unpack[1] % BASE) + (s2_unpack[2] % BASE) + (s2_unpack[3] % BASE) + (s2_unpack[4] % BASE) + (s2_unpack[5] % BASE) + (s2_unpack[6] % BASE) + (s2_unpack[7] % BASE);
       MOD(sum2);
       s2[7] = sum2;
    }

    while (len--) {
       adler += *buf++;
       sum2 += adler;
    }
    MOD(adler);
    MOD(sum2);

    /* return recombined sums */
    return adler | (sum2 << 16);
}

void *resolve_adler32(void)
{
  unsigned int eax, ebx, ecx, edx;
	signed char has_sse42 = 0;
	signed char has_avx2 = 0;

	/* Collect CPU features */
  if (!__get_cpuid (1, &eax, &ebx, &ecx, &edx))
    return adler32_default;
	has_sse42 = ((ecx & bit_SSE4_2) != 0);
#if defined(bit_AVX2)
	if (__get_cpuid_max (0, NULL) < 7)
		return adler32_default;
	__cpuid_count (7, 0, eax, ebx, ecx, edx);
	has_avx2 = ((ebx & bit_AVX2) != 0);
#endif /* defined(bit_AVX2) */

	/* Pick AVX2 version */
	if (has_avx2)
		return adler32_avx2;

  /* Pick SSE4.2 version */
  if (has_sse42)
    return adler32_sse42;

	/* Fallback to default implementation */
  return adler32_default;
}

uLong ZEXPORT adler32(uLong adler, const Bytef *buf, uInt len)  __attribute__ ((ifunc ("resolve_adler32")));
#else // x86_64
uLong ZEXPORT adler32(uLong adler, const Bytef *buf, uInt len){
  return adler32_default(adler, buf, len);
}
#endif

/* ========================================================================= */
static uLong adler32_combine_(uLong adler1, uLong adler2, z_off64_t len2)
{
    unsigned long sum1;
    unsigned long sum2;
    unsigned rem;

    /* for negative len, return invalid adler32 as a clue for debugging */
    if (len2 < 0)
        return 0xffffffffUL;

    /* the derivation of this formula is left as an exercise for the reader */
    MOD63(len2);                /* assumes len2 >= 0 */
    rem = (unsigned)len2;
    sum1 = adler1 & 0xffff;
    sum2 = rem * sum1;
    MOD(sum2);
    sum1 += (adler2 & 0xffff) + BASE - 1;
    sum2 += ((adler1 >> 16) & 0xffff) + ((adler2 >> 16) & 0xffff) + BASE - rem;
    if (sum1 >= BASE) sum1 -= BASE;
    if (sum1 >= BASE) sum1 -= BASE;
    if (sum2 >= (BASE << 1)) sum2 -= (BASE << 1);
    if (sum2 >= BASE) sum2 -= BASE;
    return sum1 | (sum2 << 16);
}

/* ========================================================================= */
uLong adler32_combine(uLong adler1, uLong adler2, z_off_t len2)
{
    return adler32_combine_(adler1, adler2, len2);
}

uLong adler32_combine64(uLong adler1, uLong adler2, z_off64_t len2)
{
    return adler32_combine_(adler1, adler2, len2);
}
