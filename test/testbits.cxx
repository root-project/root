/*
 * testbits.cxx -- unit tests for the new functionality in the TBits class
 * by Filip Pizlo, 2004
 */

#include "TROOT.h"
#include "TBits.h"

#include <stdlib.h>
#include <stdio.h>

static const char *current_test;
static unsigned test_count;

#define A(exp) do {\
    if (!(exp)) {\
        fprintf(stderr," FAILURE!\n\nIn %s:\n\n",current_test);\
        fprintf(stderr,"Assertion (%s) failed at %s:%d.\n\n",\
                       #exp,__FILE__,__LINE__);\
        exit(1);\
    }\
} while (0)

#define DO_TEST(exp) do {\
    current_test=#exp;\
    fprintf(stderr,".");\
    fflush(stderr);\
    exp;\
    ++test_count;\
} while (0)

static void test_set_bit() {
    TBits b;
    b.SetBitNumber(0);
    A(b.TestBitNumber(0));
}

static const Char_t char_bits[] = {
    0,
    1,
    3,
    7,
    15,
    31,
    63,
    127
};

static const Short_t short_bits[] = {
    0 + 1 * 256,
    3 + 7 * 256,
    15 + 31 * 256,
    63 + 127 * 256
};

static const Int_t int_bits[] = {
    0 + 1 * 256 + 3 * 65536 + 7 * 16777216,
    15 + 31 * 256 + 63 * 65536 + 127 * 16777216
};

static void set_bits_by_char(TBits &bits,
                             UInt_t nbits) {
    bits.Set(nbits, char_bits);
}

static void assert_bits_by_char(const TBits &bits,
                                UInt_t nbits) {
    Char_t buf[8];
    memset(buf,0,sizeof(buf));
    bits.Get(buf);
    for (UInt_t i=0;i<nbits;++i) {
        A((buf[i>>3]&(1<<(i&7))) == (char_bits[i>>3]&(1<<(i&7))));
    }
}

static void set_bits_by_short(TBits &bits,
                              UInt_t nbits) {
    bits.Set(nbits, short_bits);
}

static void assert_bits_by_short(const TBits &bits,
                                 UInt_t nbits) {
    Short_t buf[4];
    memset(buf,0,sizeof(buf));
    bits.Get(buf);
    for (UInt_t i=0;i<nbits;++i) {
        A((buf[i>>4]&(1<<(i&15))) == (short_bits[i>>4]&(1<<(i&15))));
    }
}

static void set_bits_by_int(TBits &bits,
                            UInt_t nbits) {
    bits.Set(nbits, int_bits);
}

static void assert_bits_by_int(const TBits &bits,
                               UInt_t nbits) {
    Int_t buf[2];
    memset(buf,0,sizeof(buf));
    bits.Get(buf);
    for (UInt_t i=0;i<nbits;++i) {
        A((buf[i>>5]&(1<<(i&31))) == (int_bits[i>>5]&(1<<(i&31))));
    }
}

static void set_bits(TBits &bits,
                     UInt_t nbits) {
    UInt_t i,j;
    for (i=0;
         i<8;
         ++i) {
        for (j=i*8;
             j<i*8+i && j<nbits;
             ++j) {
            bits.SetBitNumber(j);
        }
        for (j=i*8+i;
             j<i*8+8 && j<nbits;
             ++j) {
            bits.ResetBitNumber(j);
        }
    }
}

static void assert_bits(const TBits &bits,
                        UInt_t nbits) {
    UInt_t i,j;
    for (i=0;
         i<8;
         ++i) {
        for (j=i*8;
             j<i*8+i && j<nbits;
             ++j) {
            A(bits.TestBitNumber(j));
        }
        for (j=i*8+i;
             j<i*8+8 && j<nbits;
             ++j) {
            A(!bits.TestBitNumber(j));
        }
    }
}

static void test_set_from_char(UInt_t nbits) {
    TBits b;
    set_bits_by_char(b,nbits);
    assert_bits(b,nbits);
}

static void test_set_from_short(UInt_t nbits) {
    TBits b;
    set_bits_by_short(b,nbits);
    assert_bits(b,nbits);
}

static void test_set_from_int(UInt_t nbits) {
    TBits b;
    set_bits_by_int(b,nbits);
    assert_bits(b,nbits);
}

static void test_get_to_char(UInt_t nbits) {
    TBits b;
    set_bits(b,nbits);
    assert_bits_by_char(b,nbits);
}

static void test_get_to_short(UInt_t nbits) {
    TBits b;
    set_bits(b,nbits);
    assert_bits_by_short(b,nbits);
}

static void test_get_to_int(UInt_t nbits) {
    TBits b;
    set_bits(b,nbits);
    assert_bits_by_int(b,nbits);
}

// don't have access to a 64-bit machine at the moment so long test would be
// pointless...

int main(int /*c*/,char **v) {
    TROOT app("bits_test", "Tests the TBits class's new functionality");
    
    fprintf(stderr,"%s: ",v[0]);
    
    DO_TEST(test_set_bit());
    
    DO_TEST(test_set_from_char(64));
    DO_TEST(test_set_from_char(64-1));
    DO_TEST(test_set_from_char(64-7));
    DO_TEST(test_set_from_char(64-8));
    
    DO_TEST(test_set_from_short(64));
    DO_TEST(test_set_from_short(64-1));
    DO_TEST(test_set_from_short(64-7));
    DO_TEST(test_set_from_short(64-8));
    DO_TEST(test_set_from_short(64-15));
    DO_TEST(test_set_from_short(64-16));
    
    DO_TEST(test_set_from_int(64));
    DO_TEST(test_set_from_int(64-1));
    DO_TEST(test_set_from_int(64-7));
    DO_TEST(test_set_from_int(64-8));
    DO_TEST(test_set_from_int(64-15));
    DO_TEST(test_set_from_int(64-16));
    DO_TEST(test_set_from_int(64-23));
    DO_TEST(test_set_from_int(64-24));
    DO_TEST(test_set_from_int(64-31));
    DO_TEST(test_set_from_int(64-32));
    
    DO_TEST(test_get_to_char(64));
    DO_TEST(test_get_to_char(64-1));
    DO_TEST(test_get_to_char(64-7));
    DO_TEST(test_get_to_char(64-8));
    
    DO_TEST(test_get_to_short(64));
    DO_TEST(test_get_to_short(64-1));
    DO_TEST(test_get_to_short(64-7));
    DO_TEST(test_get_to_short(64-8));
    DO_TEST(test_get_to_short(64-15));
    DO_TEST(test_get_to_short(64-16));
    
    DO_TEST(test_get_to_int(64));
    DO_TEST(test_get_to_int(64-1));
    DO_TEST(test_get_to_int(64-7));
    DO_TEST(test_get_to_int(64-8));
    DO_TEST(test_get_to_int(64-15));
    DO_TEST(test_get_to_int(64-16));
    DO_TEST(test_get_to_int(64-23));
    DO_TEST(test_get_to_int(64-24));
    DO_TEST(test_get_to_int(64-31));
    DO_TEST(test_get_to_int(64-32));
    
    fprintf(stderr," OK! (%d tests)\n",test_count);
    
    return 0;
}


