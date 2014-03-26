/*  This file is part of the Vc library.

    Copyright (C) 2009-2012 Matthias Kretz <kretz@kde.org>

    Vc is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as
    published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    Vc is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with Vc.  If not, see <http://www.gnu.org/licenses/>.

*/

#include "unittest.h"
#include <iostream>
#include "vectormemoryhelper.h"
#include <Vc/cpuid.h>

using namespace Vc;

template<typename Vec> void testSort()
{
    typedef typename Vec::IndexType IndexType;

    const IndexType _ref(IndexesFromZero);
    Vec ref(_ref);
    Vec a;
    int maxPerm = 1;
    for (int x = Vec::Size; x > 0; --x) {
        maxPerm *= x;
    }
    for (int perm = 0; perm < maxPerm; ++perm) {
        int rest = perm;
        for (int i = 0; i < Vec::Size; ++i) {
            a[i] = 0;
            for (int j = 0; j < i; ++j) {
                if (a[i] == a[j]) {
                    ++(a[i]);
                    j = -1;
                }
            }
            a[i] += rest % (Vec::Size - i);
            rest /= (Vec::Size - i);
            for (int j = 0; j < i; ++j) {
                if (a[i] == a[j]) {
                    ++(a[i]);
                    j = -1;
                }
            }
        }
        //std::cout << a << a.sorted() << std::endl;
        COMPARE(ref, a.sorted()) << ", a: " << a;
    }

    for (int repetition = 0; repetition < 1000; ++repetition) {
        Vec test = Vec::Random();
        Vc::Memory<Vec, Vec::Size> reference;
        reference.vector(0) = test;
        std::sort(&reference[0], &reference[Vec::Size]);
        ref = reference.vector(0);
        COMPARE(ref, test.sorted());
    }
}

template<typename T, typename Mem> struct Foo
{
    Foo() : i(0) {}
    void reset() { i = 0; }
    void operator()(T v) { d[i++] = v; }
    Mem d;
    int i;
};

template<typename V> void testCall()
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    typedef typename V::Mask M;
    typedef typename I::Mask MI;
    const I _indexes(IndexesFromZero);
    const MI _odd = (_indexes & I(One)) > 0;
    const M odd(_odd);
    V a(_indexes);
    Foo<T, typename V::Memory> f;
    a.callWithValuesSorted(f);
    V b(f.d);
    COMPARE(b, a);

    f.reset();
    a(odd) -= 1;
    a.callWithValuesSorted(f);
    V c(f.d);
    for (int i = 0; i < V::Size / 2; ++i) {
        COMPARE(a[i * 2], c[i]);
    }
    for (int i = V::Size / 2; i < V::Size; ++i) {
        COMPARE(b[i], c[i]);
    }
}

template<typename V> void testForeachBit()
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    const I indexes(IndexesFromZero);
    for_all_masks(V, mask) {
        V tmp = V::Zero();
        foreach_bit(int j, mask) {
            tmp[j] = T(1);
        }
        COMPARE(tmp == V::One(), mask);

        int count = 0;
        foreach_bit(int j, mask) {
            ++count;
            if (j >= 0) {
                continue;
            }
        }
        COMPARE(count, mask.count());

        count = 0;
        foreach_bit(int j, mask) {
            if (j >= 0) {
                break;
            }
            ++count;
        }
        COMPARE(count, 0);
    }
}

template<typename V> void copySign()
{
    V v(One);
    V positive(One);
    V negative = -positive;
    COMPARE(v, v.copySign(positive));
    COMPARE(-v, v.copySign(negative));
}

#ifdef _WIN32
void bzero(void *p, size_t n) { memset(p, 0, n); }
#else
#include <strings.h>
#endif

template<typename V> void Random()
{
    typedef typename V::EntryType T;
    enum {
        NBits = 3,
        NBins = 1 << NBits,                        // short int
        TotalBits = sizeof(T) * 8,                 //    16  32
        RightShift = TotalBits - NBits,            //    13  29
        NHistograms = TotalBits - NBits + 1,       //    14  30
        LeftShift = (RightShift + 1) / NHistograms,//     1   1
        Mean = 135791,
        MinGood = Mean - Mean/10,
        MaxGood = Mean + Mean/10
    };
    const V mask((1 << NBits) - 1);
    int histogram[NHistograms][NBins];
    bzero(&histogram[0][0], sizeof(histogram));
    for (size_t i = 0; i < NBins * Mean / V::Size; ++i) {
        const V rand = V::Random();
        for (size_t hist = 0; hist < NHistograms; ++hist) {
            const V bin = ((rand << (hist * LeftShift)) >> RightShift) & mask;
            for (size_t k = 0; k < V::Size; ++k) {
                ++histogram[hist][bin[k]];
            }
        }
    }
//#define PRINT_RANDOM_HISTOGRAM
#ifdef PRINT_RANDOM_HISTOGRAM
    for (size_t hist = 0; hist < NHistograms; ++hist) {
        std::cout << "histogram[" << std::setw(2) << hist << "]: ";
        for (size_t bin = 0; bin < NBins; ++bin) {
            std::cout << std::setw(3) << (histogram[hist][bin] - Mean) * 1000 / Mean << "|";
        }
        std::cout << std::endl;
    }
#endif
    for (size_t hist = 0; hist < NHistograms; ++hist) {
        for (size_t bin = 0; bin < NBins; ++bin) {
            VERIFY(histogram[hist][bin] > MinGood)
                << " bin = " << bin << " is " << histogram[0][bin];
            VERIFY(histogram[hist][bin] < MaxGood)
                << " bin = " << bin << " is " << histogram[0][bin];
        }
    }
}

template<typename V, typename I> void FloatRandom()
{
    typedef typename V::EntryType T;
    enum {
        NBins = 64,
        NHistograms = 1,
        Mean = 135791,
        MinGood = Mean - Mean/10,
        MaxGood = Mean + Mean/10
    };
    int histogram[NHistograms][NBins];
    bzero(&histogram[0][0], sizeof(histogram));
    for (size_t i = 0; i < NBins * Mean / V::Size; ++i) {
        const V rand = V::Random();
        const I bin = static_cast<I>(rand * T(NBins));
        for (size_t k = 0; k < V::Size; ++k) {
            ++histogram[0][bin[k]];
        }
    }
#ifdef PRINT_RANDOM_HISTOGRAM
    for (size_t hist = 0; hist < NHistograms; ++hist) {
        std::cout << "histogram[" << std::setw(2) << hist << "]: ";
        for (size_t bin = 0; bin < NBins; ++bin) {
            std::cout << std::setw(3) << (histogram[hist][bin] - Mean) * 1000 / Mean << "|";
        }
        std::cout << std::endl;
    }
#endif
    for (size_t hist = 0; hist < NHistograms; ++hist) {
        for (size_t bin = 0; bin < NBins; ++bin) {
            VERIFY(histogram[hist][bin] > MinGood)
                << " bin = " << bin << " is " << histogram[0][bin];
            VERIFY(histogram[hist][bin] < MaxGood)
                << " bin = " << bin << " is " << histogram[0][bin];
        }
    }
}

template<> void Random<float_v>() { FloatRandom<float_v, int_v>(); }
template<> void Random<double_v>() { FloatRandom<double_v, int_v>(); }
template<> void Random<sfloat_v>() { FloatRandom<sfloat_v, short_v>(); }

template<typename T> T add2(T x) { return x + T(2); }

template<typename T, typename V>
class CallTester
{
    public:
        CallTester() : v(Vc::Zero), i(0) {}

        void operator()(T x) {
            v[i] = x;
            ++i;
        }

        void reset() { v.setZero(); i = 0; }

        int callCount() const { return i; }
        V callValues() const { return v; }

    private:
        V v;
        int i;
};

#if __cplusplus >= 201103 && (!defined(VC_CLANG) || VC_CLANG > 0x30000)
#define DO_LAMBDA_TESTS 1
#endif

template<typename V>
void applyAndCall()
{
    typedef typename V::EntryType T;

    const V two(T(2));
    for (int i = 0; i < 1000; ++i) {
        const V rand = V::Random();
        COMPARE(rand.apply(add2<T>), rand + two);
#ifdef DO_LAMBDA_TESTS
        COMPARE(rand.apply([](T x) { return x + T(2); }), rand + two);
#endif

        CallTester<T, V> callTester;
        rand.call(callTester);
        COMPARE(callTester.callCount(), int(V::Size));
        COMPARE(callTester.callValues(), rand);

        for_all_masks(V, mask) {
            V copy1 = rand;
            V copy2 = rand;
            copy1(mask) += two;

            COMPARE(copy2(mask).apply(add2<T>), copy1) << mask;
            COMPARE(rand.apply(add2<T>, mask), copy1) << mask;
#ifdef DO_LAMBDA_TESTS
            COMPARE(copy2(mask).apply([](T x) { return x + T(2); }), copy1) << mask;
            COMPARE(rand.apply([](T x) { return x + T(2); }, mask), copy1) << mask;
#endif

            callTester.reset();
            copy2(mask).call(callTester);
            COMPARE(callTester.callCount(), mask.count());

            callTester.reset();
            rand.call(callTester, mask);
            COMPARE(callTester.callCount(), mask.count());
        }
    }
}

template<typename T, int value> T returnConstant() { return T(value); }
template<typename T, int value> T returnConstantOffset(int i) { return T(value) + T(i); }
template<typename T, int value> T returnConstantOffset2(unsigned short i) { return T(value) + T(i); }

template<typename V> void fill()
{
    typedef typename V::EntryType T;
    typedef typename V::IndexType I;
    V test = V::Random();
    test.fill(returnConstant<T, 2>);
    COMPARE(test, V(T(2)));

    test = V::Random();
    test.fill(returnConstantOffset<T, 0>);
    COMPARE(test, static_cast<V>(I::IndexesFromZero()));

    test = V::Random();
    test.fill(returnConstantOffset2<T, 0>);
    COMPARE(test, static_cast<V>(I::IndexesFromZero()));
}

template<typename V> void shifted()
{
    typedef typename V::EntryType T;
    for (int shift = -2 * V::Size; shift <= 2 * V::Size; ++shift) {
        const V reference = V::Random();
        const V test = reference.shifted(shift);
        for (int i = 0; i < V::Size; ++i) {
            if (i + shift >= 0 && i + shift < V::Size) {
                COMPARE(test[i], reference[i + shift]) << "shift: " << shift << ", i: " << i << ", test: " << test << ", reference: " << reference;
            } else {
                COMPARE(test[i], T(0)) << "shift: " << shift << ", i: " << i << ", test: " << test << ", reference: " << reference;
            }
        }
    }
}

template<typename V> void rotated()
{
    for (int shift = -2 * V::Size; shift <= 2 * V::Size; ++shift) {
        //std::cout << "amount = " << shift % V::Size << std::endl;
        const V reference = V::Random();
        const V test = reference.rotated(shift);
        for (int i = 0; i < V::Size; ++i) {
            unsigned int refShift = i + shift;
            COMPARE(test[i], reference[refShift % V::Size]) << "shift: " << shift << ", i: " << i << ", test: " << test << ", reference: " << reference;
        }
    }
}

void testMallocAlignment()
{
    int_v *a = Vc::malloc<int_v, Vc::AlignOnVector>(10);

    unsigned long mask = VectorAlignment - 1;
    for (int i = 0; i < 10; ++i) {
        VERIFY((reinterpret_cast<unsigned long>(&a[i]) & mask) == 0);
    }
    const char *data = reinterpret_cast<const char *>(&a[0]);
    for (int i = 0; i < 10; ++i) {
        VERIFY(&data[i * int_v::Size * sizeof(int_v::EntryType)] == reinterpret_cast<const char *>(&a[i]));
    }

    a = Vc::malloc<int_v, Vc::AlignOnCacheline>(10);
    mask = CpuId::cacheLineSize() - 1;
    COMPARE((reinterpret_cast<unsigned long>(&a[0]) & mask), 0ul);

    // I don't know how to properly check page alignment. So we check for 4 KiB alignment as this is
    // the minimum page size on x86
    a = Vc::malloc<int_v, Vc::AlignOnPage>(10);
    mask = 4096 - 1;
    COMPARE((reinterpret_cast<unsigned long>(&a[0]) & mask), 0ul);
}

int main()
{
    testAllTypes(testCall);
    testAllTypes(testForeachBit);
    testAllTypes(testSort);
    testRealTypes(copySign);

    testAllTypes(shifted);
    testAllTypes(rotated);
    testAllTypes(Random);

    testAllTypes(applyAndCall);
    testAllTypes(fill);

    runTest(testMallocAlignment);

    return 0;
}
