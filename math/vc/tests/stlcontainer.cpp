/*{{{
    Copyright (C) 2012 Matthias Kretz <kretz@kde.org>

    Permission to use, copy, modify, and distribute this software
    and its documentation for any purpose and without fee is hereby
    granted, provided that the above copyright notice appear in all
    copies and that both that the copyright notice and this
    permission notice and warranty disclaimer appear in supporting
    documentation, and that the name of the author not be used in
    advertising or publicity pertaining to distribution of the
    software without specific, written prior permission.

    The author disclaim all warranties with regard to this
    software, including all implied warranties of merchantability
    and fitness.  In no event shall the author be liable for any
    special, indirect or consequential damages or any damages
    whatsoever resulting from loss of use, data or profits, whether
    in an action of contract, negligence or other tortious action,
    arising out of or in connection with the use or performance of
    this software.

}}}*/

#include <Vc/Allocator>
#include <vector>
#include "unittest.h"

#include "Vc/common/macros.h"

template<typename Vec> size_t alignmentMask()
{
    if (Vec::Size == 1) {
        // on 32bit the maximal alignment is 4 Bytes, even for 8-Byte doubles.
        return std::min(sizeof(void*), sizeof(typename Vec::EntryType)) - 1;
    }
    // sizeof(SSE::sfloat_v) is too large
    // AVX::VectorAlignment is too large
    return std::min<size_t>(sizeof(Vec), Vc::VectorAlignment) - 1;
}

template<typename T> struct SomeStruct { char a; T x; };

template<typename V> void stdVectorAlignment()
{
    const size_t mask = alignmentMask<V>();
    const char *const null = 0;

    std::vector<V> v(11);
    for (int i = 0; i < 11; ++i) {
        COMPARE((reinterpret_cast<char *>(&v[i]) - null) & mask, 0u) << "&v[i] = " << &v[i] << ", mask = " << mask << ", i = " << i;
    }

    std::vector<SomeStruct<V>, Vc::Allocator<SomeStruct<V> > > v2(11);
    for (int i = 0; i < 11; ++i) {
        COMPARE((reinterpret_cast<char *>(&v2[i]) - null) & mask, 0u) << "&v2[i] = " << &v2[i] << ", mask = " << mask << ", i = " << i;
    }

    std::vector<V> v3(v);
    std::vector<SomeStruct<V>, Vc::Allocator<SomeStruct<V> > > v4(v2);

    typedef typename V::EntryType T;
    for (int i = 1; i < 100; ++i) {
        std::vector<T, Vc::Allocator<T> > v5(i);
        const size_t expectedAlignment = Vc_ALIGNOF(V);
        COMPARE((&v5[0] - static_cast<const T *>(0)) * sizeof(T) & (expectedAlignment - 1), 0u);
    }
}

int main(int argc, char **argv)
{
    initTest(argc, argv);

    using namespace Vc;
    testAllTypes(stdVectorAlignment);
}
