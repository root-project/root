/*  This file is part of the Vc library.

    Copyright (C) 2010-2011 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SCALAR_HELPERIMPL_TCC
#define VC_SCALAR_HELPERIMPL_TCC

#include <cstdlib>
#if defined _WIN32 || defined _WIN64
#include <malloc.h>
#endif

namespace ROOT {
namespace Vc
{
namespace Internal
{

template<size_t X>
static _VC_CONSTEXPR size_t nextMultipleOf(size_t value)
{
    return (value % X) > 0 ? value + X - (value % X) : value;
}

template<Vc::MallocAlignment A>
Vc_ALWAYS_INLINE void *HelperImpl<ScalarImpl>::malloc(size_t n)
{
    void *ptr = 0;
    switch (A) {
        case Vc::AlignOnVector:
            return std::malloc(n);
        case Vc::AlignOnCacheline:
            // TODO: hardcoding 64 is not such a great idea
#ifdef _WIN32
#ifdef __GNUC__
#define _VC_ALIGNED_MALLOC __mingw_aligned_malloc
#else
#define _VC_ALIGNED_MALLOC _aligned_malloc
#endif
            ptr = _VC_ALIGNED_MALLOC(nextMultipleOf<64>(n), 64);
#else
            if (0 == posix_memalign(&ptr, 64, nextMultipleOf<64>(n))) {
                return ptr;
            }
#endif
            break;
        case Vc::AlignOnPage:
            // TODO: hardcoding 4096 is not such a great idea
#ifdef _WIN32
            ptr = _VC_ALIGNED_MALLOC(nextMultipleOf<4096>(n), 4096);
#undef _VC_ALIGNED_MALLOC
#else
            if (0 == posix_memalign(&ptr, 4096, nextMultipleOf<4096>(n))) {
                return ptr;
            }
#endif
            break;
    }
    return ptr;
}

Vc_ALWAYS_INLINE void HelperImpl<ScalarImpl>::free(void *p)
{
    std::free(p);
}

} // namespace Internal
} // namespace Vc
} // namespace ROOT

#endif // VC_SCALAR_HELPERIMPL_TCC
