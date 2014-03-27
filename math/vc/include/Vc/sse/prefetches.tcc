/*  This file is part of the Vc library.

    Copyright (C) 2010 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_SSE_PREFETCHES_TCC
#define VC_SSE_PREFETCHES_TCC

namespace ROOT {
namespace Vc
{
namespace Internal
{

Vc_ALWAYS_INLINE void HelperImpl<Vc::SSE2Impl>::prefetchForOneRead(const void *addr)
{
    _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_NTA);
}
Vc_ALWAYS_INLINE void HelperImpl<Vc::SSE2Impl>::prefetchClose(const void *addr)
{
    _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_T0);
}
Vc_ALWAYS_INLINE void HelperImpl<Vc::SSE2Impl>::prefetchMid(const void *addr)
{
    _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_T1);
}
Vc_ALWAYS_INLINE void HelperImpl<Vc::SSE2Impl>::prefetchFar(const void *addr)
{
    _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_T2);
}
Vc_ALWAYS_INLINE void HelperImpl<Vc::SSE2Impl>::prefetchForModify(const void *addr)
{
#if defined(__3dNOW__) && (!defined(VC_CLANG) || VC_CLANG >= 0x30200)
    _m_prefetchw(const_cast<void *>(addr));
#else
    _mm_prefetch(static_cast<char *>(const_cast<void *>(addr)), _MM_HINT_T0);
#endif
}

} // namespace Internal
} // namespace Vc
} // namespace ROOT

#endif // VC_SSE_PREFETCHES_TCC
