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

#ifndef VC_AVX_WRITEMASKEDVECTOR_H
#define VC_AVX_WRITEMASKEDVECTOR_H

#include "macros.h"
namespace ROOT {
namespace Vc
{
namespace AVX
{

template<typename T>
class WriteMaskedVector
{
    friend class Vector<T>;
    typedef typename VectorTypeHelper<T>::Type VectorType;
    typedef typename DetermineEntryType<T>::Type EntryType;
    enum Constants { Size = sizeof(VectorType) / sizeof(EntryType) };
    typedef typename Vc::AVX::Mask<Size, sizeof(VectorType)> Mask;
    public:
        FREE_STORE_OPERATORS_ALIGNED(32)
        //prefix
        Vector<T> Vc_ALWAYS_INLINE_L &operator++() Vc_ALWAYS_INLINE_R;
        Vector<T> Vc_ALWAYS_INLINE_L &operator--() Vc_ALWAYS_INLINE_R;
        //postfix
        Vector<T> Vc_ALWAYS_INLINE_L operator++(int) Vc_ALWAYS_INLINE_R;
        Vector<T> Vc_ALWAYS_INLINE_L operator--(int) Vc_ALWAYS_INLINE_R;

        Vector<T> Vc_ALWAYS_INLINE_L &operator+=(const Vector<T> &x) Vc_ALWAYS_INLINE_R;
        Vector<T> Vc_ALWAYS_INLINE_L &operator-=(const Vector<T> &x) Vc_ALWAYS_INLINE_R;
        Vector<T> Vc_ALWAYS_INLINE_L &operator*=(const Vector<T> &x) Vc_ALWAYS_INLINE_R;
        Vector<T> Vc_ALWAYS_INLINE_L &operator/=(const Vector<T> &x) Vc_ALWAYS_INLINE_R;
        Vector<T> Vc_ALWAYS_INLINE &operator+=(EntryType x) { return operator+=(Vector<T>(x)); }
        Vector<T> Vc_ALWAYS_INLINE &operator-=(EntryType x) { return operator-=(Vector<T>(x)); }
        Vector<T> Vc_ALWAYS_INLINE &operator*=(EntryType x) { return operator*=(Vector<T>(x)); }
        Vector<T> Vc_ALWAYS_INLINE &operator/=(EntryType x) { return operator/=(Vector<T>(x)); }

        Vector<T> Vc_ALWAYS_INLINE_L &operator=(const Vector<T> &x) Vc_ALWAYS_INLINE_R;
        Vector<T> Vc_ALWAYS_INLINE &operator=(EntryType x) { return operator=(Vector<T>(x)); }

        template<typename F> Vc_INTRINSIC void call(const F &f) const {
            return vec->call(f, mask);
        }
        template<typename F> Vc_INTRINSIC void call(F &f) const {
            return vec->call(f, mask);
        }
        template<typename F> Vc_INTRINSIC Vector<T> apply(const F &f) const {
            return vec->apply(f, mask);
        }
        template<typename F> Vc_INTRINSIC Vector<T> apply(F &f) const {
            return vec->apply(f, mask);
        }
    private:
        Vc_ALWAYS_INLINE WriteMaskedVector(Vector<T> *v, const Mask &k) : vec(v), mask(k) {}
        Vector<T> *const vec;
        Mask mask;
};

} // namespace AVX
} // namespace Vc
} // namespace ROOT
#include "writemaskedvector.tcc"
#include "undomacros.h"
#endif // VC_AVX_WRITEMASKEDVECTOR_H
