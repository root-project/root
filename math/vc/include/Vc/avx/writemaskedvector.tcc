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

namespace ROOT {
namespace Vc
{
namespace AVX
{

template<typename T>
Vc_ALWAYS_INLINE Vector<T> &WriteMaskedVector<T>::operator++()
{
    vec->data() = VectorHelper<T>::add(vec->data(),
            VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
            );
    return *vec;
}

template<typename T>
Vc_ALWAYS_INLINE Vector<T> &WriteMaskedVector<T>::operator--() {
    vec->data() = VectorHelper<T>::sub(vec->data(),
            VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
            );
    return *vec;
}

template<typename T>
Vc_ALWAYS_INLINE Vector<T> WriteMaskedVector<T>::operator++(int) {
    Vector<T> ret(*vec);
    vec->data() = VectorHelper<T>::add(vec->data(),
            VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
            );
    return ret;
}

template<typename T>
Vc_ALWAYS_INLINE Vector<T> WriteMaskedVector<T>::operator--(int) {
    Vector<T> ret(*vec);
    vec->data() = VectorHelper<T>::sub(vec->data(),
            VectorHelper<T>::notMaskedToZero(VectorHelper<T>::one(), mask.data())
            );
    return ret;
}

template<typename T>
Vc_ALWAYS_INLINE Vector<T> &WriteMaskedVector<T>::operator+=(const Vector<T> &x) {
    vec->data() = VectorHelper<T>::add(vec->data(), VectorHelper<T>::notMaskedToZero(x.data(), mask.data()));
    return *vec;
}

template<typename T>
Vc_ALWAYS_INLINE Vector<T> &WriteMaskedVector<T>::operator-=(const Vector<T> &x) {
    vec->data() = VectorHelper<T>::sub(vec->data(), VectorHelper<T>::notMaskedToZero(x.data(), mask.data()));
    return *vec;
}

template<typename T>
Vc_ALWAYS_INLINE Vector<T> &WriteMaskedVector<T>::operator*=(const Vector<T> &x) {
    vec->assign(VectorHelper<T>::mul(vec->data(), x.data()), mask);
    return *vec;
}

template<typename T>
Vc_ALWAYS_INLINE Vector<T> &WriteMaskedVector<T>::operator/=(const Vector<T> &x) {
    vec->assign(*vec / x, mask);
    return *vec;
}

template<typename T>
Vc_ALWAYS_INLINE Vector<T> &WriteMaskedVector<T>::operator=(const Vector<T> &x) {
    vec->assign(x, mask);
    return *vec;
}

} // namespace AVX
} // namespace Vc
} // namespace ROOT
