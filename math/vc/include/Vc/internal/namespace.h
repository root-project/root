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

#ifdef VC_IMPL_Scalar
# define VECTOR_NAMESPACE Vc::Scalar
#elif defined(VC_IMPL_AVX)
# define VECTOR_NAMESPACE Vc::AVX
#elif defined(VC_IMPL_SSE)
# define VECTOR_NAMESPACE Vc::SSE
#else
# error "No known Vc implementation was selected. This should not happen. The logic in Vc/global.h failed."
#endif
