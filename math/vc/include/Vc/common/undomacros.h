/*  This file is part of the Vc library.

    Copyright (C) 2010-2012 Matthias Kretz <kretz@kde.org>

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

#ifndef VC_COMMON_UNDOMACROS_H
#define VC_COMMON_UNDOMACROS_H
#undef VC_COMMON_MACROS_H

#undef Vc_ALIGNOF

#undef Vc_INTRINSIC
#undef Vc_INTRINSIC_L
#undef Vc_INTRINSIC_R
#undef Vc_CONST
#undef Vc_CONST_L
#undef Vc_CONST_R
#undef Vc_PURE
#undef Vc_PURE_L
#undef Vc_PURE_R
#undef Vc_MAY_ALIAS
#undef Vc_ALWAYS_INLINE
#undef Vc_ALWAYS_INLINE_L
#undef Vc_ALWAYS_INLINE_R
#undef VC_IS_UNLIKELY
#undef VC_IS_LIKELY
#undef VC_RESTRICT
#undef VC_DEPRECATED
#undef _VC_CONSTEXPR
#undef _VC_CONSTEXPR_L
#undef _VC_CONSTEXPR_R
#undef _VC_NOEXCEPT

#undef ALIGN
#undef STRUCT_ALIGN1
#undef STRUCT_ALIGN2
#undef ALIGNED_TYPEDEF
#undef _CAT_IMPL
#undef CAT
#undef unrolled_loop16
#undef for_all_vector_entries
#undef FREE_STORE_OPERATORS_ALIGNED

#undef VC_WARN_INLINE
#undef VC_WARN

#ifdef VC_EXTERNAL_ASSERT
#undef VC_EXTERNAL_ASSERT
#else
#undef VC_ASSERT
#endif

#undef VC_HAS_BUILTIN

#undef Vc_buildDouble
#undef Vc_buildFloat

#undef _VC_APPLY_IMPL_1
#undef _VC_APPLY_IMPL_2
#undef _VC_APPLY_IMPL_3
#undef _VC_APPLY_IMPL_4
#undef _VC_APPLY_IMPL_5

#undef VC_LIST_FLOAT_VECTOR_TYPES
#undef VC_LIST_INT_VECTOR_TYPES
#undef VC_LIST_VECTOR_TYPES
#undef VC_LIST_COMPARES
#undef VC_LIST_LOGICAL
#undef VC_LIST_BINARY
#undef VC_LIST_SHIFTS
#undef VC_LIST_ARITHMETICS

#undef VC_APPLY_0
#undef VC_APPLY_1
#undef VC_APPLY_2
#undef VC_APPLY_3
#undef VC_APPLY_4

#undef VC_ALL_COMPARES
#undef VC_ALL_LOGICAL
#undef VC_ALL_BINARY
#undef VC_ALL_SHIFTS
#undef VC_ALL_ARITHMETICS
#undef VC_ALL_FLOAT_VECTOR_TYPES
#undef VC_ALL_VECTOR_TYPES

#undef VC_EXACT_TYPE
#undef VC_ALIGNED_PARAMETER
#undef VC_OFFSETOF

#ifdef Vc_POP_GCC_DIAGNOSTIC__
#pragma GCC diagnostic pop
#undef Vc_POP_GCC_DIAGNOSTIC__
#endif

#endif // VC_COMMON_UNDOMACROS_H
