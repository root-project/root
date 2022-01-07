// Author: Enrico Guiraud, Enric Tejedor, Danilo Piparo CERN  04/2021
// Implementation adapted from from llvm::SmallVector.
// See /math/vecops/ARCHITECTURE.md for more information.

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RVec.hxx"
using namespace ROOT::VecOps;

// Check that no bytes are wasted and everything is well-aligned.
namespace {
struct Struct16B {
   alignas(16) void *X;
};
struct Struct32B {
   alignas(32) void *X;
};
}
static_assert(alignof(RVec<Struct16B>) >= alignof(Struct16B), "wrong alignment for 16-byte aligned T");
static_assert(alignof(RVec<Struct32B>) >= alignof(Struct32B), "wrong alignment for 32-byte aligned T");
static_assert(sizeof(RVec<Struct16B>) >= alignof(Struct16B), "missing padding for 16-byte aligned T");
static_assert(sizeof(RVec<Struct32B>) >= alignof(Struct32B), "missing padding for 32-byte aligned T");
static_assert(sizeof(RVecN<void *, 0>) == sizeof(int32_t) * 2 + sizeof(void *), "wasted space in RVec");
static_assert(sizeof(RVecN<void *, 1>) == sizeof(int32_t) * 2 + sizeof(void *) * 2,
              "wasted space in SmallVector size 1");
static_assert(sizeof(RVec<void *>) ==
                 sizeof(int32_t) * 2 +
                    (1 + ROOT::Internal::VecOps::RVecInlineStorageSize<void *>::value) * sizeof(void *),
              "wasted space in RVec");

void ROOT::Internal::VecOps::SmallVectorBase::report_size_overflow(size_t MinSize)
{
   std::string Reason = "RVec unable to grow. Requested capacity (" + std::to_string(MinSize) +
                        ") is larger than maximum value for size type (" + std::to_string(SizeTypeMax()) + ")";
   throw std::length_error(Reason);
}

void ROOT::Internal::VecOps::SmallVectorBase::report_at_maximum_capacity()
{
   std::string Reason = "RVec capacity unable to grow. Already at maximum size " + std::to_string(SizeTypeMax());
   throw std::length_error(Reason);
}

// Note: Moving this function into the header may cause performance regression.
void ROOT::Internal::VecOps::SmallVectorBase::grow_pod(void *FirstEl, size_t MinSize, size_t TSize)
{
   // Ensure we can fit the new capacity.
   // This is only going to be applicable when the capacity is 32 bit.
   if (MinSize > SizeTypeMax())
      report_size_overflow(MinSize);

   // Ensure we can meet the guarantee of space for at least one more element.
   // The above check alone will not catch the case where grow is called with a
   // default MinSize of 0, but the current capacity cannot be increased.
   // This is only going to be applicable when the capacity is 32 bit.
   if (capacity() == SizeTypeMax())
      report_at_maximum_capacity();

   // In theory 2*capacity can overflow if the capacity is 64 bit, but the
   // original capacity would never be large enough for this to be a problem.
   size_t NewCapacity = 2 * capacity() + 1; // Always grow.
   NewCapacity = std::min(std::max(NewCapacity, MinSize), SizeTypeMax());

   void *NewElts;
   if (fBeginX == FirstEl || !this->Owns()) {
      NewElts = malloc(NewCapacity * TSize);
      R__ASSERT(NewElts != nullptr);

      // Copy the elements over.  No need to run dtors on PODs.
      memcpy(NewElts, this->fBeginX, size() * TSize);
   } else {
      // If this wasn't grown from the inline copy, grow the allocated space.
      NewElts = realloc(this->fBeginX, NewCapacity * TSize);
      R__ASSERT(NewElts != nullptr);
   }

   this->fBeginX = NewElts;
   this->fCapacity = NewCapacity;
}

#if (_VECOPS_USE_EXTERN_TEMPLATES)

namespace ROOT {
namespace VecOps {

#define RVEC_DECLARE_UNARY_OPERATOR(T, OP) \
   template RVec<T> operator OP(const RVec<T> &);

#define RVEC_DECLARE_BINARY_OPERATOR(T, OP)                                              \
   template auto operator OP(const RVec<T> &v, const T &y) -> RVec<decltype(v[0] OP y)>; \
   template auto operator OP(const T &x, const RVec<T> &v) -> RVec<decltype(x OP v[0])>; \
   template auto operator OP(const RVec<T> &v0, const RVec<T> &v1) -> RVec<decltype(v0[0] OP v1[0])>;

#define RVEC_DECLARE_LOGICAL_OPERATOR(T, OP)                   \
   template RVec<int> operator OP(const RVec<T> &, const T &); \
   template RVec<int> operator OP(const T &, const RVec<T> &); \
   template RVec<int> operator OP(const RVec<T> &, const RVec<T> &);

#define RVEC_DECLARE_ASSIGN_OPERATOR(T, OP)             \
   template RVec<T> &operator OP(RVec<T> &, const T &); \
   template RVec<T> &operator OP(RVec<T> &, const RVec<T> &);

#define RVEC_DECLARE_FLOAT_TEMPLATE(T)  \
   template class RVec<T>;              \
   RVEC_DECLARE_UNARY_OPERATOR(T, +)    \
   RVEC_DECLARE_UNARY_OPERATOR(T, -)    \
   RVEC_DECLARE_UNARY_OPERATOR(T, !)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, +)   \
   RVEC_DECLARE_BINARY_OPERATOR(T, -)   \
   RVEC_DECLARE_BINARY_OPERATOR(T, *)   \
   RVEC_DECLARE_BINARY_OPERATOR(T, /)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, +=)  \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, -=)  \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, *=)  \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, /=)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, <)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, >)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, ==) \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, !=) \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, <=) \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, >=) \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, &&) \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, ||)

#define RVEC_DECLARE_INTEGER_TEMPLATE(T) \
   template class RVec<T>;               \
   RVEC_DECLARE_UNARY_OPERATOR(T, +)     \
   RVEC_DECLARE_UNARY_OPERATOR(T, -)     \
   RVEC_DECLARE_UNARY_OPERATOR(T, ~)     \
   RVEC_DECLARE_UNARY_OPERATOR(T, !)     \
   RVEC_DECLARE_BINARY_OPERATOR(T, +)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, -)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, *)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, /)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, %)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, &)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, |)    \
   RVEC_DECLARE_BINARY_OPERATOR(T, ^)    \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, +=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, -=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, *=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, /=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, %=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, &=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, |=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, ^=)   \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, >>=)  \
   RVEC_DECLARE_ASSIGN_OPERATOR(T, <<=)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, <)   \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, >)   \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, ==)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, !=)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, <=)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, >=)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, &&)  \
   RVEC_DECLARE_LOGICAL_OPERATOR(T, ||)

RVEC_DECLARE_INTEGER_TEMPLATE(char)
RVEC_DECLARE_INTEGER_TEMPLATE(short)
RVEC_DECLARE_INTEGER_TEMPLATE(int)
RVEC_DECLARE_INTEGER_TEMPLATE(long)
RVEC_DECLARE_INTEGER_TEMPLATE(long long)

RVEC_DECLARE_INTEGER_TEMPLATE(unsigned char)
RVEC_DECLARE_INTEGER_TEMPLATE(unsigned short)
RVEC_DECLARE_INTEGER_TEMPLATE(unsigned int)
RVEC_DECLARE_INTEGER_TEMPLATE(unsigned long)
RVEC_DECLARE_INTEGER_TEMPLATE(unsigned long long)

RVEC_DECLARE_FLOAT_TEMPLATE(float)
RVEC_DECLARE_FLOAT_TEMPLATE(double)

#define RVEC_DECLARE_UNARY_FUNCTION(T, NAME, FUNC) \
   template RVec<PromoteType<T>> NAME(const RVec<T> &);

#define RVEC_DECLARE_STD_UNARY_FUNCTION(T, F) RVEC_DECLARE_UNARY_FUNCTION(T, F, ::std::F)

#define RVEC_DECLARE_BINARY_FUNCTION(T0, T1, NAME, FUNC) \
   template RVec<PromoteTypes<T0, T1>> NAME(const RVec<T0> &v, const T1 &y); \
   template RVec<PromoteTypes<T0, T1>> NAME(const T0 &x, const RVec<T1> &v); \
   template RVec<PromoteTypes<T0, T1>> NAME(const RVec<T0> &v0, const RVec<T1> &v1);

#define RVEC_DECLARE_STD_BINARY_FUNCTION(T, F) RVEC_DECLARE_BINARY_FUNCTION(T, T, F, ::std::F)

#define RVEC_DECLARE_STD_FUNCTIONS(T)             \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, abs)        \
   RVEC_DECLARE_STD_BINARY_FUNCTION(T, fdim)      \
   RVEC_DECLARE_STD_BINARY_FUNCTION(T, fmod)      \
   RVEC_DECLARE_STD_BINARY_FUNCTION(T, remainder) \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, exp)        \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, exp2)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, expm1)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, log)        \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, log10)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, log2)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, log1p)      \
   RVEC_DECLARE_STD_BINARY_FUNCTION(T, pow)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, sqrt)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, cbrt)       \
   RVEC_DECLARE_STD_BINARY_FUNCTION(T, hypot)     \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, sin)        \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, cos)        \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, tan)        \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, asin)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, acos)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, atan)       \
   RVEC_DECLARE_STD_BINARY_FUNCTION(T, atan2)     \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, sinh)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, cosh)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, tanh)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, asinh)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, acosh)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, atanh)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, floor)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, ceil)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, trunc)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, round)      \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, lround)     \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, llround)    \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, erf)        \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, erfc)       \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, lgamma)     \
   RVEC_DECLARE_STD_UNARY_FUNCTION(T, tgamma)     \

RVEC_DECLARE_STD_FUNCTIONS(float)
RVEC_DECLARE_STD_FUNCTIONS(double)
#undef RVEC_DECLARE_STD_UNARY_FUNCTION
#undef RVEC_DECLARE_STD_BINARY_FUNCTION
#undef RVEC_DECLARE_STD_UNARY_FUNCTIONS

#ifdef R__HAS_VDT

#define RVEC_DECLARE_VDT_UNARY_FUNCTION(T, F)    \
   RVEC_DECLARE_UNARY_FUNCTION(T, F, vdt::F)

RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_expf)
RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_logf)
RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_sinf)
RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_cosf)
RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_tanf)
RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_asinf)
RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_acosf)
RVEC_DECLARE_VDT_UNARY_FUNCTION(float, fast_atanf)

RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_exp)
RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_log)
RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_sin)
RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_cos)
RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_tan)
RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_asin)
RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_acos)
RVEC_DECLARE_VDT_UNARY_FUNCTION(double, fast_atan)

#endif // R__HAS_VDT

} // namespace VecOps
} // namespace ROOT

#endif // _VECOPS_USE_EXTERN_TEMPLATES
