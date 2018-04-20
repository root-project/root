#include "ROOT/TVec.hxx"

namespace ROOT {
namespace Experimental {
namespace VecOps {

#define TVEC_DECLARE_UNARY_OPERATOR(T, OP) \
   template TVec<T> operator OP(const TVec<T> &);

#define TVEC_DECLARE_BINARY_OPERATOR(T, OP)                                         \
   template TVec<decltype((T) {} OP(T){})> operator OP(const TVec<T> &, const T &); \
   template TVec<decltype((T) {} OP(T){})> operator OP(const T &, const TVec<T> &); \
   template TVec<decltype((T) {} OP(T){})> operator OP(const TVec<T> &, const TVec<T> &);

#define TVEC_DECLARE_LOGICAL_OPERATOR(T, OP)                   \
   template TVec<int> operator OP(const TVec<T> &, const T &); \
   template TVec<int> operator OP(const T &, const TVec<T> &); \
   template TVec<int> operator OP(const TVec<T> &, const TVec<T> &);

#define TVEC_DECLARE_ASSIGN_OPERATOR(T, OP)             \
   template TVec<T> &operator OP(TVec<T> &, const T &); \
   template TVec<T> &operator OP(TVec<T> &, const TVec<T> &);

#define TVEC_DECLARE_FLOAT_TEMPLATE(T)  \
   template class TVec<T>;              \
   TVEC_DECLARE_UNARY_OPERATOR(T, +)    \
   TVEC_DECLARE_UNARY_OPERATOR(T, -)    \
   TVEC_DECLARE_UNARY_OPERATOR(T, !)    \
   TVEC_DECLARE_BINARY_OPERATOR(T, +)   \
   TVEC_DECLARE_BINARY_OPERATOR(T, -)   \
   TVEC_DECLARE_BINARY_OPERATOR(T, *)   \
   TVEC_DECLARE_BINARY_OPERATOR(T, /)   \
   TVEC_DECLARE_ASSIGN_OPERATOR(T, +=)  \
   TVEC_DECLARE_ASSIGN_OPERATOR(T, -=)  \
   TVEC_DECLARE_ASSIGN_OPERATOR(T, *=)  \
   TVEC_DECLARE_ASSIGN_OPERATOR(T, /=)  \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, <)  \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, >)  \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, ==) \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, !=) \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, <=) \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, >=) \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, &&) \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, ||)

#define TVEC_DECLARE_INTEGER_TEMPLATE(T) \
   template class TVec<T>;               \
   TVEC_DECLARE_UNARY_OPERATOR(T, +)     \
   TVEC_DECLARE_UNARY_OPERATOR(T, -)     \
   TVEC_DECLARE_UNARY_OPERATOR(T, ~)     \
   TVEC_DECLARE_UNARY_OPERATOR(T, !)     \
   TVEC_DECLARE_BINARY_OPERATOR(T, +)    \
   TVEC_DECLARE_BINARY_OPERATOR(T, -)    \
   TVEC_DECLARE_BINARY_OPERATOR(T, *)    \
   TVEC_DECLARE_BINARY_OPERATOR(T, /)    \
   TVEC_DECLARE_BINARY_OPERATOR(T, %)    \
   TVEC_DECLARE_BINARY_OPERATOR(T, &)    \
   TVEC_DECLARE_BINARY_OPERATOR(T, |)    \
   TVEC_DECLARE_BINARY_OPERATOR(T, ^)    \
   TVEC_DECLARE_ASSIGN_OPERATOR(T, +=)   \
   TVEC_DECLARE_ASSIGN_OPERATOR(T, -=)   \
   TVEC_DECLARE_ASSIGN_OPERATOR(T, *=)   \
   TVEC_DECLARE_ASSIGN_OPERATOR(T, /=)   \
   TVEC_DECLARE_ASSIGN_OPERATOR(T, %=)   \
   TVEC_DECLARE_ASSIGN_OPERATOR(T, &=)   \
   TVEC_DECLARE_ASSIGN_OPERATOR(T, |=)   \
   TVEC_DECLARE_ASSIGN_OPERATOR(T, ^=)   \
   TVEC_DECLARE_ASSIGN_OPERATOR(T, >>=)  \
   TVEC_DECLARE_ASSIGN_OPERATOR(T, <<=)  \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, <)   \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, >)   \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, ==)  \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, !=)  \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, <=)  \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, >=)  \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, &&)  \
   TVEC_DECLARE_LOGICAL_OPERATOR(T, ||)

TVEC_DECLARE_INTEGER_TEMPLATE(char)
TVEC_DECLARE_INTEGER_TEMPLATE(short)
TVEC_DECLARE_INTEGER_TEMPLATE(int)
TVEC_DECLARE_INTEGER_TEMPLATE(long)
TVEC_DECLARE_INTEGER_TEMPLATE(long long)

TVEC_DECLARE_INTEGER_TEMPLATE(unsigned char)
TVEC_DECLARE_INTEGER_TEMPLATE(unsigned short)
TVEC_DECLARE_INTEGER_TEMPLATE(unsigned int)
TVEC_DECLARE_INTEGER_TEMPLATE(unsigned long)
TVEC_DECLARE_INTEGER_TEMPLATE(unsigned long long)

TVEC_DECLARE_FLOAT_TEMPLATE(float)
TVEC_DECLARE_FLOAT_TEMPLATE(double)

} // namespace VecOps
} // namespace Experimental
} // namespace ROOT
