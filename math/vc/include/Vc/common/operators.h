#ifndef VC_ICC
// ICC ICEs if the following type-traits are in the anonymous namespace
namespace
{
#endif
template<typename Cond, typename T> struct EnableIfNeitherIntegerNorVector : public EnableIf<!CanConvertToInt<Cond>::Value, T> {};
template<typename Cond, typename T> struct EnableIfNeitherIntegerNorVector<Vector<Cond>, T>;

template<typename T> struct IsVector             { enum { Value = false }; };
template<typename T> struct IsVector<Vector<T> > { enum { Value =  true }; };

template<typename T0, typename T1, typename V0, typename V1> struct IsTypeCombinationOf
{
    enum {
        Value = IsVector<V0>::Value ? (IsVector<V1>::Value ? ( // Vec × Vec
                    (    IsEqualType<T0, V0>::Value && HasImplicitCast<T1, V1>::Value && !HasImplicitCast<T1, int>::Value) ||
                    (HasImplicitCast<T0, V0>::Value &&     IsEqualType<T1, V1>::Value && !HasImplicitCast<T0, int>::Value) ||
                    (    IsEqualType<T0, V1>::Value && HasImplicitCast<T1, V0>::Value && !HasImplicitCast<T1, int>::Value) ||
                    (HasImplicitCast<T0, V1>::Value &&     IsEqualType<T1, V0>::Value && !HasImplicitCast<T0, int>::Value)
                ) : ( // Vec × Scalar
                    (HasImplicitCast<T0, V0>::Value &&     IsEqualType<T1, V1>::Value && !HasImplicitCast<T0, int>::Value) ||
                    (    IsEqualType<T0, V1>::Value && HasImplicitCast<T1, V0>::Value && !HasImplicitCast<T1, int>::Value)
            )) : (IsVector<V1>::Value ? ( // Scalar × Vec
                    (    IsEqualType<T0, V0>::Value && HasImplicitCast<T1, V1>::Value && !HasImplicitCast<T1, int>::Value) ||
                    (HasImplicitCast<T0, V1>::Value &&     IsEqualType<T1, V0>::Value && !HasImplicitCast<T0, int>::Value)
                ) : ( // Scalar × Scalar
                    (    IsEqualType<T0, V0>::Value &&     IsEqualType<T1, V1>::Value) ||
                    (    IsEqualType<T0, V1>::Value &&     IsEqualType<T1, V0>::Value)
                    ))
    };
};

template<typename T0, typename T1, typename V> struct IsVectorOperands
{
    enum {
        Value = (HasImplicitCast<T0, V>::Value && !HasImplicitCast<T0, int>::Value && !IsEqualType<T0, V>::Value && IsEqualType<T1, V>::Value)
            ||  (HasImplicitCast<T1, V>::Value && !HasImplicitCast<T1, int>::Value && !IsEqualType<T1, V>::Value && IsEqualType<T0, V>::Value)
    };
};
#ifndef VC_ICC
}
#endif

// float-int arithmetic operators //{{{1
// These operators must be very picky about the exact types they want to handle. Once (uncontrolled)
// implicit type conversions get involved, ambiguous overloads will occur. E.g. a simple int × enum
// will become ambiguous because it can convert both to a vector type, which then can execute the
// operator. We can't argue that such code should not be used - it could break existing code, not
// under control of the developer, just by putting the Vc header somewhere on top.
//
// The following type combinations are safe (always symmetric):
// 1. Vector × Vector
// 2. Vector × Scalar (int, float, enum value, ...)
// 3. Some object that has a vector cast operator × Vector
// 4. Some object that has a vector cast operator × Scalar
//
// Additionally there are restrictions on which types combine to what resulting type:
// 1.a.        float × double_v -> double_v
// 1.b.      any int × double_v -> double_v
// 2.a.     (u)int_v ×  float_v ->  float_v
// 2.b.     (u)int_v ×    float ->  float_v
// 2.c.      any int ×  float_v ->  float_v
// 3.a.   (u)short_v × sfloat_v -> sfloat_v
// 3.b.   (u)short_v ×    float -> sfloat_v
// 3.c.        short × sfloat_v -> sfloat_v
// 4.a.        int_v ×   uint_v ->   uint_v
// 4.b.      any int ×   uint_v ->   uint_v
// 4.c. unsigned int ×    int_v ->   uint_v
// 4.d.   signed int ×    int_v ->    int_v
// 5.              shorts like ints

#define VC_OPERATOR_FORWARD_(ret, op) \
template<typename T0, typename T1> static Vc_ALWAYS_INLINE typename EnableIf< \
    IsVectorOperands<T0, T1, double_v>::Value || \
    ((IsEqualType<T0, float>::Value || IsLikeInteger<T0>::Value) && HasImplicitCast<T1, double_v>::Value && !HasImplicitCast<T1, int>::Value) || \
    ((IsEqualType<T1, float>::Value || IsLikeInteger<T1>::Value) && HasImplicitCast<T0, double_v>::Value && !HasImplicitCast<T0, int>::Value) || \
    false, double_##ret>::Value operator op(const T0 &x, const T1 &y) { return double_v(x) op double_v(y); } \
\
template<typename T0, typename T1> static Vc_ALWAYS_INLINE typename EnableIf< \
    IsVectorOperands<T0, T1, float_v>::Value || \
    IsTypeCombinationOf<T0, T1,  int_v, float_v>::Value || \
    IsTypeCombinationOf<T0, T1, uint_v, float_v>::Value || \
    IsTypeCombinationOf<T0, T1,  int_v,   float>::Value || \
    IsTypeCombinationOf<T0, T1, uint_v,   float>::Value || \
    (IsLikeInteger<T0>::Value && HasImplicitCast<T1, float_v>::Value && !HasImplicitCast<T1, int>::Value) || \
    (IsLikeInteger<T1>::Value && HasImplicitCast<T0, float_v>::Value && !HasImplicitCast<T0, int>::Value) || \
    false, float_##ret>::Value operator op(const T0 &x, const T1 &y) { return float_v(x) op float_v(y); } \
\
template<typename T0, typename T1> static Vc_ALWAYS_INLINE typename EnableIf< \
    IsVectorOperands<T0, T1, sfloat_v>::Value || \
    IsTypeCombinationOf<T0, T1,  short_v, sfloat_v>::Value || \
    IsTypeCombinationOf<T0, T1, ushort_v, sfloat_v>::Value || \
    IsTypeCombinationOf<T0, T1,  short_v,    float>::Value || \
    IsTypeCombinationOf<T0, T1, ushort_v,    float>::Value || \
    (IsLikeInteger<T0>::Value && HasImplicitCast<T1, sfloat_v>::Value && !HasImplicitCast<T1, int>::Value) || \
    (IsLikeInteger<T1>::Value && HasImplicitCast<T0, sfloat_v>::Value && !HasImplicitCast<T0, int>::Value) || \
    false, sfloat_##ret>::Value operator op(const T0 &x, const T1 &y) { return sfloat_v(x) op sfloat_v(y); } \
\
template<typename T0, typename T1> static Vc_ALWAYS_INLINE typename EnableIf< \
    IsVectorOperands<T0, T1, uint_v>::Value || \
    IsTypeCombinationOf<T0, T1, int_v, uint_v>::Value || \
    (IsUnsignedInteger<T0>::Value && HasImplicitCast<T1, int_v>::Value && !HasImplicitCast<T1, int>::Value) || \
    (IsUnsignedInteger<T1>::Value && HasImplicitCast<T0, int_v>::Value && !HasImplicitCast<T0, int>::Value) || \
    (IsLikeInteger<T0>::Value && !IsEqualType<T0, unsigned int>::Value && HasImplicitCast<T1, uint_v>::Value && !HasImplicitCast<T1, int>::Value) || \
    (IsLikeInteger<T1>::Value && !IsEqualType<T1, unsigned int>::Value && HasImplicitCast<T0, uint_v>::Value && !HasImplicitCast<T0, int>::Value) || \
    false, uint_##ret>::Value operator op(const T0 &x, const T1 &y) { return uint_v(x) op uint_v(y); } \
template<typename T0, typename T1> static Vc_ALWAYS_INLINE typename EnableIf< \
    IsVectorOperands<T0, T1, int_v>::Value || \
    (IsLikeSignedInteger<T0>::Value && !IsEqualType<T0, int>::Value && HasImplicitCast<T1, int_v>::Value && !HasImplicitCast<T1, int>::Value) || \
    (IsLikeSignedInteger<T1>::Value && !IsEqualType<T1, int>::Value && HasImplicitCast<T0, int_v>::Value && !HasImplicitCast<T0, int>::Value) || \
    false, int_##ret>::Value operator op(const T0 &x, const T1 &y) { return  int_v(x) op  int_v(y); } \
\
template<typename T0, typename T1> static Vc_ALWAYS_INLINE typename EnableIf< \
    IsVectorOperands<T0, T1, ushort_v>::Value || \
    IsTypeCombinationOf<T0, T1, short_v, ushort_v>::Value || \
    (IsUnsignedInteger<T0>::Value && HasImplicitCast<T1, short_v>::Value && !HasImplicitCast<T1, int>::Value) || \
    (IsUnsignedInteger<T1>::Value && HasImplicitCast<T0, short_v>::Value && !HasImplicitCast<T0, int>::Value) || \
    (IsLikeInteger<T0>::Value && !IsEqualType<T0, unsigned short>::Value && HasImplicitCast<T1, ushort_v>::Value && !HasImplicitCast<T1, int>::Value) || \
    (IsLikeInteger<T1>::Value && !IsEqualType<T1, unsigned short>::Value && HasImplicitCast<T0, ushort_v>::Value && !HasImplicitCast<T0, int>::Value) || \
    false, ushort_##ret>::Value operator op(const T0 &x, const T1 &y) { return ushort_v(x) op ushort_v(y); } \
template<typename T0, typename T1> static Vc_ALWAYS_INLINE typename EnableIf< \
    IsVectorOperands<T0, T1, short_v>::Value || \
    (IsLikeSignedInteger<T0>::Value && !IsEqualType<T0, short>::Value && HasImplicitCast<T1, short_v>::Value && !HasImplicitCast<T1, int>::Value) || \
    (IsLikeSignedInteger<T1>::Value && !IsEqualType<T1, short>::Value && HasImplicitCast<T0, short_v>::Value && !HasImplicitCast<T0, int>::Value) || \
    false, short_##ret>::Value operator op(const T0 &x, const T1 &y) { return  short_v(x) op  short_v(y); }


// break incorrect combinations
#define VC_OPERATOR_INTENTIONAL_ERROR_1(V, op) \
template<typename T> static inline typename EnableIfNeitherIntegerNorVector<T, Vc::Error::invalid_operands_of_types<V, T> >::Value operator op(const V &, const T &) { return Vc::Error::invalid_operands_of_types<V, T>(); } \
template<typename T> static inline typename EnableIfNeitherIntegerNorVector<T, Vc::Error::invalid_operands_of_types<T, V> >::Value operator op(const T &, const V &) { return Vc::Error::invalid_operands_of_types<T, V>(); }

#define VC_OPERATOR_INTENTIONAL_ERROR_2(V1, V2, op) \
static inline Vc::Error::invalid_operands_of_types<V1, V2> operator op(V1::AsArg, V2::AsArg) { return Vc::Error::invalid_operands_of_types<V1, V2>(); } \
static inline Vc::Error::invalid_operands_of_types<V2, V1> operator op(V2::AsArg, V1::AsArg) { return Vc::Error::invalid_operands_of_types<V2, V1>(); }

#define VC_OPERATOR_INTENTIONAL_ERROR_3(V, _T, op) \
template<typename T> static inline typename EnableIf<IsEqualType<T, _T>::Value, Vc::Error::invalid_operands_of_types<V, T> >::Value operator op(const V &, const T &) { return Vc::Error::invalid_operands_of_types<V, T>(); } \
template<typename T> static inline typename EnableIf<IsEqualType<T, _T>::Value, Vc::Error::invalid_operands_of_types<T, V> >::Value operator op(const T &, const V &) { return Vc::Error::invalid_operands_of_types<T, V>(); }

//#define VC_EXTRA_CHECKING
#ifdef VC_EXTRA_CHECKING
#define VC_OPERATOR_INTENTIONAL_ERROR(op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v, sfloat_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v,  float_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v,    int_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v,   uint_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v,  short_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(double_v, ushort_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(   int_v,  short_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(  uint_v,  short_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(   int_v, ushort_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(  uint_v, ushort_v, op) \
    VC_APPLY_1(VC_LIST_VECTOR_TYPES, VC_OPERATOR_INTENTIONAL_ERROR_1, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2( float_v,  short_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2( float_v, ushort_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(sfloat_v,  float_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(sfloat_v,    int_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_2(sfloat_v,   uint_v, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_3( float_v,   double, op) \
    VC_OPERATOR_INTENTIONAL_ERROR_3(sfloat_v,   double, op)
#else
#define VC_OPERATOR_INTENTIONAL_ERROR(op)
#endif

#define VC_OPERATOR_FORWARD_COMMUTATIVE(ret, op, op2) \
template<typename T> static Vc_ALWAYS_INLINE VC_EXACT_TYPE(T,         double, double_##ret) operator op(T x, double_v::AsArg y) { return y op2 x; } \
template<typename T> static Vc_ALWAYS_INLINE VC_EXACT_TYPE(T,          float, sfloat_##ret) operator op(T x, sfloat_v::AsArg y) { return y op2 x; } \
template<typename T> static Vc_ALWAYS_INLINE VC_EXACT_TYPE(T,          float,  float_##ret) operator op(T x,  float_v::AsArg y) { return y op2 x; } \
template<typename T> static Vc_ALWAYS_INLINE VC_EXACT_TYPE(T,            int,    int_##ret) operator op(T x,    int_v::AsArg y) { return y op2 x; } \
template<typename T> static Vc_ALWAYS_INLINE VC_EXACT_TYPE(T,   unsigned int,   uint_##ret) operator op(T x,   uint_v::AsArg y) { return y op2 x; } \
template<typename T> static Vc_ALWAYS_INLINE VC_EXACT_TYPE(T,          short,  short_##ret) operator op(T x,  short_v::AsArg y) { return y op2 x; } \
template<typename T> static Vc_ALWAYS_INLINE VC_EXACT_TYPE(T, unsigned short, ushort_##ret) operator op(T x, ushort_v::AsArg y) { return y op2 x; } \
VC_OPERATOR_FORWARD_(ret, op) \
VC_OPERATOR_INTENTIONAL_ERROR(op)

#define VC_OPERATOR_FORWARD(ret, op) \
template<typename T> static Vc_ALWAYS_INLINE VC_EXACT_TYPE(T,         double, double_##ret) operator op(T x, double_v::AsArg y) { return double_v(x) op y; } \
template<typename T> static Vc_ALWAYS_INLINE VC_EXACT_TYPE(T,          float, sfloat_##ret) operator op(T x, sfloat_v::AsArg y) { return sfloat_v(x) op y; } \
template<typename T> static Vc_ALWAYS_INLINE VC_EXACT_TYPE(T,          float,  float_##ret) operator op(T x,  float_v::AsArg y) { return  float_v(x) op y; } \
template<typename T> static Vc_ALWAYS_INLINE VC_EXACT_TYPE(T,            int,    int_##ret) operator op(T x,    int_v::AsArg y) { return    int_v(x) op y; } \
template<typename T> static Vc_ALWAYS_INLINE VC_EXACT_TYPE(T,   unsigned int,   uint_##ret) operator op(T x,   uint_v::AsArg y) { return   uint_v(x) op y; } \
template<typename T> static Vc_ALWAYS_INLINE VC_EXACT_TYPE(T,          short,  short_##ret) operator op(T x,  short_v::AsArg y) { return  short_v(x) op y; } \
template<typename T> static Vc_ALWAYS_INLINE VC_EXACT_TYPE(T, unsigned short, ushort_##ret) operator op(T x, ushort_v::AsArg y) { return ushort_v(x) op y; } \
VC_OPERATOR_FORWARD_(ret, op) \
VC_OPERATOR_INTENTIONAL_ERROR(op)

VC_OPERATOR_FORWARD_COMMUTATIVE(v, *, *)
VC_OPERATOR_FORWARD(v, /)
VC_OPERATOR_FORWARD_COMMUTATIVE(v, +, +)
VC_OPERATOR_FORWARD(v, -)
VC_OPERATOR_FORWARD_COMMUTATIVE(v, |, |)
VC_OPERATOR_FORWARD_COMMUTATIVE(v, &, &)
VC_OPERATOR_FORWARD_COMMUTATIVE(v, ^, ^)
VC_OPERATOR_FORWARD_COMMUTATIVE(m, <, >)
VC_OPERATOR_FORWARD_COMMUTATIVE(m, >, <)
VC_OPERATOR_FORWARD_COMMUTATIVE(m, <=, >=)
VC_OPERATOR_FORWARD_COMMUTATIVE(m, >=, <=)
VC_OPERATOR_FORWARD_COMMUTATIVE(m, ==, ==)
VC_OPERATOR_FORWARD_COMMUTATIVE(m, !=, !=)

#undef VC_OPERATOR_FORWARD_
#undef VC_OPERATOR_INTENTIONAL_ERROR_1
#undef VC_OPERATOR_INTENTIONAL_ERROR_2
#undef VC_OPERATOR_INTENTIONAL_ERROR
#undef VC_OPERATOR_FORWARD_COMMUTATIVE
#undef VC_OPERATOR_FORWARD

// }}}1
