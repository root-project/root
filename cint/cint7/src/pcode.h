#include "Reflex/Builder/TypeBuilder.h"
#include "value.h"

namespace Cint {
   namespace Internal {


/*************************************************************************
**************************************************************************
* Optimization level 1 runtime function
**************************************************************************
*************************************************************************/

/* templated read and write access to G__value */

template <typename T>
inline T G__getvalue_raw(const G__value& val)
{ return (T) val.obj.i; }

template <>
inline bool G__getvalue_raw<bool>(const G__value& val)
{ 
#ifdef G__BOOL4BYTE
   return (bool)val.obj.i; 
#else // G__BOOL4BYTE
   return (bool)val.obj.uch; 
#endif // G__BOOL4BYTE
}

template <>
inline unsigned char G__getvalue_raw<unsigned char>(const G__value& val)
{ return (unsigned char)val.obj.ulo; }

template <>
inline unsigned short G__getvalue_raw<unsigned short>(const G__value& val)
{ return (unsigned short)val.obj.ulo; }

template <>
inline unsigned int G__getvalue_raw<unsigned int>(const G__value& val)
{ return (unsigned int)val.obj.ulo; }

template <>
inline unsigned long G__getvalue_raw<unsigned long>(const G__value& val)
{ return (unsigned long)val.obj.ulo; }

template <>
inline G__int64 G__getvalue_raw<G__int64>(const G__value& val)
{ return val.obj.ll; }
template <>
inline G__uint64 G__getvalue_raw<G__uint64>(const G__value& val)
{ return val.obj.ull; }

template <>
inline long double G__getvalue_raw<long double>(const G__value& val)
{ return val.obj.ld; }
template <>
inline double G__getvalue_raw<double>(const G__value& val)
{ return val.obj.d; }
template <>
inline float G__getvalue_raw<float>(const G__value& val)
{ return (float)val.obj.d; }


/*************************************************************************
**************************************************************************
* TOPVALUE and TOVALUE optimization
**************************************************************************
*************************************************************************/
/*
  Template version of casting to obj.i's long
  formerly known as G__asm_tovalue_*
*/

template <typename CASTTYPE>
inline void G__asm_deref_cast(const G__value* from, G__value* to)
{
  G__setvalue(to, (*(CASTTYPE *)(from->obj.i)));
}
template<>
inline void G__asm_deref_cast<G__int64>(const G__value* from, G__value* to)
{
  G__setvalue(to, (*(G__int64*)(from->obj.i)));
}
template<>
inline void G__asm_deref_cast<G__uint64>(const G__value* from, G__value* to)
{
  G__setvalue(to, (*(G__uint64*)(from->obj.i)));
}
template<>
inline void G__asm_deref_cast<long double>(const G__value* from, G__value* to)
{
  G__setvalue(to, (*(long double*)(from->obj.i)));
}
template<>
inline void G__asm_deref_cast<float>(const G__value* from, G__value* to)
{
  G__setvalue(to, (*(float*)(from->obj.i)));
}
template<>
inline void G__asm_deref_cast<double>(const G__value* from, G__value* to)
{
  G__setvalue(to, (*(double*)(from->obj.i)));
}

template <typename CASTTYPE>
inline void G__asm_deref(G__value *result)
{
  assert(G__value_typenum(*result).FinalType().IsPointer());
  result->ref = result->obj.i;
  G__asm_deref_cast<CASTTYPE>(result, result);
  G__value_typenum(*result) = G__deref(G__value_typenum(*result));
}

/******************************************************************
* static void G__asm_tovalue_p2p(G__value* p)
******************************************************************/
inline void G__asm_tovalue_p2p(G__value *result)
{
   //deref
   G__asm_deref<long>(result);
}
 
/******************************************************************
* void Cint::Internal::G__asm_tovalue_U(G__value* p)
******************************************************************/
inline void G__asm_tovalue_U(G__value *result)
{
  assert(G__value_typenum(*result).FinalType().IsPointer());
  result->ref = result->obj.i;
  /* result->obj.i = result->obj.i; */
  G__value_typenum(*result) = G__deref(G__value_typenum(*result));
}

/******************************************************************
* G__value G__asm_toXvalue(G__value* p)
*
******************************************************************/
inline void G__asm_toXvalue(G__value* result)
{
   G__value_typenum(*result) = Reflex::PointerBuilder(G__value_typenum(*result));
   if(result->ref) result->obj.i = result->ref;
   result->ref = 0;
}

   } // namespace Internal
} // namespace Cint
