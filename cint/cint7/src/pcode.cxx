/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file pcode.c
 ************************************************************************
 * Description:
 *  Loop compilation related source code
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "pcode.h"
#include "value.h"

#include "Dict.h"
#include "Reflex/Tools.h"
#include "Reflex/Builder/TypeBuilder.h"

using namespace Cint::Internal;

#ifdef G__BORLANDCC5
   //--
   //--
   void G__get__tm__(char *buf);
   char* G__get__date__(void);
   char* G__get__time__(void);
   int G__isInt(int type);
   int G__get_LD_Rp0_p2f(int type,long *pinst);
   int G__get_ST_Rp0_p2f(int type,long *pinst);
#endif // G__BORLANDCC5

#ifdef G__ASM

#ifdef G__ASM_DBG
int Cint::Internal::G__asm_step = 0;
#endif // G__ASM_DBG

#define G__TUNEUP_W_SECURITY

//______________________________________________________________________________
//
//  Macros to access G__value.
//

//______________________________________________________________________________
#define G__intM(buf) \
   (((G__get_type(*buf) == 'f') || (G__get_type(*buf) == 'd')) ? ((long) buf->obj.d) : buf->obj.i)

//______________________________________________________________________________
#define G__longlongM(buf) G__converT<long long>(buf)

//______________________________________________________________________________
#define G__doubleM(buf) G__convertT<double>(buf)

//______________________________________________________________________________
#define G__isdoubleM(buf) \
   ((G__get_type(*buf) == 'f') || (G__get_type(*buf) == 'd'))

//______________________________________________________________________________
#define G__isunsignedM(buf) \
   ((G__get_type(*buf) == 'h') || (G__get_type(*buf) == 'k'))

//______________________________________________________________________________
//
//  Optimization level 2 runtime functions.
//

//______________________________________________________________________________
//
//  G__asm_test_X(),  Optimized comparator

namespace Cint {
namespace Internal {

//______________________________________________________________________________
static int G__asm_test_E(int* a, int* b)
{
   return *a == *b;
}

//______________________________________________________________________________
static int G__asm_test_N(int* a, int* b)
{
   return *a != *b;
}

//______________________________________________________________________________
static int G__asm_test_GE(int* a, int* b)
{
   return *a >= *b;
}

//______________________________________________________________________________
static int G__asm_test_LE(int* a, int* b)
{
   return *a <= *b;
}

//______________________________________________________________________________
static int G__asm_test_g(int* a, int* b)
{
   return *a > *b;
}

//______________________________________________________________________________
static int G__asm_test_l(int* a, int* b)
{
   return *a < *b;
}

} // namespace Internal
} // namespace Cint

//______________________________________________________________________________
//
//  Optimization level 3 runtime functions.
//

//______________________________________________________________________________
//
//  G__LD_p0 family of functions.
//
//  Note: None of these functions can be inline because we take their address.
//

//______________________________________________________________________________
template<class CASTTYPE> static void G__LD_p0(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   static Reflex::Type ty(Reflex::Type::ByTypeInfo(typeid(CASTTYPE)));
   G__value* buf = &pbuf[(*psp)++];
   G__value_typenum(*buf) = ty;
   buf->ref = (long) (G__get_offset(var) + offset);
   G__setvalue(buf, *(CASTTYPE*)buf->ref);
}

//______________________________________________________________________________
static void G__LD_p0_pointer(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* buf = &pbuf[(*psp)++];
   G__value_typenum(*buf) = var.TypeOf();
   buf->ref = ((long) G__get_offset(var)) + offset;
   buf->obj.i = *(long*)buf->ref;
}

//______________________________________________________________________________
static void G__LD_p0_struct(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* buf = &pbuf[(*psp)++];
   G__value_typenum(*buf) = var.TypeOf();
   buf->ref = ((long) G__get_offset(var)) + offset;
   buf->obj.i = buf->ref;
}

//______________________________________________________________________________
//
//  Array bounds checking.
//

//______________________________________________________________________________
static void G__nonintarrayindex(const Reflex::Member& var)
{
   G__fprinterr(G__serr, "Error: %s[] invalud type for array index", var.Name(Reflex::SCOPED).c_str());
   G__genericerror(0);
}

#ifdef G__TUNEUP_W_SECURITY
//______________________________________________________________________________
static bool G__check_idx(const G__value& vidx, const Reflex::Member& var)
{
   char type = G__get_type(vidx);
   if ((type == 'd') || (type == 'f')) {
      G__nonintarrayindex(var);
   }
   if (vidx.obj.i > G__get_varlabel(var, 1)) {
      G__arrayindexerror(var, var.Name(Reflex::SCOPED).c_str(), G__convertT<long>(&vidx));
      return false;
   }
   return true;
}
#endif // G__TUNEUP_W_SECURITY

//______________________________________________________________________________
//
//  G__LD_p1_xxx
//
//  Note: None of these functions can be inline because we take their address.
//

//______________________________________________________________________________
template <typename CASTTYPE> static void G__LD_p1(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   static Reflex::Type ty(Reflex::Type::ByTypeInfo(typeid(CASTTYPE)));
   G__value* buf = &pbuf[*psp-1];
#ifdef G__TUNEUP_W_SECURITY
   //if (G__check_idx(*buf, var))
#endif // G__TUNEUP_W_SECURITY
   {
      buf->ref = (long)(G__get_offset(var) + offset + (G__convertT<long>(buf) * sizeof(CASTTYPE)));
      G__setvalue(buf, *(CASTTYPE*)buf->ref);
   }
   G__value_typenum(*buf) = ty;
}

//______________________________________________________________________________
static void G__LD_p1_pointer(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* buf = &pbuf[*psp-1];
#ifdef G__TUNEUP_W_SECURITY
   if (G__check_idx(*buf, var))
#endif // G__TUNEUP_W_SECURITY
   {
      buf->ref = (long)G__get_offset(var) + offset + (G__convertT<long>(buf) * sizeof(long));
      buf->obj.i = *(long*)buf->ref;
   }
   G__value_typenum(*buf) = var.TypeOf();
}

//______________________________________________________________________________
static void G__LD_p1_struct(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* buf = &pbuf[*psp-1];
#ifdef G__TUNEUP_W_SECURITY
   if (G__check_idx(*buf, var))
#endif // G__TUNEUP_W_SECURITY
   {
      buf->ref = (long)G__get_offset(var) + offset + (G__convertT<long>(buf) * var.TypeOf().RawType().SizeOf());
      buf->obj.i = buf->ref;
   }
   G__value_typenum(*buf) = var.TypeOf();
}

//______________________________________________________________________________
//
//  Get pointer increment for a given array dimension.
//

//______________________________________________________________________________
static inline int G__get_p_inc(int paran, G__value* buf, const Reflex::Member& var)
{
   int ary = G__get_varlabel(var, 0);
   int p_inc = 0;
   for (int ig25 = 0; ig25 < paran; ++ig25) {
      p_inc += ary * G__int(buf[ig25]);
      ary /= G__get_varlabel(var, ig25 + 2);
   }
   return p_inc;
}

//______________________________________________________________________________
//
//  G__LD_pn_xxx
//
//  Note: None of these can be inline because we take their address.
//

//______________________________________________________________________________
template <typename CASTTYPE> static void G__LD_pn(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   static Reflex::Type ty(Reflex::Type::ByTypeInfo(typeid(CASTTYPE)));
   int paran = G__get_paran(var);
   *psp = *psp - paran;
   G__value* buf = &pbuf[*psp];
   ++(*psp);
   int p_inc = G__get_p_inc(paran, buf, var);
   G__value_typenum(*buf) = ty;
   buf->ref = (long)(G__get_offset(var) + offset + (p_inc * sizeof(CASTTYPE)));
#ifdef G__TUNEUP_W_SECURITY
   if (p_inc > G__get_varlabel(var, 1)) {
      G__arrayindexerror(var, var.Name(Reflex::SCOPED).c_str(), p_inc);
   }
   else
#endif // G__TUNEUP_W_SECURITY
   {
       G__setvalue(buf, *(CASTTYPE*)buf->ref);
   }
}

//______________________________________________________________________________
static void G__LD_pn_pointer(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   int paran = G__get_paran(var);
   *psp = *psp - paran;
   G__value* buf = &pbuf[*psp];
   ++(*psp);
   int p_inc = G__get_p_inc(paran, buf, var);
   G__value_typenum(*buf) = var.TypeOf();
   buf->ref = (long)G__get_offset(var) + offset + (p_inc * sizeof(long));
#ifdef G__TUNEUP_W_SECURITY
   //if (p_inc > G__get_varlabel(var, 1)) {
   //   G__arrayindexerror(var, var.Name(Reflex::SCOPED).c_str(), p_inc);
   //}
   //else
#endif // G__TUNEUP_W_SECURITY
   {
      buf->obj.i = *(long*)buf->ref;
   }
}

//______________________________________________________________________________
static void G__LD_pn_struct(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   int paran = G__get_paran(var);
   *psp = *psp - paran;
   G__value* buf = &pbuf[*psp];
   ++(*psp);
   int p_inc = G__get_p_inc(paran, buf, var);
   G__value_typenum(*buf) = var.TypeOf();
   buf->ref = (long)G__get_offset(var) + offset + (p_inc * var.TypeOf().RawType().SizeOf());
#ifdef G__TUNEUP_W_SECURITY
   //if (p_inc > G__get_varlabel(var, 1)) {
   //   G__arrayindexerror(var, var.Name(Reflex::SCOPED).c_str(), p_inc);
   //}
   //else
#endif // G__TUNEUP_W_SECURITY
   {
      buf->obj.i = buf->ref;
   }
}

//______________________________________________________________________________
template <typename CASTTYPE> static void G__LD_P10(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   static Reflex::Type ty(Reflex::Type::ByTypeInfo(typeid(CASTTYPE)));
   G__value* buf= &pbuf[*psp-1];
   buf->ref = *(long*)(G__get_offset(var) + offset) + (G__convertT<long>(buf) * sizeof(CASTTYPE));
   G__value_typenum(*buf) = ty;
   G__setvalue(buf, *(CASTTYPE*)buf->ref);
}

//______________________________________________________________________________
static void G__LD_P10_pointer(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* buf = &pbuf[*psp-1];
   buf->ref = *(long*)(G__get_offset(var) + offset) + (G__convertT<long>(buf) * sizeof(long));
   G__value_typenum(*buf) = var.TypeOf();
   buf->obj.i = *(long*)buf->ref;
}

//______________________________________________________________________________
static void G__LD_P10_struct(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* buf = &pbuf[*psp-1];
   buf->ref = *(long*)(G__get_offset(var) + offset) + (G__convertT<long>(buf) * var.TypeOf().RawType().SizeOf());
   G__value_typenum(*buf) = var.TypeOf();
   buf->obj.i = buf->ref;
}

//______________________________________________________________________________
//
//  G__ST_p0_xxx
//
//  Note: None of these functions can be inline because we take their address.
//

//______________________________________________________________________________
template<class CASTTYPE> static void G__ST_p0(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* val = &pbuf[*psp-1];
   *(CASTTYPE*)(G__get_offset(var) + offset) = G__convertT<CASTTYPE>(val);
}

//______________________________________________________________________________
static void G__ST_p0_pointer(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* val = &pbuf[*psp-1];
   char* address = G__get_offset(var) + offset;
   long newval = G__intM(val);
   if ((G__security & G__SECURE_GARBAGECOLLECTION) && address) {
      if (*(long*)address) {
         G__del_refcount((void*) (*(long*)address), (void**) address);
      }
      if (newval) {
         G__add_refcount((void*) newval, (void**) address);
      }
   }
   *(long*)(address) = newval;
}

//______________________________________________________________________________
static void G__ST_p0_struct(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   memcpy(G__get_offset(var) + offset, (void*) G__convertT<long>(&pbuf[*psp-1]), var.TypeOf().RawType().SizeOf());
}

//______________________________________________________________________________
//
//  G__ST_p1_xxx
//
//  Note: None of these functions can be inline because we take their address.
//

//______________________________________________________________________________
template <typename CASTTYPE> static void G__ST_p1(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* val = &pbuf[*psp-1];
#ifdef G__TUNEUP_W_SECURITY
   //if (G__check_idx(*val, var))
#endif // G__TUNEUP_W_SECURITY
   {
      *(CASTTYPE*)(G__get_offset(var) + offset + (G__convertT<long>(val) * sizeof(CASTTYPE))) = G__convertT<CASTTYPE>(&pbuf[*psp-2]);
   }
   --(*psp);
}

//______________________________________________________________________________
static void G__ST_p1_pointer(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* val = &pbuf[*psp-1];
#ifdef G__TUNEUP_W_SECURITY
   //if (G__check_idx(*val, var))
#endif // G__TUNEUP_W_SECURITY
   {
      long address = (long)(G__get_offset(var) + offset + (G__convertT<long>(val) * sizeof(long)));
      long newval = G__int(pbuf[*psp-2]);
      if ((G__security & G__SECURE_GARBAGECOLLECTION) && address) {
         if (*(long*)address) {
            G__del_refcount((void*)(*(long*)address), (void**)address);
         }
         if (newval) {
            G__add_refcount((void*)newval, (void**)address);
         }
      }
      *(long*)(address) = newval;
   }
   --(*psp);
}

//______________________________________________________________________________
static void G__ST_p1_struct(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* val = &pbuf[*psp-1];
#ifdef G__TUNEUP_W_SECURITY
   //if (G__check_idx(*val, var))
#endif // G__TUNEUP_W_SECURITY
   {
      long struct_sizeof = var.TypeOf().RawType().SizeOf();
      memcpy(G__get_offset(var) + offset + (G__convertT<long>(val) * struct_sizeof), (void*) pbuf[*psp-2].obj.i, struct_sizeof);
   }
   --(*psp);
}

//______________________________________________________________________________
//
//  G__ST_pn_xxx
//

//______________________________________________________________________________
template <typename CASTTYPE> static void G__ST_pn(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   int paran = G__get_paran(var);
   *psp = *psp - paran;
   G__value* buf = &pbuf[*psp];
   int p_inc = G__get_p_inc(paran, buf, var);
#ifdef G__TUNEUP_W_SECURITY
   //if (p_inc > G__get_varlabel(var, 1)) {
   //   G__arrayindexerror(var, var.Name(Reflex::SCOPED).c_str(), p_inc);
   //}
   //else
#endif // G__TUNEUP_W_SECURITY
   {
      *(CASTTYPE*)(G__get_offset(var) + offset + (p_inc * sizeof(CASTTYPE))) = G__convertT<CASTTYPE>(&pbuf[*psp-1]);
   }
}

//______________________________________________________________________________
static void G__ST_pn_pointer(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   int paran = G__get_paran(var);
   *psp = *psp - paran;
   G__value* buf = &pbuf[*psp];
   int p_inc = G__get_p_inc(paran, buf, var);
#ifdef G__TUNEUP_W_SECURITY
   //if (p_inc > G__get_varlabel(var, 1)) {
   //   G__arrayindexerror(var, var.Name(Reflex::SCOPED).c_str(), p_inc);
   //}
   //else
#endif // G__TUNEUP_W_SECURITY
   {
      char* address = (G__get_offset(var) + offset + (p_inc * sizeof(long)));
      long newval = G__int(pbuf[*psp-1]);
      if ((G__security & G__SECURE_GARBAGECOLLECTION) && address) {
         if (*(long*)address) {
            G__del_refcount((void*) (*(long*)address), (void**) address);
         }
         if (newval) {
            G__add_refcount((void*) newval, (void**) address);
         }
      }
      *(long*)(address) = newval;
   }
}

//______________________________________________________________________________
static void G__ST_pn_struct(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   int paran = G__get_paran(var);
   *psp = *psp - paran;
   G__value* buf = &pbuf[*psp];
   int p_inc = G__get_p_inc(paran, buf, var);
#ifdef G__TUNEUP_W_SECURITY
   //if (p_inc > G__get_varlabel(var, 1)) {
   //   G__arrayindexerror(var, var.Name(Reflex::SCOPED).c_str(), p_inc);
   //}
   //else
#endif // G__TUNEUP_W_SECURITY
   {
      long struct_sizeof = var.TypeOf().RawType().SizeOf();
      memcpy(G__get_offset(var) + offset + (p_inc * struct_sizeof), (void*) pbuf[*psp-1].obj.i, struct_sizeof);
   }
}

//______________________________________________________________________________
//
//  G__ST_P10_xxx
//
//  Note: None of these functions can be inline because we take their address.
//

//______________________________________________________________________________
template <typename CASTTYPE> static void G__ST_P10(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* val = &pbuf[*psp-1];
   *(CASTTYPE*)(*(long*)(G__get_offset(var) + offset) + (G__convertT<long>(val) * sizeof(CASTTYPE))) = G__convertT<CASTTYPE>(&pbuf[*psp-2]);
   --(*psp);
}

//______________________________________________________________________________
static void G__ST_P10_pointer(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__ST_P10<long>(pbuf, psp, offset, var);
}

//______________________________________________________________________________
static void G__ST_P10_struct(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* val = &pbuf[*psp-1];
   long struct_sizeof = var.TypeOf().RawType().SizeOf();
   memcpy((void*) (*(long*)(G__get_offset(var) + offset) + (G__convertT<long>(val) * struct_sizeof)), (void*) pbuf[*psp-2].obj.i, struct_sizeof);
   --(*psp);
}

//______________________________________________________________________________
//
//  G__LD_Rp0_xxx
//
//  type& p;
//  p;    optimize this expression
//
//  Note: None of these functions can be inline because we take their address.
//

//______________________________________________________________________________
template <typename CASTTYPE> static void G__LD_Rp0(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   static Reflex::Type ty(Reflex::Type::ByTypeInfo(typeid(CASTTYPE)));
   G__value* buf = &pbuf[(*psp)++];
   G__value_typenum(*buf) = ty;
   buf->ref = *(long*)(G__get_offset(var) + offset);
   G__setvalue(buf, *(CASTTYPE*)buf->ref);
}

//______________________________________________________________________________
static void G__LD_Rp0_pointer(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* buf = &pbuf[(*psp)++];
   G__value_typenum(*buf) = var.TypeOf();
   buf->ref = *(long*)(G__get_offset(var) + offset);
   buf->obj.i = *(long*)buf->ref;
}

//______________________________________________________________________________
static void G__LD_Rp0_struct(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* buf = &pbuf[(*psp)++];
   G__value_typenum(*buf) = var.TypeOf();
   buf->ref = *(long*)(G__get_offset(var) + offset);
   buf->obj.i = buf->ref;
}

//______________________________________________________________________________
//
//  G__ST_Rp0_xxx
//
//  type& p;
//  p = x;    optimize this expression
//
//  Note: None of these functions can be inline because we take their address.
//

//______________________________________________________________________________
template <typename CASTTYPE> static void G__ST_Rp0(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* val = &pbuf[*psp-1];
   void* addr = (void*) (*(long*)(G__get_offset(var) + offset));
   *(CASTTYPE*)addr = G__convertT<CASTTYPE>(val);
}

//______________________________________________________________________________
static void G__ST_Rp0_pointer(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__ST_Rp0<long>(pbuf, psp, offset, var);
}

//______________________________________________________________________________
static void G__ST_Rp0_struct(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   memcpy((void*) (*(long*)(G__get_offset(var) + offset)), (void*) pbuf[*psp-1].obj.i, var.TypeOf().RawType().SizeOf());
}

//______________________________________________________________________________
//
//  G__LD_RP0_xxx
//
//  type& p;
//  &p;    optimize this expression
//
//  Note: None of these functions can be inline because we take their address.
//

//______________________________________________________________________________
template <typename CASTTYPE> static void G__LD_RP0(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   static Reflex::Type ty(Reflex::Type::ByTypeInfo(typeid(CASTTYPE)));
   G__value* buf = &pbuf[(*psp)++];
   G__value_typenum(*buf) = ty;
   buf->ref = (long) (G__get_offset(var) + offset);
   G__setvalue(buf, *(CASTTYPE*)buf->ref);
}

//______________________________________________________________________________
static void G__LD_RP0_pointer(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* buf = &pbuf[(*psp)++];
   G__value_typenum(*buf) = var.TypeOf();
   buf->ref = (long)G__get_offset(var) + offset;
   buf->obj.i = *(long*)buf->ref;
}

//______________________________________________________________________________
static void G__LD_RP0_struct(G__value* pbuf, int* psp, long offset, const Reflex::Member& var)
{
   G__value* buf = &pbuf[(*psp)++];
   G__value_typenum(*buf) = Reflex::PointerBuilder(var.TypeOf());
   buf->ref = (long)G__get_offset(var) + offset;
   buf->obj.i = buf->ref;
}

//______________________________________________________________________________
//
//  G__OP2_OPTIMIZED_UU
//
//  Note: None of these functions can be inline because we take their address.
//

//______________________________________________________________________________
template <typename T> static void G__OP2_plus_T(G__value* bufm1, G__value* bufm2)
{
   static Reflex::Type ty(Reflex::Type::ByTypeInfo(typeid(T)));
   G__setvalue(bufm2, G__convertT<T>(bufm2) + G__convertT<T>(bufm1));
   G__value_typenum(*bufm2) = ty;
   bufm2->ref = 0;
}

//______________________________________________________________________________
template <typename T> static void G__OP2_minus_T(G__value* bufm1, G__value* bufm2)
{
   static Reflex::Type ty(Reflex::Type::ByTypeInfo(typeid(T)));
   G__setvalue(bufm2, G__convertT<T>(bufm2) - G__convertT<T>(bufm1));
   G__value_typenum(*bufm2) = ty;
   bufm2->ref = 0;
}

//______________________________________________________________________________
template <typename T> static void G__OP2_multiply_T(G__value* bufm1, G__value* bufm2)
{
   static Reflex::Type ty(Reflex::Type::ByTypeInfo(typeid(T)));
   G__setvalue(bufm2, G__convertT<T>(bufm2) * G__convertT<T>(bufm1));
   G__value_typenum(*bufm2) = ty;
   bufm2->ref = 0;
}

//______________________________________________________________________________
template <typename T> static void G__OP2_divide_T(G__value* bufm1, G__value* bufm2)
{
   static Reflex::Type ty(Reflex::Type::ByTypeInfo(typeid(T)));
   T bufm1v = G__convertT<T>(bufm1);
   if (bufm1v == 0) {
      G__genericerror("Error: operator '/' division by zero");
      return;
   }
   G__setvalue(bufm2, G__convertT<T>(bufm2) / bufm1v);
   G__value_typenum(*bufm2) = ty;
   bufm2->ref = 0;
}

//______________________________________________________________________________
static void G__OP2_addvoidptr(G__value* bufm1, G__value* bufm2)
{
   bufm2->obj.i += bufm1->obj.i;
}

//______________________________________________________________________________
//
//  G__OP2_OPTIMIZED
//
//  Note: None of these functions can be inline because we take their address.
//

//______________________________________________________________________________
static void G__OP2_plus(G__value* bufm1, G__value* bufm2)
{
   char fundType1 = G__get_type(*bufm1);
   char fundType2 = G__get_type(*bufm2);
   if (fundType1 == 'q' || fundType2 == 'q') {
      G__OP2_plus_T<long double>(bufm1, bufm2);
   }
   else if (fundType1 == 'd' || fundType2 == 'd') {
      G__OP2_plus_T<double>(bufm1, bufm2);
   }
   else if (fundType1 == 'f' || fundType2 == 'f') {
      G__OP2_plus_T<float>(bufm1, bufm2);
   }
   else if (fundType1 == 'm' || fundType2 == 'm') {
      G__OP2_plus_T<G__uint64>(bufm1, bufm2);
   }
   else if (fundType1 == 'n' || fundType2 == 'n') {
      G__OP2_plus_T<G__int64>(bufm1, bufm2);
   }
   else if (isupper(fundType2)) {
      bufm2->obj.i = bufm2->obj.i + (bufm1->obj.i * G__sizeof_deref(bufm2));
   }
   else if (isupper(fundType1)) {
      bufm2->obj.i = (bufm2->obj.i * G__sizeof_deref(bufm1)) + bufm1->obj.i;
      G__value_typenum(*bufm2) = G__value_typenum(*bufm1);
   }
   else if (
      (fundType1 == 'b') ||
      (fundType1 == 'r') ||
      (fundType1 == 'h') ||
      (fundType1 == 'k')
   ) {
      G__OP2_plus_T<unsigned long>(bufm1, bufm2);
   }
   else {
      G__OP2_plus_T<long>(bufm1, bufm2);
   }
   bufm2->ref = 0;
}

//______________________________________________________________________________
static void G__OP2_minus(G__value* bufm1, G__value* bufm2)
{
   static Reflex::Type int_type(Reflex::Type::ByTypeInfo(typeid(int)));
   char fundType1 = G__get_type(*bufm1);
   char fundType2 = G__get_type(*bufm2);
   if (fundType1 == 'q' || fundType2 == 'q') {
      G__OP2_minus_T<long double>(bufm1, bufm2);
   }
   else if (fundType1 == 'd' || fundType2 == 'd') {
      G__OP2_minus_T<double>(bufm1, bufm2);
   }
   else if (fundType1 == 'f' || fundType2 == 'f') {
      G__OP2_minus_T<float>(bufm1, bufm2);
   }
   else if (fundType1 == 'm' || fundType2 == 'm') {
      G__OP2_minus_T<G__uint64>(bufm1, bufm2);
   }
   else if (fundType1 == 'n' || fundType2 == 'n') {
      G__OP2_minus_T<G__int64>(bufm1, bufm2);
   }
   else if (isupper(fundType2)) {
      if (isupper(fundType1)) {
         bufm2->obj.i = (bufm2->obj.i - bufm1->obj.i) / G__sizeof_deref(bufm2);
         G__value_typenum(*bufm2) = int_type;
      }
      else {
         bufm2->obj.i -= bufm1->obj.i * G__sizeof_deref(bufm2);
         G__value_typenum(*bufm2) = G__value_typenum(*bufm1);
      }
   }
   else if (isupper(fundType1)) {
      bufm2->obj.i = (bufm2->obj.i * G__sizeof(bufm2)) - bufm1->obj.i;
      G__value_typenum(*bufm2) = G__value_typenum(*bufm1);
   }
   else if (
      (fundType1 == 'b') ||
      (fundType1 == 'r') ||
      (fundType1 == 'h') ||
      (fundType1 == 'k')
   ) {
      G__OP2_minus_T<unsigned long>(bufm1, bufm2);
   }
   else {
      G__OP2_minus_T<long>(bufm1, bufm2);
   }
   bufm2->ref = 0;
}

//______________________________________________________________________________
static void G__OP2_multiply(G__value* bufm1, G__value* bufm2)
{
   char fundType1 = G__get_type(*bufm1);
   char fundType2 = G__get_type(*bufm2);
   if (fundType1 == 'q' || fundType2 == 'q') {
      G__OP2_multiply_T<long double>(bufm1, bufm2);
   }
   else if (fundType1 == 'd' || fundType2 == 'd') {
      if (fundType1 == fundType2) {
         bufm2->obj.d = bufm2->obj.d * bufm1->obj.d;
      } else if (fundType2 == 'd') {
         bufm2->obj.d = bufm2->obj.d * G__convertT<double>(bufm1);         
      } else {
         // assert(fundType1 == 'd');
         bufm2->obj.d = G__convertT<double>(bufm2) * bufm1->obj.d; 
         G__value_typenum(*bufm2) = G__value_typenum(*bufm1); // Because we know that buf1 is a double.
      } 
   }
   else if (fundType1 == 'f' || fundType2 == 'f') {
      G__OP2_multiply_T<float>(bufm1, bufm2);
   }
   else if (fundType1 == 'm' || fundType2 == 'm') {
      G__OP2_multiply_T<G__uint64>(bufm1, bufm2);
   }
   else if (fundType1 == 'n' || fundType2 == 'n') {
      G__OP2_multiply_T<G__int64>(bufm1, bufm2);
   }
   else if (
      (fundType1 == 'b') ||
      (fundType1 == 'r') ||
      (fundType1 == 'h') ||
      (fundType1 == 'k')
   ) {
      G__OP2_multiply_T<unsigned long>(bufm1, bufm2);
   }
   else {
      G__OP2_multiply_T<long>(bufm1, bufm2);
   }
   bufm2->ref = 0;
}

//______________________________________________________________________________
template <typename T> static void G__OP2_modulus_T(G__value* bufm1, G__value* bufm2)
{
   static Reflex::Type ty(Reflex::Type::ByTypeInfo(typeid(T)));
   T bufm1v = G__convertT<T>(bufm1);
#ifdef G__TUNEUP_W_SECURITY
   if (bufm1v == 0) {
      G__genericerror("Error: operator '%%' division by zero");
      return;
   }
#endif // G__TUNEUP_W_SECURITY
   G__setvalue<T>(bufm2, G__convertT<T>(bufm2) % bufm1v);
   G__value_typenum(*bufm2) = ty;
}

//______________________________________________________________________________
static void G__OP2_modulus(G__value* bufm1, G__value* bufm2)
{
   char fundType1 = G__get_type(*bufm1);
   char fundType2 = G__get_type(*bufm2);
   if (fundType1 == 'm' || fundType2 == 'm') {
      G__OP2_modulus_T<G__uint64>(bufm1, bufm2);
   }
   else if (fundType1 == 'n' || fundType2 == 'n') {
      G__OP2_modulus_T<G__int64>(bufm1, bufm2);
   }
   else if (
      (fundType1 == 'b') ||
      (fundType1 == 'r') ||
      (fundType1 == 'h') ||
      (fundType1 == 'k')
   ) {
      G__OP2_modulus_T<unsigned long>(bufm1, bufm2);
   }
   else {
      G__OP2_modulus_T<long>(bufm1, bufm2);
   }
   bufm2->ref = 0;
}

//______________________________________________________________________________
static void G__OP2_divide(G__value* bufm1, G__value* bufm2)
{
   char fundType1 = G__get_type(*bufm1);
   char fundType2 = G__get_type(*bufm2);
   if (fundType1 == 'q' || fundType2 == 'q') {
      G__OP2_divide_T<long double>(bufm1, bufm2);
   }
   else if (fundType1 == 'd' || fundType2 == 'd') {
      G__OP2_divide_T<double>(bufm1, bufm2);
   }
   else if (fundType1 == 'f' || fundType2 == 'f') {
      G__OP2_divide_T<float>(bufm1, bufm2);
   }
   else if (fundType1 == 'm' || fundType2 == 'm') {
      G__OP2_divide_T<G__uint64>(bufm1, bufm2);
   }
   else if (fundType1 == 'n' || fundType2 == 'n') {
      G__OP2_divide_T<G__int64>(bufm1, bufm2);
   }
   else if (
      (fundType1 == 'b') ||
      (fundType1 == 'r') ||
      (fundType1 == 'h') ||
      (fundType1 == 'k')
   ) {
      G__OP2_divide_T<unsigned long>(bufm1, bufm2);
   }
   else {
      G__OP2_divide_T<long>(bufm1, bufm2);
   }
   bufm2->ref = 0;
}

//______________________________________________________________________________
static void G__OP2_logicaland(G__value* bufm1, G__value* bufm2)
{
   static Reflex::Type bool_type(Reflex::Type::ByTypeInfo(typeid(bool)));
   bool newval = G__convertT<bool>(bufm2) && G__convertT<bool>(bufm1);
   G__setvalue<bool>(bufm2, newval);
   G__value_typenum(*bufm2) = bool_type;
   bufm2->ref = 0;
}

//______________________________________________________________________________
static void G__OP2_logicalor(G__value* bufm1, G__value* bufm2)
{
   static Reflex::Type bool_type(Reflex::Type::ByTypeInfo(typeid(bool)));
   bool newval = G__convertT<bool>(bufm2) || G__convertT<bool>(bufm1);
   G__setvalue<bool>(bufm2, newval);
   G__value_typenum(*bufm2) = bool_type;
   bufm2->ref = 0;
}

//______________________________________________________________________________
struct OperModAssign 
{
   template<class Common, class Left> struct Select {
      static void Apply(const G__value* bufm1, G__value* bufm2)
      {
         // --
#ifdef G__TUNEUP_W_SECURITY
         Common right = G__convertT<Common>(bufm1);
         if (right == 0) {
            G__genericerror("Error: operator %' division by zero");
            return;
         }
         Left val = (Left) (G__getvalue_raw<Left>(*bufm2) % right);
#else // G__TUNEUP_W_SECURITY
         Left val = (G__convertT<Left>(bufm2) % G__convertT<R>(bufm1));
#endif // G__TUNEUP_W_SECURITY
         //return left % right;
         G__setvalue<Left>(bufm2, val); // G__setvalue does not change the type.
         *(Left*)bufm2->ref = val;
      }
   };
   template<class Common> struct Select<Common,bool> {
      static void Apply(const G__value*, G__value*)
      {
         // '%' : unsafe use of type 'bool' in operation
         assert(0); // FIXME: Give error message to user, do not crash!
         return;
      }
   };
};

//______________________________________________________________________________
struct OperDivAssign
{
   template<class Common, class Left> struct Select {
      static void Apply(const G__value* bufm1, G__value* bufm2)
      {
         // --
#ifdef G__TUNEUP_W_SECURITY
         Common right = G__convertT<Common>(bufm1);
         if (right == 0) {
            G__genericerror("Error: operator /' division by zero");
            return;
         }
         Left val = (Left) (G__getvalue_raw<Left>(*bufm2) / right);
#else // G__TUNEUP_W_SECURITY
         Left val = (G__convertT<Left>(bufm2) / G__convertT<R>(bufm1));
#endif // G__TUNEUP_W_SECURITY
         G__setvalue(bufm2, val);  // G__setvalue does not change the type.
         *(Left*)bufm2->ref = val;
         //return (L)(left / right);
         return;
      }
   };
   template<class Common> struct Select<Common,bool> {
      static void Apply(const G__value*, G__value*)
      {
         // '/' : unsafe use of type 'bool' in operation
         assert(0); // FIXME: Give error message to user, do not crash!
         return;
      }
   };
};

//______________________________________________________________________________
struct OperMulAssign
{
   template<class Common, class Left> 
   struct Select {
      static void Apply(const G__value* bufm1, G__value* bufm2)
      {
         Left val = (Left) (G__getvalue_raw<Left>(*bufm2) * G__convertT<Common>(bufm1));
         G__setvalue(bufm2, val);  // G__setvalue does not change the type.
         *(Left*)bufm2->ref = val;
      }
   };
};

//______________________________________________________________________________
struct OperAddAssign
{
   template<class Common, class Left> struct Select {
      static void Apply(const G__value* bufm1, G__value* bufm2)
      {
         Left val = (Left) (G__getvalue_raw<Left>(*bufm2) + G__convertT<Common>(bufm1));
         G__setvalue(bufm2, val);  // G__setvalue does not change the type.
         *(Left*)bufm2->ref = val;
      }
   };
};

//______________________________________________________________________________
struct OperSubAssign
{
   template<class Common, class Left> struct Select {
      static void Apply(const G__value* bufm1, G__value* bufm2)
      {
         Left val = (Left) (G__getvalue_raw<Left>(*bufm2) - G__convertT<Common>(bufm1));
         G__setvalue(bufm2, val);  // G__setvalue does not change the type.
         *(Left*)bufm2->ref = val;
      }
   };
};

//______________________________________________________________________________
struct OperLess
{
   template<class Common, class Left> struct Select {
      static void Apply(const G__value* bufm1, G__value* bufm2)
      {
         static Reflex::Type bool_type(Reflex::Type::ByTypeInfo(typeid(bool)));
         bool result = (G__convertT<Common>(bufm2) < G__convertT<Common>(bufm1));
         G__setvalue<bool>(bufm2, result);
         G__value_typenum(*bufm2) = bool_type;
         bufm2->ref = 0;
         //return (L)(left < right);
      }
   };
};

//______________________________________________________________________________
struct OperLessOrEqual
{
   template<class Common, class Left> struct Select {
      static void Apply(const G__value* bufm1, G__value* bufm2)
      {
         static Reflex::Type bool_type(Reflex::Type::ByTypeInfo(typeid(bool)));
         bool result = (G__convertT<Common>(bufm2) <= G__convertT<Common>(bufm1));
         G__setvalue<bool>(bufm2, result);
         G__value_typenum(*bufm2) = bool_type;
         bufm2->ref = 0;
         //return (L)(left <= right);
      }
   };
};

//______________________________________________________________________________
struct OperGreater
{
   template<class Common, class Left> struct Select {
      static void Apply(const G__value* bufm1, G__value* bufm2)
      {
         static Reflex::Type bool_type(Reflex::Type::ByTypeInfo(typeid(bool)));
         bool result = (G__convertT<Common>(bufm2) > G__convertT<Common>(bufm1));
         G__setvalue<bool>(bufm2, result);
         G__value_typenum(*bufm2) = bool_type;
         bufm2->ref = 0;
         //return (L)(left < right);
      }
   };
};

//______________________________________________________________________________
struct OperGreaterOrEqual
{
   template<class Common, class Left> struct Select {
      static void Apply(const G__value* bufm1, G__value* bufm2)
      {
         static Reflex::Type bool_type(Reflex::Type::ByTypeInfo(typeid(bool)));
         bool result = (G__convertT<Common>(bufm2) >= G__convertT<Common>(bufm1));
         G__setvalue<bool>(bufm2, result);
         G__value_typenum(*bufm2) = bool_type;
         bufm2->ref = 0;
         //return (L)(left <= right);
      }
   };
};

//______________________________________________________________________________
struct OperEqual
{
   template<class Common, class Left> struct Select {
      static void Apply(const G__value* bufm1, G__value* bufm2)
      {
         static Reflex::Type bool_type(Reflex::Type::ByTypeInfo(typeid(bool)));
         bool result = (G__convertT<Common>(bufm2) == G__convertT<Common>(bufm1));
         G__setvalue<bool>(bufm2, result);
         G__value_typenum(*bufm2) = bool_type;
         bufm2->ref = 0;
         //return (L)(left < right);
      }
   };
};

//______________________________________________________________________________
struct OperNotEqual
{
   template<class Common, class Left> struct Select {
      static void Apply(const G__value* bufm1, G__value* bufm2)
      {
         static Reflex::Type bool_type(Reflex::Type::ByTypeInfo(typeid(bool)));
         bool result = (G__convertT<Common>(bufm2) != G__convertT<Common>(bufm1));
         G__setvalue<bool>(bufm2, result);
         G__value_typenum(*bufm2) = bool_type;
         bufm2->ref = 0;
         //return (L)(left < right);
      }
   };
};

//______________________________________________________________________________
template<class Common, class Left, class Oper>
static inline void G__OP2_apply_left_right(G__value* bufm1, G__value* bufm2)
{
   Oper::template Select<Common,Left>::Apply(bufm1,bufm2); 
   //L val = Oper::Apply(G__convertT<L>(bufm2), G__convertT<R>(bufm1));
   //G__setvalue<Ret>(bufm2, val);
   //*(L*)bufm2->ref = val;
}

//______________________________________________________________________________
template<class Common, class Oper> static void G__OP2_apply_int_common(G__value* bufm1, G__value* bufm2)
{
   char fundType2 = G__get_type(*bufm2);
   switch (fundType2) {
      case 'c':
         G__OP2_apply_left_right<Common, char, Oper>(bufm1, bufm2);
         break;
      case 's':
         G__OP2_apply_left_right<Common, short, Oper>(bufm1, bufm2);
         break;
      case 'i':
         G__OP2_apply_left_right<Common, int, Oper>(bufm1, bufm2);
         break;
      case 'l':
         G__OP2_apply_left_right<Common, long, Oper>(bufm1, bufm2);
         break;
      case 'b':
         G__OP2_apply_left_right<Common, unsigned char, Oper>(bufm1, bufm2);
         break;
      case 'r':
         G__OP2_apply_left_right<Common, unsigned short, Oper>(bufm1, bufm2);
         break;
      case 'h':
         G__OP2_apply_left_right<Common, unsigned int, Oper>(bufm1, bufm2);
         break;
      case 'k':
         G__OP2_apply_left_right<Common, unsigned long, Oper>(bufm1, bufm2);
         break;
      case 'g':
         G__OP2_apply_left_right<Common, bool, Oper>(bufm1, bufm2);
         break;
      case 'n':
         G__OP2_apply_left_right<Common, G__int64, Oper>(bufm1, bufm2);
         break;
      case 'm':
         G__OP2_apply_left_right<Common, G__uint64, Oper>(bufm1, bufm2);
         break;
      default:
         assert(0); // FIXME: No crashing allowed in production code!
   }
}

//______________________________________________________________________________
template<class Oper> static void G__OP2_apply_int(G__value* bufm1, G__value* bufm2)
{
   char fundType1 = G__get_type(*bufm1);
   char fundType2 = G__get_type(*bufm2);
   if (fundType1 == 'm' || fundType2 == 'm') {
      G__OP2_apply_int_common<G__uint64, Oper>(bufm1, bufm2);
   }
   else if (fundType1 == 'n' || fundType2 == 'n') {
      G__OP2_apply_int_common<G__int64, Oper>(bufm1, bufm2);
   }
   else if (fundType1 == 'k' || fundType2 == 'k') {
      G__OP2_apply_int_common<unsigned long, Oper>(bufm1, bufm2);
   }
   else if (fundType1 == 'l' || fundType2 == 'l') {
      G__OP2_apply_int_common<long, Oper>(bufm1, bufm2);
   }
   else if (fundType1 == 'h' || fundType2 == 'h') {
      G__OP2_apply_int_common<unsigned int, Oper>(bufm1, bufm2);
   }
   else {
      G__OP2_apply_int_common<int, Oper>(bufm1, bufm2);
   }
}

//______________________________________________________________________________
template<class Common, class Oper> static void G__OP2_apply_common(G__value* bufm1, G__value* bufm2)
{
   char fundType2 = G__get_type(*bufm2);
   switch (fundType2) {
      case 'c':
         G__OP2_apply_left_right<Common, char, Oper>(bufm1, bufm2);
         break;
      case 's':
         G__OP2_apply_left_right<Common, short, Oper>(bufm1, bufm2);
         break;
      case 'i':
         G__OP2_apply_left_right<Common, int, Oper>(bufm1, bufm2);
         break;
      case 'l':
         G__OP2_apply_left_right<Common, long, Oper>(bufm1, bufm2);
         break;
      case 'b':
         G__OP2_apply_left_right<Common, unsigned char, Oper>(bufm1, bufm2);
         break;
      case 'r':
         G__OP2_apply_left_right<Common, unsigned short, Oper>(bufm1, bufm2);
         break;
      case 'h':
         G__OP2_apply_left_right<Common, unsigned int, Oper>(bufm1, bufm2);
         break;
      case 'k':
         G__OP2_apply_left_right<Common, unsigned long, Oper>(bufm1, bufm2);
         break;
      case 'g':
         G__OP2_apply_left_right<Common, bool, Oper>(bufm1, bufm2);
         break;
      case 'f':
         G__OP2_apply_left_right<Common, float, Oper>(bufm1, bufm2);
         break;
      case 'd':
         G__OP2_apply_left_right<Common, double, Oper>(bufm1, bufm2);
         break;
      case 'q':
         G__OP2_apply_left_right<Common, long double, Oper>(bufm1, bufm2);
         break;
      //case 'y': // void
      case 'n':
         G__OP2_apply_left_right<Common, G__int64, Oper>(bufm1, bufm2);
         break;
      case 'm':
         G__OP2_apply_left_right<Common, G__uint64, Oper>(bufm1, bufm2);
         break;
      default:
         if (isupper(fundType2)) {
            G__OP2_apply_left_right<Common, long, Oper>(bufm1, bufm2);
         }
         else {
            assert(0); // FIXME: No crashing in production code!
         }
   }
}

//______________________________________________________________________________
template<class Oper> static void G__OP2_apply(G__value* bufm1, G__value* bufm2)
{
   char fundType1 = G__get_type(*bufm1);
   char fundType2 = G__get_type(*bufm2);
   if (fundType1 == 'q' || fundType2 == 'q') {
      G__OP2_apply_common<long double, Oper>(bufm1, bufm2);
   }
   else if (fundType1 == 'd' || fundType2 == 'd') {
      G__OP2_apply_common<double, Oper>(bufm1, bufm2);
   }
   else if (fundType1 == 'f' || fundType2 == 'f') {
      G__OP2_apply_common<float, Oper>(bufm1, bufm2);
   }
   else if (fundType1 == 'm' || fundType2 == 'm') {
      G__OP2_apply_common<G__uint64, Oper>(bufm1, bufm2);
   }
   else if (fundType1 == 'n' || fundType2 == 'n') {
      G__OP2_apply_common<G__int64, Oper>(bufm1, bufm2);
   }
   else if (fundType1 == 'k' || fundType2 == 'k') {
      G__OP2_apply_common<unsigned long, Oper>(bufm1, bufm2);
   }
   else if (fundType1 == 'l' || fundType2 == 'l') {
      G__OP2_apply_common<long, Oper>(bufm1, bufm2);
   }
   else if (fundType1 == 'h' || fundType2 == 'h') {
      G__OP2_apply_common<unsigned int, Oper>(bufm1, bufm2);
   }
   else if (isupper(fundType1) && isupper(fundType2)) {
      // We have pointers.
      G__OP2_apply_common<long, Oper>(bufm1, bufm2);
   }
   else {
      G__OP2_apply_common<int, Oper>(bufm1, bufm2);
   }
}

//______________________________________________________________________________
void Cint::Internal::G__CMP2_equal(G__value* bufm1, G__value* bufm2)
{
   // If the value are pointers, we might first need to convert them to a 
   // potential common base class pointer.
   if(isupper(G__get_type(*bufm1)) && isupper(G__get_type(*bufm2))) {
      G__publicinheritance(bufm1,bufm2);
      OperEqual::Select<long,long>::Apply(bufm1,bufm2);
   } else {
      G__OP2_apply<OperEqual>(bufm1,bufm2);
   }
}

//______________________________________________________________________________
static void G__CMP2_notequal(G__value* bufm1, G__value* bufm2)
{
   // If the value are pointers, we might first need to convert them to a 
   // potential common base class pointer.
   if(isupper(G__get_type(*bufm1)) && isupper(G__get_type(*bufm2))) {
      G__publicinheritance(bufm1,bufm2);
      OperNotEqual::Select<long,long>::Apply(bufm1,bufm2);
   } else {
      G__OP2_apply<OperNotEqual>(bufm1,bufm2);
   }
}

//______________________________________________________________________________
//
//  G__OP1_OPTIMIZED
//
//  Note: None of these functions can be inline because we take their address.
//

//______________________________________________________________________________
template<class T> static void G__OP1_postfixinc_T(G__value* pbuf)
{
   *(T*)pbuf->ref = G__getvalue_raw<T>(*pbuf) + 1;
   pbuf->ref = (long)&(G__value_ref<T>(*pbuf));
}

//______________________________________________________________________________
template<class T> static void G__OP1_postfixdec_T(G__value* pbuf)
{
   *(T*)pbuf->ref = G__getvalue_raw<T>(*pbuf) - 1;
   pbuf->ref = (long)&(G__value_ref<T>(*pbuf));
}

//______________________________________________________________________________
template<class T> static void G__OP1_prefixinc_T(G__value* pbuf)
{
   T v = G__getvalue_raw<T>(*pbuf);
   ++v;
   G__setvalue<T>(pbuf, v);
   *(T*)pbuf->ref = v;
}

//______________________________________________________________________________
template<class T> static void G__OP1_prefixdec_T(G__value *pbuf)
{
   T v = G__getvalue_raw<T>(*pbuf);
   --v;
   G__setvalue<T>(pbuf, v);
   *(T*)pbuf->ref = v;
}

//______________________________________________________________________________
static void G__OP1_postfixinc(G__value* pbuf)
{
   char type = G__get_type(*pbuf);
   if (isupper(type)) {
      *(long*)pbuf->ref = G__getvalue_raw<long>(*pbuf) + G__sizeof_deref(pbuf);
      return;
   }
   switch (type) {
      case 'c':
         G__OP1_postfixinc_T<char>(pbuf);
         break;
      case 's':
         G__OP1_postfixinc_T<short>(pbuf);
         break;
      case 'i':
         G__OP1_postfixinc_T<int>(pbuf);
         break;
      case 'l':
         G__OP1_postfixinc_T<long>(pbuf);
         break;
      case 'b':
         G__OP1_postfixinc_T<unsigned char>(pbuf);
         break;
      case 'r':
         G__OP1_postfixinc_T<unsigned short>(pbuf);
         break;
      case 'h':
         G__OP1_postfixinc_T<unsigned int>(pbuf);
         break;
      case 'k':
         G__OP1_postfixinc_T<unsigned long>(pbuf);
         break;
      case 'g':
         // --
#ifdef G__BOOL4BYTE
         G__OP1_postfixinc_T<int>(pbuf);
#else // G__BOOL4BYTE
         G__OP1_postfixinc_T<unsigned char>(pbuf);
#endif // G__BOOL4BYTE
         break;
      case 'f':
         G__OP1_postfixinc_T<float>(pbuf);
         break;
      case 'd':
         G__OP1_postfixinc_T<double>(pbuf);
         break;
      case 'q':
         G__OP1_postfixinc_T<long double>(pbuf);
         break;
      case 'y': // not possible
         break;
      case 'n':
         G__OP1_postfixinc_T<G__int64>(pbuf);
         break;
      case 'm':
         G__OP1_postfixinc_T<G__uint64>(pbuf);
         break;
      default: // not possible
         break;
   }
}

//______________________________________________________________________________
static void G__OP1_postfixdec(G__value* pbuf)
{
   char type = G__get_type(*pbuf);
   if (isupper(type)) {
      *(long*)pbuf->ref = G__getvalue_raw<long>(*pbuf) - G__sizeof_deref(pbuf);
      return;
   }
   switch (type) {
      case 'c':
         G__OP1_postfixdec_T<char>(pbuf);
         break;
      case 's':
         G__OP1_postfixdec_T<short>(pbuf);
         break;
      case 'i':
         G__OP1_postfixdec_T<int>(pbuf);
         break;
      case 'l':
         G__OP1_postfixdec_T<long>(pbuf);
         break;
      case 'b':
         G__OP1_postfixdec_T<unsigned char>(pbuf);
         break;
      case 'r':
         G__OP1_postfixdec_T<unsigned short>(pbuf);
         break;
      case 'h':
         G__OP1_postfixdec_T<unsigned int>(pbuf);
         break;
      case 'k':
         G__OP1_postfixdec_T<unsigned long>(pbuf);
         break;
      case 'g':
         // --
#ifdef G__BOOL4BYTE
         G__OP1_postfixdec_T<int>(pbuf);
#else // G__BOOL4BYTE
         G__OP1_postfixdec_T<unsigned char>(pbuf);
#endif // G__BOOL4BYTE
         break;
      case 'f':
         G__OP1_postfixdec_T<float>(pbuf);
         break;
      case 'd':
         G__OP1_postfixdec_T<double>(pbuf);
         break;
      case 'q':
         G__OP1_postfixdec_T<long double>(pbuf);
         break;
      case 'y': // not possible
         break;
      case 'n':
         G__OP1_postfixdec_T<G__int64>(pbuf);
         break;
      case 'm':
         G__OP1_postfixdec_T<G__uint64>(pbuf);
         break;
      default: // not possible
         break;
   }
}

//______________________________________________________________________________
static void G__OP1_prefixinc(G__value* pbuf)
{
   char type = G__get_type(*pbuf);
   if (isupper(type)) {
      size_t v = G__getvalue_raw<long>(*pbuf);
      v += G__sizeof_deref(pbuf);
      G__setvalue<long>(pbuf, v);
      *(long*)pbuf->ref = v;
      return;
   }
   switch (type) {
      case 'c':
         G__OP1_prefixinc_T<char>(pbuf);
         break;
      case 's':
         G__OP1_prefixinc_T<short>(pbuf);
         break;
      case 'i':
         G__OP1_prefixinc_T<int>(pbuf);
         break;
      case 'l':
         G__OP1_prefixinc_T<long>(pbuf);
         break;
      case 'b':
         G__OP1_prefixinc_T<unsigned char>(pbuf);
         break;
      case 'r':
         G__OP1_prefixinc_T<unsigned short>(pbuf);
         break;
      case 'h':
         G__OP1_prefixinc_T<unsigned int>(pbuf);
         break;
      case 'k':
         G__OP1_prefixinc_T<unsigned long>(pbuf);
         break;
      case 'g':
         // --
#ifdef G__BOOL4BYTE
         G__OP1_prefixinc_T<int>(pbuf);
#else // G__BOOL4BYTE
         G__OP1_prefixinc_T<unsigned char>(pbuf);
#endif // G__BOOL4BYTE
         break;
      case 'f':
         G__OP1_prefixinc_T<float>(pbuf);
         break;
      case 'd':
         G__OP1_prefixinc_T<double>(pbuf);
         break;
      case 'q':
         G__OP1_prefixinc_T<long double>(pbuf);
         break;
      case 'y': // cannot happen
         break;
      case 'n':
         G__OP1_prefixinc_T<G__int64>(pbuf);
         break;
      case 'm':
         G__OP1_prefixinc_T<G__uint64>(pbuf);
         break;
      default: // cannot happen
         break;
   }
}

//______________________________________________________________________________
static void G__OP1_prefixdec(G__value* pbuf)
{
   char type = G__get_type(*pbuf);
   if (isupper(type)) {
      size_t v = G__getvalue_raw<long>(*pbuf);
      v -= G__sizeof_deref(pbuf);
      G__setvalue<long>(pbuf, v);
      *(long*)pbuf->ref = v;
      return;
   }
   switch (type) {
      case 'c':
         G__OP1_prefixdec_T<char>(pbuf);
         break;
      case 's':
         G__OP1_prefixdec_T<short>(pbuf);
         break;
      case 'i':
         G__OP1_prefixdec_T<int>(pbuf);
         break;
      case 'l':
         G__OP1_prefixdec_T<long>(pbuf);
         break;
      case 'b':
         G__OP1_prefixdec_T<unsigned char>(pbuf);
         break;
      case 'r':
         G__OP1_prefixdec_T<unsigned short>(pbuf);
         break;
      case 'h':
         G__OP1_prefixdec_T<unsigned int>(pbuf);
         break;
      case 'k':
         G__OP1_prefixdec_T<unsigned long>(pbuf);
         break;
      case 'g':
         // --
#ifdef G__BOOL4BYTE
         G__OP1_prefixdec_T<int>(pbuf);
#else // G__BOOL4BYTE
         G__OP1_prefixdec_T<unsigned char>(pbuf);
#endif // G__BOOL4BYTE
         break;
      case 'f':
         G__OP1_prefixdec_T<float>(pbuf);
         break;
      case 'd':
         G__OP1_prefixdec_T<double>(pbuf);
         break;
      case 'q':
         G__OP1_prefixdec_T<long double>(pbuf);
         break;
      case ::Reflex::kVOID: // cannot happen
         break;
      case 'n':
         G__OP1_prefixdec_T<G__int64>(pbuf);
         break;
      case 'm':
         G__OP1_prefixdec_T<G__uint64>(pbuf);
         break;
      default: // cannot happen
         break;
   }
}
//______________________________________________________________________________
template<class T> static inline void G__OP1_minus_T(G__value* pbuf)
{
   G__setvalue(pbuf, -G__getvalue_raw<T>(*pbuf));
   pbuf->ref = 0;
}

//______________________________________________________________________________
static void G__OP1_minus(G__value* pbuf)
{
   char type = G__get_type(*pbuf);
   if (isupper(type)) {
      G__genericerror("Error: Illegal pointer operation unary -");
      return;
   }
   switch (type) {
      case 'q':
         G__OP1_minus_T<long double>(pbuf);
         break;
      case 'd':
         G__OP1_minus_T<double>(pbuf);
         break;
      case 'f':
         G__OP1_minus_T<float>(pbuf);
         break;
      case 'm':
         // -unsigned still unsigned
         //G__OP1_minus_T<G__uint64>(pbuf);
         break;
      case 'n':
         G__OP1_minus_T<G__int64>(pbuf);
         break;
      case 'b':
      case 'r':
      case 'h':
      case 'k':
         // -unsigned still unsigned
         //G__OP1_minus_T<unsigned long>(pbuf);
         break;
      default:
         G__OP1_minus_T<long>(pbuf);
         break;
   }
}

//______________________________________________________________________________
//
//  Optimization level 1 functions.
//

//______________________________________________________________________________
void Cint::Internal::G__suspendbytecode()
{
   G__asm_noverflow = 0;
}

//______________________________________________________________________________
void Cint::Internal::G__resetbytecode()
{
   G__asm_noverflow = 0;
}

//______________________________________________________________________________
void Cint::Internal::G__abortbytecode()
{
   if (G__asm_dbg && G__asm_noverflow) {
      if (G__dispmsg >= G__DISPNOTE) {
         if (!G__xrefflag) {
            G__fprinterr(G__serr, "Note: Bytecode compiler stops at this line. Enclosing loop or function may be slow %d", G__asm_noverflow);
         }
         else {
            G__fprinterr(G__serr, "Note: Bytecode limitation encountered but compiler continues for Local variable cross referencing");
         }
         G__printlinenum();
      }
   }
   if (!G__xrefflag) {
      G__asm_noverflow = 0;
   }
}

//______________________________________________________________________________
int Cint::Internal::G__inc_cp_asm(int cp_inc, int dt_dec)
{
   //  Increment program counter(G__asm_cp) and decrement stack pointer
   // (G__asm_dt) at compile time.
   //  If Quasi-Assembly-Code or constant data exceeded instruction
   // and data buffer, G__asm_noverflow is reset and compilation is void.
   if (!G__xrefflag) {
      G__asm_cp += cp_inc;
      G__asm_dt -= dt_dec;
   }
   if (G__asm_instsize && (G__asm_cp > (G__asm_instsize - 8))) {
      G__asm_instsize += 0x100;
      void* p = realloc((void*) G__asm_stack, G__asm_instsize * sizeof(long));
      if (!p) {
         G__genericerror("Error: memory exhausted for bytecode instruction buffer\n");
      }
      G__asm_inst = (long*) p;
   }
   else if (!G__asm_instsize && (G__asm_cp > (G__MAXINST - 8))) {
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Warning: loop compile instruction overflow");
         G__printlinenum();
      }
      G__abortbytecode();
   }

   if (G__asm_dt < 30) {
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Warning: loop compile data overflow");
         G__printlinenum();
      }
      G__abortbytecode();
   }
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__clear_asm()
{
   // Reset instruction and data buffer.
   // This function is called at the beginning of compilation.
   G__asm_cp = 0;
   G__asm_dt = G__MAXSTACK - 1;
   G__asm_name_p = 0;
   G__asm_cond_cp = -1; // avoid wrong optimization
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__asm_clear()
{
   if (G__asm_clear_mask) {
      return 0;
   }
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "%3x,%3x: CL  FILE:%s LINE:%d  %s:%d\n", G__asm_cp, G__asm_dt, G__ifile.name, G__ifile.line_number, __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
   // Issue, value of G__CL must be unique. Otherwise, following optimization
   // causes problem. There is similar issue, but the biggest risk is here.
   if (
      (G__asm_cp >= 2) &&
      (G__asm_inst[G__asm_cp-2] == G__CL) &&
      ((G__asm_inst[G__asm_cp-1] & 0xffff0000) == 0x7fff0000)
   ) {
      G__inc_cp_asm(-2, 0);
   }
   G__asm_inst[G__asm_cp] = G__CL;
   G__asm_inst[G__asm_cp+1] = (G__ifile.line_number & G__CL_LINEMASK) + ((G__ifile.filenum & G__CL_FILEMASK) * G__CL_FILESHIFT);
   G__inc_cp_asm(2, 0);
   return 0;
}

#endif // G__ASM

#ifdef G__ASM
//______________________________________________________________________________
int Cint::Internal::G__asm_putint(int i)
{
   // --
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "%3x,%3x: LD %d from %x  %s:%d\n", G__asm_cp, G__asm_dt, i, G__asm_dt, __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
   G__asm_inst[G__asm_cp] = G__LD;
   G__asm_inst[G__asm_cp+1] = G__asm_dt;
   G__letint(&G__asm_stack[G__asm_dt], 'i', (long) i);
   G__inc_cp_asm(2, 1);
   return 0;
}
#endif // G__ASM

//______________________________________________________________________________
G__value Cint::Internal::G__getreserved(const char* item , void** /* ptr */, void** /* ppdict */)
{
   G__value buf;
   int i;

   G__abortbytecode();

   if (strcmp(item, "LINE") == 0 || strcmp(item, "_LINE__") == 0) {
      i = G__RSVD_LINE;
#ifdef G__ASM
      if (G__asm_noverflow) G__asm_putint(i);
#endif
   }
   else if (strcmp(item, "FILE") == 0 || strcmp(item, "_FILE__") == 0) {
      i = G__RSVD_FILE;
#ifdef G__ASM
      if (G__asm_noverflow) G__asm_putint(i);
#endif
   }
   else if (strcmp(item, "_DATE__") == 0) {
      i = G__RSVD_DATE;
#ifdef G__ASM
      if (G__asm_noverflow) G__asm_putint(i);
#endif
   }
   else if (strcmp(item, "_TIME__") == 0) {
      i = G__RSVD_TIME;
#ifdef G__ASM
      if (G__asm_noverflow) G__asm_putint(i);
#endif
   }
   else if (strcmp(item, "#") == 0) {
      i = G__RSVD_ARG;
#ifdef G__ASM
      if (G__asm_noverflow) G__asm_putint(i);
#endif
   }
   else if (isdigit(item[0])) {
      i = atoi(item);
#ifdef G__ASM
      if (G__asm_noverflow) G__asm_putint(i);
#endif
   }
   else {
      i = 0;
      buf = G__null;
   }

   if (i) {
      buf = G__getrsvd(i);
#ifdef G__ASM
      if (G__asm_noverflow) {
#ifdef G__ASM_DBG
         if (G__asm_dbg) G__fprinterr(G__serr, "%3x: GETRSVD $%s\n" , G__asm_cp, item);
#endif
         /* GETRSVD */
         G__asm_inst[G__asm_cp] = G__GETRSVD;
         G__inc_cp_asm(1, 0);
      }
#endif
   }
   return(buf);
}

//______________________________________________________________________________
static void G__get__tm__(char *buf)
{
  // -- Returns 'Sun Nov 28 21:40:32 1999\n' in buf.
  time_t t = time(0);
  sprintf(buf,"%s",ctime(&t));
}

//______________________________________________________________________________
static char* G__get__date__()
{
   int i = 0, j = 0;
   char buf[80];
   static char result[80];
   G__get__tm__(buf);
   while (buf[i] && !isspace(buf[i])) ++i; /* skip 'Sun' */
   while (buf[i] && isspace(buf[i])) ++i;
   while (buf[i] && !isspace(buf[i])) result[j++] = buf[i++]; /* copy 'Nov' */
   while (buf[i] && isspace(buf[i])) result[j++] = buf[i++];
   while (buf[i] && !isspace(buf[i])) result[j++] = buf[i++]; /* copy '28' */
   while (buf[i] && isspace(buf[i])) result[j++] = buf[i++];
   while (buf[i] && !isspace(buf[i])) ++i; /* skip '21:41:10' */
   while (buf[i] && isspace(buf[i])) ++i;
   while (buf[i] && !isspace(buf[i])) result[j++] = buf[i++]; /* copy '1999' */
   result[j] = 0;
   return(result);
}

//______________________________________________________________________________
static char* G__get__time__()
{
   int i = 0, j = 0;
   char buf[80];
   static char result[80];
   G__get__tm__(buf);
   while (buf[i] && !isspace(buf[i])) ++i; /* skip 'Sun' */
   while (buf[i] && isspace(buf[i])) ++i;
   while (buf[i] && !isspace(buf[i])) ++i; /* skip 'Nov' */
   while (buf[i] && isspace(buf[i])) ++i;
   while (buf[i] && !isspace(buf[i])) ++i; /* skip '28' */
   while (buf[i] && isspace(buf[i])) ++i;
   while (buf[i] && !isspace(buf[i])) result[j++] = buf[i++];/*copy '21:41:10'*/
   result[j] = 0;
   return(result);
}

//______________________________________________________________________________
G__value Cint::Internal::G__getrsvd(int i)
{
   G__value buf = G__null;
   switch (i) {
      case G__RSVD_LINE:
         G__letint(&buf, 'i', (long)G__ifile.line_number);
         break;
      case G__RSVD_FILE:
         if (0 <= G__ifile.filenum && G__ifile.filenum < G__MAXFILE &&
               G__srcfile[G__ifile.filenum].filename) {
            G__letint(&buf, 'C', (long)G__srcfile[G__ifile.filenum].filename);
         }
         else {
            G__letint(&buf, 'C', (long)0);
         }
         break;
      case G__RSVD_ARG:
         G__letint(&buf, 'i', (long)G__argn);
         break;
      case G__RSVD_DATE:
         G__letint(&buf, 'C', (long)G__get__date__());
         break;
      case G__RSVD_TIME:
         G__letint(&buf, 'C', (long)G__get__time__());
         break;
      default:
         G__letint(&buf, 'C', (long)G__arg[i]);
         break;
   }
   return buf;
}


//______________________________________________________________________________
//
//  Optimization level 2 functions.
//

//______________________________________________________________________________
long Cint::Internal::G__asm_gettest(int op, long* inst)
{
   switch (op) {
      case 'E': /* != */
         *inst = (long) G__asm_test_E;
         break;
      case 'N': /* != */
         *inst = (long) G__asm_test_N;
         break;
      case 'G': /* >= */
         *inst = (long) G__asm_test_GE;
         break;
      case 'l': /* <= */
         *inst = (long) G__asm_test_LE;
         break;
      case '<':
         *inst = (long) G__asm_test_l;
         break;
      case '>':
         *inst = (long) G__asm_test_g;
         break;
      default:
         G__fprinterr(G__serr, "Error: Loop compile optimizer, illegal conditional instruction %d(%c) FILE:%s LINE:%d\n", op, op, G__ifile.name, G__ifile.line_number);
         break;
   }
   return 0;
}

//______________________________________________________________________________
static int G__isInt(int type)
{
   switch (type) {
      case 'i':
         return 1;
      case 'l':
         if (G__LONGALLOC == G__INTALLOC) {
            return 1;
         }
         return 0;
      case 's':
         if (G__SHORTALLOC == G__INTALLOC) {
            return 1;
         }
         return 0;
   }
   return 0;
}

//______________________________________________________________________________
#define G__OP_VAR1(INST) \
   (instVar1 = ::Reflex::Member(reinterpret_cast<const Reflex::MemberBase*>(G__asm_inst[INST])))

//______________________________________________________________________________
#define G__OP_VAR2(INST) \
   (instVar2 = ::Reflex::Member(reinterpret_cast<const Reflex::MemberBase*>(G__asm_inst[INST])))

//______________________________________________________________________________
int Cint::Internal::G__asm_optimize(int* start)
{
   // Issue, value of G__LD_VAR, G__LD_MVAR, G__LD, G__CMP2, G__CNDJMP must be
   // unique. Otherwise, following optimization causes problem.
   int *pb;
   Reflex::Member instVar1;
   Reflex::Member instVar2;
   /*******************************************************
    * i<100, i<=100, i>0 i>=0 at loop header
    *
    *               before                     ----------> after
    *
    *      0      G__LD_VAR       <- check  1             JMP
    *      1      index           <- check  2  (int)      6
    *      2      paran                                   NOP
    *      3      point_level     <- check  3             NOP
    *      4      *var                     (2)            NOP
    *      5      LD              <- check  1             NOP  bydy of *b
    *      6      data_stack      <- check  2 (int)       CMPJMP
    *      7      CMP2            <- check  1             *compare()
    *      8      <,<=,>,>=,==,!=    case                 *a  var->p[]
    *      9      CNDJMP          <- check  1             *b  ptr to inst[5]
    *     10      next_pc=G__asm_cp                       next_pc=G__asm_cp
    *          .                                           .
    *     -2      JMP                                     JMP
    *     -1      next_pc                                 6
    * G__asm_cp   RTN                                     RTN
    *******************************************************/
   /* AFTER MERGE:
           1      not used
           4      Reflex::Member::Id()                    unchanged
   */
   if ((G__asm_inst[*start] == G__LD_VAR
         || (G__asm_inst[*start] == G__LD_MSTR
             && !G__asm_wholefunction
            ))
         &&  /* 1 */
         G__asm_inst[*start+5] == G__LD      &&
         G__asm_inst[*start+7] == G__CMP2    &&
         G__asm_inst[*start+9] == G__CNDJMP  &&
         G__isInt(G__get_type(G__OP_VAR1(*start + 4).TypeOf())) && /* 2 */
         G__isInt(G__get_type(G__value_typenum(G__asm_stack[G__asm_inst[*start+6]]))) &&
         G__asm_inst[*start+3] == 'p'    /* 3 */
      ) {

#ifdef G__ASM_DBG
      if (G__asm_dbg)
         G__fprinterr(G__serr, "%3x: CMPJMP i %c %d optimized\n"
                      , *start + 6, G__asm_inst[*start+8]
                      , G__int(G__asm_stack[G__asm_inst[*start+6]]));
#endif
      G__asm_gettest((int)G__asm_inst[*start+8]
                     , &G__asm_inst[*start+7]);

      G__asm_inst[*start+8] = (long) G__get_offset(instVar1);
      if (G__asm_inst[*start] == G__LD_MSTR
            && (G__get_properties(instVar1)->statictype != G__LOCALSTATIC)
         )
         G__asm_inst[*start+8] += (long)G__store_struct_offset;

      /* long to int conversion */ /* TODO, Storing ptr to temporary stack buffer, is this Bad? */
      pb = (int*)(&G__asm_inst[*start+5]);
      *pb = G__int(G__asm_stack[G__asm_inst[*start+6]]);
      G__asm_inst[*start+9] = (long)(pb);
      G__asm_inst[*start+6] = G__CMPJMP;
      G__asm_inst[*start] = G__JMP;
      G__asm_inst[*start+1] = *start + 6;
      G__asm_inst[*start+2] = G__NOP;
      G__asm_inst[*start+3] = G__NOP;
      G__asm_inst[*start+4] = G__NOP;
      G__asm_inst[*start+5] = G__NOP;

      *start += 6;
      G__asm_inst[G__asm_cp-1] = *start;
   }
   /*******************************************************
    * i<100, i<=100, i>0 i>=0 at loop header
    *
    *               before                     ----------> after
    *
    *      0      G__LD_VAR,MSTR  <- check  1             JMP
    *      1      index           <- check  2  (int)      9
    *      2      paran                                   NOP
    *      3      point_level     <- check  3             NOP
    *      4      *var                     (2)            NOP
    *      5      G__LD_VAR,MSTR  <- check  1             NOP
    *      6      index           <- check  2  (int)      NOP
    *      7      paran                                   NOP
    *      8      point_level     <- check  3             NOP
    *      9      *var                     (2)            CMPJMP
    *      10     CMP2            <- check  1             *compare()
    *      11     <,<=,>,>=,==,!=    case                 *a  var->[]
    *      12     CNDJMP          <- check  1             *b  var->[]
    *      13     next_pc=G__asm_cp                       next_pc=G__asm_pc
    *          .
    *     -2      JMP                                     JMP
    *     -1      next_pc                                 9
    * G__asm_cp   RTN                                     RTN
    *******************************************************/
   /* AFTER MERGE:
           1      not used
           4      Reflex::Member::Id()                    unchanged
           6      not used
           9      Reflex::Member::Id()                    unchanged
   */
   else if ((G__asm_inst[*start] == G__LD_VAR ||
             (G__asm_inst[*start] == G__LD_MSTR
              && !G__asm_wholefunction
             )) &&  /* 1 */
            (G__asm_inst[*start+5] == G__LD_VAR ||
             (G__asm_inst[*start+5] == G__LD_MSTR
              && !G__asm_wholefunction
             )) &&  /* 1 */
            G__asm_inst[*start+10] == G__CMP2    &&
            G__asm_inst[*start+12] == G__CNDJMP  &&
            G__isInt(G__get_type(G__OP_VAR1(*start + 4).TypeOf())) && /* 2 */
            (G__isInt(G__get_type(G__OP_VAR2(*start + 9).TypeOf())) || /* 2 */
             G__get_type(instVar2.TypeOf()) == 'p') && /* 2 */
            G__asm_inst[*start+3] == 'p'  &&  /* 3 */
            G__asm_inst[*start+8] == 'p'    /* 3 */
           ) {

#ifdef G__ASM_DBG
      if (G__asm_dbg)
         G__fprinterr(G__serr, "%3x: CMPJMP a %c b optimized\n"
                      , *start + 9, G__asm_inst[*start+11]);
#endif
      G__asm_gettest((int)G__asm_inst[*start+11] , &G__asm_inst[*start+10]);

      G__asm_inst[*start+11] = (long) G__get_offset(instVar1);
      if ((G__asm_inst[*start] == G__LD_MSTR)
            && (G__get_properties(instVar1)->statictype != G__LOCALSTATIC)
         )
         G__asm_inst[*start+11] += (long) G__store_struct_offset;

      G__asm_inst[*start+12] = (long) G__get_offset(instVar2);
      if (G__asm_inst[*start+5] == G__LD_MSTR
            && (G__get_properties(instVar2)->statictype != G__LOCALSTATIC)
         )
         G__asm_inst[*start+12] += (long)G__store_struct_offset;

      G__asm_inst[*start+9] = G__CMPJMP;
      G__asm_inst[*start] = G__JMP;
      G__asm_inst[*start+1] = *start + 9;
      G__asm_inst[*start+2] = G__NOP;
      G__asm_inst[*start+3] = G__NOP;
      G__asm_inst[*start+4] = G__NOP;
      G__asm_inst[*start+5] = G__NOP;
      G__asm_inst[*start+6] = G__NOP;
      G__asm_inst[*start+7] = G__NOP;
      G__asm_inst[*start+8] = G__NOP;

      *start += 9;
      G__asm_inst[G__asm_cp-1] = *start;

   }


   /**************************************************************
    * i++ , i-- , ++i , --i at the loop end
    *
    *               before                     ----------> after
    *
    *     -9      G__LD_VAR,LD_MSTR                        INCJMP
    *     -8      index                                    *a  var->p[]
    *     -7      paran                                    1,,-1
    *     -6      point_level                              next_pc
    *     -5      *var                                     NOP
    *     -4      OP1                                      NOP
    *     -3      opr                                      NOP
    *     -2      JMP                                      NOP
    *     -1      next_pc                                  NOP
    * G__asm_cp   RTN                                      RTN
    *******************************************************/
   if (G__asm_inst[G__asm_cp-2] == G__JMP &&
         G__asm_inst[G__asm_cp-4] == G__OP1 &&
         G__asm_inst[G__asm_cp-7] == 0      &&
         G__asm_inst[G__asm_cp-6] == 'p'    &&
         G__asm_cond_cp != G__asm_cp - 2 &&
         (G__LD_VAR == G__asm_inst[G__asm_cp-9] ||
          (G__LD_MSTR == G__asm_inst[G__asm_cp-9]
           && !G__asm_wholefunction
          )) &&
         G__isInt(G__get_type(G__OP_VAR1(G__asm_cp - 5).TypeOf()))) {

#ifdef G__ASM_DBG
      if (G__asm_dbg) G__fprinterr(G__serr, "%3x: INCJMP  i++ optimized\n", G__asm_cp - 9);
#endif
      G__asm_inst[G__asm_cp-8] = (long)G__get_offset(instVar1);
      if (G__asm_inst[G__asm_cp-9] == G__LD_MSTR
            && (G__get_properties(instVar1)->statictype != G__LOCALSTATIC)
         ) {
         G__asm_inst[G__asm_cp-8] += (long) G__store_struct_offset;
      }

      G__asm_inst[G__asm_cp-9] = G__INCJMP;

      switch (G__asm_inst[G__asm_cp-3]) {
         case G__OPR_POSTFIXINC:
         case G__OPR_PREFIXINC:
         case G__OPR_POSTFIXINC_I:
         case G__OPR_PREFIXINC_I:
            G__asm_inst[G__asm_cp-7] = 1;
            break;
         case G__OPR_POSTFIXDEC:
         case G__OPR_PREFIXDEC:
         case G__OPR_POSTFIXDEC_I:
         case G__OPR_PREFIXDEC_I:
            G__asm_inst[G__asm_cp-7] = -1;
            break;
      }
      G__asm_inst[G__asm_cp-6] = G__asm_inst[G__asm_cp-1];
      G__asm_inst[G__asm_cp-5] = G__NOP;
      G__asm_inst[G__asm_cp-4] = G__NOP;
      G__asm_inst[G__asm_cp-3] = G__NOP;

      /*
        G__asm_inst[G__asm_cp-5]=G__RETURN;
        G__asm_cp -= 5 ;
        */

   }

   /**************************************************************
    * i+=1 , i-=1 at the loop end
    *
    *               before                     ----------> after
    *
    *     -11     G__LD_VAR,LD_MSTR                        INCJMP
    *     -10     index                                    *a  var->p[]
    *     -9      paran                                    1,,-1
    *     -8      point_level                              next_pc
    *     -7      *var                                     NOP
    *     -6      G__LD                                    NOP
    *     -5      data_stack                               NOP
    *     -4      OP2                                      NOP
    *     -3      opr                                      NOP
    *     -2      JMP                                      NOP
    *     -1      next_pc                                  NOP
    * G__asm_cp   RTN                           G__asm_cp  RTN
    *******************************************************/
   else if (G__asm_inst[G__asm_cp-2] == G__JMP &&
            G__asm_inst[G__asm_cp-4] == G__OP2 &&
            (G__asm_inst[G__asm_cp-3] == G__OPR_ADDASSIGN ||
             G__asm_inst[G__asm_cp-3] == G__OPR_SUBASSIGN) &&
            G__asm_inst[G__asm_cp-9] == 0      &&
            G__asm_inst[G__asm_cp-8] == 'p'    &&
            G__asm_inst[G__asm_cp-6] == G__LD  &&
            G__asm_cond_cp != G__asm_cp - 2 &&
            (G__LD_VAR == G__asm_inst[G__asm_cp-11] ||
             (G__LD_MSTR == G__asm_inst[G__asm_cp-11]
              && !G__asm_wholefunction
             )) &&
            G__isInt(G__get_type(G__OP_VAR1(G__asm_cp - 7).TypeOf())))  {

#ifdef G__ASM_DBG
      if (G__asm_dbg) G__fprinterr(G__serr, "%3x: INCJMP  i+=1 optimized\n", G__asm_cp - 11);
#endif

      G__asm_inst[G__asm_cp-10] = (long)G__get_offset(instVar1);
      if (G__asm_inst[G__asm_cp-11] == G__LD_MSTR
            && (G__get_properties(instVar1)->statictype != G__LOCALSTATIC)
         ) {
         G__asm_inst[G__asm_cp-10] += (long)G__store_struct_offset;
      }

      G__asm_inst[G__asm_cp-11] = G__INCJMP;

      switch (G__asm_inst[G__asm_cp-3]) {
         case G__OPR_ADDASSIGN:
            G__asm_inst[G__asm_cp-9] = G__int(G__asm_stack[G__asm_inst[G__asm_cp-5]]);
            break;
         case G__OPR_SUBASSIGN:
            G__asm_inst[G__asm_cp-9] = -1 * G__int(G__asm_stack[G__asm_inst[G__asm_cp-5]]);
            break;
      }
      G__asm_inst[G__asm_cp-8] = G__asm_inst[G__asm_cp-1];
      G__asm_inst[G__asm_cp-7] = G__NOP;
      G__asm_inst[G__asm_cp-6] = G__NOP;
      G__asm_inst[G__asm_cp-5] = G__NOP;
      G__asm_inst[G__asm_cp-4] = G__NOP;
      G__asm_inst[G__asm_cp-3] = G__NOP;

      /*
        G__asm_inst[G__asm_cp-7]=G__RETURN;
        G__asm_cp -= 7 ;
        */
   }

   /*******************************************************
    * i=i+N , i=i-N at the loop end
    *
    *               before                     ----------> after
    *
    *     -16     G__LD_VAR,MSTR<- check     1             INCJMP
    *     -15     index         <- check     2             *a  var->p[]
    *     -14     paran         <- check     3             inc
    *     -13     point_level   <- check     3             next_pc
    *     -12     *var          <-          (2)            NOP
    *     -11     LD            <- check     1             NOP
    *     -10     data_stack                               NOP
    *     -9      OP2           <- check     1             NOP
    *     -8      +,-           <- check     2             NOP
    *     -7      G__ST_VAR,MSTR<- check     1             NOP
    *     -6      index         <- check     2             NOP
    *     -5      paran                                    NOP
    *     -4      point_level   <- check     3             NOP
    *     -3      *var          <-          (2)            NOP
    *     -2      JMP                                      NOP
    *     -1      next_pc                                  NOP
    * G__asm_cp   RTN                            G__asm_cp RTN
    *******************************************************/
   else if (G__asm_inst[G__asm_cp-2] == G__JMP &&
            ((G__asm_inst[G__asm_cp-7] == G__ST_VAR &&
              G__asm_inst[G__asm_cp-16] == G__LD_VAR) ||
             (G__asm_inst[G__asm_cp-7] == G__ST_MSTR &&
              G__asm_inst[G__asm_cp-16] == G__LD_MSTR
              && !G__asm_wholefunction
             )) &&
            G__asm_inst[G__asm_cp-9] == G__OP2     &&
            G__asm_inst[G__asm_cp-11] == G__LD     &&
            G__asm_inst[G__asm_cp-15] == G__asm_inst[G__asm_cp-6] && /* 2 */
            G__asm_inst[G__asm_cp-12] == G__asm_inst[G__asm_cp-3] &&
            (G__asm_inst[G__asm_cp-8] == '+' || G__asm_inst[G__asm_cp-8] == '-') &&
            G__isInt(G__get_type(G__OP_VAR1(G__asm_cp - 3).TypeOf()))       &&
            G__asm_inst[G__asm_cp-14] == 0 &&
            G__asm_inst[G__asm_cp-13] == 'p' &&   /* 3 */
            G__asm_inst[G__asm_cp-4] == 'p') {

#ifdef G__ASM_DBG
      if (G__asm_dbg) G__fprinterr(G__serr, "%3x: INCJMP  i=i+1 optimized\n", G__asm_cp - 16);
#endif
      G__asm_inst[G__asm_cp-16] = G__INCJMP;

      G__asm_inst[G__asm_cp-15] = (long)G__get_offset(instVar1);
      if (G__asm_inst[G__asm_cp-7] == G__ST_MSTR
            && (G__get_properties(instVar1)->statictype != G__LOCALSTATIC)
         )
         G__asm_inst[G__asm_cp-15] += (long)G__store_struct_offset;

      G__asm_inst[G__asm_cp-14] = G__int(G__asm_stack[G__asm_inst[G__asm_cp-10]]);
      if (G__asm_inst[G__asm_cp-8] == '-')
         G__asm_inst[G__asm_cp-14] *= -1;

      G__asm_inst[G__asm_cp-13] = G__asm_inst[G__asm_cp-1];
      G__asm_inst[G__asm_cp-12] = G__NOP;
      G__asm_inst[G__asm_cp-11] = G__NOP;
      G__asm_inst[G__asm_cp-10] = G__NOP;
      G__asm_inst[G__asm_cp-9] = G__NOP;
      G__asm_inst[G__asm_cp-8] = G__NOP;
      G__asm_inst[G__asm_cp-7] = G__NOP;
      G__asm_inst[G__asm_cp-6] = G__NOP;
      G__asm_inst[G__asm_cp-5] = G__NOP;
      G__asm_inst[G__asm_cp-4] = G__NOP;
      G__asm_inst[G__asm_cp-3] = G__NOP;

      /*
        G__asm_inst[G__asm_cp-12]=G__RETURN;
        G__asm_cp -= 12 ;
        */
   }
   // Optimization level 3.
   if (G__asm_loopcompile >= 3) {
      G__asm_optimize3(start);
   }
   return 0;
}


//______________________________________________________________________________
//
//  Optimization level 3 functions.
//

//______________________________________________________________________________
static int G__get_LD_p0_p2f(int type, long* pinst)
{
   typedef void (*LD_p0_func_t)(G__value *pbuf, int *psp, long offset, const Reflex::Member& var);
   int done = 1;
   if (isupper(type)) {
      if (type == 'Z') {
         done = 0;
      }
#ifndef G__OLDIMMPLEMENTATION1341
      else if ((type == 'P') || (type == 'O')) {
         *pinst = (long)(LD_p0_func_t)G__LD_p0<double>;
      }
#endif // G__OLDIMMPLEMENTATION1341
      else {
         *pinst = (long)(LD_p0_func_t)G__LD_p0_pointer;
      }
   }
   else {
      switch (type) {
         case 'b':
            *pinst = (long)(LD_p0_func_t)G__LD_p0<unsigned char>;
            break;
         case 'c':
            *pinst = (long)(LD_p0_func_t)G__LD_p0<char>;
            break;
         case 'r':
            *pinst = (long)(LD_p0_func_t)G__LD_p0<unsigned short>;
            break;
         case 's':
            *pinst = (long)(LD_p0_func_t)G__LD_p0<short>;
            break;
         case 'h':
            *pinst = (long)(LD_p0_func_t)G__LD_p0<unsigned int>;
            break;
         case 'i':
            *pinst = (long)(LD_p0_func_t)G__LD_p0<int>;
            break;
         case 'k':
            *pinst = (long)(LD_p0_func_t)G__LD_p0<unsigned long>;
            break;
         case 'l':
            *pinst = (long)(LD_p0_func_t)G__LD_p0<long>;
            break;
         case 'u':
            *pinst = (long)(LD_p0_func_t)G__LD_p0_struct;
            break;
         case 'f':
            *pinst = (long)(LD_p0_func_t)G__LD_p0<float>;
            break;
         case 'd':
            *pinst = (long)(LD_p0_func_t)G__LD_p0<double>;
            break;
         case 'g':
            *pinst = (long)(LD_p0_func_t)G__LD_p0<bool>;
            break;
         case 'n':
            *pinst = (long)(LD_p0_func_t)G__LD_p0<G__int64>;
            break;
         case 'm':
            *pinst = (long)(LD_p0_func_t)G__LD_p0<G__uint64>;
            break;
         case 'q':
            *pinst = (long)(LD_p0_func_t)G__LD_p0<long double>;
            break;
         default:
            done = 0;
            break;
      }
   }
   return done;
}

//______________________________________________________________________________
static int G__get_LD_p1_p2f(int type, long* pinst)
{
   typedef void(*LD_p1_func_t)(G__value *pbuf, int *psp, long offset, const Reflex::Member& var);
   int done = 1;
   if (isupper(type)) {
      if ('Z' == type) {
         done = 0;
      }
      else {
         *pinst = (long)(LD_p1_func_t)G__LD_p1_pointer;
      }
   }
   else {
      switch (type) {
         case 'b':
            *pinst = (long)(LD_p1_func_t)G__LD_p1<unsigned char>;
            break;
         case 'c':
            *pinst = (long)(LD_p1_func_t)G__LD_p1<char>;
            break;
         case 'r':
            *pinst = (long)(LD_p1_func_t)G__LD_p1<unsigned short>;
            break;
         case 's':
            *pinst = (long)(LD_p1_func_t)G__LD_p1<short>;
            break;
         case 'h':
            *pinst = (long)(LD_p1_func_t)G__LD_p1<unsigned int>;
            break;
         case 'i':
            *pinst = (long)(LD_p1_func_t)G__LD_p1<int>;
            break;
         case 'k':
            *pinst = (long)(LD_p1_func_t)G__LD_p1<unsigned long>;
            break;
         case 'l':
            *pinst = (long)(LD_p1_func_t)G__LD_p1<long>;
            break;
         case 'u':
            *pinst = (long)(LD_p1_func_t)G__LD_p1_struct;
            break;
         case 'f':
            *pinst = (long)(LD_p1_func_t)G__LD_p1<float>;
            break;
         case 'd':
            *pinst = (long)(LD_p1_func_t)G__LD_p1<double>;
            break;
         case 'g':
            *pinst = (long)(LD_p1_func_t)G__LD_p1<bool>;
            break;
         case 'n':
            *pinst = (long)(LD_p1_func_t)G__LD_p1<G__int64>;
            break;
         case 'm':
            *pinst = (long)(LD_p1_func_t)G__LD_p1<G__uint64>;
            break;
         case 'q':
            *pinst = (long)(LD_p1_func_t)G__LD_p1<long double>;
            break;
         default:
            done = 0;
            break;
      }
   }
   return done;
}

//______________________________________________________________________________
static int G__get_LD_pn_p2f(int type, long* pinst)
{
   typedef void(*LD_pn_func_t)(G__value *pbuf, int *psp, long offset, const Reflex::Member& var);
   int done = 1;
   if (isupper(type)) {
      if ('Z' == type) {
         done = 0;
      }
      else {
         *pinst = (long)(LD_pn_func_t)G__LD_pn_pointer;
      }
   }
   else {
      switch (type) {
         case 'b':
            *pinst = (long)(LD_pn_func_t)G__LD_pn<unsigned char>;
            break;
         case 'c':
            *pinst = (long)(LD_pn_func_t)G__LD_pn<char>;
            break;
         case 'r':
            *pinst = (long)(LD_pn_func_t)G__LD_pn<unsigned short>;
            break;
         case 's':
            *pinst = (long)(LD_pn_func_t)G__LD_pn<short>;
            break;
         case 'h':
            *pinst = (long)(LD_pn_func_t)G__LD_pn<unsigned int>;
            break;
         case 'i':
            *pinst = (long)(LD_pn_func_t)G__LD_pn<int>;
            break;
         case 'k':
            *pinst = (long)(LD_pn_func_t)G__LD_pn<unsigned long>;
            break;
         case 'l':
            *pinst = (long)(LD_pn_func_t)G__LD_pn<long>;
            break;
         case 'u':
            *pinst = (long)(LD_pn_func_t)G__LD_pn_struct;
            break;
         case 'f':
            *pinst = (long)(LD_pn_func_t)G__LD_pn<float>;
            break;
         case 'd':
            *pinst = (long)(LD_pn_func_t)G__LD_pn<double>;
            break;
         case 'g':
            *pinst = (long)(LD_pn_func_t)G__LD_pn<bool>;
            break;
         case 'n':
            *pinst = (long)(LD_pn_func_t)G__LD_pn<G__int64>;
            break;
         case 'm':
            *pinst = (long)(LD_pn_func_t)G__LD_pn<G__uint64>;
            break;
         case 'q':
            *pinst = (long)(LD_pn_func_t)G__LD_pn<long double>;
            break;
         default:
            done = 0;
            break;
      }
   }
   return done;
}

//______________________________________________________________________________
static int G__get_LD_P10_p2f(int type, long* pinst, int reftype)
{
   typedef void(*LD_P10_func_t)(G__value *pbuf, int *psp, long offset, const Reflex::Member& var);
   int done = 1;
   if (G__PARAP2P == reftype) {
      if ('Z' == type) {
         done = 0;
      }
      else {
         *pinst = (long)(LD_P10_func_t)G__LD_P10_pointer;
      }
   }
   else if (G__PARANORMAL == reftype) {
      switch (type) {
         case 'B':
            *pinst = (long)(LD_P10_func_t)G__LD_P10<unsigned char>;
            break;
         case 'C':
            *pinst = (long)(LD_P10_func_t)G__LD_P10<char>;
            break;
         case 'R':
            *pinst = (long)(LD_P10_func_t)G__LD_P10<unsigned short>;
            break;
         case 'S':
            *pinst = (long)(LD_P10_func_t)G__LD_P10<short>;
            break;
         case 'H':
            *pinst = (long)(LD_P10_func_t)G__LD_P10<unsigned int>;
            break;
         case 'I':
            *pinst = (long)(LD_P10_func_t)G__LD_P10<int>;
            break;
         case 'K':
            *pinst = (long)(LD_P10_func_t)G__LD_P10<unsigned long>;
            break;
         case 'L':
            *pinst = (long)(LD_P10_func_t)G__LD_P10<long>;
            break;
         case 'U':
            *pinst = (long)(LD_P10_func_t)G__LD_P10_struct;
            break;
         case 'F':
            *pinst = (long)(LD_P10_func_t)G__LD_P10<float>;
            break;
         case 'D':
            *pinst = (long)(LD_P10_func_t)G__LD_P10<double>;
            break;
         case 'G':
            *pinst = (long)(LD_P10_func_t)G__LD_P10<bool>;
            break;
         case 'N':
            *pinst = (long)(LD_P10_func_t)G__LD_P10<G__int64>;
            break;
         case 'M':
            *pinst = (long)(LD_P10_func_t)G__LD_P10<G__uint64>;
            break;
         case 'Q':
            *pinst = (long)(LD_P10_func_t)G__LD_P10<long double>;
            break;
         default:
            done = 0;
            break;
      }
   }
   else {
      done = 0;
   }
   return done;
}

//______________________________________________________________________________
static int G__get_ST_p0_p2f(int type, long* pinst)
{
   typedef void (*ST_p0_func_t)(G__value *pbuf, int *psp, long offset, const Reflex::Member& var);
   int done = 1;
   if (isupper(type)) {
      if (type == 'Z') {
         done = 0;
      }
      else {
         *pinst = (long)(ST_p0_func_t)G__ST_p0_pointer;
      }
   }
   else {
      switch (type) {
         case 'b':
            *pinst = (long)(ST_p0_func_t)G__ST_p0<unsigned char>;
            break;
         case 'c':
            *pinst = (long)(ST_p0_func_t)G__ST_p0<char>;
            break;
         case 'r':
            *pinst = (long)(ST_p0_func_t)G__ST_p0<unsigned short>;
            break;
         case 's':
            *pinst = (long)(ST_p0_func_t)G__ST_p0<short>;
            break;
         case 'h':
            *pinst = (long)(ST_p0_func_t)G__ST_p0<unsigned int>;
            break;
         case 'i':
            *pinst = (long)(ST_p0_func_t)G__ST_p0<int>;
            break;
         case 'k':
            *pinst = (long)(ST_p0_func_t)G__ST_p0<unsigned long>;
            break;
         case 'l':
            *pinst = (long)(ST_p0_func_t)G__ST_p0<long>;
            break;
         case 'u':
            *pinst = (long)(ST_p0_func_t)G__ST_p0_struct;
            break;
         case 'f':
            *pinst = (long)(ST_p0_func_t)G__ST_p0<float>;
            break;
         case 'd':
            *pinst = (long)(ST_p0_func_t)G__ST_p0<double>;
            break;
         case 'g':
            *pinst = (long)(ST_p0_func_t)G__ST_p0<bool>;
            break;
         case 'n':
            *pinst = (long)(ST_p0_func_t)G__ST_p0<G__int64>;
            break;
         case 'm':
            *pinst = (long)(ST_p0_func_t)G__ST_p0<G__uint64>;
            break;
         case 'q':
            *pinst = (long)(ST_p0_func_t)G__ST_p0<long double>;
            break;
         default:
            done = 0;
            break;
      }
   }
   return done;
}

//______________________________________________________________________________
static int G__get_ST_p1_p2f(int type, long* pinst)
{
   typedef void (*ST_p1_func_t)(G__value *pbuf, int *psp, long offset, const Reflex::Member& var);
   int done = 1;
   if (isupper(type)) {
      if (type == 'Z') {
         done = 0;
      }
      else {
         *pinst = (long)(ST_p1_func_t)G__ST_p1_pointer;
      }
   }
   else {
      switch (type) {
         case 'b':
            *pinst = (long)(ST_p1_func_t)G__ST_p1<unsigned char>;
            break;
         case 'c':
            *pinst = (long)(ST_p1_func_t)G__ST_p1<char>;
            break;
         case 'r':
            *pinst = (long)(ST_p1_func_t)G__ST_p1<unsigned short>;
            break;
         case 's':
            *pinst = (long)(ST_p1_func_t)G__ST_p1<short>;
            break;
         case 'h':
            *pinst = (long)(ST_p1_func_t)G__ST_p1<unsigned int>;
            break;
         case 'i':
            *pinst = (long)(ST_p1_func_t)G__ST_p1<int>;
            break;
         case 'k':
            *pinst = (long)(ST_p1_func_t)G__ST_p1<unsigned long>;
            break;
         case 'l':
            *pinst = (long)(ST_p1_func_t)G__ST_p1<long>;
            break;
         case 'u':
            *pinst = (long)(ST_p1_func_t)G__ST_p1_struct;
            break;
         case 'f':
            *pinst = (long)(ST_p1_func_t)G__ST_p1<float>;
            break;
         case 'd':
            *pinst = (long)(ST_p1_func_t)G__ST_p1<double>;
            break;
         case 'g':
            *pinst = (long)(ST_p1_func_t)G__ST_p1<bool>;
            break; /* to be fixed */
         case 'n':
            *pinst = (long)(ST_p1_func_t)G__ST_p1<G__int64>;
            break;
         case 'm':
            *pinst = (long)(ST_p1_func_t)G__ST_p1<G__uint64>;
            break;
         case 'q':
            *pinst = (long)(ST_p1_func_t)G__ST_p1<long double>;
            break;
         default:
            done = 0;
            break;
      }
   }
   return done;
}

//______________________________________________________________________________
static int G__get_ST_pn_p2f(int type, long* pinst)
{
   typedef void (*ST_pn_func_t)(G__value *pbuf, int *psp, long offset, const Reflex::Member& var);
   int done = 1;
   if (isupper(type)) {
      if (type == 'Z') {
         done = 0;
      }
      else {
         *pinst = (long)(ST_pn_func_t)G__ST_pn_pointer;
      }
   }
   else {
      switch (type) {
         case 'b':
            *pinst = (long)(ST_pn_func_t)G__ST_pn<unsigned char>;
            break;
         case 'c':
            *pinst = (long)(ST_pn_func_t)G__ST_pn<char>;
            break;
         case 'r':
            *pinst = (long)(ST_pn_func_t)G__ST_pn<unsigned short>;
            break;
         case 's':
            *pinst = (long)(ST_pn_func_t)G__ST_pn<short>;
            break;
         case 'h':
            *pinst = (long)(ST_pn_func_t)G__ST_pn<unsigned int>;
            break;
         case 'i':
            *pinst = (long)(ST_pn_func_t)G__ST_pn<int>;
            break;
         case 'k':
            *pinst = (long)(ST_pn_func_t)G__ST_pn<unsigned long>;
            break;
         case 'l':
            *pinst = (long)(ST_pn_func_t)G__ST_pn<long>;
            break;
         case 'u':
            *pinst = (long)(ST_pn_func_t)G__ST_pn_struct;
            break;
         case 'f':
            *pinst = (long)(ST_pn_func_t)G__ST_pn<float>;
            break;
         case 'd':
            *pinst = (long)(ST_pn_func_t)G__ST_pn<double>;
            break;
         case 'g':
            *pinst = (long)(ST_pn_func_t)G__ST_pn<bool>;
            break; /* to be fixed */
         case 'n':
            *pinst = (long)(ST_pn_func_t)G__ST_pn<G__int64>;
            break;
         case 'm':
            *pinst = (long)(ST_pn_func_t)G__ST_pn<G__uint64>;
            break;
         case 'q':
            *pinst = (long)(ST_pn_func_t)G__ST_pn<long double>;
            break;
         default:
            done = 0;
            break;
      }
   }
   return done;
}

//______________________________________________________________________________
static int G__get_ST_P10_p2f(int type, long* pinst, int reftype)
{
   typedef void (*ST_P10_func_t)(G__value *pbuf, int *psp, long offset, const Reflex::Member& var);
   int done = 1;
   if (reftype == G__PARAP2P) {
      if (type == 'Z') {
         done = 0;
      }
      else {
         *pinst = (long)(ST_P10_func_t)G__ST_P10_pointer;
      }
   }
   else if (reftype == G__PARANORMAL) {
      switch (type) {
         case 'B':
            *pinst = (long)(ST_P10_func_t)G__ST_P10<unsigned char>;
            break;
         case 'C':
            *pinst = (long)(ST_P10_func_t)G__ST_P10<char>;
            break;
         case 'R':
            *pinst = (long)(ST_P10_func_t)G__ST_P10<unsigned short>;
            break;
         case 'S':
            *pinst = (long)(ST_P10_func_t)G__ST_P10<short>;
            break;
         case 'H':
            *pinst = (long)(ST_P10_func_t)G__ST_P10<unsigned int>;
            break;
         case 'I':
            *pinst = (long)(ST_P10_func_t)G__ST_P10<int>;
            break;
         case 'K':
            *pinst = (long)(ST_P10_func_t)G__ST_P10<unsigned long>;
            break;
         case 'L':
            *pinst = (long)(ST_P10_func_t)G__ST_P10<long>;
            break;
         case 'U':
            *pinst = (long)(ST_P10_func_t)G__ST_P10_struct;
            break;
         case 'F':
            *pinst = (long)(ST_P10_func_t)G__ST_P10<float>;
            break;
         case 'D':
            *pinst = (long)(ST_P10_func_t)G__ST_P10<double>;
            break;
         case 'G':
            *pinst = (long)(ST_P10_func_t)G__ST_P10<bool>;
            break;
         case 'N':
            *pinst = (long)(ST_P10_func_t)G__ST_P10<G__int64>;
            break;
         case 'M':
            *pinst = (long)(ST_P10_func_t)G__ST_P10<G__uint64>;
            break;
         case 'Q':
            *pinst = (long)(ST_P10_func_t)G__ST_P10<long double>;
            break;
         default:
            done = 0;
            break;
      }
   }
   else {
      done = 0;
   }
   return done;
}

//______________________________________________________________________________
static int G__get_LD_Rp0_p2f(int type, long* pinst)
{
   typedef void (*LD_Rp0_func_t)(G__value *pbuf, int *psp, long offset, const Reflex::Member& var);
   int done = 1;
   if (isupper(type)) {
      if (type == 'Z') {
         done = 0;
      }
      else {
         *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0_pointer;
      }
   }
   else {
      switch (type) {
         case 'b':
            *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0<unsigned char>;
            break;
         case 'c':
            *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0<char>;
            break;
         case 'r':
            *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0<unsigned short>;
            break;
         case 's':
            *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0<short>;
            break;
         case 'h':
            *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0<unsigned int>;
            break;
         case 'i':
            *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0<int>;
            break;
         case 'k':
            *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0<unsigned long>;
            break;
         case 'l':
            *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0<long>;
            break;
         case 'u':
            *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0_struct;
            break;
         case 'f':
            *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0<float>;
            break;
         case 'd':
            *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0<double>;
            break;
         case 'g':
            *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0<bool>;
            break; /* to be fixed */
         case 'n':
            *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0<G__int64>;
            break;
         case 'm':
            *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0<G__uint64>;
            break;
         case 'q':
            *pinst = (long)(LD_Rp0_func_t)G__LD_Rp0<long double>;
            break;
         default:
            done = 0;
            break;
      }
   }
   return done;
}

//______________________________________________________________________________
static int G__get_ST_Rp0_p2f(int type, long* pinst)
{
   typedef void (*ST_Rp0_func_t)(G__value *pbuf, int *psp, long offset, const Reflex::Member& var);
   int done = 1;
   if (isupper(type)) {
      if (type == 'Z') {
         done = 0;
      }
      else {
         *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0_pointer;
      }
   }
   else {
      switch (type) {
         case 'b':
            *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0<unsigned char>;
            break;
         case 'c':
            *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0<char>;
            break;
         case 'r':
            *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0<unsigned short>;
            break;
         case 's':
            *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0<short>;
            break;
         case 'h':
            *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0<unsigned int>;
            break;
         case 'i':
            *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0<int>;
            break;
         case 'k':
            *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0<unsigned long>;
            break;
         case 'l':
            *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0<long>;
            break;
         case 'u':
            *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0_struct;
            break;
         case 'f':
            *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0<float>;
            break;
         case 'd':
            *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0<double>;
            break;
         case 'g':
            *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0<bool>;
            break; /* to be fixed */
         case 'n':
            *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0<G__int64>;
            break;
         case 'm':
            *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0<G__uint64>;
            break;
         case 'q':
            *pinst = (long)(ST_Rp0_func_t)G__ST_Rp0<long double>;
            break;
         default:
            done = 0;
            break;
      }
   }
   return done;
}

//______________________________________________________________________________
static int G__get_LD_RP0_p2f(int type, long* pinst)
{
   typedef void (*LD_RP0_func_t)(G__value *pbuf, int *psp, long offset, const Reflex::Member& var);
   int done = 1;
   if (isupper(type)) {
      if (type == 'Z') {
         done = 0;
      }
      else {
         *pinst = (long)(LD_RP0_func_t)G__LD_RP0_pointer;
      }
   }
   else {
      switch (type) {
         case 'b':
            *pinst = (long)(LD_RP0_func_t)G__LD_RP0<unsigned char>;
            break;
         case 'c':
            *pinst = (long)(LD_RP0_func_t)G__LD_RP0<char>;
            break;
         case 'r':
            *pinst = (long)(LD_RP0_func_t)G__LD_RP0<unsigned short>;
            break;
         case 's':
            *pinst = (long)(LD_RP0_func_t)G__LD_RP0<short>;
            break;
         case 'h':
            *pinst = (long)(LD_RP0_func_t)G__LD_RP0<unsigned int>;
            break;
         case 'i':
            *pinst = (long)(LD_RP0_func_t)G__LD_RP0<int>;
            break;
         case 'k':
            *pinst = (long)(LD_RP0_func_t)G__LD_RP0<unsigned long>;
            break;
         case 'l':
            *pinst = (long)(LD_RP0_func_t)G__LD_RP0<long>;
            break;
         case 'u':
            *pinst = (long)(LD_RP0_func_t)G__LD_RP0_struct;
            break;
         case 'f':
            *pinst = (long)(LD_RP0_func_t)G__LD_RP0<float>;
            break;
         case 'd':
            *pinst = (long)(LD_RP0_func_t)G__LD_RP0<double>;
            break;
         case 'g':
            *pinst = (long)(LD_RP0_func_t)G__LD_RP0<bool>;
            break; /* to be fixed */
         case 'n':
            *pinst = (long)(LD_RP0_func_t)G__LD_RP0<G__int64>;
            break;
         case 'm':
            *pinst = (long)(LD_RP0_func_t)G__LD_RP0<G__uint64>;
            break;
         case 'q':
            *pinst = (long)(LD_RP0_func_t)G__LD_RP0<long double>;
            break;
         default:
            done = 0;
            break;
      }
   }
   return done;
}

//______________________________________________________________________________
static void G__LD_Rp0_optimize(const Reflex::Member& var, int pc, long inst)
{
   long originst = G__asm_inst[pc];
   int pointlevel = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      switch (inst) {
         case G__LDST_VAR_P: /* illegal case */
            G__fprinterr(G__serr, "  G__LD_VAR REF optimized 6 G__LDST_VAR_P\n");
            break;
         case G__LDST_MSTR_P:
            G__fprinterr(G__serr, "  G__LD_MSTR REF optimized 6 G__LDST_MSTR_P\n");
            break;
         case G__LDST_LVAR_P:
            G__fprinterr(G__serr, "  G__LD_LVAR REF optimized 6 G__LDST_LVAR_P\n");
            break;
      }
   }
#endif // G__ASM_DBG
   G__asm_inst[pc] = inst;
   G__asm_inst[pc+3] = 0;
   if (!G__get_LD_Rp0_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+2])) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Error: LD_VAR,LD_MSTR REF optimize (6) error %s\n", var.Name().c_str());
      }
#endif // G__ASM_DBG
      G__asm_inst[pc] = originst;
      G__asm_inst[pc+3] = pointlevel;
   }
}

//______________________________________________________________________________
static void G__ST_Rp0_optimize(const Reflex::Member& var, int pc, long inst)
{
   long originst = G__asm_inst[pc];
   int pointlevel = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      switch (inst) {
         case G__LDST_VAR_P: /* illegal case */
            G__fprinterr(G__serr, "  G__ST_VAR REF optimized 6 G__LDST_VAR_P\n");
            break;
         case G__LDST_MSTR_P:
            G__fprinterr(G__serr, "  G__ST_MSTR REF optimized 6 G__LDST_MSTR_P\n");
            break;
         case G__LDST_LVAR_P:
            G__fprinterr(G__serr, "  G__ST_LVAR REF optimized 6 G__LDST_LVAR_P\n");
            break;
      }
   }
#endif // G__ASM_DBG
   G__asm_inst[pc] = inst;
   G__asm_inst[pc+3] = 0;
   if (!G__get_ST_Rp0_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+2])) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Error: LD_VAR,LD_MSTR REF optimize (6) error %s\n", var.Name().c_str());
      }
#endif // G__ASM_DBG
      G__asm_inst[pc] = originst;
      G__asm_inst[pc+3] = pointlevel;
   }
}

//______________________________________________________________________________
static void G__LD_RP0_optimize(const Reflex::Member& var, int pc, long inst)
{
   long originst = G__asm_inst[pc];
   int pointlevel = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      switch (inst) {
         case G__LDST_VAR_P: /* illegal case */
            G__fprinterr(G__serr, "  G__LD_VAR REF optimized 7 G__LDST_VAR_P\n");
            break;
         case G__LDST_MSTR_P:
            G__fprinterr(G__serr, "  G__LD_MSTR REF optimized 7 G__LDST_MSTR_P\n");
            break;
         case G__LDST_LVAR_P:
            G__fprinterr(G__serr, "  G__LD_LVAR REF optimized 7 G__LDST_LVAR_P\n");
            break;
      }
   }
#endif // G__ASM_DBG
   G__asm_inst[pc] = inst;
   G__asm_inst[pc+3] = 0;
   if (!G__get_LD_RP0_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+2])) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Error: LD_VAR,LD_MSTR REF optimize (7) error %s\n", var.Name().c_str());
      }
#endif // G__ASM_DBG
      G__asm_inst[pc] = originst;
      G__asm_inst[pc+3] = pointlevel;
   }
}

//______________________________________________________________________________
static void G__LD_p0_optimize(const Reflex::Member& var, int pc, long inst)
{
   long originst = G__asm_inst[pc];
   int pointlevel = G__asm_inst[pc+3];
   if (G__get_properties(var)->bitfield_width) {
      return;
   }
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      switch (inst) {
         case G__LDST_VAR_P:
            G__fprinterr(G__serr, "  G__LD_VAR optimized 6 G__LDST_VAR_P\n");
            break;
         case G__LDST_MSTR_P:
            G__fprinterr(G__serr, "  G__LD_MSTR optimized 6 G__LDST_MSTR_P\n");
            break;
         case G__LDST_LVAR_P:
            G__fprinterr(G__serr, "  G__LD_LVAR optimized 6 G__LDST_LVAR_P\n");
            break;
      }
   }
#endif // G__ASM_DBG
   G__asm_inst[pc] = inst;
   G__asm_inst[pc+3] = 0;
   if (!G__get_LD_p0_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+2])) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Error: LD_VAR,LD_MSTR optimize (6) error %s\n", var.Name().c_str());
      }
#endif // G__ASM_DBG
      G__asm_inst[pc] = originst;
      G__asm_inst[pc+3] = pointlevel;
   }
}

//______________________________________________________________________________
static void G__LD_p1_optimize(const Reflex::Member& var, int pc, long inst)
{
   long originst = G__asm_inst[pc];
   int pointlevel = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      switch (inst) {
         case G__LDST_VAR_P:
            G__fprinterr(G__serr, "  G__LD_VAR optimized 7 G__LDST_VAR_P\n");
            break;
         case G__LDST_MSTR_P:
            G__fprinterr(G__serr, "  G__LD_MSTR optimized 7 G__LDST_MSTR_P\n");
            break;
         case G__LDST_LVAR_P:
            G__fprinterr(G__serr, "  G__LD_LVAR optimized 7 G__LDST_LVAR_P\n");
            break;
      }
   }
#endif
   G__asm_inst[pc] = inst;
   G__asm_inst[pc+3] = 0;
   if (!G__get_LD_p1_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+2])) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Error: LD_VAR optimize (8) error %s\n", var.Name().c_str());
      }
#endif // G__ASM_DBG
      G__asm_inst[pc] = originst;
      G__asm_inst[pc+3] = pointlevel;
   }
}

//______________________________________________________________________________
static void G__LD_pn_optimize(const Reflex::Member& var, int pc, long inst)
{
   long originst = G__asm_inst[pc];
   int pointlevel = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      switch (inst) {
         case G__LDST_VAR_P:
            G__fprinterr(G__serr, "  G__LD_VAR optimized 8 G__LDST_VAR_P\n");
            break;
         case G__LDST_MSTR_P:
            G__fprinterr(G__serr, "  G__LD_MSTR optimized 8 G__LDST_MSTR_P\n");
            break;
         case G__LDST_LVAR_P:
            G__fprinterr(G__serr, "  G__LD_LVAR optimized 8 G__LDST_LVAR_P\n");
            break;
      }
   }
#endif // G__ASM_DBG
   G__asm_inst[pc] = inst;
   G__asm_inst[pc+3] = 0;
   if (!G__get_LD_pn_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+2])) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Error: LD_VAR optimize (8) error %s\n", var.Name().c_str());
      }
#endif // G__ASM_DBG
      G__asm_inst[pc] = originst;
      G__asm_inst[pc+3] = pointlevel;
   }
}

//______________________________________________________________________________
static void G__LD_P10_optimize(const Reflex::Member& var, int pc, long inst)
{
   long originst = G__asm_inst[pc];
   int pointlevel = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      switch (inst) {
         case G__LDST_VAR_P:
            G__fprinterr(G__serr, "  G__LD_VAR optimized 9 G__LDST_VAR_P\n");
            break;
         case G__LDST_MSTR_P:
            G__fprinterr(G__serr, "  G__LD_MSTR optimized 9 G__LDST_MSTR_P\n");
            break;
         case G__LDST_LVAR_P:
            G__fprinterr(G__serr, "  G__LD_LVAR optimized 9 G__LDST_LVAR_P\n");
            break;
      }
   }
#endif // G__ASM_DBG
   G__asm_inst[pc] = inst;
   G__asm_inst[pc+3] = 0;
   if (!G__get_LD_P10_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+2], G__get_reftype(var.TypeOf()))) {
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Error: LD_VAR optimize (9) error %s\n", var.Name().c_str());
      }
#endif // G__ASM_DBG
      G__asm_inst[pc] = originst;
      G__asm_inst[pc+3] = pointlevel;
   }
}

//______________________________________________________________________________
static void G__ST_p0_optimize(const Reflex::Member& var, int pc, long inst)
{
   long originst = G__asm_inst[pc];
   int pointlevel = G__asm_inst[pc+3];
   if (G__get_properties(var)->bitfield_width) {
      return;
   }
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      switch (inst) {
         case G__LDST_VAR_P:
            G__fprinterr(G__serr, "  G__ST_VAR optimized 8 G__LDST_VAR_P\n");
            break;
         case G__LDST_MSTR_P:
            G__fprinterr(G__serr, "  G__ST_MSTR optimized 8 G__LDST_MSTR_P\n");
            break;
         case G__LDST_LVAR_P:
            G__fprinterr(G__serr, "  G__ST_VAR optimized 8 G__LDST_LVAR_P\n");
            break;
      }
   }
#endif // G__ASM_DBG
   G__asm_inst[pc+0] = inst;
   G__asm_inst[pc+3] = 1;
   if (!G__get_ST_p0_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+2])) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Warning: ST_VAR optimize (8) error %s\n", var.Name().c_str());
      }
#endif // G__ASM_DBG
      G__asm_inst[pc] = originst;
      G__asm_inst[pc+3] = pointlevel;
   }
}

//______________________________________________________________________________
static void G__ST_p1_optimize(const Reflex::Member& var, int pc, long inst)
{
   long originst = G__asm_inst[pc];
   int pointlevel = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      switch (inst) {
         case G__LDST_VAR_P:
            G__fprinterr(G__serr, "  G__ST_VAR optimized 9 G__LDST_VAR_P\n");
            break;
         case G__LDST_MSTR_P:
            G__fprinterr(G__serr, "  G__ST_MSTR optimized 9 G__LDST_MSTR_P\n");
            break;
         case G__LDST_LVAR_P:
            G__fprinterr(G__serr, "  G__ST_VAR optimized 9 G__LDST_LVAR_P\n");
            break;
      }
   }
#endif // G__ASM_DBG
   G__asm_inst[pc+0] = inst;
   G__asm_inst[pc+3] = 1;
   if (!G__get_ST_p1_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Warning: ST_VAR optimize error %s\n", var.Name().c_str());
      }
#endif // G__ASM_DBG
      G__asm_inst[pc] = originst;
      G__asm_inst[pc+3] = pointlevel;
   }
}

//______________________________________________________________________________
static void G__ST_pn_optimize(const Reflex::Member& var, int pc, long inst)
{
   long originst = G__asm_inst[pc];
   int pointlevel = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      switch (inst) {
         case G__LDST_VAR_P:
            G__fprinterr(G__serr, "  G__ST_VAR optimized 10 G__LDST_VAR_P\n");
            break;
         case G__LDST_MSTR_P:
            G__fprinterr(G__serr, "  G__ST_MSTR optimized 10 G__LDST_MSTR_P\n");
            break;
         case G__LDST_LVAR_P:
            G__fprinterr(G__serr, "  G__ST_VAR optimized 10 G__LDST_LVAR_P\n");
            break;
      }
   }
#endif // G__ASM_DBG
   G__asm_inst[pc+0] = inst;
   G__asm_inst[pc+3] = 1;
   if (!G__get_ST_pn_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Warning: ST_VAR optimize error %s\n", var.Name().c_str());
      }
#endif // G__ASM_DBG
      G__asm_inst[pc] = originst;
      G__asm_inst[pc+3] = pointlevel;
   }
}

//______________________________________________________________________________
static void G__ST_P10_optimize(const Reflex::Member& var, int pc, long inst)
{
   long originst = G__asm_inst[pc];
   int pointlevel = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      switch (inst) {
         case G__LDST_VAR_P:
            G__fprinterr(G__serr, "  G__ST_VAR optimized 7 G__LDST_VAR_P\n");
            break;
         case G__LDST_MSTR_P:
            G__fprinterr(G__serr, "  G__ST_MSTR optimized 7 G__LDST_MSTR_P\n");
            break;
         case G__LDST_LVAR_P:
            G__fprinterr(G__serr, "  G__ST_LVAR optimized 7 G__LDST_LVAR_P\n");
            break;
      }
   }
#endif // G__ASM_DBG
   G__asm_inst[pc] = inst;
   G__asm_inst[pc+3] = 0;
   if (!G__get_ST_P10_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+2], G__get_reftype(var.TypeOf()))) {
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Error: ST_VAR optimize (7) error %s\n", var.Name().c_str());
      }
#endif // G__ASM_DBG
      G__asm_inst[pc] = originst;
      G__asm_inst[pc+3] = pointlevel;
   }
}

//______________________________________________________________________________
//
//  Array index optimization constant.
//

#define G__MAXINDEXCONST 11
static long G__indexconst[G__MAXINDEXCONST] =
   {
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
   };

//______________________________________________________________________________
static int G__LD_VAR_int_optimize(int* ppc, long* pi)
{
   // G__LDST_VAR_INDEX optimization
   int ig15;
   int done = 0;
   int pc = *ppc;
   ::Reflex::Member var;
   if (G__asm_inst[pc+9]) {
      var = G__Dict::GetDict().GetDataMember((size_t) G__asm_inst[pc+9]);
   }
   if (
      (G__asm_inst[pc+7] == 1) &&
      (G__asm_inst[pc+8] == 'p') &&
      G__asm_inst[pc+9] &&
      (G__get_paran(var) == 1) &&
      (
         islower(G__get_type(var.TypeOf())) ||
         (G__get_reftype(var.TypeOf()) == G__PARANORMAL)
      )
   ) {
      ig15 = G__asm_inst[pc+6];
      /********************************************************************
       * 0 G__LD_VAR,LVAR                    G__LDST_VAR_INDEX
       * 1 index (not used)                  *arrayindex
       * 2 paran == 0                        (*p2f)(buf,psp,0,var2,index2)
       * 3 pointer_level == p                index2 (not used)
       * 4 var id                            pc increment
       * 5 G__LD_VAR,LVAR                    local_global
       * 6 index2 (not used)                 var2 id 
       * 7 paran == 1
       * 8 point_level == p
       * 9 var2 id
       ********************************************************************/
      if ((G__asm_inst[pc+5] == G__LD_VAR) || (G__asm_inst[pc+5] == G__LD_LVAR)) {
         int flag = 0;
         if (G__asm_inst[pc] == G__LD_LVAR) {
            flag = 1;
         }
         if (G__asm_inst[pc+5] == G__LD_LVAR) {
            flag |= 2;
         }
         if (!G__get_LD_p1_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+2])) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "Error: LD_VAR,LD_VAR[1] optimize error %s  %s:%d\n", var.Name().c_str(), __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            // --
         }
         else {
            done = 1;
            G__asm_inst[pc+5] = flag;
            G__asm_inst[pc] = G__LDST_VAR_INDEX;
            G__asm_inst[pc+1] = (long) pi;
            G__asm_inst[pc+3] = G__asm_inst[pc+6];
            G__asm_inst[pc+4] = 10;
            G__asm_inst[pc+6] = G__asm_inst[pc+9];
            *ppc = pc + 5; // other 2 is incremented one level up
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "LDST_VAR_INDEX (1) optimized  %s:%d\n", __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            // --
         }
      }

      /********************************************************************
       * 0 G__LD_VAR                         G__LDST_VAR_INDEX
       * 1 index (not used)                  *arrayindex
       * 2 paran == 0                        (*p2f)(buf,psp,0,var2,index2)
       * 3 pointer_level == p                index2 (not used)
       * 4 var id                            pc increment
       * 5 G__ST_VAR                         local_global
       * 6 index2 (not used)                 var2 id    
       * 7 paran == 1
       * 8 point_level == p
       * 9 var2 id    
       ********************************************************************/
      else if (G__ST_VAR == G__asm_inst[pc+5] || G__ST_LVAR == G__asm_inst[pc+5]) {
         int flag;
         if (G__LD_LVAR == G__asm_inst[pc]) flag = 1;
         else                            flag = 0;
         if (G__ST_LVAR == G__asm_inst[pc+5]) flag |= 2;
         ig15 = G__asm_inst[pc+6];
         if (0 == G__get_ST_p1_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
            if (G__asm_dbg)
               G__fprinterr(G__serr, "Error: LD_VAR,ST_VAR[1] optimize error %s\n"
                            , var.Name().c_str());
#endif
         }
         else {
            done = 1;
            G__asm_inst[pc+5] = flag;
            G__asm_inst[pc] = G__LDST_VAR_INDEX;
            G__asm_inst[pc+1] = (long)pi;
            G__asm_inst[pc+3] = G__asm_inst[pc+6];
            G__asm_inst[pc+4] = 10;
            G__asm_inst[pc+6] = G__asm_inst[pc+9];
            *ppc = pc + 5; /* other 2 is incremented one level up */
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "LDST_VAR_INDEX (2) optimized\n");
#endif
         }
      }
   }

   /********************************************************************
    * G__LDST_VAR_INDEX_OPR optimization
    ********************************************************************/
   else if (G__LD == G__asm_inst[pc+5] &&
            'i' == G__get_type(G__value_typenum(G__asm_stack[G__asm_inst[pc+6]])) &&
            G__OP2 == G__asm_inst[pc+7] &&
            ('+' == G__asm_inst[pc+8] || '-' == G__asm_inst[pc+8]) &&
            1 == G__asm_inst[pc+11] &&
            'p' == G__asm_inst[pc+12] &&
            (var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+13])) &&
            1 == G__get_paran(var) &&
            (!var.TypeOf().FinalType().IsPointer() ||
             G__PARANORMAL == G__get_reftype(var.TypeOf()))) {
      ig15 = G__asm_inst[pc+10];
      /********************************************************************
       * 0 G__LD_VAR,LVAR                    G__LDST_VAR_INDEX_OPR
       * 1 index (not used)                  *int1
       * 2 paran == 0                        *int2
       * 3 pointer_level == p                opr +,-
       * 4 var id                            (*p2f)(buf,p2p,0,var2,index2)
       * 5 G__LD                             index3 (not used)
       * 6 stack address 'i'                 pc increment
       * 7 OP2                               local_global
       * 8 +,-                               var3 id    
       * 9 G__LD_VAR,LVAR
       *10 index3 (not used)
       *11 paran == 1
       *12 point_level == p
       *13 var3 id    
       ********************************************************************/
      if (G__LD_VAR == G__asm_inst[pc+9] || G__LD_LVAR == G__asm_inst[pc+9]) {
         int flag;
         long *pi2 = &(G__asm_stack[G__asm_inst[pc+6]].obj.i);
         long *pix;
         if (G__ASM_FUNC_COMPILE == G__asm_wholefunction) {
            if (*pi2 >= G__MAXINDEXCONST || *pi2 < 0) return(done);
            else pix = &G__indexconst[*pi2];
         }
         else {
            pix = pi2;
            if (sizeof(long) > sizeof(int)) *pix = (int)(*pi2);
         }
         if (G__LD_LVAR == G__asm_inst[pc]) flag = 1;
         else                            flag = 0;
         if (G__LD_LVAR == G__asm_inst[pc+9]) flag |= 4;
         if (0 == G__get_LD_p1_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+4])) {
#ifdef G__ASM_DBG
            if (G__asm_dbg)
               G__fprinterr(G__serr,
                            "Error: LD_VAR,LD,OP2,LD_VAR[1] optimize error %s\n"
                            , var.Name().c_str());
#endif
         }
         else {
            done = 1;
            G__asm_inst[pc+7] = flag;
            G__asm_inst[pc] = G__LDST_VAR_INDEX_OPR;
            G__asm_inst[pc+1] = (long)pi;
            G__asm_inst[pc+2] = (long)pix;
            G__asm_inst[pc+3] = G__asm_inst[pc+8];
            G__asm_inst[pc+5] = G__asm_inst[pc+10];
            G__asm_inst[pc+6] = 14;
            G__asm_inst[pc+8] = G__asm_inst[pc+13];
            *ppc = pc + 9; /* other 2 is incremented one level up */
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "LDST_VAR_INDEX_OPR (3) optimized\n");
#endif
         }
      }
      /********************************************************************
       * 0 G__LD_VAR,LVAR                    G__LDST_VAR_INDEX_OPR
       * 1 index (not used)                  *int1
       * 2 paran == 0                        *int2
       * 3 pointer_level == p                opr +,-
       * 4 var id                            (*p2f)(buf,p2p,0,var2,index2)
       * 5 G__LD                             index3 (not used)
       * 6 stack address 'i'                 pc increment
       * 7 OP2                               local_global
       * 8 +,-                               var id    
       * 9 G__ST_VAR,LVAR
       *10 index3 (not used)
       *11 paran == 1
       *12 point_level == p
       *13 var3 id    
       ********************************************************************/
      else if (G__ST_VAR == G__asm_inst[pc+9] || G__ST_LVAR == G__asm_inst[pc+9]) {
         int flag;
         long *pi2 = &(G__asm_stack[G__asm_inst[pc+6]].obj.i);
         long *pix;
         if (G__ASM_FUNC_COMPILE == G__asm_wholefunction) {
            if (*pi2 >= G__MAXINDEXCONST || *pi2 < 0) return(done);
            else pix = &G__indexconst[*pi2];
         }
         else {
            pix = pi2;
            if (sizeof(long) > sizeof(int)) *pix = (int)(*pi2);
         }
         if (G__LD_LVAR == G__asm_inst[pc]) flag = 1;
         else                            flag = 0;
         if (G__ST_LVAR == G__asm_inst[pc+9]) flag |= 4;
         if (0 == G__get_ST_p1_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+4])) {
#ifdef G__ASM_DBG
            if (G__asm_dbg)
               G__fprinterr(G__serr,
                            "Error: LD_VAR,LD,OP2,ST_VAR[1] optimize error %s\n"
                            , var.Name().c_str());
#endif
         }
         else {
            done = 1;
            G__asm_inst[pc+7] = flag;
            G__asm_inst[pc] = G__LDST_VAR_INDEX_OPR;
            G__asm_inst[pc+1] = (long)pi;
            G__asm_inst[pc+2] = (long)pix;
            G__asm_inst[pc+3] = G__asm_inst[pc+8];
            G__asm_inst[pc+5] = G__asm_inst[pc+10];
            G__asm_inst[pc+6] = 14;
            G__asm_inst[pc+8] = G__asm_inst[pc+13];
            *ppc = pc + 9; /* other 2 is incremented one level up */
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "LDST_VAR_INDEX_OPR (4) optimized\n");
#endif
         }
      }
   }

   /********************************************************************
    * 0 G__LD_VAR,LVAR                    G__LDST_VAR_INDEX_OPR
    * 1 index (not used)                  *int1
    * 2 paran == 0                        *int2
    * 3 pointer_level == p                opr +,-
    * 4 var id                            (*p2f)(buf,p2p,0,var2,index2)
    * 5 G__LD_VAR,LvAR                    index3 (not used)
    * 6 index2 (not used)                 pc increment
    * 7 paran == 0                        not use
    * 8 point_level == p                  var3 id  
    * 9 var2 id  
    *10 OP2
    *11 +,-
    *12 G__LD_VAR,LvAR
    *13 index3 (not used)
    *14 paran == 1
    *15 point_level == p
    *16 var3 id  
    ********************************************************************/

   /********************************************************************
    * 0 G__LD_VAR,LVAR                    G__LDST_VAR_INDEX_OPR
    * 1 index (not used)                  *int1
    * 2 paran == 0                        *int2
    * 3 pointer_level == p                opr +,-
    * 4 var id                            (*p2f)(buf,p2p,0,var2,index2)
    * 5 G__LD_VAR,LvAR                    index3 (not used)
    * 6 index2 (not used)                 pc increment
    * 7 paran == 0                        not used
    * 8 point_level == p                  var id  
    * 9 var2 id  
    *10 OP2
    *11 +,-
    *12 G__ST_VAR,LVAR
    *13 index3 (not used)
    *14 paran == 1
    *15 point_level == p
    *16 var3 id  
    ********************************************************************/

   return done;
}

//______________________________________________________________________________
static int G__LD_int_optimize(int* ppc, long* pi)
{
   Reflex::Member var;
   int ig15;
   int done = 0;
   int pc;
   pc = *ppc;

   /********************************************************************
    * 0 G__LD                             G__LD_VAR_INDEX
    * 1 stack address 'i'                 *arrayindex
    * 2 G__LD_VAR,LVAR                    (*p2f)(buf,psp,0,var,index)
    * 3 index (not used)                  index (not used)
    * 4 paran == 1                        pc increment
    * 5 point_level == p                  local_global
    * 6 var id                            var id  
    ********************************************************************/
   if ((G__LD_VAR == G__asm_inst[pc+2] || G__LD_LVAR == G__asm_inst[pc+2]) &&
         1 == G__asm_inst[pc+4] &&
         'p' == G__asm_inst[pc+5] &&
         (var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+6])) &&
         1 == G__get_paran(var) &&
         (!var.TypeOf().FinalType().IsPointer() ||
          G__PARANORMAL == G__get_reftype(var.TypeOf()))
         && (pc < 4 || G__JMP != G__asm_inst[pc-2] || G__asm_inst[pc-1] != pc + 2)
      ) {
      int flag;
      if (G__ASM_FUNC_COMPILE == G__asm_wholefunction) {
         if (*pi >= G__MAXINDEXCONST || *pi < 0) return(done);
         else pi = &G__indexconst[*pi];
      }
      if (G__LD_LVAR == G__asm_inst[pc+2]) flag = 2;
      else                              flag = 0;
      done = 1;
      ig15 = G__asm_inst[pc+3];
      if (0 == G__get_LD_p1_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
         if (G__asm_dbg)
            G__fprinterr(G__serr, "Error: LD,LD_VAR[1] optimize error %s\n"
                         , var.Name().c_str());
#endif
      }
      else {
         done = 1;
         G__asm_inst[pc+5] = flag;
         G__asm_inst[pc] = G__LDST_VAR_INDEX;
         G__asm_inst[pc+1] = (long)pi;
         if (sizeof(long) > sizeof(int)) { /* long to int conversion */
            *(int*)G__asm_inst[pc+1] = (int)(*(long*)pi);
         }
         G__asm_inst[pc+4] = 7;
         *ppc = pc + 5; /* other 2 is incremented one level up */
#ifdef G__ASM_DBG
         if (G__asm_dbg) G__fprinterr(G__serr, "LDST_VAR_INDEX (5) optimized\n");
#endif
      }
   }

   /********************************************************************
    * 0 G__LD                             G__LDST_VAR_INDEX
    * 1 stack address 'i'                 *arrayindex
    * 2 G__ST_VAR,LvAR                    (*p2f)(buf,psp,0,var,index)
    * 3 index (not used)                  index (not used)
    * 4 paran == 1                        pc increment
    * 5 point_level == p                  flag &1:param_lcocal,&2:array_local
    * 6 var id                            var id  
    ********************************************************************/
   else if (
      ((G__asm_inst[pc+2] == G__ST_VAR) || (G__asm_inst[pc+2] == G__ST_LVAR)) &&
      (G__asm_inst[pc+4] == 1) &&
      (G__asm_inst[pc+5] == 'p') &&
      (var = G__Dict::GetDict().GetDataMember((size_t) G__asm_inst[pc+6])) &&
      (G__get_paran(var) == 1) &&
      (
         islower(G__get_type(var.TypeOf())) ||
         (G__get_reftype(var.TypeOf()) == G__PARANORMAL)
      ) &&
      ((pc < 4) || (G__asm_inst[pc-2] != G__JMP) || (G__asm_inst[pc-1] != (pc + 2)))
   ) {
      int flag;
      if (G__ASM_FUNC_COMPILE == G__asm_wholefunction) {
         if (*pi >= G__MAXINDEXCONST || *pi < 0) return(done);
         else pi = &G__indexconst[*pi];
      }
      if (G__ST_LVAR == G__asm_inst[pc+2]) flag = 2;
      else                              flag = 0;
      ig15 = G__asm_inst[pc+3];
      if (0 == G__get_ST_p1_p2f(G__get_type(var.TypeOf()), &G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
         if (G__asm_dbg)
            G__fprinterr(G__serr, "Error: LD,ST_VAR[1] optimize error %s\n"
                         , var.Name().c_str());
#endif
      }
      else {
         done = 1;
         G__asm_inst[pc+5] = flag;
         G__asm_inst[pc] = G__LDST_VAR_INDEX;
         G__asm_inst[pc+1] = (long)pi;
         if (sizeof(long) > sizeof(int)) { /* long to int conversion */
            *(int*)G__asm_inst[pc+1] = (int)(*(long*)pi);
         }
         G__asm_inst[pc+4] = 7;
         *ppc = pc + 5; /* other 2 is incremented one level up */
#ifdef G__ASM_DBG
         if (G__asm_dbg) G__fprinterr(G__serr, "LDST_VAR_INDEX (6) optimized\n");
#endif
      }
   }

   return(done);
}


//______________________________________________________________________________
static int G__CMP2_optimize(int pc)
{
   typedef void(*opt_func_t)(G__value*, G__value*);
   G__asm_inst[pc] = G__OP2_OPTIMIZED;
   switch (G__asm_inst[pc+1]) {
      case 'E': /* == */
         G__asm_inst[pc+1] = (long)(opt_func_t) G__CMP2_equal;
         break;
      case 'N': /* != */
         G__asm_inst[pc+1] = (long)(opt_func_t) G__CMP2_notequal;
         break;
      case 'G': /* >= */
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply<OperGreaterOrEqual>;
         break;
      case 'l': /* <= */
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply<OperLessOrEqual>;
         break;
      case '<': /* <  */
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply<OperLess>;
         break;
      case '>': /* >  */
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply<OperGreater>;
         break;
   }
   return(0);
}

//______________________________________________________________________________
static int G__OP2_optimize(int pc)
{
   typedef void(*opt_func_t)(G__value*, G__value*);
   int done = 1;
   switch (G__asm_inst[pc+1]) {
      case '+':
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_plus;
         break;
      case '-':
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_minus;
         break;
      case '*':
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_multiply;
         break;
      case '/':
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_divide;
         break;
      case '%':
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_modulus;
         break;
         /*
           case '&':
             G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_bitand;
             break;
           case '|':
             G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_bitand;
             break;
           case '^':
             G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_exor;
             break;
           case '~':
             G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_bininv;
             break;
         */
      case 'A':
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_logicaland;
         break;
      case 'O':
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_logicalor;
         break;
      case '>':
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply<OperGreater>;
         break;
      case '<':
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply<OperLess>;
         break;
         /*
           case 'R':
             G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_rightshift;
             break;
           case 'L':
             G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_leftshift;
             break;
           case '@':
             G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_power;
             break;
         */
      case 'E':
         G__asm_inst[pc+1] = (long)(opt_func_t) G__CMP2_equal;
         break;
      case 'N':
         G__asm_inst[pc+1] = (long)(opt_func_t) G__CMP2_notequal;
         break;
      case 'G':
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply<OperGreaterOrEqual>;
         break;
      case 'l':
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply<OperLessOrEqual>;
         break;
      case G__OPR_ADDASSIGN:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply<OperAddAssign>;
         break;
      case G__OPR_SUBASSIGN:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply<OperSubAssign>;
         break;
      case G__OPR_MODASSIGN:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_int<OperModAssign>;
         break;
      case G__OPR_MULASSIGN:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply<OperMulAssign>;
         break;
      case G__OPR_DIVASSIGN:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply<OperDivAssign>;
         break;
      case G__OPR_ADD_UU:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_plus_T<unsigned long>;
         break;
      case G__OPR_SUB_UU:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_minus_T<unsigned long>;
         break;
      case G__OPR_MUL_UU:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_multiply_T<unsigned long>;
         break;
      case G__OPR_DIV_UU:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_divide_T<unsigned long>;
         break;
      case G__OPR_ADDASSIGN_UU:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<unsigned long, OperAddAssign>;
         break;
      case G__OPR_SUBASSIGN_UU:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<unsigned long, OperSubAssign>;
         break;
      case G__OPR_MULASSIGN_UU:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<unsigned long, OperMulAssign>;
         break;
      case G__OPR_DIVASSIGN_UU:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<unsigned long, OperDivAssign>;
         break;
      case G__OPR_ADD_II:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_plus_T<long>;
         break;
      case G__OPR_SUB_II:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_minus_T<long>;
         break;
      case G__OPR_MUL_II:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_multiply_T<long>;
         break;
      case G__OPR_DIV_II:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_divide_T<long>;
         break;
      case G__OPR_ADDASSIGN_II:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<long, OperAddAssign>;
         break;
      case G__OPR_SUBASSIGN_II:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<long, OperSubAssign>;
         break;
      case G__OPR_MULASSIGN_II:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<long, OperMulAssign>;
         break;
      case G__OPR_DIVASSIGN_II:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<long, OperDivAssign>;
         break;
      case G__OPR_ADD_DD:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_plus_T<double>;
         break;
      case G__OPR_SUB_DD:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_minus_T<double>;
         break;
      case G__OPR_MUL_DD:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_multiply_T<double>;
         break;
      case G__OPR_DIV_DD:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_divide_T<double>;
         break;
      case G__OPR_ADDASSIGN_DD:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<double,OperAddAssign>;
         break;
      case G__OPR_SUBASSIGN_DD:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<double,OperSubAssign>;
         break;
      case G__OPR_MULASSIGN_DD:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<double,OperMulAssign>;
         break;
      case G__OPR_DIVASSIGN_DD:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<double,OperDivAssign>;
         break;
      case G__OPR_ADDASSIGN_FD:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<float,OperAddAssign>;
         break;
      case G__OPR_SUBASSIGN_FD:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<float,OperSubAssign>;
         break;
      case G__OPR_MULASSIGN_FD:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<float,OperMulAssign>;
         break;
      case G__OPR_DIVASSIGN_FD:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_apply_common<float,OperDivAssign>;
         break;
      case G__OPR_ADDVOIDPTR:
         G__asm_inst[pc+1] = (long)(opt_func_t) G__OP2_addvoidptr;
         break;
      default:
         done = 0;
         break;
   }
   if (done) G__asm_inst[pc] = G__OP2_OPTIMIZED;
   return(0);
}

//______________________________________________________________________________
int Cint::Internal::G__asm_optimize3(int* start)
{
   int illegal = 0;
   Reflex::Member var;
   int ig15;
   int paran;
   int var_type;
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "Optimize 3 start\n");
   }
#endif // G__ASM_DBG
   int pc = *start; // Set program counter to start of code.
   while (pc < G__MAXINST) {
      switch (G__INST(G__asm_inst[pc])) {
         case G__LDST_VAR_P:
            /***************************************
            * inst
            * 0 G__LDST_VAR_P
            * 1 index (not used)
            * 2 void (*f)(pbuf,psp,offset,p,ctype,
            * 3 (not used)
            * 4 var id
            * stack
            * sp          <-
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
               G__fprinterr(G__serr, "%3lx: LDST_VAR_P index=%ld %s\n"
                            , pc, G__asm_inst[pc+1]
                            , var.Name().c_str());
            }
#endif
            pc += 5;
            break;

         case G__LDST_MSTR_P:
            /***************************************
            * inst
            * 0 G__LDST_MSTR_P
            * 1 index (not used)
            * 2 void (*f)(pbuf,offset,psp,p,ctype,
            * 3 (not use)
            * 4 var id      
            * stack
            * sp          <-
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
               G__fprinterr(G__serr, "%3lx: LDST_MSTR_P index=%d %s\n"
                            , pc, G__asm_inst[pc+1]
                            , var.Name().c_str());
            }
#endif
            pc += 5;
            break;

         case G__LDST_VAR_INDEX:
            /***************************************
            * inst
            * 0 G__LDST_VAR_INDEX
            * 1 *arrayindex
            * 2 void (*f)(pbuf,psp,offset,p,ctype,
            * 3 index (not used)
            * 4 pc increment
            * 5 not use
            * 6 var id      
            * stack
            * sp          <-
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+6]);
               G__fprinterr(G__serr, "%3lx: LDST_VAR_INDEX index=%d %s\n"
                            , pc, G__asm_inst[pc+3]
                            , var.Name().c_str());
            }
#endif
            pc += G__asm_inst[pc+4];
            break;

         case G__LDST_VAR_INDEX_OPR:
            /***************************************
            * inst
            * 0 G__LDST_VAR_INDEX_OPR
            * 1 *int1
            * 2 *int2
            * 3 opr +,-
            * 4 void (*f)(pbuf,psp,offset,p,ctype,
            * 5 index (not used)
            * 6 pc increment
            * 7 not use
            * 8 var id      
            * stack
            * sp          <-
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+8]);
               G__fprinterr(G__serr, "%3lx: LDST_VAR_INDEX_OPR index=%d %s\n"
                            , pc, G__asm_inst[pc+5]
                            , var.Name().c_str());
            }
#endif
            pc += G__asm_inst[pc+6];
            break;

         case G__OP2_OPTIMIZED:
            /***************************************
            * inst
            * 0 OP2_OPTIMIZED
            * 1 (*p2f)(buf,buf)
            * stack
            * sp-2  a
            * sp-1  a         <-
            * sp    G__null
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: OP2_OPTIMIZED \n", pc);
#endif
            pc += 2;
            break;

         case G__OP1_OPTIMIZED:
            /***************************************
            * inst
            * 0 OP1_OPTIMIZED
            * 1 (*p2f)(buf)
            * stack
            * sp-1  a
            * sp    G__null     <-
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: OP1_OPTIMIZED \n", pc);
#endif
            pc += 2;
            break;

         case G__LD_VAR:
            /***************************************
            * inst
            * 0 G__LD_VAR
            * 1 index (not used)
            * 2 paran
            * 3 point_level
            * 4 var id      
            * stack
            * sp-paran+1      <- sp-paran+1
            * sp-2
            * sp-1
            * sp
            ***************************************/
            var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
            ig15 = G__asm_inst[pc+1];
            paran = G__asm_inst[pc+2];
            var_type = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3lx: LD_VAR index=%d paran=%d point %c %s\n"
                            , pc, G__asm_inst[pc+1], G__asm_inst[pc+2]
                            , G__asm_inst[pc+3]
                            , var.Name().c_str());
            }
#endif
            /* need optimization */
            if ('p' == var_type &&
                  (islower(G__get_type(var.TypeOf())) || G__PARANORMAL == G__get_reftype(var.TypeOf()))) {
               if (0 == paran && 0 == G__get_paran(var)) {
                  if ('i' == G__get_type(var.TypeOf())) {
                     if (0 == G__LD_VAR_int_optimize(&pc, (long*)G__get_offset(var)))
                        G__LD_p0_optimize(var, pc, G__LDST_VAR_P);
                  }
                  else {
                     G__LD_p0_optimize(var, pc, G__LDST_VAR_P);
                  }
               }
               else if (1 == paran && 1 == G__get_paran(var)) {
                  G__LD_p1_optimize(var, pc, G__LDST_VAR_P);
               }
               else if (paran == G__get_paran(var)) {
                  G__LD_pn_optimize(var, pc, G__LDST_VAR_P);
               }
               else if (1 == paran && 0 == G__get_paran(var) && isupper(G__get_type(var.TypeOf()))) {
                  G__LD_P10_optimize(var, pc, G__LDST_VAR_P);
               }
            }
            pc += 5;
            break;

         case G__LD:
            /***************************************
            * inst
            * 0 G__LD
            * 1 address in data stack
            * stack
            * sp    a
            * sp+1             <-
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x: LD 0x%08lx,%d,%g from data stack index: 0x%lx  %s:%d\n", pc, G__int(G__asm_stack[G__asm_inst[pc+1]]), G__int(G__asm_stack[G__asm_inst[pc+1]]), G__double(G__asm_stack[G__asm_inst[pc+1]]), G__asm_inst[pc+1], __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            // no optimize
            if (G__get_type(G__value_typenum(G__asm_stack[G__asm_inst[pc+1]])) == 'i') {
               G__LD_int_optimize(&pc, &G__asm_stack[G__asm_inst[pc+1]].obj.i);
            }
            pc += 2;
            break;

         case G__CL:
            /***************************************
            * 0 CL
            *  clear stack pointer
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: CL %s:%d\n", pc
                                            , G__srcfile[G__asm_inst[pc+1] / G__CL_FILESHIFT].filename
                                            , G__asm_inst[pc+1]&G__CL_LINEMASK);
#endif
            /* no optimize */
            pc += 2;
            break;

         case G__OP2:
            /***************************************
            * inst
            * 0 OP2
            * 1 (+,-,*,/,%,@,>>,<<,&,|)
            * stack
            * sp-2  a
            * sp-1  b          <-
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_inst[pc+1] < 256 && isprint(G__asm_inst[pc+1])) {
               if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: OP2 '%c'%d \n" , pc
                                               , G__asm_inst[pc+1], G__asm_inst[pc+1]);
            }
            else {
               if (G__asm_dbg)
                  G__fprinterr(G__serr, "%3lx: OP2 %d \n", pc, G__asm_inst[pc+1]);
            }
#endif
            /* need optimization */
            G__OP2_optimize(pc);
            pc += 2;
            break;

         case G__ST_VAR:
            /***************************************
            * inst
            * 0 G__ST_VAR
            * 1 index (not used)
            * 2 paran
            * 3 point_level
            * 4 var id      
            * stack
            * sp-paran        <- sp-paran
            * sp-2
            * sp-1
            * sp
            ***************************************/
            var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
            ig15 = G__asm_inst[pc+1];
            paran = G__asm_inst[pc+2];
            var_type = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3lx: ST_VAR index=%d paran=%d point %c %s\n"
                            , pc, G__asm_inst[pc+1], G__asm_inst[pc+2]
                            , G__asm_inst[pc+3]
                            , var.Name().c_str());
            }
#endif
            /* need optimization */
            if (('p' == var_type || var_type == G__get_type(var.TypeOf())) &&
                  (islower(G__get_type(var.TypeOf())) || G__PARANORMAL == G__get_reftype(var.TypeOf()))) {
               if (0 == paran && 0 == G__get_paran(var)) {
                  G__ST_p0_optimize(var, pc, G__LDST_VAR_P);
               }
               else if (1 == paran && 1 == G__get_paran(var)) {
                  G__ST_p1_optimize(var, pc, G__LDST_VAR_P);
               }
               else if (paran == G__get_paran(var)) {
                  G__ST_pn_optimize(var, pc, G__LDST_VAR_P);
               }
               else if (1 == paran && 0 == G__get_paran(var) && isupper(G__get_type(var.TypeOf()))) {
                  G__ST_P10_optimize(var, pc, G__LDST_VAR_P);
               }
            }
            pc += 5;
            break;

         case G__LD_MSTR:
            /***************************************
            * inst
            * 0 G__LD_MSTR
            * 1 index
            * 2 paran
            * 3 point_level
            * 4 *structmem
            * stack
            * sp-paran+1      <- sp-paran+1
            * sp-2
            * sp-1
            * sp
            ***************************************/
            var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
            ig15 = G__asm_inst[pc+1];
            paran = G__asm_inst[pc+2];
            var_type = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3lx: LD_MSTR index=%d paran=%d point %c %s\n"
                            , pc, G__asm_inst[pc+1], G__asm_inst[pc+2]
                            , G__asm_inst[pc+3]
                            , var.Name().c_str());
            }
#endif
            /* need optimization */
            if ('p' == var_type &&
                  (islower(G__get_type(var.TypeOf())) || G__PARANORMAL == G__get_reftype(var.TypeOf()))) {
               long inst;
               if (G__get_properties(var)->statictype == G__LOCALSTATIC) inst = G__LDST_VAR_P;
               else                                      inst = G__LDST_MSTR_P;
               if (0 == paran && 0 == G__get_paran(var)) {
                  G__LD_p0_optimize(var, pc, inst);
               }
               else if (1 == paran && 1 == G__get_paran(var)) {
                  G__LD_p1_optimize(var, pc, inst);
               }
               else if (paran == G__get_paran(var)) {
                  G__LD_pn_optimize(var, pc, inst);
               }
               else if (1 == paran && 0 == G__get_paran(var) && isupper(G__get_type(var.TypeOf()))) {
                  G__LD_P10_optimize(var, pc, G__LDST_MSTR_P);
               }
            }
            pc += 5;
            break;

         case G__CMPJMP:
            /***************************************
            * 0 CMPJMP
            * 1 *G__asm_test_X()
            * 2 *a
            * 3 *b
            * 4 next_pc
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3lx: CMPJMP (0x%lx)%d (0x%lx)%d to %lx\n"
                            , pc
                            , G__asm_inst[pc+2], *(int *)G__asm_inst[pc+2]
                            , G__asm_inst[pc+3], *(int *)G__asm_inst[pc+3]
                            , G__asm_inst[pc+4]);
            }
#endif
            /* no optmization */
            pc += 5;
            break;

         case G__PUSHSTROS:
            /***************************************
            * inst
            * 0 G__PUSHSTROS
            * stack
            * sp           <- sp-paran
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: PUSHSTROS\n" , pc);
#endif
            /* no optmization */
            ++pc;
            break;

         case G__SETSTROS:
            /***************************************
            * inst
            * 0 G__SETSTROS
            * stack
            * sp-1         <- sp-paran
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: SETSTROS\n", pc);
#endif
            /* no optmization */
            ++pc;
            break;

         case G__POPSTROS:
            /***************************************
            * inst
            * 0 G__POPSTROS
            * stack
            * sp           <- sp-paran
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: POPSTROS\n" , pc);
#endif
            /* no optmization */
            ++pc;
            break;

         case G__ST_MSTR:
            /***************************************
            * inst
            * 0 G__ST_MSTR
            * 1 index
            * 2 paran
            * 3 point_level
            * 4 *structmem
            * stack
            * sp-paran        <- sp-paran
            * sp-2
            * sp-1
            * sp
            ***************************************/
            var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
            ig15 = G__asm_inst[pc+1];
            paran = G__asm_inst[pc+2];
            var_type = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3lx: ST_MSTR index=%d paran=%d point %c %s\n"
                            , pc, G__asm_inst[pc+1], G__asm_inst[pc+2]
                            , G__asm_inst[pc+3]
                            , var.Name().c_str());
            }
#endif
            /* need optimization */
            if ('p' == var_type &&
                  (islower(G__get_type(var.TypeOf())) || G__PARANORMAL == G__get_reftype(var.TypeOf()))) {
               long inst;
               if (G__get_properties(var)->statictype == G__LOCALSTATIC) inst = G__LDST_VAR_P;
               else                                      inst = G__LDST_MSTR_P;
               if (0 == paran && 0 == G__get_paran(var)) {
                  G__ST_p0_optimize(var, pc, inst);
               }
               else if (1 == paran && 1 == G__get_paran(var)) {
                  G__ST_p1_optimize(var, pc, inst);
               }
               else if (paran == G__get_paran(var)) {
                  G__ST_pn_optimize(var, pc, inst);
               }
               else if (1 == paran && 0 == G__get_paran(var) && isupper(G__get_type(var.TypeOf()))) {
                  G__ST_P10_optimize(var, pc, G__LDST_MSTR_P);
               }
            }
            pc += 5;
            break;

         case G__INCJMP:
            /***************************************
            * 0 INCJMP
            * 1 *cntr
            * 2 increment
            * 3 next_pc
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: INCJMP *(int*)0x%lx+%d to %x\n"
                                            , pc , G__asm_inst[pc+1] , G__asm_inst[pc+2]
                                            , G__asm_inst[pc+3]);
#endif
            /* no optimization */
            pc += 4;
            break;

         case G__CNDJMP:
            /***************************************
            * 0 CNDJMP   (jump if 0)
            * 1 next_pc
            * stack
            * sp-1         <-
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: CNDJMP to %x\n"
                                            , pc , G__asm_inst[pc+1]);
#endif
            /* no optimization */
            pc += 2;
            break;

         case G__CMP2:
            /***************************************
            * 0 CMP2
            * 1 operator
            * stack
            * sp-1         <-
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3lx: CMP2 '%c' \n" , pc , G__asm_inst[pc+1]);
            }
#endif
            /* need optimization, but not high priority */
            G__CMP2_optimize(pc);
            pc += 2;
            break;

         case G__JMP:
            /***************************************
            * 0 JMP
            * 1 next_pc
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: JMP %x\n" , pc, G__asm_inst[pc+1]);
#endif
            /* no optimization */
            pc += 2;
            break;

         case G__PUSHCPY:
            /***************************************
            * inst
            * 0 G__PUSHCPY
            * stack
            * sp
            * sp+1            <-
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: PUSHCPY\n", pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__POP:
            /***************************************
            * inst
            * 0 G__POP
            * stack
            * sp-1            <-
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: POP\n" , pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__LD_FUNC:
            /***************************************
            * inst
            * 0 G__LD_FUNC
            * 1 flag + 10 * hash 
            * 2 if flag==0 return type's Id
            *   if flag==1 function name
            *   if flag==2 function's Id
            *   if flag==3 function's Bytecode struct.
            * 3 paran
            * 4 (*func)()
            * 5 this ptr offset for multiple inheritance
            * stack
            * sp-paran+1      <- sp-paran+1
            * sp-2
            * sp-1
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               if (G__asm_inst[pc+1] < G__MAXSTRUCT) {
                  G__fprinterr(G__serr, "%3lx: LD_FUNC '%s' paran: %d  %s:%d\n", pc, "compiled", G__asm_inst[pc+3], __FILE__, __LINE__);
               }
               else {
                  G__fprinterr(G__serr, "%3lx: LD_FUNC '%s' paran: %d  %s:%d\n", pc, (char*) G__asm_inst[pc+1], G__asm_inst[pc+3], __FILE__, __LINE__);
               }
            }
#endif // G__ASM_DBG
            // No optimization.
            pc += 6;
            break;

         case G__RETURN:
            /***************************************
            * 0 RETURN
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: RETURN\n" , pc);
#endif
            /* no optimization */
            pc++;
            return(0);
            break;

         case G__CAST:
            /***************************************
            * 0 CAST
            * 1 type    - AFTER MERGE: Type
            * 2 typenum - AFTER MERGE: Type (cont'ed)
            * 3 tagnum  - AFTER MERGE: not used
            * 4 reftype - AFTER MERGE: not used
            * stack
            * sp-1    <- cast on this
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3lx: CAST to %s\n", pc, reinterpret_cast< ::Reflex::Type*>(&G__asm_inst[pc+1])->Name(::Reflex::SCOPED).c_str());
            }
#endif
            /* need optimization */
            pc += 5;
            break;

         case G__OP1:
            /***************************************
            * inst
            * 0 OP1
            * 1 (+,-)
            * stack
            * sp-1  a
            * sp    G__null     <-
            ***************************************/
#ifdef G__ASM_DBG
            if ((G__asm_inst[pc+1] < 256) && isprint(G__asm_inst[pc+1])) {
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3lx: OP1 '%c'%d\n", pc, G__asm_inst[pc+1], G__asm_inst[pc+1]);
               }
            }
            else {
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3lx: OP1 %d\n", pc, G__asm_inst[pc+1]);
               }
            }
#endif // G__ASM_DBG
            /* need optimization */
            typedef void(*OP1_func_t)(G__value*);
            switch (G__asm_inst[pc+1]) {
               case G__OPR_POSTFIXINC_I:
                  G__asm_inst[pc] = G__OP1_OPTIMIZED;
                  G__asm_inst[pc+1] = (long)(OP1_func_t) G__OP1_postfixinc_T<int>;
                  break;
               case G__OPR_POSTFIXDEC_I:
                  G__asm_inst[pc] = G__OP1_OPTIMIZED;
                  G__asm_inst[pc+1] = (long)(OP1_func_t) G__OP1_postfixdec_T<int>;
                  break;
               case G__OPR_PREFIXINC_I:
                  G__asm_inst[pc] = G__OP1_OPTIMIZED;
                  G__asm_inst[pc+1] = (long)(OP1_func_t) G__OP1_prefixinc_T<int>;
                  break;
               case G__OPR_PREFIXDEC_I:
                  G__asm_inst[pc] = G__OP1_OPTIMIZED;
                  G__asm_inst[pc+1] = (long)(OP1_func_t) G__OP1_prefixdec_T<int>;
                  break;

               case G__OPR_POSTFIXINC_D:
                  G__asm_inst[pc] = G__OP1_OPTIMIZED;
                  G__asm_inst[pc+1] = (long)(OP1_func_t) G__OP1_postfixinc_T<double>;
                  break;
               case G__OPR_POSTFIXDEC_D:
                  G__asm_inst[pc] = G__OP1_OPTIMIZED;
                  G__asm_inst[pc+1] = (long)(OP1_func_t) G__OP1_postfixdec_T<double>;
                  break;
               case G__OPR_PREFIXINC_D:
                  G__asm_inst[pc] = G__OP1_OPTIMIZED;
                  G__asm_inst[pc+1] = (long)(OP1_func_t) G__OP1_prefixinc_T<double>;
                  break;
               case G__OPR_PREFIXDEC_D:
                  G__asm_inst[pc] = G__OP1_OPTIMIZED;
                  G__asm_inst[pc+1] = (long)(OP1_func_t) G__OP1_prefixdec_T<double>;
                  break;

               case G__OPR_POSTFIXINC:
                  G__asm_inst[pc] = G__OP1_OPTIMIZED;
                  G__asm_inst[pc+1] = (long)(OP1_func_t) G__OP1_postfixinc;
                  break;
               case G__OPR_POSTFIXDEC:
                  G__asm_inst[pc] = G__OP1_OPTIMIZED;
                  G__asm_inst[pc+1] = (long)(OP1_func_t) G__OP1_postfixdec;
                  break;
               case G__OPR_PREFIXINC:
                  G__asm_inst[pc] = G__OP1_OPTIMIZED;
                  G__asm_inst[pc+1] = (long)(OP1_func_t) G__OP1_prefixinc;
                  break;
               case G__OPR_PREFIXDEC:
                  G__asm_inst[pc] = G__OP1_OPTIMIZED;
                  G__asm_inst[pc+1] = (long)(OP1_func_t) G__OP1_prefixdec;
                  break;
               case '-':
                  G__asm_inst[pc] = G__OP1_OPTIMIZED;
                  G__asm_inst[pc+1] = (long)(OP1_func_t) G__OP1_minus;
                  break;
            }
            pc += 2;
            break;

         case G__LETVVAL:
            /***************************************
            * inst
            * 0 LETVVAL
            * stack
            * sp-2  a
            * sp-1  b          <-
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: LETVVAL\n" , pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__ADDSTROS:
            /***************************************
            * inst
            * 0 ADDSTROS
            * 1 addoffset
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3lx: ADDSTROS %d\n" , pc, G__asm_inst[pc+1]);
            }
#endif
            /* no optimization */
            pc += 2;
            break;

         case G__LETPVAL:
            /***************************************
            * inst
            * 0 LETPVAL
            * stack
            * sp-2  a
            * sp-1  b          <-
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: LETPVAL\n" , pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__FREETEMP:
            /***************************************
            * 0 FREETEMP
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: FREETEMP\n" , pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__SETTEMP:
            /***************************************
            * 0 SETTEMP
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: SETTEMP\n" , pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__GETRSVD:
            /***************************************
            * 0 GETRSVD
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: GETRSVD\n" , pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__TOPNTR:
            /***************************************
            * inst
            * 0 LETVVAL
            * stack
            * sp-1  a          <-
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: TOPNTR\n" , pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__NOT:
            /***************************************
            * 0 NOT
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: NOT\n" , pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__BOOL:
            /***************************************
            * 0 BOOL
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: BOOL\n" , pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__ISDEFAULTPARA:
            /***************************************
            * 0 ISDEFAULTPARA
            * 1 next_pc
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: !ISDEFAULTPARA JMP %x\n"
                                            , pc, G__asm_inst[pc+1]);
#endif
            pc += 2;
            /* no optimization */
            break;

#ifdef G__ASM_WHOLEFUNC
         case G__LDST_LVAR_P:
            /***************************************
            * inst
            * 0 G__LDST_LVAR_P
            * 1 index (not used)
            * 2 void (*f)(pbuf,psp,offset,p,ctype,
            * 3 (not use)
            * 4 var id      
            * stack
            * sp          <-
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
               G__fprinterr(G__serr, "%3lx: LDST_LVAR_P index=%d %s\n"
                            , pc, G__asm_inst[pc+1]
                            , var.Name().c_str());
            }
#endif
            pc += 5;
            break;

         case G__LD_LVAR:
            /***************************************
            * inst
            * 0 G__LD_LVAR
            * 1 index (not used)
            * 2 paran
            * 3 point_level
            * 4 var id
            * stack
            * sp-paran+1      <- sp-paran+1
            * sp-2
            * sp-1
            * sp
            ***************************************/
            var = G__Dict::GetDict().GetDataMember((size_t) G__asm_inst[pc+4]);
            ig15 = G__asm_inst[pc+1];
            paran = G__asm_inst[pc+2];
            var_type = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               if ((var_type < 256) && isprint(var_type)) {
                  G__fprinterr(G__serr, "%3x: LD_LVAR name: '%s' index: %d paran: %d point %ld '%c'  %s:%d\n", pc, var.Name(::Reflex::SCOPED).c_str(), G__asm_inst[pc+1], G__asm_inst[pc+2], G__asm_inst[pc+3], G__asm_inst[pc+3], __FILE__, __LINE__);
               }
               else {
                  G__fprinterr(G__serr, "%3x: LD_LVAR name: '%s' index: %d paran: %d point %ld  %s:%d\n", pc, var.Name(::Reflex::SCOPED).c_str(), G__asm_inst[pc+1], G__asm_inst[pc+2], G__asm_inst[pc+3], __FILE__, __LINE__);
               }
            }
#endif // G__ASM_DBG
            // need optimization
            if (G__get_reftype(var.TypeOf()) == G__PARAREFERENCE) {
               switch (var_type) {
                  case 'P':
                     G__LD_RP0_optimize(var, pc, G__LDST_LVAR_P);
                     break;
                  case 'p':
                     G__LD_Rp0_optimize(var, pc, G__LDST_LVAR_P);
                     break;
                  case 'v':
                     break;
               }
            }
            else {
               if (
                  (var_type == 'p') &&
                  (
                     islower(G__get_type(var.TypeOf())) ||
                     (G__get_reftype(var.TypeOf()) == G__PARANORMAL)
                  )
               ) {
                  long inst;
                  if (G__get_properties(var)->statictype == G__LOCALSTATIC) {
                     inst = G__LDST_VAR_P;
                  }
                  else {
                     inst = G__LDST_LVAR_P;
                  }
                  if (!paran && !G__get_paran(var)) {
                     if (G__get_type(var.TypeOf()) == 'i') {
                        if (!G__LD_VAR_int_optimize(&pc, (long*) G__get_offset(var)))
                           G__LD_p0_optimize(var, pc, inst);
                     }
                     else {
                        G__LD_p0_optimize(var, pc, inst);
                     }
                  }
                  else if ((paran == 1) && (G__get_paran(var) == 1)) {
                     G__LD_p1_optimize(var, pc, inst);
                  }
                  else if (paran == G__get_paran(var)) {
                     G__LD_pn_optimize(var, pc, inst);
                  }
                  else if ((paran == 1) && !G__get_paran(var) && isupper(G__get_type(var.TypeOf()))) {
                     G__LD_P10_optimize(var, pc, inst);
                  }
               }
            }
            pc += 5;
            break;

         case G__ST_LVAR:
            /***************************************
            * inst
            * 0 G__ST_LVAR
            * 1 index (not used)
            * 2 paran
            * 3 point_level
            * 4 var id      
            * stack
            * sp-paran        <- sp-paran
            * sp-2
            * sp-1
            * sp
            ***************************************/
            var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
            ig15 = G__asm_inst[pc+1];
            paran = G__asm_inst[pc+2];
            var_type = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3lx: ST_LVAR index=%d paran=%d point %c %s\n"
                            , pc, G__asm_inst[pc+1], G__asm_inst[pc+2]
                            , G__asm_inst[pc+3]
                            , var.Name().c_str());
            }
#endif
            /* need optimization */
            if (G__PARAREFERENCE == G__get_reftype(var.TypeOf())) {
               switch (var_type) {
                  case 'P':
                     break;
                  case 'p':
                     G__ST_Rp0_optimize(var, pc, G__LDST_LVAR_P);
                     break;
                  case 'v':
                     break;
               }
            }
            else
               if (('p' == var_type || var_type == G__get_type(var.TypeOf())) &&
                     (islower(G__get_type(var.TypeOf())) || G__PARANORMAL == G__get_reftype(var.TypeOf()))) {
                  long inst;
                  if (G__get_properties(var)->statictype == G__LOCALSTATIC) inst = G__LDST_VAR_P;
                  else                                      inst = G__LDST_LVAR_P;
                  if (0 == paran && 0 == G__get_paran(var)) {
                     G__ST_p0_optimize(var, pc, inst);
                  }
                  else if (1 == paran && 1 == G__get_paran(var)) {
                     G__ST_p1_optimize(var, pc, inst);
                  }
                  else if (paran == G__get_paran(var)) {
                     G__ST_pn_optimize(var, pc, inst);
                  }
                  else if (1 == paran && 0 == G__get_paran(var) && isupper(G__get_type(var.TypeOf()))) {
                     G__ST_P10_optimize(var, pc, inst);
                  }
               }
            pc += 5;
            break;
#endif // G__ASM_WHOLEFUNC

         case G__REWINDSTACK:
            /***************************************
            * inst
            * 0 G__REWINDSTACK
            * 1 rewind
            * stack
            * sp-2            <-  ^
            * sp-1                | rewind
            * sp              <- ..
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3lx: REWINDSTACK %d\n" , pc, G__asm_inst[pc+1]);
            }
#endif
            /* no optimization */
            pc += 2;
            break;

         case G__CND1JMP:
            /***************************************
            * 0 CND1JMP   (jump if 1)
            * 1 next_pc
            * stack
            * sp-1         <-
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3lx: CND1JMP  to %x\n" , pc , G__asm_inst[pc+1]);
            }
#endif
            /* no optimization */
            pc += 2;
            break;

#ifdef G__ASM_IFUNC
         case G__LD_IFUNC:
            /***************************************
            * inst
            * 0 G__LD_IFUNC
            * 1 *name         // unused
            * 2 hash          // unused
            * 3 paran
            * 4 p_ifunc
            * 5 funcmatch
            * 6 memfunc_flag
            * 7 index         // unused
            * stack
            * sp-paran+1      <- sp-paran+1
            * sp-2
            * sp-1
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: LD_IFUNC %s paran=%d\n" , pc
                                            , (char *)G__asm_inst[pc+1], G__asm_inst[pc+3]);
#endif
            /* need optimization, later */
            pc += 8;
            break;

         case G__NEWALLOC:
            /***************************************
            * inst
            * 0 G__NEWALLOC
            * 1 size
            * stack
            * sp-1     <- pinc
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: NEWALLOC size(%d)\n"
                                            , pc, G__asm_inst[pc+1]);
#endif
            /* no optimization */
            pc += 3;
            break;

         case G__SET_NEWALLOC:
            /***************************************
            * inst
            * 0 G__SET_NEWALLOC
            * 1 tagnum       - AFTER MERGE: Type
            * 2 type&reftype - AFTER MERGE: Type (cont'ed)
            * stack
            * sp-1        G__store_struct_offset
            * sp       <-
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: SET_NEWALLOC\n" , pc);
#endif
            /* no optimization */
            pc += 3;
            break;

         case G__DELETEFREE:
            /***************************************
            * inst
            * 0 G__DELETEFREE
            * 1 isarray  0: simple free, 1: array, 2: virtual free
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: DELETEFREE\n", pc);
#endif
            /* no optimization */
            pc += 2;
            break;

         case G__SWAP:
            /***************************************
            * inst
            * 0 G__SWAP
            * stack
            * sp-2          sp-1
            * sp-1          sp-2
            * sp       <-   sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: SWAP\n", pc);
#endif
            /* no optimization */
            ++pc;
            break;

#endif // G__ASM_IFUNC

         case G__BASECONV:
            /***************************************
            * inst
            * 0 G__BASECONV
            * 1 formal_tagnum
            * 2 baseoffset
            * stack
            * sp-2          sp-1
            * sp-1          sp-2
            * sp       <-   sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: BASECONV %d %d\n", pc
                                            , G__asm_inst[pc+1], G__asm_inst[pc+2]);
#endif
            /* no optimization */
            pc += 3;
            break;

         case G__STORETEMP:
            /***************************************
            * 0 STORETEMP
            * stack
            * sp-1
            * sp       <-  sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: STORETEMP\n", pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__ALLOCTEMP:
            /***************************************
            * 0 ALLOCTEMP
            * 1 tagnum
            * stack
            * sp-1
            * sp       <-  sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3lx: ALLOCTEMP %s\n", pc
                            , G__struct.name[G__asm_inst[pc+1]]);
            }
#endif
            /* no optimization */
            pc += 2;
            break;

         case G__POPTEMP:
            /***************************************
            * 0 POPTEMP
            * 1 tagnum
            * stack
            * sp-1
            * sp      <-  sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               if (-1 != G__asm_inst[pc+1])
                  G__fprinterr(G__serr, "%3lx: POPTEMP %s\n" , pc
                               , G__struct.name[G__asm_inst[pc+1]]);
               else
                  G__fprinterr(G__serr, "%3lx: POPTEMP -1\n" , pc);
            }
#endif
            /* no optimization */
            pc += 2;
            break;

         case G__REORDER:
            /***************************************
            * 0 REORDER
            * 1 paran(total)
            * 2 ig25(arrayindex)
            * stack      paran=4 ig25=2    x y z w -> x y z w z w -> x y x y z w -> w z x y
            * sp-3    <-  sp-1
            * sp-2    <-  sp-3
            * sp-1    <-  sp-2
            * sp      <-  sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: REORDER paran=%d ig25=%d\n"
                                            , pc , G__asm_inst[pc+1], G__asm_inst[pc+2]);
#endif
            /* no optimization */
            pc += 3;
            break;

         case G__LD_THIS:
            /***************************************
            * 0 LD_THIS
            * 1 point_level;
            * stack
            * sp-1
            * sp
            * sp+1   <-
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: LD_THIS %s\n"
                                            , pc , G__tagnum.Name().c_str());
#endif
            /* no optimization */
            pc += 2;
            break;

         case G__RTN_FUNC:
            /***************************************
            * 0 RTN_FUNC
            * 1 isreturnvalue
            * stack
            * sp-1   -> return this
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3lx: RTN_FUNC %d\n" , pc , G__asm_inst[pc+1]);
            }
#endif
            /* no optimization */
            pc += 2;
            break;

         case G__SETMEMFUNCENV:
            /***************************************
            * 0 SETMEMFUNCENV:
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: SETMEMFUNCENV\n", pc);
#endif
            /* no optimization */
            pc += 1;
            break;

         case G__RECMEMFUNCENV:
            /***************************************
            * 0 RECMEMFUNCENV:
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: RECMEMFUNCENV\n" , pc);
#endif
            /* no optimization */
            pc += 1;
            break;

         case G__ADDALLOCTABLE:
            /***************************************
            * 0 ADDALLOCTABLE:
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: ADDALLOCTABLE\n" , pc);
#endif
            /* no optimization */
            pc += 1;
            break;

         case G__DELALLOCTABLE:
            /***************************************
            * 0 DELALLOCTABLE:
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: DELALLOCTABLE\n" , pc);
#endif
            /* no optimization */
            pc += 1;
            break;

         case G__BASEDESTRUCT:
            /***************************************
            * 0 BASEDESTRUCT:
            * 1 tagnum
            * 2 isarray
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3lx: BASECONSTRUCT tagnum=%d isarray=%d\n"
                            , pc, G__asm_inst[pc+1], G__asm_inst[pc+2]);
            }
#endif
            /* no optimization */
            pc += 3;
            break;

         case G__REDECL:
            /***************************************
            * 0 REDECL:
            * 1 ig15
            * 2 var
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: REDECL\n", pc);
#endif
            /* no optimization */
            pc += 3;
            break;

         case G__TOVALUE:
            /***************************************
            * 0 TOVALUE:
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: TOVALUE\n", pc);
#endif
            /* no optimization */
            pc += 2;
            break;

         case G__INIT_REF:
            /***************************************
            * inst
            * 0 G__INIT_REF
            * 1 index (not used)
            * 2 paran
            * 3 point_level
            * 4 var id
            * stack
            * sp-paran        <- sp-paran
            * sp-2
            * sp-1
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x: INIT_REF\n", pc);
            }
#endif // G__ASM_DBG
            pc += 5;
            break;

         case G__LETNEWVAL:
            /***************************************
            * inst
            * 0 LETNEWVAL
            * stack
            * sp-2  a
            * sp-1  b          <-
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: LETNEWVAL\n" , pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__SETGVP:
            /***************************************
            * inst
            * 0 SETGVP
            * 1 p or flag      0:use stack-1,else use this value
            * stack
            * sp-1  b          <-
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: SETGVP\n" , pc);
#endif
            /* no optimization */
            pc += 2;
            break;

#ifndef G__OLDIMPLEMENTATION1073
         case G__CTOR_SETGVP:
            /***************************************
            * inst
            * 0 CTOR_SETGVP
            * 1 index (not used)
            * 2 var id
            * 3 mode, 0 local block scope, 1 member offset
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: CTOR_SETGVP\n", pc);
#endif
            /* no optimization */
            pc += 4;
            break;
#endif // G__OLDIMPLEMENTATION1073

         case G__TOPVALUE:
            /***************************************
            * 0 TOPVALUE:
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: TOPVALUE\n", pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__TRY:
            /***************************************
            * inst
            * 0 TRY
            * 1 first_catchblock 
            * 2 endof_catchblock
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: TRY %lx %lx\n", pc
                                            , G__asm_inst[pc+1] , G__asm_inst[pc+2]);
#endif
            /* no optimization */
            pc += 3;
            break;

         case G__TYPEMATCH:
            /***************************************
            * inst
            * 0 TYPEMATCH
            * 1 address in data stack
            * stack
            * sp-1    a      <- comparee
            * sp             <- ismatch
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: TYPEMATCH\n", pc);
#endif
            /* no optimization */
            pc += 2;
            break;

         case G__ALLOCEXCEPTION:
            /***************************************
            * inst
            * 0 ALLOCEXCEPTION
            * 1 tagnum
            * stack
            * sp    a
            * sp+1             <-
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg)
               G__fprinterr(G__serr, "%3lx: ALLOCEXCEPTION %d\n", pc, G__asm_inst[pc+1]);
#endif
            /* no optimization */
            pc += 2;
            break;

         case G__DESTROYEXCEPTION:
            /***************************************
            * inst
            * 0 DESTROYEXCEPTION
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: DESTROYEXCEPTION\n", pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__THROW:
            /***************************************
            * inst
            * 0 THROW
            * stack
            * sp-1    <-
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: THROW\n", pc);
#endif
            /* no optimization */
            pc += 1;
            break;

         case G__CATCH:
            /***************************************
            * inst
            * 0 CATCH
            * 1 filenum
            * 2 linenum
            * 3 pos
            * 4  "
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: CATCH\n", pc);
#endif
            /* no optimization */
            pc += 5;
            break;

         case G__SETARYINDEX:
            /***************************************
            * inst
            * 0 SETARYINDEX
            * 1 allocflag, 1: new object, 0: auto object
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: SETARYINDEX\n", pc);
#endif
            /* no optimization */
            pc += 2;
            break;

         case G__RESETARYINDEX:
            /***************************************
            * inst
            * 0 RESETARYINDEX
            * 1 allocflag, 1: new object, 0: auto object
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: RESETARYINDEX\n", pc);
#endif
            /* no optimization */
            pc += 2;
            break;

         case G__GETARYINDEX:
            /***************************************
            * inst
            * 0 GETARYINDEX
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: GETARYINDEX\n", pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__ENTERSCOPE:
            /***************************************
            * inst
            * 0 ENTERSCOPE
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: ENTERSCOPE\n", pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__EXITSCOPE:
            /***************************************
            * inst
            * 0 EXITSCOPE
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: EXITSCOPE\n", pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__PUTAUTOOBJ:
            /***************************************
            * inst
            * 0 PUTAUTOOBJ
            * 1 var
            * 2 ig15
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: PUTAUTOOBJ\n", pc);
#endif
            /* no optimization */
            pc += 3;
            break;

         case G__CASE:
            /***************************************
            * inst
            * 0 CASE
            * 1 *casetable
            * stack
            * sp-1         <- 
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: CASE\n", pc);
#endif
            /* no optimization */
            pc += 2;
            break;

         case G__MEMCPY:
            /***************************************
            * inst
            * 0 MEMCPY
            * stack
            * sp-3        ORIG  <- sp-3
            * sp-2        DEST
            * sp-1        SIZE
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: MEMCPY\n", pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__MEMSETINT:
            /***************************************
            * inst
            * 0 MEMSETINT
            * 1 mode,  0:no offset, 1: G__store_struct_offset, 2: localmem
            * 2 numdata
            * 3 adr
            * 4 data
            * 5 adr
            * 6 data
            * ...
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: MEMSETINT %ld %ld\n", pc
                                            , G__asm_inst[pc+1], G__asm_inst[pc+2]);
#endif
            /* no optimization */
            pc += G__asm_inst[pc+2] * 2 + 3;
            break;

         case G__JMPIFVIRTUALOBJ:
            /***************************************
            * inst
            * 0 JMPIFVIRTUALOBJ
            * 1 offset
            * 2 next_pc
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: JMPIFVIRTUALOBJ %lx %lx\n", pc
                                            , G__asm_inst[pc+1], G__asm_inst[pc+2]);
#endif
            /* no optimization */
            pc += 3;
            break;

         case G__VIRTUALADDSTROS:
            /***************************************
            * inst
            * 0 VIRTUALADDSTROS
            * 1 tagnum
            * 2 baseclass
            * 3 basen
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: VIRTUALADDSTROS %lx %lx\n", pc
                                            , G__asm_inst[pc+1], G__asm_inst[pc+3]);
#endif
            /* no optimization */
            pc += 4;
            break;

         case G__ROOTOBJALLOCBEGIN:
            /***************************************
            * 0 ROOTOBJALLOCBEGIN
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: ROOTOBJALLOCBEGIN", pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__ROOTOBJALLOCEND:
            /***************************************
            * 0 ROOTOBJALLOCEND
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: ROOTOBJALLOCEND", pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__PAUSE:
            /***************************************
            * inst
            * 0 PAUSe
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: PAUSE\n", pc);
#endif
            /* no optimization */
            ++pc;
            break;

         case G__NOP:
            /***************************************
            * 0 NOP
            ***************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3lx: NOP\n" , pc);
#endif
            /* no optimization */
            ++pc;
            break;

         default:
            /***************************************
            * Illegal instruction.
            * This is a double check and should
            * never happen.
            ***************************************/
            G__fprinterr(G__serr, "%3x: illegal instruction 0x%lx\t%ld\n", pc, G__asm_inst[pc], G__asm_inst[pc]);
            ++pc;
            ++illegal;
            return 1;
            break;
      }
   }
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__dasm(FILE* fout, int isthrow)
{
   // Disassembler
   unsigned int pc = 0; // instruction program counter
   int illegal = 0;
   Reflex::Member var;
   if (!fout) {
      fout = G__serr;
   }
   while (pc < G__MAXINST) {
      switch (G__INST(G__asm_inst[pc])) {
         case G__LDST_VAR_P:
            /***************************************
            * inst
            * 0 G__LDST_VAR_P
            * 1 index (not used)
            * 2 void (*f)(pbuf,psp,offset,p,ctype,
            * 3 (not used)
            * 4 var id
            * stack
            * sp          <-
            ***************************************/
            if (0 == isthrow) {
               var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
               if (!var) return(1);
               fprintf(fout, "%3x: LDST_VAR_P index=%ld %s\n"
                       , pc, G__asm_inst[pc+1]
                       , var.Name(Reflex::SCOPED).c_str());
            }
            pc += 5;
            break;
         case G__LDST_MSTR_P:
            /***************************************
            * inst
            * 0 G__LDST_MSTR_P
            * 1 index (not used)
            * 2 void (*f)(pbuf,psp,offset,p,ctype,
            * 3 (not used)
            * 4 var id
            * stack
            * sp          <-
            ***************************************/
            if (0 == isthrow) {
               var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
               if (!var) return(1);
               fprintf(fout, "%3x: LDST_MSTR_P index=%ld %s\n"
                       , pc, G__asm_inst[pc+1]
                       , var.Name(Reflex::SCOPED).c_str());
            }
            pc += 5;
            break;
         case G__LDST_VAR_INDEX:
            /***************************************
            * inst
            * 0 G__LDST_VAR_INDEX
            * 1 *arrayindex
            * 2 void (*f)(pbuf,psp,offset,p,ctype,
            * 3 index (not used)
            * 4 pc increment
            * 5 not used
            * 6 var id 
            * stack
            * sp          <-
            ***************************************/
            if (0 == isthrow) {
               var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+6]);
               if (!var) return(1);
               fprintf(fout, "%3x: LDST_VAR_INDEX index=%ld %s\n"
                       , pc, G__asm_inst[pc+3]
                       , var.Name(Reflex::SCOPED).c_str());
            }
            pc += G__asm_inst[pc+4];
            break;
         case G__LDST_VAR_INDEX_OPR:
            /***************************************
            * inst
            * 0 G__LDST_VAR_INDEX_OPR
            * 1 *int1
            * 2 *int2
            * 3 opr +,-
            * 4 void (*f)(pbuf,psp,offset,p,ctype,
            * 5 index (not used)
            * 6 pc increment
            * 7 not use
            * 8 var id
            * stack
            * sp          <-
            ***************************************/
            if (0 == isthrow) {
               var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+8]);
               if (!var) return(1);
               fprintf(fout, "%3x: LDST_VAR_INDEX_OPR index=%ld %s\n"
                       , pc, G__asm_inst[pc+5]
                       , var.Name(Reflex::SCOPED).c_str());
            }
            pc += G__asm_inst[pc+6];
            break;
         case G__OP2_OPTIMIZED:
            /***************************************
            * inst
            * 0 OP2_OPTIMIZED
            * 1 (*p2f)(buf)
            * stack
            * sp-2  a
            * sp-1  a           <-
            * sp    G__null
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: OP2_OPTIMIZED \n", pc);
            }
            pc += 2;
            break;
         case G__OP1_OPTIMIZED:
            /***************************************
            * inst
            * 0 OP1_OPTIMIZED
            * 1 (*p2f)(buf)
            * stack
            * sp-1  a
            * sp    G__null     <-
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: OP1_OPTIMIZED \n", pc);
            }
            pc += 2;
            break;
         case G__LD_VAR:
            /***************************************
            * inst
            * 0 G__LD_VAR
            * 1 index (not used)
            * 2 paran
            * 3 point_level
            * 4 var id
            * stack
            * sp-paran+1      <- sp-paran+1
            * sp-2
            * sp-1
            * sp
            ***************************************/
            if (!isthrow) {
               var = G__Dict::GetDict().GetDataMember((size_t) G__asm_inst[pc+4]);
               if (!var) {
                  return 1;
               }
               fprintf(fout, "%3x: LD_VAR index=%ld paran=%ld point %c %s\n", pc, G__asm_inst[pc+1], G__asm_inst[pc+2], (char) G__asm_inst[pc+3], var.Name(Reflex::SCOPED).c_str());
            }
            pc += 5;
            break;
         case G__LD:
            /***************************************
            * inst
            * 0 G__LD
            * 1 address in data stack
            * stack
            * sp    a
            * sp+1             <-
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: LD %g from %lx \n"
                       , pc
                       , G__double(G__asm_stack[G__asm_inst[pc+1]])
                       , G__asm_inst[pc+1]);
            }
            pc += 2;
            break;
         case G__CL:
            /***************************************
            * 0 CL
            *  clear stack pointer
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: CL %s:%ld\n", pc
                       , G__srcfile[G__asm_inst[pc+1] / G__CL_FILESHIFT].filename
                       , G__asm_inst[pc+1]&G__CL_LINEMASK);
            }
            pc += 2;
            break;
         case G__OP2:
            /***************************************
            * inst
            * 0 OP2
            * 1 (+,-,*,/,%,@,>>,<<,&,|)
            * stack
            * sp-2  a
            * sp-1  b          <-
            * sp
            ***************************************/
            if (0 == isthrow) {
               if (G__asm_inst[pc+1] < 256 && isprint(G__asm_inst[pc+1]))
                  fprintf(fout, "%3x: OP2 '%c'%ld \n" , pc
                          , (char)G__asm_inst[pc+1], G__asm_inst[pc+1]);
               else
                  fprintf(fout, "%3x: OP2 %ld \n", pc, G__asm_inst[pc+1]);
            }
            pc += 2;
            break;
         case G__ST_VAR:
            /***************************************
            * inst
            * 0 G__ST_VAR
            * 1 index (not used)
            * 2 paran
            * 3 point_level
            * 4 var id
            * stack
            * sp-paran        <- sp-paran
            * sp-2
            * sp-1
            * sp
            ***************************************/
            if (0 == isthrow) {
               var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
               if (!var) return(1);
               fprintf(fout, "%3x: ST_VAR index=%ld paran=%ld point %c %s\n"
                       , pc, G__asm_inst[pc+1], G__asm_inst[pc+2]
                       , (char)G__asm_inst[pc+3]
                       , var.Name(Reflex::SCOPED).c_str());
            }
            pc += 5;
            break;
         case G__LD_MSTR:
            /***************************************
            * inst
            * 0 G__LD_MSTR
            * 1 index
            * 2 paran
            * 3 point_level
            * 4 *structmem
            * stack
            * sp-paran+1      <- sp-paran+1
            * sp-2
            * sp-1
            * sp
            ***************************************/
            if (0 == isthrow) {
               var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
               if (!var) return(1);
               fprintf(fout, "%3x: LD_MSTR index=%ld paran=%ld point %c %s\n"
                       , pc, G__asm_inst[pc+1], G__asm_inst[pc+2]
                       , (char)G__asm_inst[pc+3]
                       , var.Name(Reflex::SCOPED).c_str());
            }
            pc += 5;
            break;
         case G__CMPJMP:
            /***************************************
            * 0 CMPJMP
            * 1 *G__asm_test_X()
            * 2 *a
            * 3 *b
            * 4 next_pc
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: CMPJMP (0x%lx)%d (0x%lx)%d to %lx\n"
                       , pc
                       , G__asm_inst[pc+2], *(int *)G__asm_inst[pc+2]
                       , G__asm_inst[pc+3], *(int *)G__asm_inst[pc+3]
                       , G__asm_inst[pc+4]);
            }
            pc += 5;
            break;
         case G__PUSHSTROS:
            /***************************************
            * inst
            * 0 G__PUSHSTROS
            * stack
            * sp           <- sp-paran
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: PUSHSTROS\n" , pc);
            }
            ++pc;
            break;
         case G__SETSTROS:
            /***************************************
            * inst
            * 0 G__SETSTROS
            * stack
            * sp-1         <- sp-paran
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: SETSTROS\n", pc);
            }
            ++pc;
            break;
         case G__POPSTROS:
            /***************************************
            * inst
            * 0 G__POPSTROS
            * stack
            * sp           <- sp-paran
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: POPSTROS\n" , pc);
            }
            ++pc;
            break;
         case G__ST_MSTR:
            /***************************************
            * inst
            * 0 G__ST_MSTR
            * 1 index
            * 2 paran
            * 3 point_level
            * 4 *structmem
            * stack
            * sp-paran        <- sp-paran
            * sp-2
            * sp-1
            * sp
            ***************************************/
            if (0 == isthrow) {
               var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
               if (!var) return(1);
               fprintf(fout, "%3x: ST_MSTR index=%ld paran=%ld point %c %s\n"
                       , pc, G__asm_inst[pc+1], G__asm_inst[pc+2]
                       , (char)G__asm_inst[pc+3]
                       , var.Name(Reflex::SCOPED).c_str());
            }
            pc += 5;
            break;
         case G__INCJMP:
            /***************************************
            * 0 INCJMP
            * 1 *cntr
            * 2 increment
            * 3 next_pc
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: INCJMP *(int*)0x%lx+%ld to %lx\n"
                       , pc , G__asm_inst[pc+1] , G__asm_inst[pc+2]
                       , G__asm_inst[pc+3]);
            }
            pc += 4;
            break;
         case G__CNDJMP:
            /***************************************
            * 0 CNDJMP   (jump if 0)
            * 1 next_pc
            * stack
            * sp-1         <-
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: CNDJMP to %lx\n" , pc , G__asm_inst[pc+1]);
            }
            pc += 2;
            break;
         case G__CMP2:
            /***************************************
            * 0 CMP2
            * 1 operator
            * stack
            * sp-1         <-
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: CMP2 '%c' \n" , pc , (char)G__asm_inst[pc+1]);
            }
            pc += 2;
            break;
         case G__JMP:
            /***************************************
            * 0 JMP
            * 1 next_pc
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: JMP %lx\n" , pc, G__asm_inst[pc+1]);
            }
            pc += 2;
            break;
         case G__PUSHCPY:
            /***************************************
            * inst
            * 0 G__PUSHCPY
            * stack
            * sp
            * sp+1            <-
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: PUSHCPY\n", pc);
            }
            ++pc;
            break;
         case G__POP:
            /***************************************
            * inst
            * 0 G__POP
            * stack
            * sp-1            <-
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: POP\n" , pc);
            }
            ++pc;
            break;
         case G__LD_FUNC:
            /***************************************
            * inst
            * 0 G__LD_FUNC
            * 1 flag + 10 * hash 
            * 2 if flag==0 return type's Id
            *   if flag==1 function name
            *   if flag==2 function's Id
            *   if flag==3 function's Bytecode struct.
            * 3 paran
            * 4 (*func)()
            * 5 this ptr offset for multiple inheritance
            * stack
            * sp-paran+1      <- sp-paran+1
            * sp-2
            * sp-1
            * sp
            ***************************************/
            if (!isthrow) {
               if (G__asm_inst[pc+1] < G__MAXSTRUCT) {
                  fprintf(fout, "%3x: LD_FUNC '%s' paran: %ld  %s:%d\n", pc, "compiled", G__asm_inst[pc+3], __FILE__, __LINE__);
               }
               else {
                  fprintf(fout, "%3x: LD_FUNC '%s' paran: %ld  %s:%d\n", pc, (char*) G__asm_inst[pc+1], G__asm_inst[pc+3], __FILE__, __LINE__);
               }
            }
            pc += 6;
            break;
         case G__RETURN:
            /***************************************
            * 0 RETURN
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: RETURN\n" , pc);
            }
            pc++;
            return(0);
            break;
         case G__CAST:
            /***************************************
            * 0 CAST
            * 1 type    - AFTER MERGE: Type
            * 2 typenum - AFTER MERGE: Type (cont'ed)
            * 3 tagnum  - AFTER MERGE: not used
            * 4 reftype - AFTER MERGE: not used
            * stack
            * sp-1    <- cast on this
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: CAST to %s\n", pc, reinterpret_cast< ::Reflex::TypeBase*>(G__asm_inst[pc+1])->Name(::Reflex::SCOPED).c_str());
            }
            pc += 5;
            break;
         case G__OP1:
            /***************************************
            * inst
            * 0 OP1
            * 1 (+,-)
            * stack
            * sp-1  a
            * sp    G__null     <-
            ***************************************/
            if (0 == isthrow) {
               if (G__asm_inst[pc+1] < 256 && isprint(G__asm_inst[pc+1]))
                  fprintf(fout, "%3x: OP1 '%c'%ld\n", pc
                          , (char)G__asm_inst[pc+1], G__asm_inst[pc+1]);
               else
                  fprintf(fout, "%3x: OP1 %ld\n", pc, G__asm_inst[pc+1]);
            }
            pc += 2;
            break;
         case G__LETVVAL:
            /***************************************
            * inst
            * 0 LETVVAL
            * stack
            * sp-2  a
            * sp-1  b          <-
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: LETVVAL\n" , pc);
            }
            ++pc;
            break;
         case G__ADDSTROS:
            /***************************************
            * inst
            * 0 ADDSTROS
            * 1 addoffset
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: ADDSTROS %ld\n" , pc, G__asm_inst[pc+1]);
            }
            pc += 2;
            break;
         case G__LETPVAL:
            /***************************************
            * inst
            * 0 LETPVAL
            * stack
            * sp-2  a
            * sp-1  b          <-
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: LETPVAL\n" , pc);
            }
            ++pc;
            break;
         case G__FREETEMP:
            /***************************************
            * 0 FREETEMP
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: FREETEMP\n" , pc);
            }
            ++pc;
            break;
         case G__SETTEMP:
            /***************************************
            * 0 SETTEMP
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: SETTEMP\n" , pc);
            }
            ++pc;
            break;
         case G__GETRSVD:
            /***************************************
            * 0 GETRSVD
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: GETRSVD\n" , pc);
            }
            ++pc;
            break;
         case G__TOPNTR:
            /***************************************
            * inst
            * 0 LETVVAL
            * stack
            * sp-1  a          <-
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: TOPNTR\n" , pc);
            }
            ++pc;
            break;
         case G__NOT:
            /***************************************
            * 0 NOT
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: NOT\n" , pc);
            }
            ++pc;
            break;
         case G__BOOL:
            /***************************************
            * 0 BOOL
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: BOOL\n" , pc);
            }
            ++pc;
            break;
         case G__ISDEFAULTPARA:
            /***************************************
            * 0 ISDEFAULTPARA
            * 1 next_pc
            ***************************************/
            if (0 == isthrow) {
               G__fprinterr(G__serr, "%3x: !ISDEFAULTPARA JMP %lx\n", pc, G__asm_inst[pc+1]);
            }
            pc += 2;
            break;
#ifdef G__ASM_WHOLEFUNC
         case G__LDST_LVAR_P:
            /***************************************
            * inst
            * 0 G__LDST_LVAR_P
            * 1 index (not used)
            * 2 void (*f)(pbuf,psp,offset,p,ctype,
            * 3 (not use)
            * 4 var id 
            * stack
            * sp          <-
            ***************************************/
            if (0 == isthrow) {
               var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
               if (!var) return(1);
               fprintf(fout, "%3x: LDST_LVAR_P index=%ld %s\n"
                       , pc, G__asm_inst[pc+1]
                       , var.Name(Reflex::SCOPED).c_str());
            }
            pc += 5;
            break;
         case G__LD_LVAR:
            /***************************************
            * inst
            * 0 G__LD_LVAR
            * 1 index (not used)
            * 2 paran
            * 3 point_level
            * 4 var id
            * stack
            * sp-paran+1      <- sp-paran+1
            * sp-2
            * sp-1
            * sp
            ***************************************/
            if (0 == isthrow) {
               var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
               if (!var) return(1);
               fprintf(fout, "%3x: LD_LVAR index=%ld paran=%ld point %c %s\n"
                       , pc, G__asm_inst[pc+1], G__asm_inst[pc+2]
                       , (char)G__asm_inst[pc+3]
                       , var.Name(Reflex::SCOPED).c_str());
            }
            pc += 5;
            break;
         case G__ST_LVAR:
            /***************************************
            * inst
            * 0 G__ST_LVAR
            * 1 index (not used)
            * 2 paran
            * 3 point_level
            * 4 var id
            * stack
            * sp-paran        <- sp-paran
            * sp-2
            * sp-1
            * sp
            ***************************************/
            if (0 == isthrow) {
               var = G__Dict::GetDict().GetDataMember((size_t)G__asm_inst[pc+4]);
               if (!var) return(1);
               fprintf(fout, "%3x: ST_LVAR index=%ld paran=%ld point %c %s\n"
                       , pc, G__asm_inst[pc+1], G__asm_inst[pc+2]
                       , (char)G__asm_inst[pc+3]
                       , var.Name(Reflex::SCOPED).c_str());
            }
            pc += 5;
            break;
#endif // G__ASM_WHOLEFUNC
         case G__REWINDSTACK:
            /***************************************
            * inst
            * 0 G__REWINDSTACK
            * 1 rewind
            * stack
            * sp-2            <-  ^
            * sp-1                | rewind
            * sp              <- ..
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: REWINDSTACK %ld\n" , pc, G__asm_inst[pc+1]);
            }
            pc += 2;
            break;
         case G__CND1JMP:
            /***************************************
            * 0 CND1JMP   (jump if 1)
            * 1 next_pc
            * stack
            * sp-1         <-
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: CND1JMP  to %lx\n" , pc , G__asm_inst[pc+1]);
            }
            pc += 2;
            break;
#ifdef G__ASM_IFUNC
         case G__LD_IFUNC:
            /***************************************
            * inst
            * 0 G__LD_IFUNC
            * 1 *name         // unused
            * 2 hash          // unused
            * 3 paran
            * 4 p_ifunc
            * 5 funcmatch
            * 6 memfunc_flag
            * 7 index         // unused
            * stack
            * sp-paran+1      <- sp-paran+1
            * sp-2
            * sp-1
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: LD_IFUNC %s paran=%ld\n" , pc
                       , (char *)G__asm_inst[pc+1], G__asm_inst[pc+3]);
            }
            pc += 8;
            break;
         case G__NEWALLOC:
            /***************************************
            * inst
            * 0 G__NEWALLOC
            * 1 size
            * stack
            * sp-1     <- pinc
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: NEWALLOC size(%ld)\n"
                       , pc, G__asm_inst[pc+1]);
            }
            pc += 3;
            break;
         case G__SET_NEWALLOC:
            /***************************************
            * inst
            * 0 G__SET_NEWALLOC
            * 1 tagnum       - AFTER MERGE: Type
            * 2 type&reftype - AFTER MERGE: Type (cont'ed)
            * stack
            * sp-1        G__store_struct_offset
            * sp       <-
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: SET_NEWALLOC\n" , pc);
            }
            pc += 3;
            break;
         case G__DELETEFREE:
            /***************************************
            * inst
            * 0 G__DELETEFREE
            * 1 isarray  0: simple free, 1: array, 2: virtual free
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: DELETEFREE\n", pc);
            }
            pc += 2;
            break;
         case G__SWAP:
            /***************************************
            * inst
            * 0 G__SWAP
            * stack
            * sp-2          sp-1
            * sp-1          sp-2
            * sp       <-   sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: SWAP\n", pc);
            }
            ++pc;
            break;
#endif // G__ASM_IFUNC
         case G__BASECONV:
            /***************************************
            * inst
            * 0 G__BASECONV
            * 1 formal_tagnum
            * 2 baseoffset
            * stack
            * sp-2          sp-1
            * sp-1          sp-2
            * sp       <-   sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: BASECONV %ld %ld\n", pc
                       , G__asm_inst[pc+1], G__asm_inst[pc+2]);
            }
            pc += 3;
            break;
         case G__STORETEMP:
            /***************************************
            * 0 STORETEMP
            * stack
            * sp-1
            * sp       <-  sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: STORETEMP\n", pc);
            }
            ++pc;
            break;
         case G__ALLOCTEMP:
            /***************************************
            * 0 ALLOCTEMP
            * 1 tagnum
            * stack
            * sp-1
            * sp       <-  sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: ALLOCTEMP %s\n", pc, G__struct.name[G__asm_inst[pc+1]]);
            }
            pc += 2;
            break;
         case G__POPTEMP:
            /***************************************
            * 0 POPTEMP
            * 1 tagnum
            * stack
            * sp-1
            * sp      <-  sp
            ***************************************/
            if (0 == isthrow) {
               if (-1 != G__asm_inst[pc+1])
                  fprintf(fout, "%3x: POPTEMP %s\n"
                          , pc, G__struct.name[G__asm_inst[pc+1]]);
               else
                  fprintf(fout, "%3x: POPTEMP -1\n", pc);
            }
            pc += 2;
            break;
         case G__REORDER:
            /***************************************
            * 0 REORDER
            * 1 paran(total)
            * 2 ig25(arrayindex)
            * stack      paran=4 ig25=2    x y z w -> x y z w z w -> x y x y z w -> w z x y
            * sp-3    <-  sp-1
            * sp-2    <-  sp-3
            * sp-1    <-  sp-2
            * sp      <-  sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: REORDER paran=%ld ig25=%ld\n"
                       , pc , G__asm_inst[pc+1], G__asm_inst[pc+2]);
            }
            pc += 3;
            break;
         case G__LD_THIS:
            /***************************************
            * 0 LD_THIS
            * 1 point_level;
            * stack
            * sp-1
            * sp
            * sp+1   <-
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: LD_THIS %s\n"
                       , pc , G__struct.name[G__get_tagnum(G__tagnum)]);
            }
            pc += 2;
            break;
         case G__RTN_FUNC:
            /***************************************
            * 0 RTN_FUNC
            * 1 isreturnvalue
            * stack
            * sp-1   -> return this
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: RTN_FUNC %ld\n" , pc , G__asm_inst[pc+1]);
            }
            pc += 2;
            break;
         case G__SETMEMFUNCENV:
            /***************************************
            * 0 SETMEMFUNCENV:
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: SETMEMFUNCENV\n", pc);
            }
            pc += 1;
            break;
         case G__RECMEMFUNCENV:
            /***************************************
            * 0 RECMEMFUNCENV:
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: RECMEMFUNCENV\n" , pc);
            }
            pc += 1;
            break;
         case G__ADDALLOCTABLE:
            /***************************************
            * 0 ADDALLOCTABLE:
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: ADDALLOCTABLE\n" , pc);
            }
            pc += 1;
            break;
         case G__DELALLOCTABLE:
            /***************************************
            * 0 DELALLOCTABLE:
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: DELALLOCTABLE\n" , pc);
            }
            pc += 1;
            break;
         case G__BASEDESTRUCT:
            /***************************************
            * 0 BASEDESTRUCT:
            * 1 tagnum
            * 2 isarray
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: BASECONSTRUCT tagnum=%ld isarray=%ld\n"
                       , pc, G__asm_inst[pc+1], G__asm_inst[pc+2]);
            }
            pc += 3;
            break;
         case G__REDECL:
            /***************************************
            * 0 REDECL:
            * 1 ig15
            * 2 var
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: REDECL\n", pc);
            }
            pc += 3;
            break;
         case G__TOVALUE:
            /***************************************
            * 0 TOVALUE:
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: TOVALUE\n", pc);
            }
            pc += 2;
            break;
         case G__INIT_REF:
            /***************************************
            * inst
            * 0 G__INIT_REF
            * 1 index (not used)
            * 2 paran
            * 3 point_level
            * 4 var id
            * stack
            * sp-paran        <- sp-paran
            * sp-2
            * sp-1
            * sp
            ***************************************/
            if (!isthrow) {
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x: INIT_REF\n", pc);
               }
            }
            pc += 5;
            break;
         case G__LETNEWVAL:
            /***************************************
            * inst
            * 0 LETNEWVAL
            * stack
            * sp-2  a
            * sp-1  b          <-
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: LETNEWVAL\n" , pc);
            }
            ++pc;
            break;
         case G__SETGVP:
            /***************************************
            * inst
            * 0 SETGVP
            * 1 p or flag      0:use stack-1,else use this value
            * stack
            * sp-1  b          <-
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: SETGVP\n" , pc);
               /* no optimization */
            }
            pc += 2;
            break;
         case G__TOPVALUE:
            /***************************************
            * 0 TOPVALUE:
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: TOPVALUE\n", pc);
            }
            ++pc;
            break;
#ifndef G__OLDIMPLEMENTATION1073
         case G__CTOR_SETGVP:
            /***************************************
            * inst
            * 0 CTOR_SETGVP
            * 1 index (not used)
            * 2 var id
            * 3 mode, 0 local block scope, 1 member offset
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: CTOR_SETGVP\n", pc);
            }
            pc += 4;
            break;
#endif // G__OLDIMPLEMENTATION1073
         case G__TRY:
            /***************************************
            * inst
            * 0 TRY
            * 1 first_catchblock 
            * 2 endof_catchblock
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: TRY %lx %lx\n", pc
                       , G__asm_inst[pc+1] , G__asm_inst[pc+2]);
            }
            pc += 3;
            break;
         case G__TYPEMATCH:
            /***************************************
            * inst
            * 0 TYPEMATCH
            * 1 address in data stack
            * stack
            * sp-1    a      <- comparee
            * sp             <- ismatch
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: TYPEMATCH\n", pc);
            }
            pc += 2;
            break;
         case G__ALLOCEXCEPTION:
            /***************************************
            * inst
            * 0 ALLOCEXCEPTION
            * 1 tagnum
            * stack
            * sp    a
            * sp+1             <-
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: ALLOCEXCEPTION %ld\n", pc, G__asm_inst[pc+1]);
            }
            pc += 2;
            break;
         case G__DESTROYEXCEPTION:
            /***************************************
            * inst
            * 0 DESTROYEXCEPTION
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: DESTROYEXCEPTION\n", pc);
            }
            ++pc;
            break;
         case G__THROW:
            /***************************************
            * inst
            * 0 THROW
            * stack
            * sp-1    <-
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: THROW\n" , pc);
            }
            pc += 1;
            break;
         case G__CATCH:
            /***************************************
            * inst
            * 0 CATCH
            * 1 filenum
            * 2 linenum
            * 3 pos
            * 4  "
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: CATCH\n" , pc);
            }
            else {
               fpos_t store_pos;
               struct G__input_file store_ifile = G__ifile;
               G__StrBuf statement_sb(G__LONGLINE);
               char *statement = statement_sb;
#if defined(G__NONSCALARFPOS2)
               fpos_t pos;
               pos.__pos = (off_t)G__asm_inst[pc+3];
#elif defined(G__NONSCALARFPOS_QNX)
               fpos_t pos;
               pos._Off = (off_t)G__asm_inst[pc+3];
#else
               fpos_t pos = (fpos_t)G__asm_inst[pc+3];
#endif
               fgetpos(G__ifile.fp, &store_pos);
               G__ifile.filenum = (short)G__asm_inst[pc+1];
               G__ifile.line_number = G__asm_inst[pc+2];
               strcpy(G__ifile.name, G__srcfile[G__ifile.filenum].filename);
               G__ifile.fp = G__srcfile[G__ifile.filenum].fp;
               fsetpos(G__ifile.fp, &pos);
               G__asm_exec = 0;
               G__return = G__RETURN_NON;
               G__exec_catch(statement);

               G__ifile = store_ifile;
               fsetpos(G__ifile.fp, &store_pos);
               return(G__CATCH);
            }
            pc += 5;
            break;
         case G__SETARYINDEX:
            /***************************************
            * inst
            * 0 SETARYINDEX
            * 1 allocflag, 1: new object, 0: auto object
            ***************************************/
            if (isthrow) {
               G__fprinterr(G__serr, "%3x: SETARYINDEX\n", pc);
            }
            pc += 2;
            break;
         case G__RESETARYINDEX:
            /***************************************
            * inst
            * 0 RESETARYINDEX
            * 1 allocflag, 1: new object, 0: auto object
            ***************************************/
            if (isthrow) {
               G__fprinterr(G__serr, "%3x: RESETARYINDEX\n", pc);
            }
            pc += 2;
            break;
         case G__GETARYINDEX:
            /***************************************
            * inst
            * 0 GETARYINDEX
            ***************************************/
            if (isthrow) {
               G__fprinterr(G__serr, "%3x: GETARYINDEX\n", pc);
            }
            ++pc;
            break;
         case G__ENTERSCOPE:
            /***************************************
            * inst
            * 0 ENTERSCOPE
            ***************************************/
#ifdef G__ASM_DBG
            if (0 == isthrow) G__fprinterr(G__serr, "%3x: ENTERSCOPE\n", pc);
#endif
            /* no optimization */
            ++pc;
            break;
         case G__EXITSCOPE:
            /***************************************
            * inst
            * 0 EXITSCOPE
            ***************************************/
#ifdef G__ASM_DBG
            if (0 == isthrow) G__fprinterr(G__serr, "%3x: EXITSCOPE\n", pc);
#endif
            /* no optimization */
            ++pc;
            break;
         case G__PUTAUTOOBJ:
            /***************************************
            * inst
            * 0 PUTAUTOOBJ
            * 1 var
            * 2 ig15
            ***************************************/
#ifdef G__ASM_DBG
            if (0 == isthrow) G__fprinterr(G__serr, "%3x: PUTAUTOOBJ\n", pc);
#endif
            /* no optimization */
            pc += 3;
            break;
         case G__CASE:
            /***************************************
            * inst
            * 0 CASE
            * 1 *casetable
            * stack
            * sp-1         <- 
            * sp
            ***************************************/
#ifdef G__ASM_DBG
            if (0 == isthrow) G__fprinterr(G__serr, "%3x: CASE\n", pc);
#endif
            /* no optimization */
            pc += 2;
            break;
         case G__MEMCPY:
            /***************************************
            * inst
            * 0 MEMCPY
            * stack
            * sp-3        ORIG  <- sp-3
            * sp-2        DEST
            * sp-1        SIZE
            * sp
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: MEMCPY\n" , pc);
            }
            ++pc;
            break;
         case G__MEMSETINT:
            /***************************************
            * inst
            * 0 MEMSETINT
            * 1 mode,  0:no offset, 1: G__store_struct_offset, 2: localmem
            * 2 numdata
            * 3 adr
            * 4 data
            * 5 adr
            * 6 data
            * ...
            ***************************************/
#ifdef G__ASM_DBG
            if (0 == isthrow) fprintf(fout, "%3x: MEMSETINT %ld %ld\n", pc
                                         , G__asm_inst[pc+1], G__asm_inst[pc+2]);
#endif
            pc += G__asm_inst[pc+2] * 2 + 3;
            break;
         case G__JMPIFVIRTUALOBJ:
            /***************************************
            * inst
            * 0 JMPIFVIRTUALOBJ
            * 1 offset
            * 2 next_pc
            ***************************************/
#ifdef G__ASM_DBG
            if (0 == isthrow) fprintf(fout, "%3x: JMPIFVIRTUALOBJ %lx %lx\n", pc
                                         , G__asm_inst[pc+1], G__asm_inst[pc+2]);
#endif
            pc += 3;
            break;
         case G__VIRTUALADDSTROS:
            /***************************************
            * inst
            * 0 VIRTUALADDSTROS
            * 1 tagnum
            * 2 baseclass
            * 3 basen
            ***************************************/
#ifdef G__ASM_DBG
            if (0 == isthrow) fprintf(fout, "%3x: VIRTUALADDSTROS %lx %lx\n", pc
                                         , G__asm_inst[pc+1], G__asm_inst[pc+3]);
#endif
            pc += 4;
            break;
         case G__ROOTOBJALLOCBEGIN:
            /***************************************
            * 0 ROOTOBJALLOCBEGIN
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: ROOTOBJALLOCBEGIN\n" , pc);
            }
            ++pc;
            break;
         case G__ROOTOBJALLOCEND:
            /***************************************
            * 0 ROOTOBJALLOCEND
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: ROOTOBJALLOCEND\n" , pc);
            }
            ++pc;
            break;
         case G__PAUSE:
            /***************************************
            * inst
            * 0 PAUSe
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: PAUSE\n" , pc);
            }
            ++pc;
            break;
         case G__NOP:
            /***************************************
            * 0 NOP
            ***************************************/
            if (0 == isthrow) {
               fprintf(fout, "%3x: NOP\n" , pc);
            }
            ++pc;
            break;
         default:
            /***************************************
            * Illegal instruction.
            * This is a double check and should
            * never happen.
            ***************************************/
            fprintf(fout, "%3x: illegal instruction 0x%lx\t%ld\n", pc, G__asm_inst[pc], G__asm_inst[pc]);
            ++pc;
            ++illegal;
            if (illegal > 20) {
               return 0;
            }
            break;
      }
   }
   return 0;
}

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:3
 * c-continued-statement-offset:3
 * c-brace-offset:-3
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-3
 * compile-command:"make -k"
 * End:
 */
