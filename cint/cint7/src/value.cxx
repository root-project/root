/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file value.c
 ************************************************************************
 * Description:
 *  internal meta-data structure handling
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "Dict.h"
#include "value.h"
#include "pcode.h"
#include "Reflex/Builder/TypeBuilder.h"

using namespace Cint::Internal;

//______________________________________________________________________________
extern "C" void G__letdouble(G__value *buf, int type, double value)
{
   G__value_typenum(*buf) = G__get_from_type(type, 0);
   buf->obj.d = value;
   /*
   buf->tagnum = -1;
   buf->typenum = -1;
   */
}

//______________________________________________________________________________
extern "C" void G__letbool(G__value *buf, int type, long value)
{
   G__value_typenum(*buf) = G__get_from_type(type, 0);
#ifdef G__BOOL4BYTE
   buf->obj.i = value ? 1 : 0;
#else // G__BOOL4BYTE
   buf->obj.uch = value ? 1 : 0;
#endif // G__BOOL4BYTE
   // --
}

//______________________________________________________________________________
extern "C" void G__letint(G__value* buf, int type, long value)
{
   if (type != 'U') {
      G__value_typenum(*buf) = G__get_from_type(type, 1);
   }
   switch (type) {
      case 'w': // logic
      case 'r': // unsigned short
         buf->obj.ush = value;
         break;
      case 'h': // unsigned int
         buf->obj.uin = value;
         break;
#ifndef G__BOOL4BYTE
      case 'g': // boolean
#endif // G__BOOL4BYTE
      case 'b': // unsigned char
         buf->obj.uch = value;
         break;
      case 'k': // unsigned long
         buf->obj.ulo = value;
         break;
      case 'n':
         buf->obj.ll = value;
         break;
      case 'm':
         buf->obj.ull = value;
         break;
      case 'q':
         buf->obj.ld = value;
         break;
      case 'i':
         buf->obj.i = value;
         break; // should be "in", but there are too many cases where "i" is read out later
      case 'c':
         buf->obj.ch = value;
         break;
      case 's':
         buf->obj.sh = value;
         break;
      default:
         buf->obj.i = value;
   }
}

//______________________________________________________________________________
void Cint::G__letpointer(G__value *buf, long value, const ::Reflex::Type &type)
{
   G__value_typenum(*buf) = type;
   buf->obj.i = value;
}

//______________________________________________________________________________
extern "C" void G__letLonglong(G__value *buf, int type, G__int64 value)
{
   G__value_typenum(*buf) = G__get_from_type(type, 0);
   buf->obj.ll = value;
   //buf->tagnum = -1;
   //buf->typenum = -1;
   //buf->obj.reftype.reftype = G__PARANORMAL;
}

//______________________________________________________________________________
extern "C" void G__letULonglong(G__value *buf, int type, G__uint64 value)
{
   G__value_typenum(*buf) = G__get_from_type(type, 0);
   buf->obj.ull = value;
   //buf->tagnum = -1;
   //buf->typenum = -1;
   //buf->obj.reftype.reftype = G__PARANORMAL;
}

//______________________________________________________________________________
extern "C" void G__letLongdouble(G__value *buf, int type, long double value)
{
   G__value_typenum(*buf) = G__get_from_type(type, 0);
   buf->obj.ld = value;
   //buf->tagnum = -1;
   //buf->typenum = -1;
   //buf->obj.reftype.reftype = G__PARANORMAL;
}

//______________________________________________________________________________
int Cint::Internal::G__isdouble(G__value buf)
{
   switch (G__get_type(buf)) {
      case 'd':
      case 'f':
         return(1);
      default:
         return(0);
   }
}

//______________________________________________________________________________
extern "C" double G__double(G__value buf)
{
   return G__convertT<double>(&buf);
}

//______________________________________________________________________________
long Cint::Internal::G__bool(G__value buf)
{
   return G__convertT<bool>(&buf);
}

//______________________________________________________________________________
extern "C" long G__int(G__value buf)
{
   return G__convertT<long>(&buf);
}

//______________________________________________________________________________
extern "C" unsigned long G__uint(G__value buf)
{
   return G__convertT<unsigned long>(&buf);
}

//______________________________________________________________________________
extern "C" G__int64 G__Longlong(G__value buf)
{
   return G__convertT<G__int64>(&buf);
}

//______________________________________________________________________________
extern "C" G__uint64 G__ULonglong(G__value buf)
{
   return G__convertT<G__uint64>(&buf);
}

//______________________________________________________________________________
extern "C" long double G__Longdouble(G__value buf)
{
   return G__convertT<long double>(&buf);
}

//______________________________________________________________________________
G__value Cint::Internal::G__toXvalue(G__value result, int var_type)
{
   switch (var_type) {
      case 'v':
         return(G__tovalue(result));
         break;
      case 'P':
#ifdef G__ASM
         if (G__asm_noverflow) {
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: TOPVALUE\n", G__asm_cp);
#endif
            G__asm_inst[G__asm_cp] = G__TOPVALUE;
            G__inc_cp_asm(1, 0);
         }
#endif

         G__value_typenum(result) = ::Reflex::PointerBuilder(G__value_typenum(result));

         if (result.ref) result.obj.i = result.ref;
         else if (G__no_exec_compile) result.obj.i = 1;
         result.ref = 0;
         return result;
         break;
      default:
         return result;
         break;
   }
}

//______________________________________________________________________________
G__value Cint::Internal::G__tovalue(G__value p)
{
   G__value result;
   result = p;
   //Code that was there to handle some typedef case
   //if(G__value_typenum(p) && G__value_typenum(p).IsArray()) {
   //   G__value_typenum(result) = ::Reflex::Type();
   //}
#ifdef G__ASM
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: TOVALUE  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__TOVALUE;
      //G__asm_inst[G__asm_cp+1] = func_ptr; // This will be filled in later.
      G__inc_cp_asm(2, 0);
   }
   Reflex::Type final(G__value_typenum(p).FinalType());
   if (G__no_exec_compile) {
      // -- We are generating bytecode, but not executing.
      if (final.IsPointer()) {
         switch (G__get_reftype(final)) {
            case G__PARANORMAL:
               // ;result.type = tolower(p.type);
               G__value_typenum(result) = G__deref(G__value_typenum(p)); // Strip the pointer
               result.obj.i = 1;
               result.ref = p.obj.i;
               if (G__asm_noverflow) {
                  // -- We are generating bytecode, backpatch the previously generated TOVALUE instruction.
                  typedef void(*dereff_t)(G__value*);
                  switch (G__get_type(G__value_typenum(p))) {
                     case 'B':
                        G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<unsigned char>;
                        break;
                     case 'C':
                        G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<char>;
                        break;
                     case 'R':
                        G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<unsigned short>;
                        break;
                     case 'S':
                        G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<short>;
                        break;
                     case 'H':
                        G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<unsigned int>;
                        break;
                     case 'I':
                        G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<int>;
                        break;
                     case 'K':
                        G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<unsigned long>;
                        break;
                     case 'L':
                        G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<long>;
                        break;
                     case 'F':
                        G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<float>;
                        break;
                     case 'D':
                        G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<double>;
                        break;
                     case 'U':
                        G__asm_inst[G__asm_cp-1] = (long) (dereff_t) ::Cint::Internal::G__asm_tovalue_U;
                        break;
                     default:
                        break;
                  }
               }
               return result;
            case G__PARAP2P:
               result.obj.i = 1;
               result.ref = p.obj.i;
               // result.obj.reftype.reftype=G__PARANORMAL;
               G__value_typenum(result) = G__deref(G__value_typenum(p)); // Strip the pointer
               if (G__asm_noverflow) {
                  // -- We are generating bytecode, backpatch the previously generated TOVALUE instruction.
                  G__asm_inst[G__asm_cp-1] = (long) G__asm_tovalue_p2p;
               }
               return result;
            case G__PARAP2P2P:
               result.obj.i = 1;
               result.ref = p.obj.i;
               G__value_typenum(result) = G__deref(G__value_typenum(p)); // Strip the pointer
               //result.obj.reftype.reftype=G__PARAP2P;
               if (G__asm_noverflow) {
                  // -- We are generating bytecode, backpatch the previously generated TOVALUE instruction.
                  G__asm_inst[G__asm_cp-1] = (long) G__asm_tovalue_p2p;
               }
               return result;
            case G__PARAREFERENCE:
               break;
            default:
               result.obj.i = 1;
               result.ref = p.obj.i;
               G__value_typenum(result) = G__deref(G__value_typenum(p)); // Strip the pointer
               // --result.obj.reftype.reftype;
               if (G__asm_noverflow) {
                  // -- We are generating bytecode, backpatch the previously generated TOVALUE instruction.
                  G__asm_inst[G__asm_cp-1] = (long)G__asm_tovalue_p2p;
               }
               return result;
         }
      }
   }
#endif // G__ASM
   if (final.IsPointer()) {
      switch (G__get_reftype(final)) {
         case G__PARAP2P:
            result.obj.i = (long)(*(long *)(p.obj.i));
            result.ref = p.obj.i;
            G__value_typenum(result) = G__deref(G__value_typenum(p)); // Strip the pointer
            //result.obj.reftype.reftype=G__PARANORMAL;
            if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)G__asm_tovalue_p2p;
            return(result);
         case G__PARAP2P2P:
            result.obj.i = (long)(*(long *)(p.obj.i));
            result.ref = p.obj.i;
            G__value_typenum(result) = G__deref(G__value_typenum(p)); // Strip the pointer
            //result.obj.reftype.reftype=G__PARAP2P;
            if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)G__asm_tovalue_p2p;
            return(result);
         case G__PARANORMAL:
         case G__PARAREFERENCE:
            break;
         default:
            result.obj.i = (long)(*(long *)(p.obj.i));
            result.ref = p.obj.i;
            G__value_typenum(result) = G__deref(G__value_typenum(p)); // Strip the pointer
            //--result.obj.reftype.reftype;
            if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)G__asm_tovalue_p2p;
            return(result);
      }
   }
   typedef void (*dereff_t)(G__value*);
   switch (G__get_type(final)) {
      case 'N':
         G__asm_deref_cast<G__int64>(&p, &result);
         result.ref = p.obj.i;
         if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<G__int64>;
         break;
      case 'M':
         G__asm_deref_cast<G__uint64>(&p, &result);
         result.ref = p.obj.i;
         if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<G__uint64>;
         break;
      case 'Q':
         G__asm_deref_cast<long double>(&p, &result);
         result.ref = p.obj.i;
         if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<long double>;
         break;
      case 'G':
#ifdef G__BOOL4BYTE
         G__asm_deref_cast<int>(&p, &result);
         if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<int>;
         break;
#endif
      case 'B':
         G__asm_deref_cast<unsigned char>(&p, &result);
         if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<unsigned char>;
         break;
      case 'C':
         G__asm_deref_cast<char>(&p, &result);
         if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<char>;
         break;
      case 'R':
         G__asm_deref_cast<unsigned short>(&p, &result);
         if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<unsigned short>;
         break;
      case 'S':
         G__asm_deref_cast<short>(&p, &result);
         if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<short>;
         break;
      case 'H':
         G__asm_deref_cast<unsigned int>(&p, &result);
         if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<unsigned int>;
         break;
      case 'I':
         G__asm_deref_cast<int>(&p, &result);
         if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<int>;
         break;
      case 'K':
         G__asm_deref_cast<unsigned long>(&p, &result);
         if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<unsigned long>;
         break;
      case 'L':
         G__asm_deref_cast<long>(&p, &result);
         if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<long>;
         break;
      case 'F':
         G__asm_deref_cast<float>(&p, &result);
         if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<float>;
         break;
      case 'D':
         G__asm_deref_cast<double>(&p, &result);
         if (G__asm_noverflow) G__asm_inst[G__asm_cp-1] = (long)(dereff_t)G__asm_deref<double>;
         break;
      case 'U':
         result.obj.i = p.obj.i;
         if (G__asm_noverflow) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "       : assigned func: G__asm_tovalue_U  %s:%d\n", __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp-1] = (long) (dereff_t) ::Cint::Internal::G__asm_tovalue_U;
         }
         break;
      case 'u': {
         G__StrBuf refopr_sb(G__MAXNAME);
         char *refopr = refopr_sb;
         char * store_struct_offsetX = G__store_struct_offset;
         ::Reflex::Scope store_tagnumX = G__tagnum;
         int done = 0;
         G__store_struct_offset = (char*)p.obj.i;
#ifdef G__ASM
         if (G__asm_noverflow) {
            G__inc_cp_asm(-2, 0);
            G__asm_inst[G__asm_cp] = G__PUSHSTROS;
            G__asm_inst[G__asm_cp+1] = G__SETSTROS;
            G__inc_cp_asm(2, 0);
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "TOVALUE cancelled\n");
               G__fprinterr(G__serr, "%3x: PUSHSTROS\n", G__asm_cp - 2);
               G__fprinterr(G__serr, "%3x: SETSTROS\n", G__asm_cp - 1);
            }
#endif // G__ASM_DBG
            // --
         }
#endif // G__ASM
         G__set_G__tagnum(p);
         strcpy(refopr, "operator*()");
         result = G__getfunction(refopr, &done, G__TRYMEMFUNC);
         G__tagnum = store_tagnumX;
         G__store_struct_offset = store_struct_offsetX;
#ifdef G__ASM
         if (G__asm_noverflow) {
            G__asm_inst[G__asm_cp] = G__POPSTROS;
            G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x: POPSTROS\n", G__asm_cp - 1);
#endif
         }
#endif
         if (done) return(result);
         /* if 0==done, continue to default case for displaying error message */
      }
      default:
         /* if(0==G__no_exec_compile) */
         G__genericerror("Error: Illegal pointer operation (tovalue)");
         break;
   }
   G__value_typenum(result) = G__deref(G__value_typenum(p));
   result.ref = p.obj.i;
   return result;
}

//______________________________________________________________________________
G__value Cint::Internal::G__letVvalue(G__value *p, G__value result)
{
   // --
#ifdef G__ASM
   if (G__asm_noverflow) {
#ifdef G__ASM_DBG
      if (G__asm_dbg) G__fprinterr(G__serr, "%3x: LETVVAL\n", G__asm_cp);
#endif
      G__asm_inst[G__asm_cp] = G__LETVVAL;
      G__inc_cp_asm(1, 0);
   }
#endif /* G__ASM */

   if (p->ref) {
      p->obj.i = p->ref;
      p->ref = 0;
      /* if xxx *p;   (p)=xxx;  lvalue is pointer type then assign as long
       * else convert p to its' pointer type
       */
      if (G__value_typenum(*p).FinalType().IsPointer()) {
         G__value_typenum(*p) = G__get_from_type('L', 0); // p->type='L'; // FIXME: This is impossible!
      }
      else {
         G__value_typenum(*p) = ::Reflex::PointerBuilder(G__value_typenum(*p));  // p->type=toupper(p->type);
      }
      return(G__letvalue(p, result));
   }

   G__genericerror("Error: improper lvalue");
#ifdef G__ASM
#ifdef G__ASM_DBG
   if (G__asm_dbg && G__asm_noverflow)
      G__genericerror(G__LOOPCOMPILEABORT);
#endif
   G__abortbytecode();
#endif /* G__ASM */

   return(result);
}

//______________________________________________________________________________
G__value Cint::Internal::G__letPvalue(G__value *p, G__value result)
{
#ifdef G__ASM
   if (G__asm_noverflow) {
#ifdef G__ASM_DBG
      if (G__asm_dbg)
         G__fprinterr(G__serr, "%3x: LETPVAL\n", G__asm_cp);
#endif
      G__asm_inst[G__asm_cp] = G__LETPVAL;
      G__inc_cp_asm(1, 0);
   }
#endif /* G__ASM */

   return(G__letvalue(p, result));
}

//______________________________________________________________________________
G__value Cint::Internal::G__letvalue(G__value *p, G__value result)
{
   if (G__no_exec_compile) {
      if (G__value_typenum(*p) && !G__value_typenum(*p).RawType().IsEnum()) {
         switch (G__get_type(G__value_typenum(*p))) {
            case 'U':
               result = G__classassign((char*)p->obj.i, G__value_typenum(*p), result);
               break;
            case 'u': {
               G__value para;
               G__StrBuf refopr_sb(G__MAXNAME);
               char *refopr = refopr_sb;
               char *store_struct_offsetX = G__store_struct_offset;
               ::Reflex::Scope store_tagnumX = G__tagnum;
               int done = 0;
               int store_var_type = G__var_type;
               G__var_type = 'p';
#ifdef G__ASM
               if (G__asm_noverflow) {
                  if (G__LETPVAL == G__asm_inst[G__asm_cp-1] ||
                        G__LETVVAL == G__asm_inst[G__asm_cp-1]) {
#ifdef G__ASM_DBG
                     if (G__asm_dbg)
                        G__fprinterr(G__serr, "LETPVAL,LETVVAL cancelled\n");
#endif
                     G__inc_cp_asm(-1, 0);
                  }
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x: PUSHSTROS\n", G__asm_cp - 2);
                     G__fprinterr(G__serr, "%3x: SETSTROS\n", G__asm_cp - 1);
                  }
#endif
                  G__asm_inst[G__asm_cp] = G__PUSHSTROS;
                  G__asm_inst[G__asm_cp+1] = G__SETSTROS;
                  G__inc_cp_asm(2, 0);
               }
#endif
               G__store_struct_offset = (char*)p->obj.i;
               G__set_G__tagnum(*p);
               strcpy(refopr, "operator*()");
               para = G__getfunction(refopr, &done, G__TRYMEMFUNC);
               G__tagnum = store_tagnumX;
               G__store_struct_offset = store_struct_offsetX;
               G__var_type = store_var_type;
#ifdef G__ASM
               if (G__asm_noverflow) {
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x: POPSTROS\n", G__asm_cp - 2);
                  }
#endif
                  G__asm_inst[G__asm_cp] = G__POPSTROS;
                  G__inc_cp_asm(1, 0);
               }
               G__letVvalue(&para, result);
#endif
            }
            break;
         }
      }
      return(result);
   }

   if (G__value_typenum(*p) && G__value_typenum(*p).IsArray()) {
      char store_var_type = G__var_type;
      int size = G__Lsizeof(G__value_typenum(*p).Name(::Reflex::SCOPED).c_str());
      G__var_type = store_var_type;
      if ('C' == G__get_type(G__value_typenum(result)) && (int)strlen((char*)result.obj.i) < (int)size)
         size = strlen((char*)result.obj.i) + 1;
      memcpy((void*)p->obj.i, (void*)result.obj.i, size);
      return(result);
   }
   switch (G__get_reftype(G__value_typenum(*p))) {
      case G__PARAP2P:
      case G__PARAP2P2P:
         if (G__value_typenum(*p).FinalType().IsPointer()) {
            *(long *)(p->obj.i) = (long)G__int(result);
            return(result);
         }
   }
   switch (G__get_type(G__value_typenum(*p))) {
      case 'G':
      case 'B':
         *(unsigned char *)(p->obj.i) = (unsigned char)G__int(result);
         break;
      case 'C':
         *(char *)(p->obj.i) = (char)G__int(result);
         break;
      case 'R':
         *(unsigned short *)(p->obj.i) = (unsigned short)G__int(result);
         break;
      case 'S':
         *(short *)(p->obj.i) = (short)G__int(result);
         break;
      case 'H':
         *(unsigned int *)(p->obj.i) = (unsigned int)G__int(result);
         break;
      case 'I':
         *(int *)(p->obj.i) = (int)G__int(result);
         break;
      case 'K':
         *(unsigned long *)(p->obj.i) = (unsigned long)G__int(result);
         break;
      case 'L':
         *(long *)(p->obj.i) = (long)G__int(result);
         break;
      case 'F':
         *(float *)(p->obj.i) = (float)G__double(result);
         break;
      case 'D':
         *(double *)(p->obj.i) = (double)G__double(result);
         break;
      case 'U':
         result = G__classassign((char*)p->obj.i, G__value_typenum(*p), result);
         break;
      case 'u': {
         G__value para;
         G__StrBuf refopr_sb(G__MAXNAME);
         char *refopr = refopr_sb;
         char *store_struct_offsetX = G__store_struct_offset;
         ::Reflex::Scope store_tagnumX = G__tagnum;
         int done = 0;
         int store_var_type = G__var_type;
         G__var_type = 'p';
#ifdef G__ASM
         if (G__asm_noverflow) {
            if (G__LETPVAL == G__asm_inst[G__asm_cp-1] ||
                  G__LETVVAL == G__asm_inst[G__asm_cp-1]) {
#ifdef G__ASM_DBG
               if (G__asm_dbg) G__fprinterr(G__serr, "LETPVAL,LETVVAL cancelled\n");
#endif
               G__inc_cp_asm(-1, 0);
            }
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x: PUSHSTROS\n", G__asm_cp - 2);
               G__fprinterr(G__serr, "%3x: SETSTROS\n", G__asm_cp - 1);
            }
#endif
            G__asm_inst[G__asm_cp] = G__PUSHSTROS;
            G__asm_inst[G__asm_cp+1] = G__SETSTROS;
            G__inc_cp_asm(2, 0);
         }
#endif
         G__store_struct_offset = (char*)p->obj.i;
         G__set_G__tagnum(*p);
         strcpy(refopr, "operator*()");
         para = G__getfunction(refopr, &done, G__TRYMEMFUNC);
         G__tagnum = store_tagnumX;
         G__store_struct_offset = store_struct_offsetX;
         G__var_type = store_var_type;
#ifdef G__ASM
         if (G__asm_noverflow) {
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x: POPSTROS\n", G__asm_cp - 2);
            }
#endif
            G__asm_inst[G__asm_cp] = G__POPSTROS;
            G__inc_cp_asm(1, 0);
         }
         G__letVvalue(&para, result);
#endif
      }
      break;
      case 'c':
         memcpy((void*)p->ref, (void*)result.obj.i, strlen((char*)result.obj.i) + 1);
         break;
      default:
#ifdef G__ASM
#ifdef G__ASM_DBG
         if (G__asm_dbg && G__asm_noverflow)
            G__genericerror(G__LOOPCOMPILEABORT);
#endif
         G__abortbytecode();
#endif /* G__ASM */
         G__genericerror("Error: Illegal pointer operation (letvalue)");
         break;
   }

   return(result);
}

//______________________________________________________________________________
extern "C" void G__set_typenum(G__value* val, const char* type)
{
   ::Reflex::Type td = G__find_typedef(type);
   G__value_typenum(*val) = td;
   // val->type   = ::Cint::Internal::G__get_type(td);
}

//______________________________________________________________________________
extern "C" void G__set_type(G__value* val, char* type)
{
   ::Reflex::Type td = G__Dict::GetDict().GetScope(G__defined_tagname(type, 0));
   G__value_typenum(*val) = td;
   // val->type   = ::Cint::Internal::G__get_type(td);
}

//______________________________________________________________________________
extern "C" void G__set_tagnum(G__value* val, int tagnum)
{
   ::Reflex::Type td = G__Dict::GetDict().GetScope(tagnum);
   G__value_typenum(*val) = td;
   // val->type   = ::Cint::Internal::G__get_type(td);
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
