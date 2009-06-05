//* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file var.c
 ************************************************************************
 * Description:
 *  Variable initialization, assignment and referencing
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "Dict.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "Reflex/Builder/TypeBuilder.h"

using namespace Cint::Internal;

namespace Cint {
namespace Internal {
   static G__value G__allocvariable(G__value result, G__value para[], const ::Reflex::Scope& varglobal, const ::Reflex::Scope& varlocal, int paran, int /* varhash */, const char* item, std::string& varname, int parameter00, Reflex::Member &output_var);
static int G__asm_gen_stvar(long arg_G__struct_offset, const ::Reflex::Member& var, int paran, const char* item, long store_struct_offset, int var_type, G__value* presult);
} // namespace Internal
} // namespace Cint

static int G__getarraydim = 0;

//______________________________________________________________________________
//--  1
//--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
//--  9

//______________________________________________________________________________
//--  1
//--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
//--  9

//______________________________________________________________________________
//
// Static functions.
//

//______________________________________________________________________________
static void G__redecl(const ::Reflex::Member& var)
{
   // -- FIXME: Describe me!
   if (G__asm_noverflow) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: REDECL  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__REDECL;
      G__asm_inst[G__asm_cp+1] = (long) var.Id();
      G__asm_inst[G__asm_cp+2] = 0;
      G__inc_cp_asm(3, 0);
   }
}

//______________________________________________________________________________
static void G__class_2nd_decl(const ::Reflex::Member& var)
{
   // -- FIXME: Describe me!
   ::Reflex::Scope tagnum = var.TypeOf();
   int store_var_type = G__var_type;
   G__var_type = 'p';
   ::Reflex::Scope store_tagnum = G__tagnum;
   G__set_G__tagnum(tagnum);
   char* store_struct_offset = G__store_struct_offset;
   G__store_struct_offset = G__get_offset(var);
   char* store_globalvarpointer = G__globalvarpointer;
   G__globalvarpointer = G__PVOID;
   int store_cpp_aryconstruct = G__cpp_aryconstruct;
   if (G__get_varlabel(var.TypeOf(), 1) /* number of elements */ || G__get_paran(var)) {
      G__cpp_aryconstruct = G__get_varlabel(var.TypeOf(), 1) /* number of elements */;
   }
   else {
      G__cpp_aryconstruct = 0;
   }
   int store_decl = G__decl;
   G__decl = 0;
   G__StrBuf temp_sb(G__ONELINE);
   char *temp = temp_sb;
   sprintf(temp, "~%s()", tagnum.Name_c_str());
   if (G__dispsource) {
      G__fprinterr(G__serr, "\n!!!Calling destructor 0x%lx.%s for declaration of %s", G__store_struct_offset, temp, var.Name(::Reflex::SCOPED).c_str());
   }
   if (G__struct.iscpplink[G__get_tagnum(tagnum)] == G__CPPLINK) {
      // Delete current object.
      if (G__get_offset(var)) {
         int known = 0;
         G__getfunction(temp, &known, G__TRYDESTRUCTOR);
      }
      // Set newly constructed object.
      G__get_offset(var) = store_globalvarpointer;
      if (G__dispsource) {
         G__fprinterr(G__serr, " 0x%lx is set", store_globalvarpointer);
      }
   }
   else {
      if (G__cpp_aryconstruct) {
         for (int i = G__cpp_aryconstruct - 1; i >= 0; --i) {
            // Call destructor without freeing memory.
            G__store_struct_offset = G__get_offset(var) + (i * G__struct.size[G__get_tagnum(tagnum)]);
            int known = 0;
            if (G__get_offset(var)) {
               G__getfunction(temp, &known, G__TRYDESTRUCTOR);
            }
            if ((G__return > G__RETURN_NORMAL) || !known) {
               break;
            }
         }
      }
      else {
         G__store_struct_offset = G__get_offset(var);
         if (G__get_offset(var)) {
            int known = 0;
            G__getfunction(temp, &known, G__TRYDESTRUCTOR);
         }
      }
   }
   G__decl = store_decl;
   G__ASSERT(!G__decl || (G__decl == 1));
   G__cpp_aryconstruct = store_cpp_aryconstruct;
   G__globalvarpointer = store_globalvarpointer;
   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;
   G__var_type = store_var_type;
}

//______________________________________________________________________________
static void G__class_2nd_decl_i(const ::Reflex::Member& var)
{
   // -- FIXME: Describe me!
   int store_no_exec_compile = G__no_exec_compile;
   G__no_exec_compile = 1;
   ::Reflex::Scope store_tagnum = G__tagnum;
   G__set_G__tagnum(var.TypeOf());
   char* store_struct_offset = G__store_struct_offset;
   char* store_globalvarpointer = G__globalvarpointer;
   G__globalvarpointer = G__PVOID;
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "%3x,%3x: LD_VAR  %s paran=%d  %s:%d\n", G__asm_cp, G__asm_dt, var.Name(::Reflex::SCOPED).c_str(), 0, __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
   G__asm_inst[G__asm_cp] = G__LD_VAR;
   G__asm_inst[G__asm_cp+1] = 0; // ig15;
   G__asm_inst[G__asm_cp+2] = 0;
   G__asm_inst[G__asm_cp+3] = 'p';
   G__asm_inst[G__asm_cp+4] = (long) var.Id();
   G__inc_cp_asm(5, 0);
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
   G__asm_inst[G__asm_cp] = G__PUSHSTROS;
   G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
   G__asm_inst[G__asm_cp] = G__SETSTROS;
   G__inc_cp_asm(1, 0);
   G__StrBuf temp_sb(G__ONELINE);
   char *temp = temp_sb;
   sprintf(temp, "~%s()", G__tagnum.Name_c_str());
   if (G__get_varlabel(var.TypeOf(), 1) /* number of elements */ || G__get_paran(var)) {
      // array
      int size = G__struct.size[G__get_tagnum(G__tagnum)];
      int pinc = G__get_varlabel(var.TypeOf(), 1) /* number of elements */;
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: ADDSTROS %d  %s:%d\n", G__asm_cp, G__asm_dt, -size * pinc, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__ADDSTROS;
      G__asm_inst[G__asm_cp+1] = (long) (pinc * size);
      G__inc_cp_asm(2, 0);
      for (int i = pinc - 1; i >= 0; --i) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: ADDSTROS %d  %s:%d\n", G__asm_cp, G__asm_dt, -size, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__ADDSTROS;
         G__asm_inst[G__asm_cp+1] = (long) (-size);
         G__inc_cp_asm(2, 0);
         int known = 0;
         G__getfunction(temp, &known, G__TRYDESTRUCTOR);
      }
   }
   else {
      int known = 0;
      G__getfunction(temp, &known, G__TRYDESTRUCTOR);
   }
   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;
   G__no_exec_compile = store_no_exec_compile;
   G__globalvarpointer = store_globalvarpointer;
}

//______________________________________________________________________________
static void G__class_2nd_decl_c(const ::Reflex::Member& var)
{
   // -- FIXME: Describe me!
   char* store_globalvarpointer = G__globalvarpointer;
   G__globalvarpointer = G__PVOID;
   int store_no_exec_compile = G__no_exec_compile;
   G__no_exec_compile = 1;
   ::Reflex::Scope store_tagnum = G__tagnum;
   G__set_G__tagnum(var.TypeOf());
   char* store_struct_offset = G__store_struct_offset;
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "%3x,%3x: LD_VAR  %s paran=%d  %s:%d\n", G__asm_cp, G__asm_dt, var.Name(::Reflex::SCOPED).c_str(), 0, __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
   G__asm_inst[G__asm_cp] = G__LD_VAR;
   G__asm_inst[G__asm_cp+1] = 0; // ig15;
   G__asm_inst[G__asm_cp+2] = 0;
   G__asm_inst[G__asm_cp+3] = 'p';
   G__asm_inst[G__asm_cp+4] = (long) var.Id();
   G__inc_cp_asm(5, 0);
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
   G__asm_inst[G__asm_cp] = G__PUSHSTROS;
   G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
   G__asm_inst[G__asm_cp] = G__SETSTROS;
   G__inc_cp_asm(1, 0);
   G__StrBuf temp_sb(G__ONELINE);
   char *temp = temp_sb;
   sprintf(temp, "~%s()", G__tagnum.Name_c_str());
   int known = 0;
   G__getfunction(temp, &known, G__TRYDESTRUCTOR);
   G__redecl(var);
   if (store_no_exec_compile) {
      G__abortbytecode();
   }
   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;
   G__no_exec_compile = store_no_exec_compile;
   G__globalvarpointer = store_globalvarpointer;
}

//______________________________________________________________________________
static void G__getpointer2pointer(G__value* presult, const ::Reflex::Member& var, int paran)
{
   // -- FIXME: Describe me!
   switch (G__var_type) {
      case 'v':
         switch (G__get_reftype(var.TypeOf())) {
            case G__PARANORMAL:
               if (G__get_paran(var) > paran) {
                  if (tolower(G__get_type(var.TypeOf())) != 'u') {
                     G__letint(presult, G__get_type(var.TypeOf()), *((long*) presult->ref));
                  } else {
                     ::Reflex::Type ty = var.TypeOf().FinalType();
                     for (; ty.IsArray(); ty = ty.ToType()) {}
                     G__letpointer(presult, *((long*) presult->ref), ty);
                  }
               }
               break;
            case G__PARAREFERENCE:
               break;
            case G__PARAP2P:
               if (tolower(G__get_type(var.TypeOf())) != 'u') {
                  G__letint(presult, G__get_type(var.TypeOf()), *(long*)presult->ref);
               } else {
                  ::Reflex::Type ty = var.TypeOf().FinalType();
                  for (; ty.IsArray(); ty = ty.ToType()) {}
                  G__letpointer(presult, *(long*)presult->ref, G__deref(ty));
               }
               break;
            case G__PARAP2P2P:
               if (tolower(G__get_type(var.TypeOf())) != 'u') {
                  G__letint(presult, G__get_type(var.TypeOf()), *(long*)presult->ref);
               } else {
                  ::Reflex::Type ty = var.TypeOf().FinalType();
                  for (; ty.IsArray(); ty = ty.ToType()) {}
                  G__letpointer(presult, *(long*)presult->ref, G__deref(ty));
               }
               break;
            default:
               {
                  ::Reflex::Type ty = var.TypeOf().FinalType();
                  int var_ptr_count = 0;
                  for (; ty.IsPointer(); ty = ty.ToType()) {
                     ++var_ptr_count;
                  }
                  if (tolower(G__get_type(var.TypeOf())) != 'u') {
                     G__letint(presult, G__get_type(var.TypeOf()), *(long*)presult->ref);
                     for (int i = 0; i < (var_ptr_count - 2); ++i) {
                        G__value_typenum(*presult) = ::Reflex::PointerBuilder(G__value_typenum(*presult));
                     }
                  } else {
                     G__letpointer(presult, *(long*)presult->ref, var.TypeOf().RawType());
                     for (int i = 0; i < (var_ptr_count - 1); ++i) {
                        G__value_typenum(*presult) = ::Reflex::PointerBuilder(G__value_typenum(*presult));
                     }
                  }
               }
               break;
         }
         break;
      case 'p':
         if (paran < G__get_paran(var)) {
            switch (G__get_reftype(var.TypeOf())) {
               case G__PARANORMAL:
                  G__value_typenum(*presult) = ::Reflex::PointerBuilder(G__value_typenum(*presult));
                  break;
               case G__PARAP2P:
               default:
                  G__value_typenum(*presult) = ::Reflex::PointerBuilder(G__value_typenum(*presult));
                  break;
            }
            int extra_ptr_count = G__get_paran(var) - paran - 1;
            for (int i = 0; i < extra_ptr_count; ++i) {
               G__value_typenum(*presult) = ::Reflex::PointerBuilder(G__value_typenum(*presult));
            }
         }
         else if (paran == G__get_paran(var)) {
            ::Reflex::Type ty = var.TypeOf().FinalType();
            for (; ty.IsArray(); ty = ty.ToType()) {}
            int var_ptr_count = 0;
            for (; ty.IsPointer(); ty = ty.ToType()) {
               ++var_ptr_count;
            }
            ::Reflex::Type result_type(G__value_typenum(*presult).RawType(),ty.IsConst() ? Reflex::CONST : 0); // Try to preserve some constness!
            for (int i = 0; i < var_ptr_count; ++i) {
               result_type = ::Reflex::PointerBuilder(result_type);
            }
            G__value_typenum(*presult) = result_type;
         }
         break;
      case 'P':
         /* this part is not precise. Should handle like above 'p' case */
         if (G__get_paran(var) == paran) { /* must be PPTYPE */
            switch (G__get_reftype(var.TypeOf())) {
               case G__PARANORMAL:
                  G__value_typenum(*presult) = ::Reflex::PointerBuilder(G__value_typenum(*presult));
                  break;
               case G__PARAP2P:
                  G__value_typenum(*presult) = ::Reflex::PointerBuilder(G__value_typenum(*presult));
                  break;
               default:
                  ::Reflex::Type ty = var.TypeOf().FinalType();
                  for (; ty.IsArray(); ty = ty.ToType()) {}
                  int var_ptr_count = 0;
                  for (; ty.IsPointer(); ty = ty.ToType()) {
                     ++var_ptr_count;
                  }
                  ::Reflex::Type result_type(G__value_typenum(*presult).RawType(), ty.IsConst() ? Reflex::CONST : 0);
                  for (int i = 0; i < (var_ptr_count + 1); ++i) {
                     result_type = ::Reflex::PointerBuilder(result_type);
                  }
                  G__value_typenum(*presult) = result_type;
                  break;
            }
         }
         break;
      default:
         break;
   }
}

//______________________________________________________________________________
//
// Internal functions.
//

//______________________________________________________________________________
//
// -- Change a variable's value.
//

template<class CASTTYPE, class CONVFUNC, class XTYPE>
inline G__value G__assign_var(::Reflex::Member& var, char* local_G__struct_offset, int& paran, int linear_index, G__value& result, const char* item, int SIZE, CONVFUNC f, XTYPE& X)
{
   switch (G__var_type) {
      case 'v':
         // Assign by dereferencing a pointer.  *var = result;
         if (((paran + 1) == G__get_paran(var)) && !linear_index && islower(G__get_type(G__value_typenum(result)))) {
            // Pointer to array reimplementation.
            ++paran;
            // Fall through to case 'p'.
         }
         else {
            G__assign_error(item, &result);
            break;
         }
      case 'p':
         // Assign result to variable.
         if (paran >= G__get_paran(var)) {
            // MyType var[ddd]; var[xxx] = result;
            // FIXME: The greater than case requires pointer arithmetic.
            result.ref = ((long) local_G__struct_offset) + ((long) G__get_offset(var)) + (linear_index * SIZE);
            *((CASTTYPE*) result.ref) = (CASTTYPE) (*f)(result);
            G__value_typenum(result) = var.TypeOf();
            X = *((CASTTYPE*) result.ref);
            break;
         }
         G__assign_error(item, &result);
         break;
      default:
         G__assign_error(item, &result);
         break;
   }
   G__var_type = 'p';
   return result;
}

#define G__ASSIGN_VAR(SIZE, CASTTYPE, CONVFUNC, X) \
   switch (G__var_type) { \
      case 'v': \
         /* Assign by dereferencing a pointer.  *output_var = result; */ \
         if (((paran + 1) == G__get_paran(output_var)) && !linear_index && islower(G__get_type(G__value_typenum(result)))) { \
            /* Pointer to array reimplementation. */ \
            ++paran; \
            /* Fall through to case 'p'. */ \
         } \
         else { \
            G__assign_error(item, &result); \
            break; \
         } \
      case 'p': \
         /* Assign result to variable. */ \
         if (paran >= G__get_paran(output_var)) { \
            /* MyType var[ddd]; var[xxx] = result; */ \
            /* FIXME: The greater than case requires pointer arithmetic. */ \
            result.ref = ((long) local_G__struct_offset) + ((long) G__get_offset(output_var)) + (linear_index * SIZE); \
            *((CASTTYPE*) result.ref) = (CASTTYPE) CONVFUNC(result); \
            G__value_typenum(result) = output_var.TypeOf(); \
            /**/ \
            X = *((CASTTYPE*) result.ref); \
            break; \
         } \
         G__assign_error(item, &result); \
         break; \
      default: \
         G__assign_error(item, &result); \
         break; \
   } \
   G__var_type = 'p'; \
   /**/ \
   /**/ \
   /**/ \
   return result;

//______________________________________________________________________________
//
// -- Change a pointer variable's value.
//
#define G__ASSIGN_PVAR(CASTTYPE, CONVFUNC, X) \
   switch (G__var_type) { \
      case 'v': \
         /* -- Assign to what pointer is pointing at. */ \
         switch (G__get_reftype(output_var.TypeOf())) { \
            case G__PARANORMAL: \
               /* -- Assignment through a pointer dereference.  MyType* output_var; *output_var = result; */ \
               result.ref = *((long*) (local_G__struct_offset + ((long) G__get_offset(output_var)) + (linear_index * G__LONGALLOC))); \
               *((CASTTYPE*) result.ref) = (CASTTYPE) CONVFUNC(result); \
               G__value_typenum(result) = G__deref(output_var.TypeOf()); /* FIXME: We want to remove the top-level pointer node here, but not all the typedef info! */ \
               X = *((CASTTYPE*) result.ref); \
               break; \
            case G__PARAP2P: \
               /* -- Assignment through a pointer to a pointer dereference. */ \
               if (paran > G__get_paran(output_var)) { \
                  /* -- Pointer to array reimplementation. */ \
                  /* MyType** output_var[ddd]; *output_var[xxx][yyy] = result; */ \
                  char *address = local_G__struct_offset + ((long) G__get_offset(output_var)) + (linear_index * G__LONGALLOC); \
                  result.ref = *(((long*) (*((long *) address))) + secondary_linear_index); \
                  *((CASTTYPE*) result.ref) = (CASTTYPE) CONVFUNC(result); \
                  G__value_typenum(result) = G__deref(G__strip_array(output_var.TypeOf())); /* FIXME: We want to remove the top-level pointer node here, but not all the typedef info! */ \
                  X = *((CASTTYPE*) result.ref); \
               } \
               else { \
                  /* paran <= G__get_paran(output_var) */ \
                  /* MyType** var[ddd][nnn]; *var[xxx][yyy] = result; */ \
                  result.ref = *((long*) (local_G__struct_offset + ((long) G__get_offset(output_var)) + (linear_index * G__LONGALLOC))); \
                  *((long*) result.ref) = G__int(result); \
               } \
               break; \
         } \
         break; \
      case 'p': \
         /* Assign to the pointer variable itself, no dereferencing. */ \
         if (paran > G__get_paran(output_var)) { \
            /* -- Pointer to array reimplementation. */ \
            /* -- More array dimensions used than variable has, start using up pointer to pointers. */ \
            char *address = local_G__struct_offset + ((long) G__get_offset(output_var)) + (linear_index * G__LONGALLOC); \
            if (G__get_reftype(output_var.TypeOf()) == G__PARANORMAL) { \
               result.ref = (long) (((CASTTYPE*) (*((long*) address))) + secondary_linear_index); \
               *((CASTTYPE*) result.ref) = (CASTTYPE) CONVFUNC(result); \
               G__value_typenum(result) = G__deref(G__strip_array(output_var.TypeOf())); /* FIXME: We want to remove the top-level pointer node here, but not all the typedef info! */ \
               X = *((CASTTYPE*) result.ref); \
            } \
            else if (G__get_paran(output_var) == (paran - 1)) { \
               /* -- One extra dimension. */ \
               result.ref = (long) (((long*) (*((long*) address))) + secondary_linear_index); \
               *((long*) result.ref) = G__int(result); \
            } \
            else { \
               /* -- Two or more extra dimensions. */ \
               long* phyaddress = (long*) (*((long*) address)); \
               for (int ip = G__get_paran(output_var); ip < (paran - 1); ++ip) { \
                  phyaddress = (long*) phyaddress[para[ip].obj.i]; \
               } \
               switch (G__get_reftype(output_var.TypeOf()) - paran + G__get_paran(output_var)) { \
                  case G__PARANORMAL: \
                     ((CASTTYPE*) phyaddress)[para[paran-1].obj.i] = (CASTTYPE) CONVFUNC(result); \
                     break; \
                  default: \
                     ((long*) phyaddress)[para[paran-1].obj.i] = G__int(result); \
                     break; \
               } \
            } \
         } \
         else { \
            /* Same number or less than of array dimensions used as variable has, */ \
            /* assign to array element (which is a pointer). */ \
            /* MyType* var[ddd][nnn]; *var[xxx] = result; */ \
            /* MyType* var[ddd][nnn]; var[xxx][yyy] = result; */ \
            result.ref = ((long) local_G__struct_offset) + ((long) G__get_offset(output_var)) + (linear_index * G__LONGALLOC); \
            *((long*) result.ref) = G__int(result); \
         } \
         break; \
      default: \
         G__assign_error(item, &result); \
         break; \
   }

//______________________________________________________________________________
G__value Cint::Internal::G__letvariable(const char* item, G__value expression, const ::Reflex::Scope varglobal, const ::Reflex::Scope varlocal)
{
   ::Reflex::Member dummy;
   return G__letvariable(item, expression, varglobal, varlocal, dummy);
}

//______________________________________________________________________________
G__value Cint::Internal::G__letvariable(const char* item, G__value expression, const ::Reflex::Scope varglobal, const ::Reflex::Scope varlocal, ::Reflex::Member& output_var)
{
   // -- FIXME: Describe me!

   //--
   int paran = 0;
   int ig25 = 0;
   int lenitem = 0;
   int paren = 0;
   int done = 0;
   //--
   //--
   int ig2 = 0;
   int flag = 0;
   int store_var_type = 0;
   char* local_G__struct_offset = 0;
   char* tagname = 0;
   char* membername = 0;
   int varhash = 0;
   char* store_struct_offset = 0;
   ::Reflex::Scope store_tagnum;
   int store_exec_memberfunc = 0;
   int store_def_struct_member = 0;
   int store_vartype = 0;
   int store_asm_wholefunction = 0;
   int store_no_exec_compile = 0;
   int store_no_exec = 0;
   int store_getarraydim = 0;
   int store_asm_noverflow = 0;
   G__StrBuf tmp_sb(G__ONELINE);
   G__StrBuf result7_sb(G__ONELINE);
   char *result7 = result7_sb;
   G__StrBuf parameter_sb(G__MAXVARDIM * G__ONELINE);
   typedef char parameterarr_t[G__ONELINE];
   parameterarr_t *parameter = (parameterarr_t*)parameter_sb.data();
   G__StrBuf para_sb(G__MAXVARDIM * sizeof(G__value));
   G__value *para = (G__value*) para_sb.data();
   std::string varname;
   ::Reflex::Scope varscope;
   //--
   G__value result = G__null;

#ifdef G__ASM
   if (G__asm_exec) {
      output_var = G__asm_index;
      paran = G__asm_param->paran;
      for (int i = 0; i < paran; ++i) {
         para[i] = G__asm_param->para[i];
      }
      para[paran] = G__null;
      varscope = varglobal;
      if (!varlocal) {
         local_G__struct_offset = 0;
      }
      else {
         local_G__struct_offset = G__store_struct_offset;
      }
      result = expression;
      goto exec_asm_letvar;
   }
#endif // G__ASM
   parameter[0][0] = '\0';
   lenitem = std::strlen(item);
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   switch (item[0]) {
      case '*':
         // value of pointer
         if (
            (item[1] == '(') ||
            (item[1] == '+') ||
            (item[1] == '-') ||
            (item[lenitem-1] == '+') ||
            (item[lenitem-1] == '-') ||
            ((item[1] == '*') && !G__decl)
         ) {
            result = G__getexpr(item + 1);
            G__ASSERT(G__value_typenum(result).FinalType().IsPointer() || (G__get_type(G__value_typenum(result)) == 'u'));
            G__value tmp = G__letPvalue(&result, expression);



            return tmp;
         }
         {
            int pointlevel = 0;
            int i = 0;
            if (isupper(G__var_type)) {
               ++pointlevel;
            }
            while ((item[i++] == '*')) {
               ++pointlevel;
            }
            switch (pointlevel) {
               case 0:
                  break;
               case 1:
                  G__reftype = G__PARANORMAL;
                  break;
               default:
                  G__reftype = G__PARAP2P + pointlevel - 2;
                  break;
            }
            item = item + i - 1;
            if (G__var_type == 'p') {
               G__var_type = 'v';
            }
            else {
               G__var_type = toupper(G__var_type);
            }
         }
         break;
      case '(':
         {
            // For example: (xxx) = xxx;
            //          or: (xxx) xxx = xxx;
            int ig15 = 0;
            result = G__getfunction(item, &ig15, G__TRYNORMAL);
            if (G__value_typenum(result).IsConst()) {
               G__changeconsterror(item, "ignored const");



               return result;
            }
            G__value tmp = G__letVvalue(&result, expression);



            return tmp;
         }
      case '&':
         // -- Should not happen!
         G__var_type = 'P';
         item = item + 1;
         break;
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
      case '.':
      case '-':
      case '+':
         G__fprinterr(G__serr, "Error: assignment to %s", item);
         G__genericerror(0);
         break;
   }
   store_var_type = G__var_type;
   G__var_type = 'p';
   // struct, union member
   lenitem = std::strlen(item);
   ig2 = 0;
   flag = 0;
   {
      // To keep track of the quote variables
      int single_quote = 0;
      int double_quote = 0;

      while ((ig2 < lenitem) && !flag) {
         switch (item[ig2]) {
            case '.':
               if (!paren && !double_quote && !single_quote) {
                  strcpy(result7, item);
                  result7[ig2++] = '\0';
                  tagname = result7;
                  membername = result7 + ig2;
                  flag = 1;
               }
               break;
            case '-':
               if (!paren && !double_quote && !single_quote && (item[ig2+1] == '>')) {
                  strcpy(result7, item);
                  result7[ig2++] = '\0';
                  result7[ig2++] = '\0';
                  tagname = result7;
                  membername = result7 + ig2;
                  flag = 2;
               }
               break;
            case '\\':
               ++ig2;
               break;
            case '\'':
               if (!double_quote) {
                  single_quote ^= 1;
               }
               break;
            case '\"':
               if (!single_quote) {
                  double_quote ^= 1;
               }
               break;
            case '{':
            case '[':
            case '(':
               if (!single_quote && !double_quote) {
                  ++paren;
               }
               break;
            case '}':
            case ']':
            case ')':
               if (!single_quote && !double_quote) {
                  --paren;
               }
               break;
         }
         ++ig2;
      }
   }
   paren = 0;
   if (flag) {
      result = G__letstructmem(store_var_type, varname.c_str(), membername, tagname, varglobal, expression, flag, output_var);



      return result;
   }
   /************************************************************
    * !varglobal means, G__letvariable() is called from
    * G__getvariable() or G__letvariable() and G__store_struct_offset
    * is set by parent G__getvariable().
    *  This is done in different manner in case of loop
    * compilation execution.
    ************************************************************/
   if (!varglobal) {
      local_G__struct_offset = G__store_struct_offset;
   }
   else {
      local_G__struct_offset = 0;
   }
   result = expression;
   // Parse out an identifier, and possibly handle
   // a function call or array indexes.
   {
      // Start at the beginning.
      int item_cursor = 0;
      // Collect the identifier and the hash.
      // Note: We stop at a function parameter list or at array indexes.
      varhash = 0;
      char c = '\0';
      while (item_cursor < lenitem) {
         c = item[item_cursor];
         if ((c == '(') || (c == '[')) {
            // -- The identifier is terminated by function parameters or array indexes.
            break;
         }
         varname.push_back(c);
         varhash += c;
         ++item_cursor;
      }
      // Check if we have a function call.
      if ((c == '(') || !item_cursor) {
         // -- We have parsed a function call, let G__getfunction handle it and return.
         // Note: We also support an empty item string.  FIXME: Why?
         // For example: a.sub(50, 20) = b;
         int found = 0;
         if (!varglobal) {
            para[0] = G__getfunction(item, &found, G__CALLMEMFUNC);
         }
         else {
            para[0] = G__getfunction(item, &found, G__TRYNORMAL);
         }



         if (found) {
            para[1] = G__letVvalue(&para[0], expression);
            return para[1];
         }
         return G__null;
      }

      // Get any specified array indexes.
      // FIXME: Why do we allow curly braces here?
      paran = 0;
      if (item_cursor < lenitem) {
         // -- Move past the initial left square brace '['.
         ++item_cursor;
      }
      while (item_cursor < lenitem) {
         int idx = 0;
         int nest = 0;
         int single_quote = 0;
         int double_quote = 0;
         while (item_cursor < lenitem) {
            char peek = item[item_cursor];
            if ((peek == ']') && !nest && !single_quote && !double_quote) {
               break;
            }
            switch (peek) {
               case '"':
                  if (!single_quote) {
                     double_quote ^= 1;
                  }
                  break;
               case '\'':
                  if (!double_quote) {
                     single_quote ^= 1;
                  }
                  break;
               case '(':
               case '[':
               case '{':
                  if (!double_quote && !single_quote) {
                     ++nest;
                  }
                  break;
               case ')':
               case ']':
               case '}':
                  if (!double_quote && !single_quote) {
                     --nest;
                  }
                  break;
            }
            parameter[paran][idx++] = peek;
            ++item_cursor;
         }
         // Skip the terminating square bracket.
         ++item_cursor;
         if (
            (item_cursor < lenitem) &&
            (item[item_cursor] == '[')
         ) {
            // -- Skip the opening square bracket of the next index expression.
            ++item_cursor;
         }
         parameter[paran++][idx] = '\0';
         parameter[paran][0] = '\0';
      }
   }
   // Restore base environment.
   store_struct_offset = G__store_struct_offset;
   store_tagnum = G__tagnum;
   store_exec_memberfunc = G__exec_memberfunc;
   store_def_struct_member = G__def_struct_member;
   store_vartype = G__var_type;
   if (G__def_tagnum && !G__def_tagnum.IsTopScope() && G__def_struct_member) {
      G__tagnum = G__def_tagnum;
      G__store_struct_offset = 0;
      G__exec_memberfunc = 1;
      G__def_struct_member = 0;
   }
   else {
      // --
#ifdef G__ASM
      if (
         G__asm_noverflow &&
         paran &&
         ((G__store_struct_offset != G__memberfunc_struct_offset) || G__do_setmemfuncenv)
      ) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: SETMEMFUNCENV  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__SETMEMFUNCENV;
         G__inc_cp_asm(1, 0);
      }
#endif // G__ASM
      G__tagnum = G__memberfunc_tagnum;
      G__store_struct_offset = G__memberfunc_struct_offset;
      G__var_type = 'p';
   }
   store_getarraydim = G__getarraydim;
   store_asm_wholefunction = G__asm_wholefunction;
   store_no_exec_compile = G__no_exec_compile;
   store_no_exec = G__no_exec;
   store_asm_noverflow = G__asm_noverflow;
   if (G__decl) {
      G__getarraydim = 1;
      if (G__asm_wholefunction) {
         G__asm_wholefunction = 0;
         G__no_exec_compile = 0;
         G__no_exec = 0;
         G__asm_noverflow = 0;
      }
   }
   if (G__cppconstruct) {
      G__asm_noverflow = 0;
   }
   // Evaluate array index expressions and store the results.
   for (int i = 0; i < paran; ++i) {
      para[i] = G__getexpr(parameter[i]);
   }
   G__getarraydim = store_getarraydim;
   G__asm_wholefunction = store_asm_wholefunction;
   G__no_exec_compile = store_no_exec_compile;
   G__no_exec = store_no_exec;
   G__asm_noverflow = store_asm_noverflow;
   // Recover function call environment.
#ifdef G__ASM
   if (G__asm_noverflow && paran && ((G__store_struct_offset != store_struct_offset) || G__do_setmemfuncenv)) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: RECMEMFUNCENV  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__RECMEMFUNCENV;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;
   G__exec_memberfunc = store_exec_memberfunc;
   G__def_struct_member = store_def_struct_member;
   G__var_type = store_vartype;
   G__var_type = store_var_type;
   //
   // Search local and global variables.
   //
   output_var = ::Reflex::Member();
   if (!G__in_memvar_setup) {
      // Avoid searching variables when processing
      // a function-local const static during prerun.
      if (
         !G__func_now ||       // Not in a function declaration/definition.
         !G__decl ||           // Not in a declaration.
         !G__static_alloc ||   // Not a static/const/enumerator declaration.
         !G__constvar ||       // Not a const.
         !G__prerun            // Not in prerun (we are actually executing).
      ) {
         int ig15 = 0;
         output_var = G__find_variable(varname.c_str(), varhash, varlocal, varglobal, &local_G__struct_offset, &store_struct_offset, &ig15, G__decl || G__def_struct_member);
      }
   }
   //
   // Assign value.
   //
   if (output_var) {
      // -- We have found a variable.
      //
      //  Block duplicate declaration.
      //
      G__ASSERT(!G__decl || (G__decl == 1));
      if (
         (G__decl || G__cppconstruct) &&
         (G__var_type != 'p') &&
         (G__get_properties(output_var)->statictype == G__AUTO) &&
         (
            (G__get_type(output_var.TypeOf()) != G__var_type) ||
            (output_var.TypeOf().RawType().IsClass() && (output_var.TypeOf().RawType() != G__tagnum))
         )
      ) {
         G__fprinterr(G__serr, "Error: %s already declared as different type", item);
         if (
            isupper(G__get_type(output_var.TypeOf())) &&
            isupper(G__var_type) &&
            !G__get_varlabel(output_var.TypeOf(), 1) /* number of elements */ &&
            (*((long*) G__get_offset(output_var)) == 0)
         ) {
            G__fprinterr(G__serr, ". Switch to new type\n");
            //output_var->type[ig15] = G__var_type;  FIXME: Translate this!
            //output_var->p_tagtable[ig15] = G__tagnum;
            //output_var->p_typetable[ig15] = G__typenum;
         }
         else {
            if (
               (G__globalvarpointer != G__PVOID) &&
               (G__var_type == 'u') &&
               G__tagnum &&
               (G__struct.iscpplink[G__get_tagnum(G__tagnum)] == G__CPPLINK)
            ) {
               G__StrBuf protect_temp_sb(G__ONELINE);
               char *protect_temp = protect_temp_sb;
               char* protect_struct_offset = G__store_struct_offset;
               int local_done = 0;
               G__store_struct_offset = G__globalvarpointer;
               G__globalvarpointer = G__PVOID;
               std::sprintf(protect_temp, "~%s()", G__tagnum.Name_c_str());
               G__fprinterr(G__serr, ". %s called\n", protect_temp);
               G__getfunction(protect_temp, &local_done, G__TRYDESTRUCTOR);
               G__store_struct_offset = protect_struct_offset;
            }
            G__genericerror(0);



            return G__null;
         }
      } else if ((G__get_properties(output_var)->statictype == G__LOCALSTATIC) && (!output_var.DeclaringScope().IsNamespace()) && (G__get_type(output_var.TypeOf()) == 'u')) {
         //fprintf(stderr,"humm .. declaration of static variable %s\n",output_var->varnamebuf[ig15]);
         // Let's assume this is the first definition of the class static variable (CINT currently allows the
         // declaration to be there several times  ...
         // First delete the memory allocated at the time of the declaration (inside the class declaration)
         free(G__get_offset(output_var));
         G__get_offset(output_var) = 0;
         // And let's allocate the object (currently CINT does not allow a constructor in this case).
         // (this is inpired from code in G__define_var
         int vtagnum = G__get_tagnum( output_var.TypeOf() );
         if ( G__struct.iscpplink[vtagnum] == G__CPPLINK) {
            // -- The struct is compiled code.
            char temp1[G__ONELINE];
            G__value reg = G__null;
            int known;
            sprintf(temp1, "%s()", G__struct.name[vtagnum]);
            if (G__struct.parent_tagnum[vtagnum] != -1) {
               int local_store_exec_memberfunc = G__exec_memberfunc;
               ::Reflex::Scope store_memberfunc_tagnum = G__memberfunc_tagnum;
               G__exec_memberfunc = 1;
               G__memberfunc_tagnum = output_var.DeclaringScope();
               reg = G__getfunction(temp1, &known, G__CALLCONSTRUCTOR);
               G__exec_memberfunc = local_store_exec_memberfunc;
               G__memberfunc_tagnum = store_memberfunc_tagnum;
            }
            else {
               int local_store_exec_memberfunc = G__exec_memberfunc;
               ::Reflex::Scope store_memberfunc_tagnum = G__memberfunc_tagnum;
               ::Reflex::Scope store_G__tagnum = G__tagnum;
               G__exec_memberfunc = 0;
               G__memberfunc_tagnum = ::Reflex::Scope();
               G__tagnum = output_var.TypeOf().RawType();
               reg = G__getfunction(temp1, &known, G__CALLCONSTRUCTOR);
               G__exec_memberfunc = local_store_exec_memberfunc;
               G__memberfunc_tagnum = store_memberfunc_tagnum;
               G__tagnum = store_G__tagnum;
            }
            G__get_offset(output_var)  = (char*)G__int(reg);
         }
         else {
            // -- The struct is interpreted.
            // Initialize it.
            char temp1[G__ONELINE];
            // G__value reg = G__null;
            // int known;
            sprintf(temp1, "new %s", G__struct.name[vtagnum]);
            
            int local_store_exec_memberfunc = G__exec_memberfunc;
            ::Reflex::Scope store_memberfunc_tagnum = G__memberfunc_tagnum;
            ::Reflex::Scope local_store_tagnum = G__tagnum;
            int store_prerun = G__prerun;
            int local_store_vartype = G__var_type;
            G__exec_memberfunc = 0;
            G__memberfunc_tagnum = ::Reflex::Scope();
            G__prerun = 0;
            G__tagnum = output_var.TypeOf().RawType();
            G__value reg = G__getexpr(temp1);
            G__exec_memberfunc = local_store_exec_memberfunc;
            G__memberfunc_tagnum = store_memberfunc_tagnum;
            G__tagnum = local_store_tagnum;
            G__prerun = store_prerun;
            G__var_type = local_store_vartype;
            
            G__get_offset(output_var)  = (char*) G__int(reg);            
         }
         
      }
      
      //
      //
      //
      if (
         (tolower(G__get_type(output_var.TypeOf())) != 'u') &&
         (G__get_type(G__value_typenum(result)) == 'u') &&
         G__value_typenum(result)
      ) {
         if (G__asm_noverflow && paran) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: REWINDSTACK %d  %s:%d\n", G__asm_cp, G__asm_dt, paran, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__REWINDSTACK;
            G__asm_inst[G__asm_cp+1] = paran;
            G__inc_cp_asm(2, 0);
         }
         G__fundamental_conversion_operator(G__get_type(output_var.TypeOf()), G__get_tagnum(output_var.TypeOf().RawType()), output_var.TypeOf(), G__get_reftype(output_var.TypeOf()), output_var.TypeOf().IsConst(), &result);
         if (G__asm_noverflow && paran) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: REWINDSTACK %d  %s:%d\n", G__asm_cp, G__asm_dt, paran, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__REWINDSTACK;
            G__asm_inst[G__asm_cp+1] = -paran;
            G__inc_cp_asm(2, 0);
         }
      }
#ifdef G__ASM
      //
      //  Bytecode generation.
      //
      G__ASSERT(!G__decl || (G__decl == 1));
      if (G__asm_noverflow && !G__decl_obj) {
         if ((G__var_type != 'v') || (G__get_type(output_var.TypeOf()) != 'u')) {
            // --
            if (G__get_type(G__value_typenum(result))) {
               // --
               G__asm_gen_stvar((long) local_G__struct_offset, output_var, paran, item, (long) store_struct_offset, G__var_type, &result);
            }
            else if (G__var_type == 'u') {
               // --
               G__ASSERT(!G__decl || (G__decl == 1));
               if (G__decl) {
                  // --
                  if (G__reftype) {
                     // --
                     G__redecl(output_var);
                     if (G__no_exec_compile) {
                        G__abortbytecode();
                     }
                  }
                  else {
                     // --
                     G__class_2nd_decl_i(output_var);
                  }
               }
               else if (G__cppconstruct) {
                  // --
                  G__class_2nd_decl_c(output_var);
               }
            }
         }
      }
      else if (
         (G__var_type == 'u') &&
         (G__get_properties(output_var)->statictype == G__AUTO) &&
         (G__decl || G__cppconstruct)
      ) {
         // --
#ifdef G__ASM
         if (G__asm_noverflow) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: LD_VAR  %s paran=%d  %s:%d\n", G__asm_cp, G__asm_dt, output_var.Name(::Reflex::SCOPED).c_str(), 0, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__LD_VAR;
            G__asm_inst[G__asm_cp+1] = 0; // ig15;
            G__asm_inst[G__asm_cp+2] = 0;
            G__asm_inst[G__asm_cp+3] = 'p';
            G__asm_inst[G__asm_cp+4] = (long) output_var.Id();
            G__inc_cp_asm(5, 0);
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__PUSHSTROS;
            G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__SETSTROS;
            G__inc_cp_asm(1, 0);
         }
#endif // G__ASM
         G__class_2nd_decl(output_var);
         result.obj.i = (size_t) G__get_offset(output_var);
         G__value_typenum(result) = output_var.TypeOf();


         result.ref = (size_t) G__get_offset(output_var);
         G__var_type = 'p';



         return result;
      }
      exec_asm_letvar:
#endif // G__ASM
      // Static class/struct member.
      if (
         local_G__struct_offset &&
         (
            (G__get_properties(output_var)->statictype == G__LOCALSTATIC) ||
            ((G__get_properties(output_var)->statictype == G__COMPILEDGLOBAL) && (output_var.DeclaringScope().IsNamespace() && !output_var.DeclaringScope().IsTopScope()))
         )
      ) {
         local_G__struct_offset = 0;
      }
      // Assign G__null to existing variable is ignored.
      // This is in most cases duplicate declaration.
      if (!G__get_type(G__value_typenum(result))) {
         if (
            G__asm_noverflow &&
            (G__var_type == 'u') &&
            (G__get_properties(output_var)->statictype == G__AUTO) &&
            (G__decl || G__cppconstruct)
         ) {
            int local_store_asm_noverflow = G__asm_noverflow;
            G__asm_noverflow = 0;
            G__class_2nd_decl(output_var);
            G__asm_noverflow = local_store_asm_noverflow;
            result.obj.i = (size_t) G__get_offset(output_var);
            G__value_typenum(result) = output_var.TypeOf();


            result.ref = (size_t) G__get_offset(output_var);
         }
         G__var_type = 'p';
         if (G__reftype && (G__globalvarpointer != G__PVOID)) {
            G__get_offset(output_var) = G__globalvarpointer;
         }



         return result;
      }
      // In case of duplicate declaration, make it normal assignment.
      switch (G__var_type) {
         case 'p':
            // normal assignment
         case 'v':
            // *pointer assignment
         case 'P':
            // assignment to pointer, illegal
            break;
         case 'u':
            // special case, initialization of static member. So, class
            // object may have problem with block scope.
            break;
         default:
            // duplicated declaration handled this way because cint
            // does not support block scope
            G__var_type = 'p';
            break;
      }
      // Check const variable.
      { 
        int constvar = G__get_isconst(output_var.TypeOf());
        if (constvar &&
            !G__funcheader &&
            !G__is_cppmacro(output_var)) {
           if (
               ((!G__prerun && !G__decl) || (G__get_properties(output_var)->statictype == G__COMPILEDGLOBAL)) &&
               (
                 islower(G__get_type(output_var.TypeOf())) ||
                ((G__var_type == 'p') && (constvar & G__PCONSTVAR)) ||
                ((G__var_type == 'v') && (constvar & G__CONSTVAR))
                 )
               ) {
              G__changeconsterror(output_var.Name(::Reflex::SCOPED).c_str(), "ignored const");
              G__var_type = 'p';

              return result;
           }
        }
      }
      //
      // Variable found, set done flag.
      //
      ++done;
      /*************************************************
       * int v[A][B][C][D]
       * v[i][j][k][m]
       *
       * linear_index = B*C*D*i + C*D*j + D*k + m
       * secondary_linear_index =
       *************************************************/
      int linear_index = 0;
      {
         // -- Calculate linear_index
         // tmp = B*C*D
         int tmp = G__get_varlabel(output_var.TypeOf(), 0) /* stride */;
         for (ig25 = 0; (ig25 < paran) && (ig25 < G__get_paran(output_var)); ++ig25) {
            linear_index += tmp * G__int(para[ig25]);
            tmp /= G__get_varlabel(output_var.TypeOf(), ig25 + 2);
         }
      }
      int secondary_linear_index = 0;
      {
         // -- Calculate secondary_linear_index
         // tmp = j*k*m
         int tmp = G__get_varlabel(output_var.TypeOf(), ig25 + 3);
         while ((ig25 < paran) && G__get_varlabel(output_var.TypeOf(), ig25 + 4)) {
            secondary_linear_index += tmp * G__int(para[ig25]);
            tmp /= G__get_varlabel(output_var.TypeOf(), ig25 + 4);
            ++ig25;
         }
      }
#ifdef G__ASM
      if (G__no_exec_compile && ((tolower(G__get_type(output_var.TypeOf())) != 'u') || (ig25 < paran))) {
         result.obj.d = 0;
         result.obj.i = 1;
         G__value_typenum(result) = output_var.TypeOf();
         //--
         if (isupper(G__get_type(output_var.TypeOf()))) {
            switch (G__var_type) {
               case 'v':
                  G__value_typenum(result) = G__deref(output_var.TypeOf());
                  break;
               case 'P':
                  G__value_typenum(result) = output_var.TypeOf();
                  break;
               default:
                  if (G__get_paran(output_var) < paran) {
                     G__value_typenum(result) = G__deref(output_var.TypeOf());
                  }
                  else {
                     G__value_typenum(result) = output_var.TypeOf();
                  }
                  break;
            }
         }
         else {
            switch (G__var_type) {
               case 'p':
                  if (G__get_paran(output_var) <= paran) {
                     G__value_typenum(result) = output_var.TypeOf();
                  }
                  else {
                     G__value_typenum(result) = ::Reflex::PointerBuilder(output_var.TypeOf());
                  }
                  if ((G__get_type(G__value_typenum(result)) == 'u') && !G__value_typenum(result).RawType().IsEnum()) {
                     result.ref = 1;
                     G__tryindexopr(&result, para, paran, ig25);
                     para[0] = result;
                     para[0] = G__letVvalue(&para[0], expression);
                  }
                  break;
               case 'P':
                  G__value_typenum(result) = ::Reflex::PointerBuilder(output_var.TypeOf());
                  break;
               default:
                  G__reference_error(item);
                  break;
            }
         }
         G__var_type = 'p';
         //--
         //--
         //--
         return result;
      }
#endif // G__ASM
      //
      //  Check that the linear_index is in bounds.
      //
      //  0 <= linear_index <= number of elements
      //
      //  Note: We intentionally allow the index to go one past the end.
      //
      if (
         !G__no_exec_compile &&
         G__get_varlabel(output_var.TypeOf(), 1) /* number of elements */ &&
         (G__get_reftype(output_var.TypeOf()) == G__PARANORMAL) &&
         (
            (linear_index < 0) ||
            // We intentionally allow the index to go one past the end.
            (linear_index > G__get_varlabel(output_var.TypeOf(), 1) /* number of elements*/) ||
            ((ig25 < paran) && (std::tolower(G__get_type(output_var.TypeOf())) != 'u'))
         )
      ) {
         G__arrayindexerror(output_var, item, linear_index);



         return expression;
      }
#ifdef G__SECURITY
      if (
         !G__no_exec_compile &&
         (G__var_type == 'v') &&
         std::isupper(G__get_type(output_var.TypeOf())) &&
         (G__get_reftype(output_var.TypeOf()) == G__PARANORMAL) &&
         !G__get_varlabel(output_var.TypeOf(), 1) /* number of elements */ &&
         ((*((long*) (local_G__struct_offset + ((size_t) G__get_offset(output_var))))) == 0)
      ) {
         G__assign_error(item, &result);



         return G__null;
      }
#endif // G__SECURITY
      if (
         (G__security & G__SECURE_POINTER_TYPE) &&
         !G__definemacro &&
         isupper(G__get_type(output_var.TypeOf())) &&
         (G__var_type == 'p') &&
         !paran &&
#ifndef G__OLDIMPLEMENTATION2191
         (G__get_type(output_var.TypeOf()) != '1') &&
#else // G__OLDIMPLEMENTATION2191
         (G__get_type(output_var.TypeOf()) != 'Q') &&
#endif // G__OLDIMPLEMENTATION2191
         (
            (
               (G__get_type(output_var.TypeOf()) != 'Y') &&
               (G__get_type(G__value_typenum(result)) != 'Y') &&
               result.obj.i
            ) ||
            (G__security & G__SECURE_CAST2P)
         )
      ) {
         if (
            G__get_type(output_var.TypeOf()) != G__get_type(G__value_typenum(result)) ||
            (
               (G__get_type(G__value_typenum(result)) == 'U') &&
               (G__security & G__SECURE_CAST2P) &&
#ifdef G__VIRTUALBASE
               (G__ispublicbase(output_var.TypeOf(), G__value_typenum(result), (void*) G__STATICRESOLUTION2) == -1)
#else // G__VIRTUALBASE
               (G__ispublicbase(output_var.TypeOf(), G__value_typenum(result)) == -1)
#endif // G__VIRTUALBASE
               // --
            )
         ) {
            G__CHECK(G__SECURE_POINTER_TYPE, 0 != result.obj.i, return G__null);
         }
      }
      G__CHECK(G__SECURE_POINTER_AS_ARRAY, (G__get_paran(output_var) < paran && isupper(G__get_type(output_var.TypeOf()))), return(G__null));
      G__CHECK(G__SECURE_POINTER_ASSIGN, G__get_paran(output_var) > paran || isupper(G__get_type(output_var.TypeOf())), return(G__null));
#ifdef G__SECURITY
      if (
         G__security & G__SECURE_GARBAGECOLLECTION &&
         !G__no_exec_compile &&
         std::isupper(G__get_type(output_var.TypeOf())) &&
         (G__var_type != 'v') &&
         (G__var_type != 'P') &&
         (
            (!paran && !G__get_varlabel(output_var.TypeOf(), 1) /* number of elements */) ||
            ((paran == 1) && (G__get_varlabel(output_var.TypeOf(), 2) == 1) && !G__get_varlabel(output_var.TypeOf(), 3))
         )
      ) {
         char *address = local_G__struct_offset + ((size_t) G__get_offset(output_var)) + (linear_index * G__LONGALLOC);
         if (address && *((long*) address)) {
            G__del_refcount((void*)(*(long*)address), (void**) address);
         }
         if (std::isupper(G__get_type(G__value_typenum(result))) && result.obj.i && address) {
            G__add_refcount((void*) result.obj.i, (void**) address);
         }
      }
#endif // G__SECURITY
      //
      // Assign bit-field value.
      //
      if (G__get_bitfield_width(output_var) && (G__var_type == 'p')) {
         int mask;
         int finalval;
         int original;
         char *address = local_G__struct_offset + ((size_t) G__get_offset(output_var));
         original = *((int*) address);
         mask = (1 << G__get_bitfield_width(output_var)) - 1;
         mask = mask << G__get_bitfield_start(output_var);
         finalval = (original & (~mask)) + ((result.obj.i << G__get_bitfield_start(output_var)) & mask);
         (*(int*) address) = finalval;



         return result;
      }
      //
      //  Do the assignment now.
      //
      switch (G__get_type(output_var.TypeOf())) {
         case 'n':
            // G__int64
            G__ASSIGN_VAR(G__LONGLONGALLOC, G__int64, G__Longlong, result.obj.ll)
            break;
         case 'm':
            // G__uint64
            G__ASSIGN_VAR(G__LONGLONGALLOC, G__uint64, G__ULonglong, result.obj.ull)
            break;
         case 'q':
            // long double
            G__ASSIGN_VAR(G__LONGDOUBLEALLOC, long double, G__Longdouble, result.obj.ld)
            break;
         case 'g':
            // bool
            {
               result.obj.i = G__int(result) ? 1 : 0;
#ifdef G__BOOL4BYTE
               G__ASSIGN_VAR(G__INTALLOC, int, G__int, result.obj.i)
#else // G__BOOL4BYTE
               G__ASSIGN_VAR(G__CHARALLOC, unsigned char, G__int, result.obj.i)
#endif // G__BOOL4BYTE
               // --
            }
            break;
         case 'i':
            // int
            //G__ASSIGN_VAR(G__INTALLOC, int, G__int, result.obj.i)
            G__assign_var<int>(output_var, local_G__struct_offset, paran, linear_index, result, item, G__INTALLOC, G__int, result.obj.i);
            break;
         case 'd':
            // double
            G__assign_var<double>(output_var, local_G__struct_offset, paran, linear_index, result, item, G__DOUBLEALLOC, G__double, result.obj.d);
            break;
         case 'c':
            // char
            {
               //
               // Check for and handle initialization of
               // an unspecified length array first.
               if (
                  G__decl &&
                  (G__get_varlabel(output_var.TypeOf(), 1) /* number of elements */ == INT_MAX /* unspecified length flag */) &&
                  paran &&
                  (paran == G__get_paran(output_var)) &&
                  (G__var_type  == 'p') &&
                  !local_G__struct_offset &&
                  (G__get_type(G__value_typenum(result)) == 'C') &&
                  result.obj.i
               ) {
                  // -- An unspecified length array of characters initialized by a character pointer.
                  // Release any storage previously allocated.  FIXME: I don't think there is any.
                  if (G__get_offset(output_var)) {
                     free(G__get_offset(output_var));
                  }
                  // Allocate enough storage for a copy of the initializer string.
                  int len = strlen((const char*) result.obj.i);
                  G__get_offset(output_var) = (char*) malloc(len + 1);
                  // And copy the initializer into the allocated space.
                  strcpy((char*) G__get_offset(output_var), (const char*) result.obj.i);
                  // Change the variable into a fixed-size array of characters.
                  output_var = G__update_array_dimension(output_var, len);
                  // And return, we are done.



                  return result;
               }
               G__ASSIGN_VAR(G__CHARALLOC, char, G__int, result.obj.i)
            }
            break;
         case 'b':
            // unsigned char
            G__ASSIGN_VAR(G__CHARALLOC, unsigned char, G__int, result.obj.i)
            break;
         case 's':
            // short int
            G__ASSIGN_VAR(G__SHORTALLOC, short, G__int, result.obj.i)
            break;
         case 'r':
            // unsigned short int
            G__ASSIGN_VAR(G__SHORTALLOC, unsigned short, G__int, result.obj.i)
            break;
         case 'h':
            // unsigned int
            G__ASSIGN_VAR(G__INTALLOC, unsigned int, G__int, result.obj.i)
            break;
         case 'l':
            // long int
            G__ASSIGN_VAR(G__LONGALLOC, long, G__int, result.obj.i)
            break;
         case 'k':
            // unsigned long int
            G__ASSIGN_VAR(G__LONGALLOC, unsigned long, G__int, result.obj.i)
            break;
         case 'f':
            // float
            G__ASSIGN_VAR(G__FLOATALLOC, float, G__double, result.obj.d)
            break;
         case 'E':
            // file pointer
         case 'Y':
            // void pointer
#ifndef G__OLDIMPLEMENTATION2191
         case '1':
            // pointer to function
#else // G__OLDIMPLEMENTATION2191
         case 'Q':
            // pointer to function
#endif // G__OLDIMPLEMENTATION2191
         case 'C': /* char pointer */
            // char pointer
            G__ASSIGN_PVAR(char, G__int, result.obj.i)
            break;
         case 'N':
            G__ASSIGN_PVAR(G__int64, G__Longlong, result.obj.ll)
            break;
         case 'M':
            G__ASSIGN_PVAR(G__uint64, G__ULonglong, result.obj.ull)
            break;
#ifndef G__OLDIMPLEMENTATION2191
         case 'Q':
            G__ASSIGN_PVAR(long double, G__Longdouble, result.obj.ld)
            break;
#endif // G__OLDIMPLEMENTATION2191
         case 'G':
            // bool pointer
#ifdef G__BOOL4BYTE
            G__ASSIGN_PVAR(int, G__int, result.obj.i)
#else // G__BOOL4BYTE
            G__ASSIGN_PVAR(unsigned char, G__int, result.obj.i)
#endif // G__BOOL4BYTE
            break;
         case 'B':
            // unsigned char pointer
            G__ASSIGN_PVAR(unsigned char, G__int, result.obj.i)
            break;
         case 'S':
            // short pointer
            G__ASSIGN_PVAR(short, G__int, result.obj.i)
            break;
         case 'R':
            // unsigned short pointer
            G__ASSIGN_PVAR(unsigned short, G__int, result.obj.i)
            break;
         case 'I':
            // int pointer
            G__ASSIGN_PVAR(int, G__int, result.obj.i)
            break;
         case 'H':
            // unsigned int pointer
            G__ASSIGN_PVAR(unsigned int, G__int, result.obj.i)
            break;
         case 'L':
            // long int pointer
            G__ASSIGN_PVAR(long, G__int, result.obj.i)
            break;
         case 'K':
            // unsigned long int pointer
            G__ASSIGN_PVAR(unsigned long, G__int, result.obj.i)
            break;
         case 'F':
            // float pointer
            G__ASSIGN_PVAR(float, G__double, result.obj.d)
            break;
         case 'D':
            // double pointer
            G__ASSIGN_PVAR(double, G__double, result.obj.d)
            break;
         case 'u':
            // struct,union
            {
               if (ig25 < paran) {
                  //--
                  //--
                  result.ref = (long) (local_G__struct_offset + ((size_t) G__get_offset(output_var)) + (linear_index * output_var.TypeOf().RawType().SizeOf()));
                  G__letpointer(&result, result.ref, output_var.TypeOf().RawType());
                  G__tryindexopr(&result, para, paran, ig25);
                  para[0] = G__letVvalue(&result, expression);
                  //--
                  //--
                  //--
                  return para[0];
               }
               else {
                  G__letstruct(&result, linear_index, output_var, item, paran, local_G__struct_offset);
               }
            }
            break;
         // --
#ifdef G__ROOT
         case 'Z':
            // struct, union
            G__reference_error(item);
            break;
#endif // G__ROOT
         case 'U':
            // struct, union
            {
               if (
                  (ig25 < paran) &&
                  (
                     (G__get_reftype(output_var.TypeOf()) == G__PARANORMAL) ||
                     (G__get_reftype(output_var.TypeOf()) == (paran - ig25))
                  )
               ) {
                  //--
                  //--
                  result.ref = 0;
                  char *address = local_G__struct_offset + ((size_t) G__get_offset(output_var)) + (linear_index * G__LONGALLOC);
                  result.ref = ((*(long*)address) + (secondary_linear_index * output_var.TypeOf().RawType().SizeOf()));
                  G__letpointer(&result, result.ref, output_var.TypeOf().RawType());
                  G__tryindexopr(&result, para, paran, ig25);
                  para[0] = G__letVvalue(&result, expression);
                  //--
                  //--
                  //--
                  return para[0];
               }
               else {
                  G__letstructp(result, local_G__struct_offset, linear_index, output_var, paran, item, para, secondary_linear_index);
               }
            }
            break;
         case 'a':
            // pointer to member function
            G__letpointer2memfunc(output_var, paran, item, linear_index, &result, local_G__struct_offset);
            break;
         case 'T':
            // macro char*
            {
               if (
                  (G__globalcomp == G__NOLINK) &&
                  !G__prerun &&
                  (G__double(result) != G__double(G__getitem(item)))
               ) {
                  G__changeconsterror(varname.c_str(), "enforced macro");
               }
               *((long*) G__get_offset(output_var)) = result.obj.i;
            }
            break;
         case 'p':
            // macro int
         case 'P':
            // macro double
            {
               if (
                  (G__globalcomp == G__NOLINK) &&
                  !G__prerun &&
                  (G__double(result) != G__double(G__getitem(item)))
               ) {
                  G__changeconsterror(varname.c_str(), "enforced macro");
               }
            }
         default:
            // case 'X' automatic variable
            G__letautomatic(output_var, local_G__struct_offset, linear_index, result);
            break;
      }
   }
   //
   // If this is a variable declaration and the variable name
   // is not found in the local variable table, stop searching
   // for the old variable name.  The old declaration is just
   // ignored and the initializer value is stored into the
   // new variable.
   //
   if (!done) {
      // -- No old variable, allocate new variable.
      result = G__allocvariable(result, para, varglobal, varlocal, paran, varhash, item, varname, parameter[0][0], output_var);
   }
   G__var_type = 'p';
   //--
   //--
   //--
   return result;
}

#undef G_ASSIGN_VAR
#undef G_ASSIGN_PVAR

//______________________________________________________________________________
void Cint::Internal::G__letpointer2memfunc(const ::Reflex::Member& var, int paran, const char* item, int linear_index, G__value* presult, char* arg_G__struct_offset)
{
   // -- FIXME: Describe me!
   switch (G__var_type) {
      case 'p':
         // var = expr; assign to value
         if (G__get_paran(var) <= paran) {
            // -- Assign to type element
#ifdef G__PTR2MEMFUNC
            if (G__get_type(G__value_typenum(*presult)) == 'C') {
               *(long*)(arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * G__P2MFALLOC)) = presult->obj.i;
            }
            else {
               memcpy((void*)(arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * G__P2MFALLOC)), (void*)presult->obj.i, G__P2MFALLOC);
            }
#else // G__PTR2MEMFUNC
            memcpy((void*)(arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * G__P2MFALLOC)), (void*)presult->obj.i, G__P2MFALLOC);
#endif // G__PTR2MEMFUNC
            break;
         }
      default:
         G__assign_error(item, presult);
         break;
   }
}

//______________________________________________________________________________
void Cint::Internal::G__letautomatic(const ::Reflex::Member& var, char* arg_G__struct_offset, int linear_index, G__value result)
{
   // -- FIXME: Describe me!
   if (isupper(G__get_type(var.TypeOf()))) {
      *(double*)(arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * G__DOUBLEALLOC)) = G__double(result);
   }
   else {
      *(long*)(arg_G__struct_offset + (size_t)G__get_offset(var) + (linear_index * G__LONGALLOC)) = G__int(result);
   }
}

//______________________________________________________________________________
G__value Cint::Internal::G__letstructmem(int store_var_type, const char* /*varname_input*/, char* membername, char* tagname, const ::Reflex::Scope& varglobal, G__value expression, int objptr /* 1: object, 2: pointer */, Reflex::Member &output_var)
{
   // -- FIXME: Describe me!
   G__value result;
#ifndef G__OLDIMPLEMENTATION1259
   G__SIGNEDCHAR_T store_isconst;
#endif // G__OLDIMPLEMENTATION1259
   int store_do_setmemfuncenv;
   /* add pointer operater if necessary */
   std::string varname;
   if (store_var_type == 'P') {
      varname = "&";
      varname += membername;
      std::strcpy(membername, varname.c_str());
   }
   if (store_var_type == 'v') {
      varname = "*";
      varname += membername;
      std::strcpy(membername, varname.c_str());
   }
   ::Reflex::Scope store_tagnum = G__tagnum;
   char* store_struct_offset = G__store_struct_offset;
#ifndef G__OLDIMPLEMENTATION1259
   store_isconst = G__isconst;
#endif // G__OLDIMPLEMENTATION1259
#ifdef G__ASM
   if (G__asm_noverflow) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   //
   //  Get object pointer.
   //
   int flag = 0;
   if (tagname[strlen(tagname)-1] == ')') {
      // If this is a function call like:
      //      'something(xyz)'
      // then get it from G__getfunction.
      result = G__getfunction(tagname, &flag, G__TRYNORMAL);
   }
   else if (varglobal) {
       // If this is a top level like:
       //      'tag.subtag.mem'
       // then get it from G__global and G__p_local.
      result = G__getvariable(tagname, &flag, ::Reflex::Scope::GlobalScope(), G__p_local);
   }
   else {
      // get it '&tag' which is G__struct.memvar[].
      //        OR
      // member is referenced in member function
      //  'subtag.mem'
      G__incsetup_memvar(G__tagnum);
      result = G__getvariable(tagname, &flag, ::Reflex::Scope(), G__tagnum);
   }
   G__store_struct_offset = (char*) result.obj.i;
   G__set_G__tagnum(result);
#ifndef G__OLDIMPLEMENTATION1259
   G__isconst = G__get_isconst(G__value_typenum(result));
#endif // G__OLDIMPLEMENTATION1259
   if (!G__tagnum || G__tagnum.IsTopScope()) {
      // --
#ifndef G__OLDIMPLEMENTATION1259
      G__tagnum = store_tagnum;
      G__store_struct_offset = store_struct_offset;
      G__isconst = store_isconst;
#endif // G__OLDIMPLEMENTATION1259
      return G__null;
   }
   else if (!G__store_struct_offset) {
      if (!G__const_noerror) {
         G__fprinterr(G__serr, "Error: illegal pointer to class object %s 0x%lx %d ", tagname, G__store_struct_offset, G__get_tagnum(G__tagnum));
      }
      G__genericerror(0);
#ifndef G__OLDIMPLEMENTATION1259
      G__tagnum = store_tagnum;
      G__store_struct_offset = store_struct_offset;
      G__isconst = store_isconst;
#endif // G__OLDIMPLEMENTATION1259
      return expression;
   }
   if (!flag) {
      // object not found, return
      // G__getitem() will display error message
#ifndef G__OLDIMPLEMENTATION1259
      G__tagnum = store_tagnum;
      G__store_struct_offset = store_struct_offset;
      G__isconst = store_isconst;
#endif // G__OLDIMPLEMENTATION1259
      return G__null;
   }
#ifdef G__ASM
   if (G__asm_noverflow) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__SETSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   //
   //  Handle special case of std::auto_ptr<>.
   //
   if (
      (G__get_type(G__value_typenum(result)) == 'u') &&
      (objptr == 2) &&
      /**/
      !strncmp(G__value_typenum(result).RawType().Name_c_str(), "auto_ptr<", 9)
   ) {
      int knownx = 0;
      char comm[20];
      strcpy(comm, "operator->()");
      result = G__getfunction(comm, &knownx, G__TRYMEMFUNC);
      if (knownx) {
         G__set_G__tagnum(result);
         G__store_struct_offset = (char*) result.obj.i;
#ifdef G__ASM
         if (G__asm_noverflow) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__SETSTROS;
            G__inc_cp_asm(1, 0);
         }
#endif // G__ASM
         // --
      }
   }
   //
   //  Check if the member access operator used
   //  matches the left hand side.  We do check
   //  for operator-> in the class.
   //
   if (!G__value_typenum(result).FinalType().IsPointer() && (objptr == 2)) {
      char buf[30] = "operator->()";
      int local_flag = 0;
      ::Reflex::Scope local_store_tagnum = G__tagnum;
      char* local_store_struct_offset = G__store_struct_offset;
      G__set_G__tagnum(result);
      G__store_struct_offset = (char*) result.obj.i;
      result = G__getfunction(buf, &local_flag, G__TRYMEMFUNC);
      if (local_flag) {
         G__set_G__tagnum(result);
         G__store_struct_offset = (char*) result.obj.i;
#ifdef G__ASM
         if (G__asm_noverflow) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__SETSTROS;
            G__inc_cp_asm(1, 0);
         }
#endif // G__ASM
         // --
      }
      else {
         G__tagnum = local_store_tagnum;
         G__store_struct_offset = local_store_struct_offset;
         if (
            (G__dispmsg >= G__DISPROOTSTRICT) ||
            (G__ifile.filenum <= G__gettempfilenum())
         ) {
            if (G__dispmsg >= G__DISPWARN) {
               G__fprinterr(G__serr, "Warning: wrong member access operator '->'  %s:%d\n", __FILE__, __LINE__);
               G__printlinenum();
            }
         }
      }
   }
   if (G__value_typenum(result).FinalType().IsPointer() && (objptr == 1)) {
      if (
         (G__dispmsg >= G__DISPROOTSTRICT) ||
         (G__ifile.filenum <= G__gettempfilenum())
      ) {
         if (G__dispmsg >= G__DISPWARN) {
            G__fprinterr(G__serr, "Warning: wrong member access operator '.'  %s:%d\n", __FILE__, __LINE__);
            G__printlinenum();
         }
      }
   }
   //
   //  Assign variable value.
   //
   store_do_setmemfuncenv = G__do_setmemfuncenv;
   G__do_setmemfuncenv = 1;
   G__incsetup_memvar(G__tagnum);
   result = G__letvariable(membername, expression, ::Reflex::Scope(), G__tagnum, output_var);
   G__do_setmemfuncenv = store_do_setmemfuncenv;
   G__tagnum = store_tagnum;
   G__store_struct_offset = store_struct_offset;
#ifndef G__OLDIMPLEMENTATION1259
   G__isconst = store_isconst;
#endif // G__OLDIMPLEMENTATION1259
#ifdef G__ASM
   if (G__asm_noverflow) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   return result;
}

//______________________________________________________________________________
void Cint::Internal::G__letstruct(G__value* result, int linear_index, const ::Reflex::Member& var, const char* item, int paran, char* arg_G__struct_offset)
{
   // -- FIXME: Describe me!
   // Note:
   // G__letstruct and G__classassign in struct.c have special handling
   // of operator=(). When interpretation, overloaded assignment operator
   // is recognized in G__letstruct and G__classassign functions. They
   // set appropreate environment (G__store_struct_offset, G__tagnum)
   // and try to call operator=(). It may not be required to search for
   // non member operator=() function, so, some part of these functions
   // could be omitted.
   G__StrBuf tmp_sb(G__ONELINE);
   char *tmp = tmp_sb;
   G__StrBuf result7_sb(G__ONELINE);
   char *result7 = result7_sb;
   int ig2 = 0;
   char* store_struct_offset = 0;
   int largestep = 0;
   ::Reflex::Scope store_tagnum;
   G__value para = *result;
   int store_prerun = 0;
   int store_debug = 0;
   int store_step = 0;
   long store_asm_inst = 0L;
   long addr = 0L;
   if (G__asm_exec) {
      void* p1 = (void*) (arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * var.TypeOf().RawType().SizeOf()));
      void* p2 = (void*) result->obj.i ;
      size_t size = (size_t) var.TypeOf().RawType().SizeOf();
      memcpy(p1, p2, size);
      return;
   }
   switch (G__var_type) {
      case 'p': /* return value */
         if (G__get_paran(var) <= paran) {
            // -- value, struct,union
            store_prerun = G__prerun;
            G__prerun = 0;
            if (store_prerun) {
               store_debug = G__debug;
               store_step = G__step;
               G__debug = G__debugtrace;
               G__step = G__steptrace;
               G__setdebugcond();
            }
            else {
               if (G__breaksignal) {
                  G__break = 0;
                  G__setdebugcond();
                  int ret = G__pause();
                  if (ret == G__PAUSE_STEPOVER) {
                     if (G__return == G__RETURN_NON) {
                        G__step = 0;
                        G__setdebugcond();
                        largestep = 1;
                     }
                  }
                  if (G__return > G__RETURN_NORMAL) {
                     return;
                  }
               }
            }
            if (
               G__get_tagnum(G__value_typenum(*result)) != -1 &&
               (
                  (G__get_type(G__value_typenum(*result))  == 'u') ||
                  (G__get_type(G__value_typenum(*result)) == 'i')
               )
            ) {
               if (result->obj.i) {
                  sprintf(tmp, "(%s)(%ld)", G__fulltagname(G__get_tagnum(G__value_typenum(*result)), 1), result->obj.i);
               }
               else {
                  sprintf(tmp, "(%s)%ld", G__fulltagname(G__get_tagnum(G__value_typenum(*result)), 1), result->obj.i);
               }
            }
            else {
               G__valuemonitor(*result, tmp);
            }
            G__ASSERT(!G__decl || (G__decl == 1));
            if (G__decl) {
               // -- Copy constructor.
               sprintf(result7, "%s(%s)", var.TypeOf().RawType().Name(::Reflex::SCOPED).c_str(), tmp);
               store_tagnum = G__tagnum;
               G__set_G__tagnum(var.TypeOf().RawType());
               store_struct_offset = G__store_struct_offset;
               G__store_struct_offset = (arg_G__struct_offset + (size_t)G__get_offset(var) + (linear_index * var.TypeOf().RawType().SizeOf()));
               if (G__dispsource) {
                  G__fprinterr(G__serr, "\n!!!Calling constructor 0x%lx.%s for declaration", G__store_struct_offset, result7);
               }
#ifdef G__SECURITY
               G__castcheckoff = 1;
#endif // G__SECURITY
               ig2 = 0;
               G__decl = 0;
#ifndef G__OLDIMPLEMENTATION1073
               G__oprovld = 1;
#endif // G__OLDIMPLEMENTATION1073
               {
                  int store_cp = G__asm_cp;
                  int store_dt = G__asm_dt;
                  G__getfunction(result7, &ig2, G__TRYCONSTRUCTOR);
                  if (ig2 && G__asm_noverflow) {
                     G__asm_dt = store_dt;
                     if (G__asm_inst[G__asm_cp-6] == G__LD_FUNC) {
                        for (int i = 0; i < 6; ++i) {
                           G__asm_inst[store_cp+i] = G__asm_inst[G__asm_cp-6+i];
                        }
                        G__asm_cp = store_cp + 6;
                     }
                     else if (G__asm_inst[G__asm_cp-8] == G__LD_IFUNC) {
                        for (int i = 0; i < 8; ++i) {
                           G__asm_inst[store_cp+i] = G__asm_inst[G__asm_cp-8+i];
                        }
                        G__asm_cp = store_cp + 8;
                     }
                  }
                  else if (!ig2 && (G__get_type(G__value_typenum(*result)) == 'U')) {
                     G__fprinterr(G__serr, "Error: Constructor %s not found", result7);
                     G__genericerror(0);
                  }
               }
#ifndef G__OLDIMPLEMENTATION1073
               G__oprovld = 0;
               if (G__asm_wholefunction && !ig2) {
                  G__asm_gen_stvar((long) arg_G__struct_offset, var, paran, (char*) item, G__ASM_VARLOCAL, G__var_type, result);
               }
#endif // G__OLDIMPLEMENTATION1073
               G__decl = 1;
               G__store_struct_offset = store_struct_offset;
               G__tagnum = store_tagnum;
            }
            else {
               // --
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "ST_VAR or ST_MSTR replaced with LD_VAR or LD_MSTR(2)  %s:%d\n", __FILE__, __LINE__);
                  }
#endif // G__ASM_DBG
                  store_asm_inst = G__asm_inst[G__asm_cp-5];
                  if (store_asm_inst == G__ST_VAR) {
                     G__asm_inst[G__asm_cp-5] = G__LD_VAR;
                  }
                  else if (store_asm_inst == G__ST_LVAR) {
                     G__asm_inst[G__asm_cp-5] = G__LD_LVAR;
                  }
                  else {
                     G__asm_inst[G__asm_cp-5] = G__LD_MSTR;
                  }
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__PUSHSTROS;
                  G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__SETSTROS;
                  G__inc_cp_asm(1, 0);
               }
               G__oprovld = 1;
#endif // G__ASM
               // Overloading of operator=.
               //
               // Search for member function.
               //
               sprintf(result7, "operator=(%s)", tmp);
               store_tagnum = G__tagnum;
               G__set_G__tagnum(var.TypeOf().RawType());
               store_struct_offset = G__store_struct_offset;
               G__store_struct_offset = (arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * var.TypeOf().RawType().SizeOf()));
               ig2 = 0;
               para = G__getfunction(result7, &ig2, G__TRYMEMFUNC);
               if (!ig2 && (G__tagnum != G__value_typenum(*result).RawType())) {
                  // -- Copy constructor.
                  sprintf(result7, "%s(%s)", G__tagnum.Name(::Reflex::SCOPED).c_str(), tmp);
                  if (G__struct.iscpplink[G__get_tagnum(G__tagnum)] == G__CPPLINK) {
                     G__abortbytecode();
                     char* store_globalvarpointer = G__globalvarpointer;
                     G__globalvarpointer = G__store_struct_offset;
                     G__getfunction(result7, &ig2, G__TRYCONSTRUCTOR);
                     G__globalvarpointer = store_globalvarpointer;
                  }
                  else {
                     G__getfunction(result7, &ig2, G__TRYCONSTRUCTOR);
                  }
               }
               G__store_struct_offset = store_struct_offset;
               G__tagnum = store_tagnum;
               // Search for global function.
               if (!ig2) {
                  // --
#ifdef G__ASM
                  if (G__asm_noverflow) {
                     G__inc_cp_asm(-2, 0);
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr, "PUSHSTROS,SETSTROS cancelled  %s:%d", __FILE__, __LINE__);
                        G__printlinenum();
                     }
#endif // G__ASM_DBG
                     // --
                  }
#endif // G__ASM
                  addr = (long) (arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * var.TypeOf().RawType().SizeOf()));
                  if (addr < 0) {
                     sprintf(result7, "operator=((%s)(%ld),%s)", var.TypeOf().RawType().Name(::Reflex::SCOPED).c_str(), addr, tmp);
                  }
                  else {
                     sprintf(result7, "operator=((%s)%ld,%s)", var.TypeOf().RawType().Name(::Reflex::SCOPED).c_str(), addr, tmp);
                  }
                  para = G__getfunction(result7, &ig2, G__TRYNORMAL);
               }
#ifdef G__ASM
               else {
                  if (G__asm_noverflow) {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__POPSTROS;
                     G__inc_cp_asm(1, 0);
                  }
               }
               G__oprovld = 0;
#endif // G__ASM
               // --
            }
            if (store_prerun) {
               G__debug = store_debug;
               G__step = store_step;
               G__setdebugcond();
            }
            else {
               if (largestep) {
                  G__step = 1;
                  G__setdebugcond();
                  largestep = 0;
               }
            }
            G__prerun = store_prerun;
            if (ig2) {
               // In case overloaded = or constructor is found.
               *result = para;
            }
            else {
               // -- In case no overloaded = or constructor, memberwise copy.
               // Try conversion operator for class object.
               if ((G__get_type(G__value_typenum(*result)) == 'u') && G__value_typenum(*result).IsClass()) {
                  ::Reflex::Type tagnum = var.TypeOf().RawType();
                  int done = G__class_conversion_operator(tagnum, result);
                  if (done) {
                     char* pdest = arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * var.TypeOf().RawType().SizeOf());
                     G__classassign(pdest, tagnum, *result);
                     return;
                  }
               }
#ifdef G__ASM
               if (G__asm_noverflow
                  // --
#ifndef G__OLDIMPLEMENTATION1073
                  && store_asm_inst
#endif // G__OLDIMPLEMENTATION1073
                  // --
               ) {
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "ST_VAR or ST_MSTR recovered no_exec_compile=%d  %s:%d\n", G__no_exec_compile, __FILE__, __LINE__);
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp-5] = store_asm_inst;
               }
               if (G__no_exec_compile || (G__globalcomp && !G__int(*result))) {
                  // With -c-1 or -c-2 option.
                  return;
               }
#endif // G__ASM
               if (G__value_typenum(*result).RawType() == var.TypeOf().RawType()) {
                  memcpy((void*) (arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * var.TypeOf().RawType().SizeOf())), (void*) G__int(*result), (size_t) var.TypeOf().RawType().SizeOf());
               }
               else if (-1 != (addr = G__ispublicbase(var.TypeOf().RawType(), G__value_typenum(*result), 0))) {
                  int tagnum = G__get_tagnum(var.TypeOf().RawType());
                  char* pdest = arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * G__struct.size[tagnum]);
                  memcpy((void*) pdest, (void*) (G__int(*result) + addr), (size_t) G__struct.size[tagnum]);
                  if (((long) G__struct.virtual_offset[tagnum]) != -1) {
                     *(long*)(pdest + ((size_t) G__struct.virtual_offset[tagnum])) = tagnum;
                  }
               }
               else {
                  G__fprinterr(G__serr, "Error: Assignment to %s type incompatible ", item);
                  G__genericerror(0);
               }
            }
            break;
         }
         else if (G__funcheader && !paran && G__value_typenum(*result).FinalType().IsPointer()) {
            // FIXME: Remove special case for unspecified length array.
            if (G__get_offset(var) && (G__get_properties(var)->statictype != G__COMPILEDGLOBAL)) {
               free(G__get_offset(var));
            }
            G__get_offset(var) = (char*) result->obj.i;
            G__get_properties(var)->statictype = G__COMPILEDGLOBAL;
            break;
         }
      default:
         if (G__var_type == 'u') {
            G__letpointer(result, (long) (arg_G__struct_offset + ((size_t) G__get_offset(var))), var.TypeOf().RawType());
            //--
            //--
            break;
         }
         if (G__var_type == 'v') {
            G__StrBuf refopr_sb(G__MAXNAME);
            char *refopr = refopr_sb;
            char* store_struct_offsetX = G__store_struct_offset;
            ::Reflex::Scope store_tagnumX = G__tagnum;
            int done = 0;
            int store_var_type = G__var_type;
            G__var_type = 'p';
            G__store_struct_offset = (arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * var.TypeOf().RawType().SizeOf()));
            G__set_G__tagnum(var.TypeOf().RawType());
#ifdef G__ASM
            if (G__asm_noverflow) {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD_VAR  %s paran=%d  %s:%d\n", G__asm_cp, G__asm_dt, var.Name(::Reflex::SCOPED).c_str(), 0, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               if (arg_G__struct_offset) {
                  G__asm_inst[G__asm_cp] = G__LD_MSTR;
               }
               else {
                  G__asm_inst[G__asm_cp] = G__LD_VAR;
               }
               G__asm_inst[G__asm_cp+1] = 0; // ig15;
               G__asm_inst[G__asm_cp+2] = paran;
               G__asm_inst[G__asm_cp+3] = 'p';
               G__asm_inst[G__asm_cp+4] = (long) var.Id();
               G__inc_cp_asm(5, 0);
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__PUSHSTROS;
               G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__SETSTROS;
               G__inc_cp_asm(1, 0);
            }
#endif // G__ASM
            strcpy(refopr, "operator*()");
            para = G__getfunction(refopr, &done, G__TRYMEMFUNC);
            G__tagnum = store_tagnumX;
            G__store_struct_offset = store_struct_offsetX;
            G__var_type = store_var_type;
#ifdef G__ASM
            if (G__asm_noverflow) {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: POPSTROS\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__POPSTROS;
               G__inc_cp_asm(1, 0);
            }
#endif // G__ASM
            if (!done) {
               G__assign_error(item, result);
            }
            G__letVvalue(&para, *result);
         }
         else
            G__assign_error(item, result);
         break;
   }
}

//______________________________________________________________________________
void Cint::Internal::G__letstructp(G__value result, char* arg_G__struct_offset, int linear_index, const ::Reflex::Member& var, int paran, const char* item, G__value* para, int secondary_linear_index)
{
   // -- FIXME: Describe me!
   int baseoffset = 0;
   switch (G__var_type) {
      case 'v':
         // *var = result;  Assign using a pointer variable derefence.
         switch (G__get_reftype(var.TypeOf())) {
            case G__PARANORMAL:
               if (G__no_exec_compile) {
                  G__classassign(G__PVOID, var.TypeOf().RawType(), result);
               }
               else {
                  G__classassign((char*) (*(long*)(arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * G__LONGALLOC))), var.TypeOf().RawType(), result);
               }
               break;
            case G__PARAP2P:
               if (paran > G__get_paran(var)) {
                  // -- Pointer to array reimplementation.
                  if (G__no_exec_compile) {
                     G__classassign(G__PVOID, var.TypeOf().RawType(), result);
                  }
                  else {
                     char* address = arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * G__LONGALLOC);
                     G__classassign((*(((char**)(*(long*) address)) + secondary_linear_index)), var.TypeOf().RawType(), result);
                  }
               }
               else {
                  if (!G__no_exec_compile) {
                     *(long*)(*(long*) (arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * G__LONGALLOC))) = G__int(result);
                  }
               }
               break;
         }
         break;
      case 'p':
         // var = result;  Assign to a pointer variable.
         if (paran >= G__get_paran(var)) {
            if (G__get_paran(var) < paran) {
               char* address = arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * G__LONGALLOC);
               if (G__get_reftype(var.TypeOf()) == G__PARANORMAL) {
                  if (G__no_exec_compile) {
                     address = G__PVOID;
                  }
                  else {
                     address = (*(char**)address) + (secondary_linear_index * var.TypeOf().RawType().SizeOf());
                  }
                  G__classassign(address, var.TypeOf().RawType(), result);
               }
               else if (G__get_paran(var) == (paran - 1)) {
                  if (!G__no_exec_compile) {
                     *(((long*)(*(long *)address)) + secondary_linear_index) = G__int(result);
                  }
               }
               else if (G__get_paran(var) == (paran - 2)) {
                  if (!G__no_exec_compile) {
                     address = (char*)((long*)(*(long*)address) + para[0].obj.i);
                     if (G__get_reftype(var.TypeOf()) == G__PARAP2P) {
                        address = ((*((char**)address)) + para[1].obj.i * var.TypeOf().RawType().SizeOf());
                        G__classassign(address, var.TypeOf().RawType(), result);
                     }
                     else if (G__get_reftype(var.TypeOf()) > G__PARAP2P) {
                        address = (char*)((long*)(*(long*)address) + para[1].obj.i);
                        *(long*) address = G__int(result);
                     }
                  }
               }
               else if (G__get_paran(var) == (paran - 3)) {
                  if (!G__no_exec_compile) {
                     address = (char*)((long*)(*(long*)address) + para[0].obj.i);
                     address = (char*)((long*)(*(long*)address) + para[1].obj.i);
                     if (G__get_reftype(var.TypeOf()) == G__PARAP2P2P) {
                        address = (char*)((*((long*)(address))) + (para[2].obj.i * var.TypeOf().RawType().SizeOf()));
                        G__classassign(address, var.TypeOf().RawType(), result);
                     }
                     else if (G__get_reftype(var.TypeOf()) > G__PARAP2P2P) {
                        address = (char*)((long*)(*(long*)address) + para[2].obj.i);
                        *(long*) address = G__int(result);
                     }
                  }
               }
               else {
                  if (!G__no_exec_compile)
                     G__classassign((char*)((*(((long*)(*(long*)address)) + para[0].obj.i)) + para[1].obj.i), var.TypeOf().RawType(), result);
               }
            }
            else {
               // check if tagnum matches.
               // If unmatch, check for class inheritance.
               // If derived class pointer is assigned to
               // base class pointer, add offset and assign.
               if (G__no_exec_compile) {
                  // Base class casting at this position does not make sense,
                  // because ST_VAR is already generated in G__asm_gen_stvar.
                  return;
               }
               if (
                  (G__get_type(G__value_typenum(result)) != 'U') &&
                  (G__get_type(G__value_typenum(result)) != 'Y') &&
                  result.obj.i &&
                  ((G__get_type(G__value_typenum(result)) != 'u') || (result.obj.i == G__p_tempbuf->obj.ref))
               ) {
                  G__assign_error(item, &result);
                  return;
               }
               if (
                  (var.TypeOf().RawType() == G__value_typenum(result).RawType()) ||
                  !result.obj.i ||
                  (G__get_type(G__value_typenum(result)) == 'Y')
               ) {
                  // Checked.
                  *(long*)(arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * G__LONGALLOC)) = G__int(result);
               }
               else if (
                  // --
#ifdef G__VIRTUALBASE
                  ((baseoffset = G__ispublicbase(var.TypeOf().RawType(), G__value_typenum(result), (void*)result.obj.i)) != -1)
#else // G__VIRTUALBASE
                  ((baseoffset = G__ispublicbase(var.TypeOf().RawType(), G__value_typenum(result))) != -1)
#endif // G__VIRTUALBASE
                  // --
               ) {
                  *(long*)(arg_G__struct_offset + ((size_t) G__get_offset(var)) + (linear_index * G__LONGALLOC)) = G__int(result) + baseoffset;
               }
               else {
                  G__assign_error(item, &result);
               }
            }
         }
         else {
            G__assign_error(item, &result);
         }
         break;
      default:
         G__assign_error(item, &result);
         break;
   }
}

//______________________________________________________________________________
G__value Cint::Internal::G__classassign(char* pdest, const ::Reflex::Type& tagnum, G__value result)
{
   // -- FIXME: Describe me!
#ifndef G__OLDIMPLEMENTATION1823
   G__StrBuf buf_sb(G__BUFLEN*2);
   char *buf = buf_sb;
   G__StrBuf buf2_sb(G__BUFLEN*2);
   char *buf2 = buf2_sb;
   char* ttt = buf;
   char* result7 = buf2;
   int lenttt;
#else // G__OLDIMPLEMENTATION1823
   G__StrBuf ttt_sb(G__ONELINE);
   char *ttt = ttt_sb;
   G__StrBuf result7_sb(G__ONELINE);
   char *result7 = result7_sb;
#endif // G__OLDIMPLEMENTATION1823
   char* store_struct_offset = 0;
   ::Reflex::Scope store_tagnum;
   int ig2 = 0;
   G__value para;
   long store_asm_inst = 0;
   int letvvalflag = 0;
   long addstros_value = 0;
   if (G__asm_exec) {
      memcpy((void*) pdest, (void*) G__int(result), tagnum.SizeOf());
      return result;
   }
   if (G__get_type(G__value_typenum(result)) == 'u') {
      // --
#ifndef G__OLDIMPLEMENTATION1823
      // --
      lenttt = G__value_typenum(result).Name(::Reflex::SCOPED).length();
      if (lenttt > (2 * G__BUFLEN) - 10) {
         ttt = (char*) malloc(lenttt + 20);
         result7 = (char*) malloc(lenttt + 30);
      }
      G__setiparseobject(&result, ttt);
#else // G__OLDIMPLEMENTATION1823
      if (result.obj.i < 0) {
         sprintf(ttt, "(%s)(%ld)", G__value_typenum(result).Name(::Reflex::SCOPED), result.obj.i);
      }
      else {
         sprintf(ttt, "(%s)%ld", G__value_typenum(result).Name(::Reflex::SCOPED), result.obj.i);
      }
#endif // G__OLDIMPLEMENTATION1823
      // --
   }
   else {
      G__valuemonitor(result, ttt);
   }
   //
   // operator= overloading
   //
#ifdef G__ASM
   if (G__asm_noverflow) {
      if (G__asm_inst[G__asm_cp-1] == G__LETVVAL) {
         G__inc_cp_asm(-1, 0);
         letvvalflag = 1;
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "LETVVAL cancelled  %s:%d", __FILE__, __LINE__);
            G__printlinenum();
         }
#endif // G__ASM_DBG
         // --
      }
      else {
         if (G__asm_inst[G__asm_cp-2] == G__ADDSTROS) {
            addstros_value = G__asm_inst[G__asm_cp-1];
            G__inc_cp_asm(-2, 0);
         }
         else {
            addstros_value = 0;
         }
         letvvalflag = 0;
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "ST_VAR or ST_MSTR replaced with LD_VAR or LD_MSTR(1)  %s:%d\n", __FILE__, __LINE__);
            G__printlinenum();
         }
#endif // G__ASM_DBG
         store_asm_inst = G__asm_inst[G__asm_cp-5];
         if (store_asm_inst == G__ST_VAR) {
            G__asm_inst[G__asm_cp-5] = G__LD_VAR;
         }
         else if (store_asm_inst == G__ST_LVAR) {
            G__asm_inst[G__asm_cp-5] = G__LD_LVAR;
         }
         else {
            G__asm_inst[G__asm_cp-5] = G__LD_MSTR;
         }
      }
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__SETSTROS;
      G__inc_cp_asm(1, 0);
   }
   G__oprovld = 1;
#endif // G__ASM
   // searching for member function
   sprintf(result7, "operator=(%s)", ttt);
   store_tagnum = G__tagnum;
   G__set_G__tagnum(tagnum);
   store_struct_offset = G__store_struct_offset;
   G__store_struct_offset = pdest;
   ig2 = 0;
   para = G__getfunction(result7, &ig2, G__TRYMEMFUNC);
   if (!ig2 && (tagnum != G__value_typenum(result))) {
      //
      // copy constructor
      //
      char* store_globalvarpointer = 0;
#ifndef G__OLDIMPLEMENTATION1823
      std::string xp2( tagnum.Name(::Reflex::SCOPED) );
      lenttt = strlen(ttt);
      int len2 = xp2.length() + lenttt + 10;
      if (buf2 == result7) {
         if (len2 > (2 * G__BUFLEN)) {
            result7 = (char*) malloc(len2);
         }
      }
      else {
         if (len2 > (lenttt + 30)) {
            free((void*) result7);
            result7 = (char*) malloc(len2);
         }
      }
      sprintf(result7, "%s(%s)", xp2.c_str(), ttt);
#else // G__OLDIMPLEMENTATION1823
      sprintf(result7, "%s(%s)", tagnum.Name(::Reflex::SCOPED).c_str(), ttt);
#endif // G__OLDIMPLEMENTATION1823
      if (G__get_properties(tagnum)->iscpplink == G__CPPLINK) {
         G__abortbytecode();
         store_globalvarpointer = G__globalvarpointer;
         G__globalvarpointer = G__store_struct_offset;
         G__getfunction(result7, &ig2, G__TRYCONSTRUCTOR);
         G__globalvarpointer = store_globalvarpointer;
      }
      else {
         G__getfunction(result7, &ig2, G__TRYCONSTRUCTOR);
      }
   }
   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;
   // Searching for global function.
   if (!ig2) {
      // --
#ifdef G__ASM
      if (G__asm_noverflow) {
         G__inc_cp_asm(-2, 0);
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "PUSHSTROS,SETSTROS cancelled  %s:%d", __FILE__, __LINE__);
            G__printlinenum();
         }
#endif // G__ASM_DBG
         // --
      }
#endif // G__ASM
      if (pdest < 0) {
         sprintf(result7, "operator=((%s)(%p),%s)", tagnum.Name(::Reflex::SCOPED).c_str(), pdest, ttt);
      }
      else {
         sprintf(result7, "operator=((%s)%p,%s)", tagnum.Name(::Reflex::SCOPED).c_str(), pdest, ttt);
      }
      para = G__getfunction(result7, &ig2, G__TRYNORMAL);
#ifdef G__ASM
      if (G__asm_noverflow && addstros_value) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "ADDSTROS %d recovered  %s:%d\n", addstros_value, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__ADDSTROS;
         G__asm_inst[G__asm_cp+1] = addstros_value;
         G__inc_cp_asm(2, 0);
      }
#endif // G__ASM
      // --
   }
#ifdef G__ASM
   else {
      if (G__asm_noverflow) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__POPSTROS;
         G__inc_cp_asm(1, 0);
      }
   }
   G__oprovld = 0;
#endif // G__ASM
   if (ig2) {
      // in case overloaded = or constructor is found
#ifndef G__OLDIMPLEMENTATION1823
      if (buf != ttt) {
         free((void*)ttt);
      }
      if (buf2 != result7) {
         free((void*)result7);
      }
#endif // G__OLDIMPLEMENTATION1823
      return para;
   }
   // in case no overloaded = or constructor, memberwise copy.
#ifdef G__ASM
   if (G__asm_noverflow) {
      if (letvvalflag) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "LETVVAL recovered  %s:%d\n", __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__LETVVAL;
         G__inc_cp_asm(1, 0);
      }
      else {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "ST_VAR or ST_MSTR recovered no_exec_compile=%d  %s:%d\n", G__no_exec_compile, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp-5] = store_asm_inst;
         if (addstros_value) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "ADDSTROS %d recovered  %s:%d\n", addstros_value, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__ADDSTROS;
            G__asm_inst[G__asm_cp+1] = addstros_value;
            G__inc_cp_asm(2, 0);
         }
      }
   }
   // Try conversion operator for class object.
   if ((G__get_type(G__value_typenum(result)) == 'u') && G__value_typenum(result).IsClass()) {
      int done = G__class_conversion_operator(tagnum, &result);
      if (done) {
         // --
#ifndef G__OLDIMPLEMENTATION1823
         if (buf != ttt) {
            free((void*)ttt);
         }
         if (buf2 != result7) {
            free((void*)result7);
         }
#endif // G__OLDIMPLEMENTATION1823
         return G__classassign(pdest, tagnum, result);
      }
   }
   // Return from this function if this is pure bytecode compilation.
   if (G__no_exec_compile) {
      // --
#ifndef G__OLDIMPLEMENTATION1823
      if (buf != ttt) {
         free((void*)ttt);
      }
      if (buf2 != result7) {
         free((void*)result7);
      }
#endif // G__OLDIMPLEMENTATION1823
      return result;
   }
#endif // G__ASM
   if (G__value_typenum(result) == tagnum) {
      memcpy((void*) pdest, (void*) G__int(result), (size_t) tagnum.SizeOf());
   }
   else if ((addstros_value = G__ispublicbase(tagnum, G__value_typenum(result), 0)) != -1) {
      memcpy((void*) pdest, (void*) (G__int(result) + addstros_value), (size_t) tagnum.SizeOf());
      if (((long) G__struct.virtual_offset[G__get_tagnum(tagnum)]) != -1) {
         *(long*)(pdest + (size_t) G__struct.virtual_offset[G__get_tagnum(tagnum)]) = G__get_tagnum(tagnum);
      }
   }
   else {
      G__fprinterr(G__serr, "Error: Assignment type incompatible FILE:%s LINE:%d  %s:%d\n", G__ifile.name, G__ifile.line_number, __FILE__, __LINE__);
   }
#ifndef G__OLDIMPLEMENTATION1823
   if (buf != ttt) {
      free((void*)ttt);
   }
   if (buf2 != result7) {
      free((void*)result7);
   }
#endif // G__OLDIMPLEMENTATION1823
   return result;
}

//______________________________________________________________________________
int Cint::Internal::G__class_conversion_operator(const ::Reflex::Type& tagnum, G__value* presult)
{
   // -- Conversion operator for assignment to class object.
   // Note: Bytecode compilation turned off if conversion operator is found.
   G__value conv_result;
   int conv_done = 0;
   ::Reflex::Scope conv_tagnum = G__tagnum;
   ::Reflex::Type conv_typenum = G__typenum;
   int conv_constvar = G__constvar;
   int conv_reftype = G__reftype;
   int conv_var_type = G__var_type;
   char* conv_store_struct_offset = G__store_struct_offset;
   switch (G__get_tagtype(G__value_typenum(*presult))) {
      case 'c':
      case 's':
         G__set_G__tagnum(G__value_typenum(*presult));
         G__typenum = ::Reflex::Type();
         G__constvar = 0;
         G__reftype = 0;
         G__var_type = 'p';
         G__store_struct_offset = (char*) presult->obj.i;
         // Synthesize function name.
         G__StrBuf tmp_sb(G__ONELINE);
         char *tmp = tmp_sb;
         strcpy(tmp, "operator ");
         strcpy(tmp + 9, tagnum.Name(::Reflex::SCOPED).c_str());
         strcpy(tmp + strlen(tmp), "()");
         // Call conversion operator.
         conv_result = G__getfunction(tmp, &conv_done, G__TRYMEMFUNC);
         if (conv_done) {
            if (G__dispsource) {
               G__fprinterr(G__serr, "!!!Conversion operator called 0x%lx.%s\n", G__store_struct_offset, tmp);
            }
#ifdef G__ASM
            G__abortbytecode();
#endif // G__ASM
            *presult = conv_result;
         }
         G__tagnum = conv_tagnum;
         G__typenum = conv_typenum;
         G__constvar = conv_constvar;
         G__reftype = conv_reftype;
         G__var_type = conv_var_type;
         G__store_struct_offset = conv_store_struct_offset;
         break;
   }
   return conv_done;
}

//______________________________________________________________________________
int Cint::Internal::G__fundamental_conversion_operator(int type, int tagnum, ::Reflex::Type typenum, int reftype, int constvar, G__value* presult)
{
   // -- Conversion operator for assignment to fundamental type object.
   //
   // Note: Bytecode compilation is alive after conversion operator is used.
   //
   G__StrBuf tmp_sb(G__ONELINE);
   char *tmp = tmp_sb;
   G__value conv_result;
   int conv_done = 0;
   ::Reflex::Scope conv_tagnum = G__tagnum;
   ::Reflex::Type conv_typenum = G__typenum;
   int conv_constvar = G__constvar;
   int conv_reftype = G__reftype;
   int conv_var_type = G__var_type;
   char* conv_store_struct_offset = G__store_struct_offset;
   switch (G__get_tagtype(G__value_typenum(*presult))) {
      case 'c':
      case 's':
         G__set_G__tagnum(G__value_typenum(*presult));
         G__typenum = ::Reflex::Type();
         G__constvar = 0;
         G__reftype = 0;
         G__var_type = 'p';
         G__store_struct_offset = (char*) presult->obj.i;
#ifdef G__ASM
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__PUSHSTROS;
         G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__SETSTROS;
         G__inc_cp_asm(1, 0);
#endif // G__ASM
         // Synthesize function name.
         strcpy(tmp, "operator ");
         strcpy(tmp + 9, G__type2string(type, tagnum, G__get_typenum(typenum), reftype, constvar));
         strcpy(tmp + strlen(tmp), "()");
         // Call conversion operator.
         conv_result = G__getfunction(tmp, &conv_done, G__TRYMEMFUNC);
         if (!conv_done && typenum) {
            // Make another try after removing typedef alias.
            strcpy(tmp + 9, G__type2string(type, -1, -1, reftype, constvar));
            strcpy(tmp + strlen(tmp), "()");
            conv_result = G__getfunction(tmp, &conv_done, G__TRYMEMFUNC);
         }
         if (!conv_done) {
            // Make another try constness reverting.
            constvar ^= 1;
            strcpy(tmp + 9, G__type2string(type, tagnum, G__get_typenum(typenum), reftype, constvar));
            strcpy(tmp + strlen(tmp), "()");
            conv_result = G__getfunction(tmp, &conv_done, G__TRYMEMFUNC);
            if (!conv_done && typenum) {
               // Make another try after removing typedef alias.
               strcpy(tmp + 9, G__type2string(type, -1, -1, reftype, constvar));
               strcpy(tmp + strlen(tmp), "()");
               conv_result = G__getfunction(tmp, &conv_done, G__TRYMEMFUNC);
            }
         }
         if (!conv_done) {
            for (::Reflex::Type_Iterator iTypedef = ::Reflex::Type::Type_Begin(); iTypedef !=::Reflex::Type::Type_End(); ++iTypedef) {
               if (!iTypedef->IsTypedef()) {
                  continue;
               }
               if ((type == G__get_type(*iTypedef)) && (tagnum == G__get_tagnum(*iTypedef))) {
                  constvar ^= 1;
                  strcpy(tmp + 9, G__type2string(type, tagnum, G__get_typenum(*iTypedef), reftype, constvar));
                  strcpy(tmp + strlen(tmp), "()");
                  conv_result = G__getfunction(tmp, &conv_done, G__TRYMEMFUNC);
                  if (!conv_done) {
                     constvar ^= 1;
                     strcpy(tmp + 9, G__type2string(type, tagnum, G__get_typenum(typenum), reftype, constvar));
                     strcpy(tmp + strlen(tmp), "()");
                     conv_result = G__getfunction(tmp, &conv_done, G__TRYMEMFUNC);
                  }
                  if (conv_done) {
                     break;
                  }
               }
            }
         }
         if (conv_done) {
            if (G__dispsource) {
               G__fprinterr(G__serr, "!!!Conversion operator called 0x%lx.%s\n", G__store_struct_offset, tmp);
            }
            *presult = conv_result;
#ifdef G__ASM
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__POPSTROS;
            G__inc_cp_asm(1, 0);
#endif // G__ASM
            // --
         }
         else {
            // --
#ifdef G__ASM
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "PUSHSTROS, SETSTROS cancelled  %s:%d\n", __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__inc_cp_asm(-2, 0);
#endif // G__ASM
            // --
         }
         G__tagnum = conv_tagnum;
         G__typenum = conv_typenum;
         G__constvar = conv_constvar;
         G__reftype = conv_reftype;
         G__var_type = conv_var_type;
         G__store_struct_offset = conv_store_struct_offset;
         break;
   }
   return conv_done;
}

//______________________________________________________________________________
//
// -- Allocate memory for storage of a variable's value.
//
#ifdef G__ASM_WHOLEFUNC


template<class CASTTYPE, class CONVFUNC>
inline void G__alloc_var_ref(unsigned int SIZE, CONVFUNC f, const char* item, ::Reflex::Member& var, G__value& result)
{
   if (islower(G__var_type)) {
      /* -- Not a pointer, may be an array. */
      /* Allocate memory */
      if (G__get_varlabel(var.TypeOf(), 1) /* number of elements */ == INT_MAX /* unspecified length flag */) {
         /* -- Unspecified length array. */
         if (G__funcheader) {
            /* -- In a function header. */
            /* Allocate no storage, we will share it with the caller (see the initializer below). */
            /*G__get_offset(var) = 0;*/
         }
         else {
            /* -- Not in a function header. */
            if (!G__static_alloc || G__prerun) {
               /* -- Not static, const, enum; or it is prerun time. */
               /* Allocate no storage (it will be allocated later during initializer parsing). */
               /*G__get_offset(var) = 0;*/
            }
            else {
               /* -- Static, const, or enum during execution. */
               /* Copy the pointer from the global variable, no actual allocation takes place. */
               G__get_offset(var) = (char*) G__malloc(1, SIZE, item);
            }
         }
      }
      else if (G__get_varlabel(var.TypeOf(), 1) /* number of elements */) {
         /* -- An array. */
         if (G__funcheader) {
            /* -- In a function header. */
            /* Allocate no storage, we will share storage with caller (see the initializer below). */
            /*G__get_offset(var) = 0;*/
         } else {
            /* -- Not in a function header, allocate memory. */
            /* Note: If this is a static, no memory is allocated, a copy of the global var pointer is returned. */
            G__get_offset(var) = (char*) G__malloc(G__get_varlabel(var.TypeOf(), 1) /* number of elements */, SIZE, item);
            if (
               G__get_offset(var) && /* We have a pointer, and */
               !G__def_struct_member && /* We are not defining a member variable (the pointer is an offset), and */
               (G__asm_wholefunction == G__ASM_FUNC_NOP) && /* We are not bytecode compiling a function (the pointer is an offset), and */
               !(G__static_alloc && G__func_now && !G__prerun) /* Not a static variable in function scope at runtime. */
            ) {
               /*memset(G__get_offset(var), 0, G__get_varlabel(var.TypeOf(), 1) * SIZE);*/
            }
         }
      }
      else {
         /* -- Normal variable, allocate space. */
         /* Note: If this is a static, no memory is allocated, a copy of the global var pointer is returned. */
         G__get_offset(var) = (char*) G__malloc(1, SIZE, item);
         if (
            G__get_offset(var) && /* We have a pointer, and */
            !G__def_struct_member && /* We are not defining a member variable (the pointer is an offset), and */
            (G__asm_wholefunction == G__ASM_FUNC_NOP) && /* We are not bytecode compiling a function (the pointer is an offset), and */
            !(G__static_alloc && G__func_now && !G__prerun) /* Not a static variable in function scope at runtime. */
         ) {
            /*memset(G__get_offset(var), 0, SIZE);*/
         }
      }
      int varlabel_1 = G__get_varlabel(var.TypeOf(), 1);
      /* Now do initialization. */
      if (
         /* Variable has storage to initialize */
         (G__get_offset(var) || (varlabel_1 && G__funcheader) ) &&
         /* Not bytecode compiling */
         (G__asm_wholefunction == G__ASM_FUNC_NOP) &&
         (
            /* Not a member variable. */
            !G__def_struct_member ||
            /* Static, const, or enumerator member variable. */
            G__static_alloc ||
            /* Namespace member variable. */
            (G__get_properties(var)->statictype == G__LOCALSTATIC) ||
            /* Namespace member variable. */
            (G__def_tagnum.IsNamespace() && !G__def_tagnum.IsTopScope())
         ) &&
         /* Initialize const, static, and enumerator variables before running. */
         (!G__static_alloc || G__prerun) &&
         (
            /* Variable is not of class type or it is not pre-allocated. */
            (G__globalvarpointer == G__PVOID) ||
            /* Initializer is not void. */
            G__get_type(G__value_typenum(result))
         )
      ) {
         if (varlabel_1 /* number of elements */ == INT_MAX /* unspecified length flag */) {
            /* -- We are initializing an unspecified length array. */
            if (G__funcheader) {
               /* -- In a function header, we point at our actual argument. */
               G__get_offset(var) = (char*) G__int(result);
            } else {
               /* -- Syntax errror. */
            }
         }
         else if (varlabel_1 /* number of elements */) {
            /* -- We are initializing an array. */
            if (G__funcheader) {
               /* -- In a function header, we point at our actual argument. */
               G__get_offset(var) = (char*) G__int(result);
            } else {
               /* -- Syntax error. */
            }
         }
         else {
            /* -- We are initializing a non-array variable. */
            *((CASTTYPE*) G__get_offset(var)) = (CASTTYPE) (*f)(result);
         }
      }
   }
   else {
      /* -- Pointer or array of pointers. */
      /* Allocate memory */
      if (G__get_varlabel(var.TypeOf(), 1) /* number of elements */ == INT_MAX /* unspecified length flag */) {
         /* -- Unspecified length array of pointers. */
         if (G__funcheader) {
            /* -- In a function header.*/
            /* Allocate no storage, we will share it with the caller (see initializer below). */
            /*G__get_offset(var) = 0;*/
         }
         else {
            /* -- Not in a function header. */
            if (!G__static_alloc || G__prerun) {
               /* -- Not static, const, enum; or it is prerun time. */
               /* Allocate no storage (it will be allocated later during initializer parsing). */
               /*G__get_offset(var) = 0;*/
            }
            else {
               /* -- Static, const, or enum during execution. */
               /* Copy the pointer from the global variable, no actual allocation takes place. */
               G__get_offset(var) = (char*) G__malloc(1, G__LONGALLOC, item);
            }
         }
      }
      else if (G__get_varlabel(var.TypeOf(), 1) /* number of elements */) {
         /* -- Array of pointers. */
         if (G__funcheader) {
            /* -- In a function header.*/
            /* Allocate no storage, we will share it with the caller (see initializer below). */
            /*G__get_offset(var) = 0;*/
         }
         else {
            /* -- Not a function header. */
            /* Allocate storage for an array of pointers. */
            /* Note: If this is a static, no memory is allocated, a copy of the global var pointer is returned. */
            G__get_offset(var) = (char*) G__malloc(G__get_varlabel(var.TypeOf(), 1) /* number of elements */, G__LONGALLOC, item);
            if (
               G__get_offset(var) && /* We have a pointer, and */
               !G__def_struct_member && /* We are not defining a member variable (the pointer is an offset), and */
               (G__asm_wholefunction == G__ASM_FUNC_NOP) && /* We are not bytecode compiling a function (the pointer is an offset), and */
               !(G__static_alloc && G__func_now && !G__prerun) /* Not a static variable in function scope at runtime. */
            ) {
               /*memset((char*) G__get_offset(var), 0, G__get_varlabel(var.TypeOf(), 1) * G__LONGALLOC);*/
            }
         }
      }
      else {
         /* -- Normal pointer. */
         /* Allocate storage for a pointer. */
         /* Note: If this is a static, no memory is allocated, a copy of the global var pointer is returned. */
         G__get_offset(var) = (char*) G__malloc(1, G__LONGALLOC, item);
         if (
            (G__globalvarpointer == G__PVOID) && /* variable was *not* preallocated, nor a func ref param */
            G__get_offset(var) && /* We have a pointer, and */
            !G__def_struct_member && /* We are not defining a member variable (the pointer is an offset), and */
            (G__asm_wholefunction == G__ASM_FUNC_NOP) && /* We are not bytecode compiling a function (the pointer is an offset), and */
            !(G__static_alloc && G__func_now && !G__prerun) /* Not a static variable in function scope at runtime(do not overwrite a static). */
         ) {
            /**((long*) G__get_offset(var)) = 0L;*/
         }
      }
      /* Now do initialization. */
      int varlabel_1 = G__get_varlabel(var.TypeOf(), 1);
      if (
         /* Variable has storage to initialize */
         (G__get_offset(var) || (varlabel_1 && G__funcheader) ) &&
         (
            /* Not a class member and not bytecompiling. */
            (!G__def_struct_member && (G__asm_wholefunction == G__ASM_FUNC_NOP)) ||
            /* Static or const class member variable, not running. */
            (G__static_alloc && G__prerun) ||
            /* Namespace member. */
            (G__get_properties(var)->statictype == G__LOCALSTATIC) ||
            /* Namespace member variable. */
            (G__def_tagnum.IsNamespace() && !G__def_tagnum.IsTopScope())
         ) &&
         /* Initialize const and static variables before running. */
         (!G__static_alloc || G__prerun) &&
         (
            /* Variable is not of class type or it is not pre-allocated. */
            (G__globalvarpointer == G__PVOID) ||
            /* Initializer is not void. */
            G__get_type(G__value_typenum(result))
         )
      ) {
         if (varlabel_1 /* number of elements */ == INT_MAX /* unspecified length flag */) {
            /* -- We are initializing an unspecified length array of pointers. */
            if (G__funcheader) {
               /* -- In a function header, we point at our actual argument. */
               G__get_offset(var) = (char*) G__int(result);
            } else {
               /* -- Syntax errror. */
            }
         }
         else if (varlabel_1 /* number of elements */) {
            /* -- We are initializing an array of pointers. */
            if (G__funcheader) {
               /* -- In a function header, we point at our actual argument. */
               G__get_offset(var) = (char*) G__int(result);
            } else {
               /* -- Syntax error. */
            }
         }
         else {
            /* -- We are initializing a normal pointer. */
            *((long*) G__get_offset(var)) = (long) G__int(result);
         }
      }
   }
}


#endif // G__ASM_WHOLEFUNC

//______________________________________________________________________________
static G__value Cint::Internal::G__allocvariable(G__value result, G__value para[], const ::Reflex::Scope& varglobal, const ::Reflex::Scope& varlocal, int paran, int /* varhash */, const char* item, std::string& varname, int parameter00, Reflex::Member &output_var)
{
   // -- Allocate memory for a variable and initialize it.
   //
   //  Figure out which variable chain we will use.
   //
   ::Reflex::Scope varscope;
   if (G__p_local) {
      // -- equal to G__prerun == 0
      varscope = varlocal;
   }
   else {
      // -- equal to G__prerun == 1
      varscope = varglobal;
   }
   //
   //  Make sure we have a variable chain
   //  on which to append the new variable.
   //
   if (!varscope) {
      G__fprinterr(G__serr, "Error: Illegal assignment to %s", item);
      G__genericerror(0);
      return result;
   }
   //
   //  Data members of reference type are not
   //  supported by the interpreter.
   //
   if (G__def_struct_member && (G__reftype == G__PARAREFERENCE)) {
      if (
         (G__globalcomp == G__NOLINK) &&
         G__def_tagnum &&
         !G__def_tagnum.IsNamespace() &&
         (G__def_tagnum == G__tagdefining)
      ) {
         G__genericerror("Limitation: Reference member not supported. Please use pointer");
         return result;
      }
      else if (
         (G__access == G__PUBLIC) &&
         G__def_tagnum &&
         !G__def_tagnum.IsNamespace()
         // FIXME: Do we need to check G__def_tagnum == G__tagdefining like above?
      ) {
         G__fprinterr(G__serr, "Limitation: Reference member not accessible from the interpreter");
         G__printlinenum();
         G__access = G__PRIVATE;
      }
   }
   //
   //  If the new variable is a pointer to a class,
   //  check that the initializer is compatible.
   //
   if (
      (G__var_type == 'U') &&
      (G__get_type(G__value_typenum(result)) == 'U') &&
      (G__ispublicbase((::Reflex::Type) G__tagnum, G__value_typenum(result), (void*) G__STATICRESOLUTION2) == -1) &&
      (G__ispublicbase(G__value_typenum(result), (::Reflex::Type) G__tagnum, (void*) G__STATICRESOLUTION2) == -1)
   ) {
      G__fprinterr(G__serr, "Error: Illegal initialization of pointer, wrong type %s", G__type2string(G__get_type(G__value_typenum(result)), G__get_tagnum(G__value_typenum(result)), G__get_typenum(G__value_typenum(result)), G__get_reftype(G__value_typenum(result)), 0));
      G__genericerror(0);
      return result;
   }
   //
   //  Find the end of the variable chain.
   //
   //--
   //-- Not done in cint7.
   //--
   //
   //  Perform special actions for an automatic variable.
   //
   int autoobjectflag = 0;
   ::Reflex::Scope store_tagnum;
   ::Reflex::Type store_typenum;
   char* store_globalvarpointer = 0;
   int store_var_type = 0;
   if (
      G__automaticvar &&
      (G__var_type == 'p') &&
      !G__definemacro &&
      (G__globalcomp == G__NOLINK)
   ) {
      if (G__get_tagnum(G__value_typenum(result)) == -1) {
         // -- We are auto-allocating an object of a fundamental type.
         G__var_type = G__get_type(G__value_typenum(result));
         if (!G__const_noerror) {
            // To follow the example of other places. Should printlinenum
            // be replaced by G__genericerror?
            if (G__dispmsg >= G__DISPWARN) {
               G__fprinterr(G__serr, "Warning: Automatic variable %s is allocated", item);
               G__printlinenum();
            }
         }
      }
      else {
         // -- We are auto-allocating an object of class type.
         autoobjectflag = 1;
         if (G__IsInMacro()) {
            // Undeclared variable assignment of class/struct will create
            // a global object of pointer or reference.
            if (G__p_local) {
               // -- If we are using a local variable chain, switch to the global variable chain.
               varscope = varglobal;
               // Move to the end of the chain.
               //--
               //--
               //--
            }
         }
         store_var_type = G__var_type;
         store_tagnum = G__tagnum;
         store_typenum = G__typenum;
         store_globalvarpointer = G__globalvarpointer;
         ::Reflex::Type ty = G__value_typenum(result);
         G__var_type = G__get_type(ty);
         if (G__get_tagnum(ty) == -1) {
            G__tagnum = ::Reflex::Scope();
         }
         else {
            G__tagnum = ty.RawType();
         }
         G__typenum = ::Reflex::Type();
         if (isupper(G__get_type(ty))) {
            // a = new T(init);  a is a pointer
#ifndef G__ROOT
            G__fprinterr(G__serr, "Warning: Automatic variable %s* %s is allocated", G__fulltagname(G__get_tagnum(ty), 1), item);
            G__printlinenum();
#endif // G__ROOT
            G__reftype = G__PARANORMAL;
         }
         else {
            if (
               (G__get_tagnum(ty) == G__get_tagnum(G__value_typenum(G__p_tempbuf->obj))) &&
               (G__templevel == G__p_tempbuf->level)
            ) {
               // a = T(init); a is an object
               G__globalvarpointer = (char*) result.obj.i;
#ifndef G__ROOT
               G__fprinterr(G__serr, "Warning: Automatic variable %s %s is allocated", G__fulltagname(G__get_tagnum(ty), 1), item);
               G__printlinenum();
#endif // G__ROOT
               G__reftype = G__PARANORMAL;
               G__p_tempbuf->obj.obj.i = 0;
               G__pop_tempobject();
            }
            else {
               // T b;
               // a = b; a is a reference type
               G__reftype = G__PARAREFERENCE;
               if (G__asm_wholefunction == G__ASM_FUNC_NOP) {
                  G__fprinterr(G__serr, "Error: Illegal assignment to an undeclared symbol %s", item);
               }
               G__genericerror(0);
               G__tagnum = store_tagnum;
               G__typenum = store_typenum;
               G__globalvarpointer = store_globalvarpointer;
               G__var_type = store_var_type;
               return result;
            }
         }
      }
   }
   //
   //  Perform special actions for a bitfield variable.
   //
   int bitlocation = 0;
   if (G__bitfield) {
      unsigned int nvar = varscope.DataMemberSize();
      if (!nvar || !G__get_bitfield_width(varscope.DataMemberAt(nvar - 1))) {
         // the first element in the bit-field
         bitlocation = 0;
      }
      else {
         bitlocation = G__get_bitfield_start(varscope.DataMemberAt(nvar - 1)) + G__get_bitfield_width(varscope.DataMemberAt(nvar - 1));
         if (((int) (8 * G__INTALLOC)) < (bitlocation + G__bitfield)) {
            bitlocation = 0;
         }
      }
      if (G__bitfield == -1) {
         // unsigned int a : 4;
         // unsigned int   : 0; <- in case of this, new allocation unit for b
         // unsigned int b : 3;
         G__bitfield = (8 * G__INTALLOC) - bitlocation;
      }
   }
   //
   //  Allocate a new variable in the chain.
   //
   //--1
   //--2
   //--3
   //--4
   //--5
   //--6
   //--7
   //--8
   //
   //  Determine storage duration of the variable.
   //
   int var_statictype = G__AUTO;
   if (G__decl_obj == 2) {
      // -- We are a stack-allocated array of objects of class type.
      var_statictype = G__AUTOARYDISCRETEOBJ;
      if (
         (G__globalvarpointer != G__PVOID) && // We have preallocated memory, and
         !G__cppconstruct // Not in a constructor call (memory is object), and
      ) {
         // -- We have preallocated memory and we are not in a constructor call.
         var_statictype = G__COMPILEDGLOBAL;
      }
   }
   else if (
      G__def_struct_member && // We are a data member, and
      G__tagdefining && // Inner class tag is valid, and
      (G__tagdefining.IsNamespace() && !G__tagdefining.IsTopScope()) && // We are a namespace member, and
      (std::tolower(G__var_type) != 'p') // We are not a macro (macros are global)
      // FIXME: Is this 'p' test necessary?
      // FIXME: Do we need to check for type 'p' during scratch of a namespace?
   ) {
      // -- We are a namespace member.
      if ((G__globalvarpointer != G__PVOID) && !G__cppconstruct) {
         // -- We have an object preallocated, and we are not in a constructor call.
         // Note: Marking it this way means we will not free it during a scratch.
         var_statictype = G__COMPILEDGLOBAL;
      }
      else if (G__static_alloc) {
         // -- Static namespace member (affects visibility, not storage duration!).
         var_statictype = G__LOCALSTATIC;
      }
      else {
         // -- Otherwise leave as auto, even though it is not stack allocated.
      }
   }
   else if (G__static_alloc) {
      // -- We are a static, const, or enumerator variable.
      if (G__p_local) {
         // equal to G__prerun == 0
         // We are running, the local variable will be copied
         // from the global variable.
         G__set_static_varname(varname.c_str());
         // Function scope static variable
         // No real malloc(), get pointer from
         // global variable array which is suffixed
         // as varname\funcname.
         // Also, static class/struct member.
         var_statictype = G__LOCALSTATIC;
      }
      else {
         // equal to G__prerun == 1
         // We are not running, we are parsing.
         if (G__func_now) {
            // -- Function scope static variable.
            // Variable allocated to global variable
            // array named varname\funcname. The
            // variable can be exclusively accessed
            // with in a specific function.
            std::string temp;
#ifdef G__NEWINHERIT
            G__get_stack_varname(temp, varname.c_str(), G__func_now, G__get_tagnum(G__p_ifunc));
            //--
            //--
            //--
            //--
            //--
#else // G__NEWINHERIT
            // This space intentionally left blank.
            std::abort();
            //--
            //--
            //--
            //--
#endif // G__NEWINHERIT
            varname = temp;
            //--
            //--
            var_statictype = G__LOCALSTATIC;
         }
         else if (G__nfile < G__ifile.filenum) {
            // -- Semantic error, we are in a '{ }' style macro.
            if (G__dispmsg >= G__DISPWARN) {
               G__fprinterr(G__serr, "Warning: 'static' ignored in '{ }' style macro");
               G__printlinenum();
            }
            G__static_alloc = 0;
            if (
               (G__globalvarpointer != G__PVOID) && // We have preallocated memory, and
               !G__cppconstruct // Not in a constructor call (memory is object), and
            ) {
               // -- We have preallocated memory and we are not in a constructor call, and we are not a static, const, or enum.
               var_statictype = G__COMPILEDGLOBAL;
            }
         }
         else {
            // -- File scope static variable
            var_statictype = G__ifile.filenum;
            if (
               (G__globalvarpointer != G__PVOID) && // We have preallocated memory, and
               !G__cppconstruct // Not in a constructor call (memory is object), and
            ) {
               // -- We have preallocated memory and we are not in a constructor call.
               var_statictype = G__COMPILEDGLOBAL;
            }
         }
      }
   }
   else if ((G__globalvarpointer != G__PVOID) && !G__cppconstruct) {
      // -- We have preallocated memory and we are not in a constructor call.
      // Note: Marking it this way means we will not free it during a scratch.
      var_statictype = G__COMPILEDGLOBAL;
   }
   //
   //  Determine class member access control.
   //
   int var_access = G__PUBLIC;
   if (G__def_struct_member) {
      var_access = G__access;
   }
#ifndef G__NEWINHERIT
   //--
#endif // G__NEWINHERIT
   //
   //  Determine whether or not to generate a dictionary entry for the variable.
   //
   int var_globalcomp = G__NOLINK;
   if (!varscope.IsClass()) {
      // -- Variable is not a data member.
      var_globalcomp = G__NOLINK;
      if (G__default_link) {
         var_globalcomp = G__globalcomp;
      }
   }
   else {
      // -- Variable is a data member.
      var_globalcomp = G__globalcomp;
   }
   //
   //  Remember array bounds.
   //
   std::vector<int> var_varlabel;
   if (paran || G__typedefnindex) {
      long bound = 0L;
      if (paran && parameter00) {
         // -- We have a specified array bound.
         bound = G__int(para[0]);
         var_varlabel.push_back(bound);
      } else if (paran && !parameter00 && !G__typedefnindex) {
         // -- We are an unspecified length array.
         var_varlabel.push_back(INT_MAX);
      }
      for (int i = 1; i < paran; ++i) {
         // -- Convert and store the rest of the array bounds.
         bound = G__int(para[i]);
         var_varlabel.push_back(bound);
      }
      for (int i = 0; i < G__typedefnindex; ++i) {
         // -- Convert and store the array bounds from an underlying typedef.
         bound = G__typedefindex[i];
         var_varlabel.push_back(bound);
      }
      if (G__funcheader && G__asm_wholefunction && G__get_type(G__value_typenum(result))) {
         // -- We cannot support function parameters of array type in whole function bytecode compilation.
         // FIXME: This cannot happen because G__asm_wholefunction and G__funcheader are never both set???
         // FIXME: Nope that can happen, but when it does the initializer (result) is of invalid type.
         G__ASSERT(G__globalvarpointer == G__PVOID);
         G__abortbytecode();
         G__genericerror(0);
      }
   }
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //
   // Pointer to array reimplementation.
   //
   // FIXME: We have no way to support the cint5 pointer to array hack!  Ignore G__p2arylabel for now.
   //
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //
   //--
   //
   //  Convert an array variable to a pointer variable
   //  if we are doing whole function bytecode generation
   //  so that the bytecode generated for initializing
   //  function parameters can work.
   //
   //  What happens is that an array argument in a function
   //  call is passed as a pointer to the element type, and
   //  in whole function bytecode the assignment to the
   //  function parameter variable is done as a local variable
   //  assignment.  In order for the assignment to succeed
   //  the function parameter must be of pointer type.
   //
   //  All of this gross hackery is due to the fact that
   //  a G__value cannot represent an array type and
   //  function arguments are passed on the bytecode data
   //  stack as G__value's.
   //
   //
   if (
      // -- An array func param, generating bytecode for a whole function, and not static, const, or enum.
      G__funcheader && // in a function parameter declaration, and
      (
         var_varlabel.size() // we are an array,
         //--
      ) && // and
      G__asm_wholefunction && // generating bytecode for a whole function, and
      !G__static_alloc // not a static, const, or enum
   ) {
      // -- Change the array variable to a pointer variable.
      var_varlabel.clear();
      paran = 0;
      //--
      //--
      if (std::islower(G__var_type)) {
         G__var_type = toupper(G__var_type);
      }
      else {
         switch (G__reftype) {
            case G__PARANORMAL:
               G__reftype = G__PARAP2P;
               break;
            case G__PARAP2P:
               G__reftype = G__PARAP2P2P;
               break;
            default:
               ++G__reftype;
               break;
         }
      }
   }
   if (G__funcheader && !var_varlabel.empty()) {
      // -- We are a function parameter of array type.
      // We share our storage with our caller, on function
      // exit, we must not free the memory.
      var_statictype = G__COMPILEDGLOBAL;
      G__globalvarpointer = (char*) result.obj.i;
   }
   //
   //  Do special fixups for an automatic object.
   //
   if (autoobjectflag) {
      var_statictype = G__AUTO;
   }
   //
   //  Store comment information which is specific to root.
   //
   G__comment_info var_comment;
   if (G__setcomment) { // If the dictionary interface is passing us a comment string.
      var_comment.p.com = G__setcomment; // Note: We do not take ownership, we make a copy of the char pointer.  This string must have a lifetime matching the variable!
      var_comment.filenum = -2; // Flag that comment is an immediate string, not in a source file.
   }
   else {
      var_comment.p.com = 0;
      var_comment.filenum = -1;
   }
   //
   //  Store file number and line number of declaration.
   //
#ifdef G__VARIABLEFPOS
   //G__get_properties(var)->filenum = G__ifile.filenum;
   //G__get_properties(var)->linenum = G__ifile.line_number;
#endif // G__VARIABLEFPOS
   //
   //  Complain if the variable name has a minus or plus in it.
   //
   {
      const char* pp = strchr(varname.c_str(), '-');
      if (!pp) {
         pp = strchr(varname.c_str(), '+');
      }
      if (pp) {
         G__fprinterr(G__serr, "Error: Variable name has bad character '%s'", varname.c_str());
         G__genericerror(0);
      }
   }
   //--
   //--
   //--
   //
   //  Start to finalize variable type.
   //
   //
   //  Remember whether or not variable is of pointer type.
   //
   bool var_ispointer = isupper(G__var_type);
   //--
   //--
   //--
   //--
   //--
   //--
   //--
   //--
   //
   //  Store the reference type (void is special).
   //
   int var_reftype = G__reftype;
   if (
      // --
#ifndef G__OLDIMPLEMENTATION2191
      G__var_type == '1'
#else // G__OLDIMPLEMENTATION2191
      G__var_type == 'Q'
#endif // G__OLDIMPLEMENTATION2191
      // --
   ) {
      // -- Type void is special.
      var_reftype = G__PARANORMAL;
   }
   //
   //  Accumulate G__dynconst into G__constvar and clear it.
   //
   //
   //
   //  Note: G__dynconst is set by G__lock_variable
   //        and by variable initialization when
   //        the initializer is a function call
   //        to force the variable to be treated
   //        specially for thread safety (not supported).
   //
   G__constvar |= G__dynconst;
   G__dynconst = 0;
   //
   //  Store the constness of the variable.
   //
   int var_constvar = (G__SIGNEDCHAR_T) G__constvar;
   //
   //  Setup base type.
   //
   ::Reflex::Type var_type;
   bool typeFromTypenum = false;
   if (G__typenum) { // We are of typdef type, the modifiers apply to the typedef.
      var_type = G__typenum; // We are of typdef type, the modifiers apply to the typedef.
      typeFromTypenum = true;
   } else if (G__tagnum && !G__tagnum.IsTopScope()) { // We are a class, enum, struct, or union, or a namespace, the modifiers apply to that.
      var_type = G__tagnum; // We are a class, enum, struct, or union, or a namespace, the modifiers apply to that.
   } else { // We are of fundamental type.
      var_type = G__get_from_type(G__var_type, 0); // We are of fundamental type.
   }
   //
   //  Do special processing for macro types, and auto types.
   //
   switch (G__var_type) {
      case 'p':
      case 'o':
         if (G__isdouble(result)) {
            var_type = G__get_from_type(toupper(G__var_type), 0);
            typeFromTypenum = false;
         }
      case 'P':
      case 'O':
         // Automatic variable and macro
         //   p : macro int
         //   P : macro double
         //   o : auto int
         //   O : auto double
         // if not macro definition, print out warning
         if (G__get_type(G__value_typenum(result)) == 'C') {
            var_type = G__get_from_type('T', 0);
            typeFromTypenum = false;
            // var_ispointer = true;
         }
         if (!G__definemacro && (G__globalcomp == G__NOLINK)) {
            if (!var_type.RawType().IsClass()) {
               if (G__dispmsg >= G__DISPWARN) {
                  G__fprinterr(G__serr, "Warning: Undeclared data member %s", item);
                  G__genericerror(0);
               }
               return result;
            }
            if (!G__const_noerror) {
               G__fprinterr(G__serr, "Error: Undeclared variable %s", item);
               G__genericerror(0);
            }
            var_type = G__get_from_type('o', 0);
            typeFromTypenum = false;
         }
         break;
      default:
         break;
   }
   //
   //  Accumulate array bounds, and pointer, and reference qualifications into the base variable type.
   //
   if (typeFromTypenum) { // Information from the typedef was added to the modifiers, so we need to undo that.
      // FIXME we should probably change the rest of the code to avoid having to undo it here!!
      ::Reflex::Type final = var_type.FinalType();
      //FIXME: Is this right, cint5 ignores const from a typedef?
      if (final.IsConst() && var_constvar) {
         var_constvar = 0; // humm doubtful, var_constvar contains more information than that.
      }
      if (final.IsPointer() && var_ispointer) { 
         switch (G__get_reftype(final)) {
            case G__PARANORMAL:
            case G__PARAREFERENCE:
               if (var_ispointer) {
                  switch (var_reftype) {
                     case G__PARAREFERENCE:
                     case G__PARANORMAL:
                        var_ispointer = false;
                        break;
                     case G__PARAP2P:
                        var_reftype = G__PARANORMAL;
                        break;
                     default:
                        --var_reftype;
                        break;
                  }
               }
               break;
            default:
               if (G__get_reftype(final) == var_reftype) {
                  var_reftype = G__PARANORMAL;
                  var_ispointer = false;
               }
               else if ((G__get_reftype(final) + 1) == var_reftype) {
                  var_reftype = G__PARANORMAL;
               }
               else if (G__get_reftype(final) < var_reftype) {
                  var_reftype = G__PARAP2P + var_reftype - G__get_reftype(final) - 2;
               }
               break;
         }
      }
   }
   // Note: We add only the first paran elements to the varlabel, the rest are from the typedef.
   var_type = G__modify_type(var_type, var_ispointer, var_reftype, var_constvar, paran, var_varlabel.empty() ? 0 : &var_varlabel[0]);
   //
   //  A variable of type void is not allowed.
   //
   if (
      !var_ispointer &&
      (G__var_type != '1') &&
      (G__var_type != 'Q') &&
      (::Reflex::Tools::FundamentalType(var_type.FinalType()) == ::Reflex::kVOID)
   )  {
      // -- Do *not* create void variable!
      G__genericerror("Error: void type variable can not be declared");
      return result;
   }
   //
   //  A variable of abstract class type is not allowed.
   //
   if (var_type.IsAbstract()) {
      // -- Do *not* create abstract class variable!
      G__fprinterr(G__serr, "Error: 4018: abstract class object '%s %s' declared", G__tagnum.Name(::Reflex::SCOPED).c_str(), item);
      G__genericerror(0);
      G__display_purevirtualfunc(G__get_tagnum(G__tagnum));
      return result;
   }
   //
   //  Create the variable.
   //
   {
      //
      //  Lookup variable in scope to see if reflex already knows
      //  about it.  This can happen if we are called from a reflex
      //  builder callback through cintex.
      //
#if 0
      int dm_size = varscope.DataMemberSize();
      const char* varname_c_str = varname.c_str();
      for (int i = 0; i < dm_size; ++i ) {
         ::Reflex::Member mbr = varscope.DataMemberAt(i);
         if (!strcmp(mbr.Name_c_str(), varname_c_str)) {
            output_var = mbr;
            break;
         }
      }
#endif // 0
      output_var = varscope.DataMemberByName(varname, ::Reflex::INHERITEDMEMBERS_NO);
   }
   if (
      !output_var || // var does not exist, or
      (
         !G__in_memvar_setup && // we are not being called by dictionary setup, and
         (
            G__static_alloc && // we are doing a static variable, and
            G__func_now // it is a function-local static
         )
      )
   ) {
      size_t reflex_offset = 0;
      char* var_offset = 0;
      int reflex_modifiers = 0;
      // Create the variable as a data member of its continaing scope, either a namespace or a class.
      output_var = G__add_scopemember(varscope, varname.c_str(), var_type, reflex_modifiers, reflex_offset, var_offset, var_access, var_statictype);
   }
   else if (!G__in_memvar_setup && (varname != "G__virtualinfo")) {
      assert(0); // Not in dict interface, and variable is found, should be impossible.
   }
   //
   //  Set variable properties.
   //
   G__get_properties(output_var)->statictype = var_statictype;
   G__get_properties(output_var)->comment = var_comment;
   G__get_properties(output_var)->globalcomp = var_globalcomp;
#ifdef G__VARIABLEFPOS
   //
   //  Store file number and line number of declaration.
   //
   G__get_properties(output_var)->filenum = G__ifile.filenum;
   G__get_properties(output_var)->linenum = G__ifile.line_number;
#endif // G__VARIABLEFPOS
   //
   //  Fetch info about number of array elements we are declared to have.
   //
   int num_elements = G__get_varlabel(output_var.TypeOf(), 1);
   if (num_elements == INT_MAX) {
      num_elements = 0;
   }
   //
   //  Do special processing for an enumerator, change type code.
   //
   if (
      (G__var_type == 'p') && !G__macro_defining &&
      (
         G__prerun || // not executing
         G__static_alloc || // static
         G__constvar || // const
         G__def_struct_member // member variable
      )
   ) {
      // -- Enumerator.
      G__var_type = 'l';
   }
   //
   //  If we are assigning an object of class type
   //  to this variable, and the variable is of a
   //  fundamental type, try to do a type conversion.
   //
   if (
      // -- Initializer is of class type and var is fundamental
      (std::tolower(G__get_type(output_var.TypeOf())) != 'u') &&
      (G__get_type(G__value_typenum(result)) == 'u') &&
      G__value_typenum(result).RawType().IsClass()
   ) {
      // -- Try to convert the initializer (result), which is of class type, to the type of the variable.
      int store_decl = G__decl;
      G__decl = 0;
      //--
      G__fundamental_conversion_operator(G__get_type(var_type), G__get_tagnum(var_type), var_type, G__get_reftype(var_type), G__get_isconst(var_type), &result);
      G__decl = store_decl;
   }
   //
   // Bytecode generation.
   //
#ifdef G__ASM
   //
   //  Handle whole function compilation.
   //  The following part must handle function
   //  parameter initialization, and regular
   //  variable initialization for whole function
   //  compilation.
   //
#ifdef G__ASM_WHOLEFUNC
   //
   //  Do not allow a whole function to be bytecompiled
   //  if it has a local variable of class type, or of
   //  pointer to class type, or of unspecified length
   //  array type.
   //
   // The following line is temporary, limitation for having class object as local variable.
   if (G__asm_wholefunction) {
      if (
         // -- G__funcheader case must be implemented, the following line is deleted.
         (
            (G__var_type == 'u') && (G__reftype != G__PARAREFERENCE)
#ifndef G__OLDIMPLEMENTATION1073
            && G__funcheader
#endif // G__OLDIMPLEMENTATION1073
         )
         //--
      ) {
         if (!G__xrefflag) {
            G__abortbytecode();
            G__asm_wholefunc_default_cp = 0;
            G__no_exec = 1; // FIXME: Is this right?
            G__return = G__RETURN_IMMEDIATE;
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "!!!bytecode compile aborted by automatic class object. Use pointer to class obj + new");
               G__printlinenum();
            }
#endif // G__ASM_DBG
            return result;
         }
      }
   }
   // Recover from masking of default parameter evaluation. // FIXME: Is this right?
   if (G__asm_wholefunc_default_cp) {
      // -- Default param eval masked for bytecode func compilation, recovered.
      G__asm_noverflow = 1;
   }
   // Perform work, if needed.
   if (G__asm_noverflow && G__asm_wholefunction) {
      // -- Generate bytecode for variable initialization during whole function compilation.
      if (
         // -- The variable is not an array.
         !num_elements &&
         (G__get_varlabel(output_var.TypeOf(), 1) /* number of elements */ != INT_MAX /* unspecified length flag */)
      ) {
         // -- Handle variables which are not of array type.
         if (G__funcheader) {
            // -- We are a function parameter declaration.
            if (G__reftype != G__PARAREFERENCE) {
               // -- Initialize a non-reference parameter.
               G__asm_gen_stvar(0, output_var, G__get_paran(output_var), item, G__ASM_VARLOCAL, 'p', &result);
            }
            else {
               // -- Initialize a reference parameter.
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: INIT_REF  paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, paran, G__var_type, (long) output_var.Id(), __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__INIT_REF;
               G__asm_inst[G__asm_cp+1] = 0;
               G__asm_inst[G__asm_cp+2] = paran;
               G__asm_inst[G__asm_cp+3] = G__var_type;
               G__asm_inst[G__asm_cp+4] = (long) output_var.Id();
               G__inc_cp_asm(5, 0);
            }
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: POP  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__POP;
            G__inc_cp_asm(1, 0);
         }
         else if (G__get_type(G__value_typenum(result)) && !G__static_alloc) {
            // -- We have an initializer and we are not a static or const variable.
            G__asm_gen_stvar(0, output_var, paran, item, G__ASM_VARLOCAL, 'p', &result);
         }
#ifndef G__OLDIMPLEMENTATION1073
         else if (
            (G__var_type == 'u') &&
            (G__reftype != G__PARAREFERENCE) &&
            G__tagnum &&
            !G__tagnum.IsEnum()
         ) {
            if (G__struct.iscpplink[G__get_tagnum(G__tagnum)] == G__CPPLINK) {
               // precompiled class
               // Move LD_FUNC instruction
               G__inc_cp_asm(-6, 0);
               for (int i = 5; i > -1; --i) {
                  G__asm_inst[G__asm_cp+i+4] = G__asm_inst[G__asm_cp+i];
               }
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: CTOR_SETGVP %s paran=%d  %s:%d\n", G__asm_cp, G__asm_dt, item, paran, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__CTOR_SETGVP;
               G__asm_inst[G__asm_cp+1] = 0L;
               G__asm_inst[G__asm_cp+2] = (long) output_var.Id();
               G__asm_inst[G__asm_cp+3] = 0L; // This is the 'mode'. I am not sure what it should be.
               G__inc_cp_asm(4, 0);
               G__inc_cp_asm(6, 0); // increment for moved LD_FUNC instruction.
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__PUSHSTROS;
               G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__SETSTROS;
               G__inc_cp_asm(1, 0);
            }
            else {
               // -- Interpreted class.
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD_VAR  item: '%s' paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, item, paran, 'p', (long) output_var.Id(), __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD_LVAR;
               G__asm_inst[G__asm_cp+1] = 0;
               G__asm_inst[G__asm_cp+2] = paran;
               G__asm_inst[G__asm_cp+3] = 'p';
               G__asm_inst[G__asm_cp+4] = (long) output_var.Id();
               G__inc_cp_asm(5, 0);
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__PUSHSTROS;
               G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__SETSTROS;
               G__inc_cp_asm(1, 0);
            }
         }
#endif // G__OLDIMPLEMENTATION1073
         // --
      }
      else {
         // -- Array initialization.
         if (G__funcheader) {
            // -- Array initialization for a function parameter.
            if (!G__static_alloc) {
               // -- We are not a static, const, or enumerator variable.
               // Example:  f(int var[2][3][2], ...){...};
               if (G__get_varlabel(output_var.TypeOf(), 1) /* number of elements */ == INT_MAX /* unspecified length flag */) {
                  // -- This is a parameter of unspecified length array type, handle like a pointer.
                  // Example:  f(int var[], ...){...};
                  G__asm_gen_stvar(0, output_var, 0, item, G__ASM_VARLOCAL, 'p', &result);
               }
               else {
                  // -- This is *not* a parameter of unspecified length array type.
                  // Example:  f(int var[2][3][2], ...){...};
                  G__asm_gen_stvar(0, output_var, 0, item, G__ASM_VARLOCAL, 'p', &result);
               }
            }
            else {
               // -- We are a static or const variable.
               // Example: f(const int var[2][3][2], ...){...};
               // FIXME: We should probably abort bytecode generation here!
            }
         }
         else {
            // -- Array initialization for a normal variable.
            //if (result.type && !G__static_alloc) {
            //   // -- We have an initializer and we are not a static or const variable.
            //   // FIXME: Can this code ever run?  int var[2][3][2] = 0;
            //   G__asm_gen_stvar(0, ig15, paran, var, item, G__ASM_VARLOCAL, 'p', &result);
            //}
            //else if (result.type && G__static_alloc) {
            //   // -- We have an initializer and we are a static, const, or enum variable.
            //   // FIXME: Can this code ever run?  static int var[2][3][2] = 0;
            //   // FIXME: We should probably abort bytecode generation here!
            //}
         }
      }
   }
#endif // G__ASM_WHOLEFUNC
   //
   //  Not doing whole function compilation,
   //  not doing function parameters.
   //  This part should ignore function parameters
   //  and handle all other variables if not doing
   //  whole function compilation.
   //
   if (
      G__asm_noverflow &&
      !G__funcheader
#ifdef G__ASM_WHOLEFUNC
      && (G__asm_wholefunction == G__ASM_FUNC_NOP)
#endif // G__ASM_WHOLEFUNC
      // --
   ) {
      if (G__get_type(G__value_typenum(result))) {
         G__asm_gen_stvar(0, output_var, paran, item, 0, 'p', &result);
      }
      else if (G__var_type == 'u') {
         G__ASSERT(!G__decl || (G__decl == 1));
         if (G__decl) {
            if (G__reftype) {
               G__redecl(output_var);
               if (G__no_exec_compile) {
                  G__abortbytecode();
               }
            }
            else {
               G__class_2nd_decl_i(output_var);
            }
         }
         else if (G__cppconstruct) {
            G__class_2nd_decl_c(output_var);
         }
      }
   }
#endif // G__ASM
   //
   //  Security check
   //
   G__CHECK(G__SECURE_POINTER_INSTANTIATE, isupper(G__var_type) && 'E' != G__var_type, return(result));
   G__CHECK(G__SECURE_POINTER_TYPE, isupper(G__var_type) && result.obj.i && G__var_type != G__get_type(G__value_typenum(result)) && !G__funcheader && (('Y' != G__var_type && result.obj.i) || G__security & G__SECURE_CAST2P), return(result));
   G__CHECK(G__SECURE_FILE_POINTER, 'E' == G__var_type, return(result));
   // G__CHECK(G__SECURE_ARRAY, num_elements > 0, return(result));
   //
   // Allocate memory to hold the variable value.
   //
   // Note: This part must correspond to G__ALLOC_VAR_REF.
   //
   //
   //  Handle bitfields first.
   //
   if (G__bitfield) {
      G__get_properties(output_var)->bitfield_start = bitlocation;
      G__get_properties(output_var)->bitfield_width = G__bitfield;
      G__bitfield = 0;
   }
   if (G__get_bitfield_width(output_var)) {
      //--
      //--
      if (!G__get_bitfield_start(output_var)) {
         G__get_offset(output_var) = (char*) G__malloc(1, G__INTALLOC, item);
      }
      else {
         G__get_offset(output_var) = (char*) G__malloc(1, 0, item) - G__INTALLOC;
      }
      return result;
   }
   //
   //  Now allocate memory to hold the value
   //  of the variable, and possibly initialize it.
   //
   // Mark where we are going to allocate the memory (bytecode arena or not).
   // Need to explicitly call operator bool to avoid the call to operator && with 2 arguments which leads to
   // the 2nd expression to be evaluated first!
   if (G__tagdefining) {
      bool tmp = G__get_properties(G__tagdefining)->isBytecodeArena;
      G__get_properties(output_var)->isBytecodeArena = tmp;
   }
   switch (G__var_type) {
      case 'u':
         // struct, union
         if (
            G__struct.isabstract[G__get_tagnum(G__tagnum)] &&
            !G__ansiheader &&
            !G__funcheader &&
            (G__reftype == G__PARANORMAL) &&
            ((G__globalcomp != G__CPPLINK) || (G__tagdefining != G__tagnum))
         ) {
            G__fprinterr(G__serr, "Error: Attempt to allocate memory for data member of an abstract class: '%s::%s'  %s:%d", G__tagnum.Name(Reflex::SCOPED).c_str(), item, __FILE__, __LINE__);
            G__genericerror(0);
            G__fprinterr(G__serr, "Error: Pure virtual functions are:  %s:%d\n", __FILE__, __LINE__);
            G__display_purevirtualfunc(G__get_tagnum(G__tagnum));
            // --
         }
         // type var; normal variable
         G__get_offset(output_var) = (char*) G__malloc(num_elements ? num_elements : 1, G__struct.size[G__get_tagnum(G__tagnum)], item);
         if (
            G__ansiheader &&
            G__get_type(G__value_typenum(result)) &&
            (G__globalvarpointer == G__PVOID) &&
            (!G__static_alloc || !G__func_now)
         ) {
            std::memcpy(G__get_offset(output_var), (void*) G__int(result), output_var.TypeOf().SizeOf());
         }
         result.obj.i = reinterpret_cast<long>(G__get_offset(output_var));
         break;
      case 'U':
         // pointer to struct, union
         if ((num_elements > 0) && G__get_type(G__value_typenum(result))) {
            // char* argv[];
            G__get_offset(output_var) = (char*) G__int(result);
         }
         else {
            G__get_offset(output_var) = (char*) G__malloc(num_elements ? num_elements : 1, G__LONGALLOC, item);
            if (
               (G__asm_wholefunction == G__ASM_FUNC_NOP) &&
               !G__def_struct_member &&
               (!G__static_alloc || G__prerun) &&
               ((G__globalvarpointer == G__PVOID) || G__get_type(G__value_typenum(result)))
            ) {
               int baseoffset = G__ispublicbase(output_var.TypeOf(), G__value_typenum(result), (void*) result.obj.i);
               if (baseoffset != -1) {
                  *((long*) G__get_offset(output_var)) = G__int(result) + baseoffset;
               }
               else {
                  *((long*) G__get_offset(output_var)) = G__int(result);
               }
            }
         }
         // Ensure returning 0 for not running constructor.
         if (!autoobjectflag) {
            result.obj.i = 0;
         }
         break;
#ifdef G__ROOT
      case 'Z':
         // ROOT special object.
         G__get_offset(output_var) = (char*) malloc(2 * G__LONGALLOC);
         *((long*) G__get_offset(output_var)) = 0L;
         *((long*) (G__get_offset(output_var) + G__LONGALLOC)) = 0L;
         break;
#endif // G__ROOT
#ifndef G__OLDIMPLEMENTATION2191
      case '1':
         // void
         G__alloc_var_ref<char>(G__CHARALLOC, G__int, item, output_var, result);
         break;
#else // G__OLDIMPLEMENTATION2191
      case 'Q':
         // void
         G__alloc_var_ref<char>(G__CHARALLOC, G__int, item, output_var, result);
         break;
#endif // G__OLDIMPLEMENTATION2191
      case 'Y':
         // pointer to void
         G__alloc_var_ref<char>(G__CHARALLOC, G__int, item, output_var, result);
         break;
      case 'E':
         // pointer to FILE
         G__alloc_var_ref<char>(G__CHARALLOC, G__int, item, output_var, result);
         break;
      case 'C':
         // pointer to char
         G__alloc_var_ref<char>(G__CHARALLOC, G__int, item, output_var, result);
         break;
      case 'c':
         // char
         G__alloc_var_ref<char>(G__CHARALLOC, G__int, item, output_var, result);
         // Check for initialization of a character variable with a string constant.
         if ((G__var_type == 'c') && G__get_varlabel(output_var.TypeOf(), 1) /* number of elements */ && (G__get_type(G__value_typenum(result)) == 'C')) {
            // -- We are a char array being initialized with a string constant.
            if (G__asm_wholefunction != G__ASM_FUNC_COMPILE) {
               // -- Note: In whole function compilation G__get_offset(var) is an offset.
               if (
                  !G__funcheader && // Not in a function header (we share value with caller), and
                  !G__def_struct_member && // Not defining a member variable (G__get_offset(var) is an offset), and
                  !(G__static_alloc && G__func_now && !G__prerun) // Not a static variable in function scope at runtime (init was done in prerun).
               ) {
                  int len = strlen((char*) result.obj.i);
                  if (G__get_varlabel(output_var.TypeOf(), 1) /* number of elements */ == INT_MAX /* unspecified length flag */) {
                     // -- We are an unspecified length array of char being initialized with a string constant.
                     // FIXME: Can this happen?
                     G__get_offset(output_var) = (char*) malloc(len + 1);
                     strcpy(G__get_offset(output_var), (char*) result.obj.i);
                  }
                  else if (len > G__get_varlabel(output_var.TypeOf(), 1) /* number of elements */) {
                     // -- We are an array of char being initialized with a string constant that is too big.
                     // FIXME: Can this happen?
                     // FIXME: We need to give an error message here!
                     strncpy(G__get_offset(output_var), (char*) result.obj.i, G__get_varlabel(output_var.TypeOf(), 1) /* number of elements */);
                  }
                  else {
                     // -- We are an array of char being initialized with a string constant.
                     // FIXME: Can this happen?
                     strcpy(G__get_offset(output_var), (char*) result.obj.i);
                     int num_omitted = G__get_varlabel(output_var.TypeOf(), 1) /* number of elements */ - len;
                     memset(G__get_offset(output_var) + len, 0, num_omitted);
                  }
               }
            }
            else {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD_VAR '%s' paran: %d type: 'P'  %s:%d\n", G__asm_cp, G__asm_dt, output_var.Name(::Reflex::SCOPED).c_str(), 0, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD_LVAR;
               G__asm_inst[G__asm_cp+1] = 0; // index
               G__asm_inst[G__asm_cp+2] = 0; // paran
               G__asm_inst[G__asm_cp+3] = 'P'; // type
               G__asm_inst[G__asm_cp+4] = (long) output_var.Id();
               G__inc_cp_asm(5, 0);
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: SWAP  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__SWAP;
               G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD_FUNC 'strcpy' %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD_FUNC;
               G__asm_inst[G__asm_cp+1] = 1 + 10*677; // LD_FUNC flag and hash
               G__asm_inst[G__asm_cp+2] = (long) ("strcpy"); // name
               G__asm_inst[G__asm_cp+3] = 2; // paran
               G__asm_inst[G__asm_cp+4] = (long) G__compiled_func_cxx;
               G__asm_inst[G__asm_cp+5] = 0;
               G__inc_cp_asm(6, 0);
            }
         }
         break;
      case 'n':
         G__alloc_var_ref<G__int64>(G__LONGLONGALLOC, G__Longlong, item, output_var, result);
         break;
      case 'N':
         G__alloc_var_ref<G__int64>(G__LONGLONGALLOC, G__Longlong, item, output_var, result);
         break;
      case 'm':
         G__alloc_var_ref<G__int64>(G__LONGLONGALLOC, G__Longlong, item, output_var, result);
         break;
      case 'M':
         G__alloc_var_ref<G__int64>(G__LONGLONGALLOC, G__Longlong, item, output_var, result);
         break;
#ifndef G__OLDIMPLEMENTATION2191
      case 'q':
      case 'Q':
         G__alloc_var_ref<long double>(G__LONGDOUBLEALLOC, G__Longdouble, item, output_var, result);
         break;
#endif // G__OLDIMPLEMENTATION2191
      case 'g':
         // bool
         result.obj.i = G__int(result) ? 1 : 0;
#ifdef G__BOOL4BYTE
         G__alloc_var_ref<int>(G__INTALLOC, G__int, item, output_var, result);
#else // G__BOOL4BYTE
         G__alloc_var_ref<unsigned char>(G__CHARALLOC, G__int, item, output_var, result);
#endif // G__BOOL4BYTE
         break;
      case 'G':
         // bool pointer
#ifdef G__BOOL4BYTE
         G__alloc_var_ref<int>(G__INTALLOC, G__int, item, output_var, result);
#else // G__BOOL4BYTE
         G__alloc_var_ref<unsigned char>(G__CHARALLOC, G__int, item, output_var, result);
#endif // G__BOOL4BYTE
         break;
      case 'b':
         // unsigned char
         G__alloc_var_ref<unsigned char>(G__CHARALLOC, G__int, item, output_var, result);
         break;
      case 'B':
         // pointer to unsigned char
         G__alloc_var_ref<unsigned char>(G__CHARALLOC, G__int, item, output_var, result);
         break;
      case 's':
         // short int
         G__alloc_var_ref<short>(G__SHORTALLOC, G__int, item, output_var, result);
         break;
      case 'S':
         // pointer to short int
         G__alloc_var_ref<short>(G__SHORTALLOC, G__int, item, output_var, result);
         break;
      case 'r':
         // unsigned short int
         G__alloc_var_ref<unsigned short>(G__SHORTALLOC, G__int, item, output_var, result);
         break;
      case 'R':
         // pointer to unsigned short int
         G__alloc_var_ref<unsigned short>(G__SHORTALLOC, G__int, item, output_var, result);
         break;
      case 'i':
         // int
         G__alloc_var_ref<int>(G__INTALLOC, G__int, item, output_var, result);
         break;
      case 'I':
         // pointer to int
         G__alloc_var_ref<int>(G__INTALLOC, G__int, item, output_var, result);
         break;
      case 'h':
         // unsigned int
         G__alloc_var_ref<unsigned int>(G__INTALLOC, G__int, item, output_var, result);
         break;
      case 'H':
         // pointer to unsigned int
         G__alloc_var_ref<unsigned int>(G__INTALLOC, G__int, item, output_var, result);
         break;
      case 'l':
         // long int
         G__alloc_var_ref<long>(G__LONGALLOC, G__int, item, output_var, result);
         break;
      case 'L':
         // pointer to long int
         G__alloc_var_ref<long>(G__LONGALLOC, G__int, item, output_var, result);
         break;
      case 'k':
         // unsigned long int
         G__alloc_var_ref<unsigned long>(G__LONGALLOC, G__int, item, output_var, result);
         break;
      case 'K':
         // pointer to unsigned long int
         G__alloc_var_ref<unsigned long>(G__LONGALLOC, G__int, item, output_var, result);
         break;
      case 'f':
         // float
         G__alloc_var_ref<float>(G__FLOATALLOC, G__double, item, output_var, result);
         break;
      case 'F':
         // pointer to float
         G__alloc_var_ref<float>(G__FLOATALLOC, G__double, item, output_var, result);
         break;
      case 'd':
         // double
         G__alloc_var_ref<double>(G__DOUBLEALLOC, G__double, item, output_var, result);
         break;
      case 'D':
         // pointer to double
         G__alloc_var_ref<double>(G__DOUBLEALLOC, G__double, item, output_var, result);
         break;
      case 'e':
         // FILE
         G__genericerror("Limitation: FILE type variable can not be declared unless type FILE is explicitly defined");
         //--
         break;
      case 'y':
         // void
         G__genericerror("Error: void type variable can not be declared");
         //--
         break;
#ifndef G__OLDIMPLEMENTATION2191
      case 'j':
         // macro file position
#else // G__OLDIMPLEMENTATION2191
      case 'm':
         // macro file position
#endif // G__OLDIMPLEMENTATION2191
         G__get_offset(output_var) = (char*) G__malloc(1, sizeof(std::fpos_t), item);
         *((std::fpos_t*) G__get_offset(output_var)) = *((std::fpos_t*) result.obj.i);
         break;
      case 'a':
         // pointer to member function
         G__get_offset(output_var) = (char*) G__malloc(num_elements ? num_elements : 1, G__P2MFALLOC, item);
         if (
            (G__asm_wholefunction == G__ASM_FUNC_NOP) &&
            !G__def_struct_member &&
            (!G__static_alloc || G__prerun) &&
            result.obj.i &&
            ((G__globalvarpointer == G__PVOID) || G__get_type(G__value_typenum(result)))
         ) {
            // --
#ifdef G__PTR2MEMFUNC
            if (G__get_type(G__value_typenum(result)) == 'C') {
               *((long*) G__get_offset(output_var)) = result.obj.i;
            }
            else {
               std::memcpy(G__get_offset(output_var), (void*) result.obj.i, G__P2MFALLOC);
            }
#else // G__PTR2MEMFUNC
            std::memcpy(G__get_offset(output_var), (void*) result.obj.i, G__P2MFALLOC);
#endif // G__PTR2MEMFUNC
            // --
         }
         break;
#ifndef G__OLDIMPLEMENTATION2191
         // case '1':
         // function, ???Questionable???
#else // G__OLDIMPLEMENTATION2191
      case 'q':
         // function, ???Questionable???
#endif // G__OLDIMPLEMENTATION2191
         G__get_offset(output_var) = (char*) G__malloc(num_elements ? num_elements : 1, sizeof(long), item);
         break;
      default:
         //
         // Automatic variable and macro
         //   p : macro int
         //   P : macro double
         //   o : auto int
         //   O : auto double
         //
         // case 'p' macro or 'o' automatic variable
         //
         // If not macro definition, print out warning.
         if (!G__definemacro && (G__globalcomp == G__NOLINK)) {
            if (G__get_tagnum(output_var.DeclaringScope()) != -1) {
               // -- We are a data member.
               if (G__dispmsg >= G__DISPWARN) {
                  G__fprinterr(G__serr, "Warning: Undeclared data member %s", item);
                  G__genericerror(0);
               }
               return result;
            }
            if (!G__const_noerror) {
               G__fprinterr(G__serr, "Error: Undeclared variable %s", item);
               G__genericerror(0);
            }
            //output_var->type[ig15] = 'o';
         }
         //--
         //--
         //--
         //--
         //--
         //--
         //--
         //--
         //--
         //--
         //--
         //--
         //--
         // Allocate double macro or not.
         if (G__isdouble(result)) {
            // 'P' macro double, 'O' auto double.
            //--
            G__get_offset(output_var) = (char*) G__malloc(num_elements ? num_elements : 1, G__DOUBLEALLOC, item);
            if (
               (G__asm_wholefunction == G__ASM_FUNC_NOP) &&
               (!G__static_alloc || G__prerun) &&
               ((G__globalvarpointer == G__PVOID) || G__get_type(G__value_typenum(result)))
            ) {
               *(((double*) G__get_offset(output_var)) + ((num_elements ? num_elements : 1) - 1)) = G__double(result);
            }
         }
         else {
            // 'p' macro int, 'o' auto int.
            //--
            //--
            //--
            G__get_offset(output_var) = (char*) G__malloc(num_elements ? num_elements : 1, G__LONGALLOC, item);
            if (
               (G__asm_wholefunction == G__ASM_FUNC_NOP) &&
               (!G__static_alloc || G__prerun) &&
               ((G__globalvarpointer == G__PVOID) || G__get_type(G__value_typenum(result)))
            ) {
               *(((long*) G__get_offset(output_var)) + ((num_elements ? num_elements : 1) - 1)) = G__int(result);
            }
         }
         break;
   }
   //
   //  Security, check for unassigned internal pointer.
   //
   //PSRXXXG__CHECK(G__SECURE_POINTER_INIT, !G__def_struct_member && std::isupper(G__var_type) && (G__ASM_FUNC_NOP == G__asm_wholefunction) && G__get_offset(output_var) && (0 == (*((long*) G__get_offset(output_var)))), *((long*) G__get_offset(output_var)) = 0);
   if (G__security & G__SECURE_POINTER_INIT) {
      if (G__security_handle(G__SECURE_POINTER_INIT)) {
         if (!G__def_struct_member) {
            if (isupper(G__var_type)) {
               if (G__asm_wholefunction == G__ASM_FUNC_NOP) {
                  if (G__asm_wholefunction == G__ASM_FUNC_NOP) {
                     if (G__get_offset(output_var)) {
                        if (!(*((long*) G__get_offset(output_var)))) {
                           *((long*) G__get_offset(output_var)) = 0;
                        }
                     }
                  }
               }
            }
         }
      }
   }
   //
   //  Security, increment reference count on a pointed-at object.
   //
#ifdef G__SECURITY
   if (
      (G__security & G__SECURE_GARBAGECOLLECTION) &&
      !G__def_struct_member &&
      !G__no_exec_compile &&
      std::isupper(G__var_type) &&
      G__get_offset(output_var) &&
      (*((long*) G__get_offset(output_var)))
   ) {
      G__add_refcount((void*) (*((long*) G__get_offset(output_var))), (void**) G__get_offset(output_var));
   }
#endif // G__SECURITY
   //
   //  Do special fixups for an automatic object.
   //
   if (autoobjectflag) {
      //--
      G__tagnum = store_tagnum;
      G__typenum = store_typenum;
      G__globalvarpointer = store_globalvarpointer;
      G__var_type = store_var_type;
   }
   //
   //  Handle the G__virtualinfo special variable,
   //  copy its value to the virtual offset member
   //  of the class metadata.
   //
   if (
      // -- Not running, is a public data member, and is the special virtual offset member.
      G__prerun && // not running
      (G__get_tagnum(output_var.DeclaringScope()) != -1) && // data member
      (G__access == G__PUBLIC) && // public access
      (varname == "G__virtualinfo") // special virtual offset variable
   ) {
      // -- Copy the virtual offset data member value to the class metadata.
      G__struct.virtual_offset[G__get_tagnum(output_var.DeclaringScope())] = G__get_offset(output_var);
   }
   return result;
}

#undef G__ALLOC_VAR_REF

//______________________________________________________________________________
#ifdef G__ASM_DBG
static int Cint::Internal::G__asm_gen_stvar(long arg_G__struct_offset, const ::Reflex::Member& var, int paran, const char* item, long store_struct_offset, int var_type, G__value* /*presult*/)
#else
static int Cint::Internal::G__asm_gen_stvar(long arg_G__struct_offset, const ::Reflex::Member& var, int paran, const char* /* item */, long store_struct_offset, int var_type, G__value* /*presult*/)
#endif
{
   // -- FIXME: Describe me!
   ::Reflex::Type type(var.TypeOf().FinalType());
   // ST_GVAR or ST_VAR instruction.
   if (arg_G__struct_offset) {
      // --
#ifdef G__NEWINHERIT // Always
      if (arg_G__struct_offset != store_struct_offset) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: ADDSTROS %d  %s:%d\n", G__asm_cp, G__asm_dt, arg_G__struct_offset - store_struct_offset, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__ADDSTROS;
         G__asm_inst[G__asm_cp+1] = arg_G__struct_offset - store_struct_offset;
         G__inc_cp_asm(2, 0);
      }
#endif // G__NEWINHERIT
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: ST_MSTR %s paran=%d  %s:%d\n", G__asm_cp, G__asm_dt, item, paran, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__ST_MSTR;
      G__asm_inst[G__asm_cp+1] = 0;
      G__asm_inst[G__asm_cp+2] = paran;
      G__asm_inst[G__asm_cp+3] = var_type;
      G__asm_inst[G__asm_cp+4] = (long) var.Id();
      G__inc_cp_asm(5, 0);
#ifdef G__NEWINHERIT // Always
      if (arg_G__struct_offset != store_struct_offset) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: ADDSTROS %d  %s:%d\n", G__asm_cp, G__asm_dt, -arg_G__struct_offset + store_struct_offset, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__ADDSTROS;
         G__asm_inst[G__asm_cp+1] = -arg_G__struct_offset + store_struct_offset;
         G__inc_cp_asm(2, 0);
      }
#endif // G__NEWINHERIT
      // --
   }
   else if (
      // -- In a declaration of a reference and not generating code for a whole function.
      G__decl && // in a declaration
      (G__reftype == G__PARAREFERENCE) && // of a reference
      !G__asm_wholefunction // and not compiling a whole function
   ) {
      G__redecl(var);
      if (G__no_exec_compile) {
         G__abortbytecode();
      }
   }
   else {
      // -- Normal variable.
#ifdef G__ASM_WHOLEFUNC
      if (
         // -- Generating bytecode for whole function and ??? and not a static local var.
         G__asm_wholefunction &&
         (store_struct_offset  == G__ASM_VARLOCAL) &&
         (G__get_properties(var)->statictype != G__LOCALSTATIC)
      ) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: ST_LVAR item: '%s' paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, item, paran, var_type, (long) var.Id(), __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__ST_LVAR;
      }
      else {
         // -- Normal case.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: ST_VAR item: '%s' paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, item, paran, var_type, (long) var.Id(), __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__ST_VAR;
      }
#else // G__ASM_WHOLEFUNC
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: ST_VAR item: '%s' paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, item, paran, var_type, (long) var, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__ST_VAR;
#endif // G__ASM_WHOLEFUNC
      G__asm_inst[G__asm_cp+1] = 0;
      G__asm_inst[G__asm_cp+2] = paran;
      G__asm_inst[G__asm_cp+3] = var_type;
      G__asm_inst[G__asm_cp+4] = (long) var.Id();
      G__inc_cp_asm(5, 0);
   }
   return 0;
}

//______________________________________________________________________________
//
//  Get the value from a variable which is not of pointer type.
//

//--
template<class CASTTYPE, class CONVFUNC>
inline void G__get_var(int SIZE, CONVFUNC f, char TYPE, char PTYPE, const char* item, ::Reflex::Member& variable, char* local_G__struct_offset, int paran, int linear_index, G__value* result)
{
   switch (G__var_type) {
      case 'p':
         /* -- Return the variable value. */
         if (paran >= G__get_paran(variable)) {
            /* -- Value is not an array or pointer type.  MyType v = var; */
            /* FIXME: This is wrong, if greater than we should be doing pointer arithmetic. */
            result->ref = (long) (local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * SIZE));
            (*f)(result, TYPE, (CASTTYPE) (*((CASTTYPE*) result->ref)));
         }
         else {
            /* -- We are accessing part of an array, so we must do the array to pointer standard conversion. */
            /* int ary[2][3][5];  int* v = ary[1][2]; */
            /* The resulting value is a pointer to the first element. */
            G__letint(result, PTYPE, (long) (local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * SIZE)));
            /* FIXME: We have no way of representing an array return type. */
            /* FIXME: The best we can do is call it a multi-level pointer. */
            /* FIXME: This is a violation of the C++ type system. */
            if ((G__get_paran(variable) - paran) > 1) {
               /* -- We are more than one level of pointers deep, construct a pointer chain. */
               /* FIXME: This is wrong, we need to make a pointer to an array. */
               for (int i = G__get_paran(variable) - paran - 1; i > 0; --i) {
                  G__value_typenum(*result) = ::Reflex::PointerBuilder(G__value_typenum(*result));
               }
            }
         }
         break;
      case 'P':
         /* -- Return a pointer to the variable value.  MyType* v = &var; */
         G__letint(result, PTYPE, (long) (local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * SIZE)));
         break;
      default:
         /* -- case 'v': */
         G__reference_error(item);
         break;
   }
   G__var_type = 'p';
}
//--

//______________________________________________________________________________
//
//  Get the value from a variable of pointer type.
//

//--
template<class CASTTYPE, class CONVTYPE, class CONVFUNC>
inline void G__get_pvar(CONVFUNC f, char TYPE, char PTYPE, ::Reflex::Member& variable, char* local_G__struct_offset, int paran, G__value para[G__MAXVARDIM], int linear_index, int secondary_linear_index, G__value* result)
{
   switch (G__var_type) {
      case 'v':
         /* -- Return the value that the pointer variable points to.  Mytype* var; MyType v = *var; */
         switch (G__get_reftype(variable.TypeOf())) {
         case G__PARANORMAL: {
               /* -- Variable is a one-level pointer. */
               char* address = local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC);
               result->ref = *(long*)address;
               if (result->ref) {
                  (*f)(result, TYPE, (CONVTYPE) (*(CASTTYPE*)result->ref));
               }
               break;
            }
            case G__PARAP2P:
            {
               if (G__get_paran(variable) < paran) {
                  char* address = local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC);
                  result->ref = *(long*)((CASTTYPE*)(*(long*)address) + secondary_linear_index);
                  if (result->ref) {
                     (*f)(result, TYPE, (CONVTYPE) (*(CASTTYPE*)result->ref));
                  }
               }
               else {
                  char* address = local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC);
                  result->ref = *(long*)address;
                  G__letint(result, PTYPE, *(long*)result->ref);
               }
               break;
            }
         }
         break;
      case 'P': {
         /* -- Return a pointer to the pointer variable value.  MyType* var; MyType** v = &var; */
         char* address = local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC);
         if (G__get_paran(variable) == paran) {
            G__letint(result, PTYPE, (long) address);
         }
         else if (G__get_paran(variable) < paran) {
            if (G__get_reftype(variable.TypeOf()) == G__PARANORMAL) {
               G__letint(result, PTYPE, (long) ((CASTTYPE*)(*(long*)address) + secondary_linear_index));
            }
            else {
               G__letint(result, PTYPE, (long) ((long*)(*(long*)address) + secondary_linear_index));
               G__value_typenum(*result) = ::Reflex::PointerBuilder(G__value_typenum(*result));
            }
         }
         else {
            G__letint(result, PTYPE, (long) address);
         }
         break;
      }
      default: {
         /* 'p' -- Return the pointer variable value.  MyType* var; MyType* v = var; */
         if (G__get_paran(variable) == paran) {
            /* MyType* var[ddd]; MyType* v = var[xxx]; */
            result->ref = (long) (local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC));
            if (PTYPE=='1') {
               G__letpointer(result, *((long*) result->ref), G__value_typenum(*result));
            } else {
               G__letint(result, PTYPE, *((long*) result->ref));
            }
         }
         else if (paran > G__get_paran(variable)) {
            /* -- Pointer to array reimplementation. */
            /* MyType* var[ddd];  v = var[xxx][yyy]; */
            char* address = local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC);
            if (G__get_reftype(variable.TypeOf()) == G__PARANORMAL) {
               /* -- Variable is a single-level pointer. */
               /* Pointer to array reimplementation. */
               result->ref = (long) ((CASTTYPE*) (*(long*)address) + secondary_linear_index);
               (*f)(result, TYPE, (CONVTYPE)(*(CASTTYPE*)result->ref));
            }
            else if ((paran - 1) == G__get_paran(variable)) {
               /* Pointer to array reimplementation. */
               result->ref = (long)(((long*)(*((long*)address))) + secondary_linear_index);
               G__letint(result, PTYPE, (long)((CASTTYPE*)(*((long*)result->ref))));
               if (G__get_reftype(variable.TypeOf()) > G__PARAP2P) {
                  ::Reflex::Type ty = variable.TypeOf().FinalType();
                  for (; ty.IsArray(); ty = ty.ToType()) {}
                  int var_ptr_count = 0;
                  for (; ty.IsPointer(); ty = ty.ToType()) {
                     ++var_ptr_count;
                  }
                  for (int i = 0 ; i < (var_ptr_count - 2); ++i) {
                     G__value_typenum(*result) = ::Reflex::PointerBuilder(G__value_typenum(*result));
                  }
               }
            }
            else  {
               /* -- Start doing pointer arithmetic. */
               result->ref = (long)(((long*)(*((long*)address))) + para[0].obj.i);
               for (int ip = 1; ip < (paran - 1); ++ip) {
                  result->ref = (long)(((long*)(*((long*)result->ref))) + para[ip].obj.i);
               }
               ::Reflex::Type result_type = variable.TypeOf().FinalType();
               for (; result_type.IsArray(); result_type = result_type.ToType()) {}
               int ptr_to_drop = paran - G__get_paran(variable);
               for (int i = 0; i < ptr_to_drop; ++i) {
                  result_type = G__deref(result_type);
               }
               G__value_typenum(*result) = result_type;
               switch (G__get_reftype(G__value_typenum(*result))) {
                  case G__PARANORMAL:
                     result->ref = (long)(((CASTTYPE*)(*((long*)(result->ref)))) + para[paran-1].obj.i);
                     (*f)(result, TYPE, *((CASTTYPE*)result->ref));
                     break;
                  case 1:
                     result->ref = (long)((long*)(*((long*)result->ref)) + para[paran-1].obj.i);
                     G__letint(result, PTYPE, *((long*)result->ref));
                     /**/
                     break;
                  default:
                     result->ref = (long)((long*) (*((long*) result->ref)) + para[paran-1].obj.i);
                     G__letint(result, PTYPE, *((long*)result->ref));
                     ::Reflex::Type ty = variable.TypeOf().FinalType();
                     for (; ty.IsArray(); ty = ty.ToType()) {}
                     int var_ptr_count = 0;
                     for (; ty.IsPointer(); ty = ty.ToType()) {
                        ++var_ptr_count;
                     }
                     int new_ptr_count = var_ptr_count - paran + G__get_paran(variable);
                     for (int i = 0 ; i < (new_ptr_count - 1); ++i) {
                        G__value_typenum(*result) = ::Reflex::PointerBuilder(G__value_typenum(*result));
                     }
                     break;
               }
            }
         }
         else {
            /* paran < var->paran[ig15] */
            /* MyType* var[ddd][nnn]; MyType** v = var[xxx]; */
            /* FIXME: This is a syntax error if (var->paran[ig15] - paran) > 1. */
            if (local_G__struct_offset) {
               result->ref = (long)(local_G__struct_offset + (long)G__get_offset(variable));
            } else {
               result->ref = (long)(&G__get_offset(variable));
            }
            G__letint(result, PTYPE, *((long*) result->ref));
         }
         break;
      }
   }
}
//--

//______________________________________________________________________________
G__value Cint::Internal::G__getvariable(char* item, int* known, const ::Reflex::Scope& global_scope, const ::Reflex::Scope& local_scope)
{
   // -- Return the value of a simple variable expression.
   //
   // The allowed expression forms are very simple, the variable
   // name may be prefixed with one '*' or one '&', and suffixed
   // with '[ddd]...'.
   //
   // Note: If item is the empty string, then we are being
   // called from the bytecode execution engine and the variable
   // has already been looked up.
   //
   ::Reflex::Scope varscope;
   char parameter[G__MAXVARDIM][G__ONELINE]; // Text of found array index expressions.
   G__value para[G__MAXVARDIM]; // Evaluated values of found array index expressions.
   ::Reflex::Member variable;
   int paran = 0; // Number of array index expressions found.
   int ig25 = 0;
   int lenitem = 0;
   int done = 0;
   char* local_G__struct_offset = 0;
   char store_var_type = '\0';
   int varhash = 0;
   struct G__input_file store_ifile;
   int store_vartype = 0;
   char* store_struct_offset = 0;
   ::Reflex::Scope store_tagnum;
   int posbracket = 0;
   int posparenthesis = 0;
   G__value result = G__null;
#ifdef G__ASM
   //
   //  If we are called by running bytecode,
   //  skip the string parsing, that was already done
   //  during bytecode compilation.
   //
   if (G__asm_exec) {
      variable = G__asm_index;
      paran = G__asm_param->paran;
      for (int i = 0; i < paran; ++i) {
         para[i] = G__asm_param->para[i];
      }
      para[paran] = G__null;
      varscope = global_scope;
      if (!local_scope) {
         local_G__struct_offset = 0;
      }
      else {
         local_G__struct_offset = G__store_struct_offset;
      }
      goto G__exec_asm_getvar;
   }
#endif // G__ASM
   //
   //  Now do some parsing.
   //
   /////
   //
   // Check for '*varname'  or '*(pointer-expression)'
   //
   switch (item[0]) {
      case '*':
         // -- Pointer dereference.
         //
         // if '*(pointer expression)' evaluate pointer
         // expression and get data from the address and
         // return. Also *a++, *a--, *++a, *--a
         //
         lenitem = strlen(item);
         if (
            (item[1] == '(') ||
            (item[1] == '+') ||
            (item[1] == '-') ||
            (item[lenitem-1] == '+') ||
            (item[lenitem-1] == '-')
         ) {
            int local_store_var_type = G__var_type;
            G__var_type = 'p';
            *known = 1;
            G__value tmpval = G__getexpr(item + 1);
            G__value v = G__tovalue(tmpval);
            if (local_store_var_type != 'p') {
               v = G__toXvalue(v, local_store_var_type);
            }
            return v;
         }
         //
         // if '*varname'
         //
         if (G__var_type == 'p') {
            G__var_type = 'v';
         }
         else {
            G__var_type = toupper(G__var_type);
         }
         //
         // Remove '*' from expression.
         // WARNING: Passed char* parameter "item" is modified.
         //
         for (int i = 0; i < lenitem; ++i) {
            item[i] = item[i+1];
         }
         break;
      case '&':
         // -- Take address of operator.
         // if '&varname'
         // this case only happens when '&tag.varname'.
         lenitem = strlen(item);
         G__var_type = 'P'; // FIXME: This special type is set only here.
         //
         // Remove '&' from expression.
         // WARNING: Passed char* parameter "item" is modified.
         //
         for (int i = 0; i < lenitem; ++i) {
            item[i] = item[i+1];
         }
         break;
      case '(':
         // -- Casting or deeper parenthesis, we cannot handle it.
         return G__null;
   }
   lenitem = strlen(item);
   store_var_type = G__var_type;
   G__var_type = 'p';
   {
      //
      // Following while loop checks if the variable is struct or
      // union member. If unsurrounded '.' or '->' found set flag=1
      // and split tagname and membername.
      //
      //  'tag[].member[]'  or 'tag[]->member[]'
      //        ^                    ^^          set flag=1
      //                                         tagname="tag[]"
      //                                         membername="member[]"
      //
      //  'id[tag.member]'  or 'id[func(tag->member,"ab.")]'
      //              ^                              ^          ^
      //   These '.' and '->' doesn't count because they are surrounded
      //  by parenthesis or quotation.  paren, double_quote and
      //  single_quote are used to identify if they are surreunded by
      //  (){}[] or ""'' and not set flag=1.
      //
      // C++:
      //  G__getvariable() is called before G__getfunction(). So,
      // C++ member function will be handled in G__getvariable()
      // rather than G__getfunction().
      //
      //  'func().mem'
      //  'mem.func()'
      //  'tag.mem.func()'
      //
      int double_quote = 0;
      int single_quote = 0;
      int paren = 0;
      for (int i = 0; i < lenitem; ++i) {
         switch (item[i]) {
            case '.':
               // -- This is a member of struct or union accessed by member reference.
               if (!paren && !double_quote && !single_quote) {
                  // To get full struct member name path when not found.
                  G__StrBuf tmp_sb(G__ONELINE);
                  char *tmp = tmp_sb;
                  strcpy(tmp, item);
                  tmp[i++] = '\0';
                  char* tagname = tmp;
                  char* membername = tmp + i;
                  char varname[2*G__MAXNAME];
                  G__value val = G__getstructmem(store_var_type, varname, membername, tagname, known, global_scope, 1);
                  return val;
               }
               break;
            case '-':
               // -- This is a member of struct or union accessed by pointer dereference.
               if (!paren && !double_quote && !single_quote && (item[i+1] == '>')) {
                  // To get full struct member name path when not found.
                  G__StrBuf tmp_sb(G__ONELINE);
                  char *tmp = tmp_sb;
                  strncpy(tmp, item, i);
                  tmp[i++] = '\0';
                  tmp[i++] = '\0';
                  char* tagname = tmp;
                  char* membername = item + i;
                  char varname[2*G__MAXNAME];
                  G__value val = G__getstructmem(store_var_type, varname, membername, tagname, known, global_scope, 2);
                  return val;
               }
               break;
            case '\\':
               // Don't check next char, this is for escaping quotation like '"xxx\"xxxx"', '\''.
               ++i;
               break;
            case '\'':
               if (!double_quote) {
                  single_quote ^= 1;
               }
               break;
            case '\"':
               if (!single_quote) {
                  double_quote ^= 1;
               }
               break;
            case '[':
               if (!single_quote && !double_quote) {
                  if (!paren && !posbracket) {
                     posbracket = i;
                  }
                  ++paren;
               }
               break;
            case '(':
               if (!single_quote && !double_quote) {
                  if (!paren && !posparenthesis) {
                     posparenthesis = i;
                  }
                  ++paren;
               }
               break;
            case '{':  /* this shouldn't appear */
               if (!single_quote && !double_quote) {
                  ++paren;
               }
               break;
            case '}':  /* this shouldn't appear */
            case ']':
            case ')':
               if (!single_quote && !double_quote) {
                  --paren;
               }
               break;
         }
      }
   }
   local_G__struct_offset = 0;
   if (!global_scope) {
      // We have been called from ourselves or G__letvariable() and
      // our caller set G__store_struct_offset.
      local_G__struct_offset = G__store_struct_offset;
   }
   char varname[2*G__MAXNAME];
   {
      // Collect the variable name and hash value,
      // stop at parenthesis or square brackets.
      int cursor = 0;
      for (cursor = 0; (item[cursor] != '(') && (item[cursor] != '[') && (cursor < lenitem); ++cursor) {
         varname[cursor] = item[cursor];
         varhash += item[cursor];
      }
      if (item[cursor] == '(') {
         // -- We have: 'funcname(xxx)', will be handled by G__getfunction(), abort for now.
         return G__null;
      }
      if (posparenthesis && (item[cursor] == '[')) {
         // -- We have: var[x](a,b);
         item[posparenthesis] = 0;
         G__value val = G__getvariable(item, known, global_scope, local_scope);
         if (!known) {
            return G__null;
         }
         item[posparenthesis] = '(';
         val = G__pointer2func(&val, 0, item + posparenthesis, known);
         *known = 1;
         return val;
      }
      varname[cursor++] = '\0';
      if (cursor == 1) {
         // -- No variable name, only an array index.
         G__getvariable_error(item);
         *known = 1;
         return G__null;
      }
      // Scan inside square brackets '[]' to get array index if any.
      paran = 0;
      parameter[0][0] = '\0';
      while (cursor < lenitem) {
         // FIXME: We must enforce a limit on paran!
         int idx = 0;
         int nest = 0;
         int single_quote = 0;
         int double_quote = 0;
         // Collect a single array index expression.
         while ((cursor < lenitem) && ((item[cursor] != ']') || nest || single_quote || double_quote)) {
            switch (item[cursor]) {
               case '"' :
                  if (!single_quote) {
                     double_quote ^= 1;
                  }
                  break;
               case '\'' :
                  if (!double_quote) {
                     single_quote ^= 1;
                  }
                  break;
               case '(':
               case '[':
               case '{':
                  if (!double_quote && !single_quote) {
                     ++nest;
                  }
                  break;
               case ')':
               case ']':
               case '}':
                  if (!double_quote && !single_quote) {
                     --nest;
                  }
                  break;
            }
            // Collect the current character into the array index expression.
            parameter[paran][idx++] = item[cursor++];
         }
         if (cursor < lenitem) {
            // -- Skip the terminating ']' character.
            ++cursor;
         }
         if ((cursor < lenitem) && (item[cursor] == '[')) {
            // -- Skip past a following '[' character if there is one.
            ++cursor;
         }
         // Terminate this array index expression.
         parameter[paran][idx] = '\0';
         // Initialize the next array index expression.
         parameter[paran+1][0] = '\0';
         // Move on to the next array index expression.
         ++paran;
      }
      // FIXME: We should check to see that we have consumed all of the text of item at this point!
   }
#ifdef G__ASM
   if (
      G__asm_noverflow &&
      paran &&
      ((G__store_struct_offset != G__memberfunc_struct_offset) || G__do_setmemfuncenv)
   ) {
      // -- Restore base environment.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: SETMEMFUNCENV  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__SETMEMFUNCENV;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   store_struct_offset = G__store_struct_offset;
   store_tagnum = G__tagnum;
   store_vartype = G__var_type;
   G__var_type = 'p';
   G__tagnum = G__memberfunc_tagnum;
   G__store_struct_offset = G__memberfunc_struct_offset;
   //
   // Evaluate and store array index expressions.
   //
   for (int i = 0; i < paran; ++i) {
      para[i] = G__getexpr(parameter[i]);
   }
#ifdef G__ASM
   if (
      G__asm_noverflow &&
      paran &&
      ((G__store_struct_offset != store_struct_offset) || G__do_setmemfuncenv)
   ) {
      // -- Recover function call environment.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: RECMEMFUNCENV  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__RECMEMFUNCENV;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;
   G__var_type = store_vartype;
   G__var_type = store_var_type;
   //
   //  Up to here we have been doing mostly parsing.
   //  Now we have a variable name we can search for.
   //
   variable = G__find_variable(varname, varhash, local_scope, global_scope, &local_G__struct_offset, &store_struct_offset, 0, 0);
   if (!variable && (G__prerun || G__eval_localstatic) && G__func_now) {
      std::string temp;
      G__get_stack_varname(temp, varname, G__func_now, G__get_tagnum(G__tagdefining));
      //--
      //--
      //--
      //--
      //--
      int itmpx = 0;
      G__hash(temp.c_str(), varhash, itmpx);
      variable = G__find_variable(temp.c_str(), varhash, local_scope, global_scope, &local_G__struct_offset, &store_struct_offset, 0, 0);
      if (variable) {
         local_G__struct_offset = 0;
      }
      if (!variable && G__getarraydim && !G__IsInMacro()) {
         G__const_noerror = 0;
         G__genericerror("Error: Illegal array dimension (Ignore subsequent errors)");
         *known = 1;
         return G__null;
      }
   }
   if (variable) {
      // -- We found the variable, return its value.
#ifndef G__OLDIMPLEMENTATION1259
      G__value_typenum(result) = variable.TypeOf();
      //--
      //--
      //--
#endif // G__OLDIMPLEMENTATION1259
      if (
         G__getarraydim &&
         !G__IsInMacro() &&
#ifndef G__OLDIMPLEMENTATION2191
         (G__get_type(variable.TypeOf()) != 'j') &&
#else // G__OLDIMPLEMENTATION2191
         (G__get_type(variable.TypeOf()) != 'm') &&
#endif // G__OLDIMPLEMENTATION2191
         (G__get_type(variable.TypeOf()) != 'p') &&
         (
            !(G__get_isconst(variable.TypeOf()) & G__CONSTVAR) ||
            (G__get_isconst(variable.TypeOf()) & G__DYNCONST)
         )
      ) {
         G__const_noerror = 0;
         G__genericerror("Error: Non-static-const variable in array dimension");
         G__fprinterr(G__serr, " (cint allows this only in interactive command and special form macro which\n");
         G__fprinterr(G__serr, "  is special extension. It is not allowed in source code. Please ignore\n");
         G__fprinterr(G__serr, "  subsequent errors.)\n");
         *known = 1;
         return G__null;
      }
      if (!G__get_offset(variable) && !local_G__struct_offset && !G__no_exec_compile) {
         *known = 1;
         G__value val = G__null;
         G__value_typenum(val) = variable.TypeOf();
         //--
         switch (G__var_type) {
            case 'p':
               if (G__get_paran(variable) <= paran) {
                  //--
                  break;
               }
            case 'P':
               if (std::islower(G__get_type(variable.TypeOf()))) {
                  G__value_typenum(val) = ::Reflex::PointerBuilder(variable.TypeOf());
               }
               else {
                  //--
                  switch (G__get_reftype(variable.TypeOf())) {
                     //--
                     //--
                     //--
                     case G__PARAREFERENCE:
                        //--
                        break;
                     default:
                        G__value_typenum(val) = ::Reflex::PointerBuilder(variable.TypeOf());
                        break;
                  }
               }
               break;
         }
         return val;
      }
#ifdef G__ASM
      if (G__asm_noverflow) {
         // -- LD_MSTR or LD_VAR instruction
         if (local_G__struct_offset) {
            // --
#ifdef G__NEWINHERIT
            if (local_G__struct_offset != store_struct_offset) {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: ADDSTROS %d  %s:%d\n", G__asm_cp, G__asm_dt, local_G__struct_offset - store_struct_offset, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__ADDSTROS;
               G__asm_inst[G__asm_cp+1] = local_G__struct_offset - store_struct_offset;
               G__inc_cp_asm(2, 0);
            }
#endif // G__NEWINHERIT
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: LD_MSTR  item: '%s' paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, item, paran, G__var_type, (long) variable.Id(), __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__LD_MSTR;
            G__asm_inst[G__asm_cp+1] = 0;
            G__asm_inst[G__asm_cp+2] = paran;
            G__asm_inst[G__asm_cp+3] = G__var_type;
            G__asm_inst[G__asm_cp+4] = (long) variable.Id();
            G__inc_cp_asm(5, 0);
#ifdef G__NEWINHERIT
            if (local_G__struct_offset != store_struct_offset) {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: ADDSTROS %d  %s:%d\n", G__asm_cp, G__asm_dt, -(long)local_G__struct_offset + store_struct_offset, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__ADDSTROS;
               G__asm_inst[G__asm_cp+1] =  (long) store_struct_offset - (long)local_G__struct_offset;
               G__inc_cp_asm(2, 0);
            }
#endif // G__NEWINHERIT
            // --
         }
         else {
            // --
#ifdef G__ASM_WHOLEFUNC
            if (
               G__asm_wholefunction &&
               (((long) store_struct_offset) == G__ASM_VARLOCAL) &&
               (G__get_properties(variable)->statictype != G__LOCALSTATIC)
            ) {
               //--
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD_LVAR item: '%s' paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, item, paran, G__var_type, (long) variable.Id(), __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD_LVAR;
            }
            else {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD_VAR item: '%s' paran: %d type: '%c' var: %08lx   %s:%d\n", G__asm_cp, G__asm_dt, item, paran, G__var_type, (long) variable.Id(), __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD_VAR;
            }
#else // G__ASM_WHOLEFUNC
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: LD_VAR item: '%s' paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, item, paran, G__var_type, (long) variable, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__LD_VAR;
#endif // G__ASM_WHOLEFUNC
            G__asm_inst[G__asm_cp+1] = 0;
            G__asm_inst[G__asm_cp+2] = paran;
            G__asm_inst[G__asm_cp+3] = G__var_type;
            G__asm_inst[G__asm_cp+4] = (long) variable.Id();
            G__inc_cp_asm(5, 0);
         }
      }
      if (
         G__no_exec_compile &&
         (
            (G__get_isconst(variable.TypeOf()) != G__CONSTVAR) ||
            std::isupper(G__get_type(variable.TypeOf())) ||
            (G__get_reftype(variable.TypeOf()) == G__PARAREFERENCE) ||
            G__get_varlabel(variable.TypeOf(), 1) /* number of elements */ ||
            (reinterpret_cast<long>(G__get_offset(variable)) < 4096)
         ) &&
         (std::tolower(G__get_type(variable.TypeOf())) != 'p')
      ) {
         *known = 1;
         result.obj.d = 0.0;
         G__value_typenum(result) = variable.TypeOf();
         //G__get_tagnum(G__value_typenum(result));
         
         switch (G__get_type(variable.TypeOf())) {
            case 'd':
            case 'f':
               break;
            case 'T':
               G__value_typenum(result) = G__replace_rawtype(G__value_typenum(result), G__get_from_type('C', 0));
            default:
               result.obj.i = 1;
               break;
         }
         G__returnvartype(&result, variable, paran);
         if (isupper(G__get_type(variable.TypeOf()))) {
            long dmy = 0;
            result.ref = (long) &dmy;
            G__getpointer2pointer(&result, variable, paran);
         }
         //--
         result.ref = (long) (local_G__struct_offset + ((size_t) G__get_offset(variable)));
         G__var_type = 'p';
         if (tolower(G__get_type(variable.TypeOf())) == 'u') {
            int varparan = G__get_paran(variable);
            if (G__get_type(variable.TypeOf()) == 'U') {
               ++varparan;
            }
            if (G__get_reftype(variable.TypeOf()) > G__PARAREFERENCE) {
               varparan += (G__get_reftype(variable.TypeOf()) % G__PARAREF) - G__PARAP2P + 1;
            }
            int nelem = 0;
            for (; (nelem < paran) && (nelem < varparan); ++nelem) {}
            while ((nelem < paran) && G__get_varlabel(variable.TypeOf(), nelem + 4)) {
               ++nelem;
            }
            if (nelem < paran) {
               G__tryindexopr(&result, para, paran, nelem);
            }
         }
         return result;
      }
      G__exec_asm_getvar:
      G__value_typenum(result) = variable.TypeOf();
#endif // G__ASM
      // Static class/struct member.
      if (
         local_G__struct_offset &&
         (
            (G__get_properties(variable)->statictype == G__LOCALSTATIC) ||
            (
               (
                  variable.DeclaringScope().IsTopScope() &&
                  (G__get_properties(variable)->statictype == G__COMPILEDGLOBAL)
               ) ||
               (
                  !variable.DeclaringScope().IsTopScope() &&
                  variable.DeclaringScope().IsNamespace() &&
                  (variable.DeclaringScope().Name_c_str()[strlen(variable.DeclaringScope().Name_c_str())-1] != '$') // not function local scope.
               )
            )
         )
      ) {
         local_G__struct_offset = 0;
      }
      //
      //
      //
      ++done;
      //
      //
      //
      /*************************************************
       * int v[A][B][C][D]
       * v[i][j][k][l]
       *
       * tmp = B*C*D
       * linear_index = B*C*D*i + C*D*j + D*k + l
       * secondary_linear_index =
       *************************************************/
      int linear_index = 0;
      {
         int tmp = G__get_varlabel(variable.TypeOf(), 0) /* stride */;
         for (ig25 = 0; (ig25 < paran) && (ig25 < G__get_paran(variable)); ++ig25) {
            linear_index += tmp * G__int(para[ig25]);
            tmp /= G__get_varlabel(variable.TypeOf(), ig25 + 2);
         }
      }
      int secondary_linear_index = 0;
      {
         // -- Calculate secondary_linear_index.
         int tmp = G__get_varlabel(variable.TypeOf(), ig25 + 3);
         if (!tmp) {
            // questionable
            tmp = 1;
         }
         while ((ig25 < paran) && G__get_varlabel(variable.TypeOf(), ig25 + 4)) {
            secondary_linear_index += tmp * G__int(para[ig25]);
            tmp /= G__get_varlabel(variable.TypeOf(), ig25 + 4);
            ++ig25;
         }
      }
      //
      //  Check to make sure linear_index is in bounds.
      //
      //  0 <= linear_index <= number of elements
      //
      // Note: We intentionally allow going one beyond the end.
      if (
         G__get_varlabel(variable.TypeOf(), 1) /* number of elements */ &&
         (G__get_reftype(variable.TypeOf()) == G__PARANORMAL) &&
         (
            (linear_index < 0) ||
            // We intentionally allow going one beyond the end.
            (linear_index > G__get_varlabel(variable.TypeOf(), 1) /* number of elements */) ||
            ((ig25 < paran) && (tolower(G__get_type(variable.TypeOf())) != 'u'))
         )
      ) {
         G__arrayindexerror(variable, item, linear_index);
         *known = 1;
         return G__null;
      }
      // Return struct and typedef information.
      // FIXME: This is wrong because, e.g., a[i] gives an array result type!
      G__value_typenum(result) = variable.TypeOf();
      //--
      result.ref = 0;
      *known = 1;
#ifdef G__SECURITY
      if (
         !G__no_exec_compile &&
         (G__var_type == 'v') &&
         std::isupper(G__get_type(variable.TypeOf())) &&
         (G__get_reftype(variable.TypeOf()) == G__PARANORMAL) &&
         !G__get_varlabel(variable.TypeOf(), 1) /* number of elements */ &&
         ((*(long*)(local_G__struct_offset + (size_t)G__get_offset(variable))) == 0)
      ) {
         G__reference_error(item);
         return G__null;
      }
#endif // G__SECURITY
      G__CHECK(G__SECURE_POINTER_AS_ARRAY, (G__get_paran(variable) < paran && std::isupper(G__get_type(variable.TypeOf()))), return(G__null));
      G__CHECK(G__SECURE_POINTER_REFERENCE, (isupper(G__get_type(variable.TypeOf()) && (G__get_type(variable.TypeOf()) != 'E')) || (G__get_paran(variable) > paran)), return(G__null));
      // Get bit-field value.
      if (G__get_bitfield_width(variable) && 'p' == G__var_type) {
         char *address = local_G__struct_offset + (size_t) G__get_offset(variable);
         int original = *((int*) address);
         int mask = (1 << G__get_bitfield_width(variable)) - 1;
         int finalval = (original >> G__get_bitfield_start(variable)) & mask;
         G__letint(&result, G__get_type(variable.TypeOf()), finalval);
         return result;
      }
      if (G__decl && G__getarraydim && !local_G__struct_offset && (reinterpret_cast<long>(G__get_offset(variable)) < 4096)) { 
         // prevent segv in following example. A bit tricky.
         //  void f(const int n) { int a[n]; }
         G__abortbytecode();
         return result;
      }
      //
      //  Now fetch the variable value.
      //
      //  Note: We return here if we are not a pointer variable.
      int type_code = G__get_type(variable.TypeOf());
      if (variable.Name_c_str()[0] == '$') { // Hack ROOT specials, we really need to give these guys their own type.
         type_code = 'Z';
      }
      switch (type_code) {
         case 'i':
            // int
            //G__GET_VAR(G__INTALLOC, int, G__letint, 'i', 'I')
            G__get_var<int>(G__INTALLOC, G__letint, 'i', 'I', item, variable, local_G__struct_offset, paran, linear_index, &result);
            return result;
         case 'd':
            // double
            //G__GET_VAR(G__DOUBLEALLOC, double, G__letdouble, 'd', 'D')
            G__get_var<double>(G__DOUBLEALLOC, G__letdouble, 'd', 'D', item, variable, local_G__struct_offset, paran, linear_index, &result);
            return result;
         case 'c':
            // char
            //G__GET_VAR(G__CHARALLOC, char, G__letint, 'c', 'C')
            G__get_var<char>(G__CHARALLOC, G__letint, 'c', 'C', item, variable, local_G__struct_offset, paran, linear_index, &result);
            return result;
         case 'b':
            // unsigned char
            //G__GET_VAR(G__CHARALLOC, unsigned char, G__letint, 'b', 'B')
            G__get_var<unsigned char>(G__CHARALLOC, G__letint, 'b', 'B', item, variable, local_G__struct_offset, paran, linear_index, &result);
            return result;
         case 's':
            // short int
            //G__GET_VAR(G__SHORTALLOC, short, G__letint, 's', 'S')
            G__get_var<short>(G__SHORTALLOC, G__letint, 's', 'S', item, variable, local_G__struct_offset, paran, linear_index, &result);
            return result;
         case 'r':
            // unsigned short int
            //G__GET_VAR(G__SHORTALLOC, unsigned short, G__letint, 'r', 'R')
            G__get_var<unsigned short>(G__SHORTALLOC, G__letint, 'r', 'R', item, variable, local_G__struct_offset, paran, linear_index, &result);
            return result;
         case 'h':
            // unsigned int
            //G__GET_VAR(G__INTALLOC, unsigned int, G__letint, 'h', 'H')
            G__get_var<unsigned int>(G__INTALLOC, G__letint, 'h', 'H', item, variable, local_G__struct_offset, paran, linear_index, &result);
            return result;
         case 'l':
            // long int
            //G__GET_VAR(G__LONGALLOC, long, G__letint, 'l', 'L')
            G__get_var<long>(G__LONGALLOC, G__letint, 'l', 'L', item, variable, local_G__struct_offset, paran, linear_index, &result);
            return result;
         case 'k':
            // unsigned long int
            //G__GET_VAR(G__LONGALLOC, unsigned long, G__letint, 'k', 'K')
            G__get_var<unsigned long>(G__LONGALLOC, G__letint, 'k', 'K', item, variable, local_G__struct_offset, paran, linear_index, &result);
            return result;
         case 'f':
            // float
            //G__GET_VAR(G__FLOATALLOC, float, G__letdouble, 'f', 'F')
            G__get_var<float>(G__FLOATALLOC, G__letdouble, 'f', 'F', item, variable, local_G__struct_offset, paran, linear_index, &result);
            return result;
         case 'n':
            // long long
            //G__GET_VAR(G__LONGLONGALLOC, G__int64, G__letLonglong, 'n', 'N')
            G__get_var<G__int64>(G__LONGLONGALLOC, G__letLonglong, 'n', 'N', item, variable, local_G__struct_offset, paran, linear_index, &result);
            return result;
         case 'm':
            // unsigned long long
            //G__GET_VAR(G__LONGLONGALLOC, G__uint64, G__letULonglong, 'm', 'M')
            G__get_var<G__uint64>(G__LONGLONGALLOC, G__letULonglong, 'm', 'M', item, variable, local_G__struct_offset, paran, linear_index, &result);
            return result;
         case 'q':
            // long double
            //G__GET_VAR(G__LONGDOUBLEALLOC, long double, G__letLongdouble, 'q', 'Q')
            G__get_var<long double>(G__LONGDOUBLEALLOC, G__letLongdouble, 'q', 'Q', item, variable, local_G__struct_offset, paran, linear_index, &result);
            return result;
         case 'g':
            // bool
#ifdef G__BOOL4BYTE
            //G__GET_VAR(G__INTALLOC, int, G__letbool, 'g', 'G')
            G__get_var<int>(G__INTALLOC, G__letbool, 'g', 'G', item, variable, local_G__struct_offset, paran, linear_index, &result);
            return result;
#else // G__BOOL4BYTE
            //G__GET_VAR(G__CHARALLOC, unsigned char, G__letint, 'g', 'G')
            G__get_var<unsigned char>(G__CHARALLOC, G__letint, 'g', 'G', item, variable, local_G__struct_offset, paran, linear_index, &result);
            return result;
#endif // G__BOOL4BYTE
            // --
#ifndef G__OLDIMPLEMENTATION2191
         case '1':
            // void pointer
            //G__GET_PVAR(char, G__letint, long, tolower(G__get_type(variable.TypeOf())), G__get_type(variable.TypeOf()))
            G__get_pvar<char, long>(G__letint, (char) tolower(G__get_type(variable.TypeOf())), (char) G__get_type(variable.TypeOf()), variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
#else // G__OLDIMPLEMENTATION2191
         case 'Q': pointer
            // void
            //G__GET_PVAR(char, G__letint, long, tolower(G__get_type(variable.TypeOf())), G__get_type(variable.TypeOf()))
            G__get_pvar<char, long>(G__letint, (char) tolower(G__get_type(variable.TypeOf())), (char) G__get_type(variable.TypeOf()), variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
#endif // G__OLDIMPLEMENTATION2191
         case 'Y':
            // void pointer
            //G__GET_PVAR(char, G__letint, long, tolower(G__get_type(variable.TypeOf())), G__get_type(variable.TypeOf()))
            G__get_pvar<char, long>(G__letint, (char) tolower(G__get_type(variable.TypeOf())), (char) G__get_type(variable.TypeOf()), variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'E':
            // FILE pointer
            //G__GET_PVAR(char, G__letint, long, tolower(G__get_type(variable.TypeOf())), G__get_type(variable.TypeOf()))
            G__get_pvar<char, long>(G__letint, (char) tolower(G__get_type(variable.TypeOf())), (char) G__get_type(variable.TypeOf()), variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'C':
            // char pointer
            //G__GET_PVAR(char, G__letint, long, tolower(G__get_type(variable.TypeOf())), G__get_type(variable.TypeOf()))
            G__get_pvar<char, long>(G__letint, (char) tolower(G__get_type(variable.TypeOf())), (char) G__get_type(variable.TypeOf()), variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'N':
            // long long pointer
            //G__GET_PVAR(G__int64, G__letLonglong, G__int64, tolower(G__get_type(variable.TypeOf())), G__get_type(variable.TypeOf()))
            G__get_pvar<G__int64, G__int64>(G__letLonglong, (char) tolower(G__get_type(variable.TypeOf())), (char) G__get_type(variable.TypeOf()), variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'M':
            // unsigned long long pointer
            //G__GET_PVAR(G__uint64, G__letULonglong, G__uint64, tolower(G__get_type(variable.TypeOf())), G__get_type(variable.TypeOf()))
            G__get_pvar<G__uint64, G__uint64>(G__letULonglong, (char) tolower(G__get_type(variable.TypeOf())), (char) G__get_type(variable.TypeOf()), variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
#ifndef G__OLDIMPLEMENTATION2191
         case 'Q':
            // long double pointer
            //G__GET_PVAR(long double, G__letLongdouble, long, tolower(G__get_type(variable.TypeOf())), G__get_type(variable.TypeOf()))
            G__get_pvar<long double, long>(G__letLongdouble, (char) tolower(G__get_type(variable.TypeOf())), (char) G__get_type(variable.TypeOf()), variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
#endif // G__OLDIMPLEMENTATION2191
         case 'G':
            // bool pointer
            //G__GET_PVAR(unsigned char, G__letint, long, 'g', 'G')
            G__get_pvar<unsigned char, long>(G__letint, 'g', 'G', variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'B':
            // unsigned char pointer
            //G__GET_PVAR(unsigned char, G__letint, long, 'b', 'B')
            G__get_pvar<unsigned char, long>(G__letint, 'b', 'B', variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'S':
            // short int pointer
            //G__GET_PVAR(short, G__letint, long, 's', 'S')
            G__get_pvar<short, long>(G__letint, 's', 'S', variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'R':
            // unsigned short int pointer
            //G__GET_PVAR(unsigned short, G__letint, long, 'r', 'R')
            G__get_pvar<unsigned short, long>(G__letint, 'r', 'R', variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'I':
            // int pointer
            //G__GET_PVAR(int, G__letint, long, 'i', 'I')
            G__get_pvar<int, long>(G__letint, 'i', 'I', variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'H':
            // unsigned int pointer
            //G__GET_PVAR(unsigned int, G__letint, long, 'h', 'H')
            G__get_pvar<unsigned int, long>(G__letint, 'h', 'H', variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'L':
            // long int pointer
            //G__GET_PVAR(long, G__letint, long, 'l', 'L')
            G__get_pvar<long, long>(G__letint, 'l', 'L', variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'K':
            // unsigned long int pointer
            //G__GET_PVAR(unsigned long, G__letint, long, 'k', 'K')
            G__get_pvar<unsigned long, long>(G__letint, 'k', 'K', variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'F':
            // float pointer
            //G__GET_PVAR(float, G__letdouble, double, 'f', 'F')
            G__get_pvar<float, double>(G__letdouble, 'f', 'F', variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'D':
            // double pointer
            //G__GET_PVAR(double, G__letdouble, double, 'd', 'D')
            G__get_pvar<double, double>(G__letdouble, 'd', 'D', variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'u':
            // class, enum, struct, union
            if (variable.TypeOf().IsEnum()) {
               //G__GET_VAR(G__INTALLOC, int, G__letint, 'i', 'I')
               G__get_var<int>(G__INTALLOC, G__letint, 'i', 'I', item, variable, local_G__struct_offset, paran, linear_index, &result);
               return result;
            }
            switch (G__var_type) {
               case 'p':
                  // return value
                  if (G__get_paran(variable) <= paran) {
                     // value, but return pointer
                     result.ref = (long)(local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * variable.TypeOf().RawType().SizeOf()));
                     G__letpointer(&result, result.ref, G__strip_array(variable.TypeOf()));
                  }
                  else {
                     // array, pointer
                     G__letpointer(&result, (long)(local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * variable.TypeOf().RawType().SizeOf())), variable.TypeOf().RawType());
                     int ptr_count = G__get_paran(variable) - paran;
                     for (int i = 0; i < ptr_count; ++i) {
                        G__value_typenum(result) = ::Reflex::PointerBuilder(G__value_typenum(result));
                     }
                  }
                  break;
               case 'P':
                  // return pointer
                  G__letpointer(&result, (long) (local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * variable.TypeOf().RawType().SizeOf())), ::Reflex::PointerBuilder(variable.TypeOf().RawType()));
                  break;
               default :
                  // return value
                  if (G__var_type == 'v') {
                     G__StrBuf refopr_sb(G__MAXNAME);
                     char *refopr = refopr_sb;
                     char* store_struct_offsetX = G__store_struct_offset;
                     ::Reflex::Scope store_tagnumX = G__tagnum;
                     int local_done = 0;
                     int store_asm_exec = G__asm_exec;
                     int store_asm_noverflow = G__asm_noverflow;
                     G__asm_exec = 0;
                     G__asm_noverflow = 0;
                     G__store_struct_offset = local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * variable.TypeOf().RawType().SizeOf());
                     G__set_G__tagnum(variable.TypeOf());
                     strcpy(refopr, "operator*()");
                     result = G__getfunction(refopr, &local_done, G__TRYMEMFUNC);
                     G__asm_exec = store_asm_exec;
                     G__asm_noverflow = store_asm_noverflow;
                     G__tagnum = store_tagnumX;
                     G__store_struct_offset = store_struct_offsetX;
                     if (!local_done) {
                        G__reference_error(item);
                     }
                  }
                  else {
                     G__reference_error(item);
                  }
                  break;
            }
            if (ig25 < paran) {
               G__tryindexopr(&result, para, paran, ig25);
            }
            break;
         case 'U':
            // class, enum, struct, union pointer
            switch (G__var_type) {
               case 'v':
                  // -- Dereference pointer and return pointed-at value,
                  //    an asterisk prefix ('*') was seen before the variable name.
                  switch (G__get_reftype(variable.TypeOf().FinalType())) {
                  case G__PARANORMAL: {
                        // -- We are a one-level pointer to struct.
                        // MyType* p;  MyType obj = *p;
                        // MyType* p;  MyType obj = p[x]; // cannot happen here, need a '*' to get 'v' as var type
                        // MyType* p[x];  MyType obj = *p[a];
                        // MyType* p[x];  MyType obj = p[a][b]; // cannot happen here, need a '*' to get 'v' as var type
                        // Get the address of the pointer.
                        char* address = local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC);
                        //
                        // Get the pointer value and use it to make
                        // the result point at the object.
                        //
                        result.ref = *(long*)address;
                        // FIXME: Should we remove this test?
                        if (result.ref) {
                           G__letpointer(&result, result.ref, variable.TypeOf().RawType());
                        }
                        break;
                     }
                     case G__PARAP2P:
                        // -- We are a pointer to pointer to struct.
                        // MyType** p;  MyType v = *p[x];
                        // MyType** p[x];  MyType v = *p[x][y];
                        if (G__get_paran(variable) < paran) {
                           // MyType** p[x];  MyType v = *p[x][y];
                           // Get the address of the pointer.
                           char* address = local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC);
                           //
                           // Get the pointer value and increment it by the use it to make the result point at the object.
                           //
                           result.ref = *((long*)(*(long*)address) + secondary_linear_index);
                           // FIXME: Should we remove this test?
                           if (result.ref) {
                              G__letpointer(&result, result.ref, variable.TypeOf().RawType());
                           }
                        }
                        else {
                           char* address = local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC);
                           result.ref = *(long*)address;
                           G__letpointer(&result, *(long*)result.ref, ::Reflex::PointerBuilder(variable.TypeOf().RawType()));
                        }
                        break;
                     // FIXME: We need to handle multi-level pointers too!
                  }
                  break;
               case 'P':
                  // -- Return a pointer to the pointer,
                  //    an ampersand ('&') prefix was seen before the variable name.
                  if (G__get_paran(variable) == paran) {
                     char* address = local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC);
                     G__letpointer(&result, (long) address, ::Reflex::PointerBuilder(variable.TypeOf().RawType()));
                  }
                  else if (G__get_paran(variable) < paran) {
                     char* address = local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC);
                     if (G__get_reftype(variable.TypeOf()) == G__PARANORMAL) {
                        G__letpointer(&result, (*(long*)address) + (secondary_linear_index * variable.TypeOf().RawType().SizeOf()), ::Reflex::PointerBuilder(variable.TypeOf().RawType()));
                       
                     }
                     else {
                        G__letpointer(&result, (long)((long*)(*(long*)address) + secondary_linear_index), ::Reflex::PointerBuilder(variable.TypeOf().RawType()));
                        G__value_typenum(result) = ::Reflex::PointerBuilder(G__value_typenum(result));
                     }
                  }
                  else {
                     char* address = local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC);
                     G__letpointer(&result, (long) address, ::Reflex::PointerBuilder(variable.TypeOf().RawType()));
                  }
                  break;
               default:
                  // -- 'p', Return the pointer value.
                  {
                     if (G__get_paran(variable) == paran) {
                        // MyType* p;  MyType* v = p;
                        // MyType* p[a];  MyType* v = p[x];
                        // MyType** p[a];  MyType** v = p[x];
                        // MyType* p[a][b];  MyType* v = p[x][y];
                        // MyType** p[a][b];  MyType** v = p[x][y];
                        result.ref = (long) (local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC));
                        G__letpointer(&result, *((long*)result.ref), ::Reflex::PointerBuilder(variable.TypeOf().RawType()));
                     }
                     else if (G__get_paran(variable) < paran) {
                        // MyType* p;  MyType v = p[x];
                        // MyType* p[a];  MyType v = p[x][y];
                        // MyType** p[a];  MyType* v = p[x][y];
                        char *address = local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC);
                        if (G__get_reftype(variable.TypeOf()) == G__PARANORMAL) {
                           // MyType* p;  MyType v = p[x];
                           // MyType* p[a];  MyType v = p[x][y];
                           result.ref = (*(long*)address) + (secondary_linear_index * variable.TypeOf().RawType().SizeOf());
                           G__letpointer(&result, result.ref, variable.TypeOf().RawType());
                        }
                        else if (G__get_paran(variable) == (paran - 1)) {
                           // MyType** p;  MyType* v = p[x];
                           // MyType** p[a];  MyType* v = p[x][y];
                           // MyType*** p[a];  MyType** v = p[x][y];
                           // MyType**** p[a];  MyType*** v = p[x][y];
                           result.ref = (long)((long*) (*(long*)address) + secondary_linear_index);
                           G__letpointer(&result, *((long*) result.ref), ::Reflex::PointerBuilder(variable.TypeOf().RawType()));
                           if (G__get_reftype(variable.TypeOf()) > G__PARAP2P) {
                              ::Reflex::Type ty = variable.TypeOf().FinalType();
                              int var_ptr_count = 0;
                              for (; ty.IsPointer(); ty = ty.ToType()) {
                                 ++var_ptr_count;
                              }
                              for (int i = 0; i < (var_ptr_count - 2); ++i) {
                                 G__value_typenum(result) = ::Reflex::PointerBuilder(G__value_typenum(result));
                              }
                           }
                        }
                        else if (G__get_paran(variable) == (paran - 2)) {
                           // MyType** p[a];  MyType v = p[x][y][z];
                           // MyType*** p[a];  MyType* v = p[x][y][z];
                           // MyType**** p[a];  MyType** v = p[x][y][z];
                           // MyType***** p[a];  MyType*** v = p[x][y][z];
                           result.ref = (long) ((long*) (*(long*)address) + para[0].obj.i);
                           if (G__get_reftype(variable.TypeOf()) == G__PARAP2P) {
                              // MyType** p[a];  MyType v = p[x][y][z];
                              result.ref= (long) ((*((long*) result.ref)) + (para[1].obj.i * variable.TypeOf().RawType().SizeOf()));
                              G__letpointer(&result, result.ref, variable.TypeOf().RawType());
                           }
                           else if (G__get_reftype(variable.TypeOf()) > G__PARAP2P) {
                              // MyType*** p[a];  MyType* v = p[x][y][z];
                              // MyType**** p[a];  MyType** v = p[x][y][z];
                              // MyType***** p[a];  MyType*** v = p[x][y][z];
                              result.ref = (long) ((long*) (*(long*)result.ref) + para[1].obj.i);
                              G__letpointer(&result, *((long*)result.ref), ::Reflex::PointerBuilder(variable.TypeOf().RawType()));
                              if (G__get_reftype(variable.TypeOf()) > G__PARAP2P2P) {
                                 // MyType**** p[a];  MyType v = p[x][y][z];
                                 ::Reflex::Type ty = variable.TypeOf().FinalType();
                                 int var_ptr_count = 0;
                                 for (; ty.IsPointer(); ty = ty.ToType()) {
                                    ++var_ptr_count;
                                 }
                                 for (int i = 0; i < (var_ptr_count - 3); ++i) {
                                    G__value_typenum(result) = ::Reflex::PointerBuilder(G__value_typenum(result));
                                 }
                              }
                           }
                           paran -= 1;
                        }
                        else if (G__get_paran(variable) == (paran - 3)) {
                           // MyType*** p[a];  MyType v = p[x][y][z][xx];
                           // MyType**** p[a];  MyType* v = p[x][y][z][xx];
                           // MyType***** p[a];  MyType** v = p[x][y][z][xx];
                           result.ref = (long) ((long*) (*(long*)address) + para[0].obj.i);
                           result.ref = (long) ((long*) (*(long*)result.ref) + para[1].obj.i);
                           if (G__get_reftype(variable.TypeOf()) == G__PARAP2P2P) {
                              result.ref = (long) (*(long*)result.ref + (para[2].obj.i * variable.TypeOf().RawType().SizeOf()));
                              G__letpointer(&result, result.ref, variable.TypeOf().RawType());
                           }
                           else if (G__get_reftype(variable.TypeOf()) > G__PARAP2P2P) {
                              result.ref = (long) ((long*) (*(long*)result.ref) + para[2].obj.i);
                              G__letpointer(&result, *(long*)result.ref, ::Reflex::PointerBuilder(variable.TypeOf().RawType()));
                              if (G__get_reftype(variable.TypeOf()) > G__PARAP2P2P) {
                                 ::Reflex::Type ty = variable.TypeOf().FinalType();
                                 int var_ptr_count = 0;
                                 for (; ty.IsPointer(); ty = ty.ToType()) {
                                    ++var_ptr_count;
                                 }
                                 for (int i = 0; i < (var_ptr_count - 4); ++i) {
                                    G__value_typenum(result) = ::Reflex::PointerBuilder(G__value_typenum(result));
                                 }
                              }
                           }
                           paran -= 2;
                        }
                        else {
                           // MyType* p[a];  MyType v = p[x][y][z][xx][yy];
                           result.ref = (long) ((long*) (*(long*)address) + para[0].obj.i);
                           result.ref = (long) (*(long*)result.ref + (para[1].obj.i * variable.TypeOf().RawType().SizeOf()));
                           G__letpointer(&result, result.ref, variable.TypeOf().RawType());
                           paran -= 2;
                        }
                     }
                     else {
                        // MyType* p[x];  MyType** v = p;
                        // MyType** p[x];  MyType*** v = p;
                        // MyType* p[x][y];  MyType*** v = p;
                        // MyType** p[x][y];  MyType**** v = p;
                        // MyType* p[x][y];  MyType** v = p[a];
                        // MyType** p[x][y];  MyType*** v = p[a];
                        char* address = local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC);
                        result.ref = (long) (&G__get_offset(variable));
                        G__letpointer(&result, (long) address, ::Reflex::PointerBuilder(variable.TypeOf().RawType()));
                     }
                  }
                  break;
            }
            if (ig25 < paran) {
               G__tryindexopr(&result, para, paran, ig25);
            }
            break;
#ifndef G__OLDIMPLEMENTATION2191
         case 'j':
#else // G__OLDIMPLEMENTATION2191
         case 'm':
#endif // G__OLDIMPLEMENTATION2191
            // macro
            {
               fpos_t pos;
               struct G__funcmacro_stackelt* store_stack = G__funcmacro_stack;
               G__funcmacro_stack = 0;
               fgetpos(G__ifile.fp, &pos); /* ifile might already be mfp */
               store_ifile = G__ifile;
               G__ifile.fp = G__mfp;
               strcpy(G__ifile.name, G__macro);
               fsetpos(G__ifile.fp, (fpos_t*) G__get_offset(variable));
               G__nobreak = 1;
               //--
               int brace_level = 0;
               result = G__exec_statement(&brace_level);
               G__nobreak = 0;
               G__ifile = store_ifile;
               fsetpos(G__ifile.fp, &pos);
               G__funcmacro_stack = store_stack;
            }
            break;
         case 'a':
            // member function pointer
            switch (G__var_type) {
               case 'p':
                  if (G__get_paran(variable) <= paran) {
                     result.ref = (long) (local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__P2MFALLOC));
                     result.obj.i = result.ref;
                     G__value_typenum(result) = G__get_from_type('a', 0);
                     //--
                     //--
                  }
                  else {
                     // array
                     G__letint(&result, 'A', (long) (local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__P2MFALLOC)));
                  }
                  break;
               default:
                  G__reference_error(item);
                  break;
            }
            break;
#ifdef G__ROOT
         case 'Z':
            // root special
            if (G__GetSpecialObject) {
               store_var_type = G__var_type;
               if (variable.Name_c_str()[0] == '$') { // FIXME: I assume the name is not the empty string!
                  result = (*G__GetSpecialObject)(const_cast<char*>(variable.Name_c_str() + 1), (void**)G__get_offset(variable), (void**)(G__get_offset(variable) + G__LONGALLOC));
               }
               else {
                  result = (*G__GetSpecialObject)(const_cast<char*>(variable.Name_c_str()), (void**)G__get_offset(variable), (void**)(G__get_offset(variable) + G__LONGALLOC));
               }
               // G__var_type was stored in store_var_type just before the
               // call to G__GetSpecialObject which might have recursive
               // calls to G__getvariable() or G__getexpr()
               // It is restored at this point.
               G__var_type = store_var_type;
               if (!result.obj.i) { // A nil pointer was returned.
                  *known = 0;
               }
               else { // We got something, change the result type.
                  Reflex::Scope local_varscope = variable.DeclaringScope();
                  std::string name = variable.Name();  // Need to cache the name in a string (the storage might go away with the call to RemoveDataMember)
                  char* offset = G__get_offset(variable);
                  G__RflxVarProperties* prop = G__get_properties(variable);
                  local_varscope.RemoveDataMember(variable);
                  if (G__var_type == 'v') {
                     G__value_typenum(result) = G__deref(G__value_typenum(result));
                  }
                  variable = G__add_scopemember(local_varscope, name.c_str(), G__value_typenum(result), 0, 0, offset, G__PUBLIC, prop->statictype);
                  *G__get_properties(variable) = *prop; // Overwrite the new properties with the previous ones.
               }
               switch (G__var_type) {
                  case 'p':
                     break;
                  case 'v':
                     result.ref = result.obj.i;
                     break;
                  default:
                     G__reference_error(item);
                     break;
               }
            }
            break;
#endif // G__ROOT
            //--
         case 'T':
            // #define xxx "abc"
            //G__GET_PVAR(char, G__letint, long, tolower(G__get_type(variable.TypeOf())), 'C')
            G__get_pvar<char, long>(G__letint, (char) tolower(G__get_type(variable.TypeOf())), 'C', variable, local_G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'p':
	   // case macroInt$
	   G__letpointer(&result, *(long*)(local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC)),variable.TypeOf().FinalType()); // Not really a pointer operation .. but that will do since it copies obj.i (and the type is macroInt$).
            break;
         default:
            // case 'X' automatic variable
            G__var_type = 'p';
            if (isupper(G__get_type(variable.TypeOf()))) {
               G__letdouble(&result, 'd', (double)(*(double *)(local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__DOUBLEALLOC))));
            }
            else {
               G__letint(&result, 'l', *(long*)(local_G__struct_offset + ((size_t) G__get_offset(variable)) + (linear_index * G__LONGALLOC)));
            }
            break;
      }
   }
   // Ok, searched all old local and global variables.
   if (!done) {
      // -- Undefined variable.
      //
      // If variable name not found, then search for 'this' keyword.
      *known = G__getthis(&result, varname, item);
      if (*known) {
         if (paran > 0) {
            G__genericerror("Error: syntax error");
         }
         return result;
      }
      //
      //  If variable name not found, then search
      //  for function name. The identifier might
      //   be pointer to function.
      //
      G__var_type = 'p';
      //
      //  Maybe type is 'Q' instead of 'C', but
      //  type 'Q'(pointer to function) is not implemented.
      //
      G__search_func(varname, &result);
      if (!result.obj.i) {
         return G__null;
      }
      *known = 2;
#ifdef G__ASM
      if (G__asm_noverflow) {
         // -- We are generating bytecode.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: LD '%c'  %s:%d\n", G__asm_cp, G__asm_dt, G__int(result), __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__LD;
         G__asm_inst[G__asm_cp+1] = G__asm_dt;
         G__asm_stack[G__asm_dt] = result;
         G__inc_cp_asm(2, 1);
      }
#endif // G__ASM
      return result;
   }
   // -- Handling pointer to pointer in G__value.
   if (
      isupper(G__get_type(variable.TypeOf())) &&
      (G__get_type(variable.TypeOf()) != 'P') &&
      (G__get_type(variable.TypeOf()) != 'O')
   ) {
      // -- Handle pointer to pointer in value.
      G__getpointer2pointer(&result, variable, paran);
   }
   // Return value for a non-automatic variable.
   G__var_type = 'p';
   return result;
}

#undef G__GET_VAR
#undef G__GET_PVAR

//______________________________________________________________________________
G__value Cint::Internal::G__getstructmem(int store_var_type, char* varname, char* membername, char* tagname, int* known2, const ::Reflex::Scope& varglobal, int objptr /* 1 : object, 2 : pointer */)
{
   // -- FIXME: Describe me!
   // fprintf(stderr, "G__getstructmem: varname: '%s' membername: '%s' tagname: '%s' objptr: %d\n", varname, membername, tagname, objptr);
   ::Reflex::Scope store_tagnum;
   char* store_struct_offset = 0;
   int flag = 0;
#ifndef G__OLDIMPLEMENTATION1259
   G__SIGNEDCHAR_T store_isconst = 0;
#endif // G__OLDIMPLEMENTATION1259
   char* px = 0;
   int store_do_setmemfuncenv = 0;
   G__value result;
   //
   // Pointer access operators are removed at the
   // beginning of this function. Add it again to membername
   // because child G__getvariable() needs that information.
   //
   if (store_var_type == 'P') {
      sprintf(varname, "&%s", membername);
      strcpy(membername, varname);
   }
   else if (store_var_type == 'v') {
      sprintf(varname, "*%s", membername);
      strcpy(membername, varname);
   }
   store_tagnum = G__tagnum;
   store_struct_offset = G__store_struct_offset;
#ifndef G__OLDIMPLEMENTATION1259
   store_isconst = G__isconst;
#endif // G__OLDIMPLEMENTATION1259
   // --
#ifdef G__ASM
   if (G__asm_noverflow) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   flag = 0;
   if (
      ((px = strchr(tagname, '.')) && isalpha(*(px + 1))) &&
      (
         strchr(tagname, '+') ||
         strchr(tagname, '-') ||
         strchr(tagname, '*') ||
         strchr(tagname, '/') ||
         strchr(tagname, '%') ||
         strchr(tagname, '&') ||
         strchr(tagname, '|') ||
         strchr(tagname, '^') ||
         strchr(tagname, '!')
      )
   ) {
      result = G__getexpr(tagname);
      if (G__get_type(G__value_typenum(result))) {
         flag = 1;
      }
   }
   if (!flag) {
      // --
      // Get entry pointer for the struct,union and
      // store it to a global variable G__store_struct_offset.
      // In any cases, tagname is a varaible of struct,union type.
      //
      if (varglobal) {
         // --
         //
         // If this is a top level like
         //   'tag.subtag.mem'
         //    --- ----------
         // tagname membername
         // get it from G__global and G__p_local
         //
         result = G__getvariable(tagname, &flag,::Reflex::Scope::GlobalScope(), G__p_local);
      }
      else {
         // --
         //
         // If this is not a top level like
         //   'tag.subtag.mem'
         //        ------ ---
         //       tagname membername
         // get it '&tag' which is G__struct.memvar[].
         //        OR
         // member is referenced in member function
         //  'subtag.mem'
         //   ------ ---
         //
         G__incsetup_memvar(G__tagnum);
         result = G__getvariable(tagname, &flag, ::Reflex::Scope(), G__tagnum);
      }
   }
   if (!flag) {
      // --
      // Object not found as variable,
      // try function which returns struct.
      // We are referencing a freed memory area, so this
      // implementation is bad.
      //
      // We will guess whether this is a function call or an expression.
      int isexpression = 0;
      for (
         unsigned int cur = 0, nested = 0, isstring = 0, begin = 1;
         (cur < strlen(tagname)) && !isexpression;
         ++cur
      ) {
         switch (tagname[cur]) {
            case '(':
               ++nested;
               begin = 0;
               break;
            case ')':
               --nested;
               begin = 0;
               break;
            case '"':
               isstring = !isstring;
               begin = 0;
               break;
            case '-':
            case '+':
            case '%':
            case '|':
            case '!':
            case '^':
            case '/':
               if (!nested && !isstring) {
                  isexpression = 1;
               }
               break;
            case '*':
            case '&':
               if (begin && !nested && !isstring) {
                  isexpression = 1;
               }
               break;
         }
      }
      if (!isexpression) {
         if (varglobal) {
            result = G__getfunction(tagname, &flag, G__TRYNORMAL);
         }
         else {
            result = G__getfunction(tagname, &flag, G__CALLMEMFUNC);
         }
      }
      if (
         !flag &&
         (
           strchr(tagname, '+') ||
           strchr(tagname, '-') ||
           strchr(tagname, '*') ||
           strchr(tagname, '/') ||
           strchr(tagname, '%') ||
           strchr(tagname, '&') ||
           strchr(tagname, '|') ||
           strchr(tagname, '^') ||
           strchr(tagname, '!') ||
           strstr(tagname, "new ")
         )
      ) {
         result = G__getexpr(tagname);
         if (G__get_type(G__value_typenum(result))) {
            flag = 1;
         }
      }
      //
      // If no function like that then return.
      // An error message will be displayed by G__getitem().
      //
      if (!flag) {
         // --
#define G__OLDIMPLEMENTATION965 // FIXME: Remove this?
         return G__null;
      }
      else if (G__no_exec_compile && !result.obj.i) {
         result.obj.i = (long) G__PVOID;
      }
   }
   G__store_struct_offset = (char*) result.obj.i;
   G__set_G__tagnum(result);
#ifndef G__OLDIMPLEMENTATION1259
   G__isconst = G__get_isconst(G__value_typenum(result));
#endif // G__OLDIMPLEMENTATION1259
   if (
      !G__tagnum ||
      (
         G__value_typenum(result).FinalType().IsPointer() &&
         (G__get_reftype(G__value_typenum(result)) >= G__PARAP2P)
      )
   ) {
      if (membername[0] != '~') {
         if (!G__const_noerror) {
            G__fprinterr(G__serr, "Error: non class,struct,union object %s used with . or ->", tagname);
            G__genericerror(0);
         }
      }
      *known2 = 1;
#ifndef G__OLDIMPLEMENTATION1259
      G__tagnum = store_tagnum;
      G__store_struct_offset = store_struct_offset;
      G__isconst = store_isconst;
#endif // G__OLDIMPLEMENTATION1259
      return G__null;
   }
   else if (!G__store_struct_offset && (G__asm_wholefunction == G__ASM_FUNC_NOP)) {
      *known2 = 1;
      if (!G__const_noerror) {
         G__fprinterr(G__serr, "Error: illegal pointer to class object %s 0x%lx %d ", tagname, G__store_struct_offset, G__get_tagnum(G__tagnum));
      }
      G__genericerror(0);
      if (G__interactive) {
         G__fprinterr(G__serr, "!!!Input return value by 'retuurn [val]'\n");
         G__interactive_undefined = 1;
         G__pause();
         G__interactive_undefined = 0;
#ifndef G__OLDIMPLEMENTATION1259
         G__tagnum = store_tagnum;
         G__store_struct_offset = store_struct_offset;
         G__isconst = store_isconst;
#endif // G__OLDIMPLEMENTATION1259
         return G__interactivereturnvalue;
      }
      else {
         // --
#ifndef G__OLDIMPLEMENTATION1259
         G__tagnum = store_tagnum;
         G__store_struct_offset = store_struct_offset;
         G__isconst = store_isconst;
#endif // G__OLDIMPLEMENTATION1259
         return G__null;
      }
   }
#ifdef G__ASM
   if (G__asm_noverflow) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__SETSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   if (
      (G__get_type(G__value_typenum(result)) == 'u') &&
      (objptr == 2) &&
      G__value_typenum(result).IsClass() &&
      !strncmp(G__value_typenum(result).RawType().Name_c_str(), "auto_ptr<", 9)
   ) {
      int knownx = 0;
      char comm[20];
      strcpy(comm, "operator->()");
      result = G__getfunction(comm, &knownx, G__TRYMEMFUNC);
      if (knownx) {
         G__set_G__tagnum(result);
         G__store_struct_offset = (char*) result.obj.i;
#ifdef G__ASM
         if (G__asm_noverflow) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__SETSTROS;
            G__inc_cp_asm(1, 0);
         }
#endif // G__ASM
         // --
      }
   }
   //
   // Check if . or -> matches
   //
   if (!G__value_typenum(result).FinalType().IsPointer() && 2 == objptr) {
      char bufB[30] = "operator->()";
      int flagB = 0;
      ::Reflex::Scope store_tagnumB = G__tagnum;
      char* store_struct_offsetB = G__store_struct_offset;
      G__set_G__tagnum(result);
      G__store_struct_offset = (char*) result.obj.i;
      result = G__getfunction(bufB, &flagB, G__TRYMEMFUNC);
      if (flagB) {
         G__set_G__tagnum(G__value_typenum(result));
         G__store_struct_offset = (char*) result.obj.i;
#ifdef G__ASM
         if (G__asm_noverflow) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__SETSTROS;
            G__inc_cp_asm(1, 0);
         }
#endif // G__ASM
         // --
      }
      else {
         G__tagnum = store_tagnumB;
         G__store_struct_offset = store_struct_offsetB;
         if (
            /* #ifdef G__ROOT */
            G__dispmsg >= G__DISPROOTSTRICT ||
            /* #endif */
            G__ifile.filenum <= G__gettempfilenum()
         ) {
            if (G__dispmsg >= G__DISPWARN) {
               G__fprinterr(G__serr, "Warning: wrong member access operator '->'  %s:%d\n", __FILE__, __LINE__);
               G__printlinenum();
            }
         }
      }
   }
   if (G__value_typenum(result).FinalType().IsPointer() && (objptr == 1)) {
      if (
         (G__dispmsg >= G__DISPROOTSTRICT) ||
         (G__ifile.filenum <= G__gettempfilenum())
      ) {
         if (G__dispmsg >= G__DISPWARN) {
            G__fprinterr(G__serr, "Warning: wrong member access operator '.'  %s:%d\n", __FILE__, __LINE__);
            G__printlinenum();
         }
      }
   }
   if ((objptr == 2) && G__initval_eval) {
      G__dynconst = G__DYNCONST;
   }
   //
   // Get variable value.
   //
   // If membername includes another hierarchy of struct,
   // union member, G__getvariable() will be recursively
   // called from following G__getvariable().
   //
   store_do_setmemfuncenv = G__do_setmemfuncenv;
   G__do_setmemfuncenv = 1;
   G__incsetup_memvar(G__tagnum);
   result = G__getvariable(membername, known2,::Reflex::Scope(), G__tagnum);
   //
   //  if !*known2 'tag.func()'
   //               --- ------
   //               tagname membername
   //  or 'tag.func().func2()'
   //
   // should call G__getfunction() for interpreted member function.
   //
   if (!*known2) {
      if (membername[0] == '&') {
         G__var_typeB = 'P';
         result = G__getfunction(membername + 1, known2, G__CALLMEMFUNC);
         G__var_typeB = 'p';
      }
      else if (membername[0] == '*') {
         G__var_typeB = 'v';
         result = G__getfunction(membername + 1, known2, G__CALLMEMFUNC);
         G__var_typeB = 'p';
      }
      else {
         result = G__getfunction(membername, known2, G__CALLMEMFUNC);
         result = G__toXvalue(result, store_var_type);
      }
   }
   G__do_setmemfuncenv = store_do_setmemfuncenv;
   G__tagnum = store_tagnum;
   G__store_struct_offset = store_struct_offset;
#ifndef G__OLDIMPLEMENTATION1259
   G__isconst = store_isconst;
#endif // G__OLDIMPLEMENTATION1259
   // --
#ifdef G__ASM
   if (G__asm_noverflow) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   return result;
}

//______________________________________________________________________________
int Cint::Internal::G__getthis(G__value* result7, const char* varname, const char* item)
{
   // -- FIXME: Describe me!
   if (G__exec_memberfunc && !strcmp(varname, "this")) {
      if (!G__store_struct_offset) {
         G__genericerror("Error: Can't use 'this' pointer in static member func");
         return 0;
      }
#ifdef G__ASM
      if (G__asm_noverflow) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: LD_THIS %c  %s:%d\n", G__asm_cp, G__asm_dt, G__var_type, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__LD_THIS;
         G__asm_inst[G__asm_cp+1] = G__var_type;
         G__inc_cp_asm(2, 0);
      }
#endif // G__ASM
      switch (G__var_type) {
         case 'v':
            G__letpointer(result7, (long) G__store_struct_offset, G__tagnum);
            result7->ref = (long) G__store_struct_offset;
            break;
         case 'P':
            G__reference_error(item);
            break;
         case 'p':
         default:
            G__letpointer(result7, (long) G__store_struct_offset, ::Reflex::PointerBuilder(G__tagnum));
            break;
      }
      // pointer to struct, class
      G__var_type = 'p';
      //--
      //--
      result7->ref = 0;
      //--
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
void Cint::Internal::G__returnvartype(G__value* presult, const ::Reflex::Member& var, int paran)
{
   // -- FIXME: Describe this function.
   //
   char var_type = G__get_type(var.TypeOf());
   //--
   //--
   //--
   //--
   switch (var_type) {
      case 'p':
      case 'x':
         G__value_typenum(*presult) = G__replace_rawtype(G__value_typenum(*presult), G__get_from_type('i', 0));
         return;
      case 'P':
      case 'X':
         G__value_typenum(*presult) = G__replace_rawtype(G__value_typenum(*presult), G__get_from_type('d', 0));
         return;
#ifndef G__OLDIMPLEMENTATION2191
      case 'j':
#else // G__OLDIMPLEMENTATION2191
      case 'm':
#endif // G__OLDIMPLEMENTATION2191
         G__abortbytecode();
         G__value_typenum(*presult) = G__replace_rawtype(G__value_typenum(*presult), G__get_from_type('i', 0));
         return;
   }
   if (islower(var_type)) {
      switch (G__var_type) {
         case 'p':
            if (G__get_paran(var) <= paran) {
               G__value_typenum(*presult) = var.TypeOf();
            }
            else {
               G__value_typenum(*presult) = ::Reflex::PointerBuilder(var.TypeOf());
            }
            break;
         case 'P':
            G__value_typenum(*presult) = ::Reflex::PointerBuilder(var.TypeOf());
            break;
         default:
            // 'v'
            G__value_typenum(*presult) = var.TypeOf();
            break;
      }
   }
   else {
      switch (G__var_type) {
         case 'v':
            G__value_typenum(*presult) = G__deref(G__value_typenum(*presult));
            break;
         case 'P':
            G__value_typenum(*presult) = ::Reflex::PointerBuilder(var.TypeOf());
            break;
         default:
            // 'p'
            if (G__get_paran(var) == paran) {
               G__value_typenum(*presult) = var.TypeOf();
            }
            else if (G__get_paran(var) < paran) {
               for (int i = paran; i > 0; --i) {
                  G__value_typenum(*presult) = G__value_typenum(*presult).FinalType().ToType();
               }
               //--
               //--
               //--
               //--
               //--
               //--
               //--
               //--
               //--
               //--
               //--
               //--
               //--
               //--
            }
            else {
               G__value_typenum(*presult) = ::Reflex::PointerBuilder(var.TypeOf());
            }
            break;
      }
   }
}

//______________________________________________________________________________
::Reflex::Member Cint::Internal::G__getvarentry(const char* varname, int /*varhash*/, const ::Reflex::Scope& varglobal, const ::Reflex::Scope& varlocal)
{
   // -- FIXME: Describe me!
   ::Reflex::Scope varscope;
   int ilg = 0;
   int in_memfunc = 0;
#ifdef G__NEWINHERIT
   size_t basen;
   int isbase;
   int accesslimit;
   int memfunc_or_friend = 0;
   struct G__inheritance* baseclass = 0;
#endif // G__NEWINHERIT
   ilg = G__LOCAL;
   while (ilg <= (G__GLOBAL + 1)) {
      switch (ilg) {
         case G__LOCAL:
            in_memfunc = 0;
            if (varlocal && !G__def_struct_member) {
               varscope = varlocal;
               if (varglobal) {
                  if (G__exec_memberfunc) {
                     ilg = G__MEMBER;
                  }
                  else {
                     ilg = G__GLOBAL;
                  }
               }
               else {
                  ilg = G__NOTHING;
               }
            }
            else {
               varscope = varglobal;
               ilg = G__NOTHING;
            }
            break;
         case G__MEMBER:
            in_memfunc = 1;
#ifdef G__OLDIMPLEMENTATION589_YET /* not activated due to bug */
            G__ASSERT(0 <= G__memberfunc_tagnum);
            G__incsetup_memvar(G__memberfunc_tagnum);
            varscope = G__memberfunc_tagnum;
#else // G__OLDIMPLEMENTATION589_YET
            if (!G__tagnum.IsTopScope()) {
               G__incsetup_memvar(G__tagnum);
               varscope = G__tagnum;
            }
            else {
               varscope = ::Reflex::Scope::GlobalScope();
            }
#endif // // G__OLDIMPLEMENTATION589_YET
            ilg = G__GLOBAL;
            break;
         case G__GLOBAL:
            in_memfunc = 0;
            varscope = varglobal;
            ilg = G__NOTHING;
            break;
      }
      //
      // Searching for variable name
      //
#ifdef G__NEWINHERIT
      // If searching for class member, check access rule.
      if (in_memfunc || !varglobal) {
         isbase = 1;
         basen = 0;
#ifdef G__OLDIMPLEMENTATION589_YET /* not activated due to bug */
         if (in_memfunc) {
            baseclass = G__struct.baseclass[G__memberfunc_tagnum];
         }
         else {
            baseclass = G__struct.baseclass[G__tagnum];
         }
#else // G__OLDIMPLEMENTATION589_YET
         baseclass = G__struct.baseclass[G__get_tagnum(G__tagnum)];
#endif // G__OLDIMPLEMENTATION589_YET
         if (G__exec_memberfunc || G__isfriend(G__get_tagnum(G__tagnum))) {
            accesslimit = G__PUBLIC_PROTECTED_PRIVATE;
            memfunc_or_friend = 1;
         }
         else {
            accesslimit = G__PUBLIC;
            memfunc_or_friend = 0;
         }
      }
      else {
         if (G__decl) {
            accesslimit = G__PUBLIC_PROTECTED_PRIVATE;
         }
         else {
            accesslimit = G__PUBLIC;
         }
         isbase = 0;
         basen = 0;
      }
      // Search for variable name and access rule match. 
      do {
         next_base:
         ::Reflex::Member var = varscope.DataMemberByName(varname);
         if (var) {
            int statictype = G__get_properties(var)->statictype;
            int have_access = 0;
            if (statictype > -1) {
               have_access = G__filescopeaccess(G__ifile.filenum, statictype); // File scope static access match
            }
            if (
               (statictype < 0) || // Not file scope, or
               have_access // File scope static access match
            ) {
               int ret = G__test_access(var, accesslimit); // Access limit match
               if (ret) {
                  return var;
               }
            }
         }
         //--
         //--
         //--
         //--
         //--
         //--
         //--
         //--
         // Next base class if searching for class member.
         if (isbase) {
            while (baseclass && (basen < baseclass->vec.size())) {
               if (memfunc_or_friend) {
                  if (
                     (baseclass->vec[basen].baseaccess & G__PUBLIC_PROTECTED) ||
                     (baseclass->vec[basen].property & G__ISDIRECTINHERIT)
                  ) {
                     accesslimit = G__PUBLIC_PROTECTED;
                     G__incsetup_memvar(baseclass->vec[basen].basetagnum);
                     varscope = G__Dict::GetDict().GetScope(baseclass->vec[basen].basetagnum);
                     ++basen;
                     goto next_base;
                  }
               }
               else {
                  if (baseclass->vec[basen].baseaccess & G__PUBLIC) {
                     accesslimit = G__PUBLIC;
                     G__incsetup_memvar(baseclass->vec[basen].basetagnum);
                     varscope = G__Dict::GetDict().GetScope(baseclass->vec[basen].basetagnum);
                     ++basen;
                     goto next_base;
                  }
               }
               ++basen;
            }
            isbase = 0;
         }
      }
      while (isbase);
#endif // G__NEWINHERIT
      // --
   }
   return ::Reflex::Member();
}

//______________________________________________________________________________
//--  1
//--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
//-- 10

#ifdef G__FRIEND
//______________________________________________________________________________
int Cint::Internal::G__isfriend(int tagnum)
{
   // -- FIXME: Describe me!
   G__friendtag* friendtag = 0;
   if (G__exec_memberfunc) {
      if (G__get_tagnum(G__memberfunc_tagnum) == tagnum) {
         return 1;
      }
      if (G__get_tagnum(G__memberfunc_tagnum) == -1) {
         return 0;
      }
      friendtag = G__struct.friendtag[G__get_tagnum(G__memberfunc_tagnum)];
      while (friendtag) {
         if (friendtag->tagnum == tagnum) {
            return 1;
         }
         friendtag = friendtag->next;
      }
   }
   if (G__func_now) {
      friendtag = G__get_funcproperties(G__func_now)->entry.friendtag;
      while (friendtag) {
         if (friendtag->tagnum == tagnum) {
            return 1;
         }
         friendtag = friendtag->next;
      }
   }
   return 0;
}
#endif // G__FRIEND

//______________________________________________________________________________
void Cint::Internal::G__get_stack_varname(std::string& output, const char* varname, const ::Reflex::Member& m, int tagnum)
{
   // -- Retrieve the mangled 'name' of a local variable.
   std::ostringstream stream;
   stream << varname << "\\" << std::hex << m.Id() << "\\" << tagnum;
   output = stream.str();
}

//______________________________________________________________________________
::Reflex::Member Cint::Internal::G__find_variable(const char* varname_in, int varhash, const ::Reflex::Scope varlocal, const ::Reflex::Scope varglobal, char** pG__struct_offset, char** pstore_struct_offset, int* pig15, int isdecl)
{
   // -- FIXME: Describe me!
   ::Reflex::Scope var;
   int ig15 = 0;
   int ilg = 0;
   int in_memfunc = 0;
   char* scope_struct_offset = 0;
   ::Reflex::Scope scope_tagnum;
   size_t basen = 0;
   int isbase = 0;
   int accesslimit = 0;
   int memfunc_or_friend = 0;
   struct G__inheritance* baseclass = 0;
#ifdef G__ROOT
   int specialflag = 0;
#endif // G__ROOT
   ::Reflex::Scope save_scope_tagnum;
   std::string varname;
   varname.reserve(2 * strlen(varname_in));
   varname = (char*) varname_in;
#ifdef G__ROOT
   if ((varname[0] == '$') && G__GetSpecialObject && (G__GetSpecialObject != (G__value(*)(char*, void**, void**))G__getreserved)) {
      //varname = ((char*) varname_in) + 1;
      //--
      //--
      specialflag = 1;
   }
#endif // G__ROOT
   ilg = G__LOCAL;
   scope_struct_offset = G__store_struct_offset;
   if (G__def_struct_member) {
      scope_tagnum = G__get_envtagnum();
   }
   else if (G__decl && G__exec_memberfunc && (G__get_tagnum(G__memberfunc_tagnum) != -1)) {
      scope_tagnum = G__memberfunc_tagnum;
   }
   else {
      scope_tagnum = G__tagnum;
   }
   int intScopeTagnum = G__get_tagnum(scope_tagnum);
   switch (G__scopeoperator(/*FIXME*/(char*) varname.c_str(), &varhash, &scope_struct_offset, &intScopeTagnum)) {
      case G__GLOBALSCOPE:
         ilg = G__GLOBAL;
         break;
      case G__CLASSSCOPE:
         ilg = G__MEMBER;
         break;
   }
   scope_tagnum = G__Dict::GetDict().GetScope(intScopeTagnum);
   save_scope_tagnum = scope_tagnum;
   while (ilg <= (G__GLOBAL + 1)) {
      scope_tagnum = save_scope_tagnum;
      // Switch local and global for letvariable.
      switch (ilg) {
         case G__LOCAL:
            // --
#ifdef G__NEWINHERIT
            in_memfunc = 0;
#else // G__NEWINHERIT
            in_memfunc = G__def_struct_member;
#endif // G__NEWINHERIT
            // Beginning, local or global entry.
            if (varlocal) {
               // --
#ifdef G__ASM_WHOLEFUNC
               *pstore_struct_offset = (char*) G__ASM_VARLOCAL;
#endif // G__ASM_WHOLEFUNC
               var = varlocal;
               if (varglobal && !isdecl) {
                  if (G__exec_memberfunc || ((G__get_tagnum(G__tagdefining) != -1) && (G__get_tagnum(scope_tagnum) != -1))) {
                     ilg = G__MEMBER;
                  }
                  else {
                     ilg = G__GLOBAL;
                  }
               }
               else {
                  ilg = G__NOTHING;
               }
            }
            else {
               var = varglobal;
               ilg = G__NOTHING;
            }
            break;
         case G__MEMBER:
            if (G__get_tagnum(scope_tagnum) != -1) {
               in_memfunc = 1;
               *pG__struct_offset = scope_struct_offset;
               G__incsetup_memvar((scope_tagnum));
               var = scope_tagnum;
            }
            else {
               in_memfunc = 0;
               *pG__struct_offset = scope_struct_offset;
               var = ::Reflex::Scope();
            }
            ilg = G__GLOBAL;
            break;
         case G__GLOBAL:
            // -- Global entry.
            in_memfunc = 0;
            *pG__struct_offset = 0;
#ifdef G__ASM_WHOLEFUNC
            *pstore_struct_offset = (char*) G__ASM_VARGLOBAL;
#endif // G__ASM_WHOLEFUNC
            var = varglobal;
            ilg = G__NOTHING;
            break;
      }
      // Searching for hash and variable name.
      // If searching for class member, check access rule.
      if (in_memfunc || !varglobal) {
         *pstore_struct_offset = *pG__struct_offset;
         isbase = 1;
         basen = 0;
         baseclass = G__struct.baseclass[G__get_tagnum(scope_tagnum)];
         if (G__exec_memberfunc || isdecl || G__isfriend(G__get_tagnum(G__tagnum))) {
            accesslimit = G__PUBLIC_PROTECTED_PRIVATE;
            memfunc_or_friend = 1;
         }
         else {
            accesslimit = G__PUBLIC;
            memfunc_or_friend = 0;
         }
      }
      else {
         accesslimit = G__PUBLIC;
         isbase = 0;
         basen = 0;
         if (var && (var == varglobal)) {
            isbase = 1;
            baseclass = &G__globalusingnamespace;
         }
      }
      // Search for variable name and access rule match.
      do {
         next_base:
         ig15 = 0;
         const ::Reflex::Member_Iterator end( var.DataMember_End() );
         for (::Reflex::Member_Iterator iter = var.DataMember_Begin(); iter != end; ++iter, ++ig15) {
            assert( (bool)(*iter) );
            const char *mname = iter->Name_c_str();
            if (mname[0]==varname[0] && 0==strcmp(mname, varname.c_str())) { // Names match
               if (
                  (
                   (G__get_properties(*iter)->statictype < 0) || // Not file scope, or
                    G__filescopeaccess(G__ifile.filenum, G__get_properties(*iter)->filenum) // File scope access match.
                  )
               ) {
                  if (G__test_access(*iter, accesslimit)) { // Access limit match.
                     if (pig15) {
                        *pig15 = ig15;
                     }
                     return *iter;
                  }
               }
            }
         }
         // Next base class if searching for class member.
         if (
            isbase &&
            scope_tagnum &&
            scope_tagnum.IsEnum() &&
            (G__dispmsg >= G__DISPROOTSTRICT)
         ) {
            isbase = 0;
         }
         if (isbase) {
            while (baseclass && (basen < baseclass->vec.size())) {
               if (memfunc_or_friend) {
                  if (
                     (baseclass->vec[basen].baseaccess & G__PUBLIC_PROTECTED) ||
                     (baseclass->vec[basen].property & G__ISDIRECTINHERIT)
                  ) {
                     accesslimit = G__PUBLIC_PROTECTED;
                     G__incsetup_memvar(baseclass->vec[basen].basetagnum);
                     var = G__Dict::GetDict().GetScope(baseclass->vec[basen].basetagnum);
#ifdef G__VIRTUALBASE
                     if (baseclass->vec[basen].property & G__ISVIRTUALBASE) {
                        *pG__struct_offset = *pstore_struct_offset + G__getvirtualbaseoffset(*pstore_struct_offset, G__get_tagnum(scope_tagnum), baseclass, basen);
                     }
                     else {
                        *pG__struct_offset = *pstore_struct_offset + (size_t) baseclass->vec[basen].baseoffset;
                     }
#else // G__VIRTUALBASE
                     *pG__struct_offset = *pstore_struct_offset + (size_t) baseclass->vec[basen].baseoffset;
#endif // G__VIRTUALBASE
                     ++basen;
                     goto next_base;
                  }
               }
               else {
                  if (baseclass->vec[basen].baseaccess & G__PUBLIC) {
                     accesslimit = G__PUBLIC;
                     G__incsetup_memvar(baseclass->vec[basen].basetagnum);
                     var = G__Dict::GetDict().GetScope(baseclass->vec[basen].basetagnum);
#ifdef G__VIRTUALBASE
                     if (baseclass->vec[basen].property & G__ISVIRTUALBASE) {
                        *pG__struct_offset = *pstore_struct_offset + G__getvirtualbaseoffset(*pstore_struct_offset, G__get_tagnum(scope_tagnum), baseclass, basen);
                     }
                     else {
                        *pG__struct_offset = *pstore_struct_offset + (size_t) baseclass->vec[basen].baseoffset;
                     }
#else // G__VIRTUALBASE
                     *pG__struct_offset = *pstore_struct_offset + (size_t) baseclass->vec[basen].baseoffset;
#endif // G__VIRTUALBASE
                     ++basen;
                     goto next_base;
                  }
               }
               ++basen;
            }
            // Also search enclosing scopes.
            if (
               scope_tagnum &&
               (baseclass != &G__globalusingnamespace) &&
               (G__get_tagnum(scope_tagnum.DeclaringScope()) != -1)
            ) {
               scope_tagnum = scope_tagnum.DeclaringScope();
               basen = 0;
               baseclass = G__struct.baseclass[G__get_tagnum(scope_tagnum)];
               var = scope_tagnum;
               goto next_base;
            }
            isbase = 0;
         }
      }
      while (isbase);
      // Not found.
      *pG__struct_offset = *pstore_struct_offset;
   }
#ifdef G__ROOT
   if (specialflag) {
      Reflex::Scope store_local = G__p_local;
      G__p_local = Reflex::Scope();
      int store_var_type = G__var_type;
      G__var_type = 'Z';
      G__value para[1];
      std::string final_varname = varname;
      Reflex::Member m;
      G__allocvariable(G__null, para, varglobal, ::Reflex::Scope(), 0, varhash, varname.c_str(), final_varname, 0, m);
      G__var_type = store_var_type;
      G__p_local = store_local;
      // ::Reflex::Member m = G__find_variable(final_varname.c_str(), varhash, varlocal, varglobal, pG__struct_offset, pstore_struct_offset, pig15, isdecl);
      if (m) {
         G__gettingspecial = 0;
         return m;
      }
   }
#endif // G__ROOT
   return ::Reflex::Member();
}

//______________________________________________________________________________
//
// Functions in the C api.
//

//______________________________________________________________________________
extern "C" int G__IsInMacro()
{
   // -- FIXME: Describe me!
   if ((G__nfile > G__ifile.filenum) || (G__dispmsg >= G__DISPROOTSTRICT)) {
      return 0;
   }
   return 1;
}

//______________________________________________________________________________
extern "C" struct G__var_array* G__searchvariable(char* varname, int varhash, G__var_array* varlocal, G__var_array* varglobal, long* pG__struct_offset, long* pstore_struct_offset, int* pig15, int isdecl)
{
   // -- FIXME: Describe me!
   ::Reflex::Scope local = G__Dict::GetDict().GetScope(varlocal);
   ::Reflex::Scope global = G__Dict::GetDict().GetScope(varglobal);
   ::Reflex::Member m = G__find_variable(varname, varhash, local, global, (char**) pG__struct_offset, (char**) pstore_struct_offset, pig15, isdecl);
   if (m) {
      return (G__var_array*) m.DeclaringScope().Id();
   }
   return 0;
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
   //-- 10
   //--  1
   //--  2
   //--  3
   //--  4
   //--  5
   //--  6
   //--  7
   //--  8
   //--  9
}

//______________________________________________________________________________
extern "C" int G__deletevariable(const char* varname)
{
   // -- Delete variable from global variable table.  Return 1 if successful.
   char *struct_offset = 0;
   char *store_struct_offset = 0;
   int ig15 = 0;
   int varhash = 0;
   int isdecl = 0;
   ::Reflex::Member var;
   int cpplink = G__NOLINK;
   G__hash(varname, varhash, ig15);
   var = G__find_variable(varname, varhash, ::Reflex::Scope(), ::Reflex::Scope::GlobalScope(), &struct_offset, &store_struct_offset, &ig15, isdecl);
   if (var) {
      int i;
      int done;
      ::Reflex::Scope store_tagnum;
      std::string temp("~");
      switch (G__get_type(var.TypeOf())) {
         case 'u':
            store_struct_offset = G__store_struct_offset;
            store_tagnum = G__tagnum;
            G__store_struct_offset = G__get_offset(var);
            G__set_G__tagnum(var.TypeOf());
            temp += var.Name_c_str();
            temp += "()";
            // Destruction of array.
            if (G__struct.iscpplink[G__get_tagnum(G__tagnum)] == G__CPPLINK) {
               G__store_struct_offset = G__get_offset(var);
               i = G__get_varlabel(var.TypeOf(), 1) /* number of elements */;
               if ((i > 0) || G__get_paran(var)) {
                  G__cpp_aryconstruct = i;
               }
               G__getfunction((char*) temp.c_str(), &done, G__TRYDESTRUCTOR);
               G__cpp_aryconstruct = 0;
               cpplink = G__CPPLINK;
            }
            else {
               int size = G__struct.size[G__get_tagnum(G__tagnum)];
               int nelem = G__get_varlabel(var.TypeOf(), 1) /* number of elements */;
               if (!nelem) {
                  nelem = 1;
               }
               --nelem;
               for (; nelem >= 0; --nelem) {
                  G__store_struct_offset = G__get_offset(var) + (nelem * size);
                  if (G__dispsource) {
                     G__fprinterr(G__serr, "\n0x%lx.%s", G__store_struct_offset, temp.c_str());
                  }
                  done = 0;
                  G__getfunction((char*) temp.c_str(), &done, G__TRYDESTRUCTOR);
                  if (!done) {
                     break;
                  }
               }
               G__tagnum = store_tagnum;
               G__store_struct_offset = store_struct_offset;
            }
            break;
         default:
#ifdef G__SECURITY
            if (
               G__security & G__SECURE_GARBAGECOLLECTION &&
               !G__no_exec_compile &&
               isupper(G__get_type(var.TypeOf())) &&
               G__get_offset(var)
            ) {
               long address;
               i = G__get_varlabel(var.TypeOf(), 1) /* number of elements */;
               if (!i) {
                  i = 1;
               }
               --i;
               for (; i >= 0; --i) {
                  address = ((size_t) G__get_offset(var)) + (i * G__LONGALLOC);
                  if (*((long*) address)) {
                     G__del_refcount((void*) (*((long*) address)), (void**) address);
                  }
               }
            }
#endif // G__SECURITY
            break;
      }
      if ((cpplink == G__NOLINK) && G__get_offset(var)) {
         free(G__get_offset(var));
      }
      G__get_offset(var) = 0;
      var.DeclaringScope().RemoveDataMember(var);
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
extern "C" int G__deleteglobal(void* pin)
{
   // -- Delete variable from global variable table. return 1 if successful.
   char* p = (char*) pin;
   G__CriticalSection lock;
   ::Reflex::Scope var = ::Reflex::Scope::GlobalScope();
   for (::Reflex::Member_Iterator m = var.DataMember_Begin(); m != var.DataMember_End(); ++m) {
      if (p == G__get_offset(*m)) { // addr match
         G__get_offset(*m) = 0;
         var.RemoveDataMember(*m);
         break;
      }
      else if ( // pointer member matches addr
         isupper(G__get_type(m->TypeOf())) && // member is a pointer, and
         G__get_offset(*m) && // not nil, and
         (p == (*(char**)G__get_offset(*m))) // we have an addr match
      ) { // pointer member matches addr
         if (G__get_properties(*m)->globalcomp == G__AUTO) { // FIXME: A G__AUTO global var, how can that be?
            free(G__get_offset(*m));
         }
         G__get_offset(*m) = 0;
         var.RemoveDataMember(*m);
         break;
      }
   }
   return 0;
}

//--

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
