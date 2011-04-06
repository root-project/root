/* /% C %/ */
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
#include "DataMemberHandle.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

using namespace Cint;

static G__value G__allocvariable(G__value result, G__value para[], G__var_array* varglobal, G__var_array* varlocal, int paran, int varhash, const char* item, char* varname, int parameter00, G__DataMemberHandle &member);

extern "C" {


int G__filescopeaccess(int filenum, int statictype);
static int G__asm_gen_stvar(long G__struct_offset, int ig15, int paran, G__var_array* var, const char* item, long store_struct_offset, int var_type, G__value* presult);
//--
//--

static int G__getarraydim = 0;

//______________________________________________________________________________
void psrxxx_dump_gvars()
{
   G__var_array* var = &G__global;
   for (; var; var = var->next) {
      if (var->allvar) {
         fprintf(stderr, "name: '%s'\n", var->varnamebuf[0]);
      }
   }
}

//______________________________________________________________________________
void psrxxx_dump_lvars()
{
   G__var_array* var = G__p_local;
   for (; var; var = var->next) {
      if (var->allvar) {
         fprintf(stderr, "name: '%s'\n", var->varnamebuf[0]);
      }
   }
}

//______________________________________________________________________________
//
// Static functions.
//

//______________________________________________________________________________
static void G__redecl(G__var_array* var, int ig15)
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
      G__asm_inst[G__asm_cp+1] = ig15;
      G__asm_inst[G__asm_cp+2] = (long)var;
      G__inc_cp_asm(3, 0);
   }
}

//______________________________________________________________________________
static void G__class_2nd_decl(G__var_array* var, int ig15)
{
   // -- FIXME: Describe me!
   int tagnum = var->p_tagtable[ig15];
   int store_var_type = G__var_type;
   G__var_type = 'p';
   int store_tagnum = G__tagnum;
   G__tagnum = tagnum;
   long store_struct_offset = G__store_struct_offset;
   G__store_struct_offset = var->p[ig15];
   long store_globalvarpointer = G__globalvarpointer;
   G__globalvarpointer = G__PVOID;
   int store_cpp_aryconstruct = G__cpp_aryconstruct;
   if (var->varlabel[ig15][1] /* number of elements */ || var->paran[ig15]) {
      G__cpp_aryconstruct = var->varlabel[ig15][1] /* number of elements */;
   }
   else {
      G__cpp_aryconstruct = 0;
   }
   int store_decl = G__decl;
   G__decl = 0;
   G__FastAllocString temp(G__ONELINE);
   temp.Format("~%s()", G__struct.name[tagnum]);
   if (G__dispsource) {
      G__fprinterr(
           G__serr
         , "\n!!!Calling destructor 0x%lx.%s for declaration of %s  %s:%d\n"
         , G__store_struct_offset
         , temp()
         , var->varnamebuf[ig15]
         , __FILE__
         , __LINE__
      );
   }
   if (G__struct.iscpplink[tagnum] == G__CPPLINK) {
      // Delete current object.
      if (var->p[ig15]) {
         int known = 0;
         G__getfunction(temp, &known, G__TRYDESTRUCTOR);
      }
      // Set newly constructed object.
      var->p[ig15] = store_globalvarpointer;
      if (G__dispsource) {
         G__fprinterr(G__serr, " 0x%lx is set", store_globalvarpointer);
      }
   }
   else {
      if (G__cpp_aryconstruct) {
         for (int i = G__cpp_aryconstruct - 1; i >= 0; --i) {
            // Call destructor without freeing memory.
            G__store_struct_offset = var->p[ig15] + (i * G__struct.size[tagnum]);
            int known = 0;
            if (var->p[ig15]) {
               G__getfunction(temp, &known, G__TRYDESTRUCTOR);
            }
            if ((G__return > G__RETURN_NORMAL) || !known) {
               break;
            }
         }
      }
      else {
         G__store_struct_offset = var->p[ig15];
         if (var->p[ig15]) {
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
static void G__class_2nd_decl_i(G__var_array* var, int ig15)
{
   // -- FIXME: Describe me!
   int store_no_exec_compile = G__no_exec_compile;
   G__no_exec_compile = 1;
   int store_tagnum = G__tagnum;
   G__tagnum = var->p_tagtable[ig15];
   long store_struct_offset = G__store_struct_offset;
   long store_globalvarpointer = G__globalvarpointer;
   G__globalvarpointer = G__PVOID;
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "%3x,%3x: LD_VAR  %s index=%d paran=%d  %s:%d\n", G__asm_cp, G__asm_dt, var->varnamebuf[ig15], ig15, 0, __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
   G__asm_inst[G__asm_cp] = G__LD_VAR;
   G__asm_inst[G__asm_cp+1] = ig15;
   G__asm_inst[G__asm_cp+2] = 0;
   G__asm_inst[G__asm_cp+3] = 'p';
   G__asm_inst[G__asm_cp+4] = (long) var;
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
   G__FastAllocString temp(G__ONELINE);
   temp.Format("~%s()", G__struct.name[G__tagnum]);
   if (var->varlabel[ig15][1] /* number of elements */ || var->paran[ig15]) {
      // array
      int size = G__struct.size[G__tagnum];
      int pinc = var->varlabel[ig15][1] /* number of elements */;
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
static void G__class_2nd_decl_c(G__var_array* var, int ig15)
{
   // -- FIXME: Describe me!
   long store_globalvarpointer = G__globalvarpointer;
   G__globalvarpointer = G__PVOID;
   int store_no_exec_compile = G__no_exec_compile;
   G__no_exec_compile = 1;
   int store_tagnum = G__tagnum;
   G__tagnum = var->p_tagtable[ig15];
   long store_struct_offset = G__store_struct_offset;
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "%3x,%3x: LD_VAR  %s index=%d paran=%d  %s:%d\n", G__asm_cp, G__asm_dt, var->varnamebuf[ig15], ig15, 0, __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
   G__asm_inst[G__asm_cp] = G__LD_VAR;
   G__asm_inst[G__asm_cp+1] = ig15;
   G__asm_inst[G__asm_cp+2] = 0;
   G__asm_inst[G__asm_cp+3] = 'p';
   G__asm_inst[G__asm_cp+4] = (long) var;
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
   G__FastAllocString temp(G__ONELINE);
   temp.Format("~%s()", G__struct.name[G__tagnum]);
   int known = 0;
   G__getfunction(temp, &known, G__TRYDESTRUCTOR);
   G__redecl(var, ig15);
   if (store_no_exec_compile) {
      G__abortbytecode();
   }
   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;
   G__no_exec_compile = store_no_exec_compile;
   G__globalvarpointer = store_globalvarpointer;
}

//______________________________________________________________________________
static void G__getpointer2pointer(G__value* presult, G__var_array* var, int ig15, int paran)
{
   // -- FIXME: Describe me!
   switch (G__var_type) {
      case 'v':
         switch (var->reftype[ig15]) {
            case G__PARANORMAL:
               if (var->paran[ig15] > paran) {
                  G__letint(presult, var->type[ig15], *((long*) presult->ref));
               }
               break;
            case G__PARAREFERENCE:
               break;
            case G__PARAP2P:
               G__letint(presult, var->type[ig15], *(long*)presult->ref);
               presult->obj.reftype.reftype = G__PARANORMAL;
               break;
            case G__PARAP2P2P:
               G__letint(presult, var->type[ig15], *(long*)presult->ref);
               presult->obj.reftype.reftype = G__PARAP2P;
               break;
            default:
               G__letint(presult, var->type[ig15], *(long*)presult->ref);
               presult->obj.reftype.reftype = var->reftype[ig15] - 1;
               break;
         }
         break;
      case 'p':
         if (paran < var->paran[ig15]) {
            switch (var->reftype[ig15]) {
               case G__PARANORMAL:
                  presult->obj.reftype.reftype = G__PARAP2P;
                  break;
               case G__PARAP2P:
               default:
                  presult->obj.reftype.reftype = G__PARAP2P2P;
                  break;
            }
            presult->obj.reftype.reftype += var->paran[ig15] - paran - 1;


         }
         else if (paran == var->paran[ig15]) {
            presult->obj.reftype.reftype = var->reftype[ig15];
         }
         break;
      case 'P':
         /* this part is not precise. Should handle like above 'p' case */
         if (var->paran[ig15] == paran) { /* must be PPTYPE */
            switch (var->reftype[ig15]) {
               case G__PARANORMAL:
                  presult->obj.reftype.reftype = G__PARAP2P;
                  break;
               case G__PARAP2P:
                  presult->obj.reftype.reftype = G__PARAP2P2P;
                  break;
               default:
                  presult->obj.reftype.reftype = var->reftype[ig15] + 1;
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

#define G__ASSIGN_VAR(SIZE, CASTTYPE, CONVFUNC, X) \
   switch (G__var_type) { \
      case 'v': \
         /* Assign by dereferencing a pointer.  *var = result; */ \
         if (((paran + 1) == var->paran[ig15]) && !linear_index && islower(result.type)) { \
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
         if (paran >= var->paran[ig15]) { \
            /* MyType var[ddd]; var[xxx] = result; */ \
            /* FIXME: The greater than case requires pointer arithmetic. */ \
            result.ref = G__struct_offset + var->p[ig15] + (linear_index * SIZE); \
            *((CASTTYPE*) result.ref) = (CASTTYPE) CONVFUNC(result); \
            result.type = var->type[ig15]; \
            result.obj.reftype.reftype = G__PARANORMAL; \
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
   return result;

//______________________________________________________________________________
//
// -- Change a pointer variable's value.
//
#define G__ASSIGN_PVAR(CASTTYPE, CONVFUNC, X) \
   switch (G__var_type) { \
      case 'v': \
         /* -- Assign to what pointer is pointing at. */ \
         switch (var->reftype[ig15]) { \
            case G__PARANORMAL: \
               /* -- Assignment through a pointer dereference.  MyType* var; *var = result; */ \
               result.ref = *((long*) (G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC))); \
               *((CASTTYPE*) result.ref) = (CASTTYPE) CONVFUNC(result); \
               result.type = tolower(var->type[ig15]); \
               X = *((CASTTYPE*) result.ref); \
               break; \
            case G__PARAP2P: \
               /* -- Assignment through a pointer to a pointer dereference. */ \
               if (paran > var->paran[ig15]) { \
                  /* -- Pointer to array reimplementation. */ \
                  /* MyType** var[ddd]; *var[xxx][yyy] = result; */ \
                  address = G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC); \
                  result.ref = *(((long*) (*((long *) address))) + secondary_linear_index); \
                  *((CASTTYPE*) result.ref) = (CASTTYPE) CONVFUNC(result); \
                  result.type = std::tolower(var->type[ig15]); \
                  X = *((CASTTYPE*) result.ref); \
               } \
               else { \
                  /* paran <= var->paran[ig15] */ \
                  /* MyType** var[ddd][nnn]; *var[xxx][yyy] = result; */ \
                  result.ref = *((long*) (G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC))); \
                  *((long*) result.ref) = G__int(result); \
               } \
               break; \
         } \
         break; \
      case 'p': \
         /* Assign to the pointer variable itself, no dereferencing. */ \
         if (paran > var->paran[ig15]) { \
            /* -- Pointer to array reimplementation. */ \
            /* -- More array dimensions used than variable has, start using up pointer to pointers. */ \
            address = G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC); \
            if (var->reftype[ig15] == G__PARANORMAL) { \
               result.ref = (long) (((CASTTYPE*) (*((long*) address))) + secondary_linear_index); \
               *((CASTTYPE*) result.ref) = (CASTTYPE) CONVFUNC(result); \
               result.type = std::tolower(var->type[ig15]); \
               X = *((CASTTYPE*) result.ref); \
            } \
            else if (var->paran[ig15] == (paran - 1)) { \
               /* -- One extra dimension. */ \
               result.ref = (long) (((long*) (*((long*) address))) + secondary_linear_index); \
               *((long*) result.ref) = G__int(result); \
            } \
            else { \
               /* -- Two or more extra dimensions. */ \
               long* phyaddress = (long*) (*((long*) address)); \
               for (int ip = var->paran[ig15]; ip < (paran - 1); ++ip) { \
                  phyaddress = (long*) phyaddress[para[ip].obj.i]; \
               } \
               switch (var->reftype[ig15] - paran + var->paran[ig15]) { \
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
            result.ref = G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC); \
            *((long*) result.ref) = G__int(result); \
         } \
         break; \
      default: \
         G__assign_error(item, &result); \
         break; \
   }

} // extern "C"

//______________________________________________________________________________
G__value G__letvariable(G__FastAllocString &item, G__value expression, G__var_array* varglobal, G__var_array* varlocal)
{
   static G__DataMemberHandle member;
   return G__letvariable(item,expression,varglobal,varlocal,member);
}

//______________________________________________________________________________
G__value G__letvariable(G__FastAllocString &item, G__value expression, G__var_array* varglobal, G__var_array* varlocal, G__DataMemberHandle &member)
{
   // -- FIXME: Describe me!
   struct G__var_array* var = 0;
   int ig15 = 0;
   int paran = 0;
   int ig25 = 0;
   size_t lenitem = 0;
   int single_quote = 0;
   int double_quote = 0;
   int paren = 0;
   int done = 0;
   size_t linear_index = 0;
   size_t secondary_linear_index = 0;
   size_t ig2 = 0;
   int flag = 0;
   int store_var_type = 0;
   long G__struct_offset = 0L;
   char* tagname = 0;
   int membername = -1;
   int varhash = 0;
   long address = 0L;
   long store_struct_offset = 0L;
   int store_tagnum = 0;
   int store_exec_memberfunc = 0;
   int store_def_struct_member = 0;
   int store_vartype = 0;
   int store_asm_wholefunction = 0;
   int store_no_exec_compile = 0;
   int store_no_exec = 0;
   int store_getarraydim = 0;
   int store_asm_noverflow = 0;
   G__FastAllocString ttt(G__ONELINE);
   G__FastAllocString result7(G__ONELINE);
   char parameter[G__MAXVARDIM][G__ONELINE];
   G__value para[G__MAXVARDIM];
   G__FastAllocString varname(G__BUFLEN);
   //--
   G__value result = G__null;
#ifdef G__ASM
   if (G__asm_exec) {
      ig15 = G__asm_index;
      paran = G__asm_param->paran;
      for (int i = 0; i < paran; ++i) {
         para[i] = G__asm_param->para[i];
      }
      para[paran] = G__null;
      var = varglobal;
      if (!varlocal) {
         G__struct_offset = 0;
      }
      else {
         G__struct_offset = G__store_struct_offset;
      }
      result = expression;
      goto exec_asm_letvar;
   }
#endif // G__ASM
   parameter[0][0] = '\0';
   lenitem = std::strlen(item);
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
            G__ASSERT(isupper(result.type) || (result.type == 'u'));
            G__value tmp = G__letPvalue(&result, expression);
            return tmp;
         }
         {
            int pointlevel = 0;
            int i = 0;
            if (isupper(G__var_type)) {
               ++pointlevel;
            }
            while (item[i++] == '*') {
               ++pointlevel;
            }
            switch (pointlevel) {
               case 0:
                  break;
               case 1:
                  if (G__reftype != G__PARAREFERENCE) {
                     G__reftype = G__PARANORMAL;
                  }
                  break;
               default:
                  if (G__reftype != G__PARAREFERENCE) {
                     G__reftype = G__PARAP2P + pointlevel - 2;
                  }
                  else {
                     G__reftype = G__PARAREFP2P + pointlevel - 2;
                  }
                  break;
            }
            ttt = item + i - 1;
            item = ttt;
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
            if (result.isconst & G__CONSTVAR) {
               G__changeconsterror(item, "ignored const");
               return result;
            }
            G__value tmp = G__letVvalue(&result, expression);
            return tmp;
         }
      case '&':
         // -- Should not happen!
         G__var_type = 'P';
         ttt = item + 1;
         item = ttt;
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
         G__fprinterr(G__serr, "Error: assignment to %s", item());
         G__genericerror(0);
         break;
   }
   store_var_type = G__var_type;
   G__var_type = 'p';
   // struct, union member
   lenitem = std::strlen(item);
   ig2 = 0;
   flag = 0;
   while ((ig2 < lenitem) && !flag) {
      switch (item[ig2]) {
         case '.':
            if (!paren && !double_quote && !single_quote) {
               result7 = item;
               result7.Set(ig2++, 0);
               tagname = result7;
               membername = ig2;
               flag = 1;
            }
            break;
         case '-':
            if (!paren && !double_quote && !single_quote && (item[ig2+1] == '>')) {
               result7 = item;
               result7.Set(ig2++, 0);
               result7.Set(ig2++, 0);
               tagname = result7;
               membername = ig2;
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
   single_quote = 0;
   double_quote = 0;
   paren = 0;
   if (flag) {
      result = G__letstructmem(store_var_type, varname, membername, result7, tagname, varglobal, expression, flag, member);
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
      G__struct_offset = G__store_struct_offset;
   }
   else {
      G__struct_offset = 0;
   }
   result = expression;
   // Parse out an identifier, and possibly handle
   // a function call or array indexes.
   {
      // Start at the beginning.
      size_t item_cursor = 0;
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
         varname.Set(item_cursor, c);
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
      varname.Set(item_cursor, 0);
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
            char c = item[item_cursor];
            if ((c == ']') && !nest && !single_quote && !double_quote) {
               break;
            }
            switch (c) {
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
            parameter[paran][idx++] = c;
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
   if ((G__def_tagnum != -1) && G__def_struct_member) {
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
   var = 0;
   ig15 = 0;
   // Avoid searching variables when processing
   // a function-local const static during prerun.
   if (
      (G__func_now < 0) ||  // Not in a function declaration/definition.
      !G__decl ||           // Not in a declaration.
      !G__static_alloc ||   // Not a static/const/enumerator declaration.
      !G__constvar ||       // Not a const.
      !G__prerun            // Not in prerun (we are actually executing).
   ) {
      char* v = (char*) varname();
      int vlen = strlen(v);
      char* p = NULL;
      if (vlen) {
         int nest_angle = 0;
         int nest_square_bracket = 0;
         int nest_paren = 0;
         bool done = false;
         p = v + vlen - 1;
         for (; !done && (p != v); --p) {
            switch (*p) {
               case '<':
                  --nest_angle;
                  break;
               case '>':
                  ++nest_angle;
                  break;
               case '[':
                  --nest_square_bracket;
                  break;
               case ']':
                  ++nest_square_bracket;
                  break;
               case '(':
                  --nest_paren;
                  break;
               case ')':
                  ++nest_paren;
                  break;
               case ':':
                  if (nest_angle || nest_square_bracket || nest_paren) {
                     continue;
                  }
                  if (*(p - 1) == ':') {
                     done = true;
                  } 
                  break;
               default:
                  break;
            }
         }
         if (done) {
            ++p;
         }
         else {
            p = NULL;
         }
      }
      if ((p != NULL) && varglobal) {
         int qual_id_len = p - v - 1;
         if (qual_id_len > 0) {
            G__FastAllocString qual_id(G__BUFLEN);
            qual_id.Format("%*.*s", qual_id_len, qual_id_len, v);
            int qual_id_tagnum = G__defined_tagname(qual_id(), 0);
            if (qual_id_tagnum != -1) {
               int store_tagnum = G__tagnum;
               G__tagnum = qual_id_tagnum;
               var = G__searchvariable(varname, varhash, G__struct.memvar[qual_id_tagnum], varglobal, &G__struct_offset, &store_struct_offset, &ig15, 1);
               G__tagnum = store_tagnum;
            }
         }
      }
      else {
         var = G__searchvariable(varname, varhash, varlocal, varglobal, &G__struct_offset, &store_struct_offset, &ig15, G__decl || G__def_struct_member);
      }
   }
   //
   // Assign value.
   //
   if (var) {
      // -- We have found a variable.
      member.Set(var,ig15);
      //
      //  Block duplicate declaration.
      //
      G__ASSERT(!G__decl || (G__decl == 1));
      if (
         (G__decl || G__cppconstruct) &&
         (G__var_type != 'p') &&
         (var->statictype[ig15] == G__AUTO) &&
         (
            (var->type[ig15] != G__var_type) ||
            (var->p_tagtable[ig15] != G__tagnum)
         )
      ) {
         G__fprinterr(G__serr, "Error: %s already declared as different type", item());
         if (
            isupper(var->type[ig15]) &&
            isupper(G__var_type) &&
            !var->varlabel[ig15][1] /* number of elements */ &&
            (*((long*) var->p[ig15]) == 0)
         ) {
            G__fprinterr(G__serr, ". Switch to new type\n");
            var->type[ig15] = G__var_type;
            var->p_tagtable[ig15] = G__tagnum;
            var->p_typetable[ig15] = G__typenum;
         }
         else {
            if (
               (G__globalvarpointer != G__PVOID) &&
               (G__var_type == 'u') &&
               (G__tagnum != -1) &&
               (G__struct.iscpplink[G__tagnum] == G__CPPLINK)
            ) {
               G__FastAllocString protect_temp(G__ONELINE);
               long protect_struct_offset = G__store_struct_offset;
               int done = 0;
               G__store_struct_offset = G__globalvarpointer;
               G__globalvarpointer = G__PVOID;
               protect_temp.Format("~%s()", G__struct.name[G__tagnum]);
               G__fprinterr(G__serr, ". %s called\n", protect_temp());
               G__getfunction(protect_temp, &done, G__TRYDESTRUCTOR);
               G__store_struct_offset = protect_struct_offset;
            }
            G__genericerror(0);
            return G__null;
         }
      }
      //
      //
      //
      if (
         (tolower(var->type[ig15]) != 'u') &&
         (result.type == 'u') &&
         (result.tagnum != -1)
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
         G__fundamental_conversion_operator(var->type[ig15], var->p_tagtable[ig15], var->p_typetable[ig15], var->reftype[ig15], var->constvar[ig15], &result, ttt);
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
         if ((G__var_type != 'v') || (var->type[ig15] != 'u')) {
            // --
            if (result.type) {
               // --
               G__asm_gen_stvar(G__struct_offset, ig15, paran, var, item, store_struct_offset, G__var_type, &result);
            }
            else if (G__var_type == 'u') {
               // --
               G__ASSERT(!G__decl || (G__decl == 1));
               if (G__decl) {
                  // --
                  if (G__reftype) {
                     // --
                     G__redecl(var, ig15);
                     if (G__no_exec_compile) {
                        G__abortbytecode();
                     }
                  }
                  else {
                     // --
                     G__class_2nd_decl_i(var, ig15);
                  }
               }
               else if (G__cppconstruct) {
                  // --
                  G__class_2nd_decl_c(var, ig15);
               }
            }
         }
      }
      else if (
         (G__var_type == 'u') &&
         (var->statictype[ig15] == G__AUTO) &&
         (G__decl || G__cppconstruct)
      ) {
         // --
#ifdef G__ASM
         if (G__asm_noverflow) {
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: LD_VAR  %s index=%d paran=%d  %s:%d\n", G__asm_cp, G__asm_dt, var->varnamebuf[ig15], ig15, 0, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__LD_VAR;
            G__asm_inst[G__asm_cp+1] = ig15;
            G__asm_inst[G__asm_cp+2] = 0;
            G__asm_inst[G__asm_cp+3] = 'p';
            G__asm_inst[G__asm_cp+4] = (long) var;
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
         G__class_2nd_decl(var, ig15);
         result.obj.i = var->p[ig15];
         result.type = 'u';
         result.tagnum = var->p_tagtable[ig15];
         result.typenum = var->p_typetable[ig15];
         result.ref = var->p[ig15];
         G__var_type = 'p';
         return result;
      }
      exec_asm_letvar:
#endif // G__ASM
      // Static class/struct member.
      if (
         G__struct_offset &&
         (
            (var->statictype[ig15] == G__LOCALSTATIC)
            // FIXME: Need to test for G__COMPILEDGLOBAL and declaring scope is namespace here!
         )
      ) {
         G__struct_offset = 0;
      }
      // Assign G__null to existing variable is ignored.
      // This is in most cases duplicate declaration.
      if (!result.type) {
         if (
            G__asm_noverflow &&
            (G__var_type == 'u') &&
            (var->statictype[ig15] == G__AUTO) &&
            (G__decl || G__cppconstruct)
         ) {
            int store_asm_noverflow = G__asm_noverflow;
            G__asm_noverflow = 0;
            G__class_2nd_decl(var, ig15);
            G__asm_noverflow = store_asm_noverflow;
            result.obj.i = var->p[ig15];
            result.type = 'u';
            result.tagnum = var->p_tagtable[ig15];
            result.typenum = var->p_typetable[ig15];
            result.ref = var->p[ig15];
         }
         G__var_type = 'p';
         if (G__reftype && (G__globalvarpointer != G__PVOID)) {
            var->p[ig15] = G__globalvarpointer;
         }
         if (G__cppconstruct) {
            var->p[ig15] = G__globalvarpointer;
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
      if (
         var->constvar[ig15] &&
         !G__funcheader &&
         (tolower(var->type[ig15]) != 'p')
      ) {
         if (
            ((!G__prerun && !G__decl) || (var->statictype[ig15] == G__COMPILEDGLOBAL)) &&
            (
               std::islower(var->type[ig15]) ||
               ('p' == G__var_type && (var->constvar[ig15] & G__PCONSTVAR)) ||
               ('v' == G__var_type && (var->constvar[ig15] & G__CONSTVAR))
            )
         ) {
            G__changeconsterror(var->varnamebuf[ig15], "ignored const");
            G__var_type = 'p';
            return result;
         }
      }
      if ((var->p_typetable[ig15] != -1) && G__newtype.isconst[var->p_typetable[ig15]]) {
         int constvar = G__newtype.isconst[var->p_typetable[ig15]];
         if (
            ((!G__prerun && !G__decl) || (var->statictype[ig15] == G__COMPILEDGLOBAL)) &&
            (
             islower(var->type[ig15]) ||
             ((G__var_type == 'p') && (constvar & G__PCONSTVAR)) ||
             ((G__var_type == 'v') && (constvar & G__CONSTVAR))
            )
         ) {
            G__changeconsterror(var->varnamebuf[ig15], "ignored const");
            G__var_type = 'p';
            return result;
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
      linear_index = 0;
      {
         // -- Calculate linear_index
         // tmp = B*C*D
         int tmp = var->varlabel[ig15][0] /* stride */;
         for (ig25 = 0; (ig25 < paran) && (ig25 < var->paran[ig15]); ++ig25) {
            linear_index += tmp * G__int(para[ig25]);
            tmp /= var->varlabel[ig15][ig25+2];
         }
      }
      secondary_linear_index = 0;
      {
         // -- Calculate secondary_linear_index
         // tmp = j*k*m
         int tmp = var->varlabel[ig15][ig25+3];
         while ((ig25 < paran) && var->varlabel[ig15][ig25+4]) {
            secondary_linear_index += tmp * G__int(para[ig25]);
            tmp /= var->varlabel[ig15][ig25+4];
            ++ig25;
         }
      }
#ifdef G__ASM
      if (G__no_exec_compile && ((tolower(var->type[ig15]) != 'u') || (ig25 < paran))) {
         result.obj.d = 0;
         result.obj.i = 1;
         result.tagnum = var->p_tagtable[ig15];
         result.typenum = var->p_typetable[ig15];
         if (isupper(var->type[ig15])) {
            switch (G__var_type) {
               case 'v':
                  result.type = tolower(var->type[ig15]);
                  break;
               case 'P':
                  result.type = var->type[ig15];
                  break;
               default:
                  if (var->paran[ig15] < paran) {
                     result.type = tolower(var->type[ig15]);
                  }
                  else {
                     result.type = var->type[ig15];
                  }
                  break;
            }
         }
         else {
            switch (G__var_type) {
               case 'p':
                  if (var->paran[ig15] <= paran) {
                     result.type = var->type[ig15];
                  }
                  else {
                     result.type = toupper(var->type[ig15]);
                  }
                  if ((result.type == 'u') && (result.tagnum != -1) && (G__struct.type[result.tagnum] != 'e')) {
                     result.ref = 1;
                     G__tryindexopr(&result, para, paran, ig25);
                     para[0] = result;
                     para[0] = G__letVvalue(&para[0], expression);
                  }
                  break;
               case 'P':
                  result.type = toupper(var->type[ig15]);
                  break;
               default:
                  G__reference_error(item);
                  break;
            }
         }
         G__var_type = 'p';
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
         var->varlabel[ig15][1] /* number of elements */ &&
         (var->reftype[ig15] == G__PARANORMAL) &&
         (
          // We intentionally allow the index to go one past the end.
          (linear_index > var->varlabel[ig15][1] /* number of elements */) ||
          ((ig25 < paran) && (std::tolower(var->type[ig15]) != 'u'))
         )
      ) {
         G__arrayindexerror(ig15, var, item, linear_index);
         return expression;
      }
#ifdef G__SECURITY
      if (
         !G__no_exec_compile && // we are not just bytecode compiling, and
         (G__var_type == 'v') && // we are to dereference the var as a ptr, and
         std::isupper(var->type[ig15]) && // the var is of ptr type, and
         (var->reftype[ig15] == G__PARANORMAL) && // is single-level, and
         !var->varlabel[ig15][1] /* number of elements */ && // not an array,
         ((*((long*)(G__struct_offset + var->p[ig15]))) == 0) // has val zero
      ) {
         // Error, attempt to assign using a null pointer.
         G__assign_using_null_pointer_error(item);
         return G__null;
      }
#endif // G__SECURITY
      if (
         (G__security & G__SECURE_POINTER_TYPE) &&
         !G__definemacro &&
         isupper(var->type[ig15]) &&
         (G__var_type == 'p') &&
         !paran &&
#ifndef G__OLDIMPLEMENTATION2191
         (var->type[ig15] != '1') &&
#else // G__OLDIMPLEMENTATION2191
         (var->type[ig15] != 'Q') &&
#endif // G__OLDIMPLEMENTATION2191
         (
            (
               (var->type[ig15] != 'Y') &&
               (result.type != 'Y') &&
               result.obj.i
            ) ||
            (G__security & G__SECURE_CAST2P)
         )
      ) {
         if (
            (var->type[ig15] != result.type) ||
            (
               (result.type == 'U') &&
               (G__security & G__SECURE_CAST2P) &&
#ifdef G__VIRTUALBASE
               (G__ispublicbase(var->p_tagtable[ig15], result.tagnum, G__STATICRESOLUTION2) == -1)
#else // G__VIRTUALBASE
               (G__ispublicbase(var->p_tagtable[ig15], result.tagnum) == -1)
#endif // G__VIRTUALBASE
            // --
         )
         ) {
            G__CHECK(G__SECURE_POINTER_TYPE, 0 != result.obj.i, return G__null);
         }
      }
      G__CHECK(G__SECURE_POINTER_AS_ARRAY, (var->paran[ig15] < paran && isupper(var->type[ig15])), return G__null);
      G__CHECK(G__SECURE_POINTER_ASSIGN, var->paran[ig15] > paran || isupper(var->type[ig15]), return G__null);
#ifdef G__SECURITY
      if (
         G__security & G__SECURE_GARBAGECOLLECTION &&
         !G__no_exec_compile &&
         std::isupper(var->type[ig15]) &&
         (G__var_type != 'v') &&
         (G__var_type != 'P') &&
         (
            (!paran && !var->varlabel[ig15][1] /* number of elements */) ||
            ((paran == 1) && (var->varlabel[ig15][2] == 1) && !var->varlabel[ig15][3])
         )
      ) {
         address = G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC);
         if (address && *((long*) address)) {
            G__del_refcount((void*)(*(long*)address), (void**) address);
         }
         if (std::isupper(result.type) && result.obj.i && address) {
            G__add_refcount((void*) result.obj.i, (void**) address);
         }
      }
#endif // G__SECURITY
      //
      // Assign bit-field value.
      //
      if (var->bitfield[ig15] && (G__var_type == 'p')) {
         int mask;
         int finalval;
         int original;
         address = G__struct_offset + var->p[ig15];
         original = *((int*) address);
         mask = (1 << var->bitfield[ig15]) - 1;
         mask = mask << var->varlabel[ig15][G__MAXVARDIM-1];
         finalval = (original & (~mask)) + ((result.obj.i << var->varlabel[ig15][G__MAXVARDIM-1]) & mask);
         *((int*) address) = finalval;
         return result;
      }
      //
      //  Do the assignment now.
      //
      switch (var->type[ig15]) {
         case 'n':
            // G__int64
            G__ASSIGN_VAR(G__LONGLONGALLOC, G__int64, G__Longlong, result.obj.ll);
            break;
         case 'm':
            // G__uint64
            G__ASSIGN_VAR(G__LONGLONGALLOC, G__uint64, G__ULonglong, result.obj.ull);
            break;
         case 'q':
            // long double
            G__ASSIGN_VAR(G__LONGDOUBLEALLOC, long double, G__Longdouble, result.obj.ld);
            break;
         case 'g':
            // bool
            result.obj.i = G__int(result) ? 1 : 0;
#ifdef G__BOOL4BYTE
            G__ASSIGN_VAR(G__INTALLOC, int, G__int, result.obj.i);
#else // G__BOOL4BYTE
            G__ASSIGN_VAR(G__CHARALLOC, unsigned char, G__int, result.obj.i);
#endif // G__BOOL4BYTE
         case 'i':
            // int
            G__ASSIGN_VAR(G__INTALLOC, int, G__int, result.obj.i);
         case 'd':
            // double
            G__ASSIGN_VAR(G__DOUBLEALLOC, double, G__double, result.obj.d);
         case 'c':
            // char
            {
               //
               // Check for and handle initialization of
               // an unspecified length array first.
               if (
                  G__decl &&
                  (var->varlabel[ig15][1] /* number of elements */ == INT_MAX /* unspecified length flag */) &&
                  paran &&
                  (paran == var->paran[ig15]) &&
                  (G__var_type == 'p') &&
                  !G__struct_offset &&
                  (result.type == 'C') &&
                  result.obj.i
               ) {
                  // -- An unspecified length array of characters initialized by a character pointer.
                  // Release any storage previously allocated.  FIXME: I don't think there is any.
                  if (var->p[ig15]) {
                     free((void*) var->p[ig15]);
                  }
                  // Allocate enough storage for a copy of the initializer string.
                  size_t len = strlen((const char*) result.obj.i);
                  var->p[ig15] = (long) malloc(len + 1);
                  // And copy the initializer into the allocated space.
                  strcpy((char*) var->p[ig15], (const char*) result.obj.i); // Okay, we just allocated enough space
                  // Change the variable into a fixed-size array of characters.
                  var->varlabel[ig15][1] = len;
                  // And return, we are done.
                  return result;
               }
               G__ASSIGN_VAR(G__CHARALLOC, char, G__int, result.obj.i);
            }
         case 'b':
            // unsigned char
            G__ASSIGN_VAR(G__CHARALLOC, unsigned char, G__int, result.obj.i);
         case 's':
            // short int
            G__ASSIGN_VAR(G__SHORTALLOC, short, G__int, result.obj.i);
         case 'r':
            // unsigned short int
            G__ASSIGN_VAR(G__SHORTALLOC, unsigned short, G__int, result.obj.i);
         case 'h':
            // unsigned int
            G__ASSIGN_VAR(G__INTALLOC, unsigned int, G__int, result.obj.i);
         case 'l':
            // long int
            G__ASSIGN_VAR(G__LONGALLOC, long, G__int, result.obj.i);
         case 'k':
            // unsigned long int
            G__ASSIGN_VAR(G__LONGALLOC, unsigned long, G__int, result.obj.i);
         case 'f':
            // float
            G__ASSIGN_VAR(G__FLOATALLOC, float, G__double, result.obj.d);
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
         case 'C':
            // char pointer
            G__ASSIGN_PVAR(char, G__int, result.obj.i);
            break;
         case 'N':
            G__ASSIGN_PVAR(G__int64, G__Longlong, result.obj.ll);
            break;
         case 'M':
            G__ASSIGN_PVAR(G__uint64, G__ULonglong, result.obj.ull);
            break;
#ifndef G__OLDIMPLEMENTATION2191
         case 'Q':
            G__ASSIGN_PVAR(long double, G__Longdouble, result.obj.ld);
            break;
#endif // G__OLDIMPLEMENTATION2191
         case 'G':
            // bool pointer
#ifdef G__BOOL4BYTE
            G__ASSIGN_PVAR(int, G__int, result.obj.i);
#else // G__BOOL4BYTE
            G__ASSIGN_PVAR(unsigned char, G__int, result.obj.i);
#endif // G__BOOL4BYTE
            break;
         case 'B':
            // unsigned char pointer
            G__ASSIGN_PVAR(unsigned char, G__int, result.obj.i);
            break;
         case 'S':
            // short pointer
            G__ASSIGN_PVAR(short, G__int, result.obj.i);
            break;
         case 'R':
            // unsigned short pointer
            G__ASSIGN_PVAR(unsigned short, G__int, result.obj.i);
            break;
         case 'I':
            // int pointer
            G__ASSIGN_PVAR(int, G__int, result.obj.i);
            break;
         case 'H':
            // unsigned int pointer
            G__ASSIGN_PVAR(unsigned int, G__int, result.obj.i);
            break;
         case 'L':
            // long int pointer
            G__ASSIGN_PVAR(long, G__int, result.obj.i);
            break;
         case 'K':
            // unsigned long int pointer
            G__ASSIGN_PVAR(unsigned long, G__int, result.obj.i);
            break;
         case 'F':
            // float pointer
            G__ASSIGN_PVAR(float, G__double, result.obj.d);
            break;
         case 'D':
            // double pointer
            G__ASSIGN_PVAR(double, G__double, result.obj.d);
            break;
         case 'u':
            // struct,union
            {
               if (ig25 < paran) {
                  result.tagnum = var->p_tagtable[ig15];
                  result.typenum = var->p_typetable[ig15];
                  result.ref = (long) (G__struct_offset + var->p[ig15] + (linear_index * G__struct.size[var->p_tagtable[ig15]]));
                  G__letint(&result, 'u', result.ref);
                  G__tryindexopr(&result, para, paran, ig25);
                  para[0] = G__letVvalue(&result, expression);
                  return para[0];
               }
               else {
                  G__letstruct(&result, linear_index, var, ig15, item, paran, G__struct_offset);
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
                     (var->reftype[ig15] == G__PARANORMAL) ||
                     (var->reftype[ig15] == (paran - ig25))
                  )
               ) {
                  result.tagnum = var->p_tagtable[ig15];
                  result.typenum = var->p_typetable[ig15];
                  result.ref = 0;
                  address = G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC);
                  result.ref = (*(long*)address) + (secondary_linear_index * G__struct.size[var->p_tagtable[ig15]]);
                  G__letint(&result, 'u', result.ref);
                  G__tryindexopr(&result, para, paran, ig25);
                  para[0] = G__letVvalue(&result, expression);
                  return para[0];
               }
               else {
                  G__letstructp(result, G__struct_offset, ig15, linear_index, var, paran, item, para, secondary_linear_index);
               }
            }
            break;
         case 'a':
            // pointer to member function
            G__letpointer2memfunc(var, paran, ig15, item, linear_index, &result, G__struct_offset);
            break;
         case 'T':
            // macro char*
            {
               if (
                  (G__globalcomp == G__NOLINK) &&
                  !G__prerun &&
                  (G__double(result) != G__double(G__getitem(item)))
               ) {
                  G__changeconsterror(varname, "enforced macro");
               }
               *((long*) var->p[ig15]) = result.obj.i;
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
                  G__changeconsterror(varname, "enforced macro");
               }
            }
            G__letautomatic(var, ig15, G__struct_offset, linear_index, result);
            break;
         default:
            // case 'X' automatic variable
            G__letautomatic(var, ig15, G__struct_offset, linear_index, result);
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
      result = G__allocvariable(result, para, varglobal, varlocal, paran, varhash, item, varname, parameter[0][0], member);
   }
   G__var_type = 'p';
   return result;
}

#undef G__ASSIGN_VAR
#undef G__ASSIGN_PVAR

extern "C" {

//______________________________________________________________________________
void G__letpointer2memfunc(G__var_array* var, int paran, int ig15, const char* item, int linear_index, G__value* presult, long G__struct_offset)
{
   // -- FIXME: Describe me!
   switch (G__var_type) {
      case 'p':
         // var = expr; assign to value
         if (var->paran[ig15] <= paran) {
            // -- Assign to type element
#ifdef G__PTR2MEMFUNC
            if (presult->type == 'C') {
               *(long*)(G__struct_offset + var->p[ig15] + (linear_index * G__P2MFALLOC)) = presult->obj.i;
            }
            else {
               memcpy((void*)(G__struct_offset + var->p[ig15] + (linear_index * G__P2MFALLOC)), (void*)presult->obj.i, G__P2MFALLOC);
            }
#else // G__PTR2MEMFUNC
            memcpy((void*)(G__struct_offset + var->p[ig15] + (linear_index * G__P2MFALLOC)), (void*) presult->obj.i, G__P2MFALLOC);
#endif // G__PTR2MEMFUNC
            break;
         }
      default:
         G__assign_error(item, presult);
         break;
   }
}

//______________________________________________________________________________
void G__letautomatic(G__var_array* var, int ig15, long G__struct_offset, int linear_index, G__value result)
{
   // -- FIXME: Describe me!
   if (isupper(var->type[ig15])) {
      *(double*)(G__struct_offset + var->p[ig15] + (linear_index * G__DOUBLEALLOC)) = G__double(result);
   }
   else {
      *(long*)(G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC)) = G__int(result);
   }
}

} // extern "C"

//______________________________________________________________________________
G__value G__letstructmem(int store_var_type, G__FastAllocString& varname,
                         int membernameoffset, G__FastAllocString& result7,
                         char* tagname, G__var_array* varglobal,
                         G__value expression, int objptr  /* 1: object, 2: pointer */,
                         Cint::G__DataMemberHandle &member)
{
   // -- FIXME: Describe me!
   G__value result;
#ifndef G__OLDIMPLEMENTATION1259
   G__SIGNEDCHAR_T store_isconst;
#endif // G__OLDIMPLEMENTATION1259
   int store_do_setmemfuncenv;
   /* add pointer operater if necessary */

   if (store_var_type == 'P') {
      varname = "&"; varname += (result7 + membernameoffset); // Legacy, only add one character
      result7.Replace(membernameoffset, varname); // Legacy, only increase use by on charater

   }
   if (store_var_type == 'v') {
      varname = "*"; varname += (result7 + membernameoffset); // Legacy, only add one character
      result7.Replace(membernameoffset, varname); // Legacy, only add one character

   }
   int store_tagnum = G__tagnum;
   long store_struct_offset = G__store_struct_offset;
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
      result = G__getvariable(tagname, &flag, &G__global, G__p_local);
   }
   else {
      // get it '&tag' which is G__struct.memvar[].
      //        OR
      // member is referenced in member function
      //  'subtag.mem'
      G__incsetup_memvar(G__tagnum);
      result = G__getvariable(tagname, &flag, 0, G__struct.memvar[G__tagnum]);
   }
   G__store_struct_offset = result.obj.i;
   G__tagnum = result.tagnum;
#ifndef G__OLDIMPLEMENTATION1259
   G__isconst = result.isconst;
#endif // G__OLDIMPLEMENTATION1259
   if (G__tagnum < 0) {
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
         G__fprinterr(G__serr, "Error: illegal pointer to class object %s 0x%lx %d ", tagname, G__store_struct_offset, G__tagnum);
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
      (result.type == 'u') &&
      (objptr == 2) &&
      (result.tagnum != -1) &&
      !strncmp(G__struct.name[result.tagnum], "auto_ptr<", 9)
    ) {
      int knownx = 0;
      G__FastAllocString comm("operator->()");
      result = G__getfunction(comm, &knownx, G__TRYMEMFUNC);
      if (knownx) {
         G__tagnum = result.tagnum;
         G__store_struct_offset = result.obj.i;
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
   if (islower(result.type) && (objptr == 2)) {
      char buf[30] = "operator->()";
      int flag = 0;
      int store_tagnum = G__tagnum;
      long store_struct_offset = G__store_struct_offset;
      G__tagnum = result.tagnum;
      G__store_struct_offset = result.obj.i;
      result = G__getfunction(buf, &flag, G__TRYMEMFUNC);
      if (flag) {
         G__tagnum = result.tagnum;
         G__store_struct_offset = result.obj.i;
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
         G__tagnum = store_tagnum;
         G__store_struct_offset = store_struct_offset;
         if (
            (G__dispmsg >= G__DISPROOTSTRICT) ||
            (G__ifile.filenum <= G__gettempfilenum())
         ) {
            if (G__dispmsg >= G__DISPWARN) {
               G__fprinterr(G__serr, "Warning: wrong member access operator '->'");
               G__printlinenum();
            }
         }
      }
   }
   if (isupper(result.type) && (objptr == 1)) {
      if (
         (G__dispmsg >= G__DISPROOTSTRICT) ||
         (G__ifile.filenum <= G__gettempfilenum())
      ) {
         if (G__dispmsg >= G__DISPWARN) {
            G__fprinterr(G__serr, "Warning: wrong member access operator '.'");
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
   {   
      G__FastAllocString temp_membername(result7 + membernameoffset);
      result = G__letvariable(temp_membername, expression, 0, G__struct.memvar[G__tagnum], member);
   }
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

extern "C" {

//______________________________________________________________________________
void G__letstruct(G__value* result, int linear_index, G__var_array* var, int ig15, const char* item, int paran, long G__struct_offset)
{
   // Perform assignment from passed struct to passed result.
   // Note:
   // G__letstruct and G__classassign in struct.c have special handling
   // of operator=(). When interpretation, overloaded assignment operator
   // is recognized in G__letstruct and G__classassign functions. They
   // set appropreate environment (G__store_struct_offset, G__tagnum)
   // and try to call operator=(). It may not be required to search for
   // non member operator=() function, so, some part of these functions
   // could be omitted.
   G__FastAllocString tmp(G__ONELINE);
   G__FastAllocString result7(G__ONELINE);
   int ig2 = 0;
   long store_struct_offset = 0;
   int largestep = 0;
   int store_tagnum = -1;
   G__value para = *result;
   int store_prerun = 0;
   int store_debug = 0;
   int store_step = 0;
   long store_asm_inst = 0L;
   long addr = 0L;
   if (G__asm_exec) {
      void* p1 = (void*)
         (G__struct_offset + var->p[ig15] + linear_index * G__struct.size[var->p_tagtable[ig15]]);
      void* p2 = (void*) result->obj.i;
      size_t size = (size_t) G__struct.size[var->p_tagtable[ig15]];
      memcpy(p1, p2, size);
      return;
   }
   switch (G__var_type) {
      case 'p': // return by pointer, normal case, used for intermediate results as well
         if (var->paran[ig15] <= paran) {
            // Argument count is in range.
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
               (result->tagnum != -1) && // has tagnum, and
               (
                  (result->type == 'u') || // is class, struct, union, or
                  (result->type == 'i') // enum
               )
            ) {
               tmp.Format(
                    "(%s)0x%lx"
                  , G__fulltagname(result->tagnum, 1)
                  , result->obj.i
               );
            }
            else {
               // Result is of fundamental type.
               G__valuemonitor(*result, tmp);
            }
            if (G__decl) {
               // Assignment in a declaration, use a constructor.
               result7.Format("%s(%s)", G__struct.name[var->p_tagtable[ig15]], tmp());
               store_tagnum = G__tagnum;
               G__tagnum = var->p_tagtable[ig15];
               store_struct_offset = G__store_struct_offset;
               G__store_struct_offset = (G__struct_offset + var->p[ig15] + (linear_index * G__struct.size[var->p_tagtable[ig15]]));
               if (G__dispsource) {
                  G__fprinterr(
                       G__serr
                     , "\n!!!Calling constructor (%s) 0x%lx for "
                       "declaration  %s:%d\n"
                     , result7()
                     , G__store_struct_offset
                     , __FILE__
                     , __LINE__
                  );
               }
#ifdef G__SECURITY
               G__castcheckoff = 1;
#endif // G__SECURITY
               ig2 = 0;
               G__decl = 0;
#ifndef G__OLDIMPLEMENTATION1073
               G__oprovld = 1; // Tell G__getfunction() to not stack the args.
#endif // G__OLDIMPLEMENTATION1073
               {
                  int store_cp = G__asm_cp;
                  int store_dt = G__asm_dt;
                  G__getfunction(result7, &ig2 , G__TRYCONSTRUCTOR);
                  if (ig2 && G__asm_noverflow) {
                     G__asm_dt = store_dt;
                     int x;
                     if (G__LD_FUNC == G__asm_inst[G__asm_cp-8]) {
                        for (x = 0; x < 8; ++x) {
                           G__asm_inst[store_cp+x] = G__asm_inst[G__asm_cp-8+x];
                        }
                        G__asm_cp = store_cp + 8;
                     }
                     else if (G__LD_IFUNC == G__asm_inst[G__asm_cp-8]) {
                        for (x = 0; x < 8; ++x) {
                           G__asm_inst[store_cp+x] = G__asm_inst[G__asm_cp-8+x];
                        }
                        G__asm_cp = store_cp + 8;
                     }
                  }
                  else if (!ig2 && (result->type == 'U')) {
                     G__fprinterr(G__serr, "\nError: Constructor %s not found!", result7());
                     G__genericerror(0);
                  }
               }
#ifndef G__OLDIMPLEMENTATION1073
               G__oprovld = 0; // And allow G__getfunction() to stack args again.
               if (G__asm_wholefunction && !ig2) {
                  G__asm_gen_stvar(G__struct_offset, ig15, paran, var, item,
                     G__ASM_VARLOCAL, G__var_type, result);
               }
#endif // G__OLDIMPLEMENTATION1073
               G__decl = 1;
               G__store_struct_offset = store_struct_offset;
               G__tagnum = store_tagnum;
            }
            else {
               // Use operator= to do the assignment..
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // We are generating code.
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(
                          G__serr
                        , "ST_VAR or ST_MSTR replaced with "
                          "LD_VAR or LD_MSTR(2)  %s:%d\n"
                        , __FILE__
                        , __LINE__
                     );
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
               result7.Format("operator=(%s)" , tmp());
               store_tagnum = G__tagnum;
               G__tagnum = var->p_tagtable[ig15];
               store_struct_offset = G__store_struct_offset;
               G__store_struct_offset = (G__struct_offset + var->p[ig15] + linear_index * G__struct.size[var->p_tagtable[ig15]]);
               ig2 = 0;
               para = G__getfunction(result7, &ig2 , G__TRYMEMFUNC);
               if (!ig2 && (G__tagnum != result->tagnum)) {
                  // -- Copy constructor.
                  result7.Format("%s(%s)", G__struct.name[G__tagnum], tmp());
                  if (G__struct.iscpplink[G__tagnum] == G__CPPLINK) {
                     G__abortbytecode();
                     long store_globalvarpointer = G__globalvarpointer;
                     G__globalvarpointer = G__store_struct_offset;
                     G__getfunction(result7, &ig2 , G__TRYCONSTRUCTOR);
                     G__globalvarpointer = store_globalvarpointer;
                  }
                  else {
                     G__getfunction(result7, &ig2 , G__TRYCONSTRUCTOR);
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
                  addr = G__struct_offset + var->p[ig15] + (linear_index * G__struct.size[var->p_tagtable[ig15]]);
                  if (addr < 0) {
                     result7.Format("operator=((%s)(%ld),%s)", G__fulltagname(var->p_tagtable[ig15], 1), addr, tmp());
                  }
                  else {
                     result7.Format("operator=((%s)%ld,%s)", G__fulltagname(var->p_tagtable[ig15], 1), addr, tmp());
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
               // Success.
               *result = para;
            }
            else {
               // There was no operator= or constructor available.
               // Try conversion operator for class object.
               if ((result->type == 'u') && (result->tagnum != -1)) {
                  int tagnum = var->p_tagtable[ig15];
                  int done = G__class_conversion_operator(tagnum, result, tmp);
                  if (done) {
                     long pdest = G__struct_offset + var->p[ig15] + (linear_index * G__struct.size[tagnum]);
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
                  // -- With -c-1 or -c-2 option
                  return;
               }
#endif // G__ASM
               if (result->tagnum == var->p_tagtable[ig15]) {
                  std::memcpy((void*) (G__struct_offset + var->p[ig15] + (linear_index * G__struct.size[var->p_tagtable[ig15]])), (void*) G__int(*result), (size_t) G__struct.size[var->p_tagtable[ig15]]);
               }
               else if (-1 != (addr = G__ispublicbase(var->p_tagtable[ig15], result->tagnum, 0))) {
                  int tagnum = var->p_tagtable[ig15];
                  long pdest = G__struct_offset + var->p[ig15] + (linear_index * G__struct.size[tagnum]);
                  std::memcpy((void*) pdest, (void*) (G__int(*result) + addr), (size_t) G__struct.size[tagnum]);
                  if (G__struct.virtual_offset[tagnum] != -1) {
                     *((long*)(pdest + G__struct.virtual_offset[tagnum])) = tagnum;
                  }
               }
               else {
                  G__fprinterr(G__serr, "Error: Assignment to %s type incompatible " , item);
                  G__genericerror(0);
               }
            }
            break;
         }
         else if (G__funcheader && !paran && isupper(result->type)) {
            // FIXME: Remove special case for unspecified length array.
            if (var->p[ig15] && (var->statictype[ig15] != G__COMPILEDGLOBAL) && (var->statictype[ig15] != G__USING_VARIABLE) && (var->statictype[ig15] != G__USING_STATIC_VARIABLE)) {
               free((void*)var->p[ig15]);
            }
            var->p[ig15] = result->obj.i;
            var->statictype[ig15] = G__COMPILEDGLOBAL;
            break;
         }
      default:
         if (G__var_type == 'u') { // return by struct offset
            G__letint(result, 'u', G__struct_offset + var->p[ig15]);
            result->tagnum = var->p_tagtable[ig15];
            result->typenum = var->p_typetable[ig15];
            break;
         }
         if (G__var_type == 'v') { // return by value, we will have to call operator*() to get it.
            G__FastAllocString refopr(G__MAXNAME);
            long store_struct_offsetX = G__store_struct_offset;
            int store_tagnumX = G__tagnum;
            int done = 0;
            int store_var_type = G__var_type;
            G__var_type = 'p';
            G__store_struct_offset = (long) (G__struct_offset + var->p[ig15] +
               (linear_index * G__struct.size[var->p_tagtable[ig15]]));
            G__tagnum = var->p_tagtable[ig15];
#ifdef G__ASM
            if (G__asm_noverflow) {
               // We are generating bytecode.
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(
                       G__serr
                     , "%3x,%3x: LD_VAR  name: %s index: %d paran: %d  %s:%d\n"
                     , G__asm_cp
                     , G__asm_dt
                     , var->varnamebuf[ig15]
                     , ig15
                     , 0
                     , __FILE__
                     , __LINE__
                  );
               }
#endif // G__ASM_DBG
               if (G__struct_offset) {
                  G__asm_inst[G__asm_cp] = G__LD_MSTR;
               }
               else {
                  G__asm_inst[G__asm_cp] = G__LD_VAR;
               }
               G__asm_inst[G__asm_cp+1] = ig15;
               G__asm_inst[G__asm_cp+2] = paran;
               G__asm_inst[G__asm_cp+3] = 'p';
               G__asm_inst[G__asm_cp+4] = (long) var;
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
            refopr = "operator*()";
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
         else {
            G__assign_error(item, result);
         }
         break;
   }
}

//______________________________________________________________________________
void G__letstructp(G__value result, long G__struct_offset, int ig15, int linear_index, G__var_array* var, int paran, const char* item, G__value* para, int secondary_linear_index)
{
   // -- FIXME: Describe me!
   int baseoffset = 0;
   switch (G__var_type) {
      case 'v':
         // *var = result;  Assign using a pointer variable derefence.
         switch (var->reftype[ig15]) {
            case G__PARANORMAL:
               if (G__no_exec_compile) {
                  G__classassign(G__PVOID, var->p_tagtable[ig15], result);
               }
               else {
                  G__classassign(*((long*) (G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC))), var->p_tagtable[ig15], result);
               }
               break;
            case G__PARAP2P:
               if (paran > var->paran[ig15]) {
                  // -- Pointer to array reimplementation.
                  if (G__no_exec_compile) {
                     G__classassign(G__PVOID, var->p_tagtable[ig15], result);
                  }
                  else {
                     long address = G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC);
                     G__classassign(*(((long*) (*((long*) address))) + secondary_linear_index), var->p_tagtable[ig15], result);
                  }
               }
               else {
                  if (!G__no_exec_compile) {
                     *((long*) (*((long*) (G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC))))) = G__int(result);
                  }
               }
               break;
         }
         break;
      case 'p':
         // var = result;  Assign to a pointer variable.
         if (paran >= var->paran[ig15]) {
            if (var->paran[ig15] < paran) {
               long address = G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC);
               if (var->reftype[ig15] == G__PARANORMAL) {
                  if (G__no_exec_compile) {
                     address = G__PVOID;
                  }
                  else {
                     address = (*(long*)address) + (secondary_linear_index * G__struct.size[var->p_tagtable[ig15]]);
                  }
                  G__classassign(address, var->p_tagtable[ig15], result);
               }
               else if (var->paran[ig15] == (paran - 1)) {
                  if (!G__no_exec_compile) {
                     *(((long*)(*(long *)address)) + secondary_linear_index) = G__int(result);
                  }
               }
               else if (var->paran[ig15] == (paran - 2)) {
                  if (!G__no_exec_compile) {
                     address = (long)((long*)(*(long*)address) + para[0].obj.i);
                     if (var->reftype[ig15] == G__PARAP2P) {
                        address = (long)((*((long*)address)) + (para[1].obj.i * G__struct.size[var->p_tagtable[ig15]]));
                        G__classassign(address, var->p_tagtable[ig15], result);
                     }
                     else if (var->reftype[ig15] > G__PARAP2P) {
                        address = (long)((long*)(*(long*)address) + para[1].obj.i);
                        *(long*) address = G__int(result);
                     }
                  }
               }
               else if (var->paran[ig15] == (paran - 3)) {
                  if (!G__no_exec_compile) {
                     address = (long)((long*)(*(long*)address) + para[0].obj.i);
                     address = (long)((long*)(*(long*)address) + para[1].obj.i);
                     if (var->reftype[ig15] == G__PARAP2P2P) {
                        address = (long)((*((long*)(address))) + (para[2].obj.i * G__struct.size[var->p_tagtable[ig15]]));
                        G__classassign(address, var->p_tagtable[ig15], result);
                     }
                     else if (var->reftype[ig15] > G__PARAP2P2P) {
                        address = (long)((long*)(*(long*)address) + para[2].obj.i);
                        *(long*) address = G__int(result);
                     }
                  }
               }
               else {
                  if (!G__no_exec_compile)
                     G__classassign(((*(((long*)(*(long*)address)) + para[0].obj.i)) + para[1].obj.i), var->p_tagtable[ig15], result);
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
                  (result.type != 'U') &&
                  (result.type != 'Y') &&
                  result.obj.i &&
                  ((result.type != 'u') || (result.obj.i == G__p_tempbuf->obj.ref))
               ) {
                  G__assign_error(item, &result);
                  return;
               }
               if (
                  (var->p_tagtable[ig15] == result.tagnum) ||
                  !result.obj.i ||
                  (result.type == 'Y')
               ) {
                  // Checked.
                  *((long*) (G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC))) = G__int(result);
               }
               else if (
                  // --
#ifdef G__VIRTUALBASE
                  ((baseoffset = G__ispublicbase(var->p_tagtable[ig15], result.tagnum, result.obj.i)) != -1)
#else // G__VIRTUALBASE
                  ((baseoffset = G__ispublicbase(var->p_tagtable[ig15], result.tagnum)) != -1)
#endif // G__VIRTUALBASE
                  // --
               ) {
                  *((long*) (G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC))) = G__int(result) + baseoffset;
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
G__value G__classassign(long pdest, int tagnum, G__value result)
{
   // -- FIXME: Describe me!
   G__FastAllocString ttt(G__ONELINE);
   G__FastAllocString result7(G__ONELINE);
   long store_struct_offset = 0;
   int store_tagnum = -1;
   int ig2 = 0;
   G__value para;
   long store_asm_inst = 0;
   int letvvalflag = 0;
   long addstros_value = 0;
   if (G__asm_exec) {
      memcpy((void*) pdest, (void*) G__int(result), (size_t) G__struct.size[tagnum]);
      return result;
   }
   if (result.type == 'u' && result.tagnum != -1) {
      // --
      if (result.obj.i < 0) {
         ttt.Format("(%s)(%ld)", G__struct.name[result.tagnum], result.obj.i);
      }
      else {
         ttt.Format("(%s)%ld", G__struct.name[result.tagnum], result.obj.i);
      }
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
   result7.Format("operator=(%s)", ttt());
   store_tagnum = G__tagnum;
   G__tagnum = tagnum;
   store_struct_offset = G__store_struct_offset;
   G__store_struct_offset = pdest;
   ig2 = 0;
   para = G__getfunction(result7, &ig2 , G__TRYMEMFUNC);
   if (!ig2 && (tagnum != result.tagnum)) {
      //
      // copy constructor
      //
      long store_globalvarpointer = 0L;
      result7.Format("%s(%s)", G__struct.name[tagnum], ttt());
      if (G__struct.iscpplink[tagnum] == G__CPPLINK) {
         G__abortbytecode();
         store_globalvarpointer = G__globalvarpointer;
         G__globalvarpointer = G__store_struct_offset;
         G__getfunction(result7, &ig2 , G__TRYCONSTRUCTOR);
         G__globalvarpointer = store_globalvarpointer;
      }
      else {
         G__getfunction(result7, &ig2 , G__TRYCONSTRUCTOR);
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
         result7.Format("operator=((%s)(%ld),%s)", G__fulltagname(tagnum, 1), pdest, ttt());
      }
      else {
         result7.Format("operator=((%s)%ld,%s)", G__fulltagname(tagnum, 1), pdest, ttt());
      }
      para = G__getfunction(result7, &ig2 , G__TRYNORMAL);
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
   if ((result.type == 'u') && (result.tagnum != -1)) {
      int done = G__class_conversion_operator(tagnum, &result, ttt);
      if (done) {
         // --
         return G__classassign(pdest, tagnum, result);
      }
   }
   // Return from this function if this is pure bytecode compilation.
   if (G__no_exec_compile) {
      // --
      return result;
   }
#endif // G__ASM
   if (result.tagnum == tagnum) {
      memcpy((void*) pdest, (void*) G__int(result), (size_t) G__struct.size[tagnum]);
   }
   else if ((addstros_value = G__ispublicbase(tagnum, result.tagnum, 0)) != -1) {
      memcpy((void*) pdest, (void*) (G__int(result) + addstros_value), (size_t) G__struct.size[tagnum]);
      if (G__struct.virtual_offset[tagnum] != -1) {
         *(long*)(pdest + G__struct.virtual_offset[tagnum]) = tagnum;
      }
   }
   else {
      G__fprinterr(G__serr, "Error: Assignment type incompatible FILE:%s LINE:%d\n", G__ifile.name, G__ifile.line_number);
   }
   return result;
}

//______________________________________________________________________________
int G__class_conversion_operator(int tagnum, G__value* presult, char* /*ttt*/)
{
   // -- Conversion operator for assignment to class object.
   // Note: Bytecode compilation turned off if conversion operator is found.
   G__value conv_result;
   int conv_done = 0;
   int conv_tagnum = G__tagnum;
   int conv_typenum = G__typenum;
   int conv_constvar = G__constvar;
   int conv_reftype = G__reftype;
   int conv_var_type = G__var_type;
   long conv_store_struct_offset = G__store_struct_offset;
   switch (G__struct.type[presult->tagnum]) {
      case 'c':
      case 's':
         G__tagnum = presult->tagnum;
         G__typenum = -1;
         G__constvar = 0;
         G__reftype = 0;
         G__var_type = 'p';
         G__store_struct_offset = presult->obj.i;
         // Synthesize function name.
         G__FastAllocString tmp(G__ONELINE);
         tmp = "operator ";
         tmp += G__struct.name[tagnum];
         tmp += "()";
         // Call conversion operator.
         conv_result = G__getfunction(tmp, &conv_done , G__TRYMEMFUNC);
         if (conv_done) {
            if (G__dispsource) {
               G__fprinterr(G__serr, "!!!Conversion operator called 0x%lx.%s\n", G__store_struct_offset, tmp());
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
int G__fundamental_conversion_operator(int type, int tagnum, int typenum, int reftype, int constvar, G__value* presult, char* /*ttt*/)
{
   // -- Conversion operator for assignment to fundamental type object.
   //
   // Note: Bytecode compilation is alive after conversion operator is used.
   //
   G__FastAllocString tmp(G__ONELINE);
   G__value conv_result;
   int conv_done = 0;
   int conv_tagnum = G__tagnum;
   int conv_typenum = G__typenum;
   int conv_constvar = G__constvar;
   int conv_reftype = G__reftype;
   int conv_var_type = G__var_type;
   long conv_store_struct_offset = G__store_struct_offset;
   switch (G__struct.type[presult->tagnum]) {
      case 'c':
      case 's':
         G__tagnum = presult->tagnum;
         G__typenum = -1;
         G__constvar = 0;
         G__reftype = 0;
         G__var_type = 'p';
         G__store_struct_offset = presult->obj.i;
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
         tmp = "operator ";
         tmp += G__type2string(type, tagnum, typenum, reftype, constvar);
         tmp += "()";
         // Call conversion operator.
         conv_result = G__getfunction(tmp, &conv_done , G__TRYMEMFUNC);
         if (!conv_done && (typenum != -1)) {
            // Make another try after removing typedef alias.
            tmp[9] = 0;
            tmp += G__type2string(type, -1, -1 , reftype, constvar);
            tmp += "()";
            conv_result = G__getfunction(tmp, &conv_done , G__TRYMEMFUNC);
         }
         if (!conv_done) {
            // Make another try constness reverting.
            constvar ^= 1;
            tmp[9] = 0;
            tmp += G__type2string(type, tagnum, typenum, reftype, constvar);
            tmp += "()";
            conv_result = G__getfunction(tmp, &conv_done , G__TRYMEMFUNC);
            if (!conv_done && (typenum != -1)) {
               // Make another try after removing typedef alias.
               tmp[9] = 0;
               tmp += G__type2string(type, -1, -1 , reftype, constvar);
               tmp += "()";
               conv_result = G__getfunction(tmp, &conv_done , G__TRYMEMFUNC);
            }
         }
         if (!conv_done) {
            for (int itype = 0; itype < G__newtype.alltype; ++itype) {



               if ((type == G__newtype.type[itype]) && (tagnum == G__newtype.tagnum[itype])) {
                  constvar ^= 1;
                  tmp[9] = 0;
                  tmp += G__type2string(type, tagnum, itype, reftype, constvar);
                  tmp += "()";
                  conv_result = G__getfunction(tmp, &conv_done , G__TRYMEMFUNC);
                  if (!conv_done) {
                     constvar ^= 1;
                     tmp[9] = 0;
                     tmp += G__type2string(type, tagnum, typenum, reftype, constvar);
                     tmp += "()";
                     conv_result = G__getfunction(tmp, &conv_done , G__TRYMEMFUNC);
                  }
                  if (conv_done) {
                     break;
                  }
               }
            }
         }
         if (conv_done) {
            if (G__dispsource) {
               G__fprinterr(G__serr, "!!!Conversion operator called 0x%lx.%s\n", G__store_struct_offset, tmp());
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

extern "C++" {
template<class CASTTYPE, class CONVFUNC>
inline void G__alloc_var_ref(int SIZE, CONVFUNC f, const char* item, G__var_array* var, int ig15, G__value& result)
{
   if (islower(G__var_type)) {
      /* -- Not a pointer, may be an array. */
      /* Allocate memory */
      if (var->varlabel[ig15][1] /* number of elements */ == INT_MAX /* unspecified length flag */) {
         /* -- Unspecified length array. */
         if (G__funcheader) {
            /* -- In a function header. */
            /* Allocate no storage, we will share it with the caller (see the initializer below). */
            /*var->p[ig15] = 0L;*/
         }
         else {
            /* -- Not in a function header. */
            if (!G__static_alloc || G__prerun) {
               /* -- Not static, const, enum; or it is prerun time. */
               /* Allocate no storage (it will be allocated later during initializer parsing). */
               /*var->p[ig15] = 0L;*/
            }
            else {
               /* -- Static, const, or enum during execution. */
               /* Copy the pointer from the global variable, no actual allocation takes place. */
               var->p[ig15] = (long) G__malloc(1, SIZE, item);
            }
         }
      }
      else if (var->varlabel[ig15][1] /* number of elements */) {
         /* -- An array. */
         if (G__funcheader) {
            /* -- In a function header. */
            /* Allocate no storage, we will share storage with caller (see the initializer below). */
            /*var->p[ig15] = 0L;*/
         } else {
            /* -- Not in a function header, allocate memory. */
            /* Note: If this is a static, no memory is allocated, a copy of the global var pointer is returned. */
            var->p[ig15] = (long) G__malloc(var->varlabel[ig15][1] /* number of elements */, SIZE, item);
            if (
               var->p[ig15] && /* We have a pointer, and */
               !G__def_struct_member && /* We are not defining a member variable (the pointer is an offset), and */
               (G__asm_wholefunction == G__ASM_FUNC_NOP) && /* We are not bytecode compiling a function (the pointer is an offset), and */
               !(G__static_alloc && (G__func_now > -1) && !G__prerun) /* Not a static variable in function scope at runtime. */
            ) {
               /*memset((char*) var->p[ig15], 0, var->varlabel[ig15][1] * SIZE);*/
            }
         }
      }
      else {
         /* -- Normal variable, allocate space. */
         /* Note: If this is a static, no memory is allocated, a copy of the global var pointer is returned. */
         var->p[ig15] = (long) G__malloc(1, SIZE, item);
         if (
            var->p[ig15] && /* We have a pointer, and */
            !G__def_struct_member && /* We are not defining a member variable (the pointer is an offset), and */
            (G__asm_wholefunction == G__ASM_FUNC_NOP) && /* We are not bytecode compiling a function (the pointer is an offset), and */
            !(G__static_alloc && (G__func_now > -1) && !G__prerun) /* Not a static variable in function scope at runtime. */
         ) {
            /*memset((char*) var->p[ig15], 0, SIZE);*/
         }
      }
      if (
         (G__asm_wholefunction == G__ASM_FUNC_COMPILE) &&
         (var->type[ig15] == 'i') &&
         (var->constvar[ig15] & G__CONSTVAR) &&
         result.type
      ) {
         G__abortbytecode();
      }
      /* Now do initialization. */
      if (
         /* Variable has storage to initialize */
         (var->p[ig15] != 0 || (var->varlabel[ig15][1] && G__funcheader)) &&
         /* Not bytecode compiling */
         (G__asm_wholefunction == G__ASM_FUNC_NOP) &&
         (
            /* Not a member variable. */
            !G__def_struct_member ||
            /* Static, const, or enumerator member variable. */
            G__static_alloc ||
            /* Namespace member variable. */
            (var->statictype[ig15] == G__LOCALSTATIC) ||
            /* Namespace member variable. */
            (G__struct.type[G__def_tagnum] == 'n')
         ) &&
         /* Initialize const, static, and enumerator variables before running. */
         (!G__static_alloc || G__prerun) &&
         (
            /* Variable is not of class type or it is not pre-allocated. */
            (G__globalvarpointer == G__PVOID) ||
            /* Initializer is not void. */
            result.type
         )
      ) {
         if (var->varlabel[ig15][1] /* number of elements */ == INT_MAX /* unspecified length flag */) {
            /* -- We are initializing an unspecified length array. */
            if (G__funcheader) {
               /* -- In a function header, we point at our actual argument. */
               var->p[ig15] = (long) G__int(result);
            } else {
               /* -- Syntax errror. */
            }
         }
         else if (var->varlabel[ig15][1] /* number of elements */) {
            /* -- We are initializing an array. */
            if (G__funcheader) {
               /* -- In a function header, we point at our actual argument. */
               var->p[ig15] = (long) G__int(result);
            } else {
               /* -- Syntax error. */
            }
         }
         else {
            /* -- We are initializing a non-array variable. */
            *((CASTTYPE*) var->p[ig15]) = (CASTTYPE) (*f)(result);
         }
      }
   }
   else {
      /* -- Pointer or array of pointers. */
      /* Allocate memory */
      if (var->varlabel[ig15][1] /* number of elements */ == INT_MAX /* unspecified length flag */) {
         /* -- Unspecified length array of pointers. */
         if (G__funcheader) {
            /* -- In a function header.*/
            /* Allocate no storage, we will share it with the caller (see initializer below). */
            /*var->p[ig15] = 0L;*/
         }
         else {
            /* -- Not in a function header. */
            if (!G__static_alloc || G__prerun) {
               /* -- Not static, const, enum; or it is prerun time. */
               /* Allocate no storage (it will be allocated later during initializer parsing). */
               /*var->p[ig15] = 0L;*/
            }
            else {
               /* -- Static, const, or enum during execution. */
               /* Copy the pointer from the global variable, no actual allocation takes place. */
               var->p[ig15] = (long) G__malloc(1, G__LONGALLOC, item);
            }
         }
      }
      else if (var->varlabel[ig15][1] /* number of elements */) {
         /* -- Array of pointers. */
         if (G__funcheader) {
            /* -- In a function header.*/
            /* Allocate no storage, we will share it with the caller (see initializer below). */
            /*var->p[ig15] = 0L;*/
         }
         else {
            /* -- Not a function header. */
            /* Allocate storage for an array of pointers. */
            /* Note: If this is a static, no memory is allocated, a copy of the global var pointer is returned. */
            var->p[ig15] = (long) G__malloc(var->varlabel[ig15][1] /* number of elements */, G__LONGALLOC, item);
            if (
               var->p[ig15] && /* We have a pointer, and */
               !G__def_struct_member && /* We are not defining a member variable (the pointer is an offset), and */
               (G__asm_wholefunction == G__ASM_FUNC_NOP) && /* We are not bytecode compiling a function (the pointer is an offset), and */
               !(G__static_alloc && (G__func_now > -1) && !G__prerun) /* Not a static variable in function scope at runtime. */
            ) {
               /*memset((char*) var->p[ig15], 0, var->varlabel[ig15][1] * G__LONGALLOC);*/
            }
         }
      }
      else {
         /* -- Normal pointer. */
         /* Allocate storage for a pointer. */
         /* Note: If this is a static, no memory is allocated, a copy of the global var pointer is returned. */
         var->p[ig15] = (long) G__malloc(1, G__LONGALLOC, item);
         if (
             (G__globalvarpointer == G__PVOID) && /* variable was *not* preallocated, nor a func ref param */
            var->p[ig15] && /* We have a pointer, and */
            !G__def_struct_member && /* We are not defining a member variable (the pointer is an offset), and */
            (G__asm_wholefunction == G__ASM_FUNC_NOP) && /* We are not bytecode compiling a function (the pointer is an offset), and */
            !(G__static_alloc && (G__func_now > -1) && !G__prerun) /* Not a static variable in function scope at runtime(do not overwrite a static). */
         ) {
            /**((long*) var->p[ig15]) = 0L;*/
         }
      }
      /* Now do initialization. */
      if (
         /* Variable has storage to initialize */
         (var->p[ig15] != 0 || (var->varlabel[ig15][1] && G__funcheader)) &&
         (
            /* Not a class member and not bytecompiling. */
            (!G__def_struct_member && (G__asm_wholefunction == G__ASM_FUNC_NOP)) ||
            /* Static or const class member variable, not running. */
            (G__static_alloc && G__prerun) ||
            /* Namespace member. */
            (var->statictype[ig15] == G__LOCALSTATIC) ||
            /* Namespace member variable. */
            (G__struct.type[G__def_tagnum] == 'n')
         ) &&
         /* Initialize const and static variables before running. */
         (!G__static_alloc || G__prerun) &&
         (
            /* Variable is not of class type or it is not pre-allocated. */
            (G__globalvarpointer == G__PVOID) ||
            /* Initializer is not void. */
            result.type
         )
      ) {
         if (var->varlabel[ig15][1] /* number of elements */ == INT_MAX /* unspecified length flag */) {
            /* -- We are initializing an unspecified length array of pointers. */
            if (G__funcheader) {
               /* -- In a function header, we point at our actual argument. */
               var->p[ig15] = (long) G__int(result);
            } else {
               /* -- Syntax errror. */
            }
         }
         else if (var->varlabel[ig15][1] /* number of elements */) {
            /* -- We are initializing an array of pointers. */
            if (G__funcheader) {
               /* -- In a function header, we point at our actual argument. */
               var->p[ig15] = (long) G__int(result);
            } else {
               /* -- Syntax error. */
            }
         }
         else {
            /* -- We are initializing a normal pointer. */
            *((long*) var->p[ig15]) = (long) G__int(result);
         }
      }
   }
}
} // extern "C++"

#endif // G__ASM_WHOLEFUNC

} // extern "C"

//______________________________________________________________________________
static G__value G__allocvariable(G__value result, G__value para[], G__var_array* varglobal, G__var_array* varlocal, int paran, int varhash, const char* item, char* varname, int parameter00, G__DataMemberHandle &member)
{
   // -- Allocate memory for a variable and initialize it.
   if (!varname) {
      G__fprinterr(G__serr, "Error: Variable name pointer is null!");
      G__genericerror(0);
      return result;
   }
   //
   //  Complain if the variable name has a minus or plus in it.
   //
   {
      char* pp = std::strchr(varname, '-');
      if (!pp) {
         pp = std::strchr(varname, '+');
      }
      if (pp) {
         G__fprinterr(G__serr,
                      "Error: Variable name has bad character '%s'", varname);
         G__genericerror(0);
         return result;
      }
   }
   //
   //  Complain if the variable name has a space character in it, but
   //  allow the special " : <bitfield-size>" syntax in the variable name
   //  of a bitfield when generating a dictionary (What an awful cludge!).
   //
   if (std::strchr(varname, ' ')) {
      // Ok, there is a space in the variable name, this is bad unless
      // we are generating a dictionary and the name is "name : size"
      // which is the special hack for bitfields in dictionaries.
      if (
         (G__globalcomp == G__NOLINK) || // Not generating a dictionary, or
         !std::strstr(varname, " : ") // no special bitfield marker.
      ) {
         G__fprinterr(
              G__serr
            , "Error: Invalid type '%.*s' in declaration of '%s'"
            , (std::strchr(varname, ' ') - varname)
            , varname
            , std::strchr(varname, ' ') + 1
         );
         G__genericerror(0);
         return result;
      }
   }
   //
   //  Figure out which variable chain we will use.
   //
   struct G__var_array* var = 0;
   if (G__p_local) {
      // -- equal to G__prerun == 0
      var = varlocal;
   }
   else {
      // -- equal to G__prerun == 1
      var = varglobal;
   }
   //
   //  Make sure we have a variable chain
   //  on which to append the new variable.
   //
   if (!var) {
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
         (G__def_tagnum != -1) &&
         (G__struct.type[G__def_tagnum] != 'n') &&
         G__def_tagnum == G__tagdefining
      ) {
         G__genericerror("Limitation: Reference member not supported. Please use pointer");
         return result;
      }
      else if (
         (G__access == G__PUBLIC) &&
         (G__def_tagnum != -1) &&
         (G__struct.type[G__def_tagnum] != 'n')
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
      (result.type == 'U') &&
      (G__ispublicbase(G__tagnum, result.tagnum, G__STATICRESOLUTION2) == -1) &&
      (G__ispublicbase(result.tagnum, G__tagnum, G__STATICRESOLUTION2) == -1)
   ) {
      G__fprinterr(G__serr, "Error: Illegal initialization of pointer, wrong type %s", G__type2string(result.type, result.tagnum, result.typenum, result.obj.reftype.reftype, 0));
      G__genericerror(0);
      return result;
   }
   //
   //  Find the end of the variable chain.
   //
   int index_of_var = 0;
   while (var->next) {
      ++index_of_var;
      var = var->next;
   }
   //
   //  Perform special actions for an automatic variable.
   //
   int autoobjectflag = 0;
   int store_tagnum = 0;
   int store_typenum = 0;
   int store_globalvarpointer = 0;
   int store_var_type = 0;
   if (
      G__automaticvar &&
      (G__var_type == 'p') &&
      !G__definemacro &&
      (G__globalcomp == G__NOLINK)
   ) {
      if (result.tagnum == -1) {
         // -- We are auto-allocating an object of a fundamental type.
         G__var_type = result.type;
         if (!G__const_noerror) {
            // To follow the example of other places, should printlinenum
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
               var = varglobal;
               // Move to the end of the chain.
               while (var->next) {
                  var = var->next;
               }
            }
         }
         store_var_type = G__var_type;
         store_tagnum = G__tagnum;
         store_typenum = G__typenum;
         store_globalvarpointer = G__globalvarpointer;
         G__var_type = result.type;
         G__tagnum = result.tagnum;
         G__typenum = -1;
         if (std::isupper(result.type)) {
            // a = new T(init);  a is a pointer
#ifndef G__ROOT
            G__fprinterr(G__serr, "Warning: Automatic variable %s* %s is allocated", G__fulltagname(result.tagnum, 1), item);
            G__printlinenum();
#endif // G__ROOT
            G__reftype = G__PARANORMAL;
         }
         else {
            if (
               (result.tagnum == G__p_tempbuf->obj.tagnum) &&
               (G__templevel == G__p_tempbuf->level)
            ) {
               // a = T(init); a is an object
               G__globalvarpointer = result.obj.i;
#ifndef G__ROOT
               G__fprinterr(G__serr, "Warning: Automatic variable %s %s is allocated", G__fulltagname(result.tagnum, 1), item);
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
                  G__fprinterr(G__serr, "Error: Illegal Assignment to an undeclared symbol %s", item);
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
      int idx = var->allvar;
      if (!idx || !var->bitfield[idx-1]) {
         // the first element in the bit-field
         bitlocation = 0;
      }
      else {
         bitlocation = var->varlabel[idx-1][G__MAXVARDIM-1] + var->bitfield[idx-1];
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
   if (var->allvar >= G__MEMDEPTH) {
      // -- Overflow, allocate another memory page.
      var->next = (struct G__var_array*) malloc(sizeof(struct G__var_array));
      memset(var->next, 0, sizeof(struct G__var_array));
      var->next->tagnum = var->tagnum;
      var = var->next;
      ++index_of_var;
   }
   int ig15 = var->allvar;
   //
   //  Determine storage duration of the variable.
   //
   var->statictype[var->allvar] = G__AUTO;
   //--
   if (G__decl_obj == 2) {
      // -- We are a stack-allocated array of objects of class type.
      var->statictype[var->allvar] = G__AUTOARYDISCRETEOBJ;
      if (
         (G__globalvarpointer != G__PVOID) && // We have preallocated memory, and
         !G__cppconstruct // Not in a constructor call (memory is object), and
      ) {
         // -- We have preallocated memory and we are not in a constructor call.
         var->statictype[var->allvar] = G__COMPILEDGLOBAL;
         //--
      }
   }
   else if (
      G__def_struct_member && // We are a data member, and
      (G__tagdefining != -1) && // Inner class tag is valid, and
      (G__struct.type[G__tagdefining] == 'n') && // We are a namespace member, and
      (std::tolower(G__var_type) != 'p') // We are not a macro (macros are global)
      // FIXME: Is this 'p' test necessary?
      // FIXME: Do we need to check for type 'p' during scratch of a namespace?
   ) {
      // -- We are a namespace member.
      if ((G__globalvarpointer != G__PVOID) && !G__cppconstruct) {
         // -- We have an object preallocated, and we are not in a constructor call.
         // Note: Marking it this way means we will not free it during a scratch.
         var->statictype[var->allvar] = G__COMPILEDGLOBAL;
         //--
      }
      else if (G__static_alloc) {
         // -- Static namespace member (affects visibility, not storage duration!).
         if (G__using_alloc) {
            var->statictype[var->allvar] = G__USING_STATIC_VARIABLE;
         } else {
            var->statictype[var->allvar] = G__LOCALSTATIC;
         }
      }
      else {
         // -- Otherwise leave as auto, even though it is not stack allocated.
         if (G__using_alloc) {
            var->statictype[var->allvar] = G__USING_VARIABLE;
         }
      }
   }
   else if (G__static_alloc) {
      // -- We are a static, const, or enumerator variable.
      if (G__p_local) {
         // equal to G__prerun == 0
         // We are running, the local variable will be copied
         // from the global variable.
         G__varname_now = varname;
         // Function scope static variable
         // No real malloc(), get pointer from
         // global variable array which is suffixed
         // as varname\funcname.
         // Also, static class/struct member.
         if (G__using_alloc) {
            var->statictype[var->allvar] = G__USING_STATIC_VARIABLE;
         } else {
            var->statictype[var->allvar] = G__LOCALSTATIC;
         }
      }
      else {
         // equal to G__prerun == 1
         // We are not running, we are parsing.
         if (G__func_now != -1) {
            // -- Function scope static variable.
            // Variable allocated to global variable
            // array named varname\funcname. The
            // variable can be exclusively accessed
            // with in a specific function.
            G__FastAllocString ttt(G__ONELINE);
#ifdef G__NEWINHERIT
            if (G__p_ifunc->tagnum != -1) {
               ttt.Format("%s\\%x\\%x\\%x", varname, G__func_page, G__func_now, G__p_ifunc->tagnum);
            }
            else {
               ttt.Format("%s\\%x\\%x", varname, G__func_page, G__func_now);
            }
#else // G__NEWINHERIT
            if (G__p_ifunc->basetagnum[G__func_now] != -1) {
               ttt.Format("%s\\%x\\%x\\%x", varname, G__func_page, G__func_now, G__p_ifunc->basetagnum[G__func_now]);
            }
            else {
               ttt.Format("%s\\%x\\%x", varname, G__func_page, G__func_now);
            }
#endif // G__NEWINHERIT
            // We have no idea how big varname is. G__searchvariable() takes a char* which becomes
            // varname here; we would have to change that signature to know the size of varname here.
            // coverity[secure_coding]
            std::strcpy(varname, ttt);
            int junk;
            G__hash(ttt, varhash, junk);
            if (G__using_alloc) {
               var->statictype[var->allvar] = G__USING_STATIC_VARIABLE;
            } else {
               var->statictype[var->allvar] = G__LOCALSTATIC;
            }
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
               if (G__using_alloc) {
                  var->statictype[var->allvar] = G__USING_STATIC_VARIABLE;
               } else {
                  var->statictype[var->allvar] = G__COMPILEDGLOBAL;
               }
               //--
            }
         }
         else {
            // -- File scope static variable
            var->statictype[var->allvar] = G__ifile.filenum;
            if (
               (G__globalvarpointer != G__PVOID) && // We have preallocated memory, and
               !G__cppconstruct // Not in a constructor call (memory is object), and
            ) {
               // -- We have preallocated memory and we are not in a constructor call.
               if (G__using_alloc) {
                  var->statictype[var->allvar] = G__USING_STATIC_VARIABLE;
               } else {
                  var->statictype[var->allvar] = G__COMPILEDGLOBAL;
               }
               //--
            }
         }
      }
   }
   else if ((G__globalvarpointer != G__PVOID) && !G__cppconstruct) {
      // -- We have preallocated memory and we are not in a constructor call.
      // Note: Marking it this way means we will not free it during a scratch.
      if (G__using_alloc) {
         if (G__static_alloc) {
            var->statictype[var->allvar] = G__USING_STATIC_VARIABLE;
         } else {
            var->statictype[var->allvar] = G__USING_VARIABLE;            
         }
      } else {
         var->statictype[var->allvar] = G__COMPILEDGLOBAL;
      }
      //--
   } else if (G__using_alloc) {
      var->statictype[var->allvar] = G__USING_VARIABLE;
   }
   
   //
   //  Determine class member access control.
   //
   var->access[var->allvar] = G__PUBLIC;
   if (G__def_struct_member || G__enumdef) {
      var->access[var->allvar] = G__access;
   }
#ifndef G__NEWINHERIT
   var->isinherit[var->allvar] = 0;
#endif // G__NEWINHERIT
   //
   //  Determine whether or not to generate a dictionary entry for the variable.
   //
   //--
   if (var->tagnum == -1) {
      // -- Variable is not a data member.
      var->globalcomp[var->allvar] = G__default_link ? G__globalcomp : G__NOLINK;
   }
   else {
      // -- Variable is a data member.
      var->globalcomp[var->allvar] = G__globalcomp;
   }
   //
   //  Remember array bounds.
   //
   //  Given:
   //
   //    typedef myarray int[a][b][c];
   //    myarray v[A][B][C][D]
   //
   //  We set:
   //
   //    var->varlabel[varid][0] = 1 or B*C*D*a*b*c; // stride (for unspecified size array)
   //    var->varlabel[varid][1] = 0 or A*B*C*D*a*b*c; // number of elements
   //    var->varlabel[varid][2] = 0 or B;
   //    var->varlabel[varid][3] = 0 or C;
   //    var->varlabel[varid][4] = 0 or D;
   //    var->varlabel[varid][5] = 0 or a;
   //    var->varlabel[varid][6] = 0 or b;
   //    var->varlabel[varid][7] = 0 or c;
   //    var->varlabel[varid][8] = 1;
   //    var->varlabel[varid][9] = 0;
   //
   {
      int idx = 1;
      int stride = 1;
      for (int i = 0; i < paran; ++i) {
         int bound = G__int(para[i]);
         if (idx > 1) {
            stride *= bound;
         }
         if (idx >= G__MAXVARDIM) {
            // -- Number of dimensions is out of range.
            G__fprinterr(G__serr, "Limitation: Cint can handle only up to %d dimension array", G__MAXVARDIM - 1);
            G__genericerror(0);
            return result;
         }
         var->varlabel[var->allvar][idx] = bound;
         ++idx;
      }
      for (int i = 0; i < G__typedefnindex; ++i) {
         int bound = G__typedefindex[i];
         if (idx > 1) {
            stride *= bound;
         }
         if (idx >= G__MAXVARDIM) {
            // -- Number of dimensions is out of range.
            G__fprinterr(G__serr, "Limitation: Cint can handle only up to %d dimension array", G__MAXVARDIM - 1);
            G__genericerror(0);
            return result;
         }
         var->varlabel[var->allvar][idx] = bound;
         ++idx;
      }
      paran = idx - 1;
      var->varlabel[ig15][0] = stride;
   }
   var->paran[ig15] = paran;
   int num_elements = 0;
   if (!paran) {
      // -- We are *not* of array type.
      var->varlabel[ig15][1] = 0;
      var->varlabel[ig15][2] = 1;
      // Zero fill rest of array.
      for (int i = 3; i < G__MAXVARDIM; ++i) {
         var->varlabel[var->allvar][i] = 0;
      }
   }
   else {
      // -- We are of array type.
      if (paran < G__MAXVARDIM - 1) {
         var->varlabel[ig15][paran+1] = 1;
      }
      // Zero fill rest of array.
      for (int i = paran + 2; i < G__MAXVARDIM; ++i) {
         var->varlabel[var->allvar][i] = 0;
      }
      if ((parameter00 == '\0') && !G__typedefnindex) {
         // -- We have: var[][ddd][ddd];  an unspecified size array.
         num_elements = 0;
         var->varlabel[ig15][1] = INT_MAX /* unspecified length array flag */;
      }
      else {
         num_elements = var->varlabel[ig15][1] /* first array bound */ * var->varlabel[ig15][0] /* stride */;
         var->varlabel[ig15][1] = num_elements;
      }
      if (G__funcheader && G__asm_wholefunction && result.type) {
         // -- We cannot support function parameters of array type in whole function bytecode compilation.
         // FIXME: This cannot happen because G__asm_wholefunction and G__funcheader are never both set???
         G__ASSERT(G__globalvarpointer == G__PVOID);
         G__abortbytecode();
         G__genericerror(0);
      }
   }
   //
   // Pointer to array reimplementation.
   //
   // We start with:
   //
   // typedef myarray int[a][b][c];
   // myarray v[A][B][C][D]
   //
   //   var->varlabel[var_identity][0] = 1 or B*C*D*a*b*c; // stride for unspecified final index
   //   var->varlabel[var_identity][1] = 0 or A*B*C*D*a*b*c; // number of elements
   //   var->varlabel[var_identity][2] = 0 or B;
   //   var->varlabel[var_identity][3] = 0 or C;
   //   var->varlabel[var_identity][4] = 0 or D;
   //   var->varlabel[var_identity][5] = 0 or a;
   //   var->varlabel[var_identity][6] = 0 or b;
   //   var->varlabel[var_identity][7] = 0 or c;
   //   var->varlabel[var_identity][8] = 1;
   //   var->varlabel[var_identity][9] = 0;
   //
   if (std::isupper(G__var_type)) {
      // FIXME: This limits our max dimensions in an awkward way!
      int element_count = 1;
      int i = 0;
      int base = paran + 4;
      while (G__p2arylabel[i]) {
         element_count *= G__p2arylabel[i];
         if ((base + i) >= (G__MAXVARDIM - 1)) {
            // -- Number of dimensions is out of range.
            G__fprinterr(G__serr, "Limitation: Pointer to array exceeded array bounds capacity.");
            G__genericerror(0);
            return result;
         }
         var->varlabel[ig15][base+i] = G__p2arylabel[i];
         ++i;
      }
      if ((base + i) >= G__MAXVARDIM) {
         // -- Number of dimensions is out of range.
         G__fprinterr(G__serr, "Limitation: Pointer to array exceeded array bounds capacity.");
         G__genericerror(0);
         return result;
      }
      var->varlabel[ig15][base+i] = 1;
      // Since base == paran + 4 and i >= 0, the previous if statement insures that we have
      //    (paran+4+i) < G__MAXVARDIM so (paran+4) < G__MAXVARDIM
      var->varlabel[ig15][paran+3] = element_count;
   }
   G__p2arylabel[0] = 0;
   //
   // pointer to array reimplementation
   //
   // Finally:
   //
   // type array[A][B][C][D]
   //   var->varlabel[var_identity][0]=B*C*D; or 1;
   //   var->varlabel[var_identity][1]=A*B*C*D;
   //   var->varlabel[var_identity][2]=B;
   //   var->varlabel[var_identity][3]=C;
   //   var->varlabel[var_identity][4]=D;
   //   var->varlabel[var_identity][5]=1;
   //
   // if type (*pary[A][B][C][D])[x][y][z]
   //   var->varlabel[var_identity][6]=x*y*z or 1;
   //   var->varlabel[var_identity][7]=x;
   //   var->varlabel[var_identity][8]=y;
   //   var->varlabel[var_identity][9]=z;
   //   var->varlabel[var_identity][10]=1;
   //   var->varlabel[var_identity][11]=0;
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
         num_elements || // We are a fixed size array, or
         (var->varlabel[ig15][1] == INT_MAX /* unspecified length array flag */) // an unspecified size array,
      ) && // and
      G__asm_wholefunction && // generating bytecode for a whole function, and
      !G__static_alloc // not a static, const, or enum
   ) {
      // -- Change the array variable to a pointer variable.
      num_elements = 0;
      var->paran[ig15] = 0;
      var->varlabel[ig15][0] /* stride */ = 1;
      var->varlabel[ig15][1] /* number of elements */ = 0;
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
   if (G__funcheader && var->varlabel[var->allvar][1] /* num of elements */) {
      // -- We are a function parameter of array type.
      // We share our storage with our caller, on function
      // exit, we must not free the memory.
      var->statictype[var->allvar] = G__COMPILEDGLOBAL;
      //--
   }
   //--
   //--
   //--
   //--
   //--
   //--
   //
   //  Store comment information which is specific to root.
   //
   //--
   if (G__setcomment) {
      var->comment[var->allvar].p.com = G__setcomment;
      var->comment[var->allvar].filenum = -2;
   }
   else {
      var->comment[var->allvar].p.com = 0;
      var->comment[var->allvar].filenum = -1;
   }
   //
   //  Store file number and line number of declaration.
   //
#ifdef G__VARIABLEFPOS
   var->filenum[var->allvar] = G__ifile.filenum;
   var->linenum[var->allvar] = G__ifile.line_number;
#endif // G__VARIABLEFPOS
   //--
   //--
   //--
   //
   //  Store the variable name.
   //
   G__savestring(&var->varnamebuf[var->allvar], varname);
   var->hash[var->allvar] = varhash;
   //  Get the index into the variable page.
   ig15 = var->allvar;
   //  We are at the end of the chain.
   var->next = 0;
   //  Store type code.
   var->type[ig15] = G__var_type;
   //  Store class tagnum.
   var->p_tagtable[ig15] = G__tagnum;
   //  Store typedef typenum.
   var->p_typetable[ig15] = G__typenum;
   //
   //  Store the reference type (void is special).
   //
   var->reftype[var->allvar] = G__reftype;
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
      var->reftype[var->allvar] = G__PARANORMAL;
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
   var->constvar[ig15] = (G__SIGNEDCHAR_T) G__constvar;
   //
   //  Take ownership of the variable chain entry.
   //
   var->allvar++;
   //  Pass the handle back to the caller
   member.Set(var,ig15,index_of_var);
   // FIXME: Why?  This is bizzare.
   var->varlabel[var->allvar][0] = var->varlabel[var->allvar-1][0] + 1;
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
   //
   //  Do special processing for an enumerator, change type code.
   //
   if (
      ((G__var_type == 'p') && !G__macro_defining) &&
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
      (std::tolower(var->type[ig15]) != 'u') &&
      (result.type == 'u') &&
      (result.tagnum != -1)
   ) {
      // -- Try to convert the initializer (result), which is of class type, to the type of the variable.
      int store_decl = G__decl;
      G__decl = 0;
      G__FastAllocString ttt(G__ONELINE);
      G__fundamental_conversion_operator(var->type[ig15], var->p_tagtable[ig15], var->p_typetable[ig15], var->reftype[ig15], var->constvar[ig15], &result, ttt);
      G__decl = store_decl;
   }
   //
   //  Bytecode generation.
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
   //  Note: The following line is temporary, limitation for having class object as local variable.
   if (G__asm_wholefunction) {
      if (
         // -- G__funcheader case must be implemented, the following line is deleted.
         (
            (G__var_type == 'u') && (G__reftype != G__PARAREFERENCE)
#ifndef G__OLDIMPLEMENTATION1073
            && G__funcheader
#endif // G__OLDIMPLEMENTATION1073
            // --
         )
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
      G__asm_noverflow = 1; // FIXME: Is this right?
   }
   // Perform work, if needed.
   if (G__asm_noverflow && G__asm_wholefunction) {
      // -- Generate bytecode for variable initialization during whole function compilation.
      if (
         // -- The variable is not an array.
         !num_elements &&
         (var->varlabel[ig15][1] /* number of elements */ != INT_MAX /* unspecified length flag */)
      ) {
         // -- Handle variables which are not of array type.
         if (G__funcheader) {
            // -- We are a function parameter declaration.
            if (G__reftype != G__PARAREFERENCE) {
               // -- Initialize a non-reference parameter.
               G__asm_gen_stvar(0, ig15, var->paran[ig15], var, item, G__ASM_VARLOCAL, 'p', &result);
            }
            else {
               // -- Initialize a reference parameter.
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: INIT_REF  index: %d paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, ig15, paran, G__var_type, (long) var, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__INIT_REF;
               G__asm_inst[G__asm_cp+1] = ig15;
               G__asm_inst[G__asm_cp+2] = paran;
               G__asm_inst[G__asm_cp+3] = G__var_type;
               G__asm_inst[G__asm_cp+4] = (long) var;
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
         else if (result.type && !G__static_alloc) {
            // -- We have an initializer and we are not a static or const variable.
            G__asm_gen_stvar(0, ig15, paran, var, item, G__ASM_VARLOCAL, 'p', &result);
         }
#ifndef G__OLDIMPLEMENTATION1073
         else if (
            (G__var_type == 'u') &&
            (G__reftype != G__PARAREFERENCE) &&
            (G__tagnum != -1) &&
            (G__struct.type[G__tagnum] != 'e')
         ) {
            if (G__struct.iscpplink[G__tagnum] == G__CPPLINK) {
               // precompiled class
               // Move LD_FUNC instruction
               // LD_FUNC now has 6 parameters 06-06-07
               G__inc_cp_asm(-7, 0);
               for (int ix = 6; ix > -1; --ix) {
                  G__asm_inst[G__asm_cp+ix+4] = G__asm_inst[G__asm_cp+ix];
               }
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: CTOR_SETGVP %s index=%d paran=%d  %s:%d\n", G__asm_cp, G__asm_dt, item, ig15, paran, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__CTOR_SETGVP;
               G__asm_inst[G__asm_cp+1] = ig15;
               G__asm_inst[G__asm_cp+2] = (long) var;
               G__asm_inst[G__asm_cp+3] = 0L; /* This is the 'mode'. I am not sure what it should be */
               G__inc_cp_asm(4, 0);
               G__inc_cp_asm(7, 0); //  /* increment for moved LD_FUNC instruction */
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
                  G__fprinterr(G__serr, "%3x,%3x: LD_VAR  item: '%s' index: %d paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, item, ig15, paran, 'p', (long) var, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD_LVAR;
               G__asm_inst[G__asm_cp+1] = ig15;
               G__asm_inst[G__asm_cp+2] = paran;
               G__asm_inst[G__asm_cp+3] = 'p';
               G__asm_inst[G__asm_cp+4] = (long) var;
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
               if (var->varlabel[ig15][1] /* number of elements */ == INT_MAX /* unspecified length flag */) {
                  // -- This is a parameter of unspecified length array type, handle like a pointer.
                  // Example:  f(int var[], ...){...};
                  G__asm_gen_stvar(0, ig15, 0, var, item, G__ASM_VARLOCAL, 'p', &result);
               }
               else {
                  // -- This is *not* a parameter of unspecified length array type.
                  // Example:  f(int var[2][3][2], ...){...};
                  G__asm_gen_stvar(0, ig15, 0, var, item, G__ASM_VARLOCAL, 'p', &result);
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
      if (result.type) {
         G__asm_gen_stvar(0, ig15, paran, var, item, 0, 'p', &result);
      }
      else if (G__var_type == 'u') {
         G__ASSERT(!G__decl || (G__decl == 1));
         if (G__decl) {
            if (G__reftype) {
               G__redecl(var, ig15);
               if (G__no_exec_compile) {
                  G__abortbytecode();
               }
            }
            else {
               G__class_2nd_decl_i(var, ig15);
            }
         }
         else if (G__cppconstruct) {
            G__class_2nd_decl_c(var, ig15);
         }
      }
   }
#endif // G__ASM
   //
   //  Security check
   //
   G__CHECK(G__SECURE_POINTER_INSTANTIATE, isupper(G__var_type) && 'E' != G__var_type, return(result));
   G__CHECK(G__SECURE_POINTER_TYPE, isupper(G__var_type) && result.obj.i && G__var_type != result.type && !G__funcheader && (('Y' != G__var_type && result.obj.i) || G__security & G__SECURE_CAST2P), return(result));
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
   var->bitfield[ig15] = (char) G__bitfield;
   //--
   //--
   //--
   //--
   if (G__bitfield) {
      G__bitfield = 0;
      var->varlabel[ig15][G__MAXVARDIM-1] = bitlocation;
      if (!bitlocation) {
         var->p[ig15] = G__malloc(1, G__INTALLOC, item);
      }
      else {
         var->p[ig15] = G__malloc(1, 0, item) - G__INTALLOC;
      }
      return result;
   }
   //
   //  Now allocate memory to hold the value
   //  of the variable, and possibly initialize it.
   //
   switch (G__var_type) {
      case 'u':
         // struct, union
         if (
            G__struct.isabstract[G__tagnum] &&
            !G__ansiheader &&
            !G__funcheader &&
            (G__reftype == G__PARANORMAL) &&
            ((G__globalcomp != G__CPPLINK) || (G__tagdefining != G__tagnum))
         ) {
            G__fprinterr(G__serr, "Error: abstract class object '%s %s' declared", G__struct.name[G__tagnum], item);
            G__genericerror(0);
            G__display_purevirtualfunc(G__tagnum);
            var->hash[ig15] = 0;
         }
         // type var; normal variable
         var->p[ig15] = G__malloc(num_elements ? num_elements : 1, G__struct.size[G__tagnum], item);
         if (
            G__ansiheader &&
            result.type &&
            (G__globalvarpointer == G__PVOID) &&
            (!G__static_alloc || (G__func_now == -1))
         ) {
            std::memcpy((void*) var->p[ig15], (void*) G__int(result), (std::size_t) G__struct.size[var->p_tagtable[ig15]]);
         }
         result.obj.i = var->p[ig15];
         break;
      case 'U':
         // pointer to struct, union
         if ((num_elements > 0) && result.type) {
            // char* argv[];
            var->p[ig15] = G__int(result);
         }
         else {
            var->p[ig15] = G__malloc(num_elements ? num_elements : 1, G__LONGALLOC, item);
            if (
               (G__asm_wholefunction == G__ASM_FUNC_NOP) &&
               !G__def_struct_member &&
               (!G__static_alloc || G__prerun) &&
               ((G__globalvarpointer == G__PVOID) || result.type)
            ) {
               int baseoffset = G__ispublicbase(var->p_tagtable[ig15], result.tagnum, result.obj.i);
               if (baseoffset != -1) {
                  *((long *) var->p[ig15]) = G__int(result) + baseoffset;
               }
               else {
                  *((long *) var->p[ig15]) = G__int(result);
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
         var->p[ig15] = (long) malloc(2 * G__LONGALLOC);
         *((long*) var->p[ig15]) = 0;
         *((long*)(var->p[ig15] + G__LONGALLOC)) = 0;
         break;
#endif // G__ROOT
#ifndef G__OLDIMPLEMENTATION2191
      case '1':
         // void
         G__alloc_var_ref<char>(G__CHARALLOC, G__int, item, var, ig15, result);
         break;
#else // G__OLDIMPLEMENTATION2191
      case 'Q':
         // void
         G__alloc_var_ref<char>(G__CHARALLOC, G__int, item, var, ig15, result);
         break;
#endif // G__OLDIMPLEMENTATION2191
      case 'Y':
         // pointer to void
         G__alloc_var_ref<char>(G__CHARALLOC, G__int, item, var, ig15, result);
         break;
      case 'E':
         // pointer to FILE
         G__alloc_var_ref<char>(G__CHARALLOC, G__int, item, var, ig15, result);
         break;
      case 'C':
         // pointer to char
         G__alloc_var_ref<char>(G__CHARALLOC, G__int, item, var, ig15, result);
         break;
      case 'c':
         // char
         // Check for initialization of a character variable with a string constant.
         if (
            G__funcheader || // function arg initialization, or
            !var->varlabel[ig15][1] /* number of elements */ || // not array,
            (result.type != 'C') // or not init by char array
         ) {
            // -- The simple case, just allocate a char and initialize it.
            G__alloc_var_ref<char>(G__CHARALLOC, G__int, item, var, ig15, result);
         }
         else {
            // -- We are a char array being initialized with a string constant.
            if (G__asm_wholefunction != G__ASM_FUNC_COMPILE) {
               // -- Not whole func compilation, allocate mem and copy initializer in.
               if (
                  !G__funcheader && // Not in a function header (we share value with caller), and
                  !G__def_struct_member && // Not defining a member variable (var->p[ig15] is an offset), and
                  !(G__static_alloc && (G__func_now > -1) && !G__prerun) // Not a static variable in function scope at runtime (init was done in prerun).
               ) {
                  size_t len = strlen((char*) result.obj.i);
                  if (var->varlabel[ig15][1] /* number of elements */ == INT_MAX /* unspecified length flag */) {
                     // -- We are an unspecified length array of char being initialized with a string constant.
                     // FIXME: Can this happen?
                     var->p[ig15] = (long) malloc(len + 1);
                     strcpy((char*) var->p[ig15], (char*) result.obj.i); // Okay we allocated enough space
                  }
                  else if (len > var->varlabel[ig15][1] /* number of elements */) {
                     // -- We are an array of char being initialized with a string constant that is too big.
                     // FIXME: Can this happen?
                     // FIXME: We need to give an error message here!
                     G__alloc_var_ref<char>(G__CHARALLOC, G__int, item, var, ig15, result);
                     strncpy((char*) var->p[ig15], (char*) result.obj.i, (size_t) var->varlabel[ig15][1] /* number of elements */);
                  }
                  else {
                     // -- We are an array of char being initialized with a string constant.
                     // FIXME: Can this happen?
                     G__alloc_var_ref<char>(G__CHARALLOC, G__int, item, var, ig15, result);
                     strcpy((char*) var->p[ig15], (char*) result.obj.i); // Okay we allocated enough memory
                     int num_omitted = var->varlabel[ig15][1] /* number of elements */ - len;
                     memset(((char*) var->p[ig15]) + len, 0, num_omitted);
                  }
               }
            }
            else {
               // In whole function compilation, we must reserve a memory block
               // in the function local storage and generate instructions to
               // copy the initializer in.
               if (
                  !G__funcheader && // Not in a function header (we share value with caller), and
                  !G__def_struct_member && // Not defining a member variable (var->p[ig15] is an offset), and
                  !(G__static_alloc && (G__func_now > -1) && !G__prerun) // Not a static variable in function scope at runtime (init was done in prerun).
               ) {
                  size_t len = strlen((char*) result.obj.i);
                  if (var->varlabel[ig15][1] /* number of elements */ == INT_MAX /* unspecified length flag */) {
                     // -- We are an unspecified length array of char being initialized with a string constant.
                     // FIXME: Can this happen?
                     var->p[ig15] = (long) G__malloc(1, len + 1, item);
                  }
                  else if (len > var->varlabel[ig15][1] /* number of elements */) {
                     // -- We are an array of char being initialized with a string constant that is too big.
                     // FIXME: Can this happen?
                     // FIXME: We need to give an error message here!
                     var->p[ig15] = (long) G__malloc(1, len + 1, item);
                  }
                  else {
                     // -- We are an array of char being initialized with a string constant.
                     // FIXME: Can this happen?
                     var->p[ig15] = (long) G__malloc(1, var->varlabel[ig15][1], item);
                  }
               }
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD_VAR '%s' index: %d paran: %d type: 'P'  %s:%d\n", G__asm_cp, G__asm_dt, var->varnamebuf[ig15], ig15, 0, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD_LVAR;
               G__asm_inst[G__asm_cp+1] = ig15; // index
               G__asm_inst[G__asm_cp+2] = 0; // paran
               G__asm_inst[G__asm_cp+3] = 'P'; // type
               G__asm_inst[G__asm_cp+4] = (long) var;
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
                  G__fprinterr(G__serr, "%3x,%3x: LD_FUNC 'strcpy' %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__); // Okay
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD_FUNC;
               G__asm_inst[G__asm_cp+1] = (long) "strcpy"; // name. Okay
               G__asm_inst[G__asm_cp+2] = 677; // hash
               G__asm_inst[G__asm_cp+3] = 2; // paran
               G__asm_inst[G__asm_cp+4] = (long) G__compiled_func; // pfunc
               G__asm_inst[G__asm_cp+5] = 0; // this ptr adjustment
               G__asm_inst[G__asm_cp+6] = (long) G__p_ifunc; // ifunc, ignored because pfunc is set
               G__asm_inst[G__asm_cp+7] = -1; // ifn, for special func
               G__inc_cp_asm(8, 0);
            }
         }
         break;
      case 'n':
         G__alloc_var_ref<G__int64>(G__LONGLONGALLOC, G__Longlong, item, var, ig15, result);
         break;
      case 'N':
         G__alloc_var_ref<G__int64>(G__LONGLONGALLOC, G__Longlong, item, var, ig15, result);
         break;
      case 'm':
         G__alloc_var_ref<G__int64>(G__LONGLONGALLOC, G__Longlong, item, var, ig15, result);
         break;
      case 'M':
         G__alloc_var_ref<G__int64>(G__LONGLONGALLOC, G__Longlong, item, var, ig15, result);
         break;
#ifndef G__OLDIMPLEMENTATION2191
      case 'q':
      case 'Q':
         G__alloc_var_ref<long double>(G__LONGDOUBLEALLOC, G__Longdouble, item, var, ig15, result);
         break;
#endif // G__OLDIMPLEMENTATION2191
      case 'g':
         // bool
         result.obj.i = G__int(result) ? 1 : 0; // Force the value to 1 or 0.
#ifdef G__BOOL4BYTE
         G__alloc_var_ref<int>(G__INTALLOC, G__int, item, var, ig15, result);
#else // G__BOOL4BYTE
         G__alloc_var_ref<unsigned char>(G__CHARALLOC, G__int, item, var, ig15, result);
#endif // G__BOOL4BYTE
         break;
      case 'G':
         // pointer to bool
#ifdef G__BOOL4BYTE
         G__alloc_var_ref<int>(G__INTALLOC, G__int, item, var, ig15, result);
#else // G__BOOL4BYTE
         G__alloc_var_ref<unsigned char>(G__CHARALLOC, G__int, item, var, ig15, result);
#endif // G__BOOL4BYTE
         break;
      case 'b':
         // unsigned char
         G__alloc_var_ref<unsigned char>(G__CHARALLOC, G__int, item, var, ig15, result);
         break;
      case 'B':
         // pointer to unsigned char
         G__alloc_var_ref<unsigned char>(G__CHARALLOC, G__int, item, var, ig15, result);
         break;
      case 's':
         // short int
         G__alloc_var_ref<short>(G__SHORTALLOC, G__int, item, var, ig15, result);
         break;
      case 'S':
         // pointer to short int
         G__alloc_var_ref<short>(G__SHORTALLOC, G__int, item, var, ig15, result);
         break;
      case 'r':
         // unsigned short int
         G__alloc_var_ref<unsigned short>(G__SHORTALLOC, G__int, item, var, ig15, result);
         break;
      case 'R':
         // pointer to unsigned short int
         G__alloc_var_ref<unsigned short>(G__SHORTALLOC, G__int, item, var, ig15, result);
         break;
      case 'i':
         // int
         G__alloc_var_ref<int>(G__INTALLOC, G__int, item, var, ig15, result);
         break;
      case 'I':
         // pointer to int
         G__alloc_var_ref<int>(G__INTALLOC, G__int, item, var, ig15, result);
         break;
      case 'h':
         // unsigned int
         G__alloc_var_ref<unsigned int>(G__INTALLOC, G__int, item, var, ig15, result);
         break;
      case 'H':
         // pointer to unsigned int
         G__alloc_var_ref<unsigned int>(G__INTALLOC, G__int, item, var, ig15, result);
         break;
      case 'l':
         // long int
         G__alloc_var_ref<long>(G__LONGALLOC, G__int, item, var, ig15, result);
         break;
      case 'L':
         // pointer to long int
         G__alloc_var_ref<long>(G__LONGALLOC, G__int, item, var, ig15, result);
         break;
      case 'k':
         // unsigned long int
         G__alloc_var_ref<unsigned long>(G__LONGALLOC, G__int, item, var, ig15, result);
         break;
      case 'K':
         // pointer to unsigned long int
         G__alloc_var_ref<unsigned long>(G__LONGALLOC, G__int, item, var, ig15, result);
         break;
      case 'f':
         // float
         G__alloc_var_ref<float>(G__FLOATALLOC, G__double, item, var, ig15, result);
         break;
      case 'F':
         // pointer to float
         G__alloc_var_ref<float>(G__FLOATALLOC, G__double, item, var, ig15, result);
         break;
      case 'd':
         // double
         G__alloc_var_ref<double>(G__DOUBLEALLOC, G__double, item, var, ig15, result);
         break;
      case 'D':
         // pointer to double
         G__alloc_var_ref<double>(G__DOUBLEALLOC, G__double, item, var, ig15, result);
         break;
      case 'e':
         // FILE
         G__genericerror("Limitation: FILE type variable can not be declared unless type FILE is explicitly defined");
         var->hash[ig15] = 0;
         break;
      case 'y':
         // void
         G__genericerror("Error: void type variable can not be declared");
         var->hash[ig15] = 0;
         break;
#ifndef G__OLDIMPLEMENTATION2191
      case 'j':
         // macro file position
#else // G__OLDIMPLEMENTATION2191
      case 'm':
         // macro file position
#endif // G__OLDIMPLEMENTATION2191
         var->p[ig15] = G__malloc(1, sizeof(std::fpos_t), item);
         *((std::fpos_t*) var->p[ig15]) = *((std::fpos_t*) result.obj.i);
         break;
      case 'a':
         // pointer to member function
         var->p[ig15] = G__malloc(num_elements ? num_elements : 1, G__P2MFALLOC, item);
         if (
            (G__asm_wholefunction == G__ASM_FUNC_NOP) &&
            !G__def_struct_member &&
            (!G__static_alloc || G__prerun) &&
            result.obj.i &&
            ((G__globalvarpointer == G__PVOID) || result.type)
         ) {
            // --
#ifdef G__PTR2MEMFUNC
            if (result.type == 'C') {
               *((long*) var->p[ig15]) = result.obj.i;
            }
            else {
               std::memcpy((void*) var->p[ig15], (void*) result.obj.i, G__P2MFALLOC);
            }
#else // G__PTR2MEMFUNC
            std::memcpy((void*) var->p[ig15], (void*) result.obj.i, G__P2MFALLOC);
#endif // G__PTR2MEMFUNC
            // --
         }
         break;
#ifndef G__OLDIMPLEMENTATION2191
         //case '1':
         // function, ???Questionable???
         // var->p[ig15] = G__malloc(num_elements ? num_elements : 1, sizeof(long), item);
         // break;
#else // G__OLDIMPLEMENTATION2191
      case 'q':
         // function, ???Questionable???
         var->p[ig15] = G__malloc(num_elements ? num_elements : 1, sizeof(long), item);
         break;
#endif // G__OLDIMPLEMENTATION2191
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
            if (var->tagnum != -1) {
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
            var->type[ig15] = 'o';
         }
         // Reallocate array index information.
         for (int i = 0; i < paran; ++i) {
            var->varlabel[ig15][i+1] = G__int(para[i]) + 1;
         }
         {
            int tmp = 1;
            for (int i = 2; i < paran + 1; ++i) {
               tmp *= var->varlabel[ig15][i++];
            }
            var->varlabel[ig15][0] /* stride */ = tmp;
            num_elements = tmp * var->varlabel[ig15][1];
            var->varlabel[ig15][1] /* number of elements */ = num_elements;
         }
         // Allocate double macro or not.
         if (G__isdouble(result)) {
            // 'P' macro double, 'O' auto double.
            var->type[ig15] = std::toupper(var->type[ig15]);
            var->p[ig15] = G__malloc(num_elements ? num_elements : 1, G__DOUBLEALLOC, item);
            if (
               (G__asm_wholefunction == G__ASM_FUNC_NOP) &&
               (!G__static_alloc || G__prerun) &&
               ((G__globalvarpointer == G__PVOID) || result.type)
            ) {
               *(((double*) var->p[ig15]) + ((num_elements ? num_elements : 1) - 1)) = G__double(result);
            }
         }
         else {
            // 'p' macro int, 'o' auto int.
            if (result.type == 'C') {
               var->type[ig15] = 'T';
            }
            var->p[ig15] = G__malloc((num_elements ? num_elements : 1) + 1, G__LONGALLOC, item);
            if (
               (G__asm_wholefunction == G__ASM_FUNC_NOP) &&
               (!G__static_alloc || G__prerun) &&
               ((G__globalvarpointer == G__PVOID) || result.type)
            ) {
               *(((long*) var->p[ig15]) + ((num_elements ? num_elements : 1) - 1)) = G__int(result);
            }
         }
         break;
   }
   //
   //  Security, check for unassigned internal pointer.
   //
   G__CHECK(G__SECURE_POINTER_INIT, !G__def_struct_member && std::isupper(G__var_type) && (G__ASM_FUNC_NOP == G__asm_wholefunction) && !var->varlabel[ig15][1] && var->p[ig15] && (0 == (*((long*) var->p[ig15]))), *((long*) var->p[ig15]) = 0);
   //
   //  Security, increment reference count on a pointed-at object.
   //
#ifdef G__SECURITY
   if (
      (G__security & G__SECURE_GARBAGECOLLECTION) &&
      !G__def_struct_member &&
      !G__no_exec_compile &&
      std::isupper(G__var_type) &&
      var->p[ig15] &&
      (*((long*) var->p[ig15]))
   ) {
      G__add_refcount((void*)(*((long*) var->p[ig15])), (void**) var->p[ig15]);
   }
#endif // G__SECURITY
   //
   //  Do special fixups for an automatic object.
   //
   if (autoobjectflag) {
      var->statictype[ig15] = G__AUTO;
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
      (var->tagnum != -1) && // data member
      (G__access == G__PUBLIC) && // public access
      !std::strcmp(varname, "G__virtualinfo") // special virtual offset variable
   ) {
      // -- Copy the virtual offset data member value to the class metadata.
      G__struct.virtual_offset[var->tagnum] = var->p[ig15];
   }
   return result;
}

#undef G__ALLOC_VAR_REF

extern "C" {

//______________________________________________________________________________
static int G__asm_gen_stvar(long G__struct_offset, int ig15, int paran, G__var_array* var, const char*
#ifdef G__ASM_DBG
item
#endif
, long store_struct_offset, int var_type, G__value* /*presult*/)
{
   // -- FIXME: Describe me!
   //
   // ST_GVAR or ST_VAR instruction
   //
   if (G__struct_offset) {
      // --
#ifdef G__NEWINHERIT // Always
      if (G__struct_offset != store_struct_offset) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: ADDSTROS %d  %s:%d\n", G__asm_cp, G__asm_dt, G__struct_offset - store_struct_offset, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__ADDSTROS;
         G__asm_inst[G__asm_cp+1] = G__struct_offset - store_struct_offset;
         G__inc_cp_asm(2, 0);
      }
#endif // G__NEWINHERIT
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: ST_MSTR %s index=%d paran=%d  %s:%d\n", G__asm_cp, G__asm_dt, item, ig15, paran, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__ST_MSTR;
      G__asm_inst[G__asm_cp+1] = ig15;
      G__asm_inst[G__asm_cp+2] = paran;
      G__asm_inst[G__asm_cp+3] = var_type;
      G__asm_inst[G__asm_cp+4] = (long) var;
      G__inc_cp_asm(5, 0);
#ifdef G__NEWINHERIT // Always
      if (G__struct_offset != store_struct_offset) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: ADDSTROS %d  %s:%d\n", G__asm_cp, G__asm_dt, -G__struct_offset + store_struct_offset, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__ADDSTROS;
         G__asm_inst[G__asm_cp+1] = -G__struct_offset + store_struct_offset;
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
      G__redecl(var, ig15);
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
         (store_struct_offset == G__ASM_VARLOCAL) &&
         (var->statictype[ig15] != G__LOCALSTATIC)
      ) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: ST_LVAR item: '%s' index: %d paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, item, ig15, paran, var_type, (long) var, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__ST_LVAR;
      }
      else {
         // -- Normal case.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: ST_VAR item: '%s' index: %d paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, item, ig15, paran, var_type, (long) var, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__ST_VAR;
      }
#else // G__ASM_WHOLEFUNC
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: ST_VAR item: '%s' index: %d paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, item, ig15, paran, var_type, (long) var, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__ST_VAR;
#endif // G__ASM_WHOLEFUNC
      G__asm_inst[G__asm_cp+1] = ig15;
      G__asm_inst[G__asm_cp+2] = paran;
      G__asm_inst[G__asm_cp+3] = var_type;
      G__asm_inst[G__asm_cp+4] = (long) var;
      G__inc_cp_asm(5, 0);
   }
   return 0;
}

//______________________________________________________________________________
//
//  Get the value from a variable which is not of pointer type.
//
#define G__GET_VAR(SIZE, CASTTYPE, CONVFUNC, TYPE, PTYPE) \
   switch (G__var_type) { \
      case 'p': \
         /* -- Return the variable value. */ \
         if (paran >= var->paran[ig15]) { \
            /* -- Value is not an array or pointer type.  MyType v = var; */ \
            /* FIXME: This is wrong, if greater than we should be doing pointer arithmetic. */ \
            result.ref = G__struct_offset + var->p[ig15] + (linear_index * SIZE); \
            CONVFUNC(&result, TYPE, (CASTTYPE) (*((CASTTYPE*) result.ref))); \
         } \
         else { \
            /* -- We are accessing part of an array, so we must do the array to pointer standard conversion. */ \
            /* int ary[2][3][5];  int* v = ary[1][2]; */ \
            /* The resulting value is a pointer to the first element. */ \
            G__letint(&result, PTYPE, G__struct_offset + var->p[ig15] + (linear_index * SIZE)); \
            /* FIXME: We have no way of representing an array return type. */ \
            /* FIXME: The best we can do is call it a multi-level pointer. */ \
            /* FIXME: This is a violation of the C++ type system. */ \
            if ((var->paran[ig15] - paran) > 1) { \
               /* -- We are more than one level of pointers deep, construct a pointer chain. */ \
               /* FIXME: This is wrong, we need to make a pointer to an array. */ \
               result.obj.reftype.reftype = var->paran[ig15] - paran; \
               /**/ \
               /**/ \
            } \
         } \
         break; \
      case 'P': \
         /* -- Return a pointer to the variable value.  MyType* v = &var; */ \
         G__letint(&result, PTYPE, G__struct_offset + var->p[ig15] + (linear_index * SIZE)); \
         break; \
      default: \
         /* -- case 'v': */ \
         G__reference_error(item); \
         break; \
   } \
   G__var_type = 'p'; \
   return result;

//______________________________________________________________________________
//
//  Get the value from a variable of pointer type.
//

extern "C++" {
template<class CASTTYPE, class CONVTYPE, class CONVFUNC>
inline void G__get_pvar(CONVFUNC f, char TYPE, char PTYPE, struct G__var_array* var, int ig15, long G__struct_offset, int paran, G__value para[G__MAXVARDIM], size_t linear_index, size_t secondary_linear_index, G__value* result)
{
   switch (G__var_type) {
      case 'v':
         /* -- Return the value that the pointer variable points to.  Mytype* var; MyType v = *var; */
         switch (var->reftype[ig15]) {
            case G__PARANORMAL:
               /* -- Variable is a one-level pointer. */
               result->ref = (*(long*) (G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC)));
               if (result->ref) {
                  (*f)(result, TYPE, (CONVTYPE) (*((CASTTYPE*) result->ref)));
               }
               break;
            case G__PARAP2P:
               if (var->paran[ig15] < paran) {
                  long address = G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC);
                  result->ref = *(long*) (((CASTTYPE*) (*(long*) address)) + secondary_linear_index);
                  if (result->ref) {
                     (*f)(result, TYPE, (CONVTYPE) (*(CASTTYPE*) result->ref));
                  }
               }
               else {
                  result->ref = *(long*)(G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC));
                  G__letint(result, PTYPE, *(long*)(*(long*) (G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC))));
               }
               break;
         }
         break;
      case 'P':
         /* -- Return a pointer to the pointer variable value.  MyType* var; MyType** v = &var; */
         if (var->paran[ig15] == paran) {
            G__letint(result, PTYPE, G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC));
         }
         else if (var->paran[ig15] < paran) {
            long address = G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC);
            if (var->reftype[ig15] == G__PARANORMAL) {
               G__letint(result, PTYPE, (long)((CASTTYPE*)(*(long*)address) + secondary_linear_index));
            }
            else {
               G__letint(result, PTYPE, (long)((long*)(*(long*)address) + secondary_linear_index));
               result->obj.reftype.reftype = G__PARAP2P;
            }
         }
         else {
            G__letint(result, PTYPE, G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC));
         }
         break;
      default :
         /* 'p' -- Return the pointer variable value.  MyType* var; MyType* v = var; */
         if (paran == var->paran[ig15]) {
            /* MyType* var[ddd]; MyType* v = var[xxx]; */
            result->ref = (long)(G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC));
            G__letint(result, PTYPE, *((long*) result->ref));
         }
         else if (paran > var->paran[ig15]) {
            /* -- Pointer to array reimplementation. */
            /* MyType* var[ddd];  v = var[xxx][yyy]; */
            long address = G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC);
            if (var->reftype[ig15] == G__PARANORMAL) {
               /* -- Variable is a single-level pointer. */
               /* Pointer to array reimplementation. */
               result->ref = (long)((CASTTYPE*)(*(long*)address) + secondary_linear_index);
               (*f)(result, TYPE, (CONVTYPE)(*((CASTTYPE*)result->ref)));
            }
            else if ((paran - 1) == var->paran[ig15]) {
               /* Pointer to array reimplementation. */
               result->ref = (long)((long*)(*(long*)address) + secondary_linear_index);
               G__letint(result, PTYPE, (long)((CASTTYPE*)(*((long*)result->ref))));
               if (var->reftype[ig15] > G__PARAP2P) {
                  result->obj.reftype.reftype = var->reftype[ig15] - 1;
               }
            }
            else {
               /* -- Start doing pointer arithmetic. */
               result->ref = (long)((long*)(*(long*)address) + para[0].obj.i);
               for (int ip = 1; ip < (paran - 1); ++ip) {
                  result->ref = (long)((long*)(*(long*)result->ref) + para[ip].obj.i);
               }
               result->obj.reftype.reftype = var->reftype[ig15] - paran + var->paran[ig15];
               /**/
               /**/
               /**/
               switch (result->obj.reftype.reftype) {
                  case G__PARANORMAL:
                     result->ref = (long)((CASTTYPE*)(*((long*)result->ref)) + para[paran-1].obj.i);
                     (*f)(result, TYPE, *((CASTTYPE*)result->ref));
                     break;
                  case 1:
                     result->ref = (long)((long*)(*((long*)result->ref)) + para[paran-1].obj.i);
                     G__letint(result, PTYPE, *((long*)result->ref));
                     result->obj.reftype.reftype = G__PARANORMAL;
                     break;
                  default:
                     result->ref = (long)((long*)(*((long*)result->ref)) + para[paran-1].obj.i);
                     G__letint(result, PTYPE, *((long*)result->ref));
                     result->obj.reftype.reftype = var->reftype[ig15] - paran + var->paran[ig15];
                     /**/
                     /**/
                     /**/
                     break;
               }
            }
         }
         else {
            /* paran < var->paran[ig15] */
            /* MyType* var[ddd][nnn]; MyType** v = var[xxx]; */
            /* FIXME: This is a syntax error if (var->paran[ig15] - paran) > 1. */
            result->ref = (long)(&var->p[ig15]);
            G__letint(result, PTYPE, *((long*) result->ref));
         }
         break;
   }
}
} // extern "C++"

#define G__GET_PVAR(CASTTYPE, CONVFUNC, CONVTYPE, TYPE, PTYPE) \
   switch (G__var_type) { \
      case 'v': \
         /* -- Return the value that the pointer variable points to.  Mytype* var; MyType v = *var; */ \
         switch (var->reftype[ig15]) { \
            case G__PARANORMAL: \
               /* -- Variable is a one-level pointer. */ \
               result.ref = (*(long*) (G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC))); \
               if (result.ref) { \
                  CONVFUNC(&result, TYPE, (CONVTYPE) (*((CASTTYPE*) result.ref))); \
               } \
               break; \
            case G__PARAP2P: \
               if (var->paran[ig15] < paran) { \
                  long address = G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC); \
                  result.ref = *(long*) (((CASTTYPE*) (*(long*) address)) + secondary_linear_index); \
                  if (result.ref) { \
                     CONVFUNC(&result, TYPE, (CONVTYPE) (*(CASTTYPE*) result.ref)); \
                  } \
               } \
               else { \
                  result.ref = *(long*)(G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC)); \
                  G__letint(&result, PTYPE, *(long*)(*(long*) (G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC)))); \
               } \
               break; \
         } \
         break; \
      case 'P': \
         /* -- Return a pointer to the pointer variable value.  MyType* var; MyType** v = &var; */ \
         if (var->paran[ig15] == paran) { \
            G__letint(&result, PTYPE, G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC)); \
         } \
         else if (var->paran[ig15] < paran) { \
            long address = G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC); \
            if (var->reftype[ig15] == G__PARANORMAL) { \
               G__letint(&result, PTYPE, (long)((CASTTYPE*)(*(long*)address) + secondary_linear_index)); \
            } \
            else { \
               G__letint(&result, PTYPE, (long)((long*)(*(long*)address) + secondary_linear_index)); \
               result.obj.reftype.reftype = G__PARAP2P; \
            } \
         } \
         else { \
            G__letint(&result, PTYPE, G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC)); \
         } \
         break; \
      default : \
         /* 'p' -- Return the pointer variable value.  MyType* var; MyType* v = var; */ \
         if (paran == var->paran[ig15]) { \
            /* MyType* var[ddd]; MyType* v = var[xxx]; */ \
            result.ref = (long)(G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC)); \
            G__letint(&result, PTYPE, *((long*) result.ref)); \
         } \
         else if (paran > var->paran[ig15]) { \
            /* -- Pointer to array reimplementation. */ \
            /* MyType* var[ddd];  v = var[xxx][yyy]; */ \
            long address = G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC); \
            if (var->reftype[ig15] == G__PARANORMAL) { \
               /* -- Variable is a single-level pointer. */ \
               /* Pointer to array reimplementation. */ \
               result.ref = (long)((CASTTYPE*)(*(long*)address) + secondary_linear_index); \
               CONVFUNC(&result, TYPE, (CONVTYPE)(*((CASTTYPE*)result.ref))); \
            } \
            else if ((paran - 1) == var->paran[ig15]) { \
               /* Pointer to array reimplementation. */ \
               result.ref = (long)((long*)(*(long*)address) + secondary_linear_index); \
               G__letint(&result, PTYPE, (long)((CASTTYPE*)(*((long*)result.ref)))); \
               if (var->reftype[ig15] > G__PARAP2P) { \
                  result.obj.reftype.reftype = var->reftype[ig15] - 1; \
               } \
            } \
            else { \
               /* -- Start doing pointer arithmetic. */ \
               result.ref = (long)((long*)(*(long*)address) + para[0].obj.i); \
               for (int ip = 1; ip < (paran - 1); ++ip) { \
                  result.ref = (long)((long*)(*(long*)result.ref) + para[ip].obj.i); \
               } \
               result.obj.reftype.reftype = var->reftype[ig15] - paran + var->paran[ig15]; \
               /**/ \
               /**/ \
               /**/ \
               switch (result.obj.reftype.reftype) { \
                  case G__PARANORMAL: \
                     result.ref = (long)((CASTTYPE*)(*((long*)result.ref)) + para[paran-1].obj.i); \
                     CONVFUNC(&result, TYPE, *((CASTTYPE*)result.ref)); \
                     break; \
                  case 1: \
                     result.ref = (long)((long*)(*((long*)result.ref)) + para[paran-1].obj.i); \
                     G__letint(&result, PTYPE, *((long*)result.ref)); \
                     result.obj.reftype.reftype = G__PARANORMAL; \
                     break; \
                  default: \
                     result.ref = (long)((long*)(*((long*)result.ref)) + para[paran-1].obj.i); \
                     G__letint(&result, PTYPE, *((long*)result.ref)); \
                     result.obj.reftype.reftype = var->reftype[ig15] - paran + var->paran[ig15]; \
                     /**/ \
                     /**/ \
                     /**/ \
                     break; \
               } \
            } \
         } \
         else { \
            /* paran < var->paran[ig15] */ \
            /* MyType* var[ddd][nnn]; MyType** v = var[xxx]; */ \
            /* FIXME: This is a syntax error if (var->paran[ig15] - paran) > 1. */ \
            if (G__struct_offset) { \
               result.ref = (long)(G__struct_offset + var->p[ig15]);         \
            } else { \
               result.ref = (long)(&var->p[ig15]); \
            } \
            G__letint(&result, PTYPE, *((long*) result.ref)); \
         } \
         break; \
   }

//______________________________________________________________________________
G__value G__getvariable(char* item, int* known, G__var_array* varglobal, G__var_array* varlocal)
{
   // -- FIXME: Describe me!
   struct G__var_array* var = 0;
   char parameter[G__MAXVARDIM][G__ONELINE];
   G__value para[G__MAXVARDIM];
   int ig15 = 0;
   int paran = 0;
   int ig25 = 0;
   size_t lenitem = 0;
   int done = 0;
   long G__struct_offset = 0L;
   char store_var_type = '\0';
   int varhash = 0;
   //--
   struct G__input_file store_ifile;
   int store_vartype = 0;
   long store_struct_offset = 0;
   int store_tagnum = 0;
   int posbracket = 0;
   int posparenthesis = 0;
   G__value result = G__null;
   G__FastAllocString varname(2*G__MAXNAME);
#ifdef G__ASM
   //
   //  If we are called by running bytecode,
   //  skip the string parsing, that was already done
   //  during bytecode compilation.
   //
   if (G__asm_exec) {
      ig15 = G__asm_index;
      paran = G__asm_param->paran;
      for (int i = 0; i < paran; ++i) {
         para[i] = G__asm_param->para[i];
      }
      para[paran] = G__null;
      var = varglobal;
      if (!varlocal) {
         G__struct_offset = 0;
      }
      else {
         G__struct_offset = G__store_struct_offset;
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
            int store_var_type = G__var_type;
            G__var_type = 'p';
            *known = 1;
            G__value tmpval = G__getexpr(item + 1);
            G__value v = G__tovalue(tmpval);
            if (store_var_type != 'p') {
               v = G__toXvalue(v, store_var_type);
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
         // remove '*' from expression.
         // char *item is modified.
         //
         for (size_t i = 0; i < lenitem; ++i) {
            item[i] = item[i+1];
         }
         break;
      case '&':
         // -- Take address of operator.
         // if '&varname'
         // this case only happens when '&tag.varname'.
         lenitem = strlen(item);
         G__var_type = 'P'; // FIXME: This special type is set only here.
         for (size_t i = 0; i < lenitem; ++i) {
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
      for (size_t i = 0; i < lenitem; ++i) {
         switch (item[i]) {
            case '.':
               // -- This is a member of struct or union accessed by member reference.
               if (!paren && !double_quote && !single_quote) {
                  // To get full struct member name path when not found.
                  G__FastAllocString tmp(item);
                  tmp[i++] = '\0';
                  char* tagname = tmp;
                  char* membername = tmp + i;
                  G__FastAllocString varname(2*G__MAXNAME);
                  G__value val = G__getstructmem(store_var_type, varname, membername,
                                                 tmp.Capacity() - i - 1, tagname,
                                                 known, varglobal, 1);
                  return val;
               }
               break;
            case '-':
               // -- This is a member of struct or union accessed by pointer dereference.
               if (!paren && !double_quote && !single_quote && (item[i+1] == '>')) {
                  // To get full struct member name path when not found.
                  G__FastAllocString tmp(i + 2);
                  strncpy(tmp, item, i);
                  tmp[i++] = '\0';
                  tmp[i++] = '\0';
                  char* tagname = tmp;
                  char* membername = item + i;
                  G__FastAllocString varname(2*G__MAXNAME);
                  G__value val = G__getstructmem(store_var_type, varname, membername,
                                                 INT_MAX /* we don't know */, tagname,
                                                 known, varglobal, 2);
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
   G__struct_offset = 0;
   if (!varglobal) {
      // We have been called from ourselves or G__letvariable() and
      // our caller set G__store_struct_offset.
      G__struct_offset = G__store_struct_offset;
   }
   {
      // Collect the variable name and hash value,
      // stop at parenthesis or square brackets.
      size_t cursor = 0;
      for (cursor = 0; (item[cursor] != '(') && (item[cursor] != '[') && (cursor < lenitem); ++cursor) {
         varname.Set(cursor, item[cursor]);
         varhash += item[cursor];
      }
      if (item[cursor] == '(') {
         // -- We have: 'funcname(xxx)', will be handled by G__getfunction(), abort for now.
         return G__null;
      }
      if (posparenthesis && (item[cursor] == '[')) {
         // -- We have: var[x](a,b);
         item[posparenthesis] = 0;
         G__value val = G__getvariable(item, known, varglobal, varlocal);
         if (!known) {
            return G__null;
         }
         item[posparenthesis] = '(';
         val = G__pointer2func(&val, 0, item + posparenthesis, known);
         *known = 1;
         return val;
      }
      varname.Set(cursor++, 0);
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
   var = G__searchvariable(varname, varhash, varlocal, varglobal, &G__struct_offset, &store_struct_offset, &ig15, 0);
   if (!var && (G__prerun || G__eval_localstatic) && (G__func_now >= 0)) {
      G__FastAllocString temp(G__ONELINE);
      if (G__tagdefining != -1) {
         temp.Format("%s\\%x\\%x\\%x", varname(), G__func_page, G__func_now, G__tagdefining);
      }
      else {
         temp.Format("%s\\%x\\%x", varname(), G__func_page, G__func_now);
      }
      int itmpx = 0;
      G__hash(temp, varhash, itmpx);
      var = G__searchvariable(temp, varhash, varlocal, varglobal, &G__struct_offset, &store_struct_offset, &ig15, 0);
      if (var) {
         G__struct_offset = 0;
      }
      if (!var && G__getarraydim && !G__IsInMacro()) {
         G__const_noerror = 0;
         G__genericerror("Error: Illegal array dimension (Ignore subsequent errors)");
         *known = 1;
         return G__null;
      }
   }
   if (var) {
      // -- We found the variable, return its value.
#ifndef G__OLDIMPLEMENTATION1259
      result.isconst = var->constvar[ig15];
      if (var->p_typetable[ig15] != -1) {
         result.isconst |= G__newtype.isconst[var->p_typetable[ig15]];
      }
#endif // G__OLDIMPLEMENTATION1259
      if (
         G__getarraydim &&
         !G__IsInMacro() &&
#ifndef G__OLDIMPLEMENTATION2191
         (var->type[ig15] != 'j') &&
#else // G__OLDIMPLEMENTATION2191
         (var->type[ig15] != 'm') &&
#endif // G__OLDIMPLEMENTATION2191
         (var->type[ig15] != 'p') &&
         (
            !(var->constvar[ig15] & G__CONSTVAR) ||
            (var->constvar[ig15] & G__DYNCONST)
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
      if (!var->p[ig15] && !G__struct_offset && !G__no_exec_compile) {
         *known = 1;
         G__value val = G__null;
         val.tagnum = var->p_tagtable[ig15];
         val.typenum = var->p_typetable[ig15];
         switch (G__var_type) {
            case 'p':
               if (var->paran[ig15] <= paran) {
                  val.type = var->type[ig15];
                  break;
               }
            case 'P':
               if (std::islower(var->type[ig15])) {
                  val.type = std::toupper(var->type[ig15]);
               }
               else {
                  val.type = var->type[ig15];
                  switch (var->reftype[ig15]) {
                     case G__PARANORMAL:
                        val.obj.reftype.reftype = G__PARAP2P;
                        break;
                     case G__PARAREFERENCE:
                        val.obj.reftype.reftype = var->reftype[ig15];
                        break;
                     default:
                        val.obj.reftype.reftype = var->reftype[ig15] + 1;
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
         if (G__struct_offset) {
            // --
#ifdef G__NEWINHERIT
            if (G__struct_offset != store_struct_offset) {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: ADDSTROS %d  %s:%d\n", G__asm_cp, G__asm_dt, G__struct_offset - store_struct_offset, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__ADDSTROS;
               G__asm_inst[G__asm_cp+1] = G__struct_offset - store_struct_offset;
               G__inc_cp_asm(2, 0);
            }
#endif // G__NEWINHERIT
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: LD_MSTR  item: '%s' index: %d paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, item, ig15, paran, G__var_type, (long) var, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__LD_MSTR;
            G__asm_inst[G__asm_cp+1] = ig15;
            G__asm_inst[G__asm_cp+2] = paran;
            G__asm_inst[G__asm_cp+3] = G__var_type;
            G__asm_inst[G__asm_cp+4] = (long) var;
            G__inc_cp_asm(5, 0);
#ifdef G__NEWINHERIT
            if (G__struct_offset != store_struct_offset) {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: ADDSTROS %d  %s:%d\n", G__asm_cp, G__asm_dt, -G__struct_offset + store_struct_offset, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__ADDSTROS;
               G__asm_inst[G__asm_cp+1] = -G__struct_offset + store_struct_offset;
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
               (store_struct_offset == G__ASM_VARLOCAL) &&
               (var->statictype[ig15] != G__LOCALSTATIC)
            ) {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD_LVAR item: '%s' index: %d paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, item, ig15, paran, G__var_type, (long) var, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD_LVAR;
            }
            else {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD_VAR item: '%s' index: %d paran: %d type: '%c' var: %08lx   %s:%d\n", G__asm_cp, G__asm_dt, item, ig15, paran, G__var_type, (long) var, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD_VAR;
            }
#else // G__ASM_WHOLEFUNC
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: LD_VAR item: '%s' index: %d paran: %d type: '%c' var: %08lx  %s:%d\n", G__asm_cp, G__asm_dt, item, ig15, paran, G__var_type, (long) var, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__LD_VAR;
#endif // G__ASM_WHOLEFUNC
            G__asm_inst[G__asm_cp+1] = ig15;
            G__asm_inst[G__asm_cp+2] = paran;
            G__asm_inst[G__asm_cp+3] = G__var_type;
            G__asm_inst[G__asm_cp+4] = (long) var;
            G__inc_cp_asm(5, 0);
         }
      }
      if (
         G__no_exec_compile &&
         (
            (var->constvar[ig15] != G__CONSTVAR) ||
            std::isupper(var->type[ig15]) ||
            (var->reftype[ig15] == G__PARAREFERENCE) ||
            var->varlabel[ig15][1] /* number of elements */ ||
            (var->p[ig15] < 0x1000)
         ) &&
         (std::tolower(var->type[ig15]) != 'p')
      ) {
         *known = 1;
         result.obj.d = 0.0;
         switch (var->type[ig15]) {
            case 'd':
            case 'f':
               break;
            case 'T':
               result.type = 'C';
               break;
#ifdef G__ROOT
            case 'Z':
               // root special
               if (G__GetSpecialObject) {
                  store_var_type = G__var_type;
                  result = (*G__GetSpecialObject)(var->varnamebuf[ig15], (void**)var->p[ig15], (void**)(var->p[ig15] + G__LONGALLOC));
                  // G__var_type was stored in store_var_type just before the
                  // call to G__GetSpecialObject which might have recursive
                  // calls to G__getvariable() or G__getexpr()
                  // It is restored at this point.
                  G__var_type = store_var_type;
                  if (!result.obj.i) {
                     *known = 0;
                  }
                  else {
                     var->p_tagtable[ig15] = result.tagnum;
                  }
                  switch (G__var_type) {
                  case 'p':
                     break;
                  case 'v':
                     result.ref = result.obj.i;
                     result.type = tolower(result.type);
                     break;
                  default:
                     G__reference_error(item);
                     break;
                  }
               }
               break;
#endif // G__ROOT

            default:
               result.obj.i = 1;
               break;
         }
         G__returnvartype(&result, var, ig15, paran);
         if (std::isupper(var->type[ig15])) {
            long dmy = 0;
            result.ref = (long) &dmy;
            G__getpointer2pointer(&result, var, ig15, paran);
         }
         result.tagnum = var->p_tagtable[ig15];
         result.typenum = var->p_typetable[ig15];
         result.ref = G__struct_offset + var->p[ig15];
         G__var_type = 'p';
         if (tolower(var->type[ig15]) == 'u') {
            int varparan = var->paran[ig15];
            if (var->type[ig15] == 'U') {
               ++varparan;
            }
            if (var->reftype[ig15] > G__PARAREFERENCE) {
               varparan += (var->reftype[ig15] % G__PARAREF) - G__PARAP2P + 1;
            }
            int ig25 = paran;
            if (paran > varparan)
               ig25 = varparan;
            while ((ig25 < paran) && var->varlabel[ig15][ig25+4]) {
               ++ig25;
            }
            if (ig25 < paran) {
               G__tryindexopr(&result, para, paran, ig25);
            }
         }
         return result;
      }
      G__exec_asm_getvar:
#endif // G__ASM
      // Static class/struct member.
      if (
         G__struct_offset &&
         (
            (var->statictype[ig15] == G__LOCALSTATIC) ||
            (
               ((var->tagnum == -1) && (var->statictype[ig15] == G__COMPILEDGLOBAL))
               
               || ((var->tagnum != -1) && (G__struct.type[var->tagnum] == 'n'))
            )
         )
      ) {
         G__struct_offset = 0;
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
       * v[a][b][c][d]
       *
       * tmp = B*C*D
       * linear_index = a*(B*C*D) + b*(C*D) + c*D + d
       * secondary_linear_index =
       *************************************************/
      size_t linear_index = 0;
      {
         int tmp = var->varlabel[ig15][0] /* stride */;
         for (ig25 = 0; (ig25 < paran) && (ig25 < var->paran[ig15]); ++ig25) {
            linear_index += tmp * G__int(para[ig25]);
            tmp /= var->varlabel[ig15][ig25+2];
         }
      }
      size_t secondary_linear_index = 0;
      {
         // -- Calculate secondary_linear_index.
         size_t tmp = var->varlabel[ig15][ig25+3];
         if (!tmp) {
            // questionable
            tmp = 1;
         }
         while ((ig25 < paran) && var->varlabel[ig15][ig25+4]) {
            secondary_linear_index += tmp * G__int(para[ig25]);
            tmp /= var->varlabel[ig15][ig25+4];
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
         var->varlabel[ig15][1] /* number of elements */ &&
         (var->reftype[ig15] == G__PARANORMAL) &&
         (
            // We intentionally allow going one beyond the end.
            (linear_index > var->varlabel[ig15][1] /* number of elements */) ||
            ((ig25 < paran) && (std::tolower(var->type[ig15]) != 'u'))
         )
      ) {
         G__arrayindexerror(ig15, var, item, linear_index);
         *known = 1;
         return G__null;
      }
      // Return struct and typedef information.
      // FIXME: This is wrong because, e.g., a[i] gives an array result type!
      result.tagnum = var->p_tagtable[ig15];
      result.typenum = var->p_typetable[ig15];
      result.ref = 0;
      *known = 1;
#ifdef G__SECURITY
      if (
            !G__no_exec_compile &&
            (G__var_type == 'v') &&
            std::isupper(var->type[ig15]) &&
            (var->reftype[ig15] == G__PARANORMAL) &&
            !var->varlabel[ig15][1] /* number of elements */ &&
            ((*((long*)(G__struct_offset + var->p[ig15]))) == 0)
      ) {
         G__reference_error(item);
         return G__null;
      }
#endif // G__SECURITY
      G__CHECK(G__SECURE_POINTER_AS_ARRAY, (var->paran[ig15] < paran && std::isupper(var->type[ig15])), return(G__null));
      G__CHECK(G__SECURE_POINTER_REFERENCE, ((isupper(var->type[ig15]) && (var->type[ig15] != 'E')) || (var->paran[ig15] > paran)), return(G__null));
      // Get bit-field value.
      if (var->bitfield[ig15] && (G__var_type == 'p')) {
         long address = G__struct_offset + var->p[ig15];
         int original = *((int*) address);
         int mask = (1 << var->bitfield[ig15]) - 1;
         int finalval = (original >> var->varlabel[ig15][G__MAXVARDIM-1]) & mask;
         G__letint(&result, var->type[ig15], finalval);
         return result;
      }
      if (G__decl && G__getarraydim && !G__struct_offset && (var->p[ig15] < 100)) {
         // prevent segv in following example. A bit tricky.
         //  void f(const int n) { int a[n]; }
         G__abortbytecode();
         return result;
      }
      switch (var->type[ig15]) {
         case 'i':
            // int
            G__GET_VAR(G__INTALLOC, int, G__letint, 'i', 'I')
         case 'd':
            // double
            G__GET_VAR(G__DOUBLEALLOC, double, G__letdouble, 'd', 'D')
         case 'c':
            // char
            G__GET_VAR(G__CHARALLOC, char, G__letint, 'c', 'C')
         case 'b':
            // unsigned char
            G__GET_VAR(G__CHARALLOC, unsigned char, G__letint, 'b', 'B')
         case 's':
            // short int
            G__GET_VAR(G__SHORTALLOC, short, G__letint, 's', 'S')
         case 'r':
            // unsigned short int
            G__GET_VAR(G__SHORTALLOC, unsigned short, G__letint, 'r', 'R')
         case 'h':
            // unsigned int
            G__GET_VAR(G__INTALLOC, unsigned int, G__letint, 'h', 'H')
         case 'l':
            // long int
            G__GET_VAR(G__LONGALLOC, long, G__letint, 'l', 'L')
         case 'k':
            // unsigned long int
            G__GET_VAR(G__LONGALLOC, unsigned long, G__letint, 'k', 'K')
         case 'f':
            // float
            G__GET_VAR(G__FLOATALLOC, float, G__letdouble, 'f', 'F')
         case 'n':
            // long long
            G__GET_VAR(G__LONGLONGALLOC, G__int64, G__letLonglong, 'n', 'N')
         case 'm':
            // unsigned long long
            G__GET_VAR(G__LONGLONGALLOC, G__uint64, G__letULonglong, 'm', 'M')
         case 'q':
            // long double
            G__GET_VAR(G__LONGDOUBLEALLOC, long double, G__letLongdouble, 'q', 'Q')
         case 'g':
            // bool
#ifdef G__BOOL4BYTE
            G__GET_VAR(G__INTALLOC, int, G__letbool, 'g', 'G')
#else // G__BOOL4BYTE
            G__GET_VAR(G__CHARALLOC, unsigned char, G__letint, 'g', 'G')
#endif // G__BOOL4BYTE
            // --
#ifndef G__OLDIMPLEMENTATION2191
         case '1':
            // void pointer
            G__GET_PVAR(char, G__letint, long, tolower(var->type[ig15]), var->type[ig15])
            //G__get_pvar<char, long>(G__letint, (char) tolower(var->type[ig15]), (char) var->type[ig15], var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
#else // G__OLDIMPLEMENTATION2191
         case 'Q':
            // void pointer
            G__GET_PVAR(char, G__letint, long, tolower(var->type[ig15]), var->type[ig15])
            //G__get_pvar<char, long>(G__letint, (char) tolower(var->type[ig15]), (char) var->type[ig15], var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
#endif // G__OLDIMPLEMENTATION2191
         case 'Y':
            // void pointer
            G__GET_PVAR(char, G__letint, long, tolower(var->type[ig15]), var->type[ig15])
            //G__get_pvar<char, long>(G__letint, (char) tolower(var->type[ig15]), (char) var->type[ig15], var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'E':
            // FILE pointer
            G__GET_PVAR(char, G__letint, long, tolower(var->type[ig15]), var->type[ig15])
            //G__get_pvar<char, long>(G__letint, (char) tolower(var->type[ig15]), (char) var->type[ig15], var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'C':
            // char pointer
            G__GET_PVAR(char, G__letint, long, tolower(var->type[ig15]), var->type[ig15])
            //G__get_pvar<char, long>(G__letint, (char) tolower(var->type[ig15]), (char) var->type[ig15], var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'N':
            // long long pointer
            G__GET_PVAR(G__int64, G__letLonglong, G__int64, tolower(var->type[ig15]), var->type[ig15])
            //G__get_pvar<G__int64, G__int64>(G__letLonglong, (char) tolower(var->type[ig15]), (char) var->type[ig15], var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'M':
            // unsigned long long pointer
            G__GET_PVAR(G__uint64, G__letULonglong, G__uint64, tolower(var->type[ig15]), var->type[ig15])
            //G__get_pvar<G__uint64, G__uint64>(G__letULonglong, (char) tolower(var->type[ig15]), (char) var->type[ig15], var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
#ifndef G__OLDIMPLEMENTATION2191
         case 'Q':
            // long double pointer
            G__GET_PVAR(long double, G__letLongdouble, long, tolower(var->type[ig15]), var->type[ig15])
            //G__get_pvar<long double, long>(G__letLongdouble, (char) tolower(var->type[ig15]), (char) var->type[ig15], var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
#endif // G__OLDIMPLEMENTATION2191
         case 'G':
            // bool pointer
            //G__GET_PVAR(unsigned char, G__letint, long, 'g', 'G')
            G__get_pvar<unsigned char, long>(G__letint, 'g', 'G', var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'B':
            // unsigned char pointer
            G__GET_PVAR(unsigned char, G__letint, long, 'b', 'B')
            //G__get_pvar<unsigned char, long>(G__letint, 'b', 'B', var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'S':
            // short int pointer
            G__GET_PVAR(short, G__letint, long, 's', 'S')
            //G__get_pvar<short, long>(G__letint, 's', 'S', var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'R':
            // unsigned short int pointer
            G__GET_PVAR(unsigned short, G__letint, long, 'r', 'R')
            //G__get_pvar<unsigned short, long>(G__letint, 'r', 'R', var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'I':
            // int pointer
            G__GET_PVAR(int, G__letint, long, 'i', 'I')
            //G__get_pvar<int, long>(G__letint, 'i', 'I', var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'H':
            // unsigned int pointer
            G__GET_PVAR(unsigned int, G__letint, long, 'h', 'H')
            //G__get_pvar<unsigned int, long>(G__letint, 'h', 'H', var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'L':
            // long int pointer
            G__GET_PVAR(long, G__letint, long, 'l', 'L')
            //G__get_pvar<long, long>(G__letint, 'l', 'L', var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'K':
            // unsigned long int pointer
            G__GET_PVAR(unsigned long, G__letint, long, 'k', 'K')
            //G__get_pvar<unsigned long, long>(G__letint, 'k', 'K', var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'F':
            // float pointer
            G__GET_PVAR(float, G__letdouble, double, 'f', 'F')
            //G__get_pvar<float, double>(G__letdouble, 'f', 'F', var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'D':
            // double pointer
            G__GET_PVAR(double, G__letdouble, double, 'd', 'D')
            //G__get_pvar<double, double>(G__letdouble, 'd', 'D', var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         case 'u':
            // class, enum, struct, union
            //--
            //--
            //--
            //--
            //FIXME: Replace with macro!
            switch (G__var_type) {
               case 'p':
                  // return value
                  if (var->paran[ig15] <= paran) {
                     // value, but return pointer
                     result.ref = (long)(G__struct_offset + (var->p[ig15]) + linear_index * G__struct.size[var->p_tagtable[ig15]]);
                     G__letint(&result, 'u', result.ref);
                  }
                  else {
                     // array, pointer
                     G__letint(&result, 'U', (long)(G__struct_offset + var->p[ig15] + (linear_index * G__struct.size[var->p_tagtable[ig15]])));
                     if ((var->paran[ig15] - paran) > 1) {
                        result.obj.reftype.reftype = var->paran[ig15] - paran;
                     }
                  }
                  break;
               case 'P':
                  // return pointer
                  G__letint(&result, 'U', (long) (G__struct_offset + var->p[ig15] + (linear_index * G__struct.size[var->p_tagtable[ig15]])));
                  break;
               default :
                  // return value
                  if (G__var_type == 'v') {
                     long store_struct_offsetX = G__store_struct_offset;
                     int store_tagnumX = G__tagnum;
                     int done = 0;
                     int store_asm_exec = G__asm_exec;
                     int store_asm_noverflow = G__asm_noverflow;
                     G__asm_exec = 0;
                     G__asm_noverflow = 0;
                     G__store_struct_offset = (long) (G__struct_offset + var->p[ig15] + (linear_index * G__struct.size[var->p_tagtable[ig15]]));
                     G__tagnum = var->p_tagtable[ig15];
                     G__FastAllocString refopr("operator*()");
                     result = G__getfunction(refopr, &done, G__TRYMEMFUNC);
                     G__asm_exec = store_asm_exec;
                     G__asm_noverflow = store_asm_noverflow;
                     G__tagnum = store_tagnumX;
                     G__store_struct_offset = store_struct_offsetX;
                     if (!done) {
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
            //FIXME: Put macro call back!
            switch (G__var_type) {
               case 'v': /* *var; get value */
                  // return value
                  switch (var->reftype[ig15]) {
                     case G__PARANORMAL:
                        result.ref = *(long*)(G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC));
                        if (result.ref) {
                           G__letint(&result, 'u', result.ref);
                        }
                        break;
                     case G__PARAP2P:
                        if (var->paran[ig15] < paran) {
                           long address = G__struct_offset + var->p[ig15] + linear_index * G__LONGALLOC;
                           result.ref = *(((long*)(*(long*)address)) + secondary_linear_index);
                           if (result.ref) {
                              G__letint(&result, 'u', result.ref);
                           }
                        }
                        else {
                           result.ref = *(long*)(G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC));
                           G__letint(&result, 'U', *(long*)(*(long*)(G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC))));
                        }
                        break;
                  }
                  break;
               case 'P':
                  if (var->paran[ig15] == paran) {
                     G__letint(&result, 'U', G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC));
                  }
                  else if (var->paran[ig15] < paran) {
                     long address = G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC);
                     if (G__PARANORMAL == var->reftype[ig15]) {
                        G__letint(&result, 'U', (*(long*)address) + (secondary_linear_index * G__struct.size[var->p_tagtable[ig15]]));
                     }
                     else {
                        G__letint(&result, 'U', (long)((long*)(*(long*)address) + secondary_linear_index));
                        result.obj.reftype.reftype = G__PARAP2P;
                     }
                  }
                  else {
                     G__letint(&result, 'U', G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC));
                  }
                  break;
               default:
                  // 'p'
                  if (var->paran[ig15] == paran) {
                     // type *p[];  (p[x]);
                     result.ref = (long) (G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC));
                     G__letint(&result, 'U', *(long*)(result.ref));
                  }
                  else if (var->paran[ig15] < paran) {
                     // type *p[];  p[x][y];
                     long address = G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC);
                     if (var->reftype[ig15] == G__PARANORMAL) {
                        result.ref = (*(long*)address) + (secondary_linear_index * G__struct.size[var->p_tagtable[ig15]]);
                        G__letint(&result, 'u', result.ref);
                     }
                     else if (var->paran[ig15] == (paran - 1)) {
                        result.ref = (long)((long*) (*(long*)address) + secondary_linear_index);
                        G__letint(&result, 'U', *((long*) result.ref));
                        if (var->reftype[ig15] > G__PARAP2P) {
                           result.obj.reftype.reftype = var->reftype[ig15] - 1;
                        }
                     }
                     else if (var->paran[ig15] == (paran - 2)) {
                        result.ref = (long) ((long*) (*(long*)address) + para[0].obj.i);
                        if (var->reftype[ig15] == G__PARAP2P) {
                           result.ref = (long) ((*((long*) result.ref)) + (para[1].obj.i * G__struct.size[var->p_tagtable[ig15]]));
                           G__letint(&result, 'u', result.ref);
                        }
                        else if (var->reftype[ig15] > G__PARAP2P) {
                           result.ref = (long) ((long*) (*(long*)result.ref) + para[1].obj.i);
                           G__letint(&result, 'U', *((long*)result.ref));
                           if (var->reftype[ig15] > G__PARAP2P2P) {
                              result.obj.reftype.reftype = var->reftype[ig15] - 2;
                           }
                        }
                        paran -= 1;
                     }
                     else if (var->paran[ig15] == (paran - 3)) {
                        result.ref = (long) ((long*) (*(long*)address) + para[0].obj.i);
                        result.ref = (long) ((long*) (*(long*)result.ref) + para[1].obj.i);
                        if (G__PARAP2P2P == var->reftype[ig15]) {
                           result.ref = (long) (*(long*)result.ref + (para[2].obj.i * G__struct.size[var->p_tagtable[ig15]]));
                           G__letint(&result, 'u', result.ref);
                        }
                        else if (G__PARAP2P2P < var->reftype[ig15]) {
                           result.ref = (long) ((long*) (*(long*)result.ref) + para[2].obj.i);
                           G__letint(&result, 'U', *((long*) result.ref));
                           if (G__PARAP2P2P < var->reftype[ig15]) {
                              result.obj.reftype.reftype = var->reftype[ig15] - 3;
                           }
                        }
                        paran -= 2;
                     }
                     else {
                        result.ref = (long) ((long*) (*(long*)address) + para[0].obj.i);
                        result.ref = (long) (*(long*)result.ref + (para[1].obj.i * G__struct.size[var->p_tagtable[ig15]]));
                        G__letint(&result, 'u', result.ref);
                        paran -= 2;
                     }
                  }
                  else {
                     // type *p[];  p;
                     result.ref = (long) (&var->p[ig15]);
                     G__letint(&result, 'U', G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC));
                  }
                  break;
            }
            //--
            //--
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
               G__strlcpy(G__ifile.name, G__macro, G__MAXFILENAME);
               fsetpos(G__ifile.fp, (fpos_t *) var->p[ig15]);
               G__nobreak = 1;
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
                  if (var->paran[ig15] <= paran) {
                     result.ref = G__struct_offset + var->p[ig15] + (linear_index * G__P2MFALLOC);
                     result.obj.i = result.ref;
                     result.type = 'a';
                     result.tagnum = -1;
                     result.typenum = -1;
                  }
                  else {
                     // array
                     G__letint(&result, 'A', G__struct_offset + var->p[ig15] + (linear_index * G__P2MFALLOC));
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
               result = (*G__GetSpecialObject)(var->varnamebuf[ig15], (void**)var->p[ig15], (void**)(var->p[ig15] + G__LONGALLOC));
               // G__var_type was stored in store_var_type just before the
               // call to G__GetSpecialObject which might have recursive
               // calls to G__getvariable() or G__getexpr()
               // It is restored at this point.
               G__var_type = store_var_type;
               if (!result.obj.i) {
                  *known = 0;
               }
               else {
                  var->p_tagtable[ig15] = result.tagnum;
               }
               switch (G__var_type) {
                  case 'p':
                     break;
                  case 'v':
                     result.ref = result.obj.i;
                     result.type = tolower(result.type);
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
            G__GET_PVAR(char, G__letint, long, tolower(var->type[ig15]), 'C')
            //G__get_pvar<char, long>(G__letint, (char) tolower(var->type[ig15]), 'C', var, ig15, G__struct_offset, paran, para, linear_index, secondary_linear_index, &result);
            break;
         default:
            // case 'X' automatic variable
            G__var_type = 'p';
            if (isupper(var->type[ig15])) {
               G__letdouble(&result, 'd', (double)(*(double *)(G__struct_offset + var->p[ig15] + (linear_index * G__DOUBLEALLOC))));
            }
            else {
               G__letint(&result, 'l', *(long*)(G__struct_offset + var->p[ig15] + (linear_index * G__LONGALLOC)));
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
            G__fprinterr(
                 G__serr
               , "%3x,%3x: LD func ptr '%s': 0x%08lx  %s:%d\n"
               , G__asm_cp
               , G__asm_dt
               , varname()
               , G__int(result)
               , __FILE__
               , __LINE__
            );
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
      isupper(var->type[ig15]) &&
      (var->type[ig15] != 'P') &&
      (var->type[ig15] != 'O')
   ) {
      // -- Handle pointer to pointer in value.
      G__getpointer2pointer(&result, var, ig15, paran);
   }
   // Return value for non-automatic variable.
   G__var_type = 'p';
   return result;
}

} // extern "C"

#undef G__GET_VAR
#undef G__GET_PVAR

//______________________________________________________________________________
G__value G__getstructmem(int store_var_type, G__FastAllocString& varname,
                         char* membername, int memnamesize, char* tagname,
                         int* known2, G__var_array* varglobal,
                         int objptr /* 1 : object , 2 : pointer */)
{
   // -- FIXME: Describe me!
   int store_tagnum = 0;
   long store_struct_offset = 0;
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
      varname = "&"; varname += membername; // Legacy, only add one character
      G__strlcpy(membername, varname, memnamesize - 1); // Legacy, only add one character
   }
   else if (store_var_type == 'v') {
      varname = "*"; varname += membername; // Legacy, only add one character
      G__strlcpy(membername, varname, memnamesize - 1); // Legacy, only add one character
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
      if (result.type) {
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
         result = G__getvariable(tagname , &flag , &G__global , G__p_local);
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
         result = G__getvariable(tagname , &flag, 0, G__struct.memvar[G__tagnum]);
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
         if (result.type) {
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
         result.obj.i = G__PVOID;
      }
   }
   G__store_struct_offset = result.obj.i;
   G__tagnum = result.tagnum;
#ifndef G__OLDIMPLEMENTATION1259
   G__isconst = result.isconst;
#endif // G__OLDIMPLEMENTATION1259
   if (
      (G__tagnum < 0) ||
      (
         isupper(result.type) &&
         (result.obj.reftype.reftype >= G__PARAP2P)
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
         G__fprinterr(G__serr, "Error: illegal pointer to class object %s 0x%lx %d ", tagname, G__store_struct_offset, G__tagnum);
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
      (result.type == 'u') &&
      (objptr == 2) &&
      (result.tagnum != -1) &&
      !strncmp(G__struct.name[result.tagnum], "auto_ptr<", 9)
   ) {
      int knownx = 0;
      G__FastAllocString comm("operator->()");
      result = G__getfunction(comm, &knownx, G__TRYMEMFUNC);
      if (knownx) {
         G__tagnum = result.tagnum;
         G__store_struct_offset = result.obj.i;
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
   if (islower(result.type) && (objptr == 2)) {
      char bufB[30] = "operator->()";
      int flagB = 0;
      int store_tagnumB = G__tagnum;
      long store_struct_offsetB = G__store_struct_offset;
      G__tagnum = result.tagnum;
      G__store_struct_offset = result.obj.i;
      result = G__getfunction(bufB, &flagB, G__TRYMEMFUNC);
      if (flagB) {
         G__tagnum = result.tagnum;
         G__store_struct_offset = result.obj.i;
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
               G__fprinterr(G__serr, "Warning: wrong member access operator '->'");
               G__printlinenum();
            }
         }
      }
   }
   if (isupper(result.type) && (objptr == 1)) {
      if (
         (G__dispmsg >= G__DISPROOTSTRICT) ||
         (G__ifile.filenum <= G__gettempfilenum())
      ) {
         if (G__dispmsg >= G__DISPWARN) {
            G__fprinterr(G__serr, "Warning: wrong member access operator '.'");
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
   result = G__getvariable(membername, known2, 0, G__struct.memvar[G__tagnum]);
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

extern "C" {

//______________________________________________________________________________
int G__getthis(G__value* result7, const char* varname, const char* item)
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
            G__letint(result7, 'u', G__store_struct_offset);
            result7->ref = G__store_struct_offset;
            break;
         case 'P':
            G__reference_error(item);
            break;
         case 'p':
         default:
            G__letint(result7, 'U', G__store_struct_offset);
            break;
      }
      // pointer to struct, class
      G__var_type = 'p';
      result7->typenum = G__typenum;
      result7->tagnum = G__tagnum;
      result7->ref = 0;
      result7->isconst = 0;
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
void G__returnvartype(G__value* presult, G__var_array* var, int ig15, int paran)
{
   // -- FIXME: Describe this function.
   //
   char var_type = var->type[ig15];
   presult->type = var_type;
   if (isupper(var_type)) {
      presult->obj.reftype.reftype = var->reftype[ig15];
   }
   switch (var_type) {
      case 'p':
      case 'x':
         presult->type = 'i';
         return;
      case 'P':
      case 'X':
         presult->type = 'd';
         return;
#ifndef G__OLDIMPLEMENTATION2191
      case 'j':
#else // G__OLDIMPLEMENTATION2191
      case 'm':
#endif // G__OLDIMPLEMENTATION2191
         G__abortbytecode();
         presult->type = 'i';
         return;
   }
   if (islower(var_type)) {
      switch (G__var_type) {
         case 'p':
            if (var->paran[ig15] <= paran) {
               presult->type = var_type;
            }
            else {
               presult->type = toupper(var_type);
            }
            break;
         case 'P':
            presult->type = toupper(var_type);
            break;
         default:
            // 'v'
            presult->type = var_type;
            break;
      }
   }
   else {
      switch (G__var_type) {
         case 'v':
            presult->type = tolower(var_type);
            break;
         case 'P':
            presult->type = toupper(var_type);
            break;
         default:
            // 'p'
            if (var->paran[ig15] == paran) {
               presult->type = var_type;
            }
            else if (var->paran[ig15] < paran) {
               int reftype = var->reftype[ig15];
               if (!reftype) {
                  reftype = 1;
               }
               int pointlevel = reftype - paran;
               if (!pointlevel) {
                  presult->type = tolower(var_type);
                  presult->obj.reftype.reftype = G__PARANORMAL;
               }
               else if (pointlevel == 1) {
                  presult->type = toupper(var_type);
                  presult->obj.reftype.reftype = G__PARANORMAL;
               }
               else {
                  presult->type = toupper(var_type);
                  presult->obj.reftype.reftype = pointlevel;
               }
            }
            else {
               presult->type = toupper(var_type);
            }
            break;
      }
   }
}

//______________________________________________________________________________
struct G__var_array* G__getvarentry(const char* varname, int varhash, int* pi, G__var_array* varglobal, G__var_array* varlocal)
{
   // -- FIXME: Describe me!
   struct G__var_array* var = 0;
   int ilg = 0;
   int in_memfunc = 0;
#ifdef G__NEWINHERIT
   int basen;
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
               var = varlocal;
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
               var = varglobal;
               ilg = G__NOTHING;
            }
            break;
         case G__MEMBER:
            in_memfunc = 1;
#ifdef G__OLDIMPLEMENTATION589_YET /* not activated due to bug */
            G__ASSERT(0 <= G__memberfunc_tagnum);
            G__incsetup_memvar(G__memberfunc_tagnum);
            var = G__struct.memvar[G__memberfunc_tagnum];
#else // G__OLDIMPLEMENTATION589_YET
            if (G__tagnum != -1) {
               G__incsetup_memvar(G__tagnum);
               var = G__struct.memvar[G__tagnum];
            }
            else {
               var = 0;
            }
#endif // G__OLDIMPLEMENTATION589_YET
            ilg = G__GLOBAL;
            break;
         case G__GLOBAL:
            in_memfunc = 0;
            var = varglobal;
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
         if (G__tagnum > -1) {
            baseclass = G__struct.baseclass[G__tagnum];
         }
#endif // G__OLDIMPLEMENTATION589_YET
         if (G__exec_memberfunc || G__isfriend(G__tagnum)) {
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
         for (; var; var = var->next) {
            for (int ig15 = 0; ig15 < var->allvar; ++ig15) {
               if (
                  (varhash == var->hash[ig15]) &&
                  !strcmp(varname, var->varnamebuf[ig15]) &&
                  (
                     (var->statictype[ig15] < 0) ||
                     G__filescopeaccess(G__ifile.filenum, var->statictype[ig15])
                  ) &&
                  (var->access[ig15] & accesslimit)
               ) {
                  *pi = ig15;
                  return var;
               }
            }
         }
         // Next base class if searching for class member.
         if (isbase) {
            while (baseclass && (basen < baseclass->basen)) {
               if (memfunc_or_friend) {
                  if (
                     (baseclass->herit[basen]->baseaccess & G__PUBLIC_PROTECTED) ||
                     (baseclass->herit[basen]->property & G__ISDIRECTINHERIT)
                  ) {
                     accesslimit = G__PUBLIC_PROTECTED;
                     G__incsetup_memvar(baseclass->herit[basen]->basetagnum);
                     var = G__struct.memvar[baseclass->herit[basen]->basetagnum];
                     ++basen;
                     goto next_base;
                  }
               }
               else {
                  if (baseclass->herit[basen]->baseaccess & G__PUBLIC) {
                     accesslimit = G__PUBLIC;
                     G__incsetup_memvar(baseclass->herit[basen]->basetagnum);
                     var = G__struct.memvar[baseclass->herit[basen]->basetagnum];
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
   return 0;
}

//______________________________________________________________________________
int G__filescopeaccess(int filenum, int statictype)
{
   // -- FIXME: Describe this function!
   int store_filenum = filenum;
   int store_statictype = statictype;
   if (filenum == statictype) {
      return 1;
   }
   while (statictype >= 0) {
      statictype = G__srcfile[statictype].included_from;
      if (filenum == statictype) {
         return 1;
      }
   }
   statictype = store_statictype;
   while (statictype >= 0) {
      filenum = store_filenum;
      if (filenum == statictype) {
         return 1;
      }
      statictype = G__srcfile[statictype].included_from;
      while (filenum >= 0) {
         if (filenum == statictype) {
            return 1;
         }
         filenum = G__srcfile[filenum].included_from;
      }
   }
   return 0;
}

#ifdef G__FRIEND
//______________________________________________________________________________
int G__isfriend(int tagnum)
{
   // -- FIXME: Describe me!
   struct G__friendtag* friendtag = 0;
   if (G__exec_memberfunc) {
      if (G__memberfunc_tagnum == tagnum) {
         return 1;
      }
      if (G__memberfunc_tagnum < 0) {
         return 0;
      }
      friendtag = G__struct.friendtag[G__memberfunc_tagnum];
      while (friendtag) {
         if (friendtag->tagnum == tagnum) {
            return 1;
         }
         friendtag = friendtag->next;
      }
   }
   if ((G__func_now != -1) && G__p_local && G__p_local->ifunc) {
      friendtag = G__get_ifunc_internal(G__p_local->ifunc)->friendtag[G__p_local->ifn];
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
//--  1
//--  2
   //--  3
   //--  4
   //--  5
   //--  6
//--  7

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

//______________________________________________________________________________
//
// Functions in the C api.
//

//______________________________________________________________________________
int G__IsInMacro()
{
   // -- FIXME: Describe me!
   if ((G__nfile > G__ifile.filenum) || (G__dispmsg >= G__DISPROOTSTRICT)) {
      return 0;
   }
   return 1;
}

//______________________________________________________________________________
struct G__var_array* G__searchvariable(char* varname, int varhash, G__var_array* varlocal, G__var_array* varglobal, long* pG__struct_offset, long* pstore_struct_offset, int* pig15, int isdecl)
{
   // -- FIXME: Describe me!
   struct G__var_array* var = 0;
   int ilg = 0;
   int in_memfunc = 0;
   long scope_struct_offset = 0L;
   int scope_tagnum = 0;
   int basen = 0;
   int isbase = 0;
   int accesslimit = 0;
   int memfunc_or_friend = 0;
   struct G__inheritance* baseclass = 0;
#ifdef G__ROOT
   int specialflag = 0;
#endif // G__ROOT
   int save_scope_tagnum = 0;
   //--
   //--
   //--
#ifdef G__ROOT
   if ((varname[0] == '$') && G__GetSpecialObject && (G__GetSpecialObject != G__getreserved)) {
      G__FastAllocString temp(varname + 1);
      // We copy less into varname than it contained before:
      // coverity[secure_coding]
      strcpy(varname, temp);
      specialflag = 1;
   }
#endif // __ROOT
   ilg = G__LOCAL;
   scope_struct_offset = G__store_struct_offset;
   G__ASSERT(!G__decl || (G__decl == 1));
   if (G__def_struct_member) {
      scope_tagnum = G__get_envtagnum();
   }
   else if (G__decl && G__exec_memberfunc && (G__memberfunc_tagnum != -1)) {
      scope_tagnum = G__memberfunc_tagnum;
   }
   else {
      scope_tagnum = G__tagnum;

#ifdef G__NOSTUBS
      // 25-02-08 ... unknown side effects.
      // We want to be able to specify the class before calling getexpr in newlink.
      // We need a way to trigger this from G__evaluate_libp() in newlink.cxx, and since
      // I don't want to create more global variables. I'm using G__dicttype, which
      // is only used when generating the dictionary (it shouldnt have any meaning
      // when running the code)
      if(G__dicttype == (G__dictgenmode)-1)
         ilg = G__MEMBER;
#endif
   }
   //--
   switch (G__scopeoperator(varname, &varhash, &scope_struct_offset, &scope_tagnum)) {
      case G__GLOBALSCOPE:
         ilg = G__GLOBAL;
         break;
      case G__CLASSSCOPE:
         ilg = G__MEMBER;
         break;
   }
   //--
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
               *pstore_struct_offset = G__ASM_VARLOCAL;
#endif // G__ASM_WHOLEFUNC
               var = varlocal;
               if (varglobal && !isdecl) {
                  if (G__exec_memberfunc || ((G__tagdefining != -1) && (scope_tagnum != -1))) {
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
            if (scope_tagnum != -1) {
               in_memfunc = 1;
               *pG__struct_offset = scope_struct_offset;
               G__incsetup_memvar(scope_tagnum);
               var = G__struct.memvar[scope_tagnum];
            }
            else {
               in_memfunc = 0;
               *pG__struct_offset = scope_struct_offset;
               var = 0;
            }
            ilg = G__GLOBAL;
            break;
         case G__GLOBAL:
            // -- Global entry.
            in_memfunc = 0;
            *pG__struct_offset = 0;
#ifdef G__ASM_WHOLEFUNC
            *pstore_struct_offset = G__ASM_VARGLOBAL;
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
         if (scope_tagnum > -1) {
            baseclass = G__struct.baseclass[scope_tagnum];
         }
         if (G__exec_memberfunc || isdecl || G__enumdef || G__isfriend(G__tagnum)) {
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
         for (; var; var = var->next) {
            for (int ig15 = 0; ig15 < var->allvar; ++ig15) {
               if (
                  (varhash == var->hash[ig15]) && // Name hash codes match, and
                  !strcmp(varname, var->varnamebuf[ig15]) && // Names match, and
                  (
                     (var->statictype[ig15] < 0) || // Not file scope, or
                     G__filescopeaccess(G__ifile.filenum, var->statictype[ig15]) // File scope access match, and
                  ) &&
                  (var->access[ig15] & accesslimit) // Access limit match
               ) {
                  if (pig15) {
                     *pig15 = ig15;
                  }
                  return var;
               }
            }
         }
         // Next base class if searching for class member.
         if (
            isbase &&
            (0 <= scope_tagnum) &&
            (G__struct.type[scope_tagnum] == 'e') &&
            (G__dispmsg >= G__DISPROOTSTRICT)
         ) {
            isbase = 0;
         }
         if (isbase) {
            while (baseclass && (basen < baseclass->basen)) {
               if (memfunc_or_friend) {
                  if (
                     (baseclass->herit[basen]->baseaccess & G__PUBLIC_PROTECTED) ||
                     (baseclass->herit[basen]->property & G__ISDIRECTINHERIT)
                  ) {
                     accesslimit = G__PUBLIC_PROTECTED;
                     G__incsetup_memvar(baseclass->herit[basen]->basetagnum);
                     var = G__struct.memvar[baseclass->herit[basen]->basetagnum];
#ifdef G__VIRTUALBASE
                     if (baseclass->herit[basen]->property&G__ISVIRTUALBASE) {
                        if (G__store_struct_offset!=0) {
                           *pG__struct_offset = *pstore_struct_offset + G__getvirtualbaseoffset(*pstore_struct_offset, scope_tagnum, baseclass, basen);
                        } else {
                           // We don't have a real object, we can't calculate the real offset.
                           // So do nothing ...
                        }
                     }
                     else {
                        *pG__struct_offset = *pstore_struct_offset + baseclass->herit[basen]->baseoffset;
                     }
#else // G__VIRTUALBASE
                     *pG__struct_offset = *pstore_struct_offset + baseclass->herit[basen]->baseoffset;
#endif // G__VIRTUALBASE
                     ++basen;
                     goto next_base;
                  }
               }
               else {
                  if (baseclass->herit[basen]->baseaccess & G__PUBLIC) {
                     accesslimit = G__PUBLIC;
                     G__incsetup_memvar(baseclass->herit[basen]->basetagnum);
                     var = G__struct.memvar[baseclass->herit[basen]->basetagnum];
#ifdef G__VIRTUALBASE
                     if (baseclass->herit[basen]->property & G__ISVIRTUALBASE) {
                        *pG__struct_offset = *pstore_struct_offset + G__getvirtualbaseoffset(*pstore_struct_offset, scope_tagnum, baseclass, basen);
                     }
                     else {
                        *pG__struct_offset = *pstore_struct_offset + baseclass->herit[basen]->baseoffset;
                     }
#else // G__VIRTUALBASE
                     *pG__struct_offset = *pstore_struct_offset + baseclass->herit[basen]->baseoffset;
#endif // G__VIRTUALBASE
                     ++basen;
                     goto next_base;
                  }
               }
               ++basen;
            }
            // Also search enclosing scopes.
            if (
               (scope_tagnum >= 0) &&
               (baseclass != &G__globalusingnamespace) &&
               (G__struct.parent_tagnum[scope_tagnum] != -1)
            ) {
               scope_tagnum = G__struct.parent_tagnum[scope_tagnum];
               basen = 0;
               baseclass = G__struct.baseclass[scope_tagnum];
               var = G__struct.memvar[scope_tagnum];
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
      struct G__var_array* store_local = G__p_local;
      G__p_local = 0;
      int store_var_type = G__var_type;
      G__var_type = 'Z';
      G__value para[1];
      //--
      G__DataMemberHandle member;
      G__allocvariable(G__null, para, varglobal, 0, 0, varhash, varname, varname, 0, member);
      G__var_type = store_var_type;
      G__p_local = store_local;
      var = member.GetVarArray();
      *pig15 = member.GetIndex();
      if (var) {
         G__gettingspecial = 0;
         return var;
      }
   }
#endif // G__ROOT
   return 0;
}

} // extern "C"

//______________________________________________________________________________
int G__DataMemberHandle::DeleteVariable()
{
   // -- Delete variable.  Return 1 if successful.
      
   struct G__var_array* var = GetVarArray();
   int ig15 = GetIndex();
   if (var) {
      int cpplink = G__NOLINK;
      int i;
      int done;
      switch (var->type[ig15]) {
         case 'u': {
            long store_struct_offset = G__store_struct_offset;
            int store_tagnum = G__tagnum;
            G__store_struct_offset = var->p[ig15];
            G__tagnum = var->p_tagtable[ig15];
            G__FastAllocString temp( strlen( var->varnamebuf[ig15]) + 4 );
            temp.Format("~%s()", var->varnamebuf[ig15]);
            // destruction of array
            if (G__struct.iscpplink[G__tagnum] == G__CPPLINK) {
               G__store_struct_offset = var->p[ig15];
               i = var->varlabel[ig15][1] /* number of elements */;
               if ((i > 0) || var->paran[ig15]) {
                  G__cpp_aryconstruct = i;
               }
               G__getfunction(temp, &done, G__TRYDESTRUCTOR);
               G__cpp_aryconstruct = 0;
               cpplink = G__CPPLINK;
            }
            else {
               int size = G__struct.size[G__tagnum];
               int i = var->varlabel[ig15][1] /* number of elements */;
               if (!i) {
                  i = 1;
               }
               --i;
               for (; i >= 0; --i) {
                  G__store_struct_offset = var->p[ig15] + (i * size);
                  if (G__dispsource) {
                     G__fprinterr(G__serr, "\n0x%lx.%s", G__store_struct_offset, temp.data());
                  }
                  done = 0;
                  G__getfunction(temp.data(), &done, G__TRYDESTRUCTOR);
                  if (!done) {
                     break;
                  }
               }
               G__tagnum = store_tagnum;
               G__store_struct_offset = store_struct_offset;
            }
            break;
         }
         default:
#ifdef G__SECURITY
            if (
                G__security & G__SECURE_GARBAGECOLLECTION &&
                !G__no_exec_compile &&
                isupper(var->type[ig15]) &&
                var->p[ig15]
                ) {
               long address;
               i = var->varlabel[ig15][1] /* number of elements */;
               if (!i) {
                  i = 1;
               }
               --i;
               for (; i >= 0; --i) {
                  address = var->p[ig15] + (i * G__LONGALLOC);
                  if (*((long*) address)) {
                     G__del_refcount((void*)(*((long*) address)), (void**) address);
                  }
               }
            }
#endif // G__SECURITY
            break;
      }
      if ((cpplink == G__NOLINK) && var->p[ig15]) {
         free((void*) var->p[ig15]);
      }
      var->p[ig15] = 0;
      var->varnamebuf[ig15][0] = '\0';
      var->hash[ig15] = 0;
      return 1;
   }
   return 0;
}

extern "C" {

//______________________________________________________________________________
int G__deletevariable(const char* varname)
{
   // -- Delete variable from global variable table.  Return 1 if successful.
   long struct_offset = 0;
   long store_struct_offset = 0;
   int ig15 = 0;
   int varhash = 0;
   int isdecl = 0;
   struct G__var_array* var = 0;
   G__hash(varname, varhash, ig15);
   var = G__searchvariable((char*)varname, varhash, 0, &G__global, &struct_offset, &store_struct_offset, &ig15, isdecl);
   if (var) {
      return G__DataMemberHandle(var, ig15).DeleteVariable();
   }
   return 0;
}

//______________________________________________________________________________
int G__deleteglobal(void* pin)
{
   // -- Delete variable from global variable table. return 1 if successful.
   long p = (long) pin;
   G__LockCriticalSection();
   struct G__var_array* var = &G__global;
   for (; var; var = var->next) {
      for (int ig15 = 0; ig15 < var->allvar; ++ig15) {
         if (p == var->p[ig15]) {
            var->p[ig15] = 0;
            var->varnamebuf[ig15][0] = '\0';
            var->hash[ig15] = 0;
         }
         if (isupper(var->type[ig15]) && var->p[ig15] && (p == (*(long*)var->p[ig15]))) {
            if (var->globalcomp[ig15] == G__AUTO) {
               free((void*) var->p[ig15]);
            }
            var->p[ig15] = 0;
            var->varnamebuf[ig15][0] = '\0';
            var->hash[ig15] = 0;
         }
      }
   }
   G__UnlockCriticalSection();
   return 0;
}

//______________________________________________________________________________
int G__resetglobalvar(void* pin)
{
   // Delete variable from global variable table if they are 'objects' and reset
   // to zero if they are pointers.
   // return 1 if successful.
   long p = (long) pin;
   G__LockCriticalSection();
   struct G__var_array* var = &G__global;
   for (; var; var = var->next) {
      for (int ig15 = 0; ig15 < var->allvar; ++ig15) {
         if (p == var->p[ig15]) {
            var->p[ig15] = 0;
            var->varnamebuf[ig15][0] = '\0';
            var->hash[ig15] = 0;
         }
         if (isupper(var->type[ig15]) && var->p[ig15] && (p == (*(long*)var->p[ig15]))) {
            // Only zero-out pointer, do not deleted the content nor wipe the definition.
            (*(long*)var->p[ig15]) = 0;
         }
      }
   }
   G__UnlockCriticalSection();
   return 0;
}

} // extern "C"

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
