/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file expr.c
 ************************************************************************
 * Description:
 *  Parse C/C++ expression
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "value.h"
#include "Dict.h"

using namespace Cint::Internal;
using namespace std;

#ifndef G__ROOT
#define G__NOPOWEROPR
#endif // G__ROOT


//
//  Function Directory.
//

static G__value G__getpower(char* expression2);
static int G__getoperator(int newoperator, int oldoperator);

//______________________________________________________________________________
//
//  Static functions.
//

//______________________________________________________________________________
static void G__getiparseobject(G__value* result, const char* item)
{
   // --
   // '_$trc_[tagnum]_[addr]'
   const char* xtmp = item + 6;
   const char* xx = strchr(xtmp, '_');
   int type = item[2];
   int reftype = (int) (item[3] - '0');
   int isconst = (int) (item[4] - '0');
   //
   char *strtagnum = new char[xx-xtmp+1];
   strncpy(strtagnum,xtmp,xx-xtmp);
   strtagnum[xx-xtmp] = '\0';

   int tagnum = atoi(strtagnum);

   delete [] strtagnum;

   ::Reflex::Type vtype = G__Dict::GetDict().GetType(tagnum);
   if (!vtype) {
      vtype = G__get_from_type(type, 0);
   }
   G__value_typenum(*result) = G__modify_type(vtype, 0, reftype, isconst, 0, 0);

   result->obj.i = atol(xx + 2);
   if (xx[1] == 'M') {
      result->obj.i = -result->obj.i;
   }
   result->ref = result->obj.i;
}

//______________________________________________________________________________
static G__value G__conditionaloperator(G__value defined, const char* expression, int ig1, char* ebuf)
{
   // -- Evaluate a?b:c operator.
   int tempop = 0;
   int ppointer = 0;
   int store_no_exec_compile = 0;
   // Evalulate the condition.
   tempop = G__int(defined);
   ++ig1;
   G__getstream(expression, &ig1, ebuf, ":");
#ifdef G__ASM
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: CNDJMP (? opr, condition test, assigned later)  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__CNDJMP;
      G__asm_inst[G__asm_cp+1] = 0; // filled in later
      ppointer = G__asm_cp + 1;
      G__inc_cp_asm(2, 0);
   }
#endif // G__ASM
   if (tempop) {
      // -- if a?b:c, where a is true
      // Evaluate the expression for the true case.
      defined = G__getexpr(ebuf);
      // Skip the expression for the false case.
      G__getstream(expression, &ig1, ebuf, ";");
#ifdef G__ASM
      if (G__asm_noverflow) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: JMP (? opr, out of true case, assigned later)  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            G__fprinterr(G__serr, "   %x: CNDJMP assigned %x (? opr, to false case) %s:%d\n", ppointer - 1, G__asm_cp + 2, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__JMP;
         G__asm_inst[G__asm_cp+1] = 0; // filled in later
         G__inc_cp_asm(2, 0);
         G__asm_inst[ppointer] = G__asm_cp;
         ppointer = G__asm_cp - 1;
         store_no_exec_compile = G__no_exec_compile;
         // Generate code for the false case but do not execute it.
         G__no_exec_compile = 1;
         G__getexpr(ebuf);
         G__no_exec_compile = store_no_exec_compile;
      }
#endif // G__ASM
      // --
   }
   else {
      // -- if a?b:c, where a is false
#ifdef G__ASM
      if (G__asm_noverflow) {
         // -- Generate bytecode.
         // Generate code for the true case but do not execute it.
         store_no_exec_compile = G__no_exec_compile;
         G__no_exec_compile = 1;
         G__getexpr(ebuf); /* eval true case */
         G__no_exec_compile = store_no_exec_compile;
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: JMP (? opr, out of false case, assigned later)  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            G__fprinterr(G__serr, "  %x: CNDJMP assigned %x (? opr, to false case)  %s:%d\n", ppointer - 1, G__asm_cp + 2, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__JMP;
         G__asm_inst[G__asm_cp+1] = 0; // filled in later
         G__inc_cp_asm(2, 0);
         G__asm_inst[ppointer] = G__asm_cp;
         ppointer = G__asm_cp - 1;
      }
#endif // G__ASM
      // Get the expression for the false case.
      G__getstream(expression, &ig1, ebuf, ";");
      // And evaluate it.
      defined = G__getexpr(ebuf);
   }
#ifdef G__ASM
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
      // Assign jump destination.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "  JMP assigned %x (at %x) (out of ? opr)  %s:%d\n" , G__asm_cp, ppointer - 1, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[ppointer] = G__asm_cp;
      G__asm_cond_cp = G__asm_cp; // avoid wrong optimization
   }
#endif // G__ASM
   return defined;
}

//______________________________________________________________________________
static int G__iscastexpr_body(char* ebuf, int lenbuf)
{
   // --
   int result;
   char* temp = (char*) malloc(strlen(ebuf) + 1);
   if (!temp) {
      G__genericerror("Internal error: malloc, G__iscastexpr_body(), temp");
      return 0;
   }
   strcpy(temp, ebuf + 1);
   temp[lenbuf-2] = 0;
   // Using G__istypename() is questionable.
   // May need to use G__string2type() for better language compliance.
   result = G__istypename(temp);
   free((void*) temp);
   return result;
}

#ifdef G__PTR2MEMFUNC
//______________________________________________________________________________
static int G__getpointer2memberfunc(const char* item, G__value* presult)
{
   const char* p = strstr(item, "::");
   if (!p) {
      return 0;
   }
   int hash = 0;
   char* scope_struct_offset = 0;
   int scope_tagnum = -1;
   G__scopeoperator(/*FIXME*/(char*)item, &hash, &scope_struct_offset, &scope_tagnum);
   if ((scope_tagnum < 0) || (scope_tagnum >= G__struct.alltag)) {
      return 0;
   }
   G__incsetup_memfunc(scope_tagnum);
   ::Reflex::Scope memfunc(G__Dict::GetDict().GetScope(scope_tagnum));
   //
   for (
      ::Reflex::Member_Iterator ifunc = memfunc.FunctionMember_Begin();
      ifunc != memfunc.FunctionMember_End();
      ++ifunc
   ) {
      if (ifunc->Name() == item) {
         // --
         // For the time being, pointer to member function can only be handled as function name.
         if (
            (memfunc.IsNamespace() || ifunc->IsStatic()) &&
            (G__get_funcproperties(*ifunc)->entry.size < 0) &&
            G__get_funcproperties(*ifunc)->entry.tp2f
         ) {
            G__letint(presult, 'Y', (long) G__get_funcproperties(*ifunc)->entry.tp2f);
         }
         else {
            G__letint(presult, 'C', (long) ifunc->Name().c_str());
         }
         presult->ref = 0;
         return 1;
      }
   }
   return 0;
}
#endif // G__PTR2MEMFUNC

//______________________________________________________________________________
static G__value G__getpower(char* expression2)
{
   G__value defined2, reg;
   G__StrBuf ebuf2_sb(G__ONELINE);
   char *ebuf2 = ebuf2_sb;
   int operator2 /*,c */;
   int lenbuf2 = 0;
   int ig12;
   /* int length2; */
   int nest2 = 0;
   int single_quote = 0, double_quote = 0;

   if (expression2[0] == '\0') return(G__null);

   operator2 = '\0';
   defined2 = G__null;
   /* length2=strlen(expression2); */
   /* if(length2==0) return(G__null); */
   ig12 = 0;
   while (expression2[ig12] != '\0') {
      switch (expression2[ig12]) {
         case '"' : /* double quote */
            if (single_quote == 0) {
               double_quote ^= 1;
            }
            ebuf2[lenbuf2++] = expression2[ig12];
            break;
         case '\'' : /* single quote */
            if (double_quote == 0) {
               single_quote ^= 1;
            }
            ebuf2[lenbuf2++] = expression2[ig12];
            break;
         case '~': /* 1's complement */
            /* explicit destructor handled in G__getexpr(), just go through here */
         case '@':
            if ((nest2 == 0) && (single_quote == 0) && (double_quote == 0)) {
               switch (lenbuf2) {
                  case 0:
                     operator2 = G__getoperator(operator2, expression2[ig12]);
                     break;
                  default:
                     ebuf2[lenbuf2] = '\0';
                     reg = G__getitem(ebuf2);
                     G__bstore(operator2, reg, &defined2);
                     lenbuf2 = 0;
                     ebuf2[0] = '\0';
                     operator2 = expression2[ig12];
                     break;
               }
            }
            else {
               ebuf2[lenbuf2++] = expression2[ig12];
            }
            break;
         case ' ':
            if ((nest2 == 0) && (single_quote == 0) && (double_quote == 0) &&
                  (strncmp(expression2, "new", 3) == 0)) {
               return(G__new_operator(expression2 + ig12 + 1));
            }
            else {
               G__fprinterr(G__serr, "Error: G__power() expression %s ", expression2);
               G__genericerror((char*)NULL);
               return(G__null);
            }
            /* break; */
         case '(':
         case '[':
         case '{':
            if ((double_quote == 0) && (single_quote == 0)) {
               nest2++;
               ebuf2[lenbuf2++] = expression2[ig12];
            }
            else {
               ebuf2[lenbuf2++] = expression2[ig12];
            }
            break;
         case ')':
         case ']':
         case '}':
            if ((double_quote == 0) && (single_quote == 0)) {
               ebuf2[lenbuf2++] = expression2[ig12];
               nest2--;
            }
            else {
               ebuf2[lenbuf2++] = expression2[ig12];
            }
            break;

         case '\\' :
            ebuf2[lenbuf2++] = expression2[ig12++];
            ebuf2[lenbuf2++] = expression2[ig12];
            break;

         default :
            ebuf2[lenbuf2++] = expression2[ig12];
            break;
      }
      ig12++;
   }
   ebuf2[lenbuf2] = '\0';
   if ((nest2 != 0) || (single_quote != 0) || (double_quote != 0)) {
      G__parenthesiserror(expression2, "G__getpower");
      return(G__null);
   }
   reg = G__getitem(ebuf2);
   G__bstore(operator2, reg, &defined2);
   return(defined2);
}

//______________________________________________________________________________
static int G__getoperator(int newoperator, int oldoperator)
{
   // --
   switch (newoperator) {
      case '+':
         switch (oldoperator) {
            case '+':
               return('I');
            case '-':
               return('-');
            case '~':
               return(G__UNARYOP);
            case '=':
               return(G__OPR_ADDASSIGN);
            default:
               return(oldoperator);
         }
         /* break; */
      case '-':
         switch (oldoperator) {
            case '+':
               return('-');
            case '-':
               return('D');
            case '~':
               return(G__UNARYOP);
            case '=':
               return(G__OPR_SUBASSIGN);
            default:
               return(oldoperator);
         }
         /* break; */
      case '>':
         switch (oldoperator) {
            case '>':
               return('R'); /* right shift */
            case '=':
               return('G'); /* greater or equal */
            case '~':
            case '+':
            case '-':
               return(G__UNARYOP);
            default:
               return(oldoperator);
         }
         /* break; */
      case 'R': /* right shift */
         switch (oldoperator) {
            case '~':
            case '+':
            case '-':
               return(G__UNARYOP);
            case '=':
               return(G__OPR_RSFTASSIGN);
            default:
               return(oldoperator);
         }
         /* break; */
      case '<':
         switch (oldoperator) {
            case '<':
               return('L'); /* left shift */
            case '=':
               return('l'); /* less or equal */
            case '~':
            case '+':
            case '-':
               return(G__UNARYOP);
            default:
               return(oldoperator);
         }
         /* break; */
      case 'L':
         switch (oldoperator) {
            case '~':
            case '+':
            case '-':
               return(G__UNARYOP);
            case '=':
               return(G__OPR_LSFTASSIGN);
            default:
               return(oldoperator);
         }
         /* break; */
      case '&':
         switch (oldoperator) {
            case '&':
               return('A');
            case '~':
            case '+':
            case '-':
               return(G__UNARYOP);
            case '=':
               return(G__OPR_BANDASSIGN);
            default:
               return(oldoperator);
         }
         /* break; */
      case '|':
         switch (oldoperator) {
            case '|':
               return('O');
            case '~':
            case '+':
            case '-':
               return(G__UNARYOP);
            case '=':
               return(G__OPR_BORASSIGN);
            default:
               return(oldoperator);
         }
         /* break; */
      case '^':
         switch (oldoperator) {
            case '~':
            case '+':
            case '-':
               return(G__UNARYOP);
            case '=':
               return(G__OPR_EXORASSIGN);
            default:
               return(oldoperator);
         }
         /* break; */
      case '%':
         switch (oldoperator) {
            case '~':
            case '+':
            case '-':
               return(G__UNARYOP);
            case '=':
               return(G__OPR_MODASSIGN);
            default:
               return(oldoperator);
         }
         /* break; */
      case '*':
         switch (oldoperator) {
            case '/':
               return('/');
            case '*':
               return('@');
            case '~':
            case '+':
            case '-':
               return(G__UNARYOP);
            case '=':
               return(G__OPR_MULASSIGN);
            default:
               return(newoperator);
         }
         /* break; */
      case '/':
         switch (oldoperator) {
            case '/':
               return('*');
            case '*':
               return('/');
            case '~':
            case '+':
            case '-':
               return(G__UNARYOP);
            case '=':
               return(G__OPR_DIVASSIGN);
            default:
               return(newoperator);
         }
         /* break; */
      case '=':
         switch (oldoperator) {
            case '=':
               return('E');
            default:
               return(newoperator);
         }
         /* break; */
      case '!':
         switch (oldoperator) {
            case '=':
               return('N');
            default:
               return(newoperator);
         }
         /* break; */
      case 'E':
      case 'N':
      case 'G':
      case 'l':
         switch (oldoperator) {
            case '~':
            case '+':
            case '-':
               return(G__UNARYOP);
            default:
               return(newoperator);
         }
      case 'A':
         switch (oldoperator) {
            case '=':
               return(G__OPR_ANDASSIGN);
            default:
               return(newoperator);
         }
      case 'O':
         switch (oldoperator) {
            case '=':
               return(G__OPR_ORASSIGN);
            default:
               return(newoperator);
         }
   }
   return oldoperator;
}

//______________________________________________________________________________
//
//  Cint internal functions.
//

//______________________________________________________________________________
G__value Cint::Internal::G__calc_internal(char* exprwithspace)
{
   // Parse and evaluate an expression.
   //
   // Note: This function is open to public as CINT API.
   //
#ifdef G__EH_SIGNAL
   void(*fpe)();
   void(*segv)();
#ifdef SIGILL
   void(*ill)();
#endif // SIGILL
#ifdef SIGEMT
   void(*emt)();
#endif // SIGEMT
#ifdef SIGBUS
   void(*bus)();
#endif // SIGBUS
#endif // G__EH_SIGNAL
   char* exprnospace = (char*) malloc(strlen(exprwithspace) + 2);
   int iin = 0, iout = 0, ipunct = 0;
   int single_quote = 0, double_quote = 0;
   G__value result;
   int len = 0;
   int store_asm_exec = G__asm_exec;
   int store_asm_noverflow = G__asm_noverflow;
   G__asm_noverflow = 0;
   G__asm_exec = 0;
   exprnospace[0] = '\0';

   bool isdelete = false;
   bool isdeletearr = false;

   while (exprwithspace[iin] != '\0') {
      switch (exprwithspace[iin]) {
         case '"' : /* double quote */
            if (single_quote == 0) {
               double_quote ^= 1;
            }
            exprnospace[iout++] = exprwithspace[iin++] ;
            break;
         case '\'' : /* single quote */
            if (double_quote == 0) {
               single_quote ^= 1;
            }
            exprnospace[iout++] = exprwithspace[iin++] ;
            break;
         case '\n': /* end of line */
         case '\r': /* end of line */
         case ';' : /* semi-column */
         case ' ' : /* space */
         case '\t' : /* tab */
            exprnospace[iout] = '\0'; /* temporarily terminate string */
            len = strlen(exprnospace);
            if ((single_quote != 0) || (double_quote != 0)
                  || (len >= 5 + ipunct && strncmp(exprnospace + ipunct, "const", 5) == 0)
               ) {
               
               exprnospace[iout++] = exprwithspace[iin] ;
            } else if (len >= 3 + ipunct && strncmp(exprnospace + ipunct, "new", 3) == 0) {
               exprnospace[iout++] = exprwithspace[iin] ;
            } else if (len >= 8 && strncmp(exprnospace, "delete[]", 8) == 0) {
               iout = 0;
               ipunct = 0;
               isdeletearr = true;
            }
            else if (len >= 6 && strncmp(exprnospace, "delete", 6) == 0) {
               iout = 0;
               ipunct = 0;
               isdelete = true;
            }
            iin++;
            break;
         case '=':
         case '(':
         case ')':
         case ',':
         case '<':
            ipunct = iout + 1;
         default :
            exprnospace[iout++] = exprwithspace[iin++] ;
            break;
      }
   }
   exprnospace[iout++] = '\0';
   if (isdelete) {
      if (exprnospace[0] == '[') {
         G__delete_operator(exprnospace + 2, 1);
      }
      else {
         G__delete_operator(exprnospace, 0);
      }
      result = G__null;
   } else if (isdeletearr) {

      G__delete_operator(exprnospace, 1);
      result = G__null;
   } else {
#ifdef G__EH_SIGNAL
      fpe = signal(SIGFPE, G__error_handle);
      segv = signal(SIGSEGV, G__error_handle);
#ifdef SIGILL
      ill = signal(SIGILL, G__error_handle);
#endif // SIGILL
#ifdef SIGEMT
      emt = signal(SIGEMT, G__error_handle);
#endif // SIGEMT
#ifdef SIGBUS
      bus = signal(SIGBUS, G__error_handle);
#endif // SIGBUS
#endif // G__EH_SIGNAL
      result = G__getexpr(exprnospace);
      G__last_error = G__security_error;
#ifdef G__EH_SIGNAL
      signal(SIGFPE, fpe);
      signal(SIGSEGV, segv);
#ifdef SIGILL
      signal(SIGILL, ill);
#endif // SIGILL
#ifdef SIGEMT
      signal(SIGEMT, emt);
#endif // SIGEMT
#ifdef SIGBUS
      signal(SIGBUS, bus);
#endif // SIGBUS
#endif // G__EH_SIGNAL

   }

   // --
   G__asm_exec = store_asm_exec;
   G__asm_noverflow = store_asm_noverflow;
   free(exprnospace);
   return result;
}

//______________________________________________________________________________
//
//  ANSI compliant operator precedences,  smaller values have higher precedence
//

#define G__PREC_SCOPE     1 // unused
#define G__PREC_FCALL     2 // unused
#define G__PREC_UNARY     3 // unused
#define G__PREC_P2MEM     4 // unused
#define G__PREC_PWR       5 // a@b, a**b
#define G__PREC_MULT      6 // a*b, a/b, a%b
#define G__PREC_ADD       7 // a+b, a-b
#define G__PREC_SHIFT     8 // a<<b, a>>b
#define G__PREC_RELATION  9 // a<=b, a<b, a>b, a>=b
#define G__PREC_EQUAL    10 // a==b, a!=b
#define G__PREC_BITAND   11 // a&b
#define G__PREC_BITEXOR  12 // a^b
#define G__PREC_BITOR    13 // a|b
#define G__PREC_LOGICAND 14 // a&&b
#define G__PREC_LOGICOR  15 // a||b
#define G__PREC_TEST     16 // unused
#define G__PREC_ASSIGN   17 // unused
#define G__PREC_COMMA    18 // unused

//______________________________________________________________________________
struct G__expr_parse_state {
   static const int G__STACKDEPTH = 100;
   //
   G__value vstack[G__STACKDEPTH]; // evaluated value stack
   int sp; // evaluated value stack pointer
   int unaopr[G__STACKDEPTH]; // unary operator stack
   int up; // unary operator stack pointer
   int opr[G__STACKDEPTH]; // binary and ternary operator stack
   int op; // binary and ternary operator stack pointer
   int prec[G__STACKDEPTH]; // operator precedence stack
   //
   int ppointer_and[G__STACKDEPTH];
   int pp_and;
   int ppointer_or[G__STACKDEPTH];
   int pp_or;
   //
   int store_no_exec_compile_and[G__STACKDEPTH];
   int store_no_exec_compile_or[G__STACKDEPTH];
   G__value vtmp_and;
   G__value vtmp_or;
};

//______________________________________________________________________________
static void G__exec_evalall(G__expr_parse_state* parse_state)
{
   // Process unary operator.
   while (parse_state->up && (parse_state->sp >= 1)) {
      --parse_state->up;
      if (parse_state->unaopr[parse_state->up] == '*') {
         parse_state->vstack[parse_state->sp-1] = G__tovalue(parse_state->vstack[parse_state->sp-1]);
      }
      else if(parse_state->unaopr[parse_state->up] == '&') {
         parse_state->vstack[parse_state->sp-1] = G__toXvalue(parse_state->vstack[parse_state->sp-1], 'P');
      }
      else {
         parse_state->vstack[parse_state->sp] = parse_state->vstack[parse_state->sp-1];
         parse_state->vstack[parse_state->sp-1] = G__null;
         G__bstore(parse_state->unaopr[parse_state->up], parse_state->vstack[parse_state->sp], &parse_state->vstack[parse_state->sp-1]);
      }
   }
   // Process binary operator.
   while (parse_state->op && (parse_state->sp >= 2)) {
      --parse_state->op;
      --parse_state->sp;
#ifdef G__ASM
      ///
      /// G__ASSIGN_CNDJMP
      ///
      if (G__asm_noverflow && (parse_state->opr[parse_state->op] == 'O')) {
         int store_pp_and = parse_state->pp_and;
         while (parse_state->pp_and) {
            --parse_state->pp_and;
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "   CNDJMP assigned %3x&%3x  %s:%d\n", G__asm_cp, parse_state->ppointer_and[parse_state->pp_and] - 1, __FILE__, __LINE__);
            }
            if ((char*) G__asm_inst[parse_state->ppointer_and[parse_state->pp_and]] == G__PVOID) {
               G__asm_inst[parse_state->ppointer_and[parse_state->pp_and]] = G__asm_cp;
            }
         }
         parse_state->pp_and = store_pp_and;
      }
      ///
      /// end G__ASSIGN_CNDJMP
      ///
#endif // G__ASM
      G__bstore(parse_state->opr[parse_state->op], parse_state->vstack[parse_state->sp], &parse_state->vstack[parse_state->sp-1]);
   }
}

//______________________________________________________________________________
static void G__exec_binopr(G__expr_parse_state* parse_state, int oprin, int precin)
{
   /* process unary operators */
   while (parse_state->up && (parse_state->sp >= 1)) {
      --parse_state->up;
      if (parse_state->unaopr[parse_state->up] == '*') {
         parse_state->vstack[parse_state->sp-1] = G__tovalue(parse_state->vstack[parse_state->sp-1]);
      }
      else if (parse_state->unaopr[parse_state->up] == '&') {
         parse_state->vstack[parse_state->sp-1] = G__toXvalue(parse_state->vstack[parse_state->sp-1], 'P');
      } else if ((parse_state->unaopr[parse_state->up] == '-') && (oprin == '@')) {
         parse_state->vstack[parse_state->sp] = parse_state->vstack[parse_state->sp-1];
         parse_state->vstack[parse_state->sp-1] = G__getitem("-1");
         parse_state->sp++;
         parse_state->opr[parse_state->op] = '*';
         parse_state->prec[parse_state->op++] = G__PREC_PWR;
      }
      else {
         parse_state->vstack[parse_state->sp] = parse_state->vstack[parse_state->sp-1];
         parse_state->vstack[parse_state->sp-1] = G__null;
         G__bstore(parse_state->unaopr[parse_state->up], parse_state->vstack[parse_state->sp], &parse_state->vstack[parse_state->sp-1]);
      }
   }
   /* process higher precedence binary operators */
   while (parse_state->op && (parse_state->prec[parse_state->op-1] <= precin) && (parse_state->sp >= 2)) {
      --parse_state->op;
      --parse_state->sp;
#ifdef G__ASM
      ///
      /// G__ASSIGN_CNDJMP
      ///
      if (G__asm_noverflow && (parse_state->opr[parse_state->op] == 'O')) {
         int store_pp_and = parse_state->pp_and;
         while (parse_state->pp_and) {
            --parse_state->pp_and;
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "   CNDJMP assigned %3x&%3x  %s:%d\n", G__asm_cp, parse_state->ppointer_and[parse_state->pp_and] - 1, __FILE__, __LINE__);
            }
            if ((char*) G__asm_inst[parse_state->ppointer_and[parse_state->pp_and]] == G__PVOID) {
               G__asm_inst[parse_state->ppointer_and[parse_state->pp_and]] = G__asm_cp;
            }
         }
         parse_state->pp_and = store_pp_and;
      }
      ///
      /// end G__ASSIGN_CNDJMP
      ///
#endif // G__ASM
      G__bstore(parse_state->opr[parse_state->op], parse_state->vstack[parse_state->sp], &parse_state->vstack[parse_state->sp-1]);
   }
   /* set operator */
   parse_state->opr[parse_state->op] = oprin;
   parse_state->prec[parse_state->op++] = precin;
}

//______________________________________________________________________________
G__value Cint::Internal::G__getexpr(const char* expression)
{
   // Parse and evaluate an expression.  Spaces must have been removed from the expression.
   //
   // Note:  We use an operator precedence parser.
   //
   //--
   //
   //fprintf(stderr, "Begin G__getexpr('%s') ...\n", expression);
   //
   //--
   //
   // Return null for no expression.
   //
   int length = strlen(expression);
   if (!length) {
      return G__null;
   }
   //
   int store_var_type = G__var_type;
   //
   G__expr_parse_state parse_state;
   parse_state.sp = 0; // evaluated value stack pointer
   parse_state.up = 0; // unary operator stack pointer
   parse_state.op = 0; // binary and ternary operator stack pointer
   parse_state.pp_and = 0;
   parse_state.pp_or = 0;
   //
   //
   //  Parse expression from left to right.
   //
   //  Note: terms are accumulated in ebuf
   //        character by character.
   //
   G__StrBuf ebuf_sb(length + 6); // term buffer storage
   char* ebuf = ebuf_sb; // term buffer ptr
   int lenbuf = 0; // ebuf offset
   int nest = 0; // parenthesis nesting state variable
   int single_quote = 0; // quotation flags
   int double_quote = 0; // quotation flags
   int iscastexpr = 0; // whether this expression start with a cast
   int explicitdtor = 0;
   int inew = 0;
   for (int ig1 = 0; ig1 < length; ++ig1) {
      int c = expression[ig1];
      switch (c) {
         case '\\' :
            ebuf[lenbuf++] = c;
            ebuf[lenbuf++] = expression[++ig1];
            break;
         case '"':
            ebuf[lenbuf++] = c;
            if (!single_quote) {
               double_quote ^= 1;
            }
            break;
         case '\'':
            ebuf[lenbuf++] = c;
            if (!double_quote) {
               single_quote ^= 1;
            }
            break;
         case '(': // new (arena) type(),  (type) val, (expr)
         case '[':
         case '{':
            if ((c == '(') && !nest && !single_quote && !double_quote && (lenbuf == 3) && !strncmp(expression + inew, "new", 3)) {
               return G__new_operator(expression + ig1);
            }
            ebuf[lenbuf++] = c;
            if (!double_quote && !single_quote) {
               ++nest;
               inew = ig1 + 1;
            }
            break;
         case ')':
         case ']':
         case '}':
            ebuf[lenbuf++] = c;
            if (!double_quote && !single_quote) {
               --nest;
               inew = ig1 + 1;
               if (!iscastexpr && (ebuf[0] == '(')) {
                  ebuf[lenbuf] = '\0';
                  iscastexpr = (lenbuf > 3) &&
                               (ebuf[0] == '(') &&
                               (ebuf[lenbuf-1] == ')') &&
                               (
                                  (ebuf[lenbuf-2] == '*') ||
                                  (ebuf[lenbuf-2] == '&') ||
                                  G__iscastexpr_body(ebuf, lenbuf)
                               );
               }
            }
            break;
         case ' ': // new type, new (arena) type
            if (nest || single_quote || double_quote) {
               ebuf[lenbuf++] = c;
               break;
            }
            if (((lenbuf - inew) == 3) && !strncmp(expression + inew, "new", 3)) {
               return G__new_operator(expression + ig1 + 1);
            }
            // else ignore c, shoud not happen, but not sure
            inew = ig1 + 1;
            break;
         case '!': // !a, a != b
         case '~': // ~a, a->~b(), a::~b(), a.~b()
            if (nest || single_quote || double_quote) {
               ebuf[lenbuf++] = c;
               break;
            }
            if ((c == '!') && (expression[ig1+1] == '=')) { // a != b
               ++ig1;
               if (!lenbuf) {
                  G__syntaxerror(expression);
                  return G__null;
               }
               // Evaluate item.
               ebuf[lenbuf] = '\0';
               parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
               lenbuf = 0;
               iscastexpr = 0;
               G__exec_binopr(&parse_state, G__OPR_NE, G__PREC_EQUAL);
               break;
            }
            if (!lenbuf) { // ~a, !a
               parse_state.unaopr[parse_state.up++] = c;
               break;
            }
            // a->~b(), a::~b(), a.~b()
            explicitdtor = 1;
            ebuf[lenbuf++] = c;
            break;
         case '/': // a/b, a/=b
            if (nest || single_quote || double_quote) {
               ebuf[lenbuf++] = c;
               break;
            }
            if (!lenbuf) {
               G__syntaxerror(expression);
               return G__null;
            }
            if (expression[ig1+1] == '=') { // a/=b
               ++ig1;
               // Evaluate item.
               if (lenbuf) {
                  ebuf[lenbuf] = '\0';
                  parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                  lenbuf = 0;
                  iscastexpr = 0;
               }
               G__exec_evalall(&parse_state);
               if ((parse_state.sp != 1) || parse_state.op || parse_state.up) {
                  G__syntaxerror(expression);
                  return G__null;
               }
               parse_state.vstack[1] = G__getexpr(expression + ig1 + 1);
               G__bstore(G__OPR_DIVASSIGN, parse_state.vstack[1], &parse_state.vstack[0]);
               G__var_type = 'p';
               return parse_state.vstack[0];
            }
            // a/b
            // Evaluate item.
            ebuf[lenbuf] = '\0';
            parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
            lenbuf = 0;
            iscastexpr = 0;
            G__exec_binopr(&parse_state, c, G__PREC_MULT);
            break;
         case '%': // a%b, a%=b
            if (nest || single_quote || double_quote) {
               ebuf[lenbuf++] = c;
               break;
            }
            if (!lenbuf) {
               G__syntaxerror(expression);
               return G__null;
            }
            if (expression[ig1+1] == '=') { // a%=b
               ++ig1;
               // Evaluate item.
               if (lenbuf) {
                  ebuf[lenbuf] = '\0';
                  parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                  lenbuf = 0;
                  iscastexpr = 0;
               }
               G__exec_evalall(&parse_state);
               if ((parse_state.sp != 1) || parse_state.op || parse_state.up) {
                  G__syntaxerror(expression);
                  return G__null;
               }
               parse_state.vstack[1] = G__getexpr(expression + ig1 + 1);
               G__bstore(G__OPR_MODASSIGN, parse_state.vstack[1], &parse_state.vstack[0]);
               G__var_type = 'p';
               return parse_state.vstack[0];
            }
            // a%b
            // Evaluate item.
            ebuf[lenbuf] = '\0';
            parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
            lenbuf = 0;
            iscastexpr = 0;
            G__exec_binopr(&parse_state, c, G__PREC_MULT);
            break;
         case '^': // a^b, a^=b
            if (nest || single_quote || double_quote) {
               ebuf[lenbuf++] = c;
               break;
            }
            if (!lenbuf) {
               G__syntaxerror(expression);
               return G__null;
            }
            if (expression[ig1+1] == '=') { // a^=b
               ++ig1;
               // Evaluate item.
               if (lenbuf) {
                  ebuf[lenbuf] = '\0';
                  parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                  lenbuf = 0;
                  iscastexpr = 0;
               }
               G__exec_evalall(&parse_state);
               if ((parse_state.sp != 1) || parse_state.op || parse_state.up) {
                  G__syntaxerror(expression);
                  return G__null;
               }
               parse_state.vstack[1] = G__getexpr(expression + ig1 + 1);
               G__bstore(G__OPR_EXORASSIGN, parse_state.vstack[1], &parse_state.vstack[0]);
               G__var_type = 'p';
               return parse_state.vstack[0];
            }
            // a^b
            // Evaluate item.
            ebuf[lenbuf] = '\0';
            parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
            lenbuf = 0;
            iscastexpr = 0;
            G__exec_binopr(&parse_state, c, G__PREC_BITEXOR);
            break;
         case '+': // ++a, a++, +a, a+b, a+=b, 1e+10, a + +b
         case '-': // --a, a--, -a, a-b, a-=b, 1e-10, a->b , a - -b
            if (nest || single_quote || double_quote) {
               ebuf[lenbuf++] = c;
               break;
            }
            if ( // ++a, b++a, --a, b--a
               (expression[ig1+1] == c) && // we have ++,--
               ( // ++a, b++a, --a, b--a
                  !lenbuf || // ++a, --a, or
                  (
                     !isdigit(ebuf[0]) && // not 10++a, not 10--a, and
                     (ebuf[0] != '.') // not .05++a, not 0.5--a
                  )
               )
            ) { // ++a, b++a, --a, b--a
               if (lenbuf) {
                  if ((expression[ig1+2] == '=') && (G__var_type == 'v')) { // *a++=expr, *a--=expr
                     G__var_type = 'p';
                     ebuf[lenbuf++] = c;
                     ebuf[lenbuf++] = c;
                     ++ig1;
                     break;
                  }
                  if (iscastexpr) {
                     ebuf[lenbuf++] = c;
                     ebuf[lenbuf++] = c;
                     ++ig1;
                     break;
                  }
                  if ( // a+ +b, a- -b
                     isalnum(expression[ig1+2]) ||
                     (expression[ig1+2] == '.') ||
                     (expression[ig1+2] == '_')
                  ) { // a+ +b, a- -b
                     ebuf[lenbuf] = 0;
                     ++ig1;
                     // Evaluate item.
                     ebuf[lenbuf] = '\0';
                     parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                     lenbuf = 0;
                     iscastexpr = 0;
                     G__exec_binopr(&parse_state, '+', G__PREC_ADD); // FIXME: This is wrong for minus if b has an unary operator-
                     break;
                  }
                  // a++, a--
                  ++ig1;
                  if (G__var_type == 'v') {
                     parse_state.unaopr[parse_state.up++] = '*';
                     G__var_type = 'p';
                  }
                  if (c == '+') {
                     parse_state.unaopr[parse_state.up++] = G__OPR_POSTFIXINC;
                  }
                  else {
                     parse_state.unaopr[parse_state.up++] = G__OPR_POSTFIXDEC;
                  }
                  break;
               }
               // *++a = expr, *--a = expr should be handled at assignment operator
               // ++a, --a
               ++ig1;
               if (G__var_type == 'v') {
                  parse_state.unaopr[parse_state.up++] = '*';
                  G__var_type = 'p';
               }
               if (c == '+') {
                  parse_state.unaopr[parse_state.up++] = G__OPR_PREFIXINC;
               }
               else {
                  parse_state.unaopr[parse_state.up++] = G__OPR_PREFIXDEC;
               }
               break;
            }
            if (expression[ig1+1] == '=') { // +=, -=
               if (!lenbuf) {
                  G__syntaxerror(expression);
                  return G__null;
               }
               ++ig1;
               // Evaluate item.
               if (lenbuf) {
                  ebuf[lenbuf] = '\0';
                  parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                  lenbuf = 0;
                  iscastexpr = 0;
               }
               if (c == '+') {
                  G__exec_evalall(&parse_state);
                  if ((parse_state.sp != 1) || parse_state.op || parse_state.up) {
                     G__syntaxerror(expression);
                     return G__null;
                  }
                  parse_state.vstack[1] = G__getexpr(expression + ig1 + 1);
                  G__bstore(G__OPR_ADDASSIGN, parse_state.vstack[1], &parse_state.vstack[0]);
                  G__var_type = 'p';
               }
               else {
                  G__exec_evalall(&parse_state);
                  if ((parse_state.sp != 1) || parse_state.op || parse_state.up) {
                     G__syntaxerror(expression);
                     return G__null;
                  }
                  parse_state.vstack[1] = G__getexpr(expression + ig1 + 1);
                  G__bstore(G__OPR_SUBASSIGN, parse_state.vstack[1], &parse_state.vstack[0]);
                  G__var_type = 'p';
               }
               return parse_state.vstack[0];
            }
            if (expression[ig1+1] == '>') { // a->b, note there is no a+>b
               ++ig1;
               ebuf[lenbuf++] = c;
               ebuf[lenbuf++] = expression[ig1];
               break;
            }
            if (lenbuf) {
               char* pebuf;
               if ( // 1e+10, 1e-10, (double) 1e+6
                  (tolower(expression[ig1-1]) == 'e') &&
                  !(
                      (expression[0] == '0') &&
                      (tolower(expression[1]) == 'x')
                  ) &&
                  (
                     isdigit(ebuf[0]) ||
                     (ebuf[0] == '.') ||
                     (
                        (ebuf[0] == '(') &&
                        (pebuf = strchr(ebuf, ')')) &&
                        (
                           isdigit(*++pebuf) ||
                           ((*pebuf) == '.')
                        )
                     )
                  )
               ) { // 1e+10, 1e-10, (double)1e+6
                  ebuf[lenbuf++] = c;
                  break;
               }
               ebuf[lenbuf] = 0;
               if (
                  !(
                     (lenbuf > 3) &&
                     (ebuf[0] == '(') &&
                     (ebuf[lenbuf-1] == ')') &&
                     (
                        (ebuf[lenbuf-2] == '*') ||
                        (ebuf[lenbuf-2] == '&') ||
                        G__iscastexpr_body(ebuf, lenbuf)
                     )
                  )
               ) { // a+b, a-b
                  // Evaluate item.
                  ebuf[lenbuf] = '\0';
                  parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                  lenbuf = 0;
                  iscastexpr = 0;
                  G__exec_binopr(&parse_state, c, G__PREC_ADD);
                  break;
               }
               // (int) -abc
               ebuf[lenbuf++] = c; // FIXME: Is this broken for unary +
               break;
            }
            if (c == '-') { // -a
               parse_state.unaopr[parse_state.up++] = c;
               break;
            }
            // ignore +a
            break;
         case '<': // a<<b, a<b, a<=b, a<<=b
            if (nest || single_quote || double_quote || explicitdtor) {
               ebuf[lenbuf++] = c;
               break;
            }
            ebuf[lenbuf] = '\0';
            if (G__defined_templateclass(ebuf)) {
               ++ig1;
               ebuf[lenbuf++] = c;
               c = G__getstream_template(expression, &ig1, ebuf + lenbuf, ">");
               lenbuf = strlen(ebuf);
               if ((c == '>') && (ebuf[lenbuf-1] == '>')) {
                  ebuf[lenbuf++] = ' ';
               }
               ebuf[lenbuf++] = c;
               ebuf[lenbuf] = '\0';
               --ig1;
               G__defined_tagname(ebuf, 1); // try to instantiate the template
               lenbuf = strlen(ebuf);
               break;
            }
            if (
               strchr(expression + ig1, '>') &&
               (
                  G__defined_templatefunc(ebuf) ||
                  G__defined_templatememfunc(ebuf)
               )
            ) {
               ++ig1;
               ebuf[lenbuf++] = c;
               c = G__getstream_template(expression, &ig1, ebuf + lenbuf, ">");
               if (c == '>') {
                  strcat(ebuf, ">");
               }
               lenbuf = strlen(ebuf);
               c = G__getstream_template(expression, &ig1, ebuf + lenbuf, "(");
               if (c == '(') {
                  strcat(ebuf, "(");
               }
               lenbuf = strlen(ebuf);
               c = G__getstream_template(expression, &ig1, ebuf + lenbuf, ")");
               if (c == ')') {
                  strcat(ebuf, ")");
               }
               lenbuf = strlen(ebuf);
               --ig1;
               break;
            }
            if (
               !strcmp(ebuf, "dynamic_cast") ||
               !strcmp(ebuf, "static_cast") ||
               !strcmp(ebuf, "reinterpret_cast") ||
               !strcmp(ebuf, "const_cast")
            ) {
               // TODO, implement casts, may need to introduce new instruction
               ++ig1;
               ebuf[0] = '(';
               c = G__getstream_template(expression, &ig1, ebuf + 1, ">");
               lenbuf = strlen(ebuf);
               ebuf[lenbuf++] = ')';
               ebuf[lenbuf] = '\0';
               --ig1;
               break;
            }
            if (expression[ig1+1] == '<') {
               if (expression[ig1+2] == '=') { // a<<=b
                  ig1 += 2;
                  // Evaluate item.
                  if (lenbuf) {
                     ebuf[lenbuf] = '\0';
                     parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                     lenbuf = 0;
                     iscastexpr = 0;
                  }
                  G__exec_evalall(&parse_state);
                  if ((parse_state.sp != 1) || parse_state.op || parse_state.up) {
                     G__syntaxerror(expression);
                     return G__null;
                  }
                  parse_state.vstack[1] = G__getexpr(expression + ig1 + 1);
                  G__bstore(G__OPR_LSFTASSIGN, parse_state.vstack[1], &parse_state.vstack[0]);
                  G__var_type = 'p';
                  return parse_state.vstack[0];
               }
               // a<<b
               ++ig1;
               // Evaluate item.
               ebuf[lenbuf] = '\0';
               parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
               lenbuf = 0;
               iscastexpr = 0;
               G__exec_binopr(&parse_state, G__OPR_LSFT, G__PREC_SHIFT);
               break;
            }
            if (expression[ig1+1] == '=') { // a<=b 
               ++ig1;
               // Evaluate item.
               ebuf[lenbuf] = '\0';
               parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
               lenbuf = 0;
               iscastexpr = 0;
               G__exec_binopr(&parse_state, G__OPR_LE, G__PREC_RELATION);
               break;
            }
            // a<b
            // Evaluate item.
            ebuf[lenbuf] = '\0';
            parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
            lenbuf = 0;
            iscastexpr = 0;
            G__exec_binopr(&parse_state, c, G__PREC_RELATION);
            break;
         case '>': // a>>b, a>b, a>=b, a>>=b
            if (nest || single_quote || double_quote || explicitdtor) {
               ebuf[lenbuf++] = c;
               break;
            }
            if (expression[ig1+1] == '>') {
               if (expression[ig1+2] == '=') { // a>>=b
                  ig1 += 2;
                  // Evaluate item.
                  if (lenbuf) {
                     ebuf[lenbuf] = '\0';
                     parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                     lenbuf = 0;
                     iscastexpr = 0;
                  }
                  G__exec_evalall(&parse_state);
                  if ((parse_state.sp != 1) || parse_state.op || parse_state.up) {
                     G__syntaxerror(expression);
                     return G__null;
                  }
                  parse_state.vstack[1] = G__getexpr(expression + ig1 + 1);
                  G__bstore(G__OPR_RSFTASSIGN, parse_state.vstack[1], &parse_state.vstack[0]);
                  G__var_type = 'p';
                  return parse_state.vstack[0];
               }
               // a>>b
               ++ig1;
               // Evaluate item.
               ebuf[lenbuf] = '\0';
               parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
               lenbuf = 0;
               iscastexpr = 0;
               G__exec_binopr(&parse_state, G__OPR_RSFT, G__PREC_SHIFT);
               break;
            }
            if (expression[ig1+1] == '=') { // a>=b 
               ++ig1;
               // Evaluate item.
               ebuf[lenbuf] = '\0';
               parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
               lenbuf = 0;
               iscastexpr = 0;
               G__exec_binopr(&parse_state, G__OPR_GE, G__PREC_RELATION);
               break;
            }
            // a>b
            // Evaluate item.
            ebuf[lenbuf] = '\0';
            parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
            lenbuf = 0;
            iscastexpr = 0;
            G__exec_binopr(&parse_state, c, G__PREC_RELATION);
            break;
         case '@': // a@b
            if (nest || single_quote || double_quote) {
               ebuf[lenbuf++] = c;
               break;
            }
            if (!lenbuf) {
               G__syntaxerror(expression);
               return G__null;
            }
            // Evaluate item.
            ebuf[lenbuf] = '\0';
            parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
            lenbuf = 0;
            iscastexpr = 0;
            G__exec_binopr(&parse_state, c, G__PREC_PWR);
            break;
         case '*': // *a, a*b, a*=b, a**b, **a
            if (nest || single_quote || double_quote) {
               ebuf[lenbuf++] = c;
               break;
            }
            if (expression[ig1+1] == '=') { // a*=b
               ++ig1;
               // Evaluate item.
               if (lenbuf) {
                  ebuf[lenbuf] = '\0';
                  parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                  lenbuf = 0;
                  iscastexpr = 0;
               }
               G__exec_evalall(&parse_state);
               if ((parse_state.sp != 1) || parse_state.op || parse_state.up) {
                  G__syntaxerror(expression);
                  return G__null;
               }
               parse_state.vstack[1] = G__getexpr(expression + ig1 + 1);
               G__bstore(G__OPR_MULASSIGN, parse_state.vstack[1], &parse_state.vstack[0]);
               G__var_type = 'p';
               return parse_state.vstack[0];
            }
            if (expression[ig1+1] == '*') { // a**b, **a
               if (lenbuf) {
                  // --
#ifndef G__NOPOWEROPR
                  // a**b handle as power operator
                  ++ig1;
                  // Evaluate item.
                  ebuf[lenbuf] = '\0';
                  parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                  lenbuf = 0;
                  iscastexpr = 0;
                  G__exec_binopr(&parse_state, '@', G__PREC_PWR);
#else // G__NOPOWEROPR
                  // a**b handle as a * (*b)
                  // Evaluate item.
                  ebuf[lenbuf] = '\0';
                  parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                  lenbuf = 0;
                  iscastexpr = 0;
                  G__exec_binopr(&parse_state, '*', G__PREC_MULT);
                  parse_state.unaopr[parse_state.up++] = '*';
                  ++ig1;
#endif // G__NOPOWEROPR
                  break;
               }
               // **a
               ++ig1;
               parse_state.unaopr[parse_state.up++] = c;
               parse_state.unaopr[parse_state.up++] = c;
               break;
            }
            if (lenbuf) { // a*b, a*=b
               ebuf[lenbuf] = 0;
               if (
                  !(
                     (lenbuf > 3) &&
                     (ebuf[0] == '(') &&
                     (ebuf[lenbuf-1] == ')') &&
                     (
                        (ebuf[lenbuf-2] == '*') ||
                        (ebuf[lenbuf-2] == '&') ||
                        G__iscastexpr_body(ebuf, lenbuf)
                     )
                  )
               ) { // a*b
                  // Evaluate item.
                  ebuf[lenbuf] = '\0';
                  parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                  lenbuf = 0;
                  iscastexpr = 0;
                  G__exec_binopr(&parse_state, c, G__PREC_MULT);
                  break;
               }
               // (int) *abc
               ebuf[lenbuf++] = c;
               break;
            }
            // *a
            parse_state.unaopr[parse_state.up++] = c;
            break;
         case '&': // &a, a&b, a&&b, a&=b
            if (nest || single_quote || double_quote) {
               ebuf[lenbuf++] = c;
               break;
            }
            if (expression[ig1+1] == '&') { // a&&b
               ++ig1;
               // Evaluate item.
               ebuf[lenbuf] = '\0';
               parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
               lenbuf = 0;
               iscastexpr = 0;
               G__exec_binopr(&parse_state, 'A', G__PREC_LOGICAND);
               ///
               ///  G__SUSPEND_ANDOPR
               ///
               if (G__get_type(parse_state.vstack[parse_state.sp-1]) != 'u') {
                  parse_state.store_no_exec_compile_and[parse_state.pp_and] = G__no_exec_compile;
                  if (!G__no_exec_compile && (G__double(parse_state.vstack[parse_state.sp-1]) == 0.0)) {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr, "   G__no_exec_compile set\n");
                     }
#endif // G__ASM_DBG
                     G__no_exec_compile = 1;
                     parse_state.vtmp_and = parse_state.vstack[parse_state.sp-1];
                  }
#ifdef G__ASM
                  if (G__asm_noverflow) {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr, "%3x,%3x: PUSHCPY\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                        G__fprinterr(G__serr, "%3x,%3x: CNDJMP (assigned later)\n", G__asm_cp + 1, G__asm_dt, __FILE__, __LINE__);
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__PUSHCPY;
                     G__asm_inst[G__asm_cp+1] = G__CNDJMP;
                     G__asm_inst[G__asm_cp+2] = (long) G__PVOID;
                     parse_state.ppointer_and[parse_state.pp_and] = G__asm_cp + 2;
                     G__inc_cp_asm(3, 0);
                  }
#endif // G__ASM
                  ++G__templevel;
                  ++parse_state.pp_and;
               }
               break;
            }
            if (expression[ig1+1] == '=') { // a&=b
               ++ig1;
               // Evaluate item.
               if (lenbuf) {
                  ebuf[lenbuf] = '\0';
                  parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                  lenbuf = 0;
                  iscastexpr = 0;
               }
               G__exec_evalall(&parse_state);
               if ((parse_state.sp != 1) || parse_state.op || parse_state.up) {
                  G__syntaxerror(expression);
                  return G__null;
               }
               parse_state.vstack[1] = G__getexpr(expression + ig1 + 1);
               G__bstore(G__OPR_BANDASSIGN, parse_state.vstack[1], &parse_state.vstack[0]);
               G__var_type = 'p';
               return parse_state.vstack[0];
            }
            if (lenbuf) {
               ebuf[lenbuf] = 0;
               if (
                  !(
                     (lenbuf > 3) &&
                     (ebuf[0] == '(') &&
                     (ebuf[lenbuf-1] == ')') &&
                     (
                        (ebuf[lenbuf-2] == '*') ||
                        (ebuf[lenbuf-2] == '&') ||
                        G__iscastexpr_body(ebuf, lenbuf)
                     )
                  )
               ) { // a&b
                  // Evaluate item.
                  ebuf[lenbuf] = '\0';
                  parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                  lenbuf = 0;
                  iscastexpr = 0;
                  G__exec_binopr(&parse_state, c, G__PREC_BITAND);
                  break;
               }
               // (int*) &abc
               ebuf[lenbuf++] = c;
               break;
            }
            // &a
            parse_state.unaopr[parse_state.up++] = c;
            break;
         case '|': // a|b, a||b, a|=b 
            if (nest || single_quote || double_quote) {
               ebuf[lenbuf++] = c;
               break;
            }
            if (expression[ig1+1] == '|') { // a||b
               ++ig1;
               // Evaluate item.
               ebuf[lenbuf] = '\0';
               parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
               lenbuf = 0;
               iscastexpr = 0;
               G__exec_binopr(&parse_state, 'O', G__PREC_LOGICOR);
               ///
               /// G__RESTORE_NOEXEC_ANDOPR
               ///
               if (parse_state.pp_and) { \
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "   G__no_exec_compile reset %d\n", parse_state.store_no_exec_compile_and[0]);
                  }
#endif // G__ASM_DBG
                  if (!parse_state.store_no_exec_compile_and[0] && G__no_exec_compile) {
                     parse_state.vstack[parse_state.sp-1] = parse_state.vtmp_and;
                  }
                  G__no_exec_compile = parse_state.store_no_exec_compile_and[0];
               }
               ///
               /// G__RESTORE_ANDOPR
               ///
               while (parse_state.pp_and) {
                  --parse_state.pp_and;
                  G__free_tempobject();
                  --G__templevel;
#ifdef G__ASM
                  if (G__asm_noverflow) {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr,"   %3x: CNDJMP assigned for AND %3x  %s:%d\n", parse_state.ppointer_and[parse_state.pp_and] - 1, G__asm_cp, __FILE__, __LINE__);
                     }
#endif // G__ASM_DBG
                     if ((char*) G__asm_inst[parse_state.ppointer_and[parse_state.pp_and]] == G__PVOID) {
                        G__asm_inst[parse_state.ppointer_and[parse_state.pp_and]] = G__asm_cp;
                     }
                  }
#endif // G__ASM
                  // --
               }
               ///
               /// G__SUSPEND_OROPER
               ///
               if (G__get_type(parse_state.vstack[parse_state.sp-1]) != 'u') {
                  parse_state.store_no_exec_compile_or[parse_state.pp_or] = G__no_exec_compile;
                  if (!G__no_exec_compile && G__double(parse_state.vstack[parse_state.sp-1]) != 0.0) {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr,"   G__no_exec_compile set\n");
                     }
#endif // G__ASM_DBG
                     G__no_exec_compile = 1;
                     parse_state.vstack[parse_state.sp-1] = G__one;
                     parse_state.vtmp_or = parse_state.vstack[parse_state.sp-1]; // FIXME: It will always be G__one, is this right?
                  }
#ifdef G__ASM
                  if (G__asm_noverflow) {
                     // --
#ifdef G__ASM_DBG
                     if (G__asm_dbg) {
                        G__fprinterr(G__serr, "%3x,%3x: BOOL  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                        G__fprinterr(G__serr, "%3x,%3x: PUSHCPY  %s:%d\n", G__asm_cp + 1, G__asm_dt, __FILE__, __LINE__);
                        G__fprinterr(G__serr, "%3x,%3x: CND1JMP (assigned later)  %s:%d\n", G__asm_cp + 2, G__asm_dt, __FILE__, __LINE__);
                     }
#endif // G__ASM_DBG
                     G__asm_inst[G__asm_cp] = G__BOOL;
                     G__asm_inst[G__asm_cp+1] = G__PUSHCPY;
                     G__asm_inst[G__asm_cp+2] = G__CND1JMP;
                     G__asm_inst[G__asm_cp+3] = (long) G__PVOID;
                     parse_state.ppointer_or[parse_state.pp_or] = G__asm_cp + 3;
                     G__inc_cp_asm(4, 0);
                  }
#endif // G__ASM
                  ++G__templevel;
                  ++parse_state.pp_or;
               }
               break;
            }
            if (expression[ig1+1] == '=') { // a|=b
               ++ig1;
               // Evaluate item.
               if (lenbuf) {
                  ebuf[lenbuf] = '\0';
                  parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                  lenbuf = 0;
                  iscastexpr = 0;
               }
               G__exec_evalall(&parse_state);
               if ((parse_state.sp != 1) || parse_state.op || parse_state.up) {
                  G__syntaxerror(expression);
                  return G__null;
               }
               parse_state.vstack[1] = G__getexpr(expression + ig1 + 1);
               G__bstore(G__OPR_BORASSIGN, parse_state.vstack[1], &parse_state.vstack[0]);
               G__var_type = 'p';
               return parse_state.vstack[0];
            }
            if (lenbuf) { // a|b
               // Evaluate item.
               ebuf[lenbuf] = '\0';
               parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
               lenbuf = 0;
               iscastexpr = 0;
               G__exec_binopr(&parse_state, c, G__PREC_BITOR);
               break;
            }
            // |a
            parse_state.unaopr[parse_state.up++] = c; // FIXME: There is no unary '|'
            break;
         case '=': /* a==b, a=b */
            {
               if (nest || single_quote || double_quote) {
                  ebuf[lenbuf++] = c;
                  break;
               }
               if (c == expression[ig1+1]) { // a==b
                  ++ig1;
                  // Evaluate item.
                  ebuf[lenbuf] = '\0';
                  parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
                  lenbuf = 0;
                  iscastexpr = 0;
                  G__exec_binopr(&parse_state, G__OPR_EQ, G__PREC_EQUAL);
                  inew = ig1 + 1;
                  break;
               }
               // a=b
               G__var_type = 'p';
               G__value defined = G__getexpr(expression + ig1 + 1);
               strncpy(ebuf, expression, ig1);
               ebuf[ig1] = '\0';
               G__var_type = store_var_type;
               return G__letvariable(ebuf, defined, ::Reflex::Scope::GlobalScope(), G__p_local);
            }
         case '?': // a ? b : c
            if (nest || single_quote || double_quote) {
               ebuf[lenbuf++] = c;
               break;
            }
            // Evaluate item.
            if (lenbuf) {
               ebuf[lenbuf] = '\0';
               parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
               lenbuf = 0;
               iscastexpr = 0;
            }
            G__exec_evalall(&parse_state);
            if ((parse_state.sp != 1) || parse_state.op || parse_state.up) {
               G__syntaxerror(expression);
               return G__null;
            }
            ///
            /// G__RESTORE_NOEXEC_ANDOPR
            ///
            if (parse_state.pp_and) { \
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "   G__no_exec_compile reset %d\n", parse_state.store_no_exec_compile_and[0]);
               }
#endif // G__ASM_DBG
               if (!parse_state.store_no_exec_compile_and[0] && G__no_exec_compile) {
                  parse_state.vstack[parse_state.sp-1] = parse_state.vtmp_and;
               }
               G__no_exec_compile = parse_state.store_no_exec_compile_and[0];
            }
            ///
            /// G__RESTORE_NOEXEC_OROPR
            ///
            if (parse_state.pp_or) {
               // --
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr,"   G__no_exec_compile reset %d\n", parse_state.store_no_exec_compile_or[0]);
               }
#endif // G__ASM_DBG
               if (!parse_state.store_no_exec_compile_or[0] && G__no_exec_compile) {
                  parse_state.vstack[parse_state.sp-1] = parse_state.vtmp_or;
               }
               G__no_exec_compile = parse_state.store_no_exec_compile_or[0];
            }
            ///
            /// G__RESTORE_ANDOPR
            ///
            while (parse_state.pp_and) {
               --parse_state.pp_and;
               G__free_tempobject();
               --G__templevel;
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // --
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr,"   %3x: CNDJMP assigned for AND %3x  %s:%d\n", parse_state.ppointer_and[parse_state.pp_and] - 1, G__asm_cp, __FILE__, __LINE__);
                  }
#endif // G__ASM_DBG
                  if ((char*) G__asm_inst[parse_state.ppointer_and[parse_state.pp_and]] == G__PVOID) {
                     G__asm_inst[parse_state.ppointer_and[parse_state.pp_and]] = G__asm_cp;
                  }
               }
#endif // G__ASM
               // --
            }
            ///
            /// G__RESTORE_OROPR
            ///
            while (parse_state.pp_or) {
               --parse_state.pp_or;
               G__free_tempobject();
               --G__templevel;
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // --
#ifdef G__ASM_DBG // G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr,"   %x: CND1JMP assigned for OR %x  %s:%d\n", parse_state.ppointer_or[parse_state.pp_or] - 1, G__asm_cp, __FILE__, __LINE__);
                  }
#endif // G__ASM_DBG
                  G__asm_inst[parse_state.ppointer_or[parse_state.pp_or]] = G__asm_cp;
               }
#endif // G__ASM
               // --
            }
            return G__conditionaloperator(parse_state.vstack[0], expression, ig1, ebuf);
         default: // part of a term
            ebuf[lenbuf++] = c;
            break;
      }
   }
   //
   // Evaluate remaining operators on stack.
   //
   // Evaluate item.
   if (lenbuf) {
      ebuf[lenbuf] = '\0';
      parse_state.vstack[parse_state.sp++] = G__getitem(ebuf);
      lenbuf = 0;
      iscastexpr = 0;
   }
   G__exec_evalall(&parse_state);
   if ((parse_state.sp != 1) || parse_state.op || parse_state.up) {
      G__syntaxerror(expression);
      return G__null;
   }
   ///
   /// G__RESTORE_NOEXEC_ANDOPR
   ///
   if (parse_state.pp_and) { \
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "   G__no_exec_compile reset %d\n", parse_state.store_no_exec_compile_and[0]);
      }
#endif // G__ASM_DBG
      if (!parse_state.store_no_exec_compile_and[0] && G__no_exec_compile) {
         parse_state.vstack[parse_state.sp-1] = parse_state.vtmp_and;
      }
      G__no_exec_compile = parse_state.store_no_exec_compile_and[0];
   }
   ///
   /// G__RESTORE_NOEXEC_OROPR
   ///
   if (parse_state.pp_or) {
      // --
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr,"   G__no_exec_compile reset %d\n", parse_state.store_no_exec_compile_or[0]);
      }
#endif // G__ASM_DBG
      if (!parse_state.store_no_exec_compile_or[0] && G__no_exec_compile) {
         parse_state.vstack[parse_state.sp-1] = parse_state.vtmp_or;
      }
      G__no_exec_compile = parse_state.store_no_exec_compile_or[0];
   }
   ///
   /// G__RESTORE_ANDOPR
   ///
   while (parse_state.pp_and) {
      --parse_state.pp_and;
      G__free_tempobject();
      --G__templevel;
#ifdef G__ASM
      if (G__asm_noverflow) {
         // --
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr,"   %3x: CNDJMP assigned for AND %3x  %s:%d\n", parse_state.ppointer_and[parse_state.pp_and] - 1, G__asm_cp, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         if ((char*) G__asm_inst[parse_state.ppointer_and[parse_state.pp_and]] == G__PVOID) {
            G__asm_inst[parse_state.ppointer_and[parse_state.pp_and]] = G__asm_cp;
         }
      }
#endif // G__ASM
      // --
   }
   ///
   /// G__RESTORE_OROPR
   ///
   while (parse_state.pp_or) {
      --parse_state.pp_or;
      G__free_tempobject();
      --G__templevel;
#ifdef G__ASM
      if (G__asm_noverflow) {
         // --
#ifdef G__ASM_DBG // G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr,"   %x: CND1JMP assigned for OR %x  %s:%d\n", parse_state.ppointer_or[parse_state.pp_or] - 1, G__asm_cp, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[parse_state.ppointer_or[parse_state.pp_or]] = G__asm_cp;
      }
#endif // G__ASM
      // --
   }
   //
   //
   return parse_state.vstack[0];
}

//______________________________________________________________________________
//
//  ANSI compliant operator precedences,  smaller values have higher precedence
//

#undef G__PREC_SCOPE
#undef G__PREC_FCALL
#undef G__PREC_UNARY
#undef G__PREC_P2MEM
#undef G__PREC_PWR
#undef G__PREC_MULT
#undef G__PREC_ADD
#undef G__PREC_SHIFT
#undef G__PREC_RELATION
#undef G__PREC_EQUAL
#undef G__PREC_BITAND
#undef G__PREC_BITEXOR
#undef G__PREC_BITOR
#undef G__PREC_LOGICAND
#undef G__PREC_LOGICOR
#undef G__PREC_TEST
#undef G__PREC_ASSIGN
#undef G__PREC_COMMA

//______________________________________________________________________________
G__value Cint::Internal::G__getprod( char* expression1)
{
   int length1 = strlen(expression1);
   if (!length1) {
      return G__null;
   }
   G__value defined1;
   defined1 = G__null;
   G__value reg;
   G__StrBuf ebuf1_sb(G__ONELINE);
   char* ebuf1 = ebuf1_sb;
   int operator1 = '\0';
   int prodpower = 0;
   int lenbuf1 = 0;
   int ig11;
   int ig2;
   int nest1 = 0;
   int single_quote = 0;
   int double_quote = 0;
   switch (expression1[0]) {
      case '*': /* value of pointer */
         if (expression1[1] == '(') {
            reg = G__getexpr(expression1 + 1);
            defined1 = G__tovalue(reg);
            return(defined1);
         }
         G__var_type = 'v';
         for (ig2 = 0;ig2 < length1;ig2++) expression1[ig2] = expression1[ig2+1];
         break;
      default :
         break;
   }
   for (ig11 = 0; ig11 < length1; ++ig11) {
      switch (expression1[ig11]) {
         case '"' :
            if (!single_quote) {
               double_quote ^= 1;
            }
            ebuf1[lenbuf1++] = expression1[ig11];
            break;
         case '\'' :
            if (!double_quote) {
               single_quote ^= 1;
            }
            ebuf1[lenbuf1++] = expression1[ig11];
            break;
         case '*':
            if (!strncmp(expression1, "new ", 4)) {
               ebuf1[lenbuf1++] = expression1[ig11];
               break;
            }
         case '/':
         case '%':
            if (nest1 || single_quote || double_quote) {
               ebuf1[lenbuf1++] = expression1[ig11];
               break;
            }
            switch (lenbuf1) {
               case 0:
                  operator1 = G__getoperator(operator1, expression1[ig11]);
                  break;
               default:
                  if (operator1 == '\0') {
                     operator1 = '*';
                  }
                  ebuf1[lenbuf1] = '\0';
                  reg = G__getpower(ebuf1);
                  G__bstore(operator1, reg, &defined1);
                  lenbuf1 = 0;
                  ebuf1[0] = '\0';
                  operator1 = expression1[ig11];
                  break;
            }
            break;
         case '(':
         case '[':
         case '{':
            ebuf1[lenbuf1++] = expression1[ig11];
            if (!double_quote && !single_quote) {
               ++nest1;
            }
            break;
         case ')':
         case ']':
         case '}':
            ebuf1[lenbuf1++] = expression1[ig11];
            if (!double_quote && !single_quote) {
               --nest1;
            }
            break;
         case '@':
         case '~':
         case ' ':
            if (!nest1 && !single_quote && !double_quote) {
               prodpower = 1;
            }
            ebuf1[lenbuf1++] = expression1[ig11];
            break;
         case '\\' :
            ebuf1[lenbuf1++] = expression1[ig11++];
            ebuf1[lenbuf1++] = expression1[ig11];
            break;
         default:
            ebuf1[lenbuf1++] = expression1[ig11];
            break;
      }
   }
   ebuf1[lenbuf1] = '\0';
   if (nest1 || single_quote || double_quote) {
      G__parenthesiserror(expression1, "G__getprod");
      return G__null;
   }
   if (prodpower != 0) {
      reg = G__getpower(ebuf1);
   }
   else {
      reg = G__getitem(ebuf1);
   }
   G__bstore(operator1, reg, &defined1);
   return defined1;
}

//______________________________________________________________________________
G__value Cint::Internal::G__getitem(const char* item)
{
   int known;
   G__value result3;
   int c;
   char store_var_typeB;
   G__value reg;
   switch (item[0]) {
      case '0':
         c = tolower(item[1]);
         if (
            (c != '\0') &&
            (c != '.') &&
            (c != 'f') &&
            (c != 'e') &&
            (c != 'l') &&
            (c != 'u') &&
            (c != 's')
         ) {
            result3 = G__checkBase(item, &known);
#ifdef G__ASM
            if (G__asm_noverflow) { // -- We are generating bytecode.
               /**************************************
                * G__LD instruction
                * 0 LD
                * 1 address in data stack
                * put result3
                **************************************/
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD %ld from %x  %s:%d\n", G__asm_cp, G__asm_dt, G__int(result3), G__asm_dt, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD;
               G__asm_inst[G__asm_cp+1] = G__asm_dt;
               G__asm_stack[G__asm_dt] = result3;
               G__inc_cp_asm(2, 1);
            }
#endif // G__ASM
            G__value_typenum(result3) = G__modify_type(G__value_typenum(result3), 0, 0, G__CONSTVAR + G__STATICCONST, 0, 0);
            return result3;
         }
         // Intentional dropthrough.
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
         if (G__isfloat(item, &c)) {
            G__letdouble(&result3, c, atof(item));
         }
         else {
            switch (c) {
               case 'n':
                  G__letLonglong(&result3, c, G__expr_strtoll(item, 0, 10));
                  break;
               case 'm':
                  G__letULonglong(&result3, c, G__expr_strtoull(item, 0, 10));
                  break;
               default:
                  G__letint(&result3, c, strtoul(item, 0, 10));
                  break;
            }
         }
         if (c != 'u') {
            result3.ref = 0;
         }
         G__value_typenum(result3) = G__modify_type(G__value_typenum(result3), 0, 0, G__CONSTVAR + G__STATICCONST, 0, 0);
#ifdef G__ASM
         if (G__asm_noverflow) { // We are generating bytecode.
            /**************************************
             * G__LD instruction
             * 0 LD
             * 1 address in data stack
             * put result3
             **************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: LD %g from %x  %s:%d\n", G__asm_cp, G__asm_dt, G__double(result3), G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__LD;
            G__asm_inst[G__asm_cp+1] = G__asm_dt;
            G__asm_stack[G__asm_dt] = result3;
            G__inc_cp_asm(2, 1);
         }
#endif // G__ASM
         break;
      case '\'':
         result3 = G__strip_singlequotation(item); 
         G__value_typenum(result3) = G__modify_type(G__value_typenum(result3), 0, 0, G__CONSTVAR, 0, 0);
#ifdef G__ASM
         if (G__asm_noverflow) { // We are generating bytecode.
            /**************************************
             * G__LD instruction
             * 0 LD
             * 1 address in data stack
             * put result3
             **************************************/
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: LD '%c' from %x  %s:%d\n", G__asm_cp, G__asm_dt, (char) G__int(result3), G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__LD;
            G__asm_inst[G__asm_cp+1] = G__asm_dt;
            G__asm_stack[G__asm_dt] = result3;
            G__inc_cp_asm(2, 1);
         }
#endif // G__ASM
         break;
      case '"':
         result3 = G__strip_quotation(item);
#ifdef G__ASM
         if (G__asm_noverflow) { // We are generating bytecode.
            G__asm_gen_strip_quotation(&result3);
         }
#endif // G__ASM
         return result3;
      case '-':
         reg = G__getitem(item + 1);
         result3 = G__null;
         G__bstore('-', reg, &result3);
         return result3;
      case '_':
         if (item[1] == '$') { // We have _$xxxxxxx.
            G__getiparseobject(&result3, item);
            return result3;
         }
         // Intentional dropthrough.
      default:
         store_var_typeB = G__var_type;
         known = 0;
         G__var_type = 'p';
         //
         //  Try a variable.
         //
         //fprintf(stderr, "G__get_item: Lookup up variable '%s' in scope '%s'\n", item, G__p_local.Name(Reflex::SCOPED).c_str());
         result3 = G__getvariable(/*FIXME*/(char*)item, &known, ::Reflex::Scope::GlobalScope(), G__p_local);
         if (known) {
            G__var_typeB = 'p';
            return result3;
         }
         if ( // this is "a.b", we know "a", but it has no "b", give up
            (G__get_tagnum(G__value_typenum(result3)) != -1) && // we have a class part, and
            !result3.obj.i // no pointer
         ) {
            // this is "a.b", we know "a", but it has no "b", give up
            if (!G__noerr_defined && !G__definemacro) {
               return G__interactivereturn();
            }
            return G__null;
         }
         //
         //  Try a function.
         //
         G__var_typeB = store_var_typeB;
         result3 = G__getfunction(item, &known, G__TRYNORMAL);
         if (known) { // We have the function, call it.
            result3 = G__toXvalue(result3, store_var_typeB);
            if (G__initval_eval) { // Flag that getting the value involved a function call, and so cannot be used to initialize a static const variable.
               G__dynconst = G__DYNCONST;
            }
            G__var_typeB = 'p';
            return result3;
         }
         G__var_typeB = 'p';
#ifdef G__PTR2MEMFUNC
         if (!result3.obj.i) { // Still not found, try a pointer to a member function.
            known = G__getpointer2memberfunc(item, &result3);
            if (known) {
               return result3;
            }
         }
#endif // G__PTR2MEMFUNC
         //
         //  Not a variable or a function.  Try specials.
         //
         if (!strncmp(item, "__", 2)) { // We have __xxxxxxxx, try reserved.
            result3 = G__getreserved(item + 1, 0, 0);
            int type = G__get_type(result3);
            if (type) {
               return result3;
            }
         }
         else if ( // Try a ROOT special if we are allowed, and if so, return unconditionally.
            // --
#ifdef G__ROOT
            (G__dispmsg < G__DISPROOTSTRICT) &&
#endif // G__ROOT
            G__GetSpecialObject &&
            (G__GetSpecialObject != ((G__value(*)(char*, void**, void**)) G__getreserved)) &&
            !G__gettingspecial &&
            (item[0] != '$')
         ) {
            // Prepend '$' to object and try to find it again.
            int store_return = G__return;
            int store_security_error = G__security_error;
            if (G__asm_noverflow && G__no_exec_compile) {
               G__abortbytecode();
            }
            char* sbuf = (char*) malloc(strlen(item) + 2);
            if (!sbuf) {
               G__genericerror("Internal error: malloc in G__getitem(),sbuf");
               return G__null;
            }
            sprintf(sbuf, "$%s", item);
            G__gettingspecial = 1;
            G__var_type = store_var_typeB;
            result3 = G__getitem(sbuf);
            free(sbuf);
            G__gettingspecial = 0;
            if (G__const_noerror) {
               G__return = store_return;
               G__security_error = store_security_error;
            }
            return result3;
         }
         if (!result3.obj.i) {
            if (G__noerr_defined) {
               return G__null;
            }
            if (G__definemacro) {
               //G__genericerror("Limitation: This form of macro may not be expanded. Use +P or -p option");
               return G__null;
            }
            G__warnundefined(item);
            result3 = G__interactivereturn();
            return result3;
         }
         break;
   }
   return result3;
}

//______________________________________________________________________________
//______________________________________________________________________________

//______________________________________________________________________________
char* Cint::Internal::G__setiparseobject(G__value* result, char* str)
{
   // --
   sprintf(str, "_$%c%d%c_%d_%c%lu", G__get_type(*result), 0, (!G__value_typenum(*result).FinalType().IsConst()) ? '0' : '1', G__get_tagnum(G__value_typenum(*result)), (result->obj.i < 0) ? 'M' : 'P', labs(result->obj.i));
   return str;
}

//______________________________________________________________________________
//______________________________________________________________________________

//______________________________________________________________________________
int Cint::Internal::G__test(char* expression2)
{
   G__value result;
   result = G__getexpr(expression2);
   if ('u' == G__get_type(result)) return(G__iosrdstate(&result));
   return G__convertT<bool>(&result);   
}

//______________________________________________________________________________
int Cint::Internal::G__btest(int operator2, G__value lresult, G__value rresult)
{
   if (G__get_type(lresult) == 'u' || G__get_type(rresult) == 'u') {
      G__overloadopr(operator2, rresult, &lresult);
      return(G__int(lresult));
   }
   else if (G__get_type(lresult) == 'U' || G__get_type(rresult) == 'U') {
      G__publicinheritance(&lresult, &rresult);
   }
#ifdef G__ASM
   if (G__asm_noverflow) {
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x: CMP2  '%c'\n" , G__asm_cp, operator2);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__CMP2;
      G__asm_inst[G__asm_cp+1] = (long)operator2;
      G__inc_cp_asm(2, 0);
   }
#endif // G__ASM
   if (G__no_exec_compile || G__no_exec) {
      // avoid Alpha crash
      return 1;
   }
   switch (operator2) {
      case 'E': /* == */
         if (G__double(lresult) == G__double(rresult)) return(1);
         else return(0);
         /* break; */
      case 'N': /* != */
         if (G__double(lresult) != G__double(rresult)) return(1);
         else return(0);
         /* break; */
      case 'G': /* >= */
         if (G__double(lresult) >= G__double(rresult)) return(1);
         else return(0);
         /* break; */
      case 'l': /* <= */
         if (G__double(lresult) <= G__double(rresult)) return(1);
         else return(0);
         /* break; */
      case '<': /* <  */
         if (G__double(lresult) < G__double(rresult)) return(1);
         else return(0);
         /* break; */
      case '>': /* >  */
         if (G__double(lresult) > G__double(rresult)) return(1);
         else return(0);
         /* break; */
   }
   G__genericerror("Error: Unknow operator in test condition");
   return 0;
}

//______________________________________________________________________________
//
//  Functions in the C interface.
//

//______________________________________________________________________________
extern "C" int G__lasterror()
{
   // --
   return G__last_error;
}

//______________________________________________________________________________
extern "C" void G__reset_lasterror()
{
   // --
   G__last_error = G__NOERROR;
}

//______________________________________________________________________________
extern "C" G__value G__calc(const char* exprwithspace)
{
   G__value result;
   int store_security_error;
   G__LockCriticalSection();
   store_security_error = G__security_error;
   G__security_error = G__NOERROR;
   G__storerewindposition();
   result = G__calc_internal((char*)exprwithspace);
   G__security_recover(G__serr);
   G__security_error = store_security_error;
   G__UnlockCriticalSection();
   return result;
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
