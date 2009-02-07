/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file opr.c
 ************************************************************************
 * Description:
 *  Unary and binary operator handling
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "Dict.h"
#include "value.h"

using namespace Cint::Internal;
using namespace std;

//
//  Function Directory.
//

// Static functions.
static const char* G__getoperatorstring(int operatortag);

// Cint internal functions.
namespace Cint {
namespace Internal {
void G__doubleassignbyref(G__value* defined, double val);
void G__intassignbyref(G__value* defined, G__int64 val);
void G__bstore(int operatortag, G__value expressionin, G__value* defined);
int G__scopeoperator(char* name /* name is modified and this is intentional */, int* phash, char** pstruct_offset, int* ptagnum);
int G__cmp(G__value buf1, G__value buf2);
int G__getunaryop(char unaryop, char* expression, char* buf, G__value* preg);
int G__iosrdstate(G__value* pios);
int G__overloadopr(int operatortag, G__value expressionin, G__value* defined);
int G__parenthesisovldobj(G__value* result3, G__value* result, const char* realname, G__param* libp, int flag);
int G__parenthesisovld(G__value* result3, const char* funcname, G__param* libp, int flag);
int G__tryindexopr(G__value* result7, G__value* para, int paran, int ig25);
long G__op1_operator_detail(int opr, G__value* val);
long G__op2_operator_detail(int opr, G__value* lval, G__value* rval);
} // namespace Internal
} // namespace Cint

//______________________________________________________________________________
static const char* G__getoperatorstring(int operatortag)
{
   // -- FIXME: Describe this function!
   switch (operatortag) {
      case '+': /* add */
         return("+");
      case '-': /* subtract */
         return("-");
      case '*': /* multiply */
         return("*");
      case '/': /* divide */
         return("/");
      case '%': /* modulus */
         return("%");
      case '&': /* binary and */
         return("&");
      case '|': /* binary or */
         return("|");
      case '^': /* binary exclusive or */
         return("^");
      case '~': /* binary inverse */
         return("~");
      case 'A': /* logical and */
         return("&&");
      case 'O': /* logical or */
         return("||");
      case '>':
         return(">");
      case '<':
         return("<");
      case 'R': /* right shift */
         return(">>");
      case 'L': /* left shift */
         return("<<");
      case '@': /* power */
         return("@");
      case '!':
         return("!");
      case 'E': /* == */
         return("==");
      case 'N': /* != */
         return("!=");
      case 'G': /* >= */
         return(">=");
      case 'l': /* <= */
         return("<=");
      case G__OPR_ADDASSIGN:
         return("+=");
      case G__OPR_SUBASSIGN:
         return("-=");
      case G__OPR_MODASSIGN:
         return("%=");
      case G__OPR_MULASSIGN:
         return("*=");
      case G__OPR_DIVASSIGN:
         return("/=");
      case G__OPR_RSFTASSIGN:
         return(">>=");
      case G__OPR_LSFTASSIGN:
         return("<<=");
      case G__OPR_BANDASSIGN:
         return("&=");
      case G__OPR_BORASSIGN:
         return("|=");
      case G__OPR_EXORASSIGN:
         return("^=");
      case G__OPR_ANDASSIGN:
         return("&&=");
      case G__OPR_ORASSIGN:
         return("||=");
      case G__OPR_POSTFIXINC:
      case G__OPR_PREFIXINC:
         return("++");
      case G__OPR_POSTFIXDEC:
      case G__OPR_PREFIXDEC:
         return("--");
      default:
         return("(unknown operator)");
   }
}

//______________________________________________________________________________
void Cint::Internal::G__doubleassignbyref(G__value* defined, double val)
{
   // -- FIXME: Describe this function!
   if (isupper(G__get_type(G__value_typenum(*defined)))) {
      *(long*)defined->ref = (long)val;
      defined->obj.i = (long)val;
      return;
   }
   switch (G__get_type(G__value_typenum(*defined))) {
      case 'd': /* double */
         *(double*)defined->ref = val;
         G__setvalue(defined, val);
         break;
      case 'f': /* float */
         *(float*)defined->ref = (float)val;
         G__setvalue(defined, val);
         break;
      case 'l': /* long */
         *(long*)defined->ref = (long)val;
         G__setvalue(defined, (long)val);
         break;
      case 'k': /* unsigned long */
         *(unsigned long*)defined->ref = (unsigned long)val;
         G__setvalue(defined, (unsigned long)val);
         break;
      case 'i': /* int */
         *(int*)defined->ref = (int)val;
         G__setvalue(defined, (int)val);
         break;
      case 'h': /* unsigned int */
         *(unsigned int*)defined->ref = (unsigned int)val;
         G__setvalue(defined, (unsigned int)val);
         break;
      case 's': /* short */
         *(short*)defined->ref = (short)val;
         G__setvalue(defined, (short)val);
         break;
      case 'r': /* unsigned short */
         *(unsigned short*)defined->ref = (unsigned short)val;
         G__setvalue(defined, (unsigned short)val);
         break;
      case 'c': /* char */
         *(char*)defined->ref = (char)val;
         G__setvalue(defined, (char)val);
         break;
      case 'b': /* unsigned char */
         *(unsigned char*)defined->ref = (unsigned char)val;
         G__setvalue(defined, (unsigned char)val);
         break;
      case 'n': /* long long */
         *(G__int64*)defined->ref = (G__int64)val;
         G__setvalue(defined, (G__int64)val);
         break;
      case 'm': /* unsigned long long */
         *(G__uint64*)defined->ref = (G__uint64)val;
         G__setvalue(defined, (G__uint64)val);
         break;
      case 'q': /* unsigned G__int64 */
         *(long double*)defined->ref = (long double)val;
         G__setvalue(defined, (long double)val);
         break;
      case 'g': /* bool */
         // --
#ifdef G__BOOL4BYTE
         *(int*)defined->ref = (int)(val ? 1 : 0);
#else // G__BOOL4BYTE
         *(unsigned char*)defined->ref = (unsigned char)(val ? 1 : 0);
#endif // G__BOOL4BYTE
         G__setvalue(defined, (bool)val);
         break;
      default:
         G__genericerror("Invalid operation and assignment, G__doubleassignbyref");
         break;
   }
}

//______________________________________________________________________________
void Cint::Internal::G__intassignbyref(G__value* defined, G__int64 val)
{
   // -- FIXME: Describe this function!
   if (isupper(G__get_type(G__value_typenum(*defined)))) {
      if (defined->ref) *(long*)defined->ref = (long)val;
      defined->obj.i = (long)val;
      return;
   }

   switch (G__get_type(G__value_typenum(*defined))) {
      case 'i': /* int */
         if (defined->ref) *(int*)defined->ref = (int)val;
         G__setvalue(defined, (int)val);
         break;
      case 'c': /* char */
         if (defined->ref) *(char*)defined->ref = (char)val;
         G__setvalue(defined, (char)val);
         break;
      case 'l': /* long */
         if (defined->ref) *(long*)defined->ref = (long)val;
         G__setvalue(defined, (long)val);
         break;
      case 's': /* short */
         if (defined->ref) *(short*)defined->ref = (short)val;
         G__setvalue(defined, (short)val);
         break;
      case 'k': /* unsigned long */
         if (defined->ref) *(unsigned long*)defined->ref = (unsigned long)val;
         G__setvalue(defined, (unsigned long)val);
         break;
      case 'h': /* unsigned int */
         if (defined->ref) *(unsigned int*)defined->ref = (unsigned int)val;
         G__setvalue(defined, (unsigned int)val);
         break;
      case 'r': /* unsigned short */
         if (defined->ref) *(unsigned short*)defined->ref = (unsigned short)val;
         G__setvalue(defined, (unsigned short)val);
         break;
      case 'b': /* unsigned char */
         if (defined->ref) *(unsigned char*)defined->ref = (unsigned char)val;
         G__setvalue(defined, (unsigned char)val);
         break;
      case 'n': /* long long */
         if (defined->ref) *(G__int64*)defined->ref = (G__int64)val;
         G__setvalue(defined, (G__int64)val);
         break;
      case 'm': /* long long */
         if (defined->ref) *(G__uint64*)defined->ref = (G__uint64)val;
         G__setvalue(defined, (G__uint64)val);
         break;
      case 'q': /* long double */
         if (defined->ref) *(long double*)defined->ref = (long double)val;
         G__setvalue(defined, (long double)val);
         break;
      case 'g': /* bool */
         // --
#ifdef G__BOOL4BYTE
         if (defined->ref) *(int*)defined->ref = (int)(val ? 1 : 0);
#else // G__BOOL4BYTE
         if (defined->ref) *(unsigned char*)defined->ref = (unsigned char)(val ? 1 : 0);
#endif // G__BOOL4BYTE
         G__setvalue(defined, (bool)val);
         break;
      case 'd': /* double */
         if (defined->ref) *(double*)defined->ref = (double)val;
         G__setvalue(defined, (double)val);
         break;
      case 'f': /* float */
         if (defined->ref) *(float*)defined->ref = (float)val;
         G__setvalue(defined, (float)val);
         break;
      default:
         G__genericerror("Invalid operation and assignment, G__intassignbyref");
         break;
   }
}

//______________________________________________________________________________
void Cint::Internal::G__bstore(int operatortag, G__value expressionin, G__value* defined)
{
   // -- FIXME: Describe this function!
   int ig2 = 0;
   long lresult = 0L;
   double fdefined = 0.0;
   double fexpression = 0.0;
   //
   // If one of the parameter is struct(class) type, call user defined operator function.
   // Assignment operators (=,+=,-=) do not work in this way.
   //

   char defined_type = G__get_type(G__value_typenum(*defined));
   char expressionin_type = G__get_type(G__value_typenum(expressionin));
   
   if (
      (defined_type == 'u') ||
      (expressionin_type == 'u')
   ) {
      G__overloadopr(operatortag, expressionin, defined);
      return;
   }
#ifdef G__ASM
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
      if (defined_type == '\0') {
         /****************************
          * OP1 instruction
          ****************************/
         switch (operatortag) {
            case '~':
            case '!':
            case '-':
            case G__OPR_POSTFIXINC:
            case G__OPR_POSTFIXDEC:
            case G__OPR_PREFIXINC:
            case G__OPR_PREFIXDEC:
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  if (isprint(operatortag)) {
                     G__fprinterr(G__serr, "%3x,%3x: OP1 '%c' %d  %s:%d\n", G__asm_cp, G__asm_dt, operatortag, operatortag, __FILE__, __LINE__);
                  }
                  else {
                     G__fprinterr(G__serr, "%3x,%3x: OP1 %d  %s:%d\n", G__asm_cp, G__asm_dt, operatortag, __FILE__, __LINE__);
                  }
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__OP1;
               G__asm_inst[G__asm_cp+1] = G__op1_operator_detail(operatortag, &expressionin);
               G__inc_cp_asm(2, 0);
               break;
         }
      }
      else {
         /****************************
          * OP2 instruction
          ****************************/
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            if (isprint(operatortag)) {
               G__fprinterr(G__serr, "%3x,%3x: OP2 '%c' %d  %s:%d\n", G__asm_cp, G__asm_dt, operatortag, operatortag, __FILE__, __LINE__);
            }
            else {
               G__fprinterr(G__serr, "%3x,%3x: OP2 %d  %s:%d\n", G__asm_cp, G__asm_dt, operatortag, __FILE__, __LINE__);
            }
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__OP2;
         G__asm_inst[G__asm_cp+1] = G__op2_operator_detail(operatortag, defined, &expressionin);
         G__inc_cp_asm(2, 0);
      }
   }
#endif // G__ASM
   if (G__no_exec_compile || G__no_exec) {
      if (G__isdouble(expressionin)) {
         expressionin.obj.d = 0.0;
      }
      else {
         expressionin.obj.i = 0;
      }
      if (G__isdouble(*defined)) {
         defined->obj.d = 0.0;
      }
      else {
         defined->obj.i = 0;
      }
   }
   if ('q' == defined_type || 'q' == expressionin_type) {
      /****************************************************************
       * long double operator long double
       ****************************************************************/
      long double lddefined = G__Longdouble(*defined);
      long double ldexpression = G__Longdouble(expressionin);
      switch (operatortag) {
         case '\0':
            defined->ref = expressionin.ref;
            G__letLongdouble(defined, 'q', lddefined + ldexpression);
            break;
         case '+': /* add */
            G__letLongdouble(defined, 'q', lddefined + ldexpression);
            defined->ref = 0;
            break;
         case '-': /* subtract */
            G__letLongdouble(defined, 'q', lddefined - ldexpression);
            defined->ref = 0;
            break;
         case '*': /* multiply */
            if (defined_type == G__get_type(G__value_typenum(G__null))) lddefined = 1;
            G__letLongdouble(defined, 'q', lddefined*ldexpression);
            defined->ref = 0;
            break;
         case '/': /* divide */
            if (defined_type == G__get_type(G__value_typenum(G__null))) lddefined = 1;
            if (ldexpression == 0) {
               if (G__no_exec_compile) G__letdouble(defined, 'i', 0);
               else G__genericerror("Error: operator '/' divided by zero");
               return;
            }
            G__letLongdouble(defined, 'q', lddefined / ldexpression);
            defined->ref = 0;
            break;
         case '>':
            if (defined_type == G__get_type(G__value_typenum(G__null))) {
               G__letLongdouble(defined, 'i', 0 > ldexpression);
            }
            else
               G__letint(defined, 'i', lddefined > ldexpression);
            defined->ref = 0;
            break;
         case '<':
            if (defined_type == G__get_type(G__value_typenum(G__null))) {
               G__letdouble(defined, 'i', 0 < ldexpression);
            }
            else
               G__letint(defined, 'i', lddefined < ldexpression);
            defined->ref = 0;
            break;
         case '!':
            if (ldexpression == 0) G__letint(defined, 'i', 1);
            else                G__letint(defined, 'i', 0);
            defined->ref = 0;
            break;
         case 'E': /* == */
            if (defined_type == G__get_type(G__value_typenum(G__null)))
               G__letLongdouble(defined, 'q', 0); /* Expression should be false wben the var is not defined */
            else
               G__letint(defined, 'i', lddefined == ldexpression);
            defined->ref = 0;
            break;
         case 'N': /* != */
            if (defined_type == G__get_type(G__value_typenum(G__null)))
               G__letLongdouble(defined, 'q', 1); /* Expression should be true wben the var is not defined */
            else
               G__letint(defined, 'i', lddefined != ldexpression);
            defined->ref = 0;
            break;
         case 'G': /* >= */
            if (defined_type == G__get_type(G__value_typenum(G__null)))
               G__letLongdouble(defined, 'q', 0); /* Expression should be false wben the var is not defined */
            else
               G__letint(defined, 'i', lddefined >= ldexpression);
            defined->ref = 0;
            break;
         case 'l': /* <= */
            if (defined_type == G__get_type(G__value_typenum(G__null)))
               G__letLongdouble(defined, 'q', 0); /* Expression should be false wben the var is not defined */
            else
               G__letint(defined, 'i', lddefined <= ldexpression);
            defined->ref = 0;
            break;
         case G__OPR_ADDASSIGN:
            if (!G__no_exec_compile && defined->ref)
               G__doubleassignbyref(defined, lddefined + ldexpression);
            break;
         case G__OPR_SUBASSIGN:
            if (!G__no_exec_compile && defined->ref)
               G__doubleassignbyref(defined, lddefined - ldexpression);
            break;
         case G__OPR_MODASSIGN:
            if (!G__no_exec_compile && defined->ref)
               G__doubleassignbyref(defined
                                    , (double)((long)lddefined % (long)ldexpression));
            break;
         case G__OPR_MULASSIGN:
            if (!G__no_exec_compile && defined->ref)
               G__doubleassignbyref(defined, lddefined*ldexpression);
            break;
         case G__OPR_DIVASSIGN:
            if (!G__no_exec_compile && defined->ref)
               G__doubleassignbyref(defined, lddefined / ldexpression);
            break;
         case G__OPR_ANDASSIGN:
            if (!G__no_exec_compile && defined->ref)
               G__doubleassignbyref(defined
                                    , (double)((long)lddefined && (long)ldexpression));
            break;
         case G__OPR_ORASSIGN:
            if (!G__no_exec_compile && defined->ref)
               G__doubleassignbyref(defined
                                    , (double)((long)lddefined || (long)ldexpression));
            break;
      }
   }
   else if ((G__isdouble(expressionin)) || (G__isdouble(*defined))) {
      /****************************************************************
       * double operator double
       * double operator int
       * int    operator double
       ****************************************************************/
      fexpression = G__double(expressionin);
      fdefined = G__double(*defined);
      // Note Cint 5 code was reseting the typedef value (result3.typenum)
      switch (operatortag) {
         case '\0':
            defined->ref = expressionin.ref;
            G__letdouble(defined, 'd', fdefined + fexpression);
            break;
         case '+': /* add */
            G__letdouble(defined, 'd', fdefined + fexpression);
            defined->ref = 0;
            break;
         case '-': /* subtract */
            G__letdouble(defined, 'd', fdefined - fexpression);
            defined->ref = 0;
            break;
         case '*': /* multiply */
            if (defined_type == G__get_type(G__value_typenum(G__null))) fdefined = 1.0;
            G__letdouble(defined, 'd', fdefined*fexpression);
            defined->ref = 0;
            break;
         case '/': /* divide */
            if (defined_type == G__get_type(G__value_typenum(G__null))) fdefined = 1.0;
            if (fexpression == 0.0) {
               if (G__no_exec_compile) G__letdouble(defined, 'd', 0.0);
               else G__genericerror("Error: operator '/' divided by zero");
               return;
            }
            G__letdouble(defined, 'd', fdefined / fexpression);
            defined->ref = 0;
            break;
#ifdef G__NONANSIOPR
         case '%': /* modulus */
            if (fexpression == 0.0) {
               if (G__no_exec_compile) G__letdouble(defined, 'd', 0.0);
               else G__genericerror("Error: operator '%%' divided by zero");
               return;
            }
            G__letint(defined, 'i', (long)fdefined % (long)fexpression);
            defined->ref = 0;
            break;
#endif /* G__NONANSIOPR */
         case '&': /* binary and */
            /* Don't know why but this one has a problem if deleted */
            if (defined_type == G__get_type(G__value_typenum(G__null))) {
               G__letint(defined, 'i', (long)fexpression);
            }
            else {
               G__letint(defined, 'i', (long)fdefined&(long)fexpression);
               defined->ref = 0;
            }
            break;
#ifdef G__NONANSIOPR
         case '|': /* binariy or */
            G__letint(defined, 'i', (long)fdefined | (long)fexpression);
            defined->ref = 0;
            break;
         case '^': /* binary exclusive or */
            G__letint(defined, 'i', (long)fdefined ^(long)fexpression);
            defined->ref = 0;
            break;
         case '~': /* binary inverse */
            G__letint(defined, 'i', ~(long)fexpression);
            defined->ref = 0;
            break;
#endif /* G__NONANSIOPR */
         case 'A': /* logic and */
            /* printf("\n!!! %g && %g\n"); */
            G__letint(defined, 'i', 0.0 != fdefined && 0.0 != fexpression);
            defined->ref = 0;
            break;
         case 'O': /* logic or */
            G__letint(defined, 'i', 0.0 != fdefined || 0.0 != fexpression);
            defined->ref = 0;
            break;
         case '>':
            if (defined_type == G__get_type(G__value_typenum(G__null))) {
               G__letdouble(defined, 'i', 0 > fexpression);
            }
            else
               G__letint(defined, 'i', fdefined > fexpression);
            defined->ref = 0;
            break;
         case '<':
            if (defined_type == G__get_type(G__value_typenum(G__null))) {
               G__letdouble(defined, 'i', 0 < fexpression);
            }
            else
               G__letint(defined, 'i', fdefined < fexpression);
            defined->ref = 0;
            break;
#ifdef G__NONANSIOPR
         case 'R': /* right shift */
            G__letint(defined, 'i', (long)fdefined >> (long)fexpression);
            defined->ref = 0;
            break;
         case 'L': /* left shift */
            G__letint(defined, 'i', (long)fdefined << (long)fexpression);
            defined->ref = 0;
            break;
#endif /* G__NONANSIOPR */
         case '@': /* power */
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "Warning: Power operator, Cint special extension");
               G__printlinenum();
            }
            if (fdefined > 0.0) {
               /* G__letdouble(defined,'d' ,exp(fexpression*log(fdefined))); */
               G__letdouble(defined, 'd' , pow(fdefined, fexpression));
            }
            else if (fdefined == 0.0) {
               if (fexpression == 0.0) G__letdouble(defined, 'd' , 1.0);
               else                 G__letdouble(defined, 'd' , 0.0);
            }
            else if (/* fmod(fdefined,1.0)==0 && */ fmod(fexpression, 1.0) == 0 &&
                                                    fexpression >= 0) {
               double fresult = 1.0;
               for (ig2 = 0;ig2 < fexpression;ig2++) fresult *= fdefined;
               G__letdouble(defined, 'd', fresult);
               defined->ref = 0;
            }
            else {
               if (G__no_exec_compile) G__letdouble(defined, 'd', 0.0);
               else G__genericerror("Error: operator '@' or '**' negative operand");
               return;
            }
            defined->ref = 0;
            break;
         case '!':
            G__letint(defined, 'i', !fexpression);
            defined->ref = 0;
            break;
         case 'E': /* == */
            if (defined_type == G__get_type(G__value_typenum(G__null)))
               G__letdouble(defined, 'i', 0 == fexpression);
            else
               G__letint(defined, 'i', fdefined == fexpression);
            defined->ref = 0;
            break;
         case 'N': /* != */
            if (defined_type == G__get_type(G__value_typenum(G__null)))
               G__letdouble(defined, 'i', 0 != fexpression);
            else
               G__letint(defined, 'i', fdefined != fexpression);
            defined->ref = 0;
            break;
         case 'G': /* >= */
            if (defined_type == G__get_type(G__value_typenum(G__null)))
               G__letdouble(defined, 'i', 0 >= fexpression);
            else
               G__letint(defined, 'i', fdefined >= fexpression);
            defined->ref = 0;
            break;
         case 'l': /* <= */
            if (defined_type == G__get_type(G__value_typenum(G__null)))
               G__letdouble(defined, 'i', 0 <= fexpression);
            else
               G__letint(defined, 'i', fdefined <= fexpression);
            defined->ref = 0;
            break;
         case G__OPR_ADDASSIGN:
            if (!G__no_exec_compile && defined->ref)
               G__doubleassignbyref(defined, fdefined + fexpression);
            break;
         case G__OPR_SUBASSIGN:
            if (!G__no_exec_compile && defined->ref)
               G__doubleassignbyref(defined, fdefined - fexpression);
            break;
         case G__OPR_MODASSIGN:
            if (!G__no_exec_compile && defined->ref)
               G__doubleassignbyref(defined
                                    , (double)((long)fdefined % (long)fexpression));
            break;
         case G__OPR_MULASSIGN:
            if (!G__no_exec_compile && defined->ref)
               G__doubleassignbyref(defined, fdefined*fexpression);
            break;
         case G__OPR_DIVASSIGN:
            if (!G__no_exec_compile && defined->ref)
               G__doubleassignbyref(defined, fdefined / fexpression);
            break;
         case G__OPR_ANDASSIGN:
            if (!G__no_exec_compile && defined->ref)
               G__doubleassignbyref(defined
                                    , (double)((long)fdefined && (long)fexpression));
            break;
         case G__OPR_ORASSIGN:
            if (!G__no_exec_compile && defined->ref)
               G__doubleassignbyref(defined
                                    , (double)((long)fdefined || (long)fexpression));
            break;
         case G__OPR_POSTFIXINC:
            if (!G__no_exec_compile && expressionin.ref) {
               *defined = expressionin;
               G__doubleassignbyref(&expressionin, fexpression + 1);
               defined->ref = 0;
            }
            break;
         case G__OPR_POSTFIXDEC:
            if (!G__no_exec_compile && expressionin.ref) {
               *defined = expressionin;
               G__doubleassignbyref(&expressionin, fexpression - 1);
               defined->ref = 0;
            }
            break;
         case G__OPR_PREFIXINC:
            if (!G__no_exec_compile && expressionin.ref) {
               G__doubleassignbyref(&expressionin, fexpression + 1);
               *defined = expressionin;
            }
            break;
         case G__OPR_PREFIXDEC:
            if (!G__no_exec_compile && expressionin.ref) {
               G__doubleassignbyref(&expressionin, fexpression - 1);
               *defined = expressionin;
            }
            break;
         default:
            G__fprinterr(G__serr, "Error: %s ", G__getoperatorstring(operatortag));
            G__genericerror("Illegal operator for real number");
            break;
      }
   }
   else if (isupper(defined_type) || isupper(expressionin_type)) {
      /****************************************************************
      * pointer operator pointer
      * pointer operator int
      * int     operator pointer
      ****************************************************************/
      G__CHECK(G__SECURE_POINTER_CALC, '+' == operatortag || '-' == operatortag, return);
      if (isupper(defined_type)) {
         /*
          *  pointer - pointer , integer [==] pointer
          */
         if (isupper(expressionin_type)) {
            switch (operatortag) {
               case '\0': /* add */
                  defined->ref = expressionin.ref;
                  defined->obj.i = defined->obj.i + expressionin.obj.i;
                  //REMOVED: defined->obj.reftype.reftype = expressionin.obj.reftype.reftype;
                  break;
               case '-': /* subtract */
                  defined->obj.i
                  = (defined->obj.i - expressionin.obj.i) / G__sizeof_deref(defined);
                  G__value_typenum(*defined) = G__get_from_type('i', 0);
                  defined->ref = 0;
                  break;
               case 'E': /* == */
                  if ('U' == defined_type && 'U' == expressionin_type)
                     G__publicinheritance(defined, &expressionin);
                  G__letint(defined, 'i', defined->obj.i == expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case 'N': /* != */
                  if ('U' == defined_type && 'U' == expressionin_type)
                     G__publicinheritance(defined, &expressionin);
                  G__letint(defined, 'i', defined->obj.i != expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case 'G': /* >= */
                  G__letint(defined, 'i', defined->obj.i >= expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case 'l': /* <= */
                  G__letint(defined, 'i', defined->obj.i <= expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case '>': /* > */
                  G__letint(defined, 'i', defined->obj.i > expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case '<': /* < */
                  G__letint(defined, 'i', defined->obj.i < expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case 'A': /* logical and */
                  G__letint(defined, 'i', defined->obj.i && expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case 'O': /* logical or */
                  G__letint(defined, 'i', defined->obj.i || expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case G__OPR_SUBASSIGN:
                  if (!G__no_exec_compile && defined->ref)
                     G__intassignbyref(defined
                                       , (defined->obj.i - expressionin.obj.i)
                                       / G__sizeof_deref(defined));
                  break;
               default:
                  if (G__ASM_FUNC_NOP == G__asm_wholefunction) {
                     G__fprinterr(G__serr, "Error: %s ", G__getoperatorstring(operatortag));
                  }
                  G__genericerror("Illegal operator for pointer 1");
                  break;
            }
         }
         /*
          *  pointer [+-==] integer , 
          */
         else {
            switch (operatortag) {
               case '\0': /* no op */
                  defined->ref = expressionin.ref;
                  //REMOVED: defined->obj.reftype.reftype = expressionin.obj.reftype.reftype;
               case '+': /* add */
                  defined->obj.i = defined->obj.i + expressionin.obj.i * G__sizeof_deref(defined);
                  defined->ref = 0;
                  break;
               case '-': /* subtract */
                  defined->obj.i = defined->obj.i - expressionin.obj.i * G__sizeof_deref(defined);
                  defined->ref = 0;
                  break;
               case '!':
                  G__letint(defined, 'i', !expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case 'E': /* == */
                  G__letint(defined, 'i', defined->obj.i == expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case 'N': /* != */
                  G__letint(defined, 'i', defined->obj.i != expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case 'G': /* >= */
                  G__letint(defined, 'i', defined->obj.i >= expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case 'l': /* <= */
                  G__letint(defined, 'i', defined->obj.i <= expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case '>': /* > */
                  G__letint(defined, 'i', defined->obj.i > expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case '<': /* < */
                  G__letint(defined, 'i', defined->obj.i < expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case 'A': /* logical and */
                  G__letint(defined, 'i', defined->obj.i && expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case 'O': /* logical or */
                  G__letint(defined, 'i', defined->obj.i || expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case G__OPR_ADDASSIGN:
                  if (!G__no_exec_compile && defined->ref)
                     G__intassignbyref(defined
                                       , defined->obj.i + expressionin.obj.i*G__sizeof_deref(defined));
                  break;
               case G__OPR_SUBASSIGN:
                  if (!G__no_exec_compile && defined->ref)
                     G__intassignbyref(defined
                                       , defined->obj.i - expressionin.obj.i*G__sizeof_deref(defined));
                  break;
               default:
                  G__fprinterr(G__serr, "Error: %s ", G__getoperatorstring(operatortag));
                  G__genericerror("Illegal operator for pointer 2");
                  break;
            }
         }
      }
      else {
         /*
          *  integer [+-] pointer 
          */
         switch (operatortag) {
            case '\0': /* subtract */
               defined->ref = expressionin.ref;
               defined->obj.i = defined->obj.i * G__sizeof_deref(defined) + expressionin.obj.i;
               G__value_typenum(*defined) = G__value_typenum(expressionin);
               break;
            case '+': /* add */
               defined->ref = 0;
               defined->obj.i = defined->obj.i * G__sizeof_deref(defined) + expressionin.obj.i;
               G__value_typenum(*defined) = G__value_typenum(expressionin);
               break;
            case '-': /* subtract */
               defined->ref = 0;
               defined->obj.i = defined->obj.i * G__sizeof_deref(defined) - expressionin.obj.i;
               G__value_typenum(*defined) = G__value_typenum(expressionin);
               break;
            case '!':
               G__letint(defined, 'i', !expressionin.obj.i);
               defined->ref = 0;
               break;
            case 'E': /* == */
               G__letint(defined, 'i', defined->obj.i == expressionin.obj.i);
               defined->ref = 0;
               break;
            case 'N': /* != */
               G__letint(defined, 'i', defined->obj.i != expressionin.obj.i);
               defined->ref = 0;
               break;
            case 'A': /* logical and */
               G__letint(defined, 'i', defined->obj.i && expressionin.obj.i);
               defined->ref = 0;
               break;
            case 'O': /* logical or */
               G__letint(defined, 'i', defined->obj.i || expressionin.obj.i);
               defined->ref = 0;
               break;
            case G__OPR_ADDASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined
                                    , defined->obj.i*G__sizeof_deref(defined) + expressionin.obj.i);
               break;
            case G__OPR_SUBASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined
                                    , defined->obj.i*G__sizeof_deref(defined) - expressionin.obj.i);
               break;
            case G__OPR_POSTFIXINC:
               if (!G__no_exec_compile && expressionin.ref) {
                  *defined = expressionin;
                  G__intassignbyref(&expressionin
                                    , expressionin.obj.i + G__sizeof_deref(&expressionin));
                  defined->ref = 0;
               }
               break;
            case G__OPR_POSTFIXDEC:
               if (!G__no_exec_compile && expressionin.ref) {
                  *defined = expressionin;
                  G__intassignbyref(&expressionin
                                    , expressionin.obj.i - G__sizeof_deref(&expressionin));
                  defined->ref = 0;
               }
               break;
            case G__OPR_PREFIXINC:
               if (!G__no_exec_compile && expressionin.ref) {
                  G__intassignbyref(&expressionin
                                    , expressionin.obj.i + G__sizeof_deref(&expressionin));
                  *defined = expressionin;
               }
               break;
            case G__OPR_PREFIXDEC:
               if (!G__no_exec_compile && expressionin.ref) {
                  G__intassignbyref(&expressionin
                                    , expressionin.obj.i - G__sizeof_deref(&expressionin));
                  *defined = expressionin;
               }
               break;
            default:
               G__fprinterr(G__serr, "Error: %s ", G__getoperatorstring(operatortag));
               G__genericerror("Illegal operator for pointer 3");
               break;
         }
      }
   }
   else if (
      'n' == defined_type ||
      'm' == defined_type ||
      'n' == expressionin_type ||
      'm' == expressionin_type
   ) {
      /****************************************************************
       * long long operator long long
       ****************************************************************/
      int unsignedresult = 0;
      if (
         'm' == defined_type ||
         'm' == expressionin_type
      ) {
         unsignedresult = -1;
      }

      if (unsignedresult) {
         G__uint64 ulldefined = G__ULonglong(*defined);
         G__uint64 ullexpression = G__ULonglong(expressionin);
         switch (operatortag) {
            case '\0':
               defined->ref = expressionin.ref;
               G__letULonglong(defined, 'm', ulldefined + ullexpression);
               break;
            case '+': /* add */
               G__letULonglong(defined, 'm', ulldefined + ullexpression);
               defined->ref = 0;
               break;
            case '-': /* subtract */
               G__letULonglong(defined, 'm', ulldefined - ullexpression);
               defined->ref = 0;
               break;
            case '*': /* multiply */
               if (defined_type == G__get_type(G__value_typenum(G__null))) ulldefined = 1;
               G__letULonglong(defined, 'm', ulldefined*ullexpression);
               defined->ref = 0;
               break;
            case '/': /* divide */
               if (defined_type == G__get_type(G__value_typenum(G__null))) ulldefined = 1;
               if (ullexpression == 0) {
                  if (G__no_exec_compile) G__letdouble(defined, 'i', 0);
                  else G__genericerror("Error: operator '/' divided by zero");
                  return;
               }
               G__letULonglong(defined, 'm', ulldefined / ullexpression);
               defined->ref = 0;
               break;
            case '%': /* modulus */
               if (ullexpression == 0) {
                  if (G__no_exec_compile) G__letdouble(defined, 'i', 0);
                  else G__genericerror("Error: operator '%%' divided by zero");
                  return;
               }
               G__letULonglong(defined, 'm', ulldefined % ullexpression);
               defined->ref = 0;
               break;
            case '&': /* binary and */
               if (defined_type == G__get_type(G__value_typenum(G__null))) {
                  G__letULonglong(defined, 'm', ullexpression);
               }
               else {
                  G__letULonglong(defined, 'm', ulldefined&ullexpression);
               }
               defined->ref = 0;
               break;
            case '|': /* binary or */
               G__letULonglong(defined, 'm', ulldefined | ullexpression);
               defined->ref = 0;
               break;
            case '^': /* binary exclusive or */
               G__letULonglong(defined, 'm', ulldefined ^ ullexpression);
               defined->ref = 0;
               break;
            case '~': /* binary inverse */
               G__letULonglong(defined, 'm', ~ullexpression);
               defined->ref = 0;
               break;
            case 'A': /* logical and */
               G__letULonglong(defined, 'm', ulldefined && ullexpression);
               defined->ref = 0;
               break;
            case 'O': /* logical or */
               G__letULonglong(defined, 'm', ulldefined || ullexpression);
               defined->ref = 0;
               break;
            case '>':
               if (defined_type == G__get_type(G__value_typenum(G__null))) {
                  G__letULonglong(defined, 'm', 0);
               }
               else
                  G__letint(defined, 'i', ulldefined > ullexpression);
               defined->ref = 0;
               break;
            case '<':
               if (defined_type == G__get_type(G__value_typenum(G__null))) {
                  G__letULonglong(defined, 'm', 0);
               }
               else
                  G__letint(defined, 'i', ulldefined < ullexpression);
               defined->ref = 0;
               break;
            case 'R': /* right shift */
               switch (defined_type) {
                  case 'b':
                  case 'r':
                  case 'h':
                  case 'k': {
                     G__letULonglong(defined, 'm', ulldefined >> ullexpression);
                  }
                  break;
                  default:
                     G__letULonglong(defined, 'm', ulldefined >> ullexpression);
                     break;
               }
               defined->ref = 0;
               break;
            case 'L': /* left shift */
               G__letULonglong(defined, 'm', ulldefined << ullexpression);
               defined->ref = 0;
               break;
            case '!':
               G__letULonglong(defined, 'm', !ullexpression);
               defined->ref = 0;
               break;
            case 'E': /* == */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letULonglong(defined, 'm', 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, 'i', ulldefined == ullexpression);
               defined->ref = 0;
               break;
            case 'N': /* != */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letULonglong(defined, 'm', 1); /* Expression should be true wben the var is not defined */
               else
                  G__letint(defined, 'i', ulldefined != ullexpression);
               defined->ref = 0;
               break;
            case 'G': /* >= */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letULonglong(defined, 'm', 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, 'i', ulldefined >= ullexpression);
               defined->ref = 0;
               break;
            case 'l': /* <= */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letULonglong(defined, 'm', 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, 'i', ulldefined <= ullexpression);
               defined->ref = 0;
               break;
            case G__OPR_ADDASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ulldefined + ullexpression);
               break;
            case G__OPR_SUBASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ulldefined - ullexpression);
               break;
            case G__OPR_MODASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ulldefined % ullexpression);
               break;
            case G__OPR_MULASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ulldefined*ullexpression);
               break;
            case G__OPR_DIVASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ulldefined / ullexpression);
               break;
            case G__OPR_RSFTASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ulldefined >> ullexpression);
               break;
            case G__OPR_LSFTASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ulldefined << ullexpression);
               break;
            case G__OPR_BANDASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ulldefined&ullexpression);
               break;
            case G__OPR_BORASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ulldefined | ullexpression);
               break;
            case G__OPR_EXORASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ulldefined ^ ullexpression);
               break;
            case G__OPR_ANDASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ulldefined && ullexpression);
               break;
            case G__OPR_ORASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ulldefined || ullexpression);
               break;
            case G__OPR_POSTFIXINC:
               if (!G__no_exec_compile && expressionin.ref) {
                  *defined = expressionin;
                  G__intassignbyref(&expressionin, ullexpression + 1);
                  defined->ref = 0;
               }
               break;
            case G__OPR_POSTFIXDEC:
               if (!G__no_exec_compile && expressionin.ref) {
                  *defined = expressionin;
                  G__intassignbyref(&expressionin, ullexpression - 1);
                  defined->ref = 0;
               }
               break;
            case G__OPR_PREFIXINC:
               if (!G__no_exec_compile && expressionin.ref) {
                  G__intassignbyref(&expressionin, ullexpression + 1);
                  *defined = expressionin;
               }
               break;
            case G__OPR_PREFIXDEC:
               if (!G__no_exec_compile && expressionin.ref) {
                  G__intassignbyref(&expressionin, ullexpression - 1);
                  *defined = expressionin;
               }
               break;
            default:
               G__fprinterr(G__serr, "Error: %s ", G__getoperatorstring(operatortag));
               G__genericerror("Illegal operator for integer");
               break;
         }
      }
      else {
         G__int64 lldefined = G__Longlong(*defined);
         G__int64 llexpression = G__Longlong(expressionin);
         switch (operatortag) {
            case '\0':
               defined->ref = expressionin.ref;
               G__letLonglong(defined, 'n', lldefined + llexpression);
               break;
            case '+': /* add */
               G__letLonglong(defined, 'n', lldefined + llexpression);
               defined->ref = 0;
               break;
            case '-': /* subtract */
               G__letLonglong(defined, 'n', lldefined - llexpression);
               defined->ref = 0;
               break;
            case '*': /* multiply */
               if (defined_type == G__get_type(G__value_typenum(G__null))) lldefined = 1;
               G__letLonglong(defined, 'n', lldefined*llexpression);
               defined->ref = 0;
               break;
            case '/': /* divide */
               if (defined_type == G__get_type(G__value_typenum(G__null))) lldefined = 1;
               if (llexpression == 0) {
                  if (G__no_exec_compile) G__letdouble(defined, 'i', 0);
                  else G__genericerror("Error: operator '/' divided by zero");
                  return;
               }
               G__letLonglong(defined, 'n', lldefined / llexpression);
               defined->ref = 0;
               break;
            case '%': /* modulus */
               if (llexpression == 0) {
                  if (G__no_exec_compile) G__letdouble(defined, 'i', 0);
                  else G__genericerror("Error: operator '%%' divided by zero");
                  return;
               }
               G__letLonglong(defined, 'n', lldefined % llexpression);
               defined->ref = 0;
               break;
            case '&': /* binary and */
               if (defined_type == G__get_type(G__value_typenum(G__null))) {
                  G__letLonglong(defined, 'n', llexpression);
               }
               else {
                  G__letLonglong(defined, 'i', lldefined&llexpression);
               }
               defined->ref = 0;
               break;
            case '|': /* binary or */
               G__letLonglong(defined, 'i', lldefined | llexpression);
               defined->ref = 0;
               break;
            case '^': /* binary exclusive or */
               G__letULonglong(defined, 'n', lldefined ^ llexpression);
               defined->ref = 0;
               break;
            case '~': /* binary inverse */
               G__letULonglong(defined, 'n', ~llexpression);
               defined->ref = 0;
               break;
            case 'A': /* logical and */
               G__letint(defined, 'i', lldefined && llexpression);
               defined->ref = 0;
               break;
            case 'O': /* logical or */
               G__letint(defined, 'i', lldefined || llexpression);
               defined->ref = 0;
               break;
            case '>':
               if (defined_type == G__get_type(G__value_typenum(G__null))) {
                  G__letLonglong(defined, 'n', 0);
               }
               else
                  G__letint(defined, 'i', lldefined > llexpression);
               defined->ref = 0;
               break;
            case '<':
               if (defined_type == G__get_type(G__value_typenum(G__null))) {
                  G__letLonglong(defined, 'n', 0);
               }
               else
                  G__letint(defined, 'i', lldefined < llexpression);
               defined->ref = 0;
               break;
            case 'R': /* right shift */
               switch (defined_type) {
                  case 'b':
                  case 'r':
                  case 'h':
                  case 'k': {
                     G__letLonglong(defined, 'n', lldefined >> llexpression);
                  }
                  break;
                  default:
                     G__letLonglong(defined, 'n', lldefined >> llexpression);
                     break;
               }
               defined->ref = 0;
               break;
            case 'L': /* left shift */
               G__letLonglong(defined, 'n', lldefined << llexpression);
               defined->ref = 0;
               break;
            case '!':
               G__letLonglong(defined, 'n', !llexpression);
               defined->ref = 0;
               break;
            case 'E': /* == */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letLonglong(defined, 'n', 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, 'i', lldefined == llexpression);
               defined->ref = 0;
               break;
            case 'N': /* != */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letLonglong(defined, 'n', 1); /* Expression should be true wben the var is not defined */
               else
                  G__letint(defined, 'i', lldefined != llexpression);
               defined->ref = 0;
               break;
            case 'G': /* >= */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letLonglong(defined, 'n', 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, 'i', lldefined >= llexpression);
               defined->ref = 0;
               break;
            case 'l': /* <= */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letLonglong(defined, 'n', 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, 'i', lldefined <= llexpression);
               defined->ref = 0;
               break;
            case G__OPR_ADDASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, lldefined + llexpression);
               break;
            case G__OPR_SUBASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, lldefined - llexpression);
               break;
            case G__OPR_MODASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, lldefined % llexpression);
               break;
            case G__OPR_MULASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, lldefined*llexpression);
               break;
            case G__OPR_DIVASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, lldefined / llexpression);
               break;
            case G__OPR_RSFTASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, lldefined >> llexpression);
               break;
            case G__OPR_LSFTASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, lldefined << llexpression);
               break;
            case G__OPR_BANDASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, lldefined&llexpression);
               break;
            case G__OPR_BORASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, lldefined | llexpression);
               break;
            case G__OPR_EXORASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, lldefined ^ llexpression);
               break;
            case G__OPR_ANDASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, lldefined && llexpression);
               break;
            case G__OPR_ORASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, lldefined || llexpression);
               break;
            case G__OPR_POSTFIXINC:
               if (!G__no_exec_compile && expressionin.ref) {
                  *defined = expressionin;
                  G__intassignbyref(&expressionin, llexpression + 1);
                  defined->ref = 0;
               }
               break;
            case G__OPR_POSTFIXDEC:
               if (!G__no_exec_compile && expressionin.ref) {
                  *defined = expressionin;
                  G__intassignbyref(&expressionin, llexpression - 1);
                  defined->ref = 0;
               }
               break;
            case G__OPR_PREFIXINC:
               if (!G__no_exec_compile && expressionin.ref) {
                  G__intassignbyref(&expressionin, llexpression + 1);
                  *defined = expressionin;
               }
               break;
            case G__OPR_PREFIXDEC:
               if (!G__no_exec_compile && expressionin.ref) {
                  G__intassignbyref(&expressionin, llexpression - 1);
                  *defined = expressionin;
               }
               break;
            default:
               G__fprinterr(G__serr, "Error: %s ", G__getoperatorstring(operatortag));
               G__genericerror("Illegal operator for integer");
               break;
         }
      }
   }
   else {
      /****************************************************************
       * int operator int
       ****************************************************************/
      int unsignedresult = 0;
      switch (defined_type) {
         case 'h':
         case 'k':
            unsignedresult = -1;
            break;
      }
      switch (expressionin_type) {
         case 'h':
         case 'k':
            unsignedresult = -1;
            break;
      }
      bool useLong = (defined_type + expressionin_type > 'i' + 'i');
      if (!defined_type) {
         useLong = expressionin_type > 'i';
      }
      char resultTypeChar = useLong ? 'l' : 'i';
      if (unsignedresult) {
         --resultTypeChar;
      }
      
      if (unsignedresult) {
         unsigned long udefined = (unsigned long)G__uint(*defined);
         unsigned long uexpression = (unsigned long)G__uint(expressionin);
         switch (operatortag) {
            case '\0':
               defined->ref = expressionin.ref;
               G__letint(defined, resultTypeChar, udefined + uexpression);
               break;
            case '+': /* add */
               G__letint(defined, resultTypeChar, udefined + uexpression);
               defined->ref = 0;
               break;
            case '-': /* subtract */
               G__letint(defined, resultTypeChar, udefined - uexpression);
               defined->ref = 0;
               break;
            case '*': /* multiply */
               if (defined_type == G__get_type(G__value_typenum(G__null))) udefined = 1;
               G__letint(defined, resultTypeChar, udefined*uexpression);
               defined->ref = 0;
               break;
            case '/': /* divide */
               if (defined_type == G__get_type(G__value_typenum(G__null))) udefined = 1;
               if (uexpression == 0) {
                  if (G__no_exec_compile) G__letdouble(defined, resultTypeChar, 0);
                  else G__genericerror("Error: operator '/' divided by zero");
                  return;
               }
               G__letint(defined, resultTypeChar, udefined / uexpression);
               defined->ref = 0;
               break;
            case '%': /* modulus */
               if (uexpression == 0) {
                  if (G__no_exec_compile) G__letdouble(defined, resultTypeChar, 0);
                  else G__genericerror("Error: operator '%%' divided by zero");
                  return;
               }
               G__letint(defined, resultTypeChar, udefined % uexpression);
               defined->ref = 0;
               break;
            case '&': /* binary and */
               if (defined_type == G__get_type(G__value_typenum(G__null))) {
                  G__letint(defined, resultTypeChar, uexpression);
               }
               else {
                  G__letint(defined, resultTypeChar, udefined&uexpression);
               }
               defined->ref = 0;
               break;
            case '|': /* binary or */
               G__letint(defined, resultTypeChar, udefined | uexpression);
               defined->ref = 0;
               break;
            case '^': /* binary exclusive or */
               G__letint(defined, resultTypeChar, udefined ^ uexpression);
               defined->ref = 0;
               break;
            case '~': /* binary inverse */
               G__letint(defined, resultTypeChar, ~uexpression);
               defined->ref = 0;
               break;
            case 'A': /* logical and */
               G__letint(defined, resultTypeChar, udefined && uexpression);
               defined->ref = 0;
               break;
            case 'O': /* logical or */
               G__letint(defined, resultTypeChar, udefined || uexpression);
               defined->ref = 0;
               break;
            case '>':
               if (defined_type == G__get_type(G__value_typenum(G__null))) {
                  G__letint(defined, resultTypeChar, 0);
               }
               else
                  G__letint(defined, resultTypeChar, udefined > uexpression);
               defined->ref = 0;
               break;
            case '<':
               if (defined_type == G__get_type(G__value_typenum(G__null))) {
                  G__letint(defined, resultTypeChar, 0);
               }
               else
                  G__letint(defined, resultTypeChar, udefined < uexpression);
               defined->ref = 0;
               break;
            case 'R': /* right shift */
               switch (defined_type) {
                  case 'b':
                  case 'r':
                  case 'h':
                  case 'k': {
                     unsigned long uudefined = udefined;
                     G__letint(defined, 'k', uudefined >> uexpression);
                  }
                  break;
                  default:
                     G__letint(defined, resultTypeChar, udefined >> uexpression);
                     break;
               }
               defined->ref = 0;
               break;
            case 'L': /* left shift */
               G__letint(defined, resultTypeChar, udefined << uexpression);
               defined->ref = 0;
               break;
            case '@': /* power */
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "Warning: Power operator, Cint special extension");
                  G__printlinenum();
               }
               fdefined = 1.0;
               for (ig2 = 1;ig2 <= (int)uexpression;ig2++) fdefined *= udefined;
               if (fdefined > (double)LONG_MAX || fdefined < (double)LONG_MIN) {
                  G__genericerror("Error: integer overflow. Use 'double' for power operator");
               }
               lresult = (long)fdefined;
               G__letint(defined, resultTypeChar, lresult);
               defined->ref = 0;
               break;
            case '!':
               G__letint(defined, resultTypeChar, !uexpression);
               defined->ref = 0;
               break;
            case 'E': /* == */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letint(defined, resultTypeChar, 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, resultTypeChar, udefined == uexpression);
               defined->ref = 0;
               break;
            case 'N': /* != */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letint(defined, resultTypeChar, 1); /* Expression should be true wben the var is not defined */
               else
                  G__letint(defined, resultTypeChar, udefined != uexpression);
               defined->ref = 0;
               break;
            case 'G': /* >= */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letint(defined, resultTypeChar, 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, resultTypeChar, udefined >= uexpression);
               defined->ref = 0;
               break;
            case 'l': /* <= */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letint(defined, resultTypeChar, 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, resultTypeChar, udefined <= uexpression);
               defined->ref = 0;
               break;
            case G__OPR_ADDASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, udefined + uexpression);
               break;
            case G__OPR_SUBASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, udefined - uexpression);
               break;
            case G__OPR_MODASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, udefined % uexpression);
               break;
            case G__OPR_MULASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, udefined*uexpression);
               break;
            case G__OPR_DIVASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, udefined / uexpression);
               break;
            case G__OPR_RSFTASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, udefined >> uexpression);
               break;
            case G__OPR_LSFTASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, udefined << uexpression);
               break;
            case G__OPR_BANDASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, udefined&uexpression);
               break;
            case G__OPR_BORASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, udefined | uexpression);
               break;
            case G__OPR_EXORASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, udefined ^ uexpression);
               break;
            case G__OPR_ANDASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, udefined && uexpression);
               break;
            case G__OPR_ORASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, udefined || uexpression);
               break;
            case G__OPR_POSTFIXINC:
               if (!G__no_exec_compile && expressionin.ref) {
                  *defined = expressionin;
                  G__intassignbyref(&expressionin, uexpression + 1);
                  defined->ref = 0;
               }
               break;
            case G__OPR_POSTFIXDEC:
               if (!G__no_exec_compile && expressionin.ref) {
                  *defined = expressionin;
                  G__intassignbyref(&expressionin, uexpression - 1);
                  defined->ref = 0;
               }
               break;
            case G__OPR_PREFIXINC:
               if (!G__no_exec_compile && expressionin.ref) {
                  G__intassignbyref(&expressionin, uexpression + 1);
                  *defined = expressionin;
               }
               break;
            case G__OPR_PREFIXDEC:
               if (!G__no_exec_compile && expressionin.ref) {
                  G__intassignbyref(&expressionin, uexpression - 1);
                  *defined = expressionin;
               }
               break;
            default:
               G__fprinterr(G__serr, "Error: %s ", G__getoperatorstring(operatortag));
               G__genericerror("Illegal operator for integer");
               break;
         }
      }
      else {
         long ldefined = G__int(*defined);
         long lexpression = G__int(expressionin);
         switch (operatortag) {
            case '\0':
               defined->ref = expressionin.ref;
               G__letint(defined, resultTypeChar, ldefined + lexpression);
               break;
            case '+': /* add */
               G__letint(defined, resultTypeChar, ldefined + lexpression);
               defined->ref = 0;
               break;
            case '-': /* subtract */
               G__letint(defined, resultTypeChar, ldefined - lexpression);
               defined->ref = 0;
               break;
            case '*': /* multiply */
               if (defined_type == G__get_type(G__value_typenum(G__null))) ldefined = 1;
               G__letint(defined, resultTypeChar, ldefined*lexpression);
               defined->ref = 0;
               break;
            case '/': /* divide */
               if (defined_type == G__get_type(G__value_typenum(G__null))) ldefined = 1;
               if (lexpression == 0) {
                  if (G__no_exec_compile) G__letdouble(defined, resultTypeChar, 0);
                  else G__genericerror("Error: operator '/' divided by zero");
                  return;
               }
               G__letint(defined, resultTypeChar, ldefined / lexpression);
               defined->ref = 0;
               break;
            case '%': /* modulus */
               if (lexpression == 0) {
                  if (G__no_exec_compile) G__letdouble(defined, resultTypeChar, 0);
                  else G__genericerror("Error: operator '%%' divided by zero");
                  return;
               }
               G__letint(defined, resultTypeChar, ldefined % lexpression);
               defined->ref = 0;
               break;
            case '&': /* binary and */
               if (defined_type == G__get_type(G__value_typenum(G__null))) {
                  G__letint(defined, resultTypeChar, lexpression);
               }
               else {
                  G__letint(defined, resultTypeChar, ldefined&lexpression);
               }
               defined->ref = 0;
               break;
            case '|': /* binary or */
               G__letint(defined, resultTypeChar, ldefined | lexpression);
               defined->ref = 0;
               break;
            case '^': /* binary exclusive or */
               G__letint(defined, resultTypeChar, ldefined ^ lexpression);
               defined->ref = 0;
               break;
            case '~': /* binary inverse */
               G__letint(defined, resultTypeChar, ~lexpression);
               defined->ref = 0;
               break;
            case 'A': /* logical and */
               G__letint(defined, resultTypeChar, ldefined && lexpression);
               defined->ref = 0;
               break;
            case 'O': /* logical or */
               G__letint(defined, resultTypeChar, ldefined || lexpression);
               defined->ref = 0;
               break;
            case '>':
               if (defined_type == G__get_type(G__value_typenum(G__null))) {
                  G__letint(defined, resultTypeChar, 0);
               }
               else
                  G__letint(defined, resultTypeChar, ldefined > lexpression);
               defined->ref = 0;
               break;
            case '<':
               if (defined_type == G__get_type(G__value_typenum(G__null))) {
                  G__letint(defined, resultTypeChar, 0);
               }
               else
                  G__letint(defined, resultTypeChar, ldefined < lexpression);
               defined->ref = 0;
               break;
            case 'R': /* right shift */
               if (!G__prerun) {
                  unsigned long udefined = (unsigned long)G__uint(*defined);
                  unsigned long uexpression = (unsigned long)G__uint(expressionin);
                  G__letint(defined, resultTypeChar, udefined >> uexpression);
                  defined->obj.ulo = udefined >> uexpression;
               }
               else {
                  G__letint(defined, resultTypeChar, ldefined >> lexpression);
               }
               defined->ref = 0;
               break;
            case 'L': /* left shift */
               if (!G__prerun) {
                  long local_ldefined = G__int(*defined);
                  unsigned long uexpression = (unsigned long) G__uint(expressionin);
                  G__letint(defined, defined_type, local_ldefined << uexpression);
                  defined->obj.i = local_ldefined << uexpression;
               }
               else {
                  G__letint(defined, resultTypeChar, ldefined << lexpression);
               }
               defined->ref = 0;
               break;
            case '@': /* power */
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "Warning: Power operator, Cint special extension");
                  G__printlinenum();
               }
               fdefined = 1.0;
               for (ig2 = 1;ig2 <= lexpression;ig2++) fdefined *= ldefined;
               if (fdefined > (double)LONG_MAX || fdefined < (double)LONG_MIN) {
                  G__genericerror("Error: integer overflow. Use 'double' for power operator");
               }
               lresult = (long)fdefined;
               G__letint(defined, resultTypeChar, lresult);
               defined->ref = 0;
               break;
            case '!':
               G__letint(defined, resultTypeChar, !lexpression);
               defined->ref = 0;
               break;
            case 'E': /* == */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letint(defined, resultTypeChar, 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, resultTypeChar, ldefined == lexpression);
               defined->ref = 0;
               break;
            case 'N': /* != */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letint(defined, resultTypeChar, 1); /* Expression should be true wben the var is not defined */
               else
                  G__letint(defined, resultTypeChar, ldefined != lexpression);
               defined->ref = 0;
               break;
            case 'G': /* >= */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letint(defined, resultTypeChar, 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, resultTypeChar, ldefined >= lexpression);
               defined->ref = 0;
               break;
            case 'l': /* <= */
               if (defined_type == G__get_type(G__value_typenum(G__null)))
                  G__letint(defined, resultTypeChar, 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, resultTypeChar, ldefined <= lexpression);
               defined->ref = 0;
               break;
            case G__OPR_ADDASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ldefined + lexpression);
               break;
            case G__OPR_SUBASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ldefined - lexpression);
               break;
            case G__OPR_MODASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ldefined % lexpression);
               break;
            case G__OPR_MULASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ldefined*lexpression);
               break;
            case G__OPR_DIVASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ldefined / lexpression);
               break;
            case G__OPR_RSFTASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ldefined >> lexpression);
               break;
            case G__OPR_LSFTASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ldefined << lexpression);
               break;
            case G__OPR_BANDASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ldefined&lexpression);
               break;
            case G__OPR_BORASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ldefined | lexpression);
               break;
            case G__OPR_EXORASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ldefined ^ lexpression);
               break;
            case G__OPR_ANDASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ldefined && lexpression);
               break;
            case G__OPR_ORASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined, ldefined || lexpression);
               break;
            case G__OPR_POSTFIXINC:
               if (!G__no_exec_compile && expressionin.ref) {
                  *defined = expressionin;
                  G__intassignbyref(&expressionin, lexpression + 1);
                  defined->ref = 0;
               }
               break;
            case G__OPR_POSTFIXDEC:
               if (!G__no_exec_compile && expressionin.ref) {
                  *defined = expressionin;
                  G__intassignbyref(&expressionin, lexpression - 1);
                  defined->ref = 0;
               }
               break;
            case G__OPR_PREFIXINC:
               if (!G__no_exec_compile && expressionin.ref) {
                  G__intassignbyref(&expressionin, lexpression + 1);
                  *defined = expressionin;
               }
               break;
            case G__OPR_PREFIXDEC:
               if (!G__no_exec_compile && expressionin.ref) {
                  G__intassignbyref(&expressionin, lexpression - 1);
                  *defined = expressionin;
               }
               break;
            default:
               G__fprinterr(G__serr, "Error: %s ", G__getoperatorstring(operatortag));
               G__genericerror("Illegal operator for integer");
               break;
         }
      }
   }
   if (G__no_exec_compile && !defined_type) {
      *defined = expressionin;
   }
}

//______________________________________________________________________________
int Cint::Internal::G__scopeoperator(char* name /* name is modified and this is intentional */, int* phash, char** pstruct_offset, int* ptagnum)
{
   // -- FIXME: Describe this function!
   char* scope;
   char* member;
   ::Reflex::Scope scopetagnum;
   int offset;
   int offset_sum;
   int i;
   G__StrBuf temp_sb(G__MAXNAME*2);
   char *temp = temp_sb;
   re_try_after_std:
   /* search for pattern "::" */
   char* pc = G__find_first_scope_operator(name);

   /* no scope operator, return */
   char* pparen = strchr(name, '(');
   if (!pc || !strncmp(name, "operator ", 9) || (pparen && (pparen < pc))) {
      G__fixedscope = 0;
      return G__NOSCOPEOPR;
   }
   G__fixedscope = 1;
   // If scope operator found at the beginning of the name,
   // global scope or fully qualified scope!
   if (pc == name) {
      // strip scope operator, set hash and return
      strcpy(temp, name + 2);
      strcpy(name, temp);
      G__hash(name, *phash, i);
      // If we do no have anymore scope operator,
      // we know the request of for the global name space.
      pc = G__find_first_scope_operator(name);
      if (!pc) {
         return G__GLOBALSCOPE;
      }
   }
#ifndef G__STD_NAMESPACE
   if (!strncmp(name, "std::", 5) && G__ignore_stdnamespace) {
      // Strip scope operator, set hash and return
      strcpy(temp, name + 5);
      strcpy(name, temp);
      G__hash(name, (*phash), i)
      goto re_try_after_std;
   }
#endif // G__STD_NAMESPACE
   // otherwise, specific class scope
   offset_sum = 0;
   strcpy(temp, name);
   if (*name == '~') {
      scope = name + 1; // ~A::B() explicit destructor
   }
   else {
      scope = name;
   }
   // Recursive scope operator is not allowed in compiler but possible in cint.
   scopetagnum = G__get_envtagnum();
   do {
      ::Reflex::Scope save_tagdefining = G__tagdefining;
      ::Reflex::Scope save_def_tagnum = G__def_tagnum;
      G__tagdefining = scopetagnum;
      G__def_tagnum = scopetagnum;
      member = pc + 2;
      *pc = '\0';
      scopetagnum = G__Dict::GetDict().GetScope(G__defined_tagname(scope, 1));
      G__tagdefining = save_tagdefining;
      G__def_tagnum = save_def_tagnum;

#ifdef G__VIRTUALBASE
      offset = G__ispublicbase(G__get_tagnum(scopetagnum), *ptagnum, (void*)(*pstruct_offset + offset_sum));
      if (offset == -1) {
         ::Reflex::Scope store_tagnum = G__tagnum;
         G__tagnum = G__Dict::GetDict().GetScope(*ptagnum);
         offset = -G__find_virtualoffset(G__get_tagnum(scopetagnum));
         G__tagnum = store_tagnum;
      }
#else // G__VIRTUALBASE
      offset = G__ispublicbase(G__get_tagnum(scopetagnum), *ptagnum);
      if (offset == -1) {
         offset = 0;
      }
#endif // G__VIRTUALBASE
      *ptagnum = G__get_tagnum(scopetagnum);
      offset_sum += offset;
      scope = member;
   }
   while ((pc = G__find_first_scope_operator(scope)));
   *pstruct_offset += offset_sum;
#ifdef G__ASM
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: ADDSTROS 0x%x,%d  %s:%d\n", G__asm_cp, G__asm_dt, offset_sum, offset_sum, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__ADDSTROS;
      G__asm_inst[G__asm_cp+1] = offset_sum;
      G__inc_cp_asm(2, 0);
   }
#endif // G__ASM
   strcpy(temp, member);
   if (*name == '~') {
      // -- explicit destructor
      strcpy(name + 1, temp);
   }
   else {
      strcpy(name, temp);
   }
   G__hash(name, *phash, i);
   return G__CLASSSCOPE;
}

//______________________________________________________________________________
int Cint::Internal::G__cmp(G__value buf1, G__value buf2)
{
   // FIXME: Describe this function!
   switch (G__get_type(G__value_typenum(buf1))) {
      case 'a':  // G__start
      case 'z':  // G__default
      case '\0': // G__null
         if (G__get_type(G__value_typenum(buf1)) == G__get_type(G__value_typenum(buf2))) {
            return 1;
         }
         return 0;
      case 'd': // double
      case 'f': // float
         if (G__double(buf1) == G__double(buf2)) {
            return 1;
         }
         return 0;
   }
   if (G__int(buf1) == G__int(buf2)) {
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__getunaryop(char unaryop, char* expression, char* buf, G__value* preg)
{
   // FIXME: Describe this function!
   *preg = G__null;
   char prodpower = 0; // product or power operator seen
   int nest = 0; // nesting level of parenthsis, brace, and square bracket
   int i2 = 0;
   for (int i1 = 1;; ++i1) { // Loop over chars in expression
      int c = expression[i1];
      switch (c) {
         case '-':
            if (G__isexponent(buf, i2)) {
               buf[i2++] = c;
               break;
            }
         case '+':
         case '>':
         case '<':
         case '!':
         case '&':
         case '|':
         case '^':
         case '\0':
            if (!nest) {
               buf[i2] = '\0';
               G__value reg;
               if (prodpower) {
                  reg = G__getprod(buf);
               }
               else {
                  reg = G__getitem(buf);
               }
               G__bstore(unaryop, reg, preg);
               return i1;
            }
            buf[i2++] = c;
            break;
         case '*':
         case '/':
         case '%':
         case '@':
         case '~':
         case ' ':
            if (!nest) {
               prodpower = 1;
            }
            break;
         case '(':
         case '[':
         case '{':
            ++nest;
            break;
         case ')':
         case ']':
         case '}':
            --nest;
            break;
         default:
            buf[i2++] = c;
            break;
      }
   }
}

#ifdef G__VIRTUALBASE
//______________________________________________________________________________
int Cint::Internal::G__iosrdstate(G__value* pios)
{
   // -- ios rdstate condition test
   G__StrBuf buf_sb(G__MAXNAME);
   char *buf = buf_sb;
   G__value result;
   int ig2;
   char *store_struct_offset;
   ::Reflex::Scope store_tagnum;
   int rdstateflag = 0;

   if (G__value_typenum(*pios).IsEnum()) return(pios->obj.i);

   /* store member function call environment */
   store_struct_offset = G__store_struct_offset;
   store_tagnum = G__tagnum;
   G__store_struct_offset = (char*)pios->obj.i;
   G__set_G__tagnum(*pios);
#ifdef G__ASM
   if (G__asm_noverflow) {
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__asm_inst[G__asm_cp+1] = G__SETSTROS;
      G__inc_cp_asm(2, 0);
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp - 2, G__asm_dt, __FILE__, __LINE__);
         G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp - 1, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
   }
#endif // G__ASM

   /* call ios::rdstate() */
   sprintf(buf, "rdstate()" /* ,pios->obj.i */);
   result = G__getfunction(buf, &ig2, G__TRYMEMFUNC);
   if (ig2) rdstateflag = 1;

   if (0 == ig2) {
      sprintf(buf, "operator int()" /* ,pios->obj.i */);
      result = G__getfunction(buf, &ig2, G__TRYMEMFUNC);
   }
   if (0 == ig2) {
      sprintf(buf, "operator bool()" /* ,pios->obj.i */);
      result = G__getfunction(buf, &ig2, G__TRYMEMFUNC);
   }
   if (0 == ig2) {
      sprintf(buf, "operator long()" /* ,pios->obj.i */);
      result = G__getfunction(buf, &ig2, G__TRYMEMFUNC);
   }
   if (0 == ig2) {
      sprintf(buf, "operator short()" /* ,pios->obj.i */);
      result = G__getfunction(buf, &ig2, G__TRYMEMFUNC);
   }
   if (0 == ig2) {
      sprintf(buf, "operator char*()" /* ,pios->obj.i */);
      result = G__getfunction(buf, &ig2, G__TRYMEMFUNC);
   }
   if (0 == ig2) {
      sprintf(buf, "operator const char*()" /* ,pios->obj.i */);
      result = G__getfunction(buf, &ig2, G__TRYMEMFUNC);
   }

   /* restore environment */
   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;

#ifdef G__ASM
   if (G__asm_noverflow && rdstateflag) {
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      if (G__asm_dbg) G__fprinterr(G__serr, "%3x,%3x: OP1 '!'  %s:%d\n", G__asm_cp + 1, G__asm_dt, __FILE__, __LINE__);
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1, 0);
      G__asm_inst[G__asm_cp] = G__OP1;
      G__asm_inst[G__asm_cp+1] = '!';
      G__inc_cp_asm(2, 0);
   }
#endif // G__ASM

   /* test result */
   if (ig2) {
      if (rdstateflag) return(!result.obj.i);
      else            return(result.obj.i);
   }
   else {
      G__genericerror("Limitation: Cint does not support full iostream functionality in this platform");
      return(0);
   }
}
#endif // G__VIRTUALBASE

//______________________________________________________________________________
int Cint::Internal::G__overloadopr(int operatortag, G__value expressionin, G__value* defined)
{
   // Search for and run an operator function.  Try member function, then global function.
   //
   // Inputs:
   //
   // operatortag: one character token for operator
   // expressionin: first argument, which is the this pointer for a member function
   // defined: second argument, has type code 0 for an unary operator
   //
   // Outputs:
   //
   // defined: return value of operator function
   //
   char expr[G__LONGLINE];
   char* store_struct_offset;
   ::Reflex::Scope store_tagnum;
   int store_isconst;
   G__value buffer;
   char* pos;
   int postfixflag = 0;
   int store_asm_cp = 0;
   //
   //  Exit early on special operatortag.
   //
   if (!operatortag) {
      *defined = expressionin;
      return 0;
   }
   //
   //  Get short version of operator function name
   //  based on operatortag.
   //
   char opr[12]; // short version of operator function name
   switch (operatortag) {
      case '+':
      case '-':
      case '*':
      case '/':
      case '%':
      case '&':
      case '|':
      case '^':
      case '~':
      case '>':
      case '<':
      case '@': // power
      case '!':
         sprintf(opr, "operator%c", operatortag);
         break;
      case 'A': // &&, boolean and
         sprintf(opr, "operator&&");
         break;
      case 'O': // ||, boolean or
         sprintf(opr, "operator||");
         break;
      case 'R': // >>, right bit shift
         sprintf(opr, "operator>>");
         break;
      case 'L': // <<, left bit shift
         sprintf(opr, "operator<<");
         break;
      case 'E': // ==, equality
         sprintf(opr, "operator==");
         break;
      case 'N': // !=, inequality
         sprintf(opr, "operator!=");
         break;
      case 'G': // >=, greater than or equal to
         sprintf(opr, "operator>=");
         break;
      case 'l': // <=, less than or equal to
         sprintf(opr, "operator<=");
         break;
      case G__OPR_ADDASSIGN: // +=
         sprintf(opr, "operator+=");
         break;
      case G__OPR_SUBASSIGN: // -=
         sprintf(opr, "operator-=");
         break;
      case G__OPR_MODASSIGN: // %=
         sprintf(opr, "operator%%=");
         break;
      case G__OPR_MULASSIGN: // *=
         sprintf(opr, "operator*=");
         break;
      case G__OPR_DIVASSIGN: // /=
         sprintf(opr, "operator/=");
         break;
      case G__OPR_RSFTASSIGN: // >>=
         sprintf(opr, "operator>>=");
         break;
      case G__OPR_LSFTASSIGN: // <<=
         sprintf(opr, "operator<<=");
         break;
      case G__OPR_BANDASSIGN: // &=
         sprintf(opr, "operator&=");
         break;
      case G__OPR_BORASSIGN: // |=
         sprintf(opr, "operator|=");
         break;
      case G__OPR_EXORASSIGN: // ^=
         sprintf(opr, "operator^=");
         break;
      case G__OPR_ANDASSIGN: // &&=  FIXME: There is no such thing in C++!
         sprintf(opr, "operator&&=");
         break;
      case G__OPR_ORASSIGN: // ||=  FIXME: There is no such thing in C++!= 
         sprintf(opr, "operator||=");
         break;
      case G__OPR_POSTFIXINC: // x++, operator++(x&, int)
      case G__OPR_PREFIXINC: // ++x, operator++(x&)
         sprintf(opr, "operator++");
         break;
      case G__OPR_POSTFIXDEC: // x--, operator--(x&, int)
      case G__OPR_PREFIXDEC: // --x, operator--(x&)
         sprintf(opr, "operator--");
         break;
      default:
         G__genericerror("Limitation: Can't handle combination of overloading operators");
         return 0;
   }
   //
   //  Now try to find operator function and run it.
   //
   if (!G__get_type(G__value_typenum(*defined))) { // There is no second argument, so unary operator.
      //
      //  Valid operatortag and exit if not valid.
      //
      switch (operatortag) {
         case '-': // -x  NOTE: +x is missing!
         case '!': // !x
         case '~': // ~x
         case G__OPR_POSTFIXINC: // x++
         case G__OPR_POSTFIXDEC: // x--
         case G__OPR_PREFIXINC: // ++x
         case G__OPR_PREFIXDEC: // --x
            break;
         default: // Invalid unary operator.
            *defined = expressionin;
            return 0;
      }
      // Flag we are doing operator overloading.
      G__oprovld = 1;
#ifdef G__ASM
      if (G__asm_noverflow) { // Making bytecode, PUSHSTROS and SETSTROS
         // -- We are generating bytecode.
         store_asm_cp = G__asm_cp;
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp - 2, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__PUSHSTROS;
         G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp - 1, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__SETSTROS;
         G__inc_cp_asm(1, 0);
      }
#endif // G__ASM
      //
      //  Search for member function.
      //
      //--
      //
      //  Create member function name.
      //
      switch (operatortag) {
         case G__OPR_POSTFIXINC:
         case G__OPR_POSTFIXDEC:
            sprintf(expr, "%s(1)", opr);
#ifdef G__ASM
            if (G__asm_noverflow) { // Making bytecode, LD 1 for postfix dummy int arg
               // -- We are generating bytecode.
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD 1 from %lx  %s:%d\n", G__asm_cp, G__asm_dt, G__asm_dt, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD;
               G__asm_inst[G__asm_cp+1] = G__asm_dt;
               G__asm_stack[G__asm_dt] = G__one;
               G__inc_cp_asm(2, 1);
               postfixflag = 1;
            }
#endif // G__ASM
            break;
         default:
            postfixflag = 0;
            sprintf(expr, "%s()", opr);
            break;
      }
      // Save state.
      store_struct_offset = G__store_struct_offset;
      store_tagnum = G__tagnum;
      // Switch scopes and set this pointer.
      G__store_struct_offset = (char*) expressionin.obj.i; // The object this pointer is the first argument.
      G__set_G__tagnum(expressionin); // Scope to search is the class of the first argument.
      //
      //  Search for member function
      //  and run it if found.
      //
      int known = 0;
      buffer = G__getfunction(expr, &known, G__TRYUNARYOPR); // Search for and run member function.
      // Restore state.
      G__store_struct_offset = store_struct_offset;
      G__tagnum = store_tagnum;
#ifdef G__ASM
      if (known && G__asm_noverflow) { // We ran the member function, and we are making bytecode, pop structure offset.
         // -- We are generating bytecode.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__POPSTROS;
         G__inc_cp_asm(1, 0);
      }
      if (!known && G__asm_noverflow) { // No member function, making byecode, cancel LD and PUSHSTROS, SETSTROS
         // -- We are generating bytecode.
         if (postfixflag) {
            G__inc_cp_asm(-2, -1);
            postfixflag = 0;
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "LD cancelled  %s:%d\n", __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            // --
         }
         G__inc_cp_asm(store_asm_cp - G__asm_cp, 0);
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "PUSHSTROS,SETSTROS cancelled  %s:%d\n", __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         // --
      }
#endif // G__ASM
      if (!known) { // Member function not found, try global function.
         //
         //  Search for global function.
         //
         //--
         //
         //  Create global operator function name.
         //
         char arg1[G__LONGLINE];
         switch (operatortag) {
            case G__OPR_POSTFIXINC:
            case G__OPR_POSTFIXDEC:
               sprintf(expr, "%s(%s,1)", opr, G__setiparseobject(&expressionin, arg1));
#ifdef G__ASM
               if (G__asm_noverflow) { // Making bytecode, LD 1 for postfix dummy int arg
                  // -- We are generating bytecode.
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x,%3x: LD 1 from %lx  %s:%d\n", G__asm_cp, G__asm_dt, G__asm_dt, __FILE__, __LINE__);
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__LD;
                  G__asm_inst[G__asm_cp+1] = G__asm_dt;
                  G__asm_stack[G__asm_dt] = G__one;
                  G__inc_cp_asm(2, 1);
               }
#endif // G__ASM
               break;
            default:
               sprintf(expr, "%s(%s)", opr, G__setiparseobject(&expressionin, arg1));
               break;
         }
         //
         //  Search for global operator function
         //  and run it if found.
         //
         buffer = G__getfunction(expr, &known, G__TRYNORMAL); // Search for and run global function.
      }
      *defined = buffer; // Set return value.
      G__oprovld = 0; // Flag operator overload is finished.
      return 0;
   }
   //
   //  Binary operator.
   //
   G__oprovld = 1; // Flag we are doing operator overloading.
#ifdef G__ASM
   if (G__asm_noverflow) { // Making bytecode, SWAP, PUSHSTROS, SETSTROS
      // -- We are generating bytecode.
#ifdef G__ASM_IFUNC
      store_asm_cp = G__asm_cp;
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: SWAP  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__SWAP;
      G__inc_cp_asm(1, 0);
#endif // G__ASM_IFUNC
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp - 2, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp - 1, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__SETSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   int known = 0;
   //
   //  Prepare the second argument.
   //
   char arg2[G__LONGLINE];
   if (G__get_type(G__value_typenum(expressionin)) == 'u') { // If second arg is of class type, use the temporary
      G__setiparseobject(&expressionin, arg2);
   }
   else { // Otherwise, convert the value to a string
      G__valuemonitor(expressionin, arg2);
      if (expressionin.ref && (expressionin.ref != 1)) {
         pos = strchr(arg2, ')');
         *pos = '\0';
         if (expressionin.ref < 0) {
            sprintf(expr, "*%s*)(%ld)", arg2, expressionin.ref);
         }
         else {
            sprintf(expr, "*%s*)%ld", arg2, expressionin.ref);
         }
         strcpy(arg2, expr);
      } else if (G__get_type(expressionin) == 'm') {
         strcat(arg2, "ULL");
      }
      else if (G__get_type(expressionin) == 'n') {
         strcat(arg2, "LL");
      }
   }
   if (G__get_type(G__value_typenum(*defined)) == 'u') { // If first arg is of class type, try a member function.
      //
      // Search for member function.
      //
      sprintf(expr, "%s(%s)", opr, arg2); // Make the member function name.
      // Save state.
      store_struct_offset = G__store_struct_offset;
      store_tagnum = G__tagnum;
      store_isconst = G__isconst;
      // Set the this pointer, and the scope to search.
      G__store_struct_offset = (char*) defined->obj.i; // Set the this pointer.
      G__set_G__tagnum(*defined); // Set the scope to search.
      G__isconst = G__get_isconst(G__value_typenum(*defined));
      //
      //  Search for member function and
      //  run it if found.
      //
      buffer = G__getfunction(expr, &known, G__TRYBINARYOPR); // Try to run member function.
      // Restore state.
      G__isconst = store_isconst;
      G__store_struct_offset = store_struct_offset;
      G__tagnum = store_tagnum;
   }
#ifdef G__ASM
   if (known && G__asm_noverflow) { // We found it as a member function, and making bytecode, POPSTROS
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   //
   //  Try a global function if no
   //  member function found.
   //
   if (!known) {
      //
      // Search for global function.
      //
      //--
#ifdef G__ASM
      if (G__asm_noverflow) { // Making bytecode, cancel PUSHSTROS, SETSTROS
         // -- We are generating bytecode.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "PUSHSTROS,SETSTROS cancelled  %s:%d\n", __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__inc_cp_asm(store_asm_cp - G__asm_cp, 0);
      }
#endif // G__ASM
      //
      //  Prepare the first argument.
      //
      char arg1[G__LONGLINE];
      if (G__get_type(G__value_typenum(*defined)) == 'u') { // If first arg is of class type, use the temporary
         G__setiparseobject(defined, arg1);
      }
      else { // Otherwise, convert the value to a string
         G__valuemonitor(*defined, arg1);
         if (defined->ref) {
            pos = strchr(arg1, ')');
            *pos = '\0';
            if (defined->ref < 0) {
               sprintf(expr, "*%s*)(%ld)", arg1, defined->ref);
            }
            else {
               sprintf(expr, "*%s*)%ld", arg1, defined->ref);
            }
            strcpy(arg1, expr);
         }
      }
      // Make the global operator function name.
      sprintf(expr, "%s(%s,%s)", opr, arg1, arg2); // Make the global operator function name.
      //
      //  Search for the global operator function
      //  and run it if found.
      //
      buffer = G__getfunction(expr, &known, G__TRYNORMAL); // Search for operator function and run it if found.
      if (!known && (G__get_tagnum(G__value_typenum(expressionin).DeclaringScope()) > 0)) { // Not found, try scope of second argument.
         sprintf(expr, "%s::%s(%s,%s)", G__value_typenum(expressionin).DeclaringScope().Name(::Reflex::SCOPED).c_str(), opr , arg1 , arg2);
         buffer = G__getfunction(expr, &known, G__TRYNORMAL); // Search for operator function and run it if found.
      }
      if (!known && (G__get_tagnum(G__value_typenum(*defined).DeclaringScope()) > 0)) { // Not found, try scope of first argument.
         sprintf(expr, "%s::%s(%s,%s)", G__value_typenum(*defined).DeclaringScope().Name(::Reflex::SCOPED).c_str(), opr , arg1 , arg2);
         buffer = G__getfunction(expr, &known, G__TRYNORMAL); // Search for operator function and run it if found.
      }
      if (!known && ((operatortag == 'A') || (operatortag == 'O'))) { // Still not found and is operator&& or operator||
         int lval, rval;
         if ('u' == G__get_type(G__value_typenum(*defined))) {
            if (G__asm_noverflow) {
               // -- We are generating bytecode.
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: SWAP  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__SWAP;
               G__inc_cp_asm(1, 0);
            }
            lval = G__iosrdstate(defined);
            if (G__asm_noverflow) {
               // -- We are generating bytecode.
#ifdef G__ASM_DBG
               if (G__asm_dbg) G__fprinterr(G__serr, "%3x,%3x: SWAP  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__SWAP;
               G__inc_cp_asm(1, 0);
            }
         }
         else {
            lval = G__int(*defined);
         }
         if ('u' == G__get_type(G__value_typenum(expressionin))) {
            rval = G__iosrdstate(&expressionin);
         }
         else {
            rval = G__int(expressionin);
         }
         buffer.ref = 0;
         G__value_typenum(buffer) = ::Reflex::Type();
         switch (operatortag) {
            case 'A':
               G__letint(&buffer, 'i', lval && rval);
               break;
            case 'O':
               G__letint(&buffer, 'i', lval || rval);
               break;
         }
         if (G__asm_noverflow) {
            // -- We are generating bytecode.
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               if (isprint(operatortag)) {
                  G__fprinterr(G__serr, "%3x,%3x: OP2 '%c' %d  %s:%d\n", G__asm_cp, G__asm_dt, operatortag, operatortag, __FILE__, __LINE__);
               }
               else {
                  G__fprinterr(G__serr, "%3x,%3x: OP2 %d  %s:%d\n", G__asm_cp, G__asm_dt, operatortag, __FILE__, __LINE__);
               }
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__OP2;
            G__asm_inst[G__asm_cp+1] = operatortag;
            G__inc_cp_asm(2, 0);
         }
         known = 1;
      }
      if (!known) { // Error, not found.
         if (G__value_typenum(*defined)) {
            G__fprinterr(G__serr, "Error: %s not defined for '%s'", opr, G__value_typenum(*defined).Name(::Reflex::SCOPED).c_str());
         }
         else {
            G__fprinterr(G__serr, "Error: '%s' not defined", expr);
         }
         G__genericerror(0);
      }
   }
   *defined = buffer; // Set return value.
   G__oprovld = 0; // Flag operator overload is done.
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__parenthesisovldobj(G__value* result3, G__value* result, const char* realname, G__param* libp, int flag)
{
   // FIXME: Describe this function!
   //
   // Note: flag controls generation of PUSHSTROS, SETSTROS in bytecode.
   //--
   // Save state.
   int store_exec_memberfunc = G__exec_memberfunc;
   ::Reflex::Scope store_memberfunc_tagnum = G__memberfunc_tagnum;
   char* store_memberfunc_struct_offset = G__memberfunc_struct_offset;
   char* store_struct_offset = G__store_struct_offset;
   ::Reflex::Scope store_tagnum = G__tagnum;
   // Set this pointer and set scope to search.
   G__store_struct_offset = (char*) result->obj.i; // Set this pointer.
   G__set_G__tagnum(*result); // Set scope to search.
#ifdef G__ASM
   if (G__asm_noverflow && !flag) { // Making bytecode and not flag, PUSHSTROS, SETSTROS
      // -- We are generating bytecode and not flag.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp + 1, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__asm_inst[G__asm_cp+1] = G__SETSTROS;
      G__inc_cp_asm(2, 0);
   }
#endif // G__ASM
   int hash = 0;
   int known = flag; // Note: Initialization is fake, we just need to use flag so compiler does not complain.
   G__hash(realname, hash, known);
   G__fixedscope = 0;
   for (int funcmatch = G__EXACT; funcmatch <= G__USERCONV; ++funcmatch) {
      if (!G__tagnum.IsTopScope()) {
         G__incsetup_memfunc(G__tagnum);
      }
      int err = G__interpret_func(result3, realname, libp, hash, G__tagnum, funcmatch, G__CALLMEMFUNC);
      if (err == 1) { // We found and ran function, all done.
         // Restore state.
         G__store_struct_offset = store_struct_offset;
         G__tagnum = store_tagnum;
#ifdef G__ASM
         if (G__asm_noverflow) { // Making bytecode, POPSTROS
            // -- We are generating bytecode.
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__POPSTROS;
            G__inc_cp_asm(1, 0);
         }
#endif // G__ASM
         // Restore state.
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         return 1;
      }
   }
   // Restore state.
   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;
#ifdef G__ASM
   if (G__asm_noverflow) { // Making bytecode, POPSTROS
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   // Restore state.
   G__exec_memberfunc = store_exec_memberfunc;
   G__memberfunc_tagnum = store_memberfunc_tagnum;
   G__memberfunc_struct_offset = store_memberfunc_struct_offset;
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__parenthesisovld(G__value* result3, const char* funcname, G__param* libp, int flag)
{
   // FIXME: Describe this function!
   int known;
   G__value result;
   char* store_struct_offset;
   ::Reflex::Scope store_tagnum;
   int funcmatch;
   int hash;
   G__StrBuf realname_sb(G__ONELINE);
   char* realname = realname_sb;
   int store_exec_memberfunc;
   ::Reflex::Scope store_memberfunc_tagnum;
   //
   char* store_memberfunc_struct_offset;
   if (!strncmp(funcname, "operator", 8) || !strcmp(funcname, "G__ateval")) {
      return 0;
   }
   if (!funcname[0]) {
      result = *result3;
   }
   else {
      if (flag == G__CALLMEMFUNC) {
         G__incsetup_memvar(G__tagnum);
         result = G__getvariable(/*FIXME*/(char*)funcname, &known,::Reflex::Scope(), G__tagnum);
      }
      else {
         result = G__getvariable(/*FIXME*/(char*)funcname, &known,::Reflex::Scope::GlobalScope(), G__p_local);
      }
   }
   // resolve A::staticmethod(1)(2,3)
   if (
      (known != 1) ||
      (
         G__get_type(G__value_typenum(result)) != 'u'
      )
   ) {
      return 0;
   }
   //
   store_exec_memberfunc = G__exec_memberfunc;
   store_memberfunc_tagnum = G__memberfunc_tagnum;
   store_memberfunc_struct_offset = G__memberfunc_struct_offset;
   store_struct_offset = G__store_struct_offset;
   store_tagnum = G__tagnum;
   //
   G__store_struct_offset = (char*) result.obj.i; // Set this pointer.
   G__set_G__tagnum(result); // Set scope to search.
#ifdef G__ASM
   if (G__asm_noverflow) { // Making byecode, PUSHSTROS, SETSTROS
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp + 1, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__asm_inst[G__asm_cp+1] = G__SETSTROS;
      G__inc_cp_asm(2, 0);
   }
#endif // G__ASM
   sprintf(realname, "operator()");
   G__hash(realname, hash, known);
   G__fixedscope = 0;
   for (funcmatch = G__EXACT; funcmatch <= G__USERCONV; ++funcmatch) {
      if (!G__tagnum.IsTopScope()) {
         G__incsetup_memfunc(G__tagnum);
      }
      //
      //  Search for member function and
      //  run it if found.
      //
      int err = G__interpret_func(result3, realname, libp, hash, G__tagnum, funcmatch, G__CALLMEMFUNC);
      if (err == 1) { // We found it and ran it, all done.
         // Restore state.
         G__store_struct_offset = store_struct_offset;
         G__tagnum = store_tagnum;
#ifdef G__ASM
         if (G__asm_noverflow) { // Making bytecode, POPSTROS
               // -- We are generating bytecode.
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif
            G__asm_inst[G__asm_cp] = G__POPSTROS;
            G__inc_cp_asm(1, 0);
         }
#endif
         // Restore state.
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         return 1;
      }
   }
   // Restore state.
   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;
#ifdef G__ASM
   if (G__asm_noverflow) { // Making bytecode, POPSTROS
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif
   // Restore state.
   G__exec_memberfunc = store_exec_memberfunc;
   G__memberfunc_tagnum = store_memberfunc_tagnum;
   G__memberfunc_struct_offset = store_memberfunc_struct_offset;
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__tryindexopr(G__value* result7, G__value* para, int paran, int ig25)
{
   // FIXME: Describe this function!
   //
   // 1) asm
   //   * G__ST_VAR/MSTR -> LD_VAR/MSTR
   //   * paran -> ig25
   // 2) try operator[]() function while ig25<paran
   //
   //--
#ifdef G__ASM
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
      //
      //  X a[2][3];
      //  Y X::operator[]()
      //  Y::operator[]()
      //    a[x][y][z][w];   stack x y z w ->  stack w z x y
      //                                             Y X a a
      //
      if ((paran > 1) && (paran > ig25)) {
         // -- We are generating bytecode.
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%x: REORDER inserted before ST_VAR/MSTR/LD_VAR/MSTR  %s:%d\n", G__asm_cp - 5, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         for (int i = 1; i <= 5; ++i) {
            G__asm_inst[G__asm_cp-i+3] = G__asm_inst[G__asm_cp-i];
         }
         G__asm_inst[G__asm_cp-5] = G__REORDER;
         G__asm_inst[G__asm_cp-4] = paran;
         G__asm_inst[G__asm_cp-3] = ig25;
         G__inc_cp_asm(3, 0);
      }
      switch (G__asm_inst[G__asm_cp-5]) {
         case G__ST_MSTR:
            G__asm_inst[G__asm_cp-5] = G__LD_MSTR;
            break;
         case G__ST_VAR:
            G__asm_inst[G__asm_cp-5] = G__LD_VAR;
            break;
         default:
            break;
      }
      G__asm_inst[G__asm_cp-3] = ig25;
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "ST_VAR/MSTR replaced with LD_VAR/MSTR, paran=%d -> %d  %s:%d\n", paran, ig25, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      // --
   }
#endif // G__ASM 
   ::Reflex::Scope store_tagnum = G__tagnum;
   ::Reflex::Type store_typenum = G__typenum;
   char* store_struct_offset = G__store_struct_offset;
#ifdef G__ASM
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   for (; ig25 < paran; ++ig25) {
      G__oprovld = 1;
      if (G__get_type(G__value_typenum(*result7)) == 'u') {
         // -- We have T operator[]
         G__set_G__tagnum(*result7);
         G__typenum = G__value_typenum(*result7);
         G__store_struct_offset = (char*)result7->obj.i;
#ifdef G__ASM
         if (G__asm_noverflow) {
            // -- We are generating bytecode.
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__SETSTROS;
            G__inc_cp_asm(1, 0);
         }
#endif // G__ASM
         G__StrBuf expr_sb(G__ONELINE);
         char *expr = expr_sb;
         G__StrBuf arg2_sb(G__MAXNAME);
         char *arg2 = arg2_sb;
         if (G__get_type(para[ig25]) == 'u') {
            G__setiparseobject(&para[ig25], arg2);
         }
         else {
            G__valuemonitor(para[ig25], arg2);
            /* This part must be fixed when reference to pointer type
             * is supported */
            if (para[ig25].ref) {
               char* pos = strchr(arg2, ')');
               *pos = '\0';
               if (para[ig25].ref < 0) {
                  sprintf(expr, "*%s*)(%ld)", arg2, para[ig25].ref);
               }
               else {
                  sprintf(expr, "*%s*)%ld", arg2, para[ig25].ref);
               }
               strcpy(arg2, expr);
            }
         }
         sprintf(expr, "operator[](%s)", arg2);
         int store_asm_exec = G__asm_exec;
         G__asm_exec = 0;
         int known = 0;
         *result7 = G__getfunction(expr, &known, G__CALLMEMFUNC);
         G__asm_exec = store_asm_exec;
      }
      else if (isupper(G__get_type(*result7))) {
         // -- We have T* operator[]
         result7->obj.i += G__sizeof_deref(result7) * para[ig25].obj.i;
#ifdef G__ASM
         if (G__asm_noverflow) {
            // -- We are generating bytecode.
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: OP2 +  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__OP2;
            G__asm_inst[G__asm_cp+1] = '+';
            G__inc_cp_asm(2, 0);
         }
#endif // G__ASM
         *result7 = G__tovalue(*result7);
      }
   }
   G__oprovld = 0;
   G__tagnum = store_tagnum;
   G__typenum = store_typenum;
   G__store_struct_offset = store_struct_offset;
#ifdef G__ASM
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM
   return 0;
}

//______________________________________________________________________________
long Cint::Internal::G__op1_operator_detail(int opr, G__value* val)
{
   // FIXME: Describe this function!
   // don't optimze if optimize level is less than 3
   if (G__asm_loopcompile < 3) {
      return opr;
   }
   if (G__get_type(*val) == 'i') {
      switch (opr) {
         case G__OPR_POSTFIXINC:
            return G__OPR_POSTFIXINC_I;
         case G__OPR_POSTFIXDEC:
            return G__OPR_POSTFIXDEC_I;
         case G__OPR_PREFIXINC:
            return G__OPR_PREFIXINC_I;
         case G__OPR_PREFIXDEC:
            return G__OPR_PREFIXDEC_I;
      }
      return opr;
   }
   if (G__get_type(*val) == 'd') {
      switch (opr) {
         case G__OPR_POSTFIXINC:
            return G__OPR_POSTFIXINC_D;
         case G__OPR_POSTFIXDEC:
            return G__OPR_POSTFIXDEC_D;
         case G__OPR_PREFIXINC:
            return G__OPR_PREFIXINC_D;
         case G__OPR_PREFIXDEC:
            return G__OPR_PREFIXDEC_D;
      }
   }
   return opr;
}

//______________________________________________________________________________
long Cint::Internal::G__op2_operator_detail(int opr, G__value* lval, G__value* rval)
{
   // FIXME: Describe this function!
   int lisdouble, risdouble;
   int lispointer, rispointer;

   /* don't optimze if optimize level is less than 3 */
   if (G__asm_loopcompile < 3) return(opr);

   switch (G__get_type(*lval)) {
      case 'q':
      case 'n':
      case 'm':
         return(opr);
   }
   switch (G__get_type(*rval)) {
      case 'q':
      case 'n':
      case 'm':
         return(opr);
   }

   if (0 == G__get_type(*rval)
         && 0 == G__xrefflag
      ) {
      G__genericerror("Error: Binary operator oprand missing");
   }

   lisdouble = G__isdouble(*lval);
   risdouble = G__isdouble(*rval);

   if (0 == lisdouble && 0 == risdouble) {
      lispointer = isupper(G__get_type(*lval));
      rispointer = isupper(G__get_type(*rval));
      if (0 == lispointer && 0 == rispointer) {
         if ('k' == G__get_type(*lval) || 'h' == G__get_type(*lval) ||
               'k' == G__get_type(*rval) || 'h' == G__get_type(*rval)) {
            switch (opr) {
               case G__OPR_ADD:
                  return(G__OPR_ADD_UU);
               case G__OPR_SUB:
                  return(G__OPR_SUB_UU);
               case G__OPR_MUL:
                  return(G__OPR_MUL_UU);
               case G__OPR_DIV:
                  return(G__OPR_DIV_UU);
               default:
                  switch (G__get_type(*lval)) {
                     case 'i':
                        switch (opr) {
                           case G__OPR_ADDASSIGN:
                              return(G__OPR_ADDASSIGN_UU);
                           case G__OPR_SUBASSIGN:
                              return(G__OPR_SUBASSIGN_UU);
                           case G__OPR_MULASSIGN:
                              return(G__OPR_MULASSIGN_UU);
                           case G__OPR_DIVASSIGN:
                              return(G__OPR_DIVASSIGN_UU);
                        }
                  }
                  break;
            }
         }
         else {
            switch (opr) {
               case G__OPR_ADD:
                  return(G__OPR_ADD_II);
               case G__OPR_SUB:
                  return(G__OPR_SUB_II);
               case G__OPR_MUL:
                  return(G__OPR_MUL_II);
               case G__OPR_DIV:
                  return(G__OPR_DIV_II);
               default:
                  switch (G__get_type(*lval)) {
                     case 'i':
                        switch (opr) {
                           case G__OPR_ADDASSIGN:
                              return(G__OPR_ADDASSIGN_II);
                           case G__OPR_SUBASSIGN:
                              return(G__OPR_SUBASSIGN_II);
                           case G__OPR_MULASSIGN:
                              return(G__OPR_MULASSIGN_II);
                           case G__OPR_DIVASSIGN:
                              return(G__OPR_DIVASSIGN_II);
                        }
                  }
                  break;
            }
         }
      }
   }
   else if (lisdouble && risdouble) {
      switch (opr) {
         case G__OPR_ADD:
            return(G__OPR_ADD_DD);
         case G__OPR_SUB:
            return(G__OPR_SUB_DD);
         case G__OPR_MUL:
            return(G__OPR_MUL_DD);
         case G__OPR_DIV:
            return(G__OPR_DIV_DD);
         default:
            switch (G__get_type(*lval)) {
               case 'd':
                  switch (opr) {
                     case G__OPR_ADDASSIGN:
                        return(G__OPR_ADDASSIGN_DD);
                     case G__OPR_SUBASSIGN:
                        return(G__OPR_SUBASSIGN_DD);
                     case G__OPR_MULASSIGN:
                        return(G__OPR_MULASSIGN_DD);
                     case G__OPR_DIVASSIGN:
                        return(G__OPR_DIVASSIGN_DD);
                  }
               case 'f':
                  switch (opr) {
                     case G__OPR_ADDASSIGN:
                        return(G__OPR_ADDASSIGN_FD);
                     case G__OPR_SUBASSIGN:
                        return(G__OPR_SUBASSIGN_FD);
                     case G__OPR_MULASSIGN:
                        return(G__OPR_MULASSIGN_FD);
                     case G__OPR_DIVASSIGN:
                        return(G__OPR_DIVASSIGN_FD);
                  }
            }
            break;
      }
   }
   return(opr);
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
