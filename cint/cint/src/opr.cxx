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
#include "value.h"

#include <ios>

extern "C" {

//______________________________________________________________________________
static const char* G__getoperatorstring(int operatortag)
{
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
void G__doubleassignbyref(G__value* defined, double val)
{
   if (isupper(defined->type)) {
      *(long*)defined->ref = (long)val;
      defined->obj.i = (long)val;
      return;
   }

   switch (defined->type) {
      case 'd': /* double */
         *(double*)defined->ref = val;
         G__setvalue(defined, val);
         break;
      case 'f': /* float */
         *(float*)defined->ref = (float) val;
         G__setvalue(defined, val);
         break;
      case 'l': /* long */
         *(long*)defined->ref = (long) val;
         G__setvalue(defined, (long) val);
         break;
      case 'k': /* unsigned long */
         *(unsigned long*)defined->ref = (unsigned long) val;
         G__setvalue(defined, (unsigned long) val);
         break;
      case 'i': /* int */
         *(int*)defined->ref = (int) val;
         G__setvalue(defined, (int) val);
         break;
      case 'h': /* unsigned int */
         *(unsigned int*)defined->ref = (unsigned int)val;
         G__setvalue(defined, (unsigned int) val);
         break;
      case 's': /* short */
         *(short*)defined->ref = (short) val;
         G__setvalue(defined, (short) val);
         break;
      case 'r': /* unsigned short */
         *(unsigned short*)defined->ref = (unsigned short) val;
         G__setvalue(defined, (unsigned short) val);
         break;
      case 'c': /* char */
         *(char*)defined->ref = (char) val;
         G__setvalue(defined, (char) val);
         break;
      case 'b': /* unsigned char */
         *(unsigned char*)defined->ref = (unsigned char)val;
         G__setvalue(defined, (unsigned char) val);
         break;
      case 'n': /* long long */
         *(G__int64*)defined->ref = (G__int64) val;
         defined->obj.ll = (G__int64) val;
         G__setvalue(defined, (G__int64) val);
         break;
      case 'm': /* unsigned long long */
         *(G__uint64*)defined->ref = (G__uint64) val;
         G__setvalue(defined, (G__uint64) val);
         break;
      case 'q': /* unsigned G__int64 */
         *(long double*)defined->ref = (long double) val;
         G__setvalue(defined, (long double) val);
         break;
      case 'g': /* bool */
#ifdef G__BOOL4BYTE
         *(int*)defined->ref = (int) (val ? 1 : 0);
#else // G__BOOL4BYTE
         *(unsigned char*)defined->ref = (unsigned char) (val ? 1 : 0);
#endif // G__BOOL4BYTE
         G__setvalue(defined, (bool) val);
         break;
      default:
         G__genericerror("Invalid operation and assignment, G__doubleassignbyref");
         break;
   }
}

//______________________________________________________________________________
void G__intassignbyref(G__value* defined, G__int64 val)
{
   if (isupper(defined->type)) {
      if (defined->ref) *(long*)defined->ref = (long)val;
      defined->obj.i = (long)val;
      return;
   }

   switch (defined->type) {
      case 'i': /* int */
         if (defined->ref) *(int*)defined->ref = (int)val;
         G__setvalue(defined, (int) val);
         break;
      case 'c': /* char */
         if (defined->ref) *(char*)defined->ref = (char)val;
         G__setvalue(defined, (char) val);
         break;
      case 'l': /* long */
         if (defined->ref) *(long*)defined->ref = (long)val;
         G__setvalue(defined, (long) val);
         break;
      case 's': /* short */
         if (defined->ref) *(short*)defined->ref = (short)val;
         G__setvalue(defined, (short) val);
         break;
      case 'k': /* unsigned long */
         if (defined->ref) *(unsigned long*)defined->ref = (unsigned long)val;
         G__setvalue(defined, (unsigned long) val);
         break;
      case 'h': /* unsigned int */
         if (defined->ref) *(unsigned int*)defined->ref = (unsigned int)val;
         G__setvalue(defined, (unsigned int) val);
         break;
      case 'r': /* unsigned short */
         if (defined->ref) *(unsigned short*)defined->ref = (unsigned short)val;
         G__setvalue(defined, (unsigned short) val);
         break;
      case 'b': /* unsigned char */
         if (defined->ref) *(unsigned char*)defined->ref = (unsigned char)val;
         G__setvalue(defined, (unsigned char) val);
         break;
      case 'n': /* long long */
         if (defined->ref) *(G__int64*)defined->ref = (G__int64)val;
         G__setvalue(defined, (G__int64) val);
         break;
      case 'm': /* long long */
         if (defined->ref) *(G__uint64*)defined->ref = (G__uint64)val;
         G__setvalue(defined, (G__uint64) val);
         break;
      case 'q': /* long double */
         if (defined->ref) *(long double*)defined->ref = (long double)val;
         G__setvalue(defined, (long double) val);
         break;
      case 'g': /* bool */
#ifdef G__BOOL4BYTE
         if (defined->ref) *(int*)defined->ref = (int)(val ? 1 : 0);
#else // G__BOOL4BYTE
         if (defined->ref) *(unsigned char*)defined->ref = (unsigned char)(val ? 1 : 0);
#endif // G__BOOL4BYTE
         G__setvalue(defined, (bool) val);
         break;
      case 'd': /* double */
         if (defined->ref) *(double*)defined->ref = (double)val;
         G__setvalue(defined, (double) val);
         break;
      case 'f': /* float */
         if (defined->ref) *(float*)defined->ref = (float)val;
         G__setvalue(defined, (float) val);
         break;
      default:
         G__genericerror("Invalid operation and assignment, G__intassignbyref");
         break;
   }
}

//______________________________________________________________________________
void G__bstore(int operatortag, G__value expressionin, G__value* defined)
{
   int ig2;
   long lresult;
   double fdefined, fexpression;

   /*********************************************************
    * for overloading of operator
    *********************************************************/
   /****************************************************************
    * C++ 
    * If one of the parameter is struct(class) type, call user 
    * defined operator function
    * Assignment operators (=,+=,-=) do not work in this way.
    ****************************************************************/
   if (defined->type == 'u' || expressionin.type == 'u') {
      G__overloadopr(operatortag, expressionin, defined);
      return;
   }
   else {
#ifdef G__ASM
      if (G__asm_noverflow) {
         if (defined->type == '\0') {
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
                  G__asm_inst[G__asm_cp] = G__OP1;
                  G__asm_inst[G__asm_cp+1] = G__op1_operator_detail(operatortag, &expressionin);
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     if (isprint(operatortag)) {
                        G__fprinterr(G__serr, "%3x,%3x: OP1 '%c' (%d) %d %s:%d\n", G__asm_cp, G__asm_dt, operatortag, operatortag, G__asm_inst[G__asm_cp+1], __FILE__, __LINE__);
                     }
                     else {
                        static const char* oprtagnames[] = { "+=", "-=", "%=", "*=", "/=", ">>+", "<<=", "&=", "|=", "^=", "&&=", "||=", "var++", "++var", "var--", "--var", "???"
                                                           };
                        if ((operatortag > 0) && (operatortag < 18)) {
                           G__fprinterr(G__serr, "%3x,%3x: OP1 %s %d  %s:%d\n", G__asm_cp, G__asm_dt, oprtagnames[operatortag-1], G__asm_inst[G__asm_cp+1], __FILE__, __LINE__);
                        }
                        else {
                           G__fprinterr(G__serr, "%3x,%3x: OP1 %d %d  %s:%d\n", G__asm_cp, G__asm_dt, operatortag, G__asm_inst[G__asm_cp+1], __FILE__, __LINE__);
                        }
                     }
                  }
#endif // G__ASM_DBG
                  G__inc_cp_asm(2, 0);
                  break;
            }
         }
         else {
            /****************************
             * OP2 instruction
             ****************************/
            // --
            G__asm_inst[G__asm_cp] = G__OP2;
            G__asm_inst[G__asm_cp+1] = G__op2_operator_detail(operatortag, defined, &expressionin);
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               if (isprint(operatortag)) {
                  G__fprinterr(G__serr, "%3x,%3x: OP2 '%c' (%d) %d  %s:%d\n", G__asm_cp, G__asm_dt, operatortag, operatortag, G__asm_inst[G__asm_cp+1], __FILE__, __LINE__);
               }
               else {
                  G__fprinterr(G__serr, "%3x,%3x: OP2 %d %d  %s:%d\n", G__asm_cp, G__asm_dt, operatortag, G__asm_inst[G__asm_cp+1], __FILE__, __LINE__);
               }
            }
#endif // G__ASM_DBG
            G__inc_cp_asm(2, 0);
         }
      }
#endif // G__ASM
      if (G__no_exec_compile || G__no_exec) { /* avoid Alpha crash */
         if (G__isdouble(expressionin)) expressionin.obj.d = 0.0;
         else                          expressionin.obj.i = 0;
         if (G__isdouble(*defined)) defined->obj.d = 0.0;
         else                      defined->obj.i = 0;
      }
   }

   /****************************************************************
    * long double operator long double
    * 
    ****************************************************************/
   if ('q' == defined->type || 'q' == expressionin.type) {
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
            if (defined->type == G__null.type) lddefined = 1;
            G__letLongdouble(defined, 'q', lddefined*ldexpression);
            defined->ref = 0;
            break;
         case '/': /* divide */
            if (defined->type == G__null.type) lddefined = 1;
            //  IEEE 754 defines that NaN inf has to be the output in cases where division by 0 occurs.
//            if (ldexpression == 0) {
//               if (G__no_exec_compile) G__letdouble(defined, 'i', 0);
//               else G__genericerror("Error: operator '/' divided by zero");
//               return;
//            }
            if (G__no_exec_compile && ldexpression == 0.0) {
               G__letdouble(defined, 'q', 0.0);
               return;
            }
            G__letLongdouble(defined, 'q', lddefined / ldexpression);
            defined->ref = 0;
            break;

         case '>':
            if (defined->type == G__null.type) {
               G__letLongdouble(defined, 'i', 0 > ldexpression);
            }
            else
               G__letint(defined, 'i', lddefined > ldexpression);
            defined->ref = 0;
            break;
         case '<':
            if (defined->type == G__null.type) {
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
            if (defined->type == G__null.type)
               G__letLongdouble(defined, 'q', 0); /* Expression should be false wben the var is not defined */
            else
               G__letint(defined, 'i', lddefined == ldexpression);
            defined->ref = 0;
            break;
         case 'N': /* != */
            if (defined->type == G__null.type)
               G__letLongdouble(defined, 'q', 1); /* Expression should be true wben the var is not defined */
            else
               G__letint(defined, 'i', lddefined != ldexpression);
            defined->ref = 0;
            break;
         case 'G': /* >= */
            if (defined->type == G__null.type)
               G__letLongdouble(defined, 'q', 0); /* Expression should be false wben the var is not defined */
            else
               G__letint(defined, 'i', lddefined >= ldexpression);
            defined->ref = 0;
            break;
         case 'l': /* <= */
            if (defined->type == G__null.type)
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

   /****************************************************************
    * double operator double
    * double operator int
    * int    operator double
    ****************************************************************/
   else if ((G__isdouble(expressionin)) || (G__isdouble(*defined))) {
      fexpression = G__double(expressionin);
      fdefined = G__double(*defined);
      defined->typenum = -1;
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
            if (defined->type == G__null.type) fdefined = 1.0;
            G__letdouble(defined, 'd', fdefined*fexpression);
            defined->ref = 0;
            break;
         case '/': /* divide */
            if (defined->type == G__null.type) fdefined = 1.0;
            //  IEEE 754 defines that NaN inf has to be the output in cases where division by 0 occurs.
//            if (fexpression == 0.0) {
//               if (G__no_exec_compile) G__letdouble(defined, 'd', 0.0);
//               else G__genericerror("Error: operator '/' divided by zero");
//               return;
//            }
            if (G__no_exec_compile && fexpression == 0.0) {
               G__letdouble(defined, 'd', 0.0);
               return;
            }
            G__letdouble(defined, 'd', fdefined / fexpression);
            defined->ref = 0;
            break;
#ifdef G__NONANSIOPR
         case '%': /* modulus */
            if (fexpression == 0.0) {
               if (G__no_exec_compile) G__letdouble(defined, 'd', 0.0);
               else G__genericerror("Error: operator '%' divided by zero");
               return;
            }
            G__letint(defined, 'i', (long)fdefined % (long)fexpression);
            defined->ref = 0;
            break;
#endif /* G__NONANSIOPR */
         case '&': /* binary and */
            /* Don't know why but this one has a problem if deleted */
            if (defined->type == G__null.type) {
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
            if (defined->type == G__null.type) {
               G__letdouble(defined, 'i', 0 > fexpression);
            }
            else
               G__letint(defined, 'i', fdefined > fexpression);
            defined->ref = 0;
            break;
         case '<':
            if (defined->type == G__null.type) {
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
            if (defined->type == G__null.type)
               G__letdouble(defined, 'i', 0 == fexpression);
            else
               G__letint(defined, 'i', fdefined == fexpression);
            defined->ref = 0;
            break;
         case 'N': /* != */
            if (defined->type == G__null.type)
               G__letdouble(defined, 'i', 0 != fexpression);
            else
               G__letint(defined, 'i', fdefined != fexpression);
            defined->ref = 0;
            break;
         case 'G': /* >= */
            if (defined->type == G__null.type)
               G__letdouble(defined, 'i', 0 >= fexpression);
            else
               G__letint(defined, 'i', fdefined >= fexpression);
            defined->ref = 0;
            break;
         case 'l': /* <= */
            if (defined->type == G__null.type)
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

   /****************************************************************
    * pointer operator pointer
    * pointer operator int
    * int     operator pointer
    ****************************************************************/
   else if (isupper(defined->type) || isupper(expressionin.type)) {

      G__CHECK(G__SECURE_POINTER_CALC, '+' == operatortag || '-' == operatortag, return);

      if (isupper(defined->type)) {

         /*
          *  pointer - pointer , integer [==] pointer
          */
         if (isupper(expressionin.type)) {
            switch (operatortag) {
               case '\0': /* add */
                  defined->ref = expressionin.ref;
                  defined->obj.i = defined->obj.i + expressionin.obj.i;
                  defined->obj.reftype.reftype = expressionin.obj.reftype.reftype;
                  break;
               case '-': /* subtract */
                  defined->obj.i
                  = (defined->obj.i - expressionin.obj.i) / G__sizeof(defined);
                  defined->type = 'i';
                  defined->tagnum = -1;
                  defined->typenum = -1;
                  defined->ref = 0;
                  break;
               case 'E': /* == */
                  if ('U' == defined->type && 'U' == expressionin.type)
                     G__publicinheritance(defined, &expressionin);
                  G__letint(defined, 'i', defined->obj.i == expressionin.obj.i);
                  defined->ref = 0;
                  break;
               case 'N': /* != */
                  if ('U' == defined->type && 'U' == expressionin.type)
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
                                       / G__sizeof(defined));
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
                  defined->obj.reftype.reftype = expressionin.obj.reftype.reftype;
                  break;
               case '+': /* add */
                  defined->obj.i = defined->obj.i + expressionin.obj.i * G__sizeof(defined);
                  defined->ref = 0;
                  break;
               case '-': /* subtract */
                  defined->obj.i = defined->obj.i - expressionin.obj.i * G__sizeof(defined);
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
                                       , defined->obj.i + expressionin.obj.i*G__sizeof(defined));
                  break;
               case G__OPR_SUBASSIGN:
                  if (!G__no_exec_compile && defined->ref)
                     G__intassignbyref(defined
                                       , defined->obj.i - expressionin.obj.i*G__sizeof(defined));
                  break;
               default:
                  G__fprinterr(G__serr, "Error: %s ", G__getoperatorstring(operatortag));
                  G__genericerror("Illegal operator for pointer 2");
                  break;
            }
         }
      }

      /*
       *  integer [+-] pointer 
       */
      else {
         switch (operatortag) {
            case '\0': /* subtract */
               defined->ref = expressionin.ref;
               defined->type = expressionin.type;
               defined->tagnum = expressionin.tagnum;
               defined->typenum = expressionin.typenum;
               defined->obj.i = defined->obj.i * G__sizeof(defined) + expressionin.obj.i;
               defined->obj.reftype.reftype = expressionin.obj.reftype.reftype;
               break;
            case '+': /* add */
               defined->obj.i = defined->obj.i * G__sizeof(defined) + expressionin.obj.i;
               defined->type = expressionin.type;
               defined->tagnum = expressionin.tagnum;
               defined->typenum = expressionin.typenum;
               defined->obj.reftype.reftype = expressionin.obj.reftype.reftype;
               defined->ref = 0;
               break;
            case '-': /* subtract */
               defined->obj.i = defined->obj.i * G__sizeof(defined) - expressionin.obj.i;
               defined->type = expressionin.type;
               defined->tagnum = expressionin.tagnum;
               defined->typenum = expressionin.typenum;
               defined->obj.reftype.reftype = expressionin.obj.reftype.reftype;
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
                                    , defined->obj.i*G__sizeof(defined) + expressionin.obj.i);
               break;
            case G__OPR_SUBASSIGN:
               if (!G__no_exec_compile && defined->ref)
                  G__intassignbyref(defined
                                    , defined->obj.i*G__sizeof(defined) - expressionin.obj.i);
               break;
            case G__OPR_POSTFIXINC:
               if (!G__no_exec_compile && expressionin.ref) {
                  *defined = expressionin;
                  G__intassignbyref(&expressionin
                                    , expressionin.obj.i + G__sizeof(&expressionin));
                  defined->ref = 0;
               }
               break;
            case G__OPR_POSTFIXDEC:
               if (!G__no_exec_compile && expressionin.ref) {
                  *defined = expressionin;
                  G__intassignbyref(&expressionin
                                    , expressionin.obj.i - G__sizeof(&expressionin));
                  defined->ref = 0;
               }
               break;
            case G__OPR_PREFIXINC:
               if (!G__no_exec_compile && expressionin.ref) {
                  G__intassignbyref(&expressionin
                                    , expressionin.obj.i + G__sizeof(&expressionin));
                  *defined = expressionin;
               }
               break;
            case G__OPR_PREFIXDEC:
               if (!G__no_exec_compile && expressionin.ref) {
                  G__intassignbyref(&expressionin
                                    , expressionin.obj.i - G__sizeof(&expressionin));
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

   /****************************************************************
    * long long operator long long
    * 
    ****************************************************************/
   else if ('n' == defined->type     || 'm' == defined->type    ||
            'n' == expressionin.type || 'm' == expressionin.type) {
      int unsignedresult = 0;
      if ('m' == defined->type || 'm' == expressionin.type) unsignedresult = -1;
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
               if (defined->type == G__null.type) ulldefined = 1;
               G__letULonglong(defined, 'm', ulldefined*ullexpression);
               defined->ref = 0;
               break;
            case '/': /* divide */
               if (defined->type == G__null.type) ulldefined = 1;
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
                  else G__genericerror("Error: operator '%' divided by zero");
                  return;
               }
               G__letULonglong(defined, 'm', ulldefined % ullexpression);
               defined->ref = 0;
               break;
            case '&': /* binary and */
               if (defined->type == G__null.type) {
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
               if (defined->type == G__null.type) {
                  G__letULonglong(defined, 'm', 0);
               }
               else
                  G__letint(defined, 'i', ulldefined > ullexpression);
               defined->ref = 0;
               break;
            case '<':
               if (defined->type == G__null.type) {
                  G__letULonglong(defined, 'm', 0);
               }
               else
                  G__letint(defined, 'i', ulldefined < ullexpression);
               defined->ref = 0;
               break;
            case 'R': /* right shift */
               // for a>>b, the unsignedness of a defines
               // whether the result is unsigned
               switch (defined->type) {
                  case 'b':
                  case 'r':
                  case 'h':
                  case 'k': {
                     G__letULonglong(defined, 'm', ulldefined >> ullexpression);
                  }
                  break;
                  default:
                     // signed version:
                     G__letLonglong(defined, 'n', G__Longlong(*defined) >> ullexpression);
                     break;
               }
               defined->ref = 0;
               break;
            case 'L': /* left shift */
               switch (defined->type) {
                  case 'b':
                  case 'r':
                  case 'h':
                  case 'k': {
                     G__letULonglong(defined, 'm', ulldefined << ullexpression);
                  }
                  break;
                  default:
                     // signed version:
                     G__letLonglong(defined, 'n', G__Longlong(*defined) << ullexpression);
                     break;
               }
               defined->ref = 0;
               break;
            case '!':
               G__letULonglong(defined, 'm', !ullexpression);
               defined->ref = 0;
               break;
            case 'E': /* == */
               if (defined->type == G__null.type)
                  G__letULonglong(defined, 'm', 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, 'i', ulldefined == ullexpression);
               defined->ref = 0;
               break;
            case 'N': /* != */
               if (defined->type == G__null.type)
                  G__letULonglong(defined, 'm', 1); /* Expression should be true wben the var is not defined */
               else
                  G__letint(defined, 'i', ulldefined != ullexpression);
               defined->ref = 0;
               break;
            case 'G': /* >= */
               if (defined->type == G__null.type)
                  G__letULonglong(defined, 'm', 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, 'i', ulldefined >= ullexpression);
               defined->ref = 0;
               break;
            case 'l': /* <= */
               if (defined->type == G__null.type)
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
               if (defined->type == G__null.type) lldefined = 1;
               G__letLonglong(defined, 'n', lldefined*llexpression);
               defined->ref = 0;
               break;
            case '/': /* divide */
               if (defined->type == G__null.type) lldefined = 1;
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
                  else G__genericerror("Error: operator '%' divided by zero");
                  return;
               }
               G__letLonglong(defined, 'n', lldefined % llexpression);
               defined->ref = 0;
               break;
            case '&': /* binary and */
               if (defined->type == G__null.type) {
                  G__letLonglong(defined, 'n', llexpression);
               }
               else {
                  G__letint(defined, 'i', lldefined&llexpression);
               }
               defined->ref = 0;
               break;
            case '|': /* binary or */
               G__letint(defined, 'i', lldefined | llexpression);
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
               if (defined->type == G__null.type) {
                  G__letLonglong(defined, 'n', 0);
               }
               else
                  G__letint(defined, 'i', lldefined > llexpression);
               defined->ref = 0;
               break;
            case '<':
               if (defined->type == G__null.type) {
                  G__letLonglong(defined, 'n', 0);
               }
               else
                  G__letint(defined, 'i', lldefined < llexpression);
               defined->ref = 0;
               break;
            case 'R': /* right shift */
               switch (defined->type) {
                  case 'b':
                  case 'r':
                  case 'h':
                  case 'k': {
                     G__letULonglong(defined, 'n', 0);
                     defined->obj.ulo = G__ULonglong(*defined) >> llexpression;
                  }
                  break;
                  default:
                     G__letLonglong(defined, 'n', lldefined >> llexpression);
                     break;
               }
               defined->ref = 0;
               break;
            case 'L': /* left shift */
               switch (defined->type) {
                  case 'b':
                  case 'r':
                  case 'h':
                  case 'k': {
                     G__letULonglong(defined, 'n', 0);
                     defined->obj.ulo = G__ULonglong(*defined) << llexpression;
                  }
                  break;
                  default:
                     G__letLonglong(defined, 'n', lldefined << llexpression);
                     break;
               }
               defined->ref = 0;
               break;
            case '!':
               G__letLonglong(defined, 'n', !llexpression);
               defined->ref = 0;
               break;
            case 'E': /* == */
               if (defined->type == G__null.type)
                  G__letLonglong(defined, 'n', 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, 'i', lldefined == llexpression);
               defined->ref = 0;
               break;
            case 'N': /* != */
               if (defined->type == G__null.type)
                  G__letLonglong(defined, 'n', 1); /* Expression should be true wben the var is not defined */
               else
                  G__letint(defined, 'i', lldefined != llexpression);
               defined->ref = 0;
               break;
            case 'G': /* >= */
               if (defined->type == G__null.type)
                  G__letLonglong(defined, 'n', 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, 'i', lldefined >= llexpression);
               defined->ref = 0;
               break;
            case 'l': /* <= */
               if (defined->type == G__null.type)
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

   /****************************************************************
    * int operator int
    * 
    ****************************************************************/
   else {
      int unsignedresult = 0;
      switch (defined->type) {
         case 'h':
         case 'k':
            unsignedresult = -1;
            break;
      }
      switch (expressionin.type) {
         case 'h':
         case 'k':
            unsignedresult = -1;
            break;
      }
      bool useLong = (defined->type + expressionin.type > 'i' + 'i');
      if (!defined->type)
         useLong = expressionin.type > 'i';
      char resultTypeChar = useLong ? 'l' : 'i';
      if (unsignedresult) --resultTypeChar;

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
               if (defined->type == G__null.type) udefined = 1;
               G__letint(defined, resultTypeChar, udefined*uexpression);
               defined->ref = 0;
               break;
            case '/': /* divide */
               if (defined->type == G__null.type) udefined = 1;
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
                  else G__genericerror("Error: operator '%' divided by zero");
                  return;
               }
               G__letint(defined, resultTypeChar, udefined % uexpression);
               defined->ref = 0;
               break;
            case '&': /* binary and */
               if (defined->type == G__null.type) {
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
               if (defined->type == G__null.type) {
                  G__letint(defined, resultTypeChar, 0);
               }
               else
                  G__letint(defined, resultTypeChar, udefined > uexpression);
               defined->ref = 0;
               break;
            case '<':
               if (defined->type == G__null.type) {
                  G__letint(defined, resultTypeChar, 0);
               }
               else
                  G__letint(defined, resultTypeChar, udefined < uexpression);
               defined->ref = 0;
               break;
            case 'R': /* right shift */
               switch (defined->type) {
                  case 'b':
                  case 'r':
                  case 'h': {
                     unsigned int uidefined = udefined;
                     G__letint(defined, 'h', 0);
                     defined->obj.uin = uidefined >> uexpression;
                  }
                  break;
                  case 'k': {
                     unsigned long uudefined = udefined;
                     G__letint(defined, 'k', 0);
                     defined->obj.ulo = uudefined >> uexpression;
                  }
                  break;
                  default:
                     G__letint(defined, resultTypeChar, G__int(*defined) >> uexpression);
                     break;
               }
               defined->ref = 0;
               break;
            case 'L': /* left shift */
               switch (defined->type) {
                  case 'b':
                  case 'r':
                  case 'h':{
                     unsigned int uidefined = udefined;
                     G__letint(defined, 'h', 0);
                     defined->obj.uin = uidefined << uexpression;
                  }
                  break;
                  case 'k': {
                     unsigned long uudefined = udefined;
                     G__letint(defined, 'k', 0);
                     defined->obj.ulo = uudefined << uexpression;
                  }
                  break;
                  default:
                     G__letint(defined, resultTypeChar, G__int(*defined) << uexpression);
                     break;
               }
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
               if (defined->type == G__null.type)
                  G__letint(defined, resultTypeChar, 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, resultTypeChar, udefined == uexpression);
               defined->ref = 0;
               break;
            case 'N': /* != */
               if (defined->type == G__null.type)
                  G__letint(defined, resultTypeChar, 1); /* Expression should be true wben the var is not defined */
               else
                  G__letint(defined, resultTypeChar, udefined != uexpression);
               defined->ref = 0;
               break;
            case 'G': /* >= */
               if (defined->type == G__null.type)
                  G__letint(defined, resultTypeChar, 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, resultTypeChar, udefined >= uexpression);
               defined->ref = 0;
               break;
            case 'l': /* <= */
               if (defined->type == G__null.type)
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
               if (defined->type == G__null.type) ldefined = 1;
               G__letint(defined, resultTypeChar, ldefined*lexpression);
               defined->ref = 0;
               break;
            case '/': /* divide */
               if (defined->type == G__null.type) ldefined = 1;
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
                  else G__genericerror("Error: operator '%' divided by zero");
                  return;
               }
               G__letint(defined, resultTypeChar, ldefined % lexpression);
               defined->ref = 0;
               break;
            case '&': /* binary and */
               if (defined->type == G__null.type) {
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
               if (defined->type == G__null.type) {
                  G__letint(defined, resultTypeChar, 0);
               }
               else
                  G__letint(defined, resultTypeChar, ldefined > lexpression);
               defined->ref = 0;
               break;
            case '<':
               if (defined->type == G__null.type) {
                  G__letint(defined, resultTypeChar, 0);
               }
               else
                  G__letint(defined, resultTypeChar, ldefined < lexpression);
               defined->ref = 0;
               break;
            case 'R': /* right shift */
               if (!G__prerun) {
                  switch (defined->type) {
                  case 'b':
                  case 'r':
                  case 'h':
                  case 'k': {
                     unsigned long udefined = (unsigned long)G__uint(*defined);
                     G__letint(defined, defined->type, 0);
                     defined->obj.ulo = udefined >> lexpression;
                  }     
                  break;
                  default:
                     G__letint(defined, defined->type, 0);
                     defined->obj.i = ldefined >> lexpression;
                  }
               }
               else {
                  G__letint(defined, resultTypeChar, ldefined >> lexpression);
               }
               defined->ref = 0;
               break;
            case 'L': /* left shift */
               if (!G__prerun) {
                  switch (defined->type) {
                  case 'b':
                  case 'r':
                  case 'h':
                  case 'k': {
                     unsigned long udefined = (unsigned long)G__uint(*defined);
                     G__letint(defined, defined->type, 0);
                     defined->obj.ulo = udefined << lexpression;
                  }
                  break;
                  default:
                     G__letint(defined, defined->type, 0);
                     defined->obj.i = ldefined << lexpression;
                  }
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
               if (defined->type == G__null.type)
                  G__letint(defined, resultTypeChar, 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, resultTypeChar, ldefined == lexpression);
               defined->ref = 0;
               break;
            case 'N': /* != */
               if (defined->type == G__null.type)
                  G__letint(defined, resultTypeChar, 1); /* Expression should be true wben the var is not defined */
               else
                  G__letint(defined, resultTypeChar, ldefined != lexpression);
               defined->ref = 0;
               break;
            case 'G': /* >= */
               if (defined->type == G__null.type)
                  G__letint(defined, resultTypeChar, 0); /* Expression should be false wben the var is not defined */
               else
                  G__letint(defined, resultTypeChar, ldefined >= lexpression);
               defined->ref = 0;
               break;
            case 'l': /* <= */
               if (defined->type == G__null.type)
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
   if (G__no_exec_compile && 0 == defined->type) *defined = expressionin;
}

//______________________________________________________________________________
int G__scopeoperator(char * name, int* phash, long* pstruct_offset, int* ptagnum)
{
   // -- FIXME: Describe this function!
   // May need to modify this function to support multiple usage of
   // scope operator 'xxx::xxx::var'
   // name is modified and this is intentional
   char* pc;
   char* scope;
   char* member;
   int scopetagnum;
   int offset;
   int offset_sum;
   int i;
   G__FastAllocString temp(G__MAXNAME*2);
   char* pparen;
   re_try_after_std:
   // search for pattern "::"
   pc = (char*)G__find_first_scope_operator(name);
   // no scope operator, return
   pparen = strchr(name, '(');
   if (NULL == pc || strncmp(name, "operator ", 9) == 0 || (pparen && pparen < pc)) {
      G__fixedscope = 0;
      return G__NOSCOPEOPR;
   }
   G__fixedscope = 1;
   // if scope operator found at the beginning of the name, global scope
   // if scope operator found at the beginning of the name, global scope
   // or fully qualified scope!
   if (pc == name) {
      /* strip scope operator, set hash and return */
      temp = name + 2;
      strcpy(name, temp); // Okay since we 'reduce' the size of the string
      G__hash(name, (*phash), i)
      /* If we do no have anymore scope operator, we know the request of
         for the global name space */
      pc = (char*)G__find_first_scope_operator(name);
      if (pc == 0) return(G__GLOBALSCOPE);
   }
#ifndef G__STD_NAMESPACE
   if (strncmp(name, "std::", 5) == 0 && G__ignore_stdnamespace) {
      // strip scope operator, set hash and return
      temp = name + 5;
      strcpy(name, temp); // Okay since we 'reduce' the size of the string
      G__hash(name, (*phash), i)
      goto re_try_after_std;
   }
#endif
   // otherwise, specific class scope
   offset_sum = 0;
   if (*name == '~') {
      // -- Explicit destructor of the form: ~A::B().
      scope = name + 1;
   }
   else {
      scope = name;
   }
   // Recursive scope operator is not allowed in compiler but possible in cint.
   scopetagnum = G__get_envtagnum();
   do {
      int save_tagdefining, save_def_tagnum;
      save_tagdefining = G__tagdefining;
      save_def_tagnum = G__def_tagnum;
      G__tagdefining = scopetagnum;
      G__def_tagnum = scopetagnum;
      member = pc + 2;
      *pc = '\0';
      scopetagnum = G__defined_tagname(scope, 1);
      G__tagdefining = save_tagdefining;
      G__def_tagnum = save_def_tagnum;

#ifdef G__VIRTUALBASE
      if (-1 == (offset = G__ispublicbase(scopetagnum, *ptagnum
                                          , *pstruct_offset + offset_sum))) {
         int store_tagnum = G__tagnum;
         G__tagnum = *ptagnum;
         offset = -G__find_virtualoffset(scopetagnum, *pstruct_offset + offset_sum); /* NEED REFINEMENT */
         G__tagnum = store_tagnum;
      }
#else
      if (-1 == (offset = G__ispublicbase(scopetagnum, *ptagnum))) offset = 0;
#endif

      *ptagnum = scopetagnum;
      offset_sum += offset;

      scope = member;
   }
   while ((pc = (char*)G__find_first_scope_operator(scope)));
   *pstruct_offset += offset_sum;
#ifdef G__ASM
   if (G__asm_noverflow) {
#ifdef G__ASM_DBG
      if (G__asm_dbg)
         G__fprinterr(G__serr, "%3x,%3x: ADDSTROS %d  %s:%d\n", G__asm_cp, G__asm_dt, offset_sum, __FILE__, __LINE__);
#endif
      G__asm_inst[G__asm_cp] = G__ADDSTROS;
      G__asm_inst[G__asm_cp+1] = offset_sum;
      G__inc_cp_asm(2, 0);
   }
#endif
   temp = member;
   if (*name == '~') {
      // -- Explicit destructor.
      strcpy(name + 1, temp); // Okay since we 'reduce' the size of the string
   }
   else {
      strcpy(name, temp); // Okay since we 'reduce' the size of the string
   }
   G__hash(name, *phash, i)
   return G__CLASSSCOPE;
}

//______________________________________________________________________________
int G__cmp(G__value buf1, G__value buf2)
{
   switch (buf1.type) {
      case 'a':  /* G__start */
      case 'z':  /* G__default */
      case '\0': /* G__null */
         if (buf1.type == buf2.type)
            return(1);
         else
            return(0);
         /* break; */
      case 'd':
      case 'f':
         if (G__double(buf1) == G__double(buf2))
            return(1);
         else
            return(0);
         /* break; */
   }

   if (G__int(buf1) == G__int(buf2)) return(1);
   /* else */
   return(0);
}

//______________________________________________________________________________
int G__getunaryop(char unaryop, const char* expression, char* buf, G__value* preg)
{
   int nest = 0;
   int c = 0;
   int i1 = 1, i2 = 0;
   G__value reg;
   char prodpower = 0;

   *preg = G__null;
   for (;;) {
      c = expression[i1];
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
            if (0 == nest) {
               buf[i2] = '\0';
               if (prodpower) reg = G__getprod(buf);
               else          reg = G__getitem(buf);
               G__bstore(unaryop, reg, preg);
               return(i1);
            }
            buf[i2++] = c;
            break;
         case '*':
         case '/':
         case '%':
         case '@':
         case '~':
         case ' ':
            if (0 == nest) prodpower = 1;
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
      ++i1;
   }
}

//______________________________________________________________________________
#ifdef G__VIRTUALBASE
int G__iosrdstate(G__value* pios)
{
   // -- ios rdstate condition test
   G__value result;
   long store_struct_offset;
   int store_tagnum;
   int rdstateflag = 0;

   if (-1 != pios->tagnum && 'e' == G__struct.type[pios->tagnum]) return(pios->obj.i);

#ifdef G__ASM
   if (G__asm_noverflow) {
      // -- We are generating bytecode.
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "%3x,%3x: PUSHSTROS  %s:%d\n"
            , G__asm_cp
            , G__asm_dt
            , __FILE__
            , __LINE__
         );
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__inc_cp_asm(1, 0);
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "%3x,%3x: SETSTROS  %s:%d\n"
            , G__asm_cp
            , G__asm_dt
            , __FILE__
            , __LINE__
         );
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__SETSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM

   // Change member function call environment to passed object.
   store_tagnum = G__tagnum;
   G__tagnum = pios->tagnum;
   store_struct_offset = G__store_struct_offset;
   G__store_struct_offset = pios->obj.i;

   // Try to call basic_ios::rdstate().
   // FIXME: We are supposed to use basic_ios::fail() here!
   int known = 0;
   result = G__getfunction("rdstate()", &known, G__TRYMEMFUNC);
   if (known) { // If rdstate() existed, remember that.
      rdstateflag = 1;
   }

   // If no basic_ios::rdstate(), try other things.
   if (!known) {
      result = G__getfunction("operator int()", &known, G__TRYMEMFUNC);
   }
   if (!known) {
      result = G__getfunction("operator bool()", &known, G__TRYMEMFUNC);
   }
   if (!known) {
      result = G__getfunction("operator long()", &known, G__TRYMEMFUNC);
   }
   if (!known) {
      result = G__getfunction("operator short()", &known, G__TRYMEMFUNC);
   }
   if (!known) {
      result = G__getfunction("operator char*()", &known, G__TRYMEMFUNC);
   }
   if (!known) {
      result = G__getfunction("operator const char*()", &known, G__TRYMEMFUNC);
   }

#ifdef G__ASM
   if (G__asm_noverflow) {
      // -- 
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "%3x,%3x: POPSTROS  %s:%d\n"
            , G__asm_cp
            , G__asm_dt
            , __FILE__
            , __LINE__
         );
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif // G__ASM

   // Restore member function call environment.
   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;

   if (!known) {
      G__genericerror("Limitation: Cint does not support full iostream functionality in this platform");
      return 0;
   }

   if (!rdstateflag) {
      return result.obj.i;
   }

#ifdef G__ASM
   if (G__asm_noverflow) {
      // -- 
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "%3x,%3x: LD std::ios_base::failbit | std::ios_base::badbit  %s:%d\n"
            , G__asm_cp
            , G__asm_dt
            , __FILE__
            , __LINE__
         );
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__LD;
      G__asm_inst[G__asm_cp+1] = G__asm_dt;
      G__letint(&G__asm_stack[G__asm_dt], 'i', (long) (std::ios_base::failbit | std::ios_base::badbit));
      G__inc_cp_asm(2, 1);
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "%3x,%3x: OP2 '&'  %s:%d\n"
            , G__asm_cp
            , G__asm_dt
            , __FILE__
            , __LINE__
         );
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__OP2;
      G__asm_inst[G__asm_cp+1] = '&';
      G__inc_cp_asm(2, 0);
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(
              G__serr
            , "%3x,%3x: OP1 '!'  %s:%d\n"
            , G__asm_cp
            , G__asm_dt
            , __FILE__
            , __LINE__
         );
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__OP1;
      G__asm_inst[G__asm_cp+1] = '!';
      G__inc_cp_asm(2, 0);
   }
#endif // G__ASM

   return !(result.obj.i & (std::ios_base::failbit | std::ios_base::badbit));
}
#endif // G__VIRTUALBASE

//______________________________________________________________________________
int G__overloadopr(int operatortag, G__value expressionin, G__value* defined)
{
   int ig2;
   G__FastAllocString expr(G__LONGLINE);
   G__FastAllocString opr(12);
   G__FastAllocString arg1(G__LONGLINE);
   G__FastAllocString arg2(G__LONGLINE);
   long store_struct_offset;
   int store_tagnum;
   int store_isconst;
   G__value buffer;
   char* pos;
   int postfixflag = 0;
   int store_asm_cp = 0;
   switch (operatortag) {
      case '+': /* add */
      case '-': /* subtract */
      case '*': /* multiply */
      case '/': /* divide */
      case '%': /* modulus */
      case '&': /* binary and */
      case '|': /* binariy or */
      case '^': /* binary exclusive or */
      case '~': /* binary inverse */
      case '>':
      case '<':
      case '@': /* power */
      case '!':
         opr.Format("operator%c", operatortag);
         break;

      case 'A': /* logic and  && */
         opr = "operator&&";
         break;

      case 'O': /* logic or   || */
         opr = "operator||";
         break;

      case 'R': /* right shift >> */
         opr = "operator>>";
         break;
      case 'L': /* left shift  << */
         opr = "operator<<";
         break;

      case 'E':
         opr = "operator==";
         break;
      case 'N':
         opr = "operator!=";
         break;
      case 'G':
         opr = "operator>=";
         break;
      case 'l':
         opr = "operator<=";
         break;

      case '\0':
         *defined = expressionin;
         return(0);

      case G__OPR_ADDASSIGN:
         opr = "operator+=";
         break;
      case G__OPR_SUBASSIGN:
         opr = "operator-=";
         break;
      case G__OPR_MODASSIGN:
         opr = "operator%%=";
         break;
      case G__OPR_MULASSIGN:
         opr = "operator*=";
         break;
      case G__OPR_DIVASSIGN:
         opr = "operator/=";
         break;
      case G__OPR_RSFTASSIGN:
         opr = "operator>>=";
         break;
      case G__OPR_LSFTASSIGN:
         opr = "operator<<=";
         break;
      case G__OPR_BANDASSIGN:
         opr = "operator&=";
         break;
      case G__OPR_BORASSIGN:
         opr = "operator|=";
         break;
      case G__OPR_EXORASSIGN:
         opr = "operator^=";
         break;
      case G__OPR_ANDASSIGN:
         opr = "operator&&=";
         break;
      case G__OPR_ORASSIGN:
         opr = "operator||=";
         break;

      case G__OPR_POSTFIXINC:
      case G__OPR_PREFIXINC:
         opr = "operator++";
         break;
      case G__OPR_POSTFIXDEC:
      case G__OPR_PREFIXDEC:
         opr = "operator--";
         break;

      default:
         G__genericerror(
            "Limitation: Can't handle combination of overloading operators"
         );
         return(0);
   }
   if (!defined->type) {
      // -- Unary operator.
      switch (operatortag) {
         case '-':
         case '!':
         case '~':
            break;
         case G__OPR_POSTFIXINC:
         case G__OPR_POSTFIXDEC:
         case G__OPR_PREFIXINC:
         case G__OPR_PREFIXDEC:
            break;
         default:
            *defined = expressionin;
            return(0);
            /* break; */
      }

      G__oprovld = 1;
#ifdef G__ASM
      if (G__asm_noverflow) {
         store_asm_cp = G__asm_cp;
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

      /***************************************************
       * search for member function 
       ****************************************************/
      ig2 = 0;
      switch (operatortag) {
         case G__OPR_POSTFIXINC:
         case G__OPR_POSTFIXDEC:
            expr.Format("%s(1)", opr());
#ifdef G__ASM
            if (G__asm_noverflow) {
               G__asm_inst[G__asm_cp] = G__LD;
               G__asm_inst[G__asm_cp+1] = G__asm_dt;
               G__asm_stack[G__asm_dt] = G__one;
               G__inc_cp_asm(2, 1);
               postfixflag = 1;
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD %d  %s:%d\n", G__asm_cp, G__asm_dt, 1, __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
            }
#endif // G__ASM
            break;
         default:
            postfixflag = 0;
            expr.Format("%s()", opr());
            break;
      }

      store_struct_offset = G__store_struct_offset;
      store_tagnum = G__tagnum;
      G__store_struct_offset = expressionin.obj.i;
      G__tagnum = expressionin.tagnum;

      buffer = G__getfunction(expr, &ig2, G__TRYUNARYOPR);

      G__store_struct_offset = store_struct_offset;
      G__tagnum = store_tagnum;

      /***************************************************
       * search for global function
       ****************************************************/
      if (ig2 == 0) {
#ifdef G__ASM
         if (G__asm_noverflow) {
            if (postfixflag) {
               G__inc_cp_asm(-2, -1);
               postfixflag = 0;
#ifdef G__ASM_DBG
               if (G__asm_dbg) G__fprinterr(G__serr, "LD cancelled\n");
#endif
            }
            G__inc_cp_asm(store_asm_cp - G__asm_cp, 0);
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "PUSHSTROS,SETSTROS cancelled\n");
#endif
         }
#endif /* G__ASM */
         switch (operatortag) {
            case G__OPR_POSTFIXINC:
            case G__OPR_POSTFIXDEC:
               expr.Format("%s(%s,1)", opr(), G__setiparseobject(&expressionin, arg1));
#ifdef G__ASM
               if (G__asm_noverflow) {
                  // -- We are generating bytecode.
#ifdef G__ASM_DBG
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x,%3x: LD %d  %s:%d\n", G__asm_cp, G__asm_dt, 1, __FILE__, __LINE__);
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
               expr.Format("%s(%s)", opr(), G__setiparseobject(&expressionin, arg1));
               break;
         }
         buffer = G__getfunction(expr, &ig2, G__TRYNORMAL);
      }
#ifdef G__ASM
      else if (G__asm_noverflow) {
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__POPSTROS;
         G__inc_cp_asm(1, 0);
      }
#endif // G__ASM
      *defined = buffer;
      G__oprovld = 0;
   }
   else {
      // Binary operator.
      G__oprovld = 1;
#ifdef G__ASM
      if (G__asm_noverflow) {
         // We are generating bytecode.
         store_asm_cp = G__asm_cp;
#ifdef G__ASM_DBG
         if (G__asm_dbg) {
            G__fprinterr(G__serr, "%3x,%3x: SWAP  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         }
#endif // G__ASM_DBG
         G__asm_inst[G__asm_cp] = G__SWAP;
         G__inc_cp_asm(1, 0);
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
      //
      //  Search for member function
      //
      ig2 = 0;
      if (expressionin.type == 'u') {
         G__setiparseobject(&expressionin, arg2);
      }
      else {
         G__valuemonitor(expressionin, arg2);
         // This part must be fixed when reference to pointer type is supported.
         if (expressionin.ref && (expressionin.ref != 1)) {
            pos = strchr(arg2, ')');
            if (pos) {
               *pos = '\0';
               if (expressionin.ref < 0) {
                  expr.Format("*%s*)(%ld)", arg2(), expressionin.ref);
               }   
               else {
                  expr.Format("*%s*)%ld", arg2(), expressionin.ref);
               }
               arg2 = expr;
            } else {
               G__fprinterr(G__serr, "G__overloadopr: expected ')' in %s\n", arg2());
            }
         } else if (expressionin.type == 'm') {
            arg2 += "ULL";
         }
         else if (expressionin.type == 'n') {
            arg2 += "LL";
         }
      }
      if (defined->type == 'u') {
         expr.Format("%s(%s)", opr(), arg2());
         store_struct_offset = G__store_struct_offset;
         store_tagnum = G__tagnum;
         G__store_struct_offset = defined->obj.i;
         G__tagnum = defined->tagnum;
         store_isconst = G__isconst;
         G__isconst = defined->isconst;
         buffer = G__getfunction(expr, &ig2, G__TRYBINARYOPR);
         G__isconst = store_isconst;
         G__store_struct_offset = store_struct_offset;
         G__tagnum = store_tagnum;
      }
      if (!ig2) {
         // No member function, search for global function.
#ifdef G__ASM
         if (G__asm_noverflow) {
            G__bc_cancel_VIRTUALADDSTROS();
            G__inc_cp_asm(store_asm_cp - G__asm_cp, 0);
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "SWAP,PUSHSTROS,SETSTROS cancelled\n");
#endif // G__ASM_DBG
         }
#endif // G__ASM
         if (defined->type == 'u') {
            G__setiparseobject(defined, arg1);
         }
         else {
            G__valuemonitor(*defined, arg1);
            // This part must be fixed when reference to pointer type is supported.
            if (defined->ref) {
               pos = strchr(arg1, ')');
               *pos = '\0';
               if (defined->ref < 0) {
                  expr.Format("*%s*)(%ld)", arg1(), defined->ref);
               }
               else {
                  expr.Format("*%s*)%ld", arg1(), defined->ref);
               }
               arg1 = expr;
            }
         }
         expr.Format("%s(%s,%s)", opr(), arg1(), arg2());
         buffer = G__getfunction(expr, &ig2, G__TRYNORMAL);
         //
         //  Need to check ANSI/ISO standard. What happens if operator
         //  function defined in a namespace is used in other namespace.
         //
         if (!ig2 && (expressionin.tagnum != -1) && (G__struct.parent_tagnum[expressionin.tagnum] != -1)) {
            expr.Format("%s::%s(%s,%s)", G__fulltagname(G__struct.parent_tagnum[expressionin.tagnum], 1), opr(), arg1(), arg2());
            buffer = G__getfunction(expr, &ig2, G__TRYNORMAL);
         }
         if (!ig2 && (defined->tagnum != -1) && (G__struct.parent_tagnum[defined->tagnum] != -1)) {
            expr.Format("%s::%s(%s,%s)", G__fulltagname(G__struct.parent_tagnum[defined->tagnum], 1), opr(), arg1(), arg2());
            buffer = G__getfunction(expr, &ig2, G__TRYNORMAL);
         }

         if (!ig2 && ((operatortag == 'A') || (operatortag == 'O'))) {
            int lval = 0;
            int rval = 0;
            if (defined->type == 'u') {
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
                  if (G__asm_dbg) {
                     G__fprinterr(G__serr, "%3x,%3x: SWAP  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
                  }
#endif // G__ASM_DBG
                  G__asm_inst[G__asm_cp] = G__SWAP;
                  G__inc_cp_asm(1, 0);
               }
            }
            else {
               lval = G__int(*defined);
            }
            if (expressionin.type == 'u') {
               rval = G__iosrdstate(&expressionin);
            }
            else {
               rval = G__int(expressionin);
            }
            buffer.ref = 0;
            buffer.tagnum  = -1;
            buffer.typenum = -1;
            switch (operatortag) {
               case 'A':
                  G__letint(&buffer, 'i', lval && rval);
                  break;
               case 'O':
                  G__letint(&buffer, 'i', lval || rval);
                  break;
            }
            if (G__asm_noverflow) {
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  if (isprint(operatortag)) {
                     G__fprinterr(G__serr, "%3x,%3x: OP2 '%c' (%d)  %s:%d\n", G__asm_cp, G__asm_dt, operatortag, operatortag, __FILE__, __LINE__);
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
            ig2 = 1;
         }

         if (!ig2) {
            if (defined->tagnum != -1) {
               G__fprinterr(G__serr, "Error: %s not defined for %s"
                            , opr(), G__fulltagname(defined->tagnum, 1));
            }
            else {
               G__fprinterr(G__serr, "Error: %s not defined", expr());
            }
            G__genericerror(0);
         }
      }
#ifdef G__ASM
      else if (G__asm_noverflow) {
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
      *defined = buffer;
      G__oprovld = 0;
   }
   return 0;
}

//______________________________________________________________________________
int G__parenthesisovldobj(G__value* result3, G__value* result, const char* realname, G__param* libp
#ifdef G__ASM
, int flag /* flag whether to generate PUSHSTROS, SETSTROS */
#else
, int /* flag */ /* flag whether to generate PUSHSTROS, SETSTROS */
#endif
)
{
   if (result->tagnum == -1) return 0;

   int known = 0;
   long store_struct_offset;
   int store_tagnum;
   int funcmatch;
   int hash;
   int store_exec_memberfunc;
   int store_memberfunc_tagnum;
   long store_memberfunc_struct_offset;
   store_exec_memberfunc = G__exec_memberfunc;
   store_memberfunc_tagnum = G__memberfunc_tagnum;
   store_memberfunc_struct_offset = G__memberfunc_struct_offset;
   store_struct_offset = G__store_struct_offset;
   store_tagnum = G__tagnum;
   G__store_struct_offset = result->obj.i;
   G__tagnum = result->tagnum;
#ifdef G__ASM
   if (G__asm_noverflow && !flag) {
      // -- We are generating bytecode and ???
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
   G__hash(realname, hash, known);
   G__fixedscope = 0;
   for (funcmatch = G__EXACT; funcmatch <= G__USERCONV; ++funcmatch) {
      if (G__tagnum != -1) {
         G__incsetup_memfunc(G__tagnum);
      }
      int ret = G__interpret_func(result3, realname, libp, hash, G__struct.memfunc[G__tagnum], funcmatch, G__CALLMEMFUNC);
      if (ret == 1) {
         G__store_struct_offset = store_struct_offset;
         G__tagnum = store_tagnum;
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
         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         return 1;
      }
   }
   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;
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
   G__exec_memberfunc = store_exec_memberfunc;
   G__memberfunc_tagnum = store_memberfunc_tagnum;
   G__memberfunc_struct_offset = store_memberfunc_struct_offset;
   return 0;
}

//______________________________________________________________________________
int G__parenthesisovld(G__value* result3, char* funcname, G__param* libp, int flag)
{
   int known;
   G__value result;
   long store_struct_offset;
   int store_tagnum;
   int funcmatch;
   int hash;
   int store_exec_memberfunc;
   int store_memberfunc_tagnum;
   long store_memberfunc_struct_offset;

   if (strncmp(funcname, "operator", 8) == 0 || strcmp(funcname, "G__ateval") == 0)
      return(0);

   if (0 == funcname[0]) {
      known = 1;
      result = *result3;
   }
   else {

      if (flag == G__CALLMEMFUNC) {
         G__incsetup_memvar(G__tagnum);
         result = G__getvariable(funcname, &known, (struct G__var_array*)NULL
                                 , G__struct.memvar[G__tagnum]);
      }
      else {
         result = G__getvariable(funcname, &known, &G__global, G__p_local);
      }
   }
   
   /* resolve A::staticmethod(1)(2,3) */

   if (
      1 != known
      || -1 == result.tagnum) return(0);

   store_exec_memberfunc = G__exec_memberfunc;
   store_memberfunc_tagnum = G__memberfunc_tagnum;
   store_memberfunc_struct_offset = G__memberfunc_struct_offset;

   store_struct_offset = G__store_struct_offset;
   store_tagnum = G__tagnum;
   G__store_struct_offset = result.obj.i;
   G__tagnum = result.tagnum;

#ifdef G__ASM
   if (G__asm_noverflow) {
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
         G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp + 1, G__asm_dt, __FILE__, __LINE__);
      }
#endif
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__asm_inst[G__asm_cp+1] = G__SETSTROS;
      G__inc_cp_asm(2, 0);
   }
#endif

   static const char* realname = "operator()";
   G__hash(realname, hash, known);

   G__fixedscope = 0;

   for (funcmatch = G__EXACT;funcmatch <= G__USERCONV;funcmatch++) {
      if (-1 != G__tagnum) G__incsetup_memfunc(G__tagnum);
      if (G__interpret_func(result3, realname, libp, hash
                            , G__struct.memfunc[G__tagnum]
                            , funcmatch, G__CALLMEMFUNC) == 1) {
         G__store_struct_offset = store_struct_offset;
         G__tagnum = store_tagnum;

#ifdef G__ASM
         if (G__asm_noverflow) {
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
#endif
            G__asm_inst[G__asm_cp] = G__POPSTROS;
            G__inc_cp_asm(1, 0);
         }
#endif

         G__exec_memberfunc = store_exec_memberfunc;
         G__memberfunc_tagnum = store_memberfunc_tagnum;
         G__memberfunc_struct_offset = store_memberfunc_struct_offset;
         return(1);
      }
   }

   G__store_struct_offset = store_struct_offset;
   G__tagnum = store_tagnum;

#ifdef G__ASM
   if (G__asm_noverflow) {
#ifdef G__ASM_DBG
      if (G__asm_dbg) G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
#endif
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif

   G__exec_memberfunc = store_exec_memberfunc;
   G__memberfunc_tagnum = store_memberfunc_tagnum;
   G__memberfunc_struct_offset = store_memberfunc_struct_offset;
   return(0);
}

//______________________________________________________________________________
int G__tryindexopr(G__value* result7, G__value* para, int paran, int ig25)
{
   //
   // 1) asm
   //    * G__ST_VAR/MSTR -> LD_VAR/MSTR
   //    * paran -> ig25
   // 2) try operator[]() function while ig25<paran
   //
   G__FastAllocString expr(G__ONELINE);
   G__FastAllocString arg2(G__MAXNAME);
   char *pos;
   int store_tagnum;
   int store_typenum;
   long store_struct_offset;
   int known;
   int i;
   int store_asm_exec;

#ifdef G__ASM
   if (G__asm_noverflow) {
      /*  X a[2][3];
       *  Y X::operator[]()
       *  Y::operator[]()
       *    a[x][y][z][w];   stack x y z w ->  stack w z x y 
       *                                             Y X a a
       */
      if (paran > 1 && paran > ig25) {
#ifdef G__ASM_DBG
         if (G__asm_dbg)
            G__fprinterr(G__serr, "%x: REORDER inserted before ST_VAR/MSTR/LD_VAR/MSTR\n"
                         , G__asm_cp - 5);
#endif
         for (i = 1;i <= 5;i++) G__asm_inst[G__asm_cp-i+3] = G__asm_inst[G__asm_cp-i];
         G__asm_inst[G__asm_cp-5] = G__REORDER ;
         G__asm_inst[G__asm_cp-4] = paran ;
         G__asm_inst[G__asm_cp-3] = ig25 ;
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
      if (G__asm_dbg)
         G__fprinterr(G__serr, "ST_VAR/MSTR replaced to LD_VAR/MSTR, paran=%d -> %d\n"
                      , paran, ig25);
#endif
   }
#endif

   store_tagnum = G__tagnum;
   store_typenum = G__typenum;
   store_struct_offset = G__store_struct_offset;
#ifdef G__ASM
   if (G__asm_noverflow) {
#ifdef G__ASM_DBG
      if (G__asm_dbg) G__fprinterr(G__serr, "%3x,%3x: PUSHSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
#endif
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif



   while (ig25 < paran) {
      G__oprovld = 1;
      if ('u' == result7->type) {
         G__tagnum = result7->tagnum;
         G__typenum = result7->typenum;
         G__store_struct_offset = result7->obj.i;
#ifdef G__ASM
         if (G__asm_noverflow) {
#ifdef G__ASM_DBG
            if (G__asm_dbg) G__fprinterr(G__serr, "%3x,%3x: SETSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
#endif
            G__asm_inst[G__asm_cp] = G__SETSTROS;
            G__inc_cp_asm(1, 0);
         }
#endif

         if (para[ig25].type == 'u') {
            G__setiparseobject(&para[ig25], arg2);
         }
         else {
            G__valuemonitor(para[ig25], arg2);
            /* This part must be fixed when reference to pointer type
             * is supported */
            if (para[ig25].ref) {
               pos = strchr(arg2, ')');
               *pos = '\0';
               if (para[ig25].ref < 0)
                  expr.Format("*%s*)(%ld)", arg2(), para[ig25].ref);
               else
                  expr.Format("*%s*)%ld", arg2(), para[ig25].ref);
               arg2 = expr;
            }
         }

         expr.Format("operator[](%s)", arg2());
         store_asm_exec = G__asm_exec;
         G__asm_exec = 0;
         *result7 = G__getfunction(expr, &known, G__CALLMEMFUNC);
         G__asm_exec = store_asm_exec;
      }
      /* in case 'T* operator[]' */
      else if (isupper(result7->type)) {
         result7->obj.i += G__sizeof(result7) * para[ig25].obj.i;
#ifdef G__ASM
         if (G__asm_noverflow) {
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: OP2 +  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
            }
#endif // G__ASM_DBG
            G__asm_inst[G__asm_cp] = G__OP2;
            G__asm_inst[G__asm_cp+1] = '+';
            G__inc_cp_asm(2, 0);
         }
#endif //  G__ASM
         *result7 = G__tovalue(*result7);
      }

      ++ig25;
   }

   G__oprovld = 0 ;

   G__tagnum = store_tagnum;
   G__typenum = store_typenum;
   G__store_struct_offset = store_struct_offset;
#ifdef G__ASM
   if (G__asm_noverflow) {
#ifdef G__ASM_DBG
      if (G__asm_dbg) G__fprinterr(G__serr, "%3x,%3x: POPSTROS  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__);
#endif
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1, 0);
   }
#endif
   return(0);
}

//______________________________________________________________________________
long G__op1_operator_detail(int opr, G__value* val)
{
   /* int isdouble; */

   /* don't optimze if optimize level is less than 3 */
   if (G__asm_loopcompile < 3) return(opr);

   if ('i' == val->type) {
      switch (opr) {
         case G__OPR_POSTFIXINC:
            return(G__OPR_POSTFIXINC_I);
         case G__OPR_POSTFIXDEC:
            return(G__OPR_POSTFIXDEC_I);
         case G__OPR_PREFIXINC:
            return(G__OPR_PREFIXINC_I);
         case G__OPR_PREFIXDEC:
            return(G__OPR_PREFIXDEC_I);
      }
   }
   else if ('d' == val->type) {
      switch (opr) {
         case G__OPR_POSTFIXINC:
            return(G__OPR_POSTFIXINC_D);
         case G__OPR_POSTFIXDEC:
            return(G__OPR_POSTFIXDEC_D);
         case G__OPR_PREFIXINC:
            return(G__OPR_PREFIXINC_D);
         case G__OPR_PREFIXDEC:
            return(G__OPR_PREFIXDEC_D);
      }
   }
#ifdef G__NEVER /* following change rather slowed down */
   else if ('l' == val->type) {
      switch (opr) {
         case G__OPR_POSTFIXINC:
            return(G__OPR_POSTFIXINC_L);
         case G__OPR_POSTFIXDEC:
            return(G__OPR_POSTFIXDEC_L);
         case G__OPR_PREFIXINC:
            return(G__OPR_PREFIXINC_L);
         case G__OPR_PREFIXDEC:
            return(G__OPR_PREFIXDEC_L);
      }
   }
   else if ('s' == val->type) {
      switch (opr) {
         case G__OPR_POSTFIXINC:
            return(G__OPR_POSTFIXINC_S);
         case G__OPR_POSTFIXDEC:
            return(G__OPR_POSTFIXDEC_S);
         case G__OPR_PREFIXINC:
            return(G__OPR_PREFIXINC_S);
         case G__OPR_PREFIXDEC:
            return(G__OPR_PREFIXDEC_S);
      }
   }
   else if ('h' == val->type) {
      switch (opr) {
         case G__OPR_POSTFIXINC:
            return(G__OPR_POSTFIXINC_H);
         case G__OPR_POSTFIXDEC:
            return(G__OPR_POSTFIXDEC_H);
         case G__OPR_PREFIXINC:
            return(G__OPR_PREFIXINC_H);
         case G__OPR_PREFIXDEC:
            return(G__OPR_PREFIXDEC_H);
      }
   }
   else if ('R' == val->type) {
      switch (opr) {
         case G__OPR_POSTFIXINC:
            return(G__OPR_POSTFIXINC_R);
         case G__OPR_POSTFIXDEC:
            return(G__OPR_POSTFIXDEC_R);
         case G__OPR_PREFIXINC:
            return(G__OPR_PREFIXINC_R);
         case G__OPR_PREFIXDEC:
            return(G__OPR_PREFIXDEC_R);
      }
   }
   else if ('k' == val->type) {
      switch (opr) {
         case G__OPR_POSTFIXINC:
            return(G__OPR_POSTFIXINC_K);
         case G__OPR_POSTFIXDEC:
            return(G__OPR_POSTFIXDEC_K);
         case G__OPR_PREFIXINC:
            return(G__OPR_PREFIXINC_K);
         case G__OPR_PREFIXDEC:
            return(G__OPR_PREFIXDEC_K);
      }
   }
   else if ('f' == val->type) {
      switch (opr) {
         case G__OPR_POSTFIXINC:
            return(G__OPR_POSTFIXINC_F);
         case G__OPR_POSTFIXDEC:
            return(G__OPR_POSTFIXDEC_F);
         case G__OPR_PREFIXINC:
            return(G__OPR_PREFIXINC_F);
         case G__OPR_PREFIXDEC:
            return(G__OPR_PREFIXDEC_F);
      }
   }
#endif
   return(opr);
}

//______________________________________________________________________________
long G__op2_operator_detail(int opr, G__value* lval, G__value* rval)
{
   int lisdouble, risdouble;
   int lispointer, rispointer;

   /* don't optimze if optimize level is less than 3 */
   if (G__asm_loopcompile < 3) return(opr);

   switch (lval->type) {
      case 'q':
      case 'n':
      case 'm':
         return(opr);
   }
   switch (rval->type) {
      case 'q':
      case 'n':
      case 'm':
         return(opr);
   }

   if (0 == rval->type
         && 0 == G__xrefflag
      ) {
      G__genericerror("Error: Binary operator oprand missing");
   }

   lisdouble = G__isdouble(*lval);
   risdouble = G__isdouble(*rval);

   if (0 == lisdouble && 0 == risdouble) {
      lispointer = isupper(lval->type);
      rispointer = isupper(rval->type);
      if (0 == lispointer && 0 == rispointer) {
         if ('k' == lval->type || 'h' == lval->type ||
               'k' == rval->type || 'h' == rval->type) {
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
                  switch (lval->type) {
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
                  switch (lval->type) {
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
            switch (lval->type) {
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
                  break;
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

} /* extern "C" */

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
