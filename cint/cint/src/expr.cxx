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

// Static functions.
static void G__getiparseobject(G__value* result, char* item);
static G__value G__conditionaloperator(G__value defined, const char* expression, int ig1, char* ebuf);
static int G__iscastexpr_body(const char* ebuf, int lenbuf);
#ifdef G__PTR2MEMFUNC
static int G__getpointer2memberfunc(const char* item, G__value* presult);
#endif // G__PTR2MEMFUNC
static int G__getoperator(int newoperator, int oldoperator);

extern "C" {

// External functions.
char* G__setiparseobject(G__value* result, char* str);
G__value G__calc_internal(const char* exprwithspace);
G__value G__getexpr(const char* expression);
G__value G__getprod(char* expression1);
G__value G__getpower(const char* expression2);
G__value G__getitem(const char* item);
long G__test(const char* expr);
long G__btest(int operator2, G__value lresult, G__value rresult);
long double G__atolf(const char* expr);

// Functions in the C interface.
int G__lasterror();
void G__reset_lasterror();
G__value G__calc(const char* exprwithspace);

} // extern "C"

#ifndef G__ROOT
#define G__NOPOWEROPR
#endif // G__ROOT

//______________________________________________________________________________
#define G__iscastexpr(ebuf) \
   (lenbuf>3 && '('==ebuf[0] && ')'==ebuf[lenbuf-1] && \
    ('*'==ebuf[lenbuf-2] || '&'==ebuf[lenbuf-2] || \
     G__iscastexpr_body(ebuf,lenbuf)))

//______________________________________________________________________________
//
//  ANSI compliant operator precedences,  smaller the higher
//

#define G__PREC_SCOPE     1
#define G__PREC_FCALL     2
#define G__PREC_UNARY     3
#define G__PREC_P2MEM     4
#define G__PREC_PWR       5
#define G__PREC_MULT      6
#define G__PREC_ADD       7
#define G__PREC_SHIFT     8
#define G__PREC_RELATION  9
#define G__PREC_EQUAL    10
#define G__PREC_BITAND   11
#define G__PREC_BITEXOR  12
#define G__PREC_BITOR    13
#define G__PREC_LOGICAND 14
#define G__PREC_LOGICOR  15
#define G__PREC_TEST     16
#define G__PREC_ASSIGN   17
#define G__PREC_COMMA    18
#define G__PREC_NOOPR   100

//______________________________________________________________________________
#define G__expr_error \
   G__syntaxerror(expression); \
   return(G__null)

//______________________________________________________________________________
#ifdef G__ASM_DBG
#define G__ASSIGN_CNDJMP \
   if('O'==opr[op] && G__asm_noverflow) { \
      int store_pp_and = pp_and; \
      while(pp_and) { \
         if(G__asm_dbg) \
            G__fprinterr(G__serr,"   CNDJMP assigned %x&%x  %s:%d\n", G__asm_cp, ppointer_and[pp_and-1] - 1, __FILE__, __LINE__); \
         if(G__PVOID==G__asm_inst[ppointer_and[pp_and-1]]) \
            G__asm_inst[ppointer_and[--pp_and]] = G__asm_cp; \
         else --pp_and; \
      } \
      pp_and = store_pp_and; \
   }
#else // G__ASM_DBG
#define G__ASSIGN_CNDJMP \
   if('O'==opr[op] && G__asm_noverflow) { \
      int store_pp_and = pp_and; \
      while(pp_and) { \
         if(G__PVOID==G__asm_inst[ppointer_and[pp_and-1]]) \
            G__asm_inst[ppointer_and[--pp_and]] = G__asm_cp; \
         else --pp_and; \
      } \
      pp_and = store_pp_and; \
   }
#endif // G__ASM_DBG


//______________________________________________________________________________
//
//  Evaluate all operators in stack and get result as vstack[0].
//  This macro contributes to execution speed. Do not implement
//  using a function.
#define G__exec_evalall \
   /* Evaluate item */ \
   if(lenbuf) { \
      ebuf[lenbuf] = '\0'; \
      vstack[sp++] = G__getitem(ebuf); \
      lenbuf=0; \
      iscastexpr = 0; /* ON1342 */ \
   } \
   /* process unary operator */ \
   while(up && sp>=1) { \
      --up; \
      if('*'==unaopr[up]) { \
         vstack[sp-1] = G__tovalue(vstack[sp-1]); \
      } \
      else if('&'==unaopr[up]) {    /* ON717 */ \
         vstack[sp-1] = G__toXvalue(vstack[sp-1],'P'); \
      } \
      else { \
         vstack[sp] = vstack[sp-1]; \
         vstack[sp-1] = G__null; \
         G__bstore(unaopr[up],vstack[sp],&vstack[sp-1]); \
      } \
   } \
   /* process binary operator */ \
   while(op /* && opr[op-1]<=G__PROC_NOOPR */ && sp>=2) { \
      --op; \
      --sp; \
      G__ASSIGN_CNDJMP /* 1575 */ \
      G__bstore(opr[op],vstack[sp],&vstack[sp-1]); \
   } \
   if(1!=sp || op!=0 || up!=0) { G__expr_error; }

//______________________________________________________________________________
//
//  Evaluate all operators in stack and get result as vstack[0],
//  then push binary operator to operator stack.
//  This macro contributes to execution speed. Do not implement
//  using a function.
#define G__exec_binopr(oprin,precin) \
   /* evaluate left value */ \
   ebuf[lenbuf] = '\0'; \
   vstack[sp++] = G__getitem(ebuf); \
   lenbuf=0; \
   iscastexpr = 0; /* ON1342 */ \
   /* process unary operator */ \
   while(up && sp>=1) { \
      --up; \
      if('*'==unaopr[up]) { \
         vstack[sp-1] = G__tovalue(vstack[sp-1]); \
      } \
      else if('&'==unaopr[up]) {    /* ON717 */ \
         vstack[sp-1] = G__toXvalue(vstack[sp-1],'P'); \
      } else if ('-'==unaopr[up]&&oprin=='@') { \
         vstack[sp] = vstack[sp-1]; \
         vstack[sp-1] = G__getitem("-1"); \
         sp++; \
         opr[op] = '*'; \
         prec[op++] = G__PREC_PWR; \
      } \
      else { \
         vstack[sp] = vstack[sp-1]; \
         vstack[sp-1] = G__null; \
         G__bstore(unaopr[up],vstack[sp],&vstack[sp-1]); \
      } \
   } \
   /* process higher precedence operator at left */ \
   while(op && prec[op-1]<=precin && sp>=2) { \
      --op; \
      --sp; \
      G__ASSIGN_CNDJMP /* 1575 */ \
      G__bstore(opr[op],vstack[sp],&vstack[sp-1]); \
   } \
   /* set operator */ \
   opr[op] = oprin; \
   if(G__PREC_NOOPR!=precin) prec[op++] = precin

//______________________________________________________________________________
#define G__exec_unaopr(oprin) \
   unaopr[up++] = oprin

//______________________________________________________________________________
#define G__exec_oprassignopr(oprin) \
   G__exec_evalall \
   vstack[1] = G__getexpr(expression+ig1+1); \
   G__bstore(oprin,vstack[1],&vstack[0]); \
   G__var_type='p'; \
   return(vstack[0])

//______________________________________________________________________________
#define G__wrap_binassignopr(oprin,precin,assignopr) \
   if((nest==0)&&(single_quote==0)&&(double_quote==0)) { \
      if(0==lenbuf) { G__expr_error; } \
      if('='==expression[ig1+1]) { \
         /* a@=b, a@=b */ \
         ++ig1; \
         G__exec_oprassignopr(assignopr); \
      } \
      else { \
         /* a@b, a@b */ \
         G__exec_binopr(c,precin); \
      } \
   } \
   else ebuf[lenbuf++]=c

//______________________________________________________________________________
#define G__wrap_plusminus(oprin,assignopr,preincopr,postincopr) \
   if((nest==0)&&(single_quote==0)&&(double_quote==0)) { \
      if(oprin==expression[ig1+1] \
            && (!lenbuf||(!isdigit(ebuf[0])&&'.'!=ebuf[0]))  /* 1831 */ \
        ) { \
         if(lenbuf) { \
            if('='==expression[ig1+2] && 'v'==G__var_type) { \
               /* *a++=expr */ \
               G__var_type='p'; \
               ebuf[lenbuf++]=c; \
               ebuf[lenbuf++]=c; \
               ++ig1; \
            } \
            else if(iscastexpr) { /* added ON1342 */ \
               ebuf[lenbuf++]=c; \
               ebuf[lenbuf++]=c; \
               ++ig1; \
            } \
            else if(isalnum(expression[ig1+2])|| /* 2008 */ \
                    '.'==expression[ig1+2]||'_'==expression[ig1+2]) { \
               /* a+ +b, a- -b */ \
               ebuf[lenbuf]=0; \
               ++ig1; \
               G__exec_binopr('+',G__PREC_ADD); \
            } \
            else { \
               /* a++, a-- */ \
               ++ig1; \
               if('v'==G__var_type) { \
                  G__exec_unaopr('*'); \
                  G__var_type = 'p'; \
               } \
               unaopr[up++] = postincopr; \
               /* G__exec_binopr(0,G__PREC_NOOPR); */ \
            } \
         } \
         else { \
            /* *++a = expr should be handled at assignment oprerator */ \
            /* ++a, --a */ \
            ++ig1; \
            if('v'==G__var_type) { \
               G__exec_unaopr('*'); \
               G__var_type = 'p'; \
            } \
            G__exec_unaopr(preincopr); \
         } \
      } \
      else if('='==expression[ig1+1]) { \
         /* +=, -= */ \
         if(0==lenbuf) { G__expr_error; } \
         ++ig1; \
         G__exec_oprassignopr(assignopr); \
      } \
      else if('>'==expression[ig1+1]) { \
         /* a->b */ \
         ++ig1; \
         ebuf[lenbuf++]=c; \
         ebuf[lenbuf++]=expression[ig1]; \
      } \
      else if(lenbuf) { \
         char *pebuf; \
         if('e'==tolower(expression[ig1-1])&& \
               !(expression[0]=='0' && 'x'==tolower(expression[1])) &&   /* Properly handle 0x0E */ \
               (isdigit(ebuf[0])||'.'==ebuf[0]|| \
                ('('==ebuf[0]&&(pebuf=strchr(ebuf,')'))&& \
                 (isdigit(*++pebuf)||'.'==(*pebuf))))) { \
            /* 1e+10, 1e-10, (double)1e+6 */ \
            ebuf[lenbuf++]=c; \
         } \
         else { \
            ebuf[lenbuf]=0; /* ON742 */ \
            if(!G__iscastexpr(ebuf)) { \
               /* a+b, a-b */ \
               G__exec_binopr(c,G__PREC_ADD); \
            } \
            else { \
               /* (int)-abc */ \
               ebuf[lenbuf++]=c; \
            } \
            /* G__exec_binopr(c,G__PREC_ADD); ON742 */ \
         } \
      } \
      else if('-'==c) { \
         /* -a */ \
         G__exec_unaopr(oprin); \
      } \
      /* else +a , ignored */ \
   } \
   else ebuf[lenbuf++]=c

//______________________________________________________________________________
#define G__wrap_shifts(oprin,assignopr,shiftopr,relationopr) \
   if(oprin==expression[ig1+1]) { \
      if('='==expression[ig1+2]) { \
         /* a<<=b */ \
         ig1+=2; \
         G__exec_oprassignopr(assignopr); \
      } \
      else { \
         /* a<<b */ \
         ++ig1; \
         G__exec_binopr(shiftopr,G__PREC_SHIFT); \
      } \
   } \
   else if('='==expression[ig1+1]) { \
      /* a<=b */ \
      ++ig1; \
      G__exec_binopr(relationopr,G__PREC_RELATION); \
   } \
   else { \
      /* a<b */ \
      G__exec_binopr(c,G__PREC_RELATION); \
   } \

//
#ifdef G__ASM_DBG
//

//______________________________________________________________________________
#define G__SUSPEND_ANDOPR \
   if('u'!=vstack[sp-1].type) { \
      store_no_exec_compile_and[pp_and] = G__no_exec_compile; \
      if(!G__no_exec_compile && 0.0 == G__double(vstack[sp-1])) { \
         if(G__asm_dbg) G__fprinterr(G__serr,"   G__no_exec_compile set\n"); \
         G__no_exec_compile = 1; \
         vtmp_and = vstack[sp-1]; \
      } \
      if(G__asm_noverflow) { \
         if(G__asm_dbg) { \
            G__fprinterr(G__serr,"%3x,%3x: PUSHCPY\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__); \
            G__fprinterr(G__serr,"%3x,%3x: CNDJMP (assigned later)\n", G__asm_cp+1, G__asm_dt, __FILE__, __LINE__); \
         } \
         G__asm_inst[G__asm_cp]=G__PUSHCPY; \
         G__asm_inst[G__asm_cp+1]=G__CNDJMP; \
         G__asm_inst[G__asm_cp+2] = G__PVOID; /* 1575 */ \
         ppointer_and[pp_and] = G__asm_cp+2; \
         G__inc_cp_asm(3,0); \
      } \
      ++pp_and; \
   }

//______________________________________________________________________________
#define G__SUSPEND_OROPR \
   if('u'!=vstack[sp-1].type) { \
      store_no_exec_compile_or[pp_or] = G__no_exec_compile; \
      if(!G__no_exec_compile && 0.0 != G__double(vstack[sp-1])) { \
         if(G__asm_dbg) G__fprinterr(G__serr,"   G__no_exec_compile set\n"); \
         G__no_exec_compile = 1; \
         vstack[sp-1] = G__one; \
         vtmp_or = vstack[sp-1]; \
      } \
      if(G__asm_noverflow) { \
         if(G__asm_dbg) { \
            G__fprinterr(G__serr,"%3x,%3x: BOOL  %s:%d\n", G__asm_cp, G__asm_dt, __FILE__, __LINE__); \
            G__fprinterr(G__serr,"%3x,%3x: PUSHCPY  %s:%d\n", G__asm_cp + 1, G__asm_dt, __FILE__, __LINE__); \
            G__fprinterr(G__serr,"%3x,%3x: CND1JMP (assigned later)  %s:%d\n", G__asm_cp + 2, G__asm_dt, __FILE__, __LINE__); \
         } \
         G__asm_inst[G__asm_cp]=G__BOOL; \
         G__asm_inst[G__asm_cp+1]=G__PUSHCPY; \
         G__asm_inst[G__asm_cp+2]=G__CND1JMP; \
         G__asm_inst[G__asm_cp+3] = G__PVOID; /* 1575 */ \
         ppointer_or[pp_or] = G__asm_cp+3; \
         G__inc_cp_asm(4,0); \
      } \
      ++pp_or; \
   }

//______________________________________________________________________________
#define G__RESTORE_NOEXEC_ANDOPR \
   if(pp_and) { \
      if(G__asm_dbg) G__fprinterr(G__serr,"    G__no_exec_compile reset %d\n", store_no_exec_compile_and[0]); \
      if(!store_no_exec_compile_and[0]&&G__no_exec_compile) \
         vstack[sp-1] = vtmp_and; \
      G__no_exec_compile = store_no_exec_compile_and[0]; \
   }

//______________________________________________________________________________
#define DBGCOM \
   G__fprinterr(G__serr,"pp_and=%d G__decl=%d\n",pp_and,G__decl);

//______________________________________________________________________________
#define G__RESTORE_ANDOPR \
   if(G__asm_noverflow) { \
      while(pp_and) { \
         if(G__asm_dbg) \
            G__fprinterr(G__serr,"   %3x: CNDJMP assigned for AND %3x  %s:%d\n", ppointer_and[pp_and-1] - 1, G__asm_cp, __FILE__, __LINE__); \
         if(G__PVOID==G__asm_inst[ppointer_and[pp_and-1]]) /* 1575 */ \
            G__asm_inst[ppointer_and[--pp_and]] = G__asm_cp; \
         else --pp_and; /* 1575 */ \
      } \
   } \
   else while(pp_and) {--pp_and;/*1524*/}

//______________________________________________________________________________
#define G__RESTORE_NOEXEC_OROPR \
   if(pp_or) { \
      if(G__asm_dbg) G__fprinterr(G__serr,"    G__no_exec_compile reset %d\n", store_no_exec_compile_or[0]); \
      if(!store_no_exec_compile_or[0]&&G__no_exec_compile) \
         vstack[sp-1] = vtmp_or; \
      G__no_exec_compile = store_no_exec_compile_or[0]; \
   }

//______________________________________________________________________________
#define G__RESTORE_OROPR \
   if(G__asm_noverflow) { \
      while(pp_or) { \
         if(G__asm_dbg) \
            G__fprinterr(G__serr,"   %3x: CND1JMP assigned for OR %3x  %s:%d\n", ppointer_or[pp_or-1] - 1, G__asm_cp, __FILE__, __LINE__); \
         G__asm_inst[ppointer_or[--pp_or]] = G__asm_cp; \
      } \
   } \
   else while(pp_or) {--pp_or;/*1524*/}

//
#else // G__ASM_DBG
//

//______________________________________________________________________________
#define G__SUSPEND_ANDOPR \
   if('u'!=vstack[sp-1].type) { \
      store_no_exec_compile_and[pp_and] = G__no_exec_compile; \
      if(!G__no_exec_compile && 0.0 ==G__double(vstack[sp-1])) { \
         G__no_exec_compile = 1; \
         vtmp_and = vstack[sp-1]; \
      } \
      if(G__asm_noverflow) { \
         G__asm_inst[G__asm_cp]=G__PUSHCPY; \
         G__asm_inst[G__asm_cp+1]=G__CNDJMP; \
         G__asm_inst[G__asm_cp+2] = G__PVOID; \
         ppointer_and[pp_and] = G__asm_cp+2; \
         G__inc_cp_asm(3,0); \
      } \
      ++pp_and; \
   }

//______________________________________________________________________________
#define G__SUSPEND_OROPR \
   if('u'!=vstack[sp-1].type) { \
      store_no_exec_compile_or[pp_or] = G__no_exec_compile; \
      if(!G__no_exec_compile && 0.0 != G__double(vstack[sp-1])) { \
         G__no_exec_compile = 1; \
         vstack[sp-1] = G__one; \
         vtmp_or = vstack[sp-1]; \
      } \
      if(G__asm_noverflow) { \
         G__asm_inst[G__asm_cp]=G__BOOL; \
         G__asm_inst[G__asm_cp+1]=G__PUSHCPY; \
         G__asm_inst[G__asm_cp+2]=G__CND1JMP; \
         G__asm_inst[G__asm_cp+3] = G__PVOID; /* 1575 */ \
         ppointer_or[pp_or] = G__asm_cp+3; \
         G__inc_cp_asm(4,0); \
      } \
      ++pp_or; \
   }

//______________________________________________________________________________
#define G__RESTORE_NOEXEC_ANDOPR \
   if(pp_and) { \
      if(!store_no_exec_compile_and[0]&&G__no_exec_compile) \
         vstack[sp-1] = vtmp_and; \
      G__no_exec_compile = store_no_exec_compile_and[0]; \
   }

//______________________________________________________________________________
#define G__RESTORE_ANDOPR \
   if(G__asm_noverflow) { \
      while(pp_and) { \
         if(G__PVOID==G__asm_inst[ppointer_and[pp_and-1]]) \
            G__asm_inst[ppointer_and[--pp_and]] = G__asm_cp; \
         else --pp_and; \
      } \
   } \
   else while(pp_and) {--pp_and;}

//______________________________________________________________________________
#define G__RESTORE_NOEXEC_OROPR \
   if(pp_or) { \
      if(!store_no_exec_compile_or[0]&&G__no_exec_compile) \
         vstack[sp-1] = vtmp_or; \
      G__no_exec_compile = store_no_exec_compile_or[0]; \
   }

//______________________________________________________________________________
#define G__RESTORE_OROPR \
   if(G__asm_noverflow) { \
      while(pp_or) { \
         G__asm_inst[ppointer_or[--pp_or]] = G__asm_cp; \
      } \
   } \
   else while(pp_or) {--pp_or;}

//
#endif // G__ASM_DBG
//

//______________________________________________________________________________
#define G__STACKDEPTH 100

//______________________________________________________________________________
//
//  Static functions.
//

//______________________________________________________________________________
static void G__getiparseobject(G__value* result, char* item)
{
   // --
   /* '_$trc_[tagnum]_[addr]' */
   char *xtmp = item + 6;
   char *xx = strchr(xtmp, '_');
   // assert(xx != 0);
   result->type = item[2];
   result->obj.reftype.reftype = (int)(item[3] - '0');
   result->isconst = (G__SIGNEDCHAR_T)(item[4] - '0');
   result->typenum = -1;
   *xx = 0;
   result->tagnum = atoi(xtmp);
   *xx = '_';
   result->obj.i = atol(xx + 2);
   if ('M' == xx[1]) result->obj.i = -result->obj.i;
   result->ref = result->obj.i;
}

//______________________________________________________________________________
static G__value G__conditionaloperator(G__value defined, const char* expression, int ig1, char* ebuf)
{
   // -- Evaluate a?b:c operator.
   long tempop = 0;
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
static int G__iscastexpr_body(const char* ebuf, int lenbuf)
{
   // --
   int result;
   G__FastAllocString temp(ebuf+1);
   temp[lenbuf-2] = 0;
   // Using G__istypename() is questionable.
   // May need to use G__string2type() for better language compliance.
   result = G__istypename(temp);
   return result;
}

#ifdef G__PTR2MEMFUNC
//______________________________________________________________________________
static int G__getpointer2memberfunc(const char* item, G__value* presult)
{
   int hash = 0;
   long scope_struct_offset = 0;
   int scope_tagnum = -1;
   int ifn;
   struct G__ifunc_table_internal *memfunc;
   const char *p = strstr(item, "::");

   if (!p) return(0);

   G__scopeoperator((char*)item, &hash, &scope_struct_offset, &scope_tagnum);
   if (scope_tagnum < 0 || scope_tagnum >= G__struct.alltag) return(0);

   G__incsetup_memfunc(scope_tagnum);
   memfunc = G__struct.memfunc[scope_tagnum];

   while (memfunc) {
      for (ifn = 0;ifn < memfunc->allifunc;ifn++) {
         if (strcmp(item, memfunc->funcname[ifn]) == 0) {
            // --
            // For the time being, pointer to member function can only be handled as function name.
            if (('n' == G__struct.type[scope_tagnum] || memfunc->staticalloc[ifn])
                  && memfunc->pentry[ifn]->size < 0
                  && memfunc->pentry[ifn]->tp2f) {
               G__letint(presult, 'Y', (long)memfunc->pentry[ifn]->tp2f);
            }
            else {
               G__letint(presult, 'C', (long)memfunc->funcname[ifn]);
            }
            presult->tagnum = -1;
            presult->typenum = -1;
            presult->ref = 0;
            return 1;
         }
      }
      memfunc = memfunc->next;
   }
   return 0;
}
#endif // G__PTR2MEMFUNC

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
   return(oldoperator);
}

//______________________________________________________________________________
//
//  External functions.
//

//______________________________________________________________________________
char* G__setiparseobject(G__value* result, G__FastAllocString &str)
{
   // --
   str.Format("_$%c%d%c_%d_%c%lu"
           , result->type
           , 0
           , (0 == result->isconst) ? '0' : '1'
           , result->tagnum
           , (result->obj.i < 0) ? 'M' : 'P'
           , labs(result->obj.i)
          );
   return(str);
}

//______________________________________________________________________________
static bool G__IsIdentifier(int c) {
   // Check for character that is valid for an identifier.
   // If start is true, digits are not allowed
   return isalnum(c) || c == '_';
}

//______________________________________________________________________________
extern "C"
G__value G__calc_internal(const char* exprwithspace)
{
   // -- Grand entry for C/C++ expression evaluator.
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
   char *exprnospace = (char*)malloc(strlen(exprwithspace) + 2);
   int iin = 0, iout = 0;
   int single_quote = 0, double_quote = 0;
   G__value result;
   int store_asm_exec = G__asm_exec;
   int store_asm_noverflow = G__asm_noverflow;
   G__asm_noverflow = 0;
   G__asm_exec = 0;
   exprnospace[0] = '\0';

   bool isdelete = false;
   bool isdeletearr = false;

   while (exprwithspace[iin] != '\0') {
      bool next_double_quote = double_quote;
      bool next_single_quote = single_quote;
      bool skipchar = false;
      switch (exprwithspace[iin]) {
         case '"' : /* double quote */
            if (single_quote == 0) {
               next_double_quote ^= 1;
            }
            break;
         case '\'' : /* single quote */
            if (double_quote == 0) {
               next_single_quote ^= 1;
            }
            break;
         case ';' : /* semi-column */
            if (single_quote==0 && double_quote==0) skipchar = true;
            // intentional fall-through:
         case '\n': /* end of line */
         case '\r': /* end of line */
         case ' ' : /* space */
         case '\t' : /* tab */
            exprnospace[iout] = '\0'; /* temporarily terminate string */
            if (iout == 8 && strncmp(exprnospace, "delete[]", 8) == 0) {
               iout = 0;
               isdeletearr = true;
            }
            else if (iout == 6 && strncmp(exprnospace, "delete", 6) == 0) {
               iout = 0;
               isdelete = true;
            }
            break;
         default :
            break;
      }
      // adapted from fread's G__fgetstream_newtemplate_internal()
      if (iout > 0 && !single_quote && !double_quote
          && isspace(exprnospace[iout - 1])) {
         char c = exprwithspace[iin];

         // We want to append to a space. Do we keep it?
         if (isspace(c)) --iout; // replace ' ' by ' '
         else if (iout == 1) {
            // string is " " - remove leading space.
            --iout;
         } else {
            char pp = exprnospace[iout - 2];
            // We only keep spaces between "identifiers" like "new const long long"
            // and between '> >'
            if ((G__IsIdentifier(pp) && G__IsIdentifier(c)) || (pp == '>' && c == '>')) {
            } else {
               // replace previous ' '
               --iout;
            }
         }
      }
      if (!skipchar) {
         exprnospace[iout++] = exprwithspace[iin++];
      } else {
         ++iin;
      }
      double_quote = next_double_quote;
      single_quote = next_single_quote;
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
extern "C"
G__value G__getexpr(const char* expression)
{
   // -- Grand entry for C/C++ expression evaluator. Space chars must be removed.
   //printf("Begin G__getexpr('%s') ...\n", expression);
   G__value vstack[G__STACKDEPTH]; /* evaluated value stack */
   int sp = 0;                       /* stack pointer */
   int opr[G__STACKDEPTH]; /* operator stack */
   int prec[G__STACKDEPTH];/* operator precedence */
   int op = 0;               /* operator stack pointer */
   int unaopr[G__STACKDEPTH]; /* unary operator stack */
   int up = 0;                    /* unary operator stack pointer */
   char c; /* temp char */
   int ig1 = 0;  /* input expression pointer */
   int nest = 0; /* parenthesis nesting state variable */
   int single_quote = 0, double_quote = 0; /* quotation flags */
   long iscastexpr = 0; /* whether this expression start with a cast */
   G__value defined = G__null;
   char store_var_type = G__var_type;
   int explicitdtor = 0;
   size_t inew = 0; /* ON994 */
   int pp_and = 0, pp_or = 0;
   int ppointer_and[G__STACKDEPTH], ppointer_or[G__STACKDEPTH];
   int store_no_exec_compile_and[G__STACKDEPTH];
   int store_no_exec_compile_or[G__STACKDEPTH];
   G__value vtmp_and, vtmp_or;

   //
   // Return null for no expression.
   //
   size_t length = strlen(expression);
   if (!length) {
      return G__null;
   }

   G__FastAllocString ebuf(length);
   size_t lenbuf = 0;

   //
   // Operator expression.
   //
   for (ig1 = 0; ig1 < (int)length; ++ig1) {
      c = expression[ig1];
      if (!single_quote && !double_quote) {
         if (lenbuf > 1 && ebuf[lenbuf - 1] == ' ') {
            // we had a space - do we keep it?
            char beforeSpaceChar = ebuf[lenbuf - 2];
            if (((isalnum(c) || c == '_') && (isalnum(beforeSpaceChar) || beforeSpaceChar == '_'))
                || (c == '>' && beforeSpaceChar == '>')) {}
            else {
               // not two identifiers / template "> >" - replace the space
               lenbuf--;
               ebuf[lenbuf] = 0;
            }
         }
      }
      switch (c) {

            /***************************************************
             * quotation
             ****************************************************/
         case '"':
            if (single_quote == 0) double_quote ^= 1;
            ebuf[lenbuf++] = c;
            break;
         case '\'':
            if (double_quote == 0) single_quote ^= 1;
            ebuf[lenbuf++] = c;
            break;

            /***************************************************
             * parenthesis
             ****************************************************/
         case '(': /* new(arena) type(),  (type)val, (expr) */
            if ((nest == 0) && (single_quote == 0) && (double_quote == 0) &&
                  lenbuf == 3 && strncmp(expression + inew, "new", 3) == 0) { /* ON994 */
               return(G__new_operator(expression + ig1));
            }
            /* no break here */
         case '[':
         case '{':
            if ((double_quote == 0) && (single_quote == 0)) {
               nest++;
               ebuf[lenbuf++] = c;
               inew = ig1 + 1;
            }
            else ebuf[lenbuf++] = c;
            break;

         case ')':
         case ']':
         case '}':
            if ((double_quote == 0) && (single_quote == 0)) {
               nest--;
               ebuf[lenbuf++] = c;
               inew = ig1 + 1;
               if (!iscastexpr && '(' == ebuf[0]) {
                  ebuf[lenbuf] = '\0';
                  iscastexpr = G__iscastexpr(ebuf);
               }
            }
            else ebuf[lenbuf++] = c;
            break;

            /***************************************************
             * operators
             ****************************************************/
         case ' ': /* new type, new (arena) type */
            if ((nest == 0) && (single_quote == 0) && (double_quote == 0)) {
               if (lenbuf - inew == 3 && strncmp(expression + inew, "new", 3) == 0) { /* ON994 */
                  return(G__new_operator(expression + ig1 + 1));
               }
               if (lenbuf && ebuf[lenbuf - 1] != ' ') {
                  // keep space for now; if statement checking for beforeSpaceChar will
                  // later determine whether it's worth keeping this space.
                  ebuf[lenbuf++] = c;
               } else {
                  // collapse multiple spaces into one
                  inew = ig1 + 1;
               }
            }
            else ebuf[lenbuf++] = c;
            break;
         case '!': /* !a, a!=b */
            if ((nest == 0) && (single_quote == 0) && (double_quote == 0)) {
               if ('=' == expression[ig1+1]) {
                  /* a!=b */
                  ++ig1;
                  if (0 == lenbuf) {
                     G__expr_error;
                  }
                  G__exec_binopr(G__OPR_NE, G__PREC_EQUAL);
                  break;
               }
            }
            /* no break here */
         case '~': /* ~a */
            if ((nest == 0) && (single_quote == 0) && (double_quote == 0)) {
               if (lenbuf) {
                  /* a->~b(), a::~b(), a.~b() */
                  explicitdtor = 1;
                  ebuf[lenbuf++] = c;
               }
               else {
                  /* ~a, !a */
                  G__exec_unaopr(c);
               }
            }
            else ebuf[lenbuf++] = c;
            break;
         case '/': /* a/b, a/=b */
            G__wrap_binassignopr(c, G__PREC_MULT, G__OPR_DIVASSIGN);
            break;
         case '%': /* a%b, a%=b */
            G__wrap_binassignopr(c, G__PREC_MULT, G__OPR_MODASSIGN);
            break;
         case '^': /* a^b, a^=b */
            G__wrap_binassignopr(c, G__PREC_BITEXOR, G__OPR_EXORASSIGN);
            break;
         case '+': /* ++a, a++, +a, a+b, a+=b, 1e+10, a+ +b */
            G__wrap_plusminus(c, G__OPR_ADDASSIGN, G__OPR_PREFIXINC, G__OPR_POSTFIXINC);
            break;
         case '-': /* --a, a--, -a, a-b, a-=b, 1e-10, a->b , a- -b */
            G__wrap_plusminus(c, G__OPR_SUBASSIGN, G__OPR_PREFIXDEC, G__OPR_POSTFIXDEC);
            break;
         case '<': /* a<<b, a<b, a<=b, a<<=b */
            if (nest == 0 && single_quote == 0 && double_quote == 0 && explicitdtor == 0) {
               ebuf[lenbuf] = '\0';
               if (G__defined_templateclass(ebuf)) {
                  ++ig1;
                  ebuf[lenbuf++] = c;
                  c = G__getstream_template(expression, &ig1, ebuf, lenbuf, ">");
                  lenbuf = strlen(ebuf);
                  ebuf[lenbuf++] = c;
                  ebuf[lenbuf] = '\0';
                  --ig1;
                  /* try to instantiate the template */
                  (void)G__defined_tagname(ebuf, 1);
                  lenbuf = strlen(ebuf);
                  break;
               }
               else if (strchr(expression + ig1, '>') &&
                        (G__defined_templatefunc(ebuf)
                         || G__defined_templatememfunc(ebuf)
                        )) {
                  ++ig1;
                  ebuf[lenbuf++] = c;
                  c = G__getstream_template(expression, &ig1, ebuf, lenbuf, ">");
                  if ('>' == c) ebuf += ">";
                  lenbuf = strlen(ebuf);
                  c = G__getstream_template(expression, &ig1, ebuf, lenbuf, "(");
                  if ('(' == c) ebuf += "(";
                  lenbuf = strlen(ebuf);
                  c = G__getstream_template(expression, &ig1, ebuf, lenbuf, ")");
                  if (')' == c) ebuf += ")";
                  lenbuf = strlen(ebuf);
                  --ig1;
                  break;
               }
               else if (strcmp(ebuf, "dynamic_cast") == 0 ||
                        strcmp(ebuf, "static_cast") == 0 ||
                        strcmp(ebuf, "reinterpret_cast") == 0 ||
                        strcmp(ebuf, "const_cast") == 0) {
                  /* TODO, implement casts, may need to introduce new instruction */
                  ++ig1;
                  ebuf[0] = '(';
                  c = G__getstream_template(expression, &ig1, ebuf, 1, ">");
                  lenbuf = strlen(ebuf);
                  ebuf += ")";
                  ++lenbuf;
                  --ig1;
                  break;
               }
               G__wrap_shifts(c, G__OPR_LSFTASSIGN, G__OPR_LSFT, G__OPR_LE)
            }
            else ebuf[lenbuf++] = c;
            break;
         case '>': /* a>>b, a>b, a>=b, a>>=b */
            if (nest == 0 && single_quote == 0 && double_quote == 0 && explicitdtor == 0) {
               G__wrap_shifts(c, G__OPR_RSFTASSIGN, G__OPR_RSFT, G__OPR_GE)
            }
            else ebuf[lenbuf++] = c;
            break;

         case '@': /* a@b */
            if ((nest == 0) && (single_quote == 0) && (double_quote == 0)) {
               if (0 == lenbuf) {
                  G__expr_error;
               }
               G__exec_binopr(c, G__PREC_PWR);
            }
            else ebuf[lenbuf++] = c;
            break;
         case '*': /* *a, a*b, a*=b, a**b, **a */
            if ((nest == 0) && (single_quote == 0) && (double_quote == 0)) {
               if ('=' == expression[ig1+1]) {
                  /* a*=b */
                  ++ig1;
                  G__exec_oprassignopr(G__OPR_MULASSIGN);
               }
               else if (c == expression[ig1+1]) {
                  if (lenbuf) {
#ifndef G__NOPOWEROPR
                     /* a**b handle as power operator */
                     ++ig1;
                     G__exec_binopr('@', G__PREC_PWR);
#else // G__NOPOWEROPR
                     /* a**b handle as a*(*b) */
                     G__exec_binopr('*', G__PREC_MULT);
                     G__exec_unaopr('*');
                     ++ig1;
#endif // G__NOPOWEROPR
                  }
                  else {
                     /* **a */
                     ++ig1;
                     G__exec_unaopr(c);
                     G__exec_unaopr(c);
                  }
               }
               else if (lenbuf) {
                  ebuf[lenbuf] = 0;
                  if (!G__iscastexpr(ebuf)) {
                     /* a*b */
                     G__exec_binopr(c, G__PREC_MULT);
                  }
                  else {
                     /* (int)*abc */
                     ebuf[lenbuf++] = c;
                  }
               }
               else {
                  /* *a */
                  G__exec_unaopr(c);
               }
            }
            else ebuf[lenbuf++] = c;
            break;
         case '&': /* &a, a&b, a&&b, a&=b */
            if ((nest == 0) && (single_quote == 0) && (double_quote == 0)) {
               if (c == expression[ig1+1]) {
                  /* a&&b */
                  ++ig1;
                  G__exec_binopr('A', G__PREC_LOGICAND);
                  G__SUSPEND_ANDOPR;
               }
               else if ('=' == expression[ig1+1]) {
                  /* a&=b */
                  ++ig1;
                  G__exec_oprassignopr(G__OPR_BANDASSIGN);
               }
               else if (lenbuf) {
                  ebuf[lenbuf] = 0;
                  if (!G__iscastexpr(ebuf)) {
                     /* a&b */
                     G__exec_binopr(c, G__PREC_BITAND);
                  }
                  else {
                     /* (int*)&abc */
                     ebuf[lenbuf++] = c;
                  }
               }
               else {
                  /* &a */
                  G__exec_unaopr(c); /* ON717 */
               }
            }
            else ebuf[lenbuf++] = c;
            break;
         case '|': /* a|b, a||b, a|=b */
            if ((nest == 0) && (single_quote == 0) && (double_quote == 0)) {
               if (c == expression[ig1+1]) {
                  /* a||b */
                  ++ig1;
                  G__exec_binopr('O', G__PREC_LOGICOR);
                  G__RESTORE_NOEXEC_ANDOPR
                  G__RESTORE_ANDOPR
                  G__SUSPEND_OROPR;
               }
               else if ('=' == expression[ig1+1]) {
                  /* a|=b */
                  ++ig1;
                  G__exec_oprassignopr(G__OPR_BORASSIGN);
               }
               else if (lenbuf) {
                  /* a&b */
                  G__exec_binopr(c, G__PREC_BITOR);
               }
               else {
                  /* &a */
                  G__exec_unaopr(c);
               }
            }
            else ebuf[lenbuf++] = c;
            break;

            /***************************************************
             * lowest precedence, a=b and a?b:c
             ****************************************************/
         case '=': /* a==b, a=b */
            if ((nest == 0) && (single_quote == 0) && (double_quote == 0)) {
               if (c == expression[ig1+1]) {
                  /* a==b */
                  ++ig1;
                  G__exec_binopr(G__OPR_EQ, G__PREC_EQUAL);
               }
               else {
                  /* a=b */
                  G__var_type = 'p';
                  defined = G__getexpr(expression + ig1 + 1);
                  strncpy(ebuf, expression, ig1);
                  ebuf[ig1] = '\0';
                  G__var_type = store_var_type;
                  vstack[0] = G__letvariable(ebuf, defined, &G__global, G__p_local);
                  return(vstack[0]);
               }
               inew = ig1 + 1;
            }
            else ebuf[lenbuf++] = c;
            break;
         case '?': /* a?b:c */
            if (!nest && !single_quote && !double_quote) {
               G__exec_evalall
               G__RESTORE_NOEXEC_ANDOPR
               G__RESTORE_NOEXEC_OROPR
               G__RESTORE_ANDOPR
               G__RESTORE_OROPR
               vstack[1] = G__conditionaloperator(vstack[0], expression, ig1, ebuf);
               return vstack[1];
            }
            else {
               ebuf[lenbuf++] = c;
            }
            break;

         case '\\' :
            ebuf[lenbuf++] = c;
            ebuf[lenbuf++] = expression[++ig1];
            break;

            /***************************************************
             * non-operator characters
             ****************************************************/
         default:
            ebuf[lenbuf++] = c;
            break;
      }
   }
   //
   // Evaluate operators in stack.
   //
   G__exec_evalall
   G__RESTORE_NOEXEC_ANDOPR
   G__RESTORE_NOEXEC_OROPR
   G__RESTORE_ANDOPR
   G__RESTORE_OROPR
   return vstack[0];
}

//______________________________________________________________________________
extern "C"
G__value G__getprod(char* expression1)
{
   // --
   G__value defined1, reg;
   G__FastAllocString ebuf1(G__ONELINE);
   int operator1, prodpower = 0;
   int lenbuf1 = 0;
   size_t ig11, ig2;
   size_t length1;
   int nest1 = 0;
   int single_quote = 0, double_quote = 0;


   operator1 = '\0';
   defined1 = G__null;
   length1 = strlen(expression1);
   if (length1 == 0) return(G__null);

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

   for (ig11 = 0;ig11 < length1;ig11++) {
      switch (expression1[ig11]) {
         case '"' : /* double quote */
            if (single_quote == 0) {
               double_quote ^= 1;
            }
            ebuf1.Set(lenbuf1++, expression1[ig11]);
            break;
         case '\'' : /* single quote */
            if (double_quote == 0) {
               single_quote ^= 1;
            }
            ebuf1.Set(lenbuf1++, expression1[ig11]);
            break;
         case '*':
            if (strncmp(expression1, "new ", 4) == 0) {
               ebuf1.Set(lenbuf1++, expression1[ig11]);
               break;
            }
         case '/':
         case '%':
            if ((nest1 == 0) && (single_quote == 0) && (double_quote == 0)) {
               switch (lenbuf1) {
                  case 0:
                     operator1 = G__getoperator(operator1 , expression1[ig11]);
                     break;
                  default:
                     if (operator1 == '\0') operator1 = '*';
                     ebuf1.Set(lenbuf1, 0);
                     reg = G__getpower(ebuf1);
                     G__bstore(operator1, reg, &defined1);
                     lenbuf1 = 0;
                     ebuf1[0] = 0;
                     operator1 = expression1[ig11];
                     break;
               }
            }
            else {
               ebuf1.Set(lenbuf1++, expression1[ig11]);
            }
            break;
         case '(':
         case '[':
         case '{':
            if ((double_quote == 0) && (single_quote == 0)) {
               nest1++;
               ebuf1.Set(lenbuf1++, expression1[ig11]);
            }
            else {
               ebuf1.Set(lenbuf1++, expression1[ig11]);
            }
            break;
         case ')':
         case ']':
         case '}':
            if ((double_quote == 0) && (single_quote == 0)) {
               ebuf1.Set(lenbuf1++, expression1[ig11]);
               nest1--;
            }
            else {
               ebuf1.Set(lenbuf1++, expression1[ig11]);
            }
            break;
         case '@':
         case '~':
         case ' ':
            if ((nest1 == 0) && (single_quote == 0) && (double_quote == 0)) {
               prodpower = 1;
            }
            ebuf1.Set(lenbuf1++, expression1[ig11]);
            break;


         case '\\' :
            ebuf1.Set(lenbuf1++, expression1[ig11++]);
            ebuf1.Set(lenbuf1++, expression1[ig11]);
            break;

         default:
            ebuf1.Set(lenbuf1++, expression1[ig11]);
            break;
      }
   }
   ebuf1.Set(lenbuf1, 0);
   if ((nest1 != 0) || (single_quote != 0) || (double_quote != 0)) {
      G__parenthesiserror(expression1, "G__getprod");
      return(G__null);
   }
   if (prodpower != 0) {
      reg = G__getpower(ebuf1);
   }
   else {
      reg = G__getitem(ebuf1);
   }
   G__bstore(operator1, reg, &defined1);
   return(defined1);
}

//______________________________________________________________________________
extern "C"
G__value G__getpower(const char* expression2)
{
   // --
   G__value defined2, reg;
   G__FastAllocString ebuf2(G__ONELINE);
   int operator2;
   int lenbuf2 = 0;
   int ig12;
   int nest2 = 0;
   int single_quote = 0, double_quote = 0;

   if (expression2[0] == '\0') return(G__null);

   operator2 = '\0';
   defined2 = G__null;

   ig12 = 0;
   while (expression2[ig12] != '\0') {
      switch (expression2[ig12]) {
         case '"' : /* double quote */
            if (single_quote == 0) {
               double_quote ^= 1;
            }
            ebuf2.Set(lenbuf2++, expression2[ig12]);
            break;
         case '\'' : /* single quote */
            if (double_quote == 0) {
               single_quote ^= 1;
            }
            ebuf2.Set(lenbuf2++, expression2[ig12]);
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
                     ebuf2.Set(lenbuf2, 0);
                     reg = G__getitem(ebuf2);
                     G__bstore(operator2, reg, &defined2);
                     lenbuf2 = 0;
                     ebuf2[0] = 0;
                     operator2 = expression2[ig12];
                     break;
               }
            }
            else {
               ebuf2.Set(lenbuf2++, expression2[ig12]);
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
               ebuf2.Set(lenbuf2++, expression2[ig12]);
            }
            else {
               ebuf2.Set(lenbuf2++, expression2[ig12]);
            }
            break;
         case ')':
         case ']':
         case '}':
            if ((double_quote == 0) && (single_quote == 0)) {
               ebuf2.Set(lenbuf2++, expression2[ig12]);
               nest2--;
            }
            else {
               ebuf2.Set(lenbuf2++, expression2[ig12]);
            }
            break;

         case '\\' :
            ebuf2.Set(lenbuf2++, expression2[ig12++]);
            ebuf2.Set(lenbuf2++, expression2[ig12]);
            break;

         default :
            ebuf2.Set(lenbuf2++, expression2[ig12]);
            break;
      }
      ig12++;
   }
   ebuf2.Set(lenbuf2, 0);
   if ((nest2 != 0) || (single_quote != 0) || (double_quote != 0)) {
      G__parenthesiserror(expression2, "G__getpower");
      return(G__null);
   }
   reg = G__getitem(ebuf2);
   G__bstore(operator2, reg, &defined2);
   return(defined2);
}

//______________________________________________________________________________
extern "C"
G__value G__getitem(const char* item)
{
   // --
   int known;
   G__value result3;
   int c;
   char store_var_typeB;
   G__value reg;
   switch (item[0]) {
         /* constant */
      case '0':
         c = item[1];
         if (
            (c != '\0') &&
            (c != '.') &&
            ((c = tolower(c)) != 'f') &&
            (c != 'e') &&
            (c != 'l') &&
            (c != 'u') &&
            (c != 's')
         ) {
            result3 = G__checkBase(item, &known);
#ifdef G__ASM
            if (G__asm_noverflow) {
               /**************************************
                * G__LD instruction
                * 0 LD
                * 1 address in data stack
                * put result3
                **************************************/
#ifdef G__ASM_DBG
               if (G__asm_dbg) {
                  G__fprinterr(G__serr, "%3x,%3x: LD %ld  %s:%d\n", G__asm_cp, G__asm_dt, G__int(result3), __FILE__, __LINE__);
               }
#endif // G__ASM_DBG
               G__asm_inst[G__asm_cp] = G__LD;
               G__asm_inst[G__asm_cp+1] = G__asm_dt;
               G__asm_stack[G__asm_dt] = result3;
               G__inc_cp_asm(2, 1);
            }
#endif // G__ASM
            result3.tagnum = -1;
            result3.typenum = -1;
            result3.ref = 0;
            result3.isconst = G__CONSTVAR + G__STATICCONST;
            return result3;
         }
         // Intentionally fall through.
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
            if (c == 'q') {
               G__letLongdouble(&result3, c, G__atolf(item));
            } else {
               G__letdouble(&result3, c, atof(item));
            }
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
            }
         }
         if ('u' != c) {
            result3.tagnum = -1;
            result3.typenum = -1;
            result3.ref = 0;
         }
         result3.isconst = G__CONSTVAR + G__STATICCONST;
#ifdef G__ASM
         if (G__asm_noverflow) {
            /**************************************
             * G__LD instruction
             * 0 LD
             * 1 address in data stack
             * put result3
             **************************************/
            //
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: LD %ld  %s:%d\n", G__asm_cp, G__asm_dt, G__int(result3),  __FILE__, __LINE__);
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
         result3 = G__strip_singlequotation((char*)item);
         result3.tagnum = -1;
         result3.typenum = -1;
         result3.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
         result3.isconst = G__CONSTVAR;
#endif // G__OLDIMPLEMENTATION1259
         // --
#ifdef G__ASM
         if (G__asm_noverflow) {
            /**************************************
             * G__LD instruction
             * 0 LD
             * 1 address in data stack
             * put result3
             **************************************/
            // --
#ifdef G__ASM_DBG
            if (G__asm_dbg) {
               G__fprinterr(G__serr, "%3x,%3x: LD '%c'  %s:%d\n", G__asm_cp, G__asm_dt, (char) G__int(result3), __FILE__, __LINE__);
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
         result3.tagnum = -1;
         result3.typenum = -1;
         result3.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
         result3.isconst = G__CONSTVAR;
#endif // G__OLDIMPLEMENTATION1259
         // --
#ifdef G__ASM
         if (G__asm_noverflow) {
            G__asm_gen_strip_quotation(&result3);
         }
#endif // G__ASM
         return result3;
      case '-':
         reg = G__getitem(item + 1);
         result3 = G__null;
         G__bstore('-', reg, &result3);
         return result3;
      // --
      case '_':
         if ('$' == item[1]) {
            G__getiparseobject(&result3, (char*)item);
            return result3;
         }
      // --
      default:
         store_var_typeB = G__var_type;
         known = 0;
         G__var_type = 'p';
         // variable
         result3 = G__getvariable((char*)item, &known, &G__global, G__p_local);
         if (!known && (result3.tagnum != -1) && !result3.obj.i) {
            // this is "a.b", we know "a", but it has no "b" - there is no use
            // in looking at other places.
            if (G__noerr_defined == 0 && G__definemacro == 0)
               return G__interactivereturn();
            else
               return(G__null);
         }
         // function
         if (!known) {
            G__var_typeB = store_var_typeB;
            result3 = G__getfunction(item, &known, G__TRYNORMAL);
            if (known) {
               result3 = G__toXvalue(result3, store_var_typeB);
               if (G__initval_eval) {
                  G__dynconst = G__DYNCONST;
               }
            }
            G__var_typeB = 'p';
         }
#ifdef G__PTR2MEMFUNC
         if (!known && !result3.obj.i) {
            known = G__getpointer2memberfunc(item, &result3);
         }
#endif // G__PTR2MEMFUNC
         // undefined
         if (!known) {
            if (!strncmp(item, "__", 2)) {
               result3 = G__getreserved(item + 1, 0, 0);
               if (result3.type) {
                  known = 1;
               }
            }
            else {
               if (
#ifdef G__ROOT
                  (G__dispmsg < G__DISPROOTSTRICT) &&
#endif // G__ROOT
                  G__GetSpecialObject && (G__GetSpecialObject != G__getreserved)
               ) {
                  // -- Append $ to object and try to find it again.
                  if (!G__gettingspecial && (item[0] != '$')) {
                     //
                     char *sbuf;
                     int store_return = G__return;
                     int store_security_error = G__security_error;
                     // This fix should be verified very carefully.
                     if (G__asm_noverflow && G__no_exec_compile) {
                        G__abortbytecode();
                     }
                     sbuf = (char*) malloc(strlen(item) + 2);
                     if (!sbuf) {
                        G__genericerror("Internal error: malloc in G__getitem(),sbuf");
                        return G__null;
                     }
                     sprintf(sbuf, "$%s", item); // Okay, right size.
                     G__gettingspecial = 1;
                     G__var_type = store_var_typeB;
                     result3 = G__getitem(sbuf);
                     free((void*) sbuf);
                     G__gettingspecial = 0;
                     if (G__const_noerror) {
                        G__return = store_return;
                        G__security_error = store_security_error;
                     }
                     return result3;
                  }
               }
            }
            if (!known && !result3.obj.i) {
               result3 = G__null;
               if (!G__noerr_defined) {
                  if (!G__definemacro) {
                     G__warnundefined(item);
                     result3 = G__interactivereturn();
                  }
                  else {
                     /*G__genericerror("Limitation: This form of macro may not be expanded. Use +P or -p option");*/
                     return G__null;
                  }
               }
            }
         }
   }
   return result3;
}

//______________________________________________________________________________
extern "C"
long G__test(const char* expr)
{
   G__value result = G__getexpr(expr);
   if (result.type == 'u') {
      return G__iosrdstate(&result);
   }
   return G__convertT<bool>(&result);
}

//______________________________________________________________________________
extern "C"
long G__btest(int operator2, G__value lresult, G__value rresult)
{
   // --
   if (lresult.type == 'u' || rresult.type == 'u') {
      G__overloadopr(operator2, rresult, &lresult);
      return G__int(lresult);
   }
   else if (lresult.type == 'U' || rresult.type == 'U') {
      G__publicinheritance(&lresult, &rresult);
   }
#ifdef G__ASM
   if (G__asm_noverflow) {
      //
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x,%3d: CMP2 '%c'  %s:%n\n", G__asm_cp, G__asm_dt, operator2, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      G__asm_inst[G__asm_cp] = G__CMP2;
      G__asm_inst[G__asm_cp+1] = (long) operator2;
      G__inc_cp_asm(2, 0);
   }
#endif // G__ASM
   if (G__no_exec_compile || G__no_exec) {
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
extern "C"
int G__lasterror()
{
   // --
   return G__last_error;
}

//______________________________________________________________________________
extern "C"
void G__reset_lasterror()
{
   // --
   G__last_error = G__NOERROR;
}

//______________________________________________________________________________
extern "C"
G__value G__calc(const char* exprwithspace)
{
   // -- Grand entry for C/C++ expression evaluator.
   //
   // Note: This function is open to public as CINT API.
   //
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

   return(result);
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
