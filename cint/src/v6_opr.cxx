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
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#include "common.h"


#ifndef G__OLDIMPLEMENTATION999
/***********************************************************************
* G__getoperatorstring()
***********************************************************************/
static char* G__getoperatorstring(operator)
int operator;
{
  switch(operator) {
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
#endif


#ifndef G__OLDIMPLEMENTATION470
/***********************************************************************
* G__doubleassignbyref()
***********************************************************************/
void G__doubleassignbyref(defined,val)
G__value *defined;
double val;
{
  if(isupper(defined->type)) {
    *(long*)defined->ref = (long)val;
    defined->obj.i = (long)val;
    return;
  }

  switch(defined->type) {
  case 'd': /* double */
    *(double*)defined->ref = val;
    defined->obj.d = val;
    break;
  case 'f': /* float */
    *(float*)defined->ref = (float)val;
    defined->obj.d = val;
    break;
  case 'l': /* long */
    *(long*)defined->ref = (long)val;
    defined->obj.i = (long)val;
    break;
  case 'k': /* unsigned long */
    *(unsigned long*)defined->ref = (unsigned long)val;
    defined->obj.i = (unsigned long)val;
    break;
  case 'i': /* int */
    *(int*)defined->ref = (int)val;
    defined->obj.i = (int)val;
    break;
  case 'h': /* unsigned int */
    *(unsigned int*)defined->ref = (unsigned int)val;
    defined->obj.i = (unsigned int)val;
    break;
  case 's': /* short */
    *(short*)defined->ref = (short)val;
    defined->obj.i = (short)val;
    break;
  case 'r': /* unsigned short */
    *(unsigned short*)defined->ref = (unsigned short)val;
    defined->obj.i = (unsigned short)val;
    break;
  case 'c': /* char */
    *(char*)defined->ref = (char)val;
    defined->obj.i = (char)val;
    break;
  case 'b': /* unsigned char */
    *(unsigned char*)defined->ref = (unsigned char)val;
    defined->obj.i = (unsigned char)val;
    break;
#ifndef G__OLDIMPLEMENTATION2189
  case 'n': /* long long */
    *(G__int64*)defined->ref = (G__int64)val;
    defined->obj.ll = (G__int64)val;
    break;
  case 'm': /* unsigned long long */
    *(G__uint64*)defined->ref = (G__uint64)val;
    defined->obj.ull = (G__uint64)val;
    break;
  case 'q': /* unsigned G__int64 */
    *(long double*)defined->ref = (long double)val;
    defined->obj.ld = (long double)val;
    break;
#endif
#ifndef G__OLDIMPLEMENTATION1604
  case 'g': /* bool */
    *(unsigned char*)defined->ref = (unsigned char)(val?1:0);
    defined->obj.i = (int)val?1:0;
    break;
#endif
  default:
    G__genericerror("Invalid operation and assignment, G__doubleassignbyref");
    break;
  }
}

/***********************************************************************
* G__intassignbyref()
***********************************************************************/
void G__intassignbyref(defined,val)
G__value *defined;
long val;
{
  if(isupper(defined->type)) {
    if(defined->ref) *(long*)defined->ref = (long)val;
    defined->obj.i = (long)val;
    return;
  }

  switch(defined->type) {
  case 'i': /* int */
    if(defined->ref) *(int*)defined->ref = (int)val;
    defined->obj.i = (int)val;
    break;
  case 'c': /* char */
    if(defined->ref) *(char*)defined->ref = (char)val;
    defined->obj.i = (char)val;
    break;
  case 'l': /* long */
    if(defined->ref) *(long*)defined->ref = (long)val;
    defined->obj.i = (long)val;
    break;
  case 's': /* short */
    if(defined->ref) *(short*)defined->ref = (short)val;
    defined->obj.i = (short)val;
    break;
  case 'k': /* unsigned long */
    if(defined->ref) *(unsigned long*)defined->ref = (unsigned long)val;
    defined->obj.i = (unsigned long)val;
    break;
  case 'h': /* unsigned int */
    if(defined->ref) *(unsigned int*)defined->ref = (unsigned int)val;
    defined->obj.i = (unsigned int)val;
    break;
  case 'r': /* unsigned short */
    if(defined->ref) *(unsigned short*)defined->ref = (unsigned short)val;
    defined->obj.i = (unsigned short)val;
    break;
  case 'b': /* unsigned char */
    if(defined->ref) *(unsigned char*)defined->ref = (unsigned char)val;
    defined->obj.i = (unsigned char)val;
    break;
#ifndef G__OLDIMPLEMENTATION2189
  case 'n': /* long long */
    if(defined->ref) *(G__int64*)defined->ref = (G__int64)val;
    defined->obj.ll = (G__int64)val;
    break;
  case 'm': /* long long */
    if(defined->ref) *(G__uint64*)defined->ref = (G__uint64)val;
    defined->obj.ull = (G__uint64)val;
    break;
  case 'q': /* long double */
    if(defined->ref) *(long double*)defined->ref = (long double)val;
    defined->obj.ld = (long double)val;
    break;
#endif
#ifndef G__OLDIMPLEMENTATION1604
  case 'g': /* bool */
    if(defined->ref) *(unsigned char*)defined->ref = (unsigned char)(val?1:0);
    defined->obj.i = (int)val?1:0;
    break;
#endif
  case 'd': /* double */
    if(defined->ref) *(double*)defined->ref = (double)val;
    defined->obj.d = (double)val;
    break;
  case 'f': /* float */
    if(defined->ref) *(float*)defined->ref = (float)val;
    defined->obj.d = (float)val;
    break;
  default:
    G__genericerror("Invalid operation and assignment, G__intassignbyref");
    break;
  }
}
#endif


/***********************************************************************
* G__bstore() 
*
* Called by
*    G__getexpr()
*    G__getexpr()
*    G__getexpr()
*    G__getexpr()
*    G__getexpr()
*    G__getexpr()
*    G__getexpr()
*    G__getprod()
*    G__getprod()
*    G__getpower()
*    G__getpower()
*    G__exec_asm()
*    G__exec_asm()
*
***********************************************************************/

void G__bstore(operator,expressionin,defined)
int operator;
G__value expressionin;
G__value *defined;
{
  int ig2;
  long lresult;
  double fdefined,fexpression;

  /*********************************************************
   * for overloading of operator
   *********************************************************/
  /****************************************************************
   * C++ 
   * If one of the parameter is struct(class) type, call user 
   * defined operator function
   * Assignment operators (=,+=,-=) do not work in this way.
   ****************************************************************/
  if(defined->type=='u'||expressionin.type=='u') {
    G__overloadopr(operator,expressionin,defined);
    return;
  }
  else {
#ifdef G__ASM
    if(G__asm_noverflow) {
      if(defined->type=='\0') {
	/****************************
	 * OP1 instruction
	 ****************************/
	switch(operator) {
#ifndef G__OLDIMPLEMENTATION654
	case '~':
#endif
	case '!':
	case '-':
	case G__OPR_POSTFIXINC:
	case G__OPR_POSTFIXDEC:
	case G__OPR_PREFIXINC:
	case G__OPR_PREFIXDEC:
#ifdef G__ASM_DBG
	  if(G__asm_dbg) {
	    if(isprint(operator)) 
	      G__fprinterr(G__serr,"%3x: OP1  '%c' %d\n"
		      ,G__asm_cp,operator,operator);
	    else
	      G__fprinterr(G__serr,"%3x: OP1  %d\n"
		      ,G__asm_cp,operator);
	  }
#endif
	  G__asm_inst[G__asm_cp]=G__OP1;
	  G__asm_inst[G__asm_cp+1]=G__op1_operator_detail(operator
							  ,&expressionin);
	  G__inc_cp_asm(2,0);
	  break;
	}
      }
      else {
	/****************************
	 * OP2 instruction
	 ****************************/
#ifdef G__ASM_DBG
	if(G__asm_dbg) {
	  if(isprint(operator)) 
	    G__fprinterr(G__serr,"%3x: OP2  '%c' %d\n"
		    ,G__asm_cp,operator,operator);
	  else
	    G__fprinterr(G__serr,"%3x: OP2  %d\n"
		    ,G__asm_cp,operator);
	}
#endif
	G__asm_inst[G__asm_cp]=G__OP2;
	G__asm_inst[G__asm_cp+1]=G__op2_operator_detail(operator
							,defined
							,&expressionin);
	G__inc_cp_asm(2,0);
      }
    }
#endif
    if(G__no_exec_compile||G__no_exec) { /* avoid Alpha crash */
      if(G__isdouble(expressionin)) expressionin.obj.d=0.0;
      else                          expressionin.obj.i=0;
      if(G__isdouble(*defined)) defined->obj.d=0.0;
      else                      defined->obj.i=0;
    }
  }
  
  /****************************************************************
   * double operator double
   * double operator int
   * int    operator double
   ****************************************************************/
  if((G__isdouble(expressionin))||(G__isdouble(*defined))) {
    fexpression=G__double(expressionin);
    fdefined=G__double(*defined);
#ifndef G__OLDIMPLEMENTATION1132
    defined->typenum = -1;
#endif
    switch(operator) {
    case '\0':
      defined->ref=expressionin.ref;
      G__letdouble(defined,'d',fdefined+fexpression);
      break;
    case '+': /* add */
      G__letdouble(defined,'d',fdefined+fexpression);
      defined->ref=0;
      break;
    case '-': /* subtract */
      G__letdouble(defined,'d',fdefined-fexpression);
      defined->ref=0;
      break;
    case '*': /* multiply */
      if(defined->type==G__null.type) fdefined=1.0;
      G__letdouble(defined,'d',fdefined*fexpression);
      defined->ref=0;
      break;
    case '/': /* divide */
      if(defined->type==G__null.type) fdefined=1.0;
      if(fexpression==0.0) {
	if(G__no_exec_compile) G__letdouble(defined,'d',0.0);
	else G__genericerror("Error: operator '/' divided by zero");
	return;
      }
      G__letdouble(defined,'d',fdefined/fexpression);
      defined->ref=0;
      break;
#ifdef G__NONANSIOPR
    case '%': /* modulus */
      if(fexpression==0.0) {
	if(G__no_exec_compile) G__letdouble(defined,'d',0.0);
	else G__genericerror("Error: operator '%%' divided by zero");
	return;
      }
      G__letint(defined,'i',(long)fdefined%(long)fexpression);
      defined->ref=0;
      break;
#endif /* G__NONANSIOPR */
    case '&': /* binary and */ 
      /* Don't know why but this one has a problem if deleted */
      if(defined->type==G__null.type) {
	G__letint(defined,'i',(long)fexpression);
      }
      else {
	G__letint(defined,'i',(long)fdefined&(long)fexpression);
	defined->ref=0;
      }
      break;
#ifdef G__NONANSIOPR
    case '|': /* binariy or */
      G__letint(defined,'i', (long)fdefined|(long)fexpression);
      defined->ref=0;
      break;
    case '^': /* binary exclusive or */
      G__letint(defined,'i', (long)fdefined^(long)fexpression);
      defined->ref=0;
      break;
    case '~': /* binary inverse */
      G__letint(defined,'i', ~(long)fexpression);
      defined->ref=0;
      break;
#endif /* G__NONANSIOPR */
    case 'A': /* logic and */
#ifndef G__OLDIMPLEMENTATION1674
      /* printf("\n!!! %g && %g\n"); */
      G__letint(defined,'i', 0.0!=fdefined&&0.0!=fexpression);
#else
      G__letint(defined,'i', (long)fdefined&&(long)fexpression);
#endif
      defined->ref=0;
      break;
    case 'O': /* logic or */
#ifndef G__OLDIMPLEMENTATION1674
      G__letint(defined,'i', 0.0!=fdefined||0.0!=fexpression);
#else
      G__letint(defined,'i', (long)fdefined||(long)fexpression);
#endif
      defined->ref=0;
      break;
    case '>':
      if(defined->type==G__null.type) {
#ifndef G__OLDIMPLEMENTATION2230
	G__letdouble(defined,'i',0>fexpression);
#else
	G__letdouble(defined,'d',fexpression);
#endif
      }
      else
	G__letint(defined,'i',fdefined>fexpression);
      defined->ref=0;
      break;
    case '<':
      if(defined->type==G__null.type) {
#ifndef G__OLDIMPLEMENTATION2230
	G__letdouble(defined,'i',0<fexpression);
#else
	G__letdouble(defined,'d',fexpression);
#endif
      }
      else
	G__letint(defined,'i',fdefined<fexpression);
      defined->ref=0;
      break;
#ifdef G__NONANSIOPR
    case 'R': /* right shift */
      G__letint(defined,'i', (long)fdefined>>(long)fexpression);
      defined->ref=0;
      break;
    case 'L': /* left shift */
      G__letint(defined,'i', (long)fdefined<<(long)fexpression);
      defined->ref=0;
      break;
#endif /* G__NONANSIOPR */
    case '@': /* power */
#ifndef G__OLDIMPLEMENTATION1123
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"Warning: Power operator, Cint special extension");
	G__printlinenum();
      }
#endif
      if(fdefined>0.0) {
	/* G__letdouble(defined,'d' ,exp(fexpression*log(fdefined))); */
	G__letdouble(defined,'d' ,pow(fdefined,fexpression));
      }
      else if(fdefined==0.0) {
	if(fexpression==0.0) G__letdouble(defined,'d' ,1.0);
	else                 G__letdouble(defined,'d' ,0.0);
      }
      else if(/* fmod(fdefined,1.0)==0 && */ fmod(fexpression,1.0)==0 &&
	      fexpression>=0) {
	double fresult=1.0;
	for(ig2=0;ig2<fexpression;ig2++) fresult *= fdefined;
	G__letdouble(defined,'d',fresult);
	defined->ref=0;
      }
      else {
	if(G__no_exec_compile) G__letdouble(defined,'d',0.0);
	else G__genericerror("Error: operator '@' or '**' negative operand");
	return;
      }
      defined->ref=0;
      break;
#ifdef G__NONANSIOPR
    case '!': 
      if(fexpression==0) G__letdouble(defined,'i',1);
      else               G__letdouble(defined,'i',0);
      defined->ref=0;
      break;
#endif /* G__NONANSIOPR */
    case 'E': /* == */
#ifndef G__OLDIMPLEMENTATION2230
      if(defined->type==G__null.type) 
	G__letdouble(defined,'i',0==fexpression); 
      else
        G__letint(defined,'i',fdefined==fexpression);
#else
      G__letint(defined,'i',fdefined==fexpression);
#endif
      defined->ref=0;
      break;
    case 'N': /* != */
#ifndef G__OLDIMPLEMENTATION2230
      if(defined->type==G__null.type) 
	G__letdouble(defined,'i',0!=fexpression); 
      else
        G__letint(defined,'i',fdefined!=fexpression);
#else
      G__letint(defined,'i',fdefined!=fexpression);
#endif
      defined->ref=0;
      break;
    case 'G': /* >= */
#ifndef G__OLDIMPLEMENTATION2230
      if(defined->type==G__null.type) 
	G__letdouble(defined,'i',0>=fexpression); 
      else
        G__letint(defined,'i',fdefined>=fexpression);
#else
      G__letint(defined,'i',fdefined>=fexpression);
#endif
      defined->ref=0;
      break;
    case 'l': /* <= */
#ifndef G__OLDIMPLEMENTATION2230
      if(defined->type==G__null.type) 
	G__letdouble(defined,'i',0<=fexpression); 
      else
        G__letint(defined,'i',fdefined<=fexpression);
#else
      G__letint(defined,'i',fdefined<=fexpression);
#endif
      defined->ref=0;
      break;
#ifndef G__OLDIMPLEMENTATION470
    case G__OPR_ADDASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__doubleassignbyref(defined,fdefined+fexpression);
      break;
    case G__OPR_SUBASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__doubleassignbyref(defined,fdefined-fexpression);
      break;
    case G__OPR_MODASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__doubleassignbyref(defined
			     ,(double)((long)fdefined%(long)fexpression));
      break;
    case G__OPR_MULASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__doubleassignbyref(defined,fdefined*fexpression);
      break;
    case G__OPR_DIVASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__doubleassignbyref(defined,fdefined/fexpression);
      break;
    case G__OPR_ANDASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__doubleassignbyref(defined
			     ,(double)((long)fdefined&&(long)fexpression));
      break;
    case G__OPR_ORASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__doubleassignbyref(defined
			     ,(double)((long)fdefined||(long)fexpression));
      break;
#endif
#ifndef G__OLDIMPLEMENTATION471
    case G__OPR_POSTFIXINC:
      if(!G__no_exec_compile&&expressionin.ref) {
	*defined = expressionin;
	G__doubleassignbyref(&expressionin,fexpression+1);
#ifndef G__OLDIMPLEMENTATION1342
	defined->ref = 0;
#endif
      }
      break;
    case G__OPR_POSTFIXDEC:
      if(!G__no_exec_compile&&expressionin.ref) {
	*defined = expressionin;
	G__doubleassignbyref(&expressionin,fexpression-1);
#ifndef G__OLDIMPLEMENTATION1342
	defined->ref = 0;
#endif
      }
      break;
    case G__OPR_PREFIXINC:
      if(!G__no_exec_compile&&expressionin.ref) {
	G__doubleassignbyref(&expressionin,fexpression+1);
	*defined = expressionin;
      }
      break;
    case G__OPR_PREFIXDEC:
      if(!G__no_exec_compile&&expressionin.ref) {
	G__doubleassignbyref(&expressionin,fexpression-1);
	*defined = expressionin;
      }
      break;
#endif
    default:
#ifndef G__OLDIMPLEMENTATION999
	G__fprinterr(G__serr,"Error: %s ",G__getoperatorstring(operator));
#else
	G__fprinterr(G__serr,"Error: %c ",operator);
#endif
	G__genericerror("Illegal operator for real number");
	break;
    }
  }
  
  /****************************************************************
   * pointer operator pointer
   * pointer operator int
   * int     operator pointer
   ****************************************************************/
  else if(isupper(defined->type)||isupper(expressionin.type)) {
    
    G__CHECK(G__SECURE_POINTER_CALC,'+'==operator||'-'==operator,return);
    
    if(isupper(defined->type)) {
      
      /*
       *  pointer - pointer , integer [==] pointer
       */
      if(isupper(expressionin.type)) {
	switch(operator) {
	case '\0': /* add */
	  defined->ref=expressionin.ref;
	  defined->obj.i = defined->obj.i+expressionin.obj.i;
	  defined->obj.reftype.reftype = expressionin.obj.reftype.reftype;
	  break;
	case '-': /* subtract */
	  defined->obj.i
	    =(defined->obj.i-expressionin.obj.i)/G__sizeof(defined);
	  defined->type='i';
	  defined->tagnum = -1;
	  defined->typenum = -1;
	  defined->ref=0;
	  break;
	case 'E': /* == */
#ifndef G__OLDIMPLEMENTATION697
	  if('U'==defined->type && 'U'==expressionin.type)
	    G__publicinheritance(defined,&expressionin);
#endif
	  G__letint(defined,'i',defined->obj.i==expressionin.obj.i);
	  defined->ref=0;
	  break;
	case 'N': /* != */
#ifndef G__OLDIMPLEMENTATION697
	  if('U'==defined->type && 'U'==expressionin.type)
	    G__publicinheritance(defined,&expressionin);
#endif
	  G__letint(defined,'i',defined->obj.i!=expressionin.obj.i);
	  defined->ref=0;
	  break;
	case 'G': /* >= */
	  G__letint(defined,'i',defined->obj.i>=expressionin.obj.i);
	  defined->ref=0;
	  break;
	case 'l': /* <= */
	  G__letint(defined,'i',defined->obj.i<=expressionin.obj.i);
	  defined->ref=0;
	  break;
#ifndef G__OLDIMPLEMENTATION1340
	case '>': /* > */
	  G__letint(defined,'i',defined->obj.i>expressionin.obj.i);
	  defined->ref=0;
	  break;
	case '<': /* < */
	  G__letint(defined,'i',defined->obj.i<expressionin.obj.i);
	  defined->ref=0;
	  break;
#endif
	case 'A': /* logical and */
	  G__letint(defined,'i',defined->obj.i&&expressionin.obj.i);
	  defined->ref=0;
      	  break;
	case 'O': /* logical or */
      	  G__letint(defined,'i',defined->obj.i||expressionin.obj.i);
	  defined->ref=0;
      	  break;
#ifndef G__OLDIMPLEMENTATION470
	case G__OPR_SUBASSIGN:
	  if(!G__no_exec_compile&&defined->ref) 
	    G__intassignbyref(defined
			      ,(defined->obj.i-expressionin.obj.i)
			       /G__sizeof(defined));
	  break;
#endif
	default:
	  if(G__ASM_FUNC_NOP==G__asm_wholefunction) {
	    G__fprinterr(G__serr,"Error: %s ",G__getoperatorstring(operator));
	  }
	  G__genericerror("Illegal operator for pointer 1");
	  break;
	}
      }
      /*
       *  pointer [+-==] integer , 
       */
      else {
	switch(operator) {
	case '\0': /* no op */
	  defined->ref=expressionin.ref;
#ifndef G__OLDIMPLEMENTATION456
	  defined->obj.reftype.reftype = expressionin.obj.reftype.reftype;
#endif
	case '+': /* add */
	  defined->obj.i=defined->obj.i+expressionin.obj.i*G__sizeof(defined);
	  defined->ref=0;
	  break;
	case '-': /* subtract */
	  defined->obj.i=defined->obj.i-expressionin.obj.i*G__sizeof(defined);
	  defined->ref=0;
	  break;
	case '!': 
	  G__letint(defined,'i',!expressionin.obj.i);
	  defined->ref=0;
	  break;
	case 'E': /* == */
	  G__letint(defined,'i',defined->obj.i==expressionin.obj.i);
	  defined->ref=0;
	  break;
	case 'N': /* != */
	  G__letint(defined,'i',defined->obj.i!=expressionin.obj.i);
	  defined->ref=0;
	  break;
	case 'G': /* >= */
	  G__letint(defined,'i',defined->obj.i>=expressionin.obj.i);
	  defined->ref=0;
	  break;
	case 'l': /* <= */
	  G__letint(defined,'i',defined->obj.i<=expressionin.obj.i);
	  defined->ref=0;
	  break;
#ifndef G__OLDIMPLEMENTATION1340
	case '>': /* > */
	  G__letint(defined,'i',defined->obj.i>expressionin.obj.i);
	  defined->ref=0;
	  break;
	case '<': /* < */
	  G__letint(defined,'i',defined->obj.i<expressionin.obj.i);
	  defined->ref=0;
	  break;
#endif
	case 'A': /* logical and */
	  G__letint(defined,'i',defined->obj.i&&expressionin.obj.i);
	  defined->ref=0;
      	  break;
	case 'O': /* logical or */
      	  G__letint(defined,'i',defined->obj.i||expressionin.obj.i);
	  defined->ref=0;
      	  break;
#ifndef G__OLDIMPLEMENTATION470
	case G__OPR_ADDASSIGN:
	  if(!G__no_exec_compile&&defined->ref) 
	    G__intassignbyref(defined
		       ,defined->obj.i+expressionin.obj.i*G__sizeof(defined));
	  break;
	case G__OPR_SUBASSIGN:
	  if(!G__no_exec_compile&&defined->ref) 
	    G__intassignbyref(defined
	          ,defined->obj.i-expressionin.obj.i*G__sizeof(defined));
	  break;
#endif
	default:
#ifndef G__OLDIMPLEMENTATION999
	  G__fprinterr(G__serr,"Error: %s ",G__getoperatorstring(operator));
#else
	  G__fprinterr(G__serr,"Error: %c ",operator);
#endif
	  G__genericerror("Illegal operator for pointer 2");
	  break;
	}
      }
    }
    
    /*
     *  integer [+-] pointer 
     */
    else {
      switch(operator) {
      case '\0': /* subtract */
	defined->ref=expressionin.ref;
	defined->type = expressionin.type;
	defined->tagnum = expressionin.tagnum;
      	defined->typenum = expressionin.typenum;
	defined->obj.i =defined->obj.i*G__sizeof(defined) +expressionin.obj.i;
	defined->obj.reftype.reftype = expressionin.obj.reftype.reftype;
	break;
      case '+': /* add */
	defined->obj.i =defined->obj.i*G__sizeof(defined) +expressionin.obj.i;
	defined->type = expressionin.type;
	defined->tagnum = expressionin.tagnum;
      	defined->typenum = expressionin.typenum;
#ifndef G__OLDIMPLEMENTATION456
	defined->obj.reftype.reftype = expressionin.obj.reftype.reftype;
#endif
	defined->ref=0;
	break;
      case '-': /* subtract */
	defined->obj.i =defined->obj.i*G__sizeof(defined) -expressionin.obj.i;
	defined->type = expressionin.type;
	defined->tagnum = expressionin.tagnum;
      	defined->typenum = expressionin.typenum;
#ifndef G__OLDIMPLEMENTATION456
	defined->obj.reftype.reftype = expressionin.obj.reftype.reftype;
#endif
	defined->ref=0;
	break;
      case '!': 
	G__letint(defined,'i',!expressionin.obj.i);
	defined->ref=0;
	break;
#ifndef G__OLDIMPLEMENTATION1339
      case 'E': /* == */
	G__letint(defined,'i',defined->obj.i==expressionin.obj.i);
	defined->ref=0;
	break;
      case 'N': /* != */
	G__letint(defined,'i',defined->obj.i!=expressionin.obj.i);
	defined->ref=0;
	break;
#endif
      case 'A': /* logical and */
	G__letint(defined,'i',defined->obj.i&&expressionin.obj.i);
	defined->ref=0;
	break;
      case 'O': /* logical or */
	G__letint(defined,'i',defined->obj.i||expressionin.obj.i);
	defined->ref=0;
	break;
#ifndef G__OLDIMPLEMENTATION470
      case G__OPR_ADDASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined
		    ,defined->obj.i*G__sizeof(defined) +expressionin.obj.i);
	break;
      case G__OPR_SUBASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined
		    ,defined->obj.i*G__sizeof(defined) -expressionin.obj.i);
	break;
#endif
#ifndef G__OLDIMPLEMENTATION471
      case G__OPR_POSTFIXINC:
	if(!G__no_exec_compile&&expressionin.ref) {
	  *defined = expressionin;
	  G__intassignbyref(&expressionin
			    ,expressionin.obj.i+G__sizeof(&expressionin));
#ifndef G__OLDIMPLEMENTATION1342
	  defined->ref = 0;
#endif
	}
	break;
      case G__OPR_POSTFIXDEC:
	if(!G__no_exec_compile&&expressionin.ref) {
	  *defined = expressionin;
	  G__intassignbyref(&expressionin
			    ,expressionin.obj.i-G__sizeof(&expressionin));
#ifndef G__OLDIMPLEMENTATION1342
	  defined->ref = 0;
#endif
	}
	break;
      case G__OPR_PREFIXINC:
	if(!G__no_exec_compile&&expressionin.ref) {
	  G__intassignbyref(&expressionin
			    ,expressionin.obj.i+G__sizeof(&expressionin));
	  *defined = expressionin;
	}
	break;
      case G__OPR_PREFIXDEC:
	if(!G__no_exec_compile&&expressionin.ref) {
	  G__intassignbyref(&expressionin
			    ,expressionin.obj.i-G__sizeof(&expressionin));
	  *defined = expressionin;
	}
	break;
#endif
      default:
#ifndef G__OLDIMPLEMENTATION999
	G__fprinterr(G__serr,"Error: %s ",G__getoperatorstring(operator));
#else
	G__fprinterr(G__serr,"Error: %c ",operator);
#endif
	G__genericerror("Illegal operator for pointer 3");
	break;
      }
    }
  }
  
#ifndef G__OLDIMPLEMENTATION2189
  /****************************************************************
   * long double operator long double
   * 
   ****************************************************************/
  else if('q'==defined->type || 'q'==expressionin.type) { 
    long double lddefined = G__Longdouble(*defined);
    long double ldexpression = G__Longdouble(expressionin);
    switch(operator) {
    case '\0':
      defined->ref=expressionin.ref;
      G__letLongdouble(defined,'q',lddefined+ldexpression);
      break;
    case '+': /* add */
      G__letLongdouble(defined,'q',lddefined+ldexpression);
      defined->ref=0;
      break;
    case '-': /* subtract */
      G__letLongdouble(defined,'q',lddefined-ldexpression);
      defined->ref=0;
      break;
    case '*': /* multiply */
      if(defined->type==G__null.type) lddefined=1;
      G__letLongdouble(defined,'q',lddefined*ldexpression);
      defined->ref=0;
      break;
    case '/': /* divide */
      if(defined->type==G__null.type) lddefined=1;
      if(ldexpression==0) {
	if(G__no_exec_compile) G__letdouble(defined,'i',0);
	else G__genericerror("Error: operator '/' divided by zero");
	return;
      }
      G__letLongdouble(defined,'q',lddefined/ldexpression);
      defined->ref=0;
      break;

    case '>':
      if(defined->type==G__null.type) {
#ifndef G__OLDIMPLEMENTATION2230
	G__letLongdouble(defined,'i',0>ldexpression);
#else
	G__letLongdouble(defined,'q',ldexpression);
#endif
      }
      else
	G__letint(defined,'i',lddefined>ldexpression);
      defined->ref=0;
      break;
    case '<':
      if(defined->type==G__null.type) {
#ifndef G__OLDIMPLEMENTATION2230
	G__letdouble(defined,'i',0<ldexpression);
#else
	G__letdouble(defined,'q',ldexpression);
#endif
      }
      else
	G__letint(defined,'i',lddefined<ldexpression);
      defined->ref=0;
      break;
    case '!': 
      if(ldexpression==0) G__letint(defined,'i',1);
      else                G__letint(defined,'i',0);
      defined->ref=0;
      break;
    case 'E': /* == */
#ifndef G__OLDIMPLEMENTATION2230
      if(defined->type==G__null.type) 
        G__letLongdouble(defined,'q',0); /* Expression should be false wben the var is not defined */
      else
        G__letint(defined,'i',lddefined==ldexpression);
#else
      G__letint(defined,'i',lddefined==ldexpression);
#endif
      defined->ref=0;
      break;
    case 'N': /* != */
#ifndef G__OLDIMPLEMENTATION2230
      if(defined->type==G__null.type) 
        G__letLongdouble(defined,'q',1); /* Expression should be true wben the var is not defined */
      else
        G__letint(defined,'i',lddefined!=ldexpression);
#else
      G__letint(defined,'i',lddefined!=ldexpression);
#endif
      defined->ref=0;
      break;
    case 'G': /* >= */
#ifndef G__OLDIMPLEMENTATION2230
      if(defined->type==G__null.type) 
        G__letLongdouble(defined,'q',0); /* Expression should be false wben the var is not defined */
      else
        G__letint(defined,'i',lddefined>=ldexpression);
#else
      G__letint(defined,'i',lddefined>=ldexpression);
#endif
      defined->ref=0;
      break;
    case 'l': /* <= */
#ifndef G__OLDIMPLEMENTATION2230
      if(defined->type==G__null.type) 
        G__letLongdouble(defined,'q',0); /* Expression should be false wben the var is not defined */
      else
        G__letint(defined,'i',lddefined<=ldexpression);
#else
      G__letint(defined,'i',lddefined<=ldexpression);
#endif
      defined->ref=0;
      break;

    case G__OPR_ADDASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__doubleassignbyref(defined,lddefined+ldexpression);
      break;
    case G__OPR_SUBASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__doubleassignbyref(defined,lddefined-ldexpression);
      break;
    case G__OPR_MODASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__doubleassignbyref(defined
			     ,(double)((long)lddefined%(long)ldexpression));
      break;
    case G__OPR_MULASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__doubleassignbyref(defined,lddefined*ldexpression);
      break;
    case G__OPR_DIVASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__doubleassignbyref(defined,lddefined/ldexpression);
      break;
    case G__OPR_ANDASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__doubleassignbyref(defined
			     ,(double)((long)lddefined&&(long)ldexpression));
      break;
    case G__OPR_ORASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__doubleassignbyref(defined
			     ,(double)((long)lddefined||(long)ldexpression));
      break;
    }

  }
  /****************************************************************
   * long operator long
   * 
   ****************************************************************/
  else if('n'==defined->type     || 'm'==defined->type    ||
	  'n'==expressionin.type || 'm'==expressionin.type) {
    int unsignedresult=0;
    if('m'==defined->type || 'm'==expressionin.type) unsignedresult = -1;
    if(unsignedresult) {
      G__uint64 ulldefined = G__ULonglong(*defined);
      G__uint64 ullexpression = G__ULonglong(expressionin);
      switch(operator) {
      case '\0':
	defined->ref=expressionin.ref;
	G__letULonglong(defined,'m',ulldefined+ullexpression);
	break;
      case '+': /* add */
	G__letULonglong(defined,'m',ulldefined+ullexpression);
	defined->ref=0;
	break;
      case '-': /* subtract */
	G__letULonglong(defined,'m',ulldefined-ullexpression);
	defined->ref=0;
	break;
      case '*': /* multiply */
	if(defined->type==G__null.type) ulldefined=1;
	G__letULonglong(defined,'m',ulldefined*ullexpression);
	defined->ref=0;
	break;
      case '/': /* divide */
	if(defined->type==G__null.type) ulldefined=1;
	if(ullexpression==0) {
	  if(G__no_exec_compile) G__letdouble(defined,'i',0);
	  else G__genericerror("Error: operator '/' divided by zero");
	  return;
	}
	G__letULonglong(defined,'m',ulldefined/ullexpression);
	defined->ref=0;
	break;
      case '%': /* modulus */
	if(ullexpression==0) {
	  if(G__no_exec_compile) G__letdouble(defined,'i',0);
	  else G__genericerror("Error: operator '%%' divided by zero");
	  return;
	}
	G__letULonglong(defined,'m',ulldefined%ullexpression);
	defined->ref=0;
	break;
      case '&': /* binary and */
	if(defined->type==G__null.type) {
	  G__letULonglong(defined,'m',ullexpression);
	}
	else {
	  G__letULonglong(defined,'m',ulldefined&ullexpression);
	}
	defined->ref=0;
	break;
      case '|': /* binary or */
	G__letULonglong(defined,'m',ulldefined|ullexpression);
	defined->ref=0;
	break;
      case '^': /* binary exclusive or */
	G__letULonglong(defined,'m',ulldefined^ullexpression);
	defined->ref=0;
	break;
      case '~': /* binary inverse */
	G__letULonglong(defined,'m',~ullexpression);
	defined->ref=0;
	break;
      case 'A': /* logical and */
	G__letULonglong(defined,'m',ulldefined&&ullexpression);
	defined->ref=0;
	break;
      case 'O': /* logical or */
	G__letULonglong(defined,'m',ulldefined||ullexpression);
	defined->ref=0;
	break;
      case '>':
	if(defined->type==G__null.type) {
#ifndef G__OLDIMPLEMENTATION2230
	  G__letULonglong(defined,'m',0); 
#else
	  G__letULonglong(defined,'m',ullexpression);
#endif
	}
	else
	  G__letint(defined,'i',ulldefined>ullexpression);
	defined->ref=0;
	break;
      case '<':
	if(defined->type==G__null.type) {
#ifndef G__OLDIMPLEMENTATION2230
	  G__letULonglong(defined,'m',0);
#else
	  G__letULonglong(defined,'m',ullexpression);
#endif
	}
	else
	  G__letint(defined,'i',ulldefined<ullexpression);
	defined->ref=0;
	break;
      case 'R': /* right shift */
	switch(defined->type) {
	case 'b':
	case 'r':
	case 'h':
	case 'k':
	  {
	    G__letULonglong(defined,'m',ulldefined>>ullexpression);
	  }
	  break;
	default: 
	  G__letULonglong(defined,'m',ulldefined>>ullexpression);
	  break;
	}
	defined->ref=0;
	break;
      case 'L': /* left shift */
	G__letULonglong(defined,'m',ulldefined<<ullexpression);
	defined->ref=0;
	break;
      case '!': 
	G__letULonglong(defined,'m',!ullexpression);
	defined->ref=0;
	break;
      case 'E': /* == */
#ifndef G__OLDIMPLEMENTATION2230
	if(defined->type==G__null.type) 
	  G__letULonglong(defined,'m',0); /* Expression should be false wben the var is not defined */
	else
	  G__letint(defined,'i',ulldefined==ullexpression);
#else
	G__letint(defined,'i',ulldefined==ullexpression);
#endif
	defined->ref=0;
	break;
      case 'N': /* != */
#ifndef G__OLDIMPLEMENTATION2230
	if(defined->type==G__null.type) 
	  G__letULonglong(defined,'m',1); /* Expression should be true wben the var is not defined */
	else
          G__letint(defined,'i',ulldefined!=ullexpression);
#else
	G__letint(defined,'i',ulldefined!=ullexpression);
#endif
	defined->ref=0;
	break;
      case 'G': /* >= */
#ifndef G__OLDIMPLEMENTATION2230
	if(defined->type==G__null.type) 
	  G__letULonglong(defined,'m',0); /* Expression should be false wben the var is not defined */
	else
	  G__letint(defined,'i',ulldefined>=ullexpression);
#else
	G__letint(defined,'i',ulldefined>=ullexpression);
#endif
	defined->ref=0;
	break;
      case 'l': /* <= */
#ifndef G__OLDIMPLEMENTATION2230
	if(defined->type==G__null.type) 
	  G__letULonglong(defined,'m',0); /* Expression should be false wben the var is not defined */
	else
	  G__letint(defined,'i',ulldefined<=ullexpression);
#else
	G__letint(defined,'i',ulldefined<=ullexpression);
#endif
	defined->ref=0;
	break;
#ifndef G__OLDIMPLEMENTATION470
      case G__OPR_ADDASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,ulldefined+ullexpression);
	break;
      case G__OPR_SUBASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,ulldefined-ullexpression);
	break;
      case G__OPR_MODASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,ulldefined%ullexpression);
	break;
      case G__OPR_MULASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,ulldefined*ullexpression);
	break;
      case G__OPR_DIVASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,ulldefined/ullexpression);
	break;
      case G__OPR_RSFTASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,ulldefined>>ullexpression);
	break;
      case G__OPR_LSFTASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,ulldefined<<ullexpression);
	break;
      case G__OPR_BANDASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,ulldefined&ullexpression);
	break;
      case G__OPR_BORASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,ulldefined|ullexpression);
	break;
      case G__OPR_EXORASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,ulldefined^ullexpression);
	break;
      case G__OPR_ANDASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,ulldefined&&ullexpression);
	break;
      case G__OPR_ORASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,ulldefined||ullexpression);
	break;
#endif
#ifndef G__OLDIMPLEMENTATION471
      case G__OPR_POSTFIXINC:
	if(!G__no_exec_compile&&expressionin.ref) {
	  *defined = expressionin;
	  G__intassignbyref(&expressionin,ullexpression+1);
#ifndef G__OLDIMPLEMENTATION1342
	  defined->ref = 0;
#endif
	}
	break;
      case G__OPR_POSTFIXDEC:
	if(!G__no_exec_compile&&expressionin.ref) {
	  *defined = expressionin;
	  G__intassignbyref(&expressionin,ullexpression-1);
#ifndef G__OLDIMPLEMENTATION1342
	  defined->ref = 0;
#endif
	}
	break;
      case G__OPR_PREFIXINC:
      if(!G__no_exec_compile&&expressionin.ref) {
	G__intassignbyref(&expressionin,ullexpression+1);
	*defined = expressionin;
      }
      break;
      case G__OPR_PREFIXDEC:
	if(!G__no_exec_compile&&expressionin.ref) {
	  G__intassignbyref(&expressionin,ullexpression-1);
	  *defined = expressionin;
	}
	break;
#endif
      default:
#ifndef G__OLDIMPLEMENTATION999
	G__fprinterr(G__serr,"Error: %s ",G__getoperatorstring(operator));
#else
	G__fprinterr(G__serr,"Error: %c ",operator);
#endif
	G__genericerror("Illegal operator for integer");
	break;
      }
    }
    else {
      G__int64 lldefined = G__Longlong(*defined);
      G__int64 llexpression = G__Longlong(expressionin);
      switch(operator) {
      case '\0':
	defined->ref=expressionin.ref;
	G__letLonglong(defined,'n',lldefined+llexpression);
	break;
      case '+': /* add */
	G__letLonglong(defined,'n',lldefined+llexpression);
	defined->ref=0;
	break;
      case '-': /* subtract */
	G__letLonglong(defined,'n',lldefined-llexpression);
	defined->ref=0;
	break;
      case '*': /* multiply */
	if(defined->type==G__null.type) lldefined=1;
	G__letLonglong(defined,'n',lldefined*llexpression);
	defined->ref=0;
	break;
      case '/': /* divide */
	if(defined->type==G__null.type) lldefined=1;
	if(llexpression==0) {
	  if(G__no_exec_compile) G__letdouble(defined,'i',0);
	  else G__genericerror("Error: operator '/' divided by zero");
	  return;
	}
	G__letLonglong(defined,'n',lldefined/llexpression);
	defined->ref=0;
	break;
      case '%': /* modulus */
	if(llexpression==0) {
	  if(G__no_exec_compile) G__letdouble(defined,'i',0);
	  else G__genericerror("Error: operator '%%' divided by zero");
	  return;
	}
	G__letLonglong(defined,'n',lldefined%llexpression);
	defined->ref=0;
	break;
      case '&': /* binary and */
	if(defined->type==G__null.type) {
	  G__letLonglong(defined,'n',llexpression);
	}
	else {
	  G__letint(defined,'i',lldefined&llexpression);
	}
	defined->ref=0;
	break;
      case '|': /* binary or */
	G__letint(defined,'i',lldefined|llexpression);
	defined->ref=0;
	break;
      case '^': /* binary exclusive or */
	G__letULonglong(defined,'n',lldefined^llexpression);
	defined->ref=0;
	break;
      case '~': /* binary inverse */
	G__letULonglong(defined,'n',~llexpression);
	defined->ref=0;
	break;
      case 'A': /* logical and */
	G__letint(defined,'i',lldefined&&llexpression);
	defined->ref=0;
	break;
      case 'O': /* logical or */
	G__letint(defined,'i',lldefined||llexpression);
	defined->ref=0;
	break;
      case '>':
	if(defined->type==G__null.type) {
#ifndef G__OLDIMPLEMENTATION2230
	  G__letLonglong(defined,'n',0);
#else
	  G__letLonglong(defined,'n',llexpression);
#endif
	}
	else
	  G__letint(defined,'i',lldefined>llexpression);
	defined->ref=0;
	break;
      case '<':
	if(defined->type==G__null.type) {
#ifndef G__OLDIMPLEMENTATION2230
	  G__letLonglong(defined,'n',0);
#else
	  G__letLonglong(defined,'n',llexpression);
#endif
	}
	else
	  G__letint(defined,'i',lldefined<llexpression);
	defined->ref=0;
	break;
      case 'R': /* right shift */
#ifndef G__OLDIMPLEMENTATION977
	switch(defined->type) {
	case 'b':
	case 'r':
	case 'h':
	case 'k':
	  {
	    G__letLonglong(defined,'n',lldefined>>llexpression);
	  }
	  break;
	default: 
	  G__letLonglong(defined,'n',lldefined>>llexpression);
	  break;
	}
#else
	G__letLonglong(defined,'n',lldefined>>llexpression);
#endif
	defined->ref=0;
	break;
      case 'L': /* left shift */
	G__letLonglong(defined,'n',lldefined<<llexpression);
	defined->ref=0;
	break;
      case '!': 
	G__letLonglong(defined,'n',!llexpression);
	defined->ref=0;
	break;
      case 'E': /* == */
#ifndef G__OLDIMPLEMENTATION2230
	if(defined->type==G__null.type) 
	  G__letLonglong(defined,'n',0); /* Expression should be false wben the var is not defined */
	else
	  G__letint(defined,'i',lldefined==llexpression);
#else
	G__letint(defined,'i',lldefined==llexpression);
#endif
	defined->ref=0;
	break;
      case 'N': /* != */
#ifndef G__OLDIMPLEMENTATION2230
	if(defined->type==G__null.type) 
	  G__letLonglong(defined,'n',1); /* Expression should be true wben the var is not defined */
	else
	  G__letint(defined,'i',lldefined!=llexpression);
#else
	G__letint(defined,'i',lldefined!=llexpression);
#endif
	defined->ref=0;
	break;
      case 'G': /* >= */
#ifndef G__OLDIMPLEMENTATION2230
	if(defined->type==G__null.type) 
	  G__letLonglong(defined,'n',0); /* Expression should be false wben the var is not defined */
	else
	  G__letint(defined,'i',lldefined>=llexpression);
#else
	G__letint(defined,'i',lldefined>=llexpression);
#endif
	defined->ref=0;
	break;
      case 'l': /* <= */
#ifndef G__OLDIMPLEMENTATION2230
	if(defined->type==G__null.type) 
	  G__letLonglong(defined,'n',0); /* Expression should be false wben the var is not defined */
	else
	  G__letint(defined,'i',lldefined<=llexpression);
#else
	G__letint(defined,'i',lldefined<=llexpression);
#endif
	defined->ref=0;
	break;
#ifndef G__OLDIMPLEMENTATION470
      case G__OPR_ADDASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,lldefined+llexpression);
	break;
      case G__OPR_SUBASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,lldefined-llexpression);
	break;
      case G__OPR_MODASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,lldefined%llexpression);
	break;
      case G__OPR_MULASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,lldefined*llexpression);
	break;
      case G__OPR_DIVASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,lldefined/llexpression);
	break;
      case G__OPR_RSFTASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,lldefined>>llexpression);
	break;
      case G__OPR_LSFTASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,lldefined<<llexpression);
	break;
      case G__OPR_BANDASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,lldefined&llexpression);
	break;
      case G__OPR_BORASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,lldefined|llexpression);
	break;
      case G__OPR_EXORASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,lldefined^llexpression);
	break;
      case G__OPR_ANDASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,lldefined&&llexpression);
	break;
      case G__OPR_ORASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,lldefined||llexpression);
	break;
#endif
#ifndef G__OLDIMPLEMENTATION471
      case G__OPR_POSTFIXINC:
	if(!G__no_exec_compile&&expressionin.ref) {
	  *defined = expressionin;
	  G__intassignbyref(&expressionin,llexpression+1);
#ifndef G__OLDIMPLEMENTATION1342
	  defined->ref = 0;
#endif
	}
	break;
      case G__OPR_POSTFIXDEC:
	if(!G__no_exec_compile&&expressionin.ref) {
	  *defined = expressionin;
	  G__intassignbyref(&expressionin,llexpression-1);
#ifndef G__OLDIMPLEMENTATION1342
	  defined->ref = 0;
#endif
	}
	break;
      case G__OPR_PREFIXINC:
	if(!G__no_exec_compile&&expressionin.ref) {
	  G__intassignbyref(&expressionin,llexpression+1);
	  *defined = expressionin;
	}
	break;
      case G__OPR_PREFIXDEC:
	if(!G__no_exec_compile&&expressionin.ref) {
	  G__intassignbyref(&expressionin,llexpression-1);
	  *defined = expressionin;
	}
	break;
#endif
      default:
#ifndef G__OLDIMPLEMENTATION999
	G__fprinterr(G__serr,"Error: %s ",G__getoperatorstring(operator));
#else
	G__fprinterr(G__serr,"Error: %c ",operator);
#endif
	G__genericerror("Illegal operator for integer");
	break;
      }
    }
  }
#endif
  
  /****************************************************************
   * int operator int
   * 
   ****************************************************************/
  else {
#ifndef G__OLDIMPLEMENTATION918
    int unsignedresult=0;
    switch(defined->type) {
    case 'h':
    case 'k':
      unsignedresult = -1;
      break;
    }
    switch(expressionin.type) {
    case 'h':
    case 'k':
      unsignedresult = -1;
      break;
    }
#endif
#ifndef G__OLDIMPLEMENTATION1491
    if(unsignedresult) {
      unsigned long udefined=(unsigned long)G__uint(*defined);
      unsigned long uexpression=(unsigned long)G__uint(expressionin);
      switch(operator) {
      case '\0':
	defined->ref=expressionin.ref;
	G__letint(defined,'h',udefined+uexpression);
	break;
      case '+': /* add */
	G__letint(defined,'h',udefined+uexpression);
	defined->ref=0;
	break;
      case '-': /* subtract */
	G__letint(defined,'h',udefined-uexpression);
	defined->ref=0;
	break;
      case '*': /* multiply */
	if(defined->type==G__null.type) udefined=1;
	G__letint(defined,'h',udefined*uexpression);
	defined->ref=0;
	break;
      case '/': /* divide */
	if(defined->type==G__null.type) udefined=1;
	if(uexpression==0) {
	  if(G__no_exec_compile) G__letdouble(defined,'i',0);
	  else G__genericerror("Error: operator '/' divided by zero");
	  return;
	}
	G__letint(defined,'h',udefined/uexpression);
	defined->ref=0;
	break;
      case '%': /* modulus */
	if(uexpression==0) {
	  if(G__no_exec_compile) G__letdouble(defined,'i',0);
	  else G__genericerror("Error: operator '%%' divided by zero");
	  return;
	}
	G__letint(defined,'h',udefined%uexpression);
	defined->ref=0;
	break;
      case '&': /* binary and */
	if(defined->type==G__null.type) {
	  G__letint(defined,'h',uexpression);
	}
	else {
	  G__letint(defined,'h',udefined&uexpression);
	}
	defined->ref=0;
	break;
      case '|': /* binary or */
	G__letint(defined,'h',udefined|uexpression);
	defined->ref=0;
	break;
      case '^': /* binary exclusive or */
	G__letint(defined,'h',udefined^uexpression);
	defined->ref=0;
	break;
      case '~': /* binary inverse */
	G__letint(defined,'h',~uexpression);
	defined->ref=0;
	break;
      case 'A': /* logical and */
	G__letint(defined,'h',udefined&&uexpression);
	defined->ref=0;
	break;
      case 'O': /* logical or */
	G__letint(defined,'h',udefined||uexpression);
	defined->ref=0;
	break;
      case '>':
	if(defined->type==G__null.type) {
#ifndef G__OLDIMPLEMENTATION2230
	  G__letint(defined,'h',0);
#else
	  G__letint(defined,'h',uexpression);
#endif
	}
	else
	  G__letint(defined,'h',udefined>uexpression);
	defined->ref=0;
	break;
      case '<':
	if(defined->type==G__null.type) {
#ifndef G__OLDIMPLEMENTATION2230
	  G__letint(defined,'h',0);
#else
	  G__letint(defined,'h',uexpression);
#endif
	}
	else
	  G__letint(defined,'h',udefined<uexpression);
	defined->ref=0;
	break;
      case 'R': /* right shift */
#ifndef G__OLDIMPLEMENTATION977
	switch(defined->type) {
	case 'b':
	case 'r':
	case 'h':
	case 'k':
	  {
	    unsigned long uudefined=udefined;
	    G__letint(defined,'k',uudefined>>uexpression);
	  }
	  break;
	default: 
	  G__letint(defined,'h',udefined>>uexpression);
	  break;
	}
#else
	G__letint(defined,'h',udefined>>uexpression);
#endif
	defined->ref=0;
	break;
      case 'L': /* left shift */
	G__letint(defined,'h',udefined<<uexpression);
	defined->ref=0;
	break;
      case '@': /* power */
#ifndef G__OLDIMPLEMENTATION1123
	if(G__asm_dbg) {
	  G__fprinterr(G__serr,"Warning: Power operator, Cint special extension");
	  G__printlinenum();
	}
#endif
#ifndef G__OLDIMPLEMENTATION966
	fdefined=1.0;
	for(ig2=1;ig2<=(int)uexpression;ig2++) fdefined *= udefined;
	if(fdefined>(double)LONG_MAX||fdefined<(double)LONG_MIN) {
	  G__genericerror("Error: integer overflow. Use 'double' for power operator");
	}
	lresult = (long)fdefined;
#else
	lresult=1;
	for(ig2=1;ig2<=(int)uexpression;ig2++) lresult *= udefined;
#endif
	G__letint(defined,'h',lresult);
	defined->ref=0;
	break;
      case '!': 
	G__letint(defined,'h',!uexpression);
	defined->ref=0;
	break;
      case 'E': /* == */
#ifndef G__OLDIMPLEMENTATION2230
	if(defined->type==G__null.type) 
	  G__letint(defined,'h',0); /* Expression should be false wben the var is not defined */
	else
	  G__letint(defined,'h',udefined==uexpression);
#else
	G__letint(defined,'h',udefined==uexpression);
#endif
	defined->ref=0;
	break;
      case 'N': /* != */
#ifndef G__OLDIMPLEMENTATION2230
	if(defined->type==G__null.type) 
	  G__letint(defined,'h',1); /* Expression should be true wben the var is not defined */
	else
	  G__letint(defined,'h',udefined!=uexpression);
#else
	G__letint(defined,'h',udefined!=uexpression);
#endif
	defined->ref=0;
	break;
      case 'G': /* >= */
#ifndef G__OLDIMPLEMENTATION2230
	if(defined->type==G__null.type) 
	  G__letint(defined,'h',0); /* Expression should be false wben the var is not defined */
	else
	  G__letint(defined,'h',udefined>=uexpression);
#else
	G__letint(defined,'h',udefined>=uexpression);
#endif
	defined->ref=0;
	break;
      case 'l': /* <= */
#ifndef G__OLDIMPLEMENTATION2230
	if(defined->type==G__null.type) 
	  G__letint(defined,'h',0); /* Expression should be false wben the var is not defined */
	else
	  G__letint(defined,'h',udefined<=uexpression);
#else
	G__letint(defined,'h',udefined<=uexpression);
#endif
	defined->ref=0;
	break;
#ifndef G__OLDIMPLEMENTATION470
      case G__OPR_ADDASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,udefined+uexpression);
	break;
      case G__OPR_SUBASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,udefined-uexpression);
	break;
      case G__OPR_MODASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,udefined%uexpression);
	break;
      case G__OPR_MULASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,udefined*uexpression);
	break;
      case G__OPR_DIVASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,udefined/uexpression);
	break;
      case G__OPR_RSFTASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,udefined>>uexpression);
	break;
      case G__OPR_LSFTASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,udefined<<uexpression);
	break;
      case G__OPR_BANDASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,udefined&uexpression);
	break;
      case G__OPR_BORASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,udefined|uexpression);
	break;
      case G__OPR_EXORASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,udefined^uexpression);
	break;
      case G__OPR_ANDASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,udefined&&uexpression);
	break;
      case G__OPR_ORASSIGN:
	if(!G__no_exec_compile&&defined->ref) 
	  G__intassignbyref(defined,udefined||uexpression);
	break;
#endif
#ifndef G__OLDIMPLEMENTATION471
      case G__OPR_POSTFIXINC:
	if(!G__no_exec_compile&&expressionin.ref) {
	  *defined = expressionin;
	  G__intassignbyref(&expressionin,uexpression+1);
#ifndef G__OLDIMPLEMENTATION1342
	  defined->ref = 0;
#endif
	}
	break;
      case G__OPR_POSTFIXDEC:
	if(!G__no_exec_compile&&expressionin.ref) {
	  *defined = expressionin;
	  G__intassignbyref(&expressionin,uexpression-1);
#ifndef G__OLDIMPLEMENTATION1342
	  defined->ref = 0;
#endif
	}
	break;
      case G__OPR_PREFIXINC:
      if(!G__no_exec_compile&&expressionin.ref) {
	G__intassignbyref(&expressionin,uexpression+1);
	*defined = expressionin;
      }
      break;
      case G__OPR_PREFIXDEC:
	if(!G__no_exec_compile&&expressionin.ref) {
	  G__intassignbyref(&expressionin,uexpression-1);
	  *defined = expressionin;
	}
	break;
#endif
      default:
#ifndef G__OLDIMPLEMENTATION999
	G__fprinterr(G__serr,"Error: %s ",G__getoperatorstring(operator));
#else
	G__fprinterr(G__serr,"Error: %c ",operator);
#endif
	G__genericerror("Illegal operator for integer");
	break;
      }
    }
    else {
#endif
    long ldefined=G__int(*defined);
    long lexpression=G__int(expressionin);
    switch(operator) {
    case '\0':
      defined->ref=expressionin.ref;
      G__letint(defined,'i',ldefined+lexpression);
      break;
    case '+': /* add */
      G__letint(defined,'i',ldefined+lexpression);
      defined->ref=0;
      break;
    case '-': /* subtract */
      G__letint(defined,'i',ldefined-lexpression);
      defined->ref=0;
      break;
    case '*': /* multiply */
      if(defined->type==G__null.type) ldefined=1;
      G__letint(defined,'i',ldefined*lexpression);
      defined->ref=0;
      break;
    case '/': /* divide */
      if(defined->type==G__null.type) ldefined=1;
      if(lexpression==0) {
	if(G__no_exec_compile) G__letdouble(defined,'i',0);
	else G__genericerror("Error: operator '/' divided by zero");
	return;
      }
      G__letint(defined,'i',ldefined/lexpression);
      defined->ref=0;
      break;
    case '%': /* modulus */
      if(lexpression==0) {
	if(G__no_exec_compile) G__letdouble(defined,'i',0);
	else G__genericerror("Error: operator '%%' divided by zero");
	return;
      }
      G__letint(defined,'i',ldefined%lexpression);
      defined->ref=0;
      break;
    case '&': /* binary and */
      if(defined->type==G__null.type) {
	G__letint(defined,'i',lexpression);
      }
      else {
	G__letint(defined,'i',ldefined&lexpression);
      }
      defined->ref=0;
      break;
    case '|': /* binary or */
      G__letint(defined,'i',ldefined|lexpression);
      defined->ref=0;
      break;
    case '^': /* binary exclusive or */
      G__letint(defined,'i',ldefined^lexpression);
      defined->ref=0;
      break;
    case '~': /* binary inverse */
      G__letint(defined,'i',~lexpression);
      defined->ref=0;
      break;
    case 'A': /* logical and */
      G__letint(defined,'i',ldefined&&lexpression);
      defined->ref=0;
      break;
    case 'O': /* logical or */
      G__letint(defined,'i',ldefined||lexpression);
      defined->ref=0;
      break;
    case '>':
      if(defined->type==G__null.type) {
#ifndef G__OLDIMPLEMENTATION2230
	G__letint(defined,'i',0);
#else
	G__letint(defined,'i',lexpression);
#endif
      }
      else
	G__letint(defined,'i',ldefined>lexpression);
      defined->ref=0;
      break;
    case '<':
      if(defined->type==G__null.type) {
#ifndef G__OLDIMPLEMENTATION2230
	G__letint(defined,'i',0);
#else
	G__letint(defined,'i',lexpression);
#endif
      }
      else
	G__letint(defined,'i',ldefined<lexpression);
      defined->ref=0;
      break;
    case 'R': /* right shift */
#if !defined(G__OLDIMPLEMENTATION2193)
        if(!G__prerun) {
	  unsigned long udefined=(unsigned long)G__uint(*defined);
	  unsigned long uexpression=(unsigned long)G__uint(expressionin);
	  G__letint(defined,'h',udefined>>uexpression);
          defined->obj.ulo = udefined>>uexpression;
	}
	else {
	  G__letint(defined,'i',ldefined>>lexpression);
	}
#elif !defined(G__OLDIMPLEMENTATION977)
      switch(defined->type) {
      case 'b':
      case 'r':
      case 'h':
      case 'k':
	{
	  unsigned long uldefined=ldefined;
	  G__letint(defined,'l',uldefined>>lexpression);
	}
	break;
      default: 
	G__letint(defined,'h',ldefined>>lexpression);
	break;
      }
#else
      G__letint(defined,'i',ldefined>>lexpression);
#endif
      defined->ref=0;
      break;
    case 'L': /* left shift */
#if !defined(G__OLDIMPLEMENTATION2193)
        if(!G__prerun) {
	  unsigned long udefined=(unsigned long)G__uint(*defined);
	  unsigned long uexpression=(unsigned long)G__uint(expressionin);
	  G__letint(defined,'h',udefined<<uexpression);
          defined->obj.ulo = udefined<<uexpression;
	}
	else {
	  G__letint(defined,'i',ldefined<<lexpression);
	}
#else
      G__letint(defined,'i',ldefined<<lexpression);
#endif
      defined->ref=0;
      break;
    case '@': /* power */
#ifndef G__OLDIMPLEMENTATION1123
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"Warning: Power operator, Cint special extension");
	G__printlinenum();
      }
#endif
#ifndef G__OLDIMPLEMENTATION966
      fdefined=1.0;
      for(ig2=1;ig2<=lexpression;ig2++) fdefined *= ldefined;
      if(fdefined>(double)LONG_MAX||fdefined<(double)LONG_MIN) {
        G__genericerror("Error: integer overflow. Use 'double' for power operator");
      }
      lresult = (long)fdefined;
#else
      lresult=1;
      for(ig2=1;ig2<=lexpression;ig2++) lresult *= ldefined;
#endif
      G__letint(defined,'i',lresult);
      defined->ref=0;
      break;
    case '!': 
      G__letint(defined,'i',!lexpression);
      defined->ref=0;
      break;
    case 'E': /* == */
#ifndef G__OLDIMPLEMENTATION2230
      if(defined->type==G__null.type)
        G__letint(defined,'i',0); /* Expression should be false wben the var is not defined */
      else
        G__letint(defined,'i',ldefined==lexpression);
#else
      G__letint(defined,'i',ldefined==lexpression);
#endif
      defined->ref=0;
      break;
    case 'N': /* != */
#ifndef G__OLDIMPLEMENTATION2230
      if(defined->type==G__null.type)
        G__letint(defined,'i',1); /* Expression should be true wben the var is not defined */
      else
        G__letint(defined,'i',ldefined!=lexpression);
#else
      G__letint(defined,'i',ldefined!=lexpression);
#endif
      defined->ref=0;
      break;
    case 'G': /* >= */
#ifndef G__OLDIMPLEMENTATION2230
      if(defined->type==G__null.type)
        G__letint(defined,'i',0); /* Expression should be false wben the var is not defined */
      else
        G__letint(defined,'i',ldefined>=lexpression);
#else
      G__letint(defined,'i',ldefined>=lexpression);
#endif
      defined->ref=0;
      break;
    case 'l': /* <= */
#ifndef G__OLDIMPLEMENTATION2230
      if(defined->type==G__null.type)
        G__letint(defined,'i',0); /* Expression should be false wben the var is not defined */
      else
        G__letint(defined,'i',ldefined<=lexpression);
#else
      G__letint(defined,'i',ldefined<=lexpression);
#endif
      defined->ref=0;
      break;
#ifndef G__OLDIMPLEMENTATION470
    case G__OPR_ADDASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__intassignbyref(defined,ldefined+lexpression);
      break;
    case G__OPR_SUBASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__intassignbyref(defined,ldefined-lexpression);
      break;
    case G__OPR_MODASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__intassignbyref(defined,ldefined%lexpression);
      break;
    case G__OPR_MULASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__intassignbyref(defined,ldefined*lexpression);
      break;
    case G__OPR_DIVASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__intassignbyref(defined,ldefined/lexpression);
      break;
    case G__OPR_RSFTASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__intassignbyref(defined,ldefined>>lexpression);
      break;
    case G__OPR_LSFTASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__intassignbyref(defined,ldefined<<lexpression);
      break;
    case G__OPR_BANDASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__intassignbyref(defined,ldefined&lexpression);
      break;
    case G__OPR_BORASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__intassignbyref(defined,ldefined|lexpression);
      break;
    case G__OPR_EXORASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__intassignbyref(defined,ldefined^lexpression);
      break;
    case G__OPR_ANDASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__intassignbyref(defined,ldefined&&lexpression);
      break;
    case G__OPR_ORASSIGN:
      if(!G__no_exec_compile&&defined->ref) 
	G__intassignbyref(defined,ldefined||lexpression);
      break;
#endif
#ifndef G__OLDIMPLEMENTATION471
    case G__OPR_POSTFIXINC:
      if(!G__no_exec_compile&&expressionin.ref) {
	*defined = expressionin;
	G__intassignbyref(&expressionin,lexpression+1);
#ifndef G__OLDIMPLEMENTATION1342
	defined->ref = 0;
#endif
      }
      break;
    case G__OPR_POSTFIXDEC:
      if(!G__no_exec_compile&&expressionin.ref) {
	*defined = expressionin;
	G__intassignbyref(&expressionin,lexpression-1);
#ifndef G__OLDIMPLEMENTATION1342
	defined->ref = 0;
#endif
      }
      break;
    case G__OPR_PREFIXINC:
      if(!G__no_exec_compile&&expressionin.ref) {
	G__intassignbyref(&expressionin,lexpression+1);
	*defined = expressionin;
      }
      break;
    case G__OPR_PREFIXDEC:
      if(!G__no_exec_compile&&expressionin.ref) {
	G__intassignbyref(&expressionin,lexpression-1);
	*defined = expressionin;
      }
      break;
#endif
    default:
#ifndef G__OLDIMPLEMENTATION999
	G__fprinterr(G__serr,"Error: %s ",G__getoperatorstring(operator));
#else
	G__fprinterr(G__serr,"Error: %c ",operator);
#endif
	G__genericerror("Illegal operator for integer");
	break;
    }
#ifndef G__OLDIMPLEMENTATION1491
    }
#else
#ifndef G__OLDIMPLEMENTATION918
    defined->type += unsignedresult;
#endif
#endif
  }
#ifndef G__OLDIMPLEMENTATION726
  if(G__no_exec_compile&&0==defined->type) *defined = expressionin;
#endif
}

/******************************************************************
* G__scopeoperator()
*
*  May need to modify this function to support multiple usage of
*  scope operator 'xxx::xxx::var'
******************************************************************/
int G__scopeoperator(name,phash,pstruct_offset,ptagnum)
char *name;  /* name is modified and this is intentional */
int *phash;
long *pstruct_offset;
int *ptagnum;
{
  char *pc,*scope,*member;
  int scopetagnum,offset,offset_sum;
  int i;
  char temp[G__MAXNAME*2];
#ifndef G__OLDIMPLEMENTATION741
  char *pparen;
#endif

#ifndef G__OLDIMPLEMENTATION1926
    re_try_after_std:
#endif

  /* search for pattern "::" */
#ifndef G__OLDIMPLEMENTATION671
  pc = G__find_first_scope_operator(name);
#else
  pc=strstr(name,"::");
#ifndef G__OLDIMPLEMENTATION622
  while(pc&&strstr(pc+2,"::")) pc = strstr(pc+2,"::");
#endif
#endif

  /* no scope operator, return */
#ifndef G__OLDIMPLEMENTATION741
   pparen = strchr(name,'(');
#endif
  if(NULL==pc || strncmp(name,"operator ",9)==0 
#ifndef G__OLDIMPLEMENTATION741
     || (pparen && pparen<pc)
#endif
     ) {
    G__fixedscope=0;
    return(G__NOSCOPEOPR);
  }
  
  G__fixedscope=1;

  /* if scope operator found at the beginning of the name, global scope */
  /* if scope operator found at the beginning of the name, global scope 
   * or fully qualified scope!
   */
  if(pc==name) {
    /* strip scope operator, set hash and return */
    strcpy(temp,name+2);
    strcpy(name,temp);
    G__hash(name,(*phash),i)
#ifndef G__OLDIMPLEMENTATION2179
    /* If we do no have anymore scope operator, we know the request of
       for the global name space */
    pc = G__find_first_scope_operator(name);
    if (pc==0) return(G__GLOBALSCOPE);
#else    
    return(G__GLOBALSCOPE);
#endif
  }

#ifndef G__STD_NAMESPACE /* ON667 */
  if (strncmp (name, "std::", 5) == 0
#ifndef G__OLDIMPLEMENTATION1285
      && G__ignore_stdnamespace
#endif
      ) {
    /* strip scope operator, set hash and return */
    strcpy(temp,name+5);
    strcpy(name,temp);
    G__hash(name,(*phash),i)
#ifndef G__OLDIMPLEMENTATION1926
    goto re_try_after_std;
#else
    return(G__GLOBALSCOPE);
#endif
  }
#endif
  
  /* otherwise, specific class scope */
  offset_sum=0;
  strcpy(temp,name);
  if(*name=='~') scope=name+1; /* ~A::B() explicit destructor */
  else           scope=name;
  /* recursive scope operator is not allowed in compiler
   * but possible in cint */
#ifndef G__OLDIMPLEMENTATION759
  scopetagnum = G__get_envtagnum();
#endif
  do {
#ifndef G__OLDIMPLEMENTATION759
    int save_tagdefining, save_def_tagnum;
    save_tagdefining = G__tagdefining;
    save_def_tagnum = G__def_tagnum;
    G__tagdefining = scopetagnum;
    G__def_tagnum = scopetagnum;
#endif
    member=pc+2;
    *pc='\0';
#ifndef G__OLDIMPLEMENTATION2181
    scopetagnum=G__defined_tagname(scope,1);
#else
    scopetagnum=G__defined_tagname(scope,0);
#endif
#ifndef G__OLDIMPLEMENTATION759
    G__tagdefining = save_tagdefining;
    G__def_tagnum = save_def_tagnum;
#endif
    
#ifdef G__VIRTUALBASE
#ifndef G__OLDIMPLEMENTATION660
    if(-1==(offset=G__ispublicbase(scopetagnum,*ptagnum
				   ,*pstruct_offset+offset_sum))) {
      int store_tagnum = G__tagnum;
      G__tagnum = *ptagnum;
      offset = -G__find_virtualoffset(scopetagnum); /* NEED REFINEMENT */
      G__tagnum = store_tagnum;
    }
#else
    if(-1==(offset=G__ispublicbase(scopetagnum,*ptagnum
				   ,*pstruct_offset+offset_sum))) offset=0;
#endif
#else
    if(-1==(offset=G__ispublicbase(scopetagnum,*ptagnum))) offset=0;
#endif

    *ptagnum = scopetagnum;
    offset_sum += offset;
    
    scope=member;
#ifndef G__OLDIMPLEMENTATION671
  } while((pc=G__find_first_scope_operator(scope)));
#else
  } while(pc=strstr(scope,"::")); 
#endif
  
  *pstruct_offset += offset_sum;
  
#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) 
      G__fprinterr(G__serr,"%3x: ADDSTROS %d\n" ,G__asm_cp,offset_sum);
#endif
    G__asm_inst[G__asm_cp]=G__ADDSTROS;
    G__asm_inst[G__asm_cp+1]=offset_sum;
    G__inc_cp_asm(2,0);
  }
#endif
  
  strcpy(temp,member);
  if(*name=='~') strcpy(name+1,temp); /* explicit destructor */
  else           strcpy(name,temp);
  G__hash(name,*phash,i)
  return(G__CLASSSCOPE);
}

/**************************************************************************
* G__label_access_scope()
*
*  separated from G__exec_statement()
**************************************************************************/
int G__label_access_scope(statement,piout,pspaceflag,pmparen)
char *statement;
int *piout,*pspaceflag,*pmparen;
{
  int c;
  int ispntr;
  static int memfunc_def_flag=0;
  fpos_t pos;
  int line;
  char temp[G__ONELINE];
  int store_tagdefining;

  c=G__fgetc();
  /**********************************************
   * X::memberfunc() {...} ;
   *    ^  c=':'
   **********************************************/
  if(c==':') {
    /* member function definition */
    if(1==G__prerun && -1==G__func_now && 
       (
#ifndef G__OLDIMPLEMENTATION971
       (-1==G__def_tagnum||'n'==G__struct.type[G__def_tagnum])
#else
        -1==G__def_tagnum
#endif
        ||memfunc_def_flag
#define G__OLDIMPLEMENTATION694 /* Scott Snyder's patch */
#ifndef G__OLDIMPLEMENTATION694
	|| -1!=G__tmplt_def_tagnum
#endif
	)) {
#ifndef G__OLDIMPLEMENTATION971
      int store_def_tagnum = G__def_tagnum;
      int store_def_struct_member = G__def_struct_member;
#endif

      /* X<T>::TYPE X<T>::f() 
       *      ^             */
      fgetpos(G__ifile.fp,&pos);
      line = G__ifile.line_number;
      if(G__dispsource) G__disp_mask=1000;
#ifndef G__OLDIMPLEMENTATION1064
      c=G__fgetname_template(temp,"(;&*");
      if(isspace(c) || c == '&' || c == '*') {
        do {
  	  c=G__fgetspace();
        } while (c == '&' || c == '*');
#else
      c=G__fgetname_template(temp,"(;");
      if(isspace(c)) {
	c=G__fgetspace();
#endif
#ifndef G__STD_NAMESPACE /* ON780 */
	if((isalpha(c) && strcmp(temp,"operator")!=0) ||
	   (strcmp(statement,"std:")==0
#ifndef G__OLDIMPLEMENTATION1285
	    && G__ignore_stdnamespace
#endif
	   )) {
#else
	if(isalpha(c) && strcmp(temp,"operator")!=0) {
#endif
	  /* X<T>::TYPE X<T>::f() 
	   *      space^^alpha , taking as a nested class specification */
	  fsetpos(G__ifile.fp,&pos);
	  G__ifile.line_number=line;
	  if(G__dispsource) G__disp_mask=0;
	  statement[(*piout)++]=':';
	  return(0);
	}
      }
      fsetpos(G__ifile.fp,&pos);
      G__ifile.line_number=line;
      if(G__dispsource) G__disp_mask=0;
      c=':';

      statement[*piout-1] = '\0'; /* tag name */
      if('*'==statement[0]) {
	ispntr=1;
	G__var_type = toupper(G__var_type);
      }
      else {
	ispntr=0;
      }
      /* G__TEMPLATECLASS case 6) */
      G__def_tagnum = G__defined_tagname(statement+ispntr,0);
      store_tagdefining = G__tagdefining;
      G__tagdefining = G__def_tagnum;
      memfunc_def_flag=1;
      G__def_struct_member = 1;
      G__exec_statement(); /* basically, make_ifunctable */
      memfunc_def_flag=0;
#ifndef G__OLDIMPLEMENTATION971
      G__def_tagnum = store_def_tagnum;
      G__def_struct_member = store_def_struct_member;
#else
      G__def_struct_member = 0;
      G__def_tagnum = -1; /* store_tagnum ? */
#endif
      G__tagdefining = store_tagdefining;
      *piout = 0;
      *pspaceflag=0;
      if(*pmparen==0) return(1);
    }
    /* ambiguity resolution operator */
    else {
      statement[(*piout)++]=c;
    }
  }
  /**********************************************
   * public: private: protected:
   * case 0:abcde.....
   *         ^  c='a'
   **********************************************/
  else {
    fseek(G__ifile.fp,-1,SEEK_CUR);
    if(c=='\n' /* ||c=='\r' */) --G__ifile.line_number;
    if(G__dispsource) G__disp_mask=1;
    /* set public,private,protected, otherwise ignore */
    if(1==G__prerun||
       ('p'==statement[0]&&(strcmp("public:",statement)==0||
			    strcmp("private:",statement)==0||
			    strcmp("protected:",statement)==0))) {
      statement[*piout]='\0';
      G__setaccess(statement,*piout);
      *piout=0;
      *pspaceflag=0;
    }
    else { /* 0==prerun */
      /***************************************
       * trying to ignore goto label if not ?:
       ***************************************/
      statement[*piout]='\0';
      if(0==G__switch && (char*)NULL==strchr(statement,'?')) {
#ifndef G__OLDIMPLEMENTATION963
        int itmp=0,ctmp;
	ctmp = G__getstream(statement,&itmp,temp,"+-*%/&|<>=^!");
        if(ctmp && 0!=strncmp(statement,"case",4)) {
          G__fprinterr(G__serr,"Error: illegal label name %s",statement) ;
          G__genericerror((char*)NULL);
        }
#endif
	*piout=0;
	*pspaceflag=0;
#ifndef G__OLDIMPLEMENTATION842
	if(G__ASM_FUNC_COMPILE==G__asm_wholefunction)
	  G__add_label_bytecode(statement);
#endif
      }
      /* else ?: operator remains */
    }
  }
  return(0);
}


/****************************************************************
* int G__cmp(G__value buf1,G__value buf2)
* 
****************************************************************/
int G__cmp(buf1,buf2)
G__value buf1,buf2;
{
  switch(buf1.type) {
  case 'a':  /* G__start */
  case 'z':  /* G__default */
  case '\0': /* G__null */
    if(buf1.type==buf2.type)
      return(1);
    else
      return(0);
    /* break; */
  case 'd':
  case 'f':
    if(G__double(buf1)==G__double(buf2))
      return(1);
    else
      return(0);
    /* break; */
  }
  
  if(G__int(buf1)==G__int(buf2)) return(1);
  /* else */ return(0);
}

/**************************************************************************
* G__getunaryop
*  used in G__getexpr()
**************************************************************************/
int G__getunaryop(unaryop,expression,buf,preg)
char unaryop;
char *expression;
char *buf;
G__value *preg;
{
  int nest=0;
  int c=0;
  int i1=1,i2=0;
  G__value reg;
  char prodpower=0;
  
  *preg=G__null;
  for(;;) {
    c=expression[i1];
    switch(c) {
    case '-':
      if(G__isexponent(buf,i2)) {
	buf[i2++]=c;
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
      if(0==nest) {
	buf[i2]='\0';
	if(prodpower) reg=G__getprod(buf);
	else          reg=G__getitem(buf);
	G__bstore(unaryop,reg,preg);
	return(i1);
      }
      buf[i2++]=c;
      break;
    case '*':
    case '/':
    case '%':
    case '@':
    case '~':
    case ' ':
      if(0==nest) prodpower=1;
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
      buf[i2++]=c;
      break;
    }
    ++i1;
  }
}

#ifdef G__VIRTUALBASE
/**************************************************************************
* G__iosrdstate()
*
*   ios rdstate condition test
**************************************************************************/
int G__iosrdstate(pios)
G__value *pios;
{
  char buf[G__MAXNAME];
  G__value result;
  int ig2;
  long store_struct_offset;
  int store_tagnum;
#ifndef G__OLDIMPLEMENTATION1355
  int rdstateflag=0;
#endif

#ifndef G__OLDIMPLEMENTATION975
  if(-1!=pios->tagnum&&'e'==G__struct.type[pios->tagnum]) return(pios->obj.i);
#endif

  /* store member function call environment */
  store_struct_offset = G__store_struct_offset;
  store_tagnum = G__tagnum;
  G__store_struct_offset = pios->obj.i;
  G__tagnum = pios->tagnum;
#ifdef G__ASM
  if(G__asm_noverflow) {
    G__asm_inst[G__asm_cp] = G__PUSHSTROS;
    G__asm_inst[G__asm_cp+1] = G__SETSTROS;
    G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp-2);
      G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp-1);
    }
#endif
  }
#endif /* G__ASM */

  /* call ios::rdstate() */
  sprintf(buf,"rdstate()" /* ,pios->obj.i */ );
  result = G__getfunction(buf,&ig2,G__TRYMEMFUNC);
#ifndef G__OLDIMPLEMENTATION1355
  if(ig2) rdstateflag=1;
#endif

#ifndef G__OLDIMPLEMENTATION1161
  if(0==ig2) {
    sprintf(buf,"operator int()" /* ,pios->obj.i */ );
    result = G__getfunction(buf,&ig2,G__TRYMEMFUNC);
  }
#endif
#ifndef G__OLDIMPLEMENTATION1355
  if(0==ig2) {
    sprintf(buf,"operator bool()" /* ,pios->obj.i */ );
    result = G__getfunction(buf,&ig2,G__TRYMEMFUNC);
  }
  if(0==ig2) {
    sprintf(buf,"operator long()" /* ,pios->obj.i */ );
    result = G__getfunction(buf,&ig2,G__TRYMEMFUNC);
  }
  if(0==ig2) {
    sprintf(buf,"operator short()" /* ,pios->obj.i */ );
    result = G__getfunction(buf,&ig2,G__TRYMEMFUNC);
  }
#endif
#ifndef G__OLDIMPLEMENTATION1585
  if(0==ig2) {
    sprintf(buf,"operator char*()" /* ,pios->obj.i */ );
    result = G__getfunction(buf,&ig2,G__TRYMEMFUNC);
  }
  if(0==ig2) {
    sprintf(buf,"operator const char*()" /* ,pios->obj.i */ );
    result = G__getfunction(buf,&ig2,G__TRYMEMFUNC);
  }
#endif

  /* restore environment */
  G__store_struct_offset = store_struct_offset;
  G__tagnum = store_tagnum;

#ifdef G__ASM
  if(G__asm_noverflow
#ifndef G__OLDIMPLEMENTATION1355
     && rdstateflag
#endif
     ) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: OP1 '!'\n",G__asm_cp+1);
#endif
    G__asm_inst[G__asm_cp] = G__POPSTROS;
    G__inc_cp_asm(1,0); 
    G__asm_inst[G__asm_cp] = G__OP1;
    G__asm_inst[G__asm_cp+1] = '!';
    G__inc_cp_asm(2,0); 
  }
#endif /* G__ASM */

  /* test result */
  if(ig2) {
#ifndef G__OLDIMPLEMENTATION1355
    if(rdstateflag) return(!result.obj.i);
    else            return(result.obj.i);
#else
    return(!result.obj.i);
#endif
  }
  else {
    G__genericerror("Limitation: Cint does not support full iostream functionality in this platform");
    return(0);
  }
}
#endif

/**************************************************************************
* G__overloadopr()
*
*  separated from G__bstore()
**************************************************************************/
int G__overloadopr(operator,expressionin,defined)
int operator;
G__value expressionin;
G__value *defined;
{
  int ig2;
  
  /* struct G__param fpara; */
  char expr[G__LONGLINE],opr[12],arg1[G__LONGLINE],arg2[G__LONGLINE];
  long store_struct_offset; /* used to be int */
  int store_tagnum;
#ifndef G__OLDIMPLEMENTATION1904
  int store_isconst;
#endif
  G__value buffer;
  char *pos;
  int postfixflag=0;
#ifndef G__OLDIMPLEMENTATION2195
  int store_asm_cp;
#endif
  
  switch(operator) {
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
    sprintf(opr,"operator%c",operator);
    break;
    
  case 'A': /* logic and  && */
    sprintf(opr,"operator&&");
    break;
    
  case 'O': /* logic or   || */
    sprintf(opr,"operator||");
    break;
    
  case 'R': /* right shift >> */
    sprintf(opr,"operator>>");
    break;
  case 'L': /* left shift  << */
    sprintf(opr,"operator<<");
    break;
    
  case 'E': 
    sprintf(opr,"operator==");
    break;
  case 'N': 
    sprintf(opr,"operator!=");
    break;
  case 'G': 
    sprintf(opr,"operator>=");
    break;
  case 'l': 
    sprintf(opr,"operator<=");
    break;
    
  case '\0':
    *defined=expressionin;
    return(0);

  case G__OPR_ADDASSIGN:
    sprintf(opr,"operator+=");
    break;
  case G__OPR_SUBASSIGN:
    sprintf(opr,"operator-=");
    break;
  case G__OPR_MODASSIGN:
    sprintf(opr,"operator%%=");
    break;
  case G__OPR_MULASSIGN:
    sprintf(opr,"operator*=");
    break;
  case G__OPR_DIVASSIGN:
    sprintf(opr,"operator/=");
    break;
  case G__OPR_RSFTASSIGN:
    sprintf(opr,"operator>>=");
    break;
  case G__OPR_LSFTASSIGN:
    sprintf(opr,"operator<<=");
    break;
  case G__OPR_BANDASSIGN:
    sprintf(opr,"operator&=");
    break;
  case G__OPR_BORASSIGN:
    sprintf(opr,"operator|=");
    break;
  case G__OPR_EXORASSIGN:
    sprintf(opr,"operator^=");
    break;
  case G__OPR_ANDASSIGN:
    sprintf(opr,"operator&&=");
    break;
  case G__OPR_ORASSIGN:
    sprintf(opr,"operator||=");
    break;

  case G__OPR_POSTFIXINC:
  case G__OPR_PREFIXINC:
    sprintf(opr,"operator++");
    break;
  case G__OPR_POSTFIXDEC:
  case G__OPR_PREFIXDEC:
    sprintf(opr,"operator--");
    break;
    
  default:
    G__genericerror(
	    "Limitation: Can't handle combination of overloading operators"
		    );
    return(0);
  }
  
  /*****************************************************
   * Unary operator
   *****************************************************/
  if(defined->type==0) {
    
    switch(operator) {
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
      *defined=expressionin;
      return(0);
      /* break; */
    }
    
    G__oprovld=1;
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifndef G__OLDIMPLEMENTATION2195
      store_asm_cp = G__asm_cp;
#endif
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__asm_inst[G__asm_cp+1] = G__SETSTROS;
      G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp-2);
	G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp-1);
      }
#endif
    }
#endif /* G__ASM */
    
    /***************************************************
     * search for member function 
     ****************************************************/
    ig2=0;
    switch(operator) {
    case G__OPR_POSTFIXINC:
    case G__OPR_POSTFIXDEC:
      sprintf(expr,"%s(1)",opr);
#ifdef G__ASM
      if(G__asm_noverflow) {
	G__asm_inst[G__asm_cp] = G__LD;
	G__asm_inst[G__asm_cp+1]=G__asm_dt;
	G__asm_stack[G__asm_dt]=G__one;
	G__inc_cp_asm(2,1);
	postfixflag=1;
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD 0x%lx from %lx\n"
			       ,G__asm_cp ,1 ,G__asm_dt);
#endif
      }
#endif
      break;
    default:
      postfixflag=0;
      sprintf(expr,"%s()",opr);
      break;
    }
    
    store_struct_offset = G__store_struct_offset;
    store_tagnum = G__tagnum;
    G__store_struct_offset = expressionin.obj.i;
    G__tagnum = expressionin.tagnum;
    
#ifndef G__OLDIMPLEMENTATION1427
    buffer = G__getfunction(expr,&ig2,G__TRYUNARYOPR);
#else
    buffer = G__getfunction(expr,&ig2,G__TRYMEMFUNC);
#endif
    
    G__store_struct_offset = store_struct_offset;
    G__tagnum = store_tagnum;
    
    /***************************************************
     * search for global function
     ****************************************************/
    if(ig2==0) {
#ifdef G__ASM
      if(G__asm_noverflow) {
	if(postfixflag) {
	  G__inc_cp_asm(-2,-1);
	  postfixflag=0;
#ifdef G__ASM_DBG
	  if(G__asm_dbg) G__fprinterr(G__serr,"LD cancelled\n");
#endif
	}
#ifndef G__OLDIMPLEMENTATION2195
	G__inc_cp_asm(store_asm_cp-G__asm_cp,0);
#else
	G__inc_cp_asm(-2,0); 
#endif
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"PUSHSTROS,SETSTROS cancelled\n");
#endif
      }
#endif /* G__ASM */
      switch(operator) {
      case G__OPR_POSTFIXINC:
      case G__OPR_POSTFIXDEC:
#if !defined(G__OLDIMPLEMENTATION1825)
	sprintf(expr,"%s(%s,1)",opr 
		,G__setiparseobject(&expressionin,arg1));
#elif !defined(G__OLDIMPLEMENTATION719)
	if(expressionin.obj.i<0)
	  sprintf(expr,"%s((%s)(%ld),1)",opr 
		  ,G__fulltagname(expressionin.tagnum,1),expressionin.obj.i);
	else 
	  sprintf(expr,"%s((%s)%ld,1)",opr 
		  ,G__fulltagname(expressionin.tagnum,1),expressionin.obj.i);
#else
	if(expressionin.obj.i<0)
	  sprintf(expr,"%s((%s)(%ld),1)",opr 
		  ,G__struct.name[expressionin.tagnum],expressionin.obj.i);
	else 
	  sprintf(expr,"%s((%s)%ld,1)",opr 
		  ,G__struct.name[expressionin.tagnum] ,expressionin.obj.i);
#endif
#ifdef G__ASM
	if(G__asm_noverflow) {
	  G__asm_inst[G__asm_cp] = G__LD;
	  G__asm_inst[G__asm_cp+1]=G__asm_dt;
	  G__asm_stack[G__asm_dt]=G__one;
	  G__inc_cp_asm(2,1);
#ifdef G__ASM_DBG
	  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD 0x%lx from %lx\n"
				 ,G__asm_cp ,1 ,G__asm_dt);
#endif
	}
#endif
	break;
      default:
#if !defined(G__OLDIMPLEMENTATION1825)
	sprintf(expr,"%s(%s)",opr 
		,G__setiparseobject(&expressionin,arg1));
#elif !defined(G__OLDIMPLEMENTATION719)
	if(expressionin.obj.i<0)
	  sprintf(expr,"%s((%s)(%ld))",opr
		  ,G__fulltagname(expressionin.tagnum,1),expressionin.obj.i);
	else 
	  sprintf(expr,"%s((%s)%ld)" ,opr 
		  ,G__fulltagname(expressionin.tagnum,1),expressionin.obj.i);
#else
	if(expressionin.obj.i<0)
	  sprintf(expr,"%s((%s)(%ld))",opr
		  ,G__struct.name[expressionin.tagnum] ,expressionin.obj.i);
	else 
	  sprintf(expr,"%s((%s)%ld)" ,opr 
		  ,G__struct.name[expressionin.tagnum] ,expressionin.obj.i);
#endif
	break;
      }
      buffer = G__getfunction(expr,&ig2,G__TRYNORMAL);
    }
#ifdef G__ASM
    else if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1,0); 
    }
#endif /* G__ASM */
    *defined = buffer;
    
    G__oprovld=0;
  } /* end of if(defined->type==0) */

  /*****************************************************
   * Binary operator
   *****************************************************/
  else {
    
    G__oprovld=1;
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifdef G__ASM_IFUNC
#ifndef G__OLDIMPLEMENTATION2195
      store_asm_cp = G__asm_cp;
#endif
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SWAP\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp] = G__SWAP;
      G__inc_cp_asm(1,0);
#endif
      G__asm_inst[G__asm_cp] = G__PUSHSTROS;
      G__asm_inst[G__asm_cp+1] = G__SETSTROS;
      G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp-2);
	G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp-1);
      }
#endif
    }
#endif /* G__ASM */
    
    /***************************************************
     * search for member function 
     ****************************************************/
    ig2=0;
    
    if(expressionin.type=='u') {
#if !defined(G__OLDIMPLEMENTATION1825)
      G__setiparseobject(&expressionin,arg2);
#elif !defined(G__OLDIMPLEMENTATION719)
      if(expressionin.obj.i<0)
	sprintf(arg2,"(%s)(%ld)" ,G__fulltagname(expressionin.tagnum,1)
		,expressionin.obj.i);
      else
	sprintf(arg2,"(%s)%ld" ,G__fulltagname(expressionin.tagnum,1)
		,expressionin.obj.i);
#else
      if(expressionin.obj.i<0)
	sprintf(arg2,"(%s)(%ld)" ,G__struct.name[expressionin.tagnum]
		,expressionin.obj.i);
      else
	sprintf(arg2,"(%s)%ld" ,G__struct.name[expressionin.tagnum]
		,expressionin.obj.i);
#endif
    }
    else {
      G__valuemonitor(expressionin,arg2);
      /* This part must be fixed when reference to pointer type
       * is supported */
#ifndef G__OLDIMPLEMENTATION1017
      if(expressionin.ref && 1!=expressionin.ref) {
#else
      if(expressionin.ref) {
#endif
	pos=strchr(arg2,')');
	*pos = '\0';
	if(expressionin.ref<0)
	  sprintf(expr,"*%s*)(%ld)",arg2,expressionin.ref);  
	else
	  sprintf(expr,"*%s*)%ld",arg2,expressionin.ref);  
	strcpy(arg2,expr);
      }
    }

    if(defined->type=='u') {
      sprintf(expr,"%s(%s)" ,opr ,arg2);
      
      store_struct_offset = G__store_struct_offset;
      store_tagnum = G__tagnum;
      G__store_struct_offset = defined->obj.i;
      G__tagnum = defined->tagnum; 
#ifndef G__OLDIMPLEMENTATION1904
      store_isconst = G__isconst;
      G__isconst = defined->isconst;
#endif
      
#ifndef G__OLDIMPLEMENTATION1427
      buffer = G__getfunction(expr,&ig2,G__TRYBINARYOPR);
#else
      buffer = G__getfunction(expr,&ig2,G__TRYMEMFUNC);
#endif
      
#ifndef G__OLDIMPLEMENTATION1904
      G__isconst = store_isconst;
#endif
      G__store_struct_offset = store_struct_offset;
      G__tagnum = store_tagnum;
    }
    
    /***************************************************
     * search for global function
     ****************************************************/
    if(ig2==0) {
#ifdef G__ASM
      if(G__asm_noverflow) {
#ifndef G__OLDIMPLEMENTATION2152
	G__bc_cancel_VIRTUALADDSTROS();
#endif
#ifndef G__OLDIMPLEMENTATION2195
	G__inc_cp_asm(store_asm_cp-G__asm_cp,0); 
#else
#ifdef G__ASM_IFUNC
	G__inc_cp_asm(-3,0); 
#else
	G__inc_cp_asm(-2,0); 
#endif
#endif
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"PUSHSTROS,SETSTROS cancelled\n");
#endif
      }
#endif /* of G__ASM */

      if(defined->type=='u') {
#if !defined(G__OLDIMPLEMENTATION1825)
	G__setiparseobject(defined,arg1);
#elif !defined(G__OLDIMPLEMENTATION719)
	if(defined->obj.i<0)
	  sprintf(arg1,"(%s)(%ld)"
		  ,G__fulltagname(defined->tagnum,1),defined->obj.i);
	else
	  sprintf(arg1,"(%s)%ld"
		  ,G__fulltagname(defined->tagnum,1),defined->obj.i);
#else
	if(defined->obj.i<0)
	  sprintf(arg1,"(%s)(%ld)"
		  ,G__struct.name[defined->tagnum],defined->obj.i);
	else
	  sprintf(arg1,"(%s)%ld"
		  ,G__struct.name[defined->tagnum],defined->obj.i);
#endif
      }
      else {
	G__valuemonitor(*defined,arg1);
	/* This part must be fixed when reference to pointer type
	 * is supported */
#ifndef G__OLDIMPLEMENTATION995
	if(defined->ref) {
#else
	if(defined->ref && islower(defined->type)) {
#endif
	  pos=strchr(arg1,')');
	  *pos = '\0';
	  if(defined->ref<0)
	    sprintf(expr,"*%s*)(%ld)",arg1,defined->ref);  
	  else
	    sprintf(expr,"*%s*)%ld",arg1,defined->ref);  
	  strcpy(arg1,expr);
	}
      }
      sprintf(expr,"%s(%s,%s)" ,opr ,arg1 ,arg2);
      buffer = G__getfunction(expr,&ig2,G__TRYNORMAL);
      /* #ifdef G__OLDIMPLEMENTATION1286_YET */
#ifndef G__OLDIMPLEMENTATION1862
      /* Need to check ANSI/ISO standard. What happens if operator 
       * function defined in a namespace is used in other namespace */
      if(0==ig2 && -1!=expressionin.tagnum && 
	 -1!= G__struct.parent_tagnum[expressionin.tagnum]) {
	sprintf(expr,"%s::%s(%s,%s)"
		,G__fulltagname(G__struct.parent_tagnum[expressionin.tagnum],1) 
		,opr ,arg1 ,arg2);
	buffer = G__getfunction(expr,&ig2,G__TRYNORMAL);
      }
      if(0==ig2 && -1!=defined->tagnum && 
	 -1!= G__struct.parent_tagnum[defined->tagnum]) {
	sprintf(expr,"%s::%s(%s,%s)"
		,G__fulltagname(G__struct.parent_tagnum[defined->tagnum],1) 
		,opr ,arg1 ,arg2);
	buffer = G__getfunction(expr,&ig2,G__TRYNORMAL);
      }
#endif

#ifndef G__OLDIMPLEMENTATION1340
      if(0==ig2 && ('A'==operator||'O'==operator)) {
	int lval,rval;
	if('u'==defined->type) {
	  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SWAP\n",G__asm_cp);
#endif
	    G__asm_inst[G__asm_cp] = G__SWAP;
	    G__inc_cp_asm(1,0);
	  }
	  lval = G__iosrdstate(defined);
	  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SWAP\n",G__asm_cp);
#endif
	    G__asm_inst[G__asm_cp] = G__SWAP;
	    G__inc_cp_asm(1,0);
	  }
	}
	else                   lval = G__int(*defined);
	if('u'==expressionin.type) rval = G__iosrdstate(&expressionin);
	else                       rval = G__int(expressionin);
	buffer.ref=0;
	buffer.tagnum  = -1;
	buffer.typenum = -1;
	switch(operator) {
	case 'A':
	  G__letint(&buffer,'i', lval&&rval);
	  break;
	case 'O':
	  G__letint(&buffer,'i', lval||rval);
	  break;
	}
	if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	  if(G__asm_dbg) {
	    if(isprint(operator)) 
	      G__fprinterr(G__serr,"%3x: OP2  '%c' %d\n"
		      ,G__asm_cp,operator,operator);
	    else
	      G__fprinterr(G__serr,"%3x: OP2  %d\n"
		      ,G__asm_cp,operator);
	  }
#endif
	  G__asm_inst[G__asm_cp]=G__OP2;
	  G__asm_inst[G__asm_cp+1]=operator;
	  G__inc_cp_asm(2,0);
	}
	ig2=1;
      }
#endif

#ifndef G__OLDIMPLEMENTATION865
      if(0==ig2) {
#ifndef G__OLDIMPLEMENTATION1252
	if(-1!=defined->tagnum) {
	  G__fprinterr(G__serr,"Error: %s not defined for %s"
		  ,opr,G__fulltagname(defined->tagnum,1));
	}
	else {
	  G__fprinterr(G__serr,"Error: %s not defined",expr);
	}
#else
	G__fprinterr(G__serr,"Error: %s not defined for %s"
		,opr,G__fulltagname(defined->tagnum,1));
#endif
	G__genericerror((char*)NULL);
      }
#endif
    }
#ifdef G__ASM
    else if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1,0); 
    }
#endif /* G__ASM */
    *defined = buffer;
      
    G__oprovld=0;
  }  /* end ob binary operator else */
  return(0);
}

#ifndef G__OLDIMPLEMENTATION1871
/**************************************************************************
* G__parenthesisovldobj()
*
**************************************************************************/
int G__parenthesisovldobj(result3,result,realname,libp,flag)
G__value *result3;
G__value *result;
char *realname;
struct G__param *libp;
int flag; /* flag whether to generate PUSHSTROS, SETSTROS */
{
  int known;
  long store_struct_offset;
  int store_tagnum;
  int funcmatch;
  int hash;
  int store_exec_memberfunc;
  int store_memberfunc_tagnum;
  int store_memberfunc_struct_offset;

#ifndef G__OLDIMPLEMENTATION1911
  if(0 && flag) return(0);
#endif

  store_exec_memberfunc=G__exec_memberfunc;
  store_memberfunc_tagnum=G__memberfunc_tagnum;
  store_memberfunc_struct_offset=G__memberfunc_struct_offset;

  store_struct_offset = G__store_struct_offset;
  store_tagnum = G__tagnum;
  G__store_struct_offset = result->obj.i;
  G__tagnum = result->tagnum;

#ifdef G__ASM
  if(G__asm_noverflow
#ifndef G__OLDIMPLEMENTATION2120
     && !flag
#endif
     ) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp);
      G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp+1);
    }
#endif
    G__asm_inst[G__asm_cp] = G__PUSHSTROS;
    G__asm_inst[G__asm_cp+1] = G__SETSTROS;
    G__inc_cp_asm(2,0);
  }
#endif

  G__hash(realname,hash,known);

  G__fixedscope=0;

  for(funcmatch=G__EXACT;funcmatch<=G__USERCONV;funcmatch++) {
    if(-1!=G__tagnum) G__incsetup_memfunc(G__tagnum);
    if(G__interpret_func(result3,realname,libp,hash
			 ,G__struct.memfunc[G__tagnum]
			 ,funcmatch,G__CALLMEMFUNC)==1 ) {
      G__store_struct_offset = store_struct_offset;
      G__tagnum = store_tagnum;

#ifdef G__ASM
      if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
	G__asm_inst[G__asm_cp] = G__POPSTROS;
	G__inc_cp_asm(1,0);
      }
#endif

      G__exec_memberfunc=store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
      return(1);
    }
  }

  G__store_struct_offset = store_struct_offset;
  G__tagnum = store_tagnum;

#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp] = G__POPSTROS;
    G__inc_cp_asm(1,0);
  }
#endif

  G__exec_memberfunc=store_exec_memberfunc;
  G__memberfunc_tagnum=store_memberfunc_tagnum;
  G__memberfunc_struct_offset=store_memberfunc_struct_offset;
  return(0);
}

#endif

/**************************************************************************
* G__parenthesisovld()
*
**************************************************************************/
int G__parenthesisovld(result3,funcname,libp,flag)
G__value *result3;
char *funcname;
struct G__param *libp;
int flag;
{
  int known;
  G__value result;
  long store_struct_offset;
  int store_tagnum;
  int funcmatch;
  int hash;
  char realname[G__ONELINE];
  int store_exec_memberfunc;
  int store_memberfunc_tagnum;
  int store_memberfunc_struct_offset;

#ifndef G__OLDIMPLEMENTATION745
  if(strncmp(funcname,"operator",8)==0 || strcmp(funcname,"G__ateval")==0) 
    return(0);
#endif

#ifndef G__OLDIMPLEMENTATION1871
  if(0==funcname[0]) {
    result = *result3;
  }
  else 
#endif

  if(flag==G__CALLMEMFUNC) {
    G__incsetup_memvar(G__tagnum);
    result = G__getvariable(funcname,&known,(struct G__var_array*)NULL
			    ,G__struct.memvar[G__tagnum]);
  }
  else {
    result = G__getvariable(funcname,&known,&G__global,G__p_local);
  }

#ifndef G__OLDIMPLEMENTATION1902
  /* resolve A::staticmethod(1)(2,3) */
#endif

  if(
#ifndef G__OLDIMPLEMENTATION1876
     1!=known 
#else
     0==known 
#endif
     || -1 == result.tagnum) return(0);

  store_exec_memberfunc=G__exec_memberfunc;
  store_memberfunc_tagnum=G__memberfunc_tagnum;
  store_memberfunc_struct_offset=G__memberfunc_struct_offset;

  store_struct_offset = G__store_struct_offset;
  store_tagnum = G__tagnum;
  G__store_struct_offset = result.obj.i;
  G__tagnum = result.tagnum;

#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp);
      G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp+1);
    }
#endif
    G__asm_inst[G__asm_cp] = G__PUSHSTROS;
    G__asm_inst[G__asm_cp+1] = G__SETSTROS;
    G__inc_cp_asm(2,0);
  }
#endif

  sprintf(realname,"operator()");
  G__hash(realname,hash,known);

  G__fixedscope=0;

  for(funcmatch=G__EXACT;funcmatch<=G__USERCONV;funcmatch++) {
    if(-1!=G__tagnum) G__incsetup_memfunc(G__tagnum);
    if(G__interpret_func(result3,realname,libp,hash
			 ,G__struct.memfunc[G__tagnum]
			 ,funcmatch,G__CALLMEMFUNC)==1 ) {
      G__store_struct_offset = store_struct_offset;
      G__tagnum = store_tagnum;

#ifdef G__ASM
      if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
	G__asm_inst[G__asm_cp] = G__POPSTROS;
	G__inc_cp_asm(1,0);
      }
#endif

      G__exec_memberfunc=store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
      return(1);
    }
  }

  G__store_struct_offset = store_struct_offset;
  G__tagnum = store_tagnum;

#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp] = G__POPSTROS;
    G__inc_cp_asm(1,0);
  }
#endif

  G__exec_memberfunc=store_exec_memberfunc;
  G__memberfunc_tagnum=store_memberfunc_tagnum;
  G__memberfunc_struct_offset=store_memberfunc_struct_offset;
  return(0);
}


/**************************************************************************
* G__tryindexopr()
*
* 1) asm
*    * G__ST_VAR/MSTR -> LD_VAR/MSTR
*    * paran -> ig25
* 2) try operator[]() function while ig25<paran
*
**************************************************************************/
int G__tryindexopr(result7,para,paran,ig25)
G__value *result7;
G__value *para;
int paran,ig25;
{
  char expr[G__ONELINE];
  char arg2[G__MAXNAME];
  char *pos;
  int store_tagnum;
  int store_typenum;
  int store_struct_offset;
  int known;
  int i;
  int store_asm_exec;

#ifdef G__ASM
  if(G__asm_noverflow) {
    /*  X a[2][3]; 
     *  Y X::operator[]()
     *  Y::operator[]()
     *    a[x][y][z][w];   stack x y z w ->  stack w z x y 
     *                                             Y X a a
     */
    if(paran>1 && paran>ig25) {
#ifdef G__ASM_DBG
      if(G__asm_dbg)
	G__fprinterr(G__serr,"%x: REORDER inserted before ST_VAR/MSTR/LD_VAR/MSTR\n"
		,G__asm_cp-5);
#endif
      for(i=1;i<=5;i++) G__asm_inst[G__asm_cp-i+3]=G__asm_inst[G__asm_cp-i];
      G__asm_inst[G__asm_cp-5]= G__REORDER ;
      G__asm_inst[G__asm_cp-4]= paran ;
      G__asm_inst[G__asm_cp-3]= ig25 ;
      G__inc_cp_asm(3,0);
    }
    switch(G__asm_inst[G__asm_cp-5]) {
    case G__ST_MSTR:
      G__asm_inst[G__asm_cp-5]=G__LD_MSTR;
      break;
    case G__ST_VAR:
      G__asm_inst[G__asm_cp-5]=G__LD_VAR;
      break;
    default:
      break;
    }
    G__asm_inst[G__asm_cp-3]=ig25;
#ifdef G__ASM_DBG
    if(G__asm_dbg)
      G__fprinterr(G__serr,"ST_VAR/MSTR replaced to LD_VAR/MSTR, paran=%d -> %d\n"
	      ,paran,ig25);
#endif
  }
#endif

  store_tagnum = G__tagnum;
  store_typenum = G__typenum;
  store_struct_offset = G__store_struct_offset;
#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp] = G__PUSHSTROS;
    G__inc_cp_asm(1,0);
  }
#endif


#ifdef G__OLDIMPLEMENTATION860
  G__oprovld = 1;
#endif

  while(ig25<paran) {
#ifndef G__OLDIMPLEMENTATION860
    G__oprovld = 1;
#endif
#ifndef G__OLDIMPLEMENTATION492
    if('u'==result7->type) {
#endif
      G__tagnum = result7->tagnum;
      G__typenum = result7->typenum;
      G__store_struct_offset = result7->obj.i;
#ifdef G__ASM
      if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp);
#endif
	G__asm_inst[G__asm_cp] = G__SETSTROS;
	G__inc_cp_asm(1,0);
      }
#endif
      
      if(para[ig25].type=='u') {
#if !defined(G__OLDIMPLEMENTATION1825)
	G__setiparseobject(&para[ig25],arg2);
#elif !defined(G__OLDIMPLEMENTATION409)
	if(para[ig25].obj.i<0)
	  sprintf(arg2,"(%s)(%ld)",G__struct.name[para[ig25].tagnum]
		  ,para[ig25].obj.i);
	else
	  sprintf(arg2,"(%s)%ld",G__struct.name[para[ig25].tagnum]
		  ,para[ig25].obj.i);
#else
	sprintf(arg2,"(%s)%ld",G__struct.name[para[ig25].tagnum]
		,para[ig25].obj.i);
#endif
      }
      else {
	G__valuemonitor(para[ig25],arg2);
	/* This part must be fixed when reference to pointer type
	 * is supported */
#ifndef G__OLDIMPLEMENTATION995
	if(para[ig25].ref) {
#else
	if(para[ig25].ref && islower(para[ig25].type)) {
#endif
	  pos=strchr(arg2,')');
	  *pos = '\0';
#ifndef G__OLDIMPLEMENTATION409
	  if(para[ig25].ref<0)
	    sprintf(expr,"*%s*)(%ld)",arg2,para[ig25].ref);  
	  else
	    sprintf(expr,"*%s*)%ld",arg2,para[ig25].ref);  
#else
	  sprintf(expr,"*%s*)%ld",arg2,para[ig25].ref);  
#endif
	  strcpy(arg2,expr);
	}
      }
      
      sprintf(expr,"operator[](%s)",arg2);
      store_asm_exec = G__asm_exec;
      G__asm_exec=0;
      *result7 = G__getfunction(expr,&known,G__CALLMEMFUNC);
      G__asm_exec = store_asm_exec;
#ifndef G__OLDIMPLEMENTATION492
    }
    /* in case 'T* operator[]' */
    else if(isupper(result7->type)) {
      result7->obj.i += G__sizeof(result7)*para[ig25].obj.i;
#ifdef G__ASM
      if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: OP2 +\n",G__asm_cp);
#endif
	G__asm_inst[G__asm_cp] = G__OP2;
	G__asm_inst[G__asm_cp+1] = '+';
	G__inc_cp_asm(2,0);
      }
#endif
      *result7 = G__tovalue(*result7);
    }
#endif

    ++ig25;
  }

  G__oprovld = 0 ;

  G__tagnum = store_tagnum;
  G__typenum = store_typenum;
  G__store_struct_offset = store_struct_offset;
#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp] = G__POPSTROS;
    G__inc_cp_asm(1,0);
  }
#endif
  return(0);
}



#ifndef G__OLDIMPLEMENTATION572
/**************************************************************************
* G__op1_operator_detail()
*
**************************************************************************/
long G__op1_operator_detail(opr,val)
int opr;
G__value *val;
{
  /* int isdouble; */

  /* don't optimze if optimize level is less than 3 */
  if(G__asm_loopcompile<3) return(opr);

  if('i'==val->type) {
    switch(opr) {
    case G__OPR_POSTFIXINC: return(G__OPR_POSTFIXINC_I);
    case G__OPR_POSTFIXDEC: return(G__OPR_POSTFIXDEC_I);
    case G__OPR_PREFIXINC:  return(G__OPR_PREFIXINC_I);
    case G__OPR_PREFIXDEC:  return(G__OPR_PREFIXDEC_I);
    }
  }
  else if('d'==val->type) {
    switch(opr) {
    case G__OPR_POSTFIXINC: return(G__OPR_POSTFIXINC_D);
    case G__OPR_POSTFIXDEC: return(G__OPR_POSTFIXDEC_D);
    case G__OPR_PREFIXINC:  return(G__OPR_PREFIXINC_D);
    case G__OPR_PREFIXDEC:  return(G__OPR_PREFIXDEC_D);
    }
  }
#ifdef G__NEVER /* following change rather slowed down */
  else if('l'==val->type) {
    switch(opr) {
    case G__OPR_POSTFIXINC: return(G__OPR_POSTFIXINC_L);
    case G__OPR_POSTFIXDEC: return(G__OPR_POSTFIXDEC_L);
    case G__OPR_PREFIXINC:  return(G__OPR_PREFIXINC_L);
    case G__OPR_PREFIXDEC:  return(G__OPR_PREFIXDEC_L);
    }
  }
  else if('s'==val->type) {
    switch(opr) {
    case G__OPR_POSTFIXINC: return(G__OPR_POSTFIXINC_S);
    case G__OPR_POSTFIXDEC: return(G__OPR_POSTFIXDEC_S);
    case G__OPR_PREFIXINC:  return(G__OPR_PREFIXINC_S);
    case G__OPR_PREFIXDEC:  return(G__OPR_PREFIXDEC_S);
    }
  }
  else if('h'==val->type) {
    switch(opr) {
    case G__OPR_POSTFIXINC: return(G__OPR_POSTFIXINC_H);
    case G__OPR_POSTFIXDEC: return(G__OPR_POSTFIXDEC_H);
    case G__OPR_PREFIXINC:  return(G__OPR_PREFIXINC_H);
    case G__OPR_PREFIXDEC:  return(G__OPR_PREFIXDEC_H);
    }
  }
  else if('R'==val->type) {
    switch(opr) {
    case G__OPR_POSTFIXINC: return(G__OPR_POSTFIXINC_R);
    case G__OPR_POSTFIXDEC: return(G__OPR_POSTFIXDEC_R);
    case G__OPR_PREFIXINC:  return(G__OPR_PREFIXINC_R);
    case G__OPR_PREFIXDEC:  return(G__OPR_PREFIXDEC_R);
    }
  }
  else if('k'==val->type) {
    switch(opr) {
    case G__OPR_POSTFIXINC: return(G__OPR_POSTFIXINC_K);
    case G__OPR_POSTFIXDEC: return(G__OPR_POSTFIXDEC_K);
    case G__OPR_PREFIXINC:  return(G__OPR_PREFIXINC_K);
    case G__OPR_PREFIXDEC:  return(G__OPR_PREFIXDEC_K);
    }
  }
  else if('f'==val->type) {
    switch(opr) {
    case G__OPR_POSTFIXINC: return(G__OPR_POSTFIXINC_F);
    case G__OPR_POSTFIXDEC: return(G__OPR_POSTFIXDEC_F);
    case G__OPR_PREFIXINC:  return(G__OPR_PREFIXINC_F);
    case G__OPR_PREFIXDEC:  return(G__OPR_PREFIXDEC_F);
    }
  }
#endif
  return(opr);
}

/**************************************************************************
* G__op2_operator_detail()
*
**************************************************************************/
long G__op2_operator_detail(opr,lval,rval)
int opr;
G__value *lval;
G__value *rval;
{
  int lisdouble,risdouble;
  int lispointer,rispointer;

  /* don't optimze if optimize level is less than 3 */
  if(G__asm_loopcompile<3) return(opr);

#ifndef G__OLDIMPLEMENTATION2189
  switch(lval->type) {
  case 'q': case 'n': case 'm': return(opr);
  }
  switch(rval->type) {
  case 'q': case 'n': case 'm': return(opr);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1007
  if(0==rval->type
#ifndef G__OLDIMPLEMENTATION1007
     && 0==G__xrefflag
#endif
     ) {
    G__genericerror("Error: Binary operator oprand missing");
  }
#endif

  lisdouble = G__isdouble(*lval);
  risdouble = G__isdouble(*rval);

  if(0==lisdouble && 0==risdouble) {
    lispointer = isupper(lval->type);
    rispointer = isupper(rval->type);
    if(0==lispointer && 0==rispointer) {
#ifndef G__OLDIMPLEMENTATION1491
      if('k'==lval->type || 'h'==lval->type ||
	 'k'==rval->type || 'h'==rval->type) {
	switch(opr) {
	case G__OPR_ADD: return(G__OPR_ADD_UU);
	case G__OPR_SUB: return(G__OPR_SUB_UU);
	case G__OPR_MUL: return(G__OPR_MUL_UU);
	case G__OPR_DIV: return(G__OPR_DIV_UU);
	default:
	  switch(lval->type) {
	  case 'i':
	    switch(opr) {
	    case G__OPR_ADDASSIGN: return(G__OPR_ADDASSIGN_UU);
	    case G__OPR_SUBASSIGN: return(G__OPR_SUBASSIGN_UU);
	    case G__OPR_MULASSIGN: return(G__OPR_MULASSIGN_UU);
	    case G__OPR_DIVASSIGN: return(G__OPR_DIVASSIGN_UU);
	    }
	  }
	  break;
	}
      }
      else {
#endif
	switch(opr) {
	case G__OPR_ADD: return(G__OPR_ADD_II);
	case G__OPR_SUB: return(G__OPR_SUB_II);
	case G__OPR_MUL: return(G__OPR_MUL_II);
	case G__OPR_DIV: return(G__OPR_DIV_II);
	default:
	  switch(lval->type) {
	  case 'i':
	    switch(opr) {
	    case G__OPR_ADDASSIGN: return(G__OPR_ADDASSIGN_II);
	    case G__OPR_SUBASSIGN: return(G__OPR_SUBASSIGN_II);
	    case G__OPR_MULASSIGN: return(G__OPR_MULASSIGN_II);
	    case G__OPR_DIVASSIGN: return(G__OPR_DIVASSIGN_II);
	    }
	  }
	  break;
	}
#ifndef G__OLDIMPLEMENTATION1491
      }
#endif
    }
  }
  else if(lisdouble && risdouble) {
    switch(opr) {
    case G__OPR_ADD: return(G__OPR_ADD_DD);
    case G__OPR_SUB: return(G__OPR_SUB_DD);
    case G__OPR_MUL: return(G__OPR_MUL_DD);
    case G__OPR_DIV: return(G__OPR_DIV_DD);
    default:
      switch(lval->type) {
      case 'd':
	switch(opr) {
	case G__OPR_ADDASSIGN: return(G__OPR_ADDASSIGN_DD);
	case G__OPR_SUBASSIGN: return(G__OPR_SUBASSIGN_DD);
	case G__OPR_MULASSIGN: return(G__OPR_MULASSIGN_DD);
	case G__OPR_DIVASSIGN: return(G__OPR_DIVASSIGN_DD);
	}
      case 'f':
	switch(opr) {
	case G__OPR_ADDASSIGN: return(G__OPR_ADDASSIGN_FD);
	case G__OPR_SUBASSIGN: return(G__OPR_SUBASSIGN_FD);
	case G__OPR_MULASSIGN: return(G__OPR_MULASSIGN_FD);
	case G__OPR_DIVASSIGN: return(G__OPR_DIVASSIGN_FD);
	}
      }
      break;
    }
  }
  return(opr);
}
#endif

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:2
 * c-continued-statement-offset:2
 * c-brace-offset:-2
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-2
 * compile-command:"make -k"
 * End:
 */
