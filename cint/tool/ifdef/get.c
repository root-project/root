/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/* /% C %/ */

 /************************************************************************
 * get.c
 *
 *  Symbolic resolver
 *
 *  Copyright: Yokogawa-Hewlett-Packard, HSTD R&D
 *  Author   : Masaharu Goto
 *  Date     : Aug 1990
 *
 ************************************************************************/

#ifdef G__BORLANDCC5
#define G__ANSI
#endif

#ifdef G__ANSI
#define G__P(funcparam) funcparam
#else
#define G__P(funcparam) ()
#endif


 /***********************************************************************/
 /* char *G__calc(char *expression)                                     */
 /*                                                                     */
 /*   Formula evaluation program.                                       */
 /*                                                                     */
 /*   Numerical expression:                                             */
 /*    T,G,M,K,m,u,n,p,f,a can be used to define real number. For       */
 /*    example, 1k -> 1000.0, 1.01u -> 1.01E-6. upper case 'M'          */
 /*    represents mega(1E6), and lower case 'm' represents mili(1E-3).  */
 /*                                                                     */
 /*   Hex,Binary expression:                                            */
 /*    Binary,Quad,Octal and Hexadecimal expressions are available with */
 /*    %[bqohx] or 0[bqohx]. For example, %hff, %b10110010, 0xa0, 0b10. */
 /*    This kind of expression will be translated to integer.           */
 /*                                                                     */
 /*   Variable:                                                         */
 /*    Variables can be used in the expression. For example, a=315 ,    */
 /*    b=a+3 , c=a+b+(pi=3.14). Variable name should not include more   */
 /*    than 20 characters.                                              */
 /*                                                                     */
 /*   Array:                                                            */
 /*    Array can be defined by defining the largest item of the array.  */
 /*    For example, ary[1][3][2]=3.14 allocates ary(0:1,0:3,0:2).       */
 /*    Up to 10 dimension array can be used.                            */
 /*                                                                     */
 /*   Undefined variable:                                               */
 /*    If undefined variable name is found in the expression, G__calc() */
 /*    will leave undefined variable in the formula as it is.           */
 /*                                                                     */
 /*   Undefined signal expression with hex binary expression:           */
 /*    With %[bqohx] expression, undefind(U),hiz(Z) and unknown(X)      */
 /*    simbols can be used. If those simbols are found in expression,   */
 /*    the number is treated as undefined variable. For example, %hZZ   */
 /*    (or %b0X010UZZ) returns string %hZZ (or %b0X010UZZ) as it is.    */
 /*                                                                     */
 /*   Space or Tab character:                                           */
 /*    Space and Tab character are ignored in the G__calc(). This is the*/
 /*    only difference between G__getexpr() and G__calc().              */
 /*                                                                     */
 /*                                           12 Mar 1991     M.Goto    */
 /***********************************************************************/



 /***********************************************************************/
 /* char *G__getexpr(char *expression)                                  */
 /*                                                                     */
 /*   Formula evaluation program.                                       */
 /*                                                                     */
 /*   Numerical expression:                                             */
 /*    T,G,M,K,m,u,n,p,f,a can be used to define real number. For       */
 /*    example, 1k -> 1000.0, 1.01u -> 1.01E-6. upper case 'M'          */
 /*    represents mega(1E6), and lower case 'm' represents mili(1E-3).  */
 /*                                                                     */
 /*   Hex,Binary expression:                                            */
 /*    Binary,Quad,Octal and Hexadecimal expressions are available with */
 /*    %[bqohx] or 0[bqohx]. For example, %hff, %b10110010, 0xa0, 0b10. */
 /*    This kind of expression will be translated to integer.           */
 /*                                                                     */
 /*   Variable:                                                         */
 /*    Variables can be used in the expression. For example, a=315 ,    */
 /*    b=a+3 , c=a+b+(pi=3.14). Variable name should not include more   */
 /*    than 20 characters.                                              */
 /*                                                                     */
 /*   Array:                                                            */
 /*    Array can be defined by defining the largest item of the array.  */
 /*    For example, ary[1][3][2]=3.14 allocates ary(0:1,0:3,0:2).       */
 /*    Up to 10 dimension array can be used.                            */
 /*                                                                     */
 /*   Undefined variable:                                               */
 /*    If undefined variable name is found in the expression,G__getexpr()*/
 /*    will leave undefined variable in the formula as it is.           */
 /*                                                                     */
 /*   Undefined signal expression with hex binary expression:           */
 /*    With %[bqohx] expression, undefind(U),hiz(Z) and unknown(X)      */
 /*    simbols can be used. If those simbols are found in expression,   */
 /*    the number is treated as undefined variable. For example, %hZZ   */
 /*    (or %b0X010UZZ) returns string %hZZ (or %b0X010UZZ) as it is.    */
 /*                                                                     */
 /*   Space or Tab character:                                           */
 /*    Space and Tab character should not be included in the expression.*/
 /*    They are recognized as variable name by G__getexpr(). If you want*/
 /*    to use an expression with space and tab, use G__calc().          */
 /*                                                                     */
 /*                                           29 Aug 1990     M.Goto    */
 /***********************************************************************/

#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>

/**************************************************************************
 * variable definition constant.
 **************************************************************************/
#define MEMDEPTH 500
#define VARSIZE  10

/**************************************************************************
 * miscellaneous constant.
 **************************************************************************/
#define MAXPARA  40
#define ONELINE  500
#define MAXNAME  256

/* #define DEBUG */

#ifdef DEBUG
int debug=1;
#endif



/* #define G__IFDEF */

#ifdef G__IFDEF
int resolved=0;
int unresolved=0;
#endif

/**************************************************************************
 * structure for variable buffer
 *
 *  varvaluebuf[varpointer[varlabel[i][0]
 *                         +para[0]
 *                         +para[1]*varlabel[i][1]
 *                         +para[2]*varlabel[i][1]*varlabel[i][2]
 *                                 .
 *                        ]
 *             ]
 **************************************************************************/
struct G__var_array {
  int allvar;
  char varnamebuf[MEMDEPTH][MAXNAME];
  int varlabel[MEMDEPTH][MAXPARA];
  int varpointer[MEMDEPTH*2];
  char varvaluebuf[MEMDEPTH*VARSIZE];
} ;

struct G__var_array G__global ;
struct G__var_array *G__local ;

/**************************************************************************
 * structure for function and array parameter
 *
 **************************************************************************/
struct G__param {
  int paran;
  char parameter[MAXPARA][ONELINE];
  char para[MAXPARA][ONELINE];
};

/**************************************************************************
 * structure for interpleted file pointer
 *
 **************************************************************************/
struct G__input_file {
  FILE *fp;
  int line_number;
};


/**************************************************************************
 * main-functions
 **************************************************************************/
int G__exec_statement G__P((struct G__input_file *fin));
char *G__getexpr G__P((char *expression)); 
char *G__calc G__P((char *exprwithspace)); 
int isexponent G__P((char* expression4,int lenexpr));
int library_func G__P((char* result7,char* funcname,struct G__param* libp));
int G__defined G__P((char* macro));
int G__test G__P((char* expression2));


/**************************************************************************
 * sub-functions
 **************************************************************************/

char *G__letvariable G__P((char *item,char *expression,struct G__var_array *varglobal,struct G__var_array *varlocal));
char *G__getequal G__P((char *expression2));
char *G__getandor G__P((char *expression2));
char *G__getprod G__P((char *expression1));
char *G__getpower G__P((char *expression2));
char *G__getitem G__P((char *item));
char *G__getvalue G__P((char *item,int *known1));
char *G__getvariable G__P((char *item,int *known2,struct G__var_array *varglobal,struct G__var_array *varlocal));
char *G__getfunction G__P((char *item,int *known3));
char *G__checkBase G__P((char *string,int *known4));
char *G__getbase G__P((char *expression,int base,int digit));
char G__getoperator G__P((char newoperator,char oldoperator));
char G__getdigit G__P((int number));
double G__atodouble G__P((char *string));
void G__bstore G__P((char operator,char *expression3,char *defined,char *undefined));
/* int isfloating G__P(()); */
int G__recursive_check G__P((char *varname,char *result7));
void G__varmonitor G__P((struct G__var_array *var));
void G__init_var_array G__P((struct G__var_array *var));
void G__charformatter G__P((char *result, int ifmt, struct G__param *libp));
char *G__strip_quotation G__P((char *string));
char *G__add_quotation G__P((char *string));
void G__error_clear G__P((void));
int G__isvalue G__P((char *temp));
#ifndef G__OLDIMPLEMENTATION1616
int G__fgetc G__P((FILE *fp));
#else
char G__fgetc G__P((FILE *fp));
#endif

/**************************************************************************
 * error flag
 **************************************************************************/
int G__error_flag=0;
int G__new_variable=0;
int G__debug=0;
int G__eof;
int G__no_exec=0;

/***********************************************************************/
/* G__getandor(char *expression)     &&,||                             */
/* G__getequal(char *expression)     ==,!=,<,>,<=,>=                   */
/* G__getexpr(char *expression)      +,-,=                             */
/* G__getprod(char *expression)      *,/,%                             */
/* G__getpower(char *expression)     ^                                 */
/* G__getitem(char *item)             variable,function                */
/* G__getvalue(char *item,int *known)                                  */
/* G__getvariable(char *item,int *known)                               */
/* G__letvariable(char *varname, *expression)                          */
/* G__getfunction(char *item,int *known)                               */
/***********************************************************************/
char *G__getandor(expression2)
     char *expression2;
{
  static char result2[ONELINE];
  char defined2[ONELINE],undefined2[ONELINE];
  char ebuf2[ONELINE];
  char operator2;
  int lenbuf2=0;
  int ig12;
  int length2;
  int nest2=0;
  int single_quote=0,double_quote=0;
  int flag1=0;

#ifdef DEBUG
  if(debug)
    printf("getandor(%s)\n",expression2);
#endif

  operator2='\0';
  defined2[0]='\0';
  undefined2[0]='\0';
  length2=strlen(expression2);
  if(length2==0) return("");
  for(ig12=0;ig12<length2;ig12++) {
    switch(expression2[ig12]) {
    case '"' : /* double quote */
      if(double_quote==0)  double_quote=1;
      else                 double_quote=0;
      ebuf2[lenbuf2]=expression2[ig12];
      ebuf2[++lenbuf2]='\0';
      break;
    case '\'' : /* single quote */
      if(single_quote==0)  single_quote=1;
      else                 single_quote=0;
      ebuf2[lenbuf2]=expression2[ig12];
      ebuf2[++lenbuf2]='\0';
      break;
    case '&':
    case '|':
      if((nest2==0)&&(single_quote==0)&&(double_quote==0)) {
	if(expression2[ig12+1]==expression2[ig12]) {
	  switch(lenbuf2) {
	  case 0:
	    operator2=G__getoperator(operator2
				     ,expression2[ig12]);
	    break;
	  default:
	    G__bstore(operator2,G__getequal(ebuf2)
		      ,defined2,undefined2);
	    flag1=1;
	    lenbuf2=0;
	    ebuf2[0]='\0';
	    operator2=G__getoperator(expression2[ig12+1]
				     ,expression2[ig12]);
	    ig12++;
	    break;
	  }
	}
	else {
	  ebuf2[lenbuf2]=expression2[ig12];
	  ebuf2[++lenbuf2]='\0';
	}
      }
      else {
	ebuf2[lenbuf2]=expression2[ig12];
	ebuf2[++lenbuf2]='\0';
      }
      break;
    case '(':
    case '[':
    case '{':
      nest2++;
      ebuf2[lenbuf2]=expression2[ig12];
      ebuf2[++lenbuf2]='\0';
      break;
    case ')':
    case ']':
    case '}':
      ebuf2[lenbuf2]=expression2[ig12];
      ebuf2[++lenbuf2]='\0';
      nest2--;
      break;
    default :
      ebuf2[lenbuf2]=expression2[ig12];
      ebuf2[++lenbuf2]='\0';
      break;
    }
  }
  if((nest2!=0)||(single_quote!=0)||(double_quote!=0)) {
    if((G__error_flag++)==0)
      fprintf(stderr,"Syntax error: Parenthesis or quotation unmatch %s\n"
	      ,expression2);
  }
  switch(operator2) {
  case 'A':
  case 'O':
    break;
  default:
    operator2='\0';
  }
  G__bstore(operator2,G__getequal(ebuf2)
	    ,defined2,undefined2);
  if(strlen(undefined2)==0) {
    sprintf(result2,"%s",defined2);
  }
  else {
    if(strchr(undefined2,'&')==0&&strchr(undefined2,'|')==0) {
      flag1=0;
    }
    if(strlen(defined2)==0) {
      if(flag1==1) {
	sprintf(result2,"(%s)",undefined2);
      }
      else
	sprintf(result2,"%s",undefined2);
    }
    else {
      if((strcmp(defined2,"0")==0)
	 ||(strcmp(defined2,"0.0")==0))
	sprintf(result2,"%s",defined2);
      else
	sprintf(result2,"(%s%s)",defined2,undefined2);
      /*
	sprintf(result2,"%s%c%s"
	,undefined2+1
	,undefined2[0]
	,defined2);
      */
    }
  }
  return(result2);
}

char *G__getequal(expression2)
     char *expression2;
{
  static char result2[ONELINE];
  char defined2[ONELINE],undefined2[ONELINE];
  char ebuf2[ONELINE];
  char operator2;
  int lenbuf2=0;
  int ig12;
  int length2;
  int nest2=0;
  int single_quote=0,double_quote=0;
  int flag1=0;

#ifdef DEBUG
  if(debug)
    printf("getequal(%s)\n",expression2);
#endif

  operator2='\0';
  defined2[0]='\0';
  undefined2[0]='\0';
  length2=strlen(expression2);
  if(length2==0) return("");
  for(ig12=0;ig12<length2;ig12++) {
    switch(expression2[ig12]) {
    case '"' : /* double quote */
      if(double_quote==0)  double_quote=1;
      else                 double_quote=0;
      ebuf2[lenbuf2]=expression2[ig12];
      ebuf2[++lenbuf2]='\0';
      break;
    case '\'' : /* single quote */
      if(single_quote==0)  single_quote=1;
      else                 single_quote=0;
      ebuf2[lenbuf2]=expression2[ig12];
      ebuf2[++lenbuf2]='\0';
      break;
    case '=':
    case '<':
    case '>':
    case '!':
      if((nest2==0)&&(single_quote==0)&&(double_quote==0)) {
	if(expression2[ig12+1]=='='||expression2[ig12]!='=') {
	  switch(lenbuf2) {
	  case 0:
	    operator2=G__getoperator(operator2
				     ,expression2[ig12]);
	    break;
	  default:
	    G__bstore(operator2,G__getexpr(ebuf2)
		      ,defined2,undefined2);
	    flag1=1;
	    lenbuf2=0;
	    ebuf2[0]='\0';
	    if(expression2[ig12+1]=='=') {
	      operator2=G__getoperator(expression2[ig12+1]
				       ,expression2[ig12]);
	      ig12++;
	    }
	    else {
	      operator2=expression2[ig12];
	    }
	    break;
	  }
	}
	else {
	  ebuf2[lenbuf2]=expression2[ig12];
	  ebuf2[++lenbuf2]='\0';
	}
      }
      else {
	ebuf2[lenbuf2]=expression2[ig12];
	ebuf2[++lenbuf2]='\0';
      }
      break;
    case '(':
    case '[':
    case '{':
      nest2++;
      ebuf2[lenbuf2]=expression2[ig12];
      ebuf2[++lenbuf2]='\0';
      break;
    case ')':
    case ']':
    case '}':
      ebuf2[lenbuf2]=expression2[ig12];
      ebuf2[++lenbuf2]='\0';
      nest2--;
      break;
    default :
      ebuf2[lenbuf2]=expression2[ig12];
      ebuf2[++lenbuf2]='\0';
      break;
    }
  }
  if((nest2!=0)||(single_quote!=0)||(double_quote!=0)) {
    if((G__error_flag++)==0)
      fprintf(stderr,"Syntax error: Parenthesis or quotation unmatch %s\n"
	      ,expression2);
  }
  switch(operator2) {
  case 'E':
  case 'N':
  case 'G':
  case 'L':
  case '<':
  case '>':
  case '!':
    break;
  default:
    operator2='\0';
  }
  G__bstore(operator2,G__getexpr(ebuf2)
	    ,defined2,undefined2);
  if(strlen(undefined2)==0) {
    sprintf(result2,"%s",defined2);
  }
  else {
    if(strlen(defined2)==0) {
      if(flag1==1) {
	sprintf(result2,"(%s)",undefined2);
      }
      else
	sprintf(result2,"%s",undefined2);
    }
    else {
      if((strcmp(defined2,"0")==0)
	 ||(strcmp(defined2,"0.0")==0))
	sprintf(result2,"%s",defined2);
      else
	sprintf(result2,"(%s%s)",defined2,undefined2);
      /*
	sprintf(result2,"%s%c%s"
	,undefined2+1
	,undefined2[0]
	,defined2);
      */
    }
  }
  return(result2);
}

char *G__getexpr(expression)
     char *expression;
{
  static char result[ONELINE];
  char defined[ONELINE],undefined[ONELINE];
  char lbuf[ONELINE],ebuf[ONELINE];
  char operator;
  char prodoperator;
  int lenbuf=0;
  int ig1,ig2;
  int length;
  int nest=0;
  int flag=0;
  int single_quote=0,double_quote=0;

#ifdef DEBUG
  if(debug)
    printf("getexpr(%s)\n",expression);
#endif

  operator='\0';
  prodoperator='\0';
  defined[0]='\0';
  undefined[0]='\0';
  length=strlen(expression);
  if(length==0) return("");

  if(length==0) {
    strcpy(result,"");
    return(result);
  }

  if(expression[0]=='\'') {
    if(expression[length-1]=='\'') {
      expression[length-1]='\0';
    }
    return(expression+1);
  }


  strcpy(lbuf,"");

  for(ig1=0;ig1<length;ig1++) {
    switch(expression[ig1]) {
    case '"' : /* double quote */
      if(double_quote==0)  double_quote=1;
      else                 double_quote=0;
      ebuf[lenbuf]=expression[ig1];
      ebuf[++lenbuf]='\0';
      break;
    case '\'' : /* single quote */
      if(single_quote==0)  single_quote=1;
      else                 single_quote=0;
      ebuf[lenbuf]=expression[ig1];
      ebuf[++lenbuf]='\0';
      break;
    case '=':
      if((nest==0)&&(single_quote==0)&&(double_quote==0)) {
	if((operator!='\0')||(prodoperator!='\0')) {
	  strncpy(lbuf,expression,ig1-1);
	  strncpy(ebuf,expression,ig1);
	  ig2=ig1;
	}
	else {
	  sprintf(lbuf,"%s",ebuf);
	  ig2=0;
	}
	lenbuf=0;
	ig1++;
	/* ebuf[ig2++]='('; */
	while(expression[ig1]!='\0') {
	  ebuf[ig2++]=expression[ig1++] ;
	}
	/* ebuf[ig2++]=')'; */
	ebuf[ig2]='\0';
	return(G__letvariable(lbuf,ebuf,&G__global,G__local));
      }
      else {
	ebuf[lenbuf]=expression[ig1];
	ebuf[++lenbuf]='\0';
      }
      break;
    case '+':
    case '-':
    case '&':
    case '|':
      if((nest==0)&&(single_quote==0)&&(double_quote==0)) {
	switch(lenbuf) {
	case 0:
	  operator=G__getoperator(operator
				  ,expression[ig1]);
	  break;
	case 1:
	  if(operator=='\0') {
	    if(expression[ig1]=='-')
	      operator='+';
	    else
	      operator=expression[ig1];
	  }
	  G__bstore(operator,G__getitem(ebuf)
		    ,defined,undefined);
	  flag=1;
	  lenbuf=0;
	  ebuf[0]='\0';
	  operator=expression[ig1];
	  break;
	default:
	  if(isexponent(ebuf,lenbuf)==1) {
	    ebuf[lenbuf]=expression[ig1];
	    ebuf[++lenbuf]='\0';
	  }
	  else {
	    if(operator=='\0') {
	      if(expression[ig1]=='-')
		operator='+';
	      else
		operator=expression[ig1];
	    }
	    G__bstore(operator,G__getprod(ebuf)
		      ,defined,undefined);
	    flag=1;
	    lenbuf=0;
	    ebuf[0]='\0';
	    operator=expression[ig1];
	  }
	  break;
	}
      }
      else {
	ebuf[lenbuf]=expression[ig1];
	ebuf[++lenbuf]='\0';
      }
      break;
    case '(':
    case '[':
    case '{':
      nest++;
      ebuf[lenbuf]=expression[ig1];
      ebuf[++lenbuf]='\0';
      break;
    case ')':
    case ']':
    case '}':
      ebuf[lenbuf]=expression[ig1];
      ebuf[++lenbuf]='\0';
      nest--;
      break;
    case '*':
      if('='==expression[ig1+1] &&
	 ('0'==expression[ig1+2]||'1'==expression[ig1+2]) &&
	 0==expression[ig1+3]) {
	prodoperator = '\0';
	ebuf[lenbuf]=expression[ig1];
	ebuf[++lenbuf]='\0';
	break;
      }
    case '/':
    case '%':
    case '^':
      if((nest==0)&&(single_quote==0)&&(double_quote==0)) {
	prodoperator = expression[ig1] ;
      }
      ebuf[lenbuf]=expression[ig1];
      ebuf[++lenbuf]='\0';
      break;
    default :
      prodoperator = '\0';
      ebuf[lenbuf]=expression[ig1];
      ebuf[++lenbuf]='\0';
      break;
    }
  }
  if((nest!=0)||(single_quote!=0)||(double_quote!=0)) {
    if((G__error_flag++)==0)
      fprintf(stderr,"Syntax error: Parenthesis or quotation unmatch %s\n"
	      ,expression);
  }
  if((operator=='I')||(operator=='D')) {
    /* increment ++ or decrement -- operator */
    if(strcmp(ebuf,"")==0) {
      /* a++ */
      strncpy(lbuf,expression,length-2);
      lbuf[length-2]='\0';
      if(operator=='I') 
	sprintf(ebuf,"%s+1",lbuf);
      if(operator=='D') 
	sprintf(ebuf,"%s-1",lbuf);
      G__letvariable(lbuf,ebuf,&G__global,G__local);
    }
    else {
      /* ++a */
      strcpy(lbuf,ebuf);
      if(operator=='I') 
	sprintf(ebuf,"%s+1",lbuf);
      if(operator=='D') 
	sprintf(ebuf,"%s-1",lbuf);
      return(G__letvariable(lbuf,ebuf,&G__global,G__local));
    }
  }
  else {
    G__bstore(operator,G__getprod(ebuf)
	      ,defined,undefined);
  }

  if(strlen(undefined)==0) {
    sprintf(result,"%s",defined);
  }
  else {
    if(strlen(defined)==0) {
      if(flag==1) {
	undefined[0]='(';
	sprintf(result,"%s)",undefined);
      }
      else
	sprintf(result,"%s",undefined);
    }
    else {
      if((strcmp(defined,"0")==0)
	 ||(strcmp(defined,"0.0")==0))
	sprintf(result,"(%s)",undefined);
      else
	sprintf(result,"(%s%s)"
		,defined,undefined);
    }
  }
  return(result);
}

char *G__getprod(expression1)
     char *expression1;
{
  static char result1[ONELINE];
  char defined1[ONELINE],undefined1[ONELINE];
  char ebuf1[ONELINE];
  char operator1;
  int lenbuf1=0;
  int ig11;
  int length1;
  int nest1=0;
  int flag1=0;
  int single_quote=0,double_quote=0;

#ifdef DEBUG
  if(debug)
    printf("getprod(%s)\n",expression1);
#endif

  switch(expression1[0]) {
  case '*': /* value of pointer */
    /* if variable is defined as a pointer, name of it will be ~name */
    expression1[0] = '~';
    break;
  case '&': /* pointer */
    /* do nothing because this is same as 1&expression=expression */
    break;
  default :
    break;
  }

  operator1='\0';
  defined1[0]='\0';
  undefined1[0]='\0';
  length1=strlen(expression1);
  if(length1==0) return("");
  for(ig11=0;ig11<length1;ig11++) {
    switch(expression1[ig11]) {
    case '"' : /* double quote */
      if(double_quote==0)  double_quote=1;
      else                 double_quote=0;
      ebuf1[lenbuf1]=expression1[ig11];
      ebuf1[++lenbuf1]='\0';
      break;
    case '\'' : /* single quote */
      if(single_quote==0)  single_quote=1;
      else                 single_quote=0;
      ebuf1[lenbuf1]=expression1[ig11];
      ebuf1[++lenbuf1]='\0';
      break;
    case '*':
    case '/':
    case '%':
      if((nest1==0)&&(single_quote==0)&&(double_quote==0)) {
	switch(lenbuf1) {
	case 0:
	  if(expression1[ig11]=='%') {
	    ebuf1[lenbuf1]=
	      expression1[ig11];
	    ebuf1[++lenbuf1]='\0';
	  }
	  else {
	    operator1=G__getoperator(operator1
				     ,expression1[ig11]);
	  }
	  break;
	default:
	  if(operator1=='\0') operator1='*';
	  G__bstore(operator1,G__getpower(ebuf1)
		    ,defined1,undefined1);
	  flag1=1;
	  lenbuf1=0;
	  ebuf1[0]='\0';
	  operator1=expression1[ig11];
	  break;
	}
      }
      else {
	ebuf1[lenbuf1]=expression1[ig11];
	ebuf1[++lenbuf1]='\0';
      }
      break;
    case '(':
    case '[':
    case '{':
      nest1++;
      ebuf1[lenbuf1]=expression1[ig11];
      ebuf1[++lenbuf1]='\0';
      break;
    case ')':
    case ']':
    case '}':
      ebuf1[lenbuf1]=expression1[ig11];
      ebuf1[++lenbuf1]='\0';
      nest1--;
      break;
    default :
      ebuf1[lenbuf1]=expression1[ig11];
      ebuf1[++lenbuf1]='\0';
      break;
    }
  }
  if((nest1!=0)||(single_quote!=0)||(double_quote!=0)) {
    if((G__error_flag++)==0)
      fprintf(stderr,"Syntax error: Parenthesis or quotation unmatch %s\n"
	      ,expression1);
  }
  G__bstore(operator1,G__getpower(ebuf1)
	    ,defined1,undefined1);
  if(strlen(undefined1)==0) {
    sprintf(result1,"%s",defined1);
  }
  else {
    if(strlen(defined1)==0) {
      if(flag1==1) {
	undefined1[0]='(';
	sprintf(result1,"%s)",undefined1);
	/*
	  sprintf(result1,"%s",undefined1+1);
	*/
      }
      else
	sprintf(result1,"%s",undefined1);
    }
    else {
      if((strcmp(defined1,"0")==0)
	 ||(strcmp(defined1,"0.0")==0))
	sprintf(result1,"%s",defined1);
      else
	sprintf(result1,"(%s%s)",defined1,undefined1);
      /*
	sprintf(result1,"%s%c%s"
	,undefined1+1
	,undefined1[0]
	,defined1);
      */
    }
  }
  switch(result1[0]) {
  case '~': /* value of pointer */
    /* if variable is defined as a pointer, name of it will be *name */
    result1[0] = '*';
    break;
  default :
    break;
  }
  return(result1);
}

char *G__getpower(expression2)
     char *expression2;
{
  static char result2[ONELINE];
  char defined2[ONELINE],undefined2[ONELINE];
  char ebuf2[ONELINE];
  char operator2;
  int lenbuf2=0;
  int ig12;
  int length2;
  int nest2=0;
  int single_quote=0,double_quote=0;
  int flag1=0;

#ifdef DEBUG
  if(debug)
    printf("getpower(%s)\n",expression2);
#endif

  operator2='\0';
  defined2[0]='\0';
  undefined2[0]='\0';
  length2=strlen(expression2);
  if(length2==0) return("");
  for(ig12=0;ig12<length2;ig12++) {
    switch(expression2[ig12]) {
    case '"' : /* double quote */
      if(double_quote==0)  double_quote=1;
      else                 double_quote=0;
      ebuf2[lenbuf2]=expression2[ig12];
      ebuf2[++lenbuf2]='\0';
      break;
    case '\'' : /* single quote */
      if(single_quote==0)  single_quote=1;
      else                 single_quote=0;
      ebuf2[lenbuf2]=expression2[ig12];
      ebuf2[++lenbuf2]='\0';
      break;
    case '^':
      if((nest2==0)&&(single_quote==0)&&(double_quote==0)) {
	switch(lenbuf2) {
	case 0:
	  operator2=G__getoperator(operator2
				   ,expression2[ig12]);
	  break;
	default:
	  G__bstore(operator2,G__getitem(ebuf2)
		    ,defined2,undefined2);
	  flag1=1;
	  lenbuf2=0;
	  ebuf2[0]='\0';
	  operator2=expression2[ig12];
	  break;
	}
      }
      else {
	ebuf2[lenbuf2]=expression2[ig12];
	ebuf2[++lenbuf2]='\0';
      }
      break;
    case '(':
    case '[':
    case '{':
      nest2++;
      ebuf2[lenbuf2]=expression2[ig12];
      ebuf2[++lenbuf2]='\0';
      break;
    case ')':
    case ']':
    case '}':
      ebuf2[lenbuf2]=expression2[ig12];
      ebuf2[++lenbuf2]='\0';
      nest2--;
      break;
    default :
      ebuf2[lenbuf2]=expression2[ig12];
      ebuf2[++lenbuf2]='\0';
      break;
    }
  }
  if((nest2!=0)||(single_quote!=0)||(double_quote!=0)) {
    if((G__error_flag++)==0)
      fprintf(stderr,"Syntax error: Parenthesis or quotation unmatch %s\n"
	      ,expression2);
  }
  G__bstore(operator2,G__getitem(ebuf2)
	    ,defined2,undefined2);
  if(strlen(undefined2)==0) {
    sprintf(result2,"%s",defined2);
  }
  else {
    if(strlen(defined2)==0) {
      if(flag1==1) 
	sprintf(result2,"(%s)",undefined2);
      else
	sprintf(result2,"%s",undefined2);
    }
    else {
      if(undefined2[0]=='^') {
	sprintf(result2,"(%s%s)"
		,defined2,undefined2);
	if((strcmp(defined2,"1")==0)
	   ||(strcmp(defined2,"1.0")==0))
	  sprintf(result2,"%s"
		  ,defined2);
      }
      else {
	sprintf(result2,"(%s%s)"
		,undefined2,defined2);
	if(strcmp(defined2,"^0")==0)
	  sprintf(result2,"1");
	if(strcmp(defined2,"^0.0")==0)
	  sprintf(result2,"1.0");
	if((strcmp(defined2,"^1")==0)
	   ||(strcmp(defined2,"^1.0")==0))
	  sprintf(result2,"%s"
		  ,undefined2);
      }
    }
  }
  return(result2);
}

char *G__getitem(item)
     char *item;
{
  int known=0;
  static char result3[ONELINE];

#ifdef DEBUG
  if(debug)
    printf("getitem(%s)\n",item);
#endif


  sprintf(result3,"%s ",G__getvalue(item,&known));
  if(known==0) {
    strcpy(result3,G__getvariable(item,&known,&G__global,G__local));
#ifdef G__IFDEF
    if(known) {resolved++;}
#endif
  }
  if(known==0) {
    strcpy(result3,G__getfunction(item,&known));
#ifdef G__IFDEF
    if(known) {
      /* resolved++;*/
    }
#endif
  }
  if(known==0) {
    strcpy(result3,item);
#ifdef G__IFDEF
    unresolved++;
#endif
  }


  return(result3);
}

char *G__getvalue(item,known1)
     char *item;
     int *known1;
{
  if(  (strlen(item)>2)
       &&((item[0]=='%')||(item[0]=='0'))
       &&((isdigit(item[1])==0)&&(item[1]!='.'))  ) {
    return(G__checkBase(item,known1));
  }
  else {
    if((isdigit(item[0]))
       ||(item[0]=='0')
       ||(item[0]=='.')) {
      *known1=1;
      return(item);
    }
    else {
      return(item);
    }
  }
}

char *G__letvariable(item,expression,varglobal,varlocal)
     char *item;
     char *expression;
     struct G__var_array *varglobal,*varlocal;
{
  struct G__var_array *var;
  static char result[ONELINE];
  char varname[MAXNAME];
  char parameter[MAXPARA][ONELINE];
  char para[MAXPARA][ONELINE];
  char result7[ONELINE];
  int ig15=0,paran=0,ig35=0,ig25,ig45,ig55,ary;
  int lenitem,nest=0;
  int newpointer,nextpointer,endpointer;
  int digit=0;
  char basecom[5];
  int base=10;
  int single_quote=0,double_quote=0;
  int done=0;

  switch(item[0]) {
  case '*': /* value of pointer */
    /* if variable is defined as a pointer, name of it will be ~name */
    item[0] = '~';
    break;
  case '&': /* pointer */
    /* do nothing because this is same as 1&expression=expression */
    break;
  default :
    break;
  }

  lenitem=strlen(item);

  /* Check base expression */
  if((item[ig15]=='%')||(item[ig15]=='0')) {
    ig15++;
    while(isdigit(item[ig15])) {
      digit = item[ig15] - '0' + digit*10;
      ig15++;
    }
    switch(item[ig15]) {
    case 'b':
    case 'B':
      sprintf(basecom,"bin");
      base = 2;
      break;
    case 'q':
    case 'Q':
      sprintf(basecom,"quad");
      base = 4;
      break;
    case 'o':
    case 'O':
      sprintf(basecom,"oct");
      base = 8;
      break;
    case 'h':
    case 'H':
    case 'x':
    case 'X':
      sprintf(basecom,"hex");
      base = 16;
      break;
    default:
      base = 10;
      break;
    }
    ig15++;
  }

  if(ig15!=0) {
    /* sprintf(result,"%s(%s,%d)",basecom,expression,digit);
       strcpy(result7,G__getexpr(result)); */
    strcpy(result7,G__getbase(expression,base,digit));
  }
  else
    strcpy(result7,G__getexpr(expression));

  /* Separate variable name */
  while((item[ig15]!='(')&&(item[ig15]!='[')&&(ig15<lenitem)) {
    varname[ig15]=item[ig15];
    ig15++;
  }

  if(item[ig15]=='(') {
    /* if 'func(xxxx)' return */
    if((G__error_flag++)==0) {
      fprintf(stderr,"Error: Trying to change function %s\n",item);
    }
    strcpy(result,result7);
    return(result);
  }

  varname[ig15++]='\0';

  if(ig15==1) {
    /* if '[xxxx]' return */
    if((G__error_flag++)==0) {
      fprintf(stderr,"Error: expression %s\n",item);
    }
    strcpy(result,result7);
    return(result);
  }

  /* Get Parenthesis */
  if(ig15<lenitem) {
    while((item[ig15]!='!')&&(ig15<lenitem)) {
      nest=0;
      single_quote=0;
      double_quote=0;
      while(((item[ig15]!=']')||(nest>0)
	     ||(single_quote>0)||(double_quote>0))
	    &&(ig15<lenitem)) {
	switch(item[ig15]) {
	case '"' : /* double quote */
	  if(double_quote==0)  double_quote=1;
	  else                 double_quote=0;
	  break;
	case '\'' : /* single quote */
	  if(single_quote==0)  single_quote=1;
	  else                 single_quote=0;
	  break;
	case '(':
	case '[':
	case '{':
	  nest++;
	  break;
	case ')':
	case ']':
	case '}':
	  nest--;
	  break;
	}
	parameter[paran][ig35++]=item[ig15++];
      }
      ig15++;
      if((item[ig15]=='[')&&(ig15<lenitem)) ig15++;
      parameter[paran++][ig35]='\0';
      parameter[paran][0]='\0';
      ig35 = 0;
    }
  }

  /* Evaluate parameter */
  for(ig15=0;ig15<paran;ig15++) {
    strcpy(para[ig15],G__getexpr(parameter[ig15]));
  }

  if(varlocal!=NULL) {
    var=varlocal;

    /* Searching for variable name */
    ig15=0;
    while((ig15<(var->allvar))&&
	  (strcmp(varname,var->varnamebuf[ig15])!=0)) ig15++;

    /* Let value */
    if(ig15<(var->allvar)) {
      /* old variable */
      done++;

      /* Get start pointer of the variable */
      ig35= var->varlabel[ig15][0];
      ary=1;
      for(ig25=0;ig25<paran;ig25++) {
	if(ig25!=0) ary = ary*((var->varlabel[ig15][ig25])+1);
	ig35 += atoi(para[ig25]) * ary ;
      }
      newpointer = var->varpointer[ig35];
      nextpointer = var->varpointer[ig35+1];
      endpointer = var->varpointer[var->varlabel[var->allvar][0]];

      ig25 = MEMDEPTH*VARSIZE ;
      for(ig45=endpointer;nextpointer <= ig45;ig45--) {
	var->varvaluebuf[--ig25]=var->varvaluebuf[ig45];
      }

      ig45=0;
      while(result7[ig45]!='\0') {
	var->varvaluebuf[newpointer++]=result7[ig45++];
      }
      var->varvaluebuf[newpointer++]='\0';

      ig55= var->varlabel[var->allvar][0];
      for(ig45=ig35+1;ig45<=ig55;ig45++) {
	var->varpointer[ig45] += newpointer-nextpointer;
      }

      ig45=newpointer;
      for(ig35=ig25;ig35<MEMDEPTH*VARSIZE;ig35++) {
	var->varvaluebuf[ig45++]=var->varvaluebuf[ig35];
      }

    }
  }
  if(done==0) {
    var=varglobal;

    /* Searching for variable name */
    ig15=0;
    while((ig15<(var->allvar))&&(strcmp(varname,var->varnamebuf[ig15])!=0)) ig15++;

    /* Let value */
    if(ig15<(var->allvar)) {
      /* old local variable */
      done++;

      /* Get start pointer of the variable */
      ig35= var->varlabel[ig15][0];
      ary=1;
      for(ig25=0;ig25<paran;ig25++) {
	if(ig25!=0) ary = ary*((var->varlabel[ig15][ig25])+1);
	ig35 += atoi(para[ig25]) * ary ;
      }
      newpointer = var->varpointer[ig35];
      nextpointer = var->varpointer[ig35+1];
      endpointer = var->varpointer[var->varlabel[var->allvar][0]];

      ig25 = MEMDEPTH*VARSIZE ;
      for(ig45=endpointer;nextpointer <= ig45;ig45--) {
	var->varvaluebuf[--ig25]=var->varvaluebuf[ig45];
      }

      ig45=0;
      while(result7[ig45]!='\0') {
	var->varvaluebuf[newpointer++]=result7[ig45++];
      }
      var->varvaluebuf[newpointer++]='\0';

      ig55= var->varlabel[var->allvar][0];
      for(ig45=ig35+1;ig45<=ig55;ig45++) {
	var->varpointer[ig45] += newpointer-nextpointer;
      }

      ig45=newpointer;
      for(ig35=ig25;ig35<MEMDEPTH*VARSIZE;ig35++) {
	var->varvaluebuf[ig45++]=var->varvaluebuf[ig35];
      }

    }
  }

  if(done==0) {
    if(G__local!=NULL) var=varlocal;
    else            var=varglobal;

    /* new variable */
    G__new_variable++;

    for(ig25=0;ig25<paran;ig25++) {
      var->varlabel[var->allvar][ig25+1]=atoi(para[ig25]);
    }
    ig15 = var->allvar;

    ig35= var->varlabel[ig15][0];
    ary=1;
    for(ig25=0;ig25<paran;ig25++) {
      if(ig25!=0) ary = ary*(var->varlabel[ig15][ig25]+1);
      ig35 += atoi(para[ig25]) * ary ;
    }
    for(ig25= var->varlabel[var->allvar][0]+1;ig25<=ig35;ig25++) {
      var->varpointer[ig25] = var->varpointer[ig25-1]+1;
    }
    newpointer = var->varpointer[ig35];
    var->varlabel[(var->allvar)+1][0]=ig35+1;

    for(ig45=var->varpointer[var->varlabel[var->allvar][0]];ig45<=newpointer;ig45++) {
      var->varvaluebuf[ig45]='\0';
    }

    ig45=0;
    while(result7[ig45]!='\0') {
      var->varvaluebuf[newpointer++]=result7[ig45++];
    }
    var->varvaluebuf[newpointer++]='\0';
    sprintf(var->varnamebuf[var->allvar],"%s",varname);
    var->varpointer[var->varlabel[++(var->allvar)][0]]=newpointer;
  }

  strcpy(result,result7);
  return(result);
}

char *G__getvariable(item,known2,varglobal,varlocal)
     char *item;
     int *known2;
     struct G__var_array *varglobal,*varlocal;
{
  struct G__var_array *var;
  char varname[MAXNAME];
  char parameter[MAXPARA][ONELINE];
  char para[MAXPARA][ONELINE];
  static char result7[ONELINE];
  int ig15=0,paran=0,ig35=0,ig25,ary;
  int lenitem,nest=0;
  int single_quote=0,double_quote=0;
  int done=0;

  if(item[0]=='"') return("");

  lenitem=strlen(item);

  /* Separate variable name */
  while((item[ig15]!='(')&&(item[ig15]!='[')&&(ig15<lenitem)) {
    varname[ig15]=item[ig15];
    ig15++;
  }

  if(item[ig15]=='(') {
    /* if 'func(xxxx)' return */
    return("");
  }

  varname[ig15++]='\0';

  if(ig15==1) {
    /* if '[xxxx]' return */
    if((G__error_flag++)==0) {
      fprintf(stderr,"Error: expression %s\n",item);
    }
    *known2=1;
    return("");
  }

  /* Get Parenthesis */
  if(ig15<lenitem) {
    while((item[ig15]!='!')&&(ig15<lenitem)) {
      nest=0;
      single_quote=0;
      double_quote=0;
      while(((item[ig15]!=']')||(nest>0)
	     ||(single_quote>0)||(double_quote>0))
	    &&(ig15<lenitem)) {
	switch(item[ig15]) {
	case '"' : /* double quote */
	  if(double_quote==0)  double_quote=1;
	  else                 double_quote=0;
	  break;
	case '\'' : /* single quote */
	  if(single_quote==0)  single_quote=1;
	  else                 single_quote=0;
	  break;
	case '(':
	case '[':
	case '{':
	  nest++;
	  break;
	case ')':
	case ']':
	case '}':
	  nest--;
	  break;
	}
	parameter[paran][ig35++]=item[ig15++];
      }
      ig15++;
      if((item[ig15]=='[')&&(ig15<lenitem)) ig15++;
      parameter[paran++][ig35]='\0';
      parameter[paran][0]='\0';
      ig35 = 0;
    }
  }

  /* Evaluate parameter */
  for(ig15=0;ig15<paran;ig15++) {
    strcpy(para[ig15],G__getexpr(parameter[ig15]));
  }

  if(varlocal!=NULL) {
    var=varlocal;

    /* Searching for variable name */
    ig15=0;
    while((ig15<(var->allvar))&&(strcmp(varname,var->varnamebuf[ig15])!=0)) ig15++;

    /* Get value if defined*/
    if(ig15<(var->allvar)) {

      done++;

      /* Get start pointer of the variable */
      ig35= var->varlabel[ig15][0];
      ary=1;
      for(ig25=0;ig25<paran;ig25++) {
	if(ig25!=0) ary = ary*(var->varlabel[ig15][ig25]+1);
	ig35 += atoi(para[ig25]) * ary ;
      }
      ig35 = var->varpointer[ig35];

      /* Copy value to result */
      ig25=0;
      while(var->varvaluebuf[ig35]!='\0') {
	result7[ig25++]=var->varvaluebuf[ig35++];
      }
      result7[ig25]='\0';

      if((strcmp(result7,"?")!=0)&&(strcmp(result7,"")!=0)) {
	*known2=1;
	if(result7[0]=='"') return(result7);
	if(G__recursive_check(varname,result7)==0) {
	  return(G__getexpr(result7));
	}
	else {
	  if((G__error_flag++)==0) {
	    fprintf(stderr
		    ,"Warning: Recursive definition %s=%s\n"
		    ,item,result7);
	  }
	  return(result7);
	}
      }
      else {
	return(item);
      }
    }
  }

  if(done==0) {
    var=varglobal;

    /* Searching for variable name */
    ig15=0;
    while((ig15<(var->allvar))&&(strcmp(varname,var->varnamebuf[ig15])!=0)) ig15++;

    /* Get value if defined*/
    if(ig15<(var->allvar)) {

      done++;

      /* Get start pointer of the variable */
      ig35= var->varlabel[ig15][0];
      ary=1;
      for(ig25=0;ig25<paran;ig25++) {
	if(ig25!=0) ary = ary*(var->varlabel[ig15][ig25]+1);
	ig35 += atoi(para[ig25]) * ary ;
      }
      ig35 = var->varpointer[ig35];

      /* Copy value to result */
      ig25=0;
      while(var->varvaluebuf[ig35]!='\0') {
	result7[ig25++]=var->varvaluebuf[ig35++];
      }
      result7[ig25]='\0';

      if((strcmp(result7,"?")!=0)&&(strcmp(result7,"")!=0)) {
	if(*known2==10) {
	  return(result7);
	}
	*known2=1;
	if(result7[0]=='"') return(result7);
	if(G__recursive_check(varname,result7)==0) {
	  return(G__getexpr(result7));
	}
	else {
	  if((G__error_flag++)==0) {
	    fprintf(stderr
		    ,"Warning: Recursive definition %s=%s\n"
		    ,item,result7);
	  }
	  return(result7);
	}
      }
      else {
	return(item);
      }
    }
    return(item);
  }

  return("");

}

char *G__getfunction(item,known3)
     char *item;
     int *known3;
{
  static char result[ONELINE];
  char funcname[MAXNAME];
  char result7[ONELINE];
  int ig15=0,ig35=0,ipara;
  int lenitem,nest=0;
  int single_quote=0,double_quote=0;
  struct G__param fpara;
  
  if(item[0]=='"') return("");
  
  lenitem=strlen(item);
  
  /* Separate function name */
  while((item[ig15]!='(')&&(ig15<lenitem)) {
    funcname[ig15]=item[ig15];
    ig15++;
  }
  if(item[ig15]!='(') {
    /* if no parenthesis , this is not a function */
    return(item);
  }
  funcname[ig15++]='\0';
  
  fpara.paran=0;
  
  /* Get Parenthesis */
  if(ig15<lenitem) {
    while(/*(item[ig15]!='!')&& */(ig15<lenitem)) { /* ??? */
      nest=0;
      single_quote=0;
      double_quote=0;
      while((((item[ig15]!=',')
	      &&(item[ig15]!=')'))
	     ||(nest>0)||(single_quote>0)||(double_quote>0))
	    &&(ig15<lenitem)) {
	switch(item[ig15]) {
	case '"' : /* double quote */
	  if(double_quote==0)  double_quote=1;
	  else                 double_quote=0;
	  break;
	case '\'' : /* single quote */
	  if(single_quote==0)  single_quote=1;
	  else                 single_quote=0;
	  break;
	case '(':
	case '[':
	case '{':
	  nest++;
	  break;
	case ')':
	case ']':
	case '}':
	  nest--;
	  break;
	}
	fpara.parameter[fpara.paran][ig35++]=item[ig15++];
      }
      ig15++;
      fpara.parameter[fpara.paran++][ig35]='\0';
      fpara.parameter[fpara.paran][0]='\0';
      ig35 = 0;
    }
  }
  
  if(strlen(fpara.parameter[0])==0) fpara.paran=0;
  
  /* Evaluate parameters  parameter:string expression , 
     para:evaluated expression */
  /* int *a;  func(a)      
     parameter='a'         para='a'   don't use this expression*/
  /* int *a;  func(*a)     
     parameter='*a'        para=value don't change value */
  /* int  a;  func(a)      
     parameter='a'         para=value don't change value */
  /* int  a;  func(&a)     
     parameter='&a'        para=value */
  for(ig15=0;ig15< fpara.paran;ig15++) {
    /* strcpy(fpara.para[ig15],G__getexpr(fpara.parameter[ig15])); */
    strcpy(fpara.para[ig15],G__getandor(fpara.parameter[ig15]));
  }

  if(strcmp(funcname,"")==0) {
    /* (expression) */
    strcpy(result,fpara.para[0]);
    *known3 = 1;
    return(result);
  }
  
  sprintf(result7,"%s(",funcname);
  for(ipara=1;ipara<= fpara.paran;ipara++) {
    if(ipara!=1) sprintf(result7,"%s,",result7);
    sprintf(result7,"%s%s",result7,fpara.para[ipara-1]);
  }
  strcat(result7,")");
  
  /*
    if( interpret_func(result7,funcname,&fpara)==1 ) {
    return( result7 );
    }
    if( compiled_func(result7,funcname,&fpara)==1 ) {
    return( result7 );
    }
  */
  if( library_func(result7,funcname,&fpara)==1 ) {
    strcpy(result,result7);
    *known3 = 1;
    return( result );
  }
  strcpy(result,result7);
  return(result);
  
}

/***********************************************************************/
/* G__atodouble(char *string)        1.5e-12, 1T,1G,1M,1K,1m,1u,1n,1p,1a  */
/* G__checkBase(char *string,int known)  %b,%q,%o,%h,%x,0b,0q,0o,0x,0h    */
/*                                    with %[b|q|o|h|x] expression,    */
/*                                    undefined(U)|hiz(Z)|unknown(X)   */
/*                                    can be used like %hZZ,%bUUUX.    */
/*                                    with 0[b|q|o|x|h] expression,    */
/*                                    this kind of expression is not   */
/*                                    available.                       */
/***********************************************************************/

double G__atodouble(string)
     char *string;
{
  int lenstring;
  int ig16=0,ig26=0;
  char exponent[ONELINE];
  double expo,polarity=1;
  double result5=0.0;
  double ratio=0.1;

  lenstring=strlen(string);
  if(string[ig16]=='-') {
    polarity = -1;
    ig16++;
  }
  while((isdigit(string[ig16]))&&(ig16<lenstring)) {
    result5 = result5*10 + (double)(string[ig16++]-'0');
  }
  if(string[ig16]=='.') {
    ig16++;
    while((isdigit(string[ig16]))&&(ig16<lenstring)) {
      result5 += ratio * (double)(string[ig16++]-'0');
      ratio /= 10;
    }
  }
  if(ig16<lenstring) {
    switch(string[ig16]) {
    case 'e':
    case 'E':
      ig16++;
      while(ig16<lenstring) exponent[ig26++]=string[ig16++];
      exponent[ig26]='\0';
      expo = (double)(atoi(exponent));
      expo = exp(expo*log(10.0));
      result5 *= expo;
      break;
    case 't':
    case 'T':
      result5 *= 1e12;
      break;
    case 'g':
    case 'G':
      result5 *= 1e9;
      break;
    case 'M':
      result5 *= 1e6;
      break;
    case 'k':
    case 'K':
      result5 *= 1e3;
      break;
    case 'm':
      result5 *= 1e-3;
      break;
    case 'u':
    case 'U':
      result5 *= 1e-6;
      break;
    case 'n':
    case 'N':
      result5 *= 1e-9;
      break;
    case 'p':
    case 'P':
      result5 *= 1e-12;
      break;
    case 'f':
    case 'F':
      result5 *= 1e-15;
      break;
    case 'a':
    case 'A':
      result5 *= 1e-18;
      break;
    }
  }
  return(polarity*result5);
}

char *G__getbase(expression,base,digit)
     char *expression;
     int base,digit;
{
  static char result[ONELINE];
  char result1[ONELINE];
  int ig18=0,ig28=0;
  int value,k,onedig;

  if(expression[0]=='-') {
    result1[ig18++]='-';
    expression[0]=' ';
  }

  value = atoi(G__getexpr(expression));

  k = base;

  while((ig28<digit)||((digit==0)&&(value!=0))) {
    onedig = value % base ;
    result[ig28] = G__getdigit(onedig);
    value = (value - onedig)/base;
    k *= base ;
    ig28++ ;
  }
  ig28-- ;

  result1[ig18++]='%' ;
  /* result1[ig18++]='0' ; */
  switch(base) {
  case 2:
    result1[ig18++]='b' ;
    break;
  case 4:
    result1[ig18++]='q' ;
    break;
  case 8:
    result1[ig18++]='o' ;
    break;
  case 10:
    result1[ig18++]='d' ;
    break;
  case 16:
    result1[ig18++]='h' ;
    /* result1[ig18++]='x' ; */
    break;
  default:
    result1[ig18++]= base + '0' ;
  }

  while(0<=ig28) {
    result1[ig18++] = result[ig28--] ;
  }
  result1[ig18]='\0';

  strcpy(result,result1);
  return(result);
}

char G__getdigit(number)
     int number;
{
  switch(number) {
  case 0:
  case 1:
  case 2:
  case 3:
  case 4:
  case 5:
  case 6:
  case 7:
  case 8:
  case 9:
    return( number + '0' );
  case 10:
    return('a');
  case 11:
    return('b');
  case 12:
    return('c');
  case 13:
    return('d');
  case 14:
    return('e');
  case 15:
    return('f');
  default:
    return('x');
  }
}

char *G__checkBase(string,known4)
     char *string;
     int *known4;
{
  static char result4[ONELINE];
  int n=0,nchar,base;
  long int value = 0;

  nchar = strlen(string);
  while(n<nchar) 
    {
      if((string[0]!='%')&&(string[0]!='0'))
	{
	  if((G__error_flag++)==0)
	    fprintf(stderr,"Error: G__checkBase(%s)\n",string);
	}
      else
	{ 
	  switch(string[++n])
	    {
	    case 'b':
	    case 'B':
	      base=2;
	      break;
	    case 'q':
	    case 'Q':
	      base=4;
	      break;
	    case 'o':
	    case 'O':
	      base=8;
	      break;
	    case 'h':
	    case 'H':
	    case 'x':
	    case 'X':
	      base=16;
	      break;
	    default:
	      base=10;
	      break;
	    }
	  value=0;
	  while((string[++n]!=' ')&&(string[n]!='	')&&(n<nchar))
	    {
	      switch(string[n])
		{
		case '0':
		case 'L':
		case 'l':
		  value=value*base;
		  break;
		case '1':
		case 'H':
		case 'h':
		  value=value*base+1;
		  break;
		case '2':
		  value=value*base+2;
		  break;
		case '3':
		  value=value*base+3;
		  break;
		case '4':
		  value=value*base+4;
		  break;
		case '5':
		  value=value*base+5;
		  break;
		case '6':
		  value=value*base+6;
		  break;
		case '7':
		  value=value*base+7;
		  break;
		case '8':
		  value=value*base+8;
		  break;
		case '9':
		  value=value*base+9;
		  break;
		case 'a':
		case 'A':
		  value=value*base+10;
		  break;
		case 'b':
		case 'B':
		  value=value*base+11;
		  break;
		case 'c':
		case 'C':
		  value=value*base+12;
		  break;
		case 'd':
		case 'D':
		  value=value*base+13;
		  break;
		case 'e':
		case 'E':
		  value=value*base+14;
		  break;
		case 'f':
		case 'F':
		  value=value*base+15;
		  break;
		case 'x':
		case 'X':
		case 'u':
		case 'U':
		case 'z':
		case 'Z':
		case '*':
		  value = -10000;
		  break;
		}
	    }
	}
    }
  if(value>-1) {
    *known4=1;
  }
  sprintf(result4,"%ld",value);
  return(result4);
}

/***********************************************************************/
/* G__bstore(char operator,*expression,*defined,*undefined)               */
/***********************************************************************/

void G__bstore(operator,expression3,defined,undefined)
     char operator;
     char *expression3;
     char *defined,*undefined;
{
  int ig1=0,ig2;
  int lenexp;
  int lendef;
  char expressionin[ONELINE];
  long int ldefined,lexpression;
  double fdefined,fexpression;


#ifdef DEBUG
  if(debug)
    printf("defined=%s undefined=%s %c expression=%s\n",defined,undefined,operator,expression3);
#endif

  sprintf(expressionin,"%s",expression3);
  lenexp=strlen(expressionin);
  lendef=strlen(defined);
  /*if(operator=='\0') operator='+';*/
  if((isdigit(expressionin[0]))
     ||((expressionin[0]=='-')&&(isdigit(expressionin[1])))
     ||(expressionin[0]=='.')) {
    fexpression=G__atodouble(expressionin);
    fdefined=G__atodouble(defined);
    ldefined=atol(defined);
    lexpression=atol(expressionin);
    switch(operator) {
    case '+':
    case '\0':
      sprintf(defined,"%.12g",fdefined+fexpression);
      break;
    case '-':
      sprintf(defined,"%.12g",fdefined-fexpression);
      break;
    case '*':
      if(strlen(defined)==0) fdefined=1;
      sprintf(defined,"%.12g",fdefined*fexpression);
      break;
    case '/':
      if(strlen(defined)==0) fdefined=1;
      if(fexpression==0) {
	fprintf(stderr,"Error: divide by zero ignored\n");
      }
      else {
	sprintf(defined,"%.12g",fdefined/fexpression);
      }
      break;
    case '^':
      if(strlen(defined)==0) {
	sprintf(defined,"^%.12g"
		,fexpression);
      }
      else {
	sprintf(defined,"%.12g"
		,exp(fexpression*log(fdefined)));
      }
      break;
    case 'E':
      if(strlen(defined)==0) {
	if(strlen(undefined)!=0) {
	  sprintf(undefined,"%s==%.12g",undefined,fexpression);
	}
	else {
	  sprintf(undefined,"%.12g",fexpression);
	}
      }
      else sprintf(defined,"%d",fdefined==fexpression);
      break;
    case 'G':
      if(strlen(defined)==0) {
	if(strlen(undefined)!=0) {
	  sprintf(undefined,"%s>=%.12g",undefined,fexpression);
	}
	else {
	  sprintf(undefined,"%.12g",fexpression);
	}
      }
      else sprintf(defined,"%d",fdefined>=fexpression);
      break;
    case 'L':
      if(strlen(defined)==0) {
	if(strlen(undefined)!=0) {
	  sprintf(undefined,"%s<=%.12g",undefined,fexpression);
	}
	else {
	  sprintf(undefined,"%.12g",fexpression);
	}
      }
      else sprintf(defined,"%d",fdefined<=fexpression);
      break;
    case '<':
      if(strlen(defined)==0) {
	if(strlen(undefined)!=0) {
	  sprintf(undefined,"%s<%.12g",undefined,fexpression);
	}
	else {
	  sprintf(undefined,"%.12g",fexpression);
	}
      }
      else sprintf(defined,"%d",fdefined<fexpression);
      break;
    case '>':
      if(strlen(defined)==0) {
	if(strlen(undefined)!=0) {
	  sprintf(undefined,"%s>%.12g",undefined,fexpression);
	}
	else {
	  sprintf(undefined,"%.12g",fexpression);
	}
      }
      else sprintf(defined,"%d",fdefined>fexpression);
      break;
    case 'N':
      if(strlen(defined)==0) {
	if(strlen(undefined)!=0) {
	  sprintf(undefined,"%s!=%.12g",undefined,fexpression);
	}
	else {
	  sprintf(undefined,"%.12g",fexpression);
	}
      }
      else sprintf(defined,"%d",fdefined!=fexpression);
      break;
    case '!':
      sprintf(defined,"%d",!lexpression);
      break;
    case '&':
      if(strlen(defined)==0) sprintf(defined,"%ld",lexpression);
      else sprintf(defined,"%ld",ldefined&lexpression);
      break;
    case '|':
      sprintf(defined,"%ld",ldefined|lexpression);
      break;
    case 'A':
      if(strlen(defined)==0) {
	if(lexpression==0) {
	  strcpy(undefined,"");
	  strcpy(defined,"0");
	}
      }
      else {
	sprintf(defined,"%d",ldefined&&lexpression);
      }
      break;
    case 'O':
      if(strlen(defined)==0) {
	if(lexpression==1) {
	  strcpy(undefined,"");
	  strcpy(defined,"1");
	}
      }
      else {
	sprintf(defined,"%d",ldefined||lexpression);
      }
      break;
    case '%':
      sprintf(defined,"%ld",ldefined%lexpression);
      break;
      /*
	case '^':
	if(strlen(defined)==0) {
	sprintf(defined,"^%d"
	,lexpression);
	}
	else {
	lresult=1;
	for(ig2=1;ig2<=lexpression;ig2++)
	lresult *= ldefined;
	sprintf(defined,"%d",lresult);
	}
	break;
      */
    }
  }
  else {
    ig1=strlen(undefined);
    ig2=0;
    if(operator!='\0') {
      switch(operator) {
      case 'A':
	if(strlen(defined)!=0) {
	  fdefined=G__atodouble(defined);
	  if(fdefined==0.0) {
	    strcpy(undefined,"");
	  }
	  else {
	    strcpy(defined,"");
	    strcpy(undefined,expressionin);
	  }
	  return;
	}
	undefined[ig1++]='&';
	undefined[ig1++]='&';
	break;
      case 'O':
	if(strlen(defined)!=0) {
	  fdefined=G__atodouble(defined);
	  if(fdefined==1.0) {
	    strcpy(undefined,"");
	  }
	  else {
	    strcpy(defined,"");
	    strcpy(undefined,expressionin);
	  }
	  return;
	}
	undefined[ig1++]='|';
	undefined[ig1++]='|';
	break;
      case 'E':
	undefined[ig1++]='=';
	undefined[ig1++]='=';
	break;
      case 'N':
	undefined[ig1++]='!';
	undefined[ig1++]='=';
	break;
      case 'L':
	undefined[ig1++]='<';
	undefined[ig1++]='=';
	break;
      case 'G':
	undefined[ig1++]='>';
	undefined[ig1++]='=';
	break;
      default :
	undefined[ig1++]=operator;
	break;
      }
    }
    while(expressionin[ig2]!='\0') {
      if(ig1+lendef<ONELINE-1) {
	undefined[ig1++]=expressionin[ig2++];
      }
      else {
	fprintf(stderr,"Fatal Error: Expression too long %s\n"
		,undefined);
	exit(EXIT_FAILURE);
      }
    }
    undefined[ig1]='\0';
  }
}


/***********************************************************************/
/* isoperator(char c)   +,-,*,/,^,%,&,|                                */
/* isexponent(char *expression,int lenexpr)                            */
/* isfloat(char *string)                                               */
/***********************************************************************/

int isfloat(string)
     char *string;
{
  int lenstring;
  int ig17;
  int floating=0;
  lenstring=strlen(string);
  for(ig17=0;ig17<lenstring;ig17++) {
    switch(string[ig17]) {
    case '.':
    case 'e':
    case 'E':
    case 't':
    case 'T':
    case 'g':
    case 'G':
    case 'M':
    case 'k':
    case 'K':
    case 'm':
    case 'u':
    case 'U':
    case 'n':
    case 'N':
    case 'p':
    case 'P':
    case 'f':
    case 'F':
    case 'a':
    case 'A':
      floating=1;
      break;
    }
  }
  return(floating);
}

int isoperator(c)
     char c;
{
  switch(c) {
  case '+':
  case '-':
  case '*':
  case '/':
  case '^':
  case '&':
  case '%':
  case '|':
    return(1);
  default:
    return(0);
  }
}

int isexponent(expression4,lenexpr)
     char *expression4;
     int lenexpr;
{
  if(  ((isdigit(expression4[0]))
	||(expression4[0]=='0')
	||(expression4[0]=='.'))
       &&(toupper(expression4[lenexpr-1])=='E')  ) {
    return(1);
  }
  else {
    return(0);
  }
}

/***********************************************************************/
/* G__getoperator(char *newoperator,*oldoperator)                         */
/***********************************************************************/

char G__getoperator(newoperator,oldoperator)
     char newoperator,oldoperator;
{
  switch(newoperator) {
  case '+':
    switch(oldoperator) {
    case '+':
      return('I');
    case '-':
      return('-');
    default:
      return(oldoperator);
    }
    break;
  case '-':
    switch(oldoperator) {
    case '+':
      return('-');
    case '-':
      return('D');
    default:
      return(oldoperator);
    }
    break;
  case '&':
    switch(oldoperator) {
    case '&':
      return('A');
    default:
      return(oldoperator);
    }
    break;
  case '|':
    switch(oldoperator) {
    case '|':
      return('O');
    default:
      return(oldoperator);
    }
    break;
  case '*':
    switch(oldoperator) {
    case '/':
      return('/');
    case '*':
      return('^');
    default:
      return(newoperator);
    }
    break;
  case '/':
    switch(oldoperator) {
    case '/':
      return('*');
    case '*':
      return('/');
    default:
      return(newoperator);
    }
    break;
  case '!':
  case '<':
  case '>':
    switch(oldoperator) {
    default:
      return(newoperator);
    }
    break;
  case '=':
    switch(oldoperator) {
    case '=':
      return('E');
    case '!':
      return('N');
    case '<':
      return('L');
    case '>':
      return('G');
    default:
      return(newoperator);
    }
    break;
  default:
    return(oldoperator);
  }
}


/***********************************************************************/
/* VARIABLE MANAGEMENT MONITOR                                         */
/***********************************************************************/

void G__varmonitor(var)
     struct G__var_array *var;
     /* check global 
	int allvar=0;
	char varnamebuf[MEMDEPTH][MAXNAME];
	int varlabel[MEMDEPTH][MAXPARA];
	int varpointer[MEMDEPTH*2];
	char varvaluebuf[MEMDEPTH*VARSIZE];
*/
{
  int imon1,imon2;

  fprintf(stderr,"varvaluebuf");
  for(imon1=0;imon1<5;imon1++) {
    fprintf(stderr,"\n%3d~%3d  ",imon1*50,(imon1+1)*50-1);
    for(imon2=0;imon2<50;imon2++) {
      if(var->varvaluebuf[imon1*50+imon2]!='\0')
	fprintf(stderr,"%c",var->varvaluebuf[imon1*50+imon2]);
      else
	fprintf(stderr,"\\");
    }
  }
  fprintf(stderr,"\n");

  fprintf(stderr,"varpointer\n");
  for(imon1=0;imon1<30;imon1++) {
    fprintf(stderr,"%d ",var->varpointer[imon1]);
  }
  fprintf(stderr,"\n");
	
  fprintf(stderr,"varlabel\n");
  for(imon1=0;imon1<10;imon1++) {
    fprintf(stderr,"varnamebuf=%-10s , varlabel[%3d] = "
	    ,var->varnamebuf[imon1],imon1);
    for(imon2=0;imon2<5;imon2++) {
      fprintf(stderr," %d ",var->varlabel[imon1][imon2]);
    }
    fprintf(stderr,"\n");
  }
  fprintf(stderr,"allvar=%d\n",var->allvar);
}


/***********************************************************************/
/* char *G__calc(char *exprwithspace)                                  */
/*                                                                     */
/*  Main entry of the entire program.                                  */
/*  Remove space and tab from the expression and give it to G__getexpr()*/
/***********************************************************************/

char *G__calc(exprwithspace)
     char *exprwithspace;
{
  char exprnospace[ONELINE];
  static char result[ONELINE];
  int iin,iout;
  int single_quote=0,double_quote=0;

  G__error_flag=0;
  G__new_variable=0;
  iin = 0;
  iout = 0;
  while( exprwithspace[iin] != '\0' ) {
    switch( exprwithspace[iin] ) {
    case '"' : /* double quote */
      if(single_quote==0) {
	if(double_quote==0)  double_quote=1;
	else                 double_quote=0;
      }
      exprnospace[iout++] = exprwithspace[iin++] ;
      break;
    case '\'' : /* single quote */
      if(double_quote==0) {
	if(single_quote==0)  single_quote=1;
	else                 single_quote=0;
      }
      exprnospace[iout++] = exprwithspace[iin++] ;
      break;
    case '\n': /* end of line */
    case ';' : /* semi-column */
    case ' ' : /* space */
    case '	' : /* tab */
      if((single_quote!=0)||(double_quote!=0)) {
	exprnospace[iout++] = exprwithspace[iin] ;
      }
      iin++;
      break;
    default :
      exprnospace[iout++] = exprwithspace[iin++] ;
      break;
    }
  }
  exprnospace[iout++] = '\0';


  /* return(G__getexpr(exprnospace)); */
  strcpy(result,G__getandor(exprnospace));
  return(result);
}



int G__recursive_check(varname,result7)
     char *varname,*result7;
{
  char subvar[MAXNAME];
  int ig35,ig25;
  int recursive=0;
  int isub=0;
  subvar[0]='\0';
  ig25=strlen(result7);
  for(ig35=0;ig35<=ig25;ig35++) {
    switch(result7[ig35]) {
    case '+':
    case '-':
    case '&':
    case '|':
    case '=':
    case '^':
    case '%':
    case '/':
    case '*':
    case '\0':
    case '(':
    case ')':
    case '[':
    case ']':
    case ',':
      subvar[isub]='\0';
      if(strcmp(subvar,varname)==0)
	recursive = 1;
      subvar[0]='\0';
      isub=0;
      break;
    default:
      subvar[isub++]=result7[ig35];
      break;
    }
  }
  return(recursive);
}



void G__init_var_array(var)
     struct G__var_array *var;
{
  char temp[ONELINE];
  var->allvar = 0;
  sprintf(temp,"stdout=%ld",(long)stdout); G__getexpr(temp);
  sprintf(temp,"stderr=%ld",(long)stderr); G__getexpr(temp);
  sprintf(temp,"stdin=%ld",(long)stdin); G__getexpr(temp);
}


int library_func(result7,funcname,libp)
/*  return 1 if function is executed */
/*  return 0 if function isn't executed */
char *result7,*funcname;
struct G__param *libp;
{
	char temp[ONELINE],temp1[ONELINE];
	FILE *fopen();
	int fp;

	if(strcmp(funcname,"printf")==0) {
		/* para[0]:description, para[1~paran-1]: */
		G__charformatter(result7,0,libp);
		printf("%s",result7);
		strcpy(result7,"");
		return(1);
	}

	if(strcmp(funcname,"fopen")==0) {
		/* para[0]:"filename", para[1]:"mode" "rw+" */
		/* return file pointer */
		strcpy(temp,G__strip_quotation(libp->para[0]));
		strcpy(temp1,G__strip_quotation(libp->para[1]));
		sprintf(result7,"%ld",(long)fopen(temp,temp1));
		return(1);
	}

	if(strcmp(funcname,"fclose")==0) {
		/* para[0]:filepointer */
		sprintf(result7,"%d",fclose((FILE*)atoi(libp->para[0])));
		return(1);
	}

	if(strcmp(funcname,"fgets")==0) {
		/* parameter[0]:varname, para[1]:nchar,para[2]:filepointer */
		fp=atoi(libp->para[2]);
		sprintf(result7,"%ld"
			,(long)fgets(temp,atoi(libp->para[1]),(FILE*)fp));
		G__letvariable(libp->parameter[0]
			    ,G__add_quotation(temp),&G__global,G__local);
		sprintf(temp,"%d",fp);
		G__letvariable(libp->parameter[1],temp,&G__global,G__local);
		return(1);
	}

	if(strcmp(funcname,"fprintf")==0) {
		/* parameter[0]:pointer ,parameter[1]:description, para[2~paran-1]: */
		G__charformatter(result7,1,libp);
		fprintf((FILE*)atoi(libp->para[0]),"%s",result7);
		strcpy(result7,"");
		return(1);
	}

	if(strcmp(funcname,"sprintf")==0) {
		/* parameter[0]:charname ,para[1]:description, para[2~paran-1]: */
		G__charformatter(result7,1,libp);
		G__letvariable(libp->parameter[0]
			    ,G__add_quotation(result7),&G__global,G__local);
		strcpy(result7,"");
		return(1);
	}

	if(strcmp(funcname,"strlen")==0) {
		sprintf(result7,"%ld",(long)strlen(libp->para[0]));
		return(1);
	}

	if(  ((isdigit(libp->para[0][0]))
	   ||((libp->para[0][0]=='-')&&(isdigit(libp->para[0][1]))))
	   &&((isdigit(libp->para[1][0]))
	    ||((libp->para[1][0]=='-')&&(isdigit(libp->para[1][1])))) ) {
		/* if para[0] and para[1] are defined numbers */

		if(strcmp(funcname,"and")==0) {
			sprintf(result7,"%d" ,atoi(libp->para[0])&atoi(libp->para[1]));
			return(1);
		}
		if(strcmp(funcname,"or")==0) {
			sprintf(result7,"%d" ,atoi(libp->para[0])|atoi(libp->para[1]));
			return(1);
		}
		if(strcmp(funcname,"bin")==0) {
			sprintf(result7,"%s", G__getbase(libp->para[0],2,atoi(libp->para[1])));
			return(1);
		}
		if(strcmp(funcname,"quad")==0) {
			sprintf(result7,"%s", G__getbase(libp->para[0],4,atoi(libp->para[1])));
			return(1);
		}
		if(strcmp(funcname,"oct")==0) {
			sprintf(result7,"%s", G__getbase(libp->para[0],8,atoi(libp->para[1])));
			return(1);
		}
		if(strcmp(funcname,"hex")==0) {
			sprintf(result7,"%s", G__getbase(libp->para[0],16,atoi(libp->para[1])));
			return(1);
		}
	}


	if(  ((isdigit(libp->para[0][0]))
	    ||((libp->para[0][0]=='-')&&(isdigit(libp->para[0][1]))))
	   &&((isdigit(libp->para[1][0]))
	    ||((libp->para[1][0]=='-')&&(isdigit(libp->para[1][1]))))
	   &&((isdigit(libp->para[2][0]))
	    ||((libp->para[2][0]=='-')&&(isdigit(libp->para[2][1]))))) {
		/* if para[0~2] are defined numbers */

		if(strcmp(funcname,"and")==0) {
			sprintf(result7,"%d" ,atoi(libp->para[0]) &atoi(libp->para[1]) &atoi(libp->para[2]));
			return(1);
		}
		if(strcmp(funcname,"or")==0) {
			sprintf(result7,"%d" ,atoi(libp->para[0]) |atoi(libp->para[1]) |atoi(libp->para[2]));
			return(1);
		}
	}


	if((isdigit(libp->para[0][0]))
	 ||((libp->para[0][0]=='-')&&(isdigit(libp->para[0][1])))) {
		/* if para[0] is a defined number */

		if(strcmp(funcname,"abs")==0) {
			sprintf(result7,"%.12g" ,fabs(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"fabs")==0) {
			sprintf(result7,"%.12g" ,fabs(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"acos")==0) {
			sprintf(result7,"%.12g" ,acos(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"asin")==0) {
			sprintf(result7,"%.12g" ,asin(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"atan")==0) {
			sprintf(result7,"%.12g" ,atan(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"atan2")==0) {
			sprintf(result7,"%.12g" ,atan2(G__atodouble(libp->para[0]),
						       G__atodouble(libp->para[1])));
			return(1);
		}
		if(strcmp(funcname,"cos")==0) {
			sprintf(result7,"%.12g" ,cos(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"cosh")==0) {
			sprintf(result7,"%.12g" ,cosh(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"exp")==0) {
			sprintf(result7,"%.12g" ,exp(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"floor")==0) {
			sprintf(result7,"%.12g" ,floor(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"fmod")==0) {
			sprintf(result7,"%.12g" ,fmod(G__atodouble(libp->para[0]),
						      G__atodouble(libp->para[1])));
			return(1);
		}
		if(strcmp(funcname,"mod")==0) {
			sprintf(result7,"%.12g" ,fmod(G__atodouble(libp->para[0]),
						      G__atodouble(libp->para[1])));
			return(1);
		}
		if(strcmp(funcname,"int")==0) {
			sprintf(result7,"%d" ,atoi(libp->para[0]));
			return(1);
		}
		if(strcmp(funcname,"log")==0) {
			sprintf(result7,"%.12g" ,log(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"log10")==0) {
			sprintf(result7,"%.12g" ,log10(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"pow")==0) {
			sprintf(result7,"%.12g" ,pow(G__atodouble(libp->para[0]),
						     G__atodouble(libp->para[1])));
			return(1);
		}
		if(strcmp(funcname,"sin")==0) {
			sprintf(result7,"%.12g" ,sin(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"sinh")==0) {
			sprintf(result7,"%.12g" ,sinh(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"sqrt")==0) {
			sprintf(result7,"%.12g" ,sqrt(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"sqr")==0) {
			sprintf(result7,"%.12g" ,sqrt(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"srand")==0) {
			srand(atoi(libp->para[0]));
			strcpy(result7,"");
			return(1);
		}
		if(strcmp(funcname,"rand")==0) {
			sprintf(result7,"%d" ,rand());
			return(1);
		}
		if(strcmp(funcname,"tan")==0) {
			sprintf(result7,"%.12g" ,tan(G__atodouble(libp->para[0])));
			return(1);
		}
		if(strcmp(funcname,"tanh")==0) {
			sprintf(result7,"%.12g" ,tanh(G__atodouble(libp->para[0])));
			return(1);
		}
	}

#ifdef G__IFDEF
	if(strcmp(funcname,"defined")==0) {
		/* para[0]:description, para[1~paran-1]: */
		switch(G__defined(libp->parameter[0])) {
		case 1:
		        resolved++; 
			sprintf(result7,"1");
			break;
		case -1:
		        resolved++; 
			sprintf(result7,"0");
			break;
		case 0:
		        /* resolved--; */
			unresolved++;
			sprintf(result7,"defined(%s)",libp->para[0]);
			break;
		}
		return(1);
	}
#endif

	return(0);
       
}


void G__charformatter(result,ifmt,libp)
char *result;
int ifmt;
struct G__param *libp;
/************************************************************
* libp->para[ifmt] = "text format %s %d %x like this.\n"
* libp->para[ifmt+1] ~ libp->para[libp->paran-1] = parameter
* result formatted result
*************************************************************/
{
	int ipara,ichar,lenfmt;
	int ionefmt=0,fmtflag=0;
	char onefmt[ONELINE],fmt[ONELINE];

	strcpy(result,"");
	ipara=ifmt+1;
	lenfmt = strlen(libp->para[ifmt]);
	for(ichar=1;ichar<lenfmt;ichar++) {
		switch(libp->para[ifmt][ichar]) {
		case '\\': /* new line */
			if(libp->para[ifmt][ichar+1]=='n') {
				onefmt[ionefmt]='\0';
				sprintf(fmt,"%%s%s\n",onefmt);
				sprintf(result,fmt,result);
				ionefmt=0;
				ichar++;
			}
			else {
				onefmt[ionefmt++]=libp->para[ifmt][ichar];
			}
                        break;
		case '"': /* end of the format */
			onefmt[ionefmt]='\0';
			sprintf(fmt,"%%s%s",onefmt);
			sprintf(result,fmt,result);
			ionefmt=0;
			break;
		case 's': /* string */
			onefmt[ionefmt++]=libp->para[ifmt][ichar];
			if(fmtflag==1) {
				onefmt[ionefmt]='\0';
				sprintf(fmt,"%%s%s",onefmt);
				sprintf(result,fmt,result
					,G__strip_quotation(libp->para[ipara]));
				ipara++;
				ionefmt=0;
			}
			break;
		case 'd': /* int */
		case 'i': /* int */
		case 'u': /* unsigned int */
		case 'c': /* char */
		case 'o': /* octal */
		case 'x': /* hex */
		case 'X': /* HEX */
			onefmt[ionefmt++]=libp->para[ifmt][ichar];
			if(fmtflag==1) {
				onefmt[ionefmt]='\0';
				sprintf(fmt,"%%s%s",onefmt);
				sprintf(result,fmt
					,result,atoi(libp->para[ipara++]));
				ionefmt=0;
			}
			break;
		case 'e': /* exponential form */
		case 'E': /* Exponential form */
		case 'f': /* floating */
		case 'g': /* floating or exponential */
		case 'G': /* floating or exponential */
			onefmt[ionefmt++]=libp->para[ifmt][ichar];
			if(fmtflag==1) {
				onefmt[ionefmt]='\0';
				sprintf(fmt,"%%s%s",onefmt);
				sprintf(result,fmt
					,result,atof(libp->para[ipara++]));
				ionefmt=0;
			}
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
		case '-':
		case '+':
		case 'l': /* long int */
		case 'L': /* long double */
		case 'h': /* short int unsinged int */
			onefmt[ionefmt++]=libp->para[ifmt][ichar];
			break;
		case '%':
			if(fmtflag==0) fmtflag=1;
			else           fmtflag=0;
			onefmt[ionefmt++]=libp->para[ifmt][ichar];
			break;
		default:
			fmtflag=0;
			onefmt[ionefmt++]=libp->para[ifmt][ichar];
			break;
		}
	}

}


char *G__strip_quotation(string)
char *string;
{
	int itemp;
	static char temp[ONELINE];
	if((string[0]=='"')||(string[0]=='\'')) {
		for(itemp=1;itemp<strlen(string)-1;itemp++ ) {
			temp[itemp-1] = string[itemp];
		}
		temp[itemp-1]='\0';
		return( temp ) ;
	}
	else {
		return( string ) ;
	}
}

char *G__add_quotation(string)
char *string;
{
	static char temp[ONELINE];
	sprintf(temp,"\"%s\"",string);
	return(temp);
}

void G__error_clear()
{
	G__error_flag=0;
	G__new_variable=0;
}


/***********************************************************************/
/* char *G__exec_statement(fp)                                         */
/*                                                                     */
/*  Main entry of the entire program.                                  */
/*  Remove space and tab from the expression and give it to G__getexpr()*/
/***********************************************************************/

int G__exec_statement(fin)
struct G__input_file *fin;
{
	char statement[ONELINE];
	char c;
	int mparen=0;
	int iout=0;
	int spaceflag=0;
	int sparenflag=0;
	int single_quote=0,double_quote=0;
	char condition[ONELINE];
	int icond=0,nest=0;

	/* for,while,do statement file position buffer*/
	fpos_t store_fpos;
	int    store_line_number;

	G__error_clear();


	while(1) {
		c=G__fgetc(fin->fp);

		switch( c ) {
		case EOF : /* end of file */
			G__eof=1;
			goto exit_statement;
			break;

		case '{' : 
			if((single_quote!=0)||(double_quote!=0)) {
				statement[iout++] = c ;
			}
			else {
				mparen++;
			}
			break;

		case '}' :
			if((single_quote!=0)||(double_quote!=0)) {
				statement[iout++] = c ;
			}
			else {
				if((--mparen)==0) {
					goto exit_statement;
				}
			}
			break;

		case ';' : /* semi-column */
			if((single_quote!=0)||(double_quote!=0)) {
				statement[iout++] = c ;
			}
			else {
				statement[iout++] = '\0';
				if(G__no_exec==0) {
					if(strcmp(statement,"break")==0) 
						goto exit_statement;
					G__getexpr(statement);
					G__error_clear();
					iout=0;
					spaceflag=0;
					sparenflag=0;
					if(mparen==0) goto exit_statement;
				}
			}
			break;

		case '"' : /* double quote */
			if(single_quote==0) {
				if(double_quote==0)  double_quote=1;
				else                 double_quote=0;
			}
			statement[iout++] = c ;
			spaceflag=5;
			break;

		case '\'' : /* single quote */
			if(double_quote==0) {
				if(single_quote==0)  single_quote=1;
				else                 single_quote=0;
			}
			statement[iout++] = c ;
			spaceflag=5;
			break;

		case '\n': /* end of line */
			fin->line_number++;
			if((G__debug!=0)&&(G__no_exec==0)) 
				fprintf(stderr,"%-5d",fin->line_number);
		case ' ' : /* space */
		case '	' : /* tab */
			/* ignore these character */
			if((single_quote!=0)||(double_quote!=0)) {
				statement[iout++] = c ;
			}
			else {
				if(spaceflag==1) {
					statement[iout] = '\0' ;
					/* search keyword */
					if(strcmp(statement,"#define")==0) {
					}
					if(strcmp(statement,"int")==0) {
					}
					if(strcmp(statement,"short")==0) {
					}
					if(strcmp(statement,"char")==0) {
					}
					if(strcmp(statement,"double")==0) {
					}
					if(strcmp(statement,"do")==0) {
					}
					spaceflag=2;
				}
			}
			break;

		case '(' : /* parenthesis */
			statement[iout++] = c ;

			if((spaceflag==1)&&(G__no_exec==0)) {
				statement[iout] = '\0' ;

				/* search keyword */

				if(strcmp(statement,"if(")==0) {
					statement[0] = '\0' ;
					iout=0;
					icond=0;
					nest=1;
					while(nest!=0) {
						c=G__fgetc(fin->fp);
						switch(c) {
						case EOF : /* end of file */
							G__eof=1;
							goto exit_statement;
							break;
						case '(':
							if((single_quote==0)||
							   (double_quote==0))
								nest++;
							condition[icond++]=c;
							break;
						case ')':
							if((single_quote==0)||
							   (double_quote==0))
								nest--;
							condition[icond++]=c;
							break;
						case '"':
							if(single_quote==0) {
								if(double_quote==0)  double_quote=1;
								else                 double_quote=0;
							}
							condition[icond++] = c ;
							break;
						case '\'':
							if(double_quote==0) {
								if(single_quote==0) single_quote=1;
								else                 single_quote=0;
							}
							condition[icond++] = c ;
							break;
						default :
							condition[icond++]=c;
							break;
						}
					}
					condition[--icond]='\0';
					if(G__test(condition)) {
						G__no_exec=0;
						G__exec_statement(fin);
						G__no_exec=0;
						spaceflag=0;
						sparenflag=0;
					}
					else {
						G__no_exec=1;
						G__exec_statement(fin);
						G__no_exec=0;
						spaceflag=0;
						sparenflag=0;
					}
				}
				if(strcmp(statement,"while(")==0) {
					statement[0] = '\0' ;
					iout=0;
					icond=0;
					nest=1;
					while(nest!=0) {
						c=G__fgetc(fin->fp);
						switch(c) {
						case EOF : /* end of file */
							G__eof=1;
							goto exit_statement;
							break;
						case '(':
							if((single_quote==0)||
							   (double_quote==0))
								nest++;
							condition[icond++]=c;
							break;
						case ')':
							if((single_quote==0)||
							   (double_quote==0))
								nest--;
							condition[icond++]=c;
							break;
						case '"':
							if(single_quote==0) {
								if(double_quote==0)  double_quote=1;
								else                 double_quote=0;
							}
							condition[icond++] = c ;
							break;
						case '\'':
							if(double_quote==0) {
								if(single_quote==0) single_quote=1;
								else                 single_quote=0;
							}
							condition[icond++] = c ;
							break;
						default :
							condition[icond++]=c;
							break;
						}
					}
					condition[--icond]='\0';
					fgetpos(fin->fp,&store_fpos);
					store_line_number=fin->line_number;
					while(G__test(condition)) {
						G__no_exec=0;
						G__exec_statement(fin);
						spaceflag=0;
						sparenflag=0;
						fin->line_number
							=store_line_number;
						fsetpos(fin->fp,&store_fpos);
					}
					G__no_exec=1;
					G__exec_statement(fin);
					G__no_exec=0;
				}
				if(strcmp(statement,"for(")==0) {
				}
				sparenflag=2;
			}
			break;

		default :
			statement[iout++] = c ;
			if(sparenflag==0) sparenflag=1;
			if(spaceflag==0)  spaceflag=1;
			break;
		}
	}

exit_statement:
	return(0);
}


/***********************************************************************/
/* char *G__test(char *expression2)                                    */
/*                                                                     */
/*  test <,>,<=,>=,==,!=,&&,||                                         */
/***********************************************************************/

int G__test(expression2)
char *expression2;
{
	char lresult[ONELINE];
	char rresult[ONELINE];
	char lbuf[ONELINE];
	char rbuf[ONELINE];
	char operator2;
	int lenbuf2=0;
	int ig12;
	int length2;
	int nest2=0;
	int single_quote=0,double_quote=0;

	operator2='\0';
	length2=strlen(expression2);

	if(length2==0) return(0);

	for(ig12=0;ig12<length2;ig12++) {
		switch(expression2[ig12]) {
		case '"' : /* double quote */
			if(double_quote==0)  double_quote=1;
			else                 double_quote=0;
			lbuf[lenbuf2]=expression2[ig12];
			lbuf[++lenbuf2]='\0';
			break;
		case '\'' : /* single quote */
			if(single_quote==0)  single_quote=1;
			else                 single_quote=0;
			lbuf[lenbuf2]=expression2[ig12];
			lbuf[++lenbuf2]='\0';
			break;
		case '=':
			if((nest2==0)&&(single_quote==0)&&(double_quote==0)) {
				if(expression2[ig12+1]=='=') {
					ig12+=2;
					operator2 = 'E';

					lenbuf2=0;
					while(expression2[ig12]!='\0') {
						rbuf[lenbuf2++]
							=expression2[ig12++];
					}
					rbuf[lenbuf2++]='\0';

					strcpy(lresult,G__getexpr(lbuf));
					strcpy(rresult,G__getexpr(rbuf));
					if((G__isvalue(lresult))
					   &&G__isvalue(rresult)) {
					      if(atoi(lresult)==atoi(rresult))
						      return(1);
					      else
						      return(0);
					}
					else {
						if(strcmp(lresult,rresult)==0)
							return(1);
						else
							return(0);
					}
				}
				else {
					operator2='\0';
					lbuf[lenbuf2]=expression2[ig12];
					lbuf[++lenbuf2]='\0';
				}
			}
			else {
				lbuf[lenbuf2]=expression2[ig12];
				lbuf[++lenbuf2]='\0';
			}
			break;
		case '!':
			if((nest2==0)&&(single_quote==0)&&(double_quote==0)) {
				if(expression2[ig12+1]=='=') {
					ig12+=2;
					operator2 = 'N';

					lenbuf2=0;
					while(expression2[ig12]!='\0') {
						rbuf[lenbuf2++]
							=expression2[ig12++];
					}
					rbuf[lenbuf2++]='\0';

					strcpy(lresult,G__getexpr(lbuf));
					strcpy(rresult,G__getexpr(rbuf));
					if((G__isvalue(lresult))
					   &&G__isvalue(rresult)) {
					      if(atoi(lresult)!=atoi(rresult))
						      return(1);
					      else
						      return(0);
					}
					else {
						if(strcmp(lresult,rresult)!=0)
							return(1);
						else
							return(0);
					}
				}
				else {
					operator2='n';
					while(expression2[ig12]!='\0') {
						rbuf[lenbuf2++]
							=expression2[ig12++];
					}
					rbuf[lenbuf2++]='\0';
					if(G__test(rbuf)) {
						return(0) ;
					}
					else {
						return(1) ;
					}
				}
			}
			else {
				lbuf[lenbuf2]=expression2[ig12];
				lbuf[++lenbuf2]='\0';
			}
			break;
		case '>':
			if((nest2==0)&&(single_quote==0)&&(double_quote==0)) {
				if(expression2[ig12+1]=='=') {
					ig12+=2;
					operator2 = 'G';
				}
				else {
					operator2='>';
					ig12++;
				}

				lenbuf2=0;
				while(expression2[ig12]!='\0') {
					rbuf[lenbuf2++]
						=expression2[ig12++];
				}
				rbuf[lenbuf2++]='\0';

				strcpy(lresult,G__getexpr(lbuf));
				strcpy(rresult,G__getexpr(rbuf));
				if((G__isvalue(lresult))
				   &&G__isvalue(rresult)) {
				      if(operator2=='G') {
					      if(atoi(lresult)>=atoi(rresult))
						      return(1);
					      else
						      return(0);
				      }
				      else {
					      if(atoi(lresult)>atoi(rresult))
						      return(1);
					      else
						      return(0);
				      }
				}
				else {
					if(operator2=='G') {
						if(strcmp(lresult,rresult)>=0)
							return(1);
						else
							return(0);
					}
					else {
						if(strcmp(lresult,rresult)>0)
							return(1);
						else
							return(0);
					}
				}
			}
			else {
				lbuf[lenbuf2]=expression2[ig12];
				lbuf[++lenbuf2]='\0';
			}
			break;
		case '<':
			if((nest2==0)&&(single_quote==0)&&(double_quote==0)) {
				if(expression2[ig12+1]=='=') {
					ig12+=2;
					operator2 = 'L';
				}
				else {
					operator2='<';
					ig12++;
				}

				lenbuf2=0;
				while(expression2[ig12]!='\0') {
					rbuf[lenbuf2++]
						=expression2[ig12++];
				}
				rbuf[lenbuf2++]='\0';

				strcpy(lresult,G__getexpr(lbuf));
				strcpy(rresult,G__getexpr(rbuf));
				if((G__isvalue(lresult))
				   &&G__isvalue(rresult)) {
				      if(operator2=='L') {
					      if(atoi(lresult)<=atoi(rresult))
						      return(1);
					      else
						      return(0);
				      }
				      else {
					      if(atoi(lresult)<atoi(rresult))
						      return(1);
					      else
						      return(0);
				      }
				}
				else {
					if(operator2=='L') {
						if(strcmp(lresult,rresult)<=0)
							return(1);
						else
							return(0);
					}
					else {
						if(strcmp(lresult,rresult)<0)
							return(1);
						else
							return(0);
					}
				}
			}
			else {
				lbuf[lenbuf2]=expression2[ig12];
				lbuf[++lenbuf2]='\0';
			}
			break;
		case '(':
		case '[':
		case '{':
			nest2++;
			lbuf[lenbuf2]=expression2[ig12];
			lbuf[++lenbuf2]='\0';
			break;
		case ')':
		case ']':
		case '}':
			lbuf[lenbuf2]=expression2[ig12];
			lbuf[++lenbuf2]='\0';
			nest2--;
			break;
		default :
			lbuf[lenbuf2]=expression2[ig12];
			lbuf[++lenbuf2]='\0';
			break;
		}
	}

	if((nest2!=0)||(single_quote!=0)||(double_quote!=0)) {
	   if((G__error_flag++)==0)
		fprintf(stderr,"Syntax error: Parenthesis or quotation unmatch %s\n"
			,expression2);
	}

	strcpy(lresult,G__getexpr(lbuf));
	if(G__isvalue(lresult)) {
		if(atoi(lresult)!=0) 
			return(1);
		else 
			return(0);
	}
	else {
		if(strcmp(lresult,"?")==0) 
			return(0);
		else {
			return(strlen(lresult));
		}
	}

}


int G__isvalue(temp)
char *temp;
{
	if  ( (isdigit(temp[0])) ||((temp[0]=='-')&&(isdigit(temp[0])))) {
		return(1);
	}
	else {
		return(0);
	}
}


#ifndef G__OLDIMPLEMENTATION1616
int G__fgetc(fp)
#else
char G__fgetc(fp)
#endif
FILE *fp;
{
#ifndef G__OLDIMPLEMENTATION1616
	int c;
#else
	char c;
#endif
	c=fgetc(fp);
	if((G__debug!=0)&&(G__no_exec==0)) {
		if(c != EOF) fputc(c,stderr);
		else         fprintf(stderr,"EOF\n");
	}

	return( c ) ;
}


