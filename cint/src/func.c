/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file func.c
 ************************************************************************
 * Description:
 *  Function call
 ************************************************************************
 * Copyright(c) 1995~2002  Masaharu Goto 
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


#define G__OLDIMPLEMENTATION2051
#ifndef G__OLDIMPLEMENTATION2051
extern void G__cpp_setupG__bcdict();
#endif

#ifndef G__OLDIMPLEMENTATION1932
#if defined(G__WIN32)
#include <windows.h>
#elif defined(G__POSIX)
#include <unistd.h> /* already included in G__ci.h */
#endif
#endif

#ifndef __CINT__
void G__display_tempobject G__P((char* action));
#endif

#ifndef G__OLDIMPLEMENTATION863
/* not ready yet
typedef void* G__SHLHANDLE;
G__SHLHANDLE G__dlopen G__P((char *path));
void* G__shl_findsym G__P((G__SHLHANDLE *phandle,char *sym,short type));
int G__dlclose G__P((void *handle));
*/
#endif

#ifndef G__OLDIMPLEMENTATION1937
#ifndef __CINT__
int G__set_history_size G__P((int s));
#endif
#endif

#ifndef G__OLDIMPLEMENTATION1142
#ifndef __CINT__
int G__optimizemode G__P((int optimizemode));
int G__getoptimizemode G__P(());
#endif
#endif

#ifndef G__OLDIMPLEMENTATION1103
extern int G__const_noerror;
#endif

#ifndef G__OLDIMPLEMENTATION2120
/******************************************************************
*
* G__gen_PUSHSTROS_SETSTROS() 
******************************************************************/
void G__gen_PUSHSTROS_SETSTROS() {
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

/******************************************************************
* G__dispvalue()
******************************************************************/
int G__dispvalue(fp,buf)
FILE* fp;
G__value *buf;
{
  if(buf) {
    fprintf(fp
	    ,"\nd=%g i=%ld reftype=%d type=%c tagn=%d typen=%d ref=%ld isconst=%d\n"
	    ,buf->obj.d
	    ,buf->obj.i
	    ,buf->obj.reftype.reftype
	    ,buf->type
	    ,buf->tagnum
	    ,buf->typenum
	    ,buf->ref
	    ,buf->isconst
	    );
  }
  return(1);
}

#ifndef G__OLDIMPLEMENTATION1644
static int G__exitcode = 0;
/******************************************************************
* G__getexitcode()
******************************************************************/
int G__getexitcode()
{
  int exitcode = G__exitcode;
  G__exitcode = 0 ;
  return(exitcode);
}

/******************************************************************
* G__get_return()
******************************************************************/
int G__get_return(int *exitval)
{
  if (exitval)
    *exitval = G__getexitcode();
  return G__return;
}
#endif

#ifndef G__OLDIMPLEMENTATION1618
/******************************************************************
* G__bytecodedebugmode()
******************************************************************/
int G__bytecodedebugmode(mode)
int mode;
{
  G__asm_dbg = mode;
#ifndef G__OLDIMPLEMENTATION1155
#endif
  return(G__asm_dbg);
}

/******************************************************************
* G__getoptmizemode()
******************************************************************/
int G__getbytecodedebugmode()
{
  return(G__asm_dbg);
}
#endif


#ifndef G__OLDIMPLEMENTATION1198
/******************************************************************
* G__storelasterror()
******************************************************************/
void G__storelasterror()
{
  G__lasterrorpos = G__ifile;
}

/******************************************************************
* G__lasterror_filename()
******************************************************************/
char* G__lasterror_filename()
{
  return(G__lasterrorpos.name);
}
/******************************************************************
* G__lasterror_linenum()
******************************************************************/
int G__lasterror_linenum()
{
  return(G__lasterrorpos.line_number);
}
#endif

#ifndef G__OLDIMPLEMENTATION1192
/******************************************************************
* G__checkscanfarg()
******************************************************************/
int G__checkscanfarg(fname,libp,n)
char *fname;
struct G__param *libp;
int n;
{
  int result=0;
  while(n<libp->paran) {
    if(islower(libp->para[n].type)) {
      G__fprinterr(G__serr,"Error: %s arg%d not a pointer",fname,n);
      G__genericerror((char*)NULL);
      ++result;
    }
    if(0==libp->para[n].obj.i) {
      G__fprinterr(G__serr,"Error: %s arg%d is NULL",fname,n);
      G__genericerror((char*)NULL);
      ++result;
    }
    ++n;
  }
  return(result);
}
#endif

#ifndef G__OLDIMPLEMENTATION875
/******************************************************************
******************************************************************
* Pointer to function evaluation function
******************************************************************
******************************************************************/

/******************************************************************
* G__p2f_void_void()
******************************************************************/
void G__p2f_void_void(p2f)
void *p2f;
{
  switch(G__isinterpretedp2f(p2f)) {
  case G__INTERPRETEDFUNC: 
  {
    char buf[G__ONELINE];
    char *fname;
    fname = G__p2f2funcname(p2f);
    sprintf(buf,"%s()",fname);
    if(G__asm_dbg) G__fprinterr(G__serr,"(*p2f)() %s interpreted\n",buf);
    G__calc_internal(buf);
  }
    break;
  case G__BYTECODEFUNC: 
  {
    struct G__param param;
    G__value result;
#ifdef G__ANSI
    int (*ifm)(G__value*,char*,struct G__param*,int);
    ifm = (int (*)(G__value*,char*,struct G__param*,int))G__exec_bytecode;
#else
    int (*ifm)();
    ifm = (int (*)())G__exec_bytecode;
#endif
    param.paran=0;
    if(G__asm_dbg) G__fprinterr(G__serr,"(*p2f)() bytecode\n");
    (*ifm)(&result,(char*)p2f,&param,0);
  }
    break;
  case G__COMPILEDINTERFACEMETHOD:
  {
    struct G__param param;
    G__value result;
#ifdef G__ANSI
    int (*ifm)(G__value*,char*,struct G__param*,int);
    ifm = (int (*)(G__value*,char*,struct G__param*,int))p2f;
#else
    int (*ifm)();
    ifm = (int (*)())p2f;
#endif
    param.paran=0;
    if(G__asm_dbg) G__fprinterr(G__serr,"(*p2f)() compiled interface\n");
    (*ifm)(&result,(char*)NULL,&param,0);
  }
    break;
  case G__COMPILEDTRUEFUNC:
  case G__UNKNOWNFUNC:
  {
    void (*tp2f)();
    tp2f = (void (*)())p2f;
    if(G__asm_dbg) G__fprinterr(G__serr,"(*p2f)() compiled true p2f\n");
    (*tp2f)();
  }
    break;
  }
}

/******************************************************************
* G__set_atpause
******************************************************************/
void G__set_atpause(p2f)
void (*p2f)();
{
  G__atpause = p2f;
}

/******************************************************************
* G__set_aterror
******************************************************************/
void G__set_aterror(p2f)
void (*p2f)();
{
  G__aterror= p2f;
}
#endif /* ON875 */

#ifndef G__OLDIMPLEMENTATION1548
/******************************************************************
* G__set_emergencycallback
******************************************************************/
void G__set_emergencycallback(p2f)
void (*p2f)();
{
  G__emergencycallback= p2f;
}
#endif

#ifndef G__OLDIMPLEMENTATION405
/******************************************************************
* G__getindexedvalue()
******************************************************************/
static void G__getindexedvalue(result3,cindex)
G__value *result3;
char *cindex;
{
  int size;
  int index;
  int len;
  char sindex[G__ONELINE];
#ifndef G__OLDIMPLEMENTATION1818
  char *p;
#endif
  strcpy(sindex,cindex);
#ifndef G__OLDIMPLEMENTATION1818
  p=strstr(sindex,"][");
  if(p) {
    *(p+1) = 0;
    G__getindexedvalue(result3,sindex);
    p=strstr(cindex,"][");
    strcpy(sindex,p+1);
  }
#endif
  len=strlen(sindex);
#ifdef G__OLDIMPLEMENTATION424
  /* maybe unnecessary */
  if(len>3&&'['==sindex[0]&&']'==sindex[len-1]);
#endif
  sindex[len-1]='\0';

#ifndef G__OLDIMPLEMENTATION2018

  if('u'==result3->type) {
    struct G__param fpara;
#ifndef G__OLDIMPLEMENTATION2120
#ifdef G__ASM
    if(G__asm_noverflow) G__gen_PUSHSTROS_SETSTROS();
#endif
#endif
    fpara.paran=1;
    fpara.para[0]=G__getexpr(sindex+1);
#ifndef G__OLDIMPLEMENTATION2120
    G__parenthesisovldobj(result3,result3,"operator[]",&fpara,1);
#else
    G__parenthesisovldobj(result3,result3,"operator[]",&fpara,G__TRYNORMAL);
#endif
    return;
  }

  index=G__int(G__getexpr(sindex+1));

#else /* 2018 */
  index=G__int(G__getexpr(sindex+1));

#ifndef G__OLDIMPLEMENTATION1903
  if('u'==result3->type) {
    struct G__param fpara;
    fpara.paran=1;
    G__letint(&fpara.para[0],'i',index);
    G__parenthesisovldobj(result3,result3,"operator[]",&fpara,G__TRYNORMAL);
    return;
  }
#endif
#endif /* 2018 */

  size = G__sizeof(result3);
#ifdef G__ASM
  if(G__asm_noverflow) {
    /* size arithmetic is done by OP2 in bytecode execution */
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: OP2  '%c'\n" ,G__asm_cp,'+');
#endif
    G__asm_inst[G__asm_cp]=G__OP2;
    G__asm_inst[G__asm_cp+1]=(long)('+');
    G__inc_cp_asm(2,0);
  }
#endif
  result3->obj.i += (size*index);
  *result3=G__tovalue(*result3);
}
#endif

#ifndef G__OLDIMPLEMENTATION441
/******************************************************************
* G__explicit_fundamental_typeconv()
*
*
******************************************************************/
int G__explicit_fundamental_typeconv(funcname,hash,libp,presult3)
char *funcname;
int hash;
struct G__param *libp;
G__value *presult3;
{
  int flag=0;

#ifndef G__OLDIMPLEMENTATION491
  /* 
  if('u'==presult3->type && -1!=presult3->tagnum) {
  } 
  */
#endif

  switch(hash) {
  case 3:
    if(strcmp(funcname,"int")==0) {
      presult3->type='i';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(int*)presult3->ref = (int)presult3->obj.i;
#endif
      flag=1;
    }
    break;
  case 4:
    if(strcmp(funcname,"char")==0) {
      presult3->type='c';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(char*)presult3->ref = (char)presult3->obj.i;
#endif
      flag=1;
      break;
    }
    else if(strcmp(funcname,"long")==0) {
      presult3->type='l';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
#endif
      flag=1;
    }
#ifndef G__OLDIMPLEMENTATION560
    else if(strcmp(funcname,"int*")==0) {
      presult3->type='I';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
#endif
      flag=1;
    }
#endif
#ifndef G__OLDIMPLEMENTATION1332
    else if(strcmp(funcname,"bool")==0 
#ifdef G__OLDIMPLEMENTATION1604
	    && 'u'==libp->para[0].type
#endif
	    ) {
#ifndef G__OLDIMPLEMENTATION1604
      presult3->type='g';
      presult3->obj.i = G__int(libp->para[0])?1:0;
      if(presult3->ref) 
	*(unsigned char*)presult3->ref = (unsigned char)presult3->obj.i?1:0;
      flag=1;
#else
      char ttt[G__ONELINE];
      int xtype = 'u';
      int xreftype = 0;
      int xisconst = 0;
      int xtagnum = G__defined_tagname("bool",2);
      *presult3 = libp->para[0];
      G__fundamental_conversion_operator(xtype,xtagnum ,-1 ,xreftype,xisconst
				       ,presult3,ttt);
      flag=1;
      return(flag);
#endif
    }
#endif
    break;
  case 5:
    if(strcmp(funcname,"short")==0) {
      presult3->type='s';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(short*)presult3->ref = (short)presult3->obj.i;
#endif
      flag=1;
      break;
    }
    else if(strcmp(funcname,"float")==0) {
      presult3->type='f';
      presult3->obj.d = G__double(libp->para[0]);
      if(presult3->ref) *(float*)presult3->ref = (float)presult3->obj.d;
      flag=1;
    }
    else if(strcmp(funcname,"char*")==0) {
      presult3->type='C';
      presult3->obj.i = G__int(libp->para[0]);
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
      flag=1;
      break;
    }
    else if(strcmp(funcname,"long*")==0) {
      presult3->type='L';
      presult3->obj.i = G__int(libp->para[0]);
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
      flag=1;
    }
#ifndef G__OLDIMPLEMENTATION774
    else if(strcmp(funcname,"void*")==0) {
      presult3->type = 'Y';
      presult3->obj.i = G__int(libp->para[0]);
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
      flag=1;
    }
#endif /* ON774 */
    break;
  case 6:
    if(strcmp(funcname,"double")==0) {
      presult3->type='d';
      presult3->obj.d = G__double(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(double*)presult3->ref = (double)presult3->obj.d;
#endif
      flag=1;
    }
#ifndef G__OLDIMPLEMENTATION560
    else if(strcmp(funcname,"short*")==0) {
      presult3->type='S';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
#endif
      flag=1;
      break;
    }
    else if(strcmp(funcname,"float*")==0) {
      presult3->type='F';
      presult3->obj.d = G__double(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
#endif
      flag=1;
    }
#endif
    break;
#ifndef G__OLDIMPLEMENTATION560
  case 7:
    if(strcmp(funcname,"double*")==0) {
      presult3->type='d';
      presult3->obj.d = G__double(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
#endif
      flag=1;
    }
    break;
#endif
#ifndef G__OLDIMPLEMENTATION1699
  case 8:
    if(strcmp(funcname,"unsigned")==0) {
      presult3->type='h';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(unsigned int*)presult3->ref = (unsigned int)presult3->obj.i;
#endif
      flag=1;
      break;
    }
#ifndef G__OLDIMPLEMENTATION2189
    if(strcmp(funcname,"longlong")==0) {
      presult3->type='n';
      presult3->obj.ll = G__Longlong(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(G__int64*)presult3->ref = presult3->obj.ll;
#endif
      flag=1;
    }
#endif
    break;
  case 9:
    if(strcmp(funcname,"unsigned*")==0) {
      presult3->type='H';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
#endif
      flag=1;
      break;
    }
#ifndef G__OLDIMPLEMENTATION2189
    if(strcmp(funcname,"longlong*")==0) {
      presult3->type='N';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
#endif
      flag=1;
      break;
    }
#endif
#ifndef G__OLDIMPLEMENTATION2189
    if(strcmp(funcname,"long long")==0) {
      presult3->type='n';
      presult3->obj.ll = G__Longlong(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(G__int64*)presult3->ref = presult3->obj.ll;
#endif
      flag=1;
    }
#endif
    break;
#endif
#ifndef G__OLDIMPLEMENTATION2189
  case 10:
    if(strcmp(funcname,"longdouble")==0) {
      presult3->type='n';
      presult3->obj.ld = G__Longdouble(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(long double*)presult3->ref = presult3->obj.ld;
#endif
      flag=1;
    }
    break;
#endif
  case 11:
    if(strcmp(funcname,"unsignedint")==0) {
      presult3->type='h';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(unsigned int*)presult3->ref = (unsigned int)presult3->obj.i;
#endif
      flag=1;
    }
    break;
  case 12:
    if(strcmp(funcname,"unsignedchar")==0) {
      presult3->type='b';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(unsigned char*)presult3->ref = (unsigned char)presult3->obj.i;
#endif
      flag=1;
      break;
    }
    else if(strcmp(funcname,"unsignedlong")==0) {
      presult3->type='k';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(unsigned long*)presult3->ref = (unsigned long)presult3->obj.i;
#endif
      flag=1;
    }
#ifndef G__OLDIMPLEMENTATION560
    else if(strcmp(funcname,"unsigned int")==0) {
      presult3->type='h';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(unsigned int*)presult3->ref = (unsigned int)presult3->obj.i;
#endif
      flag=1;
    }
#endif
    break;
  case 13:
    if(strcmp(funcname,"unsignedshort")==0) {
      presult3->type='r';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(unsigned short*)presult3->ref = (unsigned short)presult3->obj.i;
#endif
      flag=1;
    }
#ifndef G__OLDIMPLEMENTATION560
    else if(strcmp(funcname,"unsigned char")==0) {
      presult3->type='b';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(unsigned char*)presult3->ref = (unsigned char)presult3->obj.i;
#endif
      flag=1;
      break;
    }
    else if(strcmp(funcname,"unsigned long")==0) {
      presult3->type='k';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(unsigned long*)presult3->ref = (unsigned long)presult3->obj.i;
#endif
      flag=1;
    }
    else if(strcmp(funcname,"unsigned int*")==0) {
      presult3->type='H';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
#endif
      flag=1;
    }
#endif
    break;
#ifndef G__OLDIMPLEMENTATION560
  case 14:
    if(strcmp(funcname,"unsigned short")==0) {
      presult3->type='r';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(unsigned short*)presult3->ref = (unsigned short)presult3->obj.i;
#endif
      flag=1;
    }
    else if(strcmp(funcname,"unsigned char*")==0) {
      presult3->type='B';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
#endif
      flag=1;
      break;
    }
    else if(strcmp(funcname,"unsigned long*")==0) {
      presult3->type='K';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
#endif
      flag=1;
    }
    break;
  case 15:
    if(strcmp(funcname,"unsigned short*")==0) {
      presult3->type='R';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
#endif
      flag=1;
    }
    break;
#endif
#ifndef G__OLDIMPLEMENTATION2189
  case 16:
    if(strcmp(funcname,"unsignedlonglong")==0) {
      presult3->type='m';
      presult3->obj.ull = G__ULonglong(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(G__uint64*)presult3->ref = presult3->obj.ull;
#endif
      flag=1;
    }
    break;
  case 17:
    if(strcmp(funcname,"unsignedlonglong*")==0) {
      presult3->type='M';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
#endif
      flag=1;
    }
    break;
  case 18:
    if(strcmp(funcname,"unsigned long long")==0) {
      presult3->type='m';
      presult3->obj.ull = G__ULonglong(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(G__uint64*)presult3->ref = presult3->obj.ull;
#endif
      flag=1;
    }
    break;
  case 19:
    if(strcmp(funcname,"unsigned long long*")==0) {
      presult3->type='M';
      presult3->obj.i = G__int(libp->para[0]);
#ifndef G__OLDIMPLEMENTATION571
      if(presult3->ref) *(long*)presult3->ref = (long)presult3->obj.i;
#endif
      flag=1;
    }
    break;
#endif
  }

  if(flag) {
    presult3->tagnum = -1;
    presult3->typenum = -1;
#ifndef G__OLDIMPLEMENTATION571
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg&&G__asm_noverflow) {
	G__fprinterr(G__serr,"%3x: CAST to %c\n",G__asm_cp,presult3->type);
      }
#endif
      G__asm_inst[G__asm_cp]=G__CAST;
      G__asm_inst[G__asm_cp+1]=presult3->type;
      G__asm_inst[G__asm_cp+2]=presult3->typenum;
      G__asm_inst[G__asm_cp+3]=presult3->tagnum;
      G__asm_inst[G__asm_cp+4]=G__PARANORMAL;
      G__inc_cp_asm(5,0);
    }
#endif /* ASM */
#else
    presult3->ref = 0;
#endif
  }
#ifndef G_OLDIMPLEMENTATION1128
  if(flag && 'u'==libp->para[0].type) {
    char ttt[G__ONELINE];
    int xtype = presult3->type;
    int xreftype = 0;
    int xisconst = 0;
    *presult3 = libp->para[0];
    G__fundamental_conversion_operator(xtype,-1 ,-1 ,xreftype,xisconst
				       ,presult3,ttt);
  }
#endif
  return(flag);
}
#endif

#ifndef G__OLDIMPLEMENTATION517
/******************************************************************
* void G__gen_addstros()
*
******************************************************************/
void G__gen_addstros(addstros)
int addstros;
{
#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) 
      G__fprinterr(G__serr,"%3x: ADDSTROS %d\n" ,G__asm_cp,addstros);
#endif
    G__asm_inst[G__asm_cp]=G__ADDSTROS;
    G__asm_inst[G__asm_cp+1]=addstros;
    G__inc_cp_asm(2,0);
  }
#endif
}
#endif

#ifdef G__PTR2MEMFUNC
/******************************************************************
* G__pointer2memberfunction()
*
******************************************************************/
G__value G__pointer2memberfunction(parameter0,parameter1,known3)
char *parameter0 ;
char *parameter1;
int *known3;
{
  char buf[G__LONGLINE];
  char buf2[G__ONELINE];
  char expr[G__LONGLINE];
  char* mem;
  G__value res;
  char* opx;

  strcpy(buf,parameter0);

  if((mem=strstr(buf,".*"))) {
    *mem=0;
    mem+=2;
    opx=".";
  }
  else if((mem=strstr(buf,"->*"))) {
    *mem=0;
    mem+=3;
    opx="->";
  }
  else {
    opx="";
  }
 
  res = G__getexpr(mem);
  if(!res.type) {
    G__fprinterr(G__serr,"Error: Pointer to member function %s not found"
	    ,parameter0);
    G__genericerror((char*)NULL);
    return(G__null);
  }

  if(!res.obj.i || !*(char**)res.obj.i) {
    G__fprinterr(G__serr,"Error: Pointer to member function %s is NULL",parameter0);
    G__genericerror((char*)NULL);
    return(G__null);
  }

  /* For the time being, pointer to member function can only be handed as
   * function name */
  strcpy(buf2,*(char**)res.obj.i);

  sprintf(expr,"%s%s%s%s",buf,opx,buf2,parameter1);

  G__abortbytecode();
  return(G__getvariable(expr,known3,&G__global,G__p_local));
}
#endif

#ifndef G__OLDIMPLEMENTATION1393
/******************************************************************
* G__pointerReference()
*
******************************************************************/
G__value G__pointerReference(item,libp,known3)
char *item;
struct G__param *libp;
int *known3;
{
  G__value result3;
  int i,j;
  int store_tagnum = G__tagnum;
  int store_typenum = G__typenum;
  long store_struct_offset = G__store_struct_offset;

#ifndef G__OLDIMPLEMENTATION2166
  result3 = G__getexpr(item);
#else
  result3 = G__getitem(item);
#endif
  if(0==result3.type) return(G__null);
  *known3 = 1;

#ifndef G__OLDIMPLEMENTATION1660
  if(2==libp->paran && strstr(libp->parameter[1],"][")) {
    char arg[G__ONELINE];
    char *p=arg;
    int k;
    strcpy(p,libp->parameter[1]);
    i=1;
    while(*p) {
      k=0;
      if(*p=='[') ++p; 
      while(*p && *p!=']') libp->parameter[i][k++] = *p++;
      libp->parameter[i][k++] = 0;
      if(*p==']') ++p; 
      ++i;
    }
    libp->paran = i;
  }
#endif

  for(i=1;i<libp->paran;i++) {
    char arg[G__ONELINE];
    
    strcpy(arg,libp->parameter[i]);
    if('['==arg[0]) {
      j=0;
      while(arg[++j] && ']'!=arg[j]) arg[j-1] = arg[j];
      arg[j-1] = 0;
    }

    if('u'==result3.type) { /* operator[] overloading */
      char expr[G__ONELINE];
      /* Set member function environment */
      G__tagnum = result3.tagnum;
      G__typenum = result3.typenum;
      G__store_struct_offset = result3.obj.i;
#ifdef G__ASM
      if(G__asm_noverflow) {
#ifndef G__OLDIMPLEMENTATION1449
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp);
#endif
	G__asm_inst[G__asm_cp] = G__PUSHSTROS;
	G__inc_cp_asm(1,0);
#endif
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp);
#endif
	G__asm_inst[G__asm_cp] = G__SETSTROS;
	G__inc_cp_asm(1,0);
      }
#endif
      /* call operator[] */
      *known3 = 0;
      sprintf(expr,"operator[](%s)",arg);
      result3 = G__getfunction(expr,known3,G__CALLMEMFUNC);
      /* Restore environment */
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
    }

    else if(isupper(result3.type)) {
      G__value varg;
      varg = G__getexpr(arg);
      G__bstore('+',varg,&result3);
      result3 = G__tovalue(result3);
    }

    else {
      G__genericerror("Error: Incorrect use of operator[]");
      return(G__null);
    }
  }

  return(result3);
}
#endif

#ifndef G__OLDIMPLEMENTATION1515
#ifndef __CINT__
int G__additional_paranthesis G__P((G__value* presult,struct G__param* libp));
#endif
/******************************************************************
* G__additional_parenthesis()
******************************************************************/
int G__additional_parenthesis(presult,libp)
G__value* presult;
struct G__param* libp;
{
  char buf[G__LONGLINE];
  int known;
  int store_tagnum = G__tagnum;
  long store_struct_offset = G__store_struct_offset;

  if(-1==presult->tagnum) {
    return(0);
  }
  G__tagnum = presult->tagnum;
  G__store_struct_offset = presult->obj.i;

  sprintf(buf,"operator()%s",libp->parameter[1]);
  *presult = G__getfunction(buf,&known,G__CALLMEMFUNC);

  G__tagnum = store_tagnum;
  G__store_struct_offset = store_struct_offset;

  return(known);
}
#endif

#ifndef G__OLDIMPLEMENTATION1560
/******************************************************************
 * G__rename_templatefunc()
 ******************************************************************/
char* G__rename_templatefunc(funcname,isrealloc)
char *funcname;
int isrealloc;
{
  char *ptmplt ;
  ptmplt = strchr(funcname,'<');
  if(ptmplt) {
    *ptmplt = 0;
    if(G__defined_templatefunc(funcname)) {
      *ptmplt = 0;
    }
    else {
      *ptmplt = '<';
      ptmplt = (char*)0;
    }
  }
  if(ptmplt) {
    char funcname2[G__LONGLINE];
    char buf[G__ONELINE];
    char buf2[20];
    int typenum,tagnum,len;
    int ip=1;
    int c;
    strcpy(funcname2,funcname);
    strcat(funcname2,"<");
    do {
      c = G__getstream_template(ptmplt,&ip,buf,",>");
      len = strlen(buf)-1;
      while('*'==buf[len]||'&'==buf[len]) --len;
      ++len;
      if(buf[len]) {
	strcpy(buf2,buf+len);
	buf[len] = 0;
      }
      else buf2[0] = 0;
      typenum = G__defined_typename(buf);
      if(-1!=typenum) {
	strcpy(buf,G__fulltypename(typenum));
      }
      else {
	tagnum = G__defined_tagname(buf,1);
	if(-1!=tagnum) strcpy(buf,G__fulltagname(tagnum,1));
      }
      strcat(buf,buf2);
      strcat(funcname2,buf);
#ifndef G__OLDIMPLEMENTATION2090
      if (funcname2[strlen(funcname2)-1]=='>' && c=='>') {
	buf2[0] = ' '; buf2[1] = c; buf2[2] = 0;  
      } 
      else {
        buf2[0] = c; buf2[1] = 0;
      }
#else
      buf2[0] = c; buf2[1] = 0;
#endif
      strcat(funcname2,buf2);
    } while(c!='>');
    if(isrealloc) {
      free((void*)funcname);
      funcname = (char*)malloc(strlen(funcname2)+1);
    }
    strcpy(funcname,funcname2);
  }
  return(funcname);
}
#endif

#ifndef G__OLDIMPLEMENTATION1871
/******************************************************************
* G__operatorfunction()
*
*
******************************************************************/
G__value G__operatorfunction(presult,item,known3,result7,funcname)
G__value *presult;
char* item;
int *known3;
char* result7;
char* funcname;
{
  G__value result3 = G__null;
  struct G__param fpara;
  int ig15=0;
  int ig35=0;
  int nest=0;
  int double_quote=0, single_quote=0;
  int lenitem = strlen(item);
  int overflowflag=0;
  int castflag=0;
  int base1=0;
  int nindex=0;
  int itmp;

  fpara.paran=0;
#ifndef G__OLDIMPLEMENTATION1472
  fpara.para[0].type = 0;
#endif
  
  /* Get Parenthesis */
  
  /******************************************************
   * this if statement should be always true,
   * should be able to omit.
   ******************************************************/
  if(ig15<lenitem) {
    
    /*****************************************************
     * scan '(param1,param2,param3)'
     *****************************************************/
    while(ig15<lenitem) {
#ifndef G__OLDIMPLEMENTATION1705
      int tmpltnest=0;
#endif
      
      /*************************************************
       * scan one parameter upto 'param,' or 'param)'
       * by reading upto ',' or ')'
       *************************************************/
      ig35 = 0;
      nest=0;
      single_quote=0;
      double_quote=0;
      while((((item[ig15]!=',')&&(item[ig15]!=')'))||
	     (nest>0)||
#ifndef G__OLDIMPLEMENTATION1705
	     (tmpltnest>0)||
#endif
	     (single_quote>0)||(double_quote>0))&&(ig15<lenitem)) {
	switch(item[ig15]) {
	case '"' : /* double quote */
	  if(single_quote==0) double_quote ^= 1;
	  break;
	case '\'' : /* single quote */
	  if(double_quote==0) single_quote ^= 1;
	  break;
	case '(':
	case '[':
	case '{':
	  if((double_quote==0)&& (single_quote==0)) nest++;
	  break;
	case ')':
	case ']':
	case '}':
	  if((double_quote==0)&& (single_quote==0)) nest--;
	  break;
	case '\\':
#ifndef G__OLDIMPLEMENTATION1340
	  result7[ig35++] = item[ig15++];
#else
	  fpara.parameter[fpara.paran][ig35++]=item[ig15++];
#endif
	  break;
#ifndef G__OLDIMPLEMENTATION1705
	case '<':
	  if(double_quote==0&&single_quote==0) {
	    result7[ig35]=0;
	    if(0==strcmp(result7,"operator") ||
#ifndef G__OLDIMPLEMENTATION1772
               tmpltnest || 
#endif
	       G__defined_templateclass(result7) ) ++tmpltnest;
	  }
	  break;
	case '>':
	  if(double_quote==0&&single_quote==0) {
	    if(tmpltnest) --tmpltnest;
	  }
	  break;
#endif
	}
#ifndef G__OLDIMPLEMENTATION1340
	result7[ig35++] = item[ig15++];
#else
	fpara.parameter[fpara.paran][ig35++]=item[ig15++];
#endif
#ifndef G__OLDIMPLEMENTATION1036
	if(ig35>=G__ONELINE-1) {
#ifndef G__OLDIMPLEMENTATION1340
	  if(result7[0]=='"') {
#else
	  if(fpara.parameter[fpara.paran][0]=='"') {
#endif
	    G__value bufv;
	    char bufx[G__LONGLINE];
#ifndef G__OLDIMPLEMENTATION1340
	    strncpy(bufx,result7,G__ONELINE-1);
#else
	    strncpy(bufx,fpara.parameter[fpara.paran],G__ONELINE-1);
#endif
	    while((((item[ig15]!=',')&&(item[ig15]!=')'))||
		   (nest>0)||(single_quote>0)||
		   (double_quote>0))&&(ig15<lenitem)) {
	      switch(item[ig15]) {
	      case '"' : /* double quote */
		if(single_quote==0) double_quote ^= 1;
		break;
	      case '\'' : /* single quote */
		if(double_quote==0) single_quote ^= 1;
		break;
	      case '(':
	      case '[':
	      case '{':
		if((double_quote==0)&& (single_quote==0)) nest++;
		break;
	      case ')':
	      case ']':
	      case '}':
		if((double_quote==0)&& (single_quote==0)) nest--;
		break;
	      case '\\':
		bufx[ig35++]=item[ig15++];
		break;
	      }
	      bufx[ig35++]=item[ig15++];
	      if(ig35>=G__LONGLINE-1) {
		G__genericerror("Limitation: Too long function argument");
		return(G__null);
	      }
	    }
	    bufx[ig35]=0;
	    bufv = G__strip_quotation(bufx);
#ifndef G__OLDIMPLEMENTATION1340
	    sprintf(result7,"(char*)(%ld)",bufv.obj.i);
#else
	    sprintf(fpara.parameter[fpara.paran],"(char*)(%ld)",bufv.obj.i);
#endif
#ifndef G__OLDIMPLEMENTATION1340
	    ig35=strlen(result7)+1;
#else
	    ig35=strlen(fpara.parameter[fpara.paran])+1;
#endif
	    break;
	  }
#ifndef G__OLDIMPLEMENTATION1340
	  else if(ig35>G__LONGLINE-1) {
	    G__fprinterr(G__serr,
               "Limitation: length of one function argument be less than %d"
		    ,G__LONGLINE);
	    G__genericerror((char*)NULL);
	    G__fprinterr(G__serr,"Use temp variable as workaround.\n");
	    *known3=1;
	    return(G__null);
	  }
#endif
	  else {
#ifndef G__OLDIMPLEMENTATION1340
	    overflowflag=1;
#else
	    G__fprinterr(G__serr,
    "Limitation: length of one function argument be less than %d",G__ONELINE);
	    G__genericerror((char*)NULL);
	    G__fprinterr(G__serr,"Use temp variable as workaround.\n");
	    *known3=1;
	    return(G__null);
#endif
	  }
	}
#endif
      }
      /*************************************************
       * if ')' is found at the middle of expression,
       * this should be casting or pointer to function
       *
       *  v                    v            <-- this makes
       *  (type)expression  or (*p_func)();    castflag=1
       *       ^                       ^    <-- this makes
       *                                       castflag=2
       *************************************************/
      if((item[ig15]==')')&&(ig15<lenitem-1)) {
	if(1==castflag) {
#ifndef G__OLDIMPLEMENTATION407
	  if(('-'==item[ig15+1]&&'>'==item[ig15+2]) || '.'==item[ig15+1]) 
	    castflag=3;
#else
	  if('-'==item[ig15+1] || '.'==item[ig15+1]) castflag=3;
#endif
	  else                                       castflag=2;
	}
#ifndef G__OLDIMPLEMENTATION407
	else if(('-'==item[ig15+1]&&'>'==item[ig15+2]) ||'.'==item[ig15+1]) {
#else
	else if('-'==item[ig15+1]||'.'==item[ig15+1]) {
#endif
	  castflag=3;
	  base1=ig15+1;
	}
	else if(item[ig15+1]=='[') {
	  nindex=fpara.paran+1;
	}
#ifndef G__OLDIMPLEMENTATION1150
        else if(funcname[0] && isalnum(item[ig15+1])) {
	  G__fprinterr(G__serr,"Error: %s  Syntax error?",item);
	  /* G__genericerror((char*)NULL); , avoid risk of side-effect */
	  G__printlinenum();
	}
#endif
#ifndef G__OLDIMPLEMENTATION1871
	else {
	  ++ig15;
	  result7[ig35]='\0';
	  strcpy(fpara.parameter[fpara.paran],result7);
#ifndef G__OLDIMPLEMENTATION1876
	  if(ig35) fpara.parameter[++fpara.paran][0]='\0';
#else
	  fpara.parameter[++fpara.paran][0]='\0';
#endif
	  for(itmp=0;itmp<fpara.paran;itmp++) {
	    fpara.para[itmp] = G__getexpr(fpara.parameter[itmp]);
	  }
	  if(G__parenthesisovldobj(&result3,presult,"operator()"
				   ,&fpara,G__TRYNORMAL)) {
#ifndef G__OLDIMPLEMENTATION1883
	    *known3=1;
#endif
	    return(G__operatorfunction(&result3,item+ig15+1,known3
				       ,result7,funcname));
	  }
	}
#endif
      }
      
      /*************************************************
       * set null char to parameter list buffer.
       *************************************************/
      ig15++;
#ifndef G__OLDIMPLEMENTATION1340
      result7[ig35]='\0';
      if(ig35<G__ONELINE) {
	strcpy(fpara.parameter[fpara.paran],result7);
      }
      else {
#ifndef G__OLDIMPLEMENTATION1824
	strcpy(fpara.parameter[fpara.paran],"@");
	fpara.para[fpara.paran] = G__getexpr(result7);
#endif
      }
#ifndef G__OLDIMPLEMENTATION1876
      if(ig35) fpara.parameter[++fpara.paran][0]='\0';
#else
      fpara.parameter[++fpara.paran][0]='\0';
#endif
#else
      fpara.parameter[fpara.paran++][ig35]='\0';
      fpara.parameter[fpara.paran][0]='\0';
#endif
    }
  }

#ifndef G__OLDIMPLEMENTATION2120
#ifdef G__ASM
  if(G__asm_noverflow) G__gen_PUSHSTROS_SETSTROS();
#endif
#endif

  for(itmp=0;itmp<fpara.paran;itmp++) {
    fpara.para[itmp] = G__getexpr(fpara.parameter[itmp]);
  }
#ifndef G__OLDIMPLEMENTATION2120
  G__parenthesisovldobj(&result3,presult,"operator()",&fpara,1);
#else
  G__parenthesisovldobj(&result3,presult,"operator()",&fpara,G__TRYNORMAL);
#endif

  return(result3);
}
#endif

#ifndef G__OLDIMPLEMENTATION1902
/******************************************************************
* G__value G__getfunction_libp
*
*
******************************************************************/
G__value G__getfunction_libp(item,funcname,libp,known3,memfunc_flag)
char* item;
char* funcname;
struct G__param *libp;
int *known3;
int memfunc_flag;
{
  G__value result3;
  char result7[G__LONGLINE];
  int ig15,ipara;
#ifndef G__OLDIMPLEMENTATION809
  static struct G__param *p2ffpara=(struct G__param*)NULL;
#endif
  int hash;
  int funcmatch;
  int i,classhash;
  long store_struct_offset;
  int store_tagnum;
  int store_exec_memberfunc;
  int store_asm_noverflow;
  int store_memberfunc_tagnum;
  long store_memberfunc_struct_offset;
#ifndef G__OLDIMPLEMENTATION1741
  int store_def_tagnum,store_tagdefining;
#endif
  int tempstore;
#ifndef G__OLDIMPLEMENTATION403
  char *pfparam;
  struct G__var_array *var;
#endif
#ifndef G__OLDIMPLEMENTATION405
  int nindex=0;
#endif
#ifndef G__OLDIMPLEMENTATION1515
  int oprp=0;
#endif
#ifndef G__OLDIMPLEMENTATION1570
  int store_cp_asm=0;
#endif

  store_struct_offset = G__store_struct_offset;
  store_tagnum = G__tagnum;
  store_exec_memberfunc = G__exec_memberfunc;
  store_asm_noverflow = G__asm_noverflow;
  store_memberfunc_tagnum = G__memberfunc_tagnum;
  store_memberfunc_struct_offset=G__memberfunc_struct_offset;

  *known3 = 1; /* temporary solution */
  
  /* scope operator ::f() , A::B::f()
   * note, 
   *    G__exec_memberfunc restored at return memfunc_flag is local, 
   *   there should be no problem modifying these variables.
   *    store_struct_offset and store_tagnum are only used in the
   *   explicit type conversion section.  It is OK to use them here
   *   independently.
   */
  store_struct_offset=G__store_struct_offset;
  store_tagnum=G__tagnum;
#ifndef G__OLDIMPLEMENTATION1741
  store_def_tagnum = G__def_tagnum;
  store_tagdefining = G__tagdefining;
#endif
  switch(G__scopeoperator(funcname,&hash,&G__store_struct_offset,&G__tagnum)){
  case G__GLOBALSCOPE: /* global scope */
    G__exec_memberfunc=0;
    memfunc_flag=G__TRYNORMAL;
#ifndef G__OLDIMPLEMENTATION1741
    G__def_tagnum = -1;
    G__tagdefining = -1;
#endif
    break;
  case G__CLASSSCOPE: /* class scope */
#ifndef G__OLDIMPLEMENTATION1101
    memfunc_flag=G__CALLSTATICMEMFUNC;
#else
    memfunc_flag=G__CALLMEMFUNC;
#endif
#ifndef G__OLDIMPLEMENTATION1741
    /* This looks very risky */
    G__def_tagnum = -1;
    G__tagdefining = -1;
    G__exec_memberfunc=1;
    G__memberfunc_tagnum = G__tagnum;
#endif
    break;
  }
  
#ifdef G__DUMPFILE
  /***************************************************************
   * dump that a function is called 
   ***************************************************************/
  if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
    for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
    fprintf(G__dumpfile,"%s(",funcname);
    for(ipara=1;ipara<= libp->paran;ipara++) {
      if(ipara!=1) fprintf(G__dumpfile,",");
      G__valuemonitor(libp->para[ipara-1],result7);
      fprintf(G__dumpfile,"%s",result7);
    }
    fprintf(G__dumpfile,");/*%s %d,%lx %lx*/\n" 
	    ,G__ifile.name,G__ifile.line_number
	    ,store_struct_offset,G__store_struct_offset);
    G__dumpspace += 3;
    
  }
#endif

  /********************************************************************
   * Begin Loop to resolve overloaded function
   ********************************************************************/
  for(funcmatch=G__EXACT;funcmatch<=G__USERCONV;funcmatch++) {
    
    /***************************************************************
     * search for interpreted member function
     * if(G__exec_memberfunc)     ==>  memfunc();
     * G__TRYNORMAL!=memfunc_flag ==>  a.memfunc();
     ***************************************************************/
    if(G__exec_memberfunc || G__TRYNORMAL!=memfunc_flag) {
#ifndef G__OLDIMPLEMENTATION589
      int local_tagnum;
      if(G__exec_memberfunc&&-1==G__tagnum)
	local_tagnum = G__memberfunc_tagnum;
      else 
	local_tagnum = G__tagnum;
      if(-1!=G__tagnum) G__incsetup_memfunc(G__tagnum);
#else
      G__ASSERT(0<=G__tagnum);
      G__incsetup_memfunc(G__tagnum);
#endif
#ifndef G__OLDIMPLEMENTATION589
      if(-1!=local_tagnum&& G__interpret_func(&result3,funcname,libp,hash
			   ,G__struct.memfunc[local_tagnum]
			   ,funcmatch,memfunc_flag)==1 ) {
#else
      if(G__interpret_func(&result3,funcname,libp,hash
			   ,G__struct.memfunc[G__tagnum]
			   ,funcmatch,memfunc_flag)==1 ) {
#endif
#ifdef G__DUMPFILE
	if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
	  G__dumpspace -= 3;
	  for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
	  G__valuemonitor(result3,result7);
	  fprintf(G__dumpfile,"/* return(inp) %s.%s()=%s*/\n"
		  ,G__struct.name[G__tagnum],funcname,result7);
	}
#endif
#ifndef G__OLDIMPLEMENTATION517
	if(G__store_struct_offset!=store_struct_offset) 
	  G__gen_addstros(store_struct_offset-G__store_struct_offset);
#endif
	G__store_struct_offset = store_struct_offset;
	G__tagnum = store_tagnum;
#ifndef G__OLDIMPLEMENTATION1741
	G__def_tagnum = store_def_tagnum;
	G__tagdefining = store_tagdefining;
#endif
	G__exec_memberfunc = store_exec_memberfunc;
	G__memberfunc_tagnum=store_memberfunc_tagnum;
	G__memberfunc_struct_offset=store_memberfunc_struct_offset;
	G__setclassdebugcond(G__memberfunc_tagnum,0);
#ifndef G__OLDIMPLEMENTATION405
	if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
	   (isupper(result3.type)||'u'==result3.type)
#else
	   isupper(result3.type)
#endif
	   ) {
	  G__getindexedvalue(&result3,libp->parameter[nindex]);
	}
#endif
#ifndef G__OLDIMPLEMENTATION1515
	if(oprp) *known3 = G__additional_parenthesis(&result3,libp);
#endif
	return(result3);
      }
#define G__OLDIMPLEMENTATION1159
#ifndef G__OLDIMPLEMENTATION1159
      /* STILL WORKING , DO NOT RELEASE THIS */
      /******************************************************************
       * Search template function
       ******************************************************************/
      if((G__EXACT==funcmatch||G__USERCONV==funcmatch)) {
        int storeX_exec_memberfunc=G__exec_memberfunc;
        int storeX_memberfunc_tagnum=G__memberfunc_tagnum;
        G__exec_memberfunc = 1;
        G__memberfunc_tagnum = local_tagnum;
	if(G__templatefunc(&result3,funcname,libp,hash,funcmatch)==1){
#ifdef G__DUMPFILE
          if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
	    G__dumpspace -= 3;
	    for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
            G__valuemonitor(result3,result7);
	    fprintf(G__dumpfile,"/* return(lib) %s()=%s */\n"
                    ,funcname,result7);
          }
#endif
      
          G__exec_memberfunc = store_exec_memberfunc;
          G__memberfunc_tagnum=store_memberfunc_tagnum;
          G__memberfunc_struct_offset=store_memberfunc_struct_offset;
#ifndef G__OLDIMPLEMENTATION1515
	  if(oprp) *known3 = G__additional_parenthesis(&result3,libp);
#endif
          return(result3);
        }
        G__exec_memberfunc=storeX_exec_memberfunc;
        G__memberfunc_tagnum=storeX_memberfunc_tagnum;
      }
#endif
    }
    
    
    /***************************************************************
     * If memberfunction is called explicitly by clarifying scope
     * don't examine global function and exit from G__getfunction().
     * There are 2 cases                   G__exec_memberfunc
     *   obj.memfunc();                            1
     *   X::memfunc();                             1
     *    X();              constructor            2
     *   ~X();              destructor             2
     * If G__exec_memberfunc==2, don't display error message.
     ***************************************************************/
    /* If searching only member function */
    if(memfunc_flag
#ifndef G__OLDIMPLEMENTATION1104
       &&(G__store_struct_offset||G__CALLSTATICMEMFUNC!=memfunc_flag)
#endif
       ) {
      
      G__exec_memberfunc = store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
      
      /* If the last resolution of overloading failed */
      if(funcmatch==G__USERCONV) {
	
	if(G__TRYDESTRUCTOR==memfunc_flag) {
	  /* destructor for base calss and class members */
#ifdef G__ASM
#ifdef G__SECURITY
	  store_asm_noverflow = G__asm_noverflow;
	  if(G__security&G__SECURE_GARBAGECOLLECTION) G__abortbytecode();
#endif
#endif
#ifdef G__VIRTUALBASE
	  if(G__CPPLINK!=G__struct.iscpplink[G__tagnum]) G__basedestructor();
#else
	  G__basedestructor();
#endif
#ifdef G__ASM
#ifdef G__SECURITY
	  G__asm_noverflow = store_asm_noverflow;
#endif
#endif
	}
	else {
	  switch(memfunc_flag) {
	  case G__CALLCONSTRUCTOR:
	  case G__TRYCONSTRUCTOR:
#ifndef G__OLDIMPLEMENTATION1250
	  case G__TRYIMPLICITCONSTRUCTOR:
#endif
	    /* constructor for base class and class members default 
	     * constructor only */
#ifdef G__VIRTUALBASE
	    if(G__CPPLINK!=G__struct.iscpplink[G__tagnum])
	      G__baseconstructor(0 ,(struct G__baseparam *)NULL);
#else
	    G__baseconstructor(0 ,(struct G__baseparam *)NULL);
#endif
	  }
	}
	G__exec_memberfunc = store_exec_memberfunc;
	G__memberfunc_tagnum=store_memberfunc_tagnum;
	G__memberfunc_struct_offset=store_memberfunc_struct_offset;
	
	
	*known3=0;
	switch(memfunc_flag) {
	case G__CALLMEMFUNC:
	  if(G__parenthesisovld(&result3,funcname,libp,G__CALLMEMFUNC)) {
	    *known3=1;
#ifndef G__OLDIMPLEMENTATION517
	    if(G__store_struct_offset!=store_struct_offset) 
	      G__gen_addstros(store_struct_offset-G__store_struct_offset);
#endif
	    G__store_struct_offset = store_struct_offset;
	    G__tagnum = store_tagnum;
#ifndef G__OLDIMPLEMENTATION1741
	    G__def_tagnum = store_def_tagnum;
	    G__tagdefining = store_tagdefining;
#endif
#ifndef G__OLDIMPLEMENTATION405
	    if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
	       (isupper(result3.type)||'u'==result3.type)
#else
	       isupper(result3.type)
#endif
	       ) {
	      G__getindexedvalue(&result3,libp->parameter[nindex]);
	    }
#endif
#ifndef G__OLDIMPLEMENTATION1515
	    if(oprp) *known3 = G__additional_parenthesis(&result3,libp);
#endif
	    return(result3);
	  }
#ifndef G__OLDIMPLEMENTATION733
	  if('~'==funcname[0]) {
	    *known3=1;
	    return(G__null);
	  }
#endif
#ifndef G__OLDIMPLEMENTATION1729
	  /******************************************************************
	   * Search template function
	   ******************************************************************/
	  G__exec_memberfunc = 1;
	  G__memberfunc_tagnum=G__tagnum;
	  G__memberfunc_struct_offset=G__store_struct_offset;
	  if((G__EXACT==funcmatch||G__USERCONV==funcmatch)&&
	     G__templatefunc(&result3,funcname,libp,hash,funcmatch)==1){
	    
#ifdef G__DUMPFILE
	    if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
	      G__dumpspace -= 3;
	      for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
	      G__valuemonitor(result3,result7);
	      fprintf(G__dumpfile ,"/* return(lib) %s()=%s */\n"
		      ,funcname,result7);
	    }
#endif
	    G__exec_memberfunc = store_exec_memberfunc;
	    G__memberfunc_tagnum=store_memberfunc_tagnum;
	    G__memberfunc_struct_offset=store_memberfunc_struct_offset;
	    /* don't know why if(oprp) is needed, copied from line 2111 */
	    if(oprp) *known3 = G__additional_parenthesis(&result3,libp);
	    else *known3=1;
	    return(result3);
	  }
	  G__exec_memberfunc = store_exec_memberfunc;
	  G__memberfunc_tagnum=store_memberfunc_tagnum;
	  G__memberfunc_struct_offset=store_memberfunc_struct_offset;
#endif /* 1729 */
	case G__CALLCONSTRUCTOR:
#ifndef G__OLDIMPLEMENTATION1376
	  if(G__NOLINK > G__globalcomp) break;
#endif
#ifndef G__OLDIMPLEMENTATION1505
	  if(!G__no_exec_compile || G__asm_noverflow) {
#endif
#ifndef G__OLDIMPLEMENTATION1185
#ifndef G__OLDIMPLEMENTATION1775
	    if (0==G__const_noerror) 
	      G__fprinterr(G__serr, "Error: Can't call %s::%s in current scope"
			   ,G__struct.name[G__tagnum],item);
#else
	    G__fprinterr(G__serr, "Error: Can't call %s::%s in current scope"
			 ,G__struct.name[G__tagnum],item);
#endif
#else
	    G__fprinterr(G__serr, "Error: Can't call %s::%s() in current scope"
			 ,G__struct.name[G__tagnum],funcname);
#endif
	    G__genericerror((char*)NULL);
#ifndef G__OLDIMPLEMENTATION1505
	  }
#endif
	  store_exec_memberfunc=G__exec_memberfunc;
	  G__exec_memberfunc=1;
#ifndef G__OLDIMPLEMENTATION1103
	  if(0==G__const_noerror
#ifndef G__OLDIMPLEMENTATION1505
	     && (!G__no_exec_compile || G__asm_noverflow)
#endif
	     ) {
#endif
	    G__fprinterr(G__serr,"Possible candidates are...\n");
#ifndef G__OLDIMPLEMENTATION1079
	    {
	      char itemtmp[G__LONGLINE];
	      sprintf(itemtmp,"%s::%s",G__struct.name[G__tagnum],funcname);
	      G__display_proto(G__serr,itemtmp);
	    }
#else
	    G__listfunc(G__serr,G__PUBLIC_PROTECTED_PRIVATE,funcname
		      ,G__struct.memfunc[G__tagnum]);
#endif
#ifndef G__OLDIMPLEMENTATION1103
	  }
#endif
	  G__exec_memberfunc=store_exec_memberfunc;
	}
#ifdef G__DUMPFILE
	if(G__dumpfile!=NULL && 0==G__no_exec_compile) G__dumpspace -= 3;
#endif
	
#ifndef G__OLDIMPLEMENTATION517
	if(G__store_struct_offset!=store_struct_offset) 
	  G__gen_addstros(store_struct_offset-G__store_struct_offset);
#endif
	G__store_struct_offset = store_struct_offset;
	G__tagnum = store_tagnum;
#ifndef G__OLDIMPLEMENTATION1741
	G__def_tagnum = store_def_tagnum;
	G__tagdefining = store_tagdefining;
#endif
	if(libp->paran && 'u'==libp->para[0].type&&
#ifndef G__OLDIMPLEMENTATION1250
	   (G__TRYCONSTRUCTOR==memfunc_flag||
	    G__TRYIMPLICITCONSTRUCTOR==memfunc_flag)
#else
	   G__TRYCONSTRUCTOR==memfunc_flag
#endif
	   ) {
	  /* in case of copy constructor not found */
	  return(libp->para[0]);
	}
	else {
	  return(G__null);
	}
      }
      /* ELSE next level overloaded function resolution */
      continue;
    }
    
    
    
    /***************************************************************
     * reset G__exec_memberfunc for global function.
     * Original value(store_exec_memberfunc) is restored when exit 
     * from this function
     ***************************************************************/
    tempstore = G__exec_memberfunc;
    G__exec_memberfunc = 0;
    
    
    
    /***************************************************************
     * search for interpreted global function
     *
     ***************************************************************/
    if( G__interpret_func(&result3,funcname,libp,hash,G__p_ifunc
			  ,funcmatch,G__TRYNORMAL)==1 ) {
#ifdef G__DUMPFILE
      if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
	G__dumpspace -= 3;
	for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
	G__valuemonitor(result3,result7);
	fprintf(G__dumpfile ,"/* return(inp) %s()=%s*/\n" ,funcname,result7);
      }
#endif
      
      G__exec_memberfunc = store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
      G__setclassdebugcond(G__memberfunc_tagnum,0);
#ifndef G__OLDIMPLEMENTATION405
      if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
	 (isupper(result3.type)||'u'==result3.type)
#else
	 isupper(result3.type)
#endif
	 ) {
	G__getindexedvalue(&result3,libp->parameter[nindex]);
      }
#endif
#ifndef G__OLDIMPLEMENTATION1515
      if(oprp) *known3 = G__additional_parenthesis(&result3,libp);
#endif
      return(result3);
    }

    G__exec_memberfunc = tempstore;
    
    /* there is no function overload resolution after this point,
     * thus, if not found in G__EXACT trial, there is no chance to
     * find matched function in consequitive search
     */
    if(G__USERCONV==funcmatch) goto templatefunc;
    if(G__EXACT!=funcmatch) continue;
    
    
    
    /***************************************************************
     * search for compiled(archived) function
     *
     ***************************************************************/
    if( G__compiled_func(&result3,funcname,libp,hash)==1 ) {
      
#ifdef G__ASM
      if(G__asm_noverflow) {
	/****************************************
	 * LD_FUNC (compiled)
	 ****************************************/
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,
			       "%3x: LD_FUNC compiled %s paran=%d\n"
			       ,G__asm_cp,funcname,libp->paran);
#endif
	G__asm_inst[G__asm_cp]=G__LD_FUNC;
	G__asm_inst[G__asm_cp+1] = (long)(&G__asm_name[G__asm_name_p]);
	G__asm_inst[G__asm_cp+2]=hash;
	G__asm_inst[G__asm_cp+3]=libp->paran;
	G__asm_inst[G__asm_cp+4]=(long)G__compiled_func;
	if(G__asm_name_p+strlen(funcname)+1<G__ASM_FUNCNAMEBUF) {
	  strcpy(G__asm_name+G__asm_name_p,funcname);
	  G__asm_name_p += strlen(funcname)+1;
	  G__inc_cp_asm(5,0);
	}
	else {
	  G__abortbytecode();
#ifdef G__ASM_DBG
	  if(G__asm_dbg) {
	    G__fprinterr(G__serr,"COMPILE ABORT function name buffer overflow");
	    G__printlinenum();
	  }
#endif
	}
      }
#endif /* G__ASM */
      
#ifdef G__DUMPFILE
      if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
	G__dumpspace -= 3;
	for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
	G__valuemonitor(result3,result7);
	fprintf(G__dumpfile ,"/* return(cmp) %s()=%s */\n" ,funcname,result7);
      }
#endif
      
      G__exec_memberfunc = store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
#ifndef G__OLDIMPLEMENTATION405
      if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
	 (isupper(result3.type)||'u'==result3.type)
#else
	 isupper(result3.type)
#endif
	 ) {
	G__getindexedvalue(&result3,libp->parameter[nindex]);
      }
#endif
      return(result3);
    }
    
    
    /***************************************************************
     * search for library function which are included in G__ci.c
     *
     ***************************************************************/
    if( G__library_func(&result3,funcname,libp,hash)==1 ) {
#ifndef G__OLDIMPLEMENTATION1595
      if(G__no_exec_compile) result3.type = 'i';
#endif
#ifdef G__ASM
      if(G__asm_noverflow) {
	/****************************************
	 * LD_FUNC (library)
	 ****************************************/
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,
			       "%3x: LD_FUNC library %s paran=%d\n"
			       ,G__asm_cp,funcname,libp->paran);
#endif
	G__asm_inst[G__asm_cp]=G__LD_FUNC;
	G__asm_inst[G__asm_cp+1] = (long)(&G__asm_name[G__asm_name_p]);
	G__asm_inst[G__asm_cp+2]=hash;
	G__asm_inst[G__asm_cp+3]=libp->paran;
	G__asm_inst[G__asm_cp+4]=(long)G__library_func;
	if(G__asm_name_p+strlen(funcname)+1<G__ASM_FUNCNAMEBUF) {
	  strcpy(G__asm_name+G__asm_name_p,funcname);
	  G__asm_name_p += strlen(funcname)+1;
	  G__inc_cp_asm(5,0);
	}
	else {
	  G__abortbytecode();
#ifdef G__ASM_DBG
	  if(G__asm_dbg) 
	    G__fprinterr(G__serr,"COMPILE ABORT function name buffer overflow");
	    G__printlinenum();
#endif
	}
      }
#endif /* G__ASM */
      
#ifdef G__DUMPFILE
      if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
	G__dumpspace -= 3;
	for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
	G__valuemonitor(result3,result7);
	fprintf(G__dumpfile ,"/* return(lib) %s()=%s */\n" ,funcname,result7);
      }
#endif
	    
      G__exec_memberfunc = store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
#ifndef G__OLDIMPLEMENTATION405
      if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
	 (isupper(result3.type)||'u'==result3.type)
#else
	 isupper(result3.type)
#endif
	 ) {
	G__getindexedvalue(&result3,libp->parameter[nindex]);
      }
#endif
      return(result3);
    }
    
#ifdef G__TEMPLATEFUNC
    templatefunc:
    /******************************************************************
     * Search template function
     ******************************************************************/
    if((G__EXACT==funcmatch||G__USERCONV==funcmatch)&&
       G__templatefunc(&result3,funcname,libp,hash,funcmatch)==1){

#ifdef G__DUMPFILE
      if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
	G__dumpspace -= 3;
	for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
	G__valuemonitor(result3,result7);
	fprintf(G__dumpfile ,"/* return(lib) %s()=%s */\n" ,funcname,result7);
      }
#endif
      
      G__exec_memberfunc = store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
#ifndef G__OLDIMPLEMENTATION1515
      if(oprp) *known3 = G__additional_parenthesis(&result3,libp);
#ifndef G__OLDIMPLEMENTATION1729
      else *known3=1; /* don't know why this was missing */
#endif
#endif
      return(result3);
    }
#endif /* G__TEMPLATEFUNC */
    
    /******************************************************************
     * End Loop to resolve overloaded function
     ******************************************************************/
    
    /* next_overload_match:
       ; */
    
  }

  
  /********************************************************************
   * Explicit type conversion by searching constructors
   ********************************************************************/
  if(G__TRYNORMAL==memfunc_flag
#ifndef G__OLDIMPLEMENTATION1104
     ||G__CALLSTATICMEMFUNC==memfunc_flag
#endif
     ) {
#ifndef G__OLDIMPLEMENTATION985
    int store_var_typeX = G__var_type;
#endif
    i=G__defined_typename(funcname);
#ifndef G__OLDIMPLEMENTATION985
    G__var_type = store_var_typeX;
#endif
    if(-1!=i) {
      if(-1!=G__newtype.tagnum[i]) {
	strcpy(funcname,G__struct.name[G__newtype.tagnum[i]]);
      }
      else {
#ifndef G__OLDIMPLEMENTATION1188
	char ttt[G__ONELINE];
	result3 = libp->para[0];
	if(G__fundamental_conversion_operator(G__newtype.type[i],-1 
					      ,i ,G__newtype.reftype[i],0
					      ,&result3,ttt)) {
	  *known3=1;
#ifndef G__OLDIMPLEMENTATION1515
	  if(oprp) *known3 = G__additional_parenthesis(&result3,libp);
#endif
	  return(result3);
	}
#endif
	strcpy(funcname,G__type2string(G__newtype.type[i]
				       ,G__newtype.tagnum[i] ,-1
				       ,G__newtype.reftype[i] ,0));
      }
      G__hash(funcname,hash,i);
    }

    classhash=strlen(funcname);
    i=0;
    while(i<G__struct.alltag) {
      if((G__struct.hash[i]==classhash)&&
	 (strcmp(G__struct.name[i],funcname)==0)
#ifdef G__OLDIMPLEMENTATION1386
#ifndef G__OLDIMPLEMENTATION1332
	  &&'e'!=G__struct.type[i]
#endif
#endif
	 ) {
#ifndef G__OLDIMPLEMENTATION1386
	if('e'==G__struct.type[i] && 
	   -1!=libp->para[0].tagnum &&
	   'e'==G__struct.type[libp->para[0].tagnum]) {
	  return(libp->para[0]);
	}
#endif
	store_struct_offset=G__store_struct_offset;
	
	/* questionable part */
	/* store_exec_memfunc=G__exec_memberfunc; */
	
	store_tagnum=G__tagnum;
	G__tagnum=i;
	if(G__CPPLINK!=G__struct.iscpplink[G__tagnum]) {
	  G__alloc_tempobject(G__tagnum,-1);
	  G__store_struct_offset=G__p_tempbuf->obj.obj.i;
#ifndef G__OLDIMPLEMENTATION843
#ifdef G__ASM
	  if(G__asm_noverflow) {
#ifndef G__OLDIMPLEMENTATION2111
	    if(G__throwingexception) {
#ifdef G__ASM_DBG
	      if(G__asm_dbg) {
	        G__fprinterr(G__serr,"%3x: ALLOCEXCEPTION %d\n"
                            ,G__asm_cp,G__tagnum);
	      }
#endif
	      G__asm_inst[G__asm_cp]=G__ALLOCEXCEPTION;
	      G__asm_inst[G__asm_cp+1]=G__tagnum;
	      G__inc_cp_asm(2,0);
	    }
            else {
#ifdef G__ASM_DBG
	      if(G__asm_dbg) {
	        G__fprinterr(G__serr,"%3x: ALLOCTEMP %d\n"
                             ,G__asm_cp,G__tagnum);
	        G__fprinterr(G__serr,"%3x: SETTEMP\n",G__asm_cp);
	      }
#endif
	      G__asm_inst[G__asm_cp]=G__ALLOCTEMP;
	      G__asm_inst[G__asm_cp+1]=G__tagnum;
	      G__asm_inst[G__asm_cp+2]=G__SETTEMP;
	      G__inc_cp_asm(3,0);
	    }
#else /* ON2111 */
#ifdef G__ASM_DBG
	    if(G__asm_dbg) {
	      G__fprinterr(G__serr,"%3x: ALLOCTEMP %d\n",G__asm_cp,G__tagnum);
	      G__fprinterr(G__serr,"%3x: SETTEMP\n",G__asm_cp);
	    }
#endif
	    G__asm_inst[G__asm_cp]=G__ALLOCTEMP;
	    G__asm_inst[G__asm_cp+1]=G__tagnum;
	    G__asm_inst[G__asm_cp+2]=G__SETTEMP;
	    G__inc_cp_asm(3,0);
#endif /* ON2111 */
	  }
#endif
#endif
	}
	else {
	  G__store_struct_offset= G__PVOID;
	}
#ifndef G__OLDIMPLEMENTATION1341
	G__incsetup_memfunc(G__tagnum);
#endif
	for(funcmatch=G__EXACT;funcmatch<=G__USERCONV;funcmatch++) {
#ifdef G__OLDIMPLEMENTATION1341
	  G__incsetup_memfunc(G__tagnum);
#endif
	  *known3=G__interpret_func(&result3,funcname
				    ,libp,hash
				    ,G__struct.memfunc[G__tagnum]
				    ,funcmatch
				    ,G__TRYCONSTRUCTOR);
	  if(*known3) break;
	}
	if(G__CPPLINK==G__struct.iscpplink[G__tagnum]
#ifndef G__OLDIMPLEMENTATION2111
	   && !G__throwingexception
#endif
           ) {
	  G__store_tempobject(result3);
	  if(G__dispsource) {
	    G__fprinterr(G__serr,
		    "!!!Create temp object (%s)0x%lx,%d for %s()\n"
		    ,G__struct.name[G__tagnum] ,G__p_tempbuf->obj.obj.i
		    ,G__templevel ,funcname);
	  }
#ifdef G__ASM
	  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: STORETEMP\n",G__asm_cp);
#endif
	    G__asm_inst[G__asm_cp]=G__STORETEMP;
	    G__inc_cp_asm(1,0);
	  }
#endif
	}
	else {
	  result3.type='u';
	  result3.tagnum=G__tagnum;
	  result3.typenum = -1;
	  result3.obj.i=G__store_struct_offset;
	  result3.ref=G__store_struct_offset;
#ifdef G__OLDIMPLEMENTATION843      /* FOLLOWING PART SHOULD BE REMOVED */
#ifdef G__ASM                       /* WATCH FOR PROBLEM in stl/vec3.cxx */
	  /* Loop compilation of explicit conversion not ready yet, 
	   * mainly because of temporary object handling difficulty */
	  if(G__asm_noverflow) {
	    G__abortbytecode();
	    if(G__asm_dbg) {
	      G__fprinterr(G__serr,"COMPILE ABORT Explicit conversion");
	      G__printlinenum();
	    }
	  }
#endif
#endif
	}
	G__tagnum=store_tagnum;
	
	/* questionable part */
	G__exec_memberfunc = store_exec_memberfunc;
	G__memberfunc_tagnum=store_memberfunc_tagnum;
	G__memberfunc_struct_offset=store_memberfunc_struct_offset;
	
	G__store_struct_offset=store_struct_offset;
#ifndef G__OLDIMPLEMENTATION1500
#ifdef G__ASM
	if(G__asm_noverflow
#ifndef G__OLDIMPLEMENTATION2111
	   && (!G__throwingexception || 
	       (-1!=result3.tagnum &&
                G__CPPLINK!=G__struct.iscpplink[result3.tagnum]))
#endif
          ) {
	  G__asm_inst[G__asm_cp]=G__POPTEMP;
#ifndef G__OLDIMPLEMENTATION2111
	  if(G__throwingexception) G__asm_inst[G__asm_cp+1] = result3.tagnum;
          else                     G__asm_inst[G__asm_cp+1] = -1;
#else
	  G__asm_inst[G__asm_cp+1] = -1;
#endif
#ifdef G__ASM_DBG
	  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPTEMP %d\n"
                                      ,G__asm_cp,G__asm_inst[G__asm_cp+1]);
#endif
	  G__inc_cp_asm(2,0);
	}
#endif
#endif

	if(0 == *known3) {
#ifndef G__OLDIMPLEMENTATION1341
	  if(-1 != i && libp->paran==1 && -1 != libp->para[0].tagnum) {
	    long store_struct_offset = G__store_struct_offset;
	    long store_memberfunc_struct_offset = G__memberfunc_struct_offset;
	    int store_memberfunc_tagnum = G__memberfunc_tagnum;
	    int store_exec_memberfunc = G__exec_memberfunc;
	    store_tagnum = G__tagnum;
#ifndef G__OLDIMPLEMENTATION1500
	    G__inc_cp_asm(-5,0); /* cancel ALLOCTEMP, SETTEMP, POPTEMP */
#else
	    G__inc_cp_asm(-3,0); /* cancel ALLOCTEMP, SETTEMP */
#endif
	    G__pop_tempobject();
	    G__tagnum = libp->para[0].tagnum;
	    G__store_struct_offset = libp->para[0].obj.i;
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
#endif
	    sprintf(funcname,"operator %s",G__fulltagname(i,1));
	    G__hash(funcname,hash,i);
	    G__incsetup_memfunc(G__tagnum);
	    libp->paran = 0;
	    for(funcmatch=G__EXACT;funcmatch<=G__USERCONV;funcmatch++) {
	      *known3=G__interpret_func(&result3,funcname
					,libp,hash
					,G__struct.memfunc[G__tagnum]
					,funcmatch
					,G__TRYMEMFUNC);
	      if(*known3) {
#ifdef G__ASM
		if(G__asm_noverflow) {
		  G__asm_inst[G__asm_cp] = G__POPSTROS;
		  G__inc_cp_asm(1,0);
#ifdef G__ASM_DBG
		  if(G__asm_dbg) 
		    G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp-1);
#endif
		}
#endif
		break;
	      }
	    }
	    G__memberfunc_struct_offset = store_memberfunc_struct_offset;
	    G__memberfunc_tagnum = store_memberfunc_tagnum;
	    G__exec_memberfunc = store_exec_memberfunc;
	    G__tagnum=store_tagnum;
	    G__store_struct_offset = store_struct_offset;
	  }
#endif /* 1341 */
#ifndef G__OLDIMPLEMENTATION1554
	  else if(-1!=i && libp->paran==1) {
	    G__fprinterr(G__serr,"Error: No matching constructor for explicit conversion %s",item);
	    G__genericerror((char*)NULL);
	  }
#endif
#ifndef G__OLDIMPLEMENTATION641
	  /* omitted constructor, return uninitialized object */
	  *known3 = 1;
#ifndef G__OLDIMPLEMENTATION1515
	  if(oprp) *known3 = G__additional_parenthesis(&result3,libp);
#endif
	  return(result3);
#else
	  G__pop_tempobject();
#endif
	}
	else {
	  /* Return '*this' as result */
#ifndef G__OLDIMPLEMENTATION1515
	  if(oprp) *known3 = G__additional_parenthesis(&result3,libp);
#endif
	  return(result3);
	}
      }
      i++;
    } /* while(i<G__struct.alltag) */

#ifndef G__OLDIMPLEMENTATION571
    result3.ref=0;
#endif
    if(G__explicit_fundamental_typeconv(funcname,classhash,libp,&result3)) {
      *known3=1;
      G__exec_memberfunc = store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
#ifndef G__OLDIMPLEMENTATION1515
      if(oprp) *known3 = G__additional_parenthesis(&result3,libp);
#endif
      return(result3);
    }

  } /* if(G__TRYNORMAL==memfunc_flag) */

  
  if(G__parenthesisovld(&result3,funcname,libp,G__TRYNORMAL)) {
    *known3=1;
#ifndef G__OLDIMPLEMENTATION405
    if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
       (isupper(result3.type)||'u'==result3.type)
#else
       isupper(result3.type)
#endif
       ) {
      G__getindexedvalue(&result3,libp->parameter[nindex]);
    }
#endif
#ifndef G__OLDIMPLEMENTATION1871
    else if(nindex&&'u'==result3.type) {
      int len;
      strcpy(libp->parameter[0],libp->parameter[nindex]+1);
      len = strlen(libp->parameter[0]);
      if(len>1) libp->parameter[0][len-1]=0;
      libp->para[0] = G__getexpr(libp->parameter[0]);
      libp->paran=1;
      G__parenthesisovldobj(&result3,&result3,"operator[]",libp,G__TRYNORMAL);
    }
#endif
#ifndef G__OLDIMPLEMENTATION1515
    if(oprp) *known3 = G__additional_parenthesis(&result3,libp);
#endif
    return(result3);
  }

#ifndef G__OLDIMPLEMENTATION403
  /********************************************************************
  * pointer to function described like normal function
  * int (*p2f)(void);  p2f(); 
  ********************************************************************/
  var = G__getvarentry(funcname,hash,&ig15,&G__global,G__p_local);
  if(var) {
    sprintf(result7,"*%s",funcname);
    *known3=0;
    pfparam=strchr(item,'(');
#ifndef G__OLDIMPLEMENTATION809
    p2ffpara = libp;
#endif
    result3=G__pointer2func((G__value*)NULL,result7,pfparam,known3);
#ifndef G__OLDIMPLEMENTATION809
    p2ffpara=(struct G__param*)NULL;
#endif
    if(*known3) {
      G__exec_memberfunc = store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
#ifndef G__OLDIMPLEMENTATION405
      if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
	 (isupper(result3.type)||'u'==result3.type)
#else
	 isupper(result3.type)
#endif
	 ) {
	G__getindexedvalue(&result3,libp->parameter[nindex]);
      }
#endif
#ifndef G__OLDIMPLEMENTATION1515
      if(oprp) *known3 = G__additional_parenthesis(&result3,libp);
#endif
      return(result3);
    }
  }
#endif
  
  *known3=0;
  /* bug fix, together with G__OLDIMPLEMENTATION29 */
#ifdef G__DUMPFILE
  if(G__dumpfile!=NULL && 0==G__no_exec_compile) G__dumpspace -= 3;
#endif
  
  G__exec_memberfunc = store_exec_memberfunc;
  G__memberfunc_tagnum=store_memberfunc_tagnum;
  G__memberfunc_struct_offset=store_memberfunc_struct_offset;
  

  if(!G__oprovld) {
#ifndef G__OLDIMPLEMENTATION1570
    if(G__asm_noverflow && libp->paran) {
      G__asm_cp=store_cp_asm;
    }
    G__asm_clear_mask = 1;
#endif
    result3 = G__execfuncmacro(item,known3);
#ifndef G__OLDIMPLEMENTATION1570
    G__asm_clear_mask = 0;
#endif
    if(*known3) {
#ifndef G__OLDIMPLEMENTATION405
      if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
	 (isupper(result3.type)||'u'==result3.type)
#else
	 isupper(result3.type)
#endif
	 ) {
	G__getindexedvalue(&result3,libp->parameter[nindex]);
      }
#endif
#ifndef G__OLDIMPLEMENTATION1515
      if(oprp) *known3 = G__additional_parenthesis(&result3,libp);
#endif
      return(result3);
    }
  }

  
  return(G__null);
  /* return(result3); */
  
}
#endif /* 1902 */

/******************************************************************
* G__value G__getfunction(item,known3,memfunc_flag)
*
*
******************************************************************/
G__value G__getfunction(item,known3,memfunc_flag)
char *item;
int *known3;
int memfunc_flag;
{
  G__value result3 = G__null;
  char funcname[G__LONGLINE];
#ifndef G__OLDIMPLEMENTATION1340
  int overflowflag=0;
  char result7[G__LONGLINE];
#else
  char result7[G__ONELINE];
#endif
  int ig15,ig35,ipara;
  int lenitem,nest=0;
  int single_quote=0,double_quote=0;
  struct G__param fpara;
#ifndef G__OLDIMPLEMENTATION809
  static struct G__param *p2ffpara=(struct G__param*)NULL;
#endif
  int hash;
  short castflag;
  int funcmatch;
  int i,classhash;
#ifndef G__OLDIMPLEMENTATION1613
  long store_globalvarpointer;
#endif
  long store_struct_offset;
  int store_tagnum;
  int store_exec_memberfunc;
  int store_asm_noverflow;
  int store_var_type;
  int store_memberfunc_tagnum;
  long store_memberfunc_struct_offset;
  int store_memberfunc_var_type;
#ifndef G__OLDIMPLEMENTATION1741
  int store_def_tagnum,store_tagdefining;
#endif
  int tempstore;
#ifndef G__OLDIMPLEMENTATION403
  char *pfparam;
  struct G__var_array *var;
#endif
#ifndef G__OLDIMPLEMENTATION405
  int nindex=0;
#endif
#ifndef G__OLDIMPLEMENTATION407
  int base1=0;
#endif
#ifndef G__OLDIMPLEMENTATION1515
  int oprp=0;
#endif
#ifndef G__OLDIMPLEMENTATION1570
  int store_cp_asm=0;
#endif
#ifndef G__OLDIMPLEMENTATION1984
  int memfuncenvflag=0;
#endif


#ifdef G__DEBUG
  {
    int jdbg;
    int sizedbg=sizeof(struct G__param);
    char *pcdbg = (char*)(&fpara);
    for(jdbg=0;jdbg<(int)sizedbg;jdbg++) {
      *(pcdbg+jdbg) = (char)0xa3;
    }
  }
#endif

  /*DEBUG*/ /* fprintf(stderr,"%s %d %d\n",item,*known3,memfunc_flag); */

  store_exec_memberfunc = G__exec_memberfunc;
  store_memberfunc_tagnum = G__memberfunc_tagnum;
  store_memberfunc_struct_offset=G__memberfunc_struct_offset;
  
  
  /******************************************************
   * if string expression "string"
   * return
   ******************************************************/
  if(item[0]=='"') {
    result3=G__null;
    return(result3);
  }
  
  
  /******************************************************
   * get length of expression
   ******************************************************/
  lenitem=strlen(item);
  
  /******************************************************
   * Scan item[] until '(' to get function name and hash
   ******************************************************/
  /* Separate function name */
  ig15=0;
  hash=0;
  while((item[ig15]!='(')&&(ig15<lenitem)) {
    funcname[ig15]=item[ig15];
    hash+=item[ig15];
    ig15++;
  }

#ifndef G__OLDIMPLEMENTATION1236
  if(8==ig15 && strncmp(funcname,"operator",8)==0 && 
     strncmp(item+ig15,"()(",3)==0) {
    strcpy(funcname+8,"()");
    hash = hash + '(' + ')';
    ig15 += 2;
  }
#endif

  /******************************************************
   * if itemp[0]=='(' this is a casting or pointer to
   * function.
   ******************************************************/
  castflag=0;
  if(ig15==0) castflag=1;
  
  
  /******************************************************
   * if '(' not found in expression, this is not a function
   * call , so just return.
   * this shouldn't happen.
   ******************************************************/
  if(item[ig15]!='(') {
    /* if no parenthesis , this is not a function */
    result3=G__null;
    return(result3);
  }
  
  /******************************************************
   * put null char to the end of function name
   ******************************************************/
  funcname[ig15++]='\0';

#ifndef G__OLDIMPLEMENTATION1657
  if((strchr(funcname,'.') || strstr(funcname,"->"))
#ifndef G__OLDIMPLEMENTATION1713
     && 0!=strncmp(funcname,"operator",8) 
#endif
     ) {
    result3=G__null;
    return(result3);
  }
#endif
  
#ifndef G__OLDIMPLEMENTATION1560
  /******************************************************
   * conv<B>(x) -> conv<ns::B>(x)
   ******************************************************/
  G__rename_templatefunc(funcname,0);
#endif
  
  
  /******************************************************
   * get paramtters
   *
   *  func(parameters)
   *       ^
   ******************************************************/
  fpara.paran=0;
#ifndef G__OLDIMPLEMENTATION1472
  fpara.para[0].type = 0;
#endif
  
  /* Get Parenthesis */
  
  /******************************************************
   * this if statement should be always true,
   * should be able to omit.
   ******************************************************/
  if(ig15<lenitem) {
    
    /*****************************************************
     * scan '(param1,param2,param3)'
     *****************************************************/
    while(ig15<lenitem) {
#ifndef G__OLDIMPLEMENTATION1705
      int tmpltnest=0;
#endif
      
      /*************************************************
       * scan one parameter upto 'param,' or 'param)'
       * by reading upto ',' or ')'
       *************************************************/
      ig35 = 0;
      nest=0;
      single_quote=0;
      double_quote=0;
      while((((item[ig15]!=',')&&(item[ig15]!=')'))||
	     (nest>0)||
#ifndef G__OLDIMPLEMENTATION1705
	     (tmpltnest>0)||
#endif
	     (single_quote>0)||(double_quote>0))&&(ig15<lenitem)) {
	switch(item[ig15]) {
	case '"' : /* double quote */
	  if(single_quote==0) double_quote ^= 1;
	  break;
	case '\'' : /* single quote */
	  if(double_quote==0) single_quote ^= 1;
	  break;
	case '(':
	case '[':
	case '{':
	  if((double_quote==0)&& (single_quote==0)) nest++;
	  break;
	case ')':
	case ']':
	case '}':
	  if((double_quote==0)&& (single_quote==0)) nest--;
	  break;
	case '\\':
#ifndef G__OLDIMPLEMENTATION1340
	  result7[ig35++] = item[ig15++];
#else
	  fpara.parameter[fpara.paran][ig35++]=item[ig15++];
#endif
	  break;
#ifndef G__OLDIMPLEMENTATION1705
	case '<':
	  if(double_quote==0&&single_quote==0) {
	    result7[ig35]=0;
	    if(0==strcmp(result7,"operator") ||
#ifndef G__OLDIMPLEMENTATION1772
               tmpltnest || 
#endif
	       G__defined_templateclass(result7) ) ++tmpltnest;
	  }
	  break;
	case '>':
	  if(double_quote==0&&single_quote==0) {
	    if(tmpltnest) --tmpltnest;
	  }
	  break;
#endif
	}
#ifndef G__OLDIMPLEMENTATION1340
	result7[ig35++] = item[ig15++];
#else
	fpara.parameter[fpara.paran][ig35++]=item[ig15++];
#endif
#ifndef G__OLDIMPLEMENTATION1036
	if(ig35>=G__ONELINE-1) {
#ifndef G__OLDIMPLEMENTATION1340
	  if(result7[0]=='"') {
#else
	  if(fpara.parameter[fpara.paran][0]=='"') {
#endif
	    G__value bufv;
	    char bufx[G__LONGLINE];
#ifndef G__OLDIMPLEMENTATION1340
	    strncpy(bufx,result7,G__ONELINE-1);
#else
	    strncpy(bufx,fpara.parameter[fpara.paran],G__ONELINE-1);
#endif
	    while((((item[ig15]!=',')&&(item[ig15]!=')'))||
		   (nest>0)||(single_quote>0)||
		   (double_quote>0))&&(ig15<lenitem)) {
	      switch(item[ig15]) {
	      case '"' : /* double quote */
		if(single_quote==0) double_quote ^= 1;
		break;
	      case '\'' : /* single quote */
		if(double_quote==0) single_quote ^= 1;
		break;
	      case '(':
	      case '[':
	      case '{':
		if((double_quote==0)&& (single_quote==0)) nest++;
		break;
	      case ')':
	      case ']':
	      case '}':
		if((double_quote==0)&& (single_quote==0)) nest--;
		break;
	      case '\\':
		bufx[ig35++]=item[ig15++];
		break;
	      }
	      bufx[ig35++]=item[ig15++];
	      if(ig35>=G__LONGLINE-1) {
		G__genericerror("Limitation: Too long function argument");
		return(G__null);
	      }
	    }
	    bufx[ig35]=0;
	    bufv = G__strip_quotation(bufx);
#ifndef G__OLDIMPLEMENTATION1340
	    sprintf(result7,"(char*)(%ld)",bufv.obj.i);
#else
	    sprintf(fpara.parameter[fpara.paran],"(char*)(%ld)",bufv.obj.i);
#endif
#ifndef G__OLDIMPLEMENTATION1340
	    ig35=strlen(result7)+1;
#else
	    ig35=strlen(fpara.parameter[fpara.paran])+1;
#endif
	    break;
	  }
#ifndef G__OLDIMPLEMENTATION1340
	  else if(ig35>G__LONGLINE-1) {
	    G__fprinterr(G__serr,
               "Limitation: length of one function argument be less than %d"
		    ,G__LONGLINE);
	    G__genericerror((char*)NULL);
	    G__fprinterr(G__serr,"Use temp variable as workaround.\n");
	    *known3=1;
	    return(G__null);
	  }
#endif
	  else {
#ifndef G__OLDIMPLEMENTATION1340
	    overflowflag=1;
#else
	    G__fprinterr(G__serr,
    "Limitation: length of one function argument be less than %d",G__ONELINE);
	    G__genericerror((char*)NULL);
	    G__fprinterr(G__serr,"Use temp variable as workaround.\n");
	    *known3=1;
	    return(G__null);
#endif
	  }
	}
#endif
      }
      /*************************************************
       * if ')' is found at the middle of expression,
       * this should be casting or pointer to function
       *
       *  v                    v            <-- this makes
       *  (type)expression  or (*p_func)();    castflag=1
       *       ^                       ^    <-- this makes
       *                                       castflag=2
       *************************************************/
      if((item[ig15]==')')&&(ig15<lenitem-1)) {
	if(1==castflag) {
#ifndef G__OLDIMPLEMENTATION407
	  if(('-'==item[ig15+1]&&'>'==item[ig15+2]) || '.'==item[ig15+1]) 
	    castflag=3;
#else
	  if('-'==item[ig15+1] || '.'==item[ig15+1]) castflag=3;
#endif
	  else                                       castflag=2;
	}
#ifndef G__OLDIMPLEMENTATION407
	else if(('-'==item[ig15+1]&&'>'==item[ig15+2]) ||'.'==item[ig15+1]) {
#else
	else if('-'==item[ig15+1]||'.'==item[ig15+1]) {
#endif
	  castflag=3;
	  base1=ig15+1;
	}
	else if(item[ig15+1]=='[') {
	  nindex=fpara.paran+1;
	}
#ifndef G__OLDIMPLEMENTATION1150
        else if(funcname[0] && isalnum(item[ig15+1])) {
	  G__fprinterr(G__serr,"Error: %s  Syntax error?",item);
	  /* G__genericerror((char*)NULL); , avoid risk of side-effect */
	  G__printlinenum();
	}
#endif
#ifndef G__OLDIMPLEMENTATION1871
	else if(
#ifdef G__OLDIMPLEMENTATION1876
		ig35&&
#endif
		(isalpha(item[0]) || '_'==item[0] || '$'==item[0])) {
	  int itmp;
	  result7[ig35]='\0';
	  strcpy(fpara.parameter[fpara.paran],result7);
#ifndef G__OLDIMPLEMENTATION1876
	  if(ig35) fpara.parameter[++fpara.paran][0]='\0';
#else
	  fpara.parameter[++fpara.paran][0]='\0';
#endif
	  for(itmp=0;itmp<fpara.paran;itmp++) {
	    fpara.para[itmp] = G__getexpr(fpara.parameter[itmp]);
	  }
	  if(G__parenthesisovld(&result3,funcname,&fpara,G__TRYNORMAL)) {
#ifndef G__OLDIMPLEMENTATION1883
	    *known3=1;
#endif
	    return(G__operatorfunction(&result3,item+ig15+2,known3
				       ,result7,funcname));
	  }
#ifndef G__OLDIMPLEMENTATION1902
	  else {
	    result3=G__getfunction_libp(item,funcname,&fpara,known3
					 ,G__TRYNORMAL);
	    if(*known3) {
	      return(G__operatorfunction(&result3,item+ig15+2,known3
					 ,result7,funcname));
	    }
	  }
#endif
	}
#endif
      }
      
      /*************************************************
       * set null char to parameter list buffer.
       *************************************************/
      ig15++;
#ifndef G__OLDIMPLEMENTATION1340
      result7[ig35]='\0';
      if(ig35<G__ONELINE) {
	strcpy(fpara.parameter[fpara.paran],result7);
      }
      else {
#ifndef G__OLDIMPLEMENTATION1824
	strcpy(fpara.parameter[fpara.paran],"@");
	fpara.para[fpara.paran] = G__getexpr(result7);
#endif
      }
      fpara.parameter[++fpara.paran][0]='\0';
#else
      fpara.parameter[fpara.paran++][ig35]='\0';
      fpara.parameter[fpara.paran][0]='\0';
#endif
    }
  }

#ifndef G__OLDIMPLEMENTATION1340
  if(castflag==1&&0==funcname[0] /* &&overflowflag */) {
    result3=G__getexpr(result7);
    *known3 = 1;
    return(result3);
  }
#endif
  
  /***************************************************************
   * member access by (xxx)->xxx , (xxx).xxx
   *
   ***************************************************************/
  if(3==castflag) {
    store_var_type=G__var_type;
    G__var_type='p';
    if('.'==fpara.parameter[1][0]) i=1;
    else                           i=2;
    if(base1) {
      strncpy(fpara.parameter[0],item,base1);
      fpara.parameter[0][base1]='\0';
      strcpy(fpara.parameter[1],item+base1);
    }
#ifndef G__OLDIMPLEMENTATION1013
    if(G__CALLMEMFUNC==memfunc_flag)
      result3=G__getstructmem(store_var_type ,funcname ,fpara.parameter[1]+i
			      ,fpara.parameter[0] ,known3 
			      ,(struct G__var_array*)NULL,i);
    else 
      result3=G__getstructmem(store_var_type ,funcname ,fpara.parameter[1]+i
			      ,fpara.parameter[0] ,known3 ,&G__global,i);
#else
    if(G__CALLMEMFUNC==memfunc_flag)
      result3=G__getstructmem(store_var_type ,funcname ,fpara.parameter[1]+i
			      ,fpara.parameter[0] ,known3 
			      ,(struct G__var_array*)NULL);
    else 
      result3=G__getstructmem(store_var_type ,funcname ,fpara.parameter[1]+i
			      ,fpara.parameter[0] ,known3 ,&G__global);
#endif
    G__var_type=store_var_type;
    return(result3);
  }
  
  /***************************************************************
   * casting or pointer to function
   *
   ***************************************************************/
  if(castflag==2) {
    
    /***************************************************************
     * pointer to function
     *
     *  (*p_function)(param);
     *   ^
     *  this '*' is significant
     ***************************************************************/
    if(fpara.parameter[0][0]=='*') {
#ifndef G__OLDIMPLEMENTATION1393
      switch(fpara.parameter[1][0]) {
      case '[':
	/* function pointer */
	return(G__pointerReference(fpara.parameter[0],&fpara,known3));
      case '(':
      default:
	/* function pointer */
	return(G__pointer2func((G__value*)NULL,fpara.parameter[0],fpara.parameter[1],known3));
      }
#else
      /* function pointer */
      return(G__pointer2func((G__value*)NULL,fpara.parameter[0],fpara.parameter[1],known3));
#endif
    }

#ifdef G__PTR2MEMFUNC
    /***************************************************************
     * pointer to member function
     *
     *  (obj.*p2mf)(param);
     *  (obj->*p2mf)(param);
     ***************************************************************/
    else if('('==fpara.parameter[1][0] &&
	    (strstr(fpara.parameter[0],".*")||
	     strstr(fpara.parameter[0],"->*"))) {
      return(G__pointer2memberfunction(fpara.parameter[0]
				       ,fpara.parameter[1],known3));
    }
#endif
    
#ifndef G__OLDIMPLEMENTATION1001
    /***************************************************************
     * (expr)[n]
     ***************************************************************/
    else if(fpara.paran>=2&&'['==fpara.parameter[1][0]) {
      result3 = G__getexpr(G__catparam(&fpara,fpara.paran,""));
      *known3=1;
      return(result3);
    }
#endif

    /***************************************************************
     * casting
     *
     *  (type)expression;
     ***************************************************************/
    else {
      if(fpara.paran>2) {
#ifndef G__OLDIMPLEMENTATION1824
	if('@'==fpara.parameter[fpara.paran-1][0])
	  fpara.para[1] = fpara.para[fpara.paran-1];
	else
	  fpara.para[1] = G__getexpr(fpara.parameter[fpara.paran-1]);
#else
	fpara.para[1] = G__getexpr(fpara.parameter[fpara.paran-1]);
#endif
	result3=G__castvalue(G__catparam(&fpara,fpara.paran-1,",")
			     ,fpara.para[1]);
      }
      else {
#ifndef G__OLDIMPLEMENTATION1824
	if('@'==fpara.parameter[1][0])
	  fpara.para[1] = fpara.para[1];
	else
	  fpara.para[1] = G__getexpr(fpara.parameter[1]);
#else
	fpara.para[1] = G__getexpr(fpara.parameter[1]);
#endif
	result3=G__castvalue(fpara.parameter[0],fpara.para[1]);
      }
      *known3 = 1;
      return(result3);
    }
  }
  /***************************************************************
   * end of casting or pointer to function
   *
   ***************************************************************/
  
  
  
  /***************************************************************
   * if length of the first parameter is 0 , there are no
   * parameters. set fpara.paran to 0.
   ***************************************************************/
#ifndef G__OLDIMPLEMENTATION1221
  if(strlen(fpara.parameter[0])==0) {
    if(fpara.paran>1 
#ifndef G__OLDIMPLEMENTATION1346
       && ')'==item[strlen(item)-1]
#endif
       ) {
#ifndef G__OLDIMPLEMENTATION1515
      if(fpara.paran==2 && '('==fpara.parameter[1][0]) {
	oprp=1;
      }
      else {
	if(G__dispmsg>=G__DISPWARN) {
	  G__fprinterr(G__serr,"Warning: Empty arg%d",1);
	  G__printlinenum();
	}
      }
#else
      if(G__dispmsg>=G__DISPWARN) {
	G__fprinterr(G__serr,"Warning: Empty arg%d",1);
	G__printlinenum();
      }
#endif
    }
    fpara.paran=0;
  }
#else
  if(strlen(fpara.parameter[0])==0) fpara.paran=0;
#endif
  
  
  /***************************************************************
   * initialize type to fundamental type, and known3 to 1
   ***************************************************************/
  result3.tagnum = -1;
  result3.typenum = -1;
  result3.ref = 0;
#ifndef G__OLDIMPLEMENTATION1259
  result3.isconst = 0;
#endif
  result3.obj.reftype.reftype = G__PARANORMAL;
  
  *known3 = 1;
  
  /***************************************************************
   *  Search for sizeof(), 
   * before parameters are evaluated
   *  sizeof() is processed specially.
   ***************************************************************/
  if( G__special_func(&result3,funcname,&fpara,hash)==1 ) {
    G__var_type = 'p';
    return(result3);
  }
  
  /***************************************************************
   *  Evaluate parameters  parameter:string expression , 
   *                       para     :evaluated expression 
   ***************************************************************/
#ifdef G__ASM
  store_asm_noverflow = G__asm_noverflow;
  if(G__oprovld) {
    /* In case of operator overloading function, arguments are already
     * evaluated. Avoid duplication in argument stack by temporarily
     * reset G__asm_noverflow */
    /* G__asm_noverflow=0; */
#ifndef G__OLDIMPLEMENTATION1164
    G__suspendbytecode();
#else
    G__abortbytecode();
#endif
  }
  /*DEBUG*/ /* fprintf(stderr,"\nSET %lx=%lx %lx %s\n",G__memberfunc_struct_offset,G__store_struct_offset,store_struct_offset,item); */
  if(G__asm_noverflow&&fpara.paran&&
     (G__store_struct_offset!=G__memberfunc_struct_offset
#ifndef G__OLDIMPLEMENTATION2155
      || G__do_setmemfuncenv
#endif
      )) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETMEMFUNCENV\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__SETMEMFUNCENV;
    G__inc_cp_asm(1,0);
#ifndef G__OLDIMPLEMENTATION1984
    memfuncenvflag=1;
#endif
  }
#ifndef G__OLDIMPLEMENTATION1570
  if(G__asm_noverflow && fpara.paran) {
    store_cp_asm = G__asm_cp;
  }
#endif
#endif
  /* restore base environment */
  store_struct_offset = G__store_struct_offset;
  store_tagnum = G__tagnum;
  store_memberfunc_var_type = G__var_type;
#ifndef G__OLDIMPLEMENTATION1613
  store_globalvarpointer = G__globalvarpointer;
  G__globalvarpointer = G__PVOID;
#endif
  G__tagnum = G__memberfunc_tagnum;
  G__store_struct_offset = G__memberfunc_struct_offset;
  G__var_type = 'p';
  /* evaluate parameter */
#ifndef G__OLDIMPLEMENTATION809
  if(p2ffpara) {
    fpara = *p2ffpara;
    p2ffpara=(struct G__param*)NULL;
  }
  else {
    for(ig15=0;ig15< fpara.paran;ig15++) {
#ifndef G__OLDIMPLEMENTATION1069
      if('['==fpara.parameter[ig15][0]) {
	fpara.paran=ig15;
	break;
      }
#endif
#ifndef G__OLDIMPLEMENTATION1221
      if(0==fpara.parameter[ig15][0]) {
	if(G__dispmsg>=G__DISPWARN) {
	  G__fprinterr(G__serr,"Warning: Empty arg%d",ig15+1);
	  G__printlinenum();
	}
      }
#endif
#ifndef G__OLDIMPLEMENTATION1824
      if('@'!=fpara.parameter[ig15][0]) 
	fpara.para[ig15]=G__getexpr(fpara.parameter[ig15]);
#else
      fpara.para[ig15]=G__getexpr(fpara.parameter[ig15]);
#endif
    }
  }
#else
  for(ig15=0;ig15< fpara.paran;ig15++) {
#ifndef G__OLDIMPLEMENTATION1824
    if('@'!=fpara.parameter[ig15][0]) 
      fpara.para[ig15]=G__getexpr(fpara.parameter[ig15]);
#else
    fpara.para[ig15]=G__getexpr(fpara.parameter[ig15]);
#endif
  }
#endif
  /* recover function call environment */
#ifdef G__ASM
  /*DEBUG*/ /* fprintf(stderr,"\nREC %lx %lx=%lx %s\n",G__memberfunc_struct_offset,G__store_struct_offset,store_struct_offset,item); */
  if(G__asm_noverflow&&fpara.paran&&
#ifndef G__OLDIMPLEMENTATION1984
     memfuncenvflag
#else
     G__store_struct_offset!=store_struct_offset
#endif
     ) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: RECMEMFUNCENV\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__RECMEMFUNCENV;
    G__inc_cp_asm(1,0);
#ifndef G__OLDIMPLEMENTATION1984
    memfuncenvflag=0;
#endif
  }
#endif
  G__store_struct_offset=store_struct_offset;
  G__tagnum=store_tagnum;
  G__var_type = store_memberfunc_var_type;
#ifndef G__OLDIMPLEMENTATION1613
  G__globalvarpointer = store_globalvarpointer;
#endif

#ifdef G__ASM
  if(G__oprovld) {
    G__asm_noverflow = store_asm_noverflow;
  }
  else {
    G__asm_noverflow &= store_asm_noverflow;
  }
#endif

#ifdef G__SECURITY
#ifndef G__OLDIMPLEMENTATION1115
  if(G__return>G__RETURN_NORMAL||
     (G__security_error&&G__security!=G__SECURE_NONE)) return(G__null);
#else
  if(G__return>G__RETURN_NORMAL) return(G__null);
#endif
#endif

  
  fpara.para[fpara.paran] = G__null;
  
  /***************************************************************
   * if not function name, this is '(expr1,expr2,...,exprn)' 
   * According to ANSI-C , exprn has to be returned,
   ***************************************************************/
  if(funcname[0]=='\0') {
    /*************************************************
     *  'result3 = fpara.para[fpara.paran-1] ;'
     * should be correct as ANSI-C.
     *************************************************/
    result3=fpara.para[0];
    return(result3);
  }

  
  /* scope operator ::f() , A::B::f()
   * note, 
   *    G__exec_memberfunc restored at return memfunc_flag is local, 
   *   there should be no problem modifying these variables.
   *    store_struct_offset and store_tagnum are only used in the
   *   explicit type conversion section.  It is OK to use them here
   *   independently.
   */
  store_struct_offset=G__store_struct_offset;
  store_tagnum=G__tagnum;
#ifndef G__OLDIMPLEMENTATION1741
  store_def_tagnum = G__def_tagnum;
  store_tagdefining = G__tagdefining;
#endif
  switch(G__scopeoperator(funcname,&hash,&G__store_struct_offset,&G__tagnum)){
  case G__GLOBALSCOPE: /* global scope */
    G__exec_memberfunc=0;
    memfunc_flag=G__TRYNORMAL;
#ifndef G__OLDIMPLEMENTATION1741
    G__def_tagnum = -1;
    G__tagdefining = -1;
#endif
    break;
  case G__CLASSSCOPE: /* class scope */
#ifndef G__OLDIMPLEMENTATION1101
    memfunc_flag=G__CALLSTATICMEMFUNC;
#else
    memfunc_flag=G__CALLMEMFUNC;
#endif
#ifndef G__OLDIMPLEMENTATION1741
    /* This looks very risky */
    G__def_tagnum = -1;
    G__tagdefining = -1;
    G__exec_memberfunc=1;
    G__memberfunc_tagnum = G__tagnum;
#endif
    break;
  }
  
#ifdef G__DUMPFILE
  /***************************************************************
   * dump that a function is called 
   ***************************************************************/
  if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
    for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
    fprintf(G__dumpfile,"%s(",funcname);
    for(ipara=1;ipara<= fpara.paran;ipara++) {
      if(ipara!=1) fprintf(G__dumpfile,",");
      G__valuemonitor(fpara.para[ipara-1],result7);
      fprintf(G__dumpfile,"%s",result7);
    }
    fprintf(G__dumpfile,");/*%s %d,%lx %lx*/\n" 
	    ,G__ifile.name,G__ifile.line_number
	    ,store_struct_offset,G__store_struct_offset);
    G__dumpspace += 3;
    
  }
#endif

  /********************************************************************
   * Begin Loop to resolve overloaded function
   ********************************************************************/
  for(funcmatch=G__EXACT;funcmatch<=G__USERCONV;funcmatch++) {
    
    /***************************************************************
     * search for interpreted member function
     * if(G__exec_memberfunc)     ==>  memfunc();
     * G__TRYNORMAL!=memfunc_flag ==>  a.memfunc();
     ***************************************************************/
    if(G__exec_memberfunc || G__TRYNORMAL!=memfunc_flag) {
#ifndef G__OLDIMPLEMENTATION589
      int local_tagnum;
      if(G__exec_memberfunc&&-1==G__tagnum)
	local_tagnum = G__memberfunc_tagnum;
      else 
	local_tagnum = G__tagnum;
      if(-1!=G__tagnum) G__incsetup_memfunc(G__tagnum);
#else
      G__ASSERT(0<=G__tagnum);
      G__incsetup_memfunc(G__tagnum);
#endif
#ifndef G__OLDIMPLEMENTATION589
      if(-1!=local_tagnum&& G__interpret_func(&result3,funcname,&fpara,hash
			   ,G__struct.memfunc[local_tagnum]
			   ,funcmatch,memfunc_flag)==1 ) {
#else
      if(G__interpret_func(&result3,funcname,&fpara,hash
			   ,G__struct.memfunc[G__tagnum]
			   ,funcmatch,memfunc_flag)==1 ) {
#endif
#ifdef G__DUMPFILE
	if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
	  G__dumpspace -= 3;
	  for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
	  G__valuemonitor(result3,result7);
	  fprintf(G__dumpfile,"/* return(inp) %s.%s()=%s*/\n"
		  ,G__struct.name[G__tagnum],funcname,result7);
	}
#endif
#ifndef G__OLDIMPLEMENTATION517
	if(G__store_struct_offset!=store_struct_offset) 
	  G__gen_addstros(store_struct_offset-G__store_struct_offset);
#endif
	G__store_struct_offset = store_struct_offset;
	G__tagnum = store_tagnum;
#ifndef G__OLDIMPLEMENTATION1741
	G__def_tagnum = store_def_tagnum;
	G__tagdefining = store_tagdefining;
#endif
	G__exec_memberfunc = store_exec_memberfunc;
	G__memberfunc_tagnum=store_memberfunc_tagnum;
	G__memberfunc_struct_offset=store_memberfunc_struct_offset;
	G__setclassdebugcond(G__memberfunc_tagnum,0);
#ifndef G__OLDIMPLEMENTATION405
	if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
	   (isupper(result3.type)||'u'==result3.type)
#else
	   isupper(result3.type)
#endif
	   ) {
	  G__getindexedvalue(&result3,fpara.parameter[nindex]);
	}
#endif
#ifndef G__OLDIMPLEMENTATION1515
	if(oprp) *known3 = G__additional_parenthesis(&result3,&fpara);
#endif
	return(result3);
      }
#define G__OLDIMPLEMENTATION1159
#ifndef G__OLDIMPLEMENTATION1159
      /* STILL WORKING , DO NOT RELEASE THIS */
      /******************************************************************
       * Search template function
       ******************************************************************/
      if((G__EXACT==funcmatch||G__USERCONV==funcmatch)) {
        int storeX_exec_memberfunc=G__exec_memberfunc;
        int storeX_memberfunc_tagnum=G__memberfunc_tagnum;
        G__exec_memberfunc = 1;
        G__memberfunc_tagnum = local_tagnum;
	if(G__templatefunc(&result3,funcname,&fpara,hash,funcmatch)==1){
#ifdef G__DUMPFILE
          if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
	    G__dumpspace -= 3;
	    for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
            G__valuemonitor(result3,result7);
	    fprintf(G__dumpfile,"/* return(lib) %s()=%s */\n"
                    ,funcname,result7);
          }
#endif
      
          G__exec_memberfunc = store_exec_memberfunc;
          G__memberfunc_tagnum=store_memberfunc_tagnum;
          G__memberfunc_struct_offset=store_memberfunc_struct_offset;
#ifndef G__OLDIMPLEMENTATION1515
	  if(oprp) *known3 = G__additional_parenthesis(&result3,&fpara);
#endif
          return(result3);
        }
        G__exec_memberfunc=storeX_exec_memberfunc;
        G__memberfunc_tagnum=storeX_memberfunc_tagnum;
      }
#endif
    }
    
    
    /***************************************************************
     * If memberfunction is called explicitly by clarifying scope
     * don't examine global function and exit from G__getfunction().
     * There are 2 cases                   G__exec_memberfunc
     *   obj.memfunc();                            1
     *   X::memfunc();                             1
     *    X();              constructor            2
     *   ~X();              destructor             2
     * If G__exec_memberfunc==2, don't display error message.
     ***************************************************************/
    /* If searching only member function */
    if(memfunc_flag
#ifndef G__OLDIMPLEMENTATION1104
       &&(G__store_struct_offset||G__CALLSTATICMEMFUNC!=memfunc_flag)
#endif
       ) {
      
      G__exec_memberfunc = store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
      
      /* If the last resolution of overloading failed */
      if(funcmatch==G__USERCONV) {
	
	if(G__TRYDESTRUCTOR==memfunc_flag) {
	  /* destructor for base calss and class members */
#ifdef G__ASM
#ifdef G__SECURITY
	  store_asm_noverflow = G__asm_noverflow;
	  if(G__security&G__SECURE_GARBAGECOLLECTION) G__abortbytecode();
#endif
#endif
#ifdef G__VIRTUALBASE
	  if(G__CPPLINK!=G__struct.iscpplink[G__tagnum]) G__basedestructor();
#else
	  G__basedestructor();
#endif
#ifdef G__ASM
#ifdef G__SECURITY
	  G__asm_noverflow = store_asm_noverflow;
#endif
#endif
	}
	else {
	  switch(memfunc_flag) {
	  case G__CALLCONSTRUCTOR:
	  case G__TRYCONSTRUCTOR:
#ifndef G__OLDIMPLEMENTATION1250
	  case G__TRYIMPLICITCONSTRUCTOR:
#endif
	    /* constructor for base class and class members default 
	     * constructor only */
#ifdef G__VIRTUALBASE
	    if(G__CPPLINK!=G__struct.iscpplink[G__tagnum])
	      G__baseconstructor(0 ,(struct G__baseparam *)NULL);
#else
	    G__baseconstructor(0 ,(struct G__baseparam *)NULL);
#endif
	  }
	}
	G__exec_memberfunc = store_exec_memberfunc;
	G__memberfunc_tagnum=store_memberfunc_tagnum;
	G__memberfunc_struct_offset=store_memberfunc_struct_offset;
	
	
	*known3=0;
	switch(memfunc_flag) {
	case G__CALLMEMFUNC:
	  if(G__parenthesisovld(&result3,funcname,&fpara,G__CALLMEMFUNC)) {
	    *known3=1;
#ifndef G__OLDIMPLEMENTATION517
	    if(G__store_struct_offset!=store_struct_offset) 
	      G__gen_addstros(store_struct_offset-G__store_struct_offset);
#endif
	    G__store_struct_offset = store_struct_offset;
	    G__tagnum = store_tagnum;
#ifndef G__OLDIMPLEMENTATION1741
	    G__def_tagnum = store_def_tagnum;
	    G__tagdefining = store_tagdefining;
#endif
#ifndef G__OLDIMPLEMENTATION405
	    if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
	       (isupper(result3.type)||'u'==result3.type)
#else
	       isupper(result3.type)
#endif
	       ) {
	      G__getindexedvalue(&result3,fpara.parameter[nindex]);
	    }
#endif
#ifndef G__OLDIMPLEMENTATION1515
	    if(oprp) *known3 = G__additional_parenthesis(&result3,&fpara);
#endif
	    return(result3);
	  }
#ifndef G__OLDIMPLEMENTATION733
	  if('~'==funcname[0]) {
	    *known3=1;
	    return(G__null);
	  }
#endif
#ifndef G__OLDIMPLEMENTATION2010
	case G__CALLCONSTRUCTOR:
#endif
#ifndef G__OLDIMPLEMENTATION1729
	  /******************************************************************
	   * Search template function
	   ******************************************************************/
	  G__exec_memberfunc = 1;
	  G__memberfunc_tagnum=G__tagnum;
	  G__memberfunc_struct_offset=G__store_struct_offset;
	  if((G__EXACT==funcmatch||G__USERCONV==funcmatch)&&
	     G__templatefunc(&result3,funcname,&fpara,hash,funcmatch)==1){
	    
#ifdef G__DUMPFILE
	    if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
	      G__dumpspace -= 3;
	      for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
	      G__valuemonitor(result3,result7);
	      fprintf(G__dumpfile ,"/* return(lib) %s()=%s */\n"
		      ,funcname,result7);
	    }
#endif
	    G__exec_memberfunc = store_exec_memberfunc;
	    G__memberfunc_tagnum=store_memberfunc_tagnum;
	    G__memberfunc_struct_offset=store_memberfunc_struct_offset;
	    /* don't know why if(oprp) is needed, copied from line 2111 */
	    if(oprp) *known3 = G__additional_parenthesis(&result3,&fpara);
	    else *known3=1;
	    return(result3);
	  }
	  G__exec_memberfunc = store_exec_memberfunc;
	  G__memberfunc_tagnum=store_memberfunc_tagnum;
	  G__memberfunc_struct_offset=store_memberfunc_struct_offset;
#endif /* 1729 */
#ifdef G__OLDIMPLEMENTATION2010
	case G__CALLCONSTRUCTOR:
#endif
#ifndef G__OLDIMPLEMENTATION1376
	  if(G__NOLINK > G__globalcomp) break;
#endif
#ifndef G__OLDIMPLEMENTATION1505
	  if(!G__no_exec_compile || G__asm_noverflow) {
#endif
#ifndef G__OLDIMPLEMENTATION1185
#ifndef G__OLDIMPLEMENTATION1775
	    if (0==G__const_noerror) 
	      G__fprinterr(G__serr, "Error: Can't call %s::%s in current scope"
			   ,G__struct.name[G__tagnum],item);
#else
	    G__fprinterr(G__serr, "Error: Can't call %s::%s in current scope"
			 ,G__struct.name[G__tagnum],item);
#endif
#else
	    G__fprinterr(G__serr, "Error: Can't call %s::%s() in current scope"
			 ,G__struct.name[G__tagnum],funcname);
#endif
	    G__genericerror((char*)NULL);
#ifndef G__OLDIMPLEMENTATION1505
	  }
#endif
	  store_exec_memberfunc=G__exec_memberfunc;
	  G__exec_memberfunc=1;
#ifndef G__OLDIMPLEMENTATION1103
	  if(0==G__const_noerror
#ifndef G__OLDIMPLEMENTATION1505
	     && (!G__no_exec_compile || G__asm_noverflow)
#endif
	     ) {
#endif
	    G__fprinterr(G__serr,"Possible candidates are...\n");
#ifndef G__OLDIMPLEMENTATION1079
	    {
	      char itemtmp[G__LONGLINE];
	      sprintf(itemtmp,"%s::%s",G__struct.name[G__tagnum],funcname);
	      G__display_proto(G__serr,itemtmp);
	    }
#else
	    G__listfunc(G__serr,G__PUBLIC_PROTECTED_PRIVATE,funcname
		      ,G__struct.memfunc[G__tagnum]);
#endif
#ifndef G__OLDIMPLEMENTATION1103
	  }
#endif
	  G__exec_memberfunc=store_exec_memberfunc;
	}
#ifdef G__DUMPFILE
	if(G__dumpfile!=NULL && 0==G__no_exec_compile) G__dumpspace -= 3;
#endif
	
#ifndef G__OLDIMPLEMENTATION517
	if(G__store_struct_offset!=store_struct_offset) 
	  G__gen_addstros(store_struct_offset-G__store_struct_offset);
#endif
	G__store_struct_offset = store_struct_offset;
	G__tagnum = store_tagnum;
#ifndef G__OLDIMPLEMENTATION1741
	G__def_tagnum = store_def_tagnum;
	G__tagdefining = store_tagdefining;
#endif
	if(fpara.paran && 'u'==fpara.para[0].type&&
#ifndef G__OLDIMPLEMENTATION1250
	   (G__TRYCONSTRUCTOR==memfunc_flag||
	    G__TRYIMPLICITCONSTRUCTOR==memfunc_flag)
#else
	   G__TRYCONSTRUCTOR==memfunc_flag
#endif
	   ) {
	  /* in case of copy constructor not found */
	  return(fpara.para[0]);
	}
	else {
	  return(G__null);
	}
      }
      /* ELSE next level overloaded function resolution */
      continue;
    }
    
    
    
    /***************************************************************
     * reset G__exec_memberfunc for global function.
     * Original value(store_exec_memberfunc) is restored when exit 
     * from this function
     ***************************************************************/
    tempstore = G__exec_memberfunc;
    G__exec_memberfunc = 0;
    
    
    
    /***************************************************************
     * search for interpreted global function
     *
     ***************************************************************/
    if(
#ifndef G__OLDIMPLEMENTATION2071
	memfunc_flag!=G__CALLSTATICMEMFUNC &&
#endif
	G__interpret_func(&result3,funcname,&fpara,hash,G__p_ifunc
			  ,funcmatch,G__TRYNORMAL)==1 ) {
#ifdef G__DUMPFILE
      if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
	G__dumpspace -= 3;
	for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
	G__valuemonitor(result3,result7);
	fprintf(G__dumpfile ,"/* return(inp) %s()=%s*/\n" ,funcname,result7);
      }
#endif
      
      G__exec_memberfunc = store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
      G__setclassdebugcond(G__memberfunc_tagnum,0);
#ifndef G__OLDIMPLEMENTATION405
      if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
	 (isupper(result3.type)||'u'==result3.type)
#else
	 isupper(result3.type)
#endif
	 ) {
	G__getindexedvalue(&result3,fpara.parameter[nindex]);
      }
#endif
#ifndef G__OLDIMPLEMENTATION1515
      if(oprp) *known3 = G__additional_parenthesis(&result3,&fpara);
#endif
      return(result3);
    }

    G__exec_memberfunc = tempstore;
    
    /* there is no function overload resolution after this point,
     * thus, if not found in G__EXACT trial, there is no chance to
     * find matched function in consequitive search
     */
    if(G__USERCONV==funcmatch) goto templatefunc;
    if(G__EXACT!=funcmatch) continue;
    
    
    
    /***************************************************************
     * search for compiled(archived) function
     *
     ***************************************************************/
    if(
#ifndef G__OLDIMPLEMENTATION2071
	memfunc_flag!=G__CALLSTATICMEMFUNC &&
#endif
	G__compiled_func(&result3,funcname,&fpara,hash)==1 ) {
      
#ifdef G__ASM
      if(G__asm_noverflow) {
	/****************************************
	 * LD_FUNC (compiled)
	 ****************************************/
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,
			       "%3x: LD_FUNC compiled %s paran=%d\n"
			       ,G__asm_cp,funcname,fpara.paran);
#endif
	G__asm_inst[G__asm_cp]=G__LD_FUNC;
	G__asm_inst[G__asm_cp+1] = (long)(&G__asm_name[G__asm_name_p]);
	G__asm_inst[G__asm_cp+2]=hash;
	G__asm_inst[G__asm_cp+3]=fpara.paran;
	G__asm_inst[G__asm_cp+4]=(long)G__compiled_func;
	if(G__asm_name_p+strlen(funcname)+1<G__ASM_FUNCNAMEBUF) {
	  strcpy(G__asm_name+G__asm_name_p,funcname);
	  G__asm_name_p += strlen(funcname)+1;
	  G__inc_cp_asm(5,0);
	}
	else {
	  G__abortbytecode();
#ifdef G__ASM_DBG
	  if(G__asm_dbg) {
	    G__fprinterr(G__serr,"COMPILE ABORT function name buffer overflow");
	    G__printlinenum();
	  }
#endif
	}
      }
#endif /* G__ASM */
      
#ifdef G__DUMPFILE
      if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
	G__dumpspace -= 3;
	for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
	G__valuemonitor(result3,result7);
	fprintf(G__dumpfile ,"/* return(cmp) %s()=%s */\n" ,funcname,result7);
      }
#endif
      
      G__exec_memberfunc = store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
#ifndef G__OLDIMPLEMENTATION405
      if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
	 (isupper(result3.type)||'u'==result3.type)
#else
	 isupper(result3.type)
#endif
	 ) {
	G__getindexedvalue(&result3,fpara.parameter[nindex]);
      }
#endif
      return(result3);
    }
    
    
    /***************************************************************
     * search for library function which are included in G__ci.c
     *
     ***************************************************************/
    if( G__library_func(&result3,funcname,&fpara,hash)==1 ) {
#ifndef G__OLDIMPLEMENTATION1595
      if(G__no_exec_compile) result3.type = 'i';
#endif
#ifdef G__ASM
      if(G__asm_noverflow) {
	/****************************************
	 * LD_FUNC (library)
	 ****************************************/
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,
			       "%3x: LD_FUNC library %s paran=%d\n"
			       ,G__asm_cp,funcname,fpara.paran);
#endif
	G__asm_inst[G__asm_cp]=G__LD_FUNC;
	G__asm_inst[G__asm_cp+1] = (long)(&G__asm_name[G__asm_name_p]);
	G__asm_inst[G__asm_cp+2]=hash;
	G__asm_inst[G__asm_cp+3]=fpara.paran;
	G__asm_inst[G__asm_cp+4]=(long)G__library_func;
	if(G__asm_name_p+strlen(funcname)+1<G__ASM_FUNCNAMEBUF) {
	  strcpy(G__asm_name+G__asm_name_p,funcname);
	  G__asm_name_p += strlen(funcname)+1;
	  G__inc_cp_asm(5,0);
	}
	else {
	  G__abortbytecode();
#ifdef G__ASM_DBG
	  if(G__asm_dbg) 
	    G__fprinterr(G__serr,"COMPILE ABORT function name buffer overflow");
	    G__printlinenum();
#endif
	}
      }
#endif /* G__ASM */
      
#ifdef G__DUMPFILE
      if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
	G__dumpspace -= 3;
	for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
	G__valuemonitor(result3,result7);
	fprintf(G__dumpfile ,"/* return(lib) %s()=%s */\n" ,funcname,result7);
      }
#endif
	    
      G__exec_memberfunc = store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
#ifndef G__OLDIMPLEMENTATION405
      if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
	 (isupper(result3.type)||'u'==result3.type)
#else
	 isupper(result3.type)
#endif
	 ) {
	G__getindexedvalue(&result3,fpara.parameter[nindex]);
      }
#endif
      return(result3);
    }
    
#ifdef G__TEMPLATEFUNC
    templatefunc:
    /******************************************************************
     * Search template function
     ******************************************************************/
    if((G__EXACT==funcmatch||G__USERCONV==funcmatch)&&
       G__templatefunc(&result3,funcname,&fpara,hash,funcmatch)==1){

#ifdef G__DUMPFILE
      if(G__dumpfile!=NULL && 0==G__no_exec_compile) {
	G__dumpspace -= 3;
	for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
	G__valuemonitor(result3,result7);
	fprintf(G__dumpfile ,"/* return(lib) %s()=%s */\n" ,funcname,result7);
      }
#endif
      
      G__exec_memberfunc = store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
#ifndef G__OLDIMPLEMENTATION1515
      if(oprp) *known3 = G__additional_parenthesis(&result3,&fpara);
#ifndef G__OLDIMPLEMENTATION1729
      else *known3=1; /* don't know why this was missing */
#endif
#endif
      return(result3);
    }
#endif /* G__TEMPLATEFUNC */
    
    /******************************************************************
     * End Loop to resolve overloaded function
     ******************************************************************/
    
    /* next_overload_match:
       ; */
    
  }

  
  /********************************************************************
   * Explicit type conversion by searching constructors
   ********************************************************************/
  if(G__TRYNORMAL==memfunc_flag
#ifndef G__OLDIMPLEMENTATION1104
     ||G__CALLSTATICMEMFUNC==memfunc_flag
#endif
     ) {
#ifndef G__OLDIMPLEMENTATION985
    int store_var_typeX = G__var_type;
#endif
    i=G__defined_typename(funcname);
#ifndef G__OLDIMPLEMENTATION985
    G__var_type = store_var_typeX;
#endif
    if(-1!=i) {
      if(-1!=G__newtype.tagnum[i]) {
	strcpy(funcname,G__struct.name[G__newtype.tagnum[i]]);
      }
      else {
#ifndef G__OLDIMPLEMENTATION1188
	char ttt[G__ONELINE];
	result3 = fpara.para[0];
	if(G__fundamental_conversion_operator(G__newtype.type[i],-1 
					      ,i ,G__newtype.reftype[i],0
					      ,&result3,ttt)) {
	  *known3=1;
#ifndef G__OLDIMPLEMENTATION1515
	  if(oprp) *known3 = G__additional_parenthesis(&result3,&fpara);
#endif
	  return(result3);
	}
#endif
	strcpy(funcname,G__type2string(G__newtype.type[i]
				       ,G__newtype.tagnum[i] ,-1
				       ,G__newtype.reftype[i] ,0));
      }
      G__hash(funcname,hash,i);
    }

    classhash=strlen(funcname);
    i=0;
    while(i<G__struct.alltag) {
      if((G__struct.hash[i]==classhash)&&
	 (strcmp(G__struct.name[i],funcname)==0)
#ifdef G__OLDIMPLEMENTATION1386
#ifndef G__OLDIMPLEMENTATION1332
	  &&'e'!=G__struct.type[i]
#endif
#endif
	 ) {
#ifndef G__OLDIMPLEMENTATION1386
	if('e'==G__struct.type[i] && 
	   -1!=fpara.para[0].tagnum &&
	   'e'==G__struct.type[fpara.para[0].tagnum]) {
	  return(fpara.para[0]);
	}
#endif
	store_struct_offset=G__store_struct_offset;
	
	/* questionable part */
	/* store_exec_memfunc=G__exec_memberfunc; */
	
	store_tagnum=G__tagnum;
	G__tagnum=i;
	if(G__CPPLINK!=G__struct.iscpplink[G__tagnum]) {
	  G__alloc_tempobject(G__tagnum,-1);
	  G__store_struct_offset=G__p_tempbuf->obj.obj.i;
#ifndef G__OLDIMPLEMENTATION843
#ifdef G__ASM
	  if(G__asm_noverflow) {
#ifndef G__OLDIMPLEMENTATION2111
	    if(G__throwingexception) {
#ifdef G__ASM_DBG
	      if(G__asm_dbg) {
	        G__fprinterr(G__serr,"%3x: ALLOCEXCEPTION %d\n"
                            ,G__asm_cp,G__tagnum);
	      }
#endif
	      G__asm_inst[G__asm_cp]=G__ALLOCEXCEPTION;
	      G__asm_inst[G__asm_cp+1]=G__tagnum;
	      G__inc_cp_asm(2,0);
	    }
            else {
#ifdef G__ASM_DBG
	      if(G__asm_dbg) {
	        G__fprinterr(G__serr,"%3x: ALLOCTEMP %d\n"
                             ,G__asm_cp,G__tagnum);
	        G__fprinterr(G__serr,"%3x: SETTEMP\n",G__asm_cp);
	      }
#endif
	      G__asm_inst[G__asm_cp]=G__ALLOCTEMP;
	      G__asm_inst[G__asm_cp+1]=G__tagnum;
	      G__asm_inst[G__asm_cp+2]=G__SETTEMP;
	      G__inc_cp_asm(3,0);
	    }
#else /* 2111 */
#ifdef G__ASM_DBG
	    if(G__asm_dbg) {
	      G__fprinterr(G__serr,"%3x: ALLOCTEMP %d\n",G__asm_cp,G__tagnum);
	      G__fprinterr(G__serr,"%3x: SETTEMP\n",G__asm_cp);
	    }
#endif
	    G__asm_inst[G__asm_cp]=G__ALLOCTEMP;
	    G__asm_inst[G__asm_cp+1]=G__tagnum;
	    G__asm_inst[G__asm_cp+2]=G__SETTEMP;
	    G__inc_cp_asm(3,0);
#endif /* 2111 */
	  }
#endif
#endif
	}
	else {
	  G__store_struct_offset= G__PVOID;
	}
#ifndef G__OLDIMPLEMENTATION1341
	G__incsetup_memfunc(G__tagnum);
#endif
	for(funcmatch=G__EXACT;funcmatch<=G__USERCONV;funcmatch++) {
#ifdef G__OLDIMPLEMENTATION1341
	  G__incsetup_memfunc(G__tagnum);
#endif
	  *known3=G__interpret_func(&result3,funcname
				    ,&fpara,hash
				    ,G__struct.memfunc[G__tagnum]
				    ,funcmatch
				    ,G__TRYCONSTRUCTOR);
	  if(*known3) break;
	}
	if(G__CPPLINK==G__struct.iscpplink[G__tagnum]
#ifndef G__OLDIMPLEMENTATION2111
	   && !G__throwingexception
#endif
           ) {
	  G__store_tempobject(result3);
	  if(G__dispsource) {
	    G__fprinterr(G__serr,
		    "!!!Create temp object (%s)0x%lx,%d for %s()\n"
		    ,G__struct.name[G__tagnum] ,G__p_tempbuf->obj.obj.i
		    ,G__templevel ,funcname);
	  }
#ifdef G__ASM
	  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: STORETEMP\n",G__asm_cp);
#endif
	    G__asm_inst[G__asm_cp]=G__STORETEMP;
	    G__inc_cp_asm(1,0);
	  }
#endif
	}
	else {
	  result3.type='u';
	  result3.tagnum=G__tagnum;
	  result3.typenum = -1;
	  result3.obj.i=G__store_struct_offset;
	  result3.ref=G__store_struct_offset;
#ifdef G__OLDIMPLEMENTATION843      /* FOLLOWING PART SHOULD BE REMOVED */
#ifdef G__ASM                       /* WATCH FOR PROBLEM in stl/vec3.cxx */
	  /* Loop compilation of explicit conversion not ready yet, 
	   * mainly because of temporary object handling difficulty */
	  if(G__asm_noverflow) {
	    G__abortbytecode();
	    if(G__asm_dbg) {
	      G__fprinterr(G__serr,"COMPILE ABORT Explicit conversion");
	      G__printlinenum();
	    }
	  }
#endif
#endif
	}
	G__tagnum=store_tagnum;
	
	/* questionable part */
	G__exec_memberfunc = store_exec_memberfunc;
	G__memberfunc_tagnum=store_memberfunc_tagnum;
	G__memberfunc_struct_offset=store_memberfunc_struct_offset;
	
	G__store_struct_offset=store_struct_offset;
#ifndef G__OLDIMPLEMENTATION1500
#ifdef G__ASM
	if(G__asm_noverflow
#ifndef G__OLDIMPLEMENTATION2111
	   && (!G__throwingexception || 
	       (-1!=result3.tagnum &&
                G__CPPLINK!=G__struct.iscpplink[result3.tagnum]))
#endif
           ) {
	  G__asm_inst[G__asm_cp]=G__POPTEMP;
#ifndef G__OLDIMPLEMENTATION2111
	  if(G__throwingexception) G__asm_inst[G__asm_cp+1] = result3.tagnum;
          else                     G__asm_inst[G__asm_cp+1] = -1;
#else
	  G__asm_inst[G__asm_cp+1] = -1;
#endif
#ifdef G__ASM_DBG
	  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPTEMP %d\n"
                                      ,G__asm_cp,G__asm_inst[G__asm_cp+1]);
#endif
	  G__inc_cp_asm(2,0);
	}
#endif
#endif

	if(0 == *known3) {
#ifndef G__OLDIMPLEMENTATION1341
	  if(-1 != i && fpara.paran==1 && -1 != fpara.para[0].tagnum) {
	    long store_struct_offset = G__store_struct_offset;
	    long store_memberfunc_struct_offset = G__memberfunc_struct_offset;
	    int store_memberfunc_tagnum = G__memberfunc_tagnum;
	    int store_exec_memberfunc = G__exec_memberfunc;
	    store_tagnum = G__tagnum;
#ifndef G__OLDIMPLEMENTATION1500
	    G__inc_cp_asm(-5,0); /* cancel ALLOCTEMP, SETTEMP, POPTEMP */
#else
	    G__inc_cp_asm(-3,0); /* cancel ALLOCTEMP, SETTEMP */
#endif
	    G__pop_tempobject();
	    G__tagnum = fpara.para[0].tagnum;
	    G__store_struct_offset = fpara.para[0].obj.i;
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
#endif
	    sprintf(funcname,"operator %s",G__fulltagname(i,1));
	    G__hash(funcname,hash,i);
	    G__incsetup_memfunc(G__tagnum);
	    fpara.paran = 0;
	    for(funcmatch=G__EXACT;funcmatch<=G__USERCONV;funcmatch++) {
	      *known3=G__interpret_func(&result3,funcname
					,&fpara,hash
					,G__struct.memfunc[G__tagnum]
					,funcmatch
					,G__TRYMEMFUNC);
	      if(*known3) {
#ifdef G__ASM
		if(G__asm_noverflow) {
		  G__asm_inst[G__asm_cp] = G__POPSTROS;
		  G__inc_cp_asm(1,0);
#ifdef G__ASM_DBG
		  if(G__asm_dbg) 
		    G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp-1);
#endif
		}
#endif
		break;
	      }
	    }
	    G__memberfunc_struct_offset = store_memberfunc_struct_offset;
	    G__memberfunc_tagnum = store_memberfunc_tagnum;
	    G__exec_memberfunc = store_exec_memberfunc;
	    G__tagnum=store_tagnum;
	    G__store_struct_offset = store_struct_offset;
	  }
#endif /* 1341 */
#ifndef G__OLDIMPLEMENTATION1554
	  else if(-1!=i && fpara.paran==1) {
	    G__fprinterr(G__serr,"Error: No matching constructor for explicit conversion %s",item);
	    G__genericerror((char*)NULL);
	  }
#endif
#ifndef G__OLDIMPLEMENTATION641
	  /* omitted constructor, return uninitialized object */
	  *known3 = 1;
#ifndef G__OLDIMPLEMENTATION1515
	  if(oprp) *known3 = G__additional_parenthesis(&result3,&fpara);
#endif
	  return(result3);
#else
	  G__pop_tempobject();
#endif
	}
	else {
	  /* Return '*this' as result */
#ifndef G__OLDIMPLEMENTATION1515
	  if(oprp) *known3 = G__additional_parenthesis(&result3,&fpara);
#endif
	  return(result3);
	}
      }
      i++;
    } /* while(i<G__struct.alltag) */

#ifndef G__OLDIMPLEMENTATION571
    result3.ref=0;
#endif
    if(G__explicit_fundamental_typeconv(funcname,classhash,&fpara,&result3)) {
      *known3=1;
      G__exec_memberfunc = store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
#ifndef G__OLDIMPLEMENTATION1515
      if(oprp) *known3 = G__additional_parenthesis(&result3,&fpara);
#endif
      return(result3);
    }

  } /* if(G__TRYNORMAL==memfunc_flag) */

  
  if(G__parenthesisovld(&result3,funcname,&fpara,G__TRYNORMAL)) {
    *known3=1;
#ifndef G__OLDIMPLEMENTATION405
    if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
       (isupper(result3.type)||'u'==result3.type)
#else
       isupper(result3.type)
#endif
       ) {
      G__getindexedvalue(&result3,fpara.parameter[nindex]);
    }
#endif
#ifndef G__OLDIMPLEMENTATION1871
    else if(nindex&&'u'==result3.type) {
      int len;
      strcpy(fpara.parameter[0],fpara.parameter[nindex]+1);
      len = strlen(fpara.parameter[0]);
      if(len>1) fpara.parameter[0][len-1]=0;
      fpara.para[0] = G__getexpr(fpara.parameter[0]);
      fpara.paran=1;
      G__parenthesisovldobj(&result3,&result3,"operator[]"
			    ,&fpara,G__TRYNORMAL);
    }
#endif
#ifndef G__OLDIMPLEMENTATION1515
    if(oprp) *known3 = G__additional_parenthesis(&result3,&fpara);
#endif
    return(result3);
  }

#ifndef G__OLDIMPLEMENTATION403
  /********************************************************************
  * pointer to function described like normal function
  * int (*p2f)(void);  p2f(); 
  ********************************************************************/
  var = G__getvarentry(funcname,hash,&ig15,&G__global,G__p_local);
  if(var) {
    sprintf(result7,"*%s",funcname);
    *known3=0;
    pfparam=strchr(item,'(');
#ifndef G__OLDIMPLEMENTATION809
    p2ffpara = &fpara;
#endif
    result3=G__pointer2func((G__value*)NULL,result7,pfparam,known3);
#ifndef G__OLDIMPLEMENTATION809
    p2ffpara=(struct G__param*)NULL;
#endif
    if(*known3) {
      G__exec_memberfunc = store_exec_memberfunc;
      G__memberfunc_tagnum=store_memberfunc_tagnum;
      G__memberfunc_struct_offset=store_memberfunc_struct_offset;
#ifndef G__OLDIMPLEMENTATION405
      if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
	 (isupper(result3.type)||'u'==result3.type)
#else
	 isupper(result3.type)
#endif
	 ) {
	G__getindexedvalue(&result3,fpara.parameter[nindex]);
      }
#endif
#ifndef G__OLDIMPLEMENTATION1515
      if(oprp) *known3 = G__additional_parenthesis(&result3,&fpara);
#endif
      return(result3);
    }
  }
#endif
  
  *known3=0;
  /* bug fix, together with G__OLDIMPLEMENTATION29 */
#ifdef G__DUMPFILE
  if(G__dumpfile!=NULL && 0==G__no_exec_compile) G__dumpspace -= 3;
#endif
  
  G__exec_memberfunc = store_exec_memberfunc;
  G__memberfunc_tagnum=store_memberfunc_tagnum;
  G__memberfunc_struct_offset=store_memberfunc_struct_offset;
  

  if(!G__oprovld) {
#ifndef G__OLDIMPLEMENTATION1570
    if(G__asm_noverflow && fpara.paran) {
      G__asm_cp=store_cp_asm;
    }
    G__asm_clear_mask = 1;
#endif
    result3 = G__execfuncmacro(item,known3);
#ifndef G__OLDIMPLEMENTATION1570
    G__asm_clear_mask = 0;
#endif
    if(*known3) {
#ifndef G__OLDIMPLEMENTATION405
      if(nindex&&
#ifndef G__OLDIMPLEMENTATION1903
	 (isupper(result3.type)||'u'==result3.type)
#else
	 isupper(result3.type)
#endif
	 ) {
	G__getindexedvalue(&result3,fpara.parameter[nindex]);
      }
#endif
#ifndef G__OLDIMPLEMENTATION1515
      if(oprp) *known3 = G__additional_parenthesis(&result3,&fpara);
#endif
      return(result3);
    }
  }

  
  return(G__null);
  /* return(result3); */
  
}

#ifndef G__OLDIMPLEMENTATION1472
typedef struct {
  struct G__param* libp;
  int    ip;
} G__va_list;
/******************************************************************
 * G__va_start
 *****************************************************************/
 void G__va_start(ap)
G__value ap;
{
  struct G__var_array *local;
  struct G__ifunc_table *ifunc;
  G__va_list *va;

  local = G__p_local;
  if(!local) return;
  ifunc = local->ifunc;
  if(!ifunc) return;
  va = (G__va_list*)ap.ref;
  if(!va) return;

  va->libp = local->libp;
  va->ip = ifunc->para_nu[local->ifn];
}

#if defined(_MSC_VER) && (_MSC_VER>1200)
#pragma optimize("g",off)
#endif

/******************************************************************
 * G__va_arg
 *****************************************************************/
G__value G__va_arg(ap)
G__value ap;
{
#if defined(_MSC_VER) && (_MSC_VER==1300)
  G__genericerror("Error: You can not use va_arg because of VC++7.0(_MSC_VER==1300) problem");
  return(G__null);
#else
  G__va_list *va;
  va = (G__va_list*)ap.ref;
  if(!va || !va->libp) return(G__null);
  return(va->libp->para[va->ip++]);
#endif
}
/******************************************************************
 * G__va_end
 *****************************************************************/
void G__va_end(ap)
G__value ap;
{
  G__va_list *va;
  va = (G__va_list*)ap.ref;
  if(!va) return;
  va->libp = (struct G__param*)NULL;
}
#endif


/******************************************************************
* int G__special_func(result7,funcname,libp)
*
* Called by
*   G__getfunction()
*
******************************************************************/
int G__special_func(result7,funcname,libp,hash)
/*  return 1 if function is executed */
/*  return 0 if function isn't executed */
G__value *result7;
char *funcname;
struct G__param *libp;
int hash;
{

#ifndef G__OLDIMPLEMENTATION2068
  *result7 = G__null;
#endif

  if((hash==656)&&(strcmp(funcname,"sizeof")==0)) {
    if(libp->paran>1) {
      G__letint(result7,'i',G__Lsizeof(G__catparam(libp,libp->paran,",")));
    }
    else {
      G__letint(result7,'i',G__Lsizeof(libp->parameter[0]));
    }
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD 0x%lx from %x\n"
			     ,G__asm_cp ,G__int(*result7) ,G__asm_dt);
#endif
      G__asm_inst[G__asm_cp]=G__LD;
      G__asm_inst[G__asm_cp+1]=G__asm_dt;
      G__asm_stack[G__asm_dt] = *result7;
      G__inc_cp_asm(2,1);
    }
#endif    
    return(1);
  }
  
  if((hash==860)&&(strcmp(funcname,"offsetof")==0)) {
    if(libp->paran>2) {
      G__letint(result7,'i'
		,G__Loffsetof(G__catparam(libp,libp->paran-1,",")
			      ,libp->parameter[libp->paran-1]));
    }
    else {
      G__letint(result7,'i'
		,G__Loffsetof(libp->parameter[0],libp->parameter[1]));
    }
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD 0x%lx from %x\n"
			     ,G__asm_cp ,G__int(*result7) ,G__asm_dt);
#endif
      G__asm_inst[G__asm_cp]=G__LD;
      G__asm_inst[G__asm_cp+1]=G__asm_dt;
      G__asm_stack[G__asm_dt] = *result7;
      G__inc_cp_asm(2,1);
    }
#endif    
    return(1);
  }

#ifdef G__TYPEINFO
  if((hash==655)&&(strcmp(funcname,"typeid")==0)) {
#ifndef G__OLDIMPLEMENTATION1239
#ifdef G__ASM
    if(G__asm_noverflow) {
      G__abortbytecode();
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"COMPILE ABORT function name buffer overflow");
	G__printlinenum();
      }
#endif
    }
#endif /* G__ASM */
#endif /* 1239 */
    result7->typenum = -1;
    result7->type = 'u';
    if(G__no_exec_compile) {
      result7->tagnum = G__defined_tagname("type_info",0);
      return(1);
    }
    if(libp->paran>1) {
      G__letint(result7,'u',(long)G__typeid(G__catparam(libp,libp->paran,",")));
    }
    else {
      G__letint(result7,'u',(long)G__typeid(libp->parameter[0]));
    }
    result7->ref = result7->obj.i;
    result7->tagnum = *(int*)(result7->ref);
    return(1);
  }
#endif
  
#ifdef G__OLDIMPLEMENTATION1276
  if(((hash==466)&&(strcmp(funcname,"ASSERT")==0))||
     ((hash==658)&&(strcmp(funcname,"assert")==0))||
     ((hash==626)&&(strcmp(funcname,"Assert")==0))) {
    if(G__no_exec_compile) return(1);
    if(!G__test(libp->parameter[0])) {
#ifndef G__FONS31
      G__fprinterr(G__serr,"Assertion (%s) error: " ,libp->parameter[0]);
      G__genericerror((char*)NULL);
#else
      G__fprinterr(G__serr,"Assertion (%s) error: FILE:%s LINE:%d\n"
	      ,libp->parameter[0] ,G__ifile.name,G__ifile.line_number);
#endif
      G__letint(result7,'i',-1);
      G__pause();
    }
    else {
      G__letint(result7,'i',0);
    }
    return(1);
  }
#endif
  
#ifndef G__OLDIMPLEMENTATION1472
  if(hash==624&&strcmp(funcname,"va_arg")==0) {
    G__value x;
    if(!libp->para[0].type) x = G__getexpr(libp->parameter[0]);
    else                    x = libp->para[0];
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,
			     "%3x: LD_FUNC special %s paran=%d\n"
			     ,G__asm_cp,funcname,1);
#endif
      G__asm_inst[G__asm_cp]=G__LD_FUNC;
      G__asm_inst[G__asm_cp+1] = (long)(&G__asm_name[G__asm_name_p]);
      G__asm_inst[G__asm_cp+2]=hash;
      G__asm_inst[G__asm_cp+3]=1;
      G__asm_inst[G__asm_cp+4]=(long)G__special_func;
      if(G__asm_name_p+strlen(funcname)+1<G__ASM_FUNCNAMEBUF) {
	strcpy(G__asm_name+G__asm_name_p,funcname);
	G__asm_name_p += strlen(funcname)+1;
	G__inc_cp_asm(5,0);
      }
      else {
	G__abortbytecode();
#ifdef G__ASM_DBG
	if(G__asm_dbg) {
	  G__fprinterr(G__serr,"COMPILE ABORT function name buffer overflow");
	  G__printlinenum();
	}
#endif
      }
    }
#endif    
    if(G__no_exec_compile) return(1);
    *result7 = G__va_arg(x);
    return(1);
  }
#endif
  
  return(0);
       
}

#ifndef G__OLDIMPLEMENTATION1762
/******************************************************************
* int G__defined
*
******************************************************************/
int G__defined(tname)
char *tname;
{
  int tagnum,typenum;
  typenum = G__defined_typename(tname);
  if(-1!=typenum) return 1;
  tagnum = G__defined_tagname(tname,2);
  if(-1!=tagnum) return 1;
  return 0;
}
#endif



/******************************************************************
* int G__library_func(result7,funcname,libp,hash)
*
* Called by
*   G__getfunction()
*   G__getfunction()     referenced as pointer to assign G__asm_inst[]
*
******************************************************************/
int G__library_func(result7,funcname,libp,hash)
/*  return 1 if function is executed */
/*  return 0 if function isn't executed */
G__value *result7;
char *funcname;
struct G__param *libp;
int hash;
{
  char temp[G__LONGLINE] ;
  /* FILE *fopen(); */
  
  static int first_getopt=1;
  extern int optind;
  extern char *optarg;

#ifdef G__NO_STDLIBS
  return(0);
#endif

#ifndef G__OLDIMPLEMENTATION1192
  *result7 = G__null;
#endif
  
  /*********************************************************************
  * high priority
  *********************************************************************/
  
  if(638==hash&&strcmp(funcname,"sscanf")==0) {
    if(G__no_exec_compile) return(1);
    /* this is a fake function. not compatible with real func */
    /* para0 scan string , para1 format , para2 var pointer */
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
#endif
#ifndef G__OLDIMPLEMENTATION1192
    if(G__checkscanfarg("sscanf",libp,2)) return(1);
#endif
    switch(libp->paran) {
    case 3:
      G__letint(result7,'i'
		,sscanf((char *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2]))) ;
      break;
    case 4:
      G__letint(result7,'i'
		,sscanf((char *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3]))) ;
      break;
    case 5:
      G__letint(result7,'i'
		,sscanf((char *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4]))) ;
      break;
    case 6:
      G__letint(result7,'i'
		,sscanf((char *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4])
			,G__int(libp->para[5]))) ;
      break;
    case 7:
      G__letint(result7,'i'
		,sscanf((char *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4])
			,G__int(libp->para[5])
			,G__int(libp->para[6]))) ;
      break;
    case 8:
      G__letint(result7,'i'
		,sscanf((char *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4])
			,G__int(libp->para[5])
			,G__int(libp->para[6])
			,G__int(libp->para[7]))) ;
      break;
    case 9:
      G__letint(result7,'i'
		,sscanf((char *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4])
			,G__int(libp->para[5])
			,G__int(libp->para[6])
			,G__int(libp->para[7])
			,G__int(libp->para[8]))) ;
      break;
    case 10:
      G__letint(result7,'i'
		,sscanf((char *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4])
			,G__int(libp->para[5])
			,G__int(libp->para[6])
			,G__int(libp->para[7])
			,G__int(libp->para[8])
			,G__int(libp->para[9]))) ;
      break;
    case 11:
      G__letint(result7,'i'
		,sscanf((char *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4])
			,G__int(libp->para[5])
			,G__int(libp->para[6])
			,G__int(libp->para[7])
			,G__int(libp->para[8])
			,G__int(libp->para[9])
			,G__int(libp->para[10]))) ;
      break;
    case 12:
      G__letint(result7,'i'
		,sscanf((char *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4])
			,G__int(libp->para[5])
			,G__int(libp->para[6])
			,G__int(libp->para[7])
			,G__int(libp->para[8])
			,G__int(libp->para[9])
			,G__int(libp->para[10])
			,G__int(libp->para[11]))) ;
      break;
    default:
      G__fprinterr(G__serr,"Limitation: sscanf only takes upto 12 arguments");
      G__genericerror((char*)NULL);
      break;
    }
    return(1);
  }
  
  if(625==hash&&strcmp(funcname,"fscanf")==0) {
    if(G__no_exec_compile) return(1);
    /* this is a fake function. not compatible with real func */
    /* para0 scan string , para1 format , para2 var pointer */
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(0,'E');
    G__CHECKNONULL(1,'C');
#endif
#ifndef G__OLDIMPLEMENTATION1192
    if(G__checkscanfarg("fscanf",libp,2)) return(1);
#endif
    switch(libp->paran) {
    case 3:
      G__letint(result7,'i'
		,fscanf((FILE *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2]))) ;
      break;
    case 4:
      G__letint(result7,'i'
		,fscanf((FILE *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3]))) ;
      break;
    case 5:
      G__letint(result7,'i'
		,fscanf((FILE *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4]))) ;
      break;
    case 6:
      G__letint(result7,'i'
		,fscanf((FILE *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4])
			,G__int(libp->para[5]))) ;
      break;
    case 7:
      G__letint(result7,'i'
		,fscanf((FILE *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4])
			,G__int(libp->para[5])
			,G__int(libp->para[6]))) ;
      break;
    case 8:
      G__letint(result7,'i'
		,fscanf((FILE *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4])
			,G__int(libp->para[5])
			,G__int(libp->para[6])
			,G__int(libp->para[7]))) ;
      break;
    case 9:
      G__letint(result7,'i'
		,fscanf((FILE *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4])
			,G__int(libp->para[5])
			,G__int(libp->para[6])
			,G__int(libp->para[7])
			,G__int(libp->para[8]))) ;
      break;
    case 10:
      G__letint(result7,'i'
		,fscanf((FILE *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4])
			,G__int(libp->para[5])
			,G__int(libp->para[6])
			,G__int(libp->para[7])
			,G__int(libp->para[8])
			,G__int(libp->para[9]))) ;
      break;
    case 11:
      G__letint(result7,'i'
		,fscanf((FILE *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4])
			,G__int(libp->para[5])
			,G__int(libp->para[6])
			,G__int(libp->para[7])
			,G__int(libp->para[8])
			,G__int(libp->para[9])
			,G__int(libp->para[10]))) ;
      break;
    case 12:
      G__letint(result7,'i'
		,fscanf((FILE *)G__int(libp->para[0])
			,(char *)G__int(libp->para[1])
			,G__int(libp->para[2])
			,G__int(libp->para[3])
			,G__int(libp->para[4])
			,G__int(libp->para[5])
			,G__int(libp->para[6])
			,G__int(libp->para[7])
			,G__int(libp->para[8])
			,G__int(libp->para[9])
			,G__int(libp->para[10])
			,G__int(libp->para[11]))) ;
      break;
    default:
      G__fprinterr(G__serr,"Limitation: fscanf only takes upto 12 arguments");
      G__genericerror((char*)NULL);
      break;
    }
    return(1);
  }

  if(523==hash&&strcmp(funcname,"scanf")==0) {
    if(G__no_exec_compile) return(1);
    /* this is a fake function. not compatible with real func */
    /* para0 scan string , para1 format , para2 var pointer */
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(0,'C');
#endif
#ifndef G__OLDIMPLEMENTATION1192
    if(G__checkscanfarg("scanf",libp,1)) return(1);
#endif
    switch(libp->paran) {
#ifndef G__OLDIMPLEMENTATION713
    case 2:
      G__letint(result7,'i'
		,fscanf(G__intp_sin,(char *)G__int(libp->para[0])
		       ,G__int(libp->para[1]))) ;
      break;
    case 3:
      G__letint(result7,'i'
		,fscanf(G__intp_sin,(char *)G__int(libp->para[0])
		       ,G__int(libp->para[1])
		       ,G__int(libp->para[2]))) ;
      break;
    case 4:
      G__letint(result7,'i'
		,fscanf(G__intp_sin,(char *)G__int(libp->para[0])
		       ,G__int(libp->para[1])
		       ,G__int(libp->para[2])
		       ,G__int(libp->para[3]))) ;
      break;
    case 5:
      G__letint(result7,'i'
		,fscanf(G__intp_sin,(char *)G__int(libp->para[0])
		       ,G__int(libp->para[1])
		       ,G__int(libp->para[2])
		       ,G__int(libp->para[3])
		       ,G__int(libp->para[4]))) ;
      break;
    case 6:
      G__letint(result7,'i'
		,fscanf(G__intp_sin,(char *)G__int(libp->para[0])
		       ,G__int(libp->para[1])
		       ,G__int(libp->para[2])
		       ,G__int(libp->para[3])
		       ,G__int(libp->para[4])
		       ,G__int(libp->para[5]))) ;
      break;
    case 7:
      G__letint(result7,'i'
		,fscanf(G__intp_sin,(char *)G__int(libp->para[0])
		       ,G__int(libp->para[1])
		       ,G__int(libp->para[2])
		       ,G__int(libp->para[3])
		       ,G__int(libp->para[4])
		       ,G__int(libp->para[5])
		       ,G__int(libp->para[6]))) ;
      break;
    case 8:
      G__letint(result7,'i'
		,fscanf(G__intp_sin,(char *)G__int(libp->para[0])
		       ,G__int(libp->para[1])
		       ,G__int(libp->para[2])
		       ,G__int(libp->para[3])
		       ,G__int(libp->para[4])
		       ,G__int(libp->para[5])
		       ,G__int(libp->para[6])
		       ,G__int(libp->para[7]))) ;
      break;
    case 9:
      G__letint(result7,'i'
		,fscanf(G__intp_sin,(char *)G__int(libp->para[0])
		       ,G__int(libp->para[1])
		       ,G__int(libp->para[2])
		       ,G__int(libp->para[3])
		       ,G__int(libp->para[4])
		       ,G__int(libp->para[5])
		       ,G__int(libp->para[6])
		       ,G__int(libp->para[7])
		       ,G__int(libp->para[8]))) ;
      break;
    case 10:
      G__letint(result7,'i'
		,fscanf(G__intp_sin,(char *)G__int(libp->para[0])
		       ,G__int(libp->para[1])
		       ,G__int(libp->para[2])
		       ,G__int(libp->para[3])
		       ,G__int(libp->para[4])
		       ,G__int(libp->para[5])
		       ,G__int(libp->para[6])
		       ,G__int(libp->para[7])
		       ,G__int(libp->para[8])
		       ,G__int(libp->para[9]))) ;
      break;
    case 11:
      G__letint(result7,'i'
		,fscanf(G__intp_sin,(char *)G__int(libp->para[0])
		       ,G__int(libp->para[1])
		       ,G__int(libp->para[2])
		       ,G__int(libp->para[3])
		       ,G__int(libp->para[4])
		       ,G__int(libp->para[5])
		       ,G__int(libp->para[6])
		       ,G__int(libp->para[7])
		       ,G__int(libp->para[8])
		       ,G__int(libp->para[9])
		       ,G__int(libp->para[10]))) ;
      break;
#endif
    default:
      G__fprinterr(G__serr,"Limitation: scanf only takes upto 11 arguments");
      G__genericerror((char*)NULL);
      break;
    }
    return(1);
  }
  
  if(659==hash&&strcmp(funcname,"printf")==0) {
    if(G__no_exec_compile) return(1);
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(0,'C');
#endif
    /* para[0]:description, para[1~paran-1]: */
    G__charformatter(0,libp,temp);
#ifndef G__OLDIMPLEMENTATION713
    G__letint(result7,'i', fprintf(G__intp_sout,"%s",temp));
#else
    G__letint(result7,'i', fprintf(G__sout,"%s",temp));
#endif
    return(1);
  }
  
  if(761==hash&&strcmp(funcname,"fprintf")==0) {
    if(G__no_exec_compile) return(1);
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(0,'E');
    G__CHECKNONULL(1,'C');
#endif
    /* parameter[0]:pointer ,parameter[1]:description, para[2~paran-1]: */
    G__charformatter(1,libp,temp);
    G__letint(result7,'i',
	      fprintf((FILE *)G__int(libp->para[0]),"%s",temp));
    return(1);
  }
  
  if(774==hash&&strcmp(funcname,"sprintf")==0) {
    if(G__no_exec_compile) return(1);
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
#endif
    /* parameter[0]:charname ,para[1]:description, para[2~paran-1]: */
    G__charformatter(1,libp,temp);
    G__letint(result7,'i',
	      sprintf((char *)G__int(libp->para[0]),"%s",temp));
    return(1);
  }

  if(719==hash&&strcmp(funcname,"defined")==0) {
    /********************************************************
     * modified for multiple vartable
     ********************************************************/
    G__letint(result7,'i',G__defined_macro(libp->parameter[0]));
    return(1);
  }
  

  if(664==hash&&strcmp(funcname,"G__calc")==0) {
    if(G__no_exec_compile) return(1);
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(0,'C');
#endif
#ifndef G__OLDIMPLEMENTATION1198
    G__storerewindposition();
#endif
    *result7=G__calc_internal((char *)G__int(libp->para[0]));
#ifndef G__OLDIMPLEMENTATION1198
    G__security_recover(G__serr);
#endif
    return(1);
  }


#ifdef G__SIGNAL
  if(525==hash&&strcmp(funcname,"alarm")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'h',alarm(G__int(libp->para[0])));
    return(1);
  }

  if(638==hash&&strcmp(funcname,"signal")==0) {
    if(G__no_exec_compile) return(1);
    *result7=G__null;
    if(G__int(libp->para[1])==(long)SIG_IGN) {
      signal((int)G__int(libp->para[0]),(void (*)())SIG_IGN);
      return(1);
    }
    switch(G__int(libp->para[0])) {
    case SIGABRT:
      if(G__int(libp->para[1])==(long)SIG_DFL) {
	signal(SIGABRT,(void (*)())SIG_DFL);
      }
      else {
	G__SIGABRT = (char*)G__int(libp->para[1]);
	signal(SIGABRT,(void (*)())G__fsigabrt);
      }
      break;
    case SIGFPE:
      if(G__int(libp->para[1])==(long)SIG_DFL) {
	signal(SIGFPE,(void (*)())G__floatexception);
      }
      else {
	G__SIGFPE = (char*)G__int(libp->para[1]);
	signal(SIGFPE,(void (*)())G__fsigfpe);
      }
      break;
    case SIGILL:
      if(G__int(libp->para[1])==(long)SIG_DFL) {
	signal(SIGILL,(void (*)())SIG_DFL);
      }
      else {
	G__SIGILL = (char*)G__int(libp->para[1]);
	signal(SIGILL,(void (*)())G__fsigill);
      }
      break;
    case SIGINT:
      if(G__int(libp->para[1])==(long)SIG_DFL) {
	signal(SIGINT,(void (*)())G__breakkey);
      }
      else {
	G__SIGINT = (char*)G__int(libp->para[1]);
	signal(SIGINT,(void (*)())G__fsigint);
      }
      break;
    case SIGSEGV:
      if(G__int(libp->para[1])==(long)SIG_DFL) {
	signal(SIGSEGV,G__segmentviolation);
#ifdef SIGBUS
	signal(SIGBUS,(void (*)())G__buserror);
#endif
      }
      else {
	G__SIGSEGV = (char*)G__int(libp->para[1]);
	signal(SIGSEGV,(void (*)())G__fsigsegv);
#ifdef SIGBUS
	signal(SIGSEGV,(void (*)())G__fsigsegv);
#endif
      }
      break;
    case SIGTERM:
      if(G__int(libp->para[1])==(long)SIG_DFL) {
	signal(SIGTERM,(void (*)())SIG_DFL);
      }
      else {
	G__SIGTERM = (char*)G__int(libp->para[1]);
	signal(SIGTERM,(void (*)())G__fsigterm);
      }
      break;
#ifdef SIGHUP
    case SIGHUP:
      if(G__int(libp->para[1])==(long)SIG_DFL) {
	signal(SIGHUP,(void (*)())SIG_DFL);
      }
      else {
	G__SIGHUP = (char*)G__int(libp->para[1]);
	signal(SIGHUP,(void (*)())G__fsighup);
      }
      break;
#endif
#ifdef SIGQUIT
    case SIGQUIT:
      if(G__int(libp->para[1])==(long)SIG_DFL) {
	signal(SIGHUP,(void (*)())SIG_DFL);
      }
      else {
	G__SIGQUIT = (char*)G__int(libp->para[1]);
	signal(SIGQUIT,(void (*)())G__fsigquit);
      }
      break;
#endif
#ifdef SIGSTP
    case SIGTSTP:
      if(G__int(libp->para[1])==(long)SIG_DFL) {
	signal(SIGTSTP,(void (*)())SIG_DFL);
      }
      else {
	G__SIGTSTP = (char*)G__int(libp->para[1]);
	signal(SIGTSTP,(void (*)())G__fsigtstp);
      }
      break;
#endif
#ifdef SIGTTIN
    case SIGTTIN:
      if(G__int(libp->para[1])==(long)SIG_DFL) {
	signal(SIGTTIN,(void (*)())SIG_DFL);
      }
      else {
	G__SIGTTIN = (char*)G__int(libp->para[1]);
	signal(SIGTTIN,(void (*)())G__fsigttin);
      }
      break;
#endif
#ifdef SIGTTOU
    case SIGTTOU:
      if(G__int(libp->para[1])==(long)SIG_DFL) {
	signal(SIGTTOU,(void (*)())SIG_DFL);
      }
      else {
	G__SIGTTOU = (char*)G__int(libp->para[1]);
	signal(SIGTTOU,(void (*)())G__fsigttou);
      }
      break;
#endif
#ifdef SIGALRM
    case SIGALRM:
      if(G__int(libp->para[1])==(long)SIG_DFL) {
	signal(SIGALRM,(void (*)())SIG_DFL);
      }
      else {
	G__SIGALRM = (char*)G__int(libp->para[1]);
	signal(SIGALRM,(void (*)())G__fsigalrm);
      }
      break;
#endif
#ifdef SIGUSR1
    case SIGUSR1:
      if(G__int(libp->para[1])==(long)SIG_DFL) {
	signal(SIGUSR1,(void (*)())SIG_DFL);
      }
      else {
	G__SIGUSR1 = (char*)G__int(libp->para[1]);
	signal(SIGUSR1,(void (*)())G__fsigusr1);
      }
      break;
#endif
#ifdef SIGUSR2
    case SIGUSR2:
      if(G__int(libp->para[1])==(long)SIG_DFL) {
	signal(SIGUSR2,(void (*)())SIG_DFL);
      }
      else {
	G__SIGUSR2 = (char*)G__int(libp->para[1]);
	signal(SIGUSR2,(void (*)())G__fsigusr2);
      }
      break;
#endif
    default:
      G__genericerror("Error: Unknown signal type");
      break;
    }
    return(1);
  }
#endif
  
  if(659==hash&&strcmp(funcname,"getopt")==0) {
    if(G__no_exec_compile) return(1);
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(1,'C');
    G__CHECKNONULL(2,'C');
#endif
    if(first_getopt) {
      first_getopt=0;
      G__globalvarpointer = (long)(&optind);
      G__var_type='i';
      G__abortbytecode();
      G__getexpr("optind=1");
      G__asm_noverflow=1;
      
      G__globalvarpointer = (long)(&optarg);
      G__var_type='C';
      G__getexpr("optarg=");
    }
    G__letint(result7,'c',
	      (long)getopt((int)G__int(libp->para[0])
			   ,(char **)G__int(libp->para[1])
			   ,(char *)G__int(libp->para[2])));
    return(1);
  }

#ifndef G__OLDIMPLEMENTATION1472
  if(hash==868&&strcmp(funcname,"va_start")==0) {
    if(G__no_exec_compile) return(1);
    G__va_start(libp->para[0]);
    *result7 = G__null;
    return(1);
  }
  
  if(hash==621&&strcmp(funcname,"va_end")==0) {
    if(G__no_exec_compile) return(1);
    G__va_end(libp->para[0]);
    *result7 = G__null;
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1472
  if(hash==1835&&strcmp(funcname,"G__va_arg_setalign")==0) {
    if(G__no_exec_compile) return(1);
    G__va_arg_setalign((int)G__int(libp->para[0]));
    *result7 = G__null;
    return(1);
  }
#endif

  
  if(1093==hash&&strcmp(funcname,"G__loadfile")==0) {
    if(G__no_exec_compile) return(1);
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(0,'C');
#endif
    G__letint(result7,'i'
	      ,(long)G__loadfile((char *)G__int(libp->para[0])));
    return(1);
  }
  
  if(1320==hash&&strcmp(funcname,"G__unloadfile")==0) {
    if(G__no_exec_compile) return(1);
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(0,'C');
#endif
    G__letint(result7,'i',(long)G__unloadfile((char *)G__int(libp->para[0])));
    return(1);
  }
  
  if(1308==hash&&strcmp(funcname,"G__reloadfile")==0) {
    if(G__no_exec_compile) return(1);
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(0,'C');
#endif
    G__unloadfile((char *)G__int(libp->para[0]));
    G__letint(result7,'i'
	      ,(long)G__loadfile((char *)G__int(libp->para[0])));
    return(1);
  }

#ifndef G__OLDIMPLEMENTATION1273
  if(1882==hash&&strcmp(funcname,"G__set_smartunload")==0) {
    if(G__no_exec_compile) return(1);
    *result7=G__null;
    G__set_smartunload((int)G__int(libp->para[0]));
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION832
  if(1655==hash&&strcmp(funcname,"G__charformatter")==0) {
    if(G__no_exec_compile) {
      G__abortbytecode();
      return(1);
    }
    G__CHECKNONULL(1,'C');
    G__charformatter((int)G__int(libp->para[0]),G__p_local->libp
		     ,(char*)G__int(libp->para[1]));
    G__letint(result7,'C',G__int(libp->para[1]));
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION863
  if(1023==hash&&strcmp(funcname,"G__findsym")==0) {
    if(G__no_exec_compile) {
      G__abortbytecode();
      return(1);
    }
    G__CHECKNONULL(0,'C');
    G__letint(result7,'Y',(long)G__findsym((char*)G__int(libp->para[0])));
    return(1);
  }
#endif

  if(2210==hash&&strcmp(funcname,"G__set_sym_underscore")==0) {
    if(G__no_exec_compile) {
      G__abortbytecode();
      return(1);
    }
    G__set_sym_underscore((int)G__int(libp->para[0]));
    *result7=G__null;
    return(1);
  }

  if(2198==hash&&strcmp(funcname,"G__get_sym_underscore")==0) {
    if(G__no_exec_compile) {
      G__abortbytecode();
      return(1);
    }
    G__letint(result7,'i',(long)G__get_sym_underscore());
    return(1);
  }

#ifndef G__OLDIMPLEMENTATION1034
  if(1162==hash&&strcmp(funcname,"G__IsInMacro")==0) {
    if(G__no_exec_compile) {
      G__abortbytecode();
      return(1);
    }
    G__letint(result7,'i',(long)G__IsInMacro());
    return(1);
  }
#endif

  if(1423==hash&&strcmp(funcname,"G__getmakeinfo")==0) {
    if(G__no_exec_compile) {
      G__abortbytecode();
      return(1);
    }
    G__letint(result7,'C',(long)G__getmakeinfo((char*)G__int(libp->para[0])));
    return(1);
  }

#ifndef G__OLDIMPLEMENTATION1485
  if(1249==hash&&strcmp(funcname,"G__fprinterr")==0) {
    if(G__no_exec_compile) return(1);
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(0,'C');
#endif
    /* para[0]:description, para[1~paran-1]: */
    G__charformatter(0,libp,temp);
    G__letint(result7,'i', G__fprinterr(G__serr,"%s",temp));
    return(1);
  }
#endif

  /*********************************************************************
  * low priority 2
  *********************************************************************/
  if(569==hash&&strcmp(funcname,"qsort")==0) {
    if(G__no_exec_compile) return(1);
#ifndef G__MASKERROR
    if(4!=libp->paran)
      G__printerror("qsort",1,libp->paran);
#endif
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(3,'Y');
#endif
    qsort((void *)G__int(libp->para[0])
	  ,(size_t)G__int(libp->para[1])
	  ,(size_t)G__int(libp->para[2])
	  ,(int (*)())G__int(libp->para[3]) 
	  /* ,(int (*)(void *arg1,void *argv2))G__int(libp->para[3]) */
	  );
    *result7=G__null;
    return(1);
  }
  
  if(728==hash&&strcmp(funcname,"bsearch")==0) {
    if(G__no_exec_compile) return(1);
#ifndef G__MASKERROR
    if(5!=libp->paran)
      G__printerror("bsearch",1,libp->paran);
#endif
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(3,'Y');
#endif
    bsearch((void *)G__int(libp->para[0])
	    ,(void *)G__int(libp->para[1])
	    ,(size_t)G__int(libp->para[2])
	    ,(size_t)G__int(libp->para[3])
	    ,(int (*)())G__int(libp->para[4]) 
	    );
    *result7=G__null;
    return(1);
  }

  if(strcmp(funcname,"$read")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i'
	      ,(long)G__textprocessing((FILE *)G__int(libp->para[0])));
    return(1);
  }
  
#if defined(G__REGEXP) || defined(G__REGEXP1)
  if(strcmp(funcname,"$regex")==0) {
    if(G__no_exec_compile) return(1);
    switch(libp->paran) {
    case 1:
      G__letint(result7,'i'
		,(long)G__matchregex((char*)G__int(libp->para[0])
				     ,G__arg[0]));
      break;
    case 2:
      G__letint(result7,'i'
		,(long)G__matchregex((char*)G__int(libp->para[0])
				     ,(char*)G__int(libp->para[1])));
      break;
    }
    return(1);
  }
#endif
  
  if(strcmp(funcname,"$")==0) {
    if(G__no_exec_compile) return(1);
    *result7=G__getrsvd((int)G__int(libp->para[0]));
    return(1);
  }

#ifndef G__OLDIMPLEMENTATION464
  if(1631==hash&&strcmp(funcname,"G__exec_tempfile")==0) {
    if(G__no_exec_compile) return(1);
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(0,'C');
#endif
    *result7=G__exec_tempfile((char *)G__int(libp->para[0]));
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1546
  if(1225==hash&&strcmp(funcname,"G__load_text")==0) {
    if(G__no_exec_compile) return(1);
    G__CHECKNONULL(0,'C');
    G__storerewindposition();
    G__letint(result7,'C',(long)G__load_text((char *)G__int(libp->para[0])));
    G__security_recover(G__serr);
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1348
  if(1230==hash&&strcmp(funcname,"G__exec_text")==0) {
    if(G__no_exec_compile) return(1);
    G__CHECKNONULL(0,'C');
    G__storerewindposition();
    *result7=G__exec_text((char *)G__int(libp->para[0]));
    G__security_recover(G__serr);
    return(1);
  }

  if(1431==hash&&strcmp(funcname,"G__process_cmd")==0) {
    if(G__no_exec_compile) return(1);
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__CHECKNONULL(2,'I');
    G__storerewindposition();
    *result7 = G__null;
    G__letint(result7,'i',G__process_cmd((char*)G__int(libp->para[0])
					  ,(char*)G__int(libp->para[1])
					  ,(int*)G__int(libp->para[2])
					  ,(int*)G__int(libp->para[3])
					  ,(G__value*)G__int(libp->para[4])
					 ));
    G__security_recover(G__serr);
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1867
  if(1670==hash&&strcmp(funcname,"G__exec_text_str")==0) {
    if(G__no_exec_compile) return(1);
    G__CHECKNONULL(0,'C');
    G__CHECKNONULL(1,'C');
    G__storerewindposition();
    G__letint(result7,'C',
	      (long)G__exec_text_str((char *)G__int(libp->para[0]),
				     (char *)G__int(libp->para[1])));
    G__security_recover(G__serr);
    return(1);
  }
#endif

  /*********************************************************************
  * low priority
  *********************************************************************/

  if(442==hash&&strcmp(funcname,"exit")==0) {
    if(G__no_exec_compile) return(1);
    if(G__atexit) G__call_atexit(); /* Reduntant, also done in G__main() */
    G__return=G__RETURN_EXIT2;
    G__letint(result7,'i',G__int(libp->para[0]));
#ifndef G__OLDIMPLEMENTATION1644
    G__exitcode = result7->obj.i;
#endif
    return(1);
  }
  
  if(655==hash&&strcmp(funcname,"atexit")==0) {
    if(G__no_exec_compile) return(1);
    if(G__int(libp->para[0])==0) {
      /* function wasn't registered */
      G__letint(result7,'i',1);
    }
    else {
      /* function was registered */
      G__atexit = (char *)G__int(libp->para[0]);
      G__letint(result7,'i',0);
    }
    return(1);
  }

#ifndef G__OLDIMPLEMENTATION1276
  if(((hash==466)&&(strcmp(funcname,"ASSERT")==0))||
     ((hash==658)&&(strcmp(funcname,"assert")==0))||
     ((hash==626)&&(strcmp(funcname,"Assert")==0))) {
    if(G__no_exec_compile) return(1);
    if(
#ifndef G__OLDIMPLEMENTATION2208
       !G__bool(libp->para[0])
#else
       !G__int(libp->para[0])
#endif
       ) {
#ifndef G__FONS31
      G__fprinterr(G__serr,"Assertion (%s) error: " ,libp->parameter[0]);
      G__genericerror((char*)NULL);
#else
      G__fprinterr(G__serr,"Assertion (%s) error: FILE:%s LINE:%d\n"
	      ,libp->parameter[0] ,G__ifile.name,G__ifile.line_number);
#endif
      G__letint(result7,'i',-1);
      G__pause();
    }
    else {
      G__letint(result7,'i',0);
    }
    return(1);
  }
#endif
  
  if(803==hash&&strcmp(funcname,"G__pause")==0) {
    if(G__no_exec_compile) return(1);
    /* pause */
    G__letint(result7,'i',(long)G__pause());
    return(1);
  }

#ifndef G__OLDIMPLEMENTATION875
  if(1443==hash&&strcmp(funcname,"G__set_atpause")==0) {
    if(G__no_exec_compile) return(1);
    G__set_atpause((void (*)())G__int(libp->para[0]));
    *result7=G__null;
    return(1);
  }

  if(1455==hash&&strcmp(funcname,"G__set_aterror")==0) {
    if(G__no_exec_compile) return(1);
    G__set_aterror((void (*)())G__int(libp->para[0]));
    *result7=G__null;
    return(1);
  }
#endif
  
  if(821==hash&&strcmp(funcname,"G__input")==0) {
    if(G__no_exec_compile) return(1);
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(0,'C');
#endif
    G__letint(result7,'C',(long)G__input((char *)G__int(libp->para[0])));
    return(1);
  }

  if(strcmp(funcname,"G__add_ipath")==0) {
    if(G__no_exec_compile) return(1);
    G__add_ipath((char*)G__int(libp->para[0]));
    *result7 = G__null;
    return(1);
  }

#ifndef G__OLDIMPLEMENTATION2119
  if(strcmp(funcname,"G__delete_ipath")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i',(long)G__delete_ipath((char*)G__int(libp->para[0])));
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1963
  if(strcmp(funcname,"G__SetCINTSYSDIR")==0) {
    if(G__no_exec_compile) return(1);
    G__SetCINTSYSDIR((char*)G__int(libp->para[0]));
    *result7 = G__null;
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1964
  if(strcmp(funcname,"G__set_eolcallback")==0) {
    if(G__no_exec_compile) return(1);
    G__set_eolcallback((void*)G__int(libp->para[0]));
    *result7 = G__null;
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1937
  if(strcmp(funcname,"G__set_history_size")==0) {
    if(G__no_exec_compile) return(1);
    G__set_history_size((int)G__int(libp->para[0]));
    *result7 = G__null;
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1485
  if(strcmp(funcname,"G__set_errmsgcallback")==0) {
    if(G__no_exec_compile) return(1);
    G__set_errmsgcallback((void*)G__int(libp->para[0]));
    *result7 = G__null;
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1932
  if(strcmp(funcname,"G__chdir")==0) {
#if defined(G__WIN32) || defined(G__POSIX)
    char *stringb=(char*)G__int(libp->para[0]);
#endif
    if(G__no_exec_compile) return(1);
    G__CHECKNONULL(0,'C');
#if defined(G__WIN32)
    if(FALSE==SetCurrentDirectory(stringb))
      G__fprinterr(G__serr,"can not change directory to %s\n",stringb);
#elif defined(G__POSIX)
    if(0!=chdir(stringb))
      G__fprinterr(G__serr,"can not change directory to %s\n",stringb);
#endif
    *result7 = G__null;
    return(1);
  }
#endif

#ifdef G__SHMGLOBAL
  if(strcmp(funcname,"G__shminit")==0) {
    if(G__no_exec_compile) return(1);
    G__shminit();
    *result7 = G__null;
    return(1);
  }
  if(strcmp(funcname,"G__shmmalloc")==0) {
    if(G__no_exec_compile) return(1);
    *result7 = G__null;
    G__letint(result7,'E',(long)G__shmmalloc((int)G__int(libp->para[0])));
    return(1);
  }
  if(strcmp(funcname,"G__shmcalloc")==0) {
    if(G__no_exec_compile) return(1);
    *result7 = G__null;
    G__letint(result7,'E',(long)G__shmcalloc((int)G__int(libp->para[0]),(int)G__int(libp->para[1])));
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION562
  if(strcmp(funcname,"G__setautoconsole")==0) {
    if(G__no_exec_compile) return(1);
    G__setautoconsole((int)G__int(libp->para[0]));
    *result7 = G__null;
    return(1);
  }
  if(strcmp(funcname,"G__AllocConsole")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i',(long)G__AllocConsole());
    return(1);
  }
  if(strcmp(funcname,"G__FreeConsole")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i',(long)G__FreeConsole());
    return(1);
  }
#endif

#ifdef G__TYPEINFO
  if(strcmp(funcname,"G__type2string")==0) {
    if(G__no_exec_compile) return(1);
#ifndef G__OLDIMPLEMENTATION401
    G__letint(result7,'C' ,(long)G__type2string((int)G__int(libp->para[0]),
						(int)G__int(libp->para[1]),
						(int)G__int(libp->para[2]),
						(int)G__int(libp->para[3]),
						(int)G__int(libp->para[4])));
#else
    G__letint(result7,'C' ,(long)G__type2string((int)G__int(libp->para[0]),
						(int)G__int(libp->para[1]),
						(int)G__int(libp->para[2]),
						(int)G__int(libp->para[3])));
#endif
    return(1);
  }

  if(strcmp(funcname,"G__typeid")==0) {
    result7->typenum = -1;
    result7->type = 'u';
    if(G__no_exec_compile) {
      result7->tagnum = G__defined_tagname("type_info",0);
      return(1);
    }
    G__letint(result7,'u',(long)G__typeid((char*)G__int(libp->para[0])));
    result7->ref = result7->obj.i;
    result7->tagnum = *(int*)(result7->ref);
    return(1);
  }
#endif

#ifdef G__FONS_TYPEINFO
  if(strcmp(funcname,"G__get_classinfo")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'l' ,G__get_classinfo((char*)G__int(libp->para[0]),
					    (int)G__int(libp->para[1])));
    return(1);
  }

  if(strcmp(funcname,"G__get_variableinfo")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'l' ,G__get_variableinfo((char*)G__int(libp->para[0]),
					       (long*)G__int(libp->para[1]),
					       (long*)G__int(libp->para[2]),
					       (int)G__int(libp->para[3])));
    return(1);
  }

  if(strcmp(funcname,"G__get_functioninfo")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'l' ,G__get_functioninfo((char*)G__int(libp->para[0]),
					       (long*)G__int(libp->para[1]),
					       (long*)G__int(libp->para[2]),
					       (int)G__int(libp->para[3])));
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1198
  if(strcmp(funcname,"G__lasterror_filename")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'C',(long)G__lasterror_filename());
    return(1);
  }

  if(strcmp(funcname,"G__lasterror_linenum")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i',(long)G__lasterror_linenum());
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1207
  if(strcmp(funcname,"G__loadsystemfile")==0) {
    if(G__no_exec_compile) return(1);
#ifndef G__OLDIMPLEMENTATION575
    G__CHECKNONULL(0,'C');
#endif
    G__letint(result7,'i'
	      ,(long)G__loadsystemfile((char *)G__int(libp->para[0])));
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1210
  if(strcmp(funcname,"G__set_ignoreinclude")==0) {
    if(G__no_exec_compile) return(1);
    G__set_ignoreinclude((G__IgnoreInclude)G__int(libp->para[0]));
    *result7 = G__null;
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1781
  if(strcmp(funcname,"G__ForceBytecodecompilation")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i'
	      ,(long)G__ForceBytecodecompilation((char*)G__int(libp->para[0])
						 ,(char*)G__int(libp->para[1])
						 ));
    return(1);
  }
#endif

#ifndef G__SMALLOBJECT
  

#ifdef G__TRUEP2F
  if(strcmp(funcname,"G__p2f2funcname")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'C'
	      ,(long)G__p2f2funcname((void*)G__int(libp->para[0])));
    return(1);
  }

  if(strcmp(funcname,"G__isinterpretedp2f")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i'
	      ,(long)G__isinterpretedp2f((void*)G__int(libp->para[0])));
    return(1);
  }
#endif
  
#ifndef G__OLDIMPLEMENTATION546
  if(strcmp(funcname,"G__deleteglobal")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i'
	      ,(long)G__deleteglobal((void*)G__int(libp->para[0])));
    return(1);
  }
#endif

  if(strcmp(funcname,"G__display_tempobject")==0) {
    if(G__no_exec_compile) return(1);
    *result7 = G__null;
    G__display_tempobject("");
    return(1);
  }

  if(strcmp(funcname,"G__cmparray")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i'
	      ,(long)G__cmparray((short *)G__int(libp->para[0])
				 ,(short *)G__int(libp->para[1])
				 ,(int)G__int(libp->para[2])
				 ,(short)G__int(libp->para[3])));
    return(1);
  }
  
  if(strcmp(funcname,"G__setarray")==0) {
    if(G__no_exec_compile) return(1);
    G__setarray((short *)G__int(libp->para[0])
		,(int)G__int(libp->para[1])
		,(short)G__int(libp->para[2])
		,(char *)G__int(libp->para[3]));
    *result7=G__null;
    return(1);
  }

#ifndef G__OLDIMPLEMENTATION488
  if(strcmp(funcname,"G__deletevariable")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i'
	      ,(long)G__deletevariable((char *)G__int(libp->para[0])));
    return(1);
  }
#endif

#ifndef G__NEVER
  if(strcmp(funcname,"G__split")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i'
	      ,(long)G__split((char *)G__int(libp->para[0]),
			      (char *)G__int(libp->para[1]),
			      (int *)G__int(libp->para[2]),
			      (char **)G__int(libp->para[3])));
    return(1);
  }
  
  if(strcmp(funcname,"G__readline")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i'
	      ,(long)G__readline((FILE*)G__int(libp->para[0]),
				 (char*)G__int(libp->para[1]),
				 (char*)G__int(libp->para[2]),
				 (int*)G__int(libp->para[3]),
				 (char**)G__int(libp->para[4])));
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION564
  if(strcmp(funcname,"G__setbreakpoint")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i',(long)G__setbreakpoint((char*)G__int(libp->para[0]),(char*)G__int(libp->para[1])));
    return(1);
  }
#endif
  
#ifndef G__OLDIMPLEMENTATION564
  if(strcmp(funcname,"G__tracemode")==0||
     strcmp(funcname,"G__debugmode")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i',(long)G__tracemode((int)G__int(libp->para[0])));
    return(1);
  }
  
  if(strcmp(funcname,"G__stepmode")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i',(long)G__stepmode((int)G__int(libp->para[0])));
    return(1);
  }

  if(strcmp(funcname,"G__gettracemode")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i',(long)G__gettracemode());
    return(1);
  }
  
  if(strcmp(funcname,"G__getstepmode")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i',(long)G__getstepmode());
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION2226
  if(strcmp(funcname,"G__setmemtestbreak")==0) {
    if(G__no_exec_compile) return(1);
    *result7=G__null;
    G__setmemtestbreak((int)G__int(libp->para[0]),(int)G__int(libp->para[1]));
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1142
  if(strcmp(funcname,"G__optimizemode")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i',(long)G__optimizemode((int)G__int(libp->para[0])));
    return(1);
  }
  if(strcmp(funcname,"G__getoptimizemode")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i',(long)G__getoptimizemode());
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1618
  if(strcmp(funcname,"G__bytecodedebugmode")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i',(long)G__bytecodedebugmode((int)G__int(libp->para[0])));
    return(1);
  }
  if(strcmp(funcname,"G__getbytecodedebugmode")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i',(long)G__getbytecodedebugmode());
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION478
  if(strcmp(funcname,"G__clearerror")==0) {
    if(G__no_exec_compile) return(1);
    G__return=G__RETURN_NON;
    G__security_error=G__NOERROR;
    *result7 = G__null;
    return(1);
  }
#endif
  
  if(strcmp(funcname,"G__setbreakpoint")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i'
	      ,(long)G__setbreakpoint((char*)G__int(libp->para[0])
				      ,(char*)G__int(libp->para[1])));
    return(1);
  }
  
  if(strcmp(funcname,"G__showstack")==0) {
    if(G__no_exec_compile) return(1);
    G__showstack((FILE*)G__int(libp->para[0]));
    *result7=G__null;
    return(1);
  }
  
  if(strcmp(funcname,"G__graph")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i'
	      ,(long)G__graph((double *)G__int(libp->para[0]),
			      (double *)G__int(libp->para[1]),
			      (int)G__int(libp->para[2]),
			      (char*)G__int(libp->para[3]),
			      (int)G__int(libp->para[4])));
    return(1);
  }

#ifndef G__NSEARCHMEMBER
  if(strcmp(funcname,"G__search_next_member")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'C'
	      ,(long)G__search_next_member((char *)G__int(libp->para[0])
					   ,(int)G__int(libp->para[1])));
    return(1);
  }
  
  if(strcmp(funcname,"G__what_type")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'Y',(long)G__what_type((char *)G__int(libp->para[0])
					     ,(char *)G__int(libp->para[1])
					     ,(char *)G__int(libp->para[2])
					     ,(char *)G__int(libp->para[3])
					     ));
    return(1);
  }

#ifndef G__OLDIMPLEMENTATION1815
  if(strcmp(funcname,"G__SetCatchException")==0) {
    if(G__no_exec_compile) return(1);
    G__SetCatchException((int)G__int(libp->para[0]));
    *result7=G__null;
    return(1);
  }
#endif
  
#endif
  
#ifndef G__NSTOREOBJECT
  if(strcmp(funcname,"G__storeobject")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i',(long)G__storeobject((&libp->para[0]),
					       (&libp->para[1])));
    return(1);
  }
  
  if(strcmp(funcname,"G__scanobject")==0) {
    if(G__no_exec_compile) return(1);
    
    G__letint(result7,'i',(long)G__scanobject((&libp->para[0])));
    return(1);
  }

  if(strcmp(funcname,"G__dumpobject")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i'
	      ,(long)G__dumpobject((char *)G__int(libp->para[0])
				   ,(void *)G__int(libp->para[1])
				   ,(int)G__int(libp->para[2])));
    return(1);
  }
  
  if(strcmp(funcname,"G__loadobject")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i'
	      ,(long)G__loadobject((char *)G__int(libp->para[0])
				   ,(void *)G__int(libp->para[1])
				   ,(int)G__int(libp->para[2])));
    return(1);
  }
#endif
  
  if(strcmp(funcname,"G__lock_variable")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i'
	      ,(long)G__lock_variable((char *)G__int(libp->para[0])));
    return(1);
  }
  
  if(strcmp(funcname,"G__unlock_variable")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i'
	      ,(long)G__unlock_variable((char *)G__int(libp->para[0])));
    return(1);
  }

  if(strcmp(funcname,"G__dispvalue")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i'
	      ,(long)G__dispvalue((FILE*)libp->para[0].obj.i,&libp->para[1]));
    return(1);
  }

#ifndef G__OLDIMPLEMENTATION2014
  if(strcmp(funcname,"G__set_class_autoloading_table")==0) {
    if(G__no_exec_compile) return(1);
    G__set_class_autoloading_table((char *)G__int(libp->para[0])
				   ,(char*)G__int(libp->para[1]));
    *result7 = G__null;
    return(1);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1762
  if(strcmp(funcname,"G__defined")==0) {
    if(G__no_exec_compile) return(1);
    G__letint(result7,'i',(long)G__defined((char *)G__int(libp->para[0])));
    return(1);
  }
#endif


#ifndef G__OLDIMPLEMENTATION2051
  if(strcmp(funcname,"G__cintv6")==0) {
    if(G__no_exec_compile) return(1);
    *result7 = G__null;
    G__cpp_setupG__API();
    G__cpp_setupG__bcdict();
    return(1);
  }
#endif


#endif /* G__SMALLOBJECT */
  


  
  return(0);
  
}

#ifndef G__OLDIMPLEMENTATION827
/******************************************************************
* G__printf_error()
******************************************************************/
void G__printf_error() 
{
  G__fprinterr(G__serr,"Limitation: printf string too long. Upto %d. Use fputs()"
	  ,G__LONGLINE);
  G__genericerror((char*)NULL);
}

#define G__PRINTF_ERROR(COND)                       \
   if(COND) { G__printf_error(); return(result); }  

#endif

#ifndef G__OLDIMPLEMENTATION2189
void G__sprintformatll(char* result,const char* fmt,void *p,char* buf) {
  G__int64 *pll = (G__int64*)p;
  sprintf(buf,fmt,result,*pll);
  strcpy(result,buf);
}
void G__sprintformatull(char* result,const char* fmt,void *p,char* buf) {
  G__uint64 *pll = (G__uint64*)p;
  sprintf(buf,fmt,result,*pll);
  strcpy(result,buf);
}
void G__sprintformatld(char* result,const char* fmt,void *p,char* buf) {
  long double *pld = (long double*)p;
  sprintf(buf,fmt,result,*pld);
  strcpy(result,buf);
}
#endif

/******************************************************************
* char *G__charformatter(ifmt,libp,outbuf)
*
******************************************************************/
char *G__charformatter(ifmt,libp,result)
int ifmt;
struct G__param *libp;
char *result;
{
  int ipara,ichar,lenfmt;
  int ionefmt=0,fmtflag=0;
  char onefmt[G__LONGLINE],fmt[G__LONGLINE];
  char pformat[G__LONGLINE];
  short dig=0;
#ifndef G__OLDIMPLEMENTATION2083
  int usedpara = 0;
#endif
  
  strcpy(pformat,(char *)G__int(libp->para[ifmt]));
  result[0]='\0';
  ipara=ifmt+1;
  lenfmt = strlen(pformat);
  for(ichar=0;ichar<=lenfmt;ichar++) {
    switch(pformat[ichar]) {
    case '\0': /* end of the format */
      onefmt[ionefmt]='\0';
      sprintf(fmt,"%%s%s",onefmt);
      sprintf(onefmt,fmt,result);
      strcpy(result,onefmt);
      ionefmt=0;
      break;
    case 's': /* string */
      onefmt[ionefmt++]=pformat[ichar];
      if(fmtflag==1) {
	onefmt[ionefmt]='\0';
#ifndef G__OLDIMPLEMENTATION827
	if(libp->para[ipara].obj.i) {
#ifndef G__OLDIMPLEMENTATION2083
	  G__PRINTF_ERROR(strlen(onefmt)+strlen(result)+
	     strlen((char*)G__int(libp->para[usedpara]))>=G__LONGLINE)
#else
	  G__PRINTF_ERROR(strlen(onefmt)+strlen(result)+
	     strlen((char*)G__int(libp->para[ipara]))>=G__LONGLINE)
#endif
	  sprintf(fmt,"%%s%s",onefmt);
#ifndef G__OLDIMPLEMENTATION2083
	  sprintf(onefmt,fmt,result ,(char *)G__int(libp->para[usedpara]));
#else
	  sprintf(onefmt,fmt,result ,(char *)G__int(libp->para[ipara]));
#endif
	  strcpy(result,onefmt);
        }
#else
	sprintf(fmt,"%%s%s",onefmt);
#ifndef G__OLDIMPLEMENTATION2083
	sprintf(onefmt,fmt,result ,(char *)G__int(libp->para[usedpara]));
#else
	sprintf(onefmt,fmt,result ,(char *)G__int(libp->para[ipara]));
#endif
	strcpy(result,onefmt);
#endif
	ipara++;
	ionefmt=0;
	fmtflag=0;
      }
      break;
    case 'c': /* char */
      onefmt[ionefmt++]=pformat[ichar];
      if(fmtflag==1) {
	onefmt[ionefmt]='\0';
	sprintf(fmt,"%%s%s",onefmt);
#ifndef G__OLDIMPLEMENTATION2083
	sprintf(onefmt,fmt,result ,(char)G__int(libp->para[usedpara]));
#else
	sprintf(onefmt,fmt,result ,(char)G__int(libp->para[ipara]));
#endif
	strcpy(result,onefmt);
	ipara++;
	ionefmt=0;
	fmtflag=0;
      }
      break;
    case 'b': /* int */
      onefmt[ionefmt++]=pformat[ichar];
      if(fmtflag==1) {
	onefmt[ionefmt-1]='s';
	onefmt[ionefmt]='\0';
	sprintf(fmt,"%%s%s",onefmt);
#ifndef G__OLDIMPLEMENTATION2083
	G__logicstring(libp->para[usedpara],dig,onefmt);
	ipara++;
#else
	G__logicstring(libp->para[ipara++],dig,onefmt);
#endif
	sprintf(result,fmt,result,onefmt);
	ionefmt=0;
      }
      break;
    case 'd': /* int */
    case 'i': /* int */
    case 'u': /* unsigned int */
    case 'o': /* octal */
    case 'x': /* hex */
    case 'X': /* HEX */
#ifndef G__OLDIMPLEMENTATION935
    case 'p': /* pointer */
#endif
      onefmt[ionefmt++]=pformat[ichar];
      if(fmtflag==1) {
	onefmt[ionefmt]='\0';
	sprintf(fmt,"%%s%s",onefmt);
#ifndef G__OLDIMPLEMENTATION2189
	if('n'==libp->para[usedpara].type) {
	  G__value *pval = &libp->para[usedpara]; ipara++;
	  G__sprintformatll(result,fmt,&pval->obj.ll,onefmt);
	}
	else if('m'==libp->para[usedpara].type) {
	  G__value *pval = &libp->para[usedpara]; ipara++;
	  G__sprintformatull(result,fmt,&pval->obj.ull,onefmt);
	}
	else
#endif
#ifndef G__OLDIMPLEMENTATION1739
	if(
#ifndef G__OLDIMPLEMENTATION2083
	    'u'==libp->para[usedpara].type
#else
	    'u'==libp->para[ipara].type
#endif
	    ) {
	  char llbuf[100];
#ifndef G__OLDIMPLEMENTATION2083
	  G__value *pval = &libp->para[usedpara]; ipara++;
#else
	  G__value *pval = &libp->para[ipara++];
#endif
	  if(strcmp(G__struct.name[pval->tagnum],"G__longlong")==0) {
	    sprintf(llbuf
	    ,"G__printformatll((char*)(%ld),(const char*)(%ld),(void*)(%ld))"
		    ,(long)fmt,(long)onefmt,pval->obj.i);
	    G__getitem(llbuf);
	    strcat(result,fmt);
	  }
	  else if(strcmp(G__struct.name[pval->tagnum],"G__ulonglong")==0) {
	    sprintf(llbuf
	    ,"G__printformatull((char*)(%ld),(const char*)(%ld),(void*)(%ld))"
		    ,(long)fmt,(long)onefmt,pval->obj.i);
	    G__getitem(llbuf);
	    strcat(result,fmt);
	  }
	  else {
#if !defined(G__OLDIMPLEMENTATION2175)
	    ++usedpara;
	    sprintf(onefmt,fmt ,result,G__int(libp->para[usedpara])); ipara++; 
#elif !defined(G__OLDIMPLEMENTATION2083)
	    sprintf(onefmt,fmt ,result,G__int(libp->para[usedpara])); ipara++;
#else
	    sprintf(onefmt,fmt ,result,G__int(libp->para[ipara++]));
#endif
	    strcpy(result,onefmt);
	  }
	}
	else {
#ifndef G__OLDIMPLEMENTATION2083
	  sprintf(onefmt,fmt ,result,G__int(libp->para[usedpara])); ipara++;
#else
	  sprintf(onefmt,fmt ,result,G__int(libp->para[ipara++]));
#endif
	  strcpy(result,onefmt);
	}
#else
#ifndef G__OLDIMPLEMENTATION2083
	sprintf(onefmt,fmt ,result,G__int(libp->para[usedpara])); ipara++;
#else
	sprintf(onefmt,fmt ,result,G__int(libp->para[ipara++]));
#endif
	strcpy(result,onefmt);
#endif
	ionefmt=0;
	fmtflag=0;
      }
      break;
    case 'e': /* exponential form */
    case 'E': /* Exponential form */
    case 'f': /* floating */
    case 'g': /* floating or exponential */
    case 'G': /* floating or exponential */
      onefmt[ionefmt++]=pformat[ichar];
      if(fmtflag==1) {
	onefmt[ionefmt]='\0';
	sprintf(fmt,"%%s%s",onefmt);
#ifndef G__OLDIMPLEMENTATION2189
	if('q'==libp->para[usedpara].type) {
	  G__value *pval = &libp->para[usedpara]; ipara++;
	  G__sprintformatld(result,fmt,&pval->obj.ld,onefmt);
	}
	else 
#endif
#ifndef G__OLDIMPLEMENTATION1913
	if(
#ifndef G__OLDIMPLEMENTATION2083
           'u'==libp->para[usedpara].type
#else
           'u'==libp->para[ipara].type
#endif
            ) {
	  char llbuf[100];
#ifndef G__OLDIMPLEMENTATION2083
	  G__value *pval = &libp->para[usedpara]; ipara++;
#else
	  G__value *pval = &libp->para[ipara++];
#endif
	  if(strcmp(G__struct.name[pval->tagnum],"G__longdouble")==0) {
	    sprintf(llbuf
	    ,"G__printformatld((char*)(%ld),(const char*)(%ld),(void*)(%ld))"
		    ,(long)fmt,(long)onefmt,pval->obj.i);
	    G__getitem(llbuf);
	    strcat(result,fmt);
	  }
	  else {
#if !defined(G__OLDIMPLEMENTATION2175)
	    ++usedpara;
	    sprintf(onefmt,fmt,result,G__double(libp->para[usedpara]));ipara++;
#elif !defined(G__OLDIMPLEMENTATION2083)
	    sprintf(onefmt,fmt,result,G__double(libp->para[usedpara]));ipara++;
#else
	    sprintf(onefmt,fmt ,result,G__double(libp->para[ipara++]));
#endif
	    strcpy(result,onefmt);
	  }
	}
	else {
#ifndef G__OLDIMPLEMENTATION2083
	  sprintf(onefmt,fmt ,result,G__double(libp->para[usedpara])); ipara++;
#else
	  sprintf(onefmt,fmt ,result,G__double(libp->para[ipara++]));
#endif
	  strcpy(result,onefmt);
	}
#else
#ifndef G__OLDIMPLEMENTATION2083
	sprintf(onefmt,fmt ,result,G__double(libp->para[usedpara])); ipara++;
#else
	sprintf(onefmt,fmt ,result,G__double(libp->para[ipara++]));
#endif
	strcpy(result,onefmt);
#endif
	ionefmt=0;
	fmtflag=0;
      }
      break;
    case 'L': /* long double */
#ifdef G__OLDIMPLEMENTATION2189_YET
      if('q'==libp->para[usedpara].type) {
	G__value *pval = &libp->para[usedpara]; ipara++;
	G__sprintformatld(fmt,onefmt,result,&pval->obj.ld);
      }
      break;
#endif
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
      dig=dig*10+pformat[ichar]-'0';
    case '.':
    case '-':
    case '+':
    case 'l': /* long int */
    case 'h': /* short int unsinged int */
      onefmt[ionefmt++]=pformat[ichar];
      break;
    case '%':
#ifndef G__OLDIMPLEMENTATION2083
      if(fmtflag==0) {
        usedpara = ipara;
        fmtflag=1;
      } else {
        fmtflag=0;
      }
#else
      if(fmtflag==0) fmtflag=1;
      else           fmtflag=0;
#endif
      onefmt[ionefmt++]=pformat[ichar];
      dig=0;
      break;
#ifndef G__OLDIMPLEMENTATION1237
    case '*': /* printf("%*s",4,"*"); */
      if(fmtflag==1) {
#if !defined(G__OLDIMPLEMENTATION2175)
	sprintf(onefmt+ionefmt,"%ld",G__int(libp->para[usedpara])); ipara++; usedpara++;
#elif !defined(G__OLDIMPLEMENTATION2083)
	sprintf(onefmt+ionefmt,"%ld",G__int(libp->para[usedpara])); ipara++;
#else
	sprintf(onefmt+ionefmt,"%ld",G__int(libp->para[ipara++]));
#endif
	ionefmt = strlen(onefmt);
      }
      else {
	onefmt[ionefmt++]=pformat[ichar];
      }
      break;
#endif
#ifndef G__OLDIMPLEMENTATION2083
    case '$':
      if (fmtflag && dig) {
         usedpara = dig + ifmt;
         if (usedpara > libp->paran) {
            /* this is an error! */
            usedpara = ipara;
         }
         /* rewind the digit already printed */
         while( ionefmt>=0 && isdigit(onefmt[ionefmt-1])) --ionefmt; 
         dig=0;
      } 
      else {
         onefmt[ionefmt++]=pformat[ichar];
      }
      break;
#endif
#ifndef G__OLDIMPLEMENTATION1615
    case ' ':
    case '\t' : /* tab */
    case '\n': /* end of line */
    case '\r': /* end of line */
    case '\f': /* end of line */
      if(fmtflag) {
	if('%'!=onefmt[ionefmt-1] && !isspace(onefmt[ionefmt-1])) fmtflag=0;
	onefmt[ionefmt++]=pformat[ichar];
	break;
      }
#endif
    default:
      fmtflag=0;
      onefmt[ionefmt++]=pformat[ichar];
      break;
    }
  }
  
  return(result);
}

#ifndef G__OLDIMPLEMENTATION564
/******************************************************************
* G__tracemode()
******************************************************************/
int G__tracemode(tracemode)
int tracemode;
{
  G__debug = tracemode;
#ifndef G__OLDIMPLEMENTATION1184
  G__istrace = tracemode;
#endif
  G__setdebugcond();
  return(G__debug);
}
/******************************************************************
* G__stepmode()
******************************************************************/
int G__stepmode(stepmode)
int stepmode;
{
  switch(stepmode) {
  case 0:
    G__stepover=0;
    G__step = 0;
    break;
  case 1:
    G__stepover=0;
    G__step = 1;
    break;
  default:
    G__stepover=3;
    G__step = 1;
    break;
  }
  G__setdebugcond();
  return(G__step);
}
/******************************************************************
* G__gettracemode()
******************************************************************/
int G__gettracemode()
{
  return(G__debug);
}
/******************************************************************
* G__getstepmode()
******************************************************************/
int G__getstepmode()
{
  return(G__step);
}
#endif

#ifndef G__OLDIMPLEMENTATION1142
/******************************************************************
* G__optmizemode()
******************************************************************/
int G__optimizemode(optimizemode)
int optimizemode;
{
  G__asm_loopcompile = optimizemode;
#ifndef G__OLDIMPLEMENTATION1155
  G__asm_loopcompile_mode = G__asm_loopcompile; 
#endif
  return(G__asm_loopcompile);
}

/******************************************************************
* G__getoptmizemode()
******************************************************************/
int G__getoptimizemode()
{
  return(G__asm_loopcompile_mode);
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
