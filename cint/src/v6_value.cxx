/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file value.c
 ************************************************************************
 * Description:
 *  internal meta-data structure handling
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
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

/****************************************************************
* G__letdouble(G__value buf,char type,double value)
*   macro in G__ci.h
****************************************************************/
void G__letdouble(buf,type,value)
G__value *buf;
int type;
double value;
{
	buf->type=type;
	buf->obj.d=value;
	/*
	buf->tagnum = -1;
	buf->typenum = -1;
	*/
}

/****************************************************************
* G__letint(G__value buf,char type,int value)
*   macro in G__ci.h
****************************************************************/
void G__letint(buf,type,value)
G__value *buf;
int type;
long value; /* used to be int */
{
	buf->type=type;
	buf->obj.i=value;
	/*
	buf->tagnum = -1;
	buf->typenum = -1;
	*/
#ifndef G__OLDIMPLEMENTATION456
	buf->obj.reftype.reftype = G__PARANORMAL;
#endif
}

/****************************************************************
* int G__isdouble(G__value buf)
* 
****************************************************************/
int G__isdouble(buf)
G__value buf;
{
	switch(buf.type) {
	case 'd':
	case 'f':
		return(1);
	default:
		return(0);
	}
}

#ifdef G__NEVER
/****************************************************************
* float G__float(G__value buf)
* 
****************************************************************/
float G__float(buf)
G__value buf;
{
	float result;
	switch(buf.type) {
	case 'd': /* double */
	case 'f': /* float */
	case 'w': /* logic */
		result = (float)buf.obj.d;
		return(result);
	case 'k': /* unsigned long */
	case 'h': /* unsigned int */
	case 'r': /* unsigned short */
	case 'b': /* unsigned char */
		result = (float)((unsigned long)buf.obj.i);
		return(result);
	default:
		result = (float)buf.obj.i;
		return(result);
	}
}
#endif

/****************************************************************
* double G__double(G__value buf)
* 
****************************************************************/
double G__double(buf)
G__value buf;
{
	switch(buf.type) {
	case 'd': /* double */
	case 'f': /* float */
	case 'w': /* logic */
		return(buf.obj.d);
#ifndef G__OLDIMPLEMENTATION655
	case 'k': /* unsigned long */
	case 'h': /* unsigned int */
	case 'r': /* unsigned short */
	case 'b': /* unsigned char */
		return((double)((unsigned long)buf.obj.i));
#endif
	default:
		return((double)buf.obj.i);
	}
}

/****************************************************************
* long G__int(G__value buf)
* 
****************************************************************/
long G__int(buf) /* used to be int */
G__value buf;
{
	switch(buf.type) {
	case 'd':
	case 'f':
		return((long)buf.obj.d);
	default:
		return(buf.obj.i);
	}
}

/******************************************************************
* G__value G__toXvalue(G__value p,int var_type)
*
*
******************************************************************/
G__value G__toXvalue(result,var_type)
G__value result;
int var_type;
{
  switch(var_type) {
  case 'v':
    return(G__tovalue(result));
    break;
  case 'P':
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr("%3x: TOPVALUE\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp]=G__TOPVALUE;
      G__inc_cp_asm(1,0);
    }
#endif
#ifndef G__OLDIMPLEMENTATION879
    if(islower(result.type)) {
      result.type = toupper(result.type);
      result.obj.reftype.reftype=G__PARANORMAL;
    }
    else if(G__PARANORMAL==result.obj.reftype.reftype) {
      result.obj.reftype.reftype=G__PARAP2P;
    }
    else {
      ++result.obj.reftype.reftype;
    }
    if(result.ref) result.obj.i = result.ref;
    else if(G__no_exec_compile) result.obj.i = 1;
    result.ref = 0;
#else
    if(result.ref) {
      result.obj.i = result.ref;
      result.ref = 0;
      if(islower(result.type)) {
	result.type = toupper(result.type);
	result.obj.reftype.reftype=G__PARANORMAL;
      }
      else if(G__PARANORMAL==result.obj.reftype.reftype) {
	result.obj.reftype.reftype=G__PARAP2P;
      }
      else {
	++result.obj.reftype.reftype;
      }
    }
#endif
    return result;
    break;
  default:
    return result;
    break;
  }
}



/******************************************************************
* G__value G__tovalue(G__value p)
*
******************************************************************/
G__value G__tovalue(p)
G__value p;
{
  G__value result;

  result=p;

#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr("%3x: TOVALUE\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__TOVALUE;
#ifndef G__OLDIMPLEMENTATION1401
    G__inc_cp_asm(2,0);
#else
    G__inc_cp_asm(1,0);
#endif
  }
  if(G__no_exec_compile) {
    if(isupper(p.type)) {
      switch(p.obj.reftype.reftype) {
      case G__PARANORMAL:
	result.type = tolower(p.type);
	result.obj.i = 1;
	result.ref = p.obj.i;
#ifndef G__OLDIMPLEMENTATION1401
	if(G__asm_noverflow) {
	  switch(p.type) {
	  case 'B': G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_B; break;
	  case 'C': G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_C; break;
	  case 'R': G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_R; break;
	  case 'S': G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_S; break;
	  case 'H': G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_H; break;
	  case 'I': G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_I; break;
	  case 'K': G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_K; break;
	  case 'L': G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_L; break;
	  case 'F': G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_F; break;
	  case 'D': G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_D; break;
	  case 'U': G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_U; break;
	  default: break;
	  }
	}
#endif
	return(result);
      case G__PARAP2P:
	result.obj.i = 1;
	result.ref = p.obj.i;
	result.obj.reftype.reftype=G__PARANORMAL;
#ifndef G__OLDIMPLEMENTATION1401
	if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_p2p;
#endif
	return(result);
      case G__PARAP2P2P:
	result.obj.i = 1;
	result.ref = p.obj.i;
	result.obj.reftype.reftype=G__PARAP2P;
#ifndef G__OLDIMPLEMENTATION1401
	if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_p2p2p;
#endif
	return(result);
#ifndef G__OLDIMPLEMENTATION707
      case G__PARAREFERENCE:
	break;
      default:
	result.obj.i = 1;
	result.ref = p.obj.i;
	--result.obj.reftype.reftype;
#ifndef G__OLDIMPLEMENTATION1401
	if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_p2p2p2;
#endif
	return(result);
#endif
      }
    }
  }
#endif

  if(isupper(p.type)) {
    switch(p.obj.reftype.reftype) {
    case G__PARAP2P:
      result.obj.i = (long)(*(long *)(p.obj.i));
      result.ref = p.obj.i;
      result.obj.reftype.reftype=G__PARANORMAL;
#ifndef G__OLDIMPLEMENTATION1401
      if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_p2p;
#endif
      return(result);
    case G__PARAP2P2P:
      result.obj.i = (long)(*(long *)(p.obj.i));
      result.ref = p.obj.i;
      result.obj.reftype.reftype=G__PARAP2P;
#ifndef G__OLDIMPLEMENTATION1401
      if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_p2p2p;
#endif
      return(result);
#ifndef G__OLDIMPLEMENTATION707
    case G__PARANORMAL:
    case G__PARAREFERENCE:
      break;
    default:
      result.obj.i = (long)(*(long *)(p.obj.i));
      result.ref = p.obj.i;
      --result.obj.reftype.reftype;
#ifndef G__OLDIMPLEMENTATION1401
      if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_p2p2p2;
#endif
      return(result);
#endif
    }
  }

  switch(p.type) {
  case 'B':
    result.obj.i = (long)(*(unsigned char *)(p.obj.i));
#ifndef G__OLDIMPLEMENTATION1401
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_B;
#endif
    break;
  case 'C':
    result.obj.i = (long)(*(char *)(p.obj.i));
#ifndef G__OLDIMPLEMENTATION1401
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_C;
#endif
    break;
  case 'R':
    result.obj.i = (long)(*(unsigned short *)(p.obj.i));
#ifndef G__OLDIMPLEMENTATION1401
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_R;
#endif
    break;
  case 'S':
    result.obj.i = (long)(*(short *)(p.obj.i));
#ifndef G__OLDIMPLEMENTATION1401
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_S;
#endif
    break;
  case 'H':
    result.obj.i = (long)(*(unsigned int *)(p.obj.i));
#ifndef G__OLDIMPLEMENTATION1401
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_H;
#endif
    break;
  case 'I':
    result.obj.i = (long)(*(int *)(p.obj.i));
#ifndef G__OLDIMPLEMENTATION1401
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_I;
#endif
    break;
  case 'K':
    result.obj.i = (long)(*(unsigned long *)(p.obj.i));
#ifndef G__OLDIMPLEMENTATION1401
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_K;
#endif
    break;
  case 'L':
    result.obj.i = (long)(*(long *)(p.obj.i));
#ifndef G__OLDIMPLEMENTATION1401
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_L;
#endif
    break;
  case 'F':
    result.obj.d = (double)(*(float *)(p.obj.i));
#ifndef G__OLDIMPLEMENTATION1401
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_F;
#endif
    break;
  case 'D':
    result.obj.d = (double)(*(double *)(p.obj.i));
#ifndef G__OLDIMPLEMENTATION1401
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_D;
#endif
    break;
  case 'U':
    result.obj.i = p.obj.i;
#ifndef G__OLDIMPLEMENTATION1401
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_U;
#endif
    break;
#ifndef G__OLDIMPLEMENTATION740
  case 'u': 
    { 
      char refopr[G__MAXNAME];
      long store_struct_offsetX = G__store_struct_offset;
      int store_tagnumX = G__tagnum;
      int done=0;
      G__store_struct_offset = p.obj.i;
#ifndef G__OLDIMPLEMENTATION747
#ifdef G__ASM
      if(G__asm_noverflow) {
#ifndef G__OLDIMPLEMENTATION1401
	G__inc_cp_asm(-2,0);
#else
	G__inc_cp_asm(-1,0);
#endif
	G__asm_inst[G__asm_cp] = G__PUSHSTROS;
	G__asm_inst[G__asm_cp+1] = G__SETSTROS;
	G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
	if(G__asm_dbg) {
	  G__fprinterr("TOVALUE cancelled\n");
	  G__fprinterr("%3x: PUSHSTROS\n",G__asm_cp-2);
	  G__fprinterr("%3x: SETSTROS\n",G__asm_cp-1);
	}
#endif
      }
#endif
#endif
      G__tagnum = p.tagnum;
      strcpy(refopr,"operator*()");
      result=G__getfunction(refopr,&done,G__TRYMEMFUNC);
      G__tagnum = store_tagnumX;
      G__store_struct_offset = store_struct_offsetX; 
#ifndef G__OLDIMPLEMENTATION747
#ifdef G__ASM
      if(G__asm_noverflow) {
	G__asm_inst[G__asm_cp] = G__POPSTROS;
	G__inc_cp_asm(1,0);
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr("%3x: POPSTROS\n",G__asm_cp-1);
#endif
      }
#endif
#endif
      if(done) return(result);
      /* if 0==done, continue to default case for displaying error message */
    }
#endif
  default:
    /* if(0==G__no_exec_compile) */
    G__genericerror("Error: Illegal pointer operation (tovalue)");
    break;
  }
  result.type = tolower(p.type);
  result.ref = p.obj.i;
  
  return(result);
}

/******************************************************************
* G__value G__letVvalue(G__value *p,G__value expression)
*
******************************************************************/
G__value G__letVvalue(p,result)
G__value *p,result;
{
#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr("%3x: LETVVAL\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__LETVVAL;
    G__inc_cp_asm(1,0);
  }
#endif /* G__ASM */

  if(p->ref) {
    p->obj.i = p->ref;
    p->ref=0;
    /* if xxx *p;   (p)=xxx;  lvalue is pointer type then assign as long
     * else convert p to its' pointer type
     */
    if(isupper(p->type)) p->type='L';
    else                 p->type=toupper(p->type);
    return(G__letvalue(p,result));
  }

  G__genericerror("Error: improper lvalue");
#ifdef G__ASM
#ifdef G__ASM_DBG
  if(G__asm_dbg&&G__asm_noverflow)
    G__genericerror(G__LOOPCOMPILEABORT);
#endif
  G__abortbytecode();
#endif /* G__ASM */

  return(result);

}

/******************************************************************
* G__value G__letPvalue(G__value *p,G__value expression)
*
******************************************************************/
G__value G__letPvalue(p,result)
G__value *p,result;
{
#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg)
      G__fprinterr("%3x: LETPVAL\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__LETPVAL;
    G__inc_cp_asm(1,0);
  }
#endif /* G__ASM */

  return(G__letvalue(p,result));
}

/******************************************************************
* G__value G__letvalue(G__value *p,G__value expression)
*
* Called by
*   G__letvariable
*   G__initary
*
******************************************************************/
G__value G__letvalue(p,result)
G__value *p,result;
{
  if(G__no_exec_compile) {
    return(result);
  }

#ifndef G__OLDIMPLEMENTATION1329
  if(-1!=p->typenum && G__newtype.nindex[p->typenum]) {
    char store_var_type = G__var_type;
    int size = G__Lsizeof(G__newtype.name[p->typenum]);
    G__var_type = store_var_type;
    if('C'==result.type && strlen((char*)result.obj.i)<size)
      size = strlen((char*)result.obj.i)+1;
    memcpy((void*)p->obj.i,(void*)result.obj.i,size);
    return(result);
  }
#endif
#ifndef G__OLDIMPLEMENTATION1384
  switch(p->obj.reftype.reftype) {
  case G__PARAP2P:
  case G__PARAP2P2P:
    if(isupper(p->type)) {
      *(long *)(p->obj.i)=(long)G__int(result);
      return(result);
    }
  }
#endif
  switch(p->type) {
  case 'B':
    *(unsigned char *)(p->obj.i)=(unsigned char)G__int(result);
    break;
  case 'C':
    *(char *)(p->obj.i)=(char)G__int(result);
    break;
  case 'R':
    *(unsigned short *)(p->obj.i)=(unsigned short)G__int(result);
    break;
  case 'S':
    *(short *)(p->obj.i)=(short)G__int(result);
    break;
  case 'H':
    *(unsigned int *)(p->obj.i)=(unsigned int)G__int(result);
    break;
  case 'I':
    *(int *)(p->obj.i)=(int)G__int(result);
    break;
  case 'K':
    *(unsigned long *)(p->obj.i)=(unsigned long)G__int(result);
    break;
  case 'L':
    *(long *)(p->obj.i)=(long)G__int(result);
    break;
  case 'F':
    *(float *)(p->obj.i)=(float)G__double(result);
    break;
  case 'D':
    *(double *)(p->obj.i)=(double)G__double(result);
    break;
  case 'U':
    result=G__classassign(p->obj.i,p->tagnum, result);
    break;
  default:
#ifdef G__ASM
#ifdef G__ASM_DBG
    if(G__asm_dbg&&G__asm_noverflow)
      G__genericerror(G__LOOPCOMPILEABORT);
#endif
    G__abortbytecode();
#endif /* G__ASM */
    G__genericerror("Error: Illegal pointer operation (letvalue)");
    break;
  }
  
  return(result);
}



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
