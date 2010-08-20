/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file value.c
 ************************************************************************
 * Description:
 *  internal meta-data structure handling
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "value.h"

extern "C" {

int G__Lsizeof(const char *typenamein);

/****************************************************************
* G__letdouble(G__value buf,char type,double value)
*   macro in G__ci.h
****************************************************************/
void G__letdouble(G__value *buf,int type,double value)
{
        buf->type=type;
        buf->obj.d=value;
        /*
        buf->tagnum = -1;
        buf->typenum = -1;
        */
}

/****************************************************************
* G__letbool(G__value buf,char type,int value)
*   macro in G__ci.h
****************************************************************/
void G__letbool(G__value *buf,int type,long value)
{
        buf->type=type;
#ifdef G__BOOL4BYTE
        buf->obj.i=value?1:0;
#else
        buf->obj.uch=value?1:0;
#endif
        /*
        buf->tagnum = -1;
        buf->typenum = -1;
        */
        buf->obj.reftype.reftype = G__PARANORMAL;
}

/****************************************************************
* G__letint(G__value buf,char type,int value)
*   macro in G__ci.h
****************************************************************/
void G__letint(G__value *buf,int type,long value)
{
      buf->type=type;

      switch(buf->type) {
      case 'w': /* logic */
      case 'r': /* unsigned short */
          buf->obj.ush = value; break;	
      case 'h': /* unsigned int */
        buf->obj.uin = value; break;
#ifndef G__BOOL4BYTE
      case 'g':
#endif
      case 'b': /* unsigned char */
          buf->obj.uch = value; break;
      case 'k': /* unsigned long */
         buf->obj.ulo = value; break;
      case 'n':  buf->obj.ll = value; break;
      case 'm':buf->obj.ull = value; break;
      case 'q': buf->obj.ld = value; break;
      case 'i': buf->obj.i = value; break; // should be "in", but there are too many cases where "i" is read out later
      case 'c': buf->obj.ch = value; break;
      case 's':  buf->obj.sh = value; break;
      default: buf->obj.i=value;;
      }

      //        buf->type=type;
      //  
        /*
        buf->tagnum = -1;
        buf->typenum = -1;
        */
        buf->obj.reftype.reftype = G__PARANORMAL;
}

/****************************************************************
* G__letLonglong(G__value buf,char type,int value)
*   macro in G__ci.h
****************************************************************/
void G__letLonglong(G__value *buf,int type,G__int64 value)
{
  buf->type=type;
  buf->obj.ll=value;
  /*
    buf->tagnum = -1;
    buf->typenum = -1;
  */
  /*buf->obj.reftype.reftype = G__PARANORMAL; */
}

/****************************************************************
* G__letULonglong(G__value buf,char type,int value)
*   macro in G__ci.h
****************************************************************/
void G__letULonglong(G__value *buf,int type,G__uint64 value)
{
  buf->type=type;
  buf->obj.ull=value;
  /*
    buf->tagnum = -1;
    buf->typenum = -1;
  */
  /* buf->obj.reftype.reftype = G__PARANORMAL; */
}

/****************************************************************
* G__letLongdouble(G__value buf,char type,int value)
*   macro in G__ci.h
****************************************************************/
void G__letLongdouble(G__value *buf,int type,long double value)
{
  buf->type=type;
  buf->obj.ld=value;
  /*
    buf->tagnum = -1;
    buf->typenum = -1;
  */
  /* buf->obj.reftype.reftype = G__PARANORMAL; */
}

/****************************************************************
* int G__isdouble(G__value buf)
* 
****************************************************************/
int G__isdouble(G__value buf)
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
float G__float(G__value buf)
{
   return G__convertT<float>(&buf);
}
#endif

/****************************************************************
* double G__double(G__value buf)
* 
****************************************************************/
double G__double(G__value buf)
{
   return G__convertT<double>(&buf);
}

/****************************************************************
* long G__bool(G__value buf)
* 
****************************************************************/
long G__bool(G__value buf)
{
   return G__convertT<bool>(&buf);
}

/****************************************************************
* long G__int(G__value buf)
* 
****************************************************************/
long G__int(G__value buf)
{
   return G__convertT<long>(&buf);
}

/****************************************************************
* long G__uint(G__value buf)
* 
****************************************************************/
unsigned long G__uint(G__value buf)
{
   return G__convertT<unsigned long>(&buf);
}

/****************************************************************
* G__int64 G__Longlong(G__value buf)
* 
****************************************************************/
G__int64 G__Longlong(G__value buf)
{
   return G__convertT<G__int64>(&buf);
}

/****************************************************************
* G__uint64 G__Longlong(G__value buf)
* 
****************************************************************/
G__uint64 G__ULonglong(G__value buf)
{
   return G__convertT<G__uint64>(&buf);
}

/****************************************************************
* long double G__Longdouble(G__value buf)
* 
****************************************************************/
long double G__Longdouble(G__value buf)
{
   return G__convertT<long double>(&buf);
}

/******************************************************************
* G__value G__toXvalue(G__value p,int var_type)
*
*
******************************************************************/
G__value G__toXvalue(G__value result,int var_type)
{
  switch(var_type) {
  case 'v':
    return(G__tovalue(result));
    break;
  case 'P':
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: TOPVALUE\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp]=G__TOPVALUE;
      G__inc_cp_asm(1,0);
    }
#endif
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
G__value G__tovalue(G__value p)
{
  G__value result;

  result=p;

  if(-1!=p.typenum && G__newtype.nindex[p.typenum]) {
    result.typenum = -1;
  }

#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: TOVALUE\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__TOVALUE;
    G__inc_cp_asm(2,0);
  }
  if(G__no_exec_compile) {
    if(isupper(p.type)) {
      switch(p.obj.reftype.reftype) {
      case G__PARANORMAL:
        result.type = tolower(p.type);
        result.obj.i = 1;
        result.ref = p.obj.i;
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
        return(result);
      case G__PARAP2P:
        result.obj.i = 1;
        result.ref = p.obj.i;
        result.obj.reftype.reftype=G__PARANORMAL;
        if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_p2p;
        return(result);
      case G__PARAP2P2P:
        result.obj.i = 1;
        result.ref = p.obj.i;
        result.obj.reftype.reftype=G__PARAP2P;
        if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_p2p2p;
        return(result);
      case G__PARAREFERENCE:
        break;
      default:
        result.obj.i = 1;
        result.ref = p.obj.i;
        --result.obj.reftype.reftype;
        if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_p2p2p2;
        return(result);
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
      if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_p2p;
      return(result);
    case G__PARAP2P2P:
      result.obj.i = (long)(*(long *)(p.obj.i));
      result.ref = p.obj.i;
      result.obj.reftype.reftype=G__PARAP2P;
      if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_p2p2p;
      return(result);
    case G__PARANORMAL:
    case G__PARAREFERENCE:
      break;
    default:
      result.obj.i = (long)(*(long *)(p.obj.i));
      result.ref = p.obj.i;
      --result.obj.reftype.reftype;
      if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_p2p2p2;
      return(result);
    }
  }

  switch(p.type) {
  case 'N':
    result.obj.ll = (*(G__int64*)(p.obj.i));
    result.ref = p.obj.i;
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_LL;
    break;
  case 'M':
    result.obj.ull = (*(G__uint64*)(p.obj.i));
    result.ref = p.obj.i;
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_ULL;
    break;
  case 'Q':
    result.obj.ld = (*(long double*)(p.obj.i));
    result.ref = p.obj.i;
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_LD;
    break;
  case 'G':
#ifdef G__BOOL4BYTE
    result.obj.i = (long)(*(int*)(p.obj.i));
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_I;
    break;
#endif
  case 'B':
    result.obj.uch = (*(unsigned char *)(p.obj.i));
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_B;
    break;
  case 'C':
    result.obj.ch = (*(char *)(p.obj.i));
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_C;
    break;
  case 'R':
    result.obj.ush = (*(unsigned short *)(p.obj.i));
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_R;
    break;
  case 'S':
    result.obj.sh = (*(short *)(p.obj.i));
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_S;
    break;
  case 'H':
    result.obj.uin = (*(unsigned int *)(p.obj.i));
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_H;
    break;
  case 'I':
    result.obj.in = (*(int *)(p.obj.i));
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_I;
    break;
  case 'K':
    result.obj.ulo = (*(unsigned long *)(p.obj.i));
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_K;
    break;
  case 'L':
    result.obj.i = (long)(*(long *)(p.obj.i));
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_L;
    break;
  case 'F':
    result.obj.d = (double)(*(float *)(p.obj.i));
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_F;
    break;
  case 'D':
    result.obj.d = (double)(*(double *)(p.obj.i));
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_D;
    break;
  case 'U':
    result.obj.i = p.obj.i;
    if(G__asm_noverflow) G__asm_inst[G__asm_cp-1]=(long)G__asm_tovalue_U;
    break;
  case 'u': 
    { 
      long store_struct_offsetX = G__store_struct_offset;
      int store_tagnumX = G__tagnum;
      int done=0;
      G__store_struct_offset = p.obj.i;
#ifdef G__ASM
      if(G__asm_noverflow) {
        G__inc_cp_asm(-2,0);
        G__asm_inst[G__asm_cp] = G__PUSHSTROS;
        G__asm_inst[G__asm_cp+1] = G__SETSTROS;
        G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
        if(G__asm_dbg) {
          G__fprinterr(G__serr,"TOVALUE cancelled\n");
          G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp-2);
          G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp-1);
        }
#endif
      }
#endif
      G__tagnum = p.tagnum;
      G__FastAllocString refopr("operator*()");
      result=G__getfunction(refopr,&done,G__TRYMEMFUNC);
      G__tagnum = store_tagnumX;
      G__store_struct_offset = store_struct_offsetX; 
#ifdef G__ASM
      if(G__asm_noverflow) {
        G__asm_inst[G__asm_cp] = G__POPSTROS;
        G__inc_cp_asm(1,0);
#ifdef G__ASM_DBG
        if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp-1);
#endif
      }
#endif
      if(done) return(result);
      /* if 0==done, continue to default case for displaying error message */
    }
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
G__value G__letVvalue(G__value *p,G__value result)
{
#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LETVVAL\n",G__asm_cp);
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
    p->obj.reftype.reftype = 0;
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
G__value G__letPvalue(G__value *p,G__value result)
{
#ifdef G__ASM
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg)
      G__fprinterr(G__serr,"%3x: LETPVAL\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__LETPVAL;
    G__inc_cp_asm(1,0);
  }
#endif /* G__ASM */

  return(G__letvalue(p,result));
}

/******************************************************************
* G__value G__letvalue(G__value *p,G__value expression)
******************************************************************/
G__value G__letvalue(G__value *p,G__value result)
{
  if(G__no_exec_compile) {
    if(-1!=p->tagnum && 'e'!=G__struct.type[p->tagnum]) {
      switch(p->type) {
      case 'U':
        result=G__classassign(p->obj.i,p->tagnum, result);
        break;
      case 'u':
        {
          G__value para;
          long store_struct_offsetX = G__store_struct_offset;
          int store_tagnumX = G__tagnum;
          int done=0;
          int store_var_type = G__var_type;
          G__var_type='p';
#ifdef G__ASM
          if(G__asm_noverflow) {
            if(G__LETPVAL==G__asm_inst[G__asm_cp-1]||
               G__LETVVAL==G__asm_inst[G__asm_cp-1]) {
#ifdef G__ASM_DBG
              if(G__asm_dbg) 
                G__fprinterr(G__serr,"LETPVAL,LETVVAL cancelled\n");
#endif
              G__inc_cp_asm(-1,0);
            }
#ifdef G__ASM_DBG
            if(G__asm_dbg) {
              G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp-2);
              G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp-1);
            }
#endif
            G__asm_inst[G__asm_cp] = G__PUSHSTROS;
            G__asm_inst[G__asm_cp+1] = G__SETSTROS;
            G__inc_cp_asm(2,0);
          }
#endif
          G__store_struct_offset = p->obj.i;
          G__tagnum = p->tagnum;
          G__FastAllocString refopr("operator*()");
          para=G__getfunction(refopr,&done,G__TRYMEMFUNC);
          G__tagnum = store_tagnumX;
          G__store_struct_offset = store_struct_offsetX;
          G__var_type=store_var_type;
#ifdef G__ASM
          if(G__asm_noverflow) {
#ifdef G__ASM_DBG
            if(G__asm_dbg) {
              G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp-2);
            }
#endif
            G__asm_inst[G__asm_cp] = G__POPSTROS;
            G__inc_cp_asm(1,0);
          }
          G__letVvalue(&para,result);
#endif
        }
        break;
      }
    }
    return(result);
  }

  if(-1!=p->typenum && G__newtype.nindex[p->typenum]) {
    char store_var_type = G__var_type;
    int size = G__Lsizeof(G__newtype.name[p->typenum]);
    G__var_type = store_var_type;
    if (size > -1) {
       if('C'==result.type && (int)strlen((char*)result.obj.i)<size)
          size = strlen((char*)result.obj.i)+1;
       memcpy((void*)p->obj.i,(void*)result.obj.i,size);
    }
    return(result);
  }
  if(isupper(p->type)) {
     switch(p->obj.reftype.reftype) {
     case G__PARAP2P:
     case G__PARAP2P2P:
         *(long *)(p->obj.i)=(long)G__int(result);
          return(result);
     }
  }
  switch(p->type) {
  case 'G':
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
    *(unsigned long *)(p->obj.i)=(unsigned long)G__uint(result);
    break;
  case 'L':
    *(long *)(p->obj.i)=(long)G__int(result);
    break;
  case 'M':
    *(G__uint64 *)(p->obj.i)=G__ULonglong(result);
    break;
  case 'N':
    *(G__int64 *)(p->obj.i)=G__Longlong(result);
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
  case 'u':
    {
      G__value para;
      long store_struct_offsetX = G__store_struct_offset;
      int store_tagnumX = G__tagnum;
      int done=0;
      int store_var_type = G__var_type;
      G__var_type='p';
#ifdef G__ASM
      if(G__asm_noverflow) {
        if(G__LETPVAL==G__asm_inst[G__asm_cp-1]||
           G__LETVVAL==G__asm_inst[G__asm_cp-1]) {
#ifdef G__ASM_DBG
          if(G__asm_dbg) G__fprinterr(G__serr,"LETPVAL,LETVVAL cancelled\n");
#endif
          G__inc_cp_asm(-1,0);
        }
#ifdef G__ASM_DBG
        if(G__asm_dbg) {
          G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp-2);
          G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp-1);
        }
#endif
        G__asm_inst[G__asm_cp] = G__PUSHSTROS;
        G__asm_inst[G__asm_cp+1] = G__SETSTROS;
        G__inc_cp_asm(2,0);
      }
#endif
      G__store_struct_offset = p->obj.i;
      G__tagnum = p->tagnum;
      G__FastAllocString refopr("operator*()");
      para=G__getfunction(refopr,&done,G__TRYMEMFUNC);
      G__tagnum = store_tagnumX;
      G__store_struct_offset = store_struct_offsetX;
      G__var_type=store_var_type;
#ifdef G__ASM
      if(G__asm_noverflow) {
#ifdef G__ASM_DBG
        if(G__asm_dbg) {
          G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp-2);
        }
#endif
        G__asm_inst[G__asm_cp] = G__POPSTROS;
        G__inc_cp_asm(1,0);
      }
      G__letVvalue(&para,result);
#endif
    }
    break;
  case 'c':
    memcpy((void*)p->ref,(void*)result.obj.i,strlen((char*)result.obj.i)+1);
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

void G__set_tagnum(G__value* val, int tagnum)
{
   val->type = 'u';
   val->tagnum = tagnum;
}

void G__set_typenum(G__value* val, const char* type)
{
   val->typenum = G__defined_typename(type);
}

void G__set_type(G__value* val, char* type)
{
   G__value vtype = G__string2type_body(type, 1);
   val->type    = vtype.type;
   val->tagnum  = vtype.tagnum;
   val->typenum = vtype.typenum;
}

void G__letref_int(G__value* val, long p)
{
   val->obj.i = p;
   val->ref = p;
}

void G__letref_intaddr(G__value* val, long p, long addr)
{
   val->obj.i = p;
   val->ref = addr;
}

void G__letref_doubleaddr(G__value* val, double d, long addr)
{
   val->obj.d = d;
   val->ref = addr;
}

int G__value_get_type(G__value* val) 
{
   return val->type;
}

int G__value_get_tagnum(G__value* val) 
   {
      return val->tagnum;
   }
   
} /* extern "C" */

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
