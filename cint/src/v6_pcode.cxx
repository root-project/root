/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file pcode.c
 ************************************************************************
 * Description:
 *  Loop compilation related source code
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

#ifndef G__OLDIMPLEMENTATION1229
#ifdef G__ROOT
extern void* G__new_interpreted_object G__P((int size));
extern void G__delete_interpreted_object G__P((void* p));
#endif
#endif

#ifdef G__BORLANDCC5
double G__doubleM(G__value *buf);
static void G__asm_toXvalue(G__value* result);
void G__get__tm__(char *buf);
char* G__get__date__(void);
char* G__get__time__(void);
int G__isInt(int type);
int G__get_LD_Rp0_p2f(int type,long *pinst);
int G__get_ST_Rp0_p2f(int type,long *pinst);
#endif

#ifndef __CINT__
int G__asm_optimize3 G__P((int *start));
#endif

#ifdef G__ASM

#ifdef G__ASM_DBG
int G__asm_step=0;
#endif


#define G__TUNEUP_W_SECURITY

/*************************************************************************
**************************************************************************
* macros to access G__value
**************************************************************************
*************************************************************************/

#ifndef G__OLDIMPLEMENTATION2189
#define G__longlongM(buf)                                               \
  (('f'==buf->type||'d'==buf->type) ? (long)buf->obj.d : buf->obj.i )

#endif 

/****************************************************************
* G__intM()
****************************************************************/
#define G__intM(buf)                                                   \
  (('f'==buf->type||'d'==buf->type) ? (long)buf->obj.d : buf->obj.i )

/****************************************************************
* G__doubleM()
****************************************************************/
#ifdef G__ALWAYS
double G__doubleM(buf)
     G__value *buf;
{
  return (('f'==buf->type||'d'==buf->type) ? buf->obj.d :
	  ('k'==buf->type||'h'==buf->type) ? (double)(buf->obj.ulo) :
	  (double)(buf->obj.i) );
}
#else
#ifndef G__OLDIMPLEMENTATION1494
#define G__doubleM(buf)                                                \
  (('f'==buf->type||'d'==buf->type) ? buf->obj.d :                     \
   ('k'==buf->type||'h'==buf->type) ? (double)(buf->obj.ulo) :         \
                                      (double)(buf->obj.i) )
#else
#define G__doubleM(buf)                                                \
  (('f'==buf->type||'d'==buf->type) ? buf->obj.d : (double)(buf->obj.i) )
#endif
#endif

/****************************************************************
* G__isdoubleM()
****************************************************************/
#define G__isdoubleM(buf) ('f'==buf->type||'d'==buf->type)

#ifndef G__OLDIMPLEMENTATION1494
/****************************************************************
* G__isunsignedM()
****************************************************************/
#define G__isunsignedM(buf) ('h'==buf->type||'k'==buf->type)
#endif


/*************************************************************************
**************************************************************************
* Optimization level 2 runtime function
**************************************************************************
*************************************************************************/

/****************************************************************
* G__asm_test_X()
*
*  Optimized comparator
*
****************************************************************/
int G__asm_test_E(a,b)
int *a,*b;
{
  return(*a==*b);
}
int G__asm_test_N(a,b)
int *a,*b;
{
  return(*a!=*b);
}
int G__asm_test_GE(a,b)
int *a,*b;
{
  return(*a>=*b);
}
int G__asm_test_LE(a,b)
int *a,*b;
{
  return(*a<=*b);
}
int G__asm_test_g(a,b)
int *a,*b;
{
  return(*a>*b);
}
int G__asm_test_l(a,b)
int *a,*b;
{
  return(*a<*b);
}

/*************************************************************************
**************************************************************************
* TOPVALUE and TOVALUE optimization
**************************************************************************
*************************************************************************/
#ifndef G__OLDIMPLEMENTATION1400
/******************************************************************
* G__value G__asm_toXvalue(G__value* p)
*
******************************************************************/
void G__asm_toXvalue(result)
G__value* result;
{
  if(islower(result->type)) {
    result->type = toupper(result->type);
    result->obj.reftype.reftype=G__PARANORMAL;
  }
  else if(G__PARANORMAL==result->obj.reftype.reftype) {
    result->obj.reftype.reftype=G__PARAP2P;
  }
  else {
    ++result->obj.reftype.reftype;
  }
  if(result->ref) result->obj.i = result->ref;
  result->ref = 0;
}
#endif

#ifndef G__OLDIMPLEMENTATION1401
typedef void (*G__p2f_tovalue) G__P((G__value*));
/******************************************************************
* void G__asm_tovalue_p2p(G__value* p)
******************************************************************/
void G__asm_tovalue_p2p(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.i = (long)(*(long *)(result->obj.i));
  result->obj.reftype.reftype=G__PARANORMAL;
}

/******************************************************************
* void G__asm_tovalue_p2p2p(G__value* p)
******************************************************************/
void G__asm_tovalue_p2p2p(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.i = (long)(*(long *)(result->obj.i));
  result->obj.reftype.reftype=G__PARAP2P;
}

/******************************************************************
* void G__asm_tovalue_p2p2p2(G__value* p)
******************************************************************/
void G__asm_tovalue_p2p2p2(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.i = (long)(*(long *)(result->obj.i));
  --result->obj.reftype.reftype;
}

#ifndef G__OLDIMPLEMENTATION2189
/******************************************************************
* void G__asm_tovalue_LL(G__value* p)
******************************************************************/
void G__asm_tovalue_LL(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.ll = (*(G__int64*)(result->obj.i));
  result->type = tolower(result->type);
}
/******************************************************************
* void G__asm_tovalue_ULL(G__value* p)
******************************************************************/
void G__asm_tovalue_ULL(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.ull = (*(G__uint64*)(result->obj.i));
  result->type = tolower(result->type);
}
/******************************************************************
* void G__asm_tovalue_LD(G__value* p)
******************************************************************/
void G__asm_tovalue_LD(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.ld = (*(long double*)(result->obj.i));
  result->type = tolower(result->type);
}
#endif

/******************************************************************
* void G__asm_tovalue_B(G__value* p)
******************************************************************/
void G__asm_tovalue_B(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.i = (long)(*(unsigned char *)(result->obj.i));
  result->type = tolower(result->type);
}
/******************************************************************
* void G__asm_tovalue_C(G__value* p)
******************************************************************/
void G__asm_tovalue_C(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.i = (long)(*(char *)(result->obj.i));
  result->type = tolower(result->type);
}
/******************************************************************
* void G__asm_tovalue_R(G__value* p)
******************************************************************/
void G__asm_tovalue_R(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.i = (long)(*(unsigned short *)(result->obj.i));
  result->type = tolower(result->type);
}
/******************************************************************
* void G__asm_tovalue_S(G__value* p)
******************************************************************/
void G__asm_tovalue_S(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.i = (long)(*(short *)(result->obj.i));
  result->type = tolower(result->type);
}
/******************************************************************
* void G__asm_tovalue_H(G__value* p)
******************************************************************/
void G__asm_tovalue_H(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.i = (long)(*(unsigned int *)(result->obj.i));
  result->type = tolower(result->type);
}
/******************************************************************
* void G__asm_tovalue_I(G__value* p)
******************************************************************/
void G__asm_tovalue_I(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.i = (long)(*(int *)(result->obj.i));
  result->type = tolower(result->type);
}
/******************************************************************
* void G__asm_tovalue_K(G__value* p)
******************************************************************/
void G__asm_tovalue_K(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.i = (long)(*(unsigned long *)(result->obj.i));
  result->type = tolower(result->type);
}
/******************************************************************
* void G__asm_tovalue_L(G__value* p)
******************************************************************/
void G__asm_tovalue_L(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.i = (long)(*(long *)(result->obj.i));
  result->type = tolower(result->type);
}
/******************************************************************
* void G__asm_tovalue_F(G__value* p)
******************************************************************/
void G__asm_tovalue_F(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.d = (double)(*(float *)(result->obj.i));
  result->type = tolower(result->type);
}
/******************************************************************
* void G__asm_tovalue_D(G__value* p)
******************************************************************/
void G__asm_tovalue_D(result)
G__value *result;
{
  result->ref = result->obj.i;
  result->obj.d = (double)(*(double *)(result->obj.i));
  result->type = tolower(result->type);
}
/******************************************************************
* void G__asm_tovalue_U(G__value* p)
******************************************************************/
void G__asm_tovalue_U(result)
G__value *result;
{
  result->ref = result->obj.i;
  /* result->obj.i = result->obj.i; */
  result->type = tolower(result->type);
}

#endif


/*************************************************************************
**************************************************************************
* Optimization level 1 runtime function
**************************************************************************
*************************************************************************/

#ifdef G__OLDIMPLEMENTATION2163

#include "bc_exec_asm.h"

#endif /* 2163 */


/*************************************************************************
**************************************************************************
* Optimization level 3 runtime function
**************************************************************************
*************************************************************************/

/*************************************************************************
* G__LD_p0_xxx
*************************************************************************/
/****************************************************************
* G__ASM_GET_INT
****************************************************************/
#define G__ASM_GET_INT(casttype,ctype)    \
  G__value *buf= &pbuf[(*psp)++];         \
  buf->tagnum = -1;                       \
  buf->type = ctype;                      \
  buf->typenum = var->p_typetable[ig15];  \
  buf->ref = var->p[ig15]+offset;         \
  buf->obj.i = *(casttype*)buf->ref

#ifndef G__OLDIMPLEMENTATION2189
/****************************************************************
* G__LD_p0_longlong()
****************************************************************/
void G__LD_p0_longlong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = -1;              
  buf->type = 'n';             
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset;       
  buf->obj.ll = *(G__int64*)buf->ref;
}
/****************************************************************
* G__LD_p0_ulonglong()
****************************************************************/
void G__LD_p0_ulonglong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = -1;              
  buf->type = 'm';             
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset;       
  buf->obj.ull = *(G__uint64*)buf->ref;
}
/****************************************************************
* G__LD_p0_longdouble()
****************************************************************/
void G__LD_p0_longdouble(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = -1;              
  buf->type = 'q';             
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset;       
  buf->obj.ld = *(long double*)buf->ref;
}
#endif /* 2189 */

/****************************************************************
* G__LD_p0_bool()
****************************************************************/
void G__LD_p0_bool(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
#ifdef G__BOOL4BYTE
  G__ASM_GET_INT(int,'g');
#else
  G__ASM_GET_INT(unsigned char,'g');
#endif
  buf->obj.i = buf->obj.i?1:0;
}
/****************************************************************
* G__LD_p0_char()
****************************************************************/
void G__LD_p0_char(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT(char,'c');
}
/****************************************************************
* G__LD_p0_uchar()
****************************************************************/
void G__LD_p0_uchar(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT(unsigned char,'b');
}
/****************************************************************
* G__LD_p0_short()
****************************************************************/
void G__LD_p0_short(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT(short,'s');
}
/****************************************************************
* G__LD_p0_ushort()
****************************************************************/
void G__LD_p0_ushort(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT(unsigned short,'r');
}
/****************************************************************
* G__LD_p0_int()
****************************************************************/
void G__LD_p0_int(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT(int,'i');
}
/****************************************************************
* G__LD_p0_uint()
****************************************************************/
void G__LD_p0_uint(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT(unsigned int,'h');
}
/****************************************************************
* G__LD_p0_long()
****************************************************************/
void G__LD_p0_long(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT(long,'l');
}

/****************************************************************
* G__LD_p0_ulong()
****************************************************************/
void G__LD_p0_ulong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT(unsigned long,'k');
}
/****************************************************************
* G__LD_p0_pointer()
****************************************************************/
void G__LD_p0_pointer(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = var->type[ig15];
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset;
  buf->obj.i = *(long*)buf->ref;
  buf->obj.reftype.reftype=G__PARANORMAL;
}
/****************************************************************
* G__LD_p0_struct()
****************************************************************/
void G__LD_p0_struct(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = 'u';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset;
  buf->obj.i = buf->ref;
}
/****************************************************************
* G__LD_p0_float()
****************************************************************/
void G__LD_p0_float(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = -1;
  buf->type = 'f';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset;
  buf->obj.d = *(float*)buf->ref;
}
/****************************************************************
* G__LD_p0_double()
****************************************************************/
void G__LD_p0_double(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = -1;
  buf->type = 'd';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset;
  buf->obj.d = *(double*)buf->ref;
}

#ifndef G__OLDIMPLEMENTATION1969
/*************************************************************************
* G__nonintarrayindex
*************************************************************************/
#ifndef G__OLDIMPLEMENTATION1974
void G__nonintarrayindex G__P((struct G__var_array*,int));
#endif
void G__nonintarrayindex(var,ig15)
struct G__var_array *var; 
int ig15;
{
  G__fprinterr(G__serr,"Error: %s[] invalud type for array index"
	       ,var->varnamebuf[ig15]);
  G__genericerror((char*)NULL);
}
#endif

/*************************************************************************
* G__LD_p1_xxx
*************************************************************************/

/****************************************************************
* G__ASM_GET_INT_P1
****************************************************************/
#ifdef G__TUNEUP_W_SECURITY
#define G__ASM_GET_INT_P1(casttype,ctype)              \
  G__value *buf= &pbuf[*psp-1];                        \
  if('d'==buf->type||'f'==buf->type) G__nonintarrayindex(var,ig15); /*1969*/ \
  buf->tagnum = -1;                                    \
  buf->type = ctype;                                   \
  buf->typenum = var->p_typetable[ig15];               \
  buf->ref = var->p[ig15]+offset+buf->obj.i*sizeof(casttype); \
  if(buf->obj.i-1>var->varlabel[ig15][1])              \
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],buf->obj.i);  \
  else                                                 \
    buf->obj.i = *(casttype*)buf->ref
#else
#define G__ASM_GET_INT_P1(casttype,ctype)              \
  G__value *buf= &pbuf[*psp-1];                        \
  buf->tagnum = -1;                                    \
  buf->type = ctype;                                   \
  buf->typenum = var->p_typetable[ig15];               \
  buf->ref = var->p[ig15]+offset+buf->obj.i*sizeof(casttype); \
  buf->obj.i = *(casttype*)buf->ref
#endif

#ifndef G__OLDIMPLEMENTATION2189
/****************************************************************
* G__LD_p1_longlong()
****************************************************************/
void G__LD_p1_longlong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[*psp-1];
  if('d'==buf->type||'f'==buf->type) G__nonintarrayindex(var,ig15); /*1969*/ 
  buf->tagnum = -1;                                    
  buf->type = 'n';                                   
  buf->typenum = var->p_typetable[ig15];               
  buf->ref = var->p[ig15]+offset+buf->obj.i*sizeof(G__int64); 
  if(buf->obj.i-1>var->varlabel[ig15][1])              
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],buf->obj.i);  
  else                                                 
    buf->obj.ll = *(G__int64*)buf->ref;
}
/****************************************************************
* G__LD_p1_ulonglong()
****************************************************************/
void G__LD_p1_ulonglong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[*psp-1];
  if('d'==buf->type||'f'==buf->type) G__nonintarrayindex(var,ig15); /*1969*/ 
  buf->tagnum = -1;                                    
  buf->type = 'm';                                   
  buf->typenum = var->p_typetable[ig15];               
  buf->ref = var->p[ig15]+offset+buf->obj.i*sizeof(G__uint64); 
  if(buf->obj.i-1>var->varlabel[ig15][1])              
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],buf->obj.i);  
  else                                                 
    buf->obj.ull = *(G__uint64*)buf->ref;
}
/****************************************************************
* G__LD_p1_longdouble()
****************************************************************/
void G__LD_p1_longdouble(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[*psp-1];
  if('d'==buf->type||'f'==buf->type) G__nonintarrayindex(var,ig15); /*1969*/ 
  buf->tagnum = -1;                                    
  buf->type = 'q';                                   
  buf->typenum = var->p_typetable[ig15];               
  buf->ref = var->p[ig15]+offset+buf->obj.i*sizeof(long double); 
  if(buf->obj.i-1>var->varlabel[ig15][1])              
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],buf->obj.i);  
  else                                                 
    buf->obj.ld = *(long double*)buf->ref;
}
#endif /* 2189 */

/****************************************************************
* G__LD_p1_bool()
****************************************************************/
void G__LD_p1_bool(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
#ifdef G__BOOL4BYTE
  G__ASM_GET_INT_P1(int,'g');
#else
  G__ASM_GET_INT_P1(unsigned char,'g');
#endif
  buf->obj.i = buf->obj.i?1:0;
}
/****************************************************************
* G__LD_p1_char()
****************************************************************/
void G__LD_p1_char(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_P1(char,'c');
}
/****************************************************************
* G__LD_p1_uchar()
****************************************************************/
void G__LD_p1_uchar(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_P1(unsigned char,'b');
}
/****************************************************************
* G__LD_p1_short()
****************************************************************/
void G__LD_p1_short(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_P1(short,'s');
}
/****************************************************************
* G__LD_p1_ushort()
****************************************************************/
void G__LD_p1_ushort(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_P1(unsigned short,'r');
}
/****************************************************************
* G__LD_p1_int()
****************************************************************/
void G__LD_p1_int(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_P1(int,'i');
}

/****************************************************************
* G__LD_p1_uint()
****************************************************************/
void G__LD_p1_uint(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_P1(unsigned int,'h');
}
/****************************************************************
* G__LD_p1_long()
****************************************************************/
void G__LD_p1_long(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_P1(long,'l');
}

/****************************************************************
* G__LD_p1_ulong()
****************************************************************/
void G__LD_p1_ulong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_P1(unsigned long,'k');
}
/****************************************************************
* G__LD_p1_pointer()
****************************************************************/
void G__LD_p1_pointer(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[*psp-1];
#ifndef G__OLDIMPLEMENTATION1969
  if('d'==buf->type||'f'==buf->type) G__nonintarrayindex(var,ig15); 
#endif
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = var->type[ig15];
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset+buf->obj.i*sizeof(long);
#ifdef G__TUNEUP_W_SECURITY
  if(buf->obj.i-1>var->varlabel[ig15][1])
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],buf->obj.i);
  else
#endif
    buf->obj.i = *(long*)buf->ref;
  buf->obj.reftype.reftype=G__PARANORMAL;
}
/****************************************************************
* G__LD_p1_struct()
****************************************************************/
void G__LD_p1_struct(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[*psp-1];
#ifndef G__OLDIMPLEMENTATION1969
  if('d'==buf->type||'f'==buf->type) G__nonintarrayindex(var,ig15); 
#endif
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = 'u';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset+buf->obj.i*G__struct.size[buf->tagnum];
#ifdef G__TUNEUP_W_SECURITY
  if(buf->obj.i-1>var->varlabel[ig15][1])
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],buf->obj.i);
  else
#endif
    buf->obj.i = buf->ref;
}
/****************************************************************
* G__LD_p1_float()
****************************************************************/
void G__LD_p1_float(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[*psp-1];
#ifndef G__OLDIMPLEMENTATION1969
  if('d'==buf->type||'f'==buf->type) G__nonintarrayindex(var,ig15); 
#endif
  buf->tagnum = -1;
  buf->type = 'f';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset+buf->obj.i*sizeof(float);
#ifdef G__TUNEUP_W_SECURITY
  if(buf->obj.i-1>var->varlabel[ig15][1])
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],buf->obj.i);
  else
#endif
    buf->obj.d = *(float*)buf->ref;
}
/****************************************************************
* G__LD_p1_double()
****************************************************************/
void G__LD_p1_double(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[*psp-1];
#ifndef G__OLDIMPLEMENTATION1969
  if('d'==buf->type||'f'==buf->type) G__nonintarrayindex(var,ig15); 
#endif
  buf->tagnum = -1;
  buf->type = 'd';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset+buf->obj.i*sizeof(double);
#ifdef G__TUNEUP_W_SECURITY
  if(buf->obj.i-1>var->varlabel[ig15][1])
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],buf->obj.i);
  else
#endif
    buf->obj.d = *(double*)buf->ref;
}

#ifndef G__OLDIMPLEMENTATION1405
/*************************************************************************
* G__LD_pn_xxx
*************************************************************************/

/****************************************************************
* G__ASM_GET_INT_PN
****************************************************************/
#ifdef G__TUNEUP_W_SECURITY
#define G__ASM_GET_INT_PN(casttype,ctype)              \
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];\
  int ary = var->varlabel[ig15][0];                    \
  int paran = var->paran[ig15];                        \
  int p_inc=0;                                         \
  int ig25;                                            \
  ++(*psp);                                            \
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {\
    p_inc += ary*G__int(buf[ig25]);                    \
    ary /= var->varlabel[ig15][ig25+2];                \
  }                                                    \
  buf->tagnum = -1;                                    \
  buf->type = ctype;                                   \
  buf->typenum = var->p_typetable[ig15];               \
  buf->ref = var->p[ig15]+offset+p_inc*sizeof(casttype); \
  if(p_inc-1>var->varlabel[ig15][1])                   \
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);  \
  else                                                 \
    buf->obj.i = *(casttype*)buf->ref
#else
#define G__ASM_GET_INT_PN(casttype,ctype)              \
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];\
  int ary = var->varlabel[ig15][0];                    \
  int paran = var->paran[ig15];                        \
  int p_inc=0;                                         \
  int ig25;                                            \
  ++(*psp);                                            \
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {\
    p_inc += ary*G__int(buf[ig25]);                    \
    ary /= var->varlabel[ig15][ig25+2];                \
  }                                                    \
  buf->tagnum = -1;                                    \
  buf->type = ctype;                                   \
  buf->typenum = var->p_typetable[ig15];               \
  buf->ref = var->p[ig15]+offset+p_inc*sizeof(casttype); \
  buf->obj.i = *(casttype*)buf->ref
#endif

#ifndef G__OLDIMPLEMENTATION2189
/****************************************************************
* G__LD_pn_longlong()
****************************************************************/
void G__LD_pn_longlong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];
  int ary = var->varlabel[ig15][0];                    
  int paran = var->paran[ig15];                        
  int p_inc=0;                                         
  int ig25;                                            
  ++(*psp);                                            
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
    p_inc += ary*G__int(buf[ig25]);                    
    ary /= var->varlabel[ig15][ig25+2];                
  }                                                    
  buf->tagnum = -1;                                    
  buf->type = 'n';                                   
  buf->typenum = var->p_typetable[ig15];               
  buf->ref = var->p[ig15]+offset+p_inc*sizeof(G__int64); 
  if(p_inc-1>var->varlabel[ig15][1])                   
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);  
  else                                                 
    buf->obj.ll = *(G__int64*)buf->ref;
}

/****************************************************************
* G__LD_pn_ulonglong()
****************************************************************/
void G__LD_pn_ulonglong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];
  int ary = var->varlabel[ig15][0];                    
  int paran = var->paran[ig15];                        
  int p_inc=0;                                         
  int ig25;                                            
  ++(*psp);                                            
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
    p_inc += ary*G__int(buf[ig25]);                    
    ary /= var->varlabel[ig15][ig25+2];                
  }                                                    
  buf->tagnum = -1;                                    
  buf->type = 'm';                                   
  buf->typenum = var->p_typetable[ig15];               
  buf->ref = var->p[ig15]+offset+p_inc*sizeof(G__uint64); 
  if(p_inc-1>var->varlabel[ig15][1])                   
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);  
  else                                                 
    buf->obj.ull = *(G__uint64*)buf->ref;
}
/****************************************************************
* G__LD_pn_longdouble()
****************************************************************/
void G__LD_pn_longdouble(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];
  int ary = var->varlabel[ig15][0];                    
  int paran = var->paran[ig15];                        
  int p_inc=0;                                         
  int ig25;                                            
  ++(*psp);                                            
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
    p_inc += ary*G__int(buf[ig25]);                    
    ary /= var->varlabel[ig15][ig25+2];                
  }                                                    
  buf->tagnum = -1;                                    
  buf->type = 'q';                                   
  buf->typenum = var->p_typetable[ig15];               
  buf->ref = var->p[ig15]+offset+p_inc*sizeof(long double); 
  if(p_inc-1>var->varlabel[ig15][1])                   
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);  
  else                                                 
    buf->obj.ld = *(long double*)buf->ref;
}
#endif /* 2189 */

/****************************************************************
* G__LD_pn_bool()
****************************************************************/
void G__LD_pn_bool(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
#ifdef G__BOOL4BYTE
  G__ASM_GET_INT_PN(int,'g');
#else
  G__ASM_GET_INT_PN(unsigned char,'g');
#endif
  buf->obj.i = buf->obj.i?1:0;
}
/****************************************************************
* G__LD_pn_char()
****************************************************************/
void G__LD_pn_char(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_PN(char,'c');
}
/****************************************************************
* G__LD_pn_uchar()
****************************************************************/
void G__LD_pn_uchar(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_PN(unsigned char,'b');
}
/****************************************************************
* G__LD_pn_short()
****************************************************************/
void G__LD_pn_short(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_PN(short,'s');
}
/****************************************************************
* G__LD_pn_ushort()
****************************************************************/
void G__LD_pn_ushort(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_PN(unsigned short,'r');
}
/****************************************************************
* G__LD_pn_int()
****************************************************************/
void G__LD_pn_int(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_PN(int,'i');
}

/****************************************************************
* G__LD_pn_uint()
****************************************************************/
void G__LD_pn_uint(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_PN(unsigned int,'h');
}
/****************************************************************
* G__LD_pn_long()
****************************************************************/
void G__LD_pn_long(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_PN(long,'l');
}

/****************************************************************
* G__LD_pn_ulong()
****************************************************************/
void G__LD_pn_ulong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_PN(unsigned long,'k');
}
/****************************************************************
* G__LD_pn_pointer()
****************************************************************/
void G__LD_pn_pointer(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];
  int ary = var->varlabel[ig15][0];
  int paran = var->paran[ig15];
  int p_inc=0;
  int ig25;
  ++(*psp);
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
    p_inc += ary*G__int(buf[ig25]);
    ary /= var->varlabel[ig15][ig25+2];
  }
#ifndef G__OLDIMPLEMENTATION1847
  buf->tagnum = var->p_tagtable[ig15];
#else
  buf->tagnum = -1;
#endif
  buf->type = var->type[ig15];
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset+p_inc*sizeof(long);
#ifdef G__TUNEUP_W_SECURITY
  if(p_inc-1>var->varlabel[ig15][1])
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);
  else
#endif
    buf->obj.i = *(long*)buf->ref;
  buf->obj.reftype.reftype=var->reftype[ig15]; /* ?? for G__LD_p1_pointer */
}
/****************************************************************
* G__LD_pn_struct()
****************************************************************/
void G__LD_pn_struct(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];
  int ary = var->varlabel[ig15][0];
  int paran = var->paran[ig15];
  int p_inc=0;
  int ig25;
  ++(*psp);
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
    p_inc += ary*G__int(buf[ig25]);
    ary /= var->varlabel[ig15][ig25+2];
  }
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = 'u';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset+p_inc*G__struct.size[buf->tagnum];
#ifdef G__TUNEUP_W_SECURITY
  if(p_inc-1>var->varlabel[ig15][1])
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);
  else
#endif
    buf->obj.i = buf->ref;
}
/****************************************************************
* G__LD_pn_float()
****************************************************************/
void G__LD_pn_float(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];
  int ary = var->varlabel[ig15][0];
  int paran = var->paran[ig15];
  int p_inc=0;
  int ig25;
  ++(*psp);
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
    p_inc += ary*G__int(buf[ig25]);
    ary /= var->varlabel[ig15][ig25+2];
  }
  buf->tagnum = -1;
  buf->type = 'f';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset+p_inc*sizeof(float);
#ifdef G__TUNEUP_W_SECURITY
  if(p_inc-1>var->varlabel[ig15][1])
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);
  else
#endif
    buf->obj.d = *(float*)buf->ref;
}
/****************************************************************
* G__LD_pn_double()
****************************************************************/
void G__LD_pn_double(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];
  int ary = var->varlabel[ig15][0];
  int paran = var->paran[ig15];
  int p_inc=0;
  int ig25;
  ++(*psp);
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
    p_inc += ary*G__int(buf[ig25]);
    ary /= var->varlabel[ig15][ig25+2];
  }
  buf->tagnum = -1;
  buf->type = 'd';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset+p_inc*sizeof(double);
#ifdef G__TUNEUP_W_SECURITY
  if(p_inc-1>var->varlabel[ig15][1])
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);
  else
#endif
    buf->obj.d = *(double*)buf->ref;
}
#endif /* 1405 */


/*************************************************************************
* G__LD_P10_xxx
*************************************************************************/

/****************************************************************
* G__ASM_GET_INT_P10
****************************************************************/
#define G__ASM_GET_INT_P10(casttype,ctype)                              \
  G__value *buf= &pbuf[*psp-1];                                         \
  buf->tagnum = -1;                                                     \
  buf->type = ctype;                                                    \
  buf->typenum = var->p_typetable[ig15];                                \
  buf->ref = *(long*)(var->p[ig15]+offset)+buf->obj.i*sizeof(casttype); \
  buf->obj.i = *(casttype*)buf->ref

#ifndef G__OLDIMPLEMENTATION2189
/****************************************************************
* G__LD_P10_longlong()
****************************************************************/
void G__LD_P10_longlong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[*psp-1];  
  buf->tagnum = -1;              
  buf->type = 'n';               
  buf->typenum = var->p_typetable[ig15];  
  buf->ref = *(long*)(var->p[ig15]+offset)+buf->obj.i*sizeof(G__int64); 
  buf->obj.i = *(G__int64*)buf->ref;
}
/****************************************************************
* G__LD_P10_ulonglong()
****************************************************************/
void G__LD_P10_ulonglong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[*psp-1];  
  buf->tagnum = -1;              
  buf->type = 'm';               
  buf->typenum = var->p_typetable[ig15];  
  buf->ref=*(long*)(var->p[ig15]+offset)+buf->obj.i*sizeof(G__uint64);
  buf->obj.i = *(G__uint64*)buf->ref;
}
/****************************************************************
* G__LD_P10_longdouble()
****************************************************************/
void G__LD_P10_longdouble(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[*psp-1];  
  buf->tagnum = -1;              
  buf->type = 'q';               
  buf->typenum = var->p_typetable[ig15];  
  buf->ref = *(long*)(var->p[ig15]+offset)+buf->obj.i*sizeof(long double); 
  buf->obj.i = *(long double*)buf->ref;
}
#endif /* 2189 */

/****************************************************************
* G__LD_P10_bool()
****************************************************************/
void G__LD_P10_bool(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
#ifdef G__BOOL4BYTE
  G__ASM_GET_INT_P10(int,'g');
#else
  G__ASM_GET_INT_P10(unsigned char,'g');
#endif
  buf->obj.i = buf->obj.i?1:0;
}
/****************************************************************
* G__LD_P10_char()
****************************************************************/
void G__LD_P10_char(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_P10(char,'c');
}
/****************************************************************
* G__LD_P10_uchar()
****************************************************************/
void G__LD_P10_uchar(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_P10(unsigned char,'b');
}
/****************************************************************
* G__LD_P10_short()
****************************************************************/
void G__LD_P10_short(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_P10(short,'s');
}
/****************************************************************
* G__LD_P10_ushort()
****************************************************************/
void G__LD_P10_ushort(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_P10(unsigned short,'r');
}
/****************************************************************
* G__LD_P10_int()
****************************************************************/
void G__LD_P10_int(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  /* G__ASM_GET_INT_P10(int,'i'); */
  G__value *buf= &pbuf[*psp-1];   
  buf->tagnum = -1;               
  buf->type = 'i';              
  buf->typenum = var->p_typetable[ig15];   
  buf->ref = *(long*)(var->p[ig15]+offset)+buf->obj.i*sizeof(int); 
  buf->obj.i = *(int*)buf->ref;
}

/****************************************************************
* G__LD_P10_uint()
****************************************************************/
void G__LD_P10_uint(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_P10(unsigned int,'h');
}
/****************************************************************
* G__LD_P10_long()
****************************************************************/
void G__LD_P10_long(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_P10(long,'l');
}

/****************************************************************
* G__LD_P10_ulong()
****************************************************************/
void G__LD_P10_ulong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_INT_P10(unsigned long,'k');
}
/****************************************************************
* G__LD_P10_pointer()
****************************************************************/
void G__LD_P10_pointer(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[*psp-1];
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = var->type[ig15];
  buf->typenum = var->p_typetable[ig15];
  buf->ref = *(long*)(var->p[ig15]+offset)+buf->obj.i*sizeof(long);
  buf->obj.i = *(long*)buf->ref;
  buf->obj.reftype.reftype=G__PARANORMAL;
}
/****************************************************************
* G__LD_P10_struct()
****************************************************************/
void G__LD_P10_struct(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[*psp-1];
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = 'u';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = *(long*)(var->p[ig15]+offset)
	+buf->obj.i*G__struct.size[buf->tagnum];
  buf->obj.i = buf->ref;
}
/****************************************************************
* G__LD_P10_float()
****************************************************************/
void G__LD_P10_float(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[*psp-1];
  buf->tagnum = -1;
  buf->type = 'f';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = *(long*)(var->p[ig15]+offset)+buf->obj.i*sizeof(float);
  buf->obj.d = *(float*)buf->ref;
}
/****************************************************************
* G__LD_P10_double()
****************************************************************/
void G__LD_P10_double(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[*psp-1];
  buf->tagnum = -1;
  buf->type = 'd';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = *(long*)(var->p[ig15]+offset)+buf->obj.i*sizeof(double);
  buf->obj.d = *(double*)buf->ref;
}


/*************************************************************************
* G__ST_p0_xxx
*************************************************************************/

/****************************************************************
* G__ASM_ASSIGN_INT
****************************************************************/
#define G__ASM_ASSIGN_INT(casttype)                         \
  G__value *val = &pbuf[*psp-1];                            \
  *(casttype*)(var->p[ig15]+offset)=(casttype)G__intM(val)

#ifndef G__OLDIMPLEMENTATION2189
/****************************************************************
* G__ST_p0_longlong()
****************************************************************/
void G__ST_p0_longlong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];
  G__int64 llval;
  switch(val->type) {
  case 'd':
  case 'f':
    llval = (G__int64)val->obj.d;
    break;
  case 'n':
    llval = (G__int64)val->obj.ll;
    break;
  case 'm':
    llval = (G__int64)val->obj.ull;
    break;
  case 'q':
    llval = (G__int64)val->obj.ld;
    break;
  default:
    llval = (G__int64)val->obj.i;
    break;
  }  
  *(G__int64*)(var->p[ig15]+offset)=llval;
}

/****************************************************************
* G__ST_p0_ulonglong()
****************************************************************/
void G__ST_p0_ulonglong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];
  G__uint64 ullval;
  switch(val->type) {
  case 'd':
  case 'f':
    ullval = (G__uint64)val->obj.d;
    break;
  case 'n':
    ullval = (G__uint64)val->obj.ll;
    break;
  case 'm':
    ullval = (G__uint64)val->obj.ull;
    break;
  case 'q':
    ullval = (G__uint64)val->obj.ld;
    break;
  default:
    ullval = (G__uint64)val->obj.i;
    break;
  }  
  *(G__uint64*)(var->p[ig15]+offset)=ullval;
}
/****************************************************************
* G__ST_p0_longdouble()
****************************************************************/
void G__ST_p0_longdouble(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];
  long double ldval;
  switch(val->type) {
  case 'd':
  case 'f':
    ldval = (long double)val->obj.d;
    break;
  case 'n':
    ldval = (long double)val->obj.ll;
    break;
  case 'm':
#ifdef G__WIN32
    ldval = (long double)val->obj.ll;
#else
    ldval = (long double)val->obj.ull;
#endif
    break;
  case 'q':
    ldval = (long double)val->obj.ld;
    break;
  default:
    ldval = (long double)val->obj.i;
    break;
  }  
  *(long double*)(var->p[ig15]+offset)=ldval;
}
#endif /* 2189 */

#ifndef G__OLDIMPLEMENTATION1604
/****************************************************************
* G__ST_p0_bool()
****************************************************************/
void G__ST_p0_bool(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
#if 1
#ifdef G__BOOL4BYTE
  G__ASM_ASSIGN_INT(int);
#else
  G__ASM_ASSIGN_INT(unsigned char);
#endif
#else
  G__value *val = &pbuf[*psp-1];
#ifdef G__BOOL4BYTE
  *(int*)(var->p[ig15]+offset)=(int)(G__intM(val)?1:0);
#else
  *(unsigned char*)(var->p[ig15]+offset)=(unsigned char)(G__intM(val)?1:0);
#endif
#endif
}
#endif
/****************************************************************
* G__ST_p0_char()
****************************************************************/
void G__ST_p0_char(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT(char);
}
/****************************************************************
* G__ST_p0_uchar()
****************************************************************/
void G__ST_p0_uchar(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT(unsigned char);
}
/****************************************************************
* G__ST_p0_short()
****************************************************************/
void G__ST_p0_short(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT(short);
}
/****************************************************************
* G__ST_p0_ushort()
****************************************************************/
void G__ST_p0_ushort(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT(unsigned short);
}
/****************************************************************
* G__ST_p0_int()
****************************************************************/
void G__ST_p0_int(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT(int);
}
/****************************************************************
* G__ST_p0_uint()
****************************************************************/
void G__ST_p0_uint(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT(unsigned int);
}
/****************************************************************
* G__ST_p0_long()
****************************************************************/
void G__ST_p0_long(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT(long);
}

/****************************************************************
* G__ST_p0_ulong()
****************************************************************/
void G__ST_p0_ulong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT(unsigned long);
}
/****************************************************************
* G__ST_p0_pointer()
****************************************************************/
void G__ST_p0_pointer(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];
  long address = var->p[ig15]+offset;
  long newval = G__intM(val);
  if(G__security&G__SECURE_GARBAGECOLLECTION && address) {
    if(*(long*)address) {
      G__del_refcount((void*)(*(long*)address),(void**)address);
    }
    if(newval) {
      G__add_refcount((void*)newval,(void**)address);
    }
  }
  *(long*)(address)=newval;
}
/****************************************************************
* G__ST_p0_struct()
****************************************************************/
void G__ST_p0_struct(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  memcpy((void*)(var->p[ig15]+offset),(void*)pbuf[*psp-1].obj.i
	 ,G__struct.size[var->p_tagtable[ig15]]);
}
/****************************************************************
* G__ST_p0_float()
****************************************************************/
void G__ST_p0_float(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];
  *(float*)(var->p[ig15]+offset)=(float)G__doubleM(val);
}
/****************************************************************
* G__ST_p0_double()
****************************************************************/
void G__ST_p0_double(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];
  *(double*)(var->p[ig15]+offset)=(double)G__doubleM(val);
}

/*************************************************************************
* G__ST_p1_xxx
*************************************************************************/

/****************************************************************
* G__ASM_ASSIGN_INT_P1
****************************************************************/
#ifdef G__TUNEUP_W_SECURITY
#define G__ASM_ASSIGN_INT_P1(casttype)                           \
  G__value *val = &pbuf[*psp-1];                                 \
  if('d'==val->type||'f'==val->type) G__nonintarrayindex(var,ig15); /*1969*/ \
  if(val->obj.i>var->varlabel[ig15][1])                          \
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],val->obj.i);  \
  else                                                           \
    *(casttype*)(var->p[ig15]+offset+val->obj.i*sizeof(casttype)) \
            = (casttype)G__int(pbuf[*psp-2]);                    \
  --(*psp)
#else
#define G__ASM_ASSIGN_INT_P1(casttype)                           \
  G__value *val = &pbuf[*psp-1];                                 \
  *(casttype*)(var->p[ig15]+offset+val->obj.i*sizeof(casttype))  \
            = (casttype)G__int(pbuf[*psp-2]);                    \
  --(*psp)
#endif

#ifndef G__OLDIMPLEMENTATION2189
/****************************************************************
* G__ST_p1_longlong()
****************************************************************/
void G__ST_p1_longlong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];                                 
  if('d'==val->type||'f'==val->type) G__nonintarrayindex(var,ig15); /*1969*/ 
  if(val->obj.i>var->varlabel[ig15][1])                          
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],val->obj.i);  
  else                                                           
    *(G__int64*)(var->p[ig15]+offset+val->obj.i*sizeof(G__int64)) 
            = (G__int64)G__Longlong(pbuf[*psp-2]);                    
  --(*psp);
}
/****************************************************************
* G__ST_p1_ulonglong()
****************************************************************/
void G__ST_p1_ulonglong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];                                 
  if('d'==val->type||'f'==val->type) G__nonintarrayindex(var,ig15); /*1969*/ 
  if(val->obj.i>var->varlabel[ig15][1])                          
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],val->obj.i);  
  else                                                           
    *(G__uint64*)(var->p[ig15]+offset+val->obj.i*sizeof(G__uint64)) 
            = (G__uint64)G__ULonglong(pbuf[*psp-2]); 
  --(*psp);
}
/****************************************************************
* G__ST_p1_longdouble()
****************************************************************/
void G__ST_p1_longdouble(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];                                 
  if('d'==val->type||'f'==val->type) G__nonintarrayindex(var,ig15); /*1969*/ 
  if(val->obj.i>var->varlabel[ig15][1])                          
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],val->obj.i);  
  else                                                           
    *(long double*)(var->p[ig15]+offset+val->obj.i*sizeof(long double)) 
            = (long double)G__Longdouble(pbuf[*psp-2]); 
  --(*psp);
}
#endif /* 2189 */

/****************************************************************
* G__ST_p1_bool()
****************************************************************/
void G__ST_p1_bool(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
#if 1
#ifdef G__BOOL4BYTE
  G__ASM_ASSIGN_INT_P1(int);
#else
  G__ASM_ASSIGN_INT_P1(unsigned char);
#endif
#else
  G__value *val = &pbuf[*psp-1];
#ifdef G__BOOL4BYTE
  *(int*)(var->p[ig15]+offset+val->obj.i*sizeof(int))
            = (int)G__int(pbuf[*psp-2])?1:0;
#else
  *(unsigned char*)(var->p[ig15]+offset+val->obj.i*sizeof(unsigned char))
            = (unsigned char)G__int(pbuf[*psp-2])?1:0;
#endif
#endif
}
/****************************************************************
* G__ST_p1_char()
****************************************************************/
void G__ST_p1_char(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P1(char);
}
/****************************************************************
* G__ST_p1_uchar()
****************************************************************/
void G__ST_p1_uchar(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P1(unsigned char);
}
/****************************************************************
* G__ST_p1_short()
****************************************************************/
void G__ST_p1_short(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P1(short);
}
/****************************************************************
* G__ST_p1_ushort()
****************************************************************/
void G__ST_p1_ushort(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P1(unsigned short);
}
/****************************************************************
* G__ST_p1_int()
****************************************************************/
void G__ST_p1_int(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P1(int);
}

/****************************************************************
* G__ST_p1_uint()
****************************************************************/
void G__ST_p1_uint(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P1(unsigned int);
}
/****************************************************************
* G__ST_p1_long()
****************************************************************/
void G__ST_p1_long(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P1(long);
}

/****************************************************************
* G__ST_p1_ulong()
****************************************************************/
void G__ST_p1_ulong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P1(unsigned long);
}
/****************************************************************
* G__ST_p1_pointer()
****************************************************************/
void G__ST_p1_pointer(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];
#ifndef G__OLDIMPLEMENTATION1969
  if('d'==val->type||'f'==val->type) G__nonintarrayindex(var,ig15); 
#endif
  if(val->obj.i>var->varlabel[ig15][1]) {
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],val->obj.i);
  }
  else {
    long address = (var->p[ig15]+offset+val->obj.i*sizeof(long));
    long newval = G__int(pbuf[*psp-2]);
    if(G__security&G__SECURE_GARBAGECOLLECTION && address) {
      if(*(long*)address) {
	G__del_refcount((void*)(*(long*)address),(void**)address);
      }
      if(newval) {
	G__add_refcount((void*)newval,(void**)address);
      }
    }
    *(long*)(address) = newval;
  }
  --(*psp);
}
/****************************************************************
* G__ST_p1_struct()
****************************************************************/
void G__ST_p1_struct(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];
#ifndef G__OLDIMPLEMENTATION1969
  if('d'==val->type||'f'==val->type) G__nonintarrayindex(var,ig15); 
#endif
#ifdef G__TUNEUP_W_SECURITY
  if(val->obj.i>var->varlabel[ig15][1])
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],val->obj.i);
  else
#endif
    memcpy((void*)(var->p[ig15]+offset
		 +val->obj.i*G__struct.size[var->p_tagtable[ig15]])
	 ,(void*)pbuf[*psp-2].obj.i,G__struct.size[var->p_tagtable[ig15]]);
  --(*psp);
}
/****************************************************************
* G__ST_p1_float()
****************************************************************/
void G__ST_p1_float(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];
#ifndef G__OLDIMPLEMENTATION1969
  if('d'==val->type||'f'==val->type) G__nonintarrayindex(var,ig15); 
#endif
#ifdef G__TUNEUP_W_SECURITY
  if(val->obj.i>var->varlabel[ig15][1])
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],val->obj.i);
  else
#endif
    *(float*)(var->p[ig15]+offset+val->obj.i*sizeof(float))
            = (float)G__double(pbuf[*psp-2]);
  --(*psp);
}
/****************************************************************
* G__ST_p1_double()
****************************************************************/
void G__ST_p1_double(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];
#ifndef G__OLDIMPLEMENTATION1969
  if('d'==val->type||'f'==val->type) G__nonintarrayindex(var,ig15); 
#endif
#ifdef G__TUNEUP_W_SECURITY
  if(val->obj.i>var->varlabel[ig15][1])
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],val->obj.i);
  else
#endif
    *(double*)(var->p[ig15]+offset+val->obj.i*sizeof(double))
            = (double)G__double(pbuf[*psp-2]);
  --(*psp);
}

#ifndef G__OLDIMPLEMENTATION1406
/*************************************************************************
* G__ST_pn_xxx
*************************************************************************/

/****************************************************************
* G__ASM_ASSIGN_INT_PN
****************************************************************/
#ifdef G__TUNEUP_W_SECURITY
#define G__ASM_ASSIGN_INT_PN(casttype)                           \
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];          \
  int ary = var->varlabel[ig15][0];                              \
  int paran = var->paran[ig15];                                  \
  int p_inc=0;                                                   \
  int ig25;                                                      \
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {         \
    p_inc += ary*G__int(buf[ig25]);                              \
    ary /= var->varlabel[ig15][ig25+2];                          \
  }                                                              \
  if(p_inc>var->varlabel[ig15][1])                               \
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);    \
  else                                                           \
    *(casttype*)(var->p[ig15]+offset+p_inc*sizeof(casttype))     \
            = (casttype)G__int(pbuf[*psp-1])
#else
#define G__ASM_ASSIGN_INT_PN(casttype)                           \
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];          \
  int ary = var->varlabel[ig15][0];                              \
  int paran = var->paran[ig15];                                  \
  int p_inc=0;                                                   \
  int ig25;                                                      \
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {         \
    p_inc += ary*G__int(buf[ig25]);                              \
    ary /= var->varlabel[ig15][ig25+2];                          \
  }                                                              \
  *(casttype*)(var->p[ig15]+offset+p_inc*sizeof(casttype))       \
            = (casttype)G__int(pbuf[*psp-1])
#endif

#ifndef G__OLDIMPLEMENTATION2189
/****************************************************************
* G__ST_pn_longlong()
****************************************************************/
void G__ST_pn_longlong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];          
  int ary = var->varlabel[ig15][0];                              
  int paran = var->paran[ig15];                                  
  int p_inc=0;                                                   
  int ig25;                                                      
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {         
    p_inc += ary*G__int(buf[ig25]);                              
    ary /= var->varlabel[ig15][ig25+2];                          
  }                                                              
  if(p_inc>var->varlabel[ig15][1])                               
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);    
  else                                                           
    *(G__int64*)(var->p[ig15]+offset+p_inc*sizeof(G__int64))   
      = (G__int64)G__Longlong(pbuf[*psp-1]);
}

/****************************************************************
* G__ST_pn_ulonglong()
****************************************************************/
void G__ST_pn_ulonglong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];          
  int ary = var->varlabel[ig15][0];                              
  int paran = var->paran[ig15];                                  
  int p_inc=0;                                                   
  int ig25;                                                      
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {         
    p_inc += ary*G__int(buf[ig25]);                              
    ary /= var->varlabel[ig15][ig25+2];                          
  }                                                              
  if(p_inc>var->varlabel[ig15][1])                               
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);    
  else                                                           
    *(G__uint64*)(var->p[ig15]+offset+p_inc*sizeof(G__uint64))   
      = (G__uint64)G__ULonglong(pbuf[*psp-1]);
}

/****************************************************************
* G__ST_pn_longdouble()
****************************************************************/
void G__ST_pn_longdouble(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];          
  int ary = var->varlabel[ig15][0];                              
  int paran = var->paran[ig15];                                  
  int p_inc=0;                                                   
  int ig25;                                                      
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {         
    p_inc += ary*G__int(buf[ig25]);                              
    ary /= var->varlabel[ig15][ig25+2];                          
  }                                                              
  if(p_inc>var->varlabel[ig15][1])                               
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);    
  else                                                           
    *(long double*)(var->p[ig15]+offset+p_inc*sizeof(long double))   
      = (long double)G__Longdouble(pbuf[*psp-1]);
}
#endif /* 2189 */

/****************************************************************
* G__ST_pn_bool()
****************************************************************/
void G__ST_pn_bool(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
#ifdef G__BOOL4BYTE
  G__ASM_ASSIGN_INT_PN(int);
#else
  G__ASM_ASSIGN_INT_PN(unsigned char);
#endif
  /*  val?1:0 */
}
/****************************************************************
* G__ST_p0_char()
****************************************************************/
void G__ST_pn_char(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_PN(char);
}
/****************************************************************
* G__ST_pn_uchar()
****************************************************************/
void G__ST_pn_uchar(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_PN(unsigned char);
}
/****************************************************************
* G__ST_pn_short()
****************************************************************/
void G__ST_pn_short(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_PN(short);
}
/****************************************************************
* G__ST_pn_ushort()
****************************************************************/
void G__ST_pn_ushort(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_PN(unsigned short);
}
/****************************************************************
* G__ST_pn_int()
****************************************************************/
void G__ST_pn_int(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_PN(int);
}

/****************************************************************
* G__ST_pn_uint()
****************************************************************/
void G__ST_pn_uint(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_PN(unsigned int);
}
/****************************************************************
* G__ST_pn_long()
****************************************************************/
void G__ST_pn_long(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_PN(long);
}

/****************************************************************
* G__ST_pn_ulong()
****************************************************************/
void G__ST_pn_ulong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_PN(unsigned long);
}
/****************************************************************
* G__ST_pn_pointer()
****************************************************************/
void G__ST_pn_pointer(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];
  int ary = var->varlabel[ig15][0];
  int paran = var->paran[ig15];
  int p_inc=0;
  int ig25;
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
    p_inc += ary*G__int(buf[ig25]);
    ary /= var->varlabel[ig15][ig25+2];
  }
  if(p_inc>var->varlabel[ig15][1]) {
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);
  }
  else {
    long address = (var->p[ig15]+offset+p_inc*sizeof(long));
    long newval = G__int(pbuf[*psp-1]);
    if(G__security&G__SECURE_GARBAGECOLLECTION && address) {
      if(*(long*)address) {
	G__del_refcount((void*)(*(long*)address),(void**)address);
      }
      if(newval) {
	G__add_refcount((void*)newval,(void**)address);
      }
    }
    *(long*)(address) = newval;
  }
}
/****************************************************************
* G__ST_pn_struct()
****************************************************************/
void G__ST_pn_struct(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];
  int ary = var->varlabel[ig15][0];
  int paran = var->paran[ig15];
  int p_inc=0;
  int ig25;
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
    p_inc += ary*G__int(buf[ig25]);
    ary /= var->varlabel[ig15][ig25+2];
  }
#ifdef G__TUNEUP_W_SECURITY
  if(p_inc>var->varlabel[ig15][1])
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);
  else
#endif
    memcpy((void*)(var->p[ig15]+offset
		 +p_inc*G__struct.size[var->p_tagtable[ig15]])
	 ,(void*)pbuf[*psp-1].obj.i,G__struct.size[var->p_tagtable[ig15]]);
}
/****************************************************************
* G__ST_pn_float()
****************************************************************/
void G__ST_pn_float(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];
  int ary = var->varlabel[ig15][0];
  int paran = var->paran[ig15];
  int p_inc=0;
  int ig25;
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
    p_inc += ary*G__int(buf[ig25]);
    ary /= var->varlabel[ig15][ig25+2];
  }
#ifdef G__TUNEUP_W_SECURITY
  if(p_inc>var->varlabel[ig15][1])
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);
  else
#endif
    *(float*)(var->p[ig15]+offset+p_inc*sizeof(float))
            = (float)G__double(pbuf[*psp-1]);
}
/****************************************************************
* G__ST_pn_double()
****************************************************************/
void G__ST_pn_double(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];
  int ary = var->varlabel[ig15][0];
  int paran = var->paran[ig15];
  int p_inc=0;
  int ig25;
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
    p_inc += ary*G__int(buf[ig25]);
    ary /= var->varlabel[ig15][ig25+2];
  }
#ifdef G__TUNEUP_W_SECURITY
  if(p_inc>var->varlabel[ig15][1])
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);
  else
#endif
    *(double*)(var->p[ig15]+offset+p_inc*sizeof(double))
            = (double)G__double(pbuf[*psp-1]);
}
#endif /* 1406 */

/*************************************************************************
* G__ST_P10_xxx
*************************************************************************/

/****************************************************************
* G__ASM_ASSIGN_INT_P10
****************************************************************/
#define G__ASM_ASSIGN_INT_P10(casttype)                                    \
  G__value *val = &pbuf[*psp-1];                                          \
  *(casttype*)(*(long*)(var->p[ig15]+offset)+val->obj.i*sizeof(casttype)) \
            = (casttype)G__int(pbuf[*psp-2]);                             \
  --(*psp)

#ifndef G__OLDIMPLEMENTATION2189
/****************************************************************
* G__ST_p10_longlong()
****************************************************************/
void G__ST_P10_longlong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];                                          
  *(G__int64*)(*(long*)(var->p[ig15]+offset)+val->obj.i*sizeof(G__int64)) 
            = (G__int64)G__Longlong(pbuf[*psp-2]);
  --(*psp);
}
/****************************************************************
* G__ST_p10_ulonglong()
****************************************************************/
void G__ST_P10_ulonglong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];                                          
  *(G__uint64*)(*(long*)(var->p[ig15]+offset)+val->obj.i*sizeof(G__uint64)) 
            = (G__uint64)G__ULonglong(pbuf[*psp-2]);
  --(*psp);
}
/****************************************************************
* G__ST_p10_longdouble()
****************************************************************/
void G__ST_P10_longdouble(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];                                          
  *(long double*)(*(long*)(var->p[ig15]+offset)+val->obj.i*sizeof(long double)) 
            = (long double)G__Longdouble(pbuf[*psp-2]);
  --(*psp);
}
#endif /* 2189 */

/****************************************************************
* G__ST_p0_bool()
****************************************************************/
void G__ST_P10_bool(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
#if 1
#ifdef G__BOOL4BYTE
  G__ASM_ASSIGN_INT_P10(int);
#else
  G__ASM_ASSIGN_INT_P10(unsigned char);
#endif
#else
  G__value *val = &pbuf[*psp-1];
#ifdef G__BOOL4BYTE
  *(int*)(*(long*)(var->p[ig15]+offset)+val->obj.i*sizeof(int))
            = (int)G__int(pbuf[*psp-2])?1:0;                  
#else
  *(unsigned char*)(*(long*)(var->p[ig15]+offset)+val->obj.i*sizeof(unsigned char))
            = (unsigned char)G__int(pbuf[*psp-2])?1:0;
#endif
  --(*psp);
#endif
}
/****************************************************************
* G__ST_p0_char()
****************************************************************/
void G__ST_P10_char(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P10(char);
}
/****************************************************************
* G__ST_P10_uchar()
****************************************************************/
void G__ST_P10_uchar(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P10(unsigned char);
}
/****************************************************************
* G__ST_P10_short()
****************************************************************/
void G__ST_P10_short(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P10(short);
}
/****************************************************************
* G__ST_P10_ushort()
****************************************************************/
void G__ST_P10_ushort(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P10(unsigned short);
}
/****************************************************************
* G__ST_P10_int()
****************************************************************/
void G__ST_P10_int(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P10(int);
}

/****************************************************************
* G__ST_P10_uint()
****************************************************************/
void G__ST_P10_uint(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P10(unsigned int);
}
/****************************************************************
* G__ST_P10_long()
****************************************************************/
void G__ST_P10_long(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P10(long);
}

/****************************************************************
* G__ST_P10_ulong()
****************************************************************/
void G__ST_P10_ulong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P10(unsigned long);
}
/****************************************************************
* G__ST_P10_pointer()
****************************************************************/
void G__ST_P10_pointer(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_INT_P10(long);
}
/****************************************************************
* G__ST_P10_struct()
****************************************************************/
void G__ST_P10_struct(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];
  memcpy((void*)(*(long*)(var->p[ig15]+offset)
		 +val->obj.i*G__struct.size[var->p_tagtable[ig15]])
	 ,(void*)pbuf[*psp-2].obj.i,G__struct.size[var->p_tagtable[ig15]]);
  --(*psp);
}
/****************************************************************
* G__ST_P10_float()
****************************************************************/
void G__ST_P10_float(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];
  *(float*)(*(long*)(var->p[ig15]+offset)+val->obj.i*sizeof(float))
            = (float)G__double(pbuf[*psp-2]);
  --(*psp);
}
/****************************************************************
* G__ST_P10_double()
****************************************************************/
void G__ST_P10_double(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];
  *(double*)(*(long*)(var->p[ig15]+offset)+val->obj.i*sizeof(double))
            = (double)G__double(pbuf[*psp-2]);
  --(*psp);
}

/*************************************************************************
* G__LD_Rp0_xxx
*
*  type &p;
*  p;    optimize this expression
*************************************************************************/

/****************************************************************
* G__ASM_GET_REFINT
****************************************************************/
#define G__ASM_GET_REFINT(casttype,ctype) \
  G__value *buf= &pbuf[(*psp)++];         \
  buf->tagnum = -1;                       \
  buf->type = ctype;                      \
  buf->typenum = var->p_typetable[ig15];  \
  buf->ref = *(long*)(var->p[ig15]+offset);  \
  buf->obj.i = *(casttype*)buf->ref

#ifndef G__OLDIMPLEMENTATION2189
/****************************************************************
* G__LD_Rp0_longlong()
****************************************************************/
void G__LD_Rp0_longlong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];         
  buf->tagnum = -1;                       
  buf->type = 'n';                      
  buf->typenum = var->p_typetable[ig15];  
  buf->ref = *(long*)(var->p[ig15]+offset);  
  buf->obj.i = *(G__int64*)buf->ref;
}
/****************************************************************
* G__LD_Rp0_ulonglong()
****************************************************************/
void G__LD_Rp0_ulonglong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];         
  buf->tagnum = -1;                       
  buf->type = 'm';                      
  buf->typenum = var->p_typetable[ig15];  
  buf->ref = *(long*)(var->p[ig15]+offset);  
  buf->obj.i = *(G__uint64*)buf->ref;
}
/****************************************************************
* G__LD_Rp0_longdouble()
****************************************************************/
void G__LD_Rp0_longdouble(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];         
  buf->tagnum = -1;                       
  buf->type = 'q';                      
  buf->typenum = var->p_typetable[ig15];  
  buf->ref = *(long*)(var->p[ig15]+offset);  
  buf->obj.i = *(long double*)buf->ref;
}

#endif /* 2189 */

/****************************************************************
* G__LD_Rp0_bool()
****************************************************************/
void G__LD_Rp0_bool(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
#ifdef G__BOOL4BYTE
  G__ASM_GET_REFINT(int,'g');
#else
  G__ASM_GET_REFINT(unsigned char,'g');
#endif
  buf->obj.i = buf->obj.i?1:0;
}
/****************************************************************
* G__LD_Rp0_char()
****************************************************************/
void G__LD_Rp0_char(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFINT(char,'c');
}
/****************************************************************
* G__LD_Rp0_uchar()
****************************************************************/
void G__LD_Rp0_uchar(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFINT(unsigned char,'b');
}
/****************************************************************
* G__LD_Rp0_short()
****************************************************************/
void G__LD_Rp0_short(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFINT(short,'s');
}
/****************************************************************
* G__LD_Rp0_ushort()
****************************************************************/
void G__LD_Rp0_ushort(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFINT(unsigned short,'r');
}
/****************************************************************
* G__LD_Rp0_int()
****************************************************************/
void G__LD_Rp0_int(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFINT(int,'i');
}
/****************************************************************
* G__LD_Rp0_uint()
****************************************************************/
void G__LD_Rp0_uint(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFINT(unsigned int,'h');
}
/****************************************************************
* G__LD_Rp0_long()
****************************************************************/
void G__LD_Rp0_long(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFINT(long,'l');
}

/****************************************************************
* G__LD_Rp0_ulong()
****************************************************************/
void G__LD_Rp0_ulong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFINT(unsigned long,'k');
}
/****************************************************************
* G__LD_Rp0_pointer()
****************************************************************/
void G__LD_Rp0_pointer(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = var->type[ig15];
  buf->typenum = var->p_typetable[ig15];
  buf->ref = *(long*)(var->p[ig15]+offset);
  buf->obj.i = *(long*)buf->ref;
  buf->obj.reftype.reftype=G__PARANORMAL;
}
/****************************************************************
* G__LD_Rp0_struct()
****************************************************************/
void G__LD_Rp0_struct(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = 'u';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = *(long*)(var->p[ig15]+offset);
  buf->obj.i = buf->ref;
}
/****************************************************************
* G__LD_Rp0_float()
****************************************************************/
void G__LD_Rp0_float(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = -1;
  buf->type = 'f';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = *(long*)(var->p[ig15]+offset);
  buf->obj.d = *(float*)buf->ref;
}
/****************************************************************
* G__LD_Rp0_double()
****************************************************************/
void G__LD_Rp0_double(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = -1;
  buf->type = 'd';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = *(long*)(var->p[ig15]+offset);
  buf->obj.d = *(double*)buf->ref;
}

/*************************************************************************
* G__ST_Rp0_xxx
*
*  type &p;
*  p=x;    optimize this expression
*************************************************************************/

/****************************************************************
* G__ASM_ASSIGN_REFINT
****************************************************************/
#define G__ASM_ASSIGN_REFINT(casttype)                      \
  G__value *val = &pbuf[*psp-1];                            \
  long adr = *(long*)(var->p[ig15]+offset);                 \
  *(casttype*)adr=(casttype)G__intM(val)


#ifndef G__OLDIMPLEMENTATION2189 
/****************************************************************
* G__ST_Rp0_longlong()
****************************************************************/
void G__ST_Rp0_longlong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];                    
  long adr = *(long*)(var->p[ig15]+offset);         
  G__int64 llval;
  switch(val->type) {
  case 'd':
  case 'f':
    llval = (G__int64)val->obj.d;
    break;
  case 'n':
    llval = (G__int64)val->obj.ll;
    break;
  case 'm':
    llval = (G__int64)val->obj.ull;
    break;
  case 'q':
    llval = (G__int64)val->obj.ld;
    break;
  default:
    llval = (G__int64)val->obj.i;
    break;
  }  
  *(G__int64*)adr=llval;
}

/****************************************************************
* G__ST_Rp0_ulonglong()
****************************************************************/
void G__ST_Rp0_ulonglong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];                    
  long adr = *(long*)(var->p[ig15]+offset);         
  G__uint64 ullval;
  switch(val->type) {
  case 'd':
  case 'f':
    ullval = (G__uint64)val->obj.d;
    break;
  case 'n':
    ullval = (G__uint64)val->obj.ll;
    break;
  case 'm':
    ullval = (G__uint64)val->obj.ull;
    break;
  case 'q':
    ullval = (G__uint64)val->obj.ld;
    break;
  default:
    ullval = (G__uint64)val->obj.i;
    break;
  }  
  *(G__uint64*)adr=ullval;
}
/****************************************************************
* G__ST_Rp0_longdouble()
****************************************************************/
void G__ST_Rp0_longdouble(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];                    
  long adr = *(long*)(var->p[ig15]+offset);         
  long double ldval;
  switch(val->type) {
  case 'd':
  case 'f':
    ldval = (long double)val->obj.d;
    break;
  case 'n':
    ldval = (long double)val->obj.ll;
    break;
  case 'm':
#ifdef G__WIN32
    ldval = (long double)val->obj.ll;
#else
    ldval = (long double)val->obj.ull;
#endif
    break;
  case 'q':
    ldval = (long double)val->obj.ld;
    break;
  default:
    ldval = (long double)val->obj.i;
    break;
  }  
  *(long double*)adr=ldval;
}
#endif /* 2189 */

/****************************************************************
* G__ST_Rp0_bool()
****************************************************************/
void G__ST_Rp0_bool(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
#ifdef G__BOOL4BYTE
  G__ASM_ASSIGN_REFINT(int);
  /* *(int*)adr=(int)G__intM(val)?1:0; */
#else
  G__ASM_ASSIGN_REFINT(unsigned char);
  /* *(unsigned char*)adr=(unsigned char)G__intM(val)?1:0; */
#endif
}
/****************************************************************
* G__ST_Rp0_char()
****************************************************************/
void G__ST_Rp0_char(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_REFINT(char);
}
/****************************************************************
* G__ST_Rp0_uchar()
****************************************************************/
void G__ST_Rp0_uchar(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_REFINT(unsigned char);
}
/****************************************************************
* G__ST_Rp0_short()
****************************************************************/
void G__ST_Rp0_short(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_REFINT(short);
}
/****************************************************************
* G__ST_Rp0_ushort()
****************************************************************/
void G__ST_Rp0_ushort(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_REFINT(unsigned short);
}
/****************************************************************
* G__ST_Rp0_int()
****************************************************************/
void G__ST_Rp0_int(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_REFINT(int);
}
/****************************************************************
* G__ST_Rp0_uint()
****************************************************************/
void G__ST_Rp0_uint(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_REFINT(unsigned int);
}
/****************************************************************
* G__ST_Rp0_long()
****************************************************************/
void G__ST_Rp0_long(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_REFINT(long);
}

/****************************************************************
* G__ST_Rp0_ulong()
****************************************************************/
void G__ST_Rp0_ulong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_REFINT(unsigned long);
}
/****************************************************************
* G__ST_Rp0_pointer()
****************************************************************/
void G__ST_Rp0_pointer(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_ASSIGN_REFINT(long);
}
/****************************************************************
* G__ST_Rp0_struct()
****************************************************************/
void G__ST_Rp0_struct(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  memcpy((void*)(*(long*)(var->p[ig15]+offset)),(void*)pbuf[*psp-1].obj.i
	 ,G__struct.size[var->p_tagtable[ig15]]);
}
/****************************************************************
* G__ST_Rp0_float()
****************************************************************/
void G__ST_Rp0_float(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];
  *(float*)(*(long*)(var->p[ig15]+offset))=(float)G__doubleM(val);
}
/****************************************************************
* G__ST_Rp0_double()
****************************************************************/
void G__ST_Rp0_double(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *val = &pbuf[*psp-1];
  *(double*)(*(long*)(var->p[ig15]+offset))=(double)G__doubleM(val);
}

/*************************************************************************
* G__LD_RP0_xxx
*
*  type &p;
*  &p;    optimize this expression
*************************************************************************/

/****************************************************************
* G__ASM_GET_REFPINT
****************************************************************/
#define G__ASM_GET_REFPINT(casttype,ctype) \
  G__value *buf= &pbuf[(*psp)++];         \
  buf->tagnum = -1;                       \
  buf->type = ctype;                      \
  buf->typenum = var->p_typetable[ig15];  \
  buf->ref = var->p[ig15]+offset;         \
  buf->obj.i = *(long*)buf->ref

#ifndef G__OLDIMPLEMENTATION2189
/****************************************************************
* G__LD_RP0_longlong()
****************************************************************/
void G__LD_RP0_longlong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];         
  buf->tagnum = -1;                       
  buf->type = 'n';                      
  buf->typenum = var->p_typetable[ig15];  
  buf->ref = var->p[ig15]+offset;         
  buf->obj.ll = *(long*)buf->ref;
}
/****************************************************************
* G__LD_RP0_ulonglong()
****************************************************************/
void G__LD_RP0_ulonglong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];         
  buf->tagnum = -1;                       
  buf->type = 'm';                      
  buf->typenum = var->p_typetable[ig15];  
  buf->ref = var->p[ig15]+offset;         
  buf->obj.ull = *(long*)buf->ref;
}
/****************************************************************
* G__LD_RP0_longdouble()
****************************************************************/
void G__LD_RP0_longdouble(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];         
  buf->tagnum = -1;                       
  buf->type = 'q';                      
  buf->typenum = var->p_typetable[ig15];  
  buf->ref = var->p[ig15]+offset;         
  buf->obj.ld = *(long*)buf->ref;
}
#endif /* 2189 */

/****************************************************************
* G__LD_RP0_bool()
****************************************************************/
void G__LD_RP0_bool(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
#ifdef G__BOOL4BYTE
  G__ASM_GET_REFPINT(int,'G');
#else
  G__ASM_GET_REFPINT(unsigned char,'G');
#endif
  buf->obj.i = buf->obj.i?1:0;
}
/****************************************************************
* G__LD_RP0_char()
****************************************************************/
void G__LD_RP0_char(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFPINT(char,'C');
}
/****************************************************************
* G__LD_RP0_uchar()
****************************************************************/
void G__LD_RP0_uchar(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFPINT(unsigned char,'B');
}
/****************************************************************
* G__LD_RP0_short()
****************************************************************/
void G__LD_RP0_short(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFPINT(short,'S');
}
/****************************************************************
* G__LD_RP0_ushort()
****************************************************************/
void G__LD_RP0_ushort(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFPINT(unsigned short,'R');
}
/****************************************************************
* G__LD_RP0_int()
****************************************************************/
void G__LD_RP0_int(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFPINT(int,'I');
}
/****************************************************************
* G__LD_RP0_uint()
****************************************************************/
void G__LD_RP0_uint(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFPINT(unsigned int,'H');
}
/****************************************************************
* G__LD_RP0_long()
****************************************************************/
void G__LD_RP0_long(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFPINT(long,'L');
}

/****************************************************************
* G__LD_RP0_ulong()
****************************************************************/
void G__LD_RP0_ulong(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__ASM_GET_REFPINT(unsigned long,'K');
}
/****************************************************************
* G__LD_RP0_pointer()
****************************************************************/
void G__LD_RP0_pointer(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = var->type[ig15];
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset;
  buf->obj.i = *(long*)buf->ref;
  buf->obj.reftype.reftype=G__PARAP2P;
}
/****************************************************************
* G__LD_RP0_struct()
****************************************************************/
void G__LD_RP0_struct(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = 'U';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset;
  buf->obj.i = *(long*)buf->ref;
}
/****************************************************************
* G__LD_RP0_float()
****************************************************************/
void G__LD_RP0_float(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = -1;
  buf->type = 'F';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset;
  buf->obj.d = *(long*)buf->ref;
}
/****************************************************************
* G__LD_RP0_double()
****************************************************************/
void G__LD_RP0_double(pbuf,psp,offset,var,ig15)
G__value *pbuf;
int *psp;
long offset;
struct G__var_array *var;
long ig15;
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = -1;
  buf->type = 'D';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset;
  buf->obj.d = *(long*)buf->ref;
}

#ifndef G__OLDIMPLEMENTATION1491
/****************************************************************
* G__OP2_OPTIMIZED_UU
****************************************************************/

/*************************************************************************
* G__OP2_plus_uu()
*************************************************************************/
void G__OP2_plus_uu(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.ulo = bufm2->obj.ulo + bufm1->obj.ulo;
  bufm2->type = 'h';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_minus_uu()
*************************************************************************/
void G__OP2_minus_uu(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.ulo = bufm2->obj.ulo - bufm1->obj.ulo;
  bufm2->type = 'h';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_multiply_uu()
*************************************************************************/
void G__OP2_multiply_uu(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.ulo = bufm2->obj.ulo * bufm1->obj.ulo;
  bufm2->type = 'h';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_divide_uu()
*************************************************************************/
void G__OP2_divide_uu(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  if(0==bufm1->obj.ulo) {
    G__genericerror("Error: operator '/' divided by zero");
    return;
  }
  bufm2->obj.ulo = bufm2->obj.ulo / bufm1->obj.ulo;
  bufm2->type = 'h';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_addassign_uu()
*************************************************************************/
void G__OP2_addassign_uu(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.ulo += bufm1->obj.ulo;
  *(unsigned int*)bufm2->ref=(unsigned int)bufm2->obj.ulo;
}
/*************************************************************************
* G__OP2_subassign_uu()
*************************************************************************/
void G__OP2_subassign_uu(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.ulo -= bufm1->obj.ulo;
  *(unsigned int*)bufm2->ref=(unsigned int)bufm2->obj.ulo;
}
/*************************************************************************
* G__OP2_mulassign_uu()
*************************************************************************/
void G__OP2_mulassign_uu(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.ulo *= bufm1->obj.ulo;
  *(unsigned int*)bufm2->ref=(unsigned int)bufm2->obj.ulo;
}
/*************************************************************************
* G__OP2_divassign_uu()
*************************************************************************/
void G__OP2_divassign_uu(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  if(0==bufm1->obj.ulo) {
    G__genericerror("Error: operator '/' divided by zero");
    return;
  }
  bufm2->obj.ulo /= bufm1->obj.ulo;
  *(unsigned int*)bufm2->ref=(unsigned int)bufm2->obj.ulo;
}

#endif

#ifndef G__OLDIMPLEMENTATION572
/****************************************************************
* G__OP2_OPTIMIZED_II
****************************************************************/

/*************************************************************************
* G__OP2_plus_ii()
*************************************************************************/
void G__OP2_plus_ii(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.i = bufm2->obj.i + bufm1->obj.i;
  bufm2->type = 'i';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_minus_ii()
*************************************************************************/
void G__OP2_minus_ii(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.i = bufm2->obj.i - bufm1->obj.i;
  bufm2->type = 'i';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_multiply_ii()
*************************************************************************/
void G__OP2_multiply_ii(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.i = bufm2->obj.i * bufm1->obj.i;
  bufm2->type = 'i';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_divide_ii()
*************************************************************************/
void G__OP2_divide_ii(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  if(0==bufm1->obj.i) {
    G__genericerror("Error: operator '/' divided by zero");
    return;
  }
  bufm2->obj.i = bufm2->obj.i / bufm1->obj.i;
  bufm2->type = 'i';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_addassign_ii()
*************************************************************************/
void G__OP2_addassign_ii(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.i += bufm1->obj.i;
  *(int*)bufm2->ref=(int)bufm2->obj.i;
}
/*************************************************************************
* G__OP2_subassign_ii()
*************************************************************************/
void G__OP2_subassign_ii(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.i -= bufm1->obj.i;
  *(int*)bufm2->ref=(int)bufm2->obj.i;
}
/*************************************************************************
* G__OP2_mulassign_ii()
*************************************************************************/
void G__OP2_mulassign_ii(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.i *= bufm1->obj.i;
  *(int*)bufm2->ref=(int)bufm2->obj.i;
}
/*************************************************************************
* G__OP2_divassign_ii()
*************************************************************************/
void G__OP2_divassign_ii(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  if(0==bufm1->obj.i) {
    G__genericerror("Error: operator '/' divided by zero");
    return;
  }
  bufm2->obj.i /= bufm1->obj.i;
  *(int*)bufm2->ref=(int)bufm2->obj.i;
}


/****************************************************************
* G__OP2_OPTIMIZED_DD
****************************************************************/

/*************************************************************************
* G__OP2_plus_dd()
*************************************************************************/
void G__OP2_plus_dd(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.d = bufm2->obj.d + bufm1->obj.d;
  bufm2->type = 'd';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_minus_dd()
*************************************************************************/
void G__OP2_minus_dd(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.d = bufm2->obj.d - bufm1->obj.d;
  bufm2->type = 'd';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_multiply_dd()
*************************************************************************/
void G__OP2_multiply_dd(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.d = bufm2->obj.d * bufm1->obj.d;
  bufm2->type = 'd';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_divide_dd()
*************************************************************************/
void G__OP2_divide_dd(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  if(0==bufm1->obj.d) {
    G__genericerror("Error: operator '/' divided by zero");
    return;
  }
  bufm2->obj.d = bufm2->obj.d / bufm1->obj.d;
  bufm2->type = 'd';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_addassign_dd()
*************************************************************************/
void G__OP2_addassign_dd(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.d += bufm1->obj.d;
  *(double*)bufm2->ref=bufm2->obj.d;
}
/*************************************************************************
* G__OP2_subassign_dd()
*************************************************************************/
void G__OP2_subassign_dd(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.d -= bufm1->obj.d;
  *(double*)bufm2->ref=bufm2->obj.d;
}
/*************************************************************************
* G__OP2_mulassign_dd()
*************************************************************************/
void G__OP2_mulassign_dd(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.d *= bufm1->obj.d;
  *(double*)bufm2->ref=bufm2->obj.d;
}
/*************************************************************************
* G__OP2_divassign_dd()
*************************************************************************/
void G__OP2_divassign_dd(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  if(0==bufm1->obj.d) {
    G__genericerror("Error: operator '/' divided by zero");
    return;
  }
  bufm2->obj.d /= bufm1->obj.d;
  *(double*)bufm2->ref=bufm2->obj.d;
}

/****************************************************************
* G__OP2_OPTIMIZED_FD
****************************************************************/

/*************************************************************************
* G__OP2_addassign_fd()
*************************************************************************/
void G__OP2_addassign_fd(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.d += bufm1->obj.d;
  *(float*)bufm2->ref=(float)bufm2->obj.d;
}
/*************************************************************************
* G__OP2_subassign_fd()
*************************************************************************/
void G__OP2_subassign_fd(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.d -= bufm1->obj.d;
  *(float*)bufm2->ref=(float)bufm2->obj.d;
}
/*************************************************************************
* G__OP2_mulassign_fd()
*************************************************************************/
void G__OP2_mulassign_fd(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.d *= bufm1->obj.d;
  *(float*)bufm2->ref=(float)bufm2->obj.d;
}
/*************************************************************************
* G__OP2_divassign_fd()
*************************************************************************/
void G__OP2_divassign_fd(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  if(0==bufm1->obj.d) {
    G__genericerror("Error: operator '/' divided by zero");
    return;
  }
  bufm2->obj.d /= bufm1->obj.d;
  *(float*)bufm2->ref=(float)bufm2->obj.d;
}

#endif

#ifndef G__OLDIMPLEMENTATION2129
/*************************************************************************
* G__OP2_addvoidptr()
*************************************************************************/
void G__OP2_addvoidptr(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  bufm2->obj.i += bufm1->obj.i;
}
#endif

/****************************************************************
* G__OP2_OPTIMIZED
****************************************************************/

/*************************************************************************
* G__OP2_plus()
*************************************************************************/
void G__OP2_plus(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
#ifndef G__OLDIMPLEMENTATION2189
  if('q'==bufm2->type || 'q'==bufm1->type) {
    bufm2->obj.ld=G__Longdouble(*bufm2)+G__Longdouble(*bufm1);
    bufm2->type='q';
  }
  else if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.ll=G__Longlong(*bufm2)+G__Longlong(*bufm1);
    bufm2->type='n';
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.ull=G__ULonglong(*bufm2)+G__ULonglong(*bufm1);
    bufm2->type='m';
  }
  else 
#endif
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
      bufm2->obj.d = bufm2->obj.d + bufm1->obj.d;
    }
#ifndef G__OLDIMPLEMENTATION1494
    else if(G__isunsignedM(bufm1)) {
      bufm2->obj.d = bufm2->obj.d + (double)bufm1->obj.ulo;
    }
#endif
    else {
      bufm2->obj.d = bufm2->obj.d + (double)bufm1->obj.i;
    }
    bufm2->type = 'd';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  else if(G__isdoubleM(bufm1)) {
#ifndef G__OLDIMPLEMENTATION1494
    if(G__isunsignedM(bufm2)) 
      bufm2->obj.d = (double)bufm2->obj.ulo + bufm1->obj.d;
    else
      bufm2->obj.d = (double)bufm2->obj.i + bufm1->obj.d;
#else
    bufm2->obj.d = (double)bufm2->obj.i + bufm1->obj.d;
#endif
    bufm2->type = 'd';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  else if(isupper(bufm2->type)) {
    bufm2->obj.i = bufm2->obj.i + bufm1->obj.i*G__sizeof(bufm2);
  }
  else if(isupper(bufm1->type)) {
#ifndef G__OLDIMPLEMENTATION859
    bufm2->obj.reftype.reftype = bufm1->obj.reftype.reftype;
#endif
    bufm2->obj.i = bufm2->obj.i*G__sizeof(bufm1) + bufm1->obj.i;
    /* bufm2->obj.i=(bufm2->obj.i-bufm1->obj.i)/G__sizeof(bufm2); */
    bufm2->type = bufm1->type;
    bufm2->tagnum = bufm1->tagnum;
    bufm2->typenum = bufm1->typenum;
  }
#ifndef G__OLDIMPLEMENTATION1494
  else if(G__isunsignedM(bufm1)) {
    if(G__isunsignedM(bufm2)) 
      bufm2->obj.ulo = bufm2->obj.ulo + bufm1->obj.ulo;
    else
      bufm2->obj.ulo = bufm2->obj.i + bufm1->obj.ulo;
    bufm2->type = 'h';
    bufm2->tagnum = bufm2->typenum = -1;
  }
#endif
  else {
    bufm2->obj.i = bufm2->obj.i + bufm1->obj.i;
    bufm2->type = 'i';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  bufm2->ref = 0;
}

/*************************************************************************
* G__OP2_minus()
*************************************************************************/
void G__OP2_minus(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
#ifndef G__OLDIMPLEMENTATION2189
  if('q'==bufm2->type || 'q'==bufm1->type) {
    bufm2->obj.ld=G__Longdouble(*bufm2)-G__Longdouble(*bufm1);
    bufm2->type='q';
  }
  else if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.ll=G__Longlong(*bufm2)-G__Longlong(*bufm1);
    bufm2->type='n';
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.ull=G__ULonglong(*bufm2)-G__ULonglong(*bufm1);
    bufm2->type='m';
  }
  else 
#endif
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
      bufm2->obj.d = bufm2->obj.d - bufm1->obj.d;
    }
#ifndef G__OLDIMPLEMENTATION1494
    else if(G__isunsignedM(bufm1)) {
      bufm2->obj.d = bufm2->obj.d - (double)bufm1->obj.ulo;
    }
#endif
    else {
      bufm2->obj.d = bufm2->obj.d - (double)bufm1->obj.i;
    }
    bufm2->type = 'd';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  else if(G__isdoubleM(bufm1)) {
#ifndef G__OLDIMPLEMENTATION1494
    if(G__isunsignedM(bufm2)) 
      bufm2->obj.d = (double)bufm2->obj.ulo - bufm1->obj.d;
    else
      bufm2->obj.d = (double)bufm2->obj.i - bufm1->obj.d;
#else
    bufm2->obj.d = (double)bufm2->obj.i - bufm1->obj.d;
#endif
    bufm2->type = 'd';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  else if(isupper(bufm2->type)) {
    if(isupper(bufm1->type)) {
      bufm2->obj.i=(bufm2->obj.i-bufm1->obj.i)/G__sizeof(bufm2);
      bufm2->type = 'i';
      bufm2->tagnum = bufm2->typenum = -1;
    }
    else {
      bufm2->obj.i=bufm2->obj.i-bufm1->obj.i*G__sizeof(bufm2);
    }
  }
  else if(isupper(bufm1->type)) {
#ifndef G__OLDIMPLEMENTATION859
    bufm2->obj.reftype.reftype = bufm1->obj.reftype.reftype;
#endif
    bufm2->obj.i =bufm2->obj.i*G__sizeof(bufm2) -bufm1->obj.i;
    bufm2->type = bufm1->type;
    bufm2->tagnum = bufm1->tagnum;
    bufm2->typenum = bufm1->typenum;
  }
#ifndef G__OLDIMPLEMENTATION1494
  else if(G__isunsignedM(bufm1)) {
    if(G__isunsignedM(bufm2)) 
      bufm2->obj.ulo = bufm2->obj.ulo - bufm1->obj.ulo;
    else
      bufm2->obj.ulo = bufm2->obj.i - bufm1->obj.ulo;
    bufm2->type = 'h';
    bufm2->tagnum = bufm2->typenum = -1;
  }
#endif
  else {
    bufm2->obj.i = bufm2->obj.i - bufm1->obj.i;
    bufm2->type = 'i';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  bufm2->ref = 0;
}

/*************************************************************************
* G__OP2_multiply()
*************************************************************************/
void G__OP2_multiply(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
#ifndef G__OLDIMPLEMENTATION2189
  if('q'==bufm2->type || 'q'==bufm1->type) {
    bufm2->obj.ld=G__Longdouble(*bufm2)*G__Longdouble(*bufm1);
    bufm2->type='q';
  }
  else if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.ll=G__Longlong(*bufm2)*G__Longlong(*bufm1);
    bufm2->type='n';
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.ull=G__ULonglong(*bufm2)*G__ULonglong(*bufm1);
    bufm2->type='m';
  }
  else 
#endif
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
      bufm2->obj.d = bufm2->obj.d * bufm1->obj.d;
    }
#ifndef G__OLDIMPLEMENTATION1494
    else if(G__isunsignedM(bufm1)) {
      bufm2->obj.d = bufm2->obj.d * (double)bufm1->obj.ulo;
    }
#endif
    else {
      bufm2->obj.d = bufm2->obj.d * (double)bufm1->obj.i;
    }
    bufm2->type = 'd';
  }
  else if(G__isdoubleM(bufm1)) {
#ifndef G__OLDIMPLEMENTATION1494
    if(G__isunsignedM(bufm2)) 
      bufm2->obj.d = (double)bufm2->obj.ulo * bufm1->obj.d;
    else
      bufm2->obj.d = (double)bufm2->obj.i * bufm1->obj.d;
#else
    bufm2->obj.d = (double)bufm2->obj.i * bufm1->obj.d;
#endif
    bufm2->type = 'd';
  }
#ifndef G__OLDIMPLEMENTATION1494
  else if(G__isunsignedM(bufm1)) {
    if(G__isunsignedM(bufm2)) 
      bufm2->obj.ulo = bufm2->obj.ulo * bufm1->obj.ulo;
    else
      bufm2->obj.ulo = bufm2->obj.i * bufm1->obj.ulo;
    bufm2->type = 'h';
    bufm2->tagnum = bufm2->typenum = -1;
  }
#endif
  else {
    bufm2->obj.i = bufm2->obj.i * bufm1->obj.i;
    bufm2->type = 'i';
  }
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__OP2_modulus()
*************************************************************************/
void G__OP2_modulus(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
#ifndef G__OLDIMPLEMENTATION2189
  if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.ll=G__Longlong(*bufm2)%G__Longlong(*bufm1);
    bufm2->type='n';
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.ull=G__ULonglong(*bufm2)%G__ULonglong(*bufm1);
    bufm2->type='m';
  }
  else 
#endif
#ifdef G__TUNEUP_W_SECURITY
  if(0==bufm1->obj.i) {
    G__genericerror("Error: operator '%%' divided by zero");
    return;
  }
#endif
#ifndef G__OLDIMPLEMENTATION1494
  if(G__isunsignedM(bufm1)) {
    if(G__isunsignedM(bufm2)) 
      bufm2->obj.ulo = bufm2->obj.ulo % bufm1->obj.ulo;
    else
      bufm2->obj.ulo = bufm2->obj.i % bufm1->obj.ulo;
    bufm2->type = 'h';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  else if(G__isunsignedM(bufm2)) {
    bufm2->obj.ulo = bufm2->obj.ulo % bufm1->obj.i;
    bufm2->type = 'h';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  else {
    bufm2->obj.i = bufm2->obj.i % bufm1->obj.i;
    bufm2->type = 'i';
  }
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
#else
  bufm2->obj.i = bufm2->obj.i % bufm1->obj.i;
  bufm2->type = 'i';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
#endif
}

/*************************************************************************
* G__OP2_divide()
*************************************************************************/
void G__OP2_divide(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
#ifndef G__OLDIMPLEMENTATION2189
  if('q'==bufm2->type || 'q'==bufm1->type) {
    bufm2->obj.ld=G__Longdouble(*bufm2)/G__Longdouble(*bufm1);
    bufm2->type='q';
  }
  else if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.ll=G__Longlong(*bufm2)/G__Longlong(*bufm1);
    bufm2->type='n';
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.ull=G__ULonglong(*bufm2)/G__ULonglong(*bufm1);
    bufm2->type='m';
  }
  else 
#endif
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
#ifdef G__TUNEUP_W_SECURITY
      if(0==bufm1->obj.d) {
	G__genericerror("Error: operator '/' divided by zero");
	return;
      }
#endif
      bufm2->obj.d = bufm2->obj.d / bufm1->obj.d;
    }
    else {
#ifdef G__TUNEUP_W_SECURITY
      if(0==bufm1->obj.i) {
	G__genericerror("Error: operator '/' divided by zero");
	return;
      }
#endif
#ifndef G__OLDIMPLEMENTATION1494
      if(G__isunsignedM(bufm1)) 
	bufm2->obj.d = bufm2->obj.d / (double)bufm1->obj.ulo;
      else
	bufm2->obj.d = bufm2->obj.d / (double)bufm1->obj.i;
#else
      bufm2->obj.d = bufm2->obj.d / (double)bufm1->obj.i;
#endif
    }
    bufm2->type = 'd';
  }
  else if(G__isdoubleM(bufm1)) {
#ifdef G__TUNEUP_W_SECURITY
    if(0==bufm1->obj.d) {
      G__genericerror("Error: operator '/' divided by zero");
      return;
    }
#endif
#ifndef G__OLDIMPLEMENTATION1494
    if(G__isunsignedM(bufm2)) 
      bufm2->obj.d = (double)bufm2->obj.ulo / bufm1->obj.d;
    else
      bufm2->obj.d = (double)bufm2->obj.i / bufm1->obj.d;
#else
    bufm2->obj.d = (double)bufm2->obj.i / bufm1->obj.d;
#endif
    bufm2->type = 'd';
  }
#ifndef G__OLDIMPLEMENTATION1494
  else if(G__isunsignedM(bufm1)) {
#ifdef G__TUNEUP_W_SECURITY
    if(0==bufm1->obj.i) {
      G__genericerror("Error: operator '/' divided by zero");
      return;
    }
#endif
    if(G__isunsignedM(bufm2)) 
      bufm2->obj.ulo = bufm2->obj.ulo / bufm1->obj.ulo;
    else
      bufm2->obj.ulo = bufm2->obj.i / bufm1->obj.ulo;
    bufm2->type = 'h';
    bufm2->tagnum = bufm2->typenum = -1;
  }
#endif
  else {
#ifdef G__TUNEUP_W_SECURITY
    if(0==bufm1->obj.i) {
      G__genericerror("Error: operator '/' divided by zero");
      return;
    }
#endif
    bufm2->obj.i = bufm2->obj.i / bufm1->obj.i;
    bufm2->type = 'i';
  }
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__OP2_logicaland()
*************************************************************************/
void G__OP2_logicaland(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
#ifndef G__OLDIMPLEMENTATION2189
  if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.i=G__Longlong(*bufm2)&&G__Longlong(*bufm1);
    bufm2->type='i';
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.i=G__ULonglong(*bufm2)&&G__ULonglong(*bufm1);
    bufm2->type='i';
  }
  else
#endif
  {
    bufm2->obj.i = bufm2->obj.i && bufm1->obj.i;
    bufm2->type = 'i';
  }
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__OP2_logicalor()
*************************************************************************/
void G__OP2_logicalor(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
#ifndef G__OLDIMPLEMENTATION2189
  if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.i=G__Longlong(*bufm2)||G__Longlong(*bufm1);
    bufm2->type='i';
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.i=G__ULonglong(*bufm2)||G__ULonglong(*bufm1);
    bufm2->type='i';
  }
  else
#endif
  {
    bufm2->obj.i = bufm2->obj.i || bufm1->obj.i;
    bufm2->type = 'i';
  }
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__CMP2_equal()
*************************************************************************/
void G__CMP2_equal(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
#ifndef G__OLDIMPLEMENTATION697
  if('U'==bufm1->type && 'U'==bufm2->type) G__publicinheritance(bufm1,bufm2);
#endif
#ifndef G__OLDIMPLEMENTATION1511
  if(G__isdoubleM(bufm2)||G__isdoubleM(bufm1))
    bufm2->obj.i = (G__doubleM(bufm2)==G__doubleM(bufm1));
  else
    bufm2->obj.i = (bufm2->obj.i == bufm1->obj.i);
#else
  if(G__doubleM(bufm2)==G__doubleM(bufm1)) bufm2->obj.i = 1;
  else                                     bufm2->obj.i = 0;
#endif
  bufm2->type='i';
  bufm2->typenum = bufm2->tagnum= -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__CMP2_notequal()
*************************************************************************/
void G__CMP2_notequal(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
#ifndef G__OLDIMPLEMENTATION697
  if('U'==bufm1->type && 'U'==bufm2->type) G__publicinheritance(bufm1,bufm2);
#endif
#ifndef G__OLDIMPLEMENTATION1511
  if(G__isdoubleM(bufm2)||G__isdoubleM(bufm1))
    bufm2->obj.i = (G__doubleM(bufm2)!=G__doubleM(bufm1));
  else
    bufm2->obj.i = (bufm2->obj.i != bufm1->obj.i);
#else
  if(G__doubleM(bufm2)!=G__doubleM(bufm1)) bufm2->obj.i = 1;
  else                                     bufm2->obj.i = 0;
#endif
  bufm2->type='i';
  bufm2->typenum = bufm2->tagnum= -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__CMP2_greaterorequal()
*************************************************************************/
void G__CMP2_greaterorequal(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  if(G__doubleM(bufm2)>=G__doubleM(bufm1)) bufm2->obj.i = 1;
  else                                     bufm2->obj.i = 0;
  bufm2->type='i';
  bufm2->typenum = bufm2->tagnum= -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__CMP2_lessorequal()
*************************************************************************/
void G__CMP2_lessorequal(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  if(G__doubleM(bufm2)<=G__doubleM(bufm1)) bufm2->obj.i = 1;
  else                                     bufm2->obj.i = 0;
  bufm2->type='i';
  bufm2->typenum = bufm2->tagnum= -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__CMP2_greater()
*************************************************************************/
void G__CMP2_greater(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  if(G__doubleM(bufm2)>G__doubleM(bufm1)) bufm2->obj.i = 1;
  else                                    bufm2->obj.i = 0;
  bufm2->type='i';
  bufm2->typenum = bufm2->tagnum= -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__CMP2_less()
*************************************************************************/
void G__CMP2_less(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
  if(G__doubleM(bufm2)<G__doubleM(bufm1)) bufm2->obj.i = 1;
  else                                    bufm2->obj.i = 0;
  bufm2->type='i';
  bufm2->typenum = bufm2->tagnum= -1;
  bufm2->ref = 0;
}

#ifndef G__OLDIMPLEMENTATION552
/*************************************************************************
* G__realassign()
*************************************************************************/
#define G__realassign(p,v,t)      \
 switch(t) {                      \
 case 'd': *(double*)p=v; break;  \
 case 'f': *(float*)p=v;  break;  \
 }
/*************************************************************************
* G__intassign()
*************************************************************************/
#ifdef G__BOOL4BYTE
#define G__intassign(p,v,t)               \
 switch(t) {                              \
 case 'i': *(int*)p=v;            break;  \
 case 's': *(short*)p=v;          break;  \
 case 'c': *(char*)p=v;           break;  \
 case 'h': *(unsigned int*)p=v;   break;  \
 case 'r': *(unsigned short*)p=v; break;  \
 case 'b': *(unsigned char*)p=v;  break;  \
 case 'k': *(unsigned long*)p=v;  break;  \
 case 'g': *(int*)p=v?1:0;/*1604*/break;  \
 default:  *(long*)p=v;           break;  \
 }
#else
#define G__intassign(p,v,t)               \
 switch(t) {                              \
 case 'i': *(int*)p=v;            break;  \
 case 's': *(short*)p=v;          break;  \
 case 'c': *(char*)p=v;           break;  \
 case 'h': *(unsigned int*)p=v;   break;  \
 case 'r': *(unsigned short*)p=v; break;  \
 case 'b': *(unsigned char*)p=v;  break;  \
 case 'k': *(unsigned long*)p=v;  break;  \
 case 'g': *(unsigned char*)p=v?1:0;/*1604*/break;  \
 default:  *(long*)p=v;           break;  \
 }
#endif
/*************************************************************************
* G__OP2_addassign()
*************************************************************************/
void G__OP2_addassign(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
#ifndef G__OLDIMPLEMENTATION2189
  if('q'==bufm2->type || 'q'==bufm1->type) {
    bufm2->obj.ld=G__Longdouble(*bufm2)+G__Longdouble(*bufm1);
    bufm2->type='q';
    *(long double*)bufm2->ref = bufm2->obj.ld;
  }
  else if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.ll=G__Longlong(*bufm2)+G__Longlong(*bufm1);
    bufm2->type='n';
    *(G__int64*)bufm2->ref = bufm2->obj.ll;
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.ull=G__ULonglong(*bufm2)+G__ULonglong(*bufm1);
    bufm2->type='m';
    *(G__uint64*)bufm2->ref = bufm2->obj.ull;
  }
  else 
#endif
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
      bufm2->obj.d += bufm1->obj.d;
    }
    else {
      bufm2->obj.d += (double)bufm1->obj.i;
    }
    G__realassign(bufm2->ref,bufm2->obj.d,bufm2->type);
  }
  else {
    if(G__isdoubleM(bufm1)) {
      bufm2->obj.i += bufm1->obj.d;
    }
    else if(isupper(bufm2->type)) {
      bufm2->obj.i += (bufm1->obj.i*G__sizeof(bufm2));
    }
    else if(isupper(bufm1->type)) {
      /* Illegal statement */
      bufm2->obj.i = bufm2->obj.i*G__sizeof(bufm1) + bufm1->obj.i;
    }
    else {
      bufm2->obj.i += bufm1->obj.i;
    }
    G__intassign(bufm2->ref,bufm2->obj.i,bufm2->type);
  }
}

/*************************************************************************
* G__OP2_subassign()
*************************************************************************/
void G__OP2_subassign(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
#ifndef G__OLDIMPLEMENTATION2189
  if('q'==bufm2->type || 'q'==bufm1->type) {
    bufm2->obj.ld=G__Longdouble(*bufm2)-G__Longdouble(*bufm1);
    bufm2->type='q';
    *(long double*)bufm2->ref = bufm2->obj.ld;
  }
  else if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.ll=G__Longlong(*bufm2)-G__Longlong(*bufm1);
    bufm2->type='n';
    *(G__int64*)bufm2->ref = bufm2->obj.ll;
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.ull=G__ULonglong(*bufm2)-G__ULonglong(*bufm1);
    bufm2->type='m';
    *(G__uint64*)bufm2->ref = bufm2->obj.ull;
  }
  else 
#endif
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
      bufm2->obj.d -= bufm1->obj.d;
    }
    else {
      bufm2->obj.d -= (double)bufm1->obj.i;
    }
    G__realassign(bufm2->ref,bufm2->obj.d,bufm2->type);
  }
  else {
    if(G__isdoubleM(bufm1)) {
      bufm2->obj.i -= bufm1->obj.d;
    }
    else if(isupper(bufm2->type)) {
      if(isupper(bufm1->type)) {
	bufm2->obj.i=(bufm2->obj.i-bufm1->obj.i)/G__sizeof(bufm2);
      }
      else {
	bufm2->obj.i=bufm2->obj.i-bufm1->obj.i*G__sizeof(bufm2);
      }
    }
    else if(isupper(bufm1->type)) {
      /* Illegal statement */
      bufm2->obj.i =bufm2->obj.i*G__sizeof(bufm2) -bufm1->obj.i;
    }
    else {
      bufm2->obj.i = bufm2->obj.i - bufm1->obj.i;
    }
    G__intassign(bufm2->ref,bufm2->obj.i,bufm2->type);
  }
}

/*************************************************************************
* G__OP2_mulassign()
*************************************************************************/
void G__OP2_mulassign(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
#ifndef G__OLDIMPLEMENTATION2189
  if('q'==bufm2->type || 'q'==bufm1->type) {
    bufm2->obj.ld=G__Longdouble(*bufm2)*G__Longdouble(*bufm1);
    bufm2->type='q';
    *(long double*)bufm2->ref = bufm2->obj.ld;
  }
  else if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.ll=G__Longlong(*bufm2)*G__Longlong(*bufm1);
    bufm2->type='n';
    *(G__int64*)bufm2->ref = bufm2->obj.ll;
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.ull=G__ULonglong(*bufm2)*G__ULonglong(*bufm1);
    bufm2->type='m';
    *(G__uint64*)bufm2->ref = bufm2->obj.ull;
  }
  else 
#endif
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
      bufm2->obj.d *= bufm1->obj.d;
    }
    else {
      bufm2->obj.d *= (double)bufm1->obj.i;
    }
    G__realassign(bufm2->ref,bufm2->obj.d,bufm2->type);
  }
  else {
    if(G__isdoubleM(bufm1)) {
      bufm2->obj.i *= bufm1->obj.d;
    }
    else {
      bufm2->obj.i *= bufm1->obj.i;
    }
    G__intassign(bufm2->ref,bufm2->obj.i,bufm2->type);
  }
}

/*************************************************************************
* G__OP2_modassign()
*************************************************************************/
void G__OP2_modassign(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
#ifndef G__OLDIMPLEMENTATION2189
  if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.ll=G__Longlong(*bufm2)%G__Longlong(*bufm1);
    bufm2->type='n';
    *(G__int64*)bufm2->ref = bufm2->obj.ll;
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.ull=G__ULonglong(*bufm2)%G__ULonglong(*bufm1);
    bufm2->type='m';
    *(G__uint64*)bufm2->ref = bufm2->obj.ull;
  }
  else 
#endif
#ifdef G__TUNEUP_W_SECURITY
  if(0==bufm1->obj.i) {
    G__genericerror("Error: operator '%%' divided by zero");
    return;
  }
#endif
#ifndef G__OLDIMPLEMENTATION1627
  if(G__isunsignedM(bufm2)) {
    if(G__isunsignedM(bufm1)) {
      bufm2->obj.ulo %= bufm1->obj.ulo;
    }
    else {
      bufm2->obj.ulo %= bufm1->obj.i;
    }
  }
  else {
    if(G__isunsignedM(bufm1)) {
      bufm2->obj.i %= bufm1->obj.ulo;
    }
    else {
      bufm2->obj.i %= bufm1->obj.i;
    }
  }
#else
  bufm2->obj.i %= bufm1->obj.i;
#endif
  G__intassign(bufm2->ref,bufm2->obj.i,bufm2->type);
}

/*************************************************************************
* G__OP2_divassign()
*************************************************************************/
void G__OP2_divassign(bufm1,bufm2)
G__value *bufm1;
G__value *bufm2;
{
#ifndef G__OLDIMPLEMENTATION2189
  if('q'==bufm2->type || 'q'==bufm1->type) {
    bufm2->obj.ld=G__Longdouble(*bufm2)/G__Longdouble(*bufm1);
    bufm2->type='q';
    *(long double*)bufm2->ref = bufm2->obj.ld;
  }
  else if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.ll=G__Longlong(*bufm2)/G__Longlong(*bufm1);
    bufm2->type='n';
    *(G__int64*)bufm2->ref = bufm2->obj.ll;
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.ull=G__ULonglong(*bufm2)/G__ULonglong(*bufm1);
    bufm2->type='m';
    *(G__uint64*)bufm2->ref = bufm2->obj.ull;
  }
  else 
#endif
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
#ifdef G__TUNEUP_W_SECURITY
      if(0==bufm1->obj.d) {
	G__genericerror("Error: operator '/' divided by zero");
	return;
      }
#endif
      bufm2->obj.d /= bufm1->obj.d;
    }
    else {
#ifdef G__TUNEUP_W_SECURITY
      if(0==bufm1->obj.i) {
	G__genericerror("Error: operator '/' divided by zero");
	return;
      }
#endif
      bufm2->obj.d /= (double)bufm1->obj.i;
    }
    G__realassign(bufm2->ref,bufm2->obj.d,bufm2->type);
  }
  else {
    if(G__isdoubleM(bufm1)) {
#ifdef G__TUNEUP_W_SECURITY
      if(0==bufm1->obj.d) {
	G__genericerror("Error: operator '/' divided by zero");
	return;
      }
#endif
      bufm2->obj.i /= bufm1->obj.d;
    }
    else {
#ifdef G__TUNEUP_W_SECURITY
      if(0==bufm1->obj.i) {
	G__genericerror("Error: operator '/' divided by zero");
	return;
      }
#endif
#ifndef G__OLDIMPLEMENTATION1627
      if(G__isunsignedM(bufm2)) {
	if(G__isunsignedM(bufm1)) {
	  bufm2->obj.ulo /= bufm1->obj.ulo;
	}
	else {
	  bufm2->obj.ulo /= bufm1->obj.i;
	}
      }
      else {
	if(G__isunsignedM(bufm1)) {
	  bufm2->obj.i /= bufm1->obj.ulo;
	}
	else {
	  bufm2->obj.i /= bufm1->obj.i;
	}
      }
#else
      bufm2->obj.i /= bufm1->obj.i;
#endif
    }
    G__intassign(bufm2->ref,bufm2->obj.i,bufm2->type);
  }
}
#endif


/****************************************************************
* G__OP1_OPTIMIZED
****************************************************************/

#ifndef G__OLDIMPLEMENTATION578
/****************************************************************
* G__OP1_postfixinc_i()
****************************************************************/
void G__OP1_postfixinc_i(pbuf)
G__value *pbuf;
{
  *(int*)pbuf->ref = (int)pbuf->obj.i+1;
}
/****************************************************************
* G__OP1_postfixdec_i()
****************************************************************/
void G__OP1_postfixdec_i(pbuf)
G__value *pbuf;
{
  *(int*)pbuf->ref = (int)pbuf->obj.i-1;
}
/****************************************************************
* G__OP1_prefixinc_i()
****************************************************************/
void G__OP1_prefixinc_i(pbuf)
G__value *pbuf;
{
  *(int*)pbuf->ref = (int)(++pbuf->obj.i);
}
/****************************************************************
* G__OP1_prefixdec_i()
****************************************************************/
void G__OP1_prefixdec_i(pbuf)
G__value *pbuf;
{
  *(int*)pbuf->ref = (int)(--pbuf->obj.i);
}

/****************************************************************
* G__OP1_postfixinc_d()
****************************************************************/
void G__OP1_postfixinc_d(pbuf)
G__value *pbuf;
{
  *(double*)pbuf->ref = (double)pbuf->obj.d+1.0;
}
/****************************************************************
* G__OP1_postfixdec_d()
****************************************************************/
void G__OP1_postfixdec_d(pbuf)
G__value *pbuf;
{
  *(double*)pbuf->ref = (double)pbuf->obj.d-1.0;
}
/****************************************************************
* G__OP1_prefixinc_d()
****************************************************************/
void G__OP1_prefixinc_d(pbuf)
G__value *pbuf;
{
  *(double*)pbuf->ref = (double)(++pbuf->obj.d);
}
/****************************************************************
* G__OP1_prefixdec_d()
****************************************************************/
void G__OP1_prefixdec_d(pbuf)
G__value *pbuf;
{
  *(double*)pbuf->ref = (double)(--pbuf->obj.d);
}

#if G__NEVER /* following change rather slowed down */
/****************************************************************
* G__OP1_postfixinc_l()
****************************************************************/
void G__OP1_postfixinc_l(pbuf)
G__value *pbuf;
{
  *(long*)pbuf->ref = (long)pbuf->obj.i+1;
}
/****************************************************************
* G__OP1_postfixdec_l()
****************************************************************/
void G__OP1_postfixdec_l(pbuf)
G__value *pbuf;
{
  *(long*)pbuf->ref = (long)pbuf->obj.i-1;
}
/****************************************************************
* G__OP1_prefixinc_l()
****************************************************************/
void G__OP1_prefixinc_l(pbuf)
G__value *pbuf;
{
  *(long*)pbuf->ref = (long)(++pbuf->obj.i);
}
/****************************************************************
* G__OP1_prefixdec_l()
****************************************************************/
void G__OP1_prefixdec_l(pbuf)
G__value *pbuf;
{
  *(long*)pbuf->ref = (long)(--pbuf->obj.i);
}

/****************************************************************
* G__OP1_postfixinc_s()
****************************************************************/
void G__OP1_postfixinc_s(pbuf)
G__value *pbuf;
{
  *(short*)pbuf->ref = (short)pbuf->obj.i+1;
}
/****************************************************************
* G__OP1_postfixdec_s()
****************************************************************/
void G__OP1_postfixdec_s(pbuf)
G__value *pbuf;
{
  *(short*)pbuf->ref = (short)pbuf->obj.i-1;
}
/****************************************************************
* G__OP1_prefixinc_s()
****************************************************************/
void G__OP1_prefixinc_s(pbuf)
G__value *pbuf;
{
  *(short*)pbuf->ref = (short)(++pbuf->obj.i);
}
/****************************************************************
* G__OP1_prefixdec_s()
****************************************************************/
void G__OP1_prefixdec_s(pbuf)
G__value *pbuf;
{
  *(short*)pbuf->ref = (short)(--pbuf->obj.i);
}

/****************************************************************
* G__OP1_postfixinc_h()
****************************************************************/
void G__OP1_postfixinc_h(pbuf)
G__value *pbuf;
{
  *(unsigned int*)pbuf->ref = (unsigned int)pbuf->obj.i+1;
}
/****************************************************************
* G__OP1_postfixdec_h()
****************************************************************/
void G__OP1_postfixdec_h(pbuf)
G__value *pbuf;
{
  *(unsigned int*)pbuf->ref = (unsigned int)pbuf->obj.i-1;
}
/****************************************************************
* G__OP1_prefixinc_h()
****************************************************************/
void G__OP1_prefixinc_h(pbuf)
G__value *pbuf;
{
  *(unsigned int*)pbuf->ref = (unsigned int)(++pbuf->obj.i);
}
/****************************************************************
* G__OP1_prefixdec_h()
****************************************************************/
void G__OP1_prefixdec_h(pbuf)
G__value *pbuf;
{
  *(unsigned int*)pbuf->ref = (unsigned int)(--pbuf->obj.i);
}

/****************************************************************
* G__OP1_postfixinc_k()
****************************************************************/
void G__OP1_postfixinc_k(pbuf)
G__value *pbuf;
{
  *(unsigned long*)pbuf->ref = (unsigned long)pbuf->obj.i+1;
}
/****************************************************************
* G__OP1_postfixdec_k()
****************************************************************/
void G__OP1_postfixdec_k(pbuf)
G__value *pbuf;
{
  *(unsigned long*)pbuf->ref = (unsigned long)pbuf->obj.i-1;
}
/****************************************************************
* G__OP1_prefixinc_k()
****************************************************************/
void G__OP1_prefixinc_k(pbuf)
G__value *pbuf;
{
  *(unsigned long*)pbuf->ref = (unsigned long)(++pbuf->obj.i);
}
/****************************************************************
* G__OP1_prefixdec_k()
****************************************************************/
void G__OP1_prefixdec_k(pbuf)
G__value *pbuf;
{
  *(unsigned long*)pbuf->ref = (unsigned long)(--pbuf->obj.i);
}

/****************************************************************
* G__OP1_postfixinc_r()
****************************************************************/
void G__OP1_postfixinc_r(pbuf)
G__value *pbuf;
{
  *(unsigned short*)pbuf->ref = (unsigned short)pbuf->obj.i+1;
}
/****************************************************************
* G__OP1_postfixdec_r()
****************************************************************/
void G__OP1_postfixdec_r(pbuf)
G__value *pbuf;
{
  *(unsigned short*)pbuf->ref = (unsigned short)pbuf->obj.i-1;
}
/****************************************************************
* G__OP1_prefixinc_r()
****************************************************************/
void G__OP1_prefixinc_r(pbuf)
G__value *pbuf;
{
  *(unsigned short*)pbuf->ref = (unsigned short)(++pbuf->obj.i);
}
/****************************************************************
* G__OP1_prefixdec_r()
****************************************************************/
void G__OP1_prefixdec_r(pbuf)
G__value *pbuf;
{
  *(unsigned short*)pbuf->ref = (unsigned short)(--pbuf->obj.i);
}


/****************************************************************
* G__OP1_postfixinc_f()
****************************************************************/
void G__OP1_postfixinc_f(pbuf)
G__value *pbuf;
{
  *(float*)pbuf->ref = (float)pbuf->obj.d+1.0;
}
/****************************************************************
* G__OP1_postfixdec_f()
****************************************************************/
void G__OP1_postfixdec_f(pbuf)
G__value *pbuf;
{
  *(float*)pbuf->ref = (float)pbuf->obj.d-1.0;
}
/****************************************************************
* G__OP1_prefixinc_f()
****************************************************************/
void G__OP1_prefixinc_f(pbuf)
G__value *pbuf;
{
  *(float*)pbuf->ref = (float)(++pbuf->obj.d);
}
/****************************************************************
* G__OP1_prefixdec_f()
****************************************************************/
void G__OP1_prefixdec_f(pbuf)
G__value *pbuf;
{
  *(float*)pbuf->ref = (float)(--pbuf->obj.d);
}
#endif

#endif

/****************************************************************
* G__OP1_postfixinc()
****************************************************************/
void G__OP1_postfixinc(pbuf)
G__value *pbuf;
{
  long iorig;
  double dorig;
  switch(pbuf->type) {
  case 'd':
  case 'f':
    dorig = pbuf->obj.d;
    G__doubleassignbyref(pbuf,dorig+1.0);
    pbuf->obj.d=dorig;
    break;
  default:
    iorig = pbuf->obj.i;
    if(isupper(pbuf->type)) {
      G__intassignbyref(pbuf,iorig+G__sizeof(pbuf));
      pbuf->obj.i = iorig;
    }
    else {
      G__intassignbyref(pbuf,iorig+1);
      pbuf->obj.i = iorig;
    }
  }
}
/****************************************************************
* G__OP1_postfixdec()
****************************************************************/
void G__OP1_postfixdec(pbuf)
G__value *pbuf;
{
  long iorig;
  double dorig;
  switch(pbuf->type) {
  case 'd':
  case 'f':
    dorig = pbuf->obj.d;
    G__doubleassignbyref(pbuf,dorig-1.0);
    pbuf->obj.d=dorig;
    break;
  default:
    iorig = pbuf->obj.i;
    if(isupper(pbuf->type)) {
      G__intassignbyref(pbuf,iorig-G__sizeof(pbuf));
      pbuf->obj.i = iorig;
    }
    else {
      G__intassignbyref(pbuf,iorig-1);
      pbuf->obj.i = iorig;
    }
  }
}
/****************************************************************
* G__OP1_prefixinc()
****************************************************************/
void G__OP1_prefixinc(pbuf)
G__value *pbuf;
{
  switch(pbuf->type) {
  case 'd':
  case 'f':
    G__doubleassignbyref(pbuf,pbuf->obj.d+1.0);
    break;
  default:
    if(isupper(pbuf->type)) {
      G__intassignbyref(pbuf,pbuf->obj.i+G__sizeof(pbuf));
    }
    else {
      G__intassignbyref(pbuf,pbuf->obj.i+1);
    }
  }
}
/****************************************************************
* G__OP1_prefixdec()
****************************************************************/
void G__OP1_prefixdec(pbuf)
G__value *pbuf;
{
  switch(pbuf->type) {
  case 'd':
  case 'f':
    G__doubleassignbyref(pbuf,pbuf->obj.d-1.0);
    break;
  default:
    if(isupper(pbuf->type)) {
      G__intassignbyref(pbuf,pbuf->obj.i-G__sizeof(pbuf));
    }
    else {
      G__intassignbyref(pbuf,pbuf->obj.i-1);
    }
  }
}
/****************************************************************
* G__OP1_minus()
****************************************************************/
void G__OP1_minus(pbuf)
G__value *pbuf;
{
  pbuf->ref = 0;
  switch(pbuf->type) {
  case 'd':
  case 'f':
    pbuf->obj.d *= -1.0;
    break;
  default:
    if(isupper(pbuf->type)) {
      G__genericerror("Error: Illegal pointer operation unary -");
    }
    else {
      pbuf->obj.i *= -1;
    }
  }
}




/*************************************************************************
**************************************************************************
* Optimization level 1 function
**************************************************************************
*************************************************************************/

#ifndef G__OLDIMPLEMENTATION1164
/******************************************************************
* G__suspendbytecode()
******************************************************************/
void G__suspendbytecode() 
{
  if(G__asm_dbg && G__asm_noverflow) {
    if(G__dispmsg>=G__DISPNOTE) {
      G__fprinterr(G__serr,"Note: Bytecode compiler suspended(off) and resumed(on)");
      G__printlinenum();
    }
  }
  G__asm_noverflow=0;
}
/******************************************************************
* G__resetbytecode()
******************************************************************/
void G__resetbytecode() 
{
  if(G__asm_dbg && G__asm_noverflow) {
    if(G__dispmsg>=G__DISPNOTE) {
      G__fprinterr(G__serr,"Note: Bytecode compiler reset (off)");
      G__printlinenum();
    }
  }
  G__asm_noverflow=0;
}
#endif

#ifndef G__OLDIMPLEMENTATION988
/******************************************************************
* G__abortbytecode()
******************************************************************/
void G__abortbytecode() 
{
  if(G__asm_dbg && G__asm_noverflow) {
#ifndef G__OLDIMPLEMENTATION1164
    if(G__dispmsg>=G__DISPNOTE) {
      if(0==G__xrefflag) 
	G__fprinterr(G__serr,"Note: Bytecode compiler stops at this line. Enclosing loop or function may be slow %d"
		     ,G__asm_noverflow);
      else
	G__fprinterr(G__serr,"Note: Bytecode limitation encountered but compiler continuers for Local variable cross referencing");
#else
      G__fprinterr(G__serr,"Note: Bytecode compiler stops at this line. Enclosing loop or function may be slow %d"
		   ,G__asm_noverflow);
#endif
      G__printlinenum();
    }
  }
#ifndef G__OLDIMPLEMENTATION1164
  if(0==G__xrefflag) G__asm_noverflow=0;
#else
  G__asm_noverflow=0;
#endif
}
#endif

/****************************************************************
* G__inc_cp_asm(cp_inc,dt_dec)
*
*
*  Increment program counter(G__asm_cp) and decrement stack pointer
* (G__asm_dt) at compile time.
*  If Quasi-Assembly-Code or constant data exceeded instruction
* and data buffer, G__asm_noverflow is reset and compilation is
* void.
*
****************************************************************/
int G__inc_cp_asm(cp_inc,dt_dec)
int cp_inc,dt_dec;
{
#ifndef G__OLDIMPLEMENTATION1164
  if(0==G__xrefflag) {
    G__asm_cp+=cp_inc;
    G__asm_dt-=dt_dec;
  }
#else
  G__asm_cp+=cp_inc;
  G__asm_dt-=dt_dec;
#endif

#ifndef G__OLDIMPLEMENTATION2116
  if(G__asm_instsize && G__asm_cp>G__asm_instsize-8) {
    G__asm_instsize += 0x100;
    void *p = realloc((void*)G__asm_stack,sizeof(long)*G__asm_instsize);
    if(!p) G__genericerror("Error: memory exhausted for bytecode instruction buffer\n");
    G__asm_inst = (long*)p;
  }
  else if(!G__asm_instsize && G__asm_cp>G__MAXINST-8) {
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"Warning: loop compile instruction overflow");
      G__printlinenum();
    }
    G__abortbytecode();
  }
#else /* 2116 */
  if(G__asm_cp>G__MAXINST-8) {
#ifndef G__OLDIMPLEMENTATION841
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"Warning: loop compile instruction overflow");
      G__printlinenum();
    }
#endif
    G__abortbytecode();
  }
#endif /* 2116 */

  if(G__asm_dt<30) {
#ifndef G__OLDIMPLEMENTATION841
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"Warning: loop compile data overflow");
      G__printlinenum();
    }
#endif
    G__abortbytecode();
  }
  return(0);
}

/****************************************************************
* G__clear_asm()
*
* Called by
*    G__exec_do()
*    G__exec_while()
*    G__exec_while()
*    G__exec_for()
*    G__exec_for()
*
*  Reset instruction and data buffer.
* This function is called at the beginning of compilation.
*
****************************************************************/
int G__clear_asm()
{
  G__asm_cp=0;
  G__asm_dt=G__MAXSTACK-1;
  G__asm_name_p=0;
#ifdef G__OLDIMPLEMENTATION937
  if(G__ASM_FUNC_NOP==G__asm_wholefunction) G__no_exec_compile = 0;
#endif
  G__asm_cond_cp = -1; /* avoid wrong optimization */
  return(0);
}

/******************************************************************
* G__asm_clear()
*
*
******************************************************************/
int G__asm_clear()
{
#ifndef G__OLDIMPLEMENTATION1570
  if(G__asm_clear_mask) return(0);
#endif
#ifdef G__ASM_DBG
  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: CL  FILE:%s LINE:%d\n" ,G__asm_cp
			 ,G__ifile.name ,G__ifile.line_number);
#endif

#ifndef G__OLDIMPLEMENTATION2134
  /* Issue, value of G__CL must be unique. Otherwise, following optimization
   * causes problem. There is similar issue, but the biggest risk is here. */
  if(G__asm_cp>=2 && G__CL==G__asm_inst[G__asm_cp-2]
     && (G__asm_inst[G__asm_cp-1]&0xffff0000)==0x7fff0000) 
    G__inc_cp_asm(-2,0);
  G__asm_inst[G__asm_cp]=G__CL;
  G__asm_inst[G__asm_cp+1]= (G__ifile.line_number&G__CL_LINEMASK) 
                    + (G__ifile.filenum&G__CL_FILEMASK)*G__CL_FILESHIFT;
  G__inc_cp_asm(2,0);
#else /* 2134 */
  if(G__asm_cp<2 || G__CL!=G__asm_inst[G__asm_cp-2]) {
    G__asm_inst[G__asm_cp]=G__CL;
#ifndef G__OLDIMPLEMENTATION2132
    G__asm_inst[G__asm_cp+1]= (G__ifile.line_number&G__CL_LINEMASK) 
                    + (G__ifile.filenum&G__CL_FILEMASK)*G__CL_FILESHIFT;
#else
    G__asm_inst[G__asm_cp+1]=G__ifile.line_number;
#endif
    G__inc_cp_asm(2,0);
  }
#endif /* 2134 */
  return(0);
}

#endif /* G__ASM */


#ifdef G__ASM
/**************************************************************************
* G__asm_putint()
**************************************************************************/
int G__asm_putint(i)
int i;
{
#ifdef G__ASM_DBG
  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD %d from %x\n",G__asm_cp,i,G__asm_dt);
#endif
  G__asm_inst[G__asm_cp]=G__LD;
  G__asm_inst[G__asm_cp+1]=G__asm_dt;
  G__letint(&G__asm_stack[G__asm_dt],'i',(long)i);;
  G__inc_cp_asm(2,1);
  return(0);
}
#endif

/**************************************************************************
* G__value G__getreserved()
**************************************************************************/
G__value G__getreserved(item ,ptr,ppdict)
char *item;
void **ptr;
void **ppdict;
{
  G__value buf;
  int i;

#ifndef G__OLDIMPLEMENTATION1911
  if(0 && ptr && ppdict) return(G__null);
#endif

  G__abortbytecode();

  if(strcmp(item,"LINE")==0 || strcmp(item,"_LINE__")==0) {
    i = G__RSVD_LINE;
#ifdef G__ASM
    if(G__asm_noverflow) G__asm_putint(i);
#endif
  }
  else if(strcmp(item,"FILE")==0 || strcmp(item,"_FILE__")==0) {
    i = G__RSVD_FILE;
#ifdef G__ASM
    if(G__asm_noverflow) G__asm_putint(i);
#endif
  }
#ifndef G__OLDIMPLEMENTATION1234
  else if(strcmp(item,"_DATE__")==0) {
    i = G__RSVD_DATE;
#ifdef G__ASM
    if(G__asm_noverflow) G__asm_putint(i);
#endif
  }
  else if(strcmp(item,"_TIME__")==0) {
    i = G__RSVD_TIME;
#ifdef G__ASM
    if(G__asm_noverflow) G__asm_putint(i);
#endif
  }
#endif
  else if(strcmp(item,"#")==0) {
    i = G__RSVD_ARG;
#ifdef G__ASM
    if(G__asm_noverflow) G__asm_putint(i);
#endif
  }
  else if(isdigit(item[0])) {
    i=atoi(item);
#ifdef G__ASM
    if(G__asm_noverflow) G__asm_putint(i);
#endif
  }
  else {
#ifndef G__OLDIMPLEMENTATION715
#ifndef G__OLDIMPLEMENTATION1173
      i = 0;
#else
      i = -1;
#endif
      buf = G__null;
#else
    buf = G__getexpr(item);
    if(buf.type) {
      i=G__int(buf);
    }
    else {
      i = -1;
      buf = G__null;
    }
#endif
  }

#ifndef G__OLDIMPLEMENTATION1173
  if(i) {
#else
  if(-1 != i) {
#endif
    buf = G__getrsvd(i);
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: GETRSVD $%s\n" ,G__asm_cp,item);
#endif
      /* GETRSVD */
      G__asm_inst[G__asm_cp]=G__GETRSVD;
      G__inc_cp_asm(1,0);
    }
#endif
  }
  return(buf);
}

#ifndef G__OLDIMPLEMENTATION1234
/**************************************************************************
* G__get__tm__()
*
*  returns 'Sun Nov 28 21:40:32 1999\n' in buf
**************************************************************************/
void G__get__tm__(buf)
char *buf;
{
  time_t t = time(0);
  sprintf(buf,"%s",ctime(&t));
}
/**************************************************************************
* G__get__date__()
**************************************************************************/
char* G__get__date__() 
{
  int i=0,j=0;
  char buf[80];
  static char result[80];
  G__get__tm__(buf);
  while(buf[i] && !isspace(buf[i])) ++i; /* skip 'Sun' */
  while(buf[i] && isspace(buf[i])) ++i;
  while(buf[i] && !isspace(buf[i])) result[j++] = buf[i++]; /* copy 'Nov' */
  while(buf[i] && isspace(buf[i])) result[j++] = buf[i++];
  while(buf[i] && !isspace(buf[i])) result[j++] = buf[i++]; /* copy '28' */
  while(buf[i] && isspace(buf[i])) result[j++] = buf[i++];
  while(buf[i] && !isspace(buf[i])) ++i; /* skip '21:41:10' */
  while(buf[i] && isspace(buf[i])) ++i;
  while(buf[i] && !isspace(buf[i])) result[j++] = buf[i++]; /* copy '1999' */
  result[j] = 0;
  return(result);
}
/**************************************************************************
* G__get__time__()
**************************************************************************/
char* G__get__time__() 
{
  int i=0,j=0;
  char buf[80];
  static char result[80];
  G__get__tm__(buf);
  while(buf[i] && !isspace(buf[i])) ++i; /* skip 'Sun' */
  while(buf[i] && isspace(buf[i])) ++i;
  while(buf[i] && !isspace(buf[i])) ++i; /* skip 'Nov' */
  while(buf[i] && isspace(buf[i])) ++i;
  while(buf[i] && !isspace(buf[i])) ++i; /* skip '28' */
  while(buf[i] && isspace(buf[i])) ++i;
  while(buf[i] && !isspace(buf[i])) result[j++] = buf[i++];/*copy '21:41:10'*/
  result[j] = 0;
  return(result);
}
#endif

/**************************************************************************
* G__value G__getrsvd()
**************************************************************************/
G__value G__getrsvd(i)
int i;
{
  G__value buf;

  buf.tagnum = -1;
  buf.typenum = -1;
#ifndef G__OLDIMPLEMENTATION1495
  buf.ref = 0;
#endif

  switch(i) {
  case G__RSVD_LINE:
    G__letint(&buf,'i',(long)G__ifile.line_number);
    break;
  case G__RSVD_FILE:
#ifndef G__OLDIMPLEMENTATION1475
    if(0<=G__ifile.filenum && G__ifile.filenum<G__MAXFILE &&
       G__srcfile[G__ifile.filenum].filename) {
      G__letint(&buf,'C',(long)G__srcfile[G__ifile.filenum].filename);
    }
    else {
      G__letint(&buf,'C',(long)0);
    }
#else
    G__letint(&buf,'C',(long)G__ifile.name);
#endif
    break;
  case G__RSVD_ARG:
    G__letint(&buf,'i',(long)G__argn);
    break;
#ifndef G__OLDIMPLEMENTATION1234
  case G__RSVD_DATE:
    G__letint(&buf,'C',(long)G__get__date__());
    break;
  case G__RSVD_TIME:
    G__letint(&buf,'C',(long)G__get__time__());
    break;
#endif
  default:
    G__letint(&buf,'C',(long)G__arg[i]);
    break;
  }

	return(buf);
}


/*************************************************************************
**************************************************************************
* Optimization level 2 function
**************************************************************************
*************************************************************************/

/****************************************************************
* G__asm_gettest()
****************************************************************/
long G__asm_gettest(op,inst)
int op;
long *inst;
{
  switch(op) {
  case 'E': /* != */
    *inst = (long)G__asm_test_E;
    break;
  case 'N': /* != */
    *inst = (long)G__asm_test_N;
    break;
  case 'G': /* >= */
    *inst = (long)G__asm_test_GE;
    break;
  case 'l': /* <= */
    *inst = (long)G__asm_test_LE;
    break;
  case '<':
    *inst = (long)G__asm_test_l;
    break;
  case '>':
    *inst = (long)G__asm_test_g;
    break;
  default:
    G__fprinterr(G__serr,"Error: Loop compile optimizer, illegal conditional instruction %d(%c) FILE:%s LINE:%d\n"
	    ,op,op,G__ifile.name,G__ifile.line_number);
    break;
  }
  return(0);
}

#ifndef G__OLDIMPLEMENTATION1021
/****************************************************************
 * G__isInt
 ****************************************************************/
int G__isInt(type)
int type;
{
  switch(type) {
  case 'i':
    return(1);
  case 'l':
    if(G__LONGALLOC==G__INTALLOC) return(1);
    else return(0);
    break;
  case 's':
    if(G__SHORTALLOC==G__INTALLOC) return(1);
    else return(0);
    break;
  default:
    return(0);
  }
}
#endif

/****************************************************************
* G__asm_optimize(start)
*
* Called by
*
*  Quasi-Assembly-Code optimizer
*
****************************************************************/
int G__asm_optimize(start)
int *start;
{
  /* Issue, value of G__LD_VAR, G__LD_MVAR, G__LD, G__CMP2, G__CNDJMP must be
   * unique. Otherwise, following optimization causes problem. */
  int *pb;
  /*******************************************************
   * i<100, i<=100, i>0 i>=0 at loop header
   *
   *               before                     ----------> after
   *
   *      0      G__LD_VAR       <- check  1             JMP
   *      1      index           <- check  2  (int)      6
   *      2      paran                                   NOP
   *      3      point_level     <- check  3             NOP
   *      4      *var                     (2)            NOP
   *      5      LD              <- check  1             NOP  bydy of *b
   *      6      data_stack      <- check  2 (int)       CMPJMP
   *      7      CMP2            <- check  1             *compare()
   *      8      <,<=,>,>=,==,!=    case                 *a  var->p[]
   *      9      CNDJMP          <- check  1             *b  ptr to inst[5]
   *     10      next_pc=G__asm_cp                       next_pc=G__asm_cp
   *          .                                           .
   *     -2      JMP                                     JMP
   *     -1      next_pc                                 6
   * G__asm_cp   RTN                                     RTN
   *******************************************************/
  if((G__asm_inst[*start]==G__LD_VAR
      || (G__asm_inst[*start]==G__LD_MSTR
#ifndef G__OLDIMPLEMENTATION1333
	  && !G__asm_wholefunction
#endif
      )) 
     &&  /* 1 */
     G__asm_inst[*start+5]==G__LD      &&
     G__asm_inst[*start+7]==G__CMP2    &&
     G__asm_inst[*start+9]==G__CNDJMP  &&
#ifndef G__OLDIMPLEMENTATION1021
     G__isInt(((struct G__var_array*)G__asm_inst[*start+4])->type[G__asm_inst[*start+1]]) && /* 2 */
     G__isInt(G__asm_stack[G__asm_inst[*start+6]].type) &&
#else
     ((struct G__var_array*)G__asm_inst[*start+4])->type[G__asm_inst[*start+1]] =='i' && /* 2 */
     G__asm_stack[G__asm_inst[*start+6]].type=='i' &&
#endif
     G__asm_inst[*start+3]=='p'    /* 3 */
     ) {

#ifdef G__ASM_DBG
    if(G__asm_dbg)
      G__fprinterr(G__serr,"%3x: CMPJMP i %c %d optimized\n"
	      ,*start+6,G__asm_inst[*start+8]
	      ,G__int(G__asm_stack[G__asm_inst[*start+6]]));
#endif
    G__asm_gettest((int)G__asm_inst[*start+8]
		   ,&G__asm_inst[*start+7]);

    G__asm_inst[*start+8]=
      ((struct G__var_array*)G__asm_inst[*start+4])->p[G__asm_inst[*start+1]];
    if(G__asm_inst[*start]==G__LD_MSTR
       && G__LOCALSTATIC!=((struct G__var_array*)G__asm_inst[*start+4])->statictype[G__asm_inst[*start+1]]
       )
      G__asm_inst[*start+8] += G__store_struct_offset;

    /* long to int conversion */ /* TODO, Storing ptr to temporary stack buffer, is this Bad? */
#ifndef G__OLDIMPLEMENTATION2113
    pb = (int*)(&G__asm_inst[*start+5]);
#else
    pb = (int*)(&(G__asm_stack[G__asm_inst[*start+6]].obj.i));  /* TODO, ??? */
#endif
    *pb = G__int(G__asm_stack[G__asm_inst[*start+6]]);
    G__asm_inst[*start+9]=(long)(pb);
    G__asm_inst[*start+6]=G__CMPJMP;
    G__asm_inst[*start]=G__JMP;
    G__asm_inst[*start+1]= *start+6;
    G__asm_inst[*start+2]=G__NOP;
    G__asm_inst[*start+3]=G__NOP;
    G__asm_inst[*start+4]=G__NOP;
    G__asm_inst[*start+5]=G__NOP;

    *start += 6;
    G__asm_inst[G__asm_cp-1] = *start;
  }
  /*******************************************************
   * i<100, i<=100, i>0 i>=0 at loop header
   *
   *               before                     ----------> after
   *
   *      0      G__LD_VAR,MSTR  <- check  1             JMP
   *      1      index           <- check  2  (int)      9
   *      2      paran                                   NOP
   *      3      point_level     <- check  3             NOP
   *      4      *var                     (2)            NOP
   *      5      G__LD_VAR,MSTR  <- check  1             NOP
   *      6      index           <- check  2  (int)      NOP
   *      7      paran                                   NOP
   *      8      point_level     <- check  3             NOP
   *      9      *var                     (2)            CMPJMP
   *      10     CMP2            <- check  1             *compare()
   *      11     <,<=,>,>=,==,!=    case                 *a  var->[]
   *      12     CNDJMP          <- check  1             *b  var->[]
   *      13     next_pc=G__asm_cp                       next_pc=G__asm_pc
   *          .
   *     -2      JMP                                     JMP
   *     -1      next_pc                                 9
   * G__asm_cp   RTN                                     RTN
   *******************************************************/
  else if((G__asm_inst[*start]==G__LD_VAR||
	   (G__asm_inst[*start]==G__LD_MSTR
#ifndef G__OLDIMPLEMENTATION1333
	    && !G__asm_wholefunction
#endif
	    )) &&  /* 1 */
	  (G__asm_inst[*start+5]==G__LD_VAR||
	   (G__asm_inst[*start+5]==G__LD_MSTR
#ifndef G__OLDIMPLEMENTATION1333
	    && !G__asm_wholefunction
#endif
	   )) &&  /* 1 */
	  G__asm_inst[*start+10]==G__CMP2    &&
	  G__asm_inst[*start+12]==G__CNDJMP  &&
#ifndef G__OLDIMPLEMENTATION1021
	  G__isInt(((struct G__var_array*)G__asm_inst[*start+4])->type[G__asm_inst[*start+1]]) && /* 2 */
	  (G__isInt(((struct G__var_array*)G__asm_inst[*start+9])->type[G__asm_inst[*start+6]]) || /* 2 */
	   ((struct G__var_array*)G__asm_inst[*start+9])->type[G__asm_inst[*start+6]] =='p' ) && /* 2 */
#else
	  ((struct G__var_array*)G__asm_inst[*start+4])->type[G__asm_inst[*start+1]] =='i' && /* 2 */
	  (((struct G__var_array*)G__asm_inst[*start+9])->type[G__asm_inst[*start+6]] =='i' || /* 2 */
	   ((struct G__var_array*)G__asm_inst[*start+9])->type[G__asm_inst[*start+6]] =='p' ) && /* 2 */
#endif
	  G__asm_inst[*start+3]=='p'  &&  /* 3 */
	  G__asm_inst[*start+8]=='p'    /* 3 */
	  ) {

#ifdef G__ASM_DBG
    if(G__asm_dbg)
      G__fprinterr(G__serr,"%3x: CMPJMP a %c b optimized\n"
	      ,*start+9,G__asm_inst[*start+11]);
#endif
    G__asm_gettest((int)G__asm_inst[*start+11] ,&G__asm_inst[*start+10]);

    G__asm_inst[*start+11]=
      ((struct G__var_array*)G__asm_inst[*start+4])->p[G__asm_inst[*start+1]];
    if(G__asm_inst[*start]==G__LD_MSTR
       && G__LOCALSTATIC!=((struct G__var_array*)G__asm_inst[*start+4])->statictype[G__asm_inst[*start+1]]
       )
      G__asm_inst[*start+11] += G__store_struct_offset;

    G__asm_inst[*start+12]=
      ((struct G__var_array*)G__asm_inst[*start+9])->p[G__asm_inst[*start+6]];
    if(G__asm_inst[*start+5]==G__LD_MSTR
       && G__LOCALSTATIC!=((struct G__var_array*)G__asm_inst[*start+9])->statictype[G__asm_inst[*start+6]]
       )
      G__asm_inst[*start+12] += G__store_struct_offset;

    G__asm_inst[*start+9]=G__CMPJMP;
    G__asm_inst[*start]=G__JMP;
    G__asm_inst[*start+1] = *start+9;
    G__asm_inst[*start+2]=G__NOP;
    G__asm_inst[*start+3]=G__NOP;
    G__asm_inst[*start+4]=G__NOP;
    G__asm_inst[*start+5]=G__NOP;
    G__asm_inst[*start+6]=G__NOP;
    G__asm_inst[*start+7]=G__NOP;
    G__asm_inst[*start+8]=G__NOP;

    *start += 9;
    G__asm_inst[G__asm_cp-1] = *start;

  }


  /**************************************************************
   * i++ , i-- , ++i , --i at the loop end
   *
   *               before                     ----------> after
   *
   *     -9      G__LD_VAR,LD_MSTR                        INCJMP
   *     -8      index                                    *a  var->p[]
   *     -7      paran                                    1,,-1
   *     -6      point_level                              next_pc
   *     -5      *var                                     NOP
   *     -4      OP1                                      NOP
   *     -3      opr                                      NOP
   *     -2      JMP                                      NOP
   *     -1      next_pc                                  NOP
   * G__asm_cp   RTN                                      RTN
   *******************************************************/
  if(G__asm_inst[G__asm_cp-2]==G__JMP &&
     G__asm_inst[G__asm_cp-4]==G__OP1 &&
     G__asm_inst[G__asm_cp-7]==0      &&
     G__asm_inst[G__asm_cp-6]=='p'    &&
#ifndef G__OLDIMPLEMENTATION599
     G__asm_cond_cp != G__asm_cp-2 &&
#endif
     (G__LD_VAR==G__asm_inst[G__asm_cp-9]||
      (G__LD_MSTR==G__asm_inst[G__asm_cp-9]
#ifndef G__OLDIMPLEMENTATION1333
       && !G__asm_wholefunction
#endif
      )) &&
#ifndef G__OLDIMPLEMENTATION1021
     G__isInt(((struct G__var_array*)G__asm_inst[G__asm_cp-5])->type[G__asm_inst[G__asm_cp-8]])) {
#else
     ((struct G__var_array*)G__asm_inst[G__asm_cp-5])->type[G__asm_inst[G__asm_cp-8]] =='i') {
#endif

#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: INCJMP  i++ optimized\n",G__asm_cp-9);
#endif

    G__asm_inst[G__asm_cp-8]=
      ((struct G__var_array*)G__asm_inst[G__asm_cp-5])->p[G__asm_inst[G__asm_cp-8]];
    if(G__asm_inst[G__asm_cp-9]==G__LD_MSTR
       && G__LOCALSTATIC!=((struct G__var_array*)G__asm_inst[G__asm_cp-5])->statictype[G__asm_inst[G__asm_cp-8]]
       ) {
      G__asm_inst[G__asm_cp-8] += G__store_struct_offset;
    }

    G__asm_inst[G__asm_cp-9]=G__INCJMP;

    switch(G__asm_inst[G__asm_cp-3]) {
    case G__OPR_POSTFIXINC:
    case G__OPR_PREFIXINC:
    case G__OPR_POSTFIXINC_I:
    case G__OPR_PREFIXINC_I:
      G__asm_inst[G__asm_cp-7]= 1;
      break;
    case G__OPR_POSTFIXDEC:
    case G__OPR_PREFIXDEC:
    case G__OPR_POSTFIXDEC_I:
    case G__OPR_PREFIXDEC_I:
      G__asm_inst[G__asm_cp-7]= -1;
      break;
    }
    G__asm_inst[G__asm_cp-6]=G__asm_inst[G__asm_cp-1];
    G__asm_inst[G__asm_cp-5] = G__NOP;
    G__asm_inst[G__asm_cp-4] = G__NOP;
    G__asm_inst[G__asm_cp-3] = G__NOP;
#ifdef G__OLDIMPLEMENTATION1022
    G__asm_inst[G__asm_cp-2] = G__NOP;
    G__asm_inst[G__asm_cp-1] = G__NOP;
#endif

    /*
      G__asm_inst[G__asm_cp-5]=G__RETURN;
      G__asm_cp -= 5 ;
      */

  }

  /**************************************************************
   * i+=1 , i-=1 at the loop end
   *
   *               before                     ----------> after
   *
   *     -11     G__LD_VAR,LD_MSTR                        INCJMP
   *     -10     index                                    *a  var->p[]
   *     -9      paran                                    1,,-1
   *     -8      point_level                              next_pc
   *     -7      *var                                     NOP
   *     -6      G__LD                                    NOP
   *     -5      data_stack                               NOP
   *     -4      OP2                                      NOP
   *     -3      opr                                      NOP
   *     -2      JMP                                      NOP
   *     -1      next_pc                                  NOP
   * G__asm_cp   RTN                           G__asm_cp  RTN
   *******************************************************/
  else if(G__asm_inst[G__asm_cp-2]==G__JMP &&
	  G__asm_inst[G__asm_cp-4]==G__OP2 &&
	  (G__asm_inst[G__asm_cp-3]==G__OPR_ADDASSIGN ||
	   G__asm_inst[G__asm_cp-3]==G__OPR_SUBASSIGN) &&
	  G__asm_inst[G__asm_cp-9]==0      &&
	  G__asm_inst[G__asm_cp-8]=='p'    &&
	  G__asm_inst[G__asm_cp-6]==G__LD  &&
#ifndef G__OLDIMPLEMENTATION599
	  G__asm_cond_cp != G__asm_cp-2 &&
#endif
	  (G__LD_VAR==G__asm_inst[G__asm_cp-11]||
	   (G__LD_MSTR==G__asm_inst[G__asm_cp-11]
#ifndef G__OLDIMPLEMENTATION1333
	    && !G__asm_wholefunction
#endif
	    )) &&
#ifndef G__OLDIMPLEMENTATION1021
	  G__isInt(((struct G__var_array*)G__asm_inst[G__asm_cp-7])->type[G__asm_inst[G__asm_cp-10]]))  {
#else
	  ((struct G__var_array*)G__asm_inst[G__asm_cp-7])->type[G__asm_inst[G__asm_cp-10]] =='i')  {
#endif

#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: INCJMP  i+=1 optimized\n",G__asm_cp-11);
#endif

    G__asm_inst[G__asm_cp-10]=
      ((struct G__var_array*)G__asm_inst[G__asm_cp-7])->p[G__asm_inst[G__asm_cp-10]];
    if(G__asm_inst[G__asm_cp-11]==G__LD_MSTR
       && G__LOCALSTATIC!=((struct G__var_array*)G__asm_inst[G__asm_cp-7])->statictype[G__asm_inst[G__asm_cp-10]]
       ) {
      G__asm_inst[G__asm_cp-10] += G__store_struct_offset;
    }

    G__asm_inst[G__asm_cp-11]=G__INCJMP;

    switch(G__asm_inst[G__asm_cp-3]) {
    case G__OPR_ADDASSIGN:
      G__asm_inst[G__asm_cp-9]=G__int(G__asm_stack[G__asm_inst[G__asm_cp-5]]);
      break;
    case G__OPR_SUBASSIGN:
      G__asm_inst[G__asm_cp-9]= -1*G__int(G__asm_stack[G__asm_inst[G__asm_cp-5]]);
      break;
    }
    G__asm_inst[G__asm_cp-8]=G__asm_inst[G__asm_cp-1];
    G__asm_inst[G__asm_cp-7] = G__NOP;
    G__asm_inst[G__asm_cp-6] = G__NOP;
    G__asm_inst[G__asm_cp-5] = G__NOP;
    G__asm_inst[G__asm_cp-4] = G__NOP;
    G__asm_inst[G__asm_cp-3] = G__NOP;
#ifdef G__OLDIMPLEMENTATION1022
    G__asm_inst[G__asm_cp-2] = G__NOP;
    G__asm_inst[G__asm_cp-1] = G__NOP;
#endif

    /*
      G__asm_inst[G__asm_cp-7]=G__RETURN;
      G__asm_cp -= 7 ;
      */
  }

  /*******************************************************
   * i=i+N , i=i-N at the loop end
   *
   *               before                     ----------> after
   *
   *     -16     G__LD_VAR,MSTR<- check     1             INCJMP
   *     -15     index         <- check     2             *a  var->p[]
   *     -14     paran         <- check     3             inc
   *     -13     point_level   <- check     3             next_pc
   *     -12     *var          <-          (2)            NOP
   *     -11     LD            <- check     1             NOP
   *     -10     data_stack                               NOP
   *     -9      OP2           <- check     1             NOP
   *     -8      +,-           <- check     2             NOP
   *     -7      G__ST_VAR,MSTR<- check     1             NOP
   *     -6      index         <- check     2             NOP
   *     -5      paran                                    NOP
   *     -4      point_level   <- check     3             NOP
   *     -3      *var          <-          (2)            NOP
   *     -2      JMP                                      NOP
   *     -1      next_pc                                  NOP
   * G__asm_cp   RTN                            G__asm_cp RTN
   *******************************************************/
  else if(G__asm_inst[G__asm_cp-2]==G__JMP &&
	  ((G__asm_inst[G__asm_cp-7]==G__ST_VAR&&
	    G__asm_inst[G__asm_cp-16]==G__LD_VAR) ||
	   (G__asm_inst[G__asm_cp-7]==G__ST_MSTR&&
	    G__asm_inst[G__asm_cp-16]==G__LD_MSTR
#ifndef G__OLDIMPLEMENTATION1333
	    && !G__asm_wholefunction
#endif
	   )) &&
	  G__asm_inst[G__asm_cp-9]==G__OP2     &&
	  G__asm_inst[G__asm_cp-11]==G__LD     &&
	  G__asm_inst[G__asm_cp-15]==G__asm_inst[G__asm_cp-6] && /* 2 */
	  G__asm_inst[G__asm_cp-12]==G__asm_inst[G__asm_cp-3] &&
	  (G__asm_inst[G__asm_cp-8]=='+'||G__asm_inst[G__asm_cp-8]=='-') &&
#ifndef G__OLDIMPLEMENTATION1021
	  G__isInt(((struct G__var_array*)G__asm_inst[G__asm_cp-3])->type[G__asm_inst[G__asm_cp-6]])       &&
#else
	  ((struct G__var_array*)G__asm_inst[G__asm_cp-3])->type[G__asm_inst[G__asm_cp-6]] =='i'       &&
#endif
	  G__asm_inst[G__asm_cp-14]==0 &&
	  G__asm_inst[G__asm_cp-13]=='p' &&   /* 3 */
	  G__asm_inst[G__asm_cp-4]=='p') {

#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: INCJMP  i=i+1 optimized\n",G__asm_cp-16);
#endif
    G__asm_inst[G__asm_cp-16]=G__INCJMP;

    G__asm_inst[G__asm_cp-15]=
      ((struct G__var_array*)G__asm_inst[G__asm_cp-3])->p[G__asm_inst[G__asm_cp-6]];
    if(G__asm_inst[G__asm_cp-7]==G__ST_MSTR
       && G__LOCALSTATIC!=((struct G__var_array*)G__asm_inst[G__asm_cp-3])->statictype[G__asm_inst[G__asm_cp-6]]
       )
      G__asm_inst[G__asm_cp-15] += G__store_struct_offset;

    G__asm_inst[G__asm_cp-14]=G__int(G__asm_stack[G__asm_inst[G__asm_cp-10]]);
    if(G__asm_inst[G__asm_cp-8]=='-')
      G__asm_inst[G__asm_cp-14] *= -1;

    G__asm_inst[G__asm_cp-13]=G__asm_inst[G__asm_cp-1];
    G__asm_inst[G__asm_cp-12] = G__NOP;
    G__asm_inst[G__asm_cp-11] = G__NOP;
    G__asm_inst[G__asm_cp-10] = G__NOP;
    G__asm_inst[G__asm_cp-9] = G__NOP;
    G__asm_inst[G__asm_cp-8] = G__NOP;
    G__asm_inst[G__asm_cp-7] = G__NOP;
    G__asm_inst[G__asm_cp-6] = G__NOP;
    G__asm_inst[G__asm_cp-5] = G__NOP;
    G__asm_inst[G__asm_cp-4] = G__NOP;
    G__asm_inst[G__asm_cp-3] = G__NOP;
#ifdef G__OLDIMPLEMENTATION1022
    G__asm_inst[G__asm_cp-2] = G__NOP;
    G__asm_inst[G__asm_cp-1] = G__NOP;
#endif

    /*
      G__asm_inst[G__asm_cp-12]=G__RETURN;
      G__asm_cp -= 12 ;
      */
  }

  /****************************************************************
  * Optimization level 3
  ****************************************************************/
  if(G__asm_loopcompile>=3) {
    G__asm_optimize3(start);
  }

  return(0);

}


#ifndef G__OLDIMPLEMENTATION482
/*************************************************************************
**************************************************************************
* Optimization level 3 function
**************************************************************************
*************************************************************************/

/*************************************************************************
* G__get_LD_p0_p2f()
*************************************************************************/
int G__get_LD_p0_p2f(type,pinst)
int type;
long *pinst;
{
  int done = 1;
  if(isupper(type)) {
    if('Z'==type) done=0;
#ifndef G__OLDIMMPLEMENTATION1341
    else if('P'==type || 'O'==type) *pinst = (long)G__LD_p0_double; 
#endif
    else *pinst = (long)G__LD_p0_pointer;
  }
  else {
    switch(type) {
    case 'b': *pinst = (long)G__LD_p0_uchar; break;
    case 'c': *pinst = (long)G__LD_p0_char; break;
    case 'r': *pinst = (long)G__LD_p0_ushort; break;
    case 's': *pinst = (long)G__LD_p0_short; break;
    case 'h': *pinst = (long)G__LD_p0_uint; break;
    case 'i': *pinst = (long)G__LD_p0_int; break;
    case 'k': *pinst = (long)G__LD_p0_ulong; break;
    case 'l': *pinst = (long)G__LD_p0_long; break;
    case 'u': *pinst = (long)G__LD_p0_struct; break;
    case 'f': *pinst = (long)G__LD_p0_float; break;
    case 'd': *pinst = (long)G__LD_p0_double; break;
#ifndef G__OLDIMPLEMENTATION1604
    case 'g': *pinst = (long)G__LD_p0_bool; break;
#endif
#ifndef G__OLDIMPLEMENTATION2189
    case 'n': *pinst = (long)G__LD_p0_longlong; break;
    case 'm': *pinst = (long)G__LD_p0_ulonglong; break;
    case 'q': *pinst = (long)G__LD_p0_longdouble; break;
#endif
    default: done=0; break;
    }
  }
  return(done);
}

/*************************************************************************
* G__get_LD_p1_p2f()
*************************************************************************/
int G__get_LD_p1_p2f(type,pinst)
int type;
long *pinst;
{
  int done = 1;
  if(isupper(type)) {
    if('Z'==type) done=0;
    else *pinst = (long)G__LD_p1_pointer;
  }
  else {
    switch(type) {
    case 'b': *pinst = (long)G__LD_p1_uchar; break;
    case 'c': *pinst = (long)G__LD_p1_char; break;
    case 'r': *pinst = (long)G__LD_p1_ushort; break;
    case 's': *pinst = (long)G__LD_p1_short; break;
    case 'h': *pinst = (long)G__LD_p1_uint; break;
    case 'i': *pinst = (long)G__LD_p1_int; break;
    case 'k': *pinst = (long)G__LD_p1_ulong; break;
    case 'l': *pinst = (long)G__LD_p1_long; break;
    case 'u': *pinst = (long)G__LD_p1_struct; break;
    case 'f': *pinst = (long)G__LD_p1_float; break;
    case 'd': *pinst = (long)G__LD_p1_double; break;
#ifndef G__OLDIMPLEMENTATION1604
    case 'g': *pinst = (long)G__LD_p1_bool; break;
#endif
#ifndef G__OLDIMPLEMENTATION2189
    case 'n': *pinst = (long)G__LD_p1_longlong; break;
    case 'm': *pinst = (long)G__LD_p1_ulonglong; break;
    case 'q': *pinst = (long)G__LD_p1_longdouble; break;
#endif
    default: done=0; break;
    }
  }
  return(done);
}

#ifndef G__OLDIMPLEMENTATION1405
/*************************************************************************
* G__get_LD_pn_p2f()
*************************************************************************/
int G__get_LD_pn_p2f(type,pinst)
int type;
long *pinst;
{
  int done = 1;
  if(isupper(type)) {
    if('Z'==type) done=0;
    else *pinst = (long)G__LD_pn_pointer;
  }
  else {
    switch(type) {
    case 'b': *pinst = (long)G__LD_pn_uchar; break;
    case 'c': *pinst = (long)G__LD_pn_char; break;
    case 'r': *pinst = (long)G__LD_pn_ushort; break;
    case 's': *pinst = (long)G__LD_pn_short; break;
    case 'h': *pinst = (long)G__LD_pn_uint; break;
    case 'i': *pinst = (long)G__LD_pn_int; break;
    case 'k': *pinst = (long)G__LD_pn_ulong; break;
    case 'l': *pinst = (long)G__LD_pn_long; break;
    case 'u': *pinst = (long)G__LD_pn_struct; break;
    case 'f': *pinst = (long)G__LD_pn_float; break;
    case 'd': *pinst = (long)G__LD_pn_double; break;
#ifndef G__OLDIMPLEMENTATION1604
    case 'g': *pinst = (long)G__LD_pn_bool; break;
#endif
#ifndef G__OLDIMPLEMENTATION2189
    case 'n': *pinst = (long)G__LD_pn_longlong; break;
    case 'm': *pinst = (long)G__LD_pn_ulonglong; break;
    case 'q': *pinst = (long)G__LD_pn_longdouble; break;
#endif
    default: done=0; break;
    }
  }
  return(done);
}
#endif /* 1405 */

/*************************************************************************
* G__get_LD_P10_p2f()
*************************************************************************/
int G__get_LD_P10_p2f(type,pinst,reftype)
int type;
long *pinst;
int reftype;
{
  int done = 1;
  if(G__PARAP2P==reftype) {
    if('Z'==type) done=0;
    else *pinst = (long)G__LD_P10_pointer;
  }
  else if(G__PARANORMAL==reftype) {
    switch(type) {
    case 'B': *pinst = (long)G__LD_P10_uchar; break;
    case 'C': *pinst = (long)G__LD_P10_char; break;
    case 'R': *pinst = (long)G__LD_P10_ushort; break;
    case 'S': *pinst = (long)G__LD_P10_short; break;
    case 'H': *pinst = (long)G__LD_P10_uint; break;
    case 'I': *pinst = (long)G__LD_P10_int; break;
    case 'K': *pinst = (long)G__LD_P10_ulong; break;
    case 'L': *pinst = (long)G__LD_P10_long; break;
    case 'U': *pinst = (long)G__LD_P10_struct; break;
    case 'F': *pinst = (long)G__LD_P10_float; break;
    case 'D': *pinst = (long)G__LD_P10_double; break;
#ifndef G__OLDIMPLEMENTATION2188
    case 'G': *pinst = (long)G__LD_P10_bool; break;
#endif
#ifndef G__OLDIMPLEMENTATION2189
    case 'N': *pinst = (long)G__LD_P10_longlong; break;
    case 'M': *pinst = (long)G__LD_P10_ulonglong; break;
    case 'Q': *pinst = (long)G__LD_P10_longdouble; break;
#endif
    default: done=0; break;
    }
  }
  else {
    done=0;
  }
  return(done);
}

/*************************************************************************
* G__get_ST_p0_p2f()
*************************************************************************/
int G__get_ST_p0_p2f(type,pinst)
int type;
long *pinst;
{
  int done = 1;
  if(isupper(type)) {
    if('Z'==type) done=0;
    else *pinst = (long)G__ST_p0_pointer;
  }
  else {
    switch(type) {
    case 'b': *pinst = (long)G__ST_p0_uchar; break;
    case 'c': *pinst = (long)G__ST_p0_char; break;
    case 'r': *pinst = (long)G__ST_p0_ushort; break;
    case 's': *pinst = (long)G__ST_p0_short; break;
    case 'h': *pinst = (long)G__ST_p0_uint; break;
    case 'i': *pinst = (long)G__ST_p0_int; break;
    case 'k': *pinst = (long)G__ST_p0_ulong; break;
    case 'l': *pinst = (long)G__ST_p0_long; break;
    case 'u': *pinst = (long)G__ST_p0_struct; break;
    case 'f': *pinst = (long)G__ST_p0_float; break;
    case 'd': *pinst = (long)G__ST_p0_double; break;
#ifndef G__OLDIMPLEMENTATION1604
    case 'g': *pinst = (long)G__ST_p0_bool; break;
#endif
#ifndef G__OLDIMPLEMENTATION2189
    case 'n': *pinst = (long)G__ST_p0_longlong; break;
    case 'm': *pinst = (long)G__ST_p0_ulonglong; break;
    case 'q': *pinst = (long)G__ST_p0_longdouble; break;
#endif
    default: done=0; break;
    }
  }
  return(done);
}

/*************************************************************************
* G__get_ST_p1_p2f()
*************************************************************************/
int G__get_ST_p1_p2f(type,pinst)
int type;
long *pinst;
{
  int done = 1;
  if(isupper(type)) {
    if('Z'==type) done=0;
    else *pinst = (long)G__ST_p1_pointer;
  }
  else {
    switch(type) {
    case 'b': *pinst = (long)G__ST_p1_uchar; break;
    case 'c': *pinst = (long)G__ST_p1_char; break;
    case 'r': *pinst = (long)G__ST_p1_ushort; break;
    case 's': *pinst = (long)G__ST_p1_short; break;
    case 'h': *pinst = (long)G__ST_p1_uint; break;
    case 'i': *pinst = (long)G__ST_p1_int; break;
    case 'k': *pinst = (long)G__ST_p1_ulong; break;
    case 'l': *pinst = (long)G__ST_p1_long; break;
    case 'u': *pinst = (long)G__ST_p1_struct; break;
    case 'f': *pinst = (long)G__ST_p1_float; break;
    case 'd': *pinst = (long)G__ST_p1_double; break;
#ifndef G__OLDIMPLEMENTATION1604
    case 'g': *pinst = (long)G__ST_p1_bool; break; /* to be fixed */
#endif
#ifndef G__OLDIMPLEMENTATION2189
    case 'n': *pinst = (long)G__ST_p1_longlong; break;
    case 'm': *pinst = (long)G__ST_p1_ulonglong; break;
    case 'q': *pinst = (long)G__ST_p1_longdouble; break;
#endif
    default: done=0; break;
    }
  }
  return(done);
}

#ifndef G__OLDIMPLEMENTATION1406
/*************************************************************************
* G__get_ST_pn_p2f()
*************************************************************************/
int G__get_ST_pn_p2f(type,pinst)
int type;
long *pinst;
{
  int done = 1;
  if(isupper(type)) {
    if('Z'==type) done=0;
    else *pinst = (long)G__ST_pn_pointer;
  }
  else {
    switch(type) {
    case 'b': *pinst = (long)G__ST_pn_uchar; break;
    case 'c': *pinst = (long)G__ST_pn_char; break;
    case 'r': *pinst = (long)G__ST_pn_ushort; break;
    case 's': *pinst = (long)G__ST_pn_short; break;
    case 'h': *pinst = (long)G__ST_pn_uint; break;
    case 'i': *pinst = (long)G__ST_pn_int; break;
    case 'k': *pinst = (long)G__ST_pn_ulong; break;
    case 'l': *pinst = (long)G__ST_pn_long; break;
    case 'u': *pinst = (long)G__ST_pn_struct; break;
    case 'f': *pinst = (long)G__ST_pn_float; break;
    case 'd': *pinst = (long)G__ST_pn_double; break;
#ifndef G__OLDIMPLEMENTATION1604
    case 'g': *pinst = (long)G__ST_pn_bool; break; /* to be fixed */
#endif
#ifndef G__OLDIMPLEMENTATION2189
    case 'n': *pinst = (long)G__ST_pn_longlong; break;
    case 'm': *pinst = (long)G__ST_pn_ulonglong; break;
    case 'q': *pinst = (long)G__ST_pn_longdouble; break;
#endif
    default: done=0; break;
    }
  }
  return(done);
}
#endif /* 1406 */

/*************************************************************************
* G__get_ST_P10_p2f()
*************************************************************************/
int G__get_ST_P10_p2f(type,pinst,reftype)
int type;
long *pinst;
int reftype;
{
  int done = 1;
  if(G__PARAP2P==reftype) {
    if('Z'==type) done=0;
    else *pinst = (long)G__ST_P10_pointer;
  }
  else if(G__PARANORMAL==reftype) {
    switch(type) {
    case 'B': *pinst = (long)G__ST_P10_uchar; break;
    case 'C': *pinst = (long)G__ST_P10_char; break;
    case 'R': *pinst = (long)G__ST_P10_ushort; break;
    case 'S': *pinst = (long)G__ST_P10_short; break;
    case 'H': *pinst = (long)G__ST_P10_uint; break;
    case 'I': *pinst = (long)G__ST_P10_int; break;
    case 'K': *pinst = (long)G__ST_P10_ulong; break;
    case 'L': *pinst = (long)G__ST_P10_long; break;
    case 'U': *pinst = (long)G__ST_P10_struct; break;
    case 'F': *pinst = (long)G__ST_P10_float; break;
    case 'D': *pinst = (long)G__ST_P10_double; break;
#ifndef G__OLDIMPLEMENTATION2188
    case 'G': *pinst = (long)G__ST_P10_bool; break;
#endif
#ifndef G__OLDIMPLEMENTATION2189
    case 'N': *pinst = (long)G__ST_P10_longlong; break;
    case 'M': *pinst = (long)G__ST_P10_ulonglong; break;
    case 'Q': *pinst = (long)G__ST_P10_longdouble; break;
#endif
    default: done=0; break;
    }
  }
  else {
    done=0;
  }
  return(done);
}

/*************************************************************************
* G__get_LD_Rp0_p2f()
*************************************************************************/
int G__get_LD_Rp0_p2f(type,pinst)
int type;
long *pinst;
{
  int done = 1;
  if(isupper(type)) {
    if('Z'==type) done=0;
    else *pinst = (long)G__LD_Rp0_pointer;
  }
  else {
    switch(type) {
    case 'b': *pinst = (long)G__LD_Rp0_uchar; break;
    case 'c': *pinst = (long)G__LD_Rp0_char; break;
    case 'r': *pinst = (long)G__LD_Rp0_ushort; break;
    case 's': *pinst = (long)G__LD_Rp0_short; break;
    case 'h': *pinst = (long)G__LD_Rp0_uint; break;
    case 'i': *pinst = (long)G__LD_Rp0_int; break;
    case 'k': *pinst = (long)G__LD_Rp0_ulong; break;
    case 'l': *pinst = (long)G__LD_Rp0_long; break;
    case 'u': *pinst = (long)G__LD_Rp0_struct; break;
    case 'f': *pinst = (long)G__LD_Rp0_float; break;
    case 'd': *pinst = (long)G__LD_Rp0_double; break;
#ifndef G__OLDIMPLEMENTATION1604
    case 'g': *pinst = (long)G__LD_Rp0_bool; break; /* to be fixed */
#endif
#ifndef G__OLDIMPLEMENTATION2189
    case 'n': *pinst = (long)G__LD_Rp0_longlong; break;
    case 'm': *pinst = (long)G__LD_Rp0_ulonglong; break;
    case 'q': *pinst = (long)G__LD_Rp0_longdouble; break;
#endif
    default: done=0; break;
    }
  }
  return(done);
}
/*************************************************************************
* G__get_ST_Rp0_p2f()
*************************************************************************/
int G__get_ST_Rp0_p2f(type,pinst)
int type;
long *pinst;
{
  int done = 1;
  if(isupper(type)) {
    if('Z'==type) done=0;
    else *pinst = (long)G__ST_Rp0_pointer;
  }
  else {
    switch(type) {
    case 'b': *pinst = (long)G__ST_Rp0_uchar; break;
    case 'c': *pinst = (long)G__ST_Rp0_char; break;
    case 'r': *pinst = (long)G__ST_Rp0_ushort; break;
    case 's': *pinst = (long)G__ST_Rp0_short; break;
    case 'h': *pinst = (long)G__ST_Rp0_uint; break;
    case 'i': *pinst = (long)G__ST_Rp0_int; break;
    case 'k': *pinst = (long)G__ST_Rp0_ulong; break;
    case 'l': *pinst = (long)G__ST_Rp0_long; break;
    case 'u': *pinst = (long)G__ST_Rp0_struct; break;
    case 'f': *pinst = (long)G__ST_Rp0_float; break;
    case 'd': *pinst = (long)G__ST_Rp0_double; break;
#ifndef G__OLDIMPLEMENTATION1604
    case 'g': *pinst = (long)G__ST_Rp0_bool; break; /* to be fixed */
#endif
#ifndef G__OLDIMPLEMENTATION2189
    case 'n': *pinst = (long)G__ST_Rp0_longlong; break;
    case 'm': *pinst = (long)G__ST_Rp0_ulonglong; break;
    case 'q': *pinst = (long)G__ST_Rp0_longdouble; break;
#endif
    default: done=0; break;
    }
  }
  return(done);
}
/*************************************************************************
* G__get_LD_RP0_p2f()
*************************************************************************/
int G__get_LD_RP0_p2f(type,pinst)
int type;
long *pinst;
{
  int done = 1;
  if(isupper(type)) {
    if('Z'==type) done=0;
    else *pinst = (long)G__LD_RP0_pointer;
  }
  else {
    switch(type) {
    case 'b': *pinst = (long)G__LD_RP0_uchar; break;
    case 'c': *pinst = (long)G__LD_RP0_char; break;
    case 'r': *pinst = (long)G__LD_RP0_ushort; break;
    case 's': *pinst = (long)G__LD_RP0_short; break;
    case 'h': *pinst = (long)G__LD_RP0_uint; break;
    case 'i': *pinst = (long)G__LD_RP0_int; break;
    case 'k': *pinst = (long)G__LD_RP0_ulong; break;
    case 'l': *pinst = (long)G__LD_RP0_long; break;
    case 'u': *pinst = (long)G__LD_RP0_struct; break;
    case 'f': *pinst = (long)G__LD_RP0_float; break;
    case 'd': *pinst = (long)G__LD_RP0_double; break;
#ifndef G__OLDIMPLEMENTATION1604
    case 'g': *pinst = (long)G__LD_RP0_bool; break; /* to be fixed */
#endif
#ifndef G__OLDIMPLEMENTATION2189
    case 'n': *pinst = (long)G__LD_RP0_longlong; break;
    case 'm': *pinst = (long)G__LD_RP0_ulonglong; break;
    case 'q': *pinst = (long)G__LD_RP0_longdouble; break;
#endif
    default: done=0; break;
    }
  }
  return(done);
}

/*************************************************************************
* G__LD_Rp0_optimize()
*************************************************************************/
void G__LD_Rp0_optimize(var,ig15,pc,inst)
struct G__var_array *var;
int ig15;
int pc;
long inst;
{
  long originst=G__asm_inst[pc];
  int pointlevel=G__asm_inst[pc+3];
#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    switch(inst) {
    case G__LDST_VAR_P: /* illegal case */
      G__fprinterr(G__serr,"  G__LD_VAR REF optimized 6 G__LDST_VAR_P\n");
      break;
    case G__LDST_MSTR_P:
      G__fprinterr(G__serr,"  G__LD_MSTR REF optimized 6 G__LDST_MSTR_P\n");
      break;
    case G__LDST_LVAR_P:
      G__fprinterr(G__serr,"  G__LD_LVAR REF optimized 6 G__LDST_LVAR_P\n");
      break;
    }
  }
#endif
  G__asm_inst[pc] = inst;
  G__asm_inst[pc+3] = 0;
  if(0==G__get_LD_Rp0_p2f(var->type[ig15],&G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"Error: LD_VAR,LD_MSTR REF optimize (6) error %s\n"
	      ,var->varnamebuf[ig15]);
    }
#endif
    G__asm_inst[pc] = originst;
    G__asm_inst[pc+3] = pointlevel;
  }
}
/*************************************************************************
* G__ST_Rp0_optimize()
*************************************************************************/
void G__ST_Rp0_optimize(var,ig15,pc,inst)
struct G__var_array *var;
int ig15;
int pc;
long inst;
{
  long originst=G__asm_inst[pc];
  int pointlevel=G__asm_inst[pc+3];
#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    switch(inst) {
    case G__LDST_VAR_P: /* illegal case */
      G__fprinterr(G__serr,"  G__ST_VAR REF optimized 6 G__LDST_VAR_P\n");
      break;
    case G__LDST_MSTR_P:
      G__fprinterr(G__serr,"  G__ST_MSTR REF optimized 6 G__LDST_MSTR_P\n");
      break;
    case G__LDST_LVAR_P:
      G__fprinterr(G__serr,"  G__ST_LVAR REF optimized 6 G__LDST_LVAR_P\n");
      break;
    }
  }
#endif
  G__asm_inst[pc] = inst;
  G__asm_inst[pc+3] = 0;
  if(0==G__get_ST_Rp0_p2f(var->type[ig15],&G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"Error: LD_VAR,LD_MSTR REF optimize (6) error %s\n"
	      ,var->varnamebuf[ig15]);
    }
#endif
    G__asm_inst[pc] = originst;
    G__asm_inst[pc+3] = pointlevel;
  }
}
/*************************************************************************
* G__LD_RP0_optimize()
*************************************************************************/
void G__LD_RP0_optimize(var,ig15,pc,inst)
struct G__var_array *var;
int ig15;
int pc;
long inst;
{
  long originst=G__asm_inst[pc];
  int pointlevel=G__asm_inst[pc+3];
#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    switch(inst) {
    case G__LDST_VAR_P: /* illegal case */
      G__fprinterr(G__serr,"  G__LD_VAR REF optimized 7 G__LDST_VAR_P\n");
      break;
    case G__LDST_MSTR_P:
      G__fprinterr(G__serr,"  G__LD_MSTR REF optimized 7 G__LDST_MSTR_P\n");
      break;
    case G__LDST_LVAR_P:
      G__fprinterr(G__serr,"  G__LD_LVAR REF optimized 7 G__LDST_LVAR_P\n");
      break;
    }
  }
#endif
  G__asm_inst[pc] = inst;
  G__asm_inst[pc+3] = 0;
  if(0==G__get_LD_RP0_p2f(var->type[ig15],&G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"Error: LD_VAR,LD_MSTR REF optimize (7) error %s\n"
	      ,var->varnamebuf[ig15]);
    }
#endif
    G__asm_inst[pc] = originst;
    G__asm_inst[pc+3] = pointlevel;
  }
}

/*************************************************************************
* G__LD_p0_optimize()
*************************************************************************/
void G__LD_p0_optimize(var,ig15,pc,inst)
struct G__var_array *var;
int ig15;
int pc;
long inst;
{
  long originst=G__asm_inst[pc];
  int pointlevel=G__asm_inst[pc+3];
#ifndef G__OLDIMPLEMENTATION910
  if(var->bitfield[ig15]) return;
#endif
#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    switch(inst) {
    case G__LDST_VAR_P:
      G__fprinterr(G__serr,"  G__LD_VAR optimized 6 G__LDST_VAR_P\n");
      break;
    case G__LDST_MSTR_P:
      G__fprinterr(G__serr,"  G__LD_MSTR optimized 6 G__LDST_MSTR_P\n");
      break;
    case G__LDST_LVAR_P:
      G__fprinterr(G__serr,"  G__LD_LvAR optimized 6 G__LDST_LVAR_P\n");
      break;
    }
  }
#endif
  G__asm_inst[pc] = inst;
  G__asm_inst[pc+3] = 0;
  if(0==G__get_LD_p0_p2f(var->type[ig15],&G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"Error: LD_VAR,LD_MSTR optimize (6) error %s\n"
	      ,var->varnamebuf[ig15]);
    }
#endif
    G__asm_inst[pc] = originst;
    G__asm_inst[pc+3] = pointlevel;
  }
}

/*************************************************************************
* G__LD_p1_optimize()
*************************************************************************/
void G__LD_p1_optimize(var,ig15,pc,inst)
struct G__var_array *var;
int ig15;
int pc;
long inst;
{
  long originst=G__asm_inst[pc];
  int pointlevel=G__asm_inst[pc+3];
#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    switch(inst) {
    case G__LDST_VAR_P:
      G__fprinterr(G__serr,"  G__LD_VAR optimized 7 G__LDST_VAR_P\n");
      break;
    case G__LDST_MSTR_P:
      G__fprinterr(G__serr,"  G__LD_MSTR optimized 7 G__LDST_MSTR_P\n");
      break;
    case G__LDST_LVAR_P:
      G__fprinterr(G__serr,"  G__LD_LVAR optimized 7 G__LDST_LVAR_P\n");
      break;
    }
  }
#endif
  G__asm_inst[pc] = inst;
  G__asm_inst[pc+3] = 0;
  if(0==G__get_LD_p1_p2f(var->type[ig15],&G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"Error: LD_VAR optimize (8) error %s\n"
			   ,var->varnamebuf[ig15]);
#endif
    G__asm_inst[pc] = originst;
    G__asm_inst[pc+3] = pointlevel;
  }
}

#ifndef G__OLDIMPLEMENTATION1405
/*************************************************************************
* G__LD_pn_optimize()
*************************************************************************/
void G__LD_pn_optimize(var,ig15,pc,inst)
struct G__var_array *var;
int ig15;
int pc;
long inst;
{
  long originst=G__asm_inst[pc];
  int pointlevel=G__asm_inst[pc+3];
#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    switch(inst) {
    case G__LDST_VAR_P:
      G__fprinterr(G__serr,"  G__LD_VAR optimized 8 G__LDST_VAR_P\n");
      break;
    case G__LDST_MSTR_P:
      G__fprinterr(G__serr,"  G__LD_MSTR optimized 8 G__LDST_MSTR_P\n");
      break;
    case G__LDST_LVAR_P:
      G__fprinterr(G__serr,"  G__LD_LVAR optimized 8 G__LDST_LVAR_P\n");
      break;
    }
  }
#endif
  G__asm_inst[pc] = inst;
  G__asm_inst[pc+3] = 0;
  if(0==G__get_LD_pn_p2f(var->type[ig15],&G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"Error: LD_VAR optimize (8) error %s\n"
			   ,var->varnamebuf[ig15]);
#endif
    G__asm_inst[pc] = originst;
    G__asm_inst[pc+3] = pointlevel;
  }
}
#endif /* 1405 */

/*************************************************************************
* G__LD_P10_optimize()
*************************************************************************/
void G__LD_P10_optimize(var,ig15,pc,inst)
struct G__var_array *var;
int ig15;
int pc;
long inst;
{
  long originst=G__asm_inst[pc];
  int pointlevel=G__asm_inst[pc+3];
#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    switch(inst) {
    case G__LDST_VAR_P:
      G__fprinterr(G__serr,"  G__LD_VAR optimized 9 G__LDST_VAR_P\n");
      break;
    case G__LDST_MSTR_P:
      G__fprinterr(G__serr,"  G__LD_MSTR optimized 9 G__LDST_MSTR_P\n");
      break;
    case G__LDST_LVAR_P:
      G__fprinterr(G__serr,"  G__LD_LVAR optimized 9 G__LDST_LVAR_P\n");
      break;
    }
  }
#endif
  G__asm_inst[pc] = inst;
  G__asm_inst[pc+3] = 0;
  if(0==G__get_LD_P10_p2f(var->type[ig15],&G__asm_inst[pc+2]
			  ,var->reftype[ig15])) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"Error: LD_VAR optimize (9) error %s\n"
			   ,var->varnamebuf[ig15]);
#endif
    G__asm_inst[pc] = originst;
    G__asm_inst[pc+3] = pointlevel;
  }
}


/*************************************************************************
* G__ST_p0_optimize()
*************************************************************************/
void G__ST_p0_optimize(var,ig15,pc,inst)
struct G__var_array *var;
int ig15;
int pc;
long inst;
{
  long originst=G__asm_inst[pc];
  int pointlevel=G__asm_inst[pc+3];
#ifndef G__OLDIMPLEMENTATION910
  if(var->bitfield[ig15]) return;
#endif
#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    switch(inst) {
    case G__LDST_VAR_P:
      G__fprinterr(G__serr,"  G__ST_VAR optimized 8 G__LDST_VAR_P\n");
      break;
    case G__LDST_MSTR_P:
      G__fprinterr(G__serr,"  G__ST_MSTR optimized 8 G__LDST_MSTR_P\n");
      break;
    case G__LDST_LVAR_P:
      G__fprinterr(G__serr,"  G__ST_VAR optimized 8 G__LDST_LVAR_P\n");
      break;
    }
  }
#endif
  G__asm_inst[pc+0] = inst;
  G__asm_inst[pc+3] = 1;
  if(0==G__get_ST_p0_p2f(var->type[ig15],&G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"Warning: ST_VAR optimize (8) error %s\n"
			   ,var->varnamebuf[ig15]);
#endif
    G__asm_inst[pc] = originst;
    G__asm_inst[pc+3] = pointlevel;
  }
}

/*************************************************************************
* G__ST_p1_optimize()
*************************************************************************/
void G__ST_p1_optimize(var,ig15,pc,inst)
struct G__var_array *var;
int ig15;
int pc;
long inst;
{
  long originst=G__asm_inst[pc];
  int pointlevel=G__asm_inst[pc+3];
#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    switch(inst) {
    case G__LDST_VAR_P:
      G__fprinterr(G__serr,"  G__ST_VAR optimized 9 G__LDST_VAR_P\n");
      break;
    case G__LDST_MSTR_P:
      G__fprinterr(G__serr,"  G__ST_MSTR optimized 9 G__LDST_MSTR_P\n");
      break;
    case G__LDST_LVAR_P:
      G__fprinterr(G__serr,"  G__ST_VAR optimized 9 G__LDST_LVAR_P\n");
      break;
    }
  }
#endif
  G__asm_inst[pc+0] = inst;
  G__asm_inst[pc+3] = 1;
  if(0==G__get_ST_p1_p2f(var->type[ig15],&G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"Warning: ST_VAR optimize error %s\n"
			   ,var->varnamebuf[ig15]);
#endif
    G__asm_inst[pc] = originst;
    G__asm_inst[pc+3] = pointlevel;
  }
}

#ifndef G__OLDIMPLEMENTATION1406
/*************************************************************************
* G__ST_pn_optimize()
*************************************************************************/
void G__ST_pn_optimize(var,ig15,pc,inst)
struct G__var_array *var;
int ig15;
int pc;
long inst;
{
  long originst=G__asm_inst[pc];
  int pointlevel=G__asm_inst[pc+3];
#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    switch(inst) {
    case G__LDST_VAR_P:
      G__fprinterr(G__serr,"  G__ST_VAR optimized 10 G__LDST_VAR_P\n");
      break;
    case G__LDST_MSTR_P:
      G__fprinterr(G__serr,"  G__ST_MSTR optimized 10 G__LDST_MSTR_P\n");
      break;
    case G__LDST_LVAR_P:
      G__fprinterr(G__serr,"  G__ST_VAR optimized 10 G__LDST_LVAR_P\n");
      break;
    }
  }
#endif
  G__asm_inst[pc+0] = inst;
  G__asm_inst[pc+3] = 1;
  if(0==G__get_ST_pn_p2f(var->type[ig15],&G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"Warning: ST_VAR optimize error %s\n"
			   ,var->varnamebuf[ig15]);
#endif
    G__asm_inst[pc] = originst;
    G__asm_inst[pc+3] = pointlevel;
  }
}
#endif

/*************************************************************************
* G__ST_P10_optimize()
*************************************************************************/
void G__ST_P10_optimize(var,ig15,pc,inst)
struct G__var_array *var;
int ig15;
int pc;
long inst;
{
  long originst=G__asm_inst[pc];
  int pointlevel=G__asm_inst[pc+3];
#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    switch(inst) {
    case G__LDST_VAR_P:
      G__fprinterr(G__serr,"  G__ST_VAR optimized 7 G__LDST_VAR_P\n");
      break;
    case G__LDST_MSTR_P:
      G__fprinterr(G__serr,"  G__ST_MSTR optimized 7 G__LDST_MSTR_P\n");
      break;
    case G__LDST_LVAR_P:
      G__fprinterr(G__serr,"  G__ST_LVAR optimized 7 G__LDST_LVAR_P\n");
      break;
    }
  }
#endif
  G__asm_inst[pc] = inst;
  G__asm_inst[pc+3] = 0;
  if(0==G__get_ST_P10_p2f(var->type[ig15],&G__asm_inst[pc+2]
			  ,var->reftype[ig15])) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"Error: ST_VAR optimize (7) error %s\n"
			   ,var->varnamebuf[ig15]);
#endif
    G__asm_inst[pc] = originst;
    G__asm_inst[pc+3] = pointlevel;
  }
}

/*************************************************************************
* array index optimization constant
*************************************************************************/
#ifndef G__OLDIMPLEMENTATION822
#define G__MAXINDEXCONST 11
static int G__indexconst[G__MAXINDEXCONST] = {0,1,2,3,4,5,6,7,8,9,10};
#endif

/*************************************************************************
* G__LD_VAR_int_optimize()
*************************************************************************/
int G__LD_VAR_int_optimize(ppc,pi)
int *ppc;
int *pi;
{
  struct G__var_array *var;
  int ig15;
  int pc;
  int done=0;
  pc = *ppc;

  /********************************************************************
   * G__LDST_VAR_INDEX optimization
   ********************************************************************/
  if(1==G__asm_inst[pc+7] && 'p' == G__asm_inst[pc+8] &&
     (var = (struct G__var_array*)G__asm_inst[pc+9]) &&
     1==var->paran[G__asm_inst[pc+6]] &&
     (islower(var->type[G__asm_inst[pc+6]])||
      G__PARANORMAL==var->reftype[G__asm_inst[pc+6]])) {
    ig15 = G__asm_inst[pc+6];
    /********************************************************************
     * 0 G__LD_VAR,LVAR                    G__LDST_VAR_INDEX
     * 1 index                             *arrayindex
     * 2 paran == 0                        (*p2f)(buf,psp,0,var2,index2)
     * 3 pointer_level == p                index2
     * 4 var_array pointer                 pc increment
     * 5 G__LD_VAR,LVAR                    local_global
     * 6 index2                            var_array2 pointer
     * 7 paran == 1
     * 8 point_level == p
     * 9 var_array2 pointer
     ********************************************************************/
    if(G__LD_VAR==G__asm_inst[pc+5] || G__LD_LVAR==G__asm_inst[pc+5]) {
      int flag;
      if(G__LD_LVAR==G__asm_inst[pc]) flag = 1;
      else                            flag = 0;
      if(G__LD_LVAR==G__asm_inst[pc+5]) flag |= 2;
      if(0==G__get_LD_p1_p2f(var->type[ig15],&G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
	if(G__asm_dbg)
	  G__fprinterr(G__serr,"Error: LD_VAR,LD_VAR[1] optimize error %s\n"
		  ,var->varnamebuf[ig15]);
#endif
      }
      else {
	done=1;
	G__asm_inst[pc+5] = flag;
	G__asm_inst[pc] = G__LDST_VAR_INDEX;
	G__asm_inst[pc+1] = (long)pi;
	G__asm_inst[pc+3] = G__asm_inst[pc+6];
	G__asm_inst[pc+4] = 10;
	G__asm_inst[pc+6] = G__asm_inst[pc+9];
	*ppc = pc+5; /* other 2 is incremented one level up */
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"LDST_VAR_INDEX (1) optimized\n");
#endif
      }
    }

    /********************************************************************
     * 0 G__LD_VAR                         G__LDST_VAR_INDEX
     * 1 index                             *arrayindex
     * 2 paran == 0                        (*p2f)(buf,psp,0,var2,index2)
     * 3 pointer_level == p                index2
     * 4 var_array pointer                 pc increment
     * 5 G__ST_VAR                         local_global
     * 6 index2                            var_array2 pointer
     * 7 paran == 1
     * 8 point_level == p
     * 9 var_array2 pointer
     ********************************************************************/
    else if(G__ST_VAR==G__asm_inst[pc+5] || G__ST_LVAR==G__asm_inst[pc+5]) {
      int flag;
      if(G__LD_LVAR==G__asm_inst[pc]) flag = 1;
      else                            flag = 0;
      if(G__ST_LVAR==G__asm_inst[pc+5]) flag |= 2;
      ig15 = G__asm_inst[pc+6];
      if(0==G__get_ST_p1_p2f(var->type[ig15],&G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
	if(G__asm_dbg)
	  G__fprinterr(G__serr,"Error: LD_VAR,ST_VAR[1] optimize error %s\n"
		  ,var->varnamebuf[ig15]);
#endif
      }
      else {
	done=1;
	G__asm_inst[pc+5] = flag;
	G__asm_inst[pc] = G__LDST_VAR_INDEX;
	G__asm_inst[pc+1] = (long)pi;
	G__asm_inst[pc+3] = G__asm_inst[pc+6];
	G__asm_inst[pc+4] = 10;
	G__asm_inst[pc+6] = G__asm_inst[pc+9];
	*ppc = pc+5; /* other 2 is incremented one level up */
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"LDST_VAR_INDEX (2) optimized\n");
#endif
      }
    }
  }

  /********************************************************************
   * G__LDST_VAR_INDEX_OPR optimization
   ********************************************************************/
  else if(G__LD==G__asm_inst[pc+5] &&
	  'i'==G__asm_stack[G__asm_inst[pc+6]].type &&
	  G__OP2 == G__asm_inst[pc+7] &&
	  ('+' == G__asm_inst[pc+8] || '-' == G__asm_inst[pc+8]) &&
	  1==G__asm_inst[pc+11] &&
	  'p'==G__asm_inst[pc+12] &&
	  (var = (struct G__var_array*)G__asm_inst[pc+13]) &&
	  1==var->paran[G__asm_inst[pc+10]] &&
	  (islower(var->type[G__asm_inst[pc+10]])||
	   G__PARANORMAL==var->reftype[G__asm_inst[pc+10]])) {
    ig15 = G__asm_inst[pc+10];
    /********************************************************************
     * 0 G__LD_VAR,LVAR                    G__LDST_VAR_INDEX_OPR
     * 1 index                             *int1
     * 2 paran == 0                        *int2
     * 3 pointer_level == p                opr +,-
     * 4 var_array pointer                 (*p2f)(buf,p2p,0,var2,index2)
     * 5 G__LD                             index3
     * 6 stack address 'i'                 pc increment
     * 7 OP2                               local_global
     * 8 +,-                               var_array3 pointer
     * 9 G__LD_VAR,LVAR
     *10 index3
     *11 paran == 1
     *12 point_level == p
     *13 var_array3 pointer
     ********************************************************************/
    if(G__LD_VAR==G__asm_inst[pc+9] || G__LD_LVAR==G__asm_inst[pc+9]) {
      int flag;
#ifndef G__OLDIMPLEMENTATION822
      long *pi2 = &(G__asm_stack[G__asm_inst[pc+6]].obj.i);
      int  *pix;
      if(G__ASM_FUNC_COMPILE==G__asm_wholefunction) {
	if(*pi2>=G__MAXINDEXCONST||*pi2<0) return(done);
	else pix = &G__indexconst[*pi2];
      }
      else {
	pix = (int*)pi2;
	if(sizeof(long)>sizeof(int)) *pix = (int)(*pi2);
      }
#endif
      if(G__LD_LVAR==G__asm_inst[pc]) flag=1;
      else                            flag=0;
#ifndef G__OLDIMPLEMENTATION822
      if(G__LD_LVAR==G__asm_inst[pc+9]) flag |= 4;
#else
      if(G__LD_VAR==G__asm_inst[pc+9]) flag |= 3;
#endif
      if(0==G__get_LD_p1_p2f(var->type[ig15],&G__asm_inst[pc+4])) {
#ifdef G__ASM_DBG
	if(G__asm_dbg)
	  G__fprinterr(G__serr,
		  "Error: LD_VAR,LD,OP2,LD_VAR[1] optimize error %s\n"
		  ,var->varnamebuf[ig15]);
#endif
      }
      else {
	done=1;
	G__asm_inst[pc+7] = flag;
	G__asm_inst[pc] = G__LDST_VAR_INDEX_OPR;
	G__asm_inst[pc+1] = (long)pi;
#ifndef G__OLDIMPLEMENTATION822
	G__asm_inst[pc+2] = (long)pix;
#else
	G__asm_inst[pc+2] = (long)(&(G__asm_stack[G__asm_inst[pc+6]].obj.i));
	if(sizeof(long)>sizeof(int)) { /* long to int conversion */
	  *(int*)G__asm_inst[pc+2]=(int)G__asm_stack[G__asm_inst[pc+6]].obj.i;
	}
#endif
	G__asm_inst[pc+3] = G__asm_inst[pc+8];
	G__asm_inst[pc+5] = G__asm_inst[pc+10];
	G__asm_inst[pc+6] = 14;
	G__asm_inst[pc+8] = G__asm_inst[pc+13];
	*ppc = pc+9; /* other 2 is incremented one level up */
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"LDST_VAR_INDEX_OPR (3) optimized\n");
#endif
      }
    }
    /********************************************************************
     * 0 G__LD_VAR,LVAR                    G__LDST_VAR_INDEX_OPR
     * 1 index                             *int1
     * 2 paran == 0                        *int2
     * 3 pointer_level == p                opr +,-
     * 4 var_array pointer                 (*p2f)(buf,p2p,0,var2,index2)
     * 5 G__LD                             index3
     * 6 stack address 'i'                 pc increment
     * 7 OP2                               local_global
     * 8 +,-                               var_array3 pointer
     * 9 G__ST_VAR,LVAR
     *10 index3
     *11 paran == 1
     *12 point_level == p
     *13 var_array3 pointer
     ********************************************************************/
    else if(G__ST_VAR==G__asm_inst[pc+9] || G__ST_LVAR==G__asm_inst[pc+9]) {
      int flag;
#ifndef G__OLDIMPLEMENTATION822
      long *pi2 = &(G__asm_stack[G__asm_inst[pc+6]].obj.i);
      int  *pix;
      if(G__ASM_FUNC_COMPILE==G__asm_wholefunction) {
	if(*pi2>=G__MAXINDEXCONST||*pi2<0) return(done);
	else pix = &G__indexconst[*pi2];
      }
      else {
	pix = (int*)pi2;
	if(sizeof(long)>sizeof(int)) *pix = (int)(*pi2);
      }
#endif
      if(G__LD_LVAR==G__asm_inst[pc]) flag=1;
      else                            flag=0;
#ifndef G__OLDIMPLEMENTATION822
      if(G__ST_LVAR==G__asm_inst[pc+9]) flag |= 4;
#else
      if(G__ST_VAR==G__asm_inst[pc+9]) flag |= 3;
#endif
      if(0==G__get_ST_p1_p2f(var->type[ig15],&G__asm_inst[pc+4])) {
#ifdef G__ASM_DBG
	if(G__asm_dbg)
	  G__fprinterr(G__serr,
		  "Error: LD_VAR,LD,OP2,ST_VAR[1] optimize error %s\n"
		  ,var->varnamebuf[ig15]);
#endif
      }
      else {
	done=1;
	G__asm_inst[pc+7] = flag;
	G__asm_inst[pc] = G__LDST_VAR_INDEX_OPR;
	G__asm_inst[pc+1] = (long)pi;
#ifndef G__OLDIMPLEMENTATION822
	G__asm_inst[pc+2] = (long)pix;
#else
	G__asm_inst[pc+2] = (long)(&(G__asm_stack[G__asm_inst[pc+6]].obj.i));
	if(sizeof(long)>sizeof(int)) { /* long to int conversion */
	  *(int*)G__asm_inst[pc+2]=(int)G__asm_stack[G__asm_inst[pc+6]].obj.i;
	}
#endif
	G__asm_inst[pc+3] = G__asm_inst[pc+8];
	G__asm_inst[pc+5] = G__asm_inst[pc+10];
	G__asm_inst[pc+6] = 14;
	G__asm_inst[pc+8] = G__asm_inst[pc+13];
	*ppc = pc+9; /* other 2 is incremented one level up */
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"LDST_VAR_INDEX_OPR (4) optimized\n");
#endif
      }
    }
  }

  /********************************************************************
   * 0 G__LD_VAR,LVAR                    G__LDST_VAR_INDEX_OPR
   * 1 index                             *int1
   * 2 paran == 0                        *int2
   * 3 pointer_level == p                opr +,-
   * 4 var_array pointer                 (*p2f)(buf,p2p,0,var2,index2)
   * 5 G__LD_VAR,LvAR                    index3
   * 6 index2                            pc increment
   * 7 paran == 0                        not use
   * 8 point_level == p                  var_array3 pointer
   * 9 var_array2 pointer
   *10 OP2
   *11 +,-
   *12 G__LD_VAR,LvAR
   *13 index3
   *14 paran == 1
   *15 point_level == p
   *16 var_array3 pointer
   ********************************************************************/

  /********************************************************************
   * 0 G__LD_VAR,LVAR                    G__LDST_VAR_INDEX_OPR
   * 1 index                             *int1
   * 2 paran == 0                        *int2
   * 3 pointer_level == p                opr +,-
   * 4 var_array pointer                 (*p2f)(buf,p2p,0,var2,index2)
   * 5 G__LD_VAR,LvAR                    index3
   * 6 index2                            pc increment
   * 7 paran == 0                        not use
   * 8 point_level == p                  var_array3 pointer
   * 9 var_array2 pointer
   *10 OP2
   *11 +,-
   *12 G__ST_VAR,LVAR
   *13 index3
   *14 paran == 1
   *15 point_level == p
   *16 var_array3 pointer
   ********************************************************************/

  return(done);
}

/*************************************************************************
* G__LD_int_optimize()
*************************************************************************/
int G__LD_int_optimize(ppc,pi)
int *ppc;
int *pi;
{
  struct G__var_array *var;
  int ig15;
  int done=0;
  int pc;
  pc = *ppc;

  /********************************************************************
   * 0 G__LD                             G__LD_VAR_INDEX
   * 1 stack address 'i'                 *arrayindex
   * 2 G__LD_VAR,LVAR                    (*p2f)(buf,psp,0,var,index)
   * 3 index                             index
   * 4 paran == 1                        pc increment
   * 5 point_level == p                  local_global
   * 6 var_array pointer                 var_array pointer
   ********************************************************************/
  if((G__LD_VAR==G__asm_inst[pc+2] || G__LD_LVAR==G__asm_inst[pc+2]) && 
     1==G__asm_inst[pc+4] &&
     'p' == G__asm_inst[pc+5] &&
     (var = (struct G__var_array*)G__asm_inst[pc+6]) &&
     1 == var->paran[G__asm_inst[pc+3]] &&
     (islower(var->type[G__asm_inst[pc+3]])||
      G__PARANORMAL==var->reftype[G__asm_inst[pc+3]])
#ifndef G__OLDIMPLEMENTATION876
     && (pc<4 || G__JMP!=G__asm_inst[pc-2] || G__asm_inst[pc-1]!=pc+2)
#endif
     ) {
    int flag;
#ifndef G__OLDIMPLEMENTATION822
    if(G__ASM_FUNC_COMPILE==G__asm_wholefunction) {
      if(*pi>=G__MAXINDEXCONST||*pi<0) return(done);
      else pi = &G__indexconst[*pi];
    }
#endif
    if(G__LD_LVAR==G__asm_inst[pc+2]) flag = 2;
    else                              flag = 0;
    done = 1;
    ig15 = G__asm_inst[pc+3];
    if(0==G__get_LD_p1_p2f(var->type[ig15],&G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
      if(G__asm_dbg)
	G__fprinterr(G__serr,"Error: LD,LD_VAR[1] optimize error %s\n"
		,var->varnamebuf[ig15]);
#endif
    }
    else {
      done=1;
      G__asm_inst[pc+5] = flag;
      G__asm_inst[pc] = G__LDST_VAR_INDEX;
      G__asm_inst[pc+1] = (long)pi;
      if(sizeof(long)>sizeof(int)) { /* long to int conversion */
	*(int*)G__asm_inst[pc+1]= (int)(*(long*)pi);
      }
      G__asm_inst[pc+4] = 7;
      *ppc = pc+5; /* other 2 is incremented one level up */
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"LDST_VAR_INDEX (5) optimized\n");
#endif
    }
  }

  /********************************************************************
   * 0 G__LD                             G__LDST_VAR_INDEX
   * 1 stack address 'i'                 *arrayindex
   * 2 G__ST_VAR,LvAR                    (*p2f)(buf,psp,0,var,index)
   * 3 index                             index
   * 4 paran == 1                        pc increment
   * 5 point_level == p                  flag &1:param_lcocal,&2:array_local
   * 6 var_array pointer                 var_array pointer
   ********************************************************************/
  else if((G__ST_VAR==G__asm_inst[pc+2] || G__ST_LVAR==G__asm_inst[pc+2]) &&
	  1==G__asm_inst[pc+4] &&
	  'p' == G__asm_inst[pc+5] &&
	  (var = (struct G__var_array*)G__asm_inst[pc+6]) &&
	  1 == var->paran[G__asm_inst[pc+3]] &&
	  1 == var->paran[G__asm_inst[pc+3]] &&
	  (islower(var->type[G__asm_inst[pc+3]])||
	   G__PARANORMAL==var->reftype[G__asm_inst[pc+3]])
#ifndef G__OLDIMPLEMENTATION876
	  && (pc<4 || G__JMP!=G__asm_inst[pc-2] || G__asm_inst[pc-1]!=pc+2)
#endif
	  ) {
    int flag;
#ifndef G__OLDIMPLEMENTATION822
    if(G__ASM_FUNC_COMPILE==G__asm_wholefunction) {
      if(*pi>=G__MAXINDEXCONST||*pi<0) return(done);
      else pi = &G__indexconst[*pi];
    }
#endif
    if(G__ST_LVAR==G__asm_inst[pc+2]) flag = 2;
    else                              flag = 0;
    ig15 = G__asm_inst[pc+3];
    if(0==G__get_ST_p1_p2f(var->type[ig15],&G__asm_inst[pc+2])) {
#ifdef G__ASM_DBG
      if(G__asm_dbg)
	G__fprinterr(G__serr,"Error: LD,ST_VAR[1] optimize error %s\n"
		,var->varnamebuf[ig15]);
#endif
    }
    else {
      done=1;
      G__asm_inst[pc+5] = flag;
      G__asm_inst[pc] = G__LDST_VAR_INDEX;
      G__asm_inst[pc+1] = (long)pi;
      if(sizeof(long)>sizeof(int)) { /* long to int conversion */
	*(int*)G__asm_inst[pc+1]= (int)(*(long*)pi);
      }
      G__asm_inst[pc+4] = 7;
      *ppc = pc+5; /* other 2 is incremented one level up */
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"LDST_VAR_INDEX (6) optimized\n");
#endif
    }
  }

  return(done);
}


/*************************************************************************
* G__CMP2_optimize()
*************************************************************************/
int G__CMP2_optimize(pc)
int pc;
{
  G__asm_inst[pc] = G__OP2_OPTIMIZED;
  switch(G__asm_inst[pc+1]) {
  case 'E': /* == */
    G__asm_inst[pc+1] = (long)G__CMP2_equal;
    break;
  case 'N': /* != */
    G__asm_inst[pc+1] = (long)G__CMP2_notequal;
    break;
  case 'G': /* >= */
    G__asm_inst[pc+1] = (long)G__CMP2_greaterorequal;
    break;
  case 'l': /* <= */
    G__asm_inst[pc+1] = (long)G__CMP2_lessorequal;
    break;
  case '<': /* <  */
    G__asm_inst[pc+1] = (long)G__CMP2_less;
    break;
  case '>': /* >  */
    G__asm_inst[pc+1] = (long)G__CMP2_greater;
    break;
  }
  return(0);
}

/*************************************************************************
* G__OP2_optimize()
*************************************************************************/
int G__OP2_optimize(pc)
int pc;
{
  int done=1;
  switch(G__asm_inst[pc+1]) {
  case '+':
    G__asm_inst[pc+1] = (long)G__OP2_plus;
    break;
  case '-':
    G__asm_inst[pc+1] = (long)G__OP2_minus;
    break;
  case '*':
    G__asm_inst[pc+1] = (long)G__OP2_multiply;
    break;
  case '/':
    G__asm_inst[pc+1] = (long)G__OP2_divide;
    break;
  case '%':
    G__asm_inst[pc+1] = (long)G__OP2_modulus;
    break;
/*
  case '&':
    G__asm_inst[pc+1] = (long)G__OP2_bitand;
    break;
  case '|':
    G__asm_inst[pc+1] = (long)G__OP2_bitand;
    break;
  case '^':
    G__asm_inst[pc+1] = (long)G__OP2_exor;
    break;
  case '~':
    G__asm_inst[pc+1] = (long)G__OP2_bininv;
    break;
*/
  case 'A':
    G__asm_inst[pc+1] = (long)G__OP2_logicaland;
    break;
  case 'O':
    G__asm_inst[pc+1] = (long)G__OP2_logicalor;
    break;
  case '>':
    G__asm_inst[pc+1] = (long)G__CMP2_greater;
    break;
  case '<':
    G__asm_inst[pc+1] = (long)G__CMP2_less;
    break;
/*
  case 'R':
    G__asm_inst[pc+1] = (long)G__OP2_rightshift;
    break;
  case 'L':
    G__asm_inst[pc+1] = (long)G__OP2_leftshift;
    break;
  case '@':
    G__asm_inst[pc+1] = (long)G__OP2_power;
    break;
*/
  case 'E':
    G__asm_inst[pc+1] = (long)G__CMP2_equal;
    break;
  case 'N':
    G__asm_inst[pc+1] = (long)G__CMP2_notequal;
    break;
  case 'G':
    G__asm_inst[pc+1] = (long)G__CMP2_greaterorequal;
    break;
  case 'l':
    G__asm_inst[pc+1] = (long)G__CMP2_lessorequal;
    break;
  case G__OPR_ADDASSIGN:
    G__asm_inst[pc+1] = (long)G__OP2_addassign;
    break;
  case G__OPR_SUBASSIGN:
    G__asm_inst[pc+1] = (long)G__OP2_subassign;
    break;
  case G__OPR_MODASSIGN:
    G__asm_inst[pc+1] = (long)G__OP2_modassign;
    break;
  case G__OPR_MULASSIGN:
    G__asm_inst[pc+1] = (long)G__OP2_mulassign;
    break;
  case G__OPR_DIVASSIGN:
    G__asm_inst[pc+1] = (long)G__OP2_divassign;
    break;
#ifndef G__OLDIMPLEMENTATION1491
  case G__OPR_ADD_UU:
    G__asm_inst[pc+1] = (long)G__OP2_plus_uu;
    break;
  case G__OPR_SUB_UU:
    G__asm_inst[pc+1] = (long)G__OP2_minus_uu;
    break;
  case G__OPR_MUL_UU:
    G__asm_inst[pc+1] = (long)G__OP2_multiply_uu;
    break;
  case G__OPR_DIV_UU:
    G__asm_inst[pc+1] = (long)G__OP2_divide_uu;
    break;
  case G__OPR_ADDASSIGN_UU:
    G__asm_inst[pc+1] = (long)G__OP2_addassign_uu;
    break;
  case G__OPR_SUBASSIGN_UU:
    G__asm_inst[pc+1] = (long)G__OP2_subassign_uu;
    break;
  case G__OPR_MULASSIGN_UU:
    G__asm_inst[pc+1] = (long)G__OP2_mulassign_uu;
    break;
  case G__OPR_DIVASSIGN_UU:
    G__asm_inst[pc+1] = (long)G__OP2_divassign_uu;
    break;
#endif
  case G__OPR_ADD_II:
    G__asm_inst[pc+1] = (long)G__OP2_plus_ii;
    break;
  case G__OPR_SUB_II:
    G__asm_inst[pc+1] = (long)G__OP2_minus_ii;
    break;
  case G__OPR_MUL_II:
    G__asm_inst[pc+1] = (long)G__OP2_multiply_ii;
    break;
  case G__OPR_DIV_II:
    G__asm_inst[pc+1] = (long)G__OP2_divide_ii;
    break;
  case G__OPR_ADDASSIGN_II:
    G__asm_inst[pc+1] = (long)G__OP2_addassign_ii;
    break;
  case G__OPR_SUBASSIGN_II:
    G__asm_inst[pc+1] = (long)G__OP2_subassign_ii;
    break;
  case G__OPR_MULASSIGN_II:
    G__asm_inst[pc+1] = (long)G__OP2_mulassign_ii;
    break;
  case G__OPR_DIVASSIGN_II:
    G__asm_inst[pc+1] = (long)G__OP2_divassign_ii;
    break;
  case G__OPR_ADD_DD:
    G__asm_inst[pc+1] = (long)G__OP2_plus_dd;
    break;
  case G__OPR_SUB_DD:
    G__asm_inst[pc+1] = (long)G__OP2_minus_dd;
    break;
  case G__OPR_MUL_DD:
    G__asm_inst[pc+1] = (long)G__OP2_multiply_dd;
    break;
  case G__OPR_DIV_DD:
    G__asm_inst[pc+1] = (long)G__OP2_divide_dd;
    break;
  case G__OPR_ADDASSIGN_DD:
    G__asm_inst[pc+1] = (long)G__OP2_addassign_dd;
    break;
  case G__OPR_SUBASSIGN_DD:
    G__asm_inst[pc+1] = (long)G__OP2_subassign_dd;
    break;
  case G__OPR_MULASSIGN_DD:
    G__asm_inst[pc+1] = (long)G__OP2_mulassign_dd;
    break;
  case G__OPR_DIVASSIGN_DD:
    G__asm_inst[pc+1] = (long)G__OP2_divassign_dd;
    break;
  case G__OPR_ADDASSIGN_FD:
    G__asm_inst[pc+1] = (long)G__OP2_addassign_fd;
    break;
  case G__OPR_SUBASSIGN_FD:
    G__asm_inst[pc+1] = (long)G__OP2_subassign_fd;
    break;
  case G__OPR_MULASSIGN_FD:
    G__asm_inst[pc+1] = (long)G__OP2_mulassign_fd;
    break;
  case G__OPR_DIVASSIGN_FD:
    G__asm_inst[pc+1] = (long)G__OP2_divassign_fd;
    break;
#ifndef G__OLDIMPLEMENTATION2129
  case G__OPR_ADDVOIDPTR:
    G__asm_inst[pc+1] = (long)G__OP2_addvoidptr;
    break;
#endif
  default:
    done=0;
    break;
  }
  if(done) G__asm_inst[pc] = G__OP2_OPTIMIZED;
  return(0);
}


/*************************************************************************
* G__asm_optimze3()
*************************************************************************/
int G__asm_optimize3(start)
int *start;
{
  int pc;               /* instruction program counter */
  int illegal=0;
  struct G__var_array *var;
  int ig15;
  int paran;
  int var_type;

#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    G__fprinterr(G__serr,"Optimize 3 start\n");
  }
#endif

  pc = *start;

  while(pc<G__MAXINST) {

    switch(G__INST(G__asm_inst[pc])) {

    case G__LDST_VAR_P:
      /***************************************
      * inst
      * 0 G__LDST_VAR_P
      * 1 index
      * 2 void (*f)(pbuf,psp,offset,p,ctype,
      * 3 (not use)
      * 4 var_array pointer
      * stack
      * sp          <-
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	var = (struct G__var_array*)G__asm_inst[pc+4];
	G__fprinterr(G__serr,"%3lx: LDST_VAR_P index=%ld %s\n"
		,pc,G__asm_inst[pc+1]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
#endif
      pc+=5;
      break;

    case G__LDST_MSTR_P:
      /***************************************
      * inst
      * 0 G__LDST_MSTR_P
      * 1 index
      * 2 void (*f)(pbuf,offset,psp,p,ctype,
      * 3 (not use)
      * 4 var_array pointer
      * stack
      * sp          <-
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	var = (struct G__var_array*)G__asm_inst[pc+4];
	G__fprinterr(G__serr,"%3lx: LDST_MSTR_P index=%d %s\n"
		,pc,G__asm_inst[pc+1]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
#endif
      pc+=5;
      break;

    case G__LDST_VAR_INDEX:
      /***************************************
      * inst
      * 0 G__LDST_VAR_INDEX
      * 1 *arrayindex
      * 2 void (*f)(pbuf,psp,offset,p,ctype,
      * 3 index
      * 4 pc increment
      * 5 not use
      * 6 var_array pointer
      * stack
      * sp          <-
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	var = (struct G__var_array*)G__asm_inst[pc+6];
	G__fprinterr(G__serr,"%3lx: LDST_VAR_INDEX index=%d %s\n"
		,pc,G__asm_inst[pc+3]
		,var->varnamebuf[G__asm_inst[pc+3]]);
      }
#endif
      pc+=G__asm_inst[pc+4];
      break;

    case G__LDST_VAR_INDEX_OPR:
      /***************************************
      * inst
      * 0 G__LDST_VAR_INDEX_OPR
      * 1 *int1
      * 2 *int2
      * 3 opr +,-
      * 4 void (*f)(pbuf,psp,offset,p,ctype,
      * 5 index
      * 6 pc increment
      * 7 not use
      * 8 var_array pointer
      * stack
      * sp          <-
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	var = (struct G__var_array*)G__asm_inst[pc+8];
	G__fprinterr(G__serr,"%3lx: LDST_VAR_INDEX_OPR index=%d %s\n"
		,pc,G__asm_inst[pc+5]
		,var->varnamebuf[G__asm_inst[pc+5]]);
      }
#endif
      pc+=G__asm_inst[pc+6];
      break;

    case G__OP2_OPTIMIZED:
      /***************************************
      * inst
      * 0 OP2_OPTIMIZED
      * 1 (*p2f)(buf,buf)
      * stack
      * sp-2  a
      * sp-1  a         <-
      * sp    G__null
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: OP2_OPTIMIZED \n",pc);
#endif
      pc+=2;
      break;

    case G__OP1_OPTIMIZED:
      /***************************************
      * inst
      * 0 OP1_OPTIMIZED
      * 1 (*p2f)(buf)
      * stack
      * sp-1  a
      * sp    G__null     <-
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: OP1_OPTIMIZED \n",pc);
#endif
      pc+=2;
      break;

    case G__LD_VAR:
      /***************************************
      * inst
      * 0 G__LD_VAR
      * 1 index
      * 2 paran
      * 3 point_level
      * 4 var_array pointer
      * stack
      * sp-paran+1      <- sp-paran+1
      * sp-2
      * sp-1
      * sp
      ***************************************/
      var = (struct G__var_array*)G__asm_inst[pc+4];
      ig15 = G__asm_inst[pc+1];
      paran = G__asm_inst[pc+2];
      var_type = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3lx: LD_VAR index=%d paran=%d point %c %s\n"
		,pc,G__asm_inst[pc+1],G__asm_inst[pc+2]
		,G__asm_inst[pc+3]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
#endif
      /* need optimization */
      if('p'==var_type &&
	 (islower(var->type[ig15])||G__PARANORMAL==var->reftype[ig15])) {
	if(0==paran && 0==var->paran[ig15]) {
	  if('i'==var->type[ig15]) {
	    if(0==G__LD_VAR_int_optimize(&pc,(int*)var->p[ig15]))
	      G__LD_p0_optimize(var,ig15,pc,G__LDST_VAR_P);
	  }
	  else {
	    G__LD_p0_optimize(var,ig15,pc,G__LDST_VAR_P);
	  }
	}
	else if(1==paran && 1==var->paran[ig15]) {
	  G__LD_p1_optimize(var,ig15,pc,G__LDST_VAR_P);
	}
#ifndef G__OLDIMPLEMENTATION1405
	else if(paran==var->paran[ig15]) {
	  G__LD_pn_optimize(var,ig15,pc,G__LDST_VAR_P);
	}
#endif
	else if(1==paran && 0==var->paran[ig15] && isupper(var->type[ig15])) {
	  G__LD_P10_optimize(var,ig15,pc,G__LDST_VAR_P);
	}
      }
      pc+=5;
      break;

    case G__LD:
      /***************************************
      * inst
      * 0 G__LD
      * 1 address in data stack
      * stack
      * sp    a
      * sp+1             <-
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: LD %g from %x \n"
			     ,pc
			     ,G__double(G__asm_stack[G__asm_inst[pc+1]])
			     ,G__asm_inst[pc+1]);
#endif
      /* no optimize */
      if('i'==G__asm_stack[G__asm_inst[pc+1]].type) {
	G__LD_int_optimize(&pc,(int*)(&(G__asm_stack[G__asm_inst[pc+1]].obj.i)));
      }
      pc+=2;
      break;

    case G__CL:
      /***************************************
      * 0 CL
      *  clear stack pointer
      ***************************************/
#ifdef G__ASM_DBG
#ifndef G__OLDIMPLEMENTATION2132
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: CL %s:%d\n",pc
		 ,G__srcfile[G__asm_inst[pc+1]/G__CL_FILESHIFT].filename
				  ,G__asm_inst[pc+1]&G__CL_LINEMASK);
#else
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: CL %d\n",pc,G__asm_inst[pc+1]);
#endif
#endif
      /* no optimize */
      pc+=2;
      break;

    case G__OP2:
      /***************************************
      * inst
      * 0 OP2
      * 1 (+,-,*,/,%,@,>>,<<,&,|)
      * stack
      * sp-2  a
      * sp-1  b          <-
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(isprint(G__asm_inst[pc+1])) {
	if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: OP2 '%c'%d \n" ,pc
		,G__asm_inst[pc+1],G__asm_inst[pc+1]);
      }
      else {
	if(G__asm_dbg)
	  G__fprinterr(G__serr,"%3lx: OP2 %d \n",pc,G__asm_inst[pc+1]);
      }
#endif
      /* need optimization */
      G__OP2_optimize(pc);
      pc+=2;
      break;

    case G__ST_VAR:
      /***************************************
      * inst
      * 0 G__ST_VAR
      * 1 index
      * 2 paran
      * 3 point_level
      * 4 var_array pointer
      * stack
      * sp-paran        <- sp-paran
      * sp-2
      * sp-1
      * sp
      ***************************************/
      var = (struct G__var_array*)G__asm_inst[pc+4];
      ig15 = G__asm_inst[pc+1];
      paran = G__asm_inst[pc+2];
      var_type = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3lx: ST_VAR index=%d paran=%d point %c %s\n"
		,pc,G__asm_inst[pc+1],G__asm_inst[pc+2]
		,G__asm_inst[pc+3]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
#endif
      /* need optimization */
      if(('p'==var_type || var_type == var->type[ig15])&&
	 (islower(var->type[ig15])||G__PARANORMAL==var->reftype[ig15])) {
	if(0==paran && 0==var->paran[ig15]) {
	  G__ST_p0_optimize(var,ig15,pc,G__LDST_VAR_P);
	}
	else if(1==paran && 1==var->paran[ig15]) {
	  G__ST_p1_optimize(var,ig15,pc,G__LDST_VAR_P);
	}
#ifndef G__OLDIMPLEMENTATION1406
	else if(paran==var->paran[ig15]) {
	  G__ST_pn_optimize(var,ig15,pc,G__LDST_VAR_P);
	}
#endif
	else if(1==paran && 0==var->paran[ig15] && isupper(var->type[ig15])) {
	  G__ST_P10_optimize(var,ig15,pc,G__LDST_VAR_P);
	}
      }
      pc+=5;
      break;

    case G__LD_MSTR:
      /***************************************
      * inst
      * 0 G__LD_MSTR
      * 1 index
      * 2 paran
      * 3 point_level
      * 4 *structmem
      * stack
      * sp-paran+1      <- sp-paran+1
      * sp-2
      * sp-1
      * sp
      ***************************************/
      var = (struct G__var_array*)G__asm_inst[pc+4];
      ig15 = G__asm_inst[pc+1];
      paran = G__asm_inst[pc+2];
      var_type = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3lx: LD_MSTR index=%d paran=%d point %c %s\n"
		,pc,G__asm_inst[pc+1],G__asm_inst[pc+2]
		,G__asm_inst[pc+3]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
#endif
      /* need optimization */
      if('p'==var_type &&
	 (islower(var->type[ig15])||G__PARANORMAL==var->reftype[ig15])) {
	long inst;
	if(G__LOCALSTATIC==var->statictype[ig15]) inst = G__LDST_VAR_P;
	else                                      inst = G__LDST_MSTR_P;
	if(0==paran && 0==var->paran[ig15]) {
	  G__LD_p0_optimize(var,ig15,pc,inst);
	}
	else if(1==paran && 1==var->paran[ig15]) {
	  G__LD_p1_optimize(var,ig15,pc,inst);
	}
#ifndef G__OLDIMPLEMENTATION1405
	else if(paran==var->paran[ig15]) {
	  G__LD_pn_optimize(var,ig15,pc,inst);
	}
#endif
	else if(1==paran && 0==var->paran[ig15] && isupper(var->type[ig15])) {
	  G__LD_P10_optimize(var,ig15,pc,G__LDST_MSTR_P);
	}
      }
      pc+=5;
      break;

    case G__CMPJMP:
      /***************************************
      * 0 CMPJMP
      * 1 *G__asm_test_X()
      * 2 *a
      * 3 *b
      * 4 next_pc
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3lx: CMPJMP (0x%lx)%d (0x%lx)%d to %lx\n"
		,pc
		,G__asm_inst[pc+2],*(int *)G__asm_inst[pc+2]
		,G__asm_inst[pc+3],*(int *)G__asm_inst[pc+3]
		,G__asm_inst[pc+4]);
      }
#endif
      /* no optmization */
      pc+=5;
      break;

    case G__PUSHSTROS:
      /***************************************
      * inst
      * 0 G__PUSHSTROS
      * stack
      * sp           <- sp-paran
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: PUSHSTROS\n" ,pc);
#endif
      /* no optmization */
      ++pc;
      break;

    case G__SETSTROS:
      /***************************************
      * inst
      * 0 G__SETSTROS
      * stack
      * sp-1         <- sp-paran
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: SETSTROS\n",pc);
#endif
      /* no optmization */
      ++pc;
      break;

    case G__POPSTROS:
      /***************************************
      * inst
      * 0 G__POPSTROS
      * stack
      * sp           <- sp-paran
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: POPSTROS\n" ,pc);
#endif
      /* no optmization */
      ++pc;
      break;

    case G__ST_MSTR:
      /***************************************
      * inst
      * 0 G__ST_MSTR
      * 1 index
      * 2 paran
      * 3 point_level
      * 4 *structmem
      * stack
      * sp-paran        <- sp-paran
      * sp-2
      * sp-1
      * sp
      ***************************************/
      var = (struct G__var_array*)G__asm_inst[pc+4];
      ig15 = G__asm_inst[pc+1];
      paran = G__asm_inst[pc+2];
      var_type = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3lx: ST_MSTR index=%d paran=%d point %c %s\n"
		,pc,G__asm_inst[pc+1],G__asm_inst[pc+2]
		,G__asm_inst[pc+3]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
#endif
      /* need optimization */
      if('p'==var_type &&
	 (islower(var->type[ig15])||G__PARANORMAL==var->reftype[ig15])) {
	long inst;
	if(G__LOCALSTATIC==var->statictype[ig15]) inst = G__LDST_VAR_P;
	else                                      inst = G__LDST_MSTR_P;
	if(0==paran && 0==var->paran[ig15]) {
	  G__ST_p0_optimize(var,ig15,pc,inst);
	}
	else if(1==paran && 1==var->paran[ig15]) {
	  G__ST_p1_optimize(var,ig15,pc,inst);
	}
#ifndef G__OLDIMPLEMENTATION1406
	else if(paran==var->paran[ig15]) {
	  G__ST_pn_optimize(var,ig15,pc,inst);
	}
#endif
	else if(1==paran && 0==var->paran[ig15] && isupper(var->type[ig15])) {
	  G__ST_P10_optimize(var,ig15,pc,G__LDST_MSTR_P);
	}
      }
      pc+=5;
      break;

    case G__INCJMP:
      /***************************************
      * 0 INCJMP
      * 1 *cntr
      * 2 increment
      * 3 next_pc
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: INCJMP *(int*)0x%lx+%d to %x\n"
			     ,pc ,G__asm_inst[pc+1] ,G__asm_inst[pc+2]
			     ,G__asm_inst[pc+3]);
#endif
      /* no optimization */
      pc+=4;
      break;

    case G__CNDJMP:
      /***************************************
      * 0 CNDJMP   (jump if 0)
      * 1 next_pc
      * stack
      * sp-1         <-
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: CNDJMP to %x\n"
			     ,pc ,G__asm_inst[pc+1]);
#endif
      /* no optimization */
      pc+=2;
      break;

    case G__CMP2:
      /***************************************
      * 0 CMP2
      * 1 operator
      * stack
      * sp-1         <-
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3lx: CMP2 '%c' \n" ,pc ,G__asm_inst[pc+1]);
      }
#endif
      /* need optimization, but not high priority */
      G__CMP2_optimize(pc);
      pc+=2;
      break;

    case G__JMP:
      /***************************************
      * 0 JMP
      * 1 next_pc
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: JMP %x\n" ,pc,G__asm_inst[pc+1]);
#endif
      /* no optimization */
      pc+=2;
      break;

    case G__PUSHCPY:
      /***************************************
      * inst
      * 0 G__PUSHCPY
      * stack
      * sp
      * sp+1            <-
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: PUSHCPY\n",pc);
#endif
      /* no optimization */
      ++pc;
      break;

    case G__POP:
      /***************************************
      * inst
      * 0 G__POP
      * stack
      * sp-1            <-
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: POP\n" ,pc);
#endif
      /* no optimization */
      ++pc;
      break;

    case G__LD_FUNC:
      /***************************************
      * inst
      * 0 G__LD_FUNC
      * 1 *name
      * 2 hash
      * 3 paran
      * 4 (*func)()
      * stack
      * sp-paran+1      <- sp-paran+1
      * sp-2
      * sp-1
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_inst[pc+1]<G__MAXSTRUCT) {
	if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: LD_FUNC %s paran=%d\n" ,pc
		,"compiled",G__asm_inst[pc+3]);
      }
      else {
	if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: LD_FUNC %s paran=%d\n" ,pc
		,(char *)G__asm_inst[pc+1],G__asm_inst[pc+3]);
      }
#endif
      /* no optimization */
      pc+=5;
      break;

    case G__RETURN:
      /***************************************
      * 0 RETURN
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: RETURN\n" ,pc);
#endif
      /* no optimization */
      pc++;
      return(0);
      break;

    case G__CAST:
      /***************************************
      * 0 CAST
      * 1 type
      * 2 typenum
      * 3 tagnum
      * 4 reftype 
      * stack
      * sp-1    <- cast on this
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3lx: CAST to %c type%d tag%d\n" ,pc
		,(char)G__asm_inst[pc+1],G__asm_inst[pc+2],G__asm_inst[pc+3]);
      }
#endif
      /* need optimization */
      pc+=5;
      break;

    case G__OP1:
      /***************************************
      * inst
      * 0 OP1
      * 1 (+,-)
      * stack
      * sp-1  a
      * sp    G__null     <-
      ***************************************/
#ifdef G__ASM_DBG
      if(isprint(G__asm_inst[pc+1])){
	if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: OP1 '%c'%d\n",pc
		,G__asm_inst[pc+1],G__asm_inst[pc+1] );
      }
      else {
	if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: OP1 %d\n",pc,G__asm_inst[pc+1]);
      }
#endif
      /* need optimization */
      switch(G__asm_inst[pc+1]) {
#ifndef G__OLDIMPLEMENTATION578

      case G__OPR_POSTFIXINC_I:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixinc_i;
	break;
      case G__OPR_POSTFIXDEC_I:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixdec_i;
	break;
      case G__OPR_PREFIXINC_I:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixinc_i;
	break;
      case G__OPR_PREFIXDEC_I:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixdec_i;
	break;

      case G__OPR_POSTFIXINC_D:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixinc_d;
	break;
      case G__OPR_POSTFIXDEC_D:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixdec_d;
	break;
      case G__OPR_PREFIXINC_D:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixinc_d;
	break;
      case G__OPR_PREFIXDEC_D:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixdec_d;
	break;

#if G__NEVER /* following change rather slowed down */
      case G__OPR_POSTFIXINC_S:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixinc_s;
	break;
      case G__OPR_POSTFIXDEC_S:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixdec_s;
	break;
      case G__OPR_PREFIXINC_S:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixinc_s;
	break;
      case G__OPR_PREFIXDEC_S:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixdec_s;
	break;

      case G__OPR_POSTFIXINC_L:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixinc_l;
	break;
      case G__OPR_POSTFIXDEC_L:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixdec_l;
	break;
      case G__OPR_PREFIXINC_L:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixinc_l;
	break;
      case G__OPR_PREFIXDEC_L:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixdec_l;
	break;

      case G__OPR_POSTFIXINC_H:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixinc_h;
	break;
      case G__OPR_POSTFIXDEC_H:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixdec_h;
	break;
      case G__OPR_PREFIXINC_H:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixinc_h;
	break;
      case G__OPR_PREFIXDEC_H:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixdec_h;
	break;

      case G__OPR_POSTFIXINC_R:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixinc_r;
	break;
      case G__OPR_POSTFIXDEC_R:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixdec_r;
	break;
      case G__OPR_PREFIXINC_R:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixinc_r;
	break;
      case G__OPR_PREFIXDEC_R:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixdec_r;
	break;

      case G__OPR_POSTFIXINC_K:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixinc_k;
	break;
      case G__OPR_POSTFIXDEC_K:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixdec_k;
	break;
      case G__OPR_PREFIXINC_K:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixinc_k;
	break;
      case G__OPR_PREFIXDEC_K:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixdec_k;
	break;


      case G__OPR_POSTFIXINC_F:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixinc_f;
	break;
      case G__OPR_POSTFIXDEC_F:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixdec_f;
	break;
      case G__OPR_PREFIXINC_F:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixinc_f;
	break;
      case G__OPR_PREFIXDEC_F:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixdec_f;
	break;
#endif
#endif
      case G__OPR_POSTFIXINC:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixinc;
	break;
      case G__OPR_POSTFIXDEC:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_postfixdec;
	break;
      case G__OPR_PREFIXINC:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixinc;
	break;
      case G__OPR_PREFIXDEC:
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_prefixdec;
	break;
      case '-':
	G__asm_inst[pc] = G__OP1_OPTIMIZED;
	G__asm_inst[pc+1] = (long)G__OP1_minus;
	break;
      }
      pc+=2;
      break;

    case G__LETVVAL:
      /***************************************
      * inst
      * 0 LETVVAL
      * stack
      * sp-2  a
      * sp-1  b          <-
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: LETVVAL\n" ,pc);
#endif
      /* no optimization */
      ++pc;
      break;

    case G__ADDSTROS:
      /***************************************
      * inst
      * 0 ADDSTROS
      * 1 addoffset
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3lx: ADDSTROS %d\n" ,pc,G__asm_inst[pc+1]);
      }
#endif
      /* no optimization */
      pc+=2;
      break;

    case G__LETPVAL:
      /***************************************
      * inst
      * 0 LETPVAL
      * stack
      * sp-2  a
      * sp-1  b          <-
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: LETPVAL\n" ,pc);
#endif
      /* no optimization */
      ++pc;
      break;

    case G__FREETEMP:
      /***************************************
      * 0 FREETEMP
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: FREETEMP\n" ,pc);
#endif
      /* no optimization */
      ++pc;
      break;

    case G__SETTEMP:
      /***************************************
      * 0 SETTEMP
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: SETTEMP\n" ,pc);
#endif
      /* no optimization */
      ++pc;
      break;


    case G__GETRSVD:
      /***************************************
      * 0 GETRSVD
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: GETRSVD\n" ,pc);
#endif
      /* no optimization */
      ++pc;
      break;

    case G__TOPNTR:
      /***************************************
      * inst
      * 0 LETVVAL
      * stack
      * sp-1  a          <-
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: TOPNTR\n" ,pc);
#endif
      /* no optimization */
      ++pc;
      break;

    case G__NOT:
      /***************************************
      * 0 NOT
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: NOT\n" ,pc);
#endif
      /* no optimization */
      ++pc;
      break;

#ifndef G__OLDIMPLEMENTATION1399
    case G__BOOL:
      /***************************************
      * 0 BOOL
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: BOOL\n" ,pc);
#endif
      /* no optimization */
      ++pc;
      break;
#endif

    case G__ISDEFAULTPARA:
      /***************************************
      * 0 ISDEFAULTPARA
      * 1 next_pc
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: !ISDEFAULTPARA JMP %x\n"
			     ,pc,G__asm_inst[pc+1]);
#endif
      pc+=2;
      /* no optimization */
      break;

#ifdef G__ASM_WHOLEFUNC
    case G__LDST_LVAR_P:
      /***************************************
      * inst
      * 0 G__LDST_LVAR_P
      * 1 index
      * 2 void (*f)(pbuf,psp,offset,p,ctype,
      * 3 (not use)
      * 4 var_array pointer
      * stack
      * sp          <-
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	var = (struct G__var_array*)G__asm_inst[pc+4];
	G__fprinterr(G__serr,"%3lx: LDST_LVAR_P index=%d %s\n"
		,pc,G__asm_inst[pc+1]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
#endif
      pc+=5;
      break;

    case G__LD_LVAR:
      /***************************************
      * inst
      * 0 G__LD_LVAR
      * 1 index
      * 2 paran
      * 3 point_level
      * 4 var_array pointer
      * stack
      * sp-paran+1      <- sp-paran+1
      * sp-2
      * sp-1
      * sp
      ***************************************/
      var = (struct G__var_array*)G__asm_inst[pc+4];
      ig15 = G__asm_inst[pc+1];
      paran = G__asm_inst[pc+2];
      var_type = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3lx: LD_LVAR index=%d paran=%d point %c %s\n"
		,pc,G__asm_inst[pc+1],G__asm_inst[pc+2]
		,G__asm_inst[pc+3]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
#endif
      /* need optimization */
      if(G__PARAREFERENCE==var->reftype[ig15]) {
	switch(var_type) {
	case 'P':
	  G__LD_RP0_optimize(var,ig15,pc,G__LDST_LVAR_P);
          break;
	case 'p':
	  G__LD_Rp0_optimize(var,ig15,pc,G__LDST_LVAR_P);
	  break;
	case 'v':
	  break;
	}
      } 
      else 
      if('p'==var_type &&
	 (islower(var->type[ig15])||G__PARANORMAL==var->reftype[ig15])) {
	long inst;
	if(G__LOCALSTATIC==var->statictype[ig15]) inst = G__LDST_VAR_P;
	else                                      inst = G__LDST_LVAR_P;
	if(0==paran && 0==var->paran[ig15]) {
	  if('i'==var->type[ig15]) {
	    if(0==G__LD_VAR_int_optimize(&pc,(int*)var->p[ig15]))
	      G__LD_p0_optimize(var,ig15,pc,inst);
	  }
	  else {
	    G__LD_p0_optimize(var,ig15,pc,inst);
	  }
	}
	else if(1==paran && 1==var->paran[ig15]) {
	  G__LD_p1_optimize(var,ig15,pc,inst);
	}
#ifndef G__OLDIMPLEMENTATION1405
	else if(paran==var->paran[ig15]) {
	  G__LD_pn_optimize(var,ig15,pc,inst);
	}
#endif
	else if(1==paran && 0==var->paran[ig15] && isupper(var->type[ig15])) {
	  G__LD_P10_optimize(var,ig15,pc,inst);
	}
      }
      pc+=5;
      break;

    case G__ST_LVAR:
      /***************************************
      * inst
      * 0 G__ST_LVAR
      * 1 index
      * 2 paran
      * 3 point_level
      * 4 var_array pointer
      * stack
      * sp-paran        <- sp-paran
      * sp-2
      * sp-1
      * sp
      ***************************************/
      var = (struct G__var_array*)G__asm_inst[pc+4];
      ig15 = G__asm_inst[pc+1];
      paran = G__asm_inst[pc+2];
      var_type = G__asm_inst[pc+3];
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3lx: ST_LVAR index=%d paran=%d point %c %s\n"
		,pc,G__asm_inst[pc+1],G__asm_inst[pc+2]
		,G__asm_inst[pc+3]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
#endif
      /* need optimization */
      if(G__PARAREFERENCE==var->reftype[ig15]) {
	switch(var_type) {
	case 'P':
          break;
	case 'p':
	  G__ST_Rp0_optimize(var,ig15,pc,G__LDST_LVAR_P);
	  break;
	case 'v':
	  break;
	}
      } 
      else 
      if(('p'==var_type || var_type == var->type[ig15]) &&
	 (islower(var->type[ig15])||G__PARANORMAL==var->reftype[ig15])) {
	long inst;
	if(G__LOCALSTATIC==var->statictype[ig15]) inst = G__LDST_VAR_P;
	else                                      inst = G__LDST_LVAR_P;
	if(0==paran && 0==var->paran[ig15]) {
	  G__ST_p0_optimize(var,ig15,pc,inst);
	}
	else if(1==paran && 1==var->paran[ig15]) {
	  G__ST_p1_optimize(var,ig15,pc,inst);
	}
#ifndef G__OLDIMPLEMENTATION1406
	else if(paran==var->paran[ig15]) {
	  G__ST_pn_optimize(var,ig15,pc,inst);
	}
#endif
	else if(1==paran && 0==var->paran[ig15] && isupper(var->type[ig15])) {
	  G__ST_P10_optimize(var,ig15,pc,inst);
	}
      }
      pc+=5;
      break;
#endif

    case G__REWINDSTACK:
      /***************************************
      * inst
      * 0 G__REWINDSTACK
      * 1 rewind
      * stack
      * sp-2            <-  ^
      * sp-1                | rewind
      * sp              <- ..
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3lx: REWINDSTACK %d\n" ,pc,G__asm_inst[pc+1]);
      }
#endif
      /* no optimization */
      pc+=2;
      break;

    case G__CND1JMP:
      /***************************************
      * 0 CND1JMP   (jump if 1)
      * 1 next_pc
      * stack
      * sp-1         <-
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3lx: CND1JMP  to %x\n" ,pc ,G__asm_inst[pc+1]);
      }
#endif
      /* no optimization */
      pc+=2;
      break;

#ifdef G__ASM_IFUNC
    case G__LD_IFUNC:
      /***************************************
      * inst
      * 0 G__LD_IFUNC
      * 1 *name
      * 2 hash          // unused
      * 3 paran
      * 4 p_ifunc
      * 5 funcmatch
      * 6 memfunc_flag
      * 7 index
      * stack
      * sp-paran+1      <- sp-paran+1
      * sp-2
      * sp-1
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: LD_IFUNC %s paran=%d\n" ,pc
			     ,(char *)G__asm_inst[pc+1],G__asm_inst[pc+3]);
#endif
      /* need optimization, later */
      pc+=8;
      break;

    case G__NEWALLOC:
      /***************************************
      * inst
      * 0 G__NEWALLOC
      * 1 size
      * stack
      * sp-1     <- pinc
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: NEWALLOC size(%d)\n"
			     ,pc,G__asm_inst[pc+1]);
#endif
      /* no optimization */
      pc+=3;
      break;

    case G__SET_NEWALLOC:
      /***************************************
      * inst
      * 0 G__SET_NEWALLOC
      * 1 tagnum
      * stack
      * sp-1        G__store_struct_offset
      * sp       <-
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: SET_NEWALLOC\n" ,pc);
#endif
      /* no optimization */
      pc+=3;
      break;

    case G__DELETEFREE:
      /***************************************
      * inst
      * 0 G__DELETEFREE
      * 1 isarray  0: simple free, 1: array, 2: virtual free
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: DELETEFREE\n",pc);
#endif
      /* no optimization */
      pc+=2;
      break;

    case G__SWAP:
      /***************************************
      * inst
      * 0 G__SWAP
      * stack
      * sp-2          sp-1
      * sp-1          sp-2
      * sp       <-   sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: SWAP\n",pc);
#endif
      /* no optimization */
      ++pc;
      break;

#endif /* G__ASM_IFUNC */

    case G__BASECONV:
      /***************************************
      * inst
      * 0 G__BASECONV
      * 1 formal_tagnum
      * 2 baseoffset
      * stack
      * sp-2          sp-1
      * sp-1          sp-2
      * sp       <-   sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: BASECONV %d %d\n",pc
			     ,G__asm_inst[pc+1],G__asm_inst[pc+2]);
#endif
      /* no optimization */
      pc+=3;
      break;

    case G__STORETEMP:
      /***************************************
      * 0 STORETEMP
      * stack
      * sp-1
      * sp       <-  sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: STORETEMP\n",pc);
#endif
      /* no optimization */
      ++pc;
      break;

    case G__ALLOCTEMP:
      /***************************************
      * 0 ALLOCTEMP
      * 1 tagnum
      * stack
      * sp-1
      * sp       <-  sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3lx: ALLOCTEMP %s\n",pc
		,G__struct.name[G__asm_inst[pc+1]]);
      }
#endif
      /* no optimization */
      pc+=2;
      break;

    case G__POPTEMP:
      /***************************************
      * 0 POPTEMP
      * 1 tagnum
      * stack
      * sp-1
      * sp      <-  sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	if(-1!=G__asm_inst[pc+1])
	  G__fprinterr(G__serr,"%3lx: POPTEMP %s\n" ,pc
		       ,G__struct.name[G__asm_inst[pc+1]]);
	else 
	  G__fprinterr(G__serr,"%3lx: POPTEMP -1\n" ,pc);
      }
#endif
      /* no optimization */
      pc+=2;
      break;

    case G__REORDER:
      /***************************************
      * 0 REORDER
      * 1 paran(total)
      * 2 ig25(arrayindex)
      * stack      paran=4 ig25=2    x y z w -> x y z w z w -> x y x y z w -> w z x y
      * sp-3    <-  sp-1
      * sp-2    <-  sp-3
      * sp-1    <-  sp-2
      * sp      <-  sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: REORDER paran=%d ig25=%d\n"
			     ,pc ,G__asm_inst[pc+1],G__asm_inst[pc+2]);
#endif
      /* no optimization */
      pc+=3;
      break;

    case G__LD_THIS:
      /***************************************
      * 0 LD_THIS
      * 1 point_level;
      * stack
      * sp-1
      * sp
      * sp+1   <-
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: LD_THIS %s\n"
			     ,pc ,G__struct.name[G__tagnum]);
#endif
      /* no optimization */
      pc+=2;
      break;

    case G__RTN_FUNC:
      /***************************************
      * 0 RTN_FUNC
      * 1 isreturnvalue
      * stack
      * sp-1   -> return this
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3lx: RTN_FUNC %d\n" ,pc ,G__asm_inst[pc+1]);
      }
#endif
      /* no optimization */
      pc+=2;
      break;

    case G__SETMEMFUNCENV:
      /***************************************
      * 0 SETMEMFUNCENV:
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: SETMEMFUNCENV\n",pc);
#endif
      /* no optimization */
      pc+=1;
      break;

    case G__RECMEMFUNCENV:
      /***************************************
      * 0 RECMEMFUNCENV:
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: RECMEMFUNCENV\n" ,pc);
#endif
      /* no optimization */
      pc+=1;
      break;

    case G__ADDALLOCTABLE:
      /***************************************
      * 0 ADDALLOCTABLE:
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: ADDALLOCTABLE\n" ,pc);
#endif
      /* no optimization */
      pc+=1;
      break;

    case G__DELALLOCTABLE:
      /***************************************
      * 0 DELALLOCTABLE:
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: DELALLOCTABLE\n" ,pc);
#endif
      /* no optimization */
      pc+=1;
      break;

    case G__BASEDESTRUCT:
      /***************************************
      * 0 BASEDESTRUCT:
      * 1 tagnum
      * 2 isarray
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"%3lx: BASECONSTRUCT tagnum=%d isarray=%d\n"
		,pc,G__asm_inst[pc+1],G__asm_inst[pc+2]);
      }
#endif
      /* no optimization */
      pc+=3;
      break;

    case G__REDECL:
      /***************************************
      * 0 REDECL:
      * 1 ig15
      * 2 var
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: REDECL\n",pc);
#endif
      /* no optimization */
      pc+=3;
      break;

    case G__TOVALUE:
      /***************************************
      * 0 TOVALUE:
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: TOVALUE\n",pc);
#endif
      /* no optimization */
#ifndef G__OLDIMPLEMENTATION1401
      pc+=2;
#else
      ++pc;
#endif
      break;

#ifndef G__OLDIMPLEMENTATION523
    case G__INIT_REF:
      /***************************************
      * inst
      * 0 G__INIT_REF
      * 1 index
      * 2 paran
      * 3 point_level
      * 4 var_array pointer
      * stack
      * sp-paran        <- sp-paran
      * sp-2
      * sp-1
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: INIT_REF\n",pc);
#endif
      pc+=5;
      break;
#endif

    case G__LETNEWVAL:
      /***************************************
      * inst
      * 0 LETNEWVAL
      * stack
      * sp-2  a
      * sp-1  b          <-
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: LETNEWVAL\n" ,pc);
#endif
      /* no optimization */
      ++pc;
      break;

    case G__SETGVP:
      /***************************************
      * inst
      * 0 SETGVP
      * 1 p or flag      0:use stack-1,else use this value
      * stack
      * sp-1  b          <-
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: SETGVP\n" ,pc);
#endif
      /* no optimization */
      pc+=2;
      break;

#ifndef G__OLDIMPLEMENTATION1073
    case G__CTOR_SETGVP:
      /***************************************
      * inst
      * 0 CTOR_SETGVP
      * 1 index
      * 2 var_array pointer
      * 3 mode, 0 local block scope, 1 member offset
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: CTOR_SETGVP\n",pc);
#endif
      /* no optimization */
      pc+=4;
      break;
#endif

    case G__TOPVALUE:
      /***************************************
      * 0 TOPVALUE:
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: TOPVALUE\n",pc);
#endif
      /* no optimization */
      ++pc;
      break;

#ifndef G__OLDIMPLEMENTATION2109
    case G__TRY:
      /***************************************
      * inst
      * 0 TRY
      * 1 first_catchblock 
      * 2 endof_catchblock
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: TRY %lx %lx\n",pc
				  ,G__asm_inst[pc+1] ,G__asm_inst[pc+2]);
#endif
      /* no optimization */
      pc+=3;
      break;

    case G__TYPEMATCH:
      /***************************************
      * inst
      * 0 TYPEMATCH
      * 1 address in data stack
      * stack
      * sp-1    a      <- comparee
      * sp             <- ismatch
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: TYPEMATCH\n",pc);
#endif
      /* no optimization */
      pc+=2;
      break;

    case G__ALLOCEXCEPTION:
      /***************************************
      * inst
      * 0 ALLOCEXCEPTION
      * 1 tagnum
      * stack
      * sp    a
      * sp+1             <-
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) 
        G__fprinterr(G__serr,"%3lx: ALLOCEXCEPTION %d\n",pc,G__asm_inst[pc+1]);
#endif
      /* no optimization */
      pc+=2;
      break;

    case G__DESTROYEXCEPTION:
      /***************************************
      * inst
      * 0 DESTROYEXCEPTION
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: DESTROYEXCEPTION\n",pc);
#endif
      /* no optimization */
      ++pc;
      break;
#endif /* 2109 */

#ifndef G__OLDIMPLEMENTATION1270
    case G__THROW:
      /***************************************
      * inst
      * 0 THROW
      * stack
      * sp-1    <-
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: THROW\n",pc);
#endif
      /* no optimization */
      pc+=1;
      break;

    case G__CATCH:
      /***************************************
      * inst
      * 0 CATCH
      * 1 filenum
      * 2 linenum
      * 3 pos
      * 4  "
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: CATCH\n",pc);
#endif
      /* no optimization */
      pc+=5;
      break;
#endif

#ifndef G__OLDIMPLEMENTATION1437
    case G__SETARYINDEX:
      /***************************************
      * inst
      * 0 SETARYINDEX
      * 1 allocflag, 1: new object, 0: auto object
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: SETARYINDEX\n",pc);
#endif
      /* no optimization */
      pc+=2;
      break;

    case G__RESETARYINDEX:
      /***************************************
      * inst
      * 0 RESETARYINDEX
      * 1 allocflag, 1: new object, 0: auto object
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: RESETARYINDEX\n",pc);
#endif
      /* no optimization */
      pc+=2;
      break;

    case G__GETARYINDEX:
      /***************************************
      * inst
      * 0 GETARYINDEX
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: GETARYINDEX\n",pc);
#endif
      /* no optimization */
      ++pc;
      break;
#endif

#ifndef G__OLDIMPLEMENTATION2042
    case G__ENTERSCOPE:
      /***************************************
      * inst
      * 0 ENTERSCOPE
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: ENTERSCOPE\n",pc);
#endif
      /* no optimization */
      ++pc;
      break;

    case G__EXITSCOPE:
      /***************************************
      * inst
      * 0 EXITSCOPE
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: EXITSCOPE\n",pc);
#endif
      /* no optimization */
      ++pc;
      break;

    case G__PUTAUTOOBJ:
      /***************************************
      * inst
      * 0 PUTAUTOOBJ
      * 1 var
      * 2 ig15
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: PUTAUTOOBJ\n",pc);
#endif
      /* no optimization */
      pc+=3;
      break;

#ifndef G__OLDIMPLEMENTATION2160
    case G__CASE:
      /***************************************
      * inst
      * 0 CASE
      * 1 *casetable
      * stack
      * sp-1         <- 
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: CASE\n",pc);
#endif
      /* no optimization */
      pc+=2;
      break;
#endif 


#if 0
    case G__SETARYCTOR:
      /***************************************
      * inst
      * 0 SETARYCTOR
      * 1 num,  num>=0 set this value, -1 set stack value (, 0 reset)
      * stack
      * sp-1         <- sp-paran
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETARYCTOR\n",pc);
#endif
      /* no optimization */
      pc+=2;
      break;
#endif /* 0 */
#endif /* 2042 */

#ifndef G__OLDIMPLEMENTATION2140
    case G__MEMCPY:
      /***************************************
      * inst
      * 0 MEMCPY
      * stack
      * sp-3        ORIG  <- sp-3
      * sp-2        DEST
      * sp-1        SIZE
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: MEMCPY\n",pc);
#endif
      /* no optimization */
      ++pc;
      break;
#endif

#ifndef G__OLDIMPLEMENTATION2147
    case G__MEMSETINT:
      /***************************************
      * inst
      * 0 MEMSETINT
      * 1 mode,  0:no offset, 1: G__store_struct_offset, 2: localmem
      * 2 numdata
      * 3 adr
      * 4 data
      * 5 adr
      * 6 data
      * ...
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: MEMSETINT %ld %ld\n",pc
				  ,G__asm_inst[pc+1],G__asm_inst[pc+2]);
#endif
      /* no optimization */
      pc+=G__asm_inst[pc+2]*2+3;
      break;

    case G__JMPIFVIRTUALOBJ:
      /***************************************
      * inst
      * 0 JMPIFVIRTUALOBJ
      * 1 offset
      * 2 next_pc
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: JMPIFVIRTUALOBJ %lx %lx\n",pc
				  ,G__asm_inst[pc+1],G__asm_inst[pc+2]);
#endif
      /* no optimization */
      pc+=3;
      break;
#endif /* 2147 */

#ifndef G__OLDIMPLEMENTATION2152
    case G__VIRTUALADDSTROS:
      /***************************************
      * inst
      * 0 VIRTUALADDSTROS
      * 1 tagnum
      * 2 baseclass
      * 3 basen
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: VIRTUALADDSTROS %lx %lx\n",pc
				  ,G__asm_inst[pc+1],G__asm_inst[pc+3]);
#endif
      /* no optimization */
      pc+=4;
      break;
#endif /* 2152 */

#ifndef G__OLDIMPLEMENTATION2142
    case G__PAUSE:
      /***************************************
      * inst
      * 0 PAUSe
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: PAUSE\n",pc);
#endif
      /* no optimization */
      ++pc;
      break;
#endif

    case G__NOP:
      /***************************************
      * 0 NOP
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: NOP\n" ,pc);
#endif
      /* no optimization */
      ++pc;
      break;

    default:
      /***************************************
      * Illegal instruction.
      * This is a double check and should
      * never happen.
      ***************************************/
      G__fprinterr(G__serr,"%3x: illegal instruction 0x%lx\t%ld\n"
	      ,pc,G__asm_inst[pc],G__asm_inst[pc]);
      ++pc;
      ++illegal;
      return(1);
      break;
    }

  }

  return(0);
}
#endif

#if defined(G__ASM_DBG) || !defined(G__OLDIMPLEMENTATION1270)
/****************************************************************
* G__dasm()
*
*  Disassembler
*
****************************************************************/
int G__dasm(fout,isthrow)
FILE *fout;
int isthrow;
{
  unsigned int pc;               /* instruction program counter */
  int illegal=0;
  struct G__var_array *var;

  if(!fout) fout=G__serr;
  pc=0;

  while(pc<G__MAXINST) {

    switch(G__INST(G__asm_inst[pc])) {

    case G__LDST_VAR_P:
      /***************************************
      * inst
      * 0 G__LDST_VAR_P
      * 1 index
      * 2 void (*f)(pbuf,psp,offset,p,ctype,
      * 3 (not use)
      * 4 var_array pointer
      * stack
      * sp          <-
      ***************************************/
      if(0==isthrow) {
	var = (struct G__var_array*)G__asm_inst[pc+4];
	if(!var) return(1);
	fprintf(fout,"%3x: LDST_VAR_P index=%ld %s\n"
		,pc,G__asm_inst[pc+1]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
      pc+=5;
      break;

    case G__LDST_MSTR_P:
      /***************************************
      * inst
      * 0 G__LDST_MSTR_P
      * 1 index
      * 2 void (*f)(pbuf,psp,offset,p,ctype,
      * 3 (not use)
      * 4 var_array pointer
      * stack
      * sp          <-
      ***************************************/
      if(0==isthrow) {
	var = (struct G__var_array*)G__asm_inst[pc+4];
	if(!var) return(1);
	fprintf(fout,"%3x: LDST_MSTR_P index=%ld %s\n"
		,pc,G__asm_inst[pc+1]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
      pc+=5;
      break;

    case G__LDST_VAR_INDEX:
      /***************************************
      * inst
      * 0 G__LDST_VAR_INDEX
      * 1 *arrayindex
      * 2 void (*f)(pbuf,psp,offset,p,ctype,
      * 3 index
      * 4 pc increment
      * 5 not use
      * 6 var_array pointer
      * stack
      * sp          <-
      ***************************************/
      if(0==isthrow) {
	var = (struct G__var_array*)G__asm_inst[pc+6];
	if(!var) return(1);
	fprintf(fout,"%3x: LDST_VAR_INDEX index=%ld %s\n"
		,pc,G__asm_inst[pc+3]
		,var->varnamebuf[G__asm_inst[pc+3]]);
      }
      pc+=G__asm_inst[pc+4];
      break;

    case G__LDST_VAR_INDEX_OPR:
      /***************************************
      * inst
      * 0 G__LDST_VAR_INDEX_OPR
      * 1 *int1
      * 2 *int2
      * 3 opr +,-
      * 4 void (*f)(pbuf,psp,offset,p,ctype,
      * 5 index
      * 6 pc increment
      * 7 not use
      * 8 var_array pointer
      * stack
      * sp          <-
      ***************************************/
      if(0==isthrow) {
	var = (struct G__var_array*)G__asm_inst[pc+8];
	if(!var) return(1);
	fprintf(fout,"%3x: LDST_VAR_INDEX_OPR index=%ld %s\n"
		,pc,G__asm_inst[pc+5]
		,var->varnamebuf[G__asm_inst[pc+5]]);
      }
      pc+=G__asm_inst[pc+6];
      break;

    case G__OP2_OPTIMIZED:
      /***************************************
      * inst
      * 0 OP2_OPTIMIZED
      * 1 (*p2f)(buf)
      * stack
      * sp-2  a
      * sp-1  a           <-
      * sp    G__null
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: OP2_OPTIMIZED \n",pc);
      }
      pc+=2;
      break;

    case G__OP1_OPTIMIZED:
      /***************************************
      * inst
      * 0 OP1_OPTIMIZED
      * 1 (*p2f)(buf)
      * stack
      * sp-1  a
      * sp    G__null     <-
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: OP1_OPTIMIZED \n",pc);
      }
      pc+=2;
      break;

    case G__LD_VAR:
      /***************************************
      * inst
      * 0 G__LD_VAR
      * 1 index
      * 2 paran
      * 3 point_level
      * 4 var_array pointer
      * stack
      * sp-paran+1      <- sp-paran+1
      * sp-2
      * sp-1
      * sp
      ***************************************/
      if(0==isthrow) {
	var = (struct G__var_array*)G__asm_inst[pc+4];
	if(!var) return(1);
	fprintf(fout,"%3x: LD_VAR index=%ld paran=%ld point %c %s\n"
		,pc,G__asm_inst[pc+1],G__asm_inst[pc+2]
		,(char)G__asm_inst[pc+3]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
      pc+=5;
      break;

    case G__LD:
      /***************************************
      * inst
      * 0 G__LD
      * 1 address in data stack
      * stack
      * sp    a
      * sp+1             <-
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: LD %g from %lx \n"
		,pc
		,G__double(G__asm_stack[G__asm_inst[pc+1]])
		,G__asm_inst[pc+1]);
      }
      pc+=2;
      break;

    case G__CL:
      /***************************************
      * 0 CL
      *  clear stack pointer
      ***************************************/
      if(0==isthrow) {
#ifndef G__OLDIMPLEMENTATION2132
	fprintf(fout,"%3x: CL %s:%ld\n",pc
		 ,G__srcfile[G__asm_inst[pc+1]/G__CL_FILESHIFT].filename
				  ,G__asm_inst[pc+1]&G__CL_LINEMASK);
#else
	fprintf(fout,"%3x: CL %ld\n",pc,G__asm_inst[pc+1]);
#endif
      }
      pc+=2;
      break;

    case G__OP2:
      /***************************************
      * inst
      * 0 OP2
      * 1 (+,-,*,/,%,@,>>,<<,&,|)
      * stack
      * sp-2  a
      * sp-1  b          <-
      * sp
      ***************************************/
      if(0==isthrow) {
	if(isprint(G__asm_inst[pc+1]))
	  fprintf(fout,"%3x: OP2 '%c'%ld \n" ,pc
		  ,(char)G__asm_inst[pc+1],G__asm_inst[pc+1]);
	else
	  fprintf(fout,"%3x: OP2 %ld \n",pc,G__asm_inst[pc+1]);
      }
      pc+=2;
      break;

    case G__ST_VAR:
      /***************************************
      * inst
      * 0 G__ST_VAR
      * 1 index
      * 2 paran
      * 3 point_level
      * 4 var_array pointer
      * stack
      * sp-paran        <- sp-paran
      * sp-2
      * sp-1
      * sp
      ***************************************/
      if(0==isthrow) {
	var = (struct G__var_array*)G__asm_inst[pc+4];
	if(!var) return(1);
	fprintf(fout,"%3x: ST_VAR index=%ld paran=%ld point %c %s\n"
		,pc,G__asm_inst[pc+1],G__asm_inst[pc+2]
		,(char)G__asm_inst[pc+3]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
      pc+=5;
      break;

    case G__LD_MSTR:
      /***************************************
      * inst
      * 0 G__LD_MSTR
      * 1 index
      * 2 paran
      * 3 point_level
      * 4 *structmem
      * stack
      * sp-paran+1      <- sp-paran+1
      * sp-2
      * sp-1
      * sp
      ***************************************/
      if(0==isthrow) {
	var = (struct G__var_array*)G__asm_inst[pc+4];
	if(!var) return(1);
	fprintf(fout,"%3x: LD_MSTR index=%ld paran=%ld point %c %s\n"
		,pc,G__asm_inst[pc+1],G__asm_inst[pc+2]
		,(char)G__asm_inst[pc+3]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
      pc+=5;
      break;

    case G__CMPJMP:
      /***************************************
      * 0 CMPJMP
      * 1 *G__asm_test_X()
      * 2 *a
      * 3 *b
      * 4 next_pc
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: CMPJMP (0x%lx)%d (0x%lx)%d to %lx\n"
		,pc
		,G__asm_inst[pc+2],*(int *)G__asm_inst[pc+2]
		,G__asm_inst[pc+3],*(int *)G__asm_inst[pc+3]
		,G__asm_inst[pc+4]);
      }
      pc+=5;
      break;

    case G__PUSHSTROS:
      /***************************************
      * inst
      * 0 G__PUSHSTROS
      * stack
      * sp           <- sp-paran
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: PUSHSTROS\n" ,pc);
      }
      ++pc;
      break;

    case G__SETSTROS:
      /***************************************
      * inst
      * 0 G__SETSTROS
      * stack
      * sp-1         <- sp-paran
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: SETSTROS\n",pc);
      }
      ++pc;
      break;

    case G__POPSTROS:
      /***************************************
      * inst
      * 0 G__POPSTROS
      * stack
      * sp           <- sp-paran
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: POPSTROS\n" ,pc);
      }
      ++pc;
      break;

    case G__ST_MSTR:
      /***************************************
      * inst
      * 0 G__ST_MSTR
      * 1 index
      * 2 paran
      * 3 point_level
      * 4 *structmem
      * stack
      * sp-paran        <- sp-paran
      * sp-2
      * sp-1
      * sp
      ***************************************/
      if(0==isthrow) {
	var = (struct G__var_array*)G__asm_inst[pc+4];
	if(!var) return(1);
	fprintf(fout,"%3x: ST_MSTR index=%ld paran=%ld point %c %s\n"
		,pc,G__asm_inst[pc+1],G__asm_inst[pc+2]
		,(char)G__asm_inst[pc+3]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
      pc+=5;
      break;

    case G__INCJMP:
      /***************************************
      * 0 INCJMP
      * 1 *cntr
      * 2 increment
      * 3 next_pc
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: INCJMP *(int*)0x%lx+%ld to %lx\n"
		,pc ,G__asm_inst[pc+1] ,G__asm_inst[pc+2]
		,G__asm_inst[pc+3]);
      }
      pc+=4;
      break;

    case G__CNDJMP:
      /***************************************
      * 0 CNDJMP   (jump if 0)
      * 1 next_pc
      * stack
      * sp-1         <-
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: CNDJMP to %lx\n" ,pc ,G__asm_inst[pc+1]);
      }
      pc+=2;
      break;

    case G__CMP2:
      /***************************************
      * 0 CMP2
      * 1 operator
      * stack
      * sp-1         <-
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: CMP2 '%c' \n" ,pc ,(char)G__asm_inst[pc+1]);
      }
      pc+=2;
      break;

    case G__JMP:
      /***************************************
      * 0 JMP
      * 1 next_pc
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: JMP %lx\n" ,pc,G__asm_inst[pc+1]);
      }
      pc+=2;
      break;

    case G__PUSHCPY:
      /***************************************
      * inst
      * 0 G__PUSHCPY
      * stack
      * sp
      * sp+1            <-
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: PUSHCPY\n",pc);
      }
      ++pc;
      break;

    case G__POP:
      /***************************************
      * inst
      * 0 G__POP
      * stack
      * sp-1            <-
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: POP\n" ,pc);
      }
      ++pc;
      break;

    case G__LD_FUNC:
      /***************************************
      * inst
      * 0 G__LD_FUNC
      * 1 *name
      * 2 hash
      * 3 paran
      * 4 (*func)()
      * stack
      * sp-paran+1      <- sp-paran+1
      * sp-2
      * sp-1
      * sp
      ***************************************/
      if(0==isthrow) {
	if(G__asm_inst[pc+1]<G__MAXSTRUCT)
	  fprintf(fout,"%3x: LD_FUNC %s paran=%ld\n" ,pc
		  ,"compiled",G__asm_inst[pc+3]);
	else
	  fprintf(fout,"%3x: LD_FUNC %s paran=%ld\n" ,pc
		  ,(char *)G__asm_inst[pc+1],G__asm_inst[pc+3]);
      }
      pc+=5;
      break;

    case G__RETURN:
      /***************************************
      * 0 RETURN
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: RETURN\n" ,pc);
      }
      pc++;
      return(0);
      break;

    case G__CAST:
      /***************************************
      * 0 CAST
      * 1 type
      * 2 typenum
      * 3 tagnum
      * 4 reftype 
      * stack
      * sp-1    <- cast on this
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: CAST to %c type%ld tag%ld\n" ,pc
		,(char)G__asm_inst[pc+1],G__asm_inst[pc+2],G__asm_inst[pc+3]);
      }
      pc+=5;
      break;

    case G__OP1:
      /***************************************
      * inst
      * 0 OP1
      * 1 (+,-)
      * stack
      * sp-1  a
      * sp    G__null     <-
      ***************************************/
      if(0==isthrow) {
	if(isprint(G__asm_inst[pc+1]))
	  fprintf(fout,"%3x: OP1 '%c'%ld\n",pc
		  ,(char)G__asm_inst[pc+1],G__asm_inst[pc+1] );
	else
	  fprintf(fout,"%3x: OP1 %ld\n",pc,G__asm_inst[pc+1]);
      }
      pc+=2;
      break;

    case G__LETVVAL:
      /***************************************
      * inst
      * 0 LETVVAL
      * stack
      * sp-2  a
      * sp-1  b          <-
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: LETVVAL\n" ,pc);
      }
      ++pc;
      break;

    case G__ADDSTROS:
      /***************************************
      * inst
      * 0 ADDSTROS
      * 1 addoffset
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: ADDSTROS %ld\n" ,pc,G__asm_inst[pc+1]);
      }
      pc+=2;
      break;

    case G__LETPVAL:
      /***************************************
      * inst
      * 0 LETPVAL
      * stack
      * sp-2  a
      * sp-1  b          <-
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: LETPVAL\n" ,pc);
      }
      ++pc;
      break;

    case G__FREETEMP:
      /***************************************
      * 0 FREETEMP
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: FREETEMP\n" ,pc);
      }
      ++pc;
      break;

    case G__SETTEMP:
      /***************************************
      * 0 SETTEMP
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: SETTEMP\n" ,pc);
      }
      ++pc;
      break;


    case G__GETRSVD:
      /***************************************
      * 0 GETRSVD
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: GETRSVD\n" ,pc);
      }
      ++pc;
      break;

    case G__TOPNTR:
      /***************************************
      * inst
      * 0 LETVVAL
      * stack
      * sp-1  a          <-
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: TOPNTR\n" ,pc);
      }
      ++pc;
      break;

    case G__NOT:
      /***************************************
      * 0 NOT
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: NOT\n" ,pc);
      }
      ++pc;
      break;

#ifndef G__OLDIMPLEMENTATION1399
    case G__BOOL:
      /***************************************
      * 0 BOOL
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: BOOL\n" ,pc);
      }
      ++pc;
      break;
#endif

    case G__ISDEFAULTPARA:
      /***************************************
      * 0 ISDEFAULTPARA
      * 1 next_pc
      ***************************************/
      if(0==isthrow) {
	G__fprinterr(G__serr,"%3x: !ISDEFAULTPARA JMP %lx\n",pc,G__asm_inst[pc+1]);
      }
      pc+=2;
      break;

#ifdef G__ASM_WHOLEFUNC
    case G__LDST_LVAR_P:
      /***************************************
      * inst
      * 0 G__LDST_LVAR_P
      * 1 index
      * 2 void (*f)(pbuf,psp,offset,p,ctype,
      * 3 (not use)
      * 4 var_array pointer
      * stack
      * sp          <-
      ***************************************/
      if(0==isthrow) {
	var = (struct G__var_array*)G__asm_inst[pc+4];
	if(!var) return(1);
	fprintf(fout,"%3x: LDST_LVAR_P index=%ld %s\n"
		,pc,G__asm_inst[pc+1]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
      pc+=5;
      break;

    case G__LD_LVAR:
      /***************************************
      * inst
      * 0 G__LD_LVAR
      * 1 index
      * 2 paran
      * 3 point_level
      * 4 var_array pointer
      * stack
      * sp-paran+1      <- sp-paran+1
      * sp-2
      * sp-1
      * sp
      ***************************************/
      if(0==isthrow) {
	var = (struct G__var_array*)G__asm_inst[pc+4];
	if(!var) return(1);
	fprintf(fout,"%3x: LD_LVAR index=%ld paran=%ld point %c %s\n"
		,pc,G__asm_inst[pc+1],G__asm_inst[pc+2]
		,(char)G__asm_inst[pc+3]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
      pc+=5;
      break;

    case G__ST_LVAR:
      /***************************************
      * inst
      * 0 G__ST_LVAR
      * 1 index
      * 2 paran
      * 3 point_level
      * 4 var_array pointer
      * stack
      * sp-paran        <- sp-paran
      * sp-2
      * sp-1
      * sp
      ***************************************/
      if(0==isthrow) {
	var = (struct G__var_array*)G__asm_inst[pc+4];
	if(!var) return(1);
	fprintf(fout,"%3x: ST_LVAR index=%ld paran=%ld point %c %s\n"
		,pc,G__asm_inst[pc+1],G__asm_inst[pc+2]
		,(char)G__asm_inst[pc+3]
		,var->varnamebuf[G__asm_inst[pc+1]]);
      }
      pc+=5;
      break;
#endif

    case G__REWINDSTACK:
      /***************************************
      * inst
      * 0 G__REWINDSTACK
      * 1 rewind
      * stack
      * sp-2            <-  ^
      * sp-1                | rewind
      * sp              <- ..
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: REWINDSTACK %ld\n" ,pc,G__asm_inst[pc+1]);
      }
      pc+=2;
      break;

    case G__CND1JMP:
      /***************************************
      * 0 CND1JMP   (jump if 1)
      * 1 next_pc
      * stack
      * sp-1         <-
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: CND1JMP  to %lx\n" ,pc ,G__asm_inst[pc+1]);
      }
      pc+=2;
      break;

#ifdef G__ASM_IFUNC
    case G__LD_IFUNC:
      /***************************************
      * inst
      * 0 G__LD_IFUNC
      * 1 *name
      * 2 hash          // unused
      * 3 paran
      * 4 p_ifunc
      * 5 funcmatch
      * 6 memfunc_flag
      * 7 index
      * stack
      * sp-paran+1      <- sp-paran+1
      * sp-2
      * sp-1
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: LD_IFUNC %s paran=%ld\n" ,pc
		,(char *)G__asm_inst[pc+1],G__asm_inst[pc+3]);
      }
      pc+=8;
      break;

    case G__NEWALLOC:
      /***************************************
      * inst
      * 0 G__NEWALLOC
      * 1 size
      * stack
      * sp-1     <- pinc
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: NEWALLOC size(%ld)\n"
		,pc,G__asm_inst[pc+1]);
      }
      pc+=3;
      break;

    case G__SET_NEWALLOC:
      /***************************************
      * inst
      * 0 G__SET_NEWALLOC
      * 1 tagnum
      * stack
      * sp-1        G__store_struct_offset
      * sp       <-
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: SET_NEWALLOC\n" ,pc);
      }
      pc+=3;
      break;

    case G__DELETEFREE:
      /***************************************
      * inst
      * 0 G__DELETEFREE
      * 1 isarray  0: simple free, 1: array, 2: virtual free
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: DELETEFREE\n",pc);
      }
      pc+=2;
      break;

    case G__SWAP:
      /***************************************
      * inst
      * 0 G__SWAP
      * stack
      * sp-2          sp-1
      * sp-1          sp-2
      * sp       <-   sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: SWAP\n",pc);
      }
      ++pc;
      break;

#endif /* G__ASM_IFUNC */

    case G__BASECONV:
      /***************************************
      * inst
      * 0 G__BASECONV
      * 1 formal_tagnum
      * 2 baseoffset
      * stack
      * sp-2          sp-1
      * sp-1          sp-2
      * sp       <-   sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: BASECONV %ld %ld\n",pc
		,G__asm_inst[pc+1],G__asm_inst[pc+2]);
      }
      pc+=3;
      break;

    case G__STORETEMP:
      /***************************************
      * 0 STORETEMP
      * stack
      * sp-1
      * sp       <-  sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: STORETEMP\n",pc);
      }
      ++pc;
      break;

    case G__ALLOCTEMP:
      /***************************************
      * 0 ALLOCTEMP
      * 1 tagnum
      * stack
      * sp-1
      * sp       <-  sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: ALLOCTEMP %s\n",pc,G__struct.name[G__asm_inst[pc+1]]);
      }
      pc+=2;
      break;

    case G__POPTEMP:
      /***************************************
      * 0 POPTEMP
      * 1 tagnum
      * stack
      * sp-1
      * sp      <-  sp
      ***************************************/
      if(0==isthrow) {
	if(-1!=G__asm_inst[pc+1])
	  fprintf(fout,"%3x: POPTEMP %s\n" 
		  ,pc,G__struct.name[G__asm_inst[pc+1]]);
	else
	  fprintf(fout,"%3x: POPTEMP -1\n",pc);
      }
      pc+=2;
      break;

    case G__REORDER:
      /***************************************
      * 0 REORDER
      * 1 paran(total)
      * 2 ig25(arrayindex)
      * stack      paran=4 ig25=2    x y z w -> x y z w z w -> x y x y z w -> w z x y
      * sp-3    <-  sp-1
      * sp-2    <-  sp-3
      * sp-1    <-  sp-2
      * sp      <-  sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: REORDER paran=%ld ig25=%ld\n"
		,pc ,G__asm_inst[pc+1],G__asm_inst[pc+2]);
      }
      pc+=3;
      break;

    case G__LD_THIS:
      /***************************************
      * 0 LD_THIS
      * 1 point_level;
      * stack
      * sp-1
      * sp
      * sp+1   <-
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: LD_THIS %s\n"
		,pc ,G__struct.name[G__tagnum]);
      }
      pc+=2;
      break;

    case G__RTN_FUNC:
      /***************************************
      * 0 RTN_FUNC
      * 1 isreturnvalue
      * stack
      * sp-1   -> return this
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: RTN_FUNC %ld\n" ,pc ,G__asm_inst[pc+1]);
      }
      pc+=2;
      break;

    case G__SETMEMFUNCENV:
      /***************************************
      * 0 SETMEMFUNCENV:
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: SETMEMFUNCENV\n",pc);
      }
      pc+=1;
      break;

    case G__RECMEMFUNCENV:
      /***************************************
      * 0 RECMEMFUNCENV:
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: RECMEMFUNCENV\n" ,pc);
      }
      pc+=1;
      break;

    case G__ADDALLOCTABLE:
      /***************************************
      * 0 ADDALLOCTABLE:
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: ADDALLOCTABLE\n" ,pc);
      }
      pc+=1;
      break;

    case G__DELALLOCTABLE:
      /***************************************
      * 0 DELALLOCTABLE:
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: DELALLOCTABLE\n" ,pc);
      }
      pc+=1;
      break;

    case G__BASEDESTRUCT:
      /***************************************
      * 0 BASEDESTRUCT:
      * 1 tagnum
      * 2 isarray
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: BASECONSTRUCT tagnum=%ld isarray=%ld\n"
		,pc,G__asm_inst[pc+1],G__asm_inst[pc+2]);
      }
      pc+=3;
      break;

    case G__REDECL:
      /***************************************
      * 0 REDECL:
      * 1 ig15
      * 2 var
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: REDECL\n",pc);
      }
      pc+=3;
      break;

    case G__TOVALUE:
      /***************************************
      * 0 TOVALUE:
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: TOVALUE\n",pc);
      }
#ifndef G__OLDIMPLEMENTATION1401
      pc+=2;
#else
      ++pc;
#endif
      break;

#ifndef G__OLDIMPLEMENTATION523
    case G__INIT_REF:
      /***************************************
      * inst
      * 0 G__INIT_REF
      * 1 index
      * 2 paran
      * 3 point_level
      * 4 var_array pointer
      * stack
      * sp-paran        <- sp-paran
      * sp-2
      * sp-1
      * sp
      ***************************************/
      if(0==isthrow) {
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: INIT_REF\n",pc);
      }
      pc+=5;
      break;
#endif

    case G__LETNEWVAL:
      /***************************************
      * inst
      * 0 LETNEWVAL
      * stack
      * sp-2  a
      * sp-1  b          <-
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: LETNEWVAL\n" ,pc);
      }
      ++pc;
      break;

    case G__SETGVP:
      /***************************************
      * inst
      * 0 SETGVP
      * 1 p or flag      0:use stack-1,else use this value
      * stack
      * sp-1  b          <-
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: SETGVP\n" ,pc);
	/* no optimization */
      }
      pc+=2;
      break;

    case G__TOPVALUE:
      /***************************************
      * 0 TOPVALUE:
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: TOPVALUE\n",pc);
      }
      ++pc;
      break;

#ifndef G__OLDIMPLEMENTATION1073
    case G__CTOR_SETGVP:
      /***************************************
      * inst
      * 0 CTOR_SETGVP
      * 1 index
      * 2 var_array pointer
      * 3 mode, 0 local block scope, 1 member offset
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: CTOR_SETGVP\n",pc);
      }
      pc+=4;
      break;
#endif

#ifndef G__OLDIMPLEMENTATION2109
    case G__TRY:
      /***************************************
      * inst
      * 0 TRY
      * 1 first_catchblock 
      * 2 endof_catchblock
      ***************************************/
      if(0==isthrow) {
         fprintf(fout,"%3x: TRY %lx %lx\n",pc
		  ,G__asm_inst[pc+1] ,G__asm_inst[pc+2]);
      }
      pc+=3;
      break;

    case G__TYPEMATCH:
      /***************************************
      * inst
      * 0 TYPEMATCH
      * 1 address in data stack
      * stack
      * sp-1    a      <- comparee
      * sp             <- ismatch
      ***************************************/
      if(0==isthrow) {
        fprintf(fout,"%3x: TYPEMATCH\n",pc);
      }
      pc+=2;
      break;

    case G__ALLOCEXCEPTION:
      /***************************************
      * inst
      * 0 ALLOCEXCEPTION
      * 1 tagnum
      * stack
      * sp    a
      * sp+1             <-
      ***************************************/
      if(0==isthrow) {
        fprintf(fout,"%3x: ALLOCEXCEPTION %ld\n",pc,G__asm_inst[pc+1]);
      }
      pc+=2;
      break;

    case G__DESTROYEXCEPTION:
      /***************************************
      * inst
      * 0 DESTROYEXCEPTION
      ***************************************/
      if(0==isthrow) {
        fprintf(fout,"%3x: DESTROYEXCEPTION\n",pc);
      }
      ++pc;
      break;
#endif /* 2109 */

#ifndef G__OLDIMPLEMENTATION1270
    case G__THROW:
      /***************************************
      * inst
      * 0 THROW
      * stack
      * sp-1    <-
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: THROW\n" ,pc);
      }
      pc+=1;
      break;

    case G__CATCH:
      /***************************************
      * inst
      * 0 CATCH
      * 1 filenum
      * 2 linenum
      * 3 pos
      * 4  "
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: CATCH\n" ,pc);
      }
      else {
	fpos_t store_pos;
	struct G__input_file store_ifile = G__ifile;
	char statement[G__LONGLINE];
#if defined(G__NONSCALARFPOS2)
	fpos_t pos;
	pos.__pos = (off_t)G__asm_inst[pc+3];
#elif defined(G__NONSCALARFPOS_QNX)
	fpos_t pos;
	pos._Off = (off_t)G__asm_inst[pc+3];
#else
	fpos_t pos = (fpos_t)G__asm_inst[pc+3];
#endif
	fgetpos(G__ifile.fp,&store_pos);
	G__ifile.filenum = G__asm_inst[pc+1];
	G__ifile.line_number = G__asm_inst[pc+2];
	strcpy(G__ifile.name,G__srcfile[G__ifile.filenum].filename);
	G__ifile.fp = G__srcfile[G__ifile.filenum].fp;
	fsetpos(G__ifile.fp,&pos);
	G__asm_exec = 0;
	G__return=G__RETURN_NON;
	G__exec_catch(statement);

	G__ifile = store_ifile;
	fsetpos(G__ifile.fp,&store_pos);
	return(G__CATCH);
      }
      pc+=5;
      break;
#endif

#ifndef G__OLDIMPLEMENTATION1437
    case G__SETARYINDEX:
      /***************************************
      * inst
      * 0 SETARYINDEX
      * 1 allocflag, 1: new object, 0: auto object
      ***************************************/
      if(isthrow) {
	G__fprinterr(G__serr,"%3x: SETARYINDEX\n",pc);
      }
      pc+=2;
      break;

    case G__RESETARYINDEX:
      /***************************************
      * inst
      * 0 RESETARYINDEX
      * 1 allocflag, 1: new object, 0: auto object
      ***************************************/
      if(isthrow) {
	G__fprinterr(G__serr,"%3x: RESETARYINDEX\n",pc);
      }
      pc+=2;
      break;

    case G__GETARYINDEX:
      /***************************************
      * inst
      * 0 GETARYINDEX
      ***************************************/
      if(isthrow) {
	G__fprinterr(G__serr,"%3x: GETARYINDEX\n",pc);
      }
      ++pc;
      break;
#endif

#ifndef G__OLDIMPLEMENTATION2042
    case G__ENTERSCOPE:
      /***************************************
      * inst
      * 0 ENTERSCOPE
      ***************************************/
#ifdef G__ASM_DBG
      if(0==isthrow) G__fprinterr(G__serr,"%3x: ENTERSCOPE\n",pc);
#endif
      /* no optimization */
      ++pc;
      break;

    case G__EXITSCOPE:
      /***************************************
      * inst
      * 0 EXITSCOPE
      ***************************************/
#ifdef G__ASM_DBG
      if(0==isthrow) G__fprinterr(G__serr,"%3x: EXITSCOPE\n",pc);
#endif
      /* no optimization */
      ++pc;
      break;

    case G__PUTAUTOOBJ:
      /***************************************
      * inst
      * 0 PUTAUTOOBJ
      * 1 var
      * 2 ig15
      ***************************************/
#ifdef G__ASM_DBG
      if(0==isthrow) G__fprinterr(G__serr,"%3x: PUTAUTOOBJ\n",pc);
#endif
      /* no optimization */
      pc+=3;
      break;

#ifndef G__OLDIMPLEMENTATION2160
    case G__CASE:
      /***************************************
      * inst
      * 0 CASE
      * 1 *casetable
      * stack
      * sp-1         <- 
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(0==isthrow) G__fprinterr(G__serr,"%3x: CASE\n",pc);
#endif
      /* no optimization */
      pc+=2;
      break;
#endif

#if 0
    case G__SETARYCTOR:
      /***************************************
      * inst
      * 0 SETARYCTOR
      * 1 num,  num>=0 set this value, -1 set stack value (, 0 reset)
      * stack
      * sp-1         <- sp-paran
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if(0==isthrow) G__fprinterr(G__serr,"%3x: SETARYCTOR\n",pc);
#endif
      /* no optimization */
      pc+=2;
      break;
#endif /* 0 */
#endif /* 2042 */

#ifndef G__OLDIMPLEMENTATION2140
    case G__MEMCPY:
      /***************************************
      * inst
      * 0 MEMCPY
      * stack
      * sp-3        ORIG  <- sp-3
      * sp-2        DEST
      * sp-1        SIZE
      * sp
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: MEMCPY\n" ,pc);
      }
      ++pc;
      break;
#endif

#ifndef G__OLDIMPLEMENTATION2147
    case G__MEMSETINT:
      /***************************************
      * inst
      * 0 MEMSETINT
      * 1 mode,  0:no offset, 1: G__store_struct_offset, 2: localmem
      * 2 numdata
      * 3 adr
      * 4 data
      * 5 adr
      * 6 data
      * ...
      ***************************************/
#ifdef G__ASM_DBG
      if(0==isthrow) fprintf(fout,"%3x: MEMSETINT %ld %ld\n",pc
			     ,G__asm_inst[pc+1],G__asm_inst[pc+2]);
#endif
      pc+=G__asm_inst[pc+2]*2+3;
      break;

    case G__JMPIFVIRTUALOBJ:
      /***************************************
      * inst
      * 0 JMPIFVIRTUALOBJ
      * 1 offset
      * 2 next_pc
      ***************************************/
#ifdef G__ASM_DBG
      if(0==isthrow) fprintf(fout,"%3x: JMPIFVIRTUALOBJ %lx %lx\n",pc
			     ,G__asm_inst[pc+1],G__asm_inst[pc+2]);
#endif
      pc+=3;
      break;
#endif /* 2147 */

#ifndef G__OLDIMPLEMENTATION2152
    case G__VIRTUALADDSTROS:
      /***************************************
      * inst
      * 0 VIRTUALADDSTROS
      * 1 tagnum
      * 2 baseclass
      * 3 basen
      ***************************************/
#ifdef G__ASM_DBG
      if(0==isthrow) fprintf(fout,"%3x: VIRTUALADDSTROS %lx %lx\n",pc
			     ,G__asm_inst[pc+1],G__asm_inst[pc+3]);
#endif
      pc+=4;
      break;
#endif /* 2152 */

#ifndef G__OLDIMPLEMENTATION2142
    case G__PAUSE:
      /***************************************
      * inst
      * 0 PAUSe
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: PAUSE\n" ,pc);
      }
      ++pc;
      break;
#endif


    case G__NOP:
      /***************************************
      * 0 NOP
      ***************************************/
      if(0==isthrow) {
	fprintf(fout,"%3x: NOP\n" ,pc);
      }
      ++pc;
      break;

    default:
      /***************************************
      * Illegal instruction.
      * This is a double check and should
      * never happen.
      ***************************************/
      fprintf(fout,"%3x: illegal instruction 0x%lx\t%ld\n"
	      ,pc,G__asm_inst[pc],G__asm_inst[pc]);
      ++pc;
      ++illegal;
      if(illegal>20) return(0);
      break;
    }

  }

  return(0);
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
