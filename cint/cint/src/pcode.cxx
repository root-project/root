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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"
#include "value.h"

extern "C" {

#ifdef G__BORLANDCC5
double G__doubleM(G__value *buf);
static void G__asm_toXvalue(G__value* result);
void G__get__tm__(G__FastAllocBuf &buf);
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

#define G__longlongM(buf)                                               \
  G__converT<long long>(buf)


/****************************************************************
* G__intM()
****************************************************************/
#define G__intM(buf)                                                   \
  G__convertT<long>(buf)

/****************************************************************
* G__doubleM()
****************************************************************/
double G__doubleM(G__value *buf)
{
   return G__convertT<double>(buf);
}

/****************************************************************
* G__isdoubleM()
****************************************************************/
#define G__isdoubleM(buf) ('f'==buf->type||'d'==buf->type)

/****************************************************************
* G__isunsignedM()
****************************************************************/
#define G__isunsignedM(buf) ('h'==buf->type||'k'==buf->type)

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
int G__asm_test_E(int *a,int *b)
{
  return(*a==*b);
}
int G__asm_test_N(int *a,int *b)
{
  return(*a!=*b);
}
int G__asm_test_GE(int *a,int *b)
{
  return(*a>=*b);
}
int G__asm_test_LE(int *a,int *b)
{
  return(*a<=*b);
}
int G__asm_test_g(int *a,int *b)
{
  return(*a>*b);
}
int G__asm_test_l(int *a,int *b)
{
  return(*a<*b);
}

/*************************************************************************
**************************************************************************
* TOPVALUE and TOVALUE optimization
**************************************************************************
*************************************************************************/
/******************************************************************
* G__value G__asm_toXvalue(G__value* p)
*
******************************************************************/
void G__asm_toXvalue(G__value* result)
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

typedef void (*G__p2f_tovalue) G__P((G__value*));
/******************************************************************
* void G__asm_tovalue_p2p(G__value* p)
******************************************************************/
void G__asm_tovalue_p2p(G__value *result)
{
  result->ref = result->obj.i;
  result->obj.i = (long)(*(long *)(result->obj.i));
  result->obj.reftype.reftype=G__PARANORMAL;
}

/******************************************************************
* void G__asm_tovalue_p2p2p(G__value* p)
******************************************************************/
void G__asm_tovalue_p2p2p(G__value *result)
{
  result->ref = result->obj.i;
  result->obj.i = (long)(*(long *)(result->obj.i));
  result->obj.reftype.reftype=G__PARAP2P;
}

/******************************************************************
* void G__asm_tovalue_p2p2p2(G__value* p)
******************************************************************/
void G__asm_tovalue_p2p2p2(G__value *result)
{
  result->ref = result->obj.i;
  result->obj.i = (long)(*(long *)(result->obj.i));
  --result->obj.reftype.reftype;
}
} // extern "C"

template<typename T>
void G__asm_tovalue_T(G__value* result)
{
   result->ref = result->obj.i;
   G__setvalue(result, *(T*)result->obj.i);
   result->type = tolower(result->type);
}

extern "C" {

extern void G__asm_tovalue_LL(G__value *result)  { G__asm_tovalue_T<G__int64>(result);}
extern void G__asm_tovalue_ULL(G__value *result) { G__asm_tovalue_T<G__uint64>(result);}
extern void G__asm_tovalue_LD(G__value *result)  { G__asm_tovalue_T<long double>(result);}
extern void G__asm_tovalue_B(G__value *result)   { G__asm_tovalue_T<unsigned char>(result);}
extern void G__asm_tovalue_C(G__value *result)   { G__asm_tovalue_T<char>(result);}
extern void G__asm_tovalue_R(G__value *result)   { G__asm_tovalue_T<unsigned short>(result);}
extern void G__asm_tovalue_S(G__value *result)   { G__asm_tovalue_T<short>(result);}
extern void G__asm_tovalue_H(G__value *result)   { G__asm_tovalue_T<unsigned int>(result);}
extern void G__asm_tovalue_I(G__value *result)   { G__asm_tovalue_T<int>(result);}
extern void G__asm_tovalue_K(G__value *result)   { G__asm_tovalue_T<unsigned long>(result);}
extern void G__asm_tovalue_L(G__value *result)   { G__asm_tovalue_T<long>(result);}
extern void G__asm_tovalue_F(G__value *result)   { G__asm_tovalue_T<float>(result);}
extern void G__asm_tovalue_D(G__value *result)   { G__asm_tovalue_T<double>(result);}


/******************************************************************
* void G__asm_tovalue_U(G__value* p)
******************************************************************/
void G__asm_tovalue_U(G__value *result)
{
  result->ref = result->obj.i;
  /* result->obj.i = result->obj.i; */
  result->type = tolower(result->type);
}



/*************************************************************************
**************************************************************************
* Optimization level 1 runtime function
**************************************************************************
*************************************************************************/



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
/*
#define G__ASM_GET_INT(casttype,ctype)    \
  G__value *buf= &pbuf[(*psp)++];         \
  buf->tagnum = -1;                       \
  buf->type = ctype;                      \
  buf->typenum = var->p_typetable[ig15];  \
  buf->ref = var->p[ig15]+offset;         \
  buf->obj.i = *(casttype*)buf->ref
*/

} // extern "C"

#define LDARGS G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15
#define LDARGCALL pbuf,psp,offset,var,ig15

template <typename T>
void G__ASM_GET_INT(G__value* pbuf, int* psp, long offset, G__var_array* var, long ig15) 
{
  pbuf+= (*psp)++;
  pbuf->tagnum = -1;
  pbuf->type = G__gettypechar<T>();
  pbuf->typenum = var->p_typetable[ig15];
  pbuf->ref = var->p[ig15]+offset;
  G__setvalue(pbuf, *(T*)pbuf->ref);
}


template <class T> void G__AddAssign(G__value *buf, T value) 
{  
   switch(buf->type) {
   case 'd': /* double */
   case 'f': /* float */           G__value_ref<double>(*buf)+= value; break;
   case 'w': /* logic */        
   case 'r': /* unsigned short */  G__value_ref<unsigned short>(*buf) += value; break;
   case 'h': /* unsigned int */    G__value_ref<unsigned int>(*buf) += value; break;
   case 'b': /* unsigned char */   G__value_ref<unsigned char>(*buf) += value; break;
   case 'k': /* unsigned long */   G__value_ref<unsigned long>(*buf) += value; break;
   case 'n':       G__value_ref<long long>(*buf) += value; break;
   case 'm':       G__value_ref<unsigned long long>(*buf) += value; break;
   case 'q':       G__value_ref<long double>(*buf) += value; break;
   case 'i':       G__value_ref<int>(*buf) += value; break;
   case 'c':       G__value_ref<char>(*buf) += value; break;
   case 's':       G__value_ref<short>(*buf) += value; break;
   default: 
     /* long */
      G__value_ref<int>(*buf) += value; break;
   }
}
  

template <class T> void G__SubAssign(G__value *buf, T value) 
{  
   switch(buf->type) {
   case 'd': /* double */
   case 'f': /* float */           G__value_ref<double>(*buf)-= value; break;
   case 'w': /* logic */        
   case 'r': /* unsigned short */  G__value_ref<unsigned short>(*buf) -= value; break;
   case 'h': /* unsigned int */    G__value_ref<unsigned int>(*buf) -= value; break;
   case 'b': /* unsigned char */   G__value_ref<unsigned char>(*buf) -= value; break;
   case 'k': /* unsigned long */   G__value_ref<unsigned long>(*buf) -= value; break;
   case 'n':       G__value_ref<long long>(*buf) -= value; break;
   case 'm':       G__value_ref<unsigned long long>(*buf) -= value; break;
   case 'q':       G__value_ref<long double>(*buf) -= value; break;
   case 'i':       G__value_ref<int>(*buf) -= value; break;
   case 'c':       G__value_ref<char>(*buf) -= value; break;
   case 's':       G__value_ref<short>(*buf) -= value; break;
   default: 
     /* long */
      G__value_ref<int>(*buf) -= value; break;
   }
}
  
template <class T> void G__MulAssign(G__value *buf, T value) 
{  
   switch(buf->type) {
   case 'd': /* double */
   case 'f': /* float */           G__value_ref<double>(*buf) *= value; break;
   case 'w': /* logic */        
   case 'r': /* unsigned short */  G__value_ref<unsigned short>(*buf) *= value; break;
   case 'h': /* unsigned int */    G__value_ref<unsigned int>(*buf) *= value; break;
   case 'b': /* unsigned char */   G__value_ref<unsigned char>(*buf) *= value; break;
   case 'k': /* unsigned long */   G__value_ref<unsigned long>(*buf) *= value; break;
   case 'n':       G__value_ref<long long>(*buf) *= value; break;
   case 'm':       G__value_ref<unsigned long long>(*buf) *= value; break;
   case 'q':       G__value_ref<long double>(*buf) *= value; break;
   case 'i':       G__value_ref<int>(*buf) *= value; break;
   case 'c':       G__value_ref<char>(*buf) *= value; break;
   case 's':       G__value_ref<short>(*buf) *= value; break;
   default: 
     /* long */
      G__value_ref<int>(*buf) *= value; break;
   }
}
  
template <class T> void G__DivAssign(G__value *buf, T value) 
{  
   switch(buf->type) {
   case 'd': /* double */
   case 'f': /* float */           G__value_ref<double>(*buf) /= value; break;
   case 'w': /* logic */        
   case 'r': /* unsigned short */  G__value_ref<unsigned short>(*buf) /= value; break;
   case 'h': /* unsigned int */    G__value_ref<unsigned int>(*buf) /= value; break;
   case 'b': /* unsigned char */   G__value_ref<unsigned char>(*buf) /= value; break;
   case 'k': /* unsigned long */   G__value_ref<unsigned long>(*buf) /= value; break;
   case 'n':       G__value_ref<long long>(*buf) /= value; break;
   case 'm':       G__value_ref<unsigned long long>(*buf) /= value; break;
   case 'q':       G__value_ref<long double>(*buf) /= value; break;
   case 'i':       G__value_ref<int>(*buf) /= value; break;
   case 'c':       G__value_ref<char>(*buf) /= value; break;
   case 's':       G__value_ref<short>(*buf) /= value; break;
   default: 
     /* long */
      G__value_ref<int>(*buf) /= value; break;
   }
}
  
template <class T> void G__ModAssign(G__value *buf, T value) 
{  
   switch(buf->type) {
   case 'd': /* double */
   case 'f': /* float */           break; // G__value_ref<double>(*buf)%= value; break;
   case 'w': /* logic */        
   case 'r': /* unsigned short */  G__value_ref<unsigned short>(*buf) %= value; break;
   case 'h': /* unsigned int */    G__value_ref<unsigned int>(*buf) %= value; break;
   case 'b': /* unsigned char */   G__value_ref<unsigned char>(*buf) %= value; break;
   case 'k': /* unsigned long */   G__value_ref<unsigned long>(*buf) %= value; break;
   case 'n':       G__value_ref<long long>(*buf) %= value; break;
   case 'm':       G__value_ref<unsigned long long>(*buf) %= value; break;
   case 'q':       break; // G__value_ref<long double>(*buf) %= value; break;
   case 'i':       G__value_ref<int>(*buf) %= value; break;
   case 'c':       G__value_ref<char>(*buf) %= value; break;
   case 's':       G__value_ref<short>(*buf) %= value; break;
   default: 
     /* long */
      G__value_ref<int>(*buf) %= value; break;
   }
}
  

extern "C" {

void G__LD_p0_uchar(LDARGS){G__ASM_GET_INT<unsigned char>(LDARGCALL);}
void G__LD_p0_char(LDARGS){G__ASM_GET_INT<char>(LDARGCALL);}
void G__LD_p0_ushort(LDARGS){G__ASM_GET_INT<unsigned short>(LDARGCALL);}
void G__LD_p0_short(LDARGS){G__ASM_GET_INT<short>(LDARGCALL);}
void G__LD_p0_uint(LDARGS){G__ASM_GET_INT<unsigned int>(LDARGCALL);}
void G__LD_p0_int(LDARGS){G__ASM_GET_INT<int>(LDARGCALL);}
void G__LD_p0_ulong(LDARGS){G__ASM_GET_INT<unsigned long>(LDARGCALL);}
void G__LD_p0_long(LDARGS){G__ASM_GET_INT<long>(LDARGCALL);}
void G__LD_p0_ulonglong(LDARGS){G__ASM_GET_INT<G__uint64>(LDARGCALL);}
void G__LD_p0_longlong(LDARGS){G__ASM_GET_INT<G__int64>(LDARGCALL);}
void G__LD_p0_bool(LDARGS){G__ASM_GET_INT<bool>(LDARGCALL);}
void G__LD_p0_float(LDARGS){G__ASM_GET_INT<float>(LDARGCALL);}
void G__LD_p0_double(LDARGS){G__ASM_GET_INT<double>(LDARGCALL);}
void G__LD_p0_longdouble(LDARGS){G__ASM_GET_INT<long double>(LDARGCALL);}
   

/****************************************************************
* G__LD_p0_pointer()
****************************************************************/
void G__LD_p0_pointer(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
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
void G__LD_p0_struct(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = 'u';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset;
  buf->obj.i = buf->ref;
  buf->obj.reftype.reftype=G__PARANORMAL;
}
/*************************************************************************
* G__nonintarrayindex
*************************************************************************/
void G__nonintarrayindex G__P((struct G__var_array*,int));
void G__nonintarrayindex(struct G__var_array *var,int ig15)
{
  G__fprinterr(G__serr,"Error: %s[] invalud type for array index"
               ,var->varnamebuf[ig15]);
  G__genericerror((char*)NULL);
}
} // extern "C"

template <typename T>
void G__ASM_GET_INT_P1(G__value* pbuf, int* psp, long offset, G__var_array* var, long ig15)
{
  G__value* buf = &pbuf[*psp-1];

#ifdef G__TUNEUP_W_SECURITY
  if ((buf->type == 'd') || (buf->type == 'f')) {
    G__nonintarrayindex(var, ig15);
  }
#endif

  buf->ref = var->p[ig15] + offset + (G__convertT<long>(buf) * sizeof(T));

#ifdef G__TUNEUP_W_SECURITY
  /* We intentionally allow going one beyond the end. */
  if (G__convertT<size_t>(buf) > var->varlabel[ig15][1] /* num of elements */) {
    G__arrayindexerror(ig15, var, var->varnamebuf[ig15], G__convertT<long>(buf));
  }
  else
#endif
     G__setvalue(buf, *(T*)buf->ref);

  buf->tagnum = -1;
  buf->type = G__gettypechar<T>();
  buf->typenum = var->p_typetable[ig15];
}

extern "C" {
/*************************************************************************
* G__LD_p1_xxx
*************************************************************************/

void G__LD_p1_longlong(LDARGS){G__ASM_GET_INT_P1<G__int64>(LDARGCALL);}
void G__LD_p1_ulonglong(LDARGS){G__ASM_GET_INT_P1<G__uint64>(LDARGCALL);}
void G__LD_p1_longdouble(LDARGS){G__ASM_GET_INT_P1<long double>(LDARGCALL);}
void G__LD_p1_bool(LDARGS){G__ASM_GET_INT_P1<bool>(LDARGCALL);}
void G__LD_p1_char(LDARGS){G__ASM_GET_INT_P1<char>(LDARGCALL);}
void G__LD_p1_uchar(LDARGS){G__ASM_GET_INT_P1<unsigned char>(LDARGCALL);}
void G__LD_p1_short(LDARGS){G__ASM_GET_INT_P1<short>(LDARGCALL);}
void G__LD_p1_ushort(LDARGS){G__ASM_GET_INT_P1<unsigned short>(LDARGCALL);}
void G__LD_p1_int(LDARGS){G__ASM_GET_INT_P1<int>(LDARGCALL);}
void G__LD_p1_uint(LDARGS){G__ASM_GET_INT_P1<unsigned int>(LDARGCALL);}
void G__LD_p1_long(LDARGS){G__ASM_GET_INT_P1<long>(LDARGCALL);}
void G__LD_p1_ulong(LDARGS){G__ASM_GET_INT_P1<unsigned long>(LDARGCALL);}

/****************************************************************
* G__LD_p1_pointer()
****************************************************************/
void G__LD_p1_pointer(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  G__value *buf= &pbuf[*psp-1];
  if('d'==buf->type||'f'==buf->type) G__nonintarrayindex(var,ig15);
  buf->ref = var->p[ig15]+offset+G__convertT<long>(buf)*sizeof(long);
#ifdef G__TUNEUP_W_SECURITY
  // We intentionally allow going one beyond the end.
  if (G__convertT<size_t>(buf) > var->varlabel[ig15][1] /* num of elements */)
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],G__convertT<long>(buf));
  else
#endif
    buf->obj.i = *(long*)buf->ref;
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = var->type[ig15];
  buf->typenum = var->p_typetable[ig15];
  buf->obj.reftype.reftype=G__PARANORMAL;
}
/****************************************************************
* G__LD_p1_struct()
****************************************************************/
void G__LD_p1_struct(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  G__value *buf= &pbuf[*psp-1];
  if('d'==buf->type||'f'==buf->type) G__nonintarrayindex(var,ig15);
  size_t index = G__convertT<size_t>(buf);

  buf->tagnum = var->p_tagtable[ig15];
  buf->type = 'u';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset+index*G__struct.size[buf->tagnum];
#ifdef G__TUNEUP_W_SECURITY
  // We intentionally allow going one beyond the end.
  if (index > var->varlabel[ig15][1] /* num of elements */)
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],index);
  else
#endif
    buf->obj.i = buf->ref;
  buf->obj.reftype.reftype=G__PARANORMAL;
}
/****************************************************************
* G__LD_p1_float()
****************************************************************/
void G__LD_p1_float(LDARGS){G__ASM_GET_INT_P1<float>(LDARGCALL);}
void G__LD_p1_double(LDARGS){G__ASM_GET_INT_P1<double>(LDARGCALL);}
} // extern "C"

/*************************************************************************
* G__LD_pn_xxx
*************************************************************************/

template <typename T>
void G__ASM_GET_INT_PN(G__value* pbuf, int* psp, long offset, G__var_array* var, long ig15) 
{
  *psp = *psp - var->paran[ig15];
  G__value* buf = &pbuf[*psp];
  int ary = var->varlabel[ig15][0] /* stride */;
  int paran = var->paran[ig15];
  size_t p_inc = 0;
  ++(*psp);
  for (int ig25 = 0; (ig25 < paran) && (ig25 < var->paran[ig15]); ++ig25) {
    p_inc += ary * G__int(buf[ig25]);
    ary /= var->varlabel[ig15][ig25+2];
  }
  buf->tagnum = -1;
  buf->type = G__gettypechar<T>();
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15] + offset + (p_inc * sizeof(T));
#ifdef G__TUNEUP_W_SECURITY
  /* We intentionally allow going one beyond the end. */
  if (p_inc > var->varlabel[ig15][1] /* num of elements */) {
    G__arrayindexerror(ig15, var, var->varnamebuf[ig15], p_inc);
  }
  else
#endif
     G__setvalue(buf, *(T*)buf->ref);
}

extern "C" {

/****************************************************************
* G__LD_pn_longlong()
****************************************************************/
void G__LD_pn_longlong(LDARGS) {G__ASM_GET_INT_PN<G__int64>(LDARGCALL);}
void G__LD_pn_ulonglong(LDARGS) {G__ASM_GET_INT_PN<G__uint64>(LDARGCALL);}
void G__LD_pn_longdouble(LDARGS) {G__ASM_GET_INT_PN<long double>(LDARGCALL);}
void G__LD_pn_bool(LDARGS) {G__ASM_GET_INT_PN<bool>(LDARGCALL);}
void G__LD_pn_char(LDARGS) {G__ASM_GET_INT_PN<char>(LDARGCALL);}
void G__LD_pn_uchar(LDARGS) {G__ASM_GET_INT_PN<unsigned char>(LDARGCALL);}
void G__LD_pn_short(LDARGS) {G__ASM_GET_INT_PN<short>(LDARGCALL);}
void G__LD_pn_ushort(LDARGS) {G__ASM_GET_INT_PN<unsigned short>(LDARGCALL);}
void G__LD_pn_int(LDARGS) {G__ASM_GET_INT_PN<int>(LDARGCALL);}
void G__LD_pn_uint(LDARGS) {G__ASM_GET_INT_PN<unsigned int>(LDARGCALL);}
void G__LD_pn_long(LDARGS) {G__ASM_GET_INT_PN<long>(LDARGCALL);}
void G__LD_pn_ulong(LDARGS) {G__ASM_GET_INT_PN<unsigned long>(LDARGCALL);}
/****************************************************************
* G__LD_pn_pointer()
****************************************************************/
void G__LD_pn_pointer(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];
  int ary = var->varlabel[ig15][0] /* stride */;
  int paran = var->paran[ig15];
  size_t p_inc=0;
  int ig25;
  ++(*psp);
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
    p_inc += ary*G__int(buf[ig25]);
    ary /= var->varlabel[ig15][ig25+2];
  }
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = var->type[ig15];
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset+p_inc*sizeof(long);
#ifdef G__TUNEUP_W_SECURITY
  // We intentionally allow going one beyond the end.
  if (p_inc > var->varlabel[ig15][1] /* num of elements */)
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);
  else
#endif
    buf->obj.i = *(long*)buf->ref;
  buf->obj.reftype.reftype=var->reftype[ig15]; /* ?? for G__LD_p1_pointer */
}
/****************************************************************
* G__LD_pn_struct()
****************************************************************/
void G__LD_pn_struct(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];
  int ary = var->varlabel[ig15][0] /* stride */;
  int paran = var->paran[ig15];
  size_t p_inc=0;
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
  buf->obj.reftype.reftype=G__PARANORMAL;
#ifdef G__TUNEUP_W_SECURITY
  // We intentionally allow going one beyond the end.
  if (p_inc > var->varlabel[ig15][1] /* num of elements */)
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],p_inc);
  else
#endif
    buf->obj.i = buf->ref;
}
/****************************************************************
* G__LD_pn_float()
****************************************************************/
void G__LD_pn_float(LDARGS) {G__ASM_GET_INT_PN<float>(LDARGCALL);}
   void G__LD_pn_double(LDARGS) {G__ASM_GET_INT_PN<double>(LDARGCALL);}
} // extern "C"

/*************************************************************************
* G__LD_P10_xxx
*************************************************************************/

template <typename T>
void G__ASM_GET_INT_P10(G__value* pbuf, int* psp, long offset, G__var_array* var, long ig15)
{
  G__value *buf= &pbuf[*psp-1];
  buf->ref = *(long*)(var->p[ig15]+offset)+G__convertT<long>(buf)*sizeof(T);
  buf->tagnum = -1;
  buf->type = G__gettypechar<T>();
  buf->typenum = var->p_typetable[ig15];
  G__setvalue(buf, *(T*)buf->ref);
}


extern "C" {

/****************************************************************
* G__LD_P10_longlong()
****************************************************************/
void G__LD_P10_longlong(LDARGS) {G__ASM_GET_INT_P10<G__int64>(LDARGCALL);}
void G__LD_P10_ulonglong(LDARGS) {G__ASM_GET_INT_P10<G__uint64>(LDARGCALL);}
void G__LD_P10_longdouble(LDARGS) {G__ASM_GET_INT_P10<long double>(LDARGCALL);}
void G__LD_P10_bool(LDARGS) {G__ASM_GET_INT_P10<bool>(LDARGCALL);}
void G__LD_P10_char(LDARGS) {G__ASM_GET_INT_P10<char>(LDARGCALL);}
void G__LD_P10_uchar(LDARGS) {G__ASM_GET_INT_P10<unsigned char>(LDARGCALL);}
void G__LD_P10_short(LDARGS) {G__ASM_GET_INT_P10<short>(LDARGCALL);}
void G__LD_P10_ushort(LDARGS) {G__ASM_GET_INT_P10<unsigned short>(LDARGCALL);}
void G__LD_P10_int(LDARGS) {G__ASM_GET_INT_P10<int>(LDARGCALL);}
void G__LD_P10_uint(LDARGS) {G__ASM_GET_INT_P10<unsigned int>(LDARGCALL);}
void G__LD_P10_long(LDARGS) {G__ASM_GET_INT_P10<long>(LDARGCALL);}
void G__LD_P10_ulong(LDARGS) {G__ASM_GET_INT_P10<unsigned long>(LDARGCALL);}
/****************************************************************
* G__LD_P10_pointer()
****************************************************************/
void G__LD_P10_pointer(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  G__value *buf= &pbuf[*psp-1];
  buf->ref = *(long*)(var->p[ig15]+offset)+G__convertT<long>(buf)*sizeof(long);
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = var->type[ig15];
  buf->typenum = var->p_typetable[ig15];
  buf->obj.i = *(long*)buf->ref;
  buf->obj.reftype.reftype=G__PARANORMAL;
}
/****************************************************************
* G__LD_P10_struct()
****************************************************************/
void G__LD_P10_struct(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  G__value *buf= &pbuf[*psp-1];
  long index = G__convertT<long>(buf);
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = 'u';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = *(long*)(var->p[ig15]+offset)
        +index*G__struct.size[buf->tagnum];
  buf->obj.i = buf->ref;
  buf->obj.reftype.reftype=G__PARANORMAL;
}
/****************************************************************
* G__LD_P10_float()
****************************************************************/
void G__LD_P10_float(LDARGS) {G__ASM_GET_INT_P10<float>(LDARGCALL);}
   void G__LD_P10_double(LDARGS) {G__ASM_GET_INT_P10<double>(LDARGCALL);}
} // extern "C"

/*************************************************************************
* G__ST_p0_xxx
*************************************************************************/

template <typename T>
void G__ASM_ASSIGN_INT(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
   G__value *val = &pbuf[*psp-1];
   *(T*)(var->p[ig15]+offset) = G__convertT<T>(val);
}

extern "C" {

void G__ST_p0_longlong(LDARGS) {G__ASM_ASSIGN_INT<G__int64>(LDARGCALL);}
void G__ST_p0_ulonglong(LDARGS) {G__ASM_ASSIGN_INT<G__uint64>(LDARGCALL);}
void G__ST_p0_longdouble(LDARGS) {G__ASM_ASSIGN_INT<long double>(LDARGCALL);}
void G__ST_p0_bool(LDARGS) {G__ASM_ASSIGN_INT<bool>(LDARGCALL);}
void G__ST_p0_char(LDARGS) {G__ASM_ASSIGN_INT<char>(LDARGCALL);}
void G__ST_p0_uchar(LDARGS) {G__ASM_ASSIGN_INT<unsigned char>(LDARGCALL);}
void G__ST_p0_short(LDARGS) {G__ASM_ASSIGN_INT<short>(LDARGCALL);}
void G__ST_p0_ushort(LDARGS) {G__ASM_ASSIGN_INT<unsigned short>(LDARGCALL);}
void G__ST_p0_int(LDARGS) {G__ASM_ASSIGN_INT<int>(LDARGCALL);}
void G__ST_p0_uint(LDARGS) {G__ASM_ASSIGN_INT<unsigned int>(LDARGCALL);}
void G__ST_p0_long(LDARGS) {G__ASM_ASSIGN_INT<long>(LDARGCALL);}
void G__ST_p0_ulong(LDARGS) {G__ASM_ASSIGN_INT<unsigned long>(LDARGCALL);}
/****************************************************************
* G__ST_p0_pointer()
****************************************************************/
void G__ST_p0_pointer(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
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
void G__ST_p0_struct(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  memcpy((void*)(var->p[ig15]+offset),(void*)G__convertT<long>(&pbuf[*psp-1])
         ,G__struct.size[var->p_tagtable[ig15]]);
}
/****************************************************************
* G__ST_p0_float()
****************************************************************/
void G__ST_p0_float(LDARGS) {G__ASM_ASSIGN_INT<float>(LDARGCALL);}
void G__ST_p0_double(LDARGS) {G__ASM_ASSIGN_INT<double>(LDARGCALL);}
} // extern "C"

/*************************************************************************
* G__ST_p1_xxx
*************************************************************************/

template <typename T>
void G__ASM_ASSIGN_INT_P1(G__value* pbuf, int* psp, long offset, G__var_array* var, long ig15)
{
  G__value* val = &pbuf[*psp-1];
#ifdef G__TUNEUP_W_SECURITY
  if ((val->type == 'd') || (val->type == 'f'))
    G__nonintarrayindex(var, ig15);
  /* We intentionally allow going one beyond the end. */
  if (G__convertT<size_t>(val) > var->varlabel[ig15][1] /* num of elements */)
    G__arrayindexerror(ig15, var, var->varnamebuf[ig15], G__convertT<long>(val));
  else
#endif
     *((T*) (var->p[ig15] + offset + (G__convertT<long>(val) * sizeof(T)))) 
        = G__convertT<T>(&pbuf[*psp-2]);
  --(*psp);
}


extern "C" {
   
/****************************************************************
* G__ST_p1_longlong()
****************************************************************/
void G__ST_p1_longlong(LDARGS) {G__ASM_ASSIGN_INT_P1<G__int64>(LDARGCALL);}
void G__ST_p1_ulonglong(LDARGS) {G__ASM_ASSIGN_INT_P1<G__uint64>(LDARGCALL);}
void G__ST_p1_longdouble(LDARGS) {G__ASM_ASSIGN_INT_P1<long double>(LDARGCALL);}
void G__ST_p1_bool(LDARGS) {G__ASM_ASSIGN_INT_P1<bool>(LDARGCALL);}
void G__ST_p1_char(LDARGS) {G__ASM_ASSIGN_INT_P1<char>(LDARGCALL);}
void G__ST_p1_uchar(LDARGS) {G__ASM_ASSIGN_INT_P1<unsigned char>(LDARGCALL);}
void G__ST_p1_short(LDARGS) {G__ASM_ASSIGN_INT_P1<short>(LDARGCALL);}
void G__ST_p1_ushort(LDARGS) {G__ASM_ASSIGN_INT_P1<unsigned short>(LDARGCALL);}
void G__ST_p1_int(LDARGS) {G__ASM_ASSIGN_INT_P1<int>(LDARGCALL);}
void G__ST_p1_uint(LDARGS) {G__ASM_ASSIGN_INT_P1<unsigned int>(LDARGCALL);}
void G__ST_p1_long(LDARGS) {G__ASM_ASSIGN_INT_P1<long>(LDARGCALL);}
void G__ST_p1_ulong(LDARGS) {G__ASM_ASSIGN_INT_P1<unsigned long>(LDARGCALL);}
/****************************************************************
* G__ST_p1_pointer()
****************************************************************/
void G__ST_p1_pointer(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  G__value *val = &pbuf[*psp-1];
  if('d'==val->type||'f'==val->type) G__nonintarrayindex(var,ig15);
  // We intentionally allow going one beyond the end.
  if (G__convertT<size_t>(val) > var->varlabel[ig15][1] /* num of elements */) {
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],G__convertT<long>(val));
  }
  else {
    long address = (var->p[ig15]+offset+G__convertT<long>(val)*sizeof(long));
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
void G__ST_p1_struct(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  G__value *val = &pbuf[*psp-1];
  if('d'==val->type||'f'==val->type) G__nonintarrayindex(var,ig15);
#ifdef G__TUNEUP_W_SECURITY
  // We intentionally allow going one beyond the end.
  if (G__convertT<size_t>(val) > var->varlabel[ig15][1] /* num of elements */)
    G__arrayindexerror(ig15,var,var->varnamebuf[ig15],G__convertT<long>(val));
  else
#endif
    memcpy((void*)(var->p[ig15]+offset
                 +G__convertT<long>(val)*G__struct.size[var->p_tagtable[ig15]])
         ,(void*)pbuf[*psp-2].obj.i,G__struct.size[var->p_tagtable[ig15]]);
  --(*psp);
}
/****************************************************************
* G__ST_p1_float()
****************************************************************/
void G__ST_p1_float(LDARGS) {G__ASM_ASSIGN_INT_P1<float>(LDARGCALL);}
void G__ST_p1_double(LDARGS) {G__ASM_ASSIGN_INT_P1<double>(LDARGCALL);}
} // extern "C"

/*************************************************************************
* G__ST_pn_xxx
*************************************************************************/

template <typename T>
void G__ASM_ASSIGN_INT_PN(G__value* pbuf, int* psp, long offset, G__var_array* var, long ig15) 
{
  *psp = *psp-var->paran[ig15];
  G__value* buf= &pbuf[*psp];
  int ary = var->varlabel[ig15][0]; /* stride */
  int paran = var->paran[ig15];
  size_t p_inc = 0;
  int ig25;
  for(ig25 = 0; ig25 < paran && ig25 < var->paran[ig15]; ++ig25) {
    p_inc += ary * G__int(buf[ig25]);
    ary /= var->varlabel[ig15][ig25+2];
  }
#ifdef G__TUNEUP_W_SECURITY
  /* We intentionally allow going one beyond the end. */
  if (p_inc > var->varlabel[ig15][1] /* num of elements */)
    G__arrayindexerror(ig15, var, var->varnamebuf[ig15], p_inc);
  else
#endif
     *((T*) (var->p[ig15] + offset + (p_inc * sizeof(T)))) = G__convertT<T>(&pbuf[*psp-1]);
}

extern "C" {

/****************************************************************
* G__ST_pn_longlong()
****************************************************************/
void G__ST_pn_longlong(LDARGS) {G__ASM_ASSIGN_INT_PN<G__int64>(LDARGCALL);}
void G__ST_pn_ulonglong(LDARGS) {G__ASM_ASSIGN_INT_PN<G__uint64>(LDARGCALL);}
void G__ST_pn_longdouble(LDARGS) {G__ASM_ASSIGN_INT_PN<long double>(LDARGCALL);}
void G__ST_pn_bool(LDARGS) {G__ASM_ASSIGN_INT_PN<bool>(LDARGCALL);}
void G__ST_pn_char(LDARGS) {G__ASM_ASSIGN_INT_PN<char>(LDARGCALL);}
void G__ST_pn_uchar(LDARGS) {G__ASM_ASSIGN_INT_PN<unsigned char>(LDARGCALL);}
void G__ST_pn_short(LDARGS) {G__ASM_ASSIGN_INT_PN<short>(LDARGCALL);}
void G__ST_pn_ushort(LDARGS) {G__ASM_ASSIGN_INT_PN<unsigned short>(LDARGCALL);}
void G__ST_pn_int(LDARGS) {G__ASM_ASSIGN_INT_PN<int>(LDARGCALL);}
void G__ST_pn_uint(LDARGS) {G__ASM_ASSIGN_INT_PN<unsigned int>(LDARGCALL);}
void G__ST_pn_long(LDARGS) {G__ASM_ASSIGN_INT_PN<long>(LDARGCALL);}
void G__ST_pn_ulong(LDARGS) {G__ASM_ASSIGN_INT_PN<unsigned long>(LDARGCALL);}
/****************************************************************
* G__ST_pn_pointer()
****************************************************************/
void G__ST_pn_pointer(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];
  int ary = var->varlabel[ig15][0] /* stride */;
  int paran = var->paran[ig15];
  size_t p_inc=0;
  int ig25;
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
    p_inc += ary*G__int(buf[ig25]);
    ary /= var->varlabel[ig15][ig25+2];
  }
  // We intentionally allow going one beyond the end.
  if (p_inc > var->varlabel[ig15][1] /* num of elements */) {
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
void G__ST_pn_struct(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  G__value *buf= &pbuf[(*psp = *psp-var->paran[ig15])];
  int ary = var->varlabel[ig15][0] /* stride */;
  int paran = var->paran[ig15];
  size_t p_inc=0;
  int ig25;
  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
    p_inc += ary*G__int(buf[ig25]);
    ary /= var->varlabel[ig15][ig25+2];
  }
#ifdef G__TUNEUP_W_SECURITY
  // We intentionally allow going one beyond the end.
  if (p_inc > var->varlabel[ig15][1] /* num of elements */)
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
void G__ST_pn_float(LDARGS) {G__ASM_ASSIGN_INT_PN<float>(LDARGCALL);}
void G__ST_pn_double(LDARGS) {G__ASM_ASSIGN_INT_PN<double>(LDARGCALL);}
} // extern "C"

/*************************************************************************
* G__ST_P10_xxx
*************************************************************************/

template <typename T>
void G__ASM_ASSIGN_INT_P10(G__value* pbuf, int* psp, long offset, G__var_array* var, long ig15) 
{
  G__value *val = &pbuf[*psp-1];
  *(T*)(*(long*)(var->p[ig15]+offset)+G__convertT<long>(val)*sizeof(T))
            = G__convertT<T>(&pbuf[*psp-2]);
  --(*psp);
}

extern "C" {
void G__ST_P10_longlong(LDARGS){G__ASM_ASSIGN_INT_P10<G__int64>(LDARGCALL);}
void G__ST_P10_ulonglong(LDARGS){G__ASM_ASSIGN_INT_P10<G__uint64>(LDARGCALL);}
void G__ST_P10_longdouble(LDARGS){G__ASM_ASSIGN_INT_P10<long double>(LDARGCALL);}
void G__ST_P10_bool(LDARGS){G__ASM_ASSIGN_INT_P10<bool>(LDARGCALL);}
void G__ST_P10_char(LDARGS){G__ASM_ASSIGN_INT_P10<char>(LDARGCALL);}
void G__ST_P10_uchar(LDARGS){G__ASM_ASSIGN_INT_P10<unsigned char>(LDARGCALL);}
void G__ST_P10_short(LDARGS){G__ASM_ASSIGN_INT_P10<short>(LDARGCALL);}
void G__ST_P10_ushort(LDARGS){G__ASM_ASSIGN_INT_P10<unsigned short>(LDARGCALL);}
void G__ST_P10_int(LDARGS){G__ASM_ASSIGN_INT_P10<int>(LDARGCALL);}
void G__ST_P10_uint(LDARGS){G__ASM_ASSIGN_INT_P10<unsigned int>(LDARGCALL);}
void G__ST_P10_long(LDARGS){G__ASM_ASSIGN_INT_P10<long>(LDARGCALL);}
void G__ST_P10_ulong(LDARGS){G__ASM_ASSIGN_INT_P10<unsigned long>(LDARGCALL);}
/****************************************************************
* G__ST_P10_pointer()
****************************************************************/
void G__ST_P10_pointer(LDARGS){G__ASM_ASSIGN_INT_P10<long>(LDARGCALL);}
/****************************************************************
* G__ST_P10_struct()
****************************************************************/
void G__ST_P10_struct(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  G__value *val = &pbuf[*psp-1];
  memcpy((void*)(*(long*)(var->p[ig15]+offset)
                 +G__convertT<long>(val)*G__struct.size[var->p_tagtable[ig15]])
         ,(void*)pbuf[*psp-2].obj.i,G__struct.size[var->p_tagtable[ig15]]);
  --(*psp);
}

/****************************************************************
* G__ST_P10_float()
****************************************************************/
void G__ST_P10_float(LDARGS){G__ASM_ASSIGN_INT_P10<float>(LDARGCALL);}
void G__ST_P10_double(LDARGS){G__ASM_ASSIGN_INT_P10<double>(LDARGCALL);}
} // extern "C"

/*************************************************************************
* G__LD_Rp0_xxx
*
*  type &p;
*  p;    optimize this expression
*************************************************************************/

/****************************************************************
* G__ASM_GET_REFINT
****************************************************************/
template <typename T>
void G__ASM_GET_REFINT(G__value* pbuf, int* psp, long offset, G__var_array* var, long ig15) 
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = -1;
  buf->type = G__gettypechar<T>();
  buf->typenum = var->p_typetable[ig15];
  buf->ref = *(long*)(var->p[ig15]+offset);
  G__setvalue(buf, *(T*)buf->ref);
}


extern "C" {

void G__LD_Rp0_longlong(LDARGS) {G__ASM_GET_REFINT<G__int64>(LDARGCALL);}
void G__LD_Rp0_ulonglong(LDARGS) {G__ASM_GET_REFINT<G__uint64>(LDARGCALL);}
void G__LD_Rp0_longdouble(LDARGS) {G__ASM_GET_REFINT<long double>(LDARGCALL);}
void G__LD_Rp0_bool(LDARGS) {G__ASM_GET_REFINT<bool>(LDARGCALL);}
void G__LD_Rp0_char(LDARGS) {G__ASM_GET_REFINT<char>(LDARGCALL);}
void G__LD_Rp0_uchar(LDARGS) {G__ASM_GET_REFINT<unsigned char>(LDARGCALL);}
void G__LD_Rp0_short(LDARGS) {G__ASM_GET_REFINT<short>(LDARGCALL);}
void G__LD_Rp0_ushort(LDARGS) {G__ASM_GET_REFINT<unsigned short>(LDARGCALL);}
void G__LD_Rp0_int(LDARGS) {G__ASM_GET_REFINT<int>(LDARGCALL);}
void G__LD_Rp0_uint(LDARGS) {G__ASM_GET_REFINT<unsigned int>(LDARGCALL);}
void G__LD_Rp0_long(LDARGS) {G__ASM_GET_REFINT<long>(LDARGCALL);}
void G__LD_Rp0_ulong(LDARGS) {G__ASM_GET_REFINT<unsigned long>(LDARGCALL);}
/****************************************************************
* G__LD_Rp0_pointer()
****************************************************************/
void G__LD_Rp0_pointer(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
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
void G__LD_Rp0_struct(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = var->p_tagtable[ig15];
  buf->type = 'u';
  buf->typenum = var->p_typetable[ig15];
  buf->ref = *(long*)(var->p[ig15]+offset);
  buf->obj.i = buf->ref;
  buf->obj.reftype.reftype=G__PARANORMAL;
}
/****************************************************************
* G__LD_Rp0_float()
****************************************************************/
void G__LD_Rp0_float(LDARGS) {G__ASM_GET_REFINT<float>(LDARGCALL);}
void G__LD_Rp0_double(LDARGS) {G__ASM_GET_REFINT<double>(LDARGCALL);}
} // extern "C"

/*************************************************************************
* G__ST_Rp0_xxx
*
*  type &p;
*  p=x;    optimize this expression
*************************************************************************/

/****************************************************************
* G__ASM_ASSIGN_REFINT
****************************************************************/
template <typename T>
void G__ASM_ASSIGN_REFINT(G__value* pbuf, int* psp, long offset, G__var_array* var, long ig15) 
{
  G__value *val = &pbuf[*psp-1];
  T* adr = *(T**)(var->p[ig15]+offset);
  *(T*)adr=G__convertT<T>(val);
}

extern "C" {
/****************************************************************
* G__ST_Rp0_longlong()
****************************************************************/
void G__ST_Rp0_longlong(LDARGS) {G__ASM_ASSIGN_REFINT<G__int64>(LDARGCALL);}
void G__ST_Rp0_ulonglong(LDARGS) {G__ASM_ASSIGN_REFINT<G__uint64>(LDARGCALL);}
void G__ST_Rp0_longdouble(LDARGS) {G__ASM_ASSIGN_REFINT<long double>(LDARGCALL);}
void G__ST_Rp0_bool(LDARGS) {G__ASM_ASSIGN_REFINT<bool>(LDARGCALL);}
void G__ST_Rp0_char(LDARGS) {G__ASM_ASSIGN_REFINT<char>(LDARGCALL);}
void G__ST_Rp0_uchar(LDARGS) {G__ASM_ASSIGN_REFINT<unsigned char>(LDARGCALL);}
void G__ST_Rp0_short(LDARGS) {G__ASM_ASSIGN_REFINT<short>(LDARGCALL);}
void G__ST_Rp0_ushort(LDARGS) {G__ASM_ASSIGN_REFINT<unsigned short>(LDARGCALL);}
void G__ST_Rp0_int(LDARGS) {G__ASM_ASSIGN_REFINT<int>(LDARGCALL);}
void G__ST_Rp0_uint(LDARGS) {G__ASM_ASSIGN_REFINT<unsigned int>(LDARGCALL);}
void G__ST_Rp0_long(LDARGS) {G__ASM_ASSIGN_REFINT<long>(LDARGCALL);}
void G__ST_Rp0_ulong(LDARGS) {G__ASM_ASSIGN_REFINT<unsigned long>(LDARGCALL);}
/****************************************************************
* G__ST_Rp0_pointer()
****************************************************************/
void G__ST_Rp0_pointer(LDARGS) {G__ASM_ASSIGN_REFINT<long>(LDARGCALL);}
/****************************************************************
* G__ST_Rp0_struct()
****************************************************************/
void G__ST_Rp0_struct(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  memcpy((void*)(*(long*)(var->p[ig15]+offset)),(void*)pbuf[*psp-1].obj.i
         ,G__struct.size[var->p_tagtable[ig15]]);
}
/****************************************************************
* G__ST_Rp0_float()
****************************************************************/
void G__ST_Rp0_float(LDARGS) {G__ASM_ASSIGN_REFINT<float>(LDARGCALL);}
void G__ST_Rp0_double(LDARGS) {G__ASM_ASSIGN_REFINT<double>(LDARGCALL);}
} // extern "C"

/*************************************************************************
* G__LD_RP0_xxx
*
*  type &p;
*  &p;    optimize this expression
*************************************************************************/

/****************************************************************
* G__ASM_GET_REFPINT
****************************************************************/
template <typename T>
void G__ASM_GET_REFPINT(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
{
  G__value *buf= &pbuf[(*psp)++];
  buf->tagnum = -1;
  buf->type = toupper(G__gettypechar<T>());
  buf->typenum = var->p_typetable[ig15];
  buf->ref = var->p[ig15]+offset;
  G__setvalue(buf, *(T*)buf->ref);
}

extern "C" {
/****************************************************************
* G__LD_RP0_longlong()
****************************************************************/
void G__LD_RP0_longlong(LDARGS) {G__ASM_GET_REFPINT<G__int64>(LDARGCALL);}
void G__LD_RP0_ulonglong(LDARGS) {G__ASM_GET_REFPINT<G__uint64>(LDARGCALL);}
void G__LD_RP0_longdouble(LDARGS) {G__ASM_GET_REFPINT<long double>(LDARGCALL);}
void G__LD_RP0_bool(LDARGS) {G__ASM_GET_REFPINT<bool>(LDARGCALL);}
void G__LD_RP0_char(LDARGS) {G__ASM_GET_REFPINT<char>(LDARGCALL);}
void G__LD_RP0_uchar(LDARGS) {G__ASM_GET_REFPINT<unsigned char>(LDARGCALL);}
void G__LD_RP0_short(LDARGS) {G__ASM_GET_REFPINT<short>(LDARGCALL);}
void G__LD_RP0_ushort(LDARGS) {G__ASM_GET_REFPINT<unsigned short>(LDARGCALL);}
void G__LD_RP0_int(LDARGS) {G__ASM_GET_REFPINT<int>(LDARGCALL);}
void G__LD_RP0_uint(LDARGS) {G__ASM_GET_REFPINT<unsigned int>(LDARGCALL);}
void G__LD_RP0_long(LDARGS) {G__ASM_GET_REFPINT<long>(LDARGCALL);}
void G__LD_RP0_ulong(LDARGS) {G__ASM_GET_REFPINT<unsigned long>(LDARGCALL);}
/****************************************************************
* G__LD_RP0_pointer()
****************************************************************/
void G__LD_RP0_pointer(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
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
void G__LD_RP0_struct(G__value *pbuf,int *psp,long offset,struct G__var_array *var,long ig15)
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
void G__LD_RP0_float(LDARGS) {G__ASM_GET_REFPINT<float>(LDARGCALL);}
void G__LD_RP0_double(LDARGS) {G__ASM_GET_REFPINT<double>(LDARGCALL);}

} // extern "C"

/****************************************************************
* G__OP2_OPTIMIZED_UU
****************************************************************/

extern "C" {
/*************************************************************************
* G__OP2_plus_uu()
*************************************************************************/
void G__OP2_plus_uu(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.ulo = G__convertT<unsigned long>(bufm2) + G__convertT<unsigned long>(bufm1);
  bufm2->type = 'k';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_minus_uu()
*************************************************************************/
void G__OP2_minus_uu(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.ulo = G__convertT<unsigned long>(bufm2) - G__convertT<unsigned long>(bufm1);
  bufm2->type = 'k';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_multiply_uu()
*************************************************************************/
void G__OP2_multiply_uu(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.ulo = G__convertT<unsigned long>(bufm2) * G__convertT<unsigned long>(bufm1);
  bufm2->type = 'k';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_divide_uu()
*************************************************************************/
void G__OP2_divide_uu(G__value *bufm1,G__value *bufm2)
{
  if(0==bufm1->obj.ulo) {
    G__genericerror("Error: operator '/' divided by zero");
    return;
  }
  bufm2->obj.ulo = G__convertT<unsigned long>(bufm2) / G__convertT<unsigned long>(bufm1);
  bufm2->type = 'k';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_addassign_uu()
*************************************************************************/
void G__OP2_addassign_uu(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.ulo = G__convertT<unsigned long>(bufm2);
  bufm2->obj.ulo += G__convertT<unsigned long>(bufm1);
  bufm2->type = 'k';
  *(unsigned int*)bufm2->ref=(unsigned int)bufm2->obj.ulo;
}
/*************************************************************************
* G__OP2_subassign_uu()
*************************************************************************/
void G__OP2_subassign_uu(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.ulo = G__convertT<unsigned long>(bufm2);
  bufm2->obj.ulo -= G__convertT<unsigned long>(bufm1);
  bufm2->type = 'k';
  *(unsigned int*)bufm2->ref=(unsigned int)bufm2->obj.ulo;
}
/*************************************************************************
* G__OP2_mulassign_uu()
*************************************************************************/
void G__OP2_mulassign_uu(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.ulo = G__convertT<unsigned long>(bufm2);
  bufm2->obj.ulo *= G__convertT<unsigned long>(bufm1);
  bufm2->type = 'k';
  *(unsigned int*)bufm2->ref=(unsigned int)bufm2->obj.ulo;
}
/*************************************************************************
* G__OP2_divassign_uu()
*************************************************************************/
void G__OP2_divassign_uu(G__value *bufm1,G__value *bufm2)
{
  bufm1->obj.ulo = G__convertT<unsigned long>(bufm1);
  bufm2->obj.ulo = G__convertT<unsigned long>(bufm2);
  if(0==bufm1->obj.ulo) {
    G__genericerror("Error: operator '/' divided by zero");
    return;
  }
  bufm2->type = 'k';
  bufm2->obj.ulo /= bufm1->obj.ulo;
  *(unsigned int*)bufm2->ref=(unsigned int)bufm2->obj.ulo;
}


/****************************************************************
* G__OP2_OPTIMIZED_II
****************************************************************/

/*************************************************************************
* G__OP2_plus_ii()
*************************************************************************/
void G__OP2_plus_ii(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.i = G__convertT<long>(bufm2) + G__convertT<long>(bufm1);
  bufm2->type = 'l';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_minus_ii()
*************************************************************************/
void G__OP2_minus_ii(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.i = G__convertT<long>(bufm2) - G__convertT<long>(bufm1);
  bufm2->type = 'l';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_multiply_ii()
*************************************************************************/
void G__OP2_multiply_ii(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.i = G__convertT<long>(bufm2) * G__convertT<long>(bufm1);
  bufm2->type = 'l';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_divide_ii()
*************************************************************************/
void G__OP2_divide_ii(G__value *bufm1,G__value *bufm2)
{
  bufm1->obj.i = G__convertT<long>(bufm1);
  bufm2->obj.i = G__convertT<long>(bufm2);
  if(0==bufm1->obj.i) {
    G__genericerror("Error: operator '/' divided by zero");
    return;
  }
  bufm2->obj.i = bufm2->obj.i / bufm1->obj.i;
  bufm2->type = 'l';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_addassign_ii()
*************************************************************************/
void G__OP2_addassign_ii(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.i = G__convertT<long>(bufm2);
  bufm2->obj.i += G__convertT<long>(bufm1);
  bufm2->type = 'l';
  *(int*)bufm2->ref=(int)bufm2->obj.i;
}
/*************************************************************************
* G__OP2_subassign_ii()
*************************************************************************/
void G__OP2_subassign_ii(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.i = G__convertT<long>(bufm2);
  bufm2->obj.i -= G__convertT<long>(bufm1);
  bufm2->type = 'l';
  *(int*)bufm2->ref=(int)bufm2->obj.i;
}
/*************************************************************************
* G__OP2_mulassign_ii()
*************************************************************************/
void G__OP2_mulassign_ii(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.i = G__convertT<unsigned long>(bufm2);
  bufm2->obj.i *= G__convertT<long>(bufm1);
  bufm2->type = 'l';
  *(int*)bufm2->ref=(int)bufm2->obj.i;
}
/*************************************************************************
* G__OP2_divassign_ii()
*************************************************************************/
void G__OP2_divassign_ii(G__value *bufm1,G__value *bufm2)
{
  bufm1->obj.i = G__convertT<unsigned long>(bufm1);
  bufm2->obj.i = G__convertT<unsigned long>(bufm2);
  if(0==bufm1->obj.i) {
    G__genericerror("Error: operator '/' divided by zero");
    return;
  }
  bufm2->obj.i /= bufm1->obj.i;
  bufm2->type = 'l';
  *(int*)bufm2->ref=(int)bufm2->obj.i;
}


/****************************************************************
* G__OP2_OPTIMIZED_DD
****************************************************************/

/*************************************************************************
* G__OP2_plus_dd()
*************************************************************************/
void G__OP2_plus_dd(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.d = bufm2->obj.d + bufm1->obj.d;
  bufm2->type = 'd';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_minus_dd()
*************************************************************************/
void G__OP2_minus_dd(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.d = bufm2->obj.d - bufm1->obj.d;
  bufm2->type = 'd';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_multiply_dd()
*************************************************************************/
void G__OP2_multiply_dd(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.d = bufm2->obj.d * bufm1->obj.d;
  bufm2->type = 'd';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_divide_dd()
*************************************************************************/
void G__OP2_divide_dd(G__value *bufm1,G__value *bufm2)
{
//  IEEE 754 defines that NaN inf has to be the output in cases where division by 0 occurs.
//  if(0==bufm1->obj.d) {
//    G__genericerror("Error: operator '/' divided by zero");
//    return;
//  }
  bufm2->obj.d = bufm2->obj.d / bufm1->obj.d;
  bufm2->type = 'd';
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__OP2_addassign_dd()
*************************************************************************/
void G__OP2_addassign_dd(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.d += bufm1->obj.d;
  *(double*)bufm2->ref=bufm2->obj.d;
}
/*************************************************************************
* G__OP2_subassign_dd()
*************************************************************************/
void G__OP2_subassign_dd(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.d -= bufm1->obj.d;
  *(double*)bufm2->ref=bufm2->obj.d;
}
/*************************************************************************
* G__OP2_mulassign_dd()
*************************************************************************/
void G__OP2_mulassign_dd(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.d *= bufm1->obj.d;
  *(double*)bufm2->ref=bufm2->obj.d;
}
/*************************************************************************
* G__OP2_divassign_dd()
*************************************************************************/
void G__OP2_divassign_dd(G__value *bufm1,G__value *bufm2)
{
//  IEEE 754 defines that NaN inf has to be the output in cases where division by 0 occurs.
//  if(0==bufm1->obj.d) {
//    G__genericerror("Error: operator '/' divided by zero");
//    return;
//  }
  bufm2->obj.d /= bufm1->obj.d;
  *(double*)bufm2->ref=bufm2->obj.d;
}

/****************************************************************
* G__OP2_OPTIMIZED_FD
****************************************************************/

/*************************************************************************
* G__OP2_addassign_fd()
*************************************************************************/
void G__OP2_addassign_fd(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.d += bufm1->obj.d;
  *(float*)bufm2->ref=(float)bufm2->obj.d;
}
/*************************************************************************
* G__OP2_subassign_fd()
*************************************************************************/
void G__OP2_subassign_fd(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.d -= bufm1->obj.d;
  *(float*)bufm2->ref=(float)bufm2->obj.d;
}
/*************************************************************************
* G__OP2_mulassign_fd()
*************************************************************************/
void G__OP2_mulassign_fd(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.d *= bufm1->obj.d;
  *(float*)bufm2->ref=(float)bufm2->obj.d;
}
/*************************************************************************
* G__OP2_divassign_fd()
*************************************************************************/
void G__OP2_divassign_fd(G__value *bufm1,G__value *bufm2)
{
//  IEEE 754 defines that NaN inf has to be the output in cases where division by 0 occurs.
//  if(0==bufm1->obj.d) {
//    G__genericerror("Error: operator '/' divided by zero");
//    return;
//  }
  bufm2->obj.d /= bufm1->obj.d;
  *(float*)bufm2->ref=(float)bufm2->obj.d;
}


/*************************************************************************
* G__OP2_addvoidptr()
*************************************************************************/
void G__OP2_addvoidptr(G__value *bufm1,G__value *bufm2)
{
  bufm2->obj.i += bufm1->obj.i;
}

/****************************************************************
* G__OP2_OPTIMIZED
****************************************************************/

/*************************************************************************
* G__OP2_plus()
*************************************************************************/
void G__OP2_plus(G__value *bufm1,G__value *bufm2)
{
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
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
      bufm2->obj.d = bufm2->obj.d + bufm1->obj.d;
    }
    else {
      bufm2->obj.d = bufm2->obj.d + G__convertT<double>(bufm1);
    }
    bufm2->type = 'd';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  else if(G__isdoubleM(bufm1)) {
    bufm2->obj.d =  G__convertT<double>(bufm2) + bufm1->obj.d;
    bufm2->type = 'd';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  else if(isupper(bufm2->type)) {
    bufm2->obj.i = bufm2->obj.i + bufm1->obj.i*G__sizeof(bufm2);
  }
  else if(isupper(bufm1->type)) {
    bufm2->obj.reftype.reftype = bufm1->obj.reftype.reftype;
    bufm2->obj.i = bufm2->obj.i*G__sizeof(bufm1) + bufm1->obj.i;
    /* bufm2->obj.i=(bufm2->obj.i-bufm1->obj.i)/G__sizeof(bufm2); */
    bufm2->type = bufm1->type;
    bufm2->tagnum = bufm1->tagnum;
    bufm2->typenum = bufm1->typenum;
  }
  else if(G__isunsignedM(bufm1)) {
    if(G__isunsignedM(bufm2))
      bufm2->obj.ulo = G__convertT<unsigned long>(bufm2) + G__convertT<unsigned long>(bufm1);
    else
      bufm2->obj.ulo = G__convertT<long>(bufm2) + G__convertT<unsigned long>(bufm1);
    bufm2->type = 'h';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  else {
    bufm2->obj.i = G__convertT<long>(bufm2) + G__convertT<long>(bufm1);
    bufm2->type = 'l';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  bufm2->ref = 0;
}

/*************************************************************************
* G__OP2_minus()
*************************************************************************/
void G__OP2_minus(G__value *bufm1,G__value *bufm2)
{
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
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
      bufm2->obj.d = bufm2->obj.d - bufm1->obj.d;
    }
    else {
      bufm2->obj.d = bufm2->obj.d - G__convertT<double>(bufm1);
    }
    bufm2->type = 'd';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  else if(G__isdoubleM(bufm1)) {
    bufm2->obj.d = G__convertT<double>(bufm2) - bufm1->obj.d;
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
    bufm2->obj.reftype.reftype = bufm1->obj.reftype.reftype;
    bufm2->obj.i =bufm2->obj.i*G__sizeof(bufm2) -bufm1->obj.i;
    bufm2->type = bufm1->type;
    bufm2->tagnum = bufm1->tagnum;
    bufm2->typenum = bufm1->typenum;
  }
  else if(G__isunsignedM(bufm1)) {
    if(G__isunsignedM(bufm2))
      bufm2->obj.ulo = G__convertT<unsigned long>(bufm2) - G__convertT<unsigned long>(bufm1);
    else
      bufm2->obj.ulo = G__convertT<long>(bufm2) - G__convertT<unsigned long>(bufm1);
    bufm2->type = 'h';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  else {
    bufm2->obj.i = G__convertT<long>(bufm2) - G__convertT<long>(bufm1);
    bufm2->type = 'l';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  bufm2->ref = 0;
}

/*************************************************************************
* G__OP2_multiply()
*************************************************************************/
void G__OP2_multiply(G__value *bufm1,G__value *bufm2)
{
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
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
      bufm2->obj.d = bufm2->obj.d * bufm1->obj.d;
    }
    else {
      bufm2->obj.d = bufm2->obj.d * G__convertT<double>(bufm1);
    }
    bufm2->type = 'd';
  }
  else if(G__isdoubleM(bufm1)) {
    bufm2->obj.d = G__convertT<double>(bufm2) * bufm1->obj.d;
    bufm2->type = 'd';
  }
  else if(G__isunsignedM(bufm1)) {
    if(G__isunsignedM(bufm2))
      bufm2->obj.ulo = G__convertT<unsigned long>(bufm2) * G__convertT<unsigned long>(bufm1);
    else
      bufm2->obj.ulo = G__convertT<long>(bufm2) *  G__convertT<unsigned long>(bufm1);
    bufm2->type = 'h';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  else {
    bufm2->obj.i = G__convertT<long>(bufm2) * G__convertT<long>(bufm1);
    bufm2->type = 'l';
  }
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__OP2_modulus()
*************************************************************************/
void G__OP2_modulus(G__value *bufm1,G__value *bufm2)
{
  if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.ll=G__Longlong(*bufm2)%G__Longlong(*bufm1);
    bufm2->type='n';
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.ull=G__ULonglong(*bufm2)%G__ULonglong(*bufm1);
    bufm2->type='m';
  }
  else
#ifdef G__TUNEUP_W_SECURITY
  if(0==bufm1->obj.i) {
    G__genericerror("Error: operator '%' divided by zero");
    return;
  }
#endif
  if(G__isunsignedM(bufm1)) {
    if(G__isunsignedM(bufm2))
      bufm2->obj.ulo = G__convertT<unsigned long>(bufm2) % G__convertT<unsigned long>(bufm1);
    else
      bufm2->obj.ulo = G__convertT<long>(bufm2) % G__convertT<unsigned long>(bufm1);
    bufm2->type = 'h';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  else if(G__isunsignedM(bufm2)) {
    bufm2->obj.ulo = G__convertT<unsigned long>(bufm2) % G__convertT<long>(bufm1);
    bufm2->type = 'h';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  else {
    bufm2->obj.i = G__convertT<long>(bufm2) % G__convertT<long>(bufm1);
    bufm2->type = 'i';
  }
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__OP2_divide()
*************************************************************************/
void G__OP2_divide(G__value *bufm1,G__value *bufm2)
{
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
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
#ifdef G__TUNEUP_W_SECURITY
//  IEEE 754 defines that NaN inf has to be the output in cases where division by 0 occurs.
//      if(0==bufm1->obj.d) {
//        G__genericerror("Error: operator '/' divided by zero");
//        return;
//      }
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
      bufm2->obj.d = bufm2->obj.d / G__convertT<double>(bufm1);
    }
    bufm2->type = 'd';
  }
  else if(G__isdoubleM(bufm1)) {
#ifdef G__TUNEUP_W_SECURITY
//  IEEE 754 defines that NaN inf has to be the output in cases where division by 0 occurs.
//     if(0==bufm1->obj.d) {
//      G__genericerror("Error: operator '/' divided by zero");
//      return;
//    }
#endif
    bufm2->obj.d = G__convertT<double>(bufm2) / bufm1->obj.d;
    bufm2->type = 'd';
  }
  else if(G__isunsignedM(bufm1)) {
#ifdef G__TUNEUP_W_SECURITY
    if(0==bufm1->obj.i) {
      G__genericerror("Error: operator '/' divided by zero");
      return;
    }
#endif
    if(G__isunsignedM(bufm2))
      bufm2->obj.ulo = G__convertT<unsigned long>(bufm2) / G__convertT<unsigned long>(bufm1);
    else
      bufm2->obj.ulo = G__convertT<long>(bufm2) / G__convertT<unsigned long>(bufm1);
    bufm2->type = 'h';
    bufm2->tagnum = bufm2->typenum = -1;
  }
  else {
#ifdef G__TUNEUP_W_SECURITY
    if(0==bufm1->obj.i) {
      G__genericerror("Error: operator '/' divided by zero");
      return;
    }
#endif
    bufm2->obj.i = G__convertT<long>(bufm2) / G__convertT<long>(bufm1);
    bufm2->type = 'l';
  }
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__OP2_logicaland()
*************************************************************************/
void G__OP2_logicaland(G__value *bufm1,G__value *bufm2)
{
  if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.i=G__Longlong(*bufm2)&&G__Longlong(*bufm1);
    bufm2->type='l';
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.i=G__ULonglong(*bufm2)&&G__ULonglong(*bufm1);
    bufm2->type='l';
  }
  else
  {
    bufm2->obj.i = G__convertT<long>(bufm2) && G__convertT<long>(bufm1);
    bufm2->type = 'l';
  }
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__OP2_logicalor()
*************************************************************************/
void G__OP2_logicalor(G__value *bufm1,G__value *bufm2)
{
  if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.i=G__Longlong(*bufm2)||G__Longlong(*bufm1);
    bufm2->type='l';
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.i=G__ULonglong(*bufm2)||G__ULonglong(*bufm1);
    bufm2->type='l';
  }
  else
  {
    bufm2->obj.i = G__convertT<long>(bufm2) ||  G__convertT<long>(bufm1);
    bufm2->type = 'l';
  }
  bufm2->tagnum = bufm2->typenum = -1;
  bufm2->ref = 0;
}
/*************************************************************************
* G__CMP2_equal()
*************************************************************************/
void G__CMP2_equal(G__value *bufm1,G__value *bufm2)
{
  if('U'==bufm1->type && 'U'==bufm2->type) G__publicinheritance(bufm1,bufm2);
  if(G__isdoubleM(bufm2)||G__isdoubleM(bufm1))
    bufm2->obj.i = (G__doubleM(bufm2)==G__doubleM(bufm1));
  else if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.i=G__Longlong(*bufm2) == G__Longlong(*bufm1);
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.i=G__ULonglong(*bufm2)== G__ULonglong(*bufm1);
  } else {
    bufm2->obj.i = ( G__convertT<long>(bufm2)  ==  G__convertT<long>(bufm1));
  }
  bufm2->type='l';
  bufm2->typenum = bufm2->tagnum= -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__CMP2_notequal()
*************************************************************************/
void G__CMP2_notequal(G__value *bufm1,G__value *bufm2)
{
  if('U'==bufm1->type && 'U'==bufm2->type) G__publicinheritance(bufm1,bufm2);
  if(G__isdoubleM(bufm2)||G__isdoubleM(bufm1))
    bufm2->obj.i = (G__doubleM(bufm2)!=G__doubleM(bufm1));
  else if('n'==bufm2->type || 'n'==bufm1->type) {
    bufm2->obj.i=G__Longlong(*bufm2) != G__Longlong(*bufm1);
  }
  else if('m'==bufm2->type || 'm'==bufm1->type) {
    bufm2->obj.i=G__ULonglong(*bufm2)!= G__ULonglong(*bufm1);
  } else
    bufm2->obj.i = ( G__convertT<long>(bufm2) !=  G__convertT<long>(bufm1));
  bufm2->type='l';
  bufm2->typenum = bufm2->tagnum= -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__CMP2_greaterorequal()
*************************************************************************/
void G__CMP2_greaterorequal(G__value *bufm1,G__value *bufm2)
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
void G__CMP2_lessorequal(G__value *bufm1,G__value *bufm2)
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
void G__CMP2_greater(G__value *bufm1,G__value *bufm2)
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
void G__CMP2_less(G__value *bufm1,G__value *bufm2)
{
  if(G__doubleM(bufm2)<G__doubleM(bufm1)) bufm2->obj.i = 1;
  else                                    bufm2->obj.i = 0;
  bufm2->type='i';
  bufm2->typenum = bufm2->tagnum= -1;
  bufm2->ref = 0;
}

/*************************************************************************
* G__realassign()
*************************************************************************/
#define G__realassign(p,v,t)      \
 switch(t) {                      \
 case 'd': *(double*)p=(double)v; break;  \
 case 'f': *(float*)p=(float)v;  break;  \
 }
/*************************************************************************
* G__intassign()
*************************************************************************/
#ifdef G__BOOL4BYTE
#define G__intassign(p,v,t)                              \
 switch(t) {                                             \
 case 'i': *(int*)p=(int)v;                      break;  \
 case 's': *(short*)p=(short)v;                  break;  \
 case 'c': *(char*)p=(char)v;                    break;  \
 case 'h': *(unsigned int*)p=(unsigned int)v;    break;  \
 case 'r': *(unsigned short*)p=(unsigned short)v;break;  \
 case 'b': *(unsigned char*)p=(unsigned char)v;  break;  \
 case 'k': *(unsigned long*)p=(unsigned long)v;  break;  \
 case 'g': *(int*)p=(int)(v?1:0);/*1604*/        break;  \
 default:  *(long*)p=(long)v;                    break;  \
 }
#else
#define G__intassign(p,v,t)                              \
 switch(t) {                                             \
 case 'i': *(int*)p=(int)v;                      break;  \
 case 's': *(short*)p=(short)v;                  break;  \
 case 'c': *(char*)p=(char)v;                    break;  \
 case 'h': *(unsigned int*)p=(unsigned int)v;    break;  \
 case 'r': *(unsigned short*)p=(unsigned short)v;break;  \
 case 'b': *(unsigned char*)p=(unsigned char)v;  break;  \
 case 'k': *(unsigned long*)p=(unsigned long)v;  break;  \
 case 'g': *(unsigned char*)p=(unsigned char)v?1:0;/*1604*/break;  \
 default:  *(long*)p=(long)v;                    break;  \
 }
#endif
/*************************************************************************
* G__OP2_addassign()
*************************************************************************/
void G__OP2_addassign(G__value *bufm1,G__value *bufm2)
{
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
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
      bufm2->obj.d += bufm1->obj.d;
    }
    else {
      bufm2->obj.d += G__convertT<double>(bufm1);
    }
    G__realassign(bufm2->ref,bufm2->obj.d,bufm2->type);
  }
  else {
    if(G__isdoubleM(bufm1)) {
      G__AddAssign(bufm2, G__convertT<long>(bufm1));
    }
    else if(isupper(bufm2->type)) {
      bufm2->obj.i += (bufm1->obj.i*G__sizeof(bufm2));
    }
    else if(isupper(bufm1->type)) {
      /* Illegal statement */
      bufm2->obj.i =  G__convertT<long>(bufm2)*G__sizeof(bufm1) +  G__convertT<long>(bufm1);
    }
    else {
      G__AddAssign(bufm2, G__convertT<long>(bufm1));
    }
    G__intassign(bufm2->ref,bufm2->obj.i,bufm2->type);
  }
}

/*************************************************************************
* G__OP2_subassign()
*************************************************************************/
void G__OP2_subassign(G__value *bufm1,G__value *bufm2)
{
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
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
      bufm2->obj.d -= bufm1->obj.d;
    }
    else {
      bufm2->obj.d -= G__convertT<double>(bufm1);
    }
    G__realassign(bufm2->ref,bufm2->obj.d,bufm2->type);
  }
  else {
    if(G__isdoubleM(bufm1)) {
      G__SubAssign(bufm2 , G__convertT<long>(bufm1));
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
      G__SubAssign(bufm2 , G__convertT<long>(bufm1));
    }
    G__intassign(bufm2->ref,bufm2->obj.i,bufm2->type);
  }
}

/*************************************************************************
* G__OP2_mulassign()
*************************************************************************/
void G__OP2_mulassign(G__value *bufm1,G__value *bufm2)
{
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
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
      bufm2->obj.d *= bufm1->obj.d;
    }
    else {
      bufm2->obj.d *= G__convertT<double>(bufm1);
    }
    G__realassign(bufm2->ref,bufm2->obj.d,bufm2->type);
  }
  else {
    G__MulAssign(bufm2 , G__convertT<long>(bufm1));
    G__intassign(bufm2->ref,bufm2->obj.i,bufm2->type);
  }
}

/*************************************************************************
* G__OP2_modassign()
*************************************************************************/
void G__OP2_modassign(G__value *bufm1,G__value *bufm2)
{
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
#ifdef G__TUNEUP_W_SECURITY
  if(0==bufm1->obj.i) {
    G__genericerror("Error: operator '%' divided by zero");
    return;
  }
#endif
    if(G__isunsignedM(bufm1)) {
    G__ModAssign(bufm2, G__convertT<unsigned long>(bufm1));
    }
    else {
    G__ModAssign(bufm2, G__convertT<long>(bufm1));
  }
  G__intassign(bufm2->ref,bufm2->obj.i,bufm2->type);
}

/*************************************************************************
* G__OP2_divassign()
*************************************************************************/
void G__OP2_divassign(G__value *bufm1,G__value *bufm2)
{
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
  if(G__isdoubleM(bufm2)) {
    if(G__isdoubleM(bufm1)) {
#ifdef G__TUNEUP_W_SECURITY
//  IEEE 754 defines that NaN inf has to be the output in cases where division by 0 occurs.
//      if(0==bufm1->obj.d) {
//        G__genericerror("Error: operator '/' divided by zero");
//        return;
//      }
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
      bufm2->obj.d /= G__convertT<double>(bufm1);
    }
    G__realassign(bufm2->ref,bufm2->obj.d,bufm2->type);
  }
  else {
    if(G__isdoubleM(bufm1)) {
#ifdef G__TUNEUP_W_SECURITY
//  IEEE 754 defines that NaN inf has to be the output in cases where division by 0 occurs.
//      if(0==bufm1->obj.d) {
//        G__genericerror("Error: operator '/' divided by zero");
//        return;
//      }
#endif
      G__DivAssign(bufm2, G__convertT<long>(bufm1));
    }
    else {
#ifdef G__TUNEUP_W_SECURITY
      if(0==bufm1->obj.i) {
        G__genericerror("Error: operator '/' divided by zero");
        return;
      }
#endif
        if(G__isunsignedM(bufm1)) {
	G__DivAssign(bufm2, G__convertT<unsigned long>(bufm1));
      }
      else {
	G__DivAssign(bufm2, G__convertT<long>(bufm1));
      }
    }
    G__intassign(bufm2->ref,bufm2->obj.i,bufm2->type);
  }
}


/****************************************************************
* G__OP1_OPTIMIZED
****************************************************************/

/****************************************************************
* G__OP1_postfixinc_i()
****************************************************************/
void G__OP1_postfixinc_i(G__value* pbuf)
{
  *(int*)pbuf->ref = (int) pbuf->obj.i + 1;
  pbuf->ref= (long) &(pbuf->obj.i);
}

/****************************************************************
* G__OP1_postfixdec_i()
****************************************************************/
void G__OP1_postfixdec_i(G__value* pbuf)
{
  *(int*)pbuf->ref = (int) pbuf->obj.i - 1;
  pbuf->ref= (long) &(pbuf->obj.i);
}

/****************************************************************
* G__OP1_prefixinc_i()
****************************************************************/
void G__OP1_prefixinc_i(G__value* pbuf)
{
  *(int*)pbuf->ref = (int)(++pbuf->obj.i);
}

/****************************************************************
* G__OP1_prefixdec_i()
****************************************************************/
void G__OP1_prefixdec_i(G__value *pbuf)
{
  *(int*)pbuf->ref = (int)(--pbuf->obj.i);
}

/****************************************************************
* G__OP1_postfixinc_d()
****************************************************************/
void G__OP1_postfixinc_d(G__value *pbuf)
{
  *(double*)pbuf->ref = (double)pbuf->obj.d + 1.0;
  pbuf->ref= (long) &(pbuf->obj.d);
}

/****************************************************************
* G__OP1_postfixdec_d()
****************************************************************/
void G__OP1_postfixdec_d(G__value *pbuf)
{
  *(double*)pbuf->ref = (double)pbuf->obj.d-1.0;
  pbuf->ref= (long) &(pbuf->obj.d);
}
/****************************************************************
* G__OP1_prefixinc_d()
****************************************************************/
void G__OP1_prefixinc_d(G__value *pbuf)
{
  *(double*)pbuf->ref = (double)(++pbuf->obj.d);
}
/****************************************************************
* G__OP1_prefixdec_d()
****************************************************************/
void G__OP1_prefixdec_d(G__value *pbuf)
{
  *(double*)pbuf->ref = (double)(--pbuf->obj.d);
}

/****************************************************************
* G__OP1_postfixinc()
****************************************************************/
void G__OP1_postfixinc(G__value *pbuf)
{
  G__int64 iorig;
  double dorig;
  switch(pbuf->type) {
  case 'd':
  case 'f':
    dorig = pbuf->obj.d;
    G__doubleassignbyref(pbuf,dorig+1.0);
    pbuf->obj.d=dorig;
    break;
  case 'm':
  case 'n':
    iorig = G__Longlong(*pbuf);
    if(isupper(pbuf->type)) {
      G__intassignbyref(pbuf,iorig+G__sizeof(pbuf));
      pbuf->obj.ll = iorig;
    }
    else {
      G__intassignbyref(pbuf,iorig+1);
      pbuf->obj.ll = iorig;
    }
    break;
  default:
    iorig = G__Longlong(*pbuf);
    if(isupper(pbuf->type)) {
      G__intassignbyref(pbuf,iorig+G__sizeof(pbuf));
      pbuf->obj.i = (long)iorig;
    }
    else {
      G__intassignbyref(pbuf,iorig+1);
      pbuf->obj.i = (long)iorig;
    }
  }
}
/****************************************************************
* G__OP1_postfixdec()
****************************************************************/
void G__OP1_postfixdec(G__value *pbuf)
{
  G__int64 iorig;
  double dorig;
  switch(pbuf->type) {
  case 'd':
  case 'f':
    dorig = pbuf->obj.d;
    G__doubleassignbyref(pbuf,dorig-1.0);
    pbuf->obj.d=dorig;
    break;
  case 'm':
  case 'n':
    iorig = G__Longlong(*pbuf);
    if(isupper(pbuf->type)) {
      G__intassignbyref(pbuf,iorig-G__sizeof(pbuf));
      pbuf->obj.ll = iorig;
    }
    else {
      G__intassignbyref(pbuf,iorig-1);
      pbuf->obj.ll = iorig;
    }
    break;
  default:
    iorig = G__Longlong(*pbuf);
    if(isupper(pbuf->type)) {
      G__intassignbyref(pbuf,iorig-G__sizeof(pbuf));
      pbuf->obj.i = (long)iorig;
    }
    else {
      G__intassignbyref(pbuf,iorig-1);
      pbuf->obj.i = (long)iorig;
    }
  }
}
/****************************************************************
* G__OP1_prefixinc()
****************************************************************/
void G__OP1_prefixinc(G__value *pbuf)
{
  switch(pbuf->type) {
  case 'd':
  case 'f':
    G__doubleassignbyref(pbuf,pbuf->obj.d+1.0);
    break;
  default:
    if(isupper(pbuf->type)) {
      G__intassignbyref(pbuf,G__Longlong(*pbuf)+G__sizeof(pbuf));
    }
    else {
      G__intassignbyref(pbuf,G__Longlong(*pbuf)+1);
    }
  }
}
/****************************************************************
* G__OP1_prefixdec()
****************************************************************/
void G__OP1_prefixdec(G__value *pbuf)
{
  switch(pbuf->type) {
  case 'd':
  case 'f':
    G__doubleassignbyref(pbuf,pbuf->obj.d-1.0);
    break;
  default:
    if(isupper(pbuf->type)) {
      G__intassignbyref(pbuf,G__Longlong(*pbuf)-G__sizeof(pbuf));
    }
    else {
      G__intassignbyref(pbuf,G__Longlong(*pbuf)-1);
    }
  }
}
/****************************************************************
* G__OP1_minus()
****************************************************************/
void G__OP1_minus(G__value *pbuf)
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
       switch(pbuf->type) {
         case 'm':
         case 'n':
           pbuf->obj.ll *= -1;
           break;
         default:
           pbuf->obj.i *= -1;
       }
    }
  }
}




/*************************************************************************
**************************************************************************
* Optimization level 1 function
**************************************************************************
*************************************************************************/

/******************************************************************
* G__suspendbytecode()
******************************************************************/
void G__suspendbytecode()
{
   if (G__asm_dbg && G__asm_noverflow) {
      if (G__dispmsg >= G__DISPNOTE) {
         G__fprinterr(G__serr, "Note: Bytecode compiler suspended.");
         G__printlinenum();
      }
   }
   G__asm_noverflow = 0;
}

/******************************************************************
* G__resetbytecode()
******************************************************************/
void G__resetbytecode()
{
   if (G__asm_dbg && G__asm_noverflow) {
      if (G__dispmsg >= G__DISPNOTE) {
         G__fprinterr(G__serr, "Note: Bytecode compiler reset.");
         G__printlinenum();
      }
   }
   G__asm_noverflow = 0;
}

/******************************************************************
* G__abortbytecode()
******************************************************************/
void G__abortbytecode()
{
   if (G__asm_dbg && G__asm_noverflow) {
      if (G__dispmsg >= G__DISPNOTE) {
         if (!G__xrefflag) {
            G__fprinterr(G__serr, "Note: Bytecode compiler stops at this line.  Enclosing loop or function may be slow. %d", G__asm_noverflow);
         }
         else {
            G__fprinterr(G__serr, "Note: Bytecode limitation encountered but compiler continues for local variable cross-referencing.");
         }
         G__printlinenum();
      }
   }
   if (!G__xrefflag) {
      G__asm_noverflow = 0;
   }
}

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
int G__inc_cp_asm(int cp_inc, int dt_dec)
{
   if (!G__xrefflag) {
      G__asm_cp += cp_inc;
      G__asm_dt -= dt_dec;
   }
   if (G__asm_instsize && (G__asm_cp > (G__asm_instsize - 8))) {
      G__asm_instsize += 0x100;
      void* p = realloc((void*) G__asm_stack, sizeof(long) * G__asm_instsize);
      if (!p) {
         G__genericerror("Error: memory exhausted for bytecode instruction buffer\n");
      }
      G__asm_inst = (long*) p;
   }
   else if (!G__asm_instsize && (G__asm_cp > (G__MAXINST - 8))) {
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Warning: loop compile instruction overflow");
         G__printlinenum();
      }
      G__abortbytecode();
   }
   if (G__asm_dt < 30) {
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Warning: loop compile data overflow");
         G__printlinenum();
      }
      G__abortbytecode();
   }
   return 0;
}

/****************************************************************
* G__clear_asm()
****************************************************************/
int G__clear_asm()
{
   // -- Reset instruction and data buffer.
   G__asm_cp = 0;
   G__asm_dt = G__MAXSTACK - 1;
   G__asm_name_p = 0;
   G__asm_cond_cp = -1;
   return 0;
}

/******************************************************************
* G__asm_clear()
******************************************************************/
int G__asm_clear()
{
   // -- FIXME: Describe this function!
   if (G__asm_clear_mask) {
      return 0;
   }
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "%3x,%3x: CL %s:%d  %s:%d\n", G__asm_cp, G__asm_dt, G__ifile.name, G__ifile.line_number, __FILE__, __LINE__);
   }
#endif // G__ASM_DBG
   if (
      (G__asm_cp >= 2) &&
      (G__asm_inst[G__asm_cp-2] == G__CL) &&
      ((G__asm_inst[G__asm_cp-1] & 0xffff0000) == 0x7fff0000)
   ) {
      G__inc_cp_asm(-2, 0);
   }
   G__asm_inst[G__asm_cp] = G__CL;
   G__asm_inst[G__asm_cp+1] = (G__ifile.line_number & G__CL_LINEMASK) + ((G__ifile.filenum & G__CL_FILEMASK) * G__CL_FILESHIFT);
   G__inc_cp_asm(2, 0);
   return 0;
}

#endif // G__ASM

#ifdef G__ASM
/**************************************************************************
* G__asm_putint()
**************************************************************************/
int G__asm_putint(int i)
{
#ifdef G__ASM_DBG
   if (G__asm_dbg) {
      G__fprinterr(G__serr, "%3x,%3x: LD %d  %s:%d\n", G__asm_cp, G__asm_dt, i, __FILE__, __LINE__);
   }
#endif
   G__asm_inst[G__asm_cp] = G__LD;
   G__asm_inst[G__asm_cp+1] = G__asm_dt;
   G__letint(&G__asm_stack[G__asm_dt], 'i', (long) i);
   G__inc_cp_asm(2, 1);
   return 0;
}
#endif

/**************************************************************************
* G__value G__getreserved()
**************************************************************************/
G__value G__getreserved(const char *item ,void ** /* ptr */,void ** /* ppdict */)
{
  G__value buf = G__null;
  int i;

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
      i = 0;
      buf = G__null;
  }

  if(i) {
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

/**************************************************************************
* G__get__tm__()
*
*  returns 'Sun Nov 28 21:40:32 1999\n' in buf
**************************************************************************/
void G__get__tm__(G__FastAllocString &buf)
{
  time_t t = time(0);
  buf.Format("%s",ctime(&t));
}
/**************************************************************************
* G__get__date__()
**************************************************************************/
char* G__get__date__()
{
  int i=0,j=0;
  G__FastAllocString buf(80);
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
  G__FastAllocString buf(80);
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

/**************************************************************************
* G__value G__getrsvd()
**************************************************************************/
G__value G__getrsvd(int i)
{
  G__value buf;

  buf.tagnum = -1;
  buf.typenum = -1;
  buf.ref = 0;

  switch(i) {
  case G__RSVD_LINE:
    G__letint(&buf,'i',(long)G__ifile.line_number);
    break;
  case G__RSVD_FILE:
    if(0<=G__ifile.filenum && G__ifile.filenum<G__MAXFILE &&
       G__srcfile[G__ifile.filenum].filename) {
      G__letint(&buf,'C',(long)G__srcfile[G__ifile.filenum].filename);
    }
    else {
      G__letint(&buf,'C',(long)0);
    }
    break;
  case G__RSVD_ARG:
    G__letint(&buf,'i',(long)G__argn);
    break;
  case G__RSVD_DATE:
    G__letint(&buf,'C',(long)G__get__date__());
    break;
  case G__RSVD_TIME:
    G__letint(&buf,'C',(long)G__get__time__());
    break;
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
long G__asm_gettest(int op,long *inst)
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

/****************************************************************
 * G__isInt
 ****************************************************************/
int G__isInt(int type)
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

/****************************************************************
* G__asm_optimize(start)
*
* Called by
*
*  Quasi-Assembly-Code optimizer
*
****************************************************************/
int G__asm_optimize(int *start)
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
          && !G__asm_wholefunction
      ))
     &&  /* 1 */
     G__asm_inst[*start+5]==G__LD      &&
     G__asm_inst[*start+7]==G__CMP2    &&
     G__asm_inst[*start+9]==G__CNDJMP  &&
     G__isInt(((struct G__var_array*)G__asm_inst[*start+4])->type[G__asm_inst[*start+1]]) && /* 2 */
     G__isInt(G__asm_stack[G__asm_inst[*start+6]].type) &&
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
    pb = (int*)(&G__asm_inst[*start+5]);
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
            && !G__asm_wholefunction
            )) &&  /* 1 */
          (G__asm_inst[*start+5]==G__LD_VAR||
           (G__asm_inst[*start+5]==G__LD_MSTR
            && !G__asm_wholefunction
           )) &&  /* 1 */
          G__asm_inst[*start+10]==G__CMP2    &&
          G__asm_inst[*start+12]==G__CNDJMP  &&
          G__isInt(((struct G__var_array*)G__asm_inst[*start+4])->type[G__asm_inst[*start+1]]) && /* 2 */
          (G__isInt(((struct G__var_array*)G__asm_inst[*start+9])->type[G__asm_inst[*start+6]]) || /* 2 */
           ((struct G__var_array*)G__asm_inst[*start+9])->type[G__asm_inst[*start+6]] =='p' ) && /* 2 */
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
     G__asm_cond_cp != G__asm_cp-2 &&
     (G__LD_VAR==G__asm_inst[G__asm_cp-9]||
      (G__LD_MSTR==G__asm_inst[G__asm_cp-9]
       && !G__asm_wholefunction
      )) &&
     G__isInt(((struct G__var_array*)G__asm_inst[G__asm_cp-5])->type[G__asm_inst[G__asm_cp-8]])) {

#ifdef G__ASM_DBG
    if (G__asm_dbg) {
       G__fprinterr(G__serr, "   %3x: INCJMP i++ optimized  %s:%d\n", G__asm_cp - 9, __FILE__, __LINE__);
    }
#endif // G__ASM_DBG

    G__asm_inst[G__asm_cp-8]=
      ((struct G__var_array*)G__asm_inst[G__asm_cp-5])->p[G__asm_inst[G__asm_cp-8]];
    if(G__asm_inst[G__asm_cp-9]==G__LD_MSTR
       && G__LOCALSTATIC!=((struct G__var_array*)G__asm_inst[G__asm_cp-5])->statictype[G__asm_inst[G__asm_cp-8]]
       ) {
      G__asm_inst[G__asm_cp-8] += G__store_struct_offset;
    }

    G__asm_inst[G__asm_cp-9] = G__INCJMP;

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
          G__asm_cond_cp != G__asm_cp-2 &&
          (G__LD_VAR==G__asm_inst[G__asm_cp-11]||
           (G__LD_MSTR==G__asm_inst[G__asm_cp-11]
            && !G__asm_wholefunction
            )) &&
          G__isInt(((struct G__var_array*)G__asm_inst[G__asm_cp-7])->type[G__asm_inst[G__asm_cp-10]]))  {

#ifdef G__ASM_DBG
    if (G__asm_dbg) {
       G__fprinterr(G__serr, "   %3x: INCJMP i += 1 optimized  %s:%d\n", G__asm_cp - 11, __FILE__, __LINE__);
    }
#endif // G__ASM_DBG

    G__asm_inst[G__asm_cp-10]=
      ((struct G__var_array*)G__asm_inst[G__asm_cp-7])->p[G__asm_inst[G__asm_cp-10]];
    if(G__asm_inst[G__asm_cp-11]==G__LD_MSTR
       && G__LOCALSTATIC!=((struct G__var_array*)G__asm_inst[G__asm_cp-7])->statictype[G__asm_inst[G__asm_cp-10]]
       ) {
      G__asm_inst[G__asm_cp-10] += G__store_struct_offset;
    }

    G__asm_inst[G__asm_cp-11] = G__INCJMP;

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
            && !G__asm_wholefunction
           )) &&
          G__asm_inst[G__asm_cp-9]==G__OP2     &&
          G__asm_inst[G__asm_cp-11]==G__LD     &&
          G__asm_inst[G__asm_cp-15]==G__asm_inst[G__asm_cp-6] && /* 2 */
          G__asm_inst[G__asm_cp-12]==G__asm_inst[G__asm_cp-3] &&
          (G__asm_inst[G__asm_cp-8]=='+'||G__asm_inst[G__asm_cp-8]=='-') &&
          G__isInt(((struct G__var_array*)G__asm_inst[G__asm_cp-3])->type[G__asm_inst[G__asm_cp-6]])       &&
          G__asm_inst[G__asm_cp-14]==0 &&
          G__asm_inst[G__asm_cp-13]=='p' &&   /* 3 */
          G__asm_inst[G__asm_cp-4]=='p') {

#ifdef G__ASM_DBG
    if (G__asm_dbg) {
       G__fprinterr(G__serr, "   %3x: INCJMP i = i + 1 optimized  %s:%d\n", G__asm_cp - 16, __FILE__, __LINE__);
    }
#endif // G__ASM_DBG
    G__asm_inst[G__asm_cp-16] = G__INCJMP;

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


/*************************************************************************
**************************************************************************
* Optimization level 3 function
**************************************************************************
*************************************************************************/

/*************************************************************************
* G__get_LD_p0_p2f()
*************************************************************************/
int G__get_LD_p0_p2f(int type,long *pinst)
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
    case 'g': *pinst = (long)G__LD_p0_bool; break;
    case 'n': *pinst = (long)G__LD_p0_longlong; break;
    case 'm': *pinst = (long)G__LD_p0_ulonglong; break;
    case 'q': *pinst = (long)G__LD_p0_longdouble; break;
    default: done=0; break;
    }
  }
  return(done);
}

/*************************************************************************
* G__get_LD_p1_p2f()
*************************************************************************/
int G__get_LD_p1_p2f(int type,long *pinst)
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
    case 'g': *pinst = (long)G__LD_p1_bool; break;
    case 'n': *pinst = (long)G__LD_p1_longlong; break;
    case 'm': *pinst = (long)G__LD_p1_ulonglong; break;
    case 'q': *pinst = (long)G__LD_p1_longdouble; break;
    default: done=0; break;
    }
  }
  return(done);
}

/*************************************************************************
* G__get_LD_pn_p2f()
*************************************************************************/
int G__get_LD_pn_p2f(int type,long *pinst)
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
    case 'g': *pinst = (long)G__LD_pn_bool; break;
    case 'n': *pinst = (long)G__LD_pn_longlong; break;
    case 'm': *pinst = (long)G__LD_pn_ulonglong; break;
    case 'q': *pinst = (long)G__LD_pn_longdouble; break;
    default: done=0; break;
    }
  }
  return(done);
}

/*************************************************************************
* G__get_LD_P10_p2f()
*************************************************************************/
int G__get_LD_P10_p2f(int type,long *pinst,int reftype)
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
    case 'G': *pinst = (long)G__LD_P10_bool; break;
    case 'N': *pinst = (long)G__LD_P10_longlong; break;
    case 'M': *pinst = (long)G__LD_P10_ulonglong; break;
    case 'Q': *pinst = (long)G__LD_P10_longdouble; break;
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
int G__get_ST_p0_p2f(int type,long *pinst)
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
    case 'g': *pinst = (long)G__ST_p0_bool; break;
    case 'n': *pinst = (long)G__ST_p0_longlong; break;
    case 'm': *pinst = (long)G__ST_p0_ulonglong; break;
    case 'q': *pinst = (long)G__ST_p0_longdouble; break;
    default: done=0; break;
    }
  }
  return(done);
}

/*************************************************************************
* G__get_ST_p1_p2f()
*************************************************************************/
int G__get_ST_p1_p2f(int type,long *pinst)
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
    case 'g': *pinst = (long)G__ST_p1_bool; break; /* to be fixed */
    case 'n': *pinst = (long)G__ST_p1_longlong; break;
    case 'm': *pinst = (long)G__ST_p1_ulonglong; break;
    case 'q': *pinst = (long)G__ST_p1_longdouble; break;
    default: done=0; break;
    }
  }
  return(done);
}

/*************************************************************************
* G__get_ST_pn_p2f()
*************************************************************************/
int G__get_ST_pn_p2f(int type,long *pinst)
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
    case 'g': *pinst = (long)G__ST_pn_bool; break; /* to be fixed */
    case 'n': *pinst = (long)G__ST_pn_longlong; break;
    case 'm': *pinst = (long)G__ST_pn_ulonglong; break;
    case 'q': *pinst = (long)G__ST_pn_longdouble; break;
    default: done=0; break;
    }
  }
  return(done);
}

/*************************************************************************
* G__get_ST_P10_p2f()
*************************************************************************/
int G__get_ST_P10_p2f(int type,long *pinst,int reftype)
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
    case 'G': *pinst = (long)G__ST_P10_bool; break;
    case 'N': *pinst = (long)G__ST_P10_longlong; break;
    case 'M': *pinst = (long)G__ST_P10_ulonglong; break;
    case 'Q': *pinst = (long)G__ST_P10_longdouble; break;
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
int G__get_LD_Rp0_p2f(int type,long *pinst)
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
    case 'g': *pinst = (long)G__LD_Rp0_bool; break; /* to be fixed */
    case 'n': *pinst = (long)G__LD_Rp0_longlong; break;
    case 'm': *pinst = (long)G__LD_Rp0_ulonglong; break;
    case 'q': *pinst = (long)G__LD_Rp0_longdouble; break;
    default: done=0; break;
    }
  }
  return(done);
}
/*************************************************************************
* G__get_ST_Rp0_p2f()
*************************************************************************/
int G__get_ST_Rp0_p2f(int type,long *pinst)
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
    case 'g': *pinst = (long)G__ST_Rp0_bool; break; /* to be fixed */
    case 'n': *pinst = (long)G__ST_Rp0_longlong; break;
    case 'm': *pinst = (long)G__ST_Rp0_ulonglong; break;
    case 'q': *pinst = (long)G__ST_Rp0_longdouble; break;
    default: done=0; break;
    }
  }
  return(done);
}
/*************************************************************************
* G__get_LD_RP0_p2f()
*************************************************************************/
int G__get_LD_RP0_p2f(int type,long *pinst)
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
    case 'g': *pinst = (long)G__LD_RP0_bool; break; /* to be fixed */
    case 'n': *pinst = (long)G__LD_RP0_longlong; break;
    case 'm': *pinst = (long)G__LD_RP0_ulonglong; break;
    case 'q': *pinst = (long)G__LD_RP0_longdouble; break;
    default: done=0; break;
    }
  }
  return(done);
}

/*************************************************************************
* G__LD_Rp0_optimize()
*************************************************************************/
void G__LD_Rp0_optimize(struct G__var_array *var,int ig15,int pc,long inst)
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
void G__ST_Rp0_optimize(struct G__var_array *var,int ig15,int pc,long inst)
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
void G__LD_RP0_optimize(struct G__var_array *var,int ig15,int pc,long inst)
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
void G__LD_p0_optimize(struct G__var_array* var, int ig15, int pc, long inst)
{
   long originst = G__asm_inst[pc];
   int pointlevel = G__asm_inst[pc+3];
   if (var->bitfield[ig15]) {
      return;
   }
#ifdef G__ASM_DBG
   if (G__asm_dbg)
   {
      switch (inst) {
         case G__LDST_VAR_P:
            G__fprinterr(G__serr, "        G__LD_VAR optimized 6 to G__LDST_VAR_P\n");
            break;
         case G__LDST_MSTR_P:
            G__fprinterr(G__serr, "  G__LD_MSTR optimized 6 to G__LDST_MSTR_P\n");
            break;
         case G__LDST_LVAR_P:
            G__fprinterr(G__serr, "  G__LD_LVAR optimized 6 to G__LDST_LVAR_P\n");
            break;
      }
   }
#endif
   G__asm_inst[pc] = inst;
   G__asm_inst[pc+3] = 0;
   if (!G__get_LD_p0_p2f(var->type[ig15], &G__asm_inst[pc+2]))
   {
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "Error: LD_VAR,LD_MSTR optimize (6) error %s\n", var->varnamebuf[ig15]);
      }
#endif
      G__asm_inst[pc] = originst;
      G__asm_inst[pc+3] = pointlevel;
   }
}

/*************************************************************************
* G__LD_p1_optimize()
*************************************************************************/
void G__LD_p1_optimize(struct G__var_array *var,int ig15,int pc,long inst)
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

/*************************************************************************
* G__LD_pn_optimize()
*************************************************************************/
void G__LD_pn_optimize(struct G__var_array *var,int ig15,int pc,long inst)
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

/*************************************************************************
* G__LD_P10_optimize()
*************************************************************************/
void G__LD_P10_optimize(struct G__var_array *var,int ig15,int pc,long inst)
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
void G__ST_p0_optimize(struct G__var_array *var,int ig15,int pc,long inst)
{
  long originst=G__asm_inst[pc];
  int pointlevel=G__asm_inst[pc+3];
  if(var->bitfield[ig15]) return;
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
void G__ST_p1_optimize(struct G__var_array *var,int ig15,int pc,long inst)
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

/*************************************************************************
* G__ST_pn_optimize()
*************************************************************************/
void G__ST_pn_optimize(struct G__var_array *var,int ig15,int pc,long inst)
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

/*************************************************************************
* G__ST_P10_optimize()
*************************************************************************/
void G__ST_P10_optimize(struct G__var_array *var,int ig15,int pc,long inst)
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
#define G__MAXINDEXCONST 11
static long G__indexconst[G__MAXINDEXCONST] = {0,1,2,3,4,5,6,7,8,9,10};

/*************************************************************************
* G__LD_VAR_int_optimize()
*************************************************************************/
int G__LD_VAR_int_optimize(int *ppc,long *pi)
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
      long *pi2 = &(G__asm_stack[G__asm_inst[pc+6]].obj.i);
      long *pix;
      if(G__ASM_FUNC_COMPILE==G__asm_wholefunction) {
        if(*pi2>=G__MAXINDEXCONST||*pi2<0) return(done);
        else pix = &G__indexconst[*pi2];
      }
      else {
        pix = pi2;
        if(sizeof(long)>sizeof(int)) *pix = (int)(*pi2);
      }
      if(G__LD_LVAR==G__asm_inst[pc]) flag=1;
      else                            flag=0;
      if(G__LD_LVAR==G__asm_inst[pc+9]) flag |= 4;
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
        G__asm_inst[pc+2] = (long)pix;
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
      long *pi2 = &(G__asm_stack[G__asm_inst[pc+6]].obj.i);
      long *pix;
      if(G__ASM_FUNC_COMPILE==G__asm_wholefunction) {
        if(*pi2>=G__MAXINDEXCONST||*pi2<0) return(done);
        else pix = &G__indexconst[*pi2];
      }
      else {
        pix = pi2;
        if(sizeof(long)>sizeof(int)) *pix = (int)(*pi2);
      }
      if(G__LD_LVAR==G__asm_inst[pc]) flag=1;
      else                            flag=0;
      if(G__ST_LVAR==G__asm_inst[pc+9]) flag |= 4;
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
        G__asm_inst[pc+2] = (long)pix;
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
int G__LD_int_optimize(int *ppc,long *pi)
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
     && (pc<4 || G__JMP!=G__asm_inst[pc-2] || G__asm_inst[pc-1]!=pc+2)
     ) {
    int flag;
    if(G__ASM_FUNC_COMPILE==G__asm_wholefunction) {
      if(*pi>=G__MAXINDEXCONST||*pi<0) return(done);
      else pi = &G__indexconst[*pi];
    }
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
        *(int*)G__asm_inst[pc+1]= (int)(*pi);
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
          && (pc<4 || G__JMP!=G__asm_inst[pc-2] || G__asm_inst[pc-1]!=pc+2)
          ) {
    int flag;
    if(G__ASM_FUNC_COMPILE==G__asm_wholefunction) {
      if(*pi>=G__MAXINDEXCONST||*pi<0) return(done);
      else pi = &G__indexconst[*pi];
    }
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
        *(int*)G__asm_inst[pc+1]= (int)(*pi);
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
int G__CMP2_optimize(int pc)
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
int G__OP2_optimize(int pc)
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
  case G__OPR_ADDVOIDPTR:
    G__asm_inst[pc+1] = (long)G__OP2_addvoidptr;
    break;
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
int G__asm_optimize3(int *start)
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
             if(0==G__LD_VAR_int_optimize(&pc,(long*)var->p[ig15]))
                G__LD_p0_optimize(var,ig15,pc,G__LDST_VAR_P);
          }
          else {
            G__LD_p0_optimize(var,ig15,pc,G__LDST_VAR_P);
          }
        }
        else if(1==paran && 1==var->paran[ig15]) {
          G__LD_p1_optimize(var,ig15,pc,G__LDST_VAR_P);
        }
        else if(paran==var->paran[ig15]) {
          G__LD_pn_optimize(var,ig15,pc,G__LDST_VAR_P);
        }
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
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x: LD %d from %x  %s:%d\n", pc, G__int(G__asm_stack[G__asm_inst[pc+1]]), G__asm_inst[pc+1], __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
      // no optimize
      if('i'==G__asm_stack[G__asm_inst[pc+1]].type) {
         G__LD_int_optimize(&pc,&(G__asm_stack[G__asm_inst[pc+1]].obj.i));
      }
      pc+=2;
      break;

    case G__CL:
      /***************************************
      * 0 CL
      *  clear stack pointer
      ***************************************/
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3lx: CL %s:%d  %s:%d\n", pc, G__srcfile[G__asm_inst[pc+1] / G__CL_FILESHIFT].filename, G__asm_inst[pc+1]&G__CL_LINEMASK, __FILE__, __LINE__);
      }
#endif // G__ASM_DBG
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
      if(G__asm_inst[pc+1]<256 && isprint(G__asm_inst[pc+1])) {
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
        else if(paran==var->paran[ig15]) {
          G__ST_pn_optimize(var,ig15,pc,G__LDST_VAR_P);
        }
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
        else if(paran==var->paran[ig15]) {
          G__LD_pn_optimize(var,ig15,pc,inst);
        }
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
        else if(paran==var->paran[ig15]) {
          G__ST_pn_optimize(var,ig15,pc,inst);
        }
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
      if (G__asm_dbg) {
         G__fprinterr(G__serr, "%3x: INCJMP *(int*)0x%lx+%d to %x  %s:%d\n", pc, G__asm_inst[pc+1], G__asm_inst[pc+2], G__asm_inst[pc+3], __FILE__, __LINE__);
      }
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
      * 5 this ptr offset for multiple inheritance
      * 6 ifunc
      * 7 ifn
      * stack
      * sp-paran+1      <- sp-paran+1
      * sp-2
      * sp-1
      * sp
      ***************************************/
#ifdef G__ASM_DBG
      if (G__asm_dbg) {
        if (G__asm_inst[pc+1] < G__MAXSTRUCT) {
          G__fprinterr(G__serr, "%3lx: LD_FUNC '%s' paran: %d  %s:%d\n", pc, "compiled", G__asm_inst[pc+3], __FILE__, __LINE__);
        }
        else {
          G__fprinterr(G__serr, "%3lx: LD_FUNC '%s' paran: %d  %s:%d\n", pc, (char*) G__asm_inst[pc+1], G__asm_inst[pc+3], __FILE__, __LINE__);
        }
      }
      else {
        if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: LD_FUNC %s paran=%d\n" ,pc
                ,(char *)G__asm_inst[pc+1],G__asm_inst[pc+3]);
      }
#endif
      /* no optimization */
      pc += 8;
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
      if(G__asm_inst[pc+1]<256 && isprint(G__asm_inst[pc+1])){
        if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: OP1 '%c'%d\n",pc
                ,G__asm_inst[pc+1],G__asm_inst[pc+1] );
      }
      else {
        if(G__asm_dbg) G__fprinterr(G__serr,"%3lx: OP1 %d\n",pc,G__asm_inst[pc+1]);
      }
#endif
      /* need optimization */
      switch(G__asm_inst[pc+1]) {

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
             if(0==G__LD_VAR_int_optimize(&pc,(long*)var->p[ig15]))
              G__LD_p0_optimize(var,ig15,pc,inst);
          }
          else {
            G__LD_p0_optimize(var,ig15,pc,inst);
          }
        }
        else if(1==paran && 1==var->paran[ig15]) {
          G__LD_p1_optimize(var,ig15,pc,inst);
        }
        else if(paran==var->paran[ig15]) {
          G__LD_pn_optimize(var,ig15,pc,inst);
        }
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
        else if(paran==var->paran[ig15]) {
          G__ST_pn_optimize(var,ig15,pc,inst);
        }
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
      * 7 ifn
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
      pc += 8;
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
      pc+=2;
      break;

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

    case G__ROOTOBJALLOCBEGIN:
      /***************************************
      * 0 ROOTOBJALLOCBEGIN
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: ROOTOBJALLOCBEGIN\n",pc);
#endif
      /* no optimization */
      ++pc;
      break;

    case G__ROOTOBJALLOCEND:
      /***************************************
      * 0 ROOTOBJALLOCEND
      ***************************************/
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: ROOTOBJALLOCEND\n",pc);
#endif
      /* no optimization */
      ++pc;
      break;

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

/****************************************************************
* G__dasm()
*
*  Disassembler
*
****************************************************************/
int G__dasm(FILE *fout,int isthrow)
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
      if (!isthrow) {
         fprintf(fout, "%3x: CL %s:%ld  %s:%d\n", pc, G__srcfile[G__asm_inst[pc+1] / G__CL_FILESHIFT].filename, G__asm_inst[pc+1]&G__CL_LINEMASK, __FILE__, __LINE__);
      }
      pc += 2;
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
        if(G__asm_inst[pc+1]<256 && isprint(G__asm_inst[pc+1]))
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
      if (!isthrow) {
        fprintf(fout, "%3x: INCJMP *(int*)0x%lx+%ld to %lx  %s:%d\n", pc, G__asm_inst[pc+1], G__asm_inst[pc+2], G__asm_inst[pc+3], __FILE__, __LINE__);
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
      * 5 this ptr offset for multiple inheritance
      * 6 ifunc
      * 7 ifn
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
      pc += 8;
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
         if (!isthrow) {
            if ((G__asm_inst[pc+1] < 256) && isprint(G__asm_inst[pc+1])) {
               fprintf(fout, "%3x: OP1 '%c' %ld,%ld\n", pc, (char) G__asm_inst[pc+1], G__asm_inst[pc+1], G__asm_inst[pc+1]);
            }
            else {
               fprintf(fout, "%3x: OP1 %ld,%ld\n", pc, G__asm_inst[pc+1], G__asm_inst[pc+1]);
            }
         }
         pc += 2;
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

    case G__BOOL:
      /***************************************
      * 0 BOOL
      ***************************************/
      if(0==isthrow) {
        fprintf(fout,"%3x: BOOL\n" ,pc);
      }
      ++pc;
      break;

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
      * 7 ifn
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
      pc += 8;
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
      pc+=2;
      break;

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
        G__FastAllocString statement(G__LONGLINE);
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
        G__ifile.filenum = (short)G__asm_inst[pc+1];
        G__ifile.line_number = G__asm_inst[pc+2];
        G__strlcpy(G__ifile.name,G__srcfile[G__ifile.filenum].filename,G__MAXFILENAME);
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

    case G__ROOTOBJALLOCBEGIN:
      /***************************************
      * 0 ROOTOBJALLOCBEGIN
      ***************************************/
      if(0==isthrow) {
        fprintf(fout,"%3x: ROOTOBJALLOCBEGIN\n" ,pc);
      }
      ++pc;
      break;

    case G__ROOTOBJALLOCEND:
      /***************************************
      * 0 ROOTOBJALLOCEND
      ***************************************/
      if(0==isthrow) {
        fprintf(fout,"%3x: ROOTOBJALLOCEND\n" ,pc);
      }
      ++pc;
      break;

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
