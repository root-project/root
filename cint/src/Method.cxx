/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file Method.cxx
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Author                  Masaharu Goto 
 * Copyright(c) 1995~2005  Masaharu Goto 
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation. The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#include "Api.h"
#include "common.h"

/*********************************************************************
* class G__MethodInfo
*
* 
*********************************************************************/
///////////////////////////////////////////////////////////////////////////
void G__MethodInfo::Init()
{
  handle = (long)(&G__ifunc);
  index = -1;
#ifndef G__OLDIMPLEMENTATION2194
  usingIndex = -1;
#endif
  belongingclass=(G__ClassInfo*)NULL;
}
///////////////////////////////////////////////////////////////////////////
void G__MethodInfo::Init(G__ClassInfo &a)
{
  if(a.IsValid()) {
    handle=(long)G__struct.memfunc[a.Tagnum()];
    index = -1;
#ifndef G__OLDIMPLEMENTATION2194
    usingIndex = -1;
#endif
    belongingclass = &a;
    G__incsetup_memfunc((int)a.Tagnum());
  }
  else {
    handle=0;
    index = -1;
#ifndef G__OLDIMPLEMENTATION2194
    usingIndex = -1;
#endif
    belongingclass=(G__ClassInfo*)NULL;
  }
}
///////////////////////////////////////////////////////////////////////////
void G__MethodInfo::Init(long handlein,long indexin
	,G__ClassInfo *belongingclassin)
{
#ifndef G__OLDIMPLEMENTATION2194
  usingIndex = -1;
#endif
  if(handlein) {
    handle = handlein;
    index = indexin;
    if(belongingclassin && belongingclassin->IsValid()) 
      belongingclass = belongingclassin;
    else {
      belongingclass=(G__ClassInfo*)NULL;
    }

    /* Set return type */
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    type.type=ifunc->type[index];
    type.tagnum=ifunc->p_tagtable[index];
    type.typenum=ifunc->p_typetable[index];
    type.reftype=ifunc->reftype[index];
    type.isconst=ifunc->isconst[index];
#ifndef G__OLDIMPLEMENTATION1227
    type.class_property=0;
#endif
  }
#ifndef G__FONS72
  else { /* initialize if handlein==0 */
    handle=0;
    index=-1;
    belongingclass=(G__ClassInfo*)NULL;
  }
#endif
}
///////////////////////////////////////////////////////////////////////////
#ifndef G__OLDIMPLEMENTATION644
void G__MethodInfo::Init(G__ClassInfo *belongingclassin
	,long funcpage,long indexin)
{
  struct G__ifunc_table *ifunc;
  int i=0;

  if(belongingclassin->IsValid()) {
    // member function
    belongingclass = belongingclassin;
    ifunc = G__struct.memfunc[belongingclassin->Tagnum()];
  }
  else {
    // global function
    belongingclass=(G__ClassInfo*)NULL;
    ifunc = G__p_ifunc;
  }

  // reach to desired page
  for(i=0;i<funcpage&&ifunc;i++) ifunc=ifunc->next;
  G__ASSERT(ifunc->page == funcpage);

  if(ifunc) {
    handle = (long)ifunc;
    index = indexin;
    // Set return type
    type.type=ifunc->type[index];
    type.tagnum=ifunc->p_tagtable[index];
    type.typenum=ifunc->p_typetable[index];
    type.reftype=ifunc->reftype[index];
    type.isconst=ifunc->isconst[index];
#ifndef G__OLDIMPLEMENTATION1227
    type.class_property=0;
#endif
  }
  else { /* initialize if handlein==0 */
    handle=0;
    index=-1;
    belongingclass=(G__ClassInfo*)NULL;
  }
}
#endif
///////////////////////////////////////////////////////////////////////////
const char* G__MethodInfo::Name()
{
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    return(ifunc->funcname[index]);
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
int G__MethodInfo::Hash()
{
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    return(ifunc->hash[index]);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
struct G__ifunc_table* G__MethodInfo::ifunc()
{
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    return(ifunc);
  }
  else {
    return((struct G__ifunc_table*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
const char* G__MethodInfo::Title() 
{
  static char buf[G__INFO_TITLELEN];
  buf[0]='\0';
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    G__getcomment(buf,&ifunc->comment[index],ifunc->tagnum);
    return(buf);
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
long G__MethodInfo::Property()
{
  if(IsValid()) {
    long property=0;
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
#ifndef G__OLDIMPLEMENTATION2039
    if (ifunc->hash[index]==0) return property;
#endif
    switch(ifunc->access[index]) {
    case G__PUBLIC: property|=G__BIT_ISPUBLIC; break;
    case G__PROTECTED: property|=G__BIT_ISPROTECTED; break;
    case G__PRIVATE: property|=G__BIT_ISPRIVATE; break;
    }
#ifndef G__OLDIMPLEMENTATION1189
    if(ifunc->isconst[index]&G__CONSTFUNC) property|=G__BIT_ISCONSTANT; 
#endif
    if(ifunc->isconst[index]&G__CONSTVAR) property|=G__BIT_ISCONSTANT;
    if(ifunc->isconst[index]&G__PCONSTVAR) property|=G__BIT_ISPCONSTANT;
    if(isupper(ifunc->type[index])) property|=G__BIT_ISPOINTER;
    if(ifunc->staticalloc[index]) property|=G__BIT_ISSTATIC;
    if(ifunc->isvirtual[index]) property|=G__BIT_ISVIRTUAL;
    if(ifunc->ispurevirtual[index]) property|=G__BIT_ISPUREVIRTUAL;
#ifndef G__OLDIMPLEMENTATION2012
    if(ifunc->pentry[index]->size<0) property|=G__BIT_ISCOMPILED;
#else
    if(ifunc->pentry[index]->filenum<0) property|=G__BIT_ISCOMPILED;
#endif
    if(ifunc->pentry[index]->bytecode) property|=G__BIT_ISBYTECODE;
#ifndef G__OLDIMPLEMENTATION1287
    if(ifunc->isexplicit[index]) property|=G__BIT_ISEXPLICIT;
#endif
    return(property);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
int G__MethodInfo::NArg()
{
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    return(ifunc->para_nu[index]);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
int G__MethodInfo::NDefaultArg()
{
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    if(ifunc->para_nu[index]) {
      int i,defaultnu=0;
      for(i=ifunc->para_nu[index]-1;i>=0;i--) {
	if(ifunc->para_default[index][i]) ++defaultnu;
	else return(defaultnu);
      }
      return(defaultnu);
    }
    else {
      return(0);
    }
  }
  else {
    return(-1);
  }
  return(-1); // dummy 
}
///////////////////////////////////////////////////////////////////////////
int G__MethodInfo::HasVarArgs()
{
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    return(2==ifunc->ansi[index]?1:0);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
G__InterfaceMethod G__MethodInfo::InterfaceMethod()
{
#ifndef G__OLDIMPLEMENTATION1035
  G__LockCriticalSection();
#endif
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    if(
#ifndef G__OLDIMPLEMENTATION2012
       -1==ifunc->pentry[index]->size /* this means compiled class */
#else
       -1==ifunc->pentry[index]->filenum /* this meant compiled class */
#endif
       ) {
#ifndef G__OLDIMPLEMENTATION1035
      G__UnlockCriticalSection();
#endif
      return((G__InterfaceMethod)ifunc->pentry[index]->p);
    }
    else {
#ifndef G__OLDIMPLEMENTATION1035
      G__UnlockCriticalSection();
#endif
      return((G__InterfaceMethod)NULL);
    }
  }
  else {
#ifndef G__OLDIMPLEMENTATION1035
    G__UnlockCriticalSection();
#endif
    return((G__InterfaceMethod)NULL);
  }
}
#ifdef G__ASM_WHOLEFUNC
///////////////////////////////////////////////////////////////////////////
struct G__bytecodefunc *G__MethodInfo::GetBytecode()
{
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
#ifndef G__OLDIMPLEMENTATION2082
    int store_asm_loopcompile = G__asm_loopcompile;
    G__asm_loopcompile = 4;
#endif
    if(!ifunc->pentry[index]->bytecode &&
#ifndef G__OLDIMPLEMENTATION2012
       -1!=ifunc->pentry[index]->size && 
#else
       -1!=ifunc->pentry[index]->filenum && 
#endif
       G__BYTECODE_NOTYET==ifunc->pentry[index]->bytecodestatus
#ifndef G__OLDIMPLEMENTATION1842
       && G__asm_loopcompile>=4
#endif
       ) {
      G__compile_bytecode(ifunc,(int)index);
    }
#ifndef G__OLDIMPLEMENTATION2082
    G__asm_loopcompile = store_asm_loopcompile;
#endif
    return(ifunc->pentry[index]->bytecode);
  }
  else {
    return((struct G__bytecodefunc*)NULL);
  }
}
#endif
/* #ifndef G__OLDIMPLEMENTATION1163 */
///////////////////////////////////////////////////////////////////////////
G__DataMemberInfo G__MethodInfo::GetLocalVariable()
{
  G__DataMemberInfo localvar;
  localvar.Init((long)0,(long)(-1),(G__ClassInfo*)NULL);
  if(IsValid()) {
#ifndef G__OLDIMPLEMENTATION1164
    int store_fixedscope=G__fixedscope;
    extern int G__xrefflag;
    G__xrefflag=1;
    G__fixedscope=1;
#endif
    struct G__bytecodefunc* pbc = GetBytecode();
#ifndef G__OLDIMPLEMENTATION1164
    G__xrefflag=0;
    G__fixedscope=store_fixedscope;
#endif
    if(!pbc) {
      if(Property()&G__BIT_ISCOMPILED) {
	G__fprinterr(G__serr,"Limitation: can not get local variable information for compiled function %s\n",Name());
      }
      else {
	G__fprinterr(G__serr,"Limitation: function %s , failed to get local variable information\n",Name());
      }
      return(localvar);
    }
    localvar.Init((long)pbc->var,(long)(-1),(G__ClassInfo*)NULL);
    return(localvar);
  }
  else {
    return(localvar);
  }
}
/* #endif */
///////////////////////////////////////////////////////////////////////////
#ifdef G__TRUEP2F
void* G__MethodInfo::PointerToFunc()
{
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    if(
#ifndef G__OLDIMPLEMENTATION2012
       -1!=ifunc->pentry[index]->size && 
#else
       -1!=ifunc->pentry[index]->filenum && 
#endif
       G__BYTECODE_NOTYET==ifunc->pentry[index]->bytecodestatus
#ifndef G__OLDIMPLEMENTATION1842
       && G__asm_loopcompile>=4
#endif
       ) {
      G__compile_bytecode(ifunc,(int)index);
    }
#ifndef G__OLDIMPLEMENTATION1846
    if(G__BYTECODE_SUCCESS==ifunc->pentry[index]->bytecodestatus) 
      return((void*)ifunc->pentry[index]->bytecode);
      
#endif
    return(ifunc->pentry[index]->tp2f);
  }
  else {
    return((void*)NULL);
  }
}
#endif
///////////////////////////////////////////////////////////////////////////
void G__MethodInfo::SetGlobalcomp(int globalcomp)
{
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    ifunc->globalcomp[index]=globalcomp;
#ifndef G__OLDIMPLEMENTATION912
    if(G__NOLINK==globalcomp) ifunc->access[index]=G__PRIVATE;
    else                      ifunc->access[index]=G__PUBLIC;
#endif
  }
}
///////////////////////////////////////////////////////////////////////////
int G__MethodInfo::IsValid()
{
  if(handle) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    if(0<=index&&index<ifunc->allifunc) {
      return(1);
    }
    else {
      return(0);
    }
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
int G__MethodInfo::SetFilePos(const char* fname)
{
  struct G__dictposition* dict=G__get_dictpos((char*)fname);
  if(!dict) return(0);
  handle = (long)dict->ifunc;
  index = (long)(dict->ifn-1);
  belongingclass=(G__ClassInfo*)NULL;
  return(1);
}
///////////////////////////////////////////////////////////////////////////
int G__MethodInfo::Next()
{
  if(handle) {
    struct G__ifunc_table *ifunc;
#ifndef G__OLDIMPLEMENTATION1706
  nextone:
#endif
    ifunc = (struct G__ifunc_table*)handle;
    ++index;
    if(ifunc->allifunc<=index) {
#ifndef G__FONS75
      int t = ifunc->tagnum;
#endif
      ifunc=ifunc->next;
      if(ifunc) {
#ifndef G__FONS75
	ifunc->tagnum=t;
#endif
	handle=(long)ifunc;
	index = 0;
      }
      else {
	handle=0;
	index = -1;
      }
    } 
#ifndef G__OLDIMPLEMENTATION2194
    if(ifunc==0 && belongingclass==0 && 
       usingIndex<G__globalusingnamespace.basen) {
      ++usingIndex;
      index=0;
      G__incsetup_memfunc(G__globalusingnamespace.basetagnum[usingIndex]);
      ifunc=G__struct.memfunc[G__globalusingnamespace.basetagnum[usingIndex]];
      handle=(long)ifunc;
    }
#endif
    if(IsValid()) {
#ifndef G__OLDIMPLEMENTATION1706
      if(0==ifunc->hash[index]) goto nextone;
#endif
      type.type=ifunc->type[index];
      type.tagnum=ifunc->p_tagtable[index];
      type.typenum=ifunc->p_typetable[index];
      type.reftype=ifunc->reftype[index];
      type.isconst=ifunc->isconst[index];
#ifndef G__OLDIMPLEMENTATION1227
      type.class_property=0;
#endif
      return(1);
    }
    else {
      return(0);
    }
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
const char* G__MethodInfo::FileName()
{
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    if(ifunc->pentry[index]->filenum>=0) { /* 2012, keep this */
      return(G__srcfile[ifunc->pentry[index]->filenum].filename);
    }
    else {
      return("(compiled)");
    }
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
#ifndef G__OLDIMPLEMENTATION644
FILE* G__MethodInfo::FilePointer()
{
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    if(
#ifndef G__OLDIMPLEMENTATION2012
       ifunc->pentry[index]->filenum>=0 && ifunc->pentry[index]->size>=0
#else
       ifunc->pentry[index]->filenum>=0
#endif
       ) {
      return(G__srcfile[ifunc->pentry[index]->filenum].fp);
    }
    else {
      return((FILE*)NULL);
    }
  }
  else {
    return((FILE*)NULL);
  }
}
#endif
///////////////////////////////////////////////////////////////////////////
int G__MethodInfo::LineNumber()
{
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    if(
#ifndef G__OLDIMPLEMENTATION2012
       ifunc->pentry[index]->filenum>=0 && ifunc->pentry[index]->size>=0
#else
       ifunc->pentry[index]->filenum>=0
#endif
       ) {
      return(ifunc->pentry[index]->line_number);
    }
    else {
      return(0);
    }
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
#ifndef G__OLDIMPLEMENTATION644
long G__MethodInfo::FilePosition()
{ 
  // returns  'type fname(type p1,type p2)'
  //                      ^
  long invalid=0L;
  if(IsValid()) {
#ifdef G__VMS
    //Changed so that pos can be a long.
    struct G__ifunc_table_VMS *ifunc;
    ifunc = (struct G__ifunc_table_VMS*)handle;
#else
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
#endif
    if(
#ifndef G__OLDIMPLEMENTATION2012
       ifunc->pentry[index]->filenum>=0 && ifunc->pentry[index]->size>=0
#else
       ifunc->pentry[index]->filenum>=0
#endif
       ) {
#if defined(G__NONSCALARFPOS2)
      return((long)ifunc->pentry[index]->pos.__pos);
#elif defined(G__NONSCALARFPOS_QNX)      
      return((long)ifunc->pentry[index]->pos._Off);
#else
      return((long)ifunc->pentry[index]->pos);
#endif
    }
    else {
      return(invalid);
    }
  }
  else {
    return(invalid);
  }
}
#endif
///////////////////////////////////////////////////////////////////////////
int G__MethodInfo::Size()
{
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    if(
#ifndef G__OLDIMPLEMENTATION2012
       ifunc->pentry[index]->size>=0
#else
       ifunc->pentry[index]->filenum>=0
#endif
       ) {
      return(ifunc->pentry[index]->size);
    }
    else {
      return(0);
    }
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
int G__MethodInfo::IsBusy()
{
  if(IsValid()) {
    struct G__ifunc_table *ifunc;
    ifunc = (struct G__ifunc_table*)handle;
    return(ifunc->busy[index]);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
static char G__buf[G__LONGLINE];
char* G__MethodInfo::GetPrototype()
{
  strcpy(G__buf,Type()->Name());
  strcat(G__buf," ");
  if(belongingclass && belongingclass->IsValid()) {
    strcat(G__buf,belongingclass->Name());
    strcat(G__buf,"::");
  }
  strcat(G__buf,Name());
  strcat(G__buf,"(");
  G__MethodArgInfo arg(*this);
  int flag=0;
  while(arg.Next()) {
    if(flag) strcat(G__buf,",");
    flag=1;
    strcat(G__buf,arg.Type()->Name());
    strcat(G__buf," ");
    if(arg.Name()) strcat(G__buf,arg.Name());
    if(arg.DefaultValue()) {
      strcat(G__buf,"=");
      strcat(G__buf,arg.DefaultValue());
    }
  }
  strcat(G__buf,")");
  return(G__buf);
}
///////////////////////////////////////////////////////////////////////////
char* G__MethodInfo::GetMangledName()
{
  return(G__map_cpp_name(GetPrototype()));
}
///////////////////////////////////////////////////////////////////////////
#ifndef G__OLDIMPLEMENTATION1908
extern "C" int G__DLL_direct_globalfunc(G__value *result7
					,G__CONST char *funcname
					,struct G__param *libp,int hash) ;
extern "C" void* G__FindSym(const char* filename,const char* funcname);
int G__MethodInfo::LoadDLLDirect(const char* filename,const char* funcname) 
{
  void* p2f;
  struct G__ifunc_table *ifunc;
  ifunc = (struct G__ifunc_table*)handle;
  p2f = G__FindSym(filename,funcname);
  if(p2f) {
    ifunc->pentry[index]->tp2f = p2f;
    ifunc->pentry[index]->p = (void*)G__DLL_direct_globalfunc;
#ifndef G__OLDIMPLEMENTATION2012
    ifunc->pentry[index]->size = -1;
    //ifunc->pentry[index]->filenum = -1; /* not good */
#else
    ifunc->pentry[index]->filenum = -1;
#endif
    ifunc->pentry[index]->line_number = -1;
    return 1;
  }
  return 0;
}
#endif
///////////////////////////////////////////////////////////////////////////

#ifndef G__OLDIMPLEMENTATION1294
///////////////////////////////////////////////////////////////////////////
// Global function to set precompiled library linkage
///////////////////////////////////////////////////////////////////////////
extern "C" int G__SetGlobalcomp(char *funcname,char *param,int globalcomp)
{
  G__ClassInfo globalscope;
  G__MethodInfo method;
  long dummy=0;
#ifndef G__OLDIMPLEMENTATION912
  char classname[G__LONGLINE];

  // Actually find the last :: to get the full classname, including
  // namespace and/or containing classes.
  strcpy(classname,funcname);
  char *fname = 0;
  char * tmp = classname;
  while ( (tmp = strstr(tmp,"::")) ) {
    fname = tmp;
    tmp += 2;
  }
  if(fname) {
    *fname=0;
    fname+=2;
    globalscope.Init(classname);
  }
  else {
    fname = funcname;
  }

  if(strcmp(fname,"*")==0) {
    method.Init(globalscope);
    while(method.Next()) {
      method.SetGlobalcomp(globalcomp);
    }
    return(0);
  }
  method=globalscope.GetMethod(fname,param,&dummy);

#else
  method=globalscope.GetMethod(funcname,param,&dummy);
#endif
  if(method.IsValid()) {
    method.SetGlobalcomp(globalcomp);
    return(0);
  }
  else {
    G__fprinterr(G__serr,"Warning: #pragma link, function %s(%s) not found"
#ifndef G__OLDIMPLEMENTATION912
	    ,fname,param);
#else
	    ,funcname,param);
#endif
    G__printlinenum();
    return(1);
  }
}
#else
///////////////////////////////////////////////////////////////////////////
// Global function to set precompiled library linkage
///////////////////////////////////////////////////////////////////////////
extern "C" int G__SetGlobalcomp(char *funcname,char *param,int globalcomp)
{
  G__ClassInfo globalscope;
  G__MethodInfo method;
  long dummy=0;
#ifndef G__OLDIMPLEMENTATION912
  char classname[G__LONGLINE];

  strcpy(classname,funcname);
  char *fname = strstr(classname,"::");
  if(fname) {
    *fname=0;
    fname+=2;
    globalscope.Init(classname);
  }
  else {
    fname = funcname;
  }

  if(strcmp(fname,"*")==0) {
    method.Init(globalscope);
    while(method.Next()) {
      method.SetGlobalcomp(globalcomp);
    }
    return(0);
  }
  method=globalscope.GetMethod(fname,param,&dummy);

#else
  method=globalscope.GetMethod(funcname,param,&dummy);
#endif
  if(method.IsValid()) {
    method.SetGlobalcomp(globalcomp);
    return(0);
  }
  else {
    G__fprinterr(G__serr,"Warning: #pragma link, function %s(%s) not found"
#ifndef G__OLDIMPLEMENTATION912
	    ,fname,param);
#else
	    ,funcname,param);
#endif
    G__printlinenum();
    return(1);
  }
}
#endif
///////////////////////////////////////////////////////////////////////////

#ifndef G__OLDIMPLEMENTATION1781
///////////////////////////////////////////////////////////////////////////
// Global function to set precompiled library linkage
///////////////////////////////////////////////////////////////////////////
extern "C" int G__ForceBytecodecompilation(char *funcname,char *param)
{
  G__ClassInfo globalscope;
  G__MethodInfo method;
  long dummy=0;
  char classname[G__LONGLINE];

  // Actually find the last :: to get the full classname, including
  // namespace and/or containing classes.
  strcpy(classname,funcname);
  char *fname = 0;
  char * tmp = classname;
  while ( (tmp = strstr(tmp,"::")) ) {
    fname = tmp;
    tmp += 2;
  }
  if(fname) {
    *fname=0;
    fname+=2;
    globalscope.Init(classname);
  }
  else {
    fname = funcname;
  }

  method=globalscope.GetMethod(fname,param,&dummy);

  if(method.IsValid()) {
    struct G__ifunc_table *ifunc = method.ifunc();
    int ifn = method.Index();
#ifndef G__OLDIMPLEMENTATION1842
    int stat;
    int store_asm_loopcompile = G__asm_loopcompile;
    int store_asm_loopcompile_mode = G__asm_loopcompile_mode;
    G__asm_loopcompile_mode=G__asm_loopcompile=4;
    stat = G__compile_bytecode(ifunc,ifn);
    G__asm_loopcompile=store_asm_loopcompile;
    G__asm_loopcompile_mode=store_asm_loopcompile_mode;
#else
    int stat = G__compile_bytecode(ifunc,ifn);
#endif
    if(stat) return 0;
    else return 1;
  }
  else {
    G__fprinterr(G__serr,"Warning: function %s(%s) not found"
#ifndef G__OLDIMPLEMENTATION912
	    ,fname,param);
#else
	    ,funcname,param);
#endif
    G__printlinenum();
    return(1);
  }
}
#endif

#ifndef G__OLDIMPLEMENTATION2073
///////////////////////////////////////////////////////////////////////////
// SetVtblIndex
///////////////////////////////////////////////////////////////////////////
void G__MethodInfo::SetVtblIndex(int vtblindex) {
  if(!IsValid()) return;
  struct G__ifunc_table* ifunc = (struct G__ifunc_table*)handle;
  ifunc->vtblindex[index] = (short)vtblindex;
}

///////////////////////////////////////////////////////////////////////////
// SetIsVirtual
///////////////////////////////////////////////////////////////////////////
void G__MethodInfo::SetIsVirtual(int isvirtual) {
  if(!IsValid()) return;
  struct G__ifunc_table* ifunc = (struct G__ifunc_table*)handle;
  ifunc->isvirtual[index] = isvirtual;
}
#endif

#ifndef G__OLDIMPLEMENTATION2073
///////////////////////////////////////////////////////////////////////////
// SetVtblBasetagnum
///////////////////////////////////////////////////////////////////////////
void G__MethodInfo::SetVtblBasetagnum(int basetagnum) {
  if(!IsValid()) return;
  struct G__ifunc_table* ifunc = (struct G__ifunc_table*)handle;
  ifunc->vtblbasetagnum[index] = (short)basetagnum;
}
#endif

