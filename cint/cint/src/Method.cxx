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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Api.h"
#include "common.h"

extern "C" int G__xrefflag;

/*********************************************************************
* class G__MethodInfo
*
* 
*********************************************************************/
///////////////////////////////////////////////////////////////////////////
void Cint::G__MethodInfo::Init()
{
  handle = (long)(G__get_ifunc_ref(&G__ifunc));
  index = -1;
#ifndef G__OLDIMPLEMENTATION2194
  usingIndex = -1;
#endif
  belongingclass=(G__ClassInfo*)NULL;
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__MethodInfo::Init(G__ClassInfo &a)
{
  if(a.IsValid()) {
    handle=(long)G__get_ifunc_ref(G__struct.memfunc[a.Tagnum()]);
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
void Cint::G__MethodInfo::Init(long handlein,long indexin
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
      belongingclass = (G__ClassInfo*)NULL;
    }

    /* Set return type */
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    type.type=ifunc2->type[index];
    type.tagnum=ifunc2->p_tagtable[index];
    type.typenum=ifunc2->p_typetable[index];
    type.reftype=ifunc2->reftype[index];
    type.isconst=ifunc2->isconst[index];
    type.class_property=0;
  }
  else { /* initialize if handlein==0 */
    handle=0;
    index=-1;
    belongingclass = (G__ClassInfo*)NULL;
  }
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__MethodInfo::Init(G__ClassInfo *belongingclassin
	,long funcpage,long indexin)
{
  struct G__ifunc_table_internal *ifunc2;
  int i=0;

  if(belongingclassin->IsValid()) {
    // member function
    belongingclass = belongingclassin;
    ifunc2 = G__struct.memfunc[belongingclassin->Tagnum()];
  }
  else {
    // global function
    belongingclass=(G__ClassInfo*)NULL;
    ifunc2 = G__p_ifunc;
  }

  // reach to desired page
  for(i=0;i<funcpage&&ifunc2;i++) ifunc2=ifunc2->next;
  G__ASSERT(!ifunc || ifunc2->page == funcpage);

  if(ifunc2) {
    handle = (long)G__get_ifunc_ref(ifunc2);
    index = indexin;
    // Set return type
    type.type=ifunc2->type[index];
    type.tagnum=ifunc2->p_tagtable[index];
    type.typenum=ifunc2->p_typetable[index];
    type.reftype=ifunc2->reftype[index];
    type.isconst=ifunc2->isconst[index];
    type.class_property=0;
  }
  else { /* initialize if handlein==0 */
    handle=0;
    index=-1;
    belongingclass=(G__ClassInfo*)NULL;
  }
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__MethodInfo::Name()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    return(ifunc2->funcname[index]);
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__MethodInfo::Hash()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    return(ifunc2->hash[index]);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
struct G__ifunc_table* Cint::G__MethodInfo::ifunc()
{
  if(IsValid()) {
    return (struct G__ifunc_table*)handle;
  }
  else {
    return((struct G__ifunc_table*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__MethodInfo::Title() 
{
  static char buf[G__INFO_TITLELEN];
  buf[0]='\0';
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    G__getcomment(buf,&ifunc2->comment[index],ifunc2->tagnum);
    return(buf);
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
G__ClassInfo *Cint::G__MethodInfo::MemberOf()
{
   if (!memberOf && IsValid()) {
      long tagnum = ((struct G__ifunc_table*)handle)->tagnum;
      if (belongingclass && tagnum == belongingclass->Tagnum() ) {
         memberOf = belongingclass;
      } else {
         memberOf = new G__ClassInfo(tagnum);
      }
   }
   return memberOf;
}

///////////////////////////////////////////////////////////////////////////
long Cint::G__MethodInfo::Property()
{
  if(IsValid()) {
    long property=0;
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    if (ifunc2->hash[index]==0) return property;
    switch(ifunc2->access[index]) {
    case G__PUBLIC: property|=G__BIT_ISPUBLIC; break;
    case G__PROTECTED: property|=G__BIT_ISPROTECTED; break;
    case G__PRIVATE: property|=G__BIT_ISPRIVATE; break;
    }
    if(ifunc2->isconst[index]&G__CONSTFUNC) property|=G__BIT_ISCONSTANT | G__BIT_ISMETHCONSTANT; 
    if(ifunc2->isconst[index]&G__CONSTVAR) property|=G__BIT_ISCONSTANT;
    if(ifunc2->isconst[index]&G__PCONSTVAR) property|=G__BIT_ISPCONSTANT;
    if(isupper(ifunc2->type[index])) property|=G__BIT_ISPOINTER;
    if(ifunc2->staticalloc[index]) property|=G__BIT_ISSTATIC;
    if(ifunc2->isvirtual[index]) property|=G__BIT_ISVIRTUAL;
    if(ifunc2->ispurevirtual[index]) property|=G__BIT_ISPUREVIRTUAL;
    if(ifunc2->pentry[index]->size<0) property|=G__BIT_ISCOMPILED;
    if(ifunc2->pentry[index]->bytecode) property|=G__BIT_ISBYTECODE;
    if(ifunc2->isexplicit[index]) property|=G__BIT_ISEXPLICIT;
    return(property);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__MethodInfo::NArg()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    return(ifunc2->para_nu[index]);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__MethodInfo::NDefaultArg()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    if(ifunc2->para_nu[index]) {
      int i,defaultnu=0;
      for(i=ifunc2->para_nu[index]-1;i>=0;i--) {
	     if(ifunc2->param[index][i]->pdefault) ++defaultnu;
	     else return(defaultnu);
      }
      return(defaultnu);
    }
    else {
      return(0);
    }
  }
  return(-1);
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__MethodInfo::HasVarArgs()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    return(2==ifunc2->ansi[index]?1:0);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
G__InterfaceMethod Cint::G__MethodInfo::InterfaceMethod()
{
  G__LockCriticalSection();
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    if(
       -1==ifunc2->pentry[index]->size /* this means compiled class */
       ) {
      G__UnlockCriticalSection();

      // 25-05-07
      // Problem: it has to be a compiled class here, but if we
      // are using the no-stub algorithm then the InterfaceMethod
      // is 0 (because we used our registered method in funcptr)
      // so now we have to deal with that situation in
      // TCint::GetInterfaceMethodWithPrototype
      //
      // For ROOT, his happens only in
      // static TMethod *TQObject::GetMethod(TClass *cl, const char *method, const char *params)
      // and I have seen that the address is not really used
      // they only want to know if the method can be executed...
      // this is extremely shady but for the moment just pass the funcptr
      // if the interface method is zero
      if((G__InterfaceMethod)ifunc2->pentry[index]->p)
         return((G__InterfaceMethod)ifunc2->pentry[index]->p);
      else
         // WARNING We are changing the semantics of this return.
         // If there no stub function we return the address of the function.
         // I don't know the consequences of this behaviour. We will have to recheck this point.
         return((G__InterfaceMethod) G__get_funcptr(ifunc2,index));
    }
    else {
      G__UnlockCriticalSection();
      return((G__InterfaceMethod)NULL);
    }
  }
  else {
    G__UnlockCriticalSection();
    return((G__InterfaceMethod)NULL);
  }
}
#ifdef G__ASM_WHOLEFUNC
///////////////////////////////////////////////////////////////////////////
struct G__bytecodefunc *Cint::G__MethodInfo::GetBytecode()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    int store_asm_loopcompile = G__asm_loopcompile;
    G__asm_loopcompile = 4;
    if(!ifunc2->pentry[index]->bytecode &&
       -1!=ifunc2->pentry[index]->size && 
       G__BYTECODE_NOTYET==ifunc2->pentry[index]->bytecodestatus
       && G__asm_loopcompile>=4
       ) {
      G__compile_bytecode((struct G__ifunc_table*)handle,(int)index);
    }
    G__asm_loopcompile = store_asm_loopcompile;
    return(ifunc2->pentry[index]->bytecode);
  }
  else {
    return((struct G__bytecodefunc*)NULL);
  }
}
#endif
/* #ifndef G__OLDIMPLEMENTATION1163 */
///////////////////////////////////////////////////////////////////////////
G__DataMemberInfo Cint::G__MethodInfo::GetLocalVariable()
{
  G__DataMemberInfo localvar;
  localvar.Init((long)0,(long)(-1),(G__ClassInfo*)NULL);
  if(IsValid()) {
    int store_fixedscope=G__fixedscope;
    G__xrefflag=1;
    G__fixedscope=1;
    struct G__bytecodefunc* pbc = GetBytecode();
    G__xrefflag=0;
    G__fixedscope=store_fixedscope;
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
void* Cint::G__MethodInfo::PointerToFunc()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    if(
       -1!=ifunc2->pentry[index]->size && 
       G__BYTECODE_NOTYET==ifunc2->pentry[index]->bytecodestatus
       && G__asm_loopcompile>=4
       ) {
      G__compile_bytecode((struct G__ifunc_table*)handle,(int)index);
    }
    if(G__BYTECODE_SUCCESS==ifunc2->pentry[index]->bytecodestatus) 
      return((void*)ifunc2->pentry[index]->bytecode);
      
    return(ifunc2->pentry[index]->tp2f);
  }
  else {
    return((void*)NULL);
  }
}
#endif
///////////////////////////////////////////////////////////////////////////
void Cint::G__MethodInfo::SetGlobalcomp(int globalcomp)
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    ifunc2->globalcomp[index]=globalcomp;
    if(G__NOLINK==globalcomp) ifunc2->access[index]=G__PRIVATE;
    else                      ifunc2->access[index]=G__PUBLIC;
  }
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__MethodInfo::SetForceStub()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    ifunc2->funcptr[index]=(void*)-2;
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__MethodInfo::IsValid()
{
  if(handle) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    if(ifunc2 && 0<=index&&index<ifunc2->allifunc) {
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
int Cint::G__MethodInfo::SetFilePos(const char* fname)
{
  struct G__dictposition* dict=G__get_dictpos((char*)fname);
  if(!dict) return(0);
  handle = (long)dict->ifunc;
  index = (long)(dict->ifn-1);
  belongingclass=(G__ClassInfo*)NULL;
  return(1);
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__MethodInfo::Next()
{
  if(handle) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    ++index;
    if(ifunc2->allifunc<=index) {
      int t = ifunc2->tagnum;
      ifunc2=ifunc2->next;
      if(ifunc2) {
	ifunc2->tagnum=t;
	handle=(long)G__get_ifunc_ref(ifunc2);
	index = 0;
      }
      else {
	handle=0;
	index = -1;
      }
    } 
#ifndef G__OLDIMPLEMENTATION2194
    if(ifunc2==0 && belongingclass==0 && 
       usingIndex<G__globalusingnamespace.basen) {
      ++usingIndex;
      index=0;
      G__incsetup_memfunc(G__globalusingnamespace.herit[usingIndex]->basetagnum);
      ifunc2=G__struct.memfunc[G__globalusingnamespace.herit[usingIndex]->basetagnum];
      handle=(long)G__get_ifunc_ref(ifunc2);
    }
#endif
    if(IsValid()) {
      type.type=ifunc2->type[index];
      type.tagnum=ifunc2->p_tagtable[index];
      type.typenum=ifunc2->p_typetable[index];
      type.reftype=ifunc2->reftype[index];
      type.isconst=ifunc2->isconst[index];
      type.class_property=0;
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
const char* Cint::G__MethodInfo::FileName()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    if(ifunc2->pentry[index]->filenum>=0) { /* 2012, keep this */
      return(G__srcfile[ifunc2->pentry[index]->filenum].filename);
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
FILE* Cint::G__MethodInfo::FilePointer()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    if(
       ifunc2->pentry[index]->filenum>=0 && ifunc2->pentry[index]->size>=0
       ) {
      return(G__srcfile[ifunc2->pentry[index]->filenum].fp);
    }
    else {
      return((FILE*)NULL);
    }
  }
  else {
    return((FILE*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__MethodInfo::LineNumber()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    if(
       ifunc2->pentry[index]->filenum>=0 && ifunc2->pentry[index]->size>=0
       ) {
      return(ifunc2->pentry[index]->line_number);
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
long Cint::G__MethodInfo::FilePosition()
{ 
  // returns  'type fname(type p1,type p2)'
  //                      ^
  long invalid=0L;
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    if(
       ifunc2->pentry[index]->filenum>=0 && ifunc2->pentry[index]->size>=0
       ) {
#if defined(G__NONSCALARFPOS2)
      return((long)ifunc2->pentry[index]->pos.__pos);
#elif defined(G__NONSCALARFPOS_QNX)      
      return((long)ifunc2->pentry[index]->pos._Off);
#else
      return((long)ifunc2->pentry[index]->pos);
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
///////////////////////////////////////////////////////////////////////////
int Cint::G__MethodInfo::Size()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    if(
       ifunc2->pentry[index]->size>=0
       ) {
      return(ifunc2->pentry[index]->size);
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
int Cint::G__MethodInfo::IsBusy()
{
  if(IsValid()) {
    struct G__ifunc_table_internal *ifunc2;
    ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
    return(ifunc2->busy[index]);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
char* Cint::G__MethodInfo::GetPrototype()
{
  static G__FastAllocString buf(G__LONGLINE); // valid until the next call of GetPrototype, just like any static
  if (!IsValid()) return 0;
  buf = Type()->Name();
  buf += " ";
  if(belongingclass && belongingclass->IsValid()) {
    buf += belongingclass->Fullname();
    buf += "::";
  }
  buf += Name();
  buf += "(";
  G__MethodArgInfo arg(*this);
  int flag=0;
  while(arg.Next()) {
    if(flag) buf += ",";
    flag=1;
    buf += arg.Type()->Name();
    buf += " ";
    if(arg.Name()) buf += arg.Name();
    if(arg.DefaultValue()) {
      buf += "=";
      buf += arg.DefaultValue();
    }
  }
  buf += ")";
  return buf;
}
///////////////////////////////////////////////////////////////////////////
char* Cint::G__MethodInfo::GetMangledName()
{
  if (!IsValid()) return 0;
  return(G__map_cpp_name(GetPrototype()));
}
///////////////////////////////////////////////////////////////////////////
extern "C" int G__DLL_direct_globalfunc(G__value *result7
					,G__CONST char *funcname
					,struct G__param *libp,int hash) ;
extern "C" void* G__FindSym(const char* filename,const char* funcname);
int Cint::G__MethodInfo::LoadDLLDirect(const char* filename,const char* funcname) 
{
  void* p2f;
  struct G__ifunc_table_internal *ifunc2;
  ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
  p2f = G__FindSym(filename,funcname);
  if(p2f) {
    ifunc2->pentry[index]->tp2f = p2f;
    ifunc2->pentry[index]->p = (void*)G__DLL_direct_globalfunc;
    ifunc2->pentry[index]->size = -1;
    //ifunc2->pentry[index]->filenum = -1; /* not good */
    ifunc2->pentry[index]->line_number = -1;
    return 1;
  }
  return 0;
}
///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
// Global function to set precompiled library linkage
///////////////////////////////////////////////////////////////////////////
int Cint::G__SetGlobalcomp(char *funcname,char *param,int globalcomp)
{
  G__ClassInfo globalscope;
  G__MethodInfo method;
  long dummy=0;
  G__FastAllocString classname(funcname);

  // Actually find the last :: to get the full classname, including
  // namespace and/or containing classes.
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

  if(method.IsValid()) {
    method.SetGlobalcomp(globalcomp);
    return(0);
  }
  else {
    G__fprinterr(G__serr,"Warning: #pragma link, function %s(%s) not found"
	    ,fname,param);
    G__printlinenum();
    return(1);
  }
}
///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
// Global function to force the creation of the stub when using G__NOSTUBS
///////////////////////////////////////////////////////////////////////////
int Cint::G__SetForceStub(char *funcname,char *param)
{
  G__ClassInfo globalscope;
  G__MethodInfo method;
  long dummy=0;
  G__FastAllocString classname(funcname);

  // Actually find the last :: to get the full classname, including
  // namespace and/or containing classes.
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
      method.SetForceStub();
    }
    return(0);
  }
  method=globalscope.GetMethod(fname,param,&dummy);

  if(method.IsValid()) {
    method.SetForceStub();
    return(0);
  }
  else {
    G__fprinterr(G__serr,"Warning: #pragma link, function %s(%s) not found"
	    ,fname,param);
    G__printlinenum();
    return(1);
  }
}
///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
// Global function to set precompiled library linkage
///////////////////////////////////////////////////////////////////////////
int Cint::G__ForceBytecodecompilation(char *funcname,char *param)
{
  G__ClassInfo globalscope;
  G__MethodInfo method;
  long dummy=0;
  G__FastAllocString classname(funcname);

  // Actually find the last :: to get the full classname, including
  // namespace and/or containing classes.
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
    struct G__ifunc_table *ifunc2 = method.ifunc();
    int ifn = method.Index();
    int stat;
    int store_asm_loopcompile = G__asm_loopcompile;
    int store_asm_loopcompile_mode = G__asm_loopcompile_mode;
    G__asm_loopcompile_mode=G__asm_loopcompile=4;
    stat = G__compile_bytecode(ifunc2,ifn);
    G__asm_loopcompile=store_asm_loopcompile;
    G__asm_loopcompile_mode=store_asm_loopcompile_mode;
    if(stat) return 0;
    else return 1;
  }
  else {
    G__fprinterr(G__serr,"Warning: function %s(%s) not found"
	    ,fname,param);
    G__printlinenum();
    return(1);
  }
}

///////////////////////////////////////////////////////////////////////////
// SetVtblIndex
///////////////////////////////////////////////////////////////////////////
void Cint::G__MethodInfo::SetVtblIndex(int vtblindex) {
  if(!IsValid()) return;
  struct G__ifunc_table_internal* ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
  ifunc2->vtblindex[index] = (short)vtblindex;
}

///////////////////////////////////////////////////////////////////////////
// SetIsVirtual
///////////////////////////////////////////////////////////////////////////
void Cint::G__MethodInfo::SetIsVirtual(int isvirtual) {
  if(!IsValid()) return;
  struct G__ifunc_table_internal* ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
  ifunc2->isvirtual[index] = isvirtual;
}

///////////////////////////////////////////////////////////////////////////
// SetVtblBasetagnum
///////////////////////////////////////////////////////////////////////////
void Cint::G__MethodInfo::SetVtblBasetagnum(int basetagnum) {
  if(!IsValid()) return;
  struct G__ifunc_table_internal* ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
  ifunc2->vtblbasetagnum[index] = (short)basetagnum;
}

///////////////////////////////////////////////////////////////////////////
// GetFriendInfo
///////////////////////////////////////////////////////////////////////////
G__friendtag*  Cint::G__MethodInfo::GetFriendInfo() { 
   if(IsValid()) {
      struct G__ifunc_table_internal* ifunc2 = G__get_ifunc_internal((struct G__ifunc_table*)handle);
      return(ifunc2->friendtag[index]);
   }
   else return 0;
}
///////////////////////////////////////////////////////////////////////////
// GetDefiningScopeTagnum
///////////////////////////////////////////////////////////////////////////
int Cint::G__MethodInfo::GetDefiningScopeTagnum()
{
   if (IsValid()) {
      return ifunc()->tagnum;
   } 
   else return -1;
}
///////////////////////////////////////////////////////////////////////////
// SetUserParam
///////////////////////////////////////////////////////////////////////////
void Cint::G__MethodInfo::SetUserParam(void *user) 
{
   if (IsValid()) {
      struct G__ifunc_table_internal* ifunc_internal = G__get_ifunc_internal((struct G__ifunc_table*)ifunc());
      ifunc_internal->userparam[index] = user;
   }
}
///////////////////////////////////////////////////////////////////////////
// GetUserParam
///////////////////////////////////////////////////////////////////////////
void *Cint::G__MethodInfo::GetUserParam()
{
   if (IsValid()) {
      struct G__ifunc_table_internal* ifunc_internal = G__get_ifunc_internal((struct G__ifunc_table*)ifunc());
      return ifunc_internal->userparam[index];
   }
   else return 0;
}
///////////////////////////////////////////////////////////////////////////
// GetThisPointerOffset 
// Return: Return the this-pointer offset, to adjust it in case of non left-most
// multiple inheritance
///////////////////////////////////////////////////////////////////////////
long Cint::G__MethodInfo::GetThisPointerOffset()
{
   if (IsValid()) {
      struct G__ifunc_table_internal* ifunc_internal = G__get_ifunc_internal((struct G__ifunc_table*)ifunc());
      return ifunc_internal->entry[0].ptradjust;
   }
   else return 0;
}
