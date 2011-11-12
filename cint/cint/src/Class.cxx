/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file Class.cxx
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Author                  Masaharu Goto 
 * Copyright(c) 1995~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Api.h"
#include "common.h"
#include "FastAllocString.h"

extern "C" void G__exec_alloc_lock();
extern "C" void G__exec_alloc_unlock();

#ifndef G__OLDIMPLEMENTATION1586
static char G__buf[G__ONELINE];
#endif

/*********************************************************************
* class Cint::G__ClassInfo
*********************************************************************/

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::Init()
{
  tagnum = -1;
  class_property = 0;
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::Init(const char *classname)
{
   if (strchr(classname,'<')) {
      // G__defined_tagnum might modify classname.
      G__FastAllocString tmp(1+strlen(classname)*2);
      tmp = classname;
      tagnum = G__defined_tagname(tmp,1);
   } else {
      tagnum = G__defined_tagname(classname,1);
   }
   class_property = 0;
}
///////////////////////////////////////////////////////////////////////////
Cint::G__ClassInfo::G__ClassInfo(const G__value &value_for_type) : tagnum(0), class_property(0)
{
   Init(value_for_type.tagnum);
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::Init(int tagnumin)
{
  tagnum = tagnumin;
  class_property = 0;
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::operator==(const Cint::G__ClassInfo& a)
{
  return(tagnum == a.tagnum);
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::operator!=(const Cint::G__ClassInfo& a)
{
  return(tagnum != a.tagnum);
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::Name()
{
  if(IsValid()) {
    return(G__struct.name[tagnum]);
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::Fullname()
{
  if(IsValid()) {
#ifndef G__OLDIMPLEMENTATION1586
   strncpy(G__buf,G__fulltagname((int)tagnum,1), sizeof(G__buf) - 1);
#if defined(_MSC_VER) && (_MSC_VER < 1300) /*vc6*/
   char *ptr = strstr(G__buf, "long long");
   if (ptr) {
      memcpy(ptr, " __int64 ", strlen( " __int64 "));
   }
#endif
    return(G__buf);
#else
    return(G__fulltagname((int)tagnum,1));
#endif
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::Title() 
{
  static char buf[G__INFO_TITLELEN];
  buf[0]='\0';
  if(IsValid()) {
    G__getcomment(buf,&G__struct.comment[tagnum],(int)tagnum);
    return(buf);
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::Size() 
{
  if(IsValid()) {
    return(G__struct.size[tagnum]);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
long Cint::G__ClassInfo::Property()
{
  if (class_property) return class_property;
  if(IsValid()) {
    long property=0;
    switch(G__struct.type[tagnum]) {
    case 'e': property |= G__BIT_ISENUM; break;
    case 'c': property |= G__BIT_ISCLASS; break;
    case 's': property |= G__BIT_ISSTRUCT; break;
    case 'u': property |= G__BIT_ISUNION; break;
    case 'n': property |= G__BIT_ISNAMESPACE; break;
    }
    if(G__struct.istypedefed[tagnum]) property |= G__BIT_ISTYPEDEF;
    if(G__struct.isabstract[tagnum]) property |= G__BIT_ISABSTRACT;
    switch(G__struct.iscpplink[tagnum]) {
    case G__CPPLINK: property |= G__BIT_ISCPPCOMPILED; break;
    case G__CLINK: property |= G__BIT_ISCCOMPILED; break;
    case G__NOLINK: break;
    default: break;
    }
    class_property = property;
    return(property);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::NDataMembers()
{
  struct G__var_array *var;
  int ndatamembers=0;
  if(IsValid()) {
    G__incsetup_memvar((int)tagnum);
    var = G__struct.memvar[tagnum];
    while(var) {
      ndatamembers += var->allvar;
      var=var->next;
    }
    return(ndatamembers);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::NMethods()
{
  struct G__ifunc_table_internal *ifunc;
  int nmethod=0;
  if(IsValid()) {
    G__incsetup_memfunc((int)tagnum);
    ifunc = G__struct.memfunc[tagnum];
    while(ifunc) {
      nmethod += ifunc->allifunc;
      ifunc=ifunc->next;
    }
    return(nmethod);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
long Cint::G__ClassInfo::IsBase(const char *classname)
{
  Cint::G__ClassInfo base(classname);
  return(IsBase(base));
}
///////////////////////////////////////////////////////////////////////////
long Cint::G__ClassInfo::IsBase(G__ClassInfo& a)
{
  G__inheritance *baseclass;
  int i;
  long isbase=0;
  if(IsValid()) {
    baseclass = G__struct.baseclass[tagnum];
    for(i=0;i<baseclass->basen;i++) {
      if(a.Tagnum() == baseclass->herit[i]->basetagnum) {
	switch(baseclass->herit[i]->baseaccess) {
	case G__PUBLIC: isbase = G__BIT_ISPUBLIC; break;
	case G__PROTECTED: isbase = G__BIT_ISPROTECTED; break;
	case G__PRIVATE: isbase = G__BIT_ISPRIVATE; break;
	default: isbase = 0; break;
	}
	if(baseclass->herit[i]->property&G__ISDIRECTINHERIT) 
	  isbase |= G__BIT_ISDIRECTINHERIT;
	if(baseclass->herit[i]->property&G__ISVIRTUALBASE) 
	  isbase |= G__BIT_ISVIRTUALBASE;
	return(isbase);
      }
    }
    return(0);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
void* Cint::G__ClassInfo::DynamicCast(G__ClassInfo& to, void* obj)
{
   // Cast an object (pointed to by obj) of this class to an object
   // of class "to", return pointer to object. Sort of corresponds to
   //    dynamic_cast<to*>((this)obj)
   return G__dynamiccast(to.Tagnum(), Tagnum(), obj);
}
///////////////////////////////////////////////////////////////////////////
G__ClassInfo Cint::G__ClassInfo::EnclosingClass()
{
  if(IsValid()) {
    Cint::G__ClassInfo enclosingclass(G__struct.parent_tagnum[tagnum]);
    return(enclosingclass);
  }
  else {
    Cint::G__ClassInfo enclosingclass;
    return(enclosingclass);
  }
}
///////////////////////////////////////////////////////////////////////////
G__ClassInfo Cint::G__ClassInfo::EnclosingSpace()
{
  if(IsValid()) {
    int enclosed_tag = G__struct.parent_tagnum[tagnum];
    while (enclosed_tag>=0 && (G__struct.type[enclosed_tag]!='n')) {
       enclosed_tag = G__struct.parent_tagnum[enclosed_tag];
    }
    Cint::G__ClassInfo enclosingclass(enclosed_tag);
    return(enclosingclass);
  }
  else {
    Cint::G__ClassInfo enclosingclass;
    return(enclosingclass);
  }
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::SetGlobalcomp(G__SIGNEDCHAR_T globalcomp)
{
  if(IsValid()) {
    G__struct.globalcomp[tagnum] = globalcomp;
  }
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::SetProtectedAccess(int protectedaccess)
{
  if(IsValid()) {
    G__struct.protectedaccess[tagnum] = protectedaccess;
  }
}
///////////////////////////////////////////////////////////////////////////
#ifndef G__OLDIMPLEMENTATION1218_YET
int Cint::G__ClassInfo::IsValid()
{
  if(0<=tagnum && tagnum<G__struct.alltag) {
    return(1);
  }
  else {
    return(0);
  }
}
#endif
///////////////////////////////////////////////////////////////////////////
unsigned char Cint::G__ClassInfo::FuncFlag() { 
   return (IsValid()? G__struct.funcs[tagnum] : (unsigned char)0); 
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::IsLoaded()
{
  if(IsValid() && 
     (G__NOLINK!=G__struct.iscpplink[tagnum]||-1!=G__struct.filenum[tagnum])) {
    return(1);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::SetFilePos(const char *fname)
{
  struct G__dictposition* dict=G__get_dictpos((char*)fname);
  if(!dict) return(0);
  tagnum=dict->tagnum-1;
  class_property = 0;
  return(1);
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::Next()
{
  ++tagnum;
  class_property = 0;
  return(IsValid());
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::Linkage()
{
  return G__struct.globalcomp[tagnum];
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::FileName()
{
  if(IsValid()) {
#ifndef G__OLDIMPLEMENtATION2012
    if(-1!=G__struct.filenum[tagnum]) {
      return(G__srcfile[G__struct.filenum[tagnum]].filename);
    }
    else {
      switch(G__struct.iscpplink[tagnum]) {
      case G__CLINK:
	return("(C compiled)");
      case G__CPPLINK:
	return("(C++ compiled)");
      default:
	return((char*)NULL);
      }
    }
#else
    switch(G__struct.iscpplink[tagnum]) {
    case G__CLINK:
      return("(C compiled)");
    case G__CPPLINK:
      return("(C++ compiled)");
    case G__NOLINK:
      if(-1!=G__struct.filenum[tagnum]) {
	return(G__srcfile[G__struct.filenum[tagnum]].filename);
      }
      else {
	return("(unkonwn)");
      }
    default:
      return((char*)NULL);
    }
#endif
  }
  return((char*)NULL);
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::LineNumber()
{
  if(IsValid()) {
    switch(G__struct.iscpplink[tagnum]) {
    case G__CLINK:
      return(0);
    case G__CPPLINK:
      return(0);
    case G__NOLINK:
      if(-1!=G__struct.filenum[tagnum]) {
	return(G__struct.line_number[tagnum]);
      }
      else {
	return(-1);
      }
    default:
      return(-1);
    }
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::IsTmplt()
{
  if(IsValid()) {
    char *p = strchr((char*)Name(),'<');
    if(p) return(1);
    else return(0);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::TmpltName()
{
  static char buf[G__ONELINE];
  if(IsValid()) {
    char *p;
    strncpy(buf, Name(), sizeof(buf) - 1);
    p = strchr(buf,'<');
    if(p) *p = 0;
    return(buf);
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::TmpltArg()
{
  static char buf[G__ONELINE];
  if(IsValid()) {
    char *p = strchr((char*)Name(),'<');
    if(p) {
      strncpy(buf,p+1, sizeof(buf) - 1);
      p=strrchr(buf,'>');
      if(p) { 
	*p=0;
        while(isspace(*(--p))) *p=0;
      }
      return(buf);
    }
    else {
      return((char*)NULL);
    }
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////

/*********************************************************************
* ROOT project special requirements
*********************************************************************/
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::SetDefFile(char *deffilein)
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->deffile = deffilein;
  }
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::SetDefLine(int deflinein)
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->defline = deflinein;
  }
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::SetImpFile(char *impfilein)
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->impfile = impfilein;
  }
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::SetImpLine(int implinein)
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->impline = implinein;
  }
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::SetVersion(int versionin)
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->version = versionin;
  }
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::DefFile()
{
  if(IsValid()) {
    CheckValidRootInfo();
    return(G__struct.rootspecial[tagnum]->deffile);
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::DefLine()
{
  if(IsValid()) {
    CheckValidRootInfo();
    return(G__struct.rootspecial[tagnum]->defline);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::ImpFile()
{
  if(IsValid()) {
    CheckValidRootInfo();
    return(G__struct.rootspecial[tagnum]->impfile);
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::ImpLine()
{
  if(IsValid()) {
    CheckValidRootInfo();
    return(G__struct.rootspecial[tagnum]->impline);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::Version()
{
  if(IsValid()) {
    CheckValidRootInfo();
    return(G__struct.rootspecial[tagnum]->version);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::InstanceCount() 
{
  if(IsValid()) {
    CheckValidRootInfo();
    return(G__struct.rootspecial[tagnum]->instancecount);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::ResetInstanceCount()
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->instancecount = 0;
  }
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::IncInstanceCount()
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->instancecount += 1;
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::HeapInstanceCount() 
{
  if(IsValid()) {
    CheckValidRootInfo();
    return(G__struct.rootspecial[tagnum]->heapinstancecount);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::ResetHeapInstanceCount()
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->heapinstancecount = 0;
  }
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::IncHeapInstanceCount()
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->heapinstancecount += 1;
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::RootFlag()
{
  return G__struct.rootflag[tagnum];
}
///////////////////////////////////////////////////////////////////////////
G__InterfaceMethod Cint::G__ClassInfo::GetInterfaceMethod(const char* fname
						    ,const char* arg
						    ,long* poffset
						    ,MatchMode mode
                                                    ,InheritanceMode imode
						)
{
  struct G__ifunc_table_internal *ifunc;
  char *funcname;
  char *param;
  long index;

  /* Search for method */
  if(-1==tagnum) ifunc = &G__ifunc;
  else           ifunc = G__struct.memfunc[tagnum];
  funcname = (char*)fname;
  param = (char*)arg;
  ifunc = G__get_ifunc_internal(G__get_methodhandle(funcname,param,G__get_ifunc_ref(ifunc),&index,poffset
			      ,(mode==ConversionMatch)?1:0
                              ,imode
			      ));

  if(
     ifunc && -1==ifunc->pentry[index]->size
     ) {
    return((G__InterfaceMethod)ifunc->pentry[index]->p);
  }
  else {
    return((G__InterfaceMethod)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
G__MethodInfo Cint::G__ClassInfo::GetMethod(const char* fname,const char* arg
				      ,long* poffset
				      ,MatchMode mode
				      ,InheritanceMode imode
				      )
{
  struct G__ifunc_table_internal *ifunc;
  char *funcname;
  char *param;
  long index;

  /* Search for method */
  if(-1==tagnum) ifunc = &G__ifunc;
  else           ifunc = G__struct.memfunc[tagnum];
  funcname = (char*)fname;
  param = (char*)arg;
  int convmode;
  switch(mode) {
  case ExactMatch:              convmode=0; break;
  case ConversionMatch:         convmode=1; break;
  case ConversionMatchBytecode: convmode=2; break;
  default:                      convmode=0; break;
  }
  G__ifunc_table* iref = G__get_methodhandle(funcname,param,G__get_ifunc_ref(ifunc),&index,poffset
			      ,convmode
			      ,(imode==WithInheritance)?1:0
			      );

  /* Initialize method object */
  G__MethodInfo method;
  method.Init((long)iref,index,this);
  return(method);
}
///////////////////////////////////////////////////////////////////////////
G__MethodInfo Cint::G__ClassInfo::GetMethod(const char* fname,struct G__param* libp
				      ,long* poffset
				      ,MatchMode mode
				      ,InheritanceMode imode
                                      )
{
  struct G__ifunc_table_internal *ifunc;
  char *funcname = (char*)fname;
  long index;

  /* Search for method */
  if(-1==tagnum) ifunc = &G__ifunc;
  else           ifunc = G__struct.memfunc[tagnum];

  G__ifunc_table* iref = G__get_methodhandle2(funcname,libp,G__get_ifunc_ref(ifunc),&index,poffset
			       ,(mode==ConversionMatch)?1:0
			      ,(imode==WithInheritance)?1:0
                               );

  /* Initialize method object */
  G__MethodInfo method;
  method.Init((long)iref,index,this);
  return(method);
}
///////////////////////////////////////////////////////////////////////////
G__MethodInfo Cint::G__ClassInfo::GetDefaultConstructor() {
  // TODO, reserve location for default ctor for tune up
  long dmy;
  G__MethodInfo method;
  G__FastAllocString fname(Name());
  method = GetMethod(fname,"",&dmy,ExactMatch,InThisScope);
  return(method);
}
///////////////////////////////////////////////////////////////////////////
G__MethodInfo Cint::G__ClassInfo::GetCopyConstructor() {
  // TODO, reserve location for copy ctor for tune up
  long dmy;
  G__MethodInfo method;
  G__FastAllocString fname(Name());
  G__FastAllocString arg(strlen(Name())+10);
  arg.Format("const %s&", Name());
  method = GetMethod(fname,arg,&dmy,ExactMatch,InThisScope);
  return(method);
}
///////////////////////////////////////////////////////////////////////////
G__MethodInfo Cint::G__ClassInfo::GetDestructor() {
  // TODO, dtor location is already reserved, ready for tune up
  long dmy;
  G__MethodInfo method;
  G__FastAllocString fname(strlen(Name())+2);
  fname.Format("~%s",Name());
  method = GetMethod(fname,"",&dmy,ExactMatch,InThisScope);
  return(method);
}
///////////////////////////////////////////////////////////////////////////
G__MethodInfo Cint::G__ClassInfo::GetAssignOperator() {
  // TODO, reserve operator= location for tune up
  long dmy;
  G__MethodInfo method;
  G__FastAllocString arg(strlen(Name()) + 10);
  arg.Format("const %s&", Name());
  method = GetMethod("operator=",arg,&dmy,ExactMatch,InThisScope);
  return(method);
}
///////////////////////////////////////////////////////////////////////////
G__DataMemberInfo Cint::G__ClassInfo::GetDataMember(const char* name,long* poffset)
{
  char *varname;
  int hash;
  int temp;
  long original=0;
  int ig15;
  struct G__var_array *var;
  int store_tagnum;
  
  /* search for variable */
  G__hash(name,hash,temp);
  varname=(char*)name;
  *poffset = 0;
  if(-1==tagnum) var = &G__global;
  else           var = G__struct.memvar[tagnum];
  store_tagnum=G__tagnum;
  G__tagnum = (int)tagnum;
  var = G__searchvariable(varname,hash,var,(struct G__var_array*)NULL
			  ,poffset,&original,&ig15,0);
  G__tagnum=store_tagnum;
  /* Set data member object */
  G__DataMemberInfo datamember;
  datamember.Init((long)var,(long)ig15,this);
  return(datamember);
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::HasDefaultConstructor()
{
  if(IsValid()) {
     CheckValidRootInfo();
     return(G__struct.rootspecial[tagnum]->defaultconstructor!=0);
  } else {
     return 0;
  }
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::HasMethod(const char *fname)
{
  struct G__ifunc_table_internal *ifunc;
  int ifn;
  int hash;
  if(IsValid()) {
    G__hash(fname,hash,ifn);
    G__incsetup_memfunc((int)tagnum);
    ifunc=G__struct.memfunc[tagnum];
    while(ifunc) {
      for(ifn=0;ifn<ifunc->allifunc;ifn++) {
	if(hash==ifunc->hash[ifn] &&
	   strcmp(fname,ifunc->funcname[ifn])==0) {
	  return(1);
	}
      }
      ifunc=ifunc->next;
    }
  }
  return(0);
}
///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::HasDataMember(const char *name)
{
  struct G__var_array *var;
  int ig15;
  int hash;
  if(IsValid()) {
    G__hash(name,hash,ig15);
    G__incsetup_memvar((int)tagnum);
    var=G__struct.memvar[tagnum];
    while(var) {
      for(ig15=0;ig15<var->allvar;ig15++) {
	if(hash==var->hash[ig15] &&
	   strcmp(name,var->varnamebuf[ig15])==0) {
	  return(1);
	}
      }
      var=var->next;
    }
  }
  return(0);
}
///////////////////////////////////////////////////////////////////////////
void* Cint::G__ClassInfo::New()
{
  if(IsValid()) {
    void *p = (void*)NULL;
    G__value buf=G__null;
    if (!class_property) Property();
    if(class_property&G__BIT_ISCPPCOMPILED) {
      // C++ precompiled class,struct
      struct G__param *para = new G__param();
      G__InterfaceMethod defaultconstructor;
      para->paran=0;
      if(!G__struct.rootspecial[tagnum]) CheckValidRootInfo();

      /* If we have a stub for the default constructor */
       defaultconstructor
          =(G__InterfaceMethod)G__struct.rootspecial[tagnum]->defaultconstructor;
       if(defaultconstructor) {
          long ltagnum = tagnum;
          G__CurrentCall(G__DELETEFREE, this, &ltagnum);
          (*defaultconstructor)(&buf,(char*)NULL,para,0);
          G__CurrentCall(G__NOP, 0, 0);
          p = (void*)G__int(buf);
       }
#ifdef G__NOSTUBS
       else{
          /* If we have a function pointer */
          G__ifunc_table_internal* internalifuncconst 
             = G__get_ifunc_internal(G__struct.rootspecial[tagnum]->defaultconstructorifunc);
          if(internalifuncconst && internalifuncconst->funcptr[0]){
             long ltagnum = tagnum;
             G__CurrentCall(G__DELETEFREE, this, &ltagnum);
             G__stub_method_calling(&buf, para, internalifuncconst, 0);
             G__CurrentCall(G__NOP, 0, 0);
             p = (void*)G__int(buf);
          }
       }
#endif
       delete para;
    }
    else if(class_property&G__BIT_ISCCOMPILED) {
      // C precompiled class,struct
      p = new char[G__struct.size[tagnum]];
    }
    else {
      // Interpreted class,struct
      long store_struct_offset;
      long store_tagnum;
      G__FastAllocString temp(G__ONELINE);
      int known=0;
      p = new char[G__struct.size[tagnum]];
      store_tagnum = G__tagnum;
      store_struct_offset = G__store_struct_offset;
      G__tagnum = tagnum;
      G__store_struct_offset = (long)p;
      temp.Format("%s()",G__struct.name[tagnum]);
      G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
      G__store_struct_offset = store_struct_offset;
      G__tagnum = (int)store_tagnum;
    }
    return(p);
  }
  else {
    return((void*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
void* Cint::G__ClassInfo::New(int n)
{
  if(IsValid() && n>0 ) {
    void *p = (void*)NULL;
    G__value buf=G__null;
    if (!class_property) Property();
    if(class_property&G__BIT_ISCPPCOMPILED) {
      // C++ precompiled class,struct
      struct G__param* para = new G__param();
      G__InterfaceMethod defaultconstructor;
      para->paran=0;
      if(!G__struct.rootspecial[tagnum]) CheckValidRootInfo();
      
      // DMS 13-II-2007

      /* If we have a stub for the default constructor */
      defaultconstructor
          =(G__InterfaceMethod)G__struct.rootspecial[tagnum]->defaultconstructor;
       if(defaultconstructor) {
          long ltagnum = tagnum;
          G__CurrentCall(G__DELETEFREE, this, &ltagnum);
          (*defaultconstructor)(&buf,(char*)NULL,para,0);
          G__CurrentCall(G__NOP, 0, 0);
          p = (void*)G__int(buf);
       }
#ifdef G__NOSTUBS
       else{
          /* If we have a function pointer */
          G__ifunc_table_internal* internalifuncconst 
             = G__get_ifunc_internal(G__struct.rootspecial[tagnum]->defaultconstructorifunc);
          if(internalifuncconst->funcptr[0]){
             long ltagnum = tagnum;
             G__CurrentCall(G__DELETEFREE, this, &ltagnum);
             G__stub_method_calling(&buf, para, internalifuncconst, 0);
             G__CurrentCall(G__NOP, 0, 0);
             p = (void*)G__int(buf);
          }
       }
#endif

      // Record that we have allocated an array, and how many
      // elements that array has, for use by the G__calldtor function.
      G__alloc_newarraylist((long) p, n);
      delete para;
    }
    else if(class_property&G__BIT_ISCCOMPILED) {
       // C precompiled class,struct
       p = new char[G__struct.size[tagnum]*n];
    }
    else {
      // Interpreted class,struct
      int i;
      long store_struct_offset;
      long store_tagnum;
      int known=0;
      p = new char[G__struct.size[tagnum]*n];
      // Record that we have allocated an array, and how many
      // elements that array has, for use by the G__calldtor function.
      G__alloc_newarraylist((long) p, n);
      store_tagnum = G__tagnum;
      store_struct_offset = G__store_struct_offset;
      G__tagnum = tagnum;
      G__store_struct_offset = (long)p;
      //// Do it this way for an array cookie implementation.
      ////p = new char[(G__struct.size[tagnum]*n)+(2*sizeof(int))];
      ////int* pp = (int*) p;
      ////pp[0] = G__struct.size[tagnum];
      ////pp[1] = n;
      ////G__store_struct_offset = (long)(((char*)p) + (2*sizeof(int)));
      ////... at end adjust returned pointer address ...
      ////p = ((char*) p) + (2 * sizeof(int));
      G__FastAllocString temp(G__struct.name[tagnum]);
      temp += "()";
      for(i=0;i<n;i++) {
	G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
	if(!known) break;
	G__store_struct_offset += G__struct.size[tagnum];
      }
      G__store_struct_offset = store_struct_offset;
      G__tagnum = (int)store_tagnum;
    }
    return(p);
  }
  else {
    return((void*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
void* Cint::G__ClassInfo::New(void *arena)
{
  if(IsValid()) {
    void *p = 0;
    G__value buf=G__null;
    if (!class_property) Property();
    if(class_property&G__BIT_ISCPPCOMPILED) {
      // C++ precompiled class,struct
      struct G__param* para = new G__param();
      G__InterfaceMethod defaultconstructor;
      para->paran=0;
      if(!G__struct.rootspecial[tagnum]) CheckValidRootInfo();
    
      // DMS 13-II-2007

      /* If we have a stub for the default constructor */
      defaultconstructor
          =(G__InterfaceMethod)G__struct.rootspecial[tagnum]->defaultconstructor;
       if(defaultconstructor) {
          long ltagnum = tagnum;
          G__CurrentCall(G__DELETEFREE, this, &ltagnum);
#ifdef G__ROOT
         G__exec_alloc_lock();
#endif
          (*defaultconstructor)(&buf,(char*)NULL,para,0);
          G__CurrentCall(G__NOP, 0, 0);
          p = (void*)G__int(buf);
       }
#ifdef G__NOSTUBS
       else{
          /* If we have a function pointer */
          G__ifunc_table_internal* internalifuncconst 
             = G__get_ifunc_internal(G__struct.rootspecial[tagnum]->defaultconstructorifunc);
          if(internalifuncconst->funcptr[0]){
             long ltagnum = tagnum;
             G__CurrentCall(G__DELETEFREE, this, &ltagnum);
#ifdef G__ROOT
         G__exec_alloc_lock();
#endif
             G__stub_method_calling(&buf, para, internalifuncconst, 0);
             G__CurrentCall(G__NOP, 0, 0);
             p = (void*)G__int(buf);
          }
       }
#endif
       delete para;
    }
    else if(class_property&G__BIT_ISCCOMPILED) {
      // C precompiled class,struct
      p = arena;
    }
    else {
      // Interpreted class,struct
      long store_struct_offset;
      long store_tagnum;
      int known=0;
      p = arena;
      store_tagnum = G__tagnum;
      store_struct_offset = G__store_struct_offset;
      G__tagnum = tagnum;
      G__store_struct_offset = (long)p;
      G__FastAllocString temp(G__struct.name[tagnum]);
      temp += "()";
      G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
      G__store_struct_offset = store_struct_offset;
      G__tagnum = (int)store_tagnum;
    }
    return(p);
  }
  else {
    return((void*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
void* Cint::G__ClassInfo::New(int n, void *arena)
{
  if(IsValid() && (n > 0)) {
    void *p = (void*)NULL ;
    G__value buf=G__null;
    if (!class_property) Property();
    if(class_property&G__BIT_ISCPPCOMPILED) {
      // C++ precompiled class,struct
      struct G__param* para = new G__param();
      G__InterfaceMethod defaultconstructor;
      para->paran=0;
      if(!G__struct.rootspecial[tagnum]) CheckValidRootInfo();

        // DMS 13-II-2007

      /* If we have a stub for the default constructor */
      defaultconstructor
          =(G__InterfaceMethod)G__struct.rootspecial[tagnum]->defaultconstructor;

       if(defaultconstructor) {
          G__cpp_aryconstruct = n;
          G__setgvp((long)arena);
          long ltagnum = tagnum;
          G__CurrentCall(G__DELETEFREE, this, &ltagnum);
          (*defaultconstructor)(&buf,(char*)NULL,para,0);
          G__CurrentCall(G__NOP, 0, 0);
          G__setgvp((long)G__PVOID);
          G__cpp_aryconstruct = 0;
          p = (void*)G__int(buf);
          // Record that we have allocated an array, and how many
          // elements that array has, for use by the G__calldtor function.
          G__alloc_newarraylist((long) p, n);
       }
#ifdef G__NOSTUBS
       else{
          /* If we have a function pointer */
          G__ifunc_table_internal* internalifuncconst 
             = G__get_ifunc_internal(G__struct.rootspecial[tagnum]->defaultconstructorifunc);
          if(internalifuncconst->funcptr[0]){
             
             G__cpp_aryconstruct = n;
             G__setgvp((long)arena);
             G__CurrentCall(G__DELETEFREE, this, &tagnum);
             G__stub_method_calling(&buf, para, internalifuncconst, 0);
             G__CurrentCall(G__NOP, 0, 0);
             G__setgvp((long)G__PVOID);
             G__cpp_aryconstruct = 0;
             p = (void*)G__int(buf);
             // Record that we have allocated an array, and how many
             // elements that array has, for use by the G__calldtor function.
             G__alloc_newarraylist((long) p, n);
          }
       }
#endif
       delete para;
    }
    else if(class_property&G__BIT_ISCCOMPILED) {
      // C precompiled class,struct
      p = arena;
    }
    else {
      // Interpreted class,struct
      long store_struct_offset;
      long store_tagnum;
      int known=0;
      p = arena;
      // Record that we have allocated an array, and how many
      // elements that array has, for use by the delete[] operator.
      G__alloc_newarraylist((long) p, n);
      store_tagnum = G__tagnum;
      store_struct_offset = G__store_struct_offset;
      G__tagnum = tagnum;
      G__store_struct_offset = (long) p;
      //// Do it this way for an array cookie implementation.
      ////p = arena;
      ////int* pp = (int*) p;
      ////pp[0] = G__struct.size[tagnum];
      ////pp[1] = n;
      ////G__store_struct_offset = (long)(((char*)p) + (2*sizeof(int)));
      ////... at end adjust returned pointer address ...
      ////p = ((char*) p) + (2 * sizeof(int));
      G__FastAllocString temp(G__struct.name[tagnum]);
      temp += "()";
      for (int i = 0; i < n; ++i) {
        G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
        if (!known) break;
	G__store_struct_offset += G__struct.size[tagnum];
      }
      G__store_struct_offset = store_struct_offset;
      G__tagnum = (int)store_tagnum;
    }
    return(p);
  }
  else {
    return((void*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::Delete(void* p) const { G__calldtor(p,tagnum,1); }
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::Destruct(void* p) const { G__calldtor(p,tagnum,0); }
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::DeleteArray(void* ary, int dtorOnly)
{
  // Array Destruction, with optional deletion.
  if (!IsValid()) return;
  if (!class_property) {
    Property();
  }
  if (class_property & G__BIT_ISCPPCOMPILED) {
    // C++ precompiled class,struct
    // Fetch the number of elements in the array that
    // we saved when we originally allocated it.
    G__cpp_aryconstruct = G__free_newarraylist((long) ary);
    if (dtorOnly) {
      Destruct(ary);
    } else {
      Delete(ary);
    }
    G__cpp_aryconstruct = 0;
  }
  else if (class_property & G__BIT_ISCCOMPILED) {
    // C precompiled class,struct
    if (!dtorOnly) {
      free(ary);
    }
  }
  else {
    // Interpreted class,struct
    // Fetch the number of elements in the array that
    // we saved when we originally allocated it.
    int n = G__free_newarraylist((long) ary);
    int element_size = G__struct.size[tagnum];
    //// Do it this way for an array cookie implementation.
    ////int* pp = (int*) ary;
    ////int n = pp[-1];
    ////int element_size = pp[-2];
    char* r = ((char*) ary) + ((n - 1) * element_size) ;
    ////int status = 0;
    for (int i = n; i > 0; --i) {
      /*status =*/ G__calldtor(r, tagnum, 0);
      // ???FIX ME:  What does status mean here?
      // if (!status) break;
      r -= element_size;
    }
    if (!dtorOnly) {
      free(ary);
    }
  }
  return;
}
///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::CheckValidRootInfo()
{
  long offset;
  if(G__struct.rootspecial[tagnum]) return;
  
  G__struct.rootspecial[tagnum]
    =(struct G__RootSpecial*)malloc(sizeof(struct G__RootSpecial));
  G__struct.rootspecial[tagnum]->deffile=(char*)NULL;
  G__struct.rootspecial[tagnum]->impfile=(char*)NULL;
  G__struct.rootspecial[tagnum]->defline=0;
  G__struct.rootspecial[tagnum]->impline=0;
  G__struct.rootspecial[tagnum]->version=0;
  G__struct.rootspecial[tagnum]->instancecount=0;
  G__struct.rootspecial[tagnum]->heapinstancecount=0;
  G__struct.rootspecial[tagnum]->defaultconstructor
    = (void*)GetInterfaceMethod(G__struct.name[tagnum],"",&offset);
  long dmy;
  G__MethodInfo method = GetMethod(G__struct.name[tagnum],"",&dmy,ExactMatch,InThisScope);

  G__ifunc_table* constructorIfunc = method.ifunc();
  if (G__get_funcptr(G__get_ifunc_internal(constructorIfunc),0))
     G__struct.rootspecial[tagnum]->defaultconstructorifunc = constructorIfunc;
  else
     G__struct.rootspecial[tagnum]->defaultconstructorifunc = 0;           

  
}
///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
static long G__ClassInfo_MemberFunctionProperty(long& property,int tagnum)
{
  struct G__ifunc_table_internal *ifunc;
  int ifn;
  ifunc = G__struct.memfunc[tagnum];
  while(ifunc) {
    for(ifn=0;ifn<ifunc->allifunc;ifn++) {
      if(strcmp(G__struct.name[tagnum],ifunc->funcname[ifn])==0) {
	/* explicit constructor */
	property |= G__CLS_HASEXPLICITCTOR;
	if(0==ifunc->para_nu[ifn] || ifunc->param[ifn][0]->pdefault) {
	  property |= G__CLS_HASDEFAULTCTOR;
	}
      }
      else if('~'==ifunc->funcname[ifn][0] &&
	      strcmp(G__struct.name[tagnum],ifunc->funcname[ifn]+1)==0) {
	/* explicit destructor */
	property |= G__CLS_HASEXPLICITDTOR;
      }
      else if(strcmp("operator=",ifunc->funcname[ifn])==0 
	      /* && (G__struct.funcs[ifn]&G__HAS_ASSIGNMENTOPERATOR) */ ) {
	// TODO, above condition has to be refined.
	property |= G__CLS_HASASSIGNOPR;
      }
      if(ifunc->isvirtual[ifn]) {
	property|=G__CLS_HASVIRTUAL;
	if((property&G__CLS_HASEXPLICITCTOR)==0) 
	  property |= G__CLS_HASIMPLICITCTOR;
      }
    }
    ifunc=ifunc->next;
  }
  return property;
}
///////////////////////////////////////////////////////////////////////////
static long G__ClassInfo_BaseClassProperty(long& property
					   ,G__ClassInfo& classinfo)
{
  G__BaseClassInfo baseinfo(classinfo);
  while(baseinfo.Next()) {
    long baseprop = baseinfo.ClassProperty();
    if(0==(property&G__CLS_HASEXPLICITCTOR) && (baseprop&G__CLS_HASCTOR))
      property |= (G__CLS_HASIMPLICITCTOR|G__CLS_HASDEFAULTCTOR);
    if(0==(property&G__CLS_HASEXPLICITDTOR) && (baseprop&G__CLS_HASDTOR))
      property |= G__CLS_HASIMPLICITDTOR;
    if(baseprop&G__CLS_HASVIRTUAL) property |= G__CLS_HASVIRTUAL;
  }
  return property;
}
///////////////////////////////////////////////////////////////////////////
static long G__ClassInfo_DataMemberProperty(long& property,int tagnum)
{
  struct G__var_array *var;
  int ig15;
  var = G__struct.memvar[tagnum];
  while(var) {
    for(ig15=0;ig15<var->allvar;ig15++) {
      if('u'==var->type[ig15] && G__PARANORMAL==var->reftype[ig15]) {
	G__ClassInfo classinfo(G__struct.name[var->p_tagtable[ig15]]);
	long baseprop = classinfo.ClassProperty();
	if(0==(property&G__CLS_HASEXPLICITCTOR) && (baseprop&G__CLS_HASCTOR))
	  property |= (G__CLS_HASIMPLICITCTOR|G__CLS_HASDEFAULTCTOR);
	if(0==(property&G__CLS_HASEXPLICITDTOR) && (baseprop&G__CLS_HASDTOR))
	  property |= G__CLS_HASIMPLICITDTOR;
      }
    }
    var=var->next;
  }
  return property;
}
///////////////////////////////////////////////////////////////////////////
long Cint::G__ClassInfo::ClassProperty()
{
  long property=0;
  if(IsValid()) {
    switch(G__struct.type[tagnum]) {
    case 'e': 
    case 'u': 
      return(property);
    case 'c': 
    case 's': 
      property |= G__CLS_VALID;
    }
    if(G__struct.isabstract[tagnum]) property |= G__CLS_ISABSTRACT;
    G__ClassInfo_MemberFunctionProperty(property,(int)tagnum);
    G__ClassInfo_BaseClassProperty(property,*this);
    G__ClassInfo_DataMemberProperty(property,(int)tagnum);
    return(property);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
struct G__friendtag* Cint::G__ClassInfo::GetFriendInfo() { 
  if(IsValid()) return(G__struct.friendtag[tagnum]);
  else return 0;
}
///////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
G__MethodInfo Cint::G__ClassInfo::AddMethod(const char* typenam,const char* fname
                                            ,const char *arg
                                            ,int isstatic,int isvirtual,
                                            void *methodAddress) {
  struct G__ifunc_table_internal *ifunc;
  long index;

  if(-1==tagnum) ifunc = &G__ifunc;
  else           ifunc = G__struct.memfunc[tagnum];

  //////////////////////////////////////////////////
  // Add a method entry
  while(ifunc->next) ifunc=ifunc->next;
  index = ifunc->allifunc;
  if(ifunc->allifunc==G__MAXIFUNC) {
    ifunc->next=(struct G__ifunc_table_internal *)malloc(sizeof(struct G__ifunc_table_internal));
    memset(ifunc->next,0,sizeof(struct G__ifunc_table_internal));
    ifunc->next->allifunc=0;
    ifunc->next->next=(struct G__ifunc_table_internal *)NULL;
    ifunc->next->page = ifunc->page+1;
    ifunc->next->tagnum = ifunc->tagnum;
    ifunc = ifunc->next;
    for(int ix=0;ix<G__MAXIFUNC;ix++) {
      ifunc->funcname[ix] = (char*)NULL;
      ifunc->userparam[ix] = 0;
    }
    index=0;
  }

  //////////////////////////////////////////////////
  // save function name 
  G__savestring(&ifunc->funcname[index],(char*)fname);
  int tmp;
  G__hash(ifunc->funcname[index],ifunc->hash[index],tmp);
  ifunc->param[index][0]->name=(char*)NULL;

  //////////////////////////////////////////////////
  // save type information
  G__TypeInfo type(typenam);
  ifunc->type[index] = type.Type();
  ifunc->p_typetable[index] = type.Typenum();
  ifunc->p_tagtable[index] = (short)type.Tagnum();
  ifunc->reftype[index] =   type.Reftype();
  ifunc->isconst[index] =   type.Isconst();

  // flags
  ifunc->staticalloc[index] = isstatic;
  ifunc->isvirtual[index] = isvirtual;
  ifunc->ispurevirtual[index] = 0;
  ifunc->access[index] = G__PUBLIC;

  // miscellaneous flags
  ifunc->isexplicit[index] = 0;
  ifunc->iscpp[index] = 1;
  ifunc->ansi[index] = 1;
  ifunc->busy[index] = 0;
  ifunc->friendtag[index] = (struct G__friendtag*)NULL;
  ifunc->globalcomp[index] = G__NOLINK;
  ifunc->comment[index].p.com = (char*)NULL;
  ifunc->comment[index].filenum = -1;


  ifunc->userparam[index] = (void*)NULL;
  ifunc->vtblindex[index] = -1;
  ifunc->vtblbasetagnum[index] = -1;

  //////////////////////////////////////////////////
  // set argument infomation
  if (strcmp(arg,"ellipsis")==0) {
     // mimick ellipsis
     ifunc->para_nu[index] = -1;
     ifunc->ansi[index] = 0;
  } else {
     char *argtype = (char*)arg;
     struct G__param* para = new G__param();
     G__argtype2param(argtype,para,0,0);

     ifunc->para_nu[index] = para->paran;
     for(int i=0;i<para->paran;i++) {
        ifunc->param[index][i]->type = para->para[i].type;
        if(para->para[i].type!='d' && para->para[i].type!='f') 
           ifunc->param[index][i]->reftype = para->para[i].obj.reftype.reftype;
        else 
           ifunc->param[index][i]->reftype = G__PARANORMAL;
        ifunc->param[index][i]->p_tagtable = para->para[i].tagnum;
        ifunc->param[index][i]->p_typetable = para->para[i].typenum;
        ifunc->param[index][i]->name = (char*)malloc(20);
        snprintf(ifunc->param[index][i]->name,20,"G__p%d",i);
        ifunc->param[index][i]->pdefault = (G__value*)NULL;
        ifunc->param[index][i]->def = (char*)NULL;
     }
     delete para;
  }

  //////////////////////////////////////////////////
  ifunc->pentry[index] = &ifunc->entry[index];
  //ifunc->entry[index].pos
  if (methodAddress) {
     ifunc->entry[index].p = methodAddress;
     ifunc->entry[index].line_number = -1;
     ifunc->entry[index].filenum = -1;
     ifunc->entry[index].size = -1;
     ifunc->entry[index].tp2f = (char*)NULL;
     ifunc->entry[index].bytecode = 0;
     ifunc->entry[index].bytecodestatus = G__BYTECODE_NOTYET;
  } else {
     if (tagnum > -1) {
        ifunc->entry[index].p = G__srcfile[G__struct.filenum[tagnum]].fp;
        ifunc->entry[index].line_number=(-1==tagnum)?0:G__struct.line_number[tagnum];
        ifunc->entry[index].filenum=(-1==tagnum)?0:G__struct.filenum[tagnum];
     }
     ifunc->entry[index].size = 1;
     ifunc->entry[index].tp2f = (char*)NULL;
     ifunc->entry[index].bytecode = 0;
     ifunc->entry[index].bytecodestatus = G__BYTECODE_NOTYET;
  }

  //////////////////////////////////////////////////
  ++ifunc->allifunc;

  /* Initialize method object */
  G__MethodInfo method;
  method.Init((long)G__get_ifunc_ref(ifunc),index,this);
  return(method);
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::GetNumClasses()
{
   // Return the number of classes (autoload entries etc) known to the system
   return G__struct.alltag;
}
