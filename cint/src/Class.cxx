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
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#include "Api.h"
#include "common.h"

#ifndef G__OLDIMPLEMENTATION1586
static char G__buf[G__ONELINE];
#endif

/*********************************************************************
* class G__ClassInfo
*********************************************************************/

///////////////////////////////////////////////////////////////////////////
void G__ClassInfo::Init()
{
  tagnum = -1;
#ifndef G__OLDIMPLEMENTATION1218
  class_property = 0;
#endif
}
///////////////////////////////////////////////////////////////////////////
void G__ClassInfo::Init(const char *classname)
{
#ifndef G__OLDIMPLEMENTATION770
  tagnum = G__defined_tagname(classname,1);
#else
  tagnum = G__defined_tagname(classname,2);
#endif
#ifndef G__OLDIMPLEMENTATION1218
  class_property = 0;
#endif
}
///////////////////////////////////////////////////////////////////////////
void G__ClassInfo::Init(int tagnumin)
{
  tagnum = tagnumin;
#ifndef G__OLDIMPLEMENTATION1218
  class_property = 0;
#endif
}
///////////////////////////////////////////////////////////////////////////
int G__ClassInfo::operator==(const G__ClassInfo& a)
{
  return(tagnum == a.tagnum);
}
///////////////////////////////////////////////////////////////////////////
int G__ClassInfo::operator!=(const G__ClassInfo& a)
{
  return(tagnum != a.tagnum);
}
///////////////////////////////////////////////////////////////////////////
const char* G__ClassInfo::Name()
{
  if(IsValid()) {
    return(G__struct.name[tagnum]);
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
const char* G__ClassInfo::Fullname()
{
  if(IsValid()) {
#ifndef G__OLDIMPLEMENTATION1586
    strcpy(G__buf,G__fulltagname((int)tagnum,1));
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
const char* G__ClassInfo::Title() 
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
int G__ClassInfo::Size() 
{
  if(IsValid()) {
    return(G__struct.size[tagnum]);
  }
  else {
    return(-1);
  }
}
///////////////////////////////////////////////////////////////////////////
long G__ClassInfo::Property()
{
#ifndef G__OLDIMPLEMENTATION1218
  if (class_property) return class_property;
#else
  long property=0;
#endif
  if(IsValid()) {
#ifndef G__OLDIMPLEMENTATION1218
    long property=0;
#endif
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
#ifndef G__OLDIMPLEMENTATION1218
    class_property = property;
#endif
    return(property);
  }
  else {
    return(0);
  }
}
///////////////////////////////////////////////////////////////////////////
int G__ClassInfo::NDataMembers()
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
int G__ClassInfo::NMethods()
{
  struct G__ifunc_table *ifunc;
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
long G__ClassInfo::IsBase(const char *classname)
{
  G__ClassInfo base(classname);
  return(IsBase(base));
}
///////////////////////////////////////////////////////////////////////////
long G__ClassInfo::IsBase(G__ClassInfo& a)
{
  G__inheritance *baseclass;
  int i;
  long isbase=0;
  if(IsValid()) {
    baseclass = G__struct.baseclass[tagnum];
    for(i=0;i<baseclass->basen;i++) {
      if(a.Tagnum() == baseclass->basetagnum[i]) {
	switch(baseclass->baseaccess[i]) {
	case G__PUBLIC: isbase = G__BIT_ISPUBLIC; break;
	case G__PROTECTED: isbase = G__BIT_ISPROTECTED; break;
	case G__PRIVATE: isbase = G__BIT_ISPRIVATE; break;
	default: isbase = 0; break;
	}
	if(baseclass->property[i]&G__ISDIRECTINHERIT) 
	  isbase |= G__BIT_ISDIRECTINHERIT;
	if(baseclass->property[i]&G__ISVIRTUALBASE) 
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
G__ClassInfo G__ClassInfo::EnclosingClass()
{
  if(IsValid()) {
    G__ClassInfo enclosingclass(G__struct.parent_tagnum[tagnum]);
    return(enclosingclass);
  }
  else {
    G__ClassInfo enclosingclass;
    return(enclosingclass);
  }
}
///////////////////////////////////////////////////////////////////////////
G__ClassInfo G__ClassInfo::EnclosingSpace()
{
  if(IsValid()) {
    int enclosed_tag = G__struct.parent_tagnum[tagnum];
    while (enclosed_tag>=0 && (G__struct.type[enclosed_tag]!='n')) {
       enclosed_tag = G__struct.parent_tagnum[enclosed_tag];
    }
    G__ClassInfo enclosingclass(enclosed_tag);
    return(enclosingclass);
  }
  else {
    G__ClassInfo enclosingclass;
    return(enclosingclass);
  }
}
///////////////////////////////////////////////////////////////////////////
void G__ClassInfo::SetGlobalcomp(int globalcomp)
{
  if(IsValid()) {
    G__struct.globalcomp[tagnum] = globalcomp;
  }
}
#ifndef G__OLDIMPLEMENTATION1334
///////////////////////////////////////////////////////////////////////////
void G__ClassInfo::SetProtectedAccess(int protectedaccess)
{
  if(IsValid()) {
    G__struct.protectedaccess[tagnum] = protectedaccess;
  }
}
#endif
///////////////////////////////////////////////////////////////////////////
#ifndef G__OLDIMPLEMENTATION1218_YET
int G__ClassInfo::IsValid()
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
#ifndef G__OLDIMPLEMENTATION2118
unsigned char G__ClassInfo::FuncFlag() { 
  return(IsValid()?G__struct.funcs[tagnum]:0); 
}
#endif
///////////////////////////////////////////////////////////////////////////
int G__ClassInfo::IsLoaded()
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
int G__ClassInfo::SetFilePos(const char *fname)
{
  struct G__dictposition* dict=G__get_dictpos((char*)fname);
  if(!dict) return(0);
  tagnum=dict->tagnum-1;
#ifndef G__OLDIMPLEMENTATION1218
  class_property = 0;
#endif
  return(1);
}
///////////////////////////////////////////////////////////////////////////
int G__ClassInfo::Next()
{
  ++tagnum;
#ifndef G__OLDIMPLEMENTATION1218
  class_property = 0;
#endif
  return(IsValid());
}
///////////////////////////////////////////////////////////////////////////
int G__ClassInfo::Linkage()
{
  return G__struct.globalcomp[tagnum];
}
///////////////////////////////////////////////////////////////////////////
const char* G__ClassInfo::FileName()
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
int G__ClassInfo::LineNumber()
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
int G__ClassInfo::IsTmplt()
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
const char* G__ClassInfo::TmpltName()
{
  static char buf[G__ONELINE];
  if(IsValid()) {
    char *p;
    strcpy(buf,Name());
    p = strchr(buf,'<');
    if(p) *p = 0;
    return(buf);
  }
  else {
    return((char*)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
const char* G__ClassInfo::TmpltArg()
{
  static char buf[G__ONELINE];
  if(IsValid()) {
    char *p = strchr((char*)Name(),'<');
    if(p) {
      strcpy(buf,p+1);
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

#ifdef G__ROOTSPECIAL
/*********************************************************************
* ROOT project special requirements
*********************************************************************/
///////////////////////////////////////////////////////////////////////////
void G__ClassInfo::SetDefFile(char *deffilein)
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->deffile = deffilein;
  }
}
///////////////////////////////////////////////////////////////////////////
void G__ClassInfo::SetDefLine(int deflinein)
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->defline = deflinein;
  }
}
///////////////////////////////////////////////////////////////////////////
void G__ClassInfo::SetImpFile(char *impfilein)
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->impfile = impfilein;
  }
}
///////////////////////////////////////////////////////////////////////////
void G__ClassInfo::SetImpLine(int implinein)
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->impline = implinein;
  }
}
///////////////////////////////////////////////////////////////////////////
void G__ClassInfo::SetVersion(int versionin)
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->version = versionin;
  }
}
///////////////////////////////////////////////////////////////////////////
const char* G__ClassInfo::DefFile()
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
int G__ClassInfo::DefLine()
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
const char* G__ClassInfo::ImpFile()
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
int G__ClassInfo::ImpLine()
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
int G__ClassInfo::Version()
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
int G__ClassInfo::InstanceCount() 
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
void G__ClassInfo::ResetInstanceCount()
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->instancecount = 0;
  }
}
///////////////////////////////////////////////////////////////////////////
void G__ClassInfo::IncInstanceCount()
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->instancecount += 1;
  }
}
///////////////////////////////////////////////////////////////////////////
int G__ClassInfo::HeapInstanceCount() 
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
void G__ClassInfo::ResetHeapInstanceCount()
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->heapinstancecount = 0;
  }
}
///////////////////////////////////////////////////////////////////////////
void G__ClassInfo::IncHeapInstanceCount()
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->heapinstancecount += 1;
  }
}
///////////////////////////////////////////////////////////////////////////
int G__ClassInfo::RootFlag()
{
  return G__struct.rootflag[tagnum];
}
///////////////////////////////////////////////////////////////////////////
G__InterfaceMethod G__ClassInfo::GetInterfaceMethod(const char* fname
						    ,const char* arg
						    ,long* poffset
						    ,MatchMode mode
                                                    ,InheritanceMode imode
						)
{
  struct G__ifunc_table *ifunc;
  char *funcname;
  char *param;
  long index;

  /* Search for method */
  if(-1==tagnum) ifunc = &G__ifunc;
  else           ifunc = G__struct.memfunc[tagnum];
  funcname = (char*)fname;
  param = (char*)arg;
  ifunc = G__get_methodhandle(funcname,param,ifunc,&index,poffset
#ifndef G__OLDIMPLEMENTATION1989
			      ,(mode==ConversionMatch)?1:0
#endif
                              ,imode
			      );

  if(
#ifndef G__OLDIMPLEMENTATION2035
     ifunc && -1==ifunc->pentry[index]->size
#else
     ifunc && -1==ifunc->pentry[index]->filenum
#endif
     ) {
    return((G__InterfaceMethod)ifunc->pentry[index]->p);
  }
  else {
    return((G__InterfaceMethod)NULL);
  }
}
///////////////////////////////////////////////////////////////////////////
G__MethodInfo G__ClassInfo::GetMethod(const char* fname,const char* arg
				      ,long* poffset
				      ,MatchMode mode
				      ,InheritanceMode imode
				      )
{
  struct G__ifunc_table *ifunc;
  char *funcname;
  char *param;
  long index;

  /* Search for method */
  if(-1==tagnum) ifunc = &G__ifunc;
  else           ifunc = G__struct.memfunc[tagnum];
  funcname = (char*)fname;
  param = (char*)arg;
#ifndef G__OLDIMPLEMENTATION2177
  int convmode;
  switch(mode) {
  case ExactMatch:              convmode=0; break;
  case ConversionMatch:         convmode=1; break;
  case ConversionMatchBytecode: convmode=2; break;
  default:                      convmode=0; break;
  }
#else
  int convmode = (mode==ConversionMatch)?1:0;
#endif
  ifunc = G__get_methodhandle(funcname,param,ifunc,&index,poffset
#if !defined(G__OLDIMPLEMENTATION2177)
			      ,convmode
#elif !defined(G__OLDIMPLEMENTATION1989)
			      ,(mode==ConversionMatch)?1:0
#endif
			      ,(imode==WithInheritance)?1:0
			      );

  /* Initialize method object */
  G__MethodInfo method;
  method.Init((long)ifunc,index,this);
  return(method);
}
///////////////////////////////////////////////////////////////////////////
G__MethodInfo G__ClassInfo::GetMethod(const char* fname,struct G__param* libp
				      ,long* poffset
				      ,MatchMode mode
				      ,InheritanceMode imode
                                      )
{
  struct G__ifunc_table *ifunc;
  char *funcname = (char*)fname;
  long index;

  /* Search for method */
  if(-1==tagnum) ifunc = &G__ifunc;
  else           ifunc = G__struct.memfunc[tagnum];

  ifunc = G__get_methodhandle2(funcname,libp,ifunc,&index,poffset
			       ,(mode==ConversionMatch)?1:0
			       ,(imode==WithInheritance)?1:0
                               );

  /* Initialize method object */
  G__MethodInfo method;
  method.Init((long)ifunc,index,this);
  return(method);
}
#ifndef G__OLDIMPLEMENTATION2059
///////////////////////////////////////////////////////////////////////////
G__MethodInfo G__ClassInfo::GetDefaultConstructor() {
  // TODO, reserve location for default ctor for tune up
  long dmy;
  G__MethodInfo method;
  char *fname= (char*)malloc(strlen(Name())+1);
  sprintf(fname,"%s",Name());
  method = GetMethod(fname,"",&dmy,ExactMatch,InThisScope);
  free((void*)fname);
  return(method);
}
///////////////////////////////////////////////////////////////////////////
G__MethodInfo G__ClassInfo::GetCopyConstructor() {
  // TODO, reserve location for copy ctor for tune up
  long dmy;
  G__MethodInfo method;
  char *fname= (char*)malloc(strlen(Name())+1);
  sprintf(fname,"%s",Name());
  char *arg= (char*)malloc(strlen(Name())+10);
  sprintf(arg,"const %s&",Name());
  method = GetMethod(fname,arg,&dmy,ExactMatch,InThisScope);
  free((void*)arg);
  free((void*)fname);
  return(method);
}
///////////////////////////////////////////////////////////////////////////
G__MethodInfo G__ClassInfo::GetDestructor() {
  // TODO, dtor location is already reserved, ready for tune up
  long dmy;
  G__MethodInfo method;
  char *fname= (char*)malloc(strlen(Name())+2);
  sprintf(fname,"~%s",Name());
  method = GetMethod(fname,"",&dmy,ExactMatch,InThisScope);
  free((void*)fname);
  return(method);
}
///////////////////////////////////////////////////////////////////////////
G__MethodInfo G__ClassInfo::GetAssignOperator() {
  // TODO, reserve operator= location for tune up
  long dmy;
  G__MethodInfo method;
  char *arg= (char*)malloc(strlen(Name())+10);
  sprintf(arg,"const %s&",Name());
  method = GetMethod("operator=",arg,&dmy,ExactMatch,InThisScope);
  free((void*)arg);
  return(method);
}
#endif
///////////////////////////////////////////////////////////////////////////
G__DataMemberInfo G__ClassInfo::GetDataMember(const char* name,long* poffset)
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
int G__ClassInfo::HasDefaultConstructor()
{
  if(IsValid()) {
     CheckValidRootInfo();
     return(G__struct.rootspecial[tagnum]->defaultconstructor!=0);
  } else {
     return 0;
  }
}
///////////////////////////////////////////////////////////////////////////
int G__ClassInfo::HasMethod(const char *fname)
{
  struct G__ifunc_table *ifunc;
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
int G__ClassInfo::HasDataMember(const char *name)
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
void* G__ClassInfo::New()
{
  if(IsValid()) {
#ifdef G__OLDIMPLEMENTATION1218
    long property;
#endif
    void *p;
    G__value buf=G__null;
#ifndef G__OLDIMPLEMENTATION1218
    if (!class_property) Property();
    if(class_property&G__BIT_ISCPPCOMPILED) {
#else
    property = Property();
    if(property&G__BIT_ISCPPCOMPILED) {
#endif
      // C++ precompiled class,struct
      struct G__param para;
      G__InterfaceMethod defaultconstructor;
      para.paran=0;
#ifndef G__OLDIMPLEMENTATION1218
      if(!G__struct.rootspecial[tagnum]) CheckValidRootInfo();
#else
      CheckValidRootInfo();
#endif
      defaultconstructor
	=(G__InterfaceMethod)G__struct.rootspecial[tagnum]->defaultconstructor;
      if(defaultconstructor) {
#ifndef G__OLDIMPLEMENTATION1749
	G__CurrentCall(G__DELETEFREE, this, tagnum);
	(*defaultconstructor)(&buf,(char*)NULL,&para,0);
	G__CurrentCall(G__NOP, 0, 0);
#else
	(*defaultconstructor)(&buf,(char*)NULL,&para,0);
#endif
	p = (void*)G__int(buf);
      }
      else {
	p = (void*)NULL;
      }
    }
#ifndef G__OLDIMPLEMENTATION1218
    else if(class_property&G__BIT_ISCCOMPILED) {
#else
    else if(property&G__BIT_ISCCOMPILED) {
#endif
      // C precompiled class,struct
      p = malloc(G__struct.size[tagnum]);
    }
    else {
      // Interpreted class,struct
      long store_struct_offset;
      long store_tagnum;
      char temp[G__ONELINE];
      int known=0;
      p = malloc(G__struct.size[tagnum]);
      store_tagnum = G__tagnum;
      store_struct_offset = G__store_struct_offset;
#ifndef G__PHILIPPE16
      G__tagnum = tagnum;
#endif
      G__store_struct_offset = (long)p;
      sprintf(temp,"%s()",G__struct.name[tagnum]);
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
void* G__ClassInfo::New(int n)
{
  if(IsValid() && n>0 ) {
#ifdef G__OLDIMPLEMENTATION1218
    long property;
#endif
    void *p;
    G__value buf=G__null;
#ifndef G__OLDIMPLEMENTATION1218
    if (!class_property) Property();
    if(class_property&G__BIT_ISCPPCOMPILED) {
#else
    property = Property();
    if(property&G__BIT_ISCPPCOMPILED) {
#endif
      // C++ precompiled class,struct
      struct G__param para;
      G__InterfaceMethod defaultconstructor;
      para.paran=0;
#ifndef G__OLDIMPLEMENTATION1218
      if(!G__struct.rootspecial[tagnum]) CheckValidRootInfo();
#else
      CheckValidRootInfo();
#endif
      defaultconstructor
	=(G__InterfaceMethod)G__struct.rootspecial[tagnum]->defaultconstructor;
      if(defaultconstructor) {
	if(n) G__cpp_aryconstruct = n;
#ifndef G__OLDIMPLEMENTATION1749
	G__CurrentCall(G__DELETEFREE, this, tagnum);
	(*defaultconstructor)(&buf,(char*)NULL,&para,0);
	G__CurrentCall(G__NOP, 0, 0);
#else
	(*defaultconstructor)(&buf,(char*)NULL,&para,0);
#endif
	G__cpp_aryconstruct = 0;
	p = (void*)G__int(buf);
      }
      else {
	p = (void*)NULL;
      }
    }
#ifndef G__OLDIMPLEMENTATION1218
    else if(class_property&G__BIT_ISCCOMPILED) {
#else
    else if(property&G__BIT_ISCCOMPILED) {
#endif
      // C precompiled class,struct
      p = malloc(G__struct.size[tagnum]*n);
    }
    else {
      // Interpreted class,struct
      int i;
      long store_struct_offset;
      long store_tagnum;
      char temp[G__ONELINE];
      int known=0;
      p = malloc(G__struct.size[tagnum]*n);
      store_tagnum = G__tagnum;
      store_struct_offset = G__store_struct_offset;
#ifndef G__PHILIPPE16
      G__tagnum = tagnum;
#endif
      G__store_struct_offset = (long)p;
      sprintf(temp,"%s()",G__struct.name[tagnum]);
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
void* G__ClassInfo::New(void *arena)
{
  if(IsValid()) {
#ifdef G__OLDIMPLEMENTATION1218
    long property;
#endif
    void *p;
    G__value buf=G__null;
#ifndef G__OLDIMPLEMENTATION1218
    if (!class_property) Property();
    if(class_property&G__BIT_ISCPPCOMPILED) {
#else
    property = Property();
    if(property&G__BIT_ISCPPCOMPILED) {
#endif
      // C++ precompiled class,struct
      struct G__param para;
      G__InterfaceMethod defaultconstructor;
      para.paran=0;
#ifndef G__OLDIMPLEMENTATION1218
      if(!G__struct.rootspecial[tagnum]) CheckValidRootInfo();
#else
      CheckValidRootInfo();
#endif
      defaultconstructor
	=(G__InterfaceMethod)G__struct.rootspecial[tagnum]->defaultconstructor;
      if(defaultconstructor) {
	G__setgvp((long)arena);
#ifndef G__OLDIMPLEMENTATION1749
	G__CurrentCall(G__DELETEFREE, this, tagnum);
	(*defaultconstructor)(&buf,(char*)NULL,&para,0);
	G__CurrentCall(G__NOP, 0, 0);
#else
	(*defaultconstructor)(&buf,(char*)NULL,&para,0);
#endif
	G__setgvp((long)G__PVOID);
	p = (void*)G__int(buf);
      }
      else {
	p = (void*)NULL;
      }
    }
#ifndef G__OLDIMPLEMENTATION1218
    else if(class_property&G__BIT_ISCCOMPILED) {
#else
    else if(property&G__BIT_ISCCOMPILED) {
#endif
      // C precompiled class,struct
      p = arena;
    }
    else {
      // Interpreted class,struct
      long store_struct_offset;
      long store_tagnum;
      char temp[G__ONELINE];
      int known=0;
      p = arena;
      store_tagnum = G__tagnum;
      store_struct_offset = G__store_struct_offset;
#ifndef G__PHILIPPE16
      G__tagnum = tagnum;
#endif
      G__store_struct_offset = (long)p;
      sprintf(temp,"%s()",G__struct.name[tagnum]);
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
#ifndef G__OLDIMPLEMENTATION2043
///////////////////////////////////////////////////////////////////////////
void G__ClassInfo::Delete(void* p) const { G__calldtor(p,tagnum,1); }
///////////////////////////////////////////////////////////////////////////
void G__ClassInfo::Destruct(void* p) const { G__calldtor(p,tagnum,0); }
#endif
///////////////////////////////////////////////////////////////////////////
void G__ClassInfo::CheckValidRootInfo()
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
}
///////////////////////////////////////////////////////////////////////////
#endif /* ROOTSPECIAL */

#ifndef G__OLDIMPLEMENTATION644
///////////////////////////////////////////////////////////////////////////
static long G__ClassInfo_MemberFunctionProperty(long& property,int tagnum)
{
  struct G__ifunc_table *ifunc;
  int ifn;
  ifunc = G__struct.memfunc[tagnum];
  while(ifunc) {
    for(ifn=0;ifn<ifunc->allifunc;ifn++) {
      if(strcmp(G__struct.name[tagnum],ifunc->funcname[ifn])==0) {
	/* explicit constructor */
	property |= G__CLS_HASEXPLICITCTOR;
	if(0==ifunc->para_nu[ifn] || ifunc->para_default[ifn][0]) {
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
long G__ClassInfo::ClassProperty()
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
struct G__friendtag* G__ClassInfo::GetFriendInfo() { 
  if(IsValid()) return(G__struct.friendtag[tagnum]);
  else return 0;
}
///////////////////////////////////////////////////////////////////////////
#endif /* ON644 */

#ifndef G__OLDIMPLEMENTATION2076
///////////////////////////////////////////////////////////////////////////
G__MethodInfo G__ClassInfo::AddMethod(const char* typenam,const char* fname
				     ,const char *arg
                         	     ,int isstatic,int isvirtual) {
  struct G__ifunc_table *ifunc;
  long index;

  if(-1==tagnum) ifunc = &G__ifunc;
  else           ifunc = G__struct.memfunc[tagnum];

  //////////////////////////////////////////////////
  // Add a method entry
  while(ifunc->next) ifunc=ifunc->next;
  index = ifunc->allifunc;
  if(ifunc->allifunc==G__MAXIFUNC) {
    ifunc->next=(struct G__ifunc_table *)malloc(sizeof(struct G__ifunc_table));
    ifunc->next->allifunc=0;
    ifunc->next->next=(struct G__ifunc_table *)NULL;
    ifunc->next->page = ifunc->page+1;
    ifunc->next->tagnum = ifunc->tagnum;
    ifunc = ifunc->next;
    for(int ix=0;ix<G__MAXIFUNC;ix++) {
      ifunc->funcname[ix] = (char*)NULL;
      ifunc->userparam[ix] = 0;
#ifndef G__OLDIMPLEMENTATION1706
      ifunc->override_ifunc[ix] = (struct G__ifunc_table*)NULL;
      ifunc->override_ifn[ix] = 0;
      ifunc->masking_ifunc[ix] = (struct G__ifunc_table*)NULL;
      ifunc->masking_ifn[ix] = 0;
#endif
    }
    index=0;
  }

  //////////////////////////////////////////////////
  // save function name 
  G__savestring(&ifunc->funcname[index],(char*)fname);
  int tmp;
  G__hash(ifunc->funcname[index],ifunc->hash[index],tmp);
  ifunc->para_name[index][0]=(char*)NULL;

  //////////////////////////////////////////////////
  // save type information
  G__TypeInfo type(typenam);
  ifunc->type[index] =   type.Type();
  ifunc->p_typetable[index] =   type.Typenum();
  ifunc->p_tagtable[index] =   type.Tagnum();
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
#ifdef G__FONS_COMMENT
  ifunc->comment[index].p.com = (char*)NULL;
  ifunc->comment[index].filenum = -1;
#endif

#ifndef G__OLDIMPLEMENTATION1706
  ifunc->override_ifunc[index] = (struct G__ifunc_table*)NULL;
  ifunc->override_ifn[index] = 0;
  ifunc->masking_ifunc[index] = (struct G__ifunc_table*)NULL;
  ifunc->masking_ifn[index] = 0;
#endif

  ifunc->userparam[index] = (void*)NULL;
#ifndef G__OLDIMPLEMENTATION2073
  ifunc->vtblindex[index] = -1;
#endif
#ifndef G__OLDIMPLEMENTATION2084
  ifunc->vtblbasetagnum[index] = -1;
#endif

  //////////////////////////////////////////////////
  // set argument infomation
  char *argtype = (char*)arg;
  struct G__param para;
  G__argtype2param(argtype,&para);

  ifunc->para_nu[index] = para.paran;
  for(int i=0;i<para.paran;i++) {
    ifunc->para_type[index][i] = para.para[i].type;
    if(para.para[i].type!='d' && para.para[i].type!='f') 
      ifunc->para_reftype[index][i] = para.para[i].obj.reftype.reftype;
    else 
      ifunc->para_reftype[index][i] = G__PARANORMAL;
    ifunc->para_p_tagtable[index][i] = para.para[i].tagnum;
    ifunc->para_p_typetable[index][i] = para.para[i].typenum;
    ifunc->para_name[index][i] = (char*)malloc(10);
    sprintf(ifunc->para_name[index][i],"G__p%d",i);
    ifunc->para_default[index][i] = (G__value*)NULL;
    ifunc->para_def[index][i] = (char*)NULL;
  }

  //////////////////////////////////////////////////
  ifunc->pentry[index] = &ifunc->entry[index];
  //ifunc->entry[index].pos
  ifunc->entry[index].p = G__srcfile[G__struct.filenum[tagnum]].fp;
  ifunc->entry[index].line_number=(-1==tagnum)?0:G__struct.line_number[tagnum];
  ifunc->entry[index].filenum=(-1==tagnum)?0:G__struct.filenum[tagnum];
  ifunc->entry[index].size = 1;
  ifunc->entry[index].tp2f = (char*)NULL;
  ifunc->entry[index].bytecode = (struct G__bytecodefunc*)NULL;
  ifunc->entry[index].bytecodestatus = G__BYTECODE_NOTYET;

  //////////////////////////////////////////////////
  G__memfunc_next();

  /* Initialize method object */
  G__MethodInfo method;
  method.Init((long)ifunc,index,this);
  return(method);
}
#endif

///////////////////////////////////////////////////////////////////////////

