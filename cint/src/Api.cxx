/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file Api.cxx
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Api.h"
#include "common.h"

#include "bc_eh.h"


/*********************************************************************
* $xxx object resolution function Generic form
* There is another form of interface which only allow you to return
* pointer to a class object if you want to gain a slight speed 
* advantage.
*********************************************************************/
//
// Static object to store pointer to interface function
//
static G__pMethodSpecialValue  G__UserSpecificGetSpecialValue;

//
// API to Cint internal type information translater
//
extern "C" void G__TypeInfo2G__value(G__TypeInfo* type,G__value* pvalue
				     ,long l,double d)
{
  pvalue->tagnum=(int)type->Tagnum();
  pvalue->typenum=(int)type->Typenum();
  pvalue->type=(int)type->Type();
  pvalue->ref=0;
  switch(pvalue->type) {
  case 'd':
  case 'f':
    pvalue->obj.d = d;
    break;
  default:
    pvalue->obj.i = l;
    break;
  }
}

//
// Used directly from src/expr.c:G__getitem()
// 
extern "C" G__value G__APIGetSpecialValue_layer1(char *item,void **pptr
	,void **ppdict)
{
  G__value result;
  long l;
  double d;
  G__TypeInfo typeinfo;
  (*G__UserSpecificGetSpecialValue)(item,&typeinfo,&l,&d,pptr,ppdict);
  G__TypeInfo2G__value(&typeinfo,&result,l,d);
  return(result);
}

//
// Initialization routine
//
extern "C" void G__InitGetSpecialValue(G__pMethodSpecialValue pmethod) 
{
  G__GetSpecialObject
	=(G__value (*)(char*,void**,void**))G__APIGetSpecialValue_layer1;
  G__UserSpecificGetSpecialValue = pmethod;
}


/*********************************************************************
* $xxx object resolution function for ROOT, slight speed advantage.
* Following interface library can only return pointer to a class object
*********************************************************************/
//
// Static object to store pointer to interface functionb
//
static G__pMethodSpecialObject G__UserSpecificGetSpecialObject;


//
// API to Cint internal type information translater
//
extern "C" void G__ClassInfo2G__value(G__ClassInfo* type
				     ,G__value* pvalue,long l)
{
  pvalue->tagnum=(int)type->Tagnum();
  pvalue->typenum = -1;
  pvalue->type = 'U';
  pvalue->ref=0;
  pvalue->obj.i = l;
}

//
// Used directly from src/expr.c:G__getitem()
// 
extern "C" G__value G__APIGetSpecialObject_layer1(char *item,void** pptr
	,void** ppdict)
{
  G__value result;
  long l;
  G__ClassInfo type;
  int store_prerun = G__prerun;
  G__prerun = 0;
  l=(long)((*G__UserSpecificGetSpecialObject)(item,&type,pptr,ppdict));
  G__prerun = store_prerun;
  G__ClassInfo2G__value(&type,&result,l);
  return(result);
}

//
// Initialization routine
//
extern "C" void G__InitGetSpecialObject(G__pMethodSpecialObject pmethod) 
{
  G__LockCriticalSection();
  G__GetSpecialObject
	=(G__value (*)(char*,void**,void**))G__APIGetSpecialObject_layer1;
  G__UserSpecificGetSpecialObject = pmethod;
  G__UnlockCriticalSection();
}

/*********************************************************************
* Feedback routine in case tagnum for a class changes (in case the
* dictionary of a shared lib is being re-initialized).
*********************************************************************/
//
// Static object to store pointer to interface function (defined in newlink.c)
//
extern "C" G__pMethodUpdateClassInfo G__UserSpecificUpdateClassInfo;

extern "C" void G__InitUpdateClassInfo(G__pMethodUpdateClassInfo pmethod)
{
   G__UserSpecificUpdateClassInfo = pmethod;
}

#ifdef G__ROOT
extern "C" void* G__new_interpreted_object(int size) {
  char *p = new char[size];
  return((void*)p);
}
extern "C" void G__delete_interpreted_object(void* p) {
  delete [] (char*)p;
}
#endif


/*********************************************************************
* G__SourceFileInfo
*********************************************************************/
////////////////////////////////////////////////////////////////////
void G__SourceFileInfo::Init(const char* fname) {
  for(filen=0;filen<G__nfile;filen++) {
    if(0==strcmp(fname,G__srcfile[filen].filename)) return;
  }
} 
////////////////////////////////////////////////////////////////////
const char* G__SourceFileInfo::Name() {
  if(IsValid()) {
    return(G__srcfile[filen].filename);
  }
  else {
    return((char*)NULL);
  }
} 
////////////////////////////////////////////////////////////////////
const char* G__SourceFileInfo::Prepname() {
  if(IsValid()) {
    return(G__srcfile[filen].prepname);
  }
  else {
    return((char*)NULL);
  }
} 
////////////////////////////////////////////////////////////////////
FILE* G__SourceFileInfo::fp() {
  if(IsValid()) {
    return(G__srcfile[filen].fp);
  }
  else {
    return((FILE*)NULL);
  }
}
////////////////////////////////////////////////////////////////////
int G__SourceFileInfo::MaxLine() {
  if(IsValid()) {
    return(G__srcfile[filen].maxline);
  }
  else {
    return(0);
  }
}
////////////////////////////////////////////////////////////////////
G__SourceFileInfo& G__SourceFileInfo::IncludedFrom() {
  static G__SourceFileInfo x;
  if(IsValid()) {
    x.filen = G__srcfile[filen].included_from;
  }
  else {
    x.filen = -1;
  }
  return(x);
}
////////////////////////////////////////////////////////////////////
long G__SourceFileInfo::Property() { 
  return(0); 
}
////////////////////////////////////////////////////////////////////
int G__SourceFileInfo::IsValid() { 
  if(filen<0||filen>=G__nfile) return(0);
  else return(1); 
}
////////////////////////////////////////////////////////////////////
int G__SourceFileInfo::Next() {
  ++filen;
#ifndef G__OLDIMPLEMENTATION
  while(G__srcfile[filen].hash==0 && IsValid()) ++filen;
#endif
  if(IsValid()) return(1);
  else return(0);
}
////////////////////////////////////////////////////////////////////

/*********************************************************************
* G__IncludePathInfo
*********************************************************************/
////////////////////////////////////////////////////////////////////
const char* G__IncludePathInfo::Name() {
  if(IsValid()) {
    return(p->pathname);
  }
  else {
    return((char*)NULL);
  }
} 
////////////////////////////////////////////////////////////////////
long G__IncludePathInfo::Property() { 
  return(0); 
}
////////////////////////////////////////////////////////////////////
int G__IncludePathInfo::IsValid() {
  if(p&&p->pathname) return(1);
  else  return(0);
}
////////////////////////////////////////////////////////////////////
int G__IncludePathInfo::Next() {
  if(!p) {
    p = &G__ipathentry;
  }
  else {
    p = p->next;
  }
  return(IsValid());
}
////////////////////////////////////////////////////////////////////

#ifdef G__EXCEPTIONWRAPPER
#ifdef G__STD_EXCEPTION
#include <exception>
#include <typeinfo>
#include <string>
#if !defined(__hpux) || __HP_aCC >= 53000
using namespace std;
#endif
#endif
/*********************************************************************
* G__DemangleClassname
*********************************************************************/
static int G__DemangleClassname(char *buf,const char *orig)
{
  int tagnum;

  /* try typeid.name() as is */
  strcpy(buf,orig);
  tagnum = G__defined_tagname(buf,2);
  if(-1!=tagnum) return(1);
  
  /* try eliminating digit at the beginning "9exception" -> "exception" 
   * this works for classes in global scope in g++ */
  int ox=0;
  while(isdigit(orig[ox])) ++ox; 
  strcpy(buf,orig+ox);
  tagnum = G__defined_tagname(buf,2);
  if(-1!=tagnum) return(1);

  /* try Q25abcde4hijk -> abcde::hijk 
   * this works for classes in enclosed scope in g++ 2.96 */
  int n=0;
  int nest = orig[1]-'0';
  int len;
  int totallen=0;
  ox = 2;
  buf[0]=0;
  for(n=0;n<nest;n++) {
    len=0;
    while(isdigit(orig[ox])){
      len = len*10 + orig[ox]-'0';
      ++ox; 
    }
    if(buf[0]) {
      strcat(buf,"::");
      totallen += (2+len);
    }
    else {
      totallen=len;
    }
    strcat(buf,orig+ox);
    buf[totallen] = 0;
    ox += len;
  }
  tagnum = G__defined_tagname(buf,2);
  if(-1!=tagnum) return(1);

  /* try N5abcde4hijkE -> abcde::hijk 
   * this works for classes in enclosed scope in g++ 3.x */
  totallen=0;
  ox = 1;
  buf[0]=0;
  for(;;) {
    len=0;
    while(isdigit(orig[ox])){
      len = len*10 + orig[ox]-'0';
      ++ox; 
    }
    if(buf[0]) {
      strcat(buf,"::");
      totallen += (2+len);
    }
    else {
      totallen=len;
    }
    strcat(buf,orig+ox);
    buf[totallen] = 0;
    ox += len;
    if(!isdigit(orig[ox])) break;
  }
  tagnum = G__defined_tagname(buf,2);
  if(-1!=tagnum) return(1);

  /* Give up and settle with G__exception */
  return(0);
  
}

/*********************************************************************
* G__ExceptionWrapper
*********************************************************************/
extern "C" int G__ExceptionWrapper(G__InterfaceMethod funcp
				   ,G__value* result7
				   ,char* funcname
				   ,struct G__param *libp
				   ,int hash)
{
  if(!G__catchexception) {
    return((*funcp)(result7,funcname,libp,hash));
  }
  try {
    return((*funcp)(result7,funcname,libp,hash));
  }
  catch(G__bc_exception& /* x */) {
    throw;
  }
#ifdef G__STD_EXCEPTION
  catch(std::exception& x) {
    char buf[G__LONGLINE];
#ifdef G__VISUAL
    // VC++ has problem in typeid(x).name(), so every thrown exception is
    // translated to G__exception.
    sprintf(buf,"new G__exception(\"%s\")",x.what());
#else
    char buf2[G__ONELINE];
    if(G__DemangleClassname(buf2,typeid(x).name())) {
      sprintf(buf,"new %s(*(%s*)%ld)",buf2,buf2,(long)(&x));
    }
    else {
      sprintf(buf,"new G__exception(\"%s\",\"%s\")",x.what(),buf2);
    }
#endif
    G__exceptionbuffer = G__getexpr(buf);
    G__exceptionbuffer.ref = G__exceptionbuffer.obj.i;
    G__return = G__RETURN_TRY;
    G__no_exec = 1;
  }
#endif 
  catch(int x) {
    G__letint(&G__exceptionbuffer,'i',(long)x);
    G__exceptionbuffer.ref = (long)(&x);
    G__return = G__RETURN_TRY;
    G__no_exec = 1;
  }
  catch(long x) {
    G__letint(&G__exceptionbuffer,'l',(long)x);
    G__exceptionbuffer.ref = (long)(&x);
    G__return = G__RETURN_TRY;
    G__no_exec = 1;
  }
  catch(void *x) {
    G__letint(&G__exceptionbuffer,'Y',(long)x);
    G__exceptionbuffer.ref = (long)(&x);
    G__return = G__RETURN_TRY;
    G__no_exec = 1;
  }
  catch(float x) {
    G__letdouble(&G__exceptionbuffer,'f',(double)x);
    G__exceptionbuffer.ref = (long)(&x);
    G__return = G__RETURN_TRY;
    G__no_exec = 1;
  }
  catch(double x) {
    G__letdouble(&G__exceptionbuffer,'d',x);
    G__exceptionbuffer.ref = (long)(&x);
    G__return = G__RETURN_TRY;
    G__no_exec = 1;
  }
  catch(string x) {
    G__fprinterr(G__serr,"Exception: %s\n",x.c_str());
    G__genericerror((char*)NULL);
    //G__return = G__RETURN_TRY;
    //G__no_exec = 1;
  }
  catch(...) {
    if(2==G__catchexception) {
      G__fprinterr(G__serr,"Error: Exception caught in compiled code\n");
      exit(EXIT_FAILURE);
    }
    G__genericerror("Error: C++ exception caught");
  }
  return 0;
}
#endif
////////////////////////////////////////////////////////////////////


/*********************************************************************
* New scheme operator new/delete 
*********************************************************************/

#ifdef G__NEVER
////////////////////////////////////////////////////////////////////
extern "C" void* G__operator_new(size_t size,void* p) {
  if(p && (long)p==G__getgvp() && G__PVOID!=G__getgvp()) return(p);
  return new char(size);
}

////////////////////////////////////////////////////////////////////
extern "C" void* G__operator_new_ary(size_t size,void* p) {
  if(p && (long)p==G__getgvp() && G__PVOID!=G__getgvp()) return(p);
  return new char[](size);
}

////////////////////////////////////////////////////////////////////
extern "C" void G__operator_delete(void *p) {
  if((long)p==G__getgvp() && G__PVOID!=G__getgvp()) return;
  delete p;
}

////////////////////////////////////////////////////////////////////
extern "C" void G__operator_delete_ary(void *p) {
  if((long)p==G__getgvp() && G__PVOID!=G__getgvp()) return;
  delete[] p;
}
////////////////////////////////////////////////////////////////////
#endif

#ifdef G__CPPCONSTSTRING
#include <set>
#include <string>
#if (!defined(__hpux) && !defined(_MSC_VER)) || __HP_aCC >= 53000
using namespace std;
#endif
/******************************************************************
* char* G__savestring()
******************************************************************/
static const char* G__saveconststring__dummy(const char* s)
{
  static set<string> conststring;
  string str(s);
  conststring.insert(string(str));
  set<string>::iterator p = conststring.lower_bound(str);
  return((*p).c_str());
}
extern "C" const char* G__saveconststring(const char* s)
{
  return G__saveconststring__dummy (s);
}
#endif


extern "C" void G__initcxx() 
{
#if defined(__HP_aCC)||defined(__SUNPRO_CC)||defined(__BCPLUSPLUS__)||defined(__KCC)||defined(__INTEL_COMPILER)
  char temp[G__ONELINE];
#endif
#ifdef __HP_aCC     /* HP aCC C++ compiler */
  sprintf(temp,"G__HP_aCC=%ld",(long)__HP_aCC); G__add_macro(temp);
#if __HP_aCC > 15000
  sprintf(temp,"G__ANSIISOLIB=1"); G__add_macro(temp);
#endif
#endif
#ifdef __SUNPRO_CC  /* Sun C++ compiler */
  sprintf(temp,"G__SUNPRO_CC=%ld",(long)__SUNPRO_CC); G__add_macro(temp);
#endif
#ifdef __BCPLUSPLUS__  /* Borland C++ compiler */
  sprintf(temp,"G__BCPLUSPLUS=%ld",(long)__BCPLUSPLUS__); G__add_macro(temp);
#endif
#ifdef __KCC        /* KCC  C++ compiler */
  sprintf(temp,"G__KCC=%ld",(long)__KCC); G__add_macro(temp);
#endif
#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER<810) /* icc and ecc C++ compilers */
  sprintf(temp,"G__INTEL_COMPILER=%ld",(long)__INTEL_COMPILER); G__add_macro(temp);
#endif
  /*
#ifdef __cplusplus 
  sprintf(temp,"G__CPLUSPLUS=%ld",(long)__cplusplus); G__add_macro(temp);
#endif
  */
}

#include <map>
#include <string>
#if (!defined(__hpux) && !defined(_MSC_VER)) || __HP_aCC >= 53000
using namespace std;
#endif

/******************************************************************
*  map<string,string> &G_get_symbolmacro()
******************************************************************/
//map<string,string> G__symbolmacro;
map<string,string> &G__get_symbolmacro()
{  
  static map<string,string> G__symbolmacro;
  return G__symbolmacro;
}

/******************************************************************
* void G__init_replacesymbol_body()
******************************************************************/
void G__init_replacesymbol_body() {
  G__get_symbolmacro().clear();
}

/******************************************************************
* void G__init_replacesymbol()
******************************************************************/
extern "C" void G__init_replacesymbol() {
  G__init_replacesymbol_body();
}

/******************************************************************
* void G__add_replacesymbol_body()
******************************************************************/
void G__add_replacesymbol_body(const char* s1,const char* s2) {
  map<string,string>::value_type x(s1,s2);
  G__get_symbolmacro().insert(x);
}
/******************************************************************
* void G__add_replacesymbol()
******************************************************************/
extern "C" void G__add_replacesymbol(const char* s1,const char* s2) {
  G__add_replacesymbol_body(s1,s2);
}

/******************************************************************
* char* G__replacesymbol_body()
******************************************************************/
const char* G__replacesymbol_body(const char* s) {
  map<string,string>::iterator pos = G__get_symbolmacro().find(s);
  if(pos!=G__get_symbolmacro().end()) return((*pos).second.c_str());
  else                          return(s);
}

/******************************************************************
* char* G__replacesymbol()
******************************************************************/
extern "C" const char* G__replacesymbol(const char* s) {
  return(G__replacesymbol_body(s));
}

/******************************************************************
* void G__display_replacesymbol_body()
******************************************************************/
int G__display_replacesymbol_body(FILE *fout,const char* name) {
  map<string,string>::iterator i;
  char msg[G__LONGLINE];
  for(i=G__get_symbolmacro().begin();i!=G__get_symbolmacro().end();++i) {
    if(!name || !name[0] || strcmp(name,(*i).first.c_str())==0) {
      sprintf(msg,"#define %s %s\n",(*i).first.c_str(),(*i).second.c_str());
      G__more(fout,msg);
      if(name && name[0]) return(1);
    }
  }
  return(0);
}

/******************************************************************
* void G__init_replacesymbol()
******************************************************************/
extern "C" int G__display_replacesymbol(FILE *fout,const char* name) {
  return(G__display_replacesymbol_body(fout,name));
}

