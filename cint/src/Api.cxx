/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file Api.cxx
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
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

#include "Api.h"
#include "common.h"


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
#if !defined(G__OLDIMPLEMENTATION481)
extern "C" G__value G__APIGetSpecialValue_layer1(char *item,void **pptr
	,void **ppdict)
#elif !defined(G__OLDIMPLEMENTATION455)
extern "C" G__value G__APIGetSpecialValue_layer1(char *item,void *ptr)
#else
extern "C" G__value G__APIGetSpecialValue_layer1(char *item)
#endif
{
  G__value result;
  long l;
  double d;
  G__TypeInfo typeinfo;
#if !defined(G__OLDIMPLEMENTATION481)
  (*G__UserSpecificGetSpecialValue)(item,&typeinfo,&l,&d,pptr,ppdict);
#elif !defined(G__OLDIMPLEMENTATION455)
  (*G__UserSpecificGetSpecialValue)(item,&typeinfo,&l,&d,ptr);
#else
  (*G__UserSpecificGetSpecialValue)(item,&typeinfo,&l,&d);
#endif
  G__TypeInfo2G__value(&typeinfo,&result,l,d);
  return(result);
}

//
// Initialization routine
//
extern "C" void G__InitGetSpecialValue(G__pMethodSpecialValue pmethod) 
{
#if !defined(G__OLDIMPLEMENTATION481)
  G__GetSpecialObject
	=(G__value (*)(char*,void**,void**))G__APIGetSpecialValue_layer1;
#elif !defined(G__OLDIMPLEMENTATION455)
  G__GetSpecialObject=(G__value (*)(char*,void*))G__APIGetSpecialValue_layer1;
#else
  G__GetSpecialObject = (G__value (*)(char*))G__APIGetSpecialValue_layer1;
#endif
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
#if !defined(G__OLDIMPLEMENTATION481)
extern "C" G__value G__APIGetSpecialObject_layer1(char *item,void** pptr
	,void** ppdict)
#elif !defined(G__OLDIMPLEMENTATION455)
extern "C" G__value G__APIGetSpecialObject_layer1(char *item,void* ptr)
#else
extern "C" G__value G__APIGetSpecialObject_layer1(char *item)
#endif
{
  G__value result;
  long l;
  G__ClassInfo type;
#if !defined(G__PHILIPPE4)
  int store_prerun = G__prerun;
  G__prerun = 0;
  l=(long)((*G__UserSpecificGetSpecialObject)(item,&type,pptr,ppdict));
  G__prerun = store_prerun;
#elif !defined(G__OLDIMPLEMENTATION481)
  l=(long)((*G__UserSpecificGetSpecialObject)(item,&type,pptr,ppdict));
#elif !defined(G__OLDIMPLEMENTATOIN455)
  l=(long)((*G__UserSpecificGetSpecialObject)(item,&type,ptr));
#else
  l=(long)((*G__UserSpecificGetSpecialObject)(item,&type));
#endif
  G__ClassInfo2G__value(&type,&result,l);
  return(result);
}

//
// Initialization routine
//
extern "C" void G__InitGetSpecialObject(G__pMethodSpecialObject pmethod) 
{
#ifndef G__OLDIMPLEMENTATION1035
  G__LockCriticalSection();
#endif
#if !defined(G__OLDIMPLEMENTATION481)
  G__GetSpecialObject
	=(G__value (*)(char*,void**,void**))G__APIGetSpecialObject_layer1;
#elif !defined(G__OLDIMPLEMENTATION455)
  G__GetSpecialObject=(G__value (*)(char*,void*))G__APIGetSpecialObject_layer1;
#else
  G__GetSpecialObject = (G__value (*)(char*))G__APIGetSpecialObject_layer1;
#endif
  G__UserSpecificGetSpecialObject = pmethod;
#ifndef G__OLDIMPLEMENTATION1035
  G__UnlockCriticalSection();
#endif
}

#ifndef G__OLDIMPLEMENTATION1207
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
#endif

#ifndef G__OLDIMPLEMENTATION1002
#ifdef G__ROOT
extern "C" void* G__new_interpreted_object(int size) {
  char *p = new char[size];
  return((void*)p);
}
extern "C" void G__delete_interpreted_object(void* p) {
#ifndef G__OLDIMPLEMENTATION1074
  delete [] (char*)p;
#else
  delete p;
#endif
}
#endif
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
#ifndef __hpux
using namespace std;
#endif
#endif
/*********************************************************************
* G__ExceptionWrapper
*********************************************************************/
extern "C" int G__ExceptionWrapper(G__InterfaceMethod funcp
				   ,G__value* result7
				   ,char* funcname
				   ,struct G__param *libp
				   ,int hash)
{
  try {
    (*funcp)(result7,funcname,libp,hash);
    return 1;
  }
#ifdef G__STD_EXCEPTION
  catch(exception& x) {
    char buf[G__LONGLINE];
#ifdef G__VISUAL
    sprintf(buf,"new exception(\"%s\")",x.what());
#else
    char buf2[G__ONELINE];
    int ox=0;
    strcpy(buf2,typeid(x).name());
    while(isdigit(buf2[ox])) ++ox; /* why need this ??? */
    sprintf(buf,"new %s(*(%s*)%ld)",buf2+ox,buf2+ox,(long)(&x));
#endif
    G__exceptionbuffer = G__getexpr(buf);
    G__exceptionbuffer.ref = G__exceptionbuffer.obj.i;
    G__return = G__RETURN_TRY;
  }
#endif
  catch(...) {
    G__genericerror("Error: C++ exception caught");
  }
  return 0;
}
#endif
////////////////////////////////////////////////////////////////////
