/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file Api.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 *   -I$(CINTSYSDIR) -I$(CINTSYSDIR)/src must be given at compile time
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#ifndef G__API_H
#define G__API_H

#include <string.h>


/*********************************************************************
* include header files
*********************************************************************/

#define G__API
#ifndef G__OLDIMPLEMENTATION1840
#ifndef G__DICTIONARY
#define G__DICTIONARY
#endif
#endif
#define G__CINTBODY /* may not be needed */
#include "G__ci.h"
#ifndef G__OLDIMPLEMENTATION1218
#ifndef __MAKECINT__
#include "common.h"
#ifndef G__OLDIMPLEMENTATION1749
extern "C" void G__CurrentCall(int, const void*, const int&);
#endif
#else
struct G__friendtag ;
typedef int (*G__InterfaceMethod)();
#endif
#endif /* 1218 */
#ifdef __MAKECINT__
#pragma link off all classes;
#pragma link off all functions;
#pragma link off all globals;
#pragma link off all typedefs;
#endif

#define G__INFO_BUFLEN 50
/* #define G__INFO_TITLELEN 256 */
#define G__INFO_TITLELEN G__ONELINE

#include "Property.h"
#include "Class.h"
#include "BaseCls.h"
#include "Type.h"
#include "Method.h"
#include "MethodAr.h"
#include "DataMbr.h"
#include "CallFunc.h"
#include "Typedf.h"
#include "Token.h"


/*********************************************************************
* $xxx object resolution function, pointer to a class object
*
* Usage:
*
*  extern "C" void YourGetObject(char *name,G__ClassInfo *type) {
*     void *p;
*     // Whatever you want to fill type and pointetr to the object info
*     return(p);
*  }
*
*  ROOT_Initialization() {
*     // set pointer to yourown interface method
*     G__InitGetSpecialObject(YourGetObject);
*  }
*********************************************************************/
extern "C" {
#if !defined(G__OLDIMPLEMENTATION481)
typedef void *(*G__pMethodSpecialObject)(char *item,G__ClassInfo *typeinfo
					 ,void** pptr,void** ppdict);
#elif !defined(G__OLDIMPLEMENTATION455)
typedef void *(*G__pMethodSpecialObject)(char *item,G__ClassInfo *typeinfo
					 ,void* ptr);
#else
typedef void *(*G__pMethodSpecialObject)(char *item,G__ClassInfo *typeinfo);
#endif
void G__InitGetSpecialObject(G__pMethodSpecialObject pmethod);
}

/*********************************************************************
* $xxx object resolution function, Generic
*********************************************************************/
extern "C" {
#if !defined(G__OLDIMPLEMENTATION481)
typedef void (*G__pMethodSpecialValue)(char *item,G__TypeInfo *typeinfo
				       ,long *pl,double *pd,void** pptr
				       ,void** ppdict);
#elif !defined(G__OLDIMPLEMENTATION455)
typedef void (*G__pMethodSpecialValue)(char *item,G__TypeInfo *typeinfo
				        ,long *pl,double *pd,void* ptr);
#else
typedef void (*G__pMethodSpecialValue)(char *item,G__TypeInfo *typeinfo
				        ,long *pl,double *pd);
#endif
void G__InitGetSpecialValue(G__pMethodSpecialValue pmethod);
}


#ifndef G__OLDIMPLEMENTATION1207
/*********************************************************************
* Feedback routine in case tagnum for a class changes (in case the
* dictionary of a shared lib is being re-initialized).
*********************************************************************/
extern "C" {
typedef void (*G__pMethodUpdateClassInfo)(char *item,long tagnum);
void G__InitUpdateClassInfo(G__pMethodUpdateClassInfo pmethod);
}
#endif

/*********************************************************************
* G__SourceFileInfo
*********************************************************************/
class 
#ifndef __CINT__
G__EXPORT
#endif
G__SourceFileInfo {
 public:
  G__SourceFileInfo() { Init(); }
  G__SourceFileInfo(int filenin) { filen = filenin; }
  G__SourceFileInfo(const char* fname) { Init(fname); }
  ~G__SourceFileInfo() { }
  void Init() { 
    filen = -1; 
  }
  void Init(const char* fname);
  const char *Name(); 
  const char *Prepname(); 
  FILE* fp();
  int MaxLine();
  G__SourceFileInfo& IncludedFrom();
  long Property();
  int IsValid();
  int Next();
 private:
  int filen;
};

/*********************************************************************
* G__IncludePathInfo
*********************************************************************/
#ifndef __CINT__
struct G__includepath;
#endif

class 
#ifndef __CINT__
G__EXPORT
#endif
G__IncludePathInfo {
 public:
  G__IncludePathInfo() { Init(); }
  ~G__IncludePathInfo() { }
  void Init() { p=(struct G__includepath*)NULL; }
  const char *Name(); 
  long Property();
  int IsValid();
  int Next();
 private:
#ifndef __CINT__
  struct G__includepath *p;
#endif
};

#ifdef G__EXCEPTIONWRAPPER
/*********************************************************************
* G__ExceptionWrapper
*********************************************************************/
extern "C" int G__ExceptionWrapper(G__InterfaceMethod funcp
				   ,G__value* result7
				   ,char* funcname
				   ,struct G__param *libp
				   ,int hash);
#endif

#ifdef __MAKECINT__
#pragma link off class $G__value;
#pragma link off class $G__COMPLETIONLIST;
#pragma link off class $G__linked_taginfo;
#pragma link off class G__includepath;
#endif

#endif
