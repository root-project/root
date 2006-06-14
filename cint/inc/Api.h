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

#include <string>
#include <ostream>

/*********************************************************************
* include header files
*********************************************************************/

#define G__API
#ifndef G__DICTIONARY
#define G__DICTIONARY
#endif
#include "G__ci.h"
#ifndef __MAKECINT__
#include "common.h"
extern "C" G__EXPORT void G__CurrentCall(int, const void*, const int&);
#else
struct G__friendtag ;
typedef int (*G__InterfaceMethod)();
#endif
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
typedef void *(*G__pMethodSpecialObject)(char *item,G__ClassInfo *typeinfo
					 ,void** pptr,void** ppdict);
G__EXPORT void G__InitGetSpecialObject(G__pMethodSpecialObject pmethod);
}

/*********************************************************************
* $xxx object resolution function, Generic
*********************************************************************/
extern "C" {
typedef void (*G__pMethodSpecialValue)(char *item,G__TypeInfo *typeinfo
				       ,long *pl,double *pd,void** pptr
				       ,void** ppdict);
G__EXPORT void G__InitGetSpecialValue(G__pMethodSpecialValue pmethod);
}


/*********************************************************************
* Feedback routine in case tagnum for a class changes (in case the
* dictionary of a shared lib is being re-initialized).
*********************************************************************/
extern "C" {
typedef void (*G__pMethodUpdateClassInfo)(char *item,long tagnum);
G__EXPORT void G__InitUpdateClassInfo(G__pMethodUpdateClassInfo pmethod);
}

/*********************************************************************
* G__SourceFileInfo
*********************************************************************/
class 
#ifndef __CINT__
G__EXPORT
#endif
G__SourceFileInfo {
 public:
  G__SourceFileInfo(): filen(0) { Init(); }
  G__SourceFileInfo(int filenin): filen(filenin) { }
  G__SourceFileInfo(const char* fname): filen(0) { Init(fname); }
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
  G__IncludePathInfo(): p(NULL) { Init(); }
#ifndef __CINT__
  G__IncludePathInfo(const G__IncludePathInfo& ipf): p(ipf.p) {}
#endif
  ~G__IncludePathInfo() { }
  void Init() { p=(struct G__includepath*)NULL; }
  const char *Name(); 
  long Property();
  int IsValid();
  int Next();
 private:
  G__IncludePathInfo& operator=(const G__IncludePathInfo&);
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


/*********************************************************************
* Shadow class functions
*********************************************************************/
#ifndef __CINT__
class G__EXPORT G__ShadowMaker {
public:
   static bool NeedShadowClass(G__ClassInfo& cl);
   G__ShadowMaker(std::ostream& out, const char* nsprefix, 
      bool(*needShadowClass)(G__ClassInfo &cl)=G__ShadowMaker::NeedShadowClass,
      bool(*needTypedefShadow)(G__ClassInfo &cl)=0);

   void WriteAllShadowClasses();

   void WriteShadowClass(G__ClassInfo &cl, int level = 0);
   int WriteNamespaceHeader(G__ClassInfo &cl);

   int NeedShadowCached(int tagnum) { return fCacheNeedShadow[tagnum]; }
   static bool IsSTLCont(const char *type);
   static bool IsStdPair(G__ClassInfo &cl);

   static void GetFullyQualifiedName(const char *originalName, std::string &fullyQualifiedName);
   static void GetFullyQualifiedName(G__ClassInfo &cl, std::string &fullyQualifiedName);
   static void GetFullyQualifiedName(G__TypeInfo &type, std::string &fullyQualifiedName);
   static std::string GetNonConstTypeName(G__DataMemberInfo &m, bool fullyQualified = false);
   void GetFullShadowName(G__ClassInfo &cl, std::string &fullname);

   static void VetoShadow(bool veto=true);

private:
   G__ShadowMaker(const G__ShadowMaker&); // intentionally not implemented
   G__ShadowMaker& operator =(const G__ShadowMaker&); // intentionally not implemented
   void GetFullShadowNameRecurse(G__ClassInfo &cl, std::string &fullname);

   std::ostream& fOut; // where to write to
   std::string fNSPrefix; // shadow classes are in this namespace's namespace "Shadow"
   char fCacheNeedShadow[G__MAXSTRUCT]; // whether we need a shadow for a tagnum
   static bool fgVetoShadow; // whether WritaAllShadowClasses should write the shadow
   bool (*fNeedTypedefShadow)(G__ClassInfo &cl); // func deciding whether the shadow is a tyepdef
};
#endif

#ifdef __MAKECINT__
#pragma link off class $G__value;
#pragma link off class $G__COMPLETIONLIST;
#pragma link off class $G__linked_taginfo;
#pragma link off class G__includepath;
#endif

#endif
