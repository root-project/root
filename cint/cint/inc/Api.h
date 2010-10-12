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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__API_H
#define G__API_H

/*********************************************************************
* include header files
*********************************************************************/

#define G__API
#include "G__ci.h"
#ifdef __MAKECINT__
#pragma link off all classes;
#pragma link off all functions;
#pragma link off all globals;
#pragma link off all typedefs;
#endif

#define G__INFO_BUFLEN 50
/* #define G__INFO_TITLELEN 256 */
#define G__INFO_TITLELEN G__ONELINE

#ifndef G__PROPERTY_H
#include "Property.h"
#endif
#ifndef G__CLASSINFO_H
#include "Class.h"
#endif
#ifndef G__BaseClassInfo_H
#include "BaseCls.h"
#endif
#ifndef G__TYPEINFOX_H
#include "Type.h"
#endif
#ifndef G__METHODINFO_H
#include "Method.h"
#endif
#ifndef G__METHODARGINFO_H
#include "MethodAr.h"
#endif
#ifndef G__DATAMEMBER_H
#include "DataMbr.h"
#endif
#ifndef G__CALLFUNC_H
#include "CallFunc.h"
#endif
#ifndef G__TYPEDEFINFO_H
#include "Typedf.h"
#endif
#ifndef G__TOKENINFO_H
#include "Token.h"
#endif

#include <vector>
#include <string>

#ifndef __CINT__
struct G__includepath;
#endif

extern "C" {
#ifndef __CINT__
G__EXPORT
#endif
int G__Lsizeof(const char *typenamein);
}

struct G__ConstStringList;
struct G__Preprocessfilekey;
struct G__Deffuncmacro;
struct G__Definedtemplateclass;
struct G__Definetemplatefunc;

/*********************************************************************
 * scratch upto dictionary position
 *********************************************************************/
struct G__dictposition {
   /* global variable table position */
   struct G__var_array *var;
   int ig15;
   /* struct tagnum */
   int tagnum;
   /* const string table */
   struct G__ConstStringList *conststringpos;
   /* typedef table */
   int typenum;
   /* global function table position */
   struct G__ifunc_table *ifunc;
   int ifn;
   /* include path */
   struct G__includepath *ipath;
   /* shared library file */
   int allsl;
   /* preprocessfilekey */
   struct G__Preprocessfilekey *preprocessfilekey;
   /* input file */
   int nfile;
   /* macro table */
   struct G__Deffuncmacro *deffuncmacro;
   /* template class */
   struct G__Definedtemplateclass *definedtemplateclass;
   /* function template */
   struct G__Definetemplatefunc *definedtemplatefunc;
   
   int nactives; /* number of 'active' classes */
};

namespace Cint {

/*********************************************************************
* $xxx object resolution function, pointer to a class object
*
* Usage:
*
*  void YourGetObject(char *name,G__ClassInfo *type) {
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
typedef void *(*G__pMethodSpecialObject)(char *item,G__ClassInfo *typeinfo
                                         ,void** pptr,void** ppdict);
#ifndef __CINT__
G__EXPORT
#endif
void G__InitGetSpecialObject(G__pMethodSpecialObject pmethod);

/*********************************************************************
* $xxx object resolution function, Generic
*********************************************************************/
typedef void (*G__pMethodSpecialValue)(char *item,G__TypeInfo *typeinfo
                                       ,long *pl,double *pd,void** pptr
                                       ,void** ppdict);
#ifndef __CINT__
G__EXPORT
#endif
void G__InitGetSpecialValue(G__pMethodSpecialValue pmethod);


#ifndef __CINT__
G__EXPORT
#endif
int G__SetGlobalcomp(char *funcname,char *param,int globalcomp); // Method.cxx
int G__SetForceStub(char *funcname,char *param); // Method.cxx

#ifndef __CINT__
G__EXPORT
#endif
int G__ForceBytecodecompilation(char *funcname,char *param); // Method.cxx

/*********************************************************************
* Feedback routine in case tagnum for a class changes (in case the
* dictionary of a shared lib is being re-initialized).
*********************************************************************/
typedef void (*G__pMethodUpdateClassInfo)(char *item,long tagnum);

#ifndef __CINT__
G__EXPORT
#endif
void G__InitUpdateClassInfo(G__pMethodUpdateClassInfo pmethod);

#ifndef __CINT__
G__EXPORT
#endif
void* G__new_interpreted_object(int size);

#ifndef __CINT__
G__EXPORT
#endif
void G__delete_interpreted_object(void* p);

/*********************************************************************
* Generate dictionary.
*********************************************************************/
   typedef int (*G__pGenerateDictionary)(const std::string &className,const std::vector<std::string> &headers, const std::vector<std::string> &fwdDecls, const std::vector<std::string> &unknown);

#ifndef __CINT__
G__EXPORT
#endif
void G__InitGenerateDictionary( G__pGenerateDictionary gdict );

#ifndef __CINT__
G__EXPORT
#endif
G__pGenerateDictionary G__GetGenerateDictionary();


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

  static int SerialNumber();
 private:
  int filen;
};

/*********************************************************************
* G__IncludePathInfo
*********************************************************************/
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
int G__ExceptionWrapper(G__InterfaceMethod funcp
                        ,G__value* result7
                        ,char* funcname
                        ,struct G__param *libp
                        ,int hash);
#endif

unsigned long G__long_random(unsigned long limit);
   


/*********************************************************************
* External readline interface
*********************************************************************/
typedef char* (*G__pGetline_t)(const char* prompt);
typedef void  (*G__pHistadd_t)(char* line);

#ifndef __CINT__
G__EXPORT
#endif
G__pGetline_t G__GetGetlineFunc();

#ifndef __CINT__
G__EXPORT
#endif
G__pHistadd_t G__GetHistaddFunc();

#ifndef __CINT__
G__EXPORT
#endif
   void G__SetGetlineFunc(G__pGetline_t glfcn, G__pHistadd_t hafcn);

} // namespace Cint

using namespace Cint;

#ifdef __MAKECINT__
#pragma link off class $G__value;
#pragma link off class $G__COMPLETIONLIST;
#pragma link off class $G__linked_taginfo;
#pragma link off class G__includepath;
#pragma link C++ namespace Cint;
#pragma link C++ nestedclasses;
#pragma link off namespace Cint::Internal;
#endif

#endif
