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
 * Copyright(c) 1995~2007  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__API_H
#define G__API_H

#define G__API

#include "G__ci.h"

#ifndef __CINT__
#include "Reflex/Type.h"
#else // __CINT__
#include <string>
namespace Reflex {
class Type;
class Scope;
class Type_Iterator;
} // namespace Reflex
#endif // __CINT__

#ifdef __MAKECINT__
// Prevent the dictionary generation for vector<Type> and
// the associated free functions
#include <vector>
std::vector< ::Reflex::Type>* instiator;
#pragma link off all classes;
#pragma link off all functions;
#pragma link off all globals;
#pragma link off all typedefs;
#endif // __MAKECINT__

#define G__INFO_BUFLEN 50
//#define G__INFO_TITLELEN 256
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


#include <vector>
#include <string>

#ifndef __CINT__
struct G__includepath;
#endif // __CINT__

namespace Cint {

typedef void* (*G__pMethodSpecialObject)(char* item, G__ClassInfo* typeinfo, void** pptr, void** ppdict);
#ifndef __CINT__
G__EXPORT
#endif // __CINT__
void G__InitGetSpecialObject(G__pMethodSpecialObject pmethod);

typedef void (*G__pMethodSpecialValue)(char* item, G__TypeInfo* typeinfo, long* pl, double* pd, void** pptr, void** ppdict);
#ifndef __CINT__
G__EXPORT
#endif // __CINT__
void G__InitGetSpecialValue(G__pMethodSpecialValue pmethod);


#ifndef __CINT__
G__EXPORT
#endif // __CINT__
int G__SetGlobalcomp(char* funcname, char* param, int globalcomp);

#ifndef __CINT__
G__EXPORT
#endif // __CINT__
int G__ForceBytecodecompilation(char* funcname, char* param);

typedef void (*G__pMethodUpdateClassInfo)(char* item, long tagnum);

#ifndef __CINT__
G__EXPORT
#endif // __CINT__
void G__InitUpdateClassInfo(G__pMethodUpdateClassInfo pmethod);

#ifdef G__ROOT

#ifndef __CINT__
G__EXPORT
#endif // __CINT__
void* G__new_interpreted_object(int size);

#ifndef __CINT__
G__EXPORT
#endif // __CINT__
void G__delete_interpreted_object(void* p);

#endif // G__ROOT

typedef int (*G__pGenerateDictionary)(const std::string& className, const std::vector<std::string>& headers);

#ifndef __CINT__
G__EXPORT
#endif // __CINT__
void G__InitGenerateDictionary(G__pGenerateDictionary gdict);

#ifndef __CINT__
G__EXPORT
#endif // __CINT__
G__pGenerateDictionary G__GetGenerateDictionary();

//______________________________________________________________________________
class
#ifndef __CINT__
G__EXPORT
#endif // __CINT__
G__SourceFileInfo
{
public:
   G__SourceFileInfo() : filen(0)
   {
      Init();
   }
   G__SourceFileInfo(int filenin) : filen(filenin)
   {
   }
   G__SourceFileInfo(const char* fname) : filen(0)
   {
      Init(fname);
   }
   ~G__SourceFileInfo()
   {
   }
   void Init()
   {
      filen = -1;
   }
   void Init(const char* fname);
   const char* Name();
   const char* Prepname();
   FILE* fp();
   int MaxLine();
   G__SourceFileInfo& IncludedFrom();
   long Property();
   int IsValid();
   int Next();
private:
   int filen;
};

//______________________________________________________________________________
class
#ifndef __CINT__
G__EXPORT
#endif // __CINT__
G__IncludePathInfo
{
public:
   G__IncludePathInfo() : p(0)
   {
      Init();
   }
#ifndef __CINT__
   G__IncludePathInfo(const G__IncludePathInfo& ipf) : p(ipf.p)
   {
   }
#endif // __CINT__
   ~G__IncludePathInfo()
   {
   }
   void Init() {
      p = 0;
   }
   const char* Name();
   long Property();
   int IsValid();
   int Next();
private:
   G__IncludePathInfo& operator=(const G__IncludePathInfo&);
#ifndef __CINT__
   G__includepath* p;
#endif // __CINT__
   // --
};

#ifdef G__EXCEPTIONWRAPPER
// TODO: G__ExceptionWrapper needs to return its offset as char*!
int G__ExceptionWrapper(G__InterfaceMethod funcp, G__value* result7, char* funcname, G__param* libp, int hash);
#endif // G__EXCEPTIONWRAPPER

unsigned long G__long_random(unsigned long limit);

} // namespace Cint

using namespace Cint;

//
//  scratch upto dictionary position
//  Must be declared outside namespace Cint because of use in G__ci.h
//
struct G__ConstStringList;
struct G__Preprocessfilekey;
struct G__Deffuncmacro;
struct G__Definetemplatefunc;

struct G__dictposition
{
   ::Reflex::Scope var; // global variable table position
   int ig15; // global variable table position
   int tagnum; // class table position
   G__ConstStringList* conststringpos; // string constant table position
   int typenum; // typdef table position
   ::Reflex::Scope ifunc; // global function table position
   int ifn; // global function table position
   G__includepath* ipath; // include path table position
   int allsl; // shared library file table position
   G__Preprocessfilekey* preprocessfilekey; // preprocessor file key table position
   int nfile; // input file table position
   G__Deffuncmacro* deffuncmacro; // macro table position
   G__Definedtemplateclass* definedtemplateclass; // class template table position
   G__Definetemplatefunc* definedtemplatefunc; // function template table position
   int nactives; // number of 'active' classes 
};

//
// C++ Interface methods
//

namespace Cint {
void G__letpointer(G__value* buf, long value, const ::Reflex::Type& type);
} // namespace Cint

#ifdef __MAKECINT__
#pragma link off class $G__value;
#pragma link off class $G__COMPLETIONLIST;
#pragma link off class $G__linked_taginfo;
#pragma link off class G__includepath;
#pragma link C++ namespace Cint;
#pragma link off namespace Cint::Internal;
#endif

#endif // G__API_H
