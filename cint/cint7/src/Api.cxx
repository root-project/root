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
#include "Dict.h"
#include "common.h"

#include "Reflex/Builder/TypeBuilder.h"

#include <cstring>
#include <map>
#include <string>

#ifdef G__STD_EXCEPTION
#include <exception>
#include <typeinfo>
#endif // G__STD_EXCEPTION

#if defined(__GNUC__) && __GNUC__ >= 3
#define G__HAVE_CXA_DEMANGLE
#include <cxxabi.h>
#endif

using namespace Cint::Internal;
using namespace std;

//______________________________________________________________________________
//______________________________________________________________________________
//
//  Function Directory.
//

// Static Functions.
static G__value G__APIGetSpecialValue_layer1(char* item, void** pptr, void** ppdict);
static void G__TypeInfo2G__value(Cint::G__TypeInfo* ti, G__value* pvalue, long l, double d);
static G__value G__APIGetSpecialObject_layer1(char* item, void** pptr, void** ppdict);
static void G__ClassInfo2G__value(G__ClassInfo* type, G__value* pvalue, long l);
#ifdef G__EXCEPTIONWRAPPER
static int G__DemangleClassname(char* buf, const char* orig);
#endif // G__EXCEPTIONWRAPPER
static std::map<string, string>& G__get_symbolmacro();

// Cint Functions.
namespace Cint {
void G__InitGetSpecialValue(G__pMethodSpecialValue pmethod);
void G__InitGetSpecialObject(G__pMethodSpecialObject pmethod);
void G__InitUpdateClassInfo(G__pMethodUpdateClassInfo pmethod); // Set class changed callback.
#ifdef G__ROOT
void* G__new_interpreted_object(int size);
void G__delete_interpreted_object(void* p);
#endif // G__ROOT
void G__InitGenerateDictionary(G__pGenerateDictionary gdict);
G__pGenerateDictionary G__GetGenerateDictionary();
#ifdef G__EXCEPTIONWRAPPER
int G__ExceptionWrapper(G__InterfaceMethod funcp, G__value* result7, char* funcname, G__param* libp, int hash);
#endif // G__EXCEPTIONWRAPPER
} // namespace Cint

// Cint Internal Functions.
namespace Cint {
namespace Internal {
void G__initcxx();
void G__init_replacesymbol();
void G__add_replacesymbol(const char* s1, const char* s2);
const char* G__replacesymbol(const char* s);
int G__display_replacesymbol(FILE* fout, const char* name);
} // namespace Internal
} // namespace Cint

//  Implementation of class Cint::G__SourceFileInfo.
//  Implementation of class Cint::G__IncludePathInfo.

//______________________________________________________________________________
//______________________________________________________________________________
//
//  Variables
//

// C Interface Global Variables
extern "C" int G__EnableAutoDictionary; // Boolean flag, enable automatic dictionary generation.
int G__EnableAutoDictionary = 1;

// Cint Internal Global Variables.
namespace Cint {
namespace Internal {
G__pMethodUpdateClassInfo G__UserSpecificUpdateClassInfo; // Class is changed callback.
} // namespace Internal
} // namespace Cint

// Static Global Variables.
static G__pMethodSpecialValue G__UserSpecificGetSpecialValue;
static G__pMethodSpecialObject G__UserSpecificGetSpecialObject;
static G__pGenerateDictionary G__GenerateDictionary = 0; // Dictionary generator callback.

//______________________________________________________________________________
//
//  Static Functions.
//

//______________________________________________________________________________
static G__value G__APIGetSpecialValue_layer1(char* item, void** pptr, void** ppdict)
{
   //  $xxx object resolution function Generic form
   //  There is another form of interface which only allow you
   //  to return a pointer to a class object if you want to gain
   //  a slight speed advantage.
   //
   G__value result;
   long l;
   double d;
   G__TypeInfo typeinfo;
   (*G__UserSpecificGetSpecialValue)(item, &typeinfo, &l, &d, pptr, ppdict);
   G__TypeInfo2G__value(&typeinfo, &result, l, d);
   return result;
}

//______________________________________________________________________________
static void G__TypeInfo2G__value(Cint::G__TypeInfo* ti, G__value* pvalue, long l, double d)
{
   int type = ti->Type();
   int tagnum = ti->Tagnum();
   int typenum = ti->Typenum();
   int reftype = ti->Reftype();
   int isconst = ti->Isconst();
   G__value_typenum(*pvalue) = G__cint5_tuple_to_type(type, tagnum, typenum, reftype, isconst);
   pvalue->ref = 0;
   switch (G__get_type(*pvalue)) {
      case 'd':
      case 'f':
         pvalue->obj.d = d;
         break;
      default:
         pvalue->obj.i = l;
         break;
   }
}

//______________________________________________________________________________
static G__value G__APIGetSpecialObject_layer1(char* item, void** pptr, void** ppdict)
{
   //  $xxx object resolution function for ROOT, slight speed advantage.
   //  This routine can only return a pointer to a class object.
   G__value result;
   G__ClassInfo type;
   int store_prerun = G__prerun;
   G__prerun = 0;
   long l = (long) (*G__UserSpecificGetSpecialObject)(item, &type, pptr, ppdict);
   G__prerun = store_prerun;
   G__ClassInfo2G__value(&type, &result, l);
   return result;
}

//______________________________________________________________________________
static void G__ClassInfo2G__value(G__ClassInfo* type, G__value* pvalue, long l)
{
   ::Reflex::Scope scope = G__Dict::GetDict().GetScope(type->Tagnum());
   ::Reflex::Type what = scope;
   G__value_typenum(*pvalue) = ::Reflex::PointerBuilder(what);
   pvalue->ref = 0;
   pvalue->obj.i = l;
}

#ifdef G__EXCEPTIONWRAPPER
//______________________________________________________________________________
static int G__DemangleClassname(char* buf, const char* orig)
{
   int tagnum;
   // try typeid.name() as is
   strcpy(buf, orig);
   tagnum = G__defined_tagname(buf, 2);
   if (-1 != tagnum) return(1);
   // try eliminating digit at the beginning "9exception" -> "exception"
   // this works for classes in global scope in g++
   int ox = 0;
   while (isdigit(orig[ox])) ++ox;
   strcpy(buf, orig + ox);
   tagnum = G__defined_tagname(buf, 2);
   if (-1 != tagnum) return(1);
#ifndef G__HAVE_CXA_DEMANGLE
   // try Q25abcde4hijk -> abcde::hijk
   // this works for classes in enclosed scope in g++ 2.96
   int n = 0;
   int nest = orig[1] - '0';
   int len;
   int totallen = 0;
   ox = 2;
   buf[0] = 0;
   for (n = 0;n < nest;n++) {
      len = 0;
      while (isdigit(orig[ox])) {
         len = len * 10 + orig[ox] - '0';
         ++ox;
      }
      if (buf[0]) {
         strcat(buf, "::");
         totallen += (2 + len);
      }
      else {
         totallen = len;
      }
      strcat(buf, orig + ox);
      buf[totallen] = 0;
      ox += len;
   }
   tagnum = G__defined_tagname(buf, 2);
   if (-1 != tagnum) return(1);
#else // G__HAVE_CXA_DEMANGLE
   int status = 0;
   char* cxaname =::abi::__cxa_demangle(orig, 0, 0, &status);
   strcpy(buf, cxaname);
   free(cxaname);
   tagnum = G__defined_tagname(buf, 2);
   if (-1 != tagnum) return(1);
#endif // G__HAVE_CXA_DEMANGLE
   // Give up and settle with G__exception
   return 0;
}
#endif // G__EXCEPTIONWRAPPER

//______________________________________________________________________________
static std::map<string, string>& G__get_symbolmacro()
{
   static std::map<string, string> G__symbolmacro;
   return G__symbolmacro;
}

//______________________________________________________________________________
//______________________________________________________________________________
//
//  Cint Functions.
//

//______________________________________________________________________________
void Cint::G__InitGetSpecialValue(G__pMethodSpecialValue pmethod)
{
   G__GetSpecialObject = (G__value(*)(char*, void**, void**)) G__APIGetSpecialValue_layer1;
   G__UserSpecificGetSpecialValue = pmethod;
}

//______________________________________________________________________________
void Cint::G__InitGetSpecialObject(G__pMethodSpecialObject pmethod)
{
   G__LockCriticalSection();
   G__GetSpecialObject = (G__value (*)(char*, void**, void**)) G__APIGetSpecialObject_layer1;
   G__UserSpecificGetSpecialObject = pmethod;
   G__UnlockCriticalSection();
}

//______________________________________________________________________________
void Cint::G__InitUpdateClassInfo(G__pMethodUpdateClassInfo pmethod)
{
   G__UserSpecificUpdateClassInfo = pmethod;
}

#ifdef G__ROOT
//______________________________________________________________________________
void* Cint::G__new_interpreted_object(int size)
{
   char* p = new char[size];
   return p;
}
#endif // G__ROOT

#ifdef G__ROOT
//______________________________________________________________________________
void Cint::G__delete_interpreted_object(void* p)
{
   delete[] (char*) p;
}
#endif // G__ROOT

//______________________________________________________________________________
void Cint::G__InitGenerateDictionary(G__pGenerateDictionary gdict)
{
   G__GenerateDictionary = gdict; // Usually a pointer to TCint_GenerateDictionary.
}

//______________________________________________________________________________
G__pGenerateDictionary Cint::G__GetGenerateDictionary()
{
   if (G__EnableAutoDictionary) {
      return G__GenerateDictionary;
   }
   return 0;
}

#ifdef G__EXCEPTIONWRAPPER
//______________________________________________________________________________
int Cint::G__ExceptionWrapper(G__InterfaceMethod funcp, G__value* result7, char* funcname, G__param* libp, int hash)
{
   if (!G__catchexception) {
      return (*funcp)(result7, funcname, libp, hash);
   }
#if ENABLE_CPP_EXCEPTIONS
   try {
#endif //ENABLE_CPP_EXCEPTIONS
       return (*funcp)(result7, funcname, libp, hash);
   }
#if ENABLE_CPP_EXCEPTIONS
#ifdef G__STD_EXCEPTION
   catch (std::exception& x) {
      G__StrBuf buf_sb(G__LONGLINE);
      char* buf = buf_sb;
#ifdef G__VISUAL
      // VC++ has problem in typeid(x).name(), so every
      // thrown exception is translated to G__exception.
      sprintf(buf, "new G__exception(\"%s\")", x.what());
      G__fprinterr(G__serr, "Exception: %s\n", x.what());
#else // G__VISUAL
      G__StrBuf buf2_sb(G__ONELINE);
      char* buf2 = buf2_sb;
      if (G__DemangleClassname(buf2, typeid(x).name())) {
         sprintf(buf, "new %s(*(%s*)%ld)", buf2, buf2, (long)(&x));
         G__fprinterr(G__serr, "Exception %s: %s\n", buf2, x.what());
      }
      else {
         sprintf(buf, "new G__exception(\"%s\",\"%s\")", x.what(), buf2); // TODO: why buf2?!
      }
#endif // G__VISUAL
      G__exceptionbuffer = G__getexpr(buf);
      G__exceptionbuffer.ref = G__exceptionbuffer.obj.i;
      G__return = G__RETURN_TRY;
      G__no_exec = 1;
   }
#endif // G__STD_EXCEPTION
   catch (int x) {
      G__letint(&G__exceptionbuffer, 'i', (long)x);
      G__exceptionbuffer.ref = (long)(&x);
      G__return = G__RETURN_TRY;
      G__no_exec = 1;
   }
   catch (long x) {
      G__letint(&G__exceptionbuffer, 'l', (long)x);
      G__exceptionbuffer.ref = (long)(&x);
      G__return = G__RETURN_TRY;
      G__no_exec = 1;
   }
   catch (void *x) {
      G__letint(&G__exceptionbuffer, 'Y', (long)x);
      G__exceptionbuffer.ref = (long)(&x);
      G__return = G__RETURN_TRY;
      G__no_exec = 1;
   }
   catch (float x) {
      G__letdouble(&G__exceptionbuffer, 'f', (double)x);
      G__exceptionbuffer.ref = (long)(&x);
      G__return = G__RETURN_TRY;
      G__no_exec = 1;
   }
   catch (double x) {
      G__letdouble(&G__exceptionbuffer, 'd', x);
      G__exceptionbuffer.ref = (long)(&x);
      G__return = G__RETURN_TRY;
      G__no_exec = 1;
   }
   catch (std::string x) {
      G__fprinterr(G__serr, "Exception: %s\n", x.c_str());
      G__genericerror(0);
      //G__return = G__RETURN_TRY;
      //G__no_exec = 1;
   }
   catch (...) {
      if (G__catchexception == 2) {
         G__fprinterr(G__serr, "Error: Exception caught in compiled code\n");
         exit(EXIT_FAILURE);
      }
      G__genericerror("Error: C++ exception caught");
   }
   return 0;
#endif //ENABLE_CPP_EXCEPTIONS
}
#endif // G__EXCEPTIONWRAPPER

//______________________________________________________________________________
void Cint::Internal::G__initcxx()
{
   // Create compiler feature test macros.
   //--
#if defined(__HP_aCC)||defined(__SUNPRO_CC)||defined(__BCPLUSPLUS__)||defined(__KCC)||defined(__INTEL_COMPILER)
   G__StrBuf temp_sb(G__ONELINE);
   char *temp = temp_sb;
#endif // __HP_aCC || __SUNPRO_CC || __BCPLUSPLUS__ || __KCC || __INTEL_COMPILER
#ifdef __HP_aCC // HP aCC C++ compiler
   sprintf(temp, "G__HP_aCC=%ld", (long) __HP_aCC);
   G__add_macro(temp);
#if __HP_aCC > 15000
   sprintf(temp, "G__ANSIISOLIB=1");
   G__add_macro(temp);
#endif // __HP_aCC > 15000
#endif // __HP_aCC
#ifdef __SUNPRO_CC // Sun C++ compiler
   sprintf(temp, "G__SUNPRO_CC=%ld", (long) __SUNPRO_CC);
   G__add_macro(temp);
#endif // __SUNPRO_CC
#ifdef __BCPLUSPLUS__  // Borland C++ compiler
   sprintf(temp, "G__BCPLUSPLUS=%ld", (long) __BCPLUSPLUS__);
   G__add_macro(temp);
#endif // __BCPLUSPLUS__
#ifdef __KCC // KCC C++ compiler
   sprintf(temp, "G__KCC=%ld", (long) __KCC);
   G__add_macro(temp);
#endif // __KCC
#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 810) // icc and ecc C++ compilers
   sprintf(temp, "G__INTEL_COMPILER=%ld", (long) __INTEL_COMPILER);
   G__add_macro(temp);
#endif // __INTEL_COMPILER && (__INTEL_COMPILER < 810)
   // --
}

//______________________________________________________________________________
void Cint::Internal::G__init_replacesymbol()
{
   G__get_symbolmacro().clear();
}

//______________________________________________________________________________
void Cint::Internal::G__add_replacesymbol(const char* s1, const char* s2)
{
   map<string, string>::value_type x(s1, s2);
   G__get_symbolmacro().insert(x);
}

//______________________________________________________________________________
const char* Cint::Internal::G__replacesymbol(const char* s)
{
   map<string, string>::iterator pos = G__get_symbolmacro().find(s);
   if (pos != G__get_symbolmacro().end()) {
      // This assumes the iterator does not copy the string object inside
      // the map (which is a static variable of the function G__get_symbolmacro).
      return pos->second.c_str();
   }
   return s;
}

//______________________________________________________________________________
int Cint::Internal::G__display_replacesymbol(FILE* fout, const char* name)
{
   G__StrBuf msg_sb(G__LONGLINE);
   char* msg = msg_sb;
   for (map<string, string>::iterator i = G__get_symbolmacro().begin(); i != G__get_symbolmacro().end(); ++i) {
      if (!name || !name[0] || !strcmp(name, i->first.c_str())) {
         sprintf(msg, "#define %s %s\n", i->first.c_str(), i->second.c_str());
         G__more(fout, msg);
         if (name && name[0]) {
            return 1;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
//______________________________________________________________________________
//
//  Implementation of class Cint::G__SourceFileInfo.
//

//______________________________________________________________________________
void Cint::G__SourceFileInfo::Init(const char* fname)
{
   for (filen = 0; filen < G__nfile; ++filen) {
      if (!strcmp(fname, G__srcfile[filen].filename)) {
         return;
      }
   }
}
//______________________________________________________________________________
const char* Cint::G__SourceFileInfo::Name()
{
   if (!IsValid()) {
      return 0;
   }
   return G__srcfile[filen].filename;
}

//______________________________________________________________________________
const char* Cint::G__SourceFileInfo::Prepname()
{
   if (!IsValid()) {
      return 0;
   }
   return G__srcfile[filen].prepname;
}

//______________________________________________________________________________
FILE* Cint::G__SourceFileInfo::fp()
{
   if (!IsValid()) {
      return 0;
   }
   return G__srcfile[filen].fp;
}

//______________________________________________________________________________
int Cint::G__SourceFileInfo::MaxLine()
{
   if (!IsValid()) {
      return 0;
   }
   return G__srcfile[filen].maxline;
}

//______________________________________________________________________________
Cint::G__SourceFileInfo& Cint::G__SourceFileInfo::IncludedFrom()
{
   static G__SourceFileInfo x;
   x.filen = -1;
   if (IsValid()) {
      x.filen = G__srcfile[filen].included_from;
   }
   return x;
}

//______________________________________________________________________________
long Cint::G__SourceFileInfo::Property()
{
   return 0;
}

//______________________________________________________________________________
int Cint::G__SourceFileInfo::IsValid()
{
   if ((filen < 0) || (filen >= G__nfile)) {
      return 0;
   }
   return 1;
}

//______________________________________________________________________________
int Cint::G__SourceFileInfo::Next()
{
   ++filen;
   while (!G__srcfile[filen].hash && IsValid()) {
      ++filen;
   }
   return IsValid();
}

//______________________________________________________________________________
//______________________________________________________________________________
//
//  Implementation of class Cint::G__IncludePathInfo.
//

//______________________________________________________________________________
const char* Cint::G__IncludePathInfo::Name()
{
   if (!IsValid()) {
      return 0;
   }
   return p->pathname;
}

//______________________________________________________________________________
long Cint::G__IncludePathInfo::Property()
{
   return 0;
}

//______________________________________________________________________________
int Cint::G__IncludePathInfo::IsValid()
{
   if (p && p->pathname) {
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
int Cint::G__IncludePathInfo::Next()
{
   if (!p) {
      p = &G__ipathentry;
   }
   else {
      p = p->next;
   }
   return IsValid();
}

