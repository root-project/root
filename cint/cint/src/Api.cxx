/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file Api.cxx
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Api.h"
#include "common.h"

#include "bc_eh.h"
#include <string>
#include <stdexcept>

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
  G__value result = G__null;
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
void Cint::G__InitGetSpecialValue(G__pMethodSpecialValue pmethod) 
{
  G__GetSpecialObject
	=(G__value (*)(const char*,void**,void**))G__APIGetSpecialValue_layer1;
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
  pvalue->isconst = 0; // better than unitialized - we just don't know here.
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
void Cint::G__InitGetSpecialObject(G__pMethodSpecialObject pmethod) 
{
  G__LockCriticalSection();
  if (pmethod) {
     G__GetSpecialObject
        =(G__value (*)(const char*,void**,void**))G__APIGetSpecialObject_layer1;
  } else G__GetSpecialObject = 0;
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

void Cint::G__InitUpdateClassInfo(G__pMethodUpdateClassInfo pmethod)
{
   G__UserSpecificUpdateClassInfo = pmethod;
}

void* Cint::G__new_interpreted_object(int size) {
  char *p = new char[size];
  return((void*)p);
}

void Cint::G__delete_interpreted_object(void* p) {
  delete [] (char*)p;
}

/*********************************************************************
* Generate dictionary.
*********************************************************************/
static G__pGenerateDictionary G__GenerateDictionary = 0;
extern "C" int G__EnableAutoDictionary;
int G__EnableAutoDictionary = 1;

void Cint::G__InitGenerateDictionary( G__pGenerateDictionary gdict )
{
  //gdict will be a pointer to TCint_GenerateDictionary
  G__GenerateDictionary = gdict;
}

G__pGenerateDictionary Cint::G__GetGenerateDictionary()
{
  return G__EnableAutoDictionary ? G__GenerateDictionary : 0;
}

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
////////////////////////////////////////////////////////////////////
int G__SourceFileInfo::SerialNumber() {
   // Return the serial number of the G__srcfile 'state'.
   // Use this to detect if G__srcfile has changed.

   return G__srcfile_serial;
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

#if !defined(__hpux) || __HP_aCC >= 53000
using namespace std;
#endif

#ifdef G__EXCEPTIONWRAPPER
#ifdef G__STD_EXCEPTION
#include <exception>
#include <typeinfo>
#include <string>
#endif

#if defined(__GNUC__) && __GNUC__ >= 3
#define G__HAVE_CXA_DEMANGLE
#include <cxxabi.h>
#endif

/*********************************************************************
* G__DemangleClassname
*********************************************************************/
static int G__DemangleClassname(G__FastAllocString& buf,const char *orig)
{
  int tagnum;

  /* try typeid.name() as is */
  buf = orig;
  tagnum = G__defined_tagname(buf,2);
  if(-1!=tagnum) return(1);
  
  /* try eliminating digit at the beginning "9exception" -> "exception" 
   * this works for classes in global scope in g++ */
  int ox=0;
  while(isdigit(orig[ox])) ++ox; 
  buf = orig+ox;
  tagnum = G__defined_tagname(buf,2);
  if(-1!=tagnum) return(1);

#ifndef G__HAVE_CXA_DEMANGLE
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
      buf += "::";
      totallen += (2+len);
    }
    else {
      totallen=len;
    }
    buf += orig+ox;
    buf[totallen] = 0;
    ox += len;
  }
  tagnum = G__defined_tagname(buf,2);
  if(-1!=tagnum) return(1);
#else
  int status = 0;
  char* cxaname=::abi::__cxa_demangle(orig, 0, 0, &status);
  buf = cxaname;
  free(cxaname);    
  tagnum = G__defined_tagname(buf,2);
  if(-1!=tagnum) return(1);
#endif

  /* Give up and settle with G__exception */
  return(0);
  
}

/*********************************************************************
* G__ExceptionWrapper
*********************************************************************/
int Cint::G__ExceptionWrapper(G__InterfaceMethod funcp
				   ,G__value* result7
				   ,char* funcname
				   ,struct G__param *libp
				   ,int hash)
{

  if(!G__catchexception) {

    // Stub Calling
    return (*funcp)(result7,funcname,libp,hash);
   
  }
#if ENABLE_CPP_EXCEPTIONS
  try {
#endif //ENABLE_CPP_EXCEPTIONS

	// Stub Calling
	return (*funcp)(result7,funcname,libp,hash);

#if ENABLE_CPP_EXCEPTIONS
  }
  catch(G__bc_exception& /* x */) {
    throw;
  }
#ifdef G__STD_EXCEPTION
  catch(std::exception& x) {
    G__FastAllocString buf(G__LONGLINE);
#ifdef G__VISUAL
    // VC++ has problem in typeid(x).name(), so every thrown exception is
    // translated to G__exception.
    buf.Format("new G__exception(\"%s\")",x.what());
    G__fprinterr(G__serr,"Exception: %s\n",x.what());
#else
    G__FastAllocString buf2(G__ONELINE);
    if(G__DemangleClassname(buf2,typeid(x).name())) {
       buf.Format("new %s(*(%s*)%ld)",buf2(),buf2(),(long)(&x));
       G__fprinterr(G__serr,"Exception %s: %s\n", buf2(), x.what());
    }
    else {
       if (G__defined_tagname("G__exception", 2) != -1) {
          buf.Format("new G__exception(\"%s\",\"CINT forwarded std::exception\")",x.what());
       } else {
          throw; // rethrow, better than nothing.
       }
    }
#endif
    G__exceptionbuffer = G__getexpr(buf);
    G__exceptionbuffer.ref = G__exceptionbuffer.obj.i;
    G__return = G__RETURN_TRY;
    G__no_exec = 1;

    // change from pointer to reference
    if (isupper(G__exceptionbuffer.type)) {
       G__exceptionbuffer.type = tolower(G__exceptionbuffer.type);
    }
  }
#endif

#define G__SETEXCPBUF_INT(TYPE, CTYPE)          \
  catch(TYPE x) { \
    TYPE* exc_x = new TYPE(x); \
    G__letint(&G__exceptionbuffer,CTYPE,(long)x);     \
    G__exceptionbuffer.ref = (long)(exc_x); \
    G__return = G__RETURN_TRY; \
    G__no_exec = 1; \
  }
  G__SETEXCPBUF_INT(int,   'i')
  G__SETEXCPBUF_INT(long,  'l')
  G__SETEXCPBUF_INT(void*, 'Y')
#undef G__SETEXCPBUF_INT

#define G__SETEXCPBUF_DBL(TYPE, CTYPE)          \
  catch(TYPE x) { \
    TYPE* exc_x = new TYPE(x); \
    G__letdouble(&G__exceptionbuffer,CTYPE,x);     \
    G__exceptionbuffer.ref = (long)(exc_x); \
    G__return = G__RETURN_TRY; \
    G__no_exec = 1; \
  }
  G__SETEXCPBUF_DBL(float,  'f')
  G__SETEXCPBUF_DBL(double, 'd')
#undef G__SETEXCPBUF_DBL

  catch(std::string x) {
    G__fprinterr(G__serr,"Exception: %s\n",x.c_str());
    G__genericerror((char*)NULL);
    //G__return = G__RETURN_TRY;
    //G__no_exec = 1;
  }
  catch(...) {
    if(2==G__catchexception) {
      G__fprinterr(G__serr,"Error: Exception caught in compiled code\n");
      throw std::runtime_error("CINT: Exception caught in compiled code");
    }
    //G__genericerror("Error: C++ exception caught");
    if (G__defined_tagname("G__exception", 2) != -1) {
       G__FastAllocString buf("new G__exception(\"G__exception\",\"CINT forwarded exception in compiled code\")");
       G__exceptionbuffer = G__getexpr(buf);
       G__exceptionbuffer.ref = G__exceptionbuffer.obj.i;
       G__return = G__RETURN_TRY;
       G__no_exec = 1;
    } else {
       throw; // rethrow, better than nothing.
    }
  }
 return 0;
#endif //ENABLE_CPP_EXCEPTIONS
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
  G__FastAllocString temp(G__ONELINE);
#endif
#ifdef __HP_aCC     /* HP aCC C++ compiler */
  temp.Format("G__HP_aCC=%ld",(long)__HP_aCC); G__add_macro(temp);
#if __HP_aCC > 15000
  temp.Format("G__ANSIISOLIB=1"); G__add_macro(temp);
#endif
#endif
#ifdef __SUNPRO_CC  /* Sun C++ compiler */
  temp.Format("G__SUNPRO_CC=%ld",(long)__SUNPRO_CC); G__add_macro(temp);
#endif
#ifdef __BCPLUSPLUS__  /* Borland C++ compiler */
  temp.Format("G__BCPLUSPLUS=%ld",(long)__BCPLUSPLUS__); G__add_macro(temp);
#endif
#ifdef __KCC        /* KCC  C++ compiler */
  temp.Format("G__KCC=%ld",(long)__KCC); G__add_macro(temp);
#endif
#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER<810) /* icc and ecc C++ compilers */
  temp.Format("G__INTEL_COMPILER=%ld",(long)__INTEL_COMPILER); G__add_macro(temp);
#endif
  /*
#ifdef __cplusplus 
  temp.Format("G__CPLUSPLUS=%ld",(long)__cplusplus); G__add_macro(temp);
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
  G__FastAllocString msg(G__LONGLINE);
  for(i=G__get_symbolmacro().begin();i!=G__get_symbolmacro().end();++i) {
    if(!name || !name[0] || strcmp(name,(*i).first.c_str())==0) {
       msg.Format("#define %s %s\n",(*i).first.c_str(),(*i).second.c_str());
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

