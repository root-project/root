/* @(#)root/base:$Name:  $:$Id: Rtypes.h,v 1.13 2002/01/10 10:21:31 rdm Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Rtypes
#define ROOT_Rtypes

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Rtypes                                                               //
//                                                                      //
// Basic types used by ROOT.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_RConfig
#include "RConfig.h"
#endif

#include <stdio.h>

#include "DllImport.h"


//---- forward declared class types --------------------------------------------

class TClass;
class TBuffer;
class TMemberInspector;
class TNamed;

//---- types -------------------------------------------------------------------

typedef char           Char_t;      //Signed Character 1 byte (char)
typedef unsigned char  UChar_t;     //Unsigned Character 1 byte (unsigned char)
typedef short          Short_t;     //Signed Short integer 2 bytes (short)
typedef unsigned short UShort_t;    //Unsigned Short integer 2 bytes (unsigned short)
#ifdef R__INT16
typedef long           Int_t;       //Signed integer 4 bytes
typedef unsigned long  UInt_t;      //Unsigned integer 4 bytes
#else
typedef int            Int_t;       //Signed integer 4 bytes (int)
typedef unsigned int   UInt_t;      //Unsigned integer 4 bytes (unsigned int)
#endif
#ifdef R__B64    // Note: Long_t and ULong_t are currently not portable types
typedef int            Seek_t;      //File pointer (int)
typedef long           Long_t;      //Signed long integer 8 bytes (long)
typedef unsigned long  ULong_t;     //Unsigned long integer 8 bytes (unsigned long)
#else
typedef int            Seek_t;      //File pointer (int)
typedef long           Long_t;      //Signed long integer 4 bytes (long)
typedef unsigned long  ULong_t;     //Unsigned long integer 4 bytes (unsigned long)
#endif
typedef float          Float_t;     //Float 4 bytes (float)
typedef double         Double_t;    //Float 8 bytes (double)
typedef char           Text_t;      //General string (char)
typedef unsigned char  Bool_t;      //Boolean (0=false, 1=true) (unsigned char)
typedef unsigned char  Byte_t;      //Byte (8 bits) (unsigned char)
typedef short          Version_t;   //Class version identifier (short)
typedef const char     Option_t;    //Option string (const char)
typedef int            Ssiz_t;      //String size (int)
typedef float          Real_t;      //TVector and TMatrix element type (float)

typedef void         (*Streamer_t)(TBuffer&, void*, Int_t);
typedef void         (*VoidFuncPtr_t)();  //pointer to void function


//---- constants ---------------------------------------------------------------

#ifndef NULL
#define NULL 0
#endif

const Bool_t kTRUE   = 1;
const Bool_t kFALSE  = 0;

const Int_t  kMaxInt      = 2147483647;
const Int_t  kMaxShort    = 32767;
const size_t kBitsPerByte = 8;
const Ssiz_t kNPOS        = ~(Ssiz_t)0;


//--- bit manipulation ---------------------------------------------------------

#define BIT(n)       (1 << (n))
#define SETBIT(n,i)  ((n) |= BIT(i))
#define CLRBIT(n,i)  ((n) &= ~BIT(i))
#define TESTBIT(n,i) ((Bool_t)(((n) & BIT(i)) != 0))


//---- debug global ------------------------------------------------------------

R__EXTERN Int_t gDebug;


//---- ClassDef macros ---------------------------------------------------------

extern TClass *CreateClass(const char *cname, Version_t id,
                           const char *dfil, const char *ifil,
                           Int_t dl, Int_t il);
extern void AddClass(const char *cname, Version_t id, VoidFuncPtr_t dict,
                     Int_t pragmabits);
extern void RemoveClass(const char *cname);

extern TNamed* R__RegisterClassTemplate(const char *name, 
                                        const char* file, int line);
 

#ifndef R__WIN32
// MS does not support this syntax ... instead we just 
// rely on the friend statement :(

#ifndef __CINT__
template <class Tmpl> TBuffer &operator>>(TBuffer &buf, Tmpl *&obj) 
{
   // This is a function declaration to expose the current class
   // concrete implementation.  With this declaration, the return 
   // statement below will use the non-templated version of the 
   // operator>> which will be implemented either by the user or
   // by rootcint (in the dictionary file).

   TBuffer &operator>>(TBuffer &buf, Tmpl *&obj);

   return operator>>(buf,obj);
}
#else
template <class Tmpl> TBuffer &operator>>(TBuffer &buf, Tmpl *&obj);
#endif

template <class Tmpl> TBuffer &operator>>(TBuffer &buf, const Tmpl *&obj) {
   return operator>>(buf, (Tmpl *&) obj);
}

#else // This is for windows

#define R__STILL_RELY_ON_FRIEND

#endif

// Because of the template defined here, we have to insure that
// CINT does not see this file twice, even if it is preprocessed by
// an external preprocessor.
#ifdef __CINT__
#pragma define ROOT_Rtypes_In_Cint_Intepreter
#endif
#if defined(__CINT__) && !defined(ROOT_Rtypes_In_Cint_Intepreter)
#pragma ifndef ROOT_Rtypes_For_Cint
#pragma define ROOT_Rtypes_For_Cint
#endif


class R__InitBehavior {
public:
   virtual void Register(const char *cname, Version_t id, VoidFuncPtr_t dict,
                         Int_t pragmabits) const = 0;
   virtual void Unregister(const char* classname) const = 0;
   //   virtual void Dictionary() const = 0;   
   virtual TClass* CreateClass(const char *cname, Version_t id,
                               const char *dfil, const char *ifil,
                               Int_t dl, Int_t il) const = 0;
};

class R__DefaultInitBehavior : public R__InitBehavior {
public:
   virtual void Register(const char *cname, Version_t id, VoidFuncPtr_t dict,
                         Int_t pragmabits) const {
      AddClass(cname, id, dict, pragmabits);
   }
   virtual TClass* CreateClass(const char *cname, Version_t id,
                               const char *dfil, const char *ifil,
                               Int_t dl, Int_t il) const {
      return ::CreateClass(cname, id, dfil, ifil, dl, il);
      //Class_Name(),   Class_Version(),
      //                   DeclFileName(), ImplFileName(),
      //                   DeclFileLine(), ImplFileLine());
   }
   virtual void Unregister(const char* classname) const {
      RemoveClass(classname);
   }

};

#if 0
#include "TQObject.h"
class R__TQObjectInitBehavior : public R__DefaultInitBehavior {
public:
   virtual TClass* CreateClass(const char *cname, Version_t id,
                               const char *dfil, const char *ifil,
                               Int_t dl, Int_t il) const {
      return new TQClass(cname, id,dfil, ifil,dl, il);
      //Class_Name(),   Class_Version(),
      //                   DeclFileName(), ImplFileName(),
      //                   DeclFileLine(), ImplFileLine());
   }

};
#endif

template <class T> class R__TTableInitBehavior: public R__DefaultInitBehavior {
public:
#ifdef InitBuildStruct
   static const char* fgStructName; // Need to be instantiated
   virtual void Dictionary const (   
      // or 
      R__DefaultInitBehavior::Dictionary();
      std::string structname = fgStructName;
      structname += ".h";
      
      char *structBuf = new char[strlen(fgStructName)+2+2];
      strcpy(structBuf,fgStructName);
      strcat(structBuf,".h");
      char *s = strstr(structBuf,"_st.h");
      if (s) { *s = 0;  strcat(structBuf,".h"); }
      TClass *r = ::CreateClass(fgStructName, Class_Version(),
                                structBuf, structBuf, 1,  1 );
      fgColDescriptors = new TTableDescriptor(r);
   }
#else
   virtual TClass* CreateClass(const char *cname, Version_t id,
                               const char *dfil, const char *ifil,
                               Int_t dl, Int_t il) const {
      TClass * cl = R__DefaultInitBehavior::CreateClass(cname, id,dfil, ifil,dl, il);
      T::TableDictionary();
   }
   virtual void Dictionary() const {
      R__DefaultInitBehavior::Dictionary();
      T::TableDictionary();
   }
#endif
   virtual void Unregister(const char* classname) const {
      R__DefaultInitBehavior::Unregister(classname);
      R__DefaultInitBehavior::Unregister(structname);
   }
};
#ifdef InitBuildStruct
template <class T> const char * R__TTableInitBehavior<T >::fgStructName = 0;
#endif

template <class RootClass> class R__tInit {
public:
   static const R__InitBehavior *fgAction;
   static       TClass * fgClass;
   R__tInit(Int_t pragmabits) {
      GetAction().Register(RootClass::Class_Name(),
                        RootClass::Class_Version(),
                        &Dictionary,
                        pragmabits);
   }
   static const R__InitBehavior & GetAction() {
      if (fgAction == 0) {
         fgAction = R__DefineBehavior( (RootClass*)0x0, (RootClass*)0x0 );
      }
      return *fgAction;
   }
   static void Dictionary() { GetClass(); }
   static TClass* GetClass() {
      if (!fgClass) {
        fgClass = GetAction().CreateClass(RootClass::Class_Name(), 
                                          RootClass::Class_Version(),
                                          RootClass::DeclFileName(), 
                                          GetImplFileName(),
                                          RootClass::DeclFileLine(), 
                                          GetImplFileLine());
      }
      return fgClass;
   }
   ~R__tInit() { GetAction().Unregister(RootClass::Class_Name()); }

   static void SetFromTemplate() {
      TNamed *info = R__RegisterClassTemplate(RootClass::Class_Name(),0,0);
      if (info) SetImplFile(info->GetTitle(),info->GetUniqueID());
   }
   static int SetImplFile(const char *file, int line) {
     fgImplFileName = file;
     fgImplFileLine = line;
     return 0;
   }
   static const char* GetImplFileName() {
     if (fgImplFileName) return fgImplFileName;
     else { SetFromTemplate(); return fgImplFileName; };
   }
   static       int   GetImplFileLine() { 
      if (fgImplFileLine) return fgImplFileLine; 
      else { SetFromTemplate();  return fgImplFileLine; }
   }
   static const char* fgImplFileName; // Need to be instantiated
   static       int   fgImplFileLine;
};
template <class T>       TClass * R__tInit<T >::fgClass = 0;
template <class T> const char   * R__tInit<T >::fgImplFileName = 0; 
template <class T>          int   R__tInit<T >::fgImplFileLine = 0;
template <class T> const R__InitBehavior* R__tInit<T>::fgAction = 0; // R__tInit<T>::GetAction(); // R__DefineBehavior( (T*)0x0, (T*)0x0 );

inline const R__InitBehavior* R__DefineBehavior( void* parent_type, void* actual_type ) { return new R__DefaultInitBehavior(); }

class TTable;
template <class RootClass> 
const R__TTableInitBehavior<RootClass>* R__DefineBehavior( TTable*, RootClass*) { return new R__TTableInitBehavior<RootClass>(); }


#if defined(__CINT__) && !defined(ROOT_Rtypes_In_Cint_Intepreter)
#pragma endif
#endif

#ifdef R__STILL_RELY_ON_FRIEND

#define ClassDef(name,id) \
private: \
   static TClass *fgIsA; \
   friend R__tInit<name>; \
public: \
   static TClass *Class(); \
   static const char *Class_Name(); \
   static Version_t Class_Version() { return id; } \
   static void Dictionary(); \
   virtual TClass *IsA() const { return name::Class(); } \
   virtual void ShowMembers(TMemberInspector &insp, char *parent); \
   virtual void Streamer(TBuffer &b); \
   friend TBuffer &operator>>(TBuffer &buf, name *&obj); \
   friend TBuffer &operator>>(TBuffer &buf, const name *&obj); \
   void StreamerNVirtual(TBuffer &b) { name::Streamer(b); } \
   static const char *DeclFileName() { return __FILE__; } \
   static int DeclFileLine() { return __LINE__; } \
   static const char *ImplFileName(); \
   static int ImplFileLine();

#else

#define ClassDef(name,id) \
private: \
   static TClass *fgIsA; \
public: \
   static TClass *Class(); \
   static const char *Class_Name(); \
   static Version_t Class_Version() { return id; } \
   static void Dictionary(); \
   virtual TClass *IsA() const { return name::Class(); } \
   virtual void ShowMembers(TMemberInspector &insp, char *parent); \
   virtual void Streamer(TBuffer &b); \
   void StreamerNVirtual(TBuffer &b) { name::Streamer(b); } \
   static const char *DeclFileName() { return __FILE__; } \
   static int DeclFileLine() { return __LINE__; } \
   static const char *ImplFileName(); \
   static int ImplFileLine();

#endif

#define _ClassImp_(name) 

#define ClassImp(name) \
  static int _R__UNIQUE_(R__dummyint) = R__tInit<name >::SetImplFile(__FILE__,__LINE__);

//---- ClassDefT macros for templates with one template argument ---------------
// ClassDefT  corresponds to ClassDef
// ClassDefT2 goes in the same header as ClassDefT but must be
//            outside the class scope
// ClassImpT  corresponds to ClassImp


// For now we keep ClassDefT a simple macro to avoid cint parser related issues.
#ifdef R__STILL_RELY_ON_FRIEND

#define ClassDefT(name,id) \
private: \
   static TClass *fgIsA; \
public: \
   static TClass *Class(); \
   static const char *Class_Name(); \
   static Version_t Class_Version() { return id; } \
   static void Dictionary(); \
   virtual TClass *IsA() const { return name::Class(); } \
   virtual void ShowMembers(TMemberInspector &insp, char *parent); \
   virtual void Streamer(TBuffer &b); \
   friend TBuffer &operator>>(TBuffer &buf, name *&obj); \
   friend TBuffer &operator>>(TBuffer &buf, const name *&obj); \
   void StreamerNVirtual(TBuffer &b) { name::Streamer(b); } \
   static const char *DeclFileName() { return __FILE__; } \
   static int DeclFileLine() { return __LINE__; } \
   static const char *ImplFileName(); \
   static int ImplFileLine();

#else

#define ClassDefT(name,id) \
private: \
   static TClass *fgIsA; \
public: \
   static TClass *Class(); \
   static const char *Class_Name(); \
   static Version_t Class_Version() { return id; } \
   static void Dictionary(); \
   virtual TClass *IsA() const { return name::Class(); } \
   virtual void ShowMembers(TMemberInspector &insp, char *parent); \
   virtual void Streamer(TBuffer &b); \
   void StreamerNVirtual(TBuffer &b) { name::Streamer(b); } \
   static const char *DeclFileName() { return __FILE__; } \
   static int DeclFileLine() { return __LINE__; } \
   static const char *ImplFileName(); \
   static int ImplFileLine();

#endif

#define ClassDefT2(name,Tmpl) 

#define templateClassImp(name) \
static TNamed* _R__UNIQUE_(R__dummyholder) = R__RegisterClassTemplate(_QUOTE_(name), __FILE__, __LINE__);

#define ClassImpT(name,Tmpl) templateClassImp(name)

//---- ClassDefT macros for templates with two template arguments --------------
// ClassDef2T2 goes in the same header as ClassDefT but must be
//             outside the class scope
// ClassImp2T  corresponds to ClassImpT

#define ClassDef2T2(name,Tmpl1,Tmpl2) 
#define ClassImp2T(name,Tmpl1,Tmpl2) templateClassImp(name)


//---- ClassDefT macros for templates with three template arguments ------------
// ClassDef3T2 goes in the same header as ClassDefT but must be
//             outside the class scope
// ClassImp3T  corresponds to ClassImpT

#define ClassDef3T2(name,Tmpl1,Tmpl2,Tmpl3) 
#define ClassImp3T(name,Tmpl1,Tmpl2,Tmpl3) templateClassImp(name)

#endif
