/* @(#)root/base:$Name:  $:$Id: Rtypes.h,v 1.18 2002/05/09 20:21:59 brun Exp $ */

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
#ifndef ROOT_DllImport
#include "DllImport.h"
#endif
#ifndef ROOT_Rtypeinfo
#include "Rtypeinfo.h"
#endif

#include <stdio.h>



//---- forward declared class types --------------------------------------------

class TClass;
class TBuffer;
class TMemberInspector;
class TObject;
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
typedef bool           Bool_t;      //Boolean (0=false, 1=true) (bool)
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

typedef void (*ShowMembersFunc_t)(void *obj, TMemberInspector &R__insp, char *R__parent);
typedef TClass *(*IsAFunc_t)(const void *obj);

// This is implemented in TBuffer.h
template <class Tmpl> TBuffer &operator>>(TBuffer &buf, Tmpl *&obj);

// This might get used if we implement setting a class version.
// template <class RootClass> Short_t GetClassVersion(RootClass *);

namespace ROOT {

   class GenericClassInfo;
   template <class RootClass> Short_t SetClassVersion();

   extern TClass *CreateClass(const char *cname, Version_t id,
                              const type_info &info, IsAFunc_t isa,
                              ShowMembersFunc_t show,
                              const char *dfil, const char *ifil,
                              Int_t dl, Int_t il);
   extern void AddClass(const char *cname, Version_t id, const type_info &info,
                        VoidFuncPtr_t dict, Int_t pragmabits);
   extern void RemoveClass(const char *cname);
   extern void ResetClassVersion(TClass*, const char*, Short_t);

   extern TNamed *RegisterClassTemplate(const char *name,
                                        const char *file, Int_t line);


#if 0
   // This function is only implemented in the dictionary file.
   // The parameter is 'only' for overloading resolution.
   // Used to be a template <class T> GenericClassInfo *GenerateInitInstance(const T*);
   template <class T> GenericClassInfo *GetClassInfo(const T* t) {
      GenericClassInfo *GenerateInitInstance(const T*);
      return CreateInitInstance(t);
   };
#endif

   // Because of the template defined here, we have to insure that
   // CINT does not see this file twice, even if it is preprocessed by
   // an external preprocessor.
   #ifdef __CINT__
   #pragma define ROOT_Rtypes_In_Cint_Interpreter
   #endif
   #if defined(__CINT__) && !defined(ROOT_Rtypes_In_Cint_Interpreter)
   #pragma ifndef ROOT_Rtypes_For_Cint
   #pragma define ROOT_Rtypes_For_Cint
   #endif

   class InitBehavior {
      // This class defines the interface for the class registration and
      // the TClass creation. To modify the default behavior, one would
      // inherit from this class and overload ROOT::DefineBehavior().
      // See TQObject.h and star/inc/Ttypes.h for examples.
   public:
      virtual void Register(const char *cname, Version_t id, const type_info &info,
                            VoidFuncPtr_t dict, Int_t pragmabits) const = 0;
      virtual void Unregister(const char *classname) const = 0;
      virtual TClass *CreateClass(const char *cname, Version_t id,
                                  const type_info &info, IsAFunc_t isa,
                                  ShowMembersFunc_t show,
                                  const char *dfil, const char *ifil,
                                  Int_t dl, Int_t il) const = 0;
   };

   class DefaultInitBehavior : public InitBehavior {
   public:
      virtual void Register(const char *cname, Version_t id, const type_info &info,
                            VoidFuncPtr_t dict, Int_t pragmabits) const {
         ROOT::AddClass(cname, id, info, dict, pragmabits);
      }
      virtual void Unregister(const char *classname) const {
         ROOT::RemoveClass(classname);
      }
      virtual TClass *CreateClass(const char *cname, Version_t id,
                                  const type_info &info, IsAFunc_t isa,
                                  ShowMembersFunc_t show,
                                  const char *dfil, const char *ifil,
                                  Int_t dl, Int_t il) const {
         return ROOT::CreateClass(cname, id, info, isa, show, dfil, ifil, dl, il);
      }
   };

   class GenericClassInfo {
      // This class in not inlined because it is used is non time critical
      // section (the dictionaries) and inline would lead to too much
      // repetition of the code (once per class!).

      const InitBehavior  *fAction;
      TClass              *fClass;
      const char          *fClassName;
      const char          *fDeclFileName;
      Int_t                fDeclFileLine;
      VoidFuncPtr_t        fDictionary;
      const type_info     &fInfo;
      const char          *fImplFileName;
      Int_t                fImplFileLine;
      IsAFunc_t            fIsA;
      void                *fShowMembers;
      Int_t                fVersion;

   public:
      GenericClassInfo(const char *fullClassname,
                       const char *declFileName, Int_t declFileLine,
                       const type_info &info, const InitBehavior *action,
                       void *showmembers, VoidFuncPtr_t dictionary,
                       IsAFunc_t isa, Int_t pragmabits);

      GenericClassInfo(const char *fullClassname, Int_t version,
                       const char *declFileName, Int_t declFileLine,
                       const type_info &info, const InitBehavior *action,
                       void *showmembers,  VoidFuncPtr_t dictionary,
                       IsAFunc_t isa, Int_t pragmabits);

      GenericClassInfo(const char *fullClassname, Int_t version,
                       const char *declFileName, Int_t declFileLine,
                       const type_info &info, const InitBehavior *action,
                       VoidFuncPtr_t dictionary, Int_t pragmabits);

      GenericClassInfo(const char *fullClassname, Int_t version,
                       const char *declFileName, Int_t declFileLine,
                       const type_info &info, const InitBehavior *action,
                       void *showmembers, VoidFuncPtr_t dictionary, Int_t pragmabits);

      void Init(Int_t pragmabits);
      ~GenericClassInfo();

      const InitBehavior &GetAction() const;
      TClass *GetClass();
      const char *GetClassName() const;
      const type_info &GetInfo() const;
      void *GetShowMembers() const;
      Short_t SetVersion(Short_t version);
      void SetFromTemplate();
      Int_t SetImplFile(const char *file, Int_t line);
      const char *GetDeclFileName() const;
      Int_t GetDeclFileLine() const;
      const char *GetImplFileName();
      Int_t GetImplFileLine();
      Int_t GetVersion() const;
      TClass *IsA(const void *obj);
      IsAFunc_t GetIsA() const;
   };

  #if defined(__CINT__) && !defined(ROOT_Rtypes_In_Cint_Interpreter)
  #pragma endif
  #endif

} // End of namespace ROOT

// Common part of ClassDef definition.
// ImplFileLine() is not part of it since CINT uses that as trigger for
// the class comment string.
#define _ClassDef_(name,id) \
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
   static const char *ImplFileName();


#if !defined(R__CONCRETE_INPUT_OPERATOR)
#if !defined(R__ACCESS_IN_SYMBOL) || defined(__CINT__)

#define ClassDef(name,id) \
private: \
   static TClass *fgIsA; \
public: \
   _ClassDef_(name,id) \
   static int ImplFileLine();

#else

#define ClassDef(name,id) \
private: \
   static TClass *fgIsA; \
public: \
   friend void ROOT__ShowMembersFunc(name *obj, TMemberInspector &R__insp, \
                                     char *R__parent); \
   _ClassDef_(name,id) \
   static int ImplFileLine();

#endif

#else

#define ClassDef(name,id) \
private: \
   static TClass *fgIsA; \
public: \
   friend TBuffer &operator>>(TBuffer &buf, name *&obj); \
   friend TBuffer &operator>>(TBuffer &buf, const name *&obj); \
   _ClassDef_(name,id) \
   static int ImplFileLine();

#endif


#define ClassImp(name) \
namespace ROOT { \
   GenericClassInfo *GenerateInitInstance(const name*); \
   static int _R__UNIQUE_(R__dummyint) = \
            GenerateInitInstance((name*)0x0)->SetImplFile(__FILE__, __LINE__);  \
}

//---- ClassDefT macros for templates with one template argument ---------------
// ClassDefT  corresponds to ClassDef
// ClassDefT2 goes in the same header as ClassDefT but must be
//            outside the class scope
// ClassImpT  corresponds to ClassImp


// This ClassDefT is stricly redundant and is kept only for
// backward compatibility. Using #define ClassDef ClassDefT in confusing
// the CINT parser.
#if !defined(R__ACCESS_IN_SYMBOL) || defined(__CINT__)

#define ClassDefT(name,id) \
private: \
   static TClass *fgIsA; \
public: \
   _ClassDef_(name,id) \
   static int ImplFileLine();

#else

#define ClassDefT(name,id) \
private: \
   static TClass *fgIsA; \
public: \
   friend void ROOT__ShowMembersFunc(name *obj, TMemberInspector &R__insp, \
                                     char *R__parent); \
   _ClassDef_(name,id) \
   static int ImplFileLine();

#define newClassDefT2(name,Tmpl) \
   template <class Tmpl> \
   TBuffer &operator>>(TBuffer &buf, name<Tmpl> *&obj); \
   template <class Tmpl> \
   TBuffer &operator>>(TBuffer &buf, const name<Tmpl> *&obj) \
      { return operator>>(buf, (name<Tmpl> *&) obj); }

#endif

#define ClassDefT2(name,Tmpl)

#define templateClassImp(name) \
static TNamed *_R__UNIQUE_(R__dummyholder) = \
                ROOT::RegisterClassTemplate(_QUOTE_(name), __FILE__, __LINE__);

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


//---- Macro to set the class version of non instrumented classes --------------

#define RootClassVersion(name, VersionNumber) \
namespace ROOT { \
   GenericClassInfo *GenerateInitInstance(const name*); \
   static Short_t _R__UNIQUE_(R__dummyVersionNumber) = \
           GenerateInitInstance((name*)0x0)->SetVersion(VersionNumber); \
}

#endif
