/* @(#)root/base:$Name:  $:$Id: Rtypes.h,v 1.15 2002/02/26 11:11:19 brun Exp $ */

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

#include <stdio.h>
#include "Rtypeinfo.h"



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

#ifndef __CINT__
template <class Tmpl> TBuffer &operator>>(TBuffer &buf, Tmpl *&obj)
{
   // Read TObject derived classes from a TBuffer. Need to provide
   // custom version for non-TObject derived classes. The const
   // version below is correct for any class.

   // This implementation only works for classes inheriting from
   // TObject.  This enables a clearer error message from the compiler.
   const TObject *verify = obj; if (verify) { }
   obj = (Tmpl *) buf.ReadObject(Tmpl::Class());
   return buf;
}
#else
template <class Tmpl> TBuffer &operator>>(TBuffer &buf, Tmpl *&obj);
#endif

// This might get used if we implement set a class version.
// template <class RootClass> Short_t GetClassVersion(RootClass *);

namespace ROOT {
   // NOTE: Cint typeid is not fully functional yet, so these classes can not
   // be made available yet.
   template <class T> TClass *IsA(T *obj) { return gROOT->GetClass(typeid(*obj)); }
   template <class T> TClass *IsA(const T *obj) { return IsA((T*)obj); }

   template <class RootClass> class ClassInfo;

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

   // This function is only implemented in the dictionary file.
   // The parameter is 'only' for overloading resolution.
   template <class T> ClassInfo<T> &GenerateInitInstance(const T*);

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
      // the TClass creation.  To modify the default behavior, one would 
      // inherit from this class and overload ROOT::DefineBehavior.
      // Set TQObject.h and star/inc/Ttypes.h for examples.
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
   
   template <class RootClass > class ClassInfo {
      // This class is the static registry of all information related to 
      // a Class.  It is filled from code generated in dictionary file and
      // by calls to ClassImp.
   public:
      typedef void (*ShowMembersFunc_t)(RootClass *obj, TMemberInspector &R__insp, 
                                        char *R__parent);
   protected:
#ifdef R__WIN32
     friend ClassInfo<RootClass > &GenerateInitInstance(const RootClass*);
#else
     friend ClassInfo<RootClass > &GenerateInitInstance<RootClass >(const RootClass*);
#endif
     
     static const InitBehavior  *fgAction;
     static TClass              *fgClass;
     static Int_t                fgVersion;
     static const char          *fgClassName;
     static const char          *fgImplFileName;
     static Int_t                fgImplFileLine;
     static const char          *fgDeclFileName;
     static Int_t                fgDeclFileLine;
     static ShowMembersFunc_t    fgShowMembers;
     
     ClassInfo(const char *fullClassname,
               const char *declFilename, Int_t declFileline,
               ShowMembersFunc_t showmembers, Int_t pragmabits) {
        // The basic type global varible are initialized to 0
        Int_t version = 1; // This is the default version number. 
        if (fgVersion !=0) version = fgVersion;
        Init(fullClassname, version, 
             declFilename, declFileline,
             showmembers, pragmabits);
     }
     ClassInfo(const char *fullClassname, Int_t version,
               const char *declFilename, Int_t declFileline,
               ShowMembersFunc_t showmembers, Int_t pragmabits) {
        Init(fullClassname, version, 
             declFilename, declFileline,
             showmembers, pragmabits);
     }
     
     void Init(const char *fullClassname, Int_t version,
               const char *declFilename, Int_t declFileline,
               ShowMembersFunc_t showmembers, Int_t pragmabits) {
        GetAction().Register(fullClassname,
                             version,
                             typeid(RootClass),
                             &Dictionary,
                             pragmabits);
        fgShowMembers = showmembers;
        fgVersion = version;
        fgClassName = fullClassname;
        fgDeclFileName = declFilename;
        fgDeclFileLine = declFileline;
     }
     
  public:
     ~ClassInfo() { GetAction().Unregister(GetClassName()); }

     static const InitBehavior &GetAction() {
        if (!fgAction) {
           RootClass *ptr = 0;
           fgAction = DefineBehavior(ptr, ptr);
        }
        return *fgAction;
     }

     static void Dictionary() { GetClass(); }
     
     static TClass *GetClass() {
        if (!fgClass) {
           GenerateInitInstance((const RootClass*)0x0);
           fgClass = GetAction().CreateClass(GetClassName(),
                                             GetVersion(),
                                             typeid(RootClass),
                                             &IsA,
                                             &ShowMembers,
                                             GetDeclFileName(),
                                             GetImplFileName(),
                                             GetDeclFileLine(),
                                             GetImplFileLine());
        }
        return fgClass;
     }

     static const char *GetClassName() {
        return fgClassName;
     }
     
     static ShowMembersFunc_t GetShowMembers() {
        return fgShowMembers;
     }

     static Short_t SetVersion(Short_t version) {
        ROOT::ResetClassVersion(fgClass, GetClassName(),version);
        fgVersion = version;
        return version;
     }
     
     static void SetFromTemplate() {
        TNamed *info = ROOT::RegisterClassTemplate(GetClassName(), 0, 0);
        if (info) SetImplFile(info->GetTitle(), info->GetUniqueID());
     }
     
     static int SetImplFile(const char *file, Int_t line) {
        fgImplFileName = file;
        fgImplFileLine = line;
        return 0;
     }
     
     static const char *GetDeclFileName() {
        return fgDeclFileName;
     }
     
     static Int_t GetDeclFileLine() {
        return fgDeclFileLine;
     }
     
     static const char *GetImplFileName() {
        if (!fgImplFileName) SetFromTemplate();
        return fgImplFileName;
     }
     
     static Int_t GetImplFileLine() {
        if (!fgImplFileLine) SetFromTemplate(); 
        return fgImplFileLine;
     }
     
     static Int_t GetVersion() {
        return fgVersion;
     }
     
     static void ShowMembers(RootClass *obj, TMemberInspector &R__insp, 
                             char *R__parent) {
        if (fgShowMembers) fgShowMembers(obj, R__insp, R__parent);
        // for now other part of the system seem to warn about this,
        // so we can just do as if the class was 'empty'
        // else
        //Error("R__tInit","ShowMembers not initialized for %s",GetClassName());
     }
     
     static TClass* IsA(const void *obj) {
        return ROOT::IsA( (RootClass*)obj );
     }
     
  protected:
     static void ShowMembers(void *obj, TMemberInspector &R__insp, 
                             char *R__parent) {
        if (fgShowMembers) fgShowMembers((RootClass*)obj,R__insp,R__parent);
        // for now other part of the system seem to warn about this,
        // so we can just do as if the class was 'empty'
        // else
        //Error("R__tInit","ShowMembers not initialized for %s",GetClassName());
     }  
  };

  #if defined(__CINT__) && !defined(ROOT_Rtypes_In_Cint_Interpreter)
  #pragma endif
  #endif

} // End of namespace ROOT

#if !defined(R__ACCESS_IN_SYMBOL) || defined(__CINT__)

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

#else

#define ClassDef(name,id) \
private: \
   static TClass *fgIsA; \
public: \
   friend void ROOT__ShowMembersFunc(name *obj, TMemberInspector &R__insp, char *R__parent); \
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
static int _R__UNIQUE_(R__dummyint) = \
            ROOT::ClassInfo<name >::SetImplFile(__FILE__, __LINE__);

//---- ClassDefT macros for templates with one template argument ---------------
// ClassDefT  corresponds to ClassDef
// ClassDefT2 goes in the same header as ClassDefT but must be
//            outside the class scope
// ClassImpT  corresponds to ClassImp


// This ClassDefT is stricly redundant is a kept only for
// backward compatibility.  Using #define ClassDef ClassDefT in confusing
// cint parser.
#if !defined(R__ACCESS_IN_SYMBOL) || defined(__CINT__)

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

#else

#define ClassDefT(name,id) \
private: \
   static TClass *fgIsA; \
public: \
   friend void ROOT__ShowMembersFunc(name *obj, TMemberInspector &R__insp, char *R__parent); \
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


//---- Macro to set the class version of non instrumented class an implementation file -----

#define RootClassVersion(name, VersionNumber)                  \
   static Short_t _R__UNIQUE_(R__dummyVersionNumber) =         \
           ROOT::ClassInfo<name >::SetVersion( VersionNumber );


#endif
