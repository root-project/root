/* @(#)root/base:$Id$ */

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
#include <string.h>
#include <snprintf.h>   // part of stdio.h on systems that have it
#include <strlcpy.h>    // part of string.h on systems that have it



//---- forward declared class types --------------------------------------------

class TClass;
class TBuffer;
class TDirectory;
class TMemberInspector;
class TObject;
class TNamed;
class TRootIOCtor;
class TCollection;
class TFileMergeInfo;
class TString;

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
typedef float          Float16_t;   //Float 4 bytes written with a truncated mantissa
typedef double         Double_t;    //Double 8 bytes
typedef double         Double32_t;  //Double 8 bytes in memory, written as a 4 bytes float
typedef char           Text_t;      //General string (char)
typedef bool           Bool_t;      //Boolean (0=false, 1=true) (bool)
typedef unsigned char  Byte_t;      //Byte (8 bits) (unsigned char)
typedef short          Version_t;   //Class version identifier (short)
typedef const char     Option_t;    //Option string (const char)
typedef int            Ssiz_t;      //String size (int)
typedef float          Real_t;      //TVector and TMatrix element type (float)
#if defined(R__WIN32) && !defined(__CINT__)
typedef __int64          Long64_t;  //Portable signed long integer 8 bytes
typedef unsigned __int64 ULong64_t; //Portable unsigned long integer 8 bytes
#else
typedef long long          Long64_t; //Portable signed long integer 8 bytes
typedef unsigned long long ULong64_t;//Portable unsigned long integer 8 bytes
#endif
typedef double         Axis_t;      //Axis values type (double)
typedef double         Stat_t;      //Statistics type (double)

typedef short          Font_t;      //Font number (short)
typedef short          Style_t;     //Style number (short)
typedef short          Marker_t;    //Marker number (short)
typedef short          Width_t;     //Line width (short)
typedef short          Color_t;     //Color number (short)
typedef short          SCoord_t;    //Screen coordinates (short)
typedef double         Coord_t;     //Pad world coordinates (double)
typedef float          Angle_t;     //Graphics angle (float)
typedef float          Size_t;      //Attribute size (float)

enum EColor { kWhite =0,   kBlack =1,   kGray=920,
              kRed   =632, kGreen =416, kBlue=600, kYellow=400, kMagenta=616, kCyan=432,
              kOrange=800, kSpring=820, kTeal=840, kAzure =860, kViolet =880, kPink=900 };

// There is several streamer concepts.
class TClassStreamer;   // Streamer functor for a class
class TMemberStreamer;  // Streamer functor for a data member
typedef void (*ClassStreamerFunc_t)(TBuffer&, void*);  // Streamer function for a class
typedef void (*MemberStreamerFunc_t)(TBuffer&, void*, Int_t); // Streamer function for a data member

// This class is used to implement proxy around collection classes.
class TVirtualCollectionProxy;

typedef void (*VoidFuncPtr_t)();  //pointer to void function


//---- constants ---------------------------------------------------------------

#ifndef NULL
#define NULL 0
#endif

#ifndef R__NULLPTR
#ifndef nullptr
#define nullptr 0
#endif
#endif

const Bool_t kTRUE  = true;
const Bool_t kFALSE = false;

const Int_t     kMaxUChar    = 256;
const Int_t     kMaxChar     = kMaxUChar >> 1;
const Int_t     kMinChar     = -kMaxChar - 1;

const Int_t     kMaxUShort   = 65534;
const Int_t     kMaxShort    = kMaxUShort >> 1;
const Int_t     kMinShort    = -kMaxShort - 1;

const UInt_t    kMaxUInt     = UInt_t(~0);
const Int_t     kMaxInt      = Int_t(kMaxUInt >> 1);
const Int_t     kMinInt      = -kMaxInt - 1;

const ULong_t   kMaxULong    = ULong_t(~0);
const Long_t    kMaxLong     = Long_t(kMaxULong >> 1);
const Long_t    kMinLong     = -kMaxLong - 1;

const ULong64_t kMaxULong64  = ULong64_t(~0LL);
const Long64_t  kMaxLong64   = Long64_t(kMaxULong64 >> 1);
const Long64_t  kMinLong64   = -kMaxLong64 - 1;

const size_t    kBitsPerByte = 8;
const Ssiz_t    kNPOS        = ~(Ssiz_t)0;


//--- bit manipulation ---------------------------------------------------------

#define BIT(n)       (1ULL << (n))
#define SETBIT(n,i)  ((n) |= BIT(i))
#define CLRBIT(n,i)  ((n) &= ~BIT(i))
#define TESTBIT(n,i) ((Bool_t)(((n) & BIT(i)) != 0))


//---- debug global ------------------------------------------------------------

R__EXTERN Int_t gDebug;


//---- ClassDef macros ---------------------------------------------------------

typedef void (*ShowMembersFunc_t)(void *obj, TMemberInspector &R__insp);
class TVirtualIsAProxy;
typedef TClass *(*IsAGlobalFunc_t)(const TClass*, const void *obj);

// TBuffer.h declares and implements the following 2 operators
template <class Tmpl> TBuffer &operator>>(TBuffer &buf, Tmpl *&obj);
template <class Tmpl> TBuffer &operator<<(TBuffer &buf, const Tmpl *obj);

// This might get used if we implement setting a class version.
// template <class RootClass> Short_t GetClassVersion(RootClass *);

namespace ROOT {

   class TGenericClassInfo;
   typedef void *(*NewFunc_t)(void *);
   typedef void *(*NewArrFunc_t)(Long_t size, void *arena);
   typedef void  (*DelFunc_t)(void *);
   typedef void  (*DelArrFunc_t)(void *);
   typedef void  (*DesFunc_t)(void *);
   typedef void  (*DirAutoAdd_t)(void *, TDirectory *);
   typedef Long64_t (*MergeFunc_t)(void *, TCollection *, TFileMergeInfo *);

   template <class RootClass> Short_t SetClassVersion(RootClass *);

   extern TClass *CreateClass(const char *cname, Version_t id,
                              const type_info &info, TVirtualIsAProxy *isa,
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
   // Used to be a template <class T> TGenericClassInfo *GenerateInitInstance(const T*);
   template <class T> TGenericClassInfo *GetClassInfo(const T* t) {
      TGenericClassInfo *GenerateInitInstance(const T*);
      return CreateInitInstance(t);
   };
#endif

   class TInitBehavior {
      // This class defines the interface for the class registration and
      // the TClass creation. To modify the default behavior, one would
      // inherit from this class and overload ROOT::DefineBehavior().
      // See TQObject.h and table/inc/Ttypes.h for examples.
   public:
      virtual ~TInitBehavior() { }
      virtual void Register(const char *cname, Version_t id, const type_info &info,
                            VoidFuncPtr_t dict, Int_t pragmabits) const = 0;
      virtual void Unregister(const char *classname) const = 0;
      virtual TClass *CreateClass(const char *cname, Version_t id,
                                  const type_info &info, TVirtualIsAProxy *isa,
                                  ShowMembersFunc_t show,
                                  const char *dfil, const char *ifil,
                                  Int_t dl, Int_t il) const = 0;
   };

   class TDefaultInitBehavior : public TInitBehavior {
   public:
      virtual void Register(const char *cname, Version_t id, const type_info &info,
                            VoidFuncPtr_t dict, Int_t pragmabits) const {
         ROOT::AddClass(cname, id, info, dict, pragmabits);
      }
      virtual void Unregister(const char *classname) const {
         ROOT::RemoveClass(classname);
      }
      virtual TClass *CreateClass(const char *cname, Version_t id,
                                  const type_info &info, TVirtualIsAProxy *isa,
                                  ShowMembersFunc_t show,
                                  const char *dfil, const char *ifil,
                                  Int_t dl, Int_t il) const {
         return ROOT::CreateClass(cname, id, info, isa, show, dfil, ifil, dl, il);
      }
   };

   const TInitBehavior *DefineBehavior(void * /*parent_type*/,
                                       void * /*actual_type*/);

} // End of namespace ROOT

// The macros below use TGenericClassInfo, so let's ensure it is included
#ifndef ROOT_TGenericClassInfo
#include "TGenericClassInfo.h"
#endif

// Common part of ClassDef definition.
// DeclFileLine() is not part of it since CINT uses that as trigger for
// the class comment string.
#define _ClassDef_(name,id) \
private: \
   static TClass *fgIsA; \
public: \
   static TClass *Class(); \
   static const char *Class_Name(); \
   static Version_t Class_Version() { return id; } \
   static void Dictionary(); \
   virtual TClass *IsA() const { return name::Class(); } \
   virtual void ShowMembers(TMemberInspector &insp); \
   virtual void Streamer(TBuffer &b); \
   void StreamerNVirtual(TBuffer &b) { name::Streamer(b); } \
   static const char *DeclFileName() { return __FILE__; } \
   static int ImplFileLine(); \
   static const char *ImplFileName();

// Version without any virtual functions.
#define _ClassDefNV_(name,id) \
private: \
static TClass *fgIsA; \
public: \
static TClass *Class(); \
static const char *Class_Name(); \
static Version_t Class_Version() { return id; } \
static void Dictionary(); \
TClass *IsA() const { return name::Class(); } \
void ShowMembers(TMemberInspector &insp); \
void Streamer(TBuffer &b); \
void StreamerNVirtual(TBuffer &b) { name::Streamer(b); } \
static const char *DeclFileName() { return __FILE__; } \
static int ImplFileLine(); \
static const char *ImplFileName();


#if !defined(R__CONCRETE_INPUT_OPERATOR)
#if !defined(R__ACCESS_IN_SYMBOL) || defined(__CINT__)

#define ClassDef(name,id) \
   _ClassDef_(name,id) \
   static int DeclFileLine() { return __LINE__; }

#define ClassDefNV(name,id) \
   _ClassDefNV_(name,id) \
   static int DeclFileLine() { return __LINE__; }

#else

#define ClassDef(name,id) \
   _ClassDef_(name,id) \
   friend void ROOT__ShowMembersFunc(name *obj, TMemberInspector &R__insp); \
   static int DeclFileLine() { return __LINE__; }

#define ClassDefNV(name,id) \
   _ClassDefNV_(name,id) \
   friend void ROOT__ShowMembersFunc(name *obj, TMemberInspector &R__insp); \
   static int DeclFileLine() { return __LINE__; }

#endif

#else

#define ClassDef(name,id) \
   _ClassDef_(name,id) \
   friend TBuffer &operator>>(TBuffer &buf, name *&obj); \
   friend TBuffer &operator>>(TBuffer &buf, const name *&obj); \
   static int DeclFileLine() { return __LINE__; }

#define ClassDefNV(name,id) \
   _ClassDefNV_(name,id) \
   friend TBuffer &operator>>(TBuffer &buf, name *&obj); \
   friend TBuffer &operator>>(TBuffer &buf, const name *&obj); \
   static int DeclFileLine() { return __LINE__; }

#endif

#define R__UseDummy(name) \
   class _NAME2_(name,_c) { public: _NAME2_(name,_c)() { if (name) { } } }


#if defined(__CINT__)
#define ClassImpUnique(name,key)
#define ClassImp(name)
#else
#define ClassImpUnique(name,key) \
   namespace ROOT { \
      TGenericClassInfo *GenerateInitInstance(const name*); \
      static int _R__UNIQUE_(_NAME2_(R__dummyint,key)) = \
         GenerateInitInstance((name*)0x0)->SetImplFile(__FILE__, __LINE__); \
      R__UseDummy(_R__UNIQUE_(_NAME2_(R__dummyint,key))); \
   }
#define ClassImp(name) ClassImpUnique(name,default)
#endif

// Macro for Namespace

#if defined(__CINT__)
#define NamespaceImpUnique(name,key)
#define NamespaceImp(name)
#else
#define NamespaceImpUnique(name,key) \
   namespace name { \
      namespace ROOT { \
         ::ROOT::TGenericClassInfo *GenerateInitInstance(); \
         static int _R__UNIQUE_(_NAME2_(R__dummyint,key)) = \
            GenerateInitInstance()->SetImplFile(__FILE__, __LINE__); \
         R__UseDummy(_R__UNIQUE_(_NAME2_(R__dummyint,key))); \
      } \
   }
#define NamespaceImp(name) NamespaceImpUnique(name,default)
#endif

//---- ClassDefT macros for templates with one template argument ---------------
// ClassDefT  corresponds to ClassDef
// ClassDefT2 goes in the same header as ClassDefT but must be
//            outside the class scope
// ClassImpT  corresponds to ClassImp


// This ClassDefT is stricly redundant and is kept only for
// backward compatibility. Using #define ClassDef ClassDefT is confusing
// the CINT parser.
#if !defined(R__ACCESS_IN_SYMBOL) || defined(__CINT__)

#define ClassDefT(name,id) \
   _ClassDef_(name,id) \
   static int DeclFileLine() { return __LINE__; }

#define ClassDefTNV(name,id) \
   _ClassDefNV_(name,id) \
   static int DeclFileLine() { return __LINE__; }


#else

#define ClassDefT(name,id) \
   _ClassDef_(name,id) \
   friend void ROOT__ShowMembersFunc(name *obj, TMemberInspector &R__insp); \
   static int DeclFileLine() { return __LINE__; }

#define ClassDefTNV(name,id) \
   _ClassDefNV_(name,id) \
   friend void ROOT__ShowMembersFunc(name *obj, TMemberInspector &R__insp); \
   static int DeclFileLine() { return __LINE__; }

#endif

#define ClassDefT2(name,Tmpl)



#if defined(__CINT__)
#define templateClassImpUnique(name,key)
#define templateClassImp(name)
#else
#define templateClassImpUnique(name,key) \
   namespace ROOT { \
      static TNamed *_R__UNIQUE_(_NAME2_(R__dummyholder,key)) = \
         ROOT::RegisterClassTemplate(_QUOTE_(name), __FILE__, __LINE__); \
      R__UseDummy(_R__UNIQUE_(_NAME2_(R__dummyholder,key))); \
   }
#define templateClassImp(name) templateClassImpUnique(name,default)
#endif

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

#define RootClassVersion(name,VersionNumber) \
namespace ROOT { \
   TGenericClassInfo *GenerateInitInstance(const name*); \
   static Short_t _R__UNIQUE_(R__dummyVersionNumber) = \
           GenerateInitInstance((name*)0x0)->SetVersion(VersionNumber); \
   R__UseDummy(_R__UNIQUE_(R__dummyVersionNumber)); \
}

#if defined(__CINT__)
#define RootStreamer(name,STREAMER)
#else
#define RootStreamer(name,STREAMER)                                  \
namespace ROOT {                                                     \
   TGenericClassInfo *GenerateInitInstance(const name*);             \
   static Short_t _R__UNIQUE_(R__dummyStreamer) =                    \
           GenerateInitInstance((name*)0x0)->SetStreamer(STREAMER);  \
   R__UseDummy(_R__UNIQUE_(R__dummyStreamer));                       \
}
#endif

#endif
