/* @(#)root/base:$Name:  $:$Id: Rtypes.h,v 1.4 2000/11/21 15:57:21 brun Exp $ */

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

//---- types -------------------------------------------------------------------

typedef char           Char_t;      //Signed Character 1 byte
typedef unsigned char  UChar_t;     //Unsigned Character 1 byte
typedef short          Short_t;     //Signed Short integer 2 bytes
typedef unsigned short UShort_t;    //Unsigned Short integer 2 bytes
#ifdef R__INT16
typedef long           Int_t;       //Signed integer 4 bytes
typedef unsigned long  UInt_t;      //Unsigned integer 4 bytes
#else
typedef int            Int_t;       //Signed integer 4 bytes
typedef unsigned int   UInt_t;      //Unsigned integer 4 bytes
#endif
#ifdef R__B64    // Note: Long_t and ULong_t are currently not portable types
typedef int            Seek_t;      //File pointer
typedef long           Long_t;      //Signed long integer 8 bytes
typedef unsigned long  ULong_t;     //Unsigned long integer 8 bytes
#else
typedef int            Seek_t;      //File pointer
typedef long           Long_t;      //Signed long integer 4 bytes
typedef unsigned long  ULong_t;     //Unsigned long integer 4 bytes
#endif
typedef float          Float_t;     //Float 4 bytes
typedef double         Double_t;    //Float 8 bytes
typedef char           Text_t;      //General string
typedef unsigned char  Bool_t;      //Boolean (0=false, 1=true)
typedef unsigned char  Byte_t;      //Byte (8 bits)
typedef short          Version_t;   //Class version identifier
typedef const char     Option_t;    //Option string
typedef int            Ssiz_t;      //String size
typedef float          Real_t;      //TVector and TMatrix element type

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

class TClass;
class TBuffer;
class TMemberInspector;

extern TClass *CreateClass(const char *cname, Version_t id,
                           const char *dfil, const char *ifil,
                           Int_t dl, Int_t il);
extern void AddClass(const char *cname, Version_t id, VoidFuncPtr_t dict,
                     Int_t pragmabits);
extern void RemoveClass(const char *cname);

// Cleanup this mess once HP-UX CC has been phased out (1-1-2001)
#if defined(R__HPUX) && !defined(R__ACC)
#define _ClassInit_(name) \
   class R__Init { \
      public: \
         R__Init(Int_t pragmabits = 0) { \
            AddClass(name::Class_Name(), name::Class_Version(), \
                     &name::Dictionary, pragmabits); \
         } \
         ~R__Init() { \
            RemoveClass(name::Class_Name()); \
         } \
         void *operator new(size_t sz) { return ::operator new(sz); } \
         void operator delete(void *ptr) { ::operator delete(ptr); } \
   };
#else
#define _ClassInit_(name) \
   class R__Init { \
      public: \
         R__Init(Int_t pragmabits = 0) { \
            AddClass(name::Class_Name(), name::Class_Version(), \
                     &name::Dictionary, pragmabits); \
         } \
         ~R__Init() { \
            RemoveClass(name::Class_Name()); \
         } \
   };
#endif

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
   void StreamerNVirtual(TBuffer &b) {name::Streamer(b);} \
   friend TBuffer &operator>>(TBuffer &buf, name *&obj); \
   _ClassInit_(name) \
   static const char *DeclFileName() { return __FILE__; } \
   static int DeclFileLine() { return __LINE__; } \
   static const char *ImplFileName(); \
   static int ImplFileLine();

#define _ClassImp_(name) \
   TClass *name::Class() \
      { if (!fgIsA) name::Dictionary(); return fgIsA; } \
   const char *name::ImplFileName() { return __FILE__; } \
   int name::ImplFileLine() { return __LINE__; } \
   TClass *name::fgIsA = 0;

#define ClassImp(name) \
   void name::Dictionary() { \
      fgIsA = CreateClass(Class_Name(),   Class_Version(), \
                          DeclFileName(), ImplFileName(), \
                          DeclFileLine(), ImplFileLine()); \
   } \
   _ClassImp_(name)

#define ClassImp2(namespace,name) \
   ClassImp(name); \
   const char *namespace::name::Class_Name() { \
      if (strlen(_QUOTE_(namespace)) == 0) \
         return _QUOTE_(name); \
      else \
         return _QUOTE_(namespace) "::" _QUOTE_(name); \
   } \
   static namespace::name::R__Init _NAME2_(__gR__Init,name);

//---- ClassDefT macros for templates ------------------------------------------
// ClassDefT  corresponds to ClassDef
// ClassDefT2 goes in the same header as ClassDefT but must be
//            outside the class scope
// ClassImpT  corresponds to ClassImp

#define ClassDefT(name,id) \
private: \
   static TClass *fgIsA; \
public: \
   static TClass *Class(); \
   static const char *Class_Name(); \
   static Version_t Class_Version() { return id; } \
   static void Dictionary(); \
   virtual TClass *IsA() const { return name::Class(); } \
   virtual void ShowMembers(TMemberInspector &, char *); \
   virtual void Streamer(TBuffer &); \
   void StreamerNVirtual(TBuffer &b) {name::Streamer(b);} \
   static const char *DeclFileName() { return __FILE__; } \
   static int DeclFileLine() { return __LINE__; } \
   static const char *ImplFileName(); \
   static int ImplFileLine();

#define _ClassInitT_(name,Tmpl) \
   template <class Tmpl> class _NAME2_(R__Init,name) { \
      public: \
         _NAME2_(R__Init,name)(Int_t pragmabits) { \
            AddClass(name<Tmpl>::Class_Name(), \
                     name<Tmpl>::Class_Version(), \
                     &name<Tmpl>::Dictionary, pragmabits); \
         } \
         _NAME3_(~,R__Init,name)() { \
            RemoveClass(name<Tmpl>::Class_Name()); \
         } \
   };

#define ClassDefT2(name,Tmpl) \
   template <class Tmpl> TBuffer &operator>>(TBuffer &, name<Tmpl> *&); \
   _ClassInitT_(name,Tmpl)

#define _ClassImpT_(name,Tmpl) \
   template <class Tmpl> TClass *name<Tmpl>::Class() \
      { if (!fgIsA) name<Tmpl>::Dictionary(); return fgIsA; } \
   template <class Tmpl> const char *name<Tmpl>::ImplFileName() \
      { return __FILE__; } \
   template <class Tmpl> int name<Tmpl>::ImplFileLine() { return __LINE__; } \
   template <class Tmpl> TClass *name<Tmpl>::fgIsA = 0;

#define ClassImpT(name,Tmpl) \
   template <class Tmpl> void name<Tmpl>::Dictionary() { \
      fgIsA = CreateClass(Class_Name(),   Class_Version(), \
                          DeclFileName(), ImplFileName(), \
                          DeclFileLine(), ImplFileLine()); \
   } \
   _ClassImpT_(name,Tmpl)

#endif
