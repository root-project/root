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
   void StreamerNVirtual(TBuffer &b) { name::Streamer(b); } \
   friend TBuffer &operator>>(TBuffer &buf, name *&obj); \
   friend TBuffer &operator>>(TBuffer &buf, const name *&obj); \
   _ClassInit_(name) \
   static const char *DeclFileName() { return __FILE__; } \
   static int DeclFileLine() { return __LINE__; } \
   static const char *ImplFileName(); \
   static int ImplFileLine();

#define _ClassImp_(name) \
   TBuffer &operator>>(TBuffer &buf, const name *&obj) \
      { return operator>>(buf, (name *&) obj); } \
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

//---- ClassDefT macros for templates with one template argument ---------------
// ClassDefT  corresponds to ClassDef
// ClassDefT2 goes in the same header as ClassDefT but must be
//            outside the class scope
// ClassImpT  corresponds to ClassImp

template <class Tmpl> TBuffer &operator>>(TBuffer &buf, Tmpl *&obj);
template <class Tmpl> TBuffer &operator>>(TBuffer &buf, const Tmpl *&obj) {
   return operator>>(buf, (Tmpl *&) obj);
}

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
   void StreamerNVirtual(TBuffer &b) { name::Streamer(b); } \
   static const char *DeclFileName() { return __FILE__; } \
   static int DeclFileLine() { return __LINE__; } \
   static const char *ImplFileName(); \
   static int ImplFileLine();

#include "TNamed.h"
#include "THashTable.h"

inline TNamed* R__tInit__RegisterTemplate(const char *name, const char* file, int line) {
  static THashTable table;

  if (file) {
     TNamed * obj = new TNamed(name,file);
     obj->SetUniqueID(line);
     table.Add(obj);
     return obj;
  } else {
     return (TNamed*)table.FindObject(name);
  }

}
 
template <class RootClass> class R__tInit {
public:
   R__tInit(Int_t pragmabits) {
      AddClass(RootClass::Class_Name(),
               RootClass::Class_Version(),
               &RootClass::Dictionary,pragmabits);
   }
   static void SetFromTemplate() {
      TString name = RootClass::Class_Name();
      name.Remove(name.First('<'));
      TNamed *info = R__tInit__RegisterTemplate(name,0,0);
      if (info) {
         fgImplFileName = info->GetTitle();
         fgImplFileLine = info->GetUniqueID();
      }
   }
   static int SetImplFile(char *file, int line) {
     fgImplFileName = file;
     fgImplFileLine = line;
     return 0;
   }
   static const char* fgImplFileName; // Need to be instantiated
   static       int   fgImplFileLine;
   static const char* GetImplFileName() {
     if (fgImplFileName) return fgImplFileName;
     else { SetFromTemplate(); return fgImplFileName; };
   }
   static       int   GetImplFileLine() { 
      if (fgImplFileLine) return fgImplFileLine; 
      else { SetFromTemplate();  return fgImplFileLine; }
   }
   ~R__tInit() { RemoveClass(RootClass::Class_Name()); }
};
template <class T> const char * R__tInit<T >::fgImplFileName = 0; 
template <class T>          int R__tInit<T >::fgImplFileLine = 0;

#define ClassDefT2(name,Tmpl) \


#define R__join (one,two) R__join1(one,two)
#define R__join1(one,two) R__join2(one,two)
#define R__join2(one,two) one##two
#define R__unique(x) R__join1(x, __LINE__ )

#define ClassImpT(name,Tmpl) \
   TNamed* R__unique(R__dummyholder) (  R__tInit__RegisterTemplate(_QUOTE_(name), __FILE__, __LINE__) );

#define ClassImpTS(name) \
  int R__unique(R__dummyint) = R__tInit<name >::SetImplFile(__FILE__,__LINE__);



//---- ClassDefT macros for templates with two template arguments --------------
// ClassDef2T2 goes in the same header as ClassDefT but must be
//             outside the class scope
// ClassImp2T  corresponds to ClassImpT

#define _ClassInit2T_(name,Tmpl1,Tmpl2) \

#define ClassDef2T2(name,Tmpl1,Tmpl2) \
   template <class Tmpl1, class Tmpl2> \
   TBuffer &operator>>(TBuffer &buf, name<Tmpl1, Tmpl2> *&obj); \
   template <class Tmpl1, class Tmpl2> \
   TBuffer &operator>>(TBuffer &buf, const name<Tmpl1, Tmpl2> *&obj) \
      { return operator>>(buf, (name<Tmpl1, Tmpl2> *&) obj); } \
   _ClassInit2T_(name,Tmpl1,Tmpl2)

#define _ClassImp2T_(name,Tmpl1,Tmpl2) \
   template <class Tmpl1, class Tmpl2> \
   TClass *name<Tmpl1,Tmpl2>::Class() \
      { if (!fgIsA) name<Tmpl1,Tmpl2>::Dictionary(); return fgIsA; } \

#define ClassImp2T(name,Tmpl1,Tmpl2) \
   _ClassImp2T_(name,Tmpl1,Tmpl2)


//---- ClassDefT macros for templates with three template arguments ------------
// ClassDef3T2 goes in the same header as ClassDefT but must be
//             outside the class scope
// ClassImp3T  corresponds to ClassImpT

#define _ClassInit3T_(name,Tmpl1,Tmpl2,Tmpl3) \
   template <class Tmpl1, class Tmpl2, class Tmpl3> \
   class _NAME2_(R__Init,name) { \
      public: \
         _NAME2_(R__Init,name)(Int_t pragmabits) { \
            AddClass(name<Tmpl1,Tmpl2,Tmpl3>::Class_Name(), \
                     name<Tmpl1,Tmpl2,Tmpl3>::Class_Version(), \
                     &name<Tmpl1,Tmpl2,Tmpl3>::Dictionary, pragmabits); \
         } \
         _NAME2_(~R__Init,name)() { \
            RemoveClass(name<Tmpl1,Tmpl2,Tmpl3>::Class_Name()); \
         } \
   };

#define ClassDef3T2(name,Tmpl1,Tmpl2,Tmpl3) \
   template <class Tmpl1, class Tmpl2, class Tmpl3> \
   TBuffer &operator>>(TBuffer &buf, name<Tmpl1, Tmpl2, Tmpl3> *&obj); \
   template <class Tmpl1, class Tmpl2, class Tmpl3> \
   TBuffer &operator>>(TBuffer &buf, const name<Tmpl1, Tmpl2, Tmpl3> *&obj) \
      { return operator>>(buf, (name<Tmpl1, Tmpl2, Tmpl3> *&) obj); } \
   _ClassInit3T_(name,Tmpl1,Tmpl2,Tmpl3)

#define _ClassImp3T_(name,Tmpl1,Tmpl2,Tmpl3) \
   template <class Tmpl1, class Tmpl2, class Tmpl3> \
   TClass *name<Tmpl1,Tmpl2,Tmpl3>::Class() \
      { if (!fgIsA) name<Tmpl1,Tmpl2,Tmpl3>::Dictionary(); return fgIsA; } \
   template <class Tmpl1, class Tmpl2, class Tmpl3> \
   const char *name<Tmpl1,Tmpl2,Tmpl3>::ImplFileName() \
      { return __FILE__; } \
   template <class Tmpl1, class Tmpl2, class Tmpl3> \
   int name<Tmpl1,Tmpl2,Tmpl3>::ImplFileLine() { return __LINE__; } \
   template <class Tmpl1, class Tmpl2, class Tmpl3> \
   TClass *name<Tmpl1,Tmpl2,Tmpl3>::fgIsA = 0;

#define ClassImp3T(name,Tmpl1,Tmpl2,Tmpl3) \
   template <class Tmpl1, class Tmpl2, class Tmpl3> \
   void name<Tmpl1,Tmpl2,Tmpl3>::Dictionary() { \
      fgIsA = CreateClass(Class_Name(),   Class_Version(), \
                          DeclFileName(), ImplFileName(), \
                          DeclFileLine(), ImplFileLine()); \
   } \
   _ClassImp3T_(name,Tmpl1,Tmpl2,Tmpl3)

#endif
