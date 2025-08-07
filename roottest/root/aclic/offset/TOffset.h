// @(#)root/base:$Id$
// Author: Victor Perev   08/05/02

#ifndef ROOT_TOffset
#define ROOT_TOffset


#include "Rtypes.h"

class TList;
class TObject;
class TClass;
class TOffset {

public:
   TOffset(TClass *cl,Int_t all=0);
  ~TOffset();
   TList *GetOffsetList() const {return fOffsetList;}
   Int_t GetSize() const {return fSize;}
   Int_t GetUsed() const {return fUsed;}
   Int_t GetVirt() const {return fVirt;}
   Int_t GetMult() const {return fMult;}
   Int_t GetTail() const {return fTail;}
   const char *GetName() const ;
   void  Print(const char* opt="") const;
   Int_t GetOffset(const char *name) const;

   static void Test();
   static Int_t Aligne(Int_t off,Int_t mult)
   {Int_t r = off%mult; if (r) off += mult - r; return off;}

private:
   void Init();
   void Virt();
   void DoIt();
   void Adopt(Int_t all);


//  	static members

   static Int_t fgAlign[16];
   static Int_t fgSize [16];
   static Int_t fgWhereVirt;
   static Int_t fgSolBug;

   enum EAlign {
     kAlignChar,
     kAlignShort,
     kAlignInt,
     kAlignLong,
     kAlignPoint,
     kAlignFloat,
     kAlignDouble,
     kAlignBool,
     kAlignVirt,
     kAlignVector,
     kAlignList,
     kAlignDeque,
     kAlignMap,
     kAlignMultiMap,
     kAlignSet,
     kAlignMultiSet
   };


   enum ESize {
     kSizeChar,
     kSizeShort,
     kSizeInt,
     kSizeLong,
     kSizePoint,
     kSizeFloat,
     kSizeDouble,
     kSizeBool,
     kSizeVirt,
     kSizeVector,
     kSizeList,
     kSizeDeque,
     kSizeMap,
     kSizeMultiMap,
     kSizeSet,
     kSizeMultiSet
   };


//	members
   TClass *fClass;
   Int_t  fSize;
   Int_t  fUsed;	//Wrong size in case of Solaris bug
   Int_t  fVirt;
   Int_t  fMult;
   Int_t  fTail;
//      local members
   Int_t  fLastMult; 	//Align of last base, workaround Solaris bug
   Int_t  fOffset;
   Int_t  fNBase;
   TList *fOffsetList;
};

#ifdef __ROOTCLING__
#pragma link C++ class TOffset+;
#endif

#endif
