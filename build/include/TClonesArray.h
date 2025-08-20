// @(#)root/cont:$Id$
// Author: Rene Brun   11/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClonesArray
#define ROOT_TClonesArray


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClonesArray                                                         //
//                                                                      //
// An array of clone TObjects. The array expands automatically when     //
// adding elements (shrinking can be done explicitly).                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObjArray.h"

class TClass;

class TClonesArray : public TObjArray {

protected:
   TClass       *fClass;       //!Pointer to the class of the elements
   TObjArray    *fKeep;        //!Saved copies of pointers to objects

public:
   enum EStatusBits {
      kBypassStreamer = BIT(12),  // Class Streamer not called (default)
      kForgetBits     = BIT(15)   // Do not create branches for fBits, fUniqueID
   };

   TClonesArray();
   TClonesArray(const char *classname, Int_t size = 1000, Bool_t call_dtor = kFALSE);
   TClonesArray(const TClass *cl, Int_t size = 1000, Bool_t call_dtor = kFALSE);
   TClonesArray(const TClonesArray& tc);
   virtual         ~TClonesArray();
   TClonesArray& operator=(const TClonesArray& tc);
   void             Compress() override;
   void             Clear(Option_t *option="") override;
   void             Delete(Option_t *option="") override;
   void             Expand(Int_t newSize) override;
   virtual void     ExpandCreate(Int_t n);
   virtual void     ExpandCreateFast(Int_t n);
   TClass          *GetClass() const { return fClass; }
   void             SetOwner(Bool_t enable = kTRUE) override;

   void             AddFirst(TObject *) override { MayNotUse("AddFirst"); }
   void             AddLast(TObject *) override { MayNotUse("AddLast"); }
   void             AddAt(TObject *, Int_t) override { MayNotUse("AddAt"); }
   void             AddAtAndExpand(TObject *, Int_t) override { MayNotUse("AddAtAndExpand"); }
   Int_t            AddAtFree(TObject *) override { MayNotUse("AddAtFree"); return 0; }
   void             AddAfter(const TObject *, TObject *) override { MayNotUse("AddAfter"); }
   void             AddBefore(const TObject *, TObject *) override { MayNotUse("AddBefore"); }
   void             BypassStreamer(Bool_t bypass=kTRUE);
   Bool_t           CanBypassStreamer() const { return TestBit(kBypassStreamer); }
   TObject         *ConstructedAt(Int_t idx);
   TObject         *ConstructedAt(Int_t idx, Option_t *clear_options);
   void             SetClass(const char *classname,Int_t size=1000);
   void             SetClass(const TClass *cl,Int_t size=1000);

   void             AbsorbObjects(TClonesArray *tc);
   void             AbsorbObjects(TClonesArray *tc, Int_t idx1, Int_t idx2);
   void             MultiSort(Int_t nTCs, TClonesArray** tcs, Int_t upto = kMaxInt);
   TObject         *RemoveAt(Int_t idx) override;
   TObject         *Remove(TObject *obj) override;
   void             RemoveRange(Int_t idx1, Int_t idx2) override;
   void             Sort(Int_t upto = kMaxInt) override;

   TObject         *New(Int_t idx);
   TObject         *AddrAt(Int_t idx);
   TObject         *&operator[](Int_t idx) override;
   TObject         *operator[](Int_t idx) const override;

   ClassDefOverride(TClonesArray,4)  //An array of clone objects
};

inline TObject *TClonesArray::AddrAt(Int_t idx)
{
   return operator[](idx);
}

#endif
