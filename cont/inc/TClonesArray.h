// @(#)root/cont:$Name:  $:$Id: TClonesArray.h,v 1.6 2001/03/11 23:10:00 rdm Exp $
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
// adding elements (shrinking can be done by hand).                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

class TClass;


class TClonesArray : public TObjArray {

protected:
   TClass       *fClass;       //!Pointer to the class
   TObjArray    *fKeep;        //!Saved copies of pointers to objects

public:
   enum {
      kForgetBits     = BIT(0),   // Do not create branches for fBits, fUniqueID
      kNoSplit        = BIT(1),   // Array not split by TTree::Branch
      kBypassStreamer = BIT(12)   // Class Streamer not called (default)
   };

   TClonesArray();
   TClonesArray(const char *classname, Int_t size = 1000, Bool_t call_dtor = kFALSE);
   virtual         ~TClonesArray();
   virtual void     Compress();
   virtual void     Clear(Option_t *option="");
   virtual void     Delete(Option_t *option="");
   virtual void     Expand(Int_t newSize);
   virtual void     ExpandCreate(Int_t n);
   virtual void     ExpandCreateFast(Int_t n);
   TClass          *GetClass() const { return fClass; }

   void             AddFirst(TObject *) { MayNotUse("AddFirst"); }
   void             AddLast(TObject *) { MayNotUse("AddLast"); }
   void             AddAt(TObject *, Int_t) { MayNotUse("AddAt"); }
   void             AddAtAndExpand(TObject *, Int_t) { MayNotUse("AddAtAndExpand"); }
   Int_t            AddAtFree(TObject *) { MayNotUse("AddAtFree"); return 0; }
   void             AddAfter(TObject *, TObject *) { MayNotUse("AddAfter"); }
   void             AddBefore(TObject *, TObject *) { MayNotUse("AddBefore"); }
   void             BypassStreamer(Bool_t bypass=kTRUE);
   Bool_t           CanBypassStreamer() const { return TestBit(kBypassStreamer); }

   virtual TObject *RemoveAt(Int_t idx);
   virtual TObject *Remove(TObject *obj);
   virtual void     Sort(Int_t upto = kMaxInt);

   TObject         *New(Int_t idx);
   TObject         *AddrAt(Int_t idx);
   TObject         *&operator[](Int_t idx);

   ClassDef(TClonesArray,4)  //An array of clone objects
};

inline TObject *TClonesArray::AddrAt(Int_t idx)
{
   return operator[](idx);
}

#endif
