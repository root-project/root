// @(#)root/tree:$Id$
// Author: Rene Brun   11/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEventList
#define ROOT_TEventList


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEventList                                                           //
//                                                                      //
// A list of selected entries in a TTree.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TDirectory;
class TCollection;


class TEventList : public TNamed {

protected:
   Int_t            fN;           //  Number of elements in the list
   Int_t            fSize;        //  Size of array
   Int_t            fDelta;       //  Increment size
   Bool_t           fReapply;     //  If true, TTree::Draw will 'reapply' the original cut
   Long64_t        *fList;        //[fN]Array of elements
   TDirectory      *fDirectory;   //! Pointer to directory holding this tree

public:
   TEventList();
   TEventList(const char *name, const char *title="",Int_t initsize=0, Int_t delta = 0);
   TEventList(const TEventList &list);
   virtual           ~TEventList();
   virtual void      Add(const TEventList *list);
   virtual void      Clear(Option_t *option="") {Reset(option);}
   virtual Bool_t    Contains(Long64_t entry);
   virtual Bool_t    ContainsRange(Long64_t entrymin, Long64_t entrymax);
   virtual void      DirectoryAutoAdd(TDirectory *);
   virtual void      Enter(Long64_t entry);
   TDirectory       *GetDirectory() const {return fDirectory;}
   virtual Long64_t  GetEntry(Int_t index) const;
   virtual Int_t     GetIndex(Long64_t entry) const;
   virtual Long64_t *GetList() const { return fList; }
   virtual Int_t     GetN() const { return fN; }
   virtual Bool_t    GetReapplyCut() const { return fReapply; };
   virtual Int_t     GetSize() const { return fSize; }
   virtual void      Intersect(const TEventList *list);
   virtual Int_t     Merge(TCollection *list);
   virtual void      Print(Option_t *option="") const;
   virtual void      Reset(Option_t *option="");
   virtual void      Resize(Int_t delta=0);
   virtual void      SetDelta(Int_t delta=100) {fDelta = delta;}
   virtual void      SetDirectory(TDirectory *dir);
   virtual void      SetName(const char *name); // *MENU*
   virtual void      SetReapplyCut(Bool_t apply = kFALSE) {fReapply = apply;}; // *TOGGLE*
   virtual void      Sort();
   virtual void      Subtract(const TEventList *list);

   TEventList&       operator=(const TEventList &list);

   friend TEventList operator+(const TEventList &list1, const TEventList &list2);
   friend TEventList operator-(const TEventList &list1, const TEventList &list2);
   friend TEventList operator*(const TEventList &list1, const TEventList &list2);

   ClassDef(TEventList,4);  //A list of selected entries in a TTree.
};

#endif

