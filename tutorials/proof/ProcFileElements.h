// @(#)root/tutorials/proof:$Id$
// Author: G. Ganis, Nov 2011

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_ProcFileElements
#define ROOT_ProcFileElements

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// ProcFileElements                                                     //
//                                                                      //
// This class holds information about the processed elements of a file. //
// Used for testing.                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TSortedList
#include "TSortedList.h"
#endif

class ProcFileElements : public TObject {

private:

   TString     fName;          // File name 
   class ProcFileElement : public TObject {
   public:
      Long64_t    fFirst;   // Lower bound of this range
      Long64_t    fLast;    // Upper bound of this range
      ProcFileElement(Long64_t fst = 0, Long64_t lst = -1) :
                                      fFirst(fst), fLast(lst) { }
      virtual ~ProcFileElement() { }

      Int_t       Compare(const TObject *obj) const;

      Bool_t      IsSortable() const { return kTRUE; }
      Int_t       MergeElement(ProcFileElement *);

      Int_t       Overlapping(ProcFileElement *);
      void        Print(Option_t *option="") const;

      ClassDef(ProcFileElement, 1); // ProcFileElement class
   };
   TSortedList   *fElements;      // List of processed elements

   Long64_t    fFirst;   // Overall lower bound 
   Long64_t    fLast;    // Overall Upper bound 

public:
   ProcFileElements(const char *fn = "") : fName(fn), fElements(0),
                                           fFirst(0), fLast(-1) { }
   virtual ~ProcFileElements() { if (fElements) { fElements->SetOwner();
                                                  delete fElements; } }
   const char *   GetName() const { return fName; }

   Int_t          Add(Long64_t fst = 0, Long64_t lst = -1);
   Int_t          Merge(TCollection *list);

   TSortedList   *GetListOfElements() const { return fElements; }
   Int_t          GetNumElements() const { return (fElements ? fElements->GetSize() : 0); }
   
   Long64_t       GetFirst() const { return fFirst; }
   Long64_t       GetLast() const { return fLast; }

   void           Print(Option_t *option="") const;

   ClassDef(ProcFileElements, 1);  // Processed File Elements class
};

#endif
