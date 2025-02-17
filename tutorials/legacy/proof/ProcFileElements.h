/// \file
/// \ingroup tutorial_ProcFileElements
///
/// Class to hold information about the processed elements of a file
///
/// \macro_code
///
/// \author Gerardo Ganis (gerardo.ganis@cern.ch)


#ifndef ROOT_ProcFileElements
#define ROOT_ProcFileElements


#include "TObject.h"
#include "TSortedList.h"

class ProcFileElements : public TObject {

public:
   class ProcFileElement : public TObject {
   public:
      Long64_t    fFirst;   // Lower bound of this range
      Long64_t    fLast;    // Upper bound of this range
      ProcFileElement(Long64_t fst = 0, Long64_t lst = -1) :
                                      fFirst(fst), fLast(lst) { }
      ~ProcFileElement() override { }

      Int_t       Compare(const TObject *obj) const override;

      Bool_t      IsSortable() const override { return kTRUE; }
      Int_t       MergeElement(ProcFileElement *);

      Int_t       Overlapping(ProcFileElement *);
      void        Print(Option_t *option="") const override;

      ClassDefOverride(ProcFileElement, 1); // ProcFileElement class
   };

private:
   TString     fName;          // File name
   TSortedList   *fElements;      // List of processed elements

   Long64_t    fFirst;   // Overall lower bound
   Long64_t    fLast;    // Overall Upper bound

public:
   ProcFileElements(const char *fn = "") : fName(fn), fElements(nullptr),
                                           fFirst(0), fLast(-1) { }
   ~ProcFileElements() override { if (fElements) { fElements->SetOwner();
                                                  delete fElements; } }
   const char *   GetName() const override { return fName; }
   ULong_t        Hash() const override { return fName.Hash(); }

   Int_t          Add(Long64_t fst = 0, Long64_t lst = -1);
   Int_t          Merge(TCollection *list);

   TSortedList   *GetListOfElements() const { return fElements; }
   Int_t          GetNumElements() const { return (fElements ? fElements->GetSize() : 0); }

   Long64_t       GetFirst() const { return fFirst; }
   Long64_t       GetLast() const { return fLast; }

   void           Print(Option_t *option="") const override;

   ClassDefOverride(ProcFileElements, 1);  // Processed File Elements class
};

#endif
