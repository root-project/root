// @(#)root/tree:$Id$
// Author: Rene Brun   11/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBranchObject
#define ROOT_TBranchObject


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBranchObject                                                        //
//                                                                      //
// A Branch for the case of an object.                                  //
//////////////////////////////////////////////////////////////////////////


#include "TBranch.h"

class TBranchObject : public TBranch {

protected:
   enum EStatusBits {
      kWarn = BIT(14)
   };

   /// In version of ROOT older then v6.12, kWarn was set to BIT(12)
   /// which overlaps with TBranch::kBranchObject.  Since it stored
   /// in ROOT files as part of the TBranchObject and that we want
   /// to reset in TBranchObject::Streamer, we need to keep track
   /// of the old value.
   enum EStatusBitsOldValues {
      kOldWarn = BIT(12)
   };

   TString     fClassName;        ///< Class name of referenced object
   TObject     *fOldObject;       ///< !Pointer to old object

   void Init(TTree *tree, TBranch *parent, const char *name, const char *classname, void *addobj, Int_t basketsize, Int_t splitlevel, Int_t compress, Bool_t isptrptr);

public:
   TBranchObject();
   TBranchObject(TBranch *parent, const char *name, const char *classname, void *addobj, Int_t basketsize=32000, Int_t splitlevel = 0, Int_t compress = ROOT::RCompressionSetting::EAlgorithm::kInherit, Bool_t isptrptr = kTRUE);
   TBranchObject(TTree *tree, const char *name, const char *classname, void *addobj, Int_t basketsize=32000, Int_t splitlevel = 0, Int_t compress = ROOT::RCompressionSetting::EAlgorithm::kInherit, Bool_t isptrptr = kTRUE);
   virtual ~TBranchObject();

           void        Browse(TBrowser *b) override;
           const char* GetClassName() const override { return fClassName.Data(); };
   virtual const char* GetObjClassName() { return fClassName.Data(); };
           Int_t       GetEntry(Long64_t entry=0, Int_t getall = 0) override;
           Int_t       GetExpectedType(TClass *&clptr,EDataType &type) override;
           Bool_t      IsFolder() const override;
           void        Print(Option_t *option="") const override;
           void        Reset(Option_t *option="") override;
           void        ResetAfterMerge(TFileMergeInfo *) override;
           void        SetAddress(void *addobj) override;
           void        SetAutoDelete(Bool_t autodel=kTRUE) override;
           void        SetBasketSize(Int_t buffsize) override;
           void        SetupAddresses() override;
           void        UpdateAddress() override;

private:
           Int_t       FillImpl(ROOT::Internal::TBranchIMTHelper *) override;

   ClassDefOverride(TBranchObject,1);  //Branch in case of an object
};

#endif
