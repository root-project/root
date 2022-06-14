// @(#)root/hbook:$Id$
// Author: Rene Brun   18/02/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THbookBranch
#define ROOT_THbookBranch


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THbookBranch                                                         //
//                                                                      //
// A branch for a THbookTree                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBranch.h"

class THbookBranch : public TBranch {

protected:
   TString      fBlockName;   //Hbook block name

public:
   THbookBranch() {;}
   THbookBranch(TTree *tree, const char *name, void *address, const char *leaflist, Int_t basketsize=32000, Int_t compress = ROOT::RCompressionSetting::EAlgorithm::kInherit);
   THbookBranch(TBranch *branch, const char *name, void *address, const char *leaflist, Int_t basketsize=32000, Int_t compress = ROOT::RCompressionSetting::EAlgorithm::kInherit);
   ~THbookBranch() override;
   void     Browse(TBrowser *b) override;
   Int_t    GetEntry(Long64_t entry=0, Int_t getall=0) override;
   const char      *GetBlockName() const {return fBlockName.Data();}
   void     SetAddress(void *addobj) override;
           void     SetBlockName(const char *name) {fBlockName=name;}
   void     SetEntries(Long64_t n) override {fEntries=n;}

   ClassDefOverride(THbookBranch,1)  //A branch for a THbookTree
};

#endif
