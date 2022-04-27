// @(#)root/tree:$Id$
// Author: Rene Brun   06/04/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TNtupleD
#define ROOT_TNtupleD


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNtupleD                                                             //
//                                                                      //
// A simple tree with branches of doubles.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTree.h"

class TBrowser;

class TNtupleD : public TTree {

protected:
   Int_t       fNvar;            ///<  Number of columns
   Double_t    *fArgs;           ///<! [fNvar] Array of variables

   Int_t  Fill() override;

private:
   TNtupleD(const TNtupleD&) = delete;
   TNtupleD& operator=(const TNtupleD&) = delete;

public:
   TNtupleD();
   TNtupleD(const char *name,const char *title, const char *varlist, Int_t bufsize=32000);
   virtual ~TNtupleD();

           void      Browse(TBrowser *b) override;
   virtual Int_t     Fill(const Double_t *x);
   virtual Int_t     Fill(Double_t x0, Double_t x1, Double_t x2=0, Double_t x3=0,
                          Double_t x4=0, Double_t x5=0, Double_t x6=0, Double_t x7=0,
                          Double_t x8=0, Double_t x9=0, Double_t x10=0,
                          Double_t x11=0, Double_t x12=0, Double_t x13=0,
                          Double_t x14=0);
   virtual Int_t     GetNvar() const { return fNvar; }
           Double_t *GetArgs() const { return fArgs; }
           Long64_t  ReadStream(std::istream& inputstream, const char *branchDescriptor="", char delimiter = ' ') override;
           void      ResetBranchAddress(TBranch *) override;
           void      ResetBranchAddresses() override;

   ClassDefOverride(TNtupleD,1)  //A simple tree with branches of floats.
};

#endif
