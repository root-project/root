// @(#)root/tree:$Id$
// Author: Rene Brun   06/04/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TNtuple
#define ROOT_TNtuple


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNtuple                                                              //
//                                                                      //
// A simple tree with branches of floats.                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTree.h"

class TBrowser;

class TNtuple : public TTree {

protected:
   Int_t       fNvar;            ///<  Number of columns
   Float_t    *fArgs;            ///<! [fNvar] Array of variables

   Int_t  Fill() override;

private:
   TNtuple(const TNtuple&) = delete;
   TNtuple& operator=(const TNtuple&) = delete;

public:
   TNtuple();
   TNtuple(const char *name,const char *title, const char *varlist, Int_t bufsize=32000);
   virtual ~TNtuple();

           void      Browse(TBrowser *b) override;
           TTree    *CloneTree(Long64_t nentries = -1, Option_t* option = "") override;
   virtual Int_t     Fill(const Float_t *x);
           Int_t     Fill(Int_t x0) { return Fill((Float_t)x0); }
           Int_t     Fill(Double_t x0) { return Fill((Float_t)x0); }
   virtual Int_t     Fill(Float_t x0, Float_t x1=0, Float_t x2=0, Float_t x3=0,
                          Float_t x4=0, Float_t x5=0, Float_t x6=0, Float_t x7=0,
                          Float_t x8=0, Float_t x9=0, Float_t x10=0,
                          Float_t x11=0, Float_t x12=0, Float_t x13=0,
                          Float_t x14=0);
   virtual Int_t     GetNvar() const { return fNvar; }
           Float_t  *GetArgs() const { return fArgs; }
           Long64_t  ReadStream(std::istream& inputStream, const char *branchDescriptor="", char delimiter = ' ') override;
           void      ResetBranchAddress(TBranch *) override;
           void      ResetBranchAddresses() override;

   ClassDefOverride(TNtuple,2);  //A simple tree with branches of floats.
};

#endif
