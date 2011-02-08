// @(#)root/proof:$Id$
// Author: Sangsu Ryu 22/06/2010

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSelHist
#define ROOT_TSelHist

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSelHist                                                             //
// PROOF selector for CPU-intensive benchmark test.                     //
// Events are generated and 1-D, 2-D, and/or 3-D histograms are filled. //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSelector
#include <TSelector.h>
#endif

class TH1F;
class TH2F;
class TH3F;
class TRandom3;
class TCanvas;
class TPBHistType;

class TSelHist : public TSelector {
public :

   // Specific members
   TPBHistType	   *fHistType;
   Int_t            fNHists;
   Bool_t 	    fDraw;
   TH1F           **fHist1D;//[fNHists]
   TH2F           **fHist2D;//[fNHists]
   TH3F           **fHist3D;//[fNHists]
   TRandom3        *fRandom;
   TCanvas	*fCHist1D;
   TCanvas	*fCHist2D;
   TCanvas	*fCHist3D;

   TSelHist();
   virtual ~TSelHist();
   virtual Int_t   Version() const { return 2; }
   virtual void    Begin(TTree *tree);
   virtual void    SlaveBegin(TTree *tree);
   virtual Bool_t  Process(Long64_t entry);
   virtual void    SetOption(const char *option) { fOption = option; }
   virtual void    SetObject(TObject *obj) { fObject = obj; }
   virtual void    SetInputList(TList *input) { fInput = input; }
   virtual TList  *GetOutputList() const { return fOutput; }
   virtual void    SlaveTerminate();
   virtual void    Terminate();

   ClassDef(TSelHist,0)  //PROOF selector for CPU-intensive benchmark test
};

#endif
