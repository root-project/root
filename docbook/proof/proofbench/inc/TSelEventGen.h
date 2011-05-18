// @(#)root/proof:$Id$
// Author: Sangsu Ryu 22/06/2010

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSelEventGen
#define ROOT_TSelEventGen

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSelEventGen                                                         //
//                                                                      //
// PROOF selector for event file generation.                            //
// List of files to be generated for each node is provided by client.   //
// And list of files generated is sent back.                            //
// Existing files are reused if not forced to be regenerated.           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSelector
#include <TSelector.h>
#endif
#ifndef ROOT_TTree
#include <TTree.h>
#endif
#ifndef ROOT_TString
#include <TString.h>
#endif

class TList;

class TSelEventGen : public TSelector {

private:

   TString fBaseDir;                          //directory where the generated files will be written to
   //Int_t fMaxNWorkers;
   Long64_t fNEvents;                         //number of events in a file
   Int_t fNTracks;                            //avg or min-avg number of tracks in an event
   Int_t fNTracksMax;                         //max-avg number of tracks in an event
   Int_t fRegenerate;                         //force generation of cleanup files

   TObject* fTotalGen;                        //events generated on this worker
   TList* fFilesGenerated;                    //list of files generated

protected:

   Long64_t GenerateFiles(TString filename, Long64_t sizenevents);

public :

   TTree          *fChain;   //!pointer to the analyzed TTree or TChain

//   TSelEventGen(TTree *);
   TSelEventGen();
   virtual ~TSelEventGen() { }
   virtual Int_t   Version() const {return 1;}
   virtual void    Begin(TTree *);
   virtual void    SlaveBegin(TTree *tree);
   virtual void    Init(TTree *tree);
   virtual Bool_t  Notify();
   virtual Bool_t  Process(Long64_t entry);
   virtual void    SetOption(const char *option) { fOption = option; }
   virtual void    SetObject(TObject *obj) { fObject = obj; }
   virtual void    SetInputList(TList *input) {fInput = input;}
   virtual TList  *GetOutputList() const { return fOutput; }
   virtual void    SlaveTerminate();
   virtual void    Terminate();
   virtual void    Print(Option_t *option="") const;

   ClassDef(TSelEventGen,0)     //PROOF selector for event file generation
};

#endif

#ifdef TSelEventGen_cxx

void TSelEventGen::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses of the tree
   // will be set. It is normaly not necessary to make changes to the
   // generated code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running with PROOF.

   if (tree == 0) return;
   fChain = tree;
   fChain->SetMakeClass(1);
}

Bool_t TSelEventGen::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. Typically here the branch pointers
   // will be retrieved. It is normaly not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed.

   return kTRUE;
}

#endif // #ifdef TSelEventGen_cxx
