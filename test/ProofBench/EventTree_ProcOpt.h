//////////////////////////////////////////////////////////
// This class has been automatically generated on
// Fri May  7 07:24:11 2004 by ROOT version 4.00/04
// from TTree EventTree/Event Tree
// found on file: event_tree_darkstar.root
//////////////////////////////////////////////////////////

#ifndef EventTree_ProcOpt_h
#define EventTree_ProcOpt_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>
#include "Event.h"
#include "TH1.h"

class EventTree_ProcOpt : public TSelector {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain

   // Declaration of leave types
   Event           *event;
   Char_t          fType[20];
   Int_t           fNtrack;
   Int_t           fNseg;
   Int_t           fNvertex;
   UInt_t          fFlag;
   Double32_t      fTemperature;
   Int_t           fMeasures[10];
   Double32_t      fMatrix[4][4];
   Double32_t      fClosestDistance[0];   //[fNvertex]
   EventHeader     fEvtHdr;
   TClonesArray*   fTracks;
   TRef            fLastTrack;
   TRef            fWebHistogram;
   TBits           fTriggerBits;

   //Output hist
   TH1F* fHist;

   // List of branches
   TBranch        *b_event_fType;   //!
   TBranch        *b_event_fNtrack;   //!
   TBranch        *b_event_fNseg;   //!
   TBranch        *b_event_fNvertex;   //!
   TBranch        *b_event_fFlag;   //!
   TBranch        *b_event_fTemperature;   //!
   TBranch        *b_event_fMeasures;   //!
   TBranch        *b_event_fMatrix;   //!
   TBranch        *b_fClosestDistance;   //!
   TBranch        *b_event_fEvtHdr;   //!
   TBranch        *b_fTracks;   //!
   TBranch        *b_event_fLastTrack;   //!
   TBranch        *b_event_fWebHistogram;   //!
   TBranch        *b_event_fTriggerBits;   //!

   EventTree_ProcOpt(TTree *) { }
   EventTree_ProcOpt() { }
   ~EventTree_ProcOpt() { }
   Int_t   Version() const {return 1;}
   void    Begin(TTree *);
   void    SlaveBegin(TTree *tree);
   void    Init(TTree *tree);
   Bool_t  Notify();
   Bool_t  Process(Int_t entry);
   void    SetOption(const char *option) { fOption = option; }
   void    SetObject(TObject *obj) { fObject = obj; }
   void    SetInputList(TList *input) {fInput = input;}
   TList  *GetOutputList() const { return fOutput; }
   void    SlaveTerminate();
   void    Terminate();

   ClassDef(EventTree_ProcOpt,0);
};

#endif

#ifdef EventTree_ProcOpt_cxx
void EventTree_ProcOpt::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses of the tree
   // will be set. It is normaly not necessary to make changes to the
   // generated code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running with PROOF.

   // Set branch addresses
   fTracks=0;
   if (tree == 0) return;
   fChain = tree;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("fType[20]",fType);
   fChain->SetBranchAddress("fNtrack",&fNtrack);
   fChain->SetBranchAddress("fNseg",&fNseg);
   fChain->SetBranchAddress("fNvertex",&fNvertex);
   fChain->SetBranchAddress("fFlag",&fFlag);
   fChain->SetBranchAddress("fTemperature",&fTemperature);
   fChain->SetBranchAddress("fMeasures[10]",fMeasures);
   fChain->SetBranchAddress("fMatrix[4][4]",fMatrix);
   fChain->SetBranchAddress("fClosestDistance",&fClosestDistance);
   fChain->SetBranchStatus("fClosestDistance",0);
   fChain->SetBranchAddress("fEvtHdr",&fEvtHdr);
   fChain->SetBranchAddress("fTracks",&fTracks);
   fChain->SetBranchAddress("fLastTrack",&fLastTrack);
   fChain->SetBranchAddress("fWebHistogram",&fWebHistogram);
   fChain->SetBranchAddress("fTriggerBits",&fTriggerBits);

}

Bool_t EventTree_ProcOpt::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. Typically here the branch pointers
   // will be retrieved. It is normaly not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed.

   // Get branch pointers
   b_event_fType = fChain->GetBranch("fType[20]");
   b_event_fNtrack = fChain->GetBranch("fNtrack");
   b_event_fNseg = fChain->GetBranch("fNseg");
   b_event_fNvertex = fChain->GetBranch("fNvertex");
   b_event_fFlag = fChain->GetBranch("fFlag");
   b_event_fTemperature = fChain->GetBranch("fTemperature");
   b_event_fMeasures = fChain->GetBranch("fMeasures[10]");
   b_event_fMatrix = fChain->GetBranch("fMatrix[4][4]");
   b_fClosestDistance = fChain->GetBranch("fClosestDistance");
   b_event_fEvtHdr = fChain->GetBranch("fEvtHdr");
   b_fTracks = fChain->GetBranch("fTracks");
   b_event_fLastTrack = fChain->GetBranch("fLastTrack");
   b_event_fWebHistogram = fChain->GetBranch("fWebHistogram");
   b_event_fTriggerBits = fChain->GetBranch("fTriggerBits");

   return kTRUE;
}

#endif // #ifdef EventTree_ProcOpt_cxx
