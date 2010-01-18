#ifndef ProofEventProc_h
#define ProofEventProc_h

//////////////////////////////////////////////////////////
//
// Example of TSelector implementation to process trees
// containing 'Event' structures, e.g. the files under
// http://root.cern.ch/files/data .
// See tutorials/proof/runProof.C, option "eventproc", for
// an example of how to run this selector.
//
//////////////////////////////////////////////////////////

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>
#include "Event.h"

class TH1F;
class TH2F;

class ProofEventProc : public TSelector {
public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain

   // Declaration of leave types
   Event           *event;
   Char_t          fType[20];
   Char_t          *fEventName;
   Int_t           fNtrack;
   Int_t           fNseg;
   Int_t           fNvertex;
   UInt_t          fFlag;
   Double32_t      fTemperature;
   Int_t           fMeasures[10];
   Double32_t      fMatrix[4][4];
   Double32_t      fClosestDistance[21];   //[fNvertex]
   EventHeader     fEvtHdr;
   TClonesArray    *fTracks;
   TRefArray       *fHighPt;
   TRefArray       *fMuons;
   TRef            fLastTrack;
   TRef            fWebHistogram;
   TH1F            *fH;
   TBits           fTriggerBits;
   Bool_t          fIsValid;

   // Read controller
   Bool_t          fFullRead;

   //Output hist
   TH1F* fPtHist;
   TH1F* fPzHist;
   TH2F* fPxPyHist;

   // List of branches
   TBranch        *b_event_fType;   //!
   TBranch        *b_fEventName;   //!
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
   TBranch        *b_fHighPt;   //!
   TBranch        *b_fMuons;   //!
   TBranch        *b_event_fLastTrack;   //!
   TBranch        *b_event_fWebHistogram;   //!
   TBranch        *b_fH;   //!
   TBranch        *b_event_fTriggerBits;   //!
   TBranch        *b_event_fIsValid;   //!

   ProofEventProc(TTree *) { fFullRead = kTRUE; }
   ProofEventProc() { }
   virtual ~ProofEventProc() { }
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

   ClassDef(ProofEventProc,0);
};

#endif

#ifdef ProofEventProc_cxx
void ProofEventProc::Init(TTree *tree)
{
   // The Init() function is called when the selector needs to initialize
   // a new tree or chain. Typically here the branch addresses of the tree
   // will be set. It is normaly not necessary to make changes to the
   // generated code, but the routine can be extended by the user if needed.
   // Init() will be called many times when running with PROOF.

   // Set branch addresses
   fEventName=0;
   fTracks=0;
   fHighPt=0;
   fMuons=0;
   fH=0;
   if (tree == 0) return;
   fChain = tree;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("fType[20]",fType);
   fChain->SetBranchAddress("fEventName",fEventName);
   fChain->SetBranchAddress("fNtrack",&fNtrack);
   fChain->SetBranchAddress("fNseg",&fNseg);
   fChain->SetBranchAddress("fNvertex",&fNvertex);
   fChain->SetBranchAddress("fFlag",&fFlag);
   fChain->SetBranchAddress("fTemperature",&fTemperature);
   fChain->SetBranchAddress("fMeasures[10]",fMeasures);
   fChain->SetBranchAddress("fMatrix[4][4]",fMatrix);
   fChain->SetBranchAddress("fClosestDistance",fClosestDistance);
   fChain->SetBranchAddress("fEvtHdr",&fEvtHdr);
   fChain->SetBranchAddress("fTracks",&fTracks);
   fChain->SetBranchAddress("fHighPt",&fHighPt);
   fChain->SetBranchAddress("fMuons",&fMuons);
   fChain->SetBranchAddress("fLastTrack",&fLastTrack);
   fChain->SetBranchAddress("fWebHistogram",&fWebHistogram);
   fChain->SetBranchAddress("fH",&fH);
   fChain->SetBranchAddress("fTriggerBits",&fTriggerBits);
   fChain->SetBranchAddress("fIsValid",&fIsValid);
}

Bool_t ProofEventProc::Notify()
{
   // The Notify() function is called when a new file is opened. This
   // can be either for a new TTree in a TChain or when when a new TTree
   // is started when using PROOF. Typically here the branch pointers
   // will be retrieved. It is normaly not necessary to make changes
   // to the generated code, but the routine can be extended by the
   // user if needed.

   // Get branch pointers
   b_event_fType = fChain->GetBranch("fType[20]");
   b_fEventName = fChain->GetBranch("fEventName");
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
   b_fHighPt = fChain->GetBranch("fHighPt");
   b_fMuons = fChain->GetBranch("fMuons");
   b_event_fLastTrack = fChain->GetBranch("fLastTrack");
   b_event_fWebHistogram = fChain->GetBranch("fWebHistogram");
   b_fH = fChain->GetBranch("fH");
   b_event_fTriggerBits = fChain->GetBranch("fTriggerBits");
   b_event_fIsValid = fChain->GetBranch("fIsValid");

   return kTRUE;
}

#endif // #ifdef ProofEventProc_cxx
