//////////////////////////////////////////////////////////
//   This class has been automatically generated 
//     (Wed Sep 25 17:31:23 2002 by ROOT version3.03/09)
//   from TTree T1/An example of a ROOT tree
//   found on file: Event1.root
//////////////////////////////////////////////////////////


#ifndef sel_h
#define sel_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>

class sel : public TSelector {
   public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
//Declaration of leaves types
   //Event           *event;
   Char_t          fType[20];
   Int_t           fNtrack;
   Int_t           fNseg;
   Int_t           fNvertex;
   UInt_t          fFlag;
   Float_t         fTemperature;
   Int_t           fMeasures[10];
   Float_t         fMatrix[4][4];
   Float_t         fClosestDistance[17];   //[fNvertex]
 //EventHeader     fEvtHdr;
   TRef            fLastTrack;
   TRef            fWebHistogram;

//List of branches
   TBranch        *b_fType;   //!
   TBranch        *b_fNtrack;   //!
   TBranch        *b_fNseg;   //!
   TBranch        *b_fNvertex;   //!
   TBranch        *b_fFlag;   //!
   TBranch        *b_fTemperature;   //!
   TBranch        *b_fMeasures;   //!
   TBranch        *b_fMatrix;   //!
   TBranch        *b_fClosestDistance;   //!
   TBranch        *b_fLastTrack;   //!
   TBranch        *b_fWebHistogram;   //!

//Extra members
   TString MyNameIs;
   
   sel(TTree *tree=0) { }
   ~sel() { }
   void    Begin(TTree *tree);
   void    Init(TTree *tree);
   Bool_t  Notify();
   Bool_t  Process(Int_t entry);
   Bool_t  ProcessCut(Int_t entry);
   void    ProcessFill(Int_t entry);
   void    SetOption(const char *option) { fOption = option; }
   void    SetObject(TObject *obj) { fObject = obj; }
   void    SetInputList(TList *input) {fInput = input;}
   TList  *GetOutputList() const { return fOutput; }
   void    Terminate();
};

#endif

#ifdef sel_cxx
void sel::Init(TTree *tree)
{
//   Set object pointer
   //event = 0;
//   Set branch addresses
   if (tree == 0) return;
   fChain    = tree;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("fType[20]",fType);
   fChain->SetBranchAddress("fNtrack",&fNtrack);
   fChain->SetBranchAddress("fNseg",&fNseg);
   fChain->SetBranchAddress("fNvertex",&fNvertex);
   fChain->SetBranchAddress("fFlag",&fFlag);
   fChain->SetBranchAddress("fTemperature",&fTemperature);
   fChain->SetBranchAddress("fMeasures[10]",fMeasures);
   fChain->SetBranchAddress("fMatrix[4][4]",fMatrix);
   fChain->SetBranchAddress("fClosestDistance",fClosestDistance);
   fChain->SetBranchAddress("fLastTrack",&fLastTrack);
   fChain->SetBranchAddress("fWebHistogram",&fWebHistogram);
}

Bool_t sel::Notify()
{
   // Called when loading a new file.
   // Get branch pointers.
   b_fType = fChain->GetBranch("fType[20]");
   b_fNtrack = fChain->GetBranch("fNtrack");
   b_fNseg = fChain->GetBranch("fNseg");
   b_fNvertex = fChain->GetBranch("fNvertex");
   b_fFlag = fChain->GetBranch("fFlag");
   b_fTemperature = fChain->GetBranch("fTemperature");
   b_fMeasures = fChain->GetBranch("fMeasures[10]");
   b_fMatrix = fChain->GetBranch("fMatrix[4][4]");
   b_fClosestDistance = fChain->GetBranch("fClosestDistance");
   b_fLastTrack = fChain->GetBranch("fLastTrack");
   b_fWebHistogram = fChain->GetBranch("fWebHistogram");
   return kTRUE;
}

#endif // #ifdef sel_cxx

