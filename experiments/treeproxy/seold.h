//////////////////////////////////////////////////////////
//   This class has been automatically generated 
//     (Wed Oct 23 16:19:38 2002 by ROOT version3.03/09)
//   from TTree T/An example of a ROOT tree
//   found on file: Event.new.split9.root
//////////////////////////////////////////////////////////


#ifndef seold_h
#define seold_h

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>
#include <TRef.h>
#include <TRefArray.h>
#include <TH1F.h>

   const Int_t kMaxfTracks = 617;

class seold : public TSelector {
   public :
   TTree          *fChain;   //!pointer to the analyzed TTree or TChain
//Declaration of leaves types
 //Event           *event;
   UInt_t          fUniqueID;
   UInt_t          fBits;
   Char_t          fType[20];
   Int_t           fNtrack;
   Int_t           fNseg;
   Int_t           fNvertex;
   UInt_t          fFlag;
   Float_t         fTemperature;
   Int_t           fMeasures[10];
   Float_t         fMatrix[4][4];
   Float_t         fClosestDistance[20];   //[fNvertex]
   Int_t           fEvtHdr_fEvtNum;
   Int_t           fEvtHdr_fRun;
   Int_t           fEvtHdr_fDate;
   Int_t           fTracks_;
   UInt_t          fTracks_fUniqueID[kMaxfTracks];   //[fTracks_]
   UInt_t          fTracks_fBits[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fPx[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fPy[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fPz[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fRandom[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fMass2[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fBx[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fBy[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fMeanCharge[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fXfirst[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fXlast[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fYfirst[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fYlast[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fZfirst[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fZlast[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fCharge[kMaxfTracks];   //[fTracks_]
   Float_t         fTracks_fVertex[kMaxfTracks][3];   //[fTracks_]
   Int_t           fTracks_fNpoint[kMaxfTracks];   //[fTracks_]
   Short_t         fTracks_fValid[kMaxfTracks];   //[fTracks_]
   Int_t           fTracks_fNsp[kMaxfTracks];   //[fTracks_]
   Float_t        *fTracks_fPointValue[kMaxfTracks];   //[fTracks_fNsp]
   TRef            fLastTrack;
   TRef            fWebHistogram;
   TH1F*           fH;

//List of branches
   TBranch        *b_fUniqueID;   //!
   TBranch        *b_fBits;   //!
   TBranch        *b_fType;   //!
   TBranch        *b_fNtrack;   //!
   TBranch        *b_fNseg;   //!
   TBranch        *b_fNvertex;   //!
   TBranch        *b_fFlag;   //!
   TBranch        *b_fTemperature;   //!
   TBranch        *b_fMeasures;   //!
   TBranch        *b_fMatrix;   //!
   TBranch        *b_fClosestDistance;   //!
   TBranch        *b_fEvtHdr_fEvtNum;   //!
   TBranch        *b_fEvtHdr_fRun;   //!
   TBranch        *b_fEvtHdr_fDate;   //!
   TBranch        *b_fTracks_;   //!
   TBranch        *b_fTracks_fUniqueID;   //!
   TBranch        *b_fTracks_fBits;   //!
   TBranch        *b_fTracks_fPx;   //!
   TBranch        *b_fTracks_fPy;   //!
   TBranch        *b_fTracks_fPz;   //!
   TBranch        *b_fTracks_fRandom;   //!
   TBranch        *b_fTracks_fMass2;   //!
   TBranch        *b_fTracks_fBx;   //!
   TBranch        *b_fTracks_fBy;   //!
   TBranch        *b_fTracks_fMeanCharge;   //!
   TBranch        *b_fTracks_fXfirst;   //!
   TBranch        *b_fTracks_fXlast;   //!
   TBranch        *b_fTracks_fYfirst;   //!
   TBranch        *b_fTracks_fYlast;   //!
   TBranch        *b_fTracks_fZfirst;   //!
   TBranch        *b_fTracks_fZlast;   //!
   TBranch        *b_fTracks_fCharge;   //!
   TBranch        *b_fTracks_fVertex;   //!
   TBranch        *b_fTracks_fNpoint;   //!
   TBranch        *b_fTracks_fValid;   //!
   TBranch        *b_fTracks_fNsp;   //!
   TBranch        *b_fTracks_fPointValue;   //!
   TBranch        *b_fLastTrack;   //!
   TBranch        *b_fWebHistogram;   //!

   seold(TTree *tree=0) { }
   ~seold() { }
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

#ifdef seold_cxx
void seold::Init(TTree *tree)
{
//   Set branch addresses
   if (tree == 0) return;
   fChain    = tree;
   fChain->SetMakeClass(1);

   fChain->SetBranchAddress("fUniqueID",&fUniqueID);
   fChain->SetBranchAddress("fBits",&fBits);
   fChain->SetBranchAddress("fType[20]",fType);
   fChain->SetBranchAddress("fNtrack",&fNtrack);
   fChain->SetBranchAddress("fNseg",&fNseg);
   fChain->SetBranchAddress("fNvertex",&fNvertex);
   fChain->SetBranchAddress("fFlag",&fFlag);
   fChain->SetBranchAddress("fTemperature",&fTemperature);
   fChain->SetBranchAddress("fMeasures[10]",fMeasures);
   fChain->SetBranchAddress("fMatrix[4][4]",fMatrix);
   fChain->SetBranchAddress("fClosestDistance",fClosestDistance);
   fChain->SetBranchAddress("fEvtHdr.fEvtNum",&fEvtHdr_fEvtNum);
   fChain->SetBranchAddress("fEvtHdr.fRun",&fEvtHdr_fRun);
   fChain->SetBranchAddress("fEvtHdr.fDate",&fEvtHdr_fDate);
   fChain->SetBranchAddress("fTracks",&fTracks_);
   fChain->SetBranchAddress("fTracks.fUniqueID",fTracks_fUniqueID);
   fChain->SetBranchAddress("fTracks.fBits",fTracks_fBits);
   fChain->SetBranchAddress("fTracks.fPx",fTracks_fPx);
   fChain->SetBranchAddress("fTracks.fPy",fTracks_fPy);
   fChain->SetBranchAddress("fTracks.fPz",fTracks_fPz);
   fChain->SetBranchAddress("fTracks.fRandom",fTracks_fRandom);
   fChain->SetBranchAddress("fTracks.fMass2",fTracks_fMass2);
   fChain->SetBranchAddress("fTracks.fBx",fTracks_fBx);
   fChain->SetBranchAddress("fTracks.fBy",fTracks_fBy);
   fChain->SetBranchAddress("fTracks.fMeanCharge",fTracks_fMeanCharge);
   fChain->SetBranchAddress("fTracks.fXfirst",fTracks_fXfirst);
   fChain->SetBranchAddress("fTracks.fXlast",fTracks_fXlast);
   fChain->SetBranchAddress("fTracks.fYfirst",fTracks_fYfirst);
   fChain->SetBranchAddress("fTracks.fYlast",fTracks_fYlast);
   fChain->SetBranchAddress("fTracks.fZfirst",fTracks_fZfirst);
   fChain->SetBranchAddress("fTracks.fZlast",fTracks_fZlast);
   fChain->SetBranchAddress("fTracks.fCharge",fTracks_fCharge);
   fChain->SetBranchAddress("fTracks.fVertex[3]",fTracks_fVertex);
   fChain->SetBranchAddress("fTracks.fNpoint",fTracks_fNpoint);
   fChain->SetBranchAddress("fTracks.fValid",fTracks_fValid);
   fChain->SetBranchAddress("fTracks.fNsp",fTracks_fNsp);
   fChain->SetBranchAddress("fTracks.fPointValue",fTracks_fPointValue);
   fChain->SetBranchAddress("fLastTrack",&fLastTrack);
   fChain->SetBranchAddress("fWebHistogram",&fWebHistogram);
   fChain->SetBranchAddress("fH",&fH);
}

Bool_t seold::Notify()
{
   // Called when loading a new file.
   // Get branch pointers.
   b_fUniqueID = fChain->GetBranch("fUniqueID");
   b_fBits = fChain->GetBranch("fBits");
   b_fType = fChain->GetBranch("fType[20]");
   b_fNtrack = fChain->GetBranch("fNtrack");
   b_fNseg = fChain->GetBranch("fNseg");
   b_fNvertex = fChain->GetBranch("fNvertex");
   b_fFlag = fChain->GetBranch("fFlag");
   b_fTemperature = fChain->GetBranch("fTemperature");
   b_fMeasures = fChain->GetBranch("fMeasures[10]");
   b_fMatrix = fChain->GetBranch("fMatrix[4][4]");
   b_fClosestDistance = fChain->GetBranch("fClosestDistance");
   b_fEvtHdr_fEvtNum = fChain->GetBranch("fEvtHdr.fEvtNum");
   b_fEvtHdr_fRun = fChain->GetBranch("fEvtHdr.fRun");
   b_fEvtHdr_fDate = fChain->GetBranch("fEvtHdr.fDate");
   b_fTracks_ = fChain->GetBranch("fTracks");
   b_fTracks_fUniqueID = fChain->GetBranch("fTracks.fUniqueID");
   b_fTracks_fBits = fChain->GetBranch("fTracks.fBits");
   b_fTracks_fPx = fChain->GetBranch("fTracks.fPx");
   b_fTracks_fPy = fChain->GetBranch("fTracks.fPy");
   b_fTracks_fPz = fChain->GetBranch("fTracks.fPz");
   b_fTracks_fRandom = fChain->GetBranch("fTracks.fRandom");
   b_fTracks_fMass2 = fChain->GetBranch("fTracks.fMass2");
   b_fTracks_fBx = fChain->GetBranch("fTracks.fBx");
   b_fTracks_fBy = fChain->GetBranch("fTracks.fBy");
   b_fTracks_fMeanCharge = fChain->GetBranch("fTracks.fMeanCharge");
   b_fTracks_fXfirst = fChain->GetBranch("fTracks.fXfirst");
   b_fTracks_fXlast = fChain->GetBranch("fTracks.fXlast");
   b_fTracks_fYfirst = fChain->GetBranch("fTracks.fYfirst");
   b_fTracks_fYlast = fChain->GetBranch("fTracks.fYlast");
   b_fTracks_fZfirst = fChain->GetBranch("fTracks.fZfirst");
   b_fTracks_fZlast = fChain->GetBranch("fTracks.fZlast");
   b_fTracks_fCharge = fChain->GetBranch("fTracks.fCharge");
   b_fTracks_fVertex = fChain->GetBranch("fTracks.fVertex[3]");
   b_fTracks_fNpoint = fChain->GetBranch("fTracks.fNpoint");
   b_fTracks_fValid = fChain->GetBranch("fTracks.fValid");
   b_fTracks_fNsp = fChain->GetBranch("fTracks.fNsp");
   b_fTracks_fPointValue = fChain->GetBranch("fTracks.fPointValue");
   b_fLastTrack = fChain->GetBranch("fLastTrack");
   b_fWebHistogram = fChain->GetBranch("fWebHistogram");
   return kTRUE;
}

#endif // #ifdef seold_cxx

