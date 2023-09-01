/// \file
/// \ingroup tutorial_tree
/// \notebook -nodraw
/// This example writes a tree with objects of the class Event.
/// It is a simplified version of $ROOTSYS/test/MainEvent.cxx to
/// write the tree, and $ROOTSYS/test/eventb.C
/// It shows:
///   - how to fill a Tree with an event class containing these data members:
/// ~~~
///     char           fType[20];
///     Int_t          fNtrack;
///     Int_t          fNseg;
///     Int_t          fNvertex;
///     UInt_t         fFlag;
///     Float_t        fTemperature;
///     EventHeader    fEvtHdr;
///     TClonesArray  *fTracks;            //->
///     TH1F          *fH;                 //->
///     Int_t          fMeasures[10];
///     Float_t        fMatrix[4][4];
///     Float_t       *fClosestDistance;   //[fNvertex]
/// ~~~
///   - the difference in splitting or not splitting a branch
///   - how to read selected branches of the tree, and print the first entry with less than 587 tracks.
///   - how to browse and analyze the Tree via the TBrowser and TTreeViewer
///
/// This example can be run in many different ways:
///  - way1 using the Cling interpreter:
/// ~~~
/// .x tree4.C
/// ~~~
///  - way2 using the Cling interpreter:
/// ~~~
/// .L tree4.C
/// tree4()
/// ~~~
///  - way3 using ACLIC:
/// ~~~
/// .L ../test/libEvent.so
/// .x tree4.C++
/// ~~~
/// One can also run the write and read parts in two separate sessions.
/// For example following one of the sessions above, one can start the session:
/// ~~~
///   .L tree4.C
///   tree4r();
/// ~~~
/// \macro_code
///
/// \author Rene Brun

R__LOAD_LIBRARY($ROOTSYS/test/libEvent.so)

#include "TFile.h"
#include "TTree.h"
#include "TBrowser.h"
#include "TH2.h"
#include "TRandom.h"
#include "TClassTable.h"
#include "TSystem.h"
#include "TROOT.h"
#include "../test/Event.h"

void tree4w()
{

  //create a Tree file tree4.root
  TFile f("tree4.root","RECREATE");

  // Create a ROOT Tree
  TTree t4("t4","A Tree with Events");

  // Create a pointer to an Event object
  Event *event = new Event();

  // Create two branches, split one.
  t4.Branch("event_split", &event,16000,99);
  t4.Branch("event_not_split", &event,16000,0);

  // a local variable for the event type
  char etype[20];

  // Fill the tree
  for (Int_t ev = 0; ev <100; ev++) {
    Float_t sigmat, sigmas;
    gRandom->Rannor(sigmat,sigmas);
    Int_t ntrack   = Int_t(600 + 600 *sigmat/120.);
    Float_t random = gRandom->Rndm(1);
    sprintf(etype,"type%d",ev%5);
    event->SetType(etype);
    event->SetHeader(ev, 200, 960312, random);
    event->SetNseg(Int_t(10*ntrack+20*sigmas));
    event->SetNvertex(Int_t(1+20*gRandom->Rndm()));
    event->SetFlag(UInt_t(random+0.5));
    event->SetTemperature(random+20.);

    for(UChar_t m = 0; m < 10; m++) {
      event->SetMeasure(m, Int_t(gRandom->Gaus(m,m+1)));
    }

    // fill the matrix
    for(UChar_t i0 = 0; i0 < 4; i0++) {
      for(UChar_t i1 = 0; i1 < 4; i1++) {
        event->SetMatrix(i0,i1,gRandom->Gaus(i0*i1,1));
      }
    }

    //  Create and fill the Track objects
    for (Int_t t = 0; t < ntrack; t++) event->AddTrack(random);

    // Fill the tree
    t4.Fill();

    // Clear the event before reloading it
    event->Clear();
  }

  // Write the file header
  f.Write();

  // Print the tree contents
  t4.Print();
}


void tree4r()
{
  // read the tree generated with tree4w

  //note that we use "new" to create the TFile and TTree objects !
  //because we want to keep these objects alive when we leave this function.
  TFile *f = new TFile("tree4.root");
  TTree *t4 = (TTree*)f->Get("t4");

  // create a pointer to an event object. This will be used
  // to read the branch values.
  Event *event = new Event();

  // get two branches and set the branch address
  TBranch *bntrack = t4->GetBranch("fNtrack");
  TBranch *branch  = t4->GetBranch("event_split");
  branch->SetAddress(&event);

  Long64_t nevent = t4->GetEntries();
  Int_t nselected = 0;
  Int_t nb = 0;
  for (Long64_t i=0;i<nevent;i++) {
    //read branch "fNtrack"only
    bntrack->GetEntry(i);

    //reject events with more than 587 tracks
    if (event->GetNtrack() > 587)continue;

    //read complete accepted event in memory
    nb += t4->GetEntry(i);
    nselected++;

    //print the first accepted event
    if (nselected == 1) t4->Show();

    //clear tracks array
    event->Clear();
  }

  if (gROOT->IsBatch()) return;
  new TBrowser();
  t4->StartViewer();
}

void tree4() {
   Event::Reset(); // Allow for re-run this script by cleaning static variables.
   tree4w();
   Event::Reset(); // Allow for re-run this script by cleaning static variables.
   tree4r();
}
