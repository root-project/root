#define seneutral_cxx
// The class definition in se01.h has been generated automatically
// by the ROOT utility TTree::MakeSelector().
//
// This class is derived from the ROOT class TSelector.
// The following members functions are called by the TTree::Process() functions:
//    Begin():       called everytime a loop on the tree starts,
//                   a convenient place to create your histograms.
//    Notify():      this function is called at the first entry of a new Tree
//                   in a chain.
//    ProcessCut():  called at the beginning of each entry to return a flag,
//                   true if the entry must be analyzed.
//    ProcessFill(): called in the entry loop for all entries accepted
//                   by Select.
//    Terminate():   called at the end of a loop on the tree,
//                   a convenient place to draw/fit your histograms.
//
//   To use this file, try the following session on your Tree T
//
// Root > T->Process("se01.C")
// Root > T->Process("se01.C","some options")
// Root > T->Process("se01.C+")
//
//#include "se01.h"
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "Riostream.h"

#ifdef WITH_EVENT
#include "../test/Event.h"
#endif

void se::Begin(TTree *tree)
{
   // Function called before starting the event loop.
   // Initialize the tree branches.

   Init(tree);

   TString option = GetOption();

}

Bool_t se::Process(Int_t entry)
{
   // Processing function.
   // Entry is the entry number in the current tree.
   // Read only the necessary branches to select entries.
   // To read complete event, call fChain->GetTree()->GetEntry(entry).
   // Return kFALSE to stop processing.

   return kTRUE;
}

Bool_t se::ProcessCut(Int_t entry)
{
   // Selection function.
   // Entry is the entry number in the current tree.
   // Read only the necessary branches to select entries.
   // Return kFALSE as soon as a bad entry is detected.
   // To read complete event, call fChain->GetTree()->GetEntry(entry).

   return kTRUE;
}

void se::ProcessFill(Int_t entry)
{
   // Function called for selected entries only.
   // Entry is the entry number in the current tree.
   // Read branches not processed in ProcessCut() and fill histograms.
   // To read complete event, call fChain->GetTree()->GetEntry(entry).

#ifdef seold_cxx
   fChain->GetEntry(entry);
#else
   fDirector.fEntry = entry;
#endif

   // Let's test reading a little bit;
   int ntracks = fNtrack;
   float temp = fTemperature;
   int meas0 = fMeasures[0];
   int meas1 = fMeasures[3];
   
   if (entry==1) {
      cout << ntracks << endl;
      cout << fNtrack << endl;
      cout << temp << endl;
      cout << meas0 << endl;
      cout << meas1 << endl;
      cout << fClosestDistance[2] << endl;
      cout << fType[0] << endl;
// there are problems with std::string in interpreted mode :(
#ifdef __CINT__
      // The aut-conversion to std::string does not work in CINT.
      const char *ctype = fType.c_str();
      string type = ctype;
      // the operator<<(std::string) is not available in CINT.
      //cout << type.c_str() << endl;
      cout << ctype << endl;
#else
      string type = fType;
      cout << type << endl;
#endif
      cout << "fMatrix[2][1]: " << fMatrix[2][1] << endl;
      cout << "fH->GetMean() " << fH->GetMean() << endl;

#ifdef seold_cxx
      cout << fEvtHdr_fEvtNum << endl;
      cout << fTracks_ << endl;
      cout << fTracks_fPx[0] << endl;
      cout << fTracks_fPx[1] << endl;
      cout << "fTracks.fNsp[2]: " << fTracks_fNsp[2] << endl;
      cout << "fTracks.fPointValue[2][1]: " << fTracks_fPointValue[2][1] << endl;
      cout << "fLastTrack: " << fLastTrack.GetUniqueID() << endl;
#else
      cout << fEvtHdr.fEvtNum << endl;
      cout << fTracks->GetLast()+1 << endl;
      cout << fTracks.fPx[0] << endl;
      cout << fTracks.fPx[1] << endl;
      cout << "fTracks.fNsp[0]: " << fTracks.fNsp[0] << endl;
      cout << "fTracks.fPointValue[0][0]: " << fTracks.fPointValue[0][0] << endl;
      cout << "fLastTrack: " << fLastTrack->GetUniqueID() << endl;
#endif
#ifdef WITH_EVENT
#ifdef seold_cxx
#else
      cout << "event->GetHeader()->GetEvtNum(): " << event->GetHeader()->GetEvtNum() << endl;
      // cout << "event->GetHeader()->GetEvtNum(): " << const_cast<Event*>(event)->GetHeader()->GetEvtNum() << endl;
      cout << "event->GetHeader()->GetEvtNum(): " << ((Event&)event).GetHeader()->GetEvtNum() << endl;
#endif      
#endif

   }

}

void se::Terminate()
{
   // Function called at the end of the event loop.


}
