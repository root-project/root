// This test is a derivative of what's used in http://cern.ch/root-intro.
// It should work.

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1F.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"

int assertIntroTut()
{
   // open the file
   const auto fname = "root://eospublic.cern.ch//eos/root-eos/testfiles/eventdata.root";
   TFile *f= TFile::Open(fname);
   if (f == 0) {
      // if we cannot open the file, print an error message and return immediatly
      printf("Error: cannot open %s!\n", fname);
      return 1;
   }

   // Create tyhe tree reader and its data containers
   TTreeReader myReader("EventTree", f);
   // The branch "fPosX" contains doubles; access them as particlesPosX.
   TTreeReaderArray<Double_t> particlesPosX(myReader, "fParticles.fPosX");
   // The branch "fMomentum" contains doubles, too; access those as particlesMomentum.
   TTreeReaderArray<Double_t> particlesMomentum(myReader, "fParticles.fMomentum");

   // create the TH1F histogram
   TH1F *hPosX = new TH1F("hPosX", "Position in X", 20, -5, 5);
   // enable bin errors:
   hPosX->Sumw2();

   // Loop over all entries of the TTree or TChain.
   while (myReader.Next()) {
      // Do the analysis...
      for (int iParticle = 0; iParticle < particlesPosX.GetSize(); ++iParticle) {
         if (particlesMomentum[iParticle] > 40.0)
            hPosX->Fill(particlesPosX[iParticle]);
      }
      // For testing purposes, re-run with begin / end (ROOT-7362)
      for (auto i = particlesPosX.begin(), e = particlesPosX.end(); i != e; ++i) {
         if (*i > 40.0)
            hPosX->Fill(*i);
      }
      // For testing purposes, re-run with range-for
      for (double i: particlesPosX) {
         if (i > 40.0)
            hPosX->Fill(i);
      }
   }

   // Fit the histogram:
   hPosX->Fit("pol2");
   // and draw it:
   hPosX->Draw();


   // This would enable the loop below:
   // myReader.Restart();
   TTreeReaderValue<Int_t> eventSize(myReader, "fEventSize"); // should warn
   // Variables used to store the data
   Int_t totalSize = 0; // Sum of data size (in bytes) of all events
   // Loop over all entries of the TTree or TChain.
   while (myReader.Next()) {
      // Get the data from the current TTree entry by getting
      // the value from the connected reader (eventSize):
      totalSize += *eventSize;
      printf("This loop should not be entered!\n");
      return 2;
   }

   return 0;
}
