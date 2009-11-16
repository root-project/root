#include <TBenchmark.h>

int Read(TString library, TString rootfilename, Bool_t ref = kFALSE)
{
   std::string reffilename = "NuEvents_DST.ref";
   const int tolerance = 10;

   gBenchmark = new TBenchmark(); 
   gBenchmark->Start("Read");
   gSystem->Load(library);  

   // Open the input file and prepare for event reading
   TChain input("s", "s");
   input.AddFile(rootfilename);
   TBranch *branch = input.GetBranch("s");
   NuEvent *nu = new NuEvent();
   branch->SetAddress(&nu);
   
   cout  << endl
         << "************************************************"<<endl
         << "***      Starting main loop                  ***"<<endl
         << "************************************************"<<endl;
   // Loop over all input NuEvents
   for (Int_t i=0;i<input.GetEntries();++i) {
      // Print the progress
      cout << "Reading event " << i << endl;
      
      if (i >= 10) {
         break;
      }
      
      // Get the event
      branch->GetEntry(i);
   }
   
   cout << "All events read." << endl;
   gBenchmark->Stop("Read");
   Float_t ct = gBenchmark->GetCpuTime("Read");
   Float_t refct;
   //cout << "Cputime: " << ct << endl;
   if (ref) {
      ofstream reffile(reffilename.c_str());
      reffile << ct << '\n';
      return 0; // Success.
   } else {
      ifstream reffile(reffilename.c_str());
      reffile >> refct;
      if ( TMath::Abs( (refct - ct) / refct ) > (tolerance/100.0)) {
         cout << "Reading time for " << rootfilename << " with " << library 
              << " takes " << tolerance << "% more than the reference " << ct << " vs " << refct << "\n";
         return 1; // Failure
      }
      return 0; // Success.
   }
}
