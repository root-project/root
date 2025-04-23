#include <TStopwatch.h>

#include <iostream>
#include <fstream>
using namespace std;
#include "TSystem.h"
#define ClingWorkAroundMissingDynamicScope
#include "TBenchmark.h"
#include "TChain.h"

int Read(TString library, TString rootfilename, Bool_t ref = kFALSE)
{
   std::string markfilename = "NuEvent_DST.mark";
   const int tolerance = 100;

   gSystem->Load(library);  

   gBenchmark = new TBenchmark(); 
   gBenchmark->Start("Read");

   // Open the input file and prepare for event reading
   TChain input("s", "s");
   input.AddFile(rootfilename);
   TBranch *branch = input.GetBranch("s");
#ifdef ClingWorkAroundMissingDynamicScope
   void *nu = TClass::GetClass("NuEvent")->New();
#else
   NuEvent *nu = new NuEvent();
#endif
   branch->SetAddress(&nu);

   cout  << endl
         << "************************************************"<<endl
         << "***      Starting main loop                  ***"<<endl
         << "************************************************"<<endl;
   // Loop over all input NuEvents
   for (Int_t i=0;i<input.GetEntries();++i) {
      // Print the progress
      if (i%10 == 0) cout << "Reading event " << i << endl;
      
      if (i >= 100) {
         break;
      }
      
      // Get the event
      branch->GetEntry(i);
   }
   gBenchmark->Stop("Read");

   cout << "All events read." << endl;
   Float_t ct = gBenchmark->GetCpuTime("Read");
   //Float_t refct;
   //cout << "Cputime: " << ct << endl;
   if (ref) {
      ofstream markfile(markfilename.c_str());
      markfile << ct << '\n';
      return 0; // Success.
   } else {
      Float_t refct;
#if defined( _LIBCPP_VERSION ) && defined(__CLING__)
      FILE *markfile = fopen(markfilename.c_str(),"r");
      if (markfile) {
         fscanf(markfile,"%f",&refct);
         fclose(markfile);
      }
#else
      ifstream markfile(markfilename.c_str());
      markfile >> refct;
#endif
      if ( TMath::Abs( (refct - ct) / refct ) > (tolerance/100.0)) {
         cout << "Reading time for " << rootfilename << " with " << library
              << " takes " << tolerance << "% more than the reference " << ct << " vs " << refct << "\n";
         return 1; // Failure
      }
      return 0; // Success.
   }
   return  tolerance && ref && ct;
}
