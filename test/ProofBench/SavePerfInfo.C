#include <iostream>
#include "TFile.h"
#include "TIterator.h"
#include "TRegexp.h"
#include "TProof.h"


void SavePerfInfo(const Char_t *filename)
{
   // Save PROOF timing information from TPerfStats to file 'filename'

   if (!gProof) {
      cout << "PROOF must be run to save output performance information" << endl;
      return;
   }

   TFile f(filename, "UPDATE");
   if (f.IsZombie()) {
      cout << "Could not open file " << filename << " for writing" << endl;
   } else {
      f.cd();

      TIter NextObject(gProof->GetOutputList());
      TObject* obj = 0;
      while (obj = NextObject()) {
         TString objname = obj->GetName();
         if (objname.Contains(TRegexp("^PROOF_"))) {
            // must list the objects since other PROOF_ objects exist
            // besides timing objects
            if (objname == "PROOF_PerfStats" ||
                objname == "PROOF_PacketsHist" ||
                objname == "PROOF_EventsHist" ||
                objname == "PROOF_NodeHist" ||
                objname == "PROOF_LatencyHist" ||
                objname == "PROOF_ProcTimeHist" ||
                objname == "PROOF_CpuTimeHist")
               obj->Write();
         }
      }

      f.Close();
   }

}
