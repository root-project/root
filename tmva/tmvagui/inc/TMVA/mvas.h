#ifndef mvas__HH
#define mvas__HH
#include "TLegend.h"
#include "TText.h"
#include "TH2.h"

#include "TMVA/tmvaglob.h"
#include "TMVA/Types.h"
namespace TMVA{

   // this macro plots the resulting MVA distributions (Signal and
   // Background overlayed); of different MVA methods run in TMVA
   // (e.g. running TMVAnalysis.C).


   // input: - Input file (result from TMVA);
   //        - use of TMVA plotting TStyle
   void mvas(TString dataset, TString fin = "TMVA.root", HistType htype = kMVAType, Bool_t useTMVAStyle = kTRUE );
}
#endif
