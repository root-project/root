#ifndef compareanapp__HH
#define compareanapp__HH
#include "TMVA/tmvaglob.h"
#include "TMVA/Types.h"
namespace TMVA{


#define CheckDerivedPlots 0

   void compareanapp( TString finAn = "TMVA.root", TString finApp = "TMVApp.root", 
                      HistType htype = MVAType, bool useTMVAStyle=kTRUE );
}
#endif
