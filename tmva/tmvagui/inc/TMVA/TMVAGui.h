#ifndef TMVAGui__HH
#define TMVAGui__HH
#include <iostream>
#include <vector>

#include "TList.h"
#include "TROOT.h"
#include "TKey.h"
#include "TString.h"
#include "TControlBar.h"
#include "TObjString.h"
#include "TClass.h"

#include "tmvaglob.h"
#include "TMVA/variablesMultiClass.h"
#include "TMVA/variables.h"
#include "TMVA/TMVARegGui.h"
#include "TMVA/TMVAMultiClassGui.h"
#include "TMVA/TMVAGui.h"
#include "TMVA/tmvaglob.h"
#include "TMVA/rulevisHists.h"
#include "TMVA/rulevis.h"
#include "TMVA/rulevisCorr.h"
#include "TMVA/regression_averagedevs.h"
#include "TMVA/probas.h"
#include "TMVA/PlotFoams.h"
#include "TMVA/paracoor.h"
#include "TMVA/network.h"
#include "TMVA/mvaweights.h"
#include "TMVA/mvasMulticlass.h"
#include "TMVA/mvas.h"
#include "TMVA/mvaeffs.h"
#include "TMVA/MovieMaker.h"
#include "TMVA/likelihoodrefs.h"
#include "TMVA/efficiencies.h"
#include "TMVA/deviations.h"
#include "TMVA/CorrGuiMultiClass.h"
#include "TMVA/CorrGui.h"
#include "TMVA/correlationsMultiClass.h"
#include "TMVA/correlations.h"
#include "TMVA/correlationscattersMultiClass.h"
#include "TMVA/correlationscatters.h"
#include "TMVA/compareanapp.h"
#include "TMVA/BoostControlPlots.h"
#include "TMVA/BDT_Reg.h"
#include "TMVA/BDT.h"
#include "TMVA/BDTControlPlots.h"
#include "TMVA/annconvergencetest.h"
namespace TMVA{


   TList* GetKeyList( const TString& pattern );

   // utility function
   void ActionButton( TControlBar* cbar, 
                      const TString& title, const TString& macro, const TString& comment, 
                      const TString& buttonType, TString requiredKey = ""); 

   // main GUI
   void TMVAGui( const char* fName = "TMVA.root" );

   struct  TMVAGUI {
      TMVAGUI(TString name = "TMVA.root" ) {
         TMVA::TMVAGui(name.Data());
      }
   };
   
}


#endif
