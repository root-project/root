#ifndef probas__HH
#define probas__HH
#include <iostream>
#include <iomanip>
using std::cout;
using std::endl;

#include "tmvaglob.h"

#include "RQ_OBJECT.h"

#include "TH1.h"
#include "TROOT.h"
#include "TList.h"
#include "TIterator.h"
#include "TStyle.h"
#include "TPad.h"
#include "TCanvas.h"
#include "TLatex.h"
#include "TLegend.h"
#include "TLine.h"
#include "TH2.h"
#include "TFormula.h"
#include "TFile.h"
#include "TApplication.h"
#include "TKey.h"
#include "TClass.h"
#include "TGaxis.h"

#include "TGWindow.h"
#include "TGButton.h"
#include "TGLabel.h"
#include "TGNumberEntry.h"
namespace TMVA{

   // this macro plots the MVA probability distributions (Signal and
   // Background overlayed); of different MVA methods run in TMVA
   // (e.g. running TMVAnalysis.C).

   // input: - Input file (result from TMVA);
   //        - use of TMVA plotting TStyle
   void probas( TString fin = "TMVA.root", Bool_t useTMVAStyle = kTRUE );
}
#endif
