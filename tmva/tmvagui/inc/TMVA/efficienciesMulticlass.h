#ifndef efficienciesMulticlass__HH
#define efficienciesMulticlass__HH

#include "tmvaglob.h"

class TCanvas;
class TDirectory;
class TFile;
class TGraph;
class TString;

namespace TMVA {

enum class EEfficiencyPlotType { kEffBvsEffS, kRejBvsEffS };

void efficienciesMulticlass(TString dataset, TString filename_input = "TMVAMulticlass.root",
                            EEfficiencyPlotType plotType = EEfficiencyPlotType::kRejBvsEffS,
                            Bool_t useTMVAStyle = kTRUE);

void plotEfficienciesMulticlass(EEfficiencyPlotType plotType = EEfficiencyPlotType::kRejBvsEffS,
                                TDirectory *BinDir = 0);

class EfficiencyPlotWrapper {

public:
   TCanvas *fCanvas;
   TLegend *fLegend;

   TString fClassname;
   Int_t fColor;

   UInt_t fNumMethods;

   EfficiencyPlotWrapper(TString title);
   Int_t addGraph(TGraph *graph);

   void addLegendEntry(TString methodTitle, TGraph *graph);

private:
   Float_t fx0L;
   Float_t fdxL;
   Float_t fy0H;
   Float_t fdyH;

   TCanvas *newEfficiencyCanvas(TString className);
   TLegend *newEfficiencyLegend();
};

} // namespace TMVA

#endif
