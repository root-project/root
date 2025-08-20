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

void efficienciesMulticlass1vsRest(TString dataset, TString filename_input = "TMVAMulticlass.root",
                                   EEfficiencyPlotType plotType = EEfficiencyPlotType::kRejBvsEffS,
                                   Bool_t useTMVAStyle = kTRUE);

void plotEfficienciesMulticlass1vsRest(TString dataset, EEfficiencyPlotType plotType = EEfficiencyPlotType::kRejBvsEffS,
                                       TString filename_input = "TMVAMulticlass.root");

void efficienciesMulticlass1vs1(TString dataset, TString fin);
void plotEfficienciesMulticlass1vs1(TString dataset, TString fin, TString baseClassname);

} // namespace TMVA

#endif
