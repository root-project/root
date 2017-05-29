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

} // namespace TMVA

#endif
