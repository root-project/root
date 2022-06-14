#include "TH2.h"
#include "TF1.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TEntryList.h"
#include "TPaveStats.h"
#include "TMath.h"

const Double_t dxbin = (0.17-0.13)/40;   // Bin-width
const Double_t sigma = 0.0012;

//_____________________________________________________________________
Double_t fdm5(Double_t *xx, Double_t *par)
{
   Double_t x = xx[0];
   if (x <= 0.13957) return 0;
   Double_t xp3 = (x-par[3])*(x-par[3]);
   Double_t res = dxbin*(par[0]*TMath::Power(x-0.13957, par[1])
       + par[2] / 2.5066/par[4]*TMath::Exp(-xp3/2/par[4]/par[4]));
   return res;
}

//_____________________________________________________________________
Double_t fdm2(Double_t *xx, Double_t *par)
{
   Double_t x = xx[0];
   if (x <= 0.13957) return 0;
   Double_t xp3 = (x-0.1454)*(x-0.1454);
   Double_t res = dxbin*(par[0]*TMath::Power(x-0.13957, 0.25)
       + par[1] / 2.5066/sigma*TMath::Exp(-xp3/2/sigma/sigma));
   return res;
}

