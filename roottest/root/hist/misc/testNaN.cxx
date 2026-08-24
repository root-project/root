// This script is mostly based on Hageboecks's one from https://github.com/root-project/root/issues/23125

#include "TH1.h"
#include "TMath.h"

void testNaN() {
   TH1D histo("histo", "histo", 10, 0, 10);
   histo.SetBinContent(1, TMath::Sqrt(-1));
   histo.Fill(3);
   histo.Draw();
}
