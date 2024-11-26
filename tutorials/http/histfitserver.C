/// \file
/// \ingroup tutorial_http
///  This program demonstrates simultaneous update of histogram and fitted function.
///  Every second new random entries add and histogram fitted again.
///  Required at least JSROOT version 5.1.1 to see correct fit function update in browser
///
/// \macro_code
///
/// \author  Sergey Linev


#include "THttpServer.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TF1.h"
#include "TSystem.h"


void histfitserver(void)
{
   auto serv = new THttpServer("http:8081");
   auto h1 = new TH1F("h1", "histogram 1", 100, -5, 5);
   auto c1 = new TCanvas("c1");
   auto f1 = new TF1("f1", "gaus", -10, 10);

   c1->cd();
   h1->Draw();

   while (!gSystem->ProcessEvents()) {
      h1->FillRandom("gaus", 100);
      h1->Fit(f1);
      c1->Modified();
      c1->Update();
      gSystem->Sleep(1000);
   }
}
