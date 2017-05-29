/// \file
/// \ingroup tutorial_http
///  This program creates and fills one and two dimensional histogram
///  Macro used to demonstrate usage of custom HTML page in custom.htm
///  One can use plain JavaScript to assign different actions with HTML buttons
///
/// \macro_code
///
/// \author Sergey Linev

#include <TH1.h>
#include <TH2.h>
#include <THttpServer.h>
#include <TRandom3.h>

void custom()
{
   // Create two histograms
   TH1F *hpx = new TH1F("hpx","This is the px distribution",100,-4,4);
   TH2F *hpxpy = new TH2F("hpxpy","py vs px",40,-4,4,40,-4,4);

   // http server with port 8080, use jobname as top-folder name
   THttpServer* serv = new THttpServer("http:8080");

   // use custom web page as default
   serv->SetDefaultPage("custom.htm");

   // Fill histograms randomly
   TRandom3 random;
   Float_t px, py;

   // press Ctrl-C to stop macro
   while (!gSystem->ProcessEvents()) {
      random.Rannor(px,py);
      hpx->Fill(px);
      hpxpy->Fill(px,py);
   }
}
