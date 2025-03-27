/// \file
/// \ingroup tutorial_webgui
/// \ingroup webwidgets
/// Use of `bootstrap` framework together with RWebWindow class.
///
/// In webwindow.cxx RWebWindow created
/// In `dist` directory HTML files generated with bootstrap are placed
/// One also can see usage of embed TWebCanvas in such application
///
/// \macro_code
///
/// \author Sergey Linev

#include <ROOT/RWebWindow.hxx>

#include "TCanvas.h"
#include "TWebCanvas.h"
#include "TTimer.h"

std::shared_ptr<ROOT::RWebWindow> window;

TCanvas *canvas = nullptr;
TH1I *hist = nullptr;

int counter = 0;

void ProcessData(unsigned connid, const std::string &arg)
{
   printf("Get msg %s \n", arg.c_str());

   counter++;

   if (arg == "get_text") {
      // send arbitrary text message
      window->Send(connid, TString::Format("Message%d", counter).Data());
   } else if (arg == "get_binary") {
      // send float array as binary
      float arr[10];
      for (int n = 0; n < 10; ++n)
         arr[n] = counter;
      window->SendBinary(connid, arr, sizeof(arr));
   } else if (arg == "halt") {
      // terminate ROOT
      window->TerminateROOT();
   } else if (arg.compare(0, 8, "channel:") == 0) {
      int chid = std::stoi(arg.substr(8));
      printf("Get channel request %d\n", chid);
      auto web_imp = dynamic_cast<TWebCanvas *>(canvas->GetCanvasImp());
      web_imp->ShowWebWindow({ window, connid, chid });
   }
}

void update_canvas()
{
   hist->FillRandom("gaus", 5000);
   canvas->Modified();
   canvas->Update();
}

void webwindow()
{
   // create window
   window = ROOT::RWebWindow::Create();

   // create TCanvas with pre-configure web display
   canvas = TWebCanvas::CreateWebCanvas("Canvas1", "Example of web canvas");


   hist = new TH1I("hpx", "Test histogram", 40, -5, 5);
   hist->FillRandom("gaus", 10000);
   canvas->Add(hist);

   // start regular fill of histogram and update of canvas
   auto timer = new TTimer("update_canvas()", 2000, kFALSE);
   timer->TurnOn();


   // configure default html page
   // either HTML code can be specified or just name of file after 'file:' prefix
   std::string fdir = __FILE__;
   auto pos = fdir.find("webwindow.cxx");
   if (pos > 0)
      fdir.resize(pos);
   else
      fdir = "./";
   window->SetDefaultPage("file:" + fdir + "dist/index.html");

   // this is call-back, invoked when message received from client
   window->SetDataCallBack(ProcessData);

   window->SetGeometry(1200, 800); // configure predefined geometry

   window->Show();
}
