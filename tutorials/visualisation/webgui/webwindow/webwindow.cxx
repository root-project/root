/// \file
/// \ingroup tutorial_webgui
/// \ingroup webwidgets
/// Minimal server/client code for working with RWebWindow class.
///
/// File webwindow.cxx shows how RWebWindow can be created and used
/// In webwindow.html simple client code is provided.
///
/// \macro_code
///
/// \author Sergey Linev

#include <ROOT/RWebWindow.hxx>

std::shared_ptr<ROOT::RWebWindow> window;

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
   }
}

void webwindow()
{
   // create window
   window = ROOT::RWebWindow::Create();

   // configure default html page
   // either HTML code can be specified or just name of file after 'file:' prefix
   std::string fdir = __FILE__;
   auto pos = fdir.find("webwindow.cxx");
   if (pos > 0)
      fdir.resize(pos);
   else
      fdir = gROOT->GetTutorialsDir() + std::string("/visualisation/webgui/webwindow/");
   window->SetDefaultPage("file:" + fdir + "webwindow.html");

   // this is call-back, invoked when message received from client
   window->SetDataCallBack(ProcessData);

   window->SetGeometry(300, 500); // configure predefined geometry

   window->Show();
}
