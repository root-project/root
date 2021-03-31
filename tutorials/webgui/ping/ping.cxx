/// \file
/// \ingroup tutorial_webgui
///  This is test suite for RWebWindow communication performance
///  On the first place latency of round-trip (ping-pong) packet is measured
///  File ping.cxx implements server-side code of RWebWindow
///  In ping.html client code plus visualization is provided.
///
/// \macro_code
///
/// \author Sergey Linev

#include <ROOT/RWebWindow.hxx>

std::shared_ptr<ROOT::Experimental::RWebWindow> window;

void ProcessData(unsigned connid, const std::string &arg)
{
   // printf("Get msg %s \n", arg.c_str());

   if (arg.find("PING:") == 0) {
      window->Send(connid, arg);
   } else if (arg == "halt") {
      // terminate ROOT
      window->TerminateROOT();
   }
}

void ping()
{
   // create window
   window = ROOT::Experimental::RWebWindow::Create();

   // configure default html page
   // either HTML code can be specified or just name of file after 'file:' prefix
   window->SetDefaultPage("file:ping.html");

   // this is call-back, invoked when message received from client
   window->SetDataCallBack(ProcessData);

   window->SetGeometry(300, 500); // configure predefined geometry

   window->Show();
}
