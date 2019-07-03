/// \file
/// \ingroup tutorial_webgui
///  This program demonstrates minimal server/client code for working with RWebWindow class
///  File server.cxx shows how RWebWindow can be created and used
///  In client.html simple client code is provided.
///
/// \macro_code
///
/// \author Sergey Linev

#include <ROOT/RWebWindow.hxx>

std::shared_ptr<ROOT::Experimental::RWebWindow> window;

int counter{0};

void ProcessData(unsigned connid, const std::string &arg)
{
   printf("Get msg %s \n", arg.c_str());

   counter++;

   if (arg == "get_text") {
      // send arbitrary text message
      window->Send(connid, Form("Message%d", counter));
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

void server()
{
   // create window, manager can handle many windows a time
   window = ROOT::Experimental::RWebWindow::Create();

   // configure default html page
   // either HTML code can be specified or just name of file after 'file:' prefix
   window->SetDefaultPage("file:client.html");

   // this is call-back, invoked when message received from client
   window->SetDataCallBack(ProcessData);

   window->SetGeometry(300, 500); // configure predefined geometry

   window->Show();
}
