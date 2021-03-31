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

#include <thread>
#include <iostream>

std::shared_ptr<ROOT::Experimental::RWebWindow> window;

int num_clients = 1;
bool window_terminated = false;
bool call_show = true;

void ProcessData(unsigned connid, const std::string &arg)
{
   if (arg.find("PING:") == 0) {
      window->Send(connid, arg);
   } else if (arg == "first") {
      // first message to provide config
      window->Send(connid, std::string("CLIENTS:") + std::to_string(num_clients));
   } else if (arg.find("SHOW:") == 0) {
      std::cout << arg.substr(5) << std::endl;
   } else if (arg == "halt") {
      // terminate ROOT
      window_terminated = true;
      window->TerminateROOT();
   }
}

/** Special thread to run web window */
void RunWebWindow()
{
   window->AssignThreadId();

   while (!window_terminated)
     window->Run(1.); // run event loop for 1 sec
}

////////////////////////////////////////////////////////
 /// @param nclients - number of clients
 /// @param special_thread - 0 - no thread, 1 - extra thread to process requests, 2 - use http server threads

void ping(int nclients = 1, int special_thread = 0)
{
   num_clients = nclients;

   // verify value
   if (num_clients < 1)
      num_clients = 1;
   else if (num_clients > 1000)
      num_clients = 1000;

   if (num_clients > 5)
      gEnv->SetValue("WebGui.HttpThreads", num_clients + 5);

   // let configure special thread which is used to handle incoming http requests
   // gEnv->SetValue("WebGui.HttpThrd", "yes");

   // let allocate special thread which will be used to perform data sending via websocket
   // should reduce consumption of webwindow thread when big data are send
   // gEnv->SetValue("WebGui.SenderThrds", "yes");


   // create window
   window = ROOT::Experimental::RWebWindow::Create();

   // configure maximal number of clients which allowed to connect
   window->SetConnLimit(num_clients);

   // configure default html page
   // either HTML code can be specified or just name of file after 'file:' prefix
   window->SetDefaultPage("file:ping.html");

   // configure window geometry
   window->SetGeometry(300, 500);

   // this set call-back, invoked when message received from client
   // also at this moment thread id is configured which supposed to be used to handle requests
   window->SetDataCallBack(ProcessData);

   // instead of showing window one can create URL and type it in any browser window
   if (call_show)
      window->Show();
   else
      std::cout << "Window url is: " << window->GetUrl(true) << std::endl;

   if (special_thread == 2) {
      // allow use server threads,
      // try to achieve minimal possible latency
      window->UseServerThreads();
   } else if (special_thread > 0) {
      // run special thread for RWebWindow
      std::thread thrd(RunWebWindow);
      thrd.detach();
   } else {
      // do nothing, callbacks handled by ProcessEvents
      // latency defined by the timer used in THttpServer
   }
}
