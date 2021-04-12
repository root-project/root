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
#include <iostream>

std::shared_ptr<ROOT::Experimental::RWebWindow> window;

int num_clients = 1;
bool window_terminated = false;
bool call_show = true;
bool batch_mode = false;
int current_counter = 0;

void ProcessData(unsigned connid, const std::string &arg)
{
   if (arg.find("PING:") == 0) {
      window->Send(connid, arg);
   } else if (arg == "first") {
      // first message to provide config
      window->Send(connid, std::string("CLIENTS:") + std::to_string(num_clients));
   } else if (arg.find("SHOW:") == 0) {
      std::string msg = arg.substr(5);
      if (!batch_mode)
         std::cout << msg << std::endl;
      if (msg.find("Cnt:") == 0) {
         int counter = std::stoi(msg.substr(4));
         if (counter > 0)
            current_counter = counter;
      }

   } else if (arg == "halt") {
      // terminate ROOT
      window_terminated = true;
      window->TerminateROOT();
   }
}

////////////////////////////////////////////////////////
/// @param nclients - number of clients
/// @param test_mode
///  0 - default config, no special threads
///  1 - reduce http server timer
///  2 - create special thread in THttpServer and use it
///  3 - also create special thread for RWebWindow
///  4 - directly use civetweb threads (only for experts)
/// 10 - force longpoll socket with default config

enum TestModes {
   modeDefault = 0,          // default configuration
   modeMinimalTimer = 1,     // reduce THttpServer timer
   modeHttpThread = 2,       // create and use THttpServer thread to handle window functionality
   modeHttpWindowThread = 3, // with THttpServer thread also create thread for the window
   modeCivetThread = 4       // directly use threads if civetweb, dangerous
};

enum MajorModes {
   majorDefault = 0,         // default test suite, using websockets
   majorLongpoll = 1         // force longpoll sockets
};

void ping(int nclients = 1, int test_mode = 0)
{
   num_clients = nclients;

   batch_mode = gROOT->IsBatch();

   // verify values
   if (test_mode < 0) test_mode = 0;
   int major_mode = test_mode / 10;
   test_mode = test_mode % 10;
   if (test_mode > modeCivetThread)
      test_mode = modeDefault;

   if (num_clients < 1)
      num_clients = 1;
   else if (num_clients > 1000)
      num_clients = 1000;

   // let force usage of longpoll engine instead of plain websocket
   if (major_mode == majorLongpoll)
      gEnv->SetValue("WebGui.WSLongpoll", "yes");

   if (num_clients > 5)
      gEnv->SetValue("WebGui.HttpThreads", num_clients + 5);

   // allocate special thread for THttpServer, it will be automatically used by web window
   if ((test_mode == modeHttpThread) || (test_mode == modeHttpWindowThread))
      gEnv->SetValue("WebGui.HttpThrd", "yes");

   // let reduce reaction time of THttpServer
   if (test_mode == modeMinimalTimer)
      gEnv->SetValue("WebGui.HttpTimer", 1);

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

   // allow to use server thread, which responsible for requests processing
   if (test_mode == modeCivetThread)
      window->UseServerThreads();

   if (test_mode == modeHttpWindowThread)
      window->StartThread();

   // instead of showing window one can create URL and type it in any browser window
   if (call_show)
      window->Show(batch_mode ? "headless" : "");
   else
      std::cout << "Window url is: " << window->GetUrl(true) << std::endl;

   // provide blocking method to let run
   if (batch_mode) {
      const int run_limit = 200;
      const double run_time = 50.;
      window->WaitFor([=](double tm) { return (current_counter >= run_limit) || (tm > run_time) ? 1 : 0; });
      if (current_counter >= run_limit)
         std::cout << "PING-PONG TEST COMPLETED" << std::endl;
      else
         std::cout << "PING-PONG TEST FAIL" << std::endl;
   }
}
