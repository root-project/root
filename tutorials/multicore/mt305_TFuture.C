/// \file
/// \ingroup tutorial_multicore
/// \notebook -js
/// Shows how to use the Future class of ROOT as a wrapper of std::future.
///
/// \macro_code
///
/// \author Danilo Piparo
/// \date August 2017

#include "ROOT/TFuture.hxx"

#include <future>
#include <iostream>
#include <thread>

void mt305_TFuture()
{
   using namespace ROOT::Experimental;

   // future from a packaged_task
   std::packaged_task<int()> task([]() { return 7; }); // wrap the function
   TFuture<int> f1 = task.get_future();                // get a future
   std::thread(std::move(task)).detach();              // launch on a thread

   // future from an async()
   TFuture<int> f2 = std::async(std::launch::async, []() { return 8; });

   // future from a promise
   std::promise<int> p;
   TFuture<int> f3(p.get_future());
   std::thread([&p] { p.set_value(9); }).detach();

   std::cout << "Waiting..." << std::flush;
   f1.wait();
   f2.wait();
   f3.wait();
   std::cout << "Done!\nResults are: " << f1.get() << ' ' << f2.get() << ' ' << f3.get() << '\n';
}
