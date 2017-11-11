/// \file
/// \ingroup tutorial_multicore
/// \notebook -js
/// Shows how to run items of work asynchronously with Async.
///
/// \macro_code
///
/// \author Danilo Piparo
/// \date August 2017

#include "TROOT.h"
#include "ROOT/TFuture.hxx"

#include <iostream>

int workItem0()
{
   printf("Running workItem0...\n");
   return 0;
}

void mt303_AsyncSimple()
{

   ROOT::EnableImplicitMT(1);

   auto wi0 = ROOT::Experimental::Async(workItem0);
   auto wi1 = ROOT::Experimental::Async([]() {
      printf("Running workItem1...\n");
      return 1;
   });

   printf("Running something in the \"main\" thread\n");

   std::cout << "The result of the work item 0 is " << wi0.get() << std::endl;
   std::cout << "The result of the work item 1 is " << wi1.get() << std::endl;

   printf("All work completed.\n");
}

int main()
{
   mt303_AsyncSimple();
   return 0;
}
