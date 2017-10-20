/// \file
/// \ingroup tutorial_multicore
/// \notebook -js
/// Calculate Fibonacci numbers exploiting nested parallelism through Async.
///
/// \macro_code
///
/// \author Danilo Piparo
/// \date August 2017

#include "TROOT.h"
#include "ROOT/TFuture.hxx"

#include <future>
#include <iostream>

int Fibonacci(int n)
{
   if (n < 2) {
      return n;
   } else {
      auto fut1 = ROOT::Experimental::Async(Fibonacci, n - 1);
      auto fut2 = ROOT::Experimental::Async(Fibonacci, n - 2);
      auto res = fut1.get() + fut2.get();
      return res;
   }
}

void mt304_AsyncNested()
{

   ROOT::EnableImplicitMT(4);

   std::cout << "Fibonacci(33) = " << Fibonacci(33) << std::endl;
}

int main()
{
   mt304_AsyncNested();
   return 0;
}
