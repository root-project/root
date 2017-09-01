// test that TDF correctly throws exceptions when users re-define a column already present in the dataframe
#include "ROOT/TDataFrame.hxx"
#include <stdexcept>
#include <string>
using namespace ROOT::Experimental;

int main()
{
   auto exceptionCount = 0u;
   const auto expectedExceptionCount = 2u;

   // create tree
   TDataFrame newd(1);
   newd.Define("x", [] { return 1; }).Snapshot<int>("t", "coloverride.root", {"x"});

   ROOT::Experimental::TDataFrame d("t", "coloverride.root");
   // re-define TTree variable
   try {
      auto c = d.Define("x", [] { return 2; }).Count();
   } catch (const std::runtime_error &e) {
      std::string msg(e.what());
      const auto expected_msg = "branch \"x\" already present in TTree";
      if (msg.find(expected_msg) == std::string::npos)
         throw;
      exceptionCount++;
   }

   // re-define TTree variable (jitted `Define`)
   try {
      auto c = d.Define("x", "2").Count();
   } catch (const std::runtime_error &e) {
      std::string msg(e.what());
      const auto expected_msg = "branch \"x\" already present in TTree";
      if (msg.find(expected_msg) == std::string::npos)
         throw;
      exceptionCount++;
   }

   if (exceptionCount != expectedExceptionCount)
      return 1;

   return 0;
}
