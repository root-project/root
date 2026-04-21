#include <ROOT/RNTupleReader.hxx>

#include "NtplEvolv_v3.hxx"

#include <iostream>

int main()
{
   auto reader = ROOT::RNTupleReader::Open("ntpl", "root_test_ntpl_evolution.root");

   reader->LoadEntry(0);

   auto a = reader->GetModel().GetDefaultEntry().GetPtr<NtplEvolv>("event")->fA;
   std::cout << "Result of event.fA: " << a << std::endl;

   return a != 13;
}
