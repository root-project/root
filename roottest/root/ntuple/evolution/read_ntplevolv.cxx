#include <ROOT/RNTupleReader.hxx>

#include <TObject.h>
#include <TROOT.h>
#include <TSeqCollection.h>

#include "NtplEvolv_v3.hxx"

#include <iostream>
#include <string>

int main()
{
   // At this point, we expect NtplEvolv _not_ being present in the global streamer infos
   for (auto si : TRangeDynCast<TObject>(gROOT->GetListOfStreamerInfo())) {
      if (std::string(si->GetName()) == "NtplEvolv")
         return 2;
   }

   auto reader = ROOT::RNTupleReader::Open("ntpl", "root_test_ntpl_evolution.root");

   reader->GetModel();
   // Now, reader should have loaded the streamer info for NtplEvolv
   bool streamerInfoFound = false;
   for (auto si : TRangeDynCast<TObject>(gROOT->GetListOfStreamerInfo())) {
      if (std::string(si->GetName()) == "NtplEvolv") {
         streamerInfoFound = true;
         break;
      }
   }
   if (!streamerInfoFound)
      return 3;

   reader->LoadEntry(0);

   auto a = reader->GetModel().GetDefaultEntry().GetPtr<NtplEvolv>("event")->fA;
   std::cout << "Result of event.fA: " << a << std::endl;

   return a != 14;
}
