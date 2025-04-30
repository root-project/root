#include <ROOT/RNTupleReader.hxx>

#include "Event_v3.hxx"

#include <cstdio>

int main()
{
   auto reader = ROOT::RNTupleReader::Open("ntpl", "root_test_streamerfield.root");

   auto ptrEvent = reader->GetModel().GetDefaultEntry().GetPtr<Event>("event");

   reader->LoadEntry(0);

   if (!dynamic_cast<const ROOT::RStreamerField *>(&reader->GetModel().GetConstField("event.fField")))
      return 10;

   if (!ptrEvent->fField.fPtr)
      return 1;
   auto derived = dynamic_cast<StreamerDerived *>(ptrEvent->fField.fPtr.get());
   if (!derived)
      return 2;
   if (derived->fBase != 1000)
      return 3;
   if (derived->fFirst != 1)
      return 4;
   if (derived->fSecond != 2)
      return 5;
   if (ptrEvent->fY != 42)
      return 6;

   std::remove("root_test_streamerfield.root");
   return 0;
}
