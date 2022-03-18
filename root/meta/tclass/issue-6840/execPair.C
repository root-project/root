#include "TROOT.h"
#include "TInterpreter.h"
#include "TClass.h"
#include "TStreamerInfo.h"
#include "TFile.h"
#include "TError.h"
#include "TSystem.h"


int check(const char *args, size_t old_index)
{
   TString collectionname("map");
   collectionname += args;
   TString pairname("pair");
   pairname += args;

   auto cl = TClass::GetClass(collectionname);
   if (!cl) {
      Error("pair offset-size", "Can not get the %s TClass", collectionname.Data());
      return 1;
   }
   if (!cl->IsLoaded()) {
      Error("pair offset-size", "Missing dictionary for %s TClass", collectionname.Data());
      return 2;
   }
   auto proxy = cl->GetCollectionProxy();
   if (!proxy) {
      Error("pair offset-size", "Missing collection proxy for %s TClass", collectionname.Data());
      return 3;
   }

   // Provoke the recreation of the pair's TClass.
   proxy->GetIncrement();

   cl = TClass::GetClass(pairname);
   if (!cl) {
      Error("pair offset-size", "Can not get the %s TClass after library loading", pairname.Data());
      return 4;
   }
   if ((int)cl->GetClassSize() != (int)proxy->GetIncrement()) {
      Error("pair offset-size", "The TClass for %s and the proxy for the map disagree on size: %d vs %d after library loading",
            pairname.Data(), (int)cl->GetClassSize(), (int)proxy->GetIncrement());
      return 5;
   }
   if ((int)cl->GetClassSize() != (int)cl->GetStreamerInfo()->GetSize()) {
      Error("pair offset-size", "EXPECTED FOR NOW The TClass for %s and its StreamerInfo disagree on size: %d vs %d after library loading",
            pairname.Data(), (int)cl->GetClassSize(), (int)cl->GetStreamerInfo()->GetSize());
      cl->GetStreamerInfo()->ls();
      return 6;
   }

   return 0;
}

int execPair(const char *filename = "pair.root")
{
   gInterpreter->SetClassAutoLoading(false);
   TInterpreter::SuspendAutoParsing s(gInterpreter);

   // Load "old" StreamerInfo for std::pair<SameAsShort, SameAsShort> and std::pair<short, SameAsShort>
   std::unique_ptr<TFile> file(TFile::Open(filename, "READ"));
   if (!file || file->IsZombie()) {
      Error("pair offset-size", "Could not open the file %s", filename);
      return 1;
   }

   if (!file->GetStreamerInfoList() || !file->GetStreamerInfoList()->FindObject("pair<SameAsShort,SameAsShort>")) {
      Error("pair offset-size", "StreamerInfo for pair<SameAsShort, SameAsShort> not in %s", filename);
      if (file->GetStreamerInfoList())
         file->GetStreamerInfoList()->ls("");
      return 2;
   }

   TClass *cl = TClass::GetClass("pair<SameAsShort, SameAsShort>");
   if (!cl) {
      Error("pair offset-size", "Can not get the pair<SameAsShort, SameAsShort> TClass)");
      return 3;
   }
   if (cl->IsLoaded()) {
      Error("pair offset-size", "Test is ineffective, found a dictionary for pair<SameAsShort, SameAsShort>");
      return 4;
   }
   if (cl->GetState() == TClass::kInterpreted || cl->HasInterpreterInfo()) {
      Error("pair offset-size", "Test is ineffective, found a interpreter info for pair<SameAsShort, SameAsShort>: state=%d hasInterpreterInfo=%d name=%s",
            cl->GetState(), cl->HasInterpreterInfo(), cl->GetName());
      return 5;
   }
#ifdef R__B64
   constexpr int ssExpectedSize = 32;
#else
   constexpr int ssExpectedSize = 16;
#endif
   if (cl->GetClassSize() != ssExpectedSize) {
      Error("pair offset-size", "Test is ineffective, the calculated size for pair<short, short> is no longer conservative (%d instead of %d)", cl->GetClassSize(), ssExpectedSize);
      return 6;
   }
   auto info = cl->GetStreamerInfo();
   if (!info) {
      Error("pair offset-size", "Can not get the StreamerInfo for pair<short, short> TClass)");
      return 7;
   }
   auto original_index_1 = info->GetNumber();
   if (info != gROOT->GetListOfStreamerInfo()->At(original_index_1)) {
      Error("pair offset-size", "StreamerInfo for pair<short, short> is not at the expected index in the list of StreamerInfo (index: %d)",
            original_index_1);
      return 8;
   }

   cl = TClass::GetClass("pair<short, SameAsShort>");
   if (!cl) {
      Error("pair offset-size", "Can not get the pair<short, SameAsShort> TClass)");
      return 11;
   }
   if (cl->IsLoaded()) {
      Error("pair offset-size", "Test is ineffective, found a dictionary for pair<short, SameAsShort>");
      return 12;
   }
#ifdef R__B64
   constexpr int saExpectedSize = 24;
#else
   constexpr int saExpectedSize = 12;
#endif
   if (cl->GetClassSize() != saExpectedSize) {
      Error("pair offset-size", "Test is ineffective, the calculated size for pair<short, SameAsShort> is no longer conservative (%d instead of %d)", cl->GetClassSize(), saExpectedSize);
      return 13;
   }
   info = cl->GetStreamerInfo();
   if (!info) {
      Error("pair offset-size", "Can not get the StreamerInfo for pair<short, SameAsShort> TClass)");
      return 14;
   }
   auto original_index_2 = info->GetNumber();
   if (info != gROOT->GetListOfStreamerInfo()->At(original_index_2)) {
      Error("pair offset-size", "StreamerInfo for pair<short, SameAsShort> is not at the expected index in the list of StreamerInfo (index: %d)",
            original_index_2);
      return 15;
   }



   ///
   /// Loading the library
   ///

   if (0 != gSystem->Load("libPairs")) {
      Error("Reload reproducer", "Can not load libPairs");
      return 30;
   }

   // No dictionary
   auto result = check("<short, SameAsShort>", original_index_2);
   if (result)
      return 30+result;

   // No dictionary but could have interpreter info without AutoParsing.
   result = check("<short, short>", 0);
   if (result)
      return 40+result;

   // Has dictionary
   result = check("<SameAsShort, SameAsShort>", original_index_1);
   if (result)
      return 50+result;

   return 0;
}

