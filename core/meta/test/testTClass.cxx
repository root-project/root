#include "TClass.h"
#include "THashTable.h"
#include "TInterpreter.h"
#include "TSystem.h"

#include <ROOT/TSeq.hxx>

#include "gtest/gtest.h"

TEST(TClass, DictCheck)
{
   gInterpreter->ProcessLine(".L stlDictCheck.h+");
   auto c = TClass::GetClass("B");

   THashTable classesWithoutDictionary;
   c->GetMissingDictionaries(classesWithoutDictionary, /*recursive*/ true);

   std::string errMsg("Missing dictionary for ");

   for (auto item : classesWithoutDictionary) {
      auto const cl = static_cast<TClass*>(item);
      errMsg += cl->GetName();
      errMsg += ", ";
   }

   EXPECT_STREQ(errMsg.c_str(), "Missing dictionary for C, ") << errMsg;
}

// This test is for issue #9029
TEST(TClass, GetClassWithFundType)
{
   ProcInfo_t info;
   gSystem->GetProcInfo(&info);
   auto start_mem = info.fMemResident;

   constexpr auto name = "std::vector<int>::value_type";
   TClass *cl = nullptr;
   for (auto i : ROOT::TSeqI(8192)) {
      cl = TClass::GetClass(name);
      i = (int)i; // avoids unused variable warning
   }

   gSystem->GetProcInfo(&info);
   auto end_mem = info.fMemResident;

   EXPECT_TRUE(nullptr == cl);
   EXPECT_NEAR(start_mem, end_mem, 16384);
}
