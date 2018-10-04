#include "TClass.h"
#include "THashTable.h"
#include "TInterpreter.h"

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
