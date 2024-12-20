#include "TClass.h"
#include "THashTable.h"
#include "TInterpreter.h"
#include "Math/GenVector/LorentzVector.h"
#include "Math/GenVector/PtEtaPhiM4D.h"
#include "Math/GenVector/PtEtaPhiE4D.h"

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

TEST(TClass, TypeNameDouble32)
{
   TClass *clLV32 = TClass::GetClass("ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<Double32_t> >");
   ASSERT_NE(clLV32, nullptr);
   EXPECT_EQ(strstr(clLV32->GetName(), "double"), nullptr);

   TClass *clTypeID32 = TClass::GetClass(typeid(ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> >));
   EXPECT_EQ(clLV32, clTypeID32) << "Only LV<Double32_t> should have been registered; typeid lookup should find it.";

   TClass *clLVd = TClass::GetClass("ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> >");
   ASSERT_NE(clLVd, nullptr);
   EXPECT_EQ(strstr(clLVd->GetName(), "Double32_t"), nullptr);
   EXPECT_NE(clLVd, clLV32);

   TClass *clTypeIDd = TClass::GetClass(typeid(ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double> >));
   EXPECT_EQ(clLVd, clTypeIDd) << "LV<double> should have priority; typeid lookup should find it.";
}


TEST(TClass, TypeNameDouble)
{
   TClass *clLVd = TClass::GetClass("ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >");
   ASSERT_NE(clLVd, nullptr);
   EXPECT_NE(strstr(clLVd->GetName(), "double"), nullptr);

   TClass *clTypeIDd = TClass::GetClass(typeid(ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >));
   EXPECT_EQ(clLVd, clTypeIDd) << "Only LV<Double32_t> should have been registered; typeid lookup should find it.";

   TClass *clLV32 = TClass::GetClass("ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<Double32_t> >");
   ASSERT_NE(clLV32, nullptr);
   EXPECT_NE(strstr(clLV32->GetName(), "Double32_t"), nullptr);
   EXPECT_NE(clLV32, clLVd);

   // <double> should not be overwritten by <Double32_t>.
   TClass *clTypeID32 = TClass::GetClass(typeid(ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> >));
   EXPECT_EQ(clLVd, clTypeID32) << "LV<double> should have priority; typeid lookup should find it.";
}