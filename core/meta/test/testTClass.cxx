#include "TClass.h"
#include "THashTable.h"
#include "TInterpreter.h"
#include "Math/GenVector/LorentzVector.h"
#include "Math/GenVector/PtEtaPhiM4D.h"
#include "Math/GenVector/PtEtaPhiE4D.h"

#include "gtest/gtest.h"

#include <deque>

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

// https://github.com/root-project/root/issues/18643
TEST(TClass, BuildRealData)
{
   TClass::GetClass("TClass")->BuildRealData();
}

// https://github.com/root-project/root/issues/18654
TEST(TClass, ConsistentSTLLookup)
{
   // The lookup via normalised shortened name yielded a different class
   // than lookup via typeid. This lead to crashes in cppyy.
   auto first = TClass::GetClass("unordered_map<string,char>", true, true);
   std::unordered_map<std::string, char> map;
   auto second = first->GetActualClass(&map);
   EXPECT_EQ(first, second);
}

TEST(TClass, CollectionSizeof)
{
   // https://its.cern.ch/jira/browse/ROOT-9889
   EXPECT_EQ(sizeof(std::deque<short>), TClass::GetClass("std::deque<short>")->GetClassSize());
   EXPECT_EQ(sizeof(std::deque<unsigned short>), TClass::GetClass("std::deque<unsigned short>")->GetClassSize());
   EXPECT_EQ(sizeof(std::deque<int>), TClass::GetClass("std::deque<int>")->GetClassSize());
   EXPECT_EQ(sizeof(std::deque<unsigned int>), TClass::GetClass("std::deque<unsigned int>")->GetClassSize());
   EXPECT_EQ(sizeof(std::deque<long>), TClass::GetClass("std::deque<long>")->GetClassSize());
   EXPECT_EQ(sizeof(std::deque<unsigned long>), TClass::GetClass("std::deque<unsigned long>")->GetClassSize());
}

TEST(TClass, ReSubstTemplateArg)
{
   // #18811
   gInterpreter->Declare("template <typename T> struct S {};"
                         "template <typename T1, typename T2> struct Two { using value_type = S<T2>; };"
                         "template <typename T> struct One { Two<int, int>::value_type *t; };");

   auto c = TClass::GetClass("One<std::string>");
   c->BuildRealData();
}

// This is a test case for an issue that arises when template names are not desugared
// (specifically when default template arguments are involved).
// In this case, __pool will not be fully qualified (it will be missing the
// `test__gnu_cxx` prefix). We need to desugar it before fully qualifying the template names.
TEST(TClass, TemplateTemplate)
{
   gInterpreter->ProcessLine(R"(
      namespace test__gnu_cxx {
         // This can be any template type, given we change the template parameters
         // for __common_pool_policy
         template<typename T>
         class __pool {};

         template<template<typename> class _PoolTp>
         struct __common_pool_policy {};
         template<typename _Poolp = __common_pool_policy<__pool> >
         class __mt_alloc {};
      }

      namespace double32t_test__gnu_cxx {
         template<typename T>
         class __pool {};

         template<template<typename> class _PoolTp, typename T>
         struct __common_pool_policy {};
         template<typename _Poolp = __common_pool_policy<__pool, Double32_t> >
         class __mt_alloc {};
      }

      namespace test_LHCb {
         template <typename ALLOC = test__gnu_cxx::__mt_alloc<>>
         struct FastAllocVector {};

         template <typename ALLOC = double32t_test__gnu_cxx::__mt_alloc<>>
         struct FastAllocVectorDouble32 {};
      }
   )");

   TClass *fastAllocVecClass = TClass::GetClass("test_LHCb::FastAllocVector<>");
   ASSERT_NE(fastAllocVecClass, nullptr);
   EXPECT_NE(strstr(fastAllocVecClass->GetName(), "test__gnu_cxx::__pool"), nullptr);
   // EXPECT_EQ(strcmp(fastAllocVecClass->GetName(),
   //                  "test_LHCb::FastAllocVector<test__gnu_cxx::__mt_alloc<test__gnu_cxx::__common_"
   //                  "pool_policy<test__gnu_cxx::__pool> > >"),
   //           0);

   TClass *fastAllocVecD32Class = TClass::GetClass("test_LHCb::FastAllocVectorDouble32<>");
   ASSERT_NE(fastAllocVecD32Class, nullptr);
   EXPECT_NE(strstr(fastAllocVecD32Class->GetName(), "double32t_test__gnu_cxx::__pool"), nullptr);
   // EXPECT_EQ(strcmp(fastAllocVecD32Class->GetName(),
   //                  "test_LHCb::FastAllocVectorDouble32<double32t_test__gnu_cxx::__mt_alloc<double32t_test_"
   //                  "_gnu_cxx::__common_pool_policy<double32t_test__gnu_cxx::__pool,Double32_t> > >"),
   //           0);
}

// ROOT-10728
TEST(TClass, CanSplitWithBaseWithCustomStreamer)
{
   gInterpreter->Declare("class CanSplitWithBaseWithCustomStreamer : public TH1D {\n"
                         "int a = 0;\n"
                         "ClassDef(CanSplitWithBaseWithCustomStreamer, 1)};");

   auto c = TClass::GetClass("CanSplitWithBaseWithCustomStreamer");
   EXPECT_FALSE(c->CanSplit());
}
