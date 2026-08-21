#include <ROOT/RField.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/TestSupport.hxx>

#include <TDictAttributeMap.h>
#include <TInterpreter.h>

#include <string>
#include <string_view>

#include "gtest/gtest.h"
#include "ntuple_fork.hxx"

namespace {

void EvaluateIntImpl(const char *expression, int *value)
{
   auto interpreterValue = gInterpreter->MakeInterpreterValue();
   ASSERT_TRUE(gInterpreter->Evaluate(expression, *interpreterValue));
   *value = interpreterValue->GetAsLong();
}

#define EXPECT_EVALUATE_EQ(expression, expected) \
   do {                                          \
      int _value;                                \
      EvaluateIntImpl(expression, &_value);      \
      if (::testing::Test::HasFatalFailure())    \
         return;                                 \
      EXPECT_EQ(expected, _value);               \
   } while (0)

void MakeSoALink(const std::string &recordName, const std::string &soaName)
{
   auto cl = TClass::GetClass(soaName.c_str());
   cl->CreateAttributeMap();
   cl->GetAttributeMap()->AddProperty("rntuple.SoARecord", recordName.c_str());
}

} // namespace

TEST(RNTupleEvolutionSoA, RemovedMember)
{
   ROOT::TestSupport::FileRaii fileGuard("test_ntuple_evolution_soa_removed_member.root");

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ROOT::TestSupport::CheckDiagsRAII diagRAII;
      diagRAII.requiredDiag(kWarning, "[ROOT.NTuple]", "The SoA field is experimental and still under development.",
                            true /* matchFullMessage */);

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct RemovedMemberRecord {
   int fInt1;
   int fInt2;
   int fInt3;
   ClassDefNV(RemovedMemberRecord, 2)
};
struct RemovedMemberSoA {
   ROOT::RVec<int> fInt1;
   ROOT::RVec<int> fInt2;
   ROOT::RVec<int> fInt3;
   ClassDefNV(RemovedMemberSoA, 2)
};
)"));
      MakeSoALink("RemovedMemberRecord", "RemovedMemberSoA");

      auto model = ROOT::RNTupleModel::Create();
      model->AddField(ROOT::RFieldBase::Create("f", "RemovedMemberSoA").Unwrap());

      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("RemovedMemberSoA", "ptrRemovedMember", ptr);
      ProcessLine("ptrRemovedMember->fInt1 = {11, 12};");
      ProcessLine("ptrRemovedMember->fInt2 = {13, 14};");
      ProcessLine("ptrRemovedMember->fInt3 = {15, 16};");
      writer->Fill();

      // Reset / close the writer and flush the file.
      writer.reset();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct RemovedMemberRecord {
   int fInt1;
   int fInt3;
   ClassDefNV(RemovedMemberRecord, 3)
};
struct RemovedMemberSoA {
   ROOT::RVec<int> fInt1;
   ROOT::RVec<int> fInt3;
   ClassDefNV(RemovedMemberSoA, 3)
};
)"));
   MakeSoALink("RemovedMemberRecord", "RemovedMemberSoA");

   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.requiredDiag(kWarning, "[ROOT.NTuple]", "The SoA field is experimental and still under development.",
                         true /* matchFullMessage */);

   auto reader = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("RemovedMemberSoA", "ptrRemovedMember", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrRemovedMember->fInt1.size()", 0);
   EXPECT_EVALUATE_EQ("ptrRemovedMember->fInt3.size()", 0);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrRemovedMember->fInt1[0]", 11);
   EXPECT_EVALUATE_EQ("ptrRemovedMember->fInt1[1]", 12);
   EXPECT_EVALUATE_EQ("ptrRemovedMember->fInt3[0]", 15);
   EXPECT_EVALUATE_EQ("ptrRemovedMember->fInt3[1]", 16);
}

TEST(RNTupleEvolutionSoA, TypeChange)
{
   ROOT::TestSupport::FileRaii fileGuard("test_ntuple_evolution_soa_type_change.root");

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct TypeChangeRecord {
   bool fInt1;
   long long int fInt2;
   ClassDefNV(TypeChangeRecord, 2)
};
struct TypeChangeSoA {
   ROOT::RVec<bool> fInt1;
   ROOT::RVec<long long int> fInt2;
   ClassDefNV(TypeChangeSoA, 2)
};
)"));
      MakeSoALink("TypeChangeRecord", "TypeChangeSoA");

      auto model = ROOT::RNTupleModel::Create();
      model->AddField(ROOT::RFieldBase::Create("f", "TypeChangeSoA").Unwrap());

      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("TypeChangeSoA", "ptrTypeChange", ptr);
      ProcessLine("ptrTypeChange->fInt1 = {true, false};");
      ProcessLine("ptrTypeChange->fInt2 = {137, 138};");
      writer->Fill();

      // Reset / close the writer and flush the file.
      writer.reset();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct TypeChangeRecord {
   int fInt1;
   int fInt2;
   ClassDefNV(TypeChangeRecord, 3)
};
struct TypeChangeSoA {
   ROOT::RVec<int> fInt1;
   ROOT::RVec<int> fInt2;
   ClassDefNV(TypeChangeSoA, 3)
};
)"));
   MakeSoALink("TypeChangeRecord", "TypeChangeSoA");

   auto reader = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(1, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("TypeChangeSoA", "ptrTypeChange", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrTypeChange->fInt1[0]", 1);
   EXPECT_EVALUATE_EQ("ptrTypeChange->fInt1[1]", 0);
   EXPECT_EVALUATE_EQ("ptrTypeChange->fInt2[0]", 137);
   EXPECT_EVALUATE_EQ("ptrTypeChange->fInt2[1]", 138);
}

TEST(RNTupleEvolutionSoA, AddedMember)
{
   ROOT::TestSupport::FileRaii fileGuard("test_ntuple_evolution_soa_added_member.root");

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedMemberRecord {
   int fInt1;
   ClassDefNV(AddedMemberRecord, 2)
};
struct AddedMemberSoA {
   ROOT::RVec<int> fInt1;
   ClassDefNV(AddedMemberSoA, 2)
};
)"));
      MakeSoALink("AddedMemberRecord", "AddedMemberSoA");

      auto model = ROOT::RNTupleModel::Create();
      model->AddField(ROOT::RFieldBase::Create("f", "AddedMemberSoA").Unwrap());

      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("AddedMemberSoA", "ptrAddedMember", ptr);
      ProcessLine("for (int i = 0; i < 1000; ++i) ptrAddedMember->fInt1.push_back(137);");
      writer->Fill();

      // Reset / close the writer and flush the file.
      writer.reset();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedMemberRecord {
   int fInt1;
   int fInt2;
   std::string fStr;
   ClassDefNV(AddedMemberRecord, 3)
};
struct AddedMemberSoA {
   ROOT::RVec<int> fInt1;
   ROOT::RVec<int> fInt2;
   ROOT::RVec<std::string> fStr;
   ClassDefNV(AddedMemberSoA, 3)
};
)"));
   MakeSoALink("AddedMemberRecord", "AddedMemberSoA");

   auto reader = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(1, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("AddedMemberSoA", "ptrAddedMember", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt1.size()", 1000);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt1[0]", 137);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt1[999]", 137);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt2.size()", 1000);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt2[0]", 0);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt2[999]", 0);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fStr.size()", 1000);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fStr[0].size()", 0);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fStr[999].size()", 0);
}
