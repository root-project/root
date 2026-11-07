#include "ntuple_test.hxx"

#include "ntuple_fork.hxx"
#include "ntuple_test_evolution.hxx"

#include <TInterpreter.h>

#include <string>
#include <string_view>

TEST(RNTupleMerger, MergeWithAttributesSchemaEvolve)
{
   ROOT::TestSupport::CheckDiagsRAII diagsRAII;
   diagsRAII.requiredDiag(kWarning, "ROOT.NTuple", "RNTuple Attributes are experimental", false);

   // Write two test ntuples to be merged.
   // These files have 1 attribute set with the same name and they contain a custom class field.
   // The second class field has a newer version that has been schema evolved from the first.
   FileRaii fileGuard1("test_ntuple_merge_attr_schemaevolve_1.root");
   ExecInFork([&] {
      fileGuard1.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct MyClass {
   int fInt1 = 1;
   int fInt3 = 3;
};
)"));

      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("mainInt");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
      auto attrModel = RNTupleModel::Create();
      attrModel->AddField(RFieldBase::Create("attr", "MyClass").Unwrap());
      auto attrSet = writer->CreateAttributeSet(std::move(attrModel), "AttrSet1");
      void *pAttr = attrSet->GetModel().GetDefaultEntry().GetPtr<void>("attr").get();
      auto attrRange = attrSet->BeginRange();
      DeclarePointer("MyClass", "pMyClass", pAttr);
      ProcessLine("pMyClass->fInt1 = 11;");
      ProcessLine("pMyClass->fInt3 = 33;");
      *pInt = 100;
      writer->Fill();
      attrSet->CommitRange(std::move(attrRange));

      // Reset / close the writer and flush the file.
      writer.reset();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct MyClass {
   int fInt1 = 1;
   int fInt2 = 2;
   int fInt3 = 3;
};
)"));

   FileRaii fileGuard2("test_ntuple_merge_attr_schemaevolve_2.root");
   {
      auto model = RNTupleModel::Create();
      auto pInt = model->MakeField<int>("mainInt");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
      auto attrModel = RNTupleModel::Create();
      attrModel->AddField(RFieldBase::Create("attr", "MyClass").Unwrap());
      auto attrSet = writer->CreateAttributeSet(std::move(attrModel), "AttrSet1");
      void *pAttr = attrSet->GetModel().GetDefaultEntry().GetPtr<void>("attr").get();
      auto attrRange = attrSet->BeginRange();
      DeclarePointer("MyClass", "pMyClass", pAttr);
      ProcessLine("pMyClass->fInt1 = 22;");
      ProcessLine("pMyClass->fInt2 = 44;");
      ProcessLine("pMyClass->fInt3 = 66;");
      *pInt = 200;
      writer->Fill();
      attrSet->CommitRange(std::move(attrRange));
   }

   std::vector<std::unique_ptr<RPageSource>> sources;
   sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
   sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
   std::vector<RPageSource *> sourcePtrs;
   for (const auto &s : sources) {
      sourcePtrs.push_back(s.get());
   }

   // Now merge the inputs
   for (const auto mmode : {ENTupleMergingMode::kFilter, ENTupleMergingMode::kStrict, ENTupleMergingMode::kUnion}) {
      const auto modeString = ToString(mmode);
      SCOPED_TRACE(std::string("with merging mode = ") + modeString);
      FileRaii fileGuardOut("test_ntuple_merge_attr_schemaevolve_out.root");
      {
         auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
         RNTupleMerger merger{std::move(destination)};
         RNTupleMergeOptions opts;
         opts.fMergingMode = mmode;
         auto res = merger.Merge(sourcePtrs, opts);
         ASSERT_TRUE(bool(res)) << res.GetError()->GetReport();
      }
      {
         auto reader = ROOT::RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
         EXPECT_EQ(reader->GetNEntries(), 2);
         EXPECT_EQ(reader->GetDescriptor().GetNAttributeSets(), 1);
         {
            auto attrSet1 = reader->OpenAttributeSet("AttrSet1");
            EXPECT_EQ(attrSet1->GetNEntries(), 2);
            const auto &attrEntry1 = attrSet1->GetModel().GetDefaultEntry();
            auto pAttr = attrEntry1.GetPtr<void>("attr").get();
            const auto ptrName = std::string("pMyClass_") + modeString;
            DeclarePointer("MyClass", ptrName, pAttr);
            for (auto idx : attrSet1->GetAttributes()) {
               auto range1 = attrSet1->LoadEntry(idx);
               EXPECT_EQ(range1.GetFirst(), idx);
               EXPECT_EQ(range1.GetLast(), idx);
               EXPECT_EVALUATE_EQ((ptrName + "->fInt1").c_str(), 11 * (idx + 1));
               EXPECT_EVALUATE_EQ((ptrName + "->fInt2").c_str(), idx == 1 ? 44 : 2);
               EXPECT_EVALUATE_EQ((ptrName + "->fInt3").c_str(), 33 * (idx + 1));
            }
         }
      }
   }
}
