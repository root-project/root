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

TEST(RNTupleMerger, UnionMergeLateClassLayout)
{
   struct LayoutCase {
      const char *fName;
      const char *fMembers; // the class the merging process loads, nullptr for none
      const char *fType;
      bool fMatches;
      bool fSkip = false;
   };
   const LayoutCase cases[] = {
      {"matching", "int fA = 1; int fC = 3;", "LateClass", true},
      {"version_changed", "int fA = 1; int fC = 3; ClassDefNV(LateClass, 2)", "LateClass", true},
      {"no_dictionary", nullptr, "LateClass", true},
      {"inserted", "int fA = 1; int fB = 2; int fC = 3;", "LateClass", false},
      {"reordered", "int fC = 3; int fA = 1;", "LateClass", false},
      {"type_changed", "float fA = 1; int fC = 3;", "LateClass", false},
      {"nested_inserted", "int fA = 1; int fB = 2; int fC = 3;", "std::vector<LateClass>", false},
      {"skip_inserted", "int fA = 1; int fB = 2; int fC = 3;", "LateClass", false, true},
   };
   for (const auto &testCase : cases) {
      SCOPED_TRACE(testCase.fName);
      const std::string type = testCase.fType;
      ExecInFork(([&] {
         FileRaii fileGuard1("test_ntuple_merge_late_layout_1.root");
         FileRaii fileGuard2("test_ntuple_merge_late_layout_2.root");
         ExecInFork([&] {
            fileGuard1.PreserveFile();
            fileGuard2.PreserveFile();
            ASSERT_TRUE(gInterpreter->Declare("struct LateClass { int fA = 1; int fC = 3; };"));
            {
               auto model = RNTupleModel::Create();
               *model->MakeField<int>("e") = 1;
               auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard1.GetPath());
               writer->Fill();
            }
            auto model = RNTupleModel::Create();
            *model->MakeField<int>("e") = 2;
            model->AddField(RFieldBase::Create("late", type).Unwrap());
            auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());
            DeclarePointer(type, "pLate", writer->GetModel().GetDefaultEntry().GetPtr<void>("late").get());
            if (type == "LateClass")
               ProcessLine("pLate->fA = 11; pLate->fC = 33;");
            else
               ProcessLine("pLate->emplace_back(); pLate->back().fA = 11; pLate->back().fC = 33;");
            writer->Fill();
         });

         if (testCase.fMembers) {
            ASSERT_TRUE(
               gInterpreter->Declare((std::string("struct LateClass { ") + testCase.fMembers + " };").c_str()));
         }
         std::vector<std::unique_ptr<RPageSource>> sources;
         sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
         sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
         std::vector<RPageSource *> sourcePtrs{sources[0].get(), sources[1].get()};

         FileRaii fileGuardOut("test_ntuple_merge_late_layout_out.root");
         auto res = [&] {
            ROOT::TestSupport::CheckDiagsRAII diagsRAII;
            if (!testCase.fMembers)
               diagsRAII.optionalDiag(kWarning, "TClass::Init", "no dictionary for class LateClass is available");
            if (testCase.fSkip) {
               diagsRAII.requiredDiag(kWarning, "ROOT.NTuple.Merge",
                                      "Skipping RNTuple due to: Source RNTuple has an incompatible schema", false);
            }
            auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
            RNTupleMerger merger{std::move(destination)};
            RNTupleMergeOptions opts;
            opts.fMergingMode = ENTupleMergingMode::kUnion;
            if (testCase.fSkip)
               opts.fErrBehavior = ROOT::Internal::ENTupleMergeErrBehavior::kSkip;
            return merger.Merge(sourcePtrs, opts);
         }();

         if (!testCase.fMatches && !testCase.fSkip) {
            ASSERT_FALSE(bool(res));
            ASSERT_THAT(res.GetError()->GetReport(), testing::HasSubstr("has a different layout in memory"));
            return;
         }
         ASSERT_TRUE(bool(res)) << res.GetError()->GetReport();
         auto reader = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
         if (testCase.fSkip) {
            ASSERT_EQ(1u, reader->GetNEntries());
            ASSERT_EQ(ROOT::kInvalidDescriptorId, reader->GetDescriptor().FindFieldId("late"));
            return;
         }
         ASSERT_EQ(2u, reader->GetNEntries());
         auto e = reader->GetView<int>("e");
         auto a = reader->GetView<int>("late.fA");
         auto c = reader->GetView<int>("late.fC");
         ASSERT_EQ(1, e(0));
         ASSERT_EQ(2, e(1));
         ASSERT_EQ(0, a(0));
         ASSERT_EQ(0, c(0));
         ASSERT_EQ(11, a(1));
         ASSERT_EQ(33, c(1));
      }));
   }
}
