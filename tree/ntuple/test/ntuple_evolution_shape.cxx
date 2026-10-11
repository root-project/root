#include "ntuple_test.hxx"

#include "ntuple_fork.hxx"
#include "ntuple_test_evolution.hxx"

#include <TInterpreter.h>

#include <array>
#include <string>
#include <string_view>

TEST(RNTupleEvolution, UnionMergeLateClassSchema)
{
   struct SchemaCase {
      const char *fName;
      const char *fMembers;
      bool fMatches;
      bool fNested;
      bool fProjected = false;
      bool fSkip = false;
      bool fStreamer = false;
   };
   const std::array<SchemaCase, 16> cases{{
      {"matching", "int fA = 11; int fC = 33;", true, false},
      {"inserted", "int fA = 11; int fB = 22; int fC = 33;", false, false},
      {"appended", "int fA = 11; int fC = 33; int fB = 22;", false, false},
      {"removed", "int fA = 11;", false, false},
      {"reordered", "int fC = 33; int fA = 11;", false, false},
      {"type_changed", "float fA = 11; int fC = 33;", false, false},
      {"nested_matching", "int fA = 11; int fC = 33;", true, true},
      {"nested_inserted", "int fA = 11; int fB = 22; int fC = 33;", false, true},
      {"projected_matching", "int fA = 11; int fC = 33;", true, false, true},
      {"projected_inserted", "int fA = 11; int fB = 22; int fC = 33;", false, false, true},
      {"skip_inserted", "int fA = 11; int fB = 22; int fC = 33;", false, false, false, true},
      {"version_changed", "int fA = 11; int fC = 33; ClassDefNV(LateMergeClass, 2)", false, false},
      {"streamer_common_matching", "int fA = 11; int fC = 33;", true, false, false, false, true},
      {"streamer_evolved", "int fA = 11; int fB = 22; int fC = 33; ClassDefNV(LateMergeClass, 2)", false, false, false,
       false, true},
      {"missing_dictionary", nullptr, false, false},
      {"skip_missing_dictionary", nullptr, false, false, false, true},
   }};
   for (const auto &testCase : cases) {
      SCOPED_TRACE(testCase.fName);
      // Each process gets its own dictionary; only the source writer sees the original definition.
      ExecInFork(([&] {
         FileRaii first("test_ntuple_merge_class_schema_first.root");
         FileRaii second("test_ntuple_merge_class_schema_second.root");
         FileRaii third("test_ntuple_merge_class_schema_third.root");
         FileRaii output("test_ntuple_merge_class_schema_output.root");
         const std::string typeName = testCase.fNested ? "std::vector<LateMergeClass>" : "LateMergeClass";
         ExecInFork([&] {
            first.PreserveFile();
            second.PreserveFile();
            ASSERT_TRUE(gInterpreter->Declare("struct LateMergeClass { int fA = 11; int fC = 33; };"));
            {
               auto model = RNTupleModel::Create();
               model->MakeField<int>("e");
               // Adding late streamer fields is unsupported; check matching streamer fields present from the start.
               if (testCase.fStreamer && testCase.fMatches)
                  model->AddField(std::make_unique<ROOT::RStreamerField>("record", typeName));
               else if (testCase.fProjected)
                  model->AddField(RFieldBase::Create("record", typeName).Unwrap());
               auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", first.GetPath());
               writer->Fill();
            }
            auto model = RNTupleModel::Create();
            model->MakeField<int>("e");
            *model->MakeField<int>("before_record") = 77;
            if (testCase.fStreamer)
               model->AddField(std::make_unique<ROOT::RStreamerField>("record", typeName));
            else
               model->AddField(RFieldBase::Create("record", typeName).Unwrap());
            if (testCase.fProjected) {
               model->AddProjectedField(RFieldBase::Create("alias", typeName).Unwrap(), [](const std::string &name) {
                  auto mapped = name;
                  mapped.replace(0, 5, "record");
                  return mapped;
               });
            }
            auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", second.GetPath());
            if (testCase.fNested) {
               void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("record").get();
               DeclarePointer(typeName, "values", ptr);
               ProcessLine("values->emplace_back();");
            }
            writer->Fill();
         });
         if (testCase.fMembers) {
            ASSERT_TRUE(
               gInterpreter->Declare((std::string("struct LateMergeClass { ") + testCase.fMembers + " };").c_str()));
         }
         auto firstSource = RPageSource::Create("ntpl", first.GetPath());
         auto secondSource = RPageSource::Create("ntpl", second.GetPath());
         std::vector<RPageSource *> sources{firstSource.get(), secondSource.get()};
         std::unique_ptr<RPageSource> thirdSource;
         if (testCase.fSkip) {
            {
               auto model = RNTupleModel::Create();
               model->MakeField<int>("e");
               *model->MakeField<int>("before_record") = 99;
               auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", third.GetPath());
               writer->Fill();
            }
            thirdSource = RPageSource::Create("ntpl", third.GetPath());
            sources.push_back(thirdSource.get());
         }
         auto destination = std::make_unique<RPageSinkFile>("ntpl", output.GetPath(), RNTupleWriteOptions());
         auto *sink = destination.get();
         RNTupleMerger merger{std::move(destination)};
         RNTupleMergeOptions options;
         options.fMergingMode = ENTupleMergingMode::kUnion;
         if (testCase.fSkip)
            options.fErrBehavior = ROOT::Experimental::Internal::ENTupleMergeErrBehavior::kSkip;
         auto result = [&] {
            ROOT::TestSupport::CheckDiagsRAII diags;
            diags.optionalDiag(kWarning, "RIoUring", "io_uring is not available because:", false);
            diags.optionalDiag(kWarning, "RRawFileUnix",
                               "io_uring setup failed, falling back to blocking I/O in ReadV");
            if (!testCase.fMembers)
               diags.optionalDiag(kWarning, "TClass::Init", "no dictionary for class LateMergeClass is available");
            if (testCase.fSkip) {
               diags.requiredDiag(kWarning, "ROOT.NTuple.Merge",
                                  "Skipping RNTuple due to: cannot extend destination with field `record`", false);
            }
            return merger.Merge(sources, options);
         }();
         ASSERT_FALSE(::testing::Test::HasFailure());
         if (testCase.fSkip) {
            ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
            auto reader = RNTupleReader::Open("ntpl", output.GetPath());
            ASSERT_EQ(2u, reader->GetNEntries());
            ASSERT_EQ(ROOT::kInvalidDescriptorId, reader->GetDescriptor().FindFieldId("record"));
            auto before = reader->GetView<int>("before_record");
            ASSERT_EQ(0, before(0));
            ASSERT_EQ(99, before(1));
         } else if (!testCase.fMatches) {
            ASSERT_FALSE(bool(result));
            ASSERT_NE(std::string::npos,
                      result.GetError()->GetReport().find(testCase.fMembers ? "does not match the source schema"
                                                                            : "unknown type: LateMergeClass"));
            ASSERT_EQ(ROOT::kInvalidDescriptorId,
                      sink->GetDescriptor().FindFieldId(testCase.fProjected ? "alias" : "record"));
            ASSERT_EQ(ROOT::kInvalidDescriptorId, sink->GetDescriptor().FindFieldId("before_record"));
         } else {
            ASSERT_TRUE(bool(result)) << result.GetError()->GetReport();
            auto reader = RNTupleReader::Open("ntpl", output.GetPath());
            ASSERT_EQ(2u, reader->GetNEntries());
            if (testCase.fStreamer) {
               void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("record").get();
               DeclarePointer(typeName, "mergedRecord", ptr);
               reader->LoadEntry(1);
               ASSERT_EQ(11, gInterpreter->Calc("mergedRecord->fA"));
               ASSERT_EQ(33, gInterpreter->Calc("mergedRecord->fC"));
            } else if (!testCase.fNested) {
               auto a = reader->GetView<int>("record.fA");
               auto c = reader->GetView<int>("record.fC");
               ASSERT_EQ(testCase.fProjected ? 11 : 0, a(0));
               ASSERT_EQ(testCase.fProjected ? 33 : 0, c(0));
               ASSERT_EQ(11, a(1));
               ASSERT_EQ(33, c(1));
               if (testCase.fProjected) {
                  auto aliasA = reader->GetView<int>("alias.fA");
                  auto aliasC = reader->GetView<int>("alias.fC");
                  ASSERT_EQ(11, aliasA(0));
                  ASSERT_EQ(33, aliasC(0));
                  ASSERT_EQ(11, aliasA(1));
                  ASSERT_EQ(33, aliasC(1));
               }
            } else {
               auto records = reader->GetCollectionView("record");
               ASSERT_EQ(0u, records(0));
               ASSERT_EQ(1u, records(1));
               auto a = records.GetView<int>("_0.fA");
               auto c = records.GetView<int>("_0.fC");
               ASSERT_EQ(11, a(0));
               ASSERT_EQ(33, c(0));
            }
         }
      }));
   }
}

TEST(RNTupleEvolution, AddedMember)
{
   FileRaii fileGuard("test_ntuple_evolution_added_member.root");

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedMember {
   int fInt1 = 1;
   int fInt3 = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "AddedMember").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("AddedMember", "ptrAddedMember", ptr);
      ProcessLine("ptrAddedMember->fInt1 = 71;");
      ProcessLine("ptrAddedMember->fInt3 = 93;");
      writer->Fill();

      // Reset / close the writer and flush the file.
      writer.reset();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedMember {
   int fInt1 = 1;
   int fInt2 = 2;
   int fInt3 = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("AddedMember", "ptrAddedMember", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt1", 1);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt2", 2);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt3", 3);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt1", 71);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt2", 2);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt3", 93);
}

TEST(RNTupleEvolution, AddedMemberBulkRead)
{
   FileRaii fileGuard("test_ntuple_evolution_added_member_bulk_read.root");

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedMemberBulkRead {
   int fX = 137;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "AddedMemberBulkRead").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      // Reset / close the writer and flush the file.
      writer.reset();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedMemberBulkRead {
   ROOT::RVec<int> fNew;
   int fX = 137;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(1, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("AddedMemberBulkRead", "ptrAddedMemberBulkRead", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrAddedMemberBulkRead->fX", 137);
   EXPECT_EVALUATE_EQ("ptrAddedMemberBulkRead->fNew.empty()", true);

   auto bulk = reader->GetModel().CreateBulk("f.fNew");
   auto values = static_cast<ROOT::RVec<int> *>(bulk.ReadBulk(ROOT::RNTupleLocalRange(0, 0, 1)));
   EXPECT_TRUE(values[0].empty());
}

TEST(RNTupleEvolution, AddedMemberObject)
{
   FileRaii fileGuard("test_ntuple_evolution_added_member_object.root");

   EXPECT_NO_STREAMER_OR_DICTIONARY();

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedMemberObjectI {
   int fInt;
};
struct AddedMemberObject {
   AddedMemberObjectI fMember1{1};
   AddedMemberObjectI fMember3{3};
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "AddedMemberObject").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("AddedMemberObject", "ptrAddedMemberObject", ptr);
      ProcessLine("ptrAddedMemberObject->fMember1.fInt = 71;");
      ProcessLine("ptrAddedMemberObject->fMember3.fInt = 93;");
      writer->Fill();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedMemberObjectI {
   int fInt;
};
struct AddedMemberObject {
   AddedMemberObjectI fMember1{1};
   AddedMemberObjectI fMember2{2};
   AddedMemberObjectI fMember3{3};
   TObject fMemberTObj;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("AddedMemberObject", "ptrAddedMemberObject", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrAddedMemberObject->fMember1.fInt", 1);
   EXPECT_EVALUATE_EQ("ptrAddedMemberObject->fMember2.fInt", 2);
   EXPECT_EVALUATE_EQ("ptrAddedMemberObject->fMember3.fInt", 3);
   EXPECT_EVALUATE_EQ("ptrAddedMemberObject->fMemberTObj.GetUniqueID()", 0);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrAddedMemberObject->fMember1.fInt", 71);
   EXPECT_EVALUATE_EQ("ptrAddedMemberObject->fMember2.fInt", 2);
   EXPECT_EVALUATE_EQ("ptrAddedMemberObject->fMember3.fInt", 93);
   EXPECT_EVALUATE_EQ("ptrAddedMemberObject->fMemberTObj.GetUniqueID()", 0);
}

TEST(RNTupleEvolution, RemovedMember)
{
   FileRaii fileGuard("test_ntuple_evolution_removed_member.root");

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct RemovedMember {
   int fInt1 = 1;
   int fInt2 = 2;
   int fInt3 = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "RemovedMember").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("RemovedMember", "ptrRemovedMember", ptr);
      ProcessLine("ptrRemovedMember->fInt1 = 71;");
      ProcessLine("ptrRemovedMember->fInt2 = 82;");
      ProcessLine("ptrRemovedMember->fInt3 = 93;");
      writer->Fill();

      // Reset / close the writer and flush the file.
      writer.reset();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct RemovedMember {
   int fInt1 = 1;
   // fInt2 member is missing!
   int fInt3 = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("RemovedMember", "ptrRemovedMember", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrRemovedMember->fInt1", 1);
   EXPECT_EVALUATE_EQ("ptrRemovedMember->fInt3", 3);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrRemovedMember->fInt1", 71);
   EXPECT_EVALUATE_EQ("ptrRemovedMember->fInt3", 93);
}

TEST(RNTupleEvolution, ReorderedMembers)
{
   FileRaii fileGuard("test_ntuple_evolution_reordered_members.root");

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct ReorderedMembers {
   int fInt1 = 1;
   int fInt2 = 2;
   int fInt3 = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "ReorderedMembers").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("ReorderedMembers", "ptrReorderedMembers", ptr);
      ProcessLine("ptrReorderedMembers->fInt1 = 71;");
      ProcessLine("ptrReorderedMembers->fInt2 = 82;");
      ProcessLine("ptrReorderedMembers->fInt3 = 93;");
      writer->Fill();

      // Reset / close the writer and flush the file.
      writer.reset();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct ReorderedMembers {
   int fInt3 = 3;
   int fInt2 = 2;
   int fInt1 = 1;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("ReorderedMembers", "ptrReorderedMembers", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrReorderedMembers->fInt1", 1);
   EXPECT_EVALUATE_EQ("ptrReorderedMembers->fInt2", 2);
   EXPECT_EVALUATE_EQ("ptrReorderedMembers->fInt3", 3);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrReorderedMembers->fInt1", 71);
   EXPECT_EVALUATE_EQ("ptrReorderedMembers->fInt2", 82);
   EXPECT_EVALUATE_EQ("ptrReorderedMembers->fInt3", 93);
}

TEST(RNTupleEvolution, RenamedMember)
{
   // Without explicit detection support for renamings, a renamed member is handled as a removal from the original
   // version (which is simply ignored) and an added member (which is defaulted).
   FileRaii fileGuard("test_ntuple_evolution_renamed_member.root");

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct RenamedMember {
   int fInt1 = 1;
   int fInt2 = 2;
   int fInt3 = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "RenamedMember").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("RenamedMember", "ptrRenamedMember", ptr);
      ProcessLine("ptrRenamedMember->fInt1 = 71;");
      ProcessLine("ptrRenamedMember->fInt2 = 82;");
      ProcessLine("ptrRenamedMember->fInt3 = 93;");
      writer->Fill();

      // Reset / close the writer and flush the file.
      writer.reset();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct RenamedMember {
   int fInt1 = 1;
   int fInt2Renamed = 2;
   int fInt3 = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("RenamedMember", "ptrRenamedMember", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrRenamedMember->fInt1", 1);
   EXPECT_EVALUATE_EQ("ptrRenamedMember->fInt2Renamed", 2);
   EXPECT_EVALUATE_EQ("ptrRenamedMember->fInt3", 3);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrRenamedMember->fInt1", 71);
   EXPECT_EVALUATE_EQ("ptrRenamedMember->fInt2Renamed", 2);
   EXPECT_EVALUATE_EQ("ptrRenamedMember->fInt3", 93);
}

TEST(RNTupleEvolution, RenamedMemberClass)
{
   // RNTuple currently does not support automatic schema evolution when a class is renamed.
   FileRaii fileGuard("test_ntuple_evolution_renamed_member_class.root");

   EXPECT_NO_STREAMER_OR_DICTIONARY();

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct RenamedMemberClass1 {
   int fInt = 1;
};
struct RenamedMemberClass {
   RenamedMemberClass1 fMember;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "RenamedMemberClass").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct RenamedMemberClass2 {
   int fInt = 1;
};
struct RenamedMemberClass {
   RenamedMemberClass2 fMember;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   try {
      reader->GetModel();
      FAIL() << "model reconstruction should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible type name for field"));
   }
}

TEST(RNTupleEvolution, AddedDefaultTemplateArg)
{
   // RNTuple currently does not support automatic schema evolution when a default template argument is added because it
   // may have implications for the on-disk format.
   FileRaii fileGuard("test_ntuple_evolution_added_default_template_arg.root");

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
template <typename T>
struct AddedDefaultTemplateArg {
   T fMember;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "AddedDefaultTemplateArg<int>").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      // Reset / close the writer and flush the file.
      writer.reset();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
template <typename T, typename U = int>
struct AddedDefaultTemplateArg {
   T fMember;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   try {
      reader->GetModel();
      FAIL() << "model reconstruction should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible type name for field"));
   }
}

TEST(RNTupleEvolution, AddedBaseClass)
{
   FileRaii fileGuard("test_ntuple_evolution_added_base_class.root");

   EXPECT_NO_STREAMER_OR_DICTIONARY();

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedBaseIntermediate {
   int fIntermediate = 2;
};
struct AddedBaseDerived : public AddedBaseIntermediate {
   int fDerived = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "AddedBaseDerived").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("AddedBaseDerived", "ptrAddedBaseDerived", ptr);
      ProcessLine("ptrAddedBaseDerived->fIntermediate = 82;");
      ProcessLine("ptrAddedBaseDerived->fDerived = 93;");
      writer->Fill();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedBase {
   int fBase = 1;
};
struct AddedBaseIntermediate : public AddedBase {
   int fIntermediate = 2;
};
struct AddedBaseDerived : public AddedBaseIntermediate {
   int fDerived = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("AddedBaseDerived", "ptrAddedBaseDerived", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrAddedBaseDerived->fBase", 1);
   EXPECT_EVALUATE_EQ("ptrAddedBaseDerived->fIntermediate", 2);
   EXPECT_EVALUATE_EQ("ptrAddedBaseDerived->fDerived", 3);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrAddedBaseDerived->fBase", 1);
   EXPECT_EVALUATE_EQ("ptrAddedBaseDerived->fIntermediate", 82);
   EXPECT_EVALUATE_EQ("ptrAddedBaseDerived->fDerived", 93);
}

TEST(RNTupleEvolution, AddedSecondBaseClass)
{
   FileRaii fileGuard("test_ntuple_evolution_added_second_base_class.root");

   EXPECT_NO_STREAMER_OR_DICTIONARY();

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedSecondBaseFirst {
   int fFirstBase = 1;
};
struct AddedSecondBaseDerived : public AddedSecondBaseFirst {
   int fDerived = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "AddedSecondBaseDerived").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("AddedSecondBaseDerived", "ptrAddedSecondBaseDerived", ptr);
      ProcessLine("ptrAddedSecondBaseDerived->fFirstBase = 71;");
      ProcessLine("ptrAddedSecondBaseDerived->fDerived = 93;");
      writer->Fill();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedSecondBaseFirst {
   int fFirstBase = 1;
};
struct AddedSecondBaseSecond {
   int fSecondBase = 2;
};
struct AddedSecondBaseDerived : public AddedSecondBaseFirst, public AddedSecondBaseSecond {
   int fDerived = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   try {
      reader->GetModel();
      FAIL() << "model reconstruction should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible number of base classes"));
   }
}

TEST(RNTupleEvolution, PrependSecondBaseClass)
{
   // For RNTuple, this case looks like the first base class was renamed. It would be confusing for users to
   // automatically evolve this case, even if the member fields and on-disk columns are compatible.
   FileRaii fileGuard("test_ntuple_evolution_prepend_second_base_class.root");

   EXPECT_NO_STREAMER_OR_DICTIONARY();

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct PrependSecondBaseFirst {
   int fFirstBase = 1;
};
struct PrependSecondBaseDerived : public PrependSecondBaseFirst {
   int fDerived = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "PrependSecondBaseDerived").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct PrependSecondBaseFirst {
   int fFirstBase = 1;
};
struct PrependSecondBaseSecond {
   int fSecondBase = 2;
};
struct PrependSecondBaseDerived : public PrependSecondBaseSecond, public PrependSecondBaseFirst {
   int fDerived = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   try {
      reader->GetModel();
      FAIL() << "model reconstruction should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible number of base classes"));
   }
}

TEST(RNTupleEvolution, AddedIntermediateClass)
{
   FileRaii fileGuard("test_ntuple_evolution_added_intermediate_class.root");

   EXPECT_NO_STREAMER_OR_DICTIONARY();

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedIntermediateBase {
   int fBase = 1;
};
struct AddedIntermediateDerived : public AddedIntermediateBase {
   int fDerived = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "AddedIntermediateDerived").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedIntermediateBase {
   int fBase = 1;
};
struct AddedIntermediate : public AddedIntermediateBase {
   int fIntermediate = 2;
};
struct AddedIntermediateDerived : public AddedIntermediate {
   int fDerived = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   try {
      reader->GetModel();
      FAIL() << "model reconstruction should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible type name for field"));
   }
}

TEST(RNTupleEvolution, RemovedBaseClass)
{
   FileRaii fileGuard("test_ntuple_evolution_removed_base_class.root");

   EXPECT_NO_STREAMER_OR_DICTIONARY();

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct RemovedBase {
   int fBase = 1;
};
struct RemovedBaseIntermediate : public RemovedBase {
   int fIntermediate = 2;
};
struct RemovedBaseDerived : public RemovedBaseIntermediate {
   int fDerived = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "RemovedBaseDerived").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("RemovedBaseDerived", "ptrRemovedBaseDerived", ptr);
      ProcessLine("ptrRemovedBaseDerived->fBase = 71;");
      ProcessLine("ptrRemovedBaseDerived->fIntermediate = 82;");
      ProcessLine("ptrRemovedBaseDerived->fDerived = 93;");
      writer->Fill();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct RemovedBaseIntermediate {
   int fIntermediate = 2;
};
struct RemovedBaseDerived : public RemovedBaseIntermediate {
   int fDerived = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("RemovedBaseDerived", "ptrRemovedBaseDerived", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrRemovedBaseDerived->fIntermediate", 2);
   EXPECT_EVALUATE_EQ("ptrRemovedBaseDerived->fDerived", 3);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrRemovedBaseDerived->fIntermediate", 82);
   EXPECT_EVALUATE_EQ("ptrRemovedBaseDerived->fDerived", 93);
}

TEST(RNTupleEvolution, RemovedIntermediateClass)
{
   FileRaii fileGuard("test_ntuple_evolution_removed_intermediate_class.root");

   EXPECT_NO_STREAMER_OR_DICTIONARY();

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct RemovedIntermediateBase {
   int fBase = 1;
};
struct RemovedIntermediate : public RemovedIntermediateBase {
   int fIntermediate = 2;
};
struct RemovedIntermediateDerived : public RemovedIntermediate {
   int fDerived = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "RemovedIntermediateDerived").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct RemovedIntermediateBase {
   int fBase = 1;
};
struct RemovedIntermediateDerived : public RemovedIntermediateBase {
   int fDerived = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   try {
      reader->GetModel();
      FAIL() << "model reconstruction should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible type name for field"));
   }
}

TEST(RNTupleEvolution, RemovedSecondBaseClass)
{
   FileRaii fileGuard("test_ntuple_evolution_removed_second_base_class.root");

   EXPECT_NO_STREAMER_OR_DICTIONARY();

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct RemovedSecondBaseFirst {
   int fFirstBase = 1;
};
struct RemovedSecondBaseSecond {
   int fSecondBase = 2;
};
struct RemovedSecondBaseDerived : public RemovedSecondBaseFirst, public RemovedSecondBaseSecond {
   int fDerived = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "RemovedSecondBaseDerived").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct RemovedSecondBaseFirst {
   int fFirstBase = 1;
};
struct RemovedSecondBaseDerived : public RemovedSecondBaseFirst {
   int fDerived = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   try {
      reader->GetModel();
      FAIL() << "model reconstruction should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible number of base classes"));
   }
}

TEST(RNTupleEvolution, RenamedBaseClass)
{
   // RNTuple currently does not support automatic schema evolution when a class is renamed.
   FileRaii fileGuard("test_ntuple_evolution_renamed_base_class.root");

   EXPECT_NO_STREAMER_OR_DICTIONARY();

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct RenamedBase1 {
   int fBase = 1;
};
struct RenamedBaseDerived : public RenamedBase1 {
   int fDerived = 2;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "RenamedBaseDerived").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct RenamedBase2 {
   int fBase = 1;
};
struct RenamedBaseDerived : public RenamedBase2 {
   int fDerived = 2;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   try {
      reader->GetModel();
      FAIL() << "model reconstruction should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible type name for field"));
   }
}

TEST(RNTupleEvolution, RenamedIntermediateClass)
{
   // RNTuple currently does not support automatic schema evolution when a class is renamed.
   FileRaii fileGuard("test_ntuple_evolution_renamed_intermediate_class.root");

   EXPECT_NO_STREAMER_OR_DICTIONARY();

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct RenamedIntermediateBase {
   int fBase = 1;
};
struct RenamedIntermediate1 : public RenamedIntermediateBase {
   int fInteremdiate = 2;
};
struct RenamedIntermediateDerived : public RenamedIntermediate1 {
   int fDerived = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "RenamedIntermediateDerived").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct RenamedIntermediateBase {
   int fBase = 1;
};
struct RenamedIntermediate2 : public RenamedIntermediateBase {
   int fInteremdiate = 2;
};
struct RenamedIntermediateDerived : public RenamedIntermediate2 {
   int fDerived = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   try {
      reader->GetModel();
      FAIL() << "model reconstruction should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible type name for field"));
   }
}

TEST(RNTupleEvolution, StreamerField)
{
   FileRaii fileGuard("test_ntuple_evolution_streamer_field.root");

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct StreamerField {
   int fInt = 1;
   int fAnotherInt = 3;

   ClassDefNV(StreamerField, 2)
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(std::make_unique<ROOT::RStreamerField>("f", "StreamerField"));

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("StreamerField", "ptrStreamerField", ptr);
      ProcessLine("ptrStreamerField->fInt = 2;");
      writer->Fill();

      // Reset / close the writer and flush the file.
      writer.reset();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct StreamerField {
   int fInt = 0;
   int fAdded = 137;
   // removed fAnotherInt

   ClassDefNV(StreamerField, 3)
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("StreamerField", "ptrStreamerField", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrStreamerField->fInt", 1);
   EXPECT_EVALUATE_EQ("ptrStreamerField->fAdded", 137);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrStreamerField->fInt", 2);
   EXPECT_EVALUATE_EQ("ptrStreamerField->fAdded", 137);
}

TEST(RNTupleEvolution, RecordField)
{
   FileRaii fileGuard("test_ntuple_evolution_record_field.root");
   {
      std::vector<std::unique_ptr<RFieldBase>> items;
      items.emplace_back(std::make_unique<RField<char>>("f1"));
      items.emplace_back(std::make_unique<RField<float>>("f2"));
      items.emplace_back(std::make_unique<RField<std::string>>("f3"));

      auto model = ROOT::RNTupleModel::Create();
      model->AddField(std::make_unique<ROOT::RRecordField>("r", std::move(items)));

      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      auto r = writer->GetModel().GetDefaultEntry().GetPtr<void>("r");
      const auto &rf = static_cast<const ROOT::RRecordField &>(writer->GetModel().GetConstField("r"));

      auto f1 = reinterpret_cast<char *>(reinterpret_cast<unsigned char *>(r.get()) + rf.GetOffsets()[0]);
      auto f3 = reinterpret_cast<std::string *>(reinterpret_cast<unsigned char *>(r.get()) + rf.GetOffsets()[2]);
      *f1 = 'x';
      *f3 = "ROOT";

      writer->Fill();
   }

   std::vector<std::unique_ptr<RFieldBase>> items;
   items.emplace_back(std::make_unique<RField<std::string>>("f3"));
   items.emplace_back(std::make_unique<RField<std::vector<float>>>("f4"));
   items.emplace_back(std::make_unique<RField<int>>("f1"));

   auto model = ROOT::RNTupleModel::Create();
   model->AddField(std::make_unique<ROOT::RRecordField>("r", std::move(items)));

   auto reader = RNTupleReader::Open(std::move(model), "ntpl", fileGuard.GetPath());
   auto r = reader->GetModel().GetDefaultEntry().GetPtr<void>("r");
   const auto &rf = static_cast<const ROOT::RRecordField &>(reader->GetModel().GetConstField("r"));
   auto f3 = reinterpret_cast<std::string *>(reinterpret_cast<unsigned char *>(r.get()) + rf.GetOffsets()[0]);
   auto f4 = reinterpret_cast<std::vector<float> *>(reinterpret_cast<unsigned char *>(r.get()) + rf.GetOffsets()[1]);
   auto f1 = reinterpret_cast<int *>(reinterpret_cast<unsigned char *>(r.get()) + rf.GetOffsets()[2]);
   reader->LoadEntry(0);
   EXPECT_EQ("ROOT", *f3);
   EXPECT_TRUE(f4->empty());
   EXPECT_EQ('x', *f1);
}
