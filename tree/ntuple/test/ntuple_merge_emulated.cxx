#include "ntuple_test.hxx"
#include "ntuple_fork.hxx"

static void DoMerge(std::initializer_list<std::reference_wrapper<const FileRaii>> inputPaths, const FileRaii &outPath,
                    ENTupleMergingMode mergeMode = {})
{
   std::vector<std::unique_ptr<RPageSource>> sources;
   for (const auto &path : inputPaths)
      sources.push_back(RPageSource::Create("ntuple", path.get().GetPath(), RNTupleReadOptions()));
   std::vector<RPageSource *> sourcePtrs;
   for (const auto &s : sources)
      sourcePtrs.push_back(s.get());

   auto destination = std::make_unique<RPageSinkFile>("ntuple", outPath.GetPath(), RNTupleWriteOptions());
   RNTupleMerger merger{std::move(destination)};
   auto mergeOpts = RNTupleMergeOptions();
   mergeOpts.fMergingMode = mergeMode;
   auto res = merger.Merge(sourcePtrs, mergeOpts);
   ASSERT_TRUE(bool(res)) << res.GetError()->GetReport();
}

template <typename T>
static std::ostream &operator<<(std::ostream &s, const std::optional<T> &o)
{
   if (o)
      s << *o;
   else
      s << "(nullopt)";
   return s;
}

static void CheckFieldDescriptorsAreEquivalent(const ROOT::RFieldDescriptor &a, const ROOT::RFieldDescriptor &b)
{
   std::stringstream errs;

#define CompareDescField(What) \
   if (a.What() != b.What())   \
   errs << "\n - " #What "() differs: " << a.What() << " vs " << b.What()

   CompareDescField(GetFieldVersion);
   CompareDescField(GetTypeVersion);
   CompareDescField(GetFieldName);
   CompareDescField(GetFieldDescription);
   CompareDescField(GetTypeName);
   CompareDescField(GetTypeAlias);
   CompareDescField(GetNRepetitions);
   CompareDescField(GetStructure);
   CompareDescField(GetLinkIds().size);
   CompareDescField(GetLogicalColumnIds().size);
   CompareDescField(GetTypeChecksum);
   CompareDescField(IsSoACollection);

#undef CompareDescField

   const auto errStr = errs.str();
   if (!errStr.empty()) {
      FAIL() << "field descriptors for '" << a.GetFieldName() << "' are different:" << errStr;
   }
}

static void CompareOnDiskInfo(const FileRaii &withDict, const FileRaii &withoutDict)
{
   auto opts = RNTupleDescriptor::RCreateModelOptions();
   opts.SetEmulateUnknownTypes(true);
   auto reader1 = RNTupleReader::Open(opts, "ntuple", withDict.GetPath());
   auto reader2 = RNTupleReader::Open(opts, "ntuple", withoutDict.GetPath());
   const auto &desc1 = reader1->GetDescriptor();
   const auto &desc2 = reader2->GetDescriptor();

   EXPECT_EQ(reader1->GetNEntries(), reader2->GetNEntries());

   // Verify that all fields in desc1 are also there in desc2 and they look the same.
   std::unordered_set<std::string> fieldsVisited;
   std::vector<ROOT::DescriptorId_t> toVisit;
   toVisit.push_back(desc1.GetFieldZeroId());
   while (!toVisit.empty()) {
      auto rootId = toVisit.back();
      toVisit.pop_back();
      for (const auto &fdesc1 : desc1.GetFieldIterable(rootId)) {
         const auto fqName = desc1.GetQualifiedFieldName(fdesc1.GetId());
         const auto &fdescId2 = desc2.FindFieldId(fqName);
         ASSERT_NE(fdescId2, ROOT::kInvalidDescriptorId);
         const auto &fdesc2 = desc2.GetFieldDescriptor(fdescId2);
         CheckFieldDescriptorsAreEquivalent(fdesc1, fdesc2);

         fieldsVisited.emplace(fqName);
         toVisit.emplace_back(fdesc1.GetId());
      }
   }

   // Make sure we desc2 doesn't contain any extra field
   toVisit.clear();
   toVisit.push_back(desc2.GetFieldZeroId());
   while (!toVisit.empty()) {
      auto rootId = toVisit.back();
      toVisit.pop_back();
      for (const auto &fdesc2 : desc2.GetFieldIterable(rootId)) {
         const auto fqName = desc2.GetQualifiedFieldName(fdesc2.GetId());
         EXPECT_EQ(fieldsVisited.count(fqName), 1);
         toVisit.emplace_back(fdesc2.GetId());
      }
   }
}

TEST(RNTupleMergeEmulated, MergeEmulatedSimple)
{
   // Merge two files containing a simple class field with no dictionary
   FileRaii fileGuard1("test_ntuple_merge_emulated1.root");
   FileRaii fileGuard2("test_ntuple_merge_emulated2.root");
   FileRaii fileGuardOut("test_ntuple_merge_emulated_out.root");

   EXPECT_NO_STREAMER_OR_DICTIONARY();

   ExecInFork([&] {
      fileGuard1.PreserveFile();
      fileGuard2.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
         struct Custom {
            int fInt1 = 1;
            int fInt2 = 2;
            ClassDefNV(Custom, 2);
         };
      )"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "Custom").Unwrap());

      auto writer = RNTupleWriter::Recreate(model->Clone(), "ntuple", fileGuard1.GetPath());
      auto writer2 = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      void *ptr2 = writer2->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("Custom", "ptr", ptr);
      DeclarePointer("Custom", "ptr2", ptr2);

      ProcessLine("ptr->fInt1 = 11;");
      ProcessLine("ptr->fInt2 = 22;");
      writer->Fill();

      ProcessLine("ptr2->fInt1 = 22;");
      ProcessLine("ptr2->fInt2 = 44;");
      writer2->Fill();
      ProcessLine("ptr2->fInt1 = 33;");
      ProcessLine("ptr2->fInt2 = 66;");
      writer2->Fill();

      writer.reset();
      writer2.reset();
   });

   DoMerge({fileGuard1, fileGuard2}, fileGuardOut);

   {
      auto reader = RNTupleReader::Open("ntuple", fileGuardOut.GetPath());
      try {
         reader->GetModel();
         FAIL() << "reading back the merged file without fEmulateUnknownTypes should fail";
      } catch (const ROOT::RException &ex) {
         ASSERT_THAT(ex.GetError().GetReport(), testing::HasSubstr("unknown type"));
      }
   }

   auto opts = RNTupleDescriptor::RCreateModelOptions();
   opts.SetEmulateUnknownTypes(true);
   auto reader = RNTupleReader::Open(opts, "ntuple", fileGuardOut.GetPath());
   EXPECT_EQ(reader->GetNEntries(), 3);

   auto vInt1 = reader->GetView<int>("f.fInt1");
   auto vInt2 = reader->GetView<int>("f.fInt2");
   for (auto idx : reader->GetEntryRange()) {
      EXPECT_EQ(vInt1(idx), 11 * (idx + 1));
      EXPECT_EQ(vInt2(idx), 22 * (idx + 1));
   }
}

TEST(RNTupleMergeEmulated, MergeLateEmulated)
{
   // Union-merge two files, the first containing an empty ntuple and the second a simple class field with no dictionary
   FileRaii fileGuard1("test_ntuple_merge_late_emulated1.root");
   FileRaii fileGuard2("test_ntuple_merge_late_emulated2.root");
   FileRaii fileGuardOut("test_ntuple_merge_late_emulated_out.root");
   FileRaii fileGuardOut2("test_ntuple_merge_late_emulated_out2.root");

   EXPECT_NO_STREAMER_OR_DICTIONARY();

   ExecInFork([&] {
      fileGuard1.PreserveFile();
      fileGuard2.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
         struct Custom {
            int fInt1 = 1;
            ClassDefNV(Custom, 2);
         };
      )"));

      {
         auto writer = RNTupleWriter::Recreate(ROOT::RNTupleModel::Create(), "ntuple", fileGuard1.GetPath());
         writer->Fill();
         writer->Fill();
         writer->Fill();
      }

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "Custom").Unwrap());
      auto writer2 = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());

      void *ptr2 = writer2->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("Custom", "ptr2", ptr2);

      ProcessLine("ptr2->fInt1 = 11;");
      writer2->Fill();

      ProcessLine("ptr2->fInt1 = 22;");
      writer2->Fill();

      writer2.reset();

      DoMerge({fileGuard1, fileGuard2}, fileGuardOut, ENTupleMergingMode::kUnion);
   });

   // Gather the input sources
   DoMerge({fileGuard1, fileGuard2}, fileGuardOut2, ENTupleMergingMode::kUnion);
   CompareOnDiskInfo(fileGuardOut, fileGuardOut2);
}

TEST(RNTupleMergeEmulated, MergeLateEmulatedVector)
{
   // Union-merge two files, the first containing an empty ntuple and the second a simple class field with no dictionary
   FileRaii fileGuard1("test_ntuple_merge_late_emulated_vec1.root");
   FileRaii fileGuard2("test_ntuple_merge_late_emulated_vec2.root");
   FileRaii fileGuardOut("test_ntuple_merge_late_emulated_out.root");
   FileRaii fileGuardOut2("test_ntuple_merge_late_emulated_out2.root");

   EXPECT_NO_STREAMER_OR_DICTIONARY();

   ExecInFork([&] {
      fileGuard1.PreserveFile();
      fileGuard2.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
         struct Custom {
            int fInt1 = 1;
            ClassDefNV(Custom, 2);
         };
      )"));

      {
         auto writer = RNTupleWriter::Recreate(ROOT::RNTupleModel::Create(), "ntuple", fileGuard1.GetPath());
         writer->Fill();
         writer->Fill();
         writer->Fill();
      }

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "std::vector<Custom>").Unwrap());
      auto writer2 = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard2.GetPath());

      void *ptr2 = writer2->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("std::vector<Custom>", "ptr2", ptr2);

      ProcessLine("ptr2->push_back(Custom{11});");
      writer2->Fill();

      ProcessLine("ptr2->push_back(Custom{22});");
      writer2->Fill();

      writer2.reset();

      DoMerge({fileGuard1, fileGuard2}, fileGuardOut, ENTupleMergingMode::kUnion);
   });

   // Gather the input sources
   DoMerge({fileGuard1, fileGuard2}, fileGuardOut2, ENTupleMergingMode::kUnion);
   CompareOnDiskInfo(fileGuardOut, fileGuardOut2);
}
