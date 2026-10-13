#include "ntuple_test.hxx"
#include "ntuple_fork.hxx"

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

   // Gather the input sources
   std::vector<std::unique_ptr<RPageSource>> sources;
   sources.push_back(RPageSource::Create("ntuple", fileGuard1.GetPath(), RNTupleReadOptions()));
   sources.push_back(RPageSource::Create("ntuple", fileGuard2.GetPath(), RNTupleReadOptions()));
   std::vector<RPageSource *> sourcePtrs;
   for (const auto &s : sources) {
      sourcePtrs.push_back(s.get());
   }

   {
      auto destination = std::make_unique<RPageSinkFile>("ntuple", fileGuardOut.GetPath(), RNTupleWriteOptions());
      RNTupleMerger merger{std::move(destination)};
      auto res = merger.Merge(sourcePtrs);
      ASSERT_TRUE(bool(res));
   }

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
