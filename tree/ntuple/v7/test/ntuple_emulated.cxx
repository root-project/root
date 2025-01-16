#include "ntuple_test.hxx"

#include "ntuple_fork.hxx"

TEST(RNTupleEmulated, EmulatedFields_Simple)
{
   // Write a user-defined class to RNTuple, then "forget" about it and try to load it as an emulated record field.
   // To achieve this "forgetting" we use fork() so that the parent process doesn't know about the class.

   FileRaii fileGuard("test_ntuple_emulated_fields.root");

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct Inner_Simple {
   int fInt1 = 1;
   int fInt2 = 2;

   ClassDefNV(Inner_Simple, 2);
};

struct Outer_Simple {
   int fInt1 = 1;
   Inner_Simple fInner;

   ClassDefNV(Outer_Simple, 2);
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "Outer_Simple").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("Outer_Simple", "ptrOuter", ptr);

      ProcessLine("ptrOuter->fInner.fInt1 = 71;");
      ProcessLine("ptrOuter->fInner.fInt2 = 82;");
      ProcessLine("ptrOuter->fInt1 = 93;");
      writer->Fill();

      // TStreamerInfo::Build will report a warning for interpreted classes (but only for members).
      // See also https://github.com/root-project/root/issues/9371
      ROOT::TestSupport::CheckDiagsRAII diagRAII;
      diagRAII.optionalDiag(kWarning, "TStreamerInfo::Build", "has no streamer or dictionary",
                            /*matchFullMessage=*/false);
      writer.reset();
   });

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();

   {
      RNTupleDescriptor::RCreateModelOptions opts;
      opts.fEmulateUnknownTypes = false;
      try {
         auto model = desc.CreateModel(opts);
         FAIL() << "Creating a model without fEmulateUnknownTypes should fail";
      } catch (const ROOT::RException &ex) {
         ASSERT_THAT(ex.GetError().GetReport(), testing::HasSubstr("unknown type"));
      }
   }

   {
      RNTupleDescriptor::RCreateModelOptions opts;
      opts.fEmulateUnknownTypes = true;
      auto model = desc.CreateModel(opts);
      ASSERT_NE(model, nullptr);

      const auto &outer = model->GetConstField("f");
      ASSERT_EQ(outer.GetTypeName(), "Outer_Simple");
      ASSERT_EQ(outer.GetStructure(), ENTupleStructure::kRecord);
      ASSERT_EQ(outer.GetSubFields().size(), 2);

      const auto subfields = outer.GetSubFields();
      ASSERT_EQ(subfields[0]->GetFieldName(), "fInt1");

      const auto *inner = subfields[1];
      ASSERT_EQ(inner->GetTypeName(), "Inner_Simple");
      ASSERT_EQ(inner->GetFieldName(), "fInner");
      ASSERT_EQ(inner->GetStructure(), ENTupleStructure::kRecord);
      ASSERT_NE(inner->GetTraits() & RFieldBase::kTraitEmulatedField, 0);
      ASSERT_EQ(inner->GetSubFields().size(), 2);
      ASSERT_EQ(inner->GetSubFields()[0]->GetFieldName(), "fInt1");
   }

   // Now test loading entries with a reader
   RNTupleDescriptor::RCreateModelOptions cmOpts;
   cmOpts.fEmulateUnknownTypes = true;
   reader = RNTupleReader::Open(cmOpts, "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
}

TEST(RNTupleEmulated, EmulatedFields_Vecs)
{
   // Write a user-defined class to RNTuple, then "forget" about it and try to load it as an emulated record field.
   // To achieve this "forgetting" we use fork() so that the parent process doesn't know about the class.

   FileRaii fileGuard("test_ntuple_emulated_fields_vecs.root");

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct Inner_Vecs {
   float fFlt;

   ClassDefNV(Inner_Vecs, 2);
};

struct Outer_Vecs {
   std::vector<Inner_Vecs> fInners;
   Inner_Vecs fInner;
   int f;

   ClassDefNV(Outer_Vecs, 2);
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("outers", "std::vector<Outer_Vecs>").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("outers").get();
      DeclarePointer("std::vector<Outer_Vecs>", "ptrOuters", ptr);

      ProcessLine("ptrOuters->push_back(Outer_Vecs{});");
      ProcessLine("(*ptrOuters)[0].fInners.push_back(Inner_Vecs{42.f});");
      ProcessLine("(*ptrOuters)[0].fInner.fFlt = 84.f;");
      writer->Fill();

      // TStreamerInfo::Build will report a warning for interpreted classes (but only for members).
      // See also https://github.com/root-project/root/issues/9371
      ROOT::TestSupport::CheckDiagsRAII diagRAII;
      diagRAII.optionalDiag(kWarning, "TStreamerInfo::Build", "has no streamer or dictionary",
                            /*matchFullMessage=*/false);
      writer.reset();
   });

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();

   {
      RNTupleDescriptor::RCreateModelOptions opts;
      opts.fEmulateUnknownTypes = false;
      try {
         auto model = desc.CreateModel(opts);
         FAIL() << "Creating a model without fEmulateUnknownTypes should fail";
      } catch (const ROOT::RException &ex) {
         ASSERT_THAT(ex.GetError().GetReport(), testing::HasSubstr("unknown type"));
      }
   }

   {
      RNTupleDescriptor::RCreateModelOptions opts;
      opts.fEmulateUnknownTypes = true;
      auto model = desc.CreateModel(opts);
      ASSERT_NE(model, nullptr);

      const auto &outers = model->GetConstField("outers");
      ASSERT_EQ(outers.GetTypeName(), "std::vector<Outer_Vecs>");
      ASSERT_EQ(outers.GetStructure(), ENTupleStructure::kCollection);
      ASSERT_EQ(outers.GetSubFields().size(), 1);

      const auto subfields = outers.GetSubFields();
      const auto *outer = subfields[0];
      ASSERT_EQ(outer->GetTypeName(), "Outer_Vecs");
      ASSERT_EQ(outer->GetStructure(), ENTupleStructure::kRecord);
      ASSERT_NE(outer->GetTraits() & RFieldBase::kTraitEmulatedField, 0);

      const auto outersubfields = outer->GetSubFields();
      ASSERT_EQ(outersubfields.size(), 3);

      const auto *inners = outersubfields[0];
      ASSERT_EQ(inners->GetTypeName(), "std::vector<Inner_Vecs>");
      ASSERT_EQ(inners->GetFieldName(), "fInners");
      ASSERT_EQ(inners->GetStructure(), ENTupleStructure::kCollection);
      ASSERT_EQ(inners->GetSubFields().size(), 1);
      ASSERT_EQ(inners->GetSubFields()[0]->GetFieldName(), "_0");

      const auto innersubfields = inners->GetSubFields();
      const auto *inner = innersubfields[0];
      ASSERT_EQ(inner->GetTypeName(), "Inner_Vecs");
      ASSERT_EQ(inner->GetStructure(), ENTupleStructure::kRecord);
      ASSERT_EQ(inner->GetSubFields().size(), 1);
      ASSERT_EQ(inner->GetSubFields()[0]->GetFieldName(), "fFlt");
   }

   // Now test loading entries with a reader
   RNTupleDescriptor::RCreateModelOptions cmOpts;
   cmOpts.fEmulateUnknownTypes = true;
   reader = RNTupleReader::Open(cmOpts, "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
}

TEST(RNTupleEmulated, EmulatedFields_EmptyStruct)
{
   // Test struct containing an empty struct

   FileRaii fileGuard("test_ntuple_emulated_emptystruct.root");

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct Inner_EmptyStruct {
   ClassDefNV(Inner_EmptyStruct, 2);
};

struct Outer_EmptyStruct {
   Inner_EmptyStruct fInner;
   ClassDefNV(Outer_EmptyStruct, 2);
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "Outer_EmptyStruct").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      // TStreamerInfo::Build will report a warning for interpreted classes (but only for members).
      // See also https://github.com/root-project/root/issues/9371
      ROOT::TestSupport::CheckDiagsRAII diagRAII;
      diagRAII.optionalDiag(kWarning, "TStreamerInfo::Build", "has no streamer or dictionary",
                            /*matchFullMessage=*/false);
      writer.reset();
   });

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();

   {
      RNTupleDescriptor::RCreateModelOptions opts;
      opts.fEmulateUnknownTypes = false;
      try {
         auto model = desc.CreateModel(opts);
         FAIL() << "Creating a model without fEmulateUnknownTypes should fail";
      } catch (const ROOT::RException &ex) {
         ASSERT_THAT(ex.GetError().GetReport(), testing::HasSubstr("unknown type"));
      }
   }

   {
      RNTupleDescriptor::RCreateModelOptions opts;
      opts.fEmulateUnknownTypes = true;
      auto model = desc.CreateModel(opts);
      ASSERT_NE(model, nullptr);

      const auto &outer = model->GetConstField("f");
      ASSERT_EQ(outer.GetTypeName(), "Outer_EmptyStruct");
      ASSERT_EQ(outer.GetStructure(), ENTupleStructure::kRecord);
      ASSERT_EQ(outer.GetSubFields().size(), 1);

      const auto subfields = outer.GetSubFields();
      const auto *inner = subfields[0];
      ASSERT_EQ(inner->GetTypeName(), "Inner_EmptyStruct");
      ASSERT_EQ(inner->GetFieldName(), "fInner");
      ASSERT_EQ(inner->GetStructure(), ENTupleStructure::kRecord);
      ASSERT_NE(inner->GetTraits() & RFieldBase::kTraitEmulatedField, 0);
      ASSERT_EQ(inner->GetSubFields().size(), 0);
   }

   // Now test loading entries with a reader
   RNTupleDescriptor::RCreateModelOptions cmOpts;
   cmOpts.fEmulateUnknownTypes = true;
   reader = RNTupleReader::Open(cmOpts, "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
}

TEST(RNTupleEmulated, EmulatedFields_EmptyVec)
{
   // Test vector of empty structs

   FileRaii fileGuard("test_ntuple_emulated_emptyvec.root");

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct Inner_EmptyVec {
   ClassDefNV(Inner_EmptyVec, 2);
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "std::vector<Inner_EmptyVec>").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("std::vector<Inner_EmptyVec>", "ptrInners", ptr);

      ProcessLine("ptrInners->push_back(Inner_EmptyVec{});");
      ProcessLine("ptrInners->push_back(Inner_EmptyVec{});");
      ProcessLine("ptrInners->push_back(Inner_EmptyVec{});");

      // TStreamerInfo::Build will report a warning for interpreted classes (but only for members).
      // See also https://github.com/root-project/root/issues/9371
      ROOT::TestSupport::CheckDiagsRAII diagRAII;
      diagRAII.optionalDiag(kWarning, "TStreamerInfo::Build", "has no streamer or dictionary",
                            /*matchFullMessage=*/false);
      writer.reset();
   });

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();

   {
      RNTupleDescriptor::RCreateModelOptions opts;
      opts.fEmulateUnknownTypes = false;
      try {
         auto model = desc.CreateModel(opts);
         FAIL() << "Creating a model without fEmulateUnknownTypes should fail";
      } catch (const ROOT::RException &ex) {
         ASSERT_THAT(ex.GetError().GetReport(), testing::HasSubstr("unknown type"));
      }
   }

   {
      RNTupleDescriptor::RCreateModelOptions opts;
      opts.fEmulateUnknownTypes = true;
      auto model = desc.CreateModel(opts);
      ASSERT_NE(model, nullptr);

      const auto &outer = model->GetConstField("f");
      ASSERT_EQ(outer.GetTypeName(), "std::vector<Inner_EmptyVec>");
      ASSERT_EQ(outer.GetStructure(), ENTupleStructure::kCollection);
      ASSERT_EQ(outer.GetSubFields().size(), 1);
      ASSERT_EQ(outer.GetTraits() & RFieldBase::kTraitEmulatedField, 0);

      const auto subfields = outer.GetSubFields();
      const auto *inner = subfields[0];
      ASSERT_EQ(inner->GetTypeName(), "Inner_EmptyVec");
      ASSERT_EQ(inner->GetFieldName(), "_0");
      ASSERT_EQ(inner->GetStructure(), ENTupleStructure::kRecord);
      ASSERT_NE(inner->GetTraits() & RFieldBase::kTraitEmulatedField, 0);
      ASSERT_EQ(inner->GetSubFields().size(), 0);
   }

   // Now test loading entries with a reader
   RNTupleDescriptor::RCreateModelOptions cmOpts;
   cmOpts.fEmulateUnknownTypes = true;
   reader = RNTupleReader::Open(cmOpts, "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
}

TEST(RNTupleEmulated, EmulatedFields_Write)
{
   // Write a RNTuple with user-defined types, read it as emulated fields and try to write it back.
   // It should fail.

   FileRaii fileGuard("test_ntuple_emulated_write.root");

   ExecInFork([&] {
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct Inner_Write {
   int fFlt = 42;
   
   ClassDefNV(Inner_Write, 2);
};

struct Outer_Write {
   std::vector<Inner_Write> fInners;
   
   ClassDefNV(Outer_Write, 2);
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "Outer_Write").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      // TStreamerInfo::Build will report a warning for interpreted classes (but only for members).
      // See also https://github.com/root-project/root/issues/9371
      ROOT::TestSupport::CheckDiagsRAII diagRAII;
      diagRAII.optionalDiag(kWarning, "TStreamerInfo::Build", "has no streamer or dictionary",
                            /*matchFullMessage=*/false);
      writer.reset();
   });

   RNTupleDescriptor::RCreateModelOptions cmOpts;
   cmOpts.fEmulateUnknownTypes = true;
   auto reader = RNTupleReader::Open(cmOpts, "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);

   // Try to write it back
   FileRaii fileGuard2("test_ntuple_emulated_write2.root");
   auto model = reader->GetModel().Clone();
   try {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard2.GetPath());
      FAIL() << "Creating a RNTupleWriter with emulated fields should fail.";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.GetError().GetReport(), testing::HasSubstr("unsupported"));
   }
}
