#include "ntuple_test.hxx"
#include "ntuple_fork.hxx"
#include "TSystem.h"

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
      opts.SetEmulateUnknownTypes(false);
      try {
         auto model = desc.CreateModel(opts);
         FAIL() << "Creating a model without fEmulateUnknownTypes should fail";
      } catch (const ROOT::RException &ex) {
         ASSERT_THAT(ex.GetError().GetReport(), testing::HasSubstr("unknown type"));
      }
   }

   {
      RNTupleDescriptor::RCreateModelOptions opts;
      opts.SetEmulateUnknownTypes(true);
      auto model = desc.CreateModel(opts);
      ASSERT_NE(model, nullptr);

      const auto &outer = model->GetConstField("f");
      ASSERT_EQ(outer.GetTypeName(), "Outer_Simple");
      ASSERT_EQ(outer.GetStructure(), ROOT::ENTupleStructure::kRecord);
      ASSERT_EQ(outer.GetConstSubfields().size(), 2);

      const auto subfields = outer.GetConstSubfields();
      ASSERT_EQ(subfields[0]->GetFieldName(), "fInt1");

      const auto *inner = subfields[1];
      ASSERT_EQ(inner->GetTypeName(), "Inner_Simple");
      ASSERT_EQ(inner->GetFieldName(), "fInner");
      ASSERT_EQ(inner->GetStructure(), ROOT::ENTupleStructure::kRecord);
      ASSERT_NE(inner->GetTraits() & RFieldBase::kTraitEmulatedField, 0);
      ASSERT_EQ(inner->GetConstSubfields().size(), 2);
      ASSERT_EQ(inner->GetConstSubfields()[0]->GetFieldName(), "fInt1");
   }

   // Now test loading entries with a reader.
   // NOTE: using a TFile-based reader exercises the code path where the user-defined type
   // gets loaded as an Emulated TClass, which we must make sure we handle properly.
   RNTupleDescriptor::RCreateModelOptions cmOpts;
   cmOpts.SetEmulateUnknownTypes(true);

   ROOT::TestSupport::CheckDiagsRAII diagRAII;
   diagRAII.optionalDiag(kWarning, "TClass::Init", "no dictionary for class",
                         /*matchFullMessage=*/false);
   std::unique_ptr<TFile> file(TFile::Open(fileGuard.GetPath().c_str()));
   std::unique_ptr<ROOT::RNTuple> ntpl(file->Get<ROOT::RNTuple>("ntpl"));
   reader = RNTupleReader::Open(cmOpts, *ntpl);

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
      opts.SetEmulateUnknownTypes(false);
      try {
         auto model = desc.CreateModel(opts);
         FAIL() << "Creating a model without fEmulateUnknownTypes should fail";
      } catch (const ROOT::RException &ex) {
         ASSERT_THAT(ex.GetError().GetReport(), testing::HasSubstr("unknown type"));
      }
   }

   {
      RNTupleDescriptor::RCreateModelOptions opts;
      opts.SetEmulateUnknownTypes(true);
      auto model = desc.CreateModel(opts);
      ASSERT_NE(model, nullptr);

      const auto &outers = model->GetConstField("outers");
      ASSERT_EQ(outers.GetTypeName(), "std::vector<Outer_Vecs>");
      ASSERT_EQ(outers.GetStructure(), ROOT::ENTupleStructure::kCollection);
      ASSERT_EQ(outers.GetConstSubfields().size(), 1);

      const auto subfields = outers.GetConstSubfields();
      const auto *outer = subfields[0];
      ASSERT_EQ(outer->GetTypeName(), "Outer_Vecs");
      ASSERT_EQ(outer->GetStructure(), ROOT::ENTupleStructure::kRecord);
      ASSERT_NE(outer->GetTraits() & RFieldBase::kTraitEmulatedField, 0);

      const auto outersubfields = outer->GetConstSubfields();
      ASSERT_EQ(outersubfields.size(), 3);

      const auto *inners = outersubfields[0];
      ASSERT_EQ(inners->GetTypeName(), "std::vector<Inner_Vecs>");
      ASSERT_EQ(inners->GetFieldName(), "fInners");
      ASSERT_EQ(inners->GetStructure(), ROOT::ENTupleStructure::kCollection);
      ASSERT_EQ(inners->GetConstSubfields().size(), 1);
      ASSERT_EQ(inners->GetConstSubfields()[0]->GetFieldName(), "_0");

      const auto innersubfields = inners->GetConstSubfields();
      const auto *inner = innersubfields[0];
      ASSERT_EQ(inner->GetTypeName(), "Inner_Vecs");
      ASSERT_EQ(inner->GetStructure(), ROOT::ENTupleStructure::kRecord);
      ASSERT_EQ(inner->GetConstSubfields().size(), 1);
      ASSERT_EQ(inner->GetConstSubfields()[0]->GetFieldName(), "fFlt");
   }

   // Now test loading entries with a reader
   RNTupleDescriptor::RCreateModelOptions cmOpts;
   cmOpts.SetEmulateUnknownTypes(true);
   reader = RNTupleReader::Open(cmOpts, "ntpl", fileGuard.GetPath());
   reader->LoadEntry(0);
}

TEST(RNTupleEmulated, EmulatedFields_VecsTemplatedWrapper)
{
   FileRaii fileGuard("test_ntuple_emulated_fields_vecs_templated_wrapper.root");

   ExecInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
         template <typename T>
         struct TemplatedWrapper {
            T fValue;
         };
      )"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("vec", "std::vector<TemplatedWrapper<float>>").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("vec").get();
      DeclarePointer("std::vector<TemplatedWrapper<float>>", "ptrVec", ptr);

      ProcessLine("ptrVec->push_back(TemplatedWrapper<float>{1.0f});");
      writer->Fill();

      writer.reset();
   });

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();

   {
      RNTupleDescriptor::RCreateModelOptions opts;
      opts.SetEmulateUnknownTypes(false);
      try {
         auto model = desc.CreateModel(opts);
         FAIL() << "Creating a model without fEmulateUnknownTypes should fail";
      } catch (const ROOT::RException &ex) {
         ASSERT_THAT(ex.GetError().GetReport(), testing::HasSubstr("unknown type"));
      }
   }

   {
      RNTupleDescriptor::RCreateModelOptions opts;
      opts.SetEmulateUnknownTypes(true);
      auto model = desc.CreateModel(opts);
      ASSERT_NE(model, nullptr);

      const auto &vecField = model->GetConstField("vec");
      ASSERT_EQ(vecField.GetTypeName(), "std::vector<TemplatedWrapper<float>>");
      ASSERT_EQ(vecField.GetStructure(), ROOT::ENTupleStructure::kCollection);
      ASSERT_EQ(vecField.GetConstSubfields().size(), 1);

      const auto *wrapperField = vecField.GetConstSubfields()[0];
      ASSERT_EQ(wrapperField->GetTypeName(), "TemplatedWrapper<float>");
      ASSERT_EQ(wrapperField->GetStructure(), ROOT::ENTupleStructure::kRecord);
      ASSERT_NE(wrapperField->GetTraits() & RFieldBase::kTraitEmulatedField, 0);
      ASSERT_EQ(wrapperField->GetConstSubfields().size(), 1);

      const auto *innerField = wrapperField->GetConstSubfields()[0];
      ASSERT_EQ(innerField->GetTypeName(), "float");
      ASSERT_EQ(innerField->GetFieldName(), "fValue");
      ASSERT_EQ(innerField->GetStructure(), ROOT::ENTupleStructure::kLeaf);
   }

   // Now test loading entries with a reader
   RNTupleDescriptor::RCreateModelOptions cmOpts;
   cmOpts.SetEmulateUnknownTypes(true);
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
      opts.SetEmulateUnknownTypes(false);
      try {
         auto model = desc.CreateModel(opts);
         FAIL() << "Creating a model without fEmulateUnknownTypes should fail";
      } catch (const ROOT::RException &ex) {
         ASSERT_THAT(ex.GetError().GetReport(), testing::HasSubstr("unknown type"));
      }
   }

   {
      RNTupleDescriptor::RCreateModelOptions opts;
      opts.SetEmulateUnknownTypes(true);
      auto model = desc.CreateModel(opts);
      ASSERT_NE(model, nullptr);

      const auto &outer = model->GetConstField("f");
      ASSERT_EQ(outer.GetTypeName(), "Outer_EmptyStruct");
      ASSERT_EQ(outer.GetStructure(), ROOT::ENTupleStructure::kRecord);
      ASSERT_EQ(outer.GetConstSubfields().size(), 1);

      const auto subfields = outer.GetConstSubfields();
      const auto *inner = subfields[0];
      ASSERT_EQ(inner->GetTypeName(), "Inner_EmptyStruct");
      ASSERT_EQ(inner->GetFieldName(), "fInner");
      ASSERT_EQ(inner->GetStructure(), ROOT::ENTupleStructure::kRecord);
      ASSERT_NE(inner->GetTraits() & RFieldBase::kTraitEmulatedField, 0);
      ASSERT_EQ(inner->GetConstSubfields().size(), 0);
   }

   // Now test loading entries with a reader
   RNTupleDescriptor::RCreateModelOptions cmOpts;
   cmOpts.SetEmulateUnknownTypes(true);
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
      opts.SetEmulateUnknownTypes(false);
      try {
         auto model = desc.CreateModel(opts);
         FAIL() << "Creating a model without fEmulateUnknownTypes should fail";
      } catch (const ROOT::RException &ex) {
         ASSERT_THAT(ex.GetError().GetReport(), testing::HasSubstr("unknown type"));
      }
   }

   {
      RNTupleDescriptor::RCreateModelOptions opts;
      opts.SetEmulateUnknownTypes(true);
      auto model = desc.CreateModel(opts);
      ASSERT_NE(model, nullptr);

      const auto &outer = model->GetConstField("f");
      ASSERT_EQ(outer.GetTypeName(), "std::vector<Inner_EmptyVec>");
      ASSERT_EQ(outer.GetStructure(), ROOT::ENTupleStructure::kCollection);
      ASSERT_EQ(outer.GetConstSubfields().size(), 1);
      ASSERT_EQ(outer.GetTraits() & RFieldBase::kTraitEmulatedField, 0);

      const auto subfields = outer.GetConstSubfields();
      const auto *inner = subfields[0];
      ASSERT_EQ(inner->GetTypeName(), "Inner_EmptyVec");
      ASSERT_EQ(inner->GetFieldName(), "_0");
      ASSERT_EQ(inner->GetStructure(), ROOT::ENTupleStructure::kRecord);
      ASSERT_NE(inner->GetTraits() & RFieldBase::kTraitEmulatedField, 0);
      ASSERT_EQ(inner->GetConstSubfields().size(), 0);
   }

   // Now test loading entries with a reader
   RNTupleDescriptor::RCreateModelOptions cmOpts;
   cmOpts.SetEmulateUnknownTypes(true);
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
   cmOpts.SetEmulateUnknownTypes(true);
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

TEST(RNTupleEmulated, CollectionProxy)
{
   FileRaii fileGuard("test_ntuple_emulated_collproxy.root");

   // Declare a custom type in a separate process, then write it through a custom collection proxy.
   // In the main process we load the generated RNTuple without having its dictionary available and we verify
   // we can read it (only) via field emulation (as an untyped VectorField).
   ExecInFork([&] {
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
         // Copypasted from StructUsingCollectionProxy in CustomStruct.hxx
         // Using a separate type because we don't want to have its dictionary loaded in the test.
         template <typename T>
         struct StructWithCollectionProxyForEmuTest {
            using ValueType = T;
            std::vector<T> v; //! do not accidentally store via RClassField
         };

         // Copypasted from SimpleCollectionProxy.hxx
         // Using a separate type because we don't want to have its dictionary loaded in the test.
         template <typename CollectionT>
         class SimpleCollectionProxyForEmuTest : public TVirtualCollectionProxy {
            /// The internal representation of an iterator, which in this simple test only contains a pointer to an element
            struct IteratorData {
               typename CollectionT::ValueType *ptr;
            };

            static void
            Func_CreateIterators(void *collection, void **begin_arena, void **end_arena, TVirtualCollectionProxy * /*proxy*/)
            {
               static_assert(sizeof(IteratorData) <= TVirtualCollectionProxy::fgIteratorArenaSize);
               auto &vec = static_cast<CollectionT *>(collection)->v;
               // An iterator on an array-backed container is just a pointer; thus, it can be directly stored in `*xyz_arena`,
               // saving one dereference (see TVirtualCollectionProxy documentation)
               *begin_arena = vec.data();
               *end_arena = vec.data() + vec.size();
            }

            static void *Func_Next(void *iter, const void *end)
            {
               auto _iter = static_cast<IteratorData *>(iter);
               auto _end = static_cast<const IteratorData *>(end);
               if (_iter->ptr >= _end->ptr)
                  return nullptr;
               return _iter->ptr++;
            }

            static void Func_DeleteTwoIterators(void * /*begin*/, void * /*end*/) {}

         private:
            CollectionT *fObject = nullptr;

         public:
            SimpleCollectionProxyForEmuTest()
               : TVirtualCollectionProxy(TClass::GetClass(ROOT::Internal::GetDemangledTypeName(typeid(CollectionT)).c_str()))
            {
            }
            SimpleCollectionProxyForEmuTest(const SimpleCollectionProxyForEmuTest<CollectionT> &) : SimpleCollectionProxyForEmuTest() {}

            TVirtualCollectionProxy *Generate() const override
            {
               return new SimpleCollectionProxyForEmuTest<CollectionT>(*this);
            }
            Int_t GetCollectionType() const override { return ROOT::kSTLvector; }
            ULong_t GetIncrement() const override { return sizeof(typename CollectionT::ValueType); }
            UInt_t Sizeof() const override { return sizeof(CollectionT); }
            bool HasPointers() const override { return false; }

            TClass *GetValueClass() const override
            {
               if constexpr (std::is_fundamental<typename CollectionT::ValueType>::value)
                  return nullptr;
               return TClass::GetClass(ROOT::Internal::GetDemangledTypeName(typeid(typename CollectionT::ValueType)).c_str());
            }
            EDataType GetType() const override
            {
               if constexpr (std::is_same<typename CollectionT::ValueType, char>::value)
                  return EDataType::kChar_t;
               ;
               if constexpr (std::is_same<typename CollectionT::ValueType, float>::value)
                  return EDataType::kFloat_t;
               ;
               return EDataType::kOther_t;
            }

            void PushProxy(void *objectstart) override { fObject = static_cast<CollectionT *>(objectstart); }
            void PopProxy() override { fObject = nullptr; }

            void *At(UInt_t idx) override { return &fObject->v[idx]; }
            void Clear(const char * /*opt*/ = "") override { fObject->v.clear(); }
            UInt_t Size() const override { return fObject->v.size(); }
            void *Allocate(UInt_t n, bool /*forceDelete*/) override
            {
               fObject->v.resize(n);
               return fObject;
            }
            void Commit(void *) override {}
            void Insert(const void *data, void *container, size_t size) override
            {
               auto p = static_cast<const typename CollectionT::ValueType *>(data);
               for (size_t i = 0; i < size; ++i) {
                  static_cast<CollectionT *>(container)->v.push_back(p[i]);
               }
            }

            TStreamerInfoActions::TActionSequence *
            GetConversionReadMemberWiseActions(TClass * /*oldClass*/, Int_t /*version*/) override
            {
               return nullptr;
            }
            TStreamerInfoActions::TActionSequence *GetReadMemberWiseActions(Int_t /*version*/) override { return nullptr; }
            TStreamerInfoActions::TActionSequence *GetWriteMemberWiseActions() override { return nullptr; }

            CreateIterators_t GetFunctionCreateIterators(bool /*read*/ = true) override { return &Func_CreateIterators; }
            CopyIterator_t GetFunctionCopyIterator(bool /*read*/ = true) override { return nullptr; }
            Next_t GetFunctionNext(bool /*read*/ = true) override { return &Func_Next; }
            DeleteIterator_t GetFunctionDeleteIterator(bool /*read*/ = true) override { return nullptr; }
            DeleteTwoIterators_t GetFunctionDeleteTwoIterators(bool /*read*/ = true) override
            {
               return &Func_DeleteTwoIterators;
            }
         };

         namespace ROOT {
         template <>
         struct IsCollectionProxy<StructWithCollectionProxyForEmuTest<char>> : std::true_type {
         };
         } // namespace ROOT                                       

         SimpleCollectionProxyForEmuTest<StructWithCollectionProxyForEmuTest<char>> proxyC;
         auto klassC = TClass::GetClass("StructWithCollectionProxyForEmuTest<char>");
         klassC->CopyCollectionProxy(proxyC);
      )"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("proxyC", "StructWithCollectionProxyForEmuTest<char>").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("proxyC").get();
      DeclarePointer("StructWithCollectionProxyForEmuTest<char>", "pProxyC", ptr);
      ProcessLine("for(int i = 0; i < 100; ++i) pProxyC->v.push_back(0x42);");
      writer->Fill();

      ProcessLine("pProxyC->v.clear();");
      writer->Fill();

      // TStreamerInfo::Build will report a warning for interpreted classes (but only for members).
      // See also https://github.com/root-project/root/issues/9371
      ROOT::TestSupport::CheckDiagsRAII diagRAII;
      diagRAII.optionalDiag(kWarning, "TStreamerInfo::Build", "has no streamer or dictionary",
                            /*matchFullMessage=*/false);
      writer.reset();
   });

   try {
      auto reader = ROOT::RNTupleReader::Open("ntpl", fileGuard.GetPath());
      reader->GetModel();
      FAIL() << "Reconstructing the model without emulating fields should fail";
   } catch (const ROOT::RException &ex) {
      EXPECT_THAT(ex.what(), testing::HasSubstr("unknown type: StructWithCollectionProxyForEmuTest<char>"));
   }

   auto opts = RNTupleDescriptor::RCreateModelOptions();
   opts.SetEmulateUnknownTypes(true);
   auto reader = ROOT::RNTupleReader::Open(opts, "ntpl", fileGuard.GetPath());
   EXPECT_EQ(reader->GetNEntries(), 2);

   const auto &model = reader->GetModel();
   const auto &field = model.GetConstField("proxyC");
   EXPECT_EQ(field.GetTypeName(), "StructWithCollectionProxyForEmuTest<char>");
   EXPECT_EQ(field.GetStructure(), ROOT::ENTupleStructure::kCollection);
   EXPECT_EQ(field.GetConstSubfields().size(), 1);
   EXPECT_NE(field.GetTraits() & RFieldBase::kTraitEmulatedField, 0);
   EXPECT_EQ(field.GetValueSize(), sizeof(std::vector<char>));

   const auto *vec = dynamic_cast<const ROOT::RVectorField *>(&field);
   ASSERT_NE(vec, nullptr);
   RNTupleLocalIndex collectionStart;
   ROOT::NTupleSize_t size;
   vec->GetCollectionInfo(0, &collectionStart, &size);
   EXPECT_EQ(collectionStart.GetClusterId(), 0);
   EXPECT_EQ(collectionStart.GetIndexInCluster(), 0);
   EXPECT_EQ(size, 100);
   vec->GetCollectionInfo(1, &collectionStart, &size);
   EXPECT_EQ(collectionStart.GetClusterId(), 0);
   EXPECT_EQ(collectionStart.GetIndexInCluster(), 100);
   EXPECT_EQ(size, 0);

   reader->LoadEntry(0);
}
