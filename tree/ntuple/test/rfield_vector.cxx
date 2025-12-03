#include "ntuple_test.hxx"

#include <TObject.h>

// A layer of indirection to hide std::vector's second template parameter.
// This way we can generalize tests over RVec and std::vector using a template template parameter (see below).
template <typename T>
using Vector_t = std::vector<T>;

template <template <typename> class Coll_t>
void TestClassVector(const char *fname)
{
   FileRaii fileGuard(fname);

   auto modelWrite = RNTupleModel::Create();
   auto wrKlassVec = modelWrite->MakeField<Coll_t<CustomStruct>>("klassVec");
   CustomStruct klass;
   klass.a = 42.f;
   klass.v1.emplace_back(1.f);
   wrKlassVec->emplace_back(klass);

   {
      auto writer = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath());
      writer->Fill();

      // enlarge
      wrKlassVec->emplace_back(CustomStruct{1.f, {1.f, 2.f, 3.f}, {{42.f}}, "foo", std::byte{0}});
      wrKlassVec->emplace_back(CustomStruct{2.f, {4.f, 5.f, 6.f}, {{1.f}, {2.f, 3.f}}, "bar", std::byte{0}});
      writer->Fill();

      // shrink
      wrKlassVec->clear();
      wrKlassVec->emplace_back(CustomStruct{3.f, {7.f, 8.f, 9.f}, {{4.f, 5.f}, {}}, "baz", std::byte{0}});
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("myNTuple", fileGuard.GetPath());
   EXPECT_EQ(3U, reader->GetNEntries());

   auto viewKlassVec = reader->GetCollectionView("klassVec");
   auto viewKlass = viewKlassVec.GetView<CustomStruct>("_0");
   auto viewKlassA = viewKlassVec.GetView<float>("_0.a");
   auto viewKlassV1 = viewKlassVec.GetView<std::vector<float>>("_0.v1");
   auto viewKlassV2 = viewKlassVec.GetView<std::vector<std::vector<float>>>("_0.v2");
   auto viewKlassS = viewKlassVec.GetView<std::string>("_0.s");

   // first entry, one element
   EXPECT_EQ(42.f, viewKlass(0).a);
   EXPECT_EQ(42.f, viewKlassA(0));
   EXPECT_EQ(std::vector<float>({1.f}), viewKlassV1(0));
   EXPECT_TRUE(viewKlassV2(0).empty());
   EXPECT_TRUE(viewKlassS(0).empty());
   EXPECT_EQ(viewKlass(0).v1, viewKlassV1(0));
   EXPECT_EQ(viewKlass(0).v2, viewKlassV2(0));

   // second entry, three elements
   EXPECT_EQ(42.f, viewKlass(0).a);
   EXPECT_EQ(42.f, viewKlassA(0));
   EXPECT_EQ(std::vector<float>({1.f}), viewKlassV1(0));
   EXPECT_TRUE(viewKlassV2(0).empty());
   EXPECT_TRUE(viewKlassS(0).empty());
   EXPECT_EQ(viewKlass(0).v1, viewKlassV1(0));
   EXPECT_EQ(viewKlass(0).v2, viewKlassV2(0));

   EXPECT_EQ(1.f, viewKlass(2).a);
   EXPECT_EQ(1.f, viewKlassA(2));
   EXPECT_EQ(std::vector<float>({1.f, 2.f, 3.f}), viewKlassV1(2));
   EXPECT_EQ(std::vector<std::vector<float>>({{42.f}}), viewKlassV2(2));
   EXPECT_EQ("foo", viewKlassS(2));
   EXPECT_EQ(viewKlass(2).v1, viewKlassV1(2));
   EXPECT_EQ(viewKlass(2).v2, viewKlassV2(2));

   EXPECT_EQ(2.f, viewKlass(3).a);
   EXPECT_EQ(2.f, viewKlassA(3));
   EXPECT_EQ(std::vector<float>({4.f, 5.f, 6.f}), viewKlassV1(3));
   EXPECT_EQ(std::vector<std::vector<float>>({{1.f}, {2.f, 3.f}}), viewKlassV2(3));
   EXPECT_EQ("bar", viewKlassS(3));
   EXPECT_EQ(viewKlass(3).v2, viewKlassV2(3));

   // third entry, one element
   EXPECT_EQ(3.f, viewKlass(4).a);
   EXPECT_EQ(3.f, viewKlassA(4));
   EXPECT_EQ(std::vector<float>({7.f, 8.f, 9.f}), viewKlassV1(4));
   EXPECT_EQ(std::vector<std::vector<float>>({{4.f, 5.f}, {}}), viewKlassV2(4));
   EXPECT_EQ("baz", viewKlassS(4));
   EXPECT_EQ(viewKlass(4).v2, viewKlassV2(4));
}

TEST(RNTuple, ClassVector)
{
   TestClassVector<Vector_t>("test_ntuple_classvector.root");
}

TEST(RNTuple, ClassRVec)
{
   TestClassVector<ROOT::RVec>("test_ntuple_classrvec.root");
}

TEST(RNTuple, InsideCollection)
{
   FileRaii fileGuard("test_ntuple_insidecollection.root");

   auto modelWrite = RNTupleModel::Create();
   auto wrKlassVec = modelWrite->MakeField<std::vector<CustomStruct>>("klassVec");
   CustomStruct klass;
   klass.a = 42.0;
   klass.v1.emplace_back(1.0);
   klass.v1.emplace_back(2.0);
   wrKlassVec->emplace_back(klass);

   {
      auto writer = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath());
      writer->Fill();
   }

   auto source = std::make_unique<RPageSourceFile>("myNTuple", fileGuard.GetPath(), RNTupleReadOptions());
   source->Attach();
   EXPECT_EQ(1U, source->GetNEntries());

   auto idKlassVec = source->GetSharedDescriptorGuard()->FindFieldId("klassVec");
   ASSERT_NE(idKlassVec, ROOT::kInvalidDescriptorId);
   auto idKlass = source->GetSharedDescriptorGuard()->FindFieldId("_0", idKlassVec);
   ASSERT_NE(idKlass, ROOT::kInvalidDescriptorId);
   auto idA = source->GetSharedDescriptorGuard()->FindFieldId("a", idKlass);
   ASSERT_NE(idA, ROOT::kInvalidDescriptorId);
   auto fieldInner = std::unique_ptr<RFieldBase>(RFieldBase::Create("klassVec_a", "float").Unwrap());
   fieldInner->SetOnDiskId(idA);

   auto field = std::make_unique<ROOT::RVectorField>("klassVec", std::move(fieldInner));
   field->SetOnDiskId(idKlassVec);
   ROOT::Internal::CallConnectPageSourceOnField(*field, *source);

   auto fieldCardinality64 = RFieldBase::Create("", "ROOT::RNTupleCardinality<std::uint64_t>").Unwrap();
   fieldCardinality64->SetOnDiskId(idKlassVec);
   ROOT::Internal::CallConnectPageSourceOnField(*fieldCardinality64, *source);
   auto fieldCardinality32 = RFieldBase::Create("", "ROOT::RNTupleCardinality<std::uint32_t>").Unwrap();
   fieldCardinality32->SetOnDiskId(idKlassVec);
   ROOT::Internal::CallConnectPageSourceOnField(*fieldCardinality32, *source);

   auto value = field->CreateValue();
   value.Read(0);
   auto aVec = value.GetRef<std::vector<float>>();
   EXPECT_EQ(1U, aVec.size());
   EXPECT_EQ(42.0, aVec[0]);

   auto valueCardinality64 = fieldCardinality64->CreateValue();
   valueCardinality64.Read(0);
   EXPECT_EQ(1U, valueCardinality64.GetRef<ROOT::RNTupleCardinality<std::uint64_t>>());

   auto valueCardinality32 = fieldCardinality32->CreateValue();
   valueCardinality32.Read(0);
   EXPECT_EQ(1U, valueCardinality32.GetRef<ROOT::RNTupleCardinality<std::uint32_t>>());

   // TODO: test reading of "klassVec.v1"
}


TEST(RNTuple, RVec)
{
   FileRaii fileGuard("test_ntuple_rvec.root");

   auto modelWrite = RNTupleModel::Create();
   auto wrJets = modelWrite->MakeField<ROOT::VecOps::RVec<float>>("jets");
   wrJets->push_back(42.0);
   wrJets->push_back(7.0);

   {
      auto writer = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath());
      writer->Fill();
      wrJets->clear();
      wrJets->push_back(1.0);
      writer->Fill();
   }

   auto modelReadAsRVec = RNTupleModel::Create();
   auto rdJetsAsRVec = modelReadAsRVec->MakeField<ROOT::VecOps::RVec<float>>("jets");

   auto ntupleRVec = RNTupleReader::Open(std::move(modelReadAsRVec), "myNTuple", fileGuard.GetPath());
   EXPECT_EQ(2U, ntupleRVec->GetNEntries());

   ntupleRVec->LoadEntry(0);
   EXPECT_EQ(2U, rdJetsAsRVec->size());
   EXPECT_EQ(42.0, (*rdJetsAsRVec)[0]);
   EXPECT_EQ(7.0, (*rdJetsAsRVec)[1]);

   ntupleRVec->LoadEntry(1);
   EXPECT_EQ(1U, rdJetsAsRVec->size());
   EXPECT_EQ(1.0, (*rdJetsAsRVec)[0]);

   auto modelReadAsStdVector = RNTupleModel::Create();
   auto rdJetsAsStdVector = modelReadAsStdVector->MakeField<std::vector<float>>("jets");

   auto ntupleStdVector = RNTupleReader::Open(std::move(modelReadAsStdVector), "myNTuple", fileGuard.GetPath());
   EXPECT_EQ(2U, ntupleStdVector->GetNEntries());

   ntupleStdVector->LoadEntry(0);
   EXPECT_EQ(2U, rdJetsAsStdVector->size());
   EXPECT_EQ(42.0, (*rdJetsAsStdVector)[0]);
   EXPECT_EQ(7.0, (*rdJetsAsStdVector)[1]);

   ntupleStdVector->LoadEntry(1);
   EXPECT_EQ(1U, rdJetsAsStdVector->size());
   EXPECT_EQ(1.0, (*rdJetsAsStdVector)[0]);
}

TEST(RNTuple, RVecTypeErased)
{
   FileRaii fileGuard("test_ntuple_rvec_typeerased.root");

   // write out RVec
   {
      ROOT::RVecI rvec = {1, 2, 3};

      auto m = RNTupleModel::Create();
      auto field = RFieldBase::Create("v", "ROOT::VecOps::RVec<int>").Unwrap();
      m->AddField(std::move(field));
      m->Freeze();
      m->GetDefaultEntry().BindRawPtr("v", &rvec);

      auto w = RNTupleWriter::Recreate(std::move(m), "r", fileGuard.GetPath());
      w->Fill();
      rvec.clear();
      rvec.push_back(42);
      w->Fill();
      rvec.clear();
      w->Fill();
   }

   // read back RVec with type-erased API
   auto r = RNTupleReader::Open("r", fileGuard.GetPath());
   auto v = r->GetModel().GetDefaultEntry().GetPtr<ROOT::RVec<int>>("v");

   r->LoadEntry(0);
   EXPECT_EQ(v->size(), 3);
   EXPECT_TRUE(All(*v == ROOT::RVec<int>{1, 2, 3}));

   r->LoadEntry(1);
   EXPECT_EQ(v->size(), 1);
   EXPECT_EQ(v->at(0), 42);

   r->LoadEntry(2);
   EXPECT_TRUE(v->empty());
}

TEST(RNTuple, ReadVectorAsRVec)
{
   FileRaii fileGuard("test_ntuple_vectorrvecinterop.root");

   // write out vector and RVec
   {
      auto m = RNTupleModel::Create();
      auto vec = m->MakeField<std::vector<float>>("vec");
      *vec = {1.f, 2.f, 3.f};
      auto r = RNTupleWriter::Recreate(std::move(m), "r", fileGuard.GetPath());
      r->Fill();
      vec->push_back(4.f);
      r->Fill();
      vec->clear();
      r->Fill();
   }

   // read them back as the other type
   auto m = RNTupleModel::Create();
   auto rvec = m->MakeField<ROOT::RVec<float>>("vec");
   auto r = RNTupleReader::Open(std::move(m), "r", fileGuard.GetPath());
   r->LoadEntry(0);
   EXPECT_TRUE(All(*rvec == ROOT::RVec<float>({1.f, 2.f, 3.f})));
   r->LoadEntry(1);
   EXPECT_TRUE(All(*rvec == ROOT::RVec<float>({1.f, 2.f, 3.f, 4.f})));
   r->LoadEntry(2);
   EXPECT_TRUE(rvec->empty());
}

TEST(RNTuple, BoolVector)
{
   FileRaii fileGuard("test_ntuple_boolvec.root");

   auto modelWrite = RNTupleModel::Create();
   auto wrBoolStdVec = modelWrite->MakeField<std::vector<bool>>("boolStdVec");
   auto wrBoolRVec = modelWrite->MakeField<ROOT::RVec<bool>>("boolRVec");
   wrBoolStdVec->push_back(true);
   wrBoolStdVec->push_back(false);
   wrBoolStdVec->push_back(true);
   wrBoolStdVec->push_back(false);
   wrBoolRVec->push_back(true);
   wrBoolRVec->push_back(false);
   wrBoolRVec->push_back(true);
   wrBoolRVec->push_back(false);

   modelWrite->Freeze();
   auto modelRead = modelWrite->Clone();

   {
      auto writer = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath());
      writer->Fill();
   }

   auto rdBoolStdVec = modelRead->GetDefaultEntry().GetPtr<std::vector<bool>>("boolStdVec");
   auto rdBoolRVec = modelRead->GetDefaultEntry().GetPtr<ROOT::RVec<bool>>("boolRVec");
   auto reader = RNTupleReader::Open(std::move(modelRead), "myNTuple", fileGuard.GetPath());
   EXPECT_EQ(1U, reader->GetNEntries());
   reader->LoadEntry(0);

   EXPECT_EQ(4U, rdBoolStdVec->size());
   EXPECT_TRUE((*rdBoolStdVec)[0]);
   EXPECT_FALSE((*rdBoolStdVec)[1]);
   EXPECT_TRUE((*rdBoolStdVec)[2]);
   EXPECT_FALSE((*rdBoolStdVec)[3]);
   EXPECT_EQ(4U, rdBoolRVec->size());
   EXPECT_TRUE((*rdBoolRVec)[0]);
   EXPECT_FALSE((*rdBoolRVec)[1]);
   EXPECT_TRUE((*rdBoolRVec)[2]);
   EXPECT_FALSE((*rdBoolRVec)[3]);
}

template <template <typename> class Coll_t>
void FillComplexVector(const std::string &fname)
{
   auto model = RNTupleModel::Create();
   auto wrV = model->MakeField<Coll_t<ComplexStruct>>("v");

   auto ntuple = RNTupleWriter::Recreate(std::move(model), "T", fname);
   ntuple->Fill();
   for (unsigned i = 0; i < 10; ++i)
      wrV->push_back(ComplexStruct());
   ntuple->Fill();
   wrV->clear();
   for (unsigned i = 0; i < 10000; ++i)
      wrV->push_back(ComplexStruct());
   ntuple->Fill();
   wrV->clear();
   for (unsigned i = 0; i < 100; ++i)
      wrV->push_back(ComplexStruct());
   ntuple->Fill();
   for (unsigned i = 0; i < 100; ++i)
      wrV->push_back(ComplexStruct());
   ntuple->Fill();
   wrV->clear();
   ntuple->Fill();
   wrV->push_back(ComplexStruct());
   ntuple->Fill();
}

// Note: RVec and std::vector differ in the number of construction/destruction calls when reading through
// a file. The reason is that we can control the memory re-allocation prodedure for RVec, which allows
// us to save destruction/construction calls.

TEST(RNTuple, ComplexVector)
{
   FileRaii fileGuard("test_ntuple_vec_complex.root");
   FillComplexVector<Vector_t>(fileGuard.GetPath());

   ComplexStruct::SetNCallConstructor(0);
   ComplexStruct::SetNCallDestructor(0);
   {
      auto ntuple = RNTupleReader::Open("T", fileGuard.GetPath());
      auto rdV = ntuple->GetModel().GetDefaultEntry().GetPtr<std::vector<ComplexStruct>>("v");

      ntuple->LoadEntry(0);
      EXPECT_EQ(0, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(0, ComplexStruct::GetNCallDestructor());
      EXPECT_TRUE(rdV->empty());
      ntuple->LoadEntry(1);
      EXPECT_EQ(10, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(0, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(10u, rdV->size());
      ntuple->LoadEntry(2);
      EXPECT_EQ(10010, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(10, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(10000u, rdV->size());
      ntuple->LoadEntry(3);
      EXPECT_EQ(10010, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(9910, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(100u, rdV->size());
      ntuple->LoadEntry(4);
      EXPECT_EQ(10210, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(10010, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(200u, rdV->size());
      ntuple->LoadEntry(5);
      EXPECT_EQ(10210, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(10210, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(0u, rdV->size());
      ntuple->LoadEntry(6);
      EXPECT_EQ(10211, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(10210, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(1u, rdV->size());
   }
   EXPECT_EQ(10211u, ComplexStruct::GetNCallDestructor());
}

TEST(RNTuple, ComplexRVec)
{
   FileRaii fileGuard("test_ntuple_rvec_complex.root");
   FillComplexVector<ROOT::RVec>(fileGuard.GetPath());

   ComplexStruct::SetNCallConstructor(0);
   ComplexStruct::SetNCallDestructor(0);
   {
      auto ntuple = RNTupleReader::Open("T", fileGuard.GetPath());
      auto rdV = ntuple->GetModel().GetDefaultEntry().GetPtr<ROOT::RVec<ComplexStruct>>("v");

      ntuple->LoadEntry(0);
      EXPECT_EQ(0, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(0, ComplexStruct::GetNCallDestructor());
      EXPECT_TRUE(rdV->empty());
      ntuple->LoadEntry(1);
      EXPECT_EQ(10, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(0, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(10u, rdV->size());
      ntuple->LoadEntry(2);
      EXPECT_EQ(10010, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(10, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(10000u, rdV->size());
      ntuple->LoadEntry(3);
      EXPECT_EQ(10010, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(9910, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(100u, rdV->size());
      ntuple->LoadEntry(4);
      EXPECT_EQ(10110, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(9910, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(200u, rdV->size());
      ntuple->LoadEntry(5);
      EXPECT_EQ(10110, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(10110, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(0u, rdV->size());
      ntuple->LoadEntry(6);
      EXPECT_EQ(10111, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(10110, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(1u, rdV->size());
   }
   EXPECT_EQ(10111u, ComplexStruct::GetNCallDestructor());
}

TEST(RNTuple, VectorOfString)
{
   FileRaii fileGuard("test_ntuple_vector_of_string.root");

   auto model = RNTupleModel::Create();
   auto ptrVecOfStr = model->MakeField<std::vector<std::string>>("f");
   auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
   ptrVecOfStr->emplace_back("abc");
   ptrVecOfStr->emplace_back("");
   ptrVecOfStr->emplace_back("de");
   ptrVecOfStr->emplace_back(std::string(1024, 'x'));
   writer->Fill();
   writer->CommitCluster();
   ptrVecOfStr->clear();
   writer->Fill();
   writer->CommitCluster();
   ptrVecOfStr->emplace_back("xyz");
   writer->Fill();
   writer.reset();

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ptrVecOfStr = reader->GetModel().GetDefaultEntry().GetPtr<std::vector<std::string>>("f");
   EXPECT_EQ(3u, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_EQ(4u, ptrVecOfStr->size());
   EXPECT_STREQ("abc", ptrVecOfStr->at(0).c_str());
   EXPECT_STREQ("de", ptrVecOfStr->at(2).c_str());
   EXPECT_EQ(std::string(1024, 'x'), ptrVecOfStr->at(3));
   EXPECT_TRUE(ptrVecOfStr->at(1).empty());
   reader->LoadEntry(1);
   EXPECT_TRUE(ptrVecOfStr->empty());
   reader->LoadEntry(2);
   EXPECT_EQ(1u, ptrVecOfStr->size());
   EXPECT_STREQ("xyz", ptrVecOfStr->at(0).c_str());
}

TEST(RNTuple, VectorOfBitset)
{
   FileRaii fileGuard("test_ntuple_vector_of_bitset.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrVecOfBitset = model->MakeField<std::vector<std::bitset<3>>>("f");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      ptrVecOfBitset->emplace_back("101");
      ptrVecOfBitset->emplace_back("010");
      writer->Fill();
      writer->CommitCluster();
      ptrVecOfBitset->clear();
      ptrVecOfBitset->emplace_back("111");
      ptrVecOfBitset->emplace_back("000");
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto ptrVecOfBitset = reader->GetModel().GetDefaultEntry().GetPtr<std::vector<std::bitset<3>>>("f");
   reader->LoadEntry(0);
   EXPECT_EQ(2u, ptrVecOfBitset->size());
   EXPECT_EQ("101", ptrVecOfBitset->at(0).to_string());
   EXPECT_EQ("010", ptrVecOfBitset->at(1).to_string());
   reader->LoadEntry(1);
   EXPECT_EQ(2u, ptrVecOfBitset->size());
   EXPECT_EQ("111", ptrVecOfBitset->at(0).to_string());
   EXPECT_EQ("000", ptrVecOfBitset->at(1).to_string());
}

TEST(RNTuple, VectorOfTObject)
{
   FileRaii fileGuard("test_ntuple_vector_of_tobject.root");
   {
      auto model = RNTupleModel::Create();
      auto objs = model->MakeField<std::vector<TObject>>("objs");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      TObject obj;
      obj.SetUniqueID(42);
      objs->push_back(obj);
      obj.SetUniqueID(43);
      objs->push_back(obj);
      writer->Fill();
      writer->CommitCluster();

      objs->clear();
      obj.SetUniqueID(44);
      objs->push_back(obj);
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2u, reader->GetNEntries());
   auto &entry = reader->GetModel().GetDefaultEntry();
   auto objs = entry.GetPtr<std::vector<TObject>>("objs");

   reader->LoadEntry(0);
   ASSERT_EQ(2u, objs->size());
   EXPECT_EQ(42u, objs->at(0).GetUniqueID());
   EXPECT_EQ(43u, objs->at(1).GetUniqueID());

   reader->LoadEntry(1);
   ASSERT_EQ(1u, objs->size());
   EXPECT_EQ(44u, objs->at(0).GetUniqueID());
}
