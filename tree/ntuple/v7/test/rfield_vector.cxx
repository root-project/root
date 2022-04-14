#include "ntuple_test.hxx"

TEST(RNTuple, ClassVector)
{
   FileRaii fileGuard("test_ntuple_classvector.root");

   auto modelWrite = RNTupleModel::Create();
   auto wrKlassVec = modelWrite->MakeField<std::vector<CustomStruct>>("klassVec");
   CustomStruct klass;
   klass.a = 42.0;
   klass.v1.emplace_back(1.0);
   wrKlassVec->emplace_back(klass);
   klass.a = 137.0;
   klass.v1.emplace_back(2.0);
   wrKlassVec->emplace_back(klass);

   {
      RNTupleWriter ntuple(std::move(modelWrite),
         std::make_unique<RPageSinkFile>("myNTuple", fileGuard.GetPath(), RNTupleWriteOptions()));
      ntuple.Fill();
   }

   RNTupleReader ntuple(std::make_unique<RPageSourceFile>("myNTuple", fileGuard.GetPath(), RNTupleReadOptions()));
   EXPECT_EQ(1U, ntuple.GetNEntries());

   auto viewKlassVec = ntuple.GetViewCollection("klassVec");
   auto viewKlass = viewKlassVec.GetView<CustomStruct>("_0");
   auto viewKlassA = viewKlassVec.GetView<float>("_0.a");

   // Loop over all the elements of the vector in all the entries
   for (auto elementId : viewKlassVec.GetFieldRange()) {
      if (elementId == 0) {
         EXPECT_EQ(42.0, viewKlass(elementId).a);
         EXPECT_EQ(1u, viewKlass(elementId).v1.size());
         EXPECT_EQ(1.0, viewKlass(elementId).v1[0]);
         EXPECT_EQ(42.0, viewKlassA(elementId));
      } else if (elementId == 1) {
         EXPECT_EQ(137.0, viewKlass(elementId).a);
         EXPECT_EQ(2u, viewKlass(elementId).v1.size());
         EXPECT_EQ(1.0, viewKlass(elementId).v1[0]);
         EXPECT_EQ(2.0, viewKlass(elementId).v1[1]);
         EXPECT_EQ(137.0, viewKlassA(elementId));
      } else {
         EXPECT_TRUE(false);
      }
   }
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
      RNTupleWriter ntuple(std::move(modelWrite),
         std::make_unique<RPageSinkFile>("myNTuple", fileGuard.GetPath(), RNTupleWriteOptions()));
      ntuple.Fill();
   }

   auto source = std::make_unique<RPageSourceFile>("myNTuple", fileGuard.GetPath(), RNTupleReadOptions());
   source->Attach();
   EXPECT_EQ(1U, source->GetNEntries());

   auto idKlassVec = source->GetSharedDescriptorGuard()->FindFieldId("klassVec");
   ASSERT_NE(idKlassVec, ROOT::Experimental::kInvalidDescriptorId);
   auto idKlass = source->GetSharedDescriptorGuard()->FindFieldId("_0", idKlassVec);
   ASSERT_NE(idKlass, ROOT::Experimental::kInvalidDescriptorId);
   auto idA = source->GetSharedDescriptorGuard()->FindFieldId("a", idKlass);
   ASSERT_NE(idA, ROOT::Experimental::kInvalidDescriptorId);
   auto fieldInner = std::unique_ptr<RFieldBase>(RFieldBase::Create("klassVec.a", "float").Unwrap());
   fieldInner->SetOnDiskId(idA);
   fieldInner->ConnectPageSource(*source);

   auto field = std::make_unique<ROOT::Experimental::RVectorField>("klassVec", std::move(fieldInner));
   field->SetOnDiskId(idKlassVec);
   field->ConnectPageSource(*source);

   auto value = field->GenerateValue();
   field->Read(0, &value);
   auto aVec = value.Get<std::vector<float>>();
   EXPECT_EQ(1U, aVec->size());
   EXPECT_EQ(42.0, (*aVec)[0]);
   field->DestroyValue(value);

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
      RNTupleWriter ntuple(std::move(modelWrite),
         std::make_unique<RPageSinkFile>("myNTuple", fileGuard.GetPath(), RNTupleWriteOptions()));
      ntuple.Fill();
      wrJets->clear();
      wrJets->push_back(1.0);
      ntuple.Fill();
   }

   auto modelReadAsRVec = RNTupleModel::Create();
   auto rdJetsAsRVec = modelReadAsRVec->MakeField<ROOT::VecOps::RVec<float>>("jets");

   RNTupleReader ntupleRVec(std::move(modelReadAsRVec),
      std::make_unique<RPageSourceFile>("myNTuple", fileGuard.GetPath(), RNTupleReadOptions()));
   EXPECT_EQ(2U, ntupleRVec.GetNEntries());

   ntupleRVec.LoadEntry(0);
   EXPECT_EQ(2U, rdJetsAsRVec->size());
   EXPECT_EQ(42.0, (*rdJetsAsRVec)[0]);
   EXPECT_EQ(7.0, (*rdJetsAsRVec)[1]);

   ntupleRVec.LoadEntry(1);
   EXPECT_EQ(1U, rdJetsAsRVec->size());
   EXPECT_EQ(1.0, (*rdJetsAsRVec)[0]);

   auto modelReadAsStdVector = RNTupleModel::Create();
   auto rdJetsAsStdVector = modelReadAsStdVector->MakeField<std::vector<float>>("jets");

   RNTupleReader ntupleStdVector(std::move(modelReadAsStdVector), std::make_unique<RPageSourceFile>(
      "myNTuple", fileGuard.GetPath(), RNTupleReadOptions()));
   EXPECT_EQ(2U, ntupleRVec.GetNEntries());

   ntupleStdVector.LoadEntry(0);
   EXPECT_EQ(2U, rdJetsAsStdVector->size());
   EXPECT_EQ(42.0, (*rdJetsAsStdVector)[0]);
   EXPECT_EQ(7.0, (*rdJetsAsStdVector)[1]);

   ntupleStdVector.LoadEntry(1);
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
      m->GetDefaultEntry()->CaptureValueUnsafe("v", (void *)&rvec);

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
   auto v = r->GetModel()->Get<ROOT::RVec<int>>("v");

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

   auto modelRead = modelWrite->Clone();

   {
      RNTupleWriter ntuple(std::move(modelWrite),
         std::make_unique<RPageSinkFile>("myNTuple", fileGuard.GetPath(), RNTupleWriteOptions()));
      ntuple.Fill();
   }

   auto rdBoolStdVec = modelRead->Get<std::vector<bool>>("boolStdVec");
   auto rdBoolRVec = modelRead->Get<ROOT::RVec<bool>>("boolRVec");
   RNTupleReader ntuple(std::move(modelRead),
      std::make_unique<RPageSourceFile>("myNTuple", fileGuard.GetPath(), RNTupleReadOptions()));
   EXPECT_EQ(1U, ntuple.GetNEntries());
   ntuple.LoadEntry(0);

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


TEST(RNTuple, ComplexVector)
{
   FileRaii fileGuard("test_ntuple_vec_complex.root");

   auto model = RNTupleModel::Create();
   auto wrV = model->MakeField<std::vector<ComplexStruct>>("v");

   {
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "T", fileGuard.GetPath());
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

   ComplexStruct::SetNCallConstructor(0);
   ComplexStruct::SetNCallDestructor(0);
   {
      auto ntuple = RNTupleReader::Open("T", fileGuard.GetPath());
      auto rdV = ntuple->GetModel()->GetDefaultEntry()->Get<std::vector<ComplexStruct>>("v");

      ntuple->LoadEntry(0);
      EXPECT_EQ(0, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(0, ComplexStruct::GetNCallDestructor());
      EXPECT_TRUE(rdV->empty());
      ntuple->LoadEntry(1);
      EXPECT_EQ(10, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(0, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(10u, rdV->size());
      ntuple->LoadEntry(2);
      EXPECT_EQ(10000, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(0, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(10000u, rdV->size());
      ntuple->LoadEntry(3);
      EXPECT_EQ(10000, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(9900, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(100u, rdV->size());
      ntuple->LoadEntry(4);
      EXPECT_EQ(10100, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(9900, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(200u, rdV->size());
      ntuple->LoadEntry(5);
      EXPECT_EQ(10100, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(10100, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(0u, rdV->size());
      ntuple->LoadEntry(6);
      EXPECT_EQ(10101, ComplexStruct::GetNCallConstructor());
      EXPECT_EQ(10100, ComplexStruct::GetNCallDestructor());
      EXPECT_EQ(1u, rdV->size());
   }
   EXPECT_EQ(10101u, ComplexStruct::GetNCallDestructor());
}
