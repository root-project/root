#include "ntuple_test.hxx"

namespace {
class RNoDictionary {};
} // namespace

TEST(RNTuple, TClass) {
   auto modelFail = RNTupleModel::Create();
   EXPECT_THROW(modelFail->MakeField<RNoDictionary>("nodict"), ROOT::Experimental::RException);
   EXPECT_THROW(modelFail->MakeField<Cyclic>("cyclic"), ROOT::Experimental::RException);

   auto model = RNTupleModel::Create();
   auto ptrKlass = model->MakeField<CustomStruct>("klass");

   // TDatime would be a supported class layout but it is unsplittable due to its custom streamer
   EXPECT_THROW(model->MakeField<TDatime>("datime"), RException);

   FileRaii fileGuard("test_ntuple_tclass.root");
   auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
}

TEST(RNTuple, DiamondInheritance)
{
   FileRaii fileGuard("test_ntuple_diamond_inheritance.root");

   {
      auto model = RNTupleModel::Create();
      auto d = model->MakeField<DuplicateBaseD>("d");
      EXPECT_THROW(model->MakeField<DiamondVirtualD>("vd"), RException);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      d->DuplicateBaseB::a = 1.0;
      d->DuplicateBaseC::a = 1.5;
      d->b = 2.0;
      d->c = 3.0;
      d->d = 4.0;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto d = reader->GetModel().GetDefaultEntry().GetPtr<DuplicateBaseD>("d");
   EXPECT_EQ(1u, reader->GetNEntries());

   reader->LoadEntry(0);
   EXPECT_FLOAT_EQ(1.0, d->DuplicateBaseB::a);
   EXPECT_FLOAT_EQ(1.5, d->DuplicateBaseC::a);
   EXPECT_FLOAT_EQ(2.0, d->b);
   EXPECT_FLOAT_EQ(3.0, d->c);
   EXPECT_FLOAT_EQ(4.0, d->d);
}
