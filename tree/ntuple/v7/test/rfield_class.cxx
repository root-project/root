#include "ntuple_test.hxx"

namespace {
class RNoDictionary {};
} // namespace

TEST(RNTuple, TClass) {
   auto modelFail = RNTupleModel::Create();
   EXPECT_THROW(modelFail->MakeField<RNoDictionary>("nodict"), ROOT::Experimental::RException);

   auto model = RNTupleModel::Create();
   auto ptrKlass = model->MakeField<CustomStruct>("klass");
   model->Freeze();
   const auto &fieldA = model->GetField("klass.a");
   EXPECT_EQ(std::string("The most important member of this struct."), fieldA.GetDescription());

   FileRaii fileGuard("test_ntuple_tclass.root");
   auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
}
