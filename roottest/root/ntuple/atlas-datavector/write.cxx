#include <ROOT/RField.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>

int main()
{
   auto model = ROOT::RNTupleModel::Create();
   model->AddField(ROOT::RFieldBase::Create("my_field", "AtlasLikeDataVector<CustomStruct>").Unwrap());
   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", "test_ntuple_datavector.root");
   writer->Fill();
}
