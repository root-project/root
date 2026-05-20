#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include "Event_v2.hxx"

#include <memory>

int main()
{
   auto model = ROOT::RNTupleModel::Create();
   auto field = model->MakeField<Event>("event");
   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", "root_test_streamerfield.root");

   field->fField.fPtr = std::unique_ptr<StreamerBase>(new StreamerDerived());
   writer->Fill();

   return 0;
}
