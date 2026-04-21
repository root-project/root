#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include "NtplEvolv_v2.hxx"

#include <memory>

int main()
{
   auto model = ROOT::RNTupleModel::Create();
   auto event = model->MakeField<NtplEvolv>("event");
   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", "root_test_ntpl_evolution.root");

   event->fA = 1;
   writer->Fill();

   return 0;
}
