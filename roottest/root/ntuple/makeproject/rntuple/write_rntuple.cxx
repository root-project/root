#include "ntuple_makeproject_header.h"
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>

int main()
{
   auto model = ROOT::RNTupleModel::Create();
   auto fldStlEvent = model->MakeField<MySTLEvent>("test");

   auto ntuple =
      ROOT::RNTupleWriter::Recreate(std::move(model), "events", "ntuple_makeproject_stl_example_rntuple.root");
   ntuple->Fill();
   return 0;
}
