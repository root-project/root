#include "ntuple_makeproject_header.h"
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>

using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;

int main()
{
   auto model = RNTupleModel::Create();
   auto fldStlEvent = model->MakeField<MySTLEvent>("test");

   auto ntuple = RNTupleWriter::Recreate(std::move(model), "events", "ntuple_makeproject_stl_example_rntuple.root");
   ntuple->Fill();
   return 0;
}
