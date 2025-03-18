#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <TSystem.h>

#include <string>
#include <utility>

void basics()
{
   using ROOT::Experimental::RNTupleWriter;
   using ROOT::Experimental::RNTupleReader;

   const std::string kFileName{"test_rntuple_basics.root"s};

   {
      auto model = ROOT::RNTupleModel::Create();
      *model->MakeField<int>("E") = 137;
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", kFileName);
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", kFileName);
   reader->Show(0);

   gSystem->Unlink(kFileName.c_str());
}
