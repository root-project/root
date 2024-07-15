#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <Compression.h>

#include <TSystem.h>

#include <string>
#include <utility>

void merge_gen_input_tuples(const char *fname1 = "test_rntuple_input1.root", const char *fname2 = "test_rntuple_input2.root")
{
   using namespace ROOT::Experimental;
   
   auto noPrereleaseWarning = RLogScopedVerbosity(NTupleLog(), ROOT::Experimental::ELogLevel::kError);

   {
      auto model = RNTupleModel::Create();
      model->MakeField<int>("I", 1337);
      model->MakeField<float>("F", 666.f);
      auto opts = RNTupleWriteOptions{};
      opts.SetCompression(ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kZSTD, 5));
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fname1, opts);
      writer->Fill();
   }
   {
      auto model = RNTupleModel::Create();
      model->MakeField<int>("I", 123);
      model->MakeField<float>("F", 420.f);
      auto opts = RNTupleWriteOptions{};
      opts.SetCompression(ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kZLIB, 1));
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fname2, opts);
      writer->Fill();
   }
}
