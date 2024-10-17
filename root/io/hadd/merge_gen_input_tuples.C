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
      auto fi = model->MakeField<int>("I", 1337);
      auto fl = model->MakeField<long>("L", 666);
      auto opts = RNTupleWriteOptions{};
      opts.SetCompression(ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kZSTD, 5));
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fname1, opts);
      for (int i = 0; i < 1000; ++i) {
         *fi = i;
         *fl = i;
         writer->Fill();
      }
   }
   {
      auto model = RNTupleModel::Create();
      auto fi = model->MakeField<int>("I", 123);
      auto fl = model->MakeField<long>("L", 420);
      auto opts = RNTupleWriteOptions{};
      opts.SetCompression(ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kZLIB, 1));
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fname2, opts);
      for (int i = 0; i < 100; ++i) {
         *fi = i;
         *fl = i;
         writer->Fill();
      }
   }
}
