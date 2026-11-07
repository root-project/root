#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <TSystem.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <utility>

void test_rntuple_metrics_env()
{
   constexpr std::size_t kNEntries = 1000;
   const std::string kFileName{"test_metrics_env.root"};

   auto model = ROOT::RNTupleModel::Create();
   auto pt = model->MakeField<float>("f");

   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", kFileName);
   for (std::size_t i = 0; i < kNEntries; ++i) {
      *pt = static_cast<float>(i);
      writer->Fill();
   }
   writer->CommitDataset();

   const auto &metrics = writer->GetMetrics();
   const auto *nPageCommitted = metrics.GetCounter("RNTupleWriter.RPageSinkBuf.RPageSinkFile.nPageCommitted");

   std::cout << "Metrics enabled: " << std::boolalpha << metrics.IsEnabled() << std::endl;
   std::cout << "nPageCommitted: " << (nPageCommitted ? std::to_string(nPageCommitted->GetValueAsInt()) : "?")
             << std::endl;
}
