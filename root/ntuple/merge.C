#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>

#include <TSystem.h>

#include <string>
#include <utility>
#include <iomanip>

void merge() {
  using ROOT::Experimental::RLogScopedVerbosity;
  using ROOT::Experimental::RNTupleModel;
  using ROOT::Experimental::RNTupleReader;
  using ROOT::Experimental::RNTupleWriter;

  const std::string kFileName1{"test_rntuple_merge1.root"s};
  const std::string kFileName2{"test_rntuple_merge2.root"s};
  const std::string kFileNameMerged{"test_ntuple_merged.root"s};

  auto noPrereleaseWarning = RLogScopedVerbosity(
      ROOT::Experimental::NTupleLog(), ROOT::Experimental::ELogLevel::kError);

  {
    auto model = RNTupleModel::Create();
    model->MakeField<int>("I", 1337);
    model->MakeField<float>("F", 666.f);
    auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", kFileName1);
    writer->Fill();
  }
  {
    auto model = RNTupleModel::Create();
    model->MakeField<int>("I", 123);
    model->MakeField<float>("F", 420.f);
    auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", kFileName2);
    writer->Fill();
  }

  std::string cmdLine =
      "hadd " + kFileNameMerged + " " + kFileName1 + " " + kFileName2;
  gSystem->Exec(cmdLine.c_str());

  {
    auto ntuple = RNTupleReader::Open("ntpl", kFileNameMerged);
    auto viewI = ntuple->GetView<int>("I");
    auto viewF = ntuple->GetView<float>("F");
    std::cout << viewI(0) << " " << viewI(1) << " " << std::setprecision(3)
              << viewF(0) << " " << viewF(1) << "\n";
  }

  gSystem->Unlink(kFileName1.c_str());
  gSystem->Unlink(kFileName2.c_str());
  gSystem->Unlink(kFileNameMerged.c_str());
}
