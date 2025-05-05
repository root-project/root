/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Demonstrate the RNTupleProcessor for horizontal compositions (joins) of RNTuples
///
/// \macro_image
/// \macro_code
///
/// \date November 2024
/// \author The ROOT Team

// NOTE: The RNTupleProcessor and related classes are experimental at this point.
// Functionality and interface are still subject to changes.

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleProcessor.hxx>

#include <TCanvas.h>
#include <TH1F.h>
#include <TRandom.h>

// Import classes from the `Experimental` namespace for the time being.
using ROOT::Experimental::RNTupleOpenSpec;
using ROOT::Experimental::RNTupleProcessor;

const std::string kMainNTupleName = "mainNTuple";
const std::string kMainNTuplePath = "main_ntuple.root";
const std::string kAuxNTupleName = "auxNTuple";
const std::string kAuxNTuplePath = "aux_ntuple.root";

// Number of events to generate for the auxiliary ntuple. The main ntuple will have a fifth of this number.
constexpr int kNEvents = 10000;

void WriteMain(std::string_view ntupleName, std::string_view ntupleFileName)
{
   auto model = ROOT::RNTupleModel::Create();

   auto fldI = model->MakeField<std::uint32_t>("i");
   auto fldVpx = model->MakeField<float>("vpx");

   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), ntupleName, ntupleFileName);

   // The main ntuple only contains a subset of the entries present in the auxiliary ntuple.
   for (int i = 0; i < kNEvents; i += 5) {
      *fldI = i;
      *fldVpx = gRandom->Gaus();

      writer->Fill();
   }

   std::cout << "Wrote " << writer->GetNEntries() << " to the main RNTuple" << std::endl;
}

void WriteAux(std::string_view ntupleName, std::string_view ntupleFileName)
{
   auto model = ROOT::RNTupleModel::Create();

   auto fldI = model->MakeField<std::uint32_t>("i");
   auto fldVpy = model->MakeField<float>("vpy");

   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), ntupleName, ntupleFileName);

   for (int i = 0; i < kNEvents; ++i) {
      *fldI = i;
      *fldVpy = gRandom->Gaus();

      writer->Fill();
   }

   std::cout << "Wrote " << writer->GetNEntries() << " to the auxiliary RNTuple" << std::endl;
}

void Read()
{
   auto c = new TCanvas("c", "RNTupleJoinProcessor Example", 200, 10, 700, 500);
   TH1F hPy("h", "This is the px + py distribution", 100, -4, 4);
   hPy.SetFillColor(48);

   // The first specified ntuple is the main ntuple and will be used to drive the processor loop. The subsequent
   // list of ntuples (in this case, only one) are auxiliary and will be joined with the entries from the main ntuple.
   // We specify field "i" as the join field. This field, which should be present in all ntuples specified is used to
   // identify which entries belong together. Multiple join fields can be specified, in which case the combination of
   // field values is used. It is possible to specify up to 4 join fields. Providing an empty list of join fields
   // signals to the processor that all entries are aligned.
   auto processor =
      RNTupleProcessor::CreateJoin({kMainNTupleName, kMainNTuplePath}, {kAuxNTupleName, kAuxNTuplePath}, {"i"});

   float px, py;
   for (const auto &entry : *processor) {
      // Fields from the main ntuple are accessed by their original name.
      px = *entry.GetPtr<float>("vpx");
      // Fields from auxiliary ntuples are accessed by prepending the name of the auxiliary ntuple.
      py = *entry.GetPtr<float>(kAuxNTupleName + ".vpy");

      hPy.Fill(px + py);
   }

   std::cout << "Processed a total of " << processor->GetNEntriesProcessed() << " entries" << std::endl;

   hPy.DrawCopy();
}

void ntpl015_processor_join()
{
   WriteMain(kMainNTupleName, kMainNTuplePath);
   WriteAux(kAuxNTupleName, kAuxNTuplePath);

   Read();
}
