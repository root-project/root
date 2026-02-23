/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Example creating low-precision floating point fields in RNTuple
///
/// RNTuple supports storing floating points on disk with less precision than their in-memory representation.
/// Under the right circumstances, this can in save storage space while not significantly altering the results
/// of an analysis.
///
/// Storing low-precision floats is done by setting their column representation to one of the dedicated column types:
/// - Real16 (half-precision IEEE754 fp),
/// - Real32Trunc (single-precision IEEE754 with truncated mantissa, using from 10 to 31 bits in total) and
/// - Real32Quant (floating point within a specified range, represented as an integer with N bits of precision in a
///   linear quantized form).
///
/// To use these column types in RNTuple, one creates a RField<float> or RField<double> and sets its desired column
/// representation by calling, respectively:
/// - RField<float>::SetHalfPrecision()  (for Real16)
/// - RField<float>::SetTruncated()      (for Real32Trunc)
/// - RField<float>::SetQuantized()      (for Real32Quant)
///
/// Other than these, one can also setup the field to use the ROOT `Double32_t` type, either via
/// RField<double>::SetDouble32() or by directly creating one such field via RFieldBase::Create("f", "Double32_t").
///
/// \macro_image
/// \macro_code
///
/// \date February 2026
/// \author The ROOT Team

static constexpr char const *kNTupleName = "ntpl";
static constexpr char const *kNTupleFileName = "ntpl018_low_precision_floats.root";
static constexpr int kNEvents = 50;

struct Event {
   std::vector<float> fPt;
   std::vector<double> fE;
};

static void Write()
{
   auto model = ROOT::RNTupleModel::Create();

   // Create 3 float fields: one backed by a Real16 column, one backed by a Real32Trunc column
   // and one backed by a Real32Quant column.
   // Since we need to call methods on the RField objects in order to make them into our specific column types,
   // we don't use MakeField but rather we explicitly create the RFields and then use AddField on the model.
   {
      auto fieldReal16 = std::make_unique<ROOT::RField<float>>("myReal16");
      fieldReal16->SetHalfPrecision(); // this is now a Real16-backed float field
      model->AddField(std::move(fieldReal16));
   }
   {
      auto fieldReal32Trunc = std::make_unique<ROOT::RField<float>>("myReal32Trunc");
      // Let's say we want 20 bits of precision. This means that this float's mantissa will be truncated to (20 - 9) =
      // 11 bits.
      fieldReal32Trunc->SetTruncated(20);
      model->AddField(std::move(fieldReal32Trunc));
   }
   {
      auto fieldReal32Quant = std::make_unique<ROOT::RField<float>>("myReal32Quant");
      // Declare that this field will never store values outside of the [-1, 1] range (this will be checked dynamically)
      // and that we want to dedicate 24 bits to this number on disk.
      fieldReal32Quant->SetQuantized(-1., 1., 24);
      model->AddField(std::move(fieldReal32Quant));
   }

   // We can also change the column type of a struct/class subfield:
   {
      auto fieldEvents = std::make_unique<ROOT::RField<Event>>("myEvents");
      // Note that we iterate over `*fieldEvents`, not over fieldEvents->GetMutableSubfields(), as the latter won't
      // recurse into fieldEvents's grandchildren. By iterating over the field itself we are sure to visit the entire
      // field hierarchy, including the fields we need to change.
      // The hierarchy of fieldEvents is like this:
      //
      //     myEvents: RField<Event>
      //        fPt:   RField<vector<float>>
      //           _0: RField<float>           <-- we need to change this
      //        fE:    RField<vector<double>
      //           _0: RField<double>          <-- we need to change this
      //
      for (auto &field : *fieldEvents) {
         if (auto *fldDouble = dynamic_cast<ROOT::RField<double> *>(&field)) {
            std::cout << "Setting field " << field.GetQualifiedFieldName() << " to truncated.\n";
            fldDouble->SetTruncated(16);
         } else if (auto *fldFloat = dynamic_cast<ROOT::RField<float> *>(&field)) {
            std::cout << "Setting field " << field.GetQualifiedFieldName() << " to truncated.\n";
            fldFloat->SetTruncated(16);
         }
      }
      model->AddField(std::move(fieldEvents));
   }

   // Get the pointers to the fields we just added:
   const auto &entry = model->GetDefaultEntry();
   auto myReal16 = entry.GetPtr<float>("myReal16");
   auto myReal32Trunc = entry.GetPtr<float>("myReal32Trunc");
   auto myReal32Quant = entry.GetPtr<float>("myReal32Quant");
   auto myEvents = entry.GetPtr<Event>("myEvents");

   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), kNTupleName, kNTupleFileName);

   // fill our entries
   gRandom->SetSeed();
   for (int i = 0; i < kNEvents; i++) {
      *myReal16 = gRandom->Rndm();
      *myReal32Trunc = gRandom->Rndm();
      *myReal32Quant = gRandom->Rndm();
      myEvents->fPt.push_back(i);
      myEvents->fE.push_back(i);
      writer->Fill();
   }
}

static void Read()
{
   auto reader = ROOT::RNTupleReader::Open(kNTupleName, kNTupleFileName);

   // We can read back our fields as regular floats. We can also read them as double if we impose our own model when
   // creating the reader.
   const auto &entry = reader->GetModel().GetDefaultEntry();
   auto myReal16 = entry.GetPtr<float>("myReal16");
   auto myReal32Trunc = entry.GetPtr<float>("myReal32Trunc");
   auto myReal32Quant = entry.GetPtr<float>("myReal32Quant");
   auto myEvents = entry.GetPtr<Event>("myEvents");

   for (auto idx : reader->GetEntryRange()) {
      reader->LoadEntry(idx);

      float eventsAvgPt = 0.f;
      for (float pt : myEvents->fPt)
         eventsAvgPt += pt;
      eventsAvgPt /= myEvents->fPt.size();
      double eventsAvgE = 0.f;
      for (double e : myEvents->fE)
         eventsAvgE += e;
      eventsAvgE /= myEvents->fE.size();

      std::cout << "[" << idx << "] Real16: " << *myReal16 << ", Real32Trunc: " << *myReal32Trunc
                << ", Real32Quant: " << *myReal32Quant << ", Events avg pt: " << eventsAvgPt << ", E: " << eventsAvgE
                << "\n";
   }
}

void ntpl018_low_precision_floats()
{
   Write();
   Read();
}
