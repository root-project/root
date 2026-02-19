/// \file
/// \ingroup tutorial_ntuple
/// \notebook
/// Example creating low-precision floating point fields in RNTuple
///
/// RNTuple supports 3 kinds of low-precision floating point column types:
/// Real16 (half-precision IEEE754 fp),
/// Real32Trunc (single-precision IEEE754 with truncated mantissa, using from 10 to 31 bits in total) and
/// Real32Quant (floating point within a specified range, represented as an integer with N bits of precision in a
/// linear quantized form).
///
/// To use these column types in RNTuple, one creates a RField<float> and sets its desired column representation by
/// calling, respectively:
/// - RField<float>::SetHalfPrecision()  (for Real16)
/// - RField<float>::SetTruncated()      (for Real32Trunc)
/// - RField<float>::SetQuantized()      (for Real32Quant)
///
/// \macro_image
/// \macro_code
///
/// \date February 2026
/// \author The ROOT Team

static constexpr char const *kNTupleName = "ntpl";
static constexpr char const *kNTupleFileName = "ntpl018_low_precision_floats.root";
static constexpr int kNEvents = 50;

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

   // Get the pointers to the fields we just added:
   const auto &entry = model->GetDefaultEntry();
   auto myReal16 = entry.GetPtr<float>("myReal16");
   auto myReal32Trunc = entry.GetPtr<float>("myReal32Trunc");
   auto myReal32Quant = entry.GetPtr<float>("myReal32Quant");

   auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), kNTupleName, kNTupleFileName);

   // fill our entries
   gRandom->SetSeed();
   for (int i = 0; i < kNEvents; i++) {
      *myReal16 = gRandom->Rndm();
      *myReal32Trunc = gRandom->Rndm();
      *myReal32Quant = gRandom->Rndm();
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

   for (auto idx : reader->GetEntryRange()) {
      reader->LoadEntry(idx);
      std::cout << "[0] Real16: " << *myReal16 << ", Real32Trunc: " << *myReal32Trunc
                << ", Real32Quant: " << *myReal32Quant << "\n";
   }
}

void ntpl018_low_precision_floats()
{
   Write();
   Read();
}
