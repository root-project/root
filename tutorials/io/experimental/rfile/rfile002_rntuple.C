/// \file
/// \ingroup ROOT7 tutorial_io
/// Demonstrate how to read/write an RNTuple with RFile
///
/// \macro_code
/// \macro_output
///
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!
///
/// \date 2026-10-12
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>

void write_rntuple_to_rfile(const char *fileName)
{
   // Create a new ROOT file for writing
   auto file = ROOT::Experimental::RFile::Recreate(fileName);

   // Define the RNTuple schema
   auto model = ROOT::RNTupleModel::Create();
   auto pInt = model->MakeField<int>("intField");

   // Create an RNTupleWriter to append to the RFile
   auto writer = ROOT::Experimental::RNTupleWriter_Append(std::move(model), "ntuple", *file);

   // Fill the RNTuple with data
   for (int i = 0; i < 10; ++i) {
      *pInt = i;
      writer->Fill();
   }

   // `writer` will write the data to file upon destruction. Then `file` will write itself to disk in turn.
}

void read_rntuple_from_rfile(const char *fileName)
{
   // Open an existing ROOT file for reading (will throw an exception if `fileName` cannot be read).
   auto file = ROOT::Experimental::RFile::Open(fileName);

   // Get back the RNTuple anchor from the file
   auto rntuple = file->Get<ROOT::RNTuple>("ntuple");
   R__ASSERT(rntuple); // we just wrote the RNTuple to the file so we expect to find it

   // Create an RNTupleReader from the anchor
   auto reader = ROOT::RNTupleReader::Open(*rntuple);
   auto pInt = reader->GetModel().GetDefaultEntry().GetPtr<int>("intField");

   for (auto i : reader->GetEntryRange()) {
      reader->LoadEntry(i);
      std::cout << "value[" << i << "] = " << *pInt << "\n";
   }
}

void rfile002_rntuple()
{
   const char *const fileName = "rfile_rntuple.root";

   write_rntuple_to_rfile(fileName);
   read_rntuple_from_rfile(fileName);

   gSystem->Unlink(fileName);
}
