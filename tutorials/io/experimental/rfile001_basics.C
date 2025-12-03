/// \file
/// \ingroup ROOT7 tutorial_io
/// Demonstrate the basic usage of RFile.
///
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-11-06
/// \macro_code
/// \macro_output
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

void write_hist_to_rfile(const char *fileName)
{
   // Create a histogram to write to the file
   TH1D hist("hist", "hist", 10, 0, 100);
   hist.FillRandom("gaus", 1000);

   // Create a new ROOT file for writing
   auto file = ROOT::Experimental::RFile::Recreate(fileName);

   // Put objects into the file (in this case we write the same object multiple times
   // under different paths). Note that the ownership of `hist` is untouched by `file->Put`.
   file->Put(hist.GetName(), hist);
   file->Put(std::string("a/") + hist.GetName(), hist);
   file->Put(std::string("a/b/") + hist.GetName(), hist);

   // When `file` goes out of scope it will write itself to disk.
   // To manually write the file to disk without closing it, one can use `file->Flush()`.
}

void read_hist_from_rfile(const char *fileName)
{
   // Open an existing ROOT file for reading (will throw an exception if `fileName` cannot be read).
   auto file = ROOT::Experimental::RFile::Open(fileName);
   // Iterate all keys of all objects in the file (this excludes directories by default - see the documentation of
   // ListKeys() for all the options).
   for (auto key : file->ListKeys()) {
      // Retrieve the objects from the file. `file->Get` will return a `std::unique_ptr` to the object, or `nullptr`
      // if the object isn't there.
      // Once an object is retrieved, it is fully owned by the application, so it survives even if `file` is closed.
      auto hist = file->Get<TH1D>(key.GetPath());
      if (!hist)
         continue;
      std::cout << key.GetClassName() << " at " << key.GetPath() << ';' << key.GetCycle() << ":\n";
      std::cout << "  entries: " << hist->GetEntries() << "\n";
   }
}

void rfile001_basics()
{
   const char *const fileName = "rfile_basics.root";

   write_hist_to_rfile(fileName);
   read_hist_from_rfile(fileName);

   gSystem->Unlink(fileName);
}
