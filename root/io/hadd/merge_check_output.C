#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <Compression.h>

#include <TSystem.h>

#include <string>
#include <utility>
#include <sstream>
#include <cstdint>
#include <cstring>

int merge_check_output(const char *fnameOut, const char *fnameIn1, const char *fnameIn2)
{
   using namespace ROOT::Experimental;

   auto noPrereleaseWarning = RLogScopedVerbosity(NTupleLog(), ROOT::Experimental::ELogLevel::kError);

   std::stringstream errs;
   {
      auto ntuple = RNTupleReader::Open("ntpl", fnameOut);
      auto viewI = ntuple->GetView<int>("I");
      auto viewL = ntuple->GetView<long>("L");
      if (auto v = viewI(0); v != 1337)
         errs << "Expected v to be 1337 but it's " << v << "\n";
      if (auto v = viewI(1); v != 123)
         errs << "Expected v to be 123 but it's " << v << "\n";
      if (auto v = viewL(0); v != 666)
         errs << "Expected v to be 666 but it's " << v << "\n";
      if (auto v = viewL(1); v != 420)
         errs << "Expected v to be 420 but it's " << v << "\n";
   }

   gSystem->Unlink(fnameOut);
   gSystem->Unlink(fnameIn1);
   gSystem->Unlink(fnameIn2);

   auto errsStr = errs.str();
   if (errsStr.length() > 0) {
      std::cerr << errsStr << "\n";
      return 1;
   }

   return 0;
}
