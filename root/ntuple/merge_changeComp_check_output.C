#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <Compression.h>

#include <TSystem.h>

#include <string>
#include <utility>

int merge_changeComp_check_output(int expectedCompression, const char *fnameOut, const char *fnameIn1, const char *fnameIn2)
{
   using namespace ROOT::Experimental;
   
   auto noPrereleaseWarning = RLogScopedVerbosity(NTupleLog(), ROOT::Experimental::ELogLevel::kError);

   Internal::RPageSourceFile source("ntpl", fnameOut, RNTupleReadOptions());
   source.Attach();

   // print out actual compression
   const auto &desc = source.GetSharedDescriptorGuard();
   auto clusterIter = desc->GetClusterIterable();
   for (const auto &cluster : clusterIter) {
      const auto &cDesc = desc->GetClusterDescriptor(cluster.GetId());
      int realCompression = cDesc.GetColumnRange(0).fCompressionSettings;
      if (realCompression != expectedCompression) {
         std::cerr << "Expected compression to be " << expectedCompression << " but it is " << realCompression << "\n";
         return 1;
      }
   }

   gSystem->Unlink(fnameOut);
   gSystem->Unlink(fnameIn1);
   gSystem->Unlink(fnameIn2);

   return 0;
}
