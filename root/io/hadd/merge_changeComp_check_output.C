#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <Compression.h>
#include <RZip.h>

#include <TSystem.h>

int merge_changeComp_check_output(int expectedCompression, const char *fnameOut, const char *fnameIn1,
                                  const char *fnameIn2)
{
   using namespace ROOT::Experimental;

   // Check compression of the tfile
   {
      auto f1 = std::unique_ptr<TFile>(TFile::Open(fnameOut, "READ"));
      if (f1->GetCompressionSettings() != expectedCompression) {
         std::cerr << "Expected TFile compression to be " << expectedCompression << " but it is "
                   << f1->GetCompressionSettings() << "\n";
         return 1;
      }
   }

   Internal::RPageSourceFile source("ntpl", fnameOut, RNTupleReadOptions());
   source.Attach();

   Internal::RClusterPool pool{source};

   const auto expCompAlgo = ROOT::RCompressionSetting::AlgorithmFromCompressionSettings(expectedCompression);
   const auto &desc = source.GetSharedDescriptorGuard();
   auto clusterIter = desc->GetClusterIterable();
   for (const auto &clusterDesc : clusterIter) {
      // check advertised compression
      int advertisedCompression = clusterDesc.GetColumnRange(0).fCompressionSettings;
      if (advertisedCompression != expectedCompression) {
         std::cerr << "Expected advertised compression to be " << expectedCompression << " but it is "
                   << advertisedCompression << "\n";
         return 1;
      }

      // check actual compression
      for (const auto &column : desc->GetColumnIterable()) {
         const auto &pages = clusterDesc.GetPageRange(column.GetLogicalId());
         std::uint64_t pageIdx = 0;
         for (const auto &pageInfo : pages.fPageInfos) {
            auto cluster = pool.GetCluster(clusterDesc.GetId(), {column.GetPhysicalId()});
            Internal::ROnDiskPage::Key key{column.GetPhysicalId(), pageIdx};
            auto onDiskPage = cluster->GetOnDiskPage(key);
            R__ASSERT(onDiskPage);
            const auto actualCompAlgo =
               R__getCompressionAlgorithm((const unsigned char *)onDiskPage->GetAddress(), onDiskPage->GetSize());
            if (actualCompAlgo != expCompAlgo) {
               std::cerr << "Expected actual compression to be "
                         << ROOT::RCompressionSetting::AlgorithmToString(expCompAlgo) << " but it is "
                         << ROOT::RCompressionSetting::AlgorithmToString(actualCompAlgo) << "\n";
               return 1;
            }
            ++pageIdx;
         }
      }
   }

   gSystem->Unlink(fnameOut);
   gSystem->Unlink(fnameIn1);
   gSystem->Unlink(fnameIn2);

   return 0;
}
