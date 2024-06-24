#include "ntuple_test.hxx"

#include <cstring> // for memset

void CreateCorruptedRNTuple(const std::string &uri)
{
   RNTupleWriteOptions options;
   options.SetCompression(0);

   auto model = RNTupleModel::Create();
   model->MakeField<float>("px", 1.0);
   model->MakeField<float>("py", 2.0);
   model->Freeze();
   auto modelClone = model->Clone(); // required later to write the corrupted version
   auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", uri, options);
   writer->Fill();
   writer.reset();

   // Load sealed pages to memory
   auto pageSource = RPageSource::Create("ntpl", uri);
   pageSource->Attach();
   auto descGuard = pageSource->GetSharedDescriptorGuard();
   const auto pxColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("px"), 0);
   const auto pyColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("py"), 0);
   const auto clusterId = descGuard->FindClusterId(pxColId, 0);
   RClusterIndex index{clusterId, 0};

   constexpr std::size_t bufSize = sizeof(float) + RPageStorage::kNBytesPageChecksum;
   unsigned char pxBuffer[bufSize];
   RPageStorage::RSealedPage pxSealedPage;
   pxSealedPage.SetBufferSize(bufSize);
   pxSealedPage.SetBuffer(pxBuffer);
   pageSource->LoadSealedPage(pxColId, index, pxSealedPage);
   EXPECT_EQ(bufSize, pxSealedPage.GetBufferSize());
   unsigned char pyBuffer[bufSize];
   RPageStorage::RSealedPage pySealedPage;
   pySealedPage.SetBufferSize(bufSize);
   pySealedPage.SetBuffer(pyBuffer);
   pageSource->LoadSealedPage(pyColId, index, pySealedPage);
   EXPECT_EQ(bufSize, pySealedPage.GetBufferSize());

   // Corrupt px sealed page's checksum
   memset(pxBuffer + sizeof(float), 0, RPageStorage::kNBytesPageChecksum);

   // Rewrite RNTuple with valid py page and corrupted px page
   auto pageSink = ROOT::Experimental::Internal::RPagePersistentSink::Create("ntpl", uri, options);
   pageSink->Init(*modelClone);
   pageSink->CommitSealedPage(pxColId, pxSealedPage);
   pageSink->CommitSealedPage(pyColId, pySealedPage);
   pageSink->CommitCluster(1);
   pageSink->CommitClusterGroup();
   pageSink->CommitDataset();
   modelClone.reset();
}
