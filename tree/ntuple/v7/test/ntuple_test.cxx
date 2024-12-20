#include "ntuple_test.hxx"

#include <cstring> // for memset

void CreateCorruptedRNTuple(const std::string &uri)
{
   RNTupleWriteOptions options;
   options.SetCompression(0);

   auto model = RNTupleModel::Create();
   *model->MakeField<float>("px") = 1.0;
   *model->MakeField<float>("py") = 2.0;
   *model->MakeField<float>("pz") = 3.0;
   model->Freeze();
   auto modelClone = model->Clone(); // required later to write the corrupted version
   auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", uri, options);
   writer->Fill();
   writer.reset();

   // Load sealed pages to memory
   auto pageSource = RPageSource::Create("ntpl", uri);
   pageSource->Attach();
   auto descGuard = pageSource->GetSharedDescriptorGuard();
   const auto pxColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("px"), 0, 0);
   const auto pyColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("py"), 0, 0);
   const auto pzColId = descGuard->FindPhysicalColumnId(descGuard->FindFieldId("pz"), 0, 0);
   const auto clusterId = descGuard->FindClusterId(pxColId, 0);
   RClusterIndex index{clusterId, 0};

   constexpr std::size_t bufSize = sizeof(float) + RPageStorage::kNBytesPageChecksum;
   unsigned char pxBuffer[bufSize];
   unsigned char pyBuffer[bufSize];
   unsigned char pzBuffer[bufSize];
   RPageStorage::RSealedPage pxSealedPage;
   RPageStorage::RSealedPage pySealedPage;
   RPageStorage::RSealedPage pzSealedPage;
   pxSealedPage.SetBufferSize(bufSize);
   pySealedPage.SetBufferSize(bufSize);
   pzSealedPage.SetBufferSize(bufSize);
   pxSealedPage.SetBuffer(pxBuffer);
   pySealedPage.SetBuffer(pyBuffer);
   pzSealedPage.SetBuffer(pzBuffer);
   pageSource->LoadSealedPage(pxColId, index, pxSealedPage);
   pageSource->LoadSealedPage(pyColId, index, pySealedPage);
   pageSource->LoadSealedPage(pzColId, index, pzSealedPage);
   EXPECT_EQ(bufSize, pxSealedPage.GetBufferSize());
   EXPECT_EQ(bufSize, pySealedPage.GetBufferSize());
   EXPECT_EQ(bufSize, pzSealedPage.GetBufferSize());

   // Corrupt px sealed page's checksum
   memset(pxBuffer + sizeof(float), 0, RPageStorage::kNBytesPageChecksum);

   // Corrupt py sealed page's data
   memset(pyBuffer, 0, sizeof(float));

   // Rewrite RNTuple with valid pz page and corrupted px, py page
   auto pageSink = ROOT::Experimental::Internal::RPagePersistentSink::Create("ntpl", uri, options);
   pageSink->Init(*modelClone);
   pageSink->CommitSealedPage(pxColId, pxSealedPage);
   pageSink->CommitSealedPage(pyColId, pySealedPage);
   pageSink->CommitSealedPage(pzColId, pzSealedPage);
   pageSink->CommitCluster(1);
   pageSink->CommitClusterGroup();
   pageSink->CommitDataset();
   modelClone.reset();
}
