#include "ntuple_test.hxx"

#include <ROOT/RConfig.hxx>
#ifndef R__BYTESWAP
#define IS_BIG_ENDIAN 1
#else
#define IS_BIG_ENDIAN 0
#endif

#if IS_BIG_ENDIAN
#include <Byteswap.h>
#endif

#include <xxhash.h>

#include <cstring> // for memset
#include <cstdio>

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
   RNTupleLocalIndex index{clusterId, 0};

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
   auto pageSink = ROOT::Internal::RPagePersistentSink::Create("ntpl", uri, options);
   pageSink->Init(*modelClone);
   pageSink->CommitSealedPage(pxColId, pxSealedPage);
   pageSink->CommitSealedPage(pyColId, pySealedPage);
   pageSink->CommitSealedPage(pzColId, pzSealedPage);
   pageSink->CommitCluster(1);
   pageSink->CommitClusterGroup();
   pageSink->CommitDataset();
   modelClone.reset();
}

void PatchRNTupleSection(std::string_view filePath, std::uint64_t sectionSeek, std::uint64_t sectionLen,
                         std::uint64_t patchedOffsetIntoSection, const std::byte *bytesToWrite,
                         std::size_t bytesToWriteLen, EEndianness sectionEndianness)
{
   // sanity check
   R__ASSERT(sectionLen > 6);
   R__ASSERT(patchedOffsetIntoSection + bytesToWriteLen <= sectionLen);

   FILE *file = fopen(std::string(filePath).c_str(), "r+b");

   fseek(file, sectionSeek + patchedOffsetIntoSection, SEEK_SET);
   std::size_t written = fwrite(bytesToWrite, 1, bytesToWriteLen, file);
   R__ASSERT(written == bytesToWriteLen);

   int err = fseek(file, sectionSeek, SEEK_SET);
   R__ASSERT(!err);

   // recompute checksum
   auto buf = MakeUninitArray<std::byte>(sectionLen);
   auto read = fread(buf.get(), 1, sectionLen, file);
   R__ASSERT(read == sectionLen);

   std::uint64_t checksum = XXH3_64bits(buf.get(), sectionLen);
   if ((sectionEndianness == EEndianness::BE) != IS_BIG_ENDIAN) {
      checksum = RByteSwap<8>::bswap(checksum);
   }
   // NOTE: we need to seek here to guarantee that the writing operation following the previous read will succeed
   // (see "File access flags" here https://en.cppreference.com/w/c/io/fopen).
   fseek(file, sectionSeek + sectionLen, SEEK_SET);
   fwrite(&checksum, 1, sizeof(checksum), file);
   fclose(file);
}
