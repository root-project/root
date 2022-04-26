// Override endianness detection in RColumnElement.hxx; assume big-endian machine
#define R__LITTLE_ENDIAN 0

#include "gtest/gtest.h"

#include <ROOT/RCluster.hxx>
#include <ROOT/RColumn.hxx>
#include <ROOT/RColumnModel.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPageStorage.hxx>

#include <cstring>
#include <memory>
#include <vector>

using EColumnType = ROOT::Experimental::EColumnType;
using NTupleSize_t = ROOT::Experimental::NTupleSize_t;
using RCluster = ROOT::Experimental::Detail::RCluster;
using RColumnElementBase = ROOT::Experimental::Detail::RColumnElementBase;
using RNTupleDescriptor = ROOT::Experimental::RNTupleDescriptor;
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTupleLocator = ROOT::Experimental::RNTupleLocator;
using RPage = ROOT::Experimental::Detail::RPage;
using RPageSink = ROOT::Experimental::Detail::RPageSink;
using RPageSource = ROOT::Experimental::Detail::RPageSource;
using RPageStorage = ROOT::Experimental::Detail::RPageStorage;

namespace {
class RPageSinkMock : public RPageSink {
protected:
   const RColumnElementBase &fElement;
   std::vector<RPageStorage::RSealedPage> fPages;

   void CreateImpl(const RNTupleModel &, unsigned char *, std::uint32_t) final {}
   RNTupleLocator CommitSealedPageImpl(ROOT::Experimental::DescriptorId_t,
                                       const RPageStorage::RSealedPage &) final { return {}; }
   std::uint64_t CommitClusterImpl(NTupleSize_t) final { return 0; }
   RNTupleLocator CommitClusterGroupImpl(unsigned char *, std::uint32_t) final { return {}; }
   void CommitDatasetImpl(unsigned char *, std::uint32_t) final {}

   RPage ReservePage(ColumnHandle_t, std::size_t) final { return {}; }
   void ReleasePage(RPage &) final {}
public:
   RPageSinkMock(const RColumnElementBase &elt)
     : RPageSink("test", ROOT::Experimental::RNTupleWriteOptions()), fElement(elt)
   {
      fCompressor = std::make_unique<ROOT::Experimental::Detail::RNTupleCompressor>();
   }
   RNTupleLocator CommitPageImpl(ColumnHandle_t /*columnHandle*/, const RPage &page) final {
      auto P = SealPage(page, fElement, /*compressionSetting=*/0);
      fPages.emplace_back(P.fBuffer, P.fSize, P.fNElements);
      return {};
   }

   const std::vector<RPageStorage::RSealedPage> &GetPages() const { return fPages; }
};

/**
 * An RPageSource that can be constructed given the raw content of its pages and unpacks them on demand
 */
class RPageSourceMock : public RPageSource {
protected:
   const RColumnElementBase &fElement;
   const std::vector<RPageStorage::RSealedPage> &fPages;

   RNTupleDescriptor AttachImpl() final { return RNTupleDescriptor(); }
public:
   RPageSourceMock(const std::vector<RPageStorage::RSealedPage> &pages, const RColumnElementBase &elt)
     : RPageSource("test", ROOT::Experimental::RNTupleReadOptions()), fElement(elt), fPages(pages)
   {}
   std::unique_ptr<RPageSource> Clone() const final { return nullptr; }

   RPage PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t i) final {
      auto pageBuffer = RPageSource::UnsealPage(fPages[i], fElement);
      RPage newPage(columnHandle.fId, pageBuffer.release(), fElement.GetSize(), fPages[i].fNElements);
      newPage.GrowUnchecked(fPages[i].fNElements);
      return newPage;
   }
   RPage PopulatePage(ColumnHandle_t, const ROOT::Experimental::RClusterIndex &) final { return RPage(); }
   void ReleasePage(RPage &) final {}
   void LoadSealedPage(
      ROOT::Experimental::DescriptorId_t, const ROOT::Experimental::RClusterIndex &, RSealedPage &) final
   { }
   std::vector<std::unique_ptr<RCluster>> LoadClusters(std::span<RCluster::RKey>) final
   {
      return {};
   }
};
} // anonymous namespace

TEST(RColumnElementEndian, Float)
{
   ROOT::Experimental::Detail::RColumnElement<float, EColumnType::kReal32> element(nullptr);
   EXPECT_EQ(element.IsMappable(), false);

   RPageSinkMock sink(element);
   unsigned char buf1[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
   RPage page1(0, buf1, 4, 4);
   page1.GrowUnchecked(4);
   sink.CommitPageImpl(RPageStorage::ColumnHandle_t{}, page1);

   EXPECT_EQ(0, memcmp(sink.GetPages()[0].fBuffer,
                       "\x03\x02\x01\x00\x07\x06\x05\x04\x0b\x0a\x09\x08\x0f\x0e\x0d\x0c", 16));

   RPageSourceMock source(sink.GetPages(), element);
   auto page2 = source.PopulatePage(RPageStorage::ColumnHandle_t{}, NTupleSize_t{0});
   std::unique_ptr<unsigned char[]> buf2(static_cast<unsigned char *>(page2.GetBuffer())); // adopt buffer
   EXPECT_EQ(0, memcmp(buf2.get(),
                       "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 16));
}

TEST(RColumnElementEndian, Double)
{
   ROOT::Experimental::Detail::RColumnElement<double, EColumnType::kReal64> element(nullptr);
   EXPECT_EQ(element.IsMappable(), false);

   RPageSinkMock sink(element);
   unsigned char buf1[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
   RPage page1(0, buf1, 8, 2);
   page1.GrowUnchecked(2);
   sink.CommitPageImpl(RPageStorage::ColumnHandle_t{}, page1);

   EXPECT_EQ(0, memcmp(sink.GetPages()[0].fBuffer,
                       "\x07\x06\x05\x04\x03\x02\x01\x00\x0f\x0e\x0d\x0c\x0b\x0a\x09\x08", 16));

   RPageSourceMock source(sink.GetPages(), element);
   auto page2 = source.PopulatePage(RPageStorage::ColumnHandle_t{}, NTupleSize_t{0});
   std::unique_ptr<unsigned char[]> buf2(static_cast<unsigned char *>(page2.GetBuffer())); // adopt buffer
   EXPECT_EQ(0, memcmp(buf2.get(),
                       "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 16));
}

TEST(RColumnElementEndian, Int16)
{
   ROOT::Experimental::Detail::RColumnElement<std::int16_t, EColumnType::kInt16> element(nullptr);
   EXPECT_EQ(element.IsMappable(), false);

   RPageSinkMock sink(element);
   unsigned char buf1[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
   RPage page1(0, buf1, 2, 8);
   page1.GrowUnchecked(8);
   sink.CommitPageImpl(RPageStorage::ColumnHandle_t{}, page1);

   EXPECT_EQ(0, memcmp(sink.GetPages()[0].fBuffer,
                       "\x01\x00\x03\x02\x05\x04\x07\x06\x09\x08\x0b\x0a\x0d\x0c\x0f\x0e", 16));

   RPageSourceMock source(sink.GetPages(), element);
   auto page2 = source.PopulatePage(RPageStorage::ColumnHandle_t{}, NTupleSize_t{0});
   std::unique_ptr<unsigned char[]> buf2(static_cast<unsigned char *>(page2.GetBuffer())); // adopt buffer
   EXPECT_EQ(0, memcmp(buf2.get(),
                       "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 16));
}

TEST(RColumnElementEndian, UInt16)
{
   ROOT::Experimental::Detail::RColumnElement<std::uint16_t, EColumnType::kInt16> element(nullptr);
   EXPECT_EQ(element.IsMappable(), false);

   RPageSinkMock sink(element);
   unsigned char buf1[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
   RPage page1(0, buf1, 2, 8);
   page1.GrowUnchecked(8);
   sink.CommitPageImpl(RPageStorage::ColumnHandle_t{}, page1);

   EXPECT_EQ(0, memcmp(sink.GetPages()[0].fBuffer,
                       "\x01\x00\x03\x02\x05\x04\x07\x06\x09\x08\x0b\x0a\x0d\x0c\x0f\x0e", 16));

   RPageSourceMock source(sink.GetPages(), element);
   auto page2 = source.PopulatePage(RPageStorage::ColumnHandle_t{}, NTupleSize_t{0});
   std::unique_ptr<unsigned char[]> buf2(static_cast<unsigned char *>(page2.GetBuffer())); // adopt buffer
   EXPECT_EQ(0, memcmp(buf2.get(),
                       "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 16));
}

TEST(RColumnElementEndian, Int32)
{
   ROOT::Experimental::Detail::RColumnElement<std::int32_t, EColumnType::kInt32> element(nullptr);
   EXPECT_EQ(element.IsMappable(), false);

   RPageSinkMock sink(element);
   unsigned char buf1[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
   RPage page1(0, buf1, 4, 4);
   page1.GrowUnchecked(4);
   sink.CommitPageImpl(RPageStorage::ColumnHandle_t{}, page1);

   EXPECT_EQ(0, memcmp(sink.GetPages()[0].fBuffer,
                       "\x03\x02\x01\x00\x07\x06\x05\x04\x0b\x0a\x09\x08\x0f\x0e\x0d\x0c", 16));

   RPageSourceMock source(sink.GetPages(), element);
   auto page2 = source.PopulatePage(RPageStorage::ColumnHandle_t{}, NTupleSize_t{0});
   std::unique_ptr<unsigned char[]> buf2(static_cast<unsigned char *>(page2.GetBuffer())); // adopt buffer
   EXPECT_EQ(0, memcmp(buf2.get(),
                       "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 16));
}

TEST(RColumnElementEndian, UInt32)
{
   ROOT::Experimental::Detail::RColumnElement<std::uint32_t, EColumnType::kInt32> element(nullptr);
   EXPECT_EQ(element.IsMappable(), false);

   RPageSinkMock sink(element);
   unsigned char buf1[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
   RPage page1(0, buf1, 4, 4);
   page1.GrowUnchecked(4);
   sink.CommitPageImpl(RPageStorage::ColumnHandle_t{}, page1);

   EXPECT_EQ(0, memcmp(sink.GetPages()[0].fBuffer,
                       "\x03\x02\x01\x00\x07\x06\x05\x04\x0b\x0a\x09\x08\x0f\x0e\x0d\x0c", 16));

   RPageSourceMock source(sink.GetPages(), element);
   auto page2 = source.PopulatePage(RPageStorage::ColumnHandle_t{}, NTupleSize_t{0});
   std::unique_ptr<unsigned char[]> buf2(static_cast<unsigned char *>(page2.GetBuffer())); // adopt buffer
   EXPECT_EQ(0, memcmp(buf2.get(),
                       "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 16));
}

TEST(RColumnElementEndian, Int64)
{
   ROOT::Experimental::Detail::RColumnElement<std::int64_t, EColumnType::kInt64> element(nullptr);
   EXPECT_EQ(element.IsMappable(), false);

   RPageSinkMock sink(element);
   unsigned char buf1[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
   RPage page1(0, buf1, 8, 2);
   page1.GrowUnchecked(2);
   sink.CommitPageImpl(RPageStorage::ColumnHandle_t{}, page1);

   EXPECT_EQ(0, memcmp(sink.GetPages()[0].fBuffer,
                       "\x07\x06\x05\x04\x03\x02\x01\x00\x0f\x0e\x0d\x0c\x0b\x0a\x09\x08", 16));

   RPageSourceMock source(sink.GetPages(), element);
   auto page2 = source.PopulatePage(RPageStorage::ColumnHandle_t{}, NTupleSize_t{0});
   std::unique_ptr<unsigned char[]> buf2(static_cast<unsigned char *>(page2.GetBuffer())); // adopt buffer
   EXPECT_EQ(0, memcmp(buf2.get(),
                       "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 16));
}

TEST(RColumnElementEndian, UInt64)
{
   ROOT::Experimental::Detail::RColumnElement<std::uint64_t, EColumnType::kInt64> element(nullptr);
   EXPECT_EQ(element.IsMappable(), false);

   RPageSinkMock sink(element);
   unsigned char buf1[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
   RPage page1(0, buf1, 8, 2);
   page1.GrowUnchecked(2);
   sink.CommitPageImpl(RPageStorage::ColumnHandle_t{}, page1);

   EXPECT_EQ(0, memcmp(sink.GetPages()[0].fBuffer,
                       "\x07\x06\x05\x04\x03\x02\x01\x00\x0f\x0e\x0d\x0c\x0b\x0a\x09\x08", 16));

   RPageSourceMock source(sink.GetPages(), element);
   auto page2 = source.PopulatePage(RPageStorage::ColumnHandle_t{}, NTupleSize_t{0});
   std::unique_ptr<unsigned char[]> buf2(static_cast<unsigned char *>(page2.GetBuffer())); // adopt buffer
   EXPECT_EQ(0, memcmp(buf2.get(),
                       "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 16));
}

TEST(RColumnElementEndian, Int64Int32)
{
   ROOT::Experimental::Detail::RColumnElement<std::int64_t, EColumnType::kInt32> element(nullptr);
   EXPECT_EQ(element.IsMappable(), false);

   RPageSinkMock sink(element);
   std::int64_t buf1[] = {0x0a0b0c0d11223344, 0x0a0b0c0d55667788};
   RPage page1(0, buf1, 8, 2);
   page1.GrowUnchecked(2);
   sink.CommitPageImpl(RPageStorage::ColumnHandle_t{}, page1);

   EXPECT_EQ(0, memcmp(sink.GetPages()[0].fBuffer,
                       "\x44\x33\x22\x11\x88\x77\x66\x55", 8));

   RPageSourceMock source(sink.GetPages(), element);
   auto page2 = source.PopulatePage(RPageStorage::ColumnHandle_t{}, NTupleSize_t{0});
   std::unique_ptr<unsigned char[]> buf2(static_cast<unsigned char *>(page2.GetBuffer())); // adopt buffer
   auto ints = reinterpret_cast<std::int64_t *>(buf2.get());
   EXPECT_EQ(ints[0], 0x0000000011223344);
   EXPECT_EQ(ints[1], 0x0000000055667788);
}
