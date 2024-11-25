// Override endianness detection in RColumnElement.hxx; assume big-endian machine
// These tests are simulating a big endian machine; we will turn them off on an actual big endian node.
#define R__LITTLE_ENDIAN 0
#include "../src/RColumnElement.hxx"

#include "gtest/gtest.h"

#include <ROOT/RCluster.hxx>
#include <ROOT/RColumn.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleReadOptions.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleUtil.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPageStorage.hxx>

#include <cstring>
#include <memory>
#include <vector>

using ROOT::Experimental::EColumnType;
using ROOT::Experimental::NTupleSize_t;
using ROOT::Experimental::RNTupleDescriptor;
using ROOT::Experimental::RNTupleModel;
using ROOT::Experimental::Internal::RCluster;
using ROOT::Experimental::Internal::RColumnElementBase;
using ROOT::Experimental::Internal::RPage;
using ROOT::Experimental::Internal::RPageRef;
using ROOT::Experimental::Internal::RPageSink;
using ROOT::Experimental::Internal::RPageSource;
using ROOT::Experimental::Internal::RPageStorage;

namespace {
class RPageSinkMock : public RPageSink {
protected:
   const RColumnElementBase &fElement;
   std::vector<RPageStorage::RSealedPage> fPages;

   ColumnHandle_t AddColumn(ROOT::Experimental::DescriptorId_t, ROOT::Experimental::Internal::RColumn &) final
   {
      return {};
   }

   const RNTupleDescriptor &GetDescriptor() const final
   {
      static RNTupleDescriptor descriptor;
      return descriptor;
   }

   void InitImpl(RNTupleModel &) final {}
   void UpdateSchema(const ROOT::Experimental::Internal::RNTupleModelChangeset &, NTupleSize_t) final {}
   void UpdateExtraTypeInfo(const ROOT::Experimental::RExtraTypeInfoDescriptor &) final {}
   void CommitSuppressedColumn(ColumnHandle_t) final {}
   void CommitSealedPage(ROOT::Experimental::DescriptorId_t, const RPageStorage::RSealedPage &) final {}
   void CommitSealedPageV(std::span<RPageStorage::RSealedPageGroup>) final {}
   RStagedCluster StageCluster(NTupleSize_t) final { return {}; }
   void CommitStagedClusters(std::span<RStagedCluster>) final {}
   void CommitClusterGroup() final {}
   void CommitDatasetImpl() final {}

public:
   RPageSinkMock(const RColumnElementBase &elt)
      : RPageSink("test", ROOT::Experimental::RNTupleWriteOptions()), fElement(elt)
   {
      fOptions->SetEnablePageChecksums(false);
      fCompressor = std::make_unique<ROOT::Experimental::Internal::RNTupleCompressor>();
   }
   void CommitPage(ColumnHandle_t /*columnHandle*/, const RPage &page) final
   {
      fPages.emplace_back(SealPage(page, fElement));
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

   void LoadStructureImpl() final {}
   RNTupleDescriptor AttachImpl() final { return RNTupleDescriptor(); }
   std::unique_ptr<RPageSource> CloneImpl() const final { return nullptr; }
   RPageRef LoadPageImpl(ColumnHandle_t, const RClusterInfo &, ROOT::Experimental::ClusterSize_t::ValueType) final
   {
      return RPageRef();
   }

public:
   RPageSourceMock(const std::vector<RPageStorage::RSealedPage> &pages, const RColumnElementBase &elt)
      : RPageSource("test", ROOT::Experimental::RNTupleReadOptions()), fElement(elt), fPages(pages)
   {
   }

   RPageRef LoadPage(ColumnHandle_t columnHandle, NTupleSize_t i) final
   {
      auto page = RPageSource::UnsealPage(fPages[i], fElement).Unwrap();
      ROOT::Experimental::Internal::RPagePool::RKey key{columnHandle.fPhysicalId, std::type_index(typeid(void))};
      return fPagePool.RegisterPage(std::move(page), key);
   }
   RPageRef LoadPage(ColumnHandle_t, ROOT::Experimental::RClusterIndex) final { return RPageRef(); }
   void LoadSealedPage(ROOT::Experimental::DescriptorId_t, ROOT::Experimental::RClusterIndex, RSealedPage &) final {}
   std::vector<std::unique_ptr<RCluster>> LoadClusters(std::span<RCluster::RKey>) final { return {}; }
};
} // anonymous namespace

TEST(RColumnElementEndian, ByteCopy)
{
#ifndef R__BYTESWAP
   GTEST_SKIP() << "Skipping test on big endian node";
#else
   RColumnElement<float, EColumnType::kReal32> element;
   EXPECT_EQ(element.IsMappable(), false);

   RPageSinkMock sink1(element);
   unsigned char buf1[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                           0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
   RPage page1(buf1, nullptr, 4, 4);
   page1.GrowUnchecked(4);
   sink1.CommitPage(RPageStorage::ColumnHandle_t{}, page1);

   EXPECT_EQ(0, memcmp(sink1.GetPages()[0].GetBuffer(),
                       "\x03\x02\x01\x00\x07\x06\x05\x04\x0b\x0a\x09\x08\x0f\x0e\x0d\x0c", 16));

   RPageSourceMock source1(sink1.GetPages(), element);
   auto page2Ref = source1.LoadPage(RPageStorage::ColumnHandle_t{}, NTupleSize_t{0});
   EXPECT_EQ(
      0, memcmp(page2Ref.Get().GetBuffer(), "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 16));
#endif
}

TEST(RColumnElementEndian, Cast)
{
#ifndef R__BYTESWAP
   GTEST_SKIP() << "Skipping test on big endian node";
#else
   RColumnElement<std::int64_t, EColumnType::kInt32> element;
   EXPECT_EQ(element.IsMappable(), false);

   RPageSinkMock sink1(element);
   unsigned char buf1[] = {0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                           0x07, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x08, 0x09,
                           0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x0c, 0x0d, 0x0e, 0x0f};
   RPage page1(buf1, nullptr, 8, 4);
   page1.GrowUnchecked(4);
   sink1.CommitPage(RPageStorage::ColumnHandle_t{}, page1);

   EXPECT_EQ(0, memcmp(sink1.GetPages()[0].GetBuffer(),
                       "\x03\x02\x01\x00\x07\x06\x05\x04\x0b\x0a\x09\x08\x0f\x0e\x0d\x0c", 16));

   RPageSourceMock source1(sink1.GetPages(), element);
   auto page2Ref = source1.LoadPage(RPageStorage::ColumnHandle_t{}, NTupleSize_t{0});
   EXPECT_EQ(0, memcmp(page2Ref.Get().GetBuffer(),
                       "\x00\x01\x02\x03\x00\x00\x00\x00\x04\x05\x06\x07\x00\x00\x00\x00"
                       "\x08\x09\x0a\x0b\x00\x00\x00\x00\x0c\x0d\x0e\x0f\x00\x00\x00\x00",
                       32));
#endif
}

TEST(RColumnElementEndian, Split)
{
#ifndef R__BYTESWAP
   GTEST_SKIP() << "Skipping test on big endian node";
#else
   RColumnElement<double, EColumnType::kSplitReal64> splitElement;

   RPageSinkMock sink1(splitElement);
   unsigned char buf1[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                           0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
   RPage page1(buf1, nullptr, 8, 2);
   page1.GrowUnchecked(2);
   sink1.CommitPage(RPageStorage::ColumnHandle_t{}, page1);

   EXPECT_EQ(0, memcmp(sink1.GetPages()[0].GetBuffer(),
                       "\x07\x0f\x06\x0e\x05\x0d\x04\x0c\x03\x0b\x02\x0a\x01\x09\x00\x08", 16));

   RPageSourceMock source1(sink1.GetPages(), splitElement);
   auto page2Ref = source1.LoadPage(RPageStorage::ColumnHandle_t{}, NTupleSize_t{0});
   EXPECT_EQ(
      0, memcmp(page2Ref.Get().GetBuffer(), "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f", 16));
#endif
}

TEST(RColumnElementEndian, DeltaSplit)
{
#ifndef R__BYTESWAP
   GTEST_SKIP() << "Skipping test on big endian node";
#else
   using ClusterSize_t = ROOT::Experimental::ClusterSize_t;

   RColumnElement<ClusterSize_t, EColumnType::kSplitIndex32> element;
   EXPECT_EQ(element.IsMappable(), false);

   RPageSinkMock sink1(element);
   unsigned char buf1[] = {0x00, 0x01, 0x02, 0x03, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
                           0x07, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x08, 0x09,
                           0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x0c, 0x0d, 0x0e, 0x0f};
   RPage page1(buf1, nullptr, 8, 4);
   page1.GrowUnchecked(4);
   sink1.CommitPage(RPageStorage::ColumnHandle_t{}, page1);

   EXPECT_EQ(0, memcmp(sink1.GetPages()[0].GetBuffer(),
                       "\x03\x04\x04\x04\x02\x04\x04\x04\x01\x04\x04\x04\x00\x04\x04\x04", 16));

   RPageSourceMock source1(sink1.GetPages(), element);
   auto page2Ref = source1.LoadPage(RPageStorage::ColumnHandle_t{}, NTupleSize_t{0});
   EXPECT_EQ(0, memcmp(page2Ref.Get().GetBuffer(),
                       "\x00\x01\x02\x03\x00\x00\x00\x00\x04\x05\x06\x07\x00\x00\x00\x00"
                       "\x08\x09\x0a\x0b\x00\x00\x00\x00\x0c\x0d\x0e\x0f\x00\x00\x00\x00",
                       32));
#endif
}

TEST(RColumnElementEndian, Real32Trunc)
{
#ifndef R__BYTESWAP
   GTEST_SKIP() << "Skipping test on big endian node";
#else
   RColumnElement<float, EColumnType::kReal32Trunc> element;
   element.SetBitsOnStorage(12);

   RPageSinkMock sink1(element);
   unsigned char buf1[] = {0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                           0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f};
   RPage page1(buf1, nullptr, sizeof(float), 4);
   page1.GrowUnchecked(4);
   sink1.CommitPage(RPageStorage::ColumnHandle_t{}, page1);

   EXPECT_EQ(0, memcmp(sink1.GetPages()[0].GetBuffer(), "\x00\x00\x04\x80\x00\x0c", 6));

   RPageSourceMock source1(sink1.GetPages(), element);
   auto page2Ref = source1.LoadPage(RPageStorage::ColumnHandle_t{}, NTupleSize_t{0});
   EXPECT_EQ(
      0, memcmp(page2Ref.Get().GetBuffer(), "\x00\x00\x00\x00\x04\x00\x00\x00\x08\x00\x00\x00\x0c\x00\x00\x00", 16));
#endif
}
