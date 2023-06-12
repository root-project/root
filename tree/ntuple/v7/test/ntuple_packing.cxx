#include "ntuple_test.hxx"

#include <array>
#include <cmath>
#include <cstring> // for memcmp
#include <limits>
#include <type_traits>
#include <utility>

template <typename PodT, typename NarrowT, ROOT::Experimental::EColumnType ColumnT>
struct Helper {
   using Pod_t = PodT;
   using Narrow_t = NarrowT;
   static constexpr ROOT::Experimental::EColumnType kColumnType = ColumnT;
};

template <typename NarrowT, ROOT::Experimental::EColumnType ColumnT>
struct Helper<ROOT::Experimental::ClusterSize_t, NarrowT, ColumnT> {
   using Pod_t = std::uint64_t;
   using Narrow_t = NarrowT;
   static constexpr ROOT::Experimental::EColumnType kColumnType = ColumnT;
};

template <typename HelperT>
class PackingInt : public ::testing::Test {
public:
   using Helper_t = HelperT;
};

template <typename HelperT>
class PackingReal : public ::testing::Test {
public:
   using Helper_t = HelperT;
};

template <typename HelperT>
class PackingIndex : public ::testing::Test {
public:
   using Helper_t = HelperT;
};

using PackingRealTypes = ::testing::Types<Helper<double, double, ROOT::Experimental::EColumnType::kSplitReal64>,
                                          Helper<float, float, ROOT::Experimental::EColumnType::kSplitReal32>>;
TYPED_TEST_SUITE(PackingReal, PackingRealTypes);

using PackingIntTypes =
   ::testing::Types<Helper<std::int64_t, std::int64_t, ROOT::Experimental::EColumnType::kSplitInt64>,
                    Helper<std::uint64_t, std::uint64_t, ROOT::Experimental::EColumnType::kSplitUInt64>,
                    Helper<std::int32_t, std::int32_t, ROOT::Experimental::EColumnType::kSplitInt32>,
                    Helper<std::uint32_t, std::uint32_t, ROOT::Experimental::EColumnType::kSplitUInt32>,
                    Helper<std::int16_t, std::int16_t, ROOT::Experimental::EColumnType::kSplitInt16>,
                    Helper<std::uint16_t, std::uint16_t, ROOT::Experimental::EColumnType::kSplitUInt16>>;
TYPED_TEST_SUITE(PackingInt, PackingIntTypes);

using PackingIndexTypes = ::testing::Types<
   Helper<ROOT::Experimental::ClusterSize_t, std::uint32_t, ROOT::Experimental::EColumnType::kSplitIndex32>,
   Helper<ROOT::Experimental::ClusterSize_t, std::uint64_t, ROOT::Experimental::EColumnType::kSplitIndex64>>;
TYPED_TEST_SUITE(PackingIndex, PackingIndexTypes);

TEST(Packing, Bitfield)
{
   ROOT::Experimental::Detail::RColumnElement<bool, ROOT::Experimental::EColumnType::kBit> element(nullptr);
   element.Pack(nullptr, nullptr, 0);
   element.Unpack(nullptr, nullptr, 0);

   bool b = true;
   char c = 0;
   element.Pack(&c, &b, 1);
   EXPECT_EQ(1, c);
   bool e = false;
   element.Unpack(&e, &c, 1);
   EXPECT_TRUE(e);

   bool b8[] = {true, false, true, false, false, true, false, true};
   c = 0;
   element.Pack(&c, &b8, 8);
   bool e8[] = {false, false, false, false, false, false, false, false};
   element.Unpack(&e8, &c, 8);
   for (unsigned i = 0; i < 8; ++i) {
      EXPECT_EQ(b8[i], e8[i]);
   }

   bool b9[] = {true, false, true, false, false, true, false, true, true};
   char c2[2];
   element.Pack(&c2, &b9, 9);
   bool e9[] = {false, false, false, false, false, false, false, false, false};
   element.Unpack(&e9, &c2, 9);
   for (unsigned i = 0; i < 9; ++i) {
      EXPECT_EQ(b9[i], e9[i]);
   }
}

TEST(Packing, RColumnSwitch)
{
   ROOT::Experimental::Detail::RColumnElement<ROOT::Experimental::RColumnSwitch,
                                              ROOT::Experimental::EColumnType::kSwitch>
      element(nullptr);
   element.Pack(nullptr, nullptr, 0);
   element.Unpack(nullptr, nullptr, 0);

   ROOT::Experimental::RColumnSwitch s1(ClusterSize_t{0xaa}, 0x55);
   std::uint64_t out = 0;
   element.Pack(&out, &s1, 1);
   EXPECT_NE(0, out);
   ROOT::Experimental::RColumnSwitch s2;
   element.Unpack(&s2, &out, 1);
   EXPECT_EQ(0xaa, s2.GetIndex());
   EXPECT_EQ(0x55, s2.GetTag());
}

TYPED_TEST(PackingReal, SplitReal)
{
   using Pod_t = typename TestFixture::Helper_t::Pod_t;
   using Narrow_t = typename TestFixture::Helper_t::Narrow_t;

   ROOT::Experimental::Detail::RColumnElement<Pod_t, TestFixture::Helper_t::kColumnType> element(nullptr);
   element.Pack(nullptr, nullptr, 0);
   element.Unpack(nullptr, nullptr, 0);

   std::array<Pod_t, 7> mem{0.0,
                            42.0,
                            std::numeric_limits<Narrow_t>::min(),
                            std::numeric_limits<Narrow_t>::max(),
                            std::numeric_limits<Narrow_t>::lowest(),
                            std::numeric_limits<Narrow_t>::infinity(),
                            std::numeric_limits<Narrow_t>::denorm_min()};
   std::array<Pod_t, 7> packed;
   std::array<Pod_t, 7> cmp;

   element.Pack(packed.data(), mem.data(), 7);
   element.Unpack(cmp.data(), packed.data(), 7);

   EXPECT_EQ(mem, cmp);
}

TYPED_TEST(PackingInt, SplitInt)
{
   using Pod_t = typename TestFixture::Helper_t::Pod_t;
   using Narrow_t = typename TestFixture::Helper_t::Narrow_t;

   ROOT::Experimental::Detail::RColumnElement<Pod_t, TestFixture::Helper_t::kColumnType> element(nullptr);
   element.Pack(nullptr, nullptr, 0);
   element.Unpack(nullptr, nullptr, 0);

   std::array<Pod_t, 9> mem{0, std::is_signed_v<Pod_t> ? -42 : 1, 42, std::numeric_limits<Narrow_t>::min(),
                            std::numeric_limits<Narrow_t>::min() + 1, std::numeric_limits<Narrow_t>::min() + 2,
                            std::numeric_limits<Narrow_t>::max(), std::numeric_limits<Narrow_t>::max() - 1,
                            std::numeric_limits<Narrow_t>::max() - 2};
   std::array<Pod_t, 9> packed;
   std::array<Pod_t, 9> cmp;

   element.Pack(packed.data(), mem.data(), 9);
   element.Unpack(cmp.data(), packed.data(), 9);

   EXPECT_EQ(mem, cmp);
}

TYPED_TEST(PackingIndex, SplitIndex)
{
   using Pod_t = typename TestFixture::Helper_t::Pod_t;
   using Narrow_t = typename TestFixture::Helper_t::Narrow_t;

   ROOT::Experimental::Detail::RColumnElement<ClusterSize_t, TestFixture::Helper_t::kColumnType> element(nullptr);
   element.Pack(nullptr, nullptr, 0);
   element.Unpack(nullptr, nullptr, 0);

   std::array<Pod_t, 5> mem{0, 1, 1, 42, std::numeric_limits<Narrow_t>::max()};
   std::array<Pod_t, 5> packed;
   std::array<Pod_t, 5> cmp;

   element.Pack(packed.data(), mem.data(), 5);
   element.Unpack(cmp.data(), packed.data(), 5);

   EXPECT_EQ(mem, cmp);
}

namespace {

template <typename PodT, ROOT::Experimental::EColumnType ColumnT>
static void AddField(RNTupleModel &model, const std::string &fieldName)
{
   auto fld = std::make_unique<RField<PodT>>(fieldName);
   fld->SetColumnRepresentative({ColumnT});
   model.AddField(std::move(fld));
}

} // anonymous namespace

TEST(Packing, OnDiskEncoding)
{
   FileRaii fileGuard("test_ntuple_packing_ondiskencoding.root");

   constexpr std::size_t kPageSize = 16; // minimum page size for two 8 byte elements

   auto model = RNTupleModel::Create();

   AddField<std::int16_t, ROOT::Experimental::EColumnType::kSplitInt16>(*model, "int16");
   AddField<std::int32_t, ROOT::Experimental::EColumnType::kSplitInt32>(*model, "int32");
   AddField<std::int64_t, ROOT::Experimental::EColumnType::kSplitInt64>(*model, "int64");
   AddField<std::uint16_t, ROOT::Experimental::EColumnType::kSplitUInt16>(*model, "uint16");
   AddField<std::uint32_t, ROOT::Experimental::EColumnType::kSplitUInt32>(*model, "uint32");
   AddField<std::uint64_t, ROOT::Experimental::EColumnType::kSplitUInt64>(*model, "uint64");
   AddField<float, ROOT::Experimental::EColumnType::kSplitReal32>(*model, "float");
   AddField<double, ROOT::Experimental::EColumnType::kSplitReal64>(*model, "double");
   AddField<ClusterSize_t, ROOT::Experimental::EColumnType::kSplitIndex32>(*model, "index32");
   AddField<ClusterSize_t, ROOT::Experimental::EColumnType::kSplitIndex64>(*model, "index64");
   auto fldStr = std::make_unique<RField<std::string>>("str");
   model->AddField(std::move(fldStr));
   {
      RNTupleWriteOptions options;
      options.SetCompression(0);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath(), options);
      auto e = writer->CreateEntry();

      *e->Get<std::int16_t>("int16") = 1;
      *e->Get<std::int32_t>("int32") = 0x00010203;
      *e->Get<std::int64_t>("int64") = 0x0001020304050607L;
      *e->Get<std::uint16_t>("uint16") = 1;
      *e->Get<std::uint32_t>("uint32") = 0x00010203;
      *e->Get<std::uint64_t>("uint64") = 0x0001020304050607L;
      *e->Get<float>("float") = std::nextafterf(1.f, 2.f); // 0 01111111 00000000000000000000001 == 0x3f800001
      *e->Get<double>("double") = std::nextafter(1., 2.);  // 0x3ff0 0000 0000 0001
      *e->Get<ClusterSize_t>("index32") = 39916801;        // 0x0261 1501
      *e->Get<ClusterSize_t>("index64") = 0x0706050403020100L;
      e->Get<std::string>("str")->assign("abc");

      writer->Fill(*e);

      *e->Get<std::int16_t>("int16") = -3;
      *e->Get<std::int32_t>("int32") = -0x04050607;
      *e->Get<std::int64_t>("int64") = -0x08090a0b0c0d0e0fL;
      *e->Get<std::uint16_t>("uint16") = 2;
      *e->Get<std::uint32_t>("uint32") = 0x04050607;
      *e->Get<std::uint64_t>("uint64") = 0x08090a0b0c0d0e0fL;
      *e->Get<float>("float") = std::nextafterf(1.f, 0.f);            // 0 01111110 11111111111111111111111 = 0x3f7fffff
      *e->Get<double>("double") = std::numeric_limits<double>::max(); // 0x7fef ffff ffff ffff
      *e->Get<ClusterSize_t>("index32") = 39916808;                   // d(previous) == 7
      *e->Get<ClusterSize_t>("index64") = 0x070605040302010DL;        // d(previous) == 13
      e->Get<std::string>("str")->assign("de");

      writer->Fill(*e);
   }

   auto source = RPageSource::Create("ntuple", fileGuard.GetPath());
   source->Attach();

   char buf[2 * kPageSize];
   RPageStorage::RSealedPage sealedPage(buf, 2 * kPageSize, 0);

   auto fnGetColumnId = [&source](const std::string &fieldName) {
      auto descGuard = source->GetSharedDescriptorGuard();
      return descGuard->FindPhysicalColumnId(descGuard->FindFieldId(fieldName), 0);
   };

   source->LoadSealedPage(fnGetColumnId("int16"), RClusterIndex(0, 0), sealedPage);
   unsigned char expInt16[] = {0x02, 0x05, 0x00, 0x00};
   EXPECT_EQ(memcmp(sealedPage.fBuffer, expInt16, sizeof(expInt16)), 0);

   source->LoadSealedPage(fnGetColumnId("int32"), RClusterIndex(0, 0), sealedPage);
   unsigned char expInt32[] = {0x06, 0x0d, 0x04, 0x0c, 0x02, 0x0a, 0x00, 0x08};
   EXPECT_EQ(memcmp(sealedPage.fBuffer, expInt32, sizeof(expInt32)), 0);

   source->LoadSealedPage(fnGetColumnId("int64"), RClusterIndex(0, 0), sealedPage);
   unsigned char expInt64[] = {0x0e, 0x1d, 0x0c, 0x1c, 0x0a, 0x1a, 0x08, 0x18,
                               0x06, 0x16, 0x04, 0x14, 0x02, 0x12, 0x00, 0x10};
   EXPECT_EQ(memcmp(sealedPage.fBuffer, expInt64, sizeof(expInt64)), 0);

   source->LoadSealedPage(fnGetColumnId("uint16"), RClusterIndex(0, 0), sealedPage);
   unsigned char expUInt16[] = {0x01, 0x02, 0x00, 0x00};
   EXPECT_EQ(memcmp(sealedPage.fBuffer, expUInt16, sizeof(expUInt16)), 0);

   source->LoadSealedPage(fnGetColumnId("uint32"), RClusterIndex(0, 0), sealedPage);
   unsigned char expUInt32[] = {0x03, 0x07, 0x02, 0x06, 0x01, 0x05, 0x00, 0x04};
   EXPECT_EQ(memcmp(sealedPage.fBuffer, expUInt32, sizeof(expUInt32)), 0);

   source->LoadSealedPage(fnGetColumnId("uint64"), RClusterIndex(0, 0), sealedPage);
   unsigned char expUInt64[] = {0x07, 0x0f, 0x06, 0x0e, 0x05, 0x0d, 0x04, 0x0c,
                                0x03, 0x0b, 0x02, 0x0a, 0x01, 0x09, 0x00, 0x08};
   EXPECT_EQ(memcmp(sealedPage.fBuffer, expUInt64, sizeof(expUInt64)), 0);

   source->LoadSealedPage(fnGetColumnId("float"), RClusterIndex(0, 0), sealedPage);
   unsigned char expFloat[] = {0x01, 0xff, 0x00, 0xff, 0x80, 0x7f, 0x3f, 0x3f};
   EXPECT_EQ(memcmp(sealedPage.fBuffer, expFloat, sizeof(expFloat)), 0);

   source->LoadSealedPage(fnGetColumnId("double"), RClusterIndex(0, 0), sealedPage);
   unsigned char expDouble[] = {0x01, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff,
                                0x00, 0xff, 0x00, 0xff, 0xf0, 0xef, 0x3f, 0x7f};
   EXPECT_EQ(memcmp(sealedPage.fBuffer, expDouble, sizeof(expDouble)), 0);

   source->LoadSealedPage(fnGetColumnId("index32"), RClusterIndex(0, 0), sealedPage);
   unsigned char expIndex32[] = {0x01, 0x07, 0x15, 0x00, 0x61, 0x00, 0x02, 0x00};
   EXPECT_EQ(memcmp(sealedPage.fBuffer, expIndex32, sizeof(expIndex32)), 0);

   source->LoadSealedPage(fnGetColumnId("index64"), RClusterIndex(0, 0), sealedPage);
   unsigned char expIndex64[] = {0x00, 0x0D, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00,
                                 0x04, 0x00, 0x05, 0x00, 0x06, 0x00, 0x07, 0x00};
   EXPECT_EQ(memcmp(sealedPage.fBuffer, expIndex64, sizeof(expIndex64)), 0);

   auto reader = RNTupleReader(std::move(source));
   EXPECT_EQ(EColumnType::kIndex64, reader.GetModel()->GetField("str")->GetColumnRepresentative()[0]);
   EXPECT_EQ(2u, reader.GetNEntries());
   auto viewStr = reader.GetView<std::string>("str");
   EXPECT_EQ(std::string("abc"), viewStr(0));
   EXPECT_EQ(std::string("de"), viewStr(1));
}
