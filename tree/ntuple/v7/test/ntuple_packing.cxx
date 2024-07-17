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
   ROOT::Experimental::Internal::RColumnElement<bool, ROOT::Experimental::EColumnType::kBit> element;
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

TEST(Packing, HalfPrecisionFloat)
{
   ROOT::Experimental::Internal::RColumnElement<float, ROOT::Experimental::EColumnType::kReal16> element32_16;
   ROOT::Experimental::Internal::RColumnElement<double, ROOT::Experimental::EColumnType::kReal16> element64_16;
   element32_16.Pack(nullptr, nullptr, 0);
   element32_16.Unpack(nullptr, nullptr, 0);
   element64_16.Pack(nullptr, nullptr, 0);
   element64_16.Unpack(nullptr, nullptr, 0);

   float fin = 3.14;
   unsigned char buf[2] = {0, 0};
   element32_16.Pack(buf, &fin, 1);
   // Expected bit representation: 0b01000010 01001000
   EXPECT_EQ(0x48, buf[0]);
   EXPECT_EQ(0x42, buf[1]);
   float fout = 0.;
   element32_16.Unpack(&fout, buf, 1);
   EXPECT_FLOAT_EQ(3.140625, fout);

   buf[0] = buf[1] = 0;
   double din = 3.14;
   element64_16.Pack(buf, &din, 1);
   // Expected bit representation: 0b01000010 01001000
   EXPECT_EQ(0x48, buf[0]);
   EXPECT_EQ(0x42, buf[1]);
   double dout = 0.;
   element64_16.Unpack(&dout, buf, 1);
   EXPECT_FLOAT_EQ(3.140625, dout);

   float fin4[] = {0.1, 0.2, 0.3, 0.4};
   std::uint64_t b4 = 0;
   element32_16.Pack(&b4, &fin4, 4);
   float fout4[] = {0., 0., 0., 0.};
   element32_16.Unpack(&fout4, &b4, 4);
   EXPECT_FLOAT_EQ(0.099975586, fout4[0]);
   EXPECT_FLOAT_EQ(0.199951171, fout4[1]);
   EXPECT_FLOAT_EQ(0.300048828, fout4[2]);
   EXPECT_FLOAT_EQ(0.399902343, fout4[3]);

   double din4[] = {0.1, 0.2, 0.3, 0.4};
   b4 = 0;
   element64_16.Pack(&b4, &din4, 4);
   double dout4[] = {0., 0., 0., 0.};
   element64_16.Unpack(&dout4, &b4, 4);
   EXPECT_FLOAT_EQ(0.099975586, dout4[0]);
   EXPECT_FLOAT_EQ(0.199951171, dout4[1]);
   EXPECT_FLOAT_EQ(0.300048828, dout4[2]);
   EXPECT_FLOAT_EQ(0.399902343, dout4[3]);
}

TEST(Packing, RColumnSwitch)
{
   ROOT::Experimental::Internal::RColumnElement<ROOT::Experimental::RColumnSwitch,
                                                ROOT::Experimental::EColumnType::kSwitch>
      element;
   element.Pack(nullptr, nullptr, 0);
   element.Unpack(nullptr, nullptr, 0);

   ROOT::Experimental::RColumnSwitch s1(ClusterSize_t{0xaa}, 0x55);
   unsigned char out[12];
   element.Pack(out, &s1, 1);
   ROOT::Experimental::RColumnSwitch s2;
   element.Unpack(&s2, out, 1);
   EXPECT_EQ(0xaa, s2.GetIndex());
   EXPECT_EQ(0x55, s2.GetTag());
}

TYPED_TEST(PackingReal, SplitReal)
{
   using Pod_t = typename TestFixture::Helper_t::Pod_t;
   using Narrow_t = typename TestFixture::Helper_t::Narrow_t;

   ROOT::Experimental::Internal::RColumnElement<Pod_t, TestFixture::Helper_t::kColumnType> element;
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

   ROOT::Experimental::Internal::RColumnElement<Pod_t, TestFixture::Helper_t::kColumnType> element;
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

   ROOT::Experimental::Internal::RColumnElement<ClusterSize_t, TestFixture::Helper_t::kColumnType> element;
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
   AddField<float, ROOT::Experimental::EColumnType::kReal16>(*model, "float16");
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

      *e->GetPtr<std::int16_t>("int16") = 1;
      *e->GetPtr<std::int32_t>("int32") = 0x00010203;
      *e->GetPtr<std::int64_t>("int64") = 0x0001020304050607L;
      *e->GetPtr<std::uint16_t>("uint16") = 1;
      *e->GetPtr<std::uint32_t>("uint32") = 0x00010203;
      *e->GetPtr<std::uint64_t>("uint64") = 0x0001020304050607L;
      *e->GetPtr<float>("float") = std::nextafterf(1.f, 2.f);   // 0 01111111 00000000000000000000001 == 0x3f800001
      *e->GetPtr<float>("float16") = std::nextafterf(1.f, 2.f); // 0 01111111 00000000000000000000001 == 0x3f800001
      *e->GetPtr<double>("double") = std::nextafter(1., 2.);    // 0x3ff0 0000 0000 0001
      *e->GetPtr<ClusterSize_t>("index32") = 39916801;          // 0x0261 1501
      *e->GetPtr<ClusterSize_t>("index64") = 0x0706050403020100L;
      e->GetPtr<std::string>("str")->assign("abc");

      writer->Fill(*e);

      *e->GetPtr<std::int16_t>("int16") = -3;
      *e->GetPtr<std::int32_t>("int32") = -0x04050607;
      *e->GetPtr<std::int64_t>("int64") = -0x08090a0b0c0d0e0fL;
      *e->GetPtr<std::uint16_t>("uint16") = 2;
      *e->GetPtr<std::uint32_t>("uint32") = 0x04050607;
      *e->GetPtr<std::uint64_t>("uint64") = 0x08090a0b0c0d0e0fL;
      *e->GetPtr<float>("float") = std::nextafterf(1.f, 0.f);     // 0 01111110 11111111111111111111111 = 0x3f7fffff
      *e->GetPtr<float>("float16") = std::nextafterf(0.1f, 0.2f); // 0 01111011 10011001100110011001110 = 0x3dccccce
      *e->GetPtr<double>("double") = std::numeric_limits<double>::max(); // 0x7fef ffff ffff ffff
      *e->GetPtr<ClusterSize_t>("index32") = 39916808;                   // d(previous) == 7
      *e->GetPtr<ClusterSize_t>("index64") = 0x070605040302010DL;        // d(previous) == 13
      e->GetPtr<std::string>("str")->assign("de");

      writer->Fill(*e);
   }

   auto source = RPageSource::Create("ntuple", fileGuard.GetPath());
   source->Attach();

   char buf[2 * kPageSize];
   RPageStorage::RSealedPage sealedPage(buf, 2 * kPageSize, 0);

   auto fnGetColumnId = [&source](const std::string &fieldName) {
      auto descGuard = source->GetSharedDescriptorGuard();
      return descGuard->FindPhysicalColumnId(descGuard->FindFieldId(fieldName), 0, 0);
   };

   source->LoadSealedPage(fnGetColumnId("int16"), RClusterIndex(0, 0), sealedPage);
   unsigned char expInt16[] = {0x02, 0x05, 0x00, 0x00};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expInt16, sizeof(expInt16)), 0);

   source->LoadSealedPage(fnGetColumnId("int32"), RClusterIndex(0, 0), sealedPage);
   unsigned char expInt32[] = {0x06, 0x0d, 0x04, 0x0c, 0x02, 0x0a, 0x00, 0x08};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expInt32, sizeof(expInt32)), 0);

   source->LoadSealedPage(fnGetColumnId("int64"), RClusterIndex(0, 0), sealedPage);
   unsigned char expInt64[] = {0x0e, 0x1d, 0x0c, 0x1c, 0x0a, 0x1a, 0x08, 0x18,
                               0x06, 0x16, 0x04, 0x14, 0x02, 0x12, 0x00, 0x10};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expInt64, sizeof(expInt64)), 0);

   source->LoadSealedPage(fnGetColumnId("uint16"), RClusterIndex(0, 0), sealedPage);
   unsigned char expUInt16[] = {0x01, 0x02, 0x00, 0x00};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expUInt16, sizeof(expUInt16)), 0);

   source->LoadSealedPage(fnGetColumnId("uint32"), RClusterIndex(0, 0), sealedPage);
   unsigned char expUInt32[] = {0x03, 0x07, 0x02, 0x06, 0x01, 0x05, 0x00, 0x04};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expUInt32, sizeof(expUInt32)), 0);

   source->LoadSealedPage(fnGetColumnId("uint64"), RClusterIndex(0, 0), sealedPage);
   unsigned char expUInt64[] = {0x07, 0x0f, 0x06, 0x0e, 0x05, 0x0d, 0x04, 0x0c,
                                0x03, 0x0b, 0x02, 0x0a, 0x01, 0x09, 0x00, 0x08};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expUInt64, sizeof(expUInt64)), 0);

   source->LoadSealedPage(fnGetColumnId("float"), RClusterIndex(0, 0), sealedPage);
   unsigned char expFloat[] = {0x01, 0xff, 0x00, 0xff, 0x80, 0x7f, 0x3f, 0x3f};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expFloat, sizeof(expFloat)), 0);

   source->LoadSealedPage(fnGetColumnId("float16"), RClusterIndex(0, 0), sealedPage);
   unsigned char expFloat16[] = {0x00, 0x3c, 0x66, 0x2e};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expFloat16, sizeof(expFloat16)), 0);

   source->LoadSealedPage(fnGetColumnId("double"), RClusterIndex(0, 0), sealedPage);
   unsigned char expDouble[] = {0x01, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff,
                                0x00, 0xff, 0x00, 0xff, 0xf0, 0xef, 0x3f, 0x7f};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expDouble, sizeof(expDouble)), 0);

   source->LoadSealedPage(fnGetColumnId("index32"), RClusterIndex(0, 0), sealedPage);
   unsigned char expIndex32[] = {0x01, 0x07, 0x15, 0x00, 0x61, 0x00, 0x02, 0x00};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expIndex32, sizeof(expIndex32)), 0);

   source->LoadSealedPage(fnGetColumnId("index64"), RClusterIndex(0, 0), sealedPage);
   unsigned char expIndex64[] = {0x00, 0x0D, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00,
                                 0x04, 0x00, 0x05, 0x00, 0x06, 0x00, 0x07, 0x00};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expIndex64, sizeof(expIndex64)), 0);

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(EColumnType::kIndex64, reader->GetModel().GetField("str").GetColumnRepresentative()[0]);
   EXPECT_EQ(2u, reader->GetNEntries());
   auto viewStr = reader->GetView<std::string>("str");
   EXPECT_EQ(std::string("abc"), viewStr(0));
   EXPECT_EQ(std::string("de"), viewStr(1));
}
