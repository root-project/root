#include "ntuple_test.hxx"

#include "../src/RColumnElement.hxx"

#include <array>
#include <cmath>
#include <cstring> // for memcmp
#include <limits>
#include <type_traits>
#include <utility>
#include <random>

#if defined(__GNUC__)
#pragma GCC diagnostic push
// Silence spurious warnings such as:
// ‘void* memcpy(void*, const void*, size_t)’ forming offset [1, 7] is out of the bounds [0, 1] of object ‘c’ with type
// ‘char’
#pragma GCC diagnostic ignored "-Warray-bounds"
#if !defined(__clang__)
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif
#endif

template <typename PodT, typename NarrowT, ENTupleColumnType ColumnT>
struct Helper {
   using Pod_t = PodT;
   using Narrow_t = NarrowT;
   static constexpr ENTupleColumnType kColumnType = ColumnT;
};

template <typename NarrowT, ENTupleColumnType ColumnT>
struct Helper<RColumnIndex, NarrowT, ColumnT> {
   using Pod_t = std::uint64_t;
   using Narrow_t = NarrowT;
   static constexpr ENTupleColumnType kColumnType = ColumnT;
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

using PackingRealTypes = ::testing::Types<Helper<double, double, ENTupleColumnType::kSplitReal64>,
                                          Helper<float, float, ENTupleColumnType::kSplitReal32>>;
TYPED_TEST_SUITE(PackingReal, PackingRealTypes);

using PackingIntTypes = ::testing::Types<Helper<std::int64_t, std::int64_t, ENTupleColumnType::kSplitInt64>,
                                         Helper<std::uint64_t, std::uint64_t, ENTupleColumnType::kSplitUInt64>,
                                         Helper<std::int32_t, std::int32_t, ENTupleColumnType::kSplitInt32>,
                                         Helper<std::uint32_t, std::uint32_t, ENTupleColumnType::kSplitUInt32>,
                                         Helper<std::int16_t, std::int16_t, ENTupleColumnType::kSplitInt16>,
                                         Helper<std::uint16_t, std::uint16_t, ENTupleColumnType::kSplitUInt16>>;
TYPED_TEST_SUITE(PackingInt, PackingIntTypes);

using PackingIndexTypes = ::testing::Types<Helper<RColumnIndex, std::uint32_t, ENTupleColumnType::kSplitIndex32>,
                                           Helper<RColumnIndex, std::uint64_t, ENTupleColumnType::kSplitIndex64>>;
TYPED_TEST_SUITE(PackingIndex, PackingIndexTypes);

TEST(Packing, Bitfield)
{
   auto element = RColumnElementBase::Generate<bool>(ENTupleColumnType::kBit);
   element->Pack(nullptr, nullptr, 0);
   element->Unpack(nullptr, nullptr, 0);

   bool b = true;
   char c = 0;
   element->Pack(&c, &b, 1);
   EXPECT_EQ(1, c);
   bool e = false;
   element->Unpack(&e, &c, 1);
   EXPECT_TRUE(e);

   bool b8[] = {true, false, true, false, false, true, false, true};
   c = 0;
   element->Pack(&c, &b8, 8);
   bool e8[] = {false, false, false, false, false, false, false, false};
   element->Unpack(&e8, &c, 8);
   for (unsigned i = 0; i < 8; ++i) {
      EXPECT_EQ(b8[i], e8[i]);
   }

   bool b9[] = {true, false, true, false, false, true, false, true, true};
   char c2[2];
   element->Pack(&c2, &b9, 9);
   bool e9[] = {false, false, false, false, false, false, false, false, false};
   element->Unpack(&e9, &c2, 9);
   for (unsigned i = 0; i < 9; ++i) {
      EXPECT_EQ(b9[i], e9[i]);
   }
}

TEST(Packing, HalfPrecisionFloat)
{
   auto element32_16 = RColumnElementBase::Generate<float>(ENTupleColumnType::kReal16);
   auto element64_16 = RColumnElementBase::Generate<double>(ENTupleColumnType::kReal16);
   element32_16->Pack(nullptr, nullptr, 0);
   element32_16->Unpack(nullptr, nullptr, 0);
   element64_16->Pack(nullptr, nullptr, 0);
   element64_16->Unpack(nullptr, nullptr, 0);

   float fin = 3.14;
   unsigned char buf[2] = {0, 0};
   element32_16->Pack(buf, &fin, 1);
   // Expected bit representation: 0b01000010 01001000
   EXPECT_EQ(0x48, buf[0]);
   EXPECT_EQ(0x42, buf[1]);
   float fout = 0.;
   element32_16->Unpack(&fout, buf, 1);
   EXPECT_FLOAT_EQ(3.140625, fout);

   buf[0] = buf[1] = 0;
   double din = 3.14;
   element64_16->Pack(buf, &din, 1);
   // Expected bit representation: 0b01000010 01001000
   EXPECT_EQ(0x48, buf[0]);
   EXPECT_EQ(0x42, buf[1]);
   double dout = 0.;
   element64_16->Unpack(&dout, buf, 1);
   EXPECT_FLOAT_EQ(3.140625, dout);

   float fin4[] = {0.1, 0.2, 0.3, 0.4};
   std::uint64_t b4 = 0;
   element32_16->Pack(&b4, &fin4, 4);
   float fout4[] = {0., 0., 0., 0.};
   element32_16->Unpack(&fout4, &b4, 4);
   EXPECT_FLOAT_EQ(0.099975586, fout4[0]);
   EXPECT_FLOAT_EQ(0.199951171, fout4[1]);
   EXPECT_FLOAT_EQ(0.300048828, fout4[2]);
   EXPECT_FLOAT_EQ(0.399902343, fout4[3]);

   double din4[] = {0.1, 0.2, 0.3, 0.4};
   b4 = 0;
   element64_16->Pack(&b4, &din4, 4);
   double dout4[] = {0., 0., 0., 0.};
   element64_16->Unpack(&dout4, &b4, 4);
   EXPECT_FLOAT_EQ(0.099975586, dout4[0]);
   EXPECT_FLOAT_EQ(0.199951171, dout4[1]);
   EXPECT_FLOAT_EQ(0.300048828, dout4[2]);
   EXPECT_FLOAT_EQ(0.399902343, dout4[3]);
}

TEST(Packing, RColumnSwitch)
{
   auto element = RColumnElementBase::Generate<RColumnSwitch>(ENTupleColumnType::kSwitch);
   element->Pack(nullptr, nullptr, 0);
   element->Unpack(nullptr, nullptr, 0);

   RColumnSwitch s1(RColumnIndex{0xaa}, 0x55);
   unsigned char out[12];
   element->Pack(out, &s1, 1);
   RColumnSwitch s2;
   element->Unpack(&s2, out, 1);
   EXPECT_EQ(0xaa, s2.GetIndex());
   EXPECT_EQ(0x55, s2.GetTag());
}

TYPED_TEST(PackingReal, SplitReal)
{
   using Pod_t = typename TestFixture::Helper_t::Pod_t;
   using Narrow_t = typename TestFixture::Helper_t::Narrow_t;

   auto element = RColumnElementBase::Generate<Pod_t>(TestFixture::Helper_t::kColumnType);
   element->Pack(nullptr, nullptr, 0);
   element->Unpack(nullptr, nullptr, 0);

   std::array<Pod_t, 7> mem{0.0,
                            42.0,
                            std::numeric_limits<Narrow_t>::min(),
                            std::numeric_limits<Narrow_t>::max(),
                            std::numeric_limits<Narrow_t>::lowest(),
                            std::numeric_limits<Narrow_t>::infinity(),
                            std::numeric_limits<Narrow_t>::denorm_min()};
   std::array<Pod_t, 7> packed;
   std::array<Pod_t, 7> cmp;

   element->Pack(packed.data(), mem.data(), 7);
   element->Unpack(cmp.data(), packed.data(), 7);

   EXPECT_EQ(mem, cmp);
}

TYPED_TEST(PackingInt, SplitInt)
{
   using Pod_t = typename TestFixture::Helper_t::Pod_t;
   using Narrow_t = typename TestFixture::Helper_t::Narrow_t;

   auto element = RColumnElementBase::Generate<Pod_t>(TestFixture::Helper_t::kColumnType);
   element->Pack(nullptr, nullptr, 0);
   element->Unpack(nullptr, nullptr, 0);

   std::array<Pod_t, 9> mem{0,
                            std::is_signed_v<Pod_t> ? -42 : 1,
                            42,
                            std::numeric_limits<Narrow_t>::min(),
                            std::numeric_limits<Narrow_t>::min() + 1,
                            std::numeric_limits<Narrow_t>::min() + 2,
                            std::numeric_limits<Narrow_t>::max(),
                            std::numeric_limits<Narrow_t>::max() - 1,
                            std::numeric_limits<Narrow_t>::max() - 2};
   std::array<Pod_t, 9> packed;
   std::array<Pod_t, 9> cmp;

   element->Pack(packed.data(), mem.data(), 9);
   element->Unpack(cmp.data(), packed.data(), 9);

   EXPECT_EQ(mem, cmp);
}

TYPED_TEST(PackingIndex, SplitIndex)
{
   using Pod_t = typename TestFixture::Helper_t::Pod_t;
   using Narrow_t = typename TestFixture::Helper_t::Narrow_t;

   auto element = RColumnElementBase::Generate<RColumnIndex>(TestFixture::Helper_t::kColumnType);
   element->Pack(nullptr, nullptr, 0);
   element->Unpack(nullptr, nullptr, 0);

   std::array<Pod_t, 5> mem{0, 1, 1, 42, std::numeric_limits<Narrow_t>::max()};
   std::array<Pod_t, 5> packed;
   std::array<Pod_t, 5> cmp;

   element->Pack(packed.data(), mem.data(), 5);
   element->Unpack(cmp.data(), packed.data(), 5);

   EXPECT_EQ(mem, cmp);
}

template <typename PodT, ENTupleColumnType ColumnT>
static void AddField(RNTupleModel &model, const std::string &fieldName)
{
   auto fld = std::make_unique<RField<PodT>>(fieldName);
   fld->SetColumnRepresentatives({{ColumnT}});
   model.AddField(std::move(fld));
}

static void AddReal32TruncField(RNTupleModel &model, const std::string &fieldName, std::size_t nBits)
{
   auto fld = std::make_unique<RField<float>>(fieldName);
   fld->SetColumnRepresentatives({{ENTupleColumnType::kReal32Trunc}});
   fld->SetTruncated(nBits);
   model.AddField(std::move(fld));
}

static void
AddReal32QuantField(RNTupleModel &model, const std::string &fieldName, std::size_t nBits, double min, double max)
{
   auto fld = std::make_unique<RField<float>>(fieldName);
   fld->SetColumnRepresentatives({{ENTupleColumnType::kReal32Quant}});
   fld->SetQuantized(min, max, nBits);
   model.AddField(std::move(fld));
}

namespace {

// Test writing index32/64 columns
class RFieldTestIndexColumn final : public ROOT::RSimpleField<RColumnIndex> {
protected:
   std::unique_ptr<RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RFieldTestIndexColumn>(newName);
   }
   const RColumnRepresentations &GetColumnRepresentations() const final
   {
      static RColumnRepresentations representations(
         {{ENTupleColumnType::kSplitIndex64}, {ENTupleColumnType::kSplitIndex32}}, {});
      return representations;
   }

public:
   explicit RFieldTestIndexColumn(std::string_view name) : RSimpleField(name, "ROOT::Internal::RColumnIndex") {}
   RFieldTestIndexColumn(RFieldTestIndexColumn &&other) = default;
   RFieldTestIndexColumn &operator=(RFieldTestIndexColumn &&other) = default;
   ~RFieldTestIndexColumn() override = default;
};

} // anonymous namespace

TEST(Packing, OnDiskEncoding)
{
   FileRaii fileGuard("test_ntuple_packing_ondiskencoding.root");

   constexpr std::size_t kPageSize = 16; // minimum page size for two 8 byte elements

   auto model = RNTupleModel::Create();

   AddField<std::int16_t, ENTupleColumnType::kSplitInt16>(*model, "int16");
   AddField<std::int32_t, ENTupleColumnType::kSplitInt32>(*model, "int32");
   AddField<std::int64_t, ENTupleColumnType::kSplitInt64>(*model, "int64");
   AddField<std::uint16_t, ENTupleColumnType::kSplitUInt16>(*model, "uint16");
   AddField<std::uint32_t, ENTupleColumnType::kSplitUInt32>(*model, "uint32");
   AddField<std::uint64_t, ENTupleColumnType::kSplitUInt64>(*model, "uint64");
   AddField<float, ENTupleColumnType::kSplitReal32>(*model, "float");
   AddField<float, ENTupleColumnType::kReal16>(*model, "float16");
   AddField<double, ENTupleColumnType::kSplitReal64>(*model, "double");
   AddReal32TruncField(*model, "float32Trunc", 11);
   AddReal32QuantField(*model, "float32Quant", 7, 0.0, 1.0);
   auto fldIdx32 = std::make_unique<RFieldTestIndexColumn>("index32");
   fldIdx32->SetColumnRepresentatives({{ENTupleColumnType::kSplitIndex32}});
   model->AddField(std::move(fldIdx32));
   auto fldIdx64 = std::make_unique<RFieldTestIndexColumn>("index64");
   fldIdx64->SetColumnRepresentatives({{ENTupleColumnType::kSplitIndex64}});
   model->AddField(std::move(fldIdx64));
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
      *e->GetPtr<RColumnIndex>("index32") = 39916801;           // 0x0261 1501
      *e->GetPtr<RColumnIndex>("index64") = 0x0706050403020100L;
      *e->GetPtr<float>("float32Trunc") = -3.75f; // 1 10000000 11100000000000000000000 == 0xC0700000
      *e->GetPtr<float>("float32Quant") = 0.69f;  // quantized to 88 == 0b1011000
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
      *e->GetPtr<RColumnIndex>("index32") = 39916808;                    // d(previous) == 7
      *e->GetPtr<RColumnIndex>("index64") = 0x070605040302010DL;         // d(previous) == 13
      *e->GetPtr<float>("float32Trunc") = 1.875f; // 0 01111111 11100000000000000000000 == 0x3ff00000
      *e->GetPtr<float>("float32Quant") = 0.875f; // quantized to 111: 0b1101111
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

   source->LoadSealedPage(fnGetColumnId("int16"), RNTupleLocalIndex(0, 0), sealedPage);
   unsigned char expInt16[] = {0x02, 0x05, 0x00, 0x00};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expInt16, sizeof(expInt16)), 0);

   source->LoadSealedPage(fnGetColumnId("int32"), RNTupleLocalIndex(0, 0), sealedPage);
   unsigned char expInt32[] = {0x06, 0x0d, 0x04, 0x0c, 0x02, 0x0a, 0x00, 0x08};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expInt32, sizeof(expInt32)), 0);

   source->LoadSealedPage(fnGetColumnId("int64"), RNTupleLocalIndex(0, 0), sealedPage);
   unsigned char expInt64[] = {0x0e, 0x1d, 0x0c, 0x1c, 0x0a, 0x1a, 0x08, 0x18,
                               0x06, 0x16, 0x04, 0x14, 0x02, 0x12, 0x00, 0x10};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expInt64, sizeof(expInt64)), 0);

   source->LoadSealedPage(fnGetColumnId("uint16"), RNTupleLocalIndex(0, 0), sealedPage);
   unsigned char expUInt16[] = {0x01, 0x02, 0x00, 0x00};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expUInt16, sizeof(expUInt16)), 0);

   source->LoadSealedPage(fnGetColumnId("uint32"), RNTupleLocalIndex(0, 0), sealedPage);
   unsigned char expUInt32[] = {0x03, 0x07, 0x02, 0x06, 0x01, 0x05, 0x00, 0x04};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expUInt32, sizeof(expUInt32)), 0);

   source->LoadSealedPage(fnGetColumnId("uint64"), RNTupleLocalIndex(0, 0), sealedPage);
   unsigned char expUInt64[] = {0x07, 0x0f, 0x06, 0x0e, 0x05, 0x0d, 0x04, 0x0c,
                                0x03, 0x0b, 0x02, 0x0a, 0x01, 0x09, 0x00, 0x08};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expUInt64, sizeof(expUInt64)), 0);

   source->LoadSealedPage(fnGetColumnId("float"), RNTupleLocalIndex(0, 0), sealedPage);
   unsigned char expFloat[] = {0x01, 0xff, 0x00, 0xff, 0x80, 0x7f, 0x3f, 0x3f};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expFloat, sizeof(expFloat)), 0);

   source->LoadSealedPage(fnGetColumnId("float16"), RNTupleLocalIndex(0, 0), sealedPage);
   unsigned char expFloat16[] = {0x00, 0x3c, 0x66, 0x2e};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expFloat16, sizeof(expFloat16)), 0);

   source->LoadSealedPage(fnGetColumnId("double"), RNTupleLocalIndex(0, 0), sealedPage);
   unsigned char expDouble[] = {0x01, 0xff, 0x00, 0xff, 0x00, 0xff, 0x00, 0xff,
                                0x00, 0xff, 0x00, 0xff, 0xf0, 0xef, 0x3f, 0x7f};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expDouble, sizeof(expDouble)), 0);

   source->LoadSealedPage(fnGetColumnId("index32"), RNTupleLocalIndex(0, 0), sealedPage);
   unsigned char expIndex32[] = {0x01, 0x07, 0x15, 0x00, 0x61, 0x00, 0x02, 0x00};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expIndex32, sizeof(expIndex32)), 0);

   source->LoadSealedPage(fnGetColumnId("index64"), RNTupleLocalIndex(0, 0), sealedPage);
   unsigned char expIndex64[] = {0x00, 0x0D, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00,
                                 0x04, 0x00, 0x05, 0x00, 0x06, 0x00, 0x07, 0x00};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expIndex64, sizeof(expIndex64)), 0);

   source->LoadSealedPage(fnGetColumnId("float32Trunc"), RNTupleLocalIndex(0, 0), sealedPage);
   // Two tightly packed 11bit floats: 0b0'01111111'11 + 0b1'10000000'11 = 0b11111111111000000011 = 0x03fe0f
   unsigned char expF32Trunc[] = {0x03, 0xFE, 0x0F};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expF32Trunc, sizeof(expF32Trunc)), 0);

   source->LoadSealedPage(fnGetColumnId("float32Quant"), RNTupleLocalIndex(0, 0), sealedPage);
   // Two tightly packed 7bit quantized ints: 0b1101111 + 0b1011000 = 0b11011111011000 = 0x37d8
   unsigned char expF32Quant[] = {0xd8, 0x37};
   EXPECT_EQ(memcmp(sealedPage.GetBuffer(), expF32Quant, sizeof(expF32Quant)), 0);

   auto reader = RNTupleReader::Open(RNTupleModel::Create(), "ntuple", fileGuard.GetPath());
   EXPECT_EQ(2u, reader->GetNEntries());
   auto viewStr = reader->GetView<std::string>("str");
   EXPECT_EQ(std::string("abc"), viewStr(0));
   EXPECT_EQ(std::string("de"), viewStr(1));
   auto viewFtrunc = reader->GetView<float>("float32Trunc");
   EXPECT_EQ(-3.5f, viewFtrunc(0));
   EXPECT_EQ(1.75f, viewFtrunc(1));
   auto viewFquant = reader->GetView<float>("float32Quant");
   EXPECT_NEAR(0.69, viewFquant(0), 0.005f);
   EXPECT_NEAR(0.875f, viewFquant(1), 0.005f);
}

TEST(Packing, Real32TruncFloat)
{
   namespace BitPacking = ROOT::Internal::BitPacking;
   {
      constexpr auto kBitsOnStorage = 10;
      RColumnElement<float, ENTupleColumnType::kReal32Trunc> element;
      element.SetBitsOnStorage(kBitsOnStorage);
      element.Pack(nullptr, nullptr, 0);
      element.Unpack(nullptr, nullptr, 0);

      float f1 = 3.5f;
      unsigned char out[BitPacking::MinBufSize(1, kBitsOnStorage)];
      element.Pack(out, &f1, 1);

      float f2;
      element.Unpack(&f2, out, 1);
      // Dropping all but first 10 bits:
      // 0x40600000 -> 0x40400000
      // 3.5f       -> 3.f
      EXPECT_EQ(f2, 3.f);

      float f[5] = {3.5f, 3.5f, 3.5f, 3.5f, 3.5f};
      unsigned char out2[BitPacking::MinBufSize(5, kBitsOnStorage)];
      element.Pack(out2, f, 5);

      float fout[5];
      element.Unpack(fout, out2, 5);
      for (int i = 0; i < 5; ++i)
         EXPECT_EQ(fout[i], 3.f);

      // verify that Pack() doesn't write past the end of the valid buffer
      unsigned char outExtra[BitPacking::MinBufSize(5, kBitsOnStorage) + 4];
      outExtra[BitPacking::MinBufSize(5, kBitsOnStorage)] = 44;
      outExtra[BitPacking::MinBufSize(5, kBitsOnStorage) + 1] = 11;
      outExtra[BitPacking::MinBufSize(5, kBitsOnStorage) + 2] = 22;
      outExtra[BitPacking::MinBufSize(5, kBitsOnStorage) + 3] = 33;
      element.Pack(outExtra, f, 5);

      EXPECT_EQ(outExtra[BitPacking::MinBufSize(5, kBitsOnStorage)], 44);
      EXPECT_EQ(outExtra[BitPacking::MinBufSize(5, kBitsOnStorage) + 1], 11);
      EXPECT_EQ(outExtra[BitPacking::MinBufSize(5, kBitsOnStorage) + 2], 22);
      EXPECT_EQ(outExtra[BitPacking::MinBufSize(5, kBitsOnStorage) + 3], 33);
   }

   {
      constexpr auto kBitsOnStorage = 11;
      RColumnElement<float, ENTupleColumnType::kReal32Trunc> element;
      element.SetBitsOnStorage(kBitsOnStorage);
      element.Pack(nullptr, nullptr, 0);
      element.Unpack(nullptr, nullptr, 0);

      float f1 = 992.f;
      unsigned char out[BitPacking::MinBufSize(1, kBitsOnStorage)];
      element.Pack(out, &f1, 1);

      float f2;
      element.Unpack(&f2, out, 1);
      // Dropping all but first 11 bits:
      // 0x44780000 -> 0x44600000
      // 992.f       -> 896.f
      EXPECT_EQ(f2, 896.f);

      // NOTE: 0b000'0000'0011, 0b000'0000'0111, 0b000'0000'1111, ...
      float f[5] = {4.408104e-39, 1.0285575e-38, -2.2040519e-38, 8.8162076e-38, 1.4105932e-36};
      // ... truncated to: 0b000'0000'0010, 0b000'0000'0110, 0b000'0000'1110, ...
      const float expf[5] = {2.938736e-39, 8.816207e-39, -2.0571151e-38, 8.2284604e-38, 1.3165537e-36};
      unsigned char out2[BitPacking::MinBufSize(5, kBitsOnStorage)];
      element.Pack(out2, f, 5);

      float fout[5];
      element.Unpack(fout, out2, 5);
      for (int i = 0; i < 5; ++i)
         EXPECT_EQ(fout[i], expf[i]);
   }

   {
      constexpr auto kBitsOnStorage = 31;
      RColumnElement<float, ENTupleColumnType::kReal32Trunc> element;
      element.SetBitsOnStorage(kBitsOnStorage);
      element.Pack(nullptr, nullptr, 0);
      element.Unpack(nullptr, nullptr, 0);

      float f1 = 2.126f;
      unsigned char out[BitPacking::MinBufSize(1, kBitsOnStorage)];
      element.Pack(out, &f1, 1);

      float f2;
      element.Unpack(&f2, out, 1);
      // 2.126f has a 1 in the 30th bit of the mantissa, we should have preserved it.
      EXPECT_EQ(f2, 2.126f);

      constexpr auto N = 10000;
      float f[N];
      for (int i = 0; i < N; ++i)
         f[i] = -2097176.7f;
      auto out2 = MakeUninitArray<unsigned char>(BitPacking::MinBufSize(N, kBitsOnStorage));
      element.Pack(out2.get(), f, N);

      float fout[N];
      element.Unpack(fout, out2.get(), N);
      for (int i = 0; i < N; ++i) {
         EXPECT_EQ(fout[i], -2097176.5f); // dropped last bit of mantissa
         if (fout[i] != -2097176.5f)      // prevent spamming
            break;
      }
   }

   {
      constexpr auto kBitsOnStorage = 18;
      constexpr auto N = 1000;
      RColumnElement<float, ENTupleColumnType::kReal32Trunc> element;
      element.SetBitsOnStorage(kBitsOnStorage);
      float f[N];
      for (int i = 0; i < N; ++i)
         f[i] = 2.f + (0.000001f * i);

      auto out = MakeUninitArray<unsigned char>(BitPacking::MinBufSize(N, kBitsOnStorage));
      element.Pack(out.get(), f, N);

      float fout[N];
      element.Unpack(fout, out.get(), N);
      for (int i = 0; i < N; ++i) {
         EXPECT_EQ(fout[i], 2.f);
         if (fout[i] != 2.f) // prevent spamming
            break;
      }
   }

   // Exhaustively test, for all valid bit widths, packing and unpacking of 0 to N random floats.
   std::uniform_real_distribution<float> dist(-1000000, 1000000);
   const auto &[minBits, maxBits] = RColumnElementBase::GetValidBitRange(ENTupleColumnType::kReal32Trunc);
   for (int bitWidth = minBits; bitWidth <= maxBits; ++bitWidth) {
      RColumnElement<float, ENTupleColumnType::kReal32Trunc> element;
      element.SetBitsOnStorage(bitWidth);

      std::default_random_engine rng(bitWidth);
      constexpr auto N = 2000;
      float inputs[N];
      for (int i = 0; i < N; ++i) {
         inputs[i] = dist(rng);
      }

      auto packed = MakeUninitArray<std::uint8_t>(BitPacking::MinBufSize(N, bitWidth));

      float outputs[N];
      for (int i = 0; i < N; ++i) {
         element.Pack(packed.get(), inputs, i);
         element.Unpack(outputs, packed.get(), i);
         for (int j = 0; j < i; ++j) {
            int exponent;
            std::frexp(inputs[j], &exponent);
            const int scale = std::pow(2, exponent);
            const float maxErr = scale * std::pow(2.f, 10 - bitWidth);
            EXPECT_NEAR(outputs[j], inputs[j], maxErr);
         }
      }
   }
}

TEST(Packing, Real32TruncDouble)
{
   namespace BitPacking = ROOT::Internal::BitPacking;
   {
      constexpr auto kBitsOnStorage = 10;
      RColumnElement<double, ENTupleColumnType::kReal32Trunc> element;
      element.SetBitsOnStorage(kBitsOnStorage);
      element.Pack(nullptr, nullptr, 0);
      element.Unpack(nullptr, nullptr, 0);

      double f1 = 3.5f;
      unsigned char out[BitPacking::MinBufSize(1, kBitsOnStorage)];
      element.Pack(out, &f1, 1);

      double f2;
      element.Unpack(&f2, out, 1);
      // Dropping all but first 10 bits:
      // 0x40600000 -> 0x40400000
      // 3.5f       -> 3.f
      EXPECT_FLOAT_EQ(f2, 3.f);

      double f[5] = {3.5f, 3.5f, 3.5f, 3.5f, 3.5f};
      unsigned char out2[BitPacking::MinBufSize(5, kBitsOnStorage)];
      element.Pack(out2, f, 5);

      double fout[5];
      element.Unpack(fout, out2, 5);
      for (int i = 0; i < 5; ++i)
         EXPECT_FLOAT_EQ(fout[i], 3.f);

      // verify that Pack() doesn't write past the end of the valid buffer
      unsigned char outExtra[BitPacking::MinBufSize(5, kBitsOnStorage) + 4];
      outExtra[BitPacking::MinBufSize(5, kBitsOnStorage)] = 44;
      outExtra[BitPacking::MinBufSize(5, kBitsOnStorage) + 1] = 11;
      outExtra[BitPacking::MinBufSize(5, kBitsOnStorage) + 2] = 22;
      outExtra[BitPacking::MinBufSize(5, kBitsOnStorage) + 3] = 33;
      element.Pack(outExtra, f, 5);

      EXPECT_FLOAT_EQ(outExtra[BitPacking::MinBufSize(5, kBitsOnStorage)], 44);
      EXPECT_FLOAT_EQ(outExtra[BitPacking::MinBufSize(5, kBitsOnStorage) + 1], 11);
      EXPECT_FLOAT_EQ(outExtra[BitPacking::MinBufSize(5, kBitsOnStorage) + 2], 22);
      EXPECT_FLOAT_EQ(outExtra[BitPacking::MinBufSize(5, kBitsOnStorage) + 3], 33);
   }

   {
      constexpr auto kBitsOnStorage = 11;
      RColumnElement<double, ENTupleColumnType::kReal32Trunc> element;
      element.SetBitsOnStorage(kBitsOnStorage);
      element.Pack(nullptr, nullptr, 0);
      element.Unpack(nullptr, nullptr, 0);

      double f1 = 992.f;
      unsigned char out[BitPacking::MinBufSize(1, kBitsOnStorage)];
      element.Pack(out, &f1, 1);

      double f2;
      element.Unpack(&f2, out, 1);
      // Dropping all but first 11 bits:
      // 0x44780000 -> 0x44600000
      // 992.f       -> 896.f
      EXPECT_FLOAT_EQ(f2, 896.f);

      // NOTE: 0b000'0000'0011, 0b000'0000'0111, 0b000'0000'1111, ...
      double f[5] = {4.408104e-39, 1.0285575e-38, -2.2040519e-38, 8.8162076e-38, 1.4105932e-36};
      // ... truncated to: 0b000'0000'0010, 0b000'0000'0110, 0b000'0000'1110, ...
      const double expf[5] = {2.938736e-39, 8.816207e-39, -2.0571151e-38, 8.2284604e-38, 1.3165537e-36};
      unsigned char out2[BitPacking::MinBufSize(5, kBitsOnStorage)];
      element.Pack(out2, f, 5);

      double fout[5];
      element.Unpack(fout, out2, 5);
      for (int i = 0; i < 5; ++i)
         EXPECT_FLOAT_EQ(fout[i], expf[i]);
   }

   {
      constexpr auto kBitsOnStorage = 31;
      RColumnElement<double, ENTupleColumnType::kReal32Trunc> element;
      element.SetBitsOnStorage(kBitsOnStorage);
      element.Pack(nullptr, nullptr, 0);
      element.Unpack(nullptr, nullptr, 0);

      double f1 = 2.126f;
      unsigned char out[BitPacking::MinBufSize(1, kBitsOnStorage)];
      element.Pack(out, &f1, 1);

      double f2;
      element.Unpack(&f2, out, 1);
      // 2.126f has a 1 in the 30th bit of the mantissa, we should have preserved it.
      EXPECT_FLOAT_EQ(f2, 2.126f);

      constexpr auto N = 10000;
      double f[N];
      for (int i = 0; i < N; ++i)
         f[i] = -2097176.7f;
      auto out2 = MakeUninitArray<unsigned char>(BitPacking::MinBufSize(N, kBitsOnStorage));
      element.Pack(out2.get(), f, N);

      double fout[N];
      element.Unpack(fout, out2.get(), N);
      for (int i = 0; i < N; ++i) {
         EXPECT_FLOAT_EQ(fout[i], -2097176.5f); // dropped last bit of mantissa
         if (fout[i] != -2097176.5f)            // prevent spamming
            break;
      }
   }

   {
      constexpr auto kBitsOnStorage = 18;
      constexpr auto N = 1000;
      RColumnElement<double, ENTupleColumnType::kReal32Trunc> element;
      element.SetBitsOnStorage(kBitsOnStorage);
      double f[N];
      for (int i = 0; i < N; ++i)
         f[i] = 2.f + (0.000001f * i);

      auto out = MakeUninitArray<unsigned char>(BitPacking::MinBufSize(N, kBitsOnStorage));
      element.Pack(out.get(), f, N);

      double fout[N];
      element.Unpack(fout, out.get(), N);
      for (int i = 0; i < N; ++i) {
         EXPECT_FLOAT_EQ(fout[i], 2.f);
         if (fout[i] != 2.f) // prevent spamming
            break;
      }
   }

   // Exhaustively test, for all valid bit widths, packing and unpacking of 0 to N random doubles.
   std::uniform_real_distribution<double> dist(-1000000, 1000000);
   const auto &[minBits, maxBits] = RColumnElementBase::GetValidBitRange(ENTupleColumnType::kReal32Trunc);
   for (int bitWidth = minBits; bitWidth <= maxBits; ++bitWidth) {
      RColumnElement<double, ENTupleColumnType::kReal32Trunc> element;
      element.SetBitsOnStorage(bitWidth);

      std::default_random_engine rng(bitWidth);
      constexpr auto N = 2000;
      double inputs[N];
      for (int i = 0; i < N; ++i) {
         inputs[i] = dist(rng);
      }

      auto packed = MakeUninitArray<std::uint8_t>(BitPacking::MinBufSize(N, bitWidth));

      double outputs[N];
      for (int i = 0; i < N; ++i) {
         element.Pack(packed.get(), inputs, i);
         element.Unpack(outputs, packed.get(), i);
         for (int j = 0; j < i; ++j) {
            int exponent;
            std::frexp(inputs[j], &exponent);
            const int scale = std::pow(2, exponent);
            const double maxErr = scale * std::pow(2.f, 10 - bitWidth);
            EXPECT_NEAR(outputs[j], inputs[j], maxErr);
         }
      }
   }
}

TEST(Packing, RealQuantize)
{
   using namespace Quantize;

   {
      float fs[5] = {1.f, 2.f, 3.f, 4.f, 5.f};
      Quantized_t qs[5];
      QuantizeReals(qs, fs, 5, 1., 5., 16);

      float fuqs[5];
      UnquantizeReals(fuqs, qs, 5, 1., 5., 16);
      EXPECT_NEAR(fuqs[0], 1.f, 0.001f);
      EXPECT_NEAR(fuqs[1], 2.f, 0.001f);
      EXPECT_NEAR(fuqs[2], 3.f, 0.001f);
      EXPECT_NEAR(fuqs[3], 4.f, 0.001f);
      EXPECT_NEAR(fuqs[4], 5.f, 0.001f);
   }

   {
      float fs[3] = {0.5f, 1.f, 1.5f};
      Quantized_t qs[3];
      QuantizeReals(qs, fs, 3, 0.5f, 1.5f, 3);

      float fuqs[3];
      UnquantizeReals(fuqs, qs, 3, 0.5f, 1.5f, 3);
      EXPECT_FLOAT_EQ(fuqs[0], 0.5f);
      EXPECT_NEAR(fuqs[1], 1.f, 0.1f);
      EXPECT_FLOAT_EQ(fuqs[2], 1.5f);
   }

   {
      std::default_random_engine rng{42};
      double min = -250, max = 500;
      std::uniform_real_distribution<decltype(min)> dist{min, max};

      constexpr auto N = 10000;
      constexpr auto kNbits = 20;
      auto inputs = MakeUninitArray<decltype(min)>(N);
      for (int i = 0; i < N; ++i)
         inputs.get()[i] = dist(rng);

      auto quant = MakeUninitArray<Quantized_t>(N);
      QuantizeReals(quant.get(), inputs.get(), N, min, max, kNbits);

      auto unquant = MakeUninitArray<decltype(min)>(N);
      UnquantizeReals(unquant.get(), quant.get(), N, min, max, kNbits);

      for (int i = 0; i < N; ++i)
         EXPECT_NEAR(inputs.get()[i], unquant.get()[i], 0.001);
   }

   {
      std::default_random_engine rng{1337};
      float min = 0, max = 1;
      std::uniform_real_distribution<decltype(min)> dist{min, max};

      constexpr auto N = 10000;
      constexpr auto kNbits = 8;
      auto inputs = MakeUninitArray<decltype(min)>(N);
      for (int i = 0; i < N; ++i)
         inputs.get()[i] = dist(rng);

      auto quant = MakeUninitArray<Quantized_t>(N);
      QuantizeReals(quant.get(), inputs.get(), N, min, max, kNbits);

      auto unquant = MakeUninitArray<decltype(min)>(N);
      UnquantizeReals(unquant.get(), quant.get(), N, min, max, kNbits);

      for (int i = 0; i < N; ++i)
         EXPECT_NEAR(inputs.get()[i], unquant.get()[i], 0.01);
   }
}

TEST(Packing, Real32QuantFloat)
{
   namespace BitPacking = ROOT::Internal::BitPacking;
   {
      constexpr auto kBitsOnStorage = 10;
      RColumnElement<float, ENTupleColumnType::kReal32Quant> element;
      element.SetBitsOnStorage(kBitsOnStorage);
      element.SetValueRange(-5.f, 5.f);
      element.Pack(nullptr, nullptr, 0);
      element.Unpack(nullptr, nullptr, 0);

      float f1 = 3.5f;
      unsigned char out[BitPacking::MinBufSize(1, kBitsOnStorage)];
      element.Pack(out, &f1, 1);

      float f2;
      element.Unpack(&f2, out, 1);
      EXPECT_NEAR(f2, 3.5f, 0.01f);

      float f[7] = {-4.5f, -3.f, -2.55f, 0.f, 1.f, 3.f, 5.f};
      unsigned char out2[BitPacking::MinBufSize(7, kBitsOnStorage)];
      element.Pack(out2, f, 7);

      float fout[7];
      element.Unpack(fout, out2, 7);
      for (int i = 0; i < 7; ++i)
         EXPECT_NEAR(fout[i], f[i], 0.01f);
   }

   {
      RColumnElement<float, ENTupleColumnType::kReal32Quant> element;
      element.SetBitsOnStorage(20);
      element.SetValueRange(-10.f, 10.f);

      float f[5] = {3.4f, 5.f, -6.f, 10.f, -10.f};
      unsigned char out[BitPacking::MinBufSize(std::size(f), 20)];
      element.Pack(out, f, std::size(f));
      float f2[std::size(f)];
      element.Unpack(&f2, out, std::size(f));
      for (size_t i = 0; i < std::size(f); ++i)
         EXPECT_NEAR(f[i], f2[i], 0.01f);

      f[3] = 11.f;
      // should throw out of range
      EXPECT_THROW(element.Pack(out, f, std::size(f)), ROOT::RException);
   }

   {
      constexpr auto kBitsOnStorage = 1;
      RColumnElement<float, ENTupleColumnType::kReal32Quant> element;
      element.SetBitsOnStorage(kBitsOnStorage);
      element.SetValueRange(-10.f, 10.f);

      float f1 = -10.f;
      unsigned char out[BitPacking::MinBufSize(1, kBitsOnStorage)];
      element.Pack(out, &f1, 1);
      float f2;
      element.Unpack(&f2, out, 1);
      EXPECT_EQ(f2, -10.f);

      f1 = 10.f;
      element.Pack(out, &f1, 1);
      element.Unpack(&f2, out, 1);
      EXPECT_EQ(f2, 10.f);
   }

   {
      constexpr auto kBitsOnStorage = 32;
      RColumnElement<float, ENTupleColumnType::kReal32Quant> element;
      element.SetBitsOnStorage(kBitsOnStorage);
      element.SetValueRange(-10.f, 10.f);

      float f1 = -5.f;
      unsigned char out[BitPacking::MinBufSize(1, kBitsOnStorage)];
      element.Pack(out, &f1, 1);
      float f2;
      element.Unpack(&f2, out, 1);
      EXPECT_FLOAT_EQ(f2, -5.f);
   }

   // Exhaustively test, for all valid bit widths, packing and unpacking of 0 to N random floats.
   constexpr double kMin = -1000, kMax = 1000;
   std::uniform_real_distribution<float> dist(kMin, kMax);
   const auto &[minBits, maxBits] = RColumnElementBase::GetValidBitRange(ENTupleColumnType::kReal32Quant);
   for (int bitWidth = minBits; bitWidth <= maxBits; ++bitWidth) {
      SCOPED_TRACE(std::string("At bitWidth = ") + std::to_string(bitWidth));

      RColumnElement<float, ENTupleColumnType::kReal32Quant> element;
      element.SetBitsOnStorage(bitWidth);
      element.SetValueRange(kMin, kMax);

      std::default_random_engine rng(bitWidth);
      constexpr auto N = 2000;
      float inputs[N];
      for (int i = 0; i < N; ++i) {
         inputs[i] = dist(rng);
      }

      auto packed = MakeUninitArray<std::uint8_t>(BitPacking::MinBufSize(N, bitWidth));
      float outputs[N];

      if (bitWidth == 1) {
         for (int i = 0; i < N; ++i) {
            element.Pack(packed.get(), inputs, i);
            element.Unpack(outputs, packed.get(), i);
            for (int j = 0; j < i; ++j) {
               if (inputs[j] < (kMax + kMin) * 0.5)
                  EXPECT_FLOAT_EQ(outputs[j], kMin);
               else
                  EXPECT_FLOAT_EQ(outputs[j], kMax);
            }
         }
      } else {
         const auto maxErr = (kMax - kMin) / ((1ull << bitWidth) - 1);
         for (int i = 0; i < N; ++i) {
            element.Pack(packed.get(), inputs, i);
            element.Unpack(outputs, packed.get(), i);
            for (int j = 0; j < i; ++j) {
               EXPECT_NEAR(outputs[j], inputs[j], maxErr);
            }
         }
      }
   }
}

TEST(Packing, Real32QuantFloatCornerCases)
{
   namespace BitPacking = ROOT::Internal::BitPacking;

   static constexpr double pi = 3.141592653589793;
   {
      // Passing a double-precision value as min, max to a RField<float>.
      // The min and max values should be converted to float internally and,
      // when deserializing, we should get back (float)min and (float)max.
      constexpr auto kBitsOnStorage = 32;

      unsigned char out[BitPacking::MinBufSize(2, kBitsOnStorage)];
      {
         RColumnElement<float, ENTupleColumnType::kReal32Quant> element;
         element.SetBitsOnStorage(kBitsOnStorage);
         element.SetValueRange((float)-pi, (float)pi);

         float extremesFloatIn[2] = {-pi, pi};
         element.Pack(out, extremesFloatIn, 2);
         float extremesFloatOut[2];
         element.Unpack(extremesFloatOut, out, 2);
         // Must be exactly the same
         EXPECT_EQ(extremesFloatOut[0], (float)-pi);
         EXPECT_EQ(extremesFloatOut[1], (float)pi);
      }

      {
         RColumnElement<double, ENTupleColumnType::kReal32Quant> elementDouble;
         elementDouble.SetBitsOnStorage(kBitsOnStorage);
         elementDouble.SetValueRange(-pi, pi);

         double extremesDoubleOut[2] = {-pi, pi};
         elementDouble.Unpack(extremesDoubleOut, out, 2);
         EXPECT_EQ(extremesDoubleOut[0], -pi);
         const double eps = std::numeric_limits<double>::epsilon();
         const double maxErr = std::max(1.0, std::abs(pi)) * eps;
         // in double precision the value unquantized from the max is not guaranteed to be exactly the same as the given
         // max value in the range, due to floating point error propagating when scaling it back.
         EXPECT_NEAR(extremesDoubleOut[1], pi, maxErr);
         EXPECT_LE(extremesDoubleOut[1], pi);
      }
   }

   {
      FileRaii fileGuard("test_ntuple_packing_quant_float_cornercase.root");

      // Same as above but through a RField.
      {
         auto model = RNTupleModel::Create();
         auto field = std::make_unique<RField<float>>("f");
         field->SetQuantized(-pi, pi, 32);
         model->AddField(std::move(field));
         auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
         auto pf = writer->GetModel().GetDefaultEntry().GetPtr<float>("f");
         *pf = -pi; // gets truncated to float!
         writer->Fill();
         *pf = pi; // gets truncated to float!
         writer->Fill();
      }
      {
         auto model = RNTupleModel::Create();
         // impose a model with a double field instead of float
         auto field = model->MakeField<double>("f");
         auto reader = RNTupleReader::Open(std::move(model), "ntpl", fileGuard.GetPath());
         auto viewF = reader->GetView<double>("f");
         EXPECT_EQ(static_cast<float>(viewF(0)), static_cast<float>(-pi));
         EXPECT_EQ(static_cast<float>(viewF(1)), static_cast<float>(pi));
      }
   }

   {
      FileRaii fileGuard("test_ntuple_packing_quant_float_cornercase.root");

      // Same as above but swapping float and double (write as double, read as float).
      {
         auto model = RNTupleModel::Create();
         auto field = std::make_unique<RField<double>>("f");
         field->SetQuantized(-pi, pi, 32);
         model->AddField(std::move(field));
         auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
         auto pf = writer->GetModel().GetDefaultEntry().GetPtr<double>("f");
         *pf = -pi;
         writer->Fill();
         *pf = pi;
         writer->Fill();
      }
      {
         auto model = RNTupleModel::Create();
         auto field = model->MakeField<float>("f");
         auto reader = RNTupleReader::Open(std::move(model), "ntpl", fileGuard.GetPath());
         auto viewF = reader->GetModel().GetDefaultEntry().GetPtr<float>("f");
         reader->LoadEntry(0);
         EXPECT_EQ(*viewF, static_cast<float>(-pi));
         reader->LoadEntry(1);
         EXPECT_EQ(*viewF, static_cast<float>(pi));
      }
   }

   {
      // Verify that we can handle ranges where the internal quantized precision
      // is higher than the floating point precision at that interval (meaning there
      // are multiple different quantized values that map to the same float).
      constexpr auto kBitsOnStorage = 32;

      unsigned char out[BitPacking::MinBufSize(2, kBitsOnStorage)];
      RColumnElement<float, ENTupleColumnType::kReal32Quant> element;
      element.SetBitsOnStorage(kBitsOnStorage);
      element.SetValueRange(100, 110);

      float extremesFloatIn[2] = {100, 110};
      element.Pack(out, extremesFloatIn, 2);
      float extremesFloatOut[2];
      element.Unpack(extremesFloatOut, out, 2);
      // Must be exactly the same
      EXPECT_EQ(extremesFloatOut[0], 100);
      EXPECT_EQ(extremesFloatOut[1], 110);
   }
}

TEST(Packing, Real32QuantDouble)
{
   namespace BitPacking = ROOT::Internal::BitPacking;
   {
      constexpr auto kBitsOnStorage = 10;
      RColumnElement<double, ENTupleColumnType::kReal32Quant> element;
      element.SetBitsOnStorage(kBitsOnStorage);
      element.SetValueRange(-5.f, 5.f);
      element.Pack(nullptr, nullptr, 0);
      element.Unpack(nullptr, nullptr, 0);

      double f1 = 3.5f;
      unsigned char out[BitPacking::MinBufSize(1, kBitsOnStorage)];
      element.Pack(out, &f1, 1);

      double f2;
      element.Unpack(&f2, out, 1);
      EXPECT_NEAR(f2, 3.5f, 0.01f);

      double f[7] = {-4.5f, -3.f, -2.55f, 0.f, 1.f, 3.f, 5.f};
      unsigned char out2[BitPacking::MinBufSize(7, kBitsOnStorage)];
      element.Pack(out2, f, 7);

      double fout[7];
      element.Unpack(fout, out2, 7);
      for (int i = 0; i < 7; ++i)
         EXPECT_NEAR(fout[i], f[i], 0.01f);
   }

   {
      RColumnElement<double, ENTupleColumnType::kReal32Quant> element;
      element.SetBitsOnStorage(20);
      element.SetValueRange(-10.f, 10.f);

      double f[5] = {3.4f, 5.f, -6.f, 10.f, -10.f};
      unsigned char out[BitPacking::MinBufSize(std::size(f), 20)];
      element.Pack(out, f, std::size(f));
      double f2[std::size(f)];
      element.Unpack(&f2, out, std::size(f));
      for (size_t i = 0; i < std::size(f); ++i)
         EXPECT_NEAR(f[i], f2[i], 0.01f);

      f[3] = 11.f;
      // should throw out of range
      EXPECT_THROW(element.Pack(out, f, std::size(f)), ROOT::RException);
   }

   {
      constexpr auto kBitsOnStorage = 1;
      RColumnElement<double, ENTupleColumnType::kReal32Quant> element;
      element.SetBitsOnStorage(kBitsOnStorage);
      element.SetValueRange(-10.f, 10.f);

      double f1 = -10.f;
      unsigned char out[BitPacking::MinBufSize(1, kBitsOnStorage)];
      element.Pack(out, &f1, 1);
      double f2;
      element.Unpack(&f2, out, 1);
      EXPECT_EQ(f2, -10.f);

      f1 = 10.f;
      element.Pack(out, &f1, 1);
      element.Unpack(&f2, out, 1);
      EXPECT_EQ(f2, 10.f);
   }

   {
      constexpr auto kBitsOnStorage = 32;
      RColumnElement<double, ENTupleColumnType::kReal32Quant> element;
      element.SetBitsOnStorage(kBitsOnStorage);
      element.SetValueRange(-10.f, 10.f);

      double f1 = -5.f;
      unsigned char out[BitPacking::MinBufSize(1, kBitsOnStorage)];
      element.Pack(out, &f1, 1);
      double f2;
      element.Unpack(&f2, out, 1);
      EXPECT_FLOAT_EQ(f2, -5.f);
   }

   // Exhaustively test, for all valid bit widths, packing and unpacking of 0 to N random doubles.
   constexpr double kMin = -1000, kMax = 1000;
   std::uniform_real_distribution<double> dist(kMin, kMax);
   const auto &[minBits, maxBits] = RColumnElementBase::GetValidBitRange(ENTupleColumnType::kReal32Quant);
   for (int bitWidth = minBits; bitWidth <= maxBits; ++bitWidth) {
      SCOPED_TRACE(std::string("At bitWidth = ") + std::to_string(bitWidth));

      RColumnElement<double, ENTupleColumnType::kReal32Quant> element;
      element.SetBitsOnStorage(bitWidth);
      element.SetValueRange(kMin, kMax);

      std::default_random_engine rng(bitWidth);
      constexpr auto N = 2000;
      double inputs[N];
      for (int i = 0; i < N; ++i) {
         inputs[i] = dist(rng);
      }

      auto packed = MakeUninitArray<std::uint8_t>(BitPacking::MinBufSize(N, bitWidth));
      double outputs[N];

      if (bitWidth == 1) {
         for (int i = 0; i < N; ++i) {
            element.Pack(packed.get(), inputs, i);
            element.Unpack(outputs, packed.get(), i);
            for (int j = 0; j < i; ++j) {
               if (inputs[j] < (kMax + kMin) * 0.5)
                  EXPECT_FLOAT_EQ(outputs[j], kMin);
               else
                  EXPECT_FLOAT_EQ(outputs[j], kMax);
            }
         }
      } else {
         const auto maxErr = (kMax - kMin) / ((1ull << bitWidth) - 1);
         for (int i = 0; i < N; ++i) {
            element.Pack(packed.get(), inputs, i);
            element.Unpack(outputs, packed.get(), i);
            for (int j = 0; j < i; ++j) {
               EXPECT_NEAR(outputs[j], inputs[j], maxErr);
            }
         }
      }
   }
}
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
