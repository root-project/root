#include "ntuple_test.hxx"

#include <optional>

TEST(RNTuple, Variant)
{
   FileRaii fileGuard("test_ntuple_variant.root");

   auto modelWrite = RNTupleModel::Create();
   auto wrVariant = modelWrite->MakeField<std::variant<double, int>>("variant");
   *wrVariant = 2.0;

   modelWrite->Freeze();
   auto modelRead = std::unique_ptr<RNTupleModel>(modelWrite->Clone());

   {
      auto writer = RNTupleWriter::Recreate(std::move(modelWrite), "myNTuple", fileGuard.GetPath());
      writer->Fill();
      writer->CommitCluster();
      *wrVariant = 4;
      writer->Fill();
      *wrVariant = 8.0;
      writer->Fill();
   }
   auto rdVariant = modelRead->GetDefaultEntry().GetPtr<std::variant<double, int>>("variant").get();

   auto reader = RNTupleReader::Open(std::move(modelRead), "myNTuple", fileGuard.GetPath());
   EXPECT_EQ(3U, reader->GetNEntries());

   reader->LoadEntry(0);
   EXPECT_EQ(2.0, *std::get_if<double>(rdVariant));
   reader->LoadEntry(1);
   EXPECT_EQ(4, *std::get_if<int>(rdVariant));
   reader->LoadEntry(2);
   EXPECT_EQ(8.0, *std::get_if<double>(rdVariant));
}

TEST(RNTuple, VariantSizeAlignment)
{
   using CharArray3_t = std::array<char, 3>;
   using CharArray4_t = std::array<char, 4>;
   using CharArray5_t = std::array<char, 5>;
   using VariantOfOptional_t = std::variant<std::optional<int>, float>;
   using VariantOfVariant_t = std::variant<std::variant<int, float>, float>;

   EXPECT_EQ(sizeof(std::variant<char>), RField<std::variant<char>>("f").GetValueSize());
   EXPECT_EQ(alignof(std::variant<char>), RField<std::variant<char>>("f").GetAlignment());
   EXPECT_EQ(sizeof(std::variant<CharArray3_t>), RField<std::variant<CharArray3_t>>("f").GetValueSize());
   EXPECT_EQ(alignof(std::variant<CharArray3_t>), RField<std::variant<CharArray3_t>>("f").GetAlignment());
   EXPECT_EQ(sizeof(std::variant<CharArray4_t>), RField<std::variant<CharArray4_t>>("f").GetValueSize());
   EXPECT_EQ(alignof(std::variant<CharArray4_t>), RField<std::variant<CharArray4_t>>("f").GetAlignment());
   EXPECT_EQ(sizeof(std::variant<CharArray5_t>), RField<std::variant<CharArray5_t>>("f").GetValueSize());
   EXPECT_EQ(alignof(std::variant<CharArray5_t>), RField<std::variant<CharArray5_t>>("f").GetAlignment());
   EXPECT_EQ(sizeof(std::variant<float>), RField<std::variant<float>>("f").GetValueSize());
   EXPECT_EQ(alignof(std::variant<float>), RField<std::variant<float>>("f").GetAlignment());
   EXPECT_EQ(sizeof(std::variant<double>), RField<std::variant<double>>("f").GetValueSize());
   EXPECT_EQ(alignof(std::variant<double>), RField<std::variant<double>>("f").GetAlignment());
   EXPECT_EQ(sizeof(std::variant<CustomStruct>), RField<std::variant<CustomStruct>>("f").GetValueSize());
   EXPECT_EQ(alignof(std::variant<CustomStruct>), RField<std::variant<CustomStruct>>("f").GetAlignment());
   EXPECT_EQ(sizeof(VariantOfOptional_t), RField<VariantOfOptional_t>("f").GetValueSize());
   EXPECT_EQ(alignof(VariantOfOptional_t), RField<VariantOfOptional_t>("f").GetAlignment());
   EXPECT_EQ(sizeof(VariantOfVariant_t), RField<VariantOfVariant_t>("f").GetValueSize());
   EXPECT_EQ(alignof(VariantOfVariant_t), RField<VariantOfVariant_t>("f").GetAlignment());
}

TEST(RNTuple, VariantLimits)
{
   // clang-format off
   EXPECT_THROW(RFieldBase::Create("f",
      "std::variant<char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,"
      "             char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,"
      "             char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,"
      "             char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,"
      "             char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,"
      "             char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,"
      "             char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,"
      "             char,char,char,char,char,char,char,char,char,char,char,char,char,char>"),
      ROOT::RException);

   using HugeVariant_t =
      std::variant<char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,
                   char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,
                   char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,
                   char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,
                   char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,
                   char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,
                   char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,char,
                   char,char,char,char,char,char,char,char,char,char,char,char,bool>;
   // clang-format on

   FileRaii fileGuard("test_ntuple_variant_limits.root");
   {
      auto model = RNTupleModel::Create();
      auto ptrV = model->MakeField<HugeVariant_t>("v");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrV = true;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto ptrV = reader->GetModel().GetDefaultEntry().GetPtr<HugeVariant_t>("v");
   EXPECT_EQ(1u, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_EQ(124, ptrV->index());
   EXPECT_TRUE(std::get<124>(*ptrV));
}

TEST(RNTuple, VariantException)
{
   FileRaii fileGuard("test_ntuple_variant_exception.root");
   {
      auto model = RNTupleModel::Create();
      auto ptrV = model->MakeField<std::variant<std::string, ThrowForVariant>>("v");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrV = "str";
      writer->Fill();

      try {
         *ptrV = ThrowForVariant();
         FAIL() << "setting varant to ThrowForVariant show throw";
      } catch (const std::runtime_error &) {
      }
      EXPECT_TRUE(ptrV->valueless_by_exception());

      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto ptrV = reader->GetModel().GetDefaultEntry().GetPtr<std::variant<std::string, ThrowForVariant>>("v");
   EXPECT_EQ(2u, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_STREQ("str", std::get<0>(*ptrV).c_str());
   reader->LoadEntry(1);
   EXPECT_TRUE(ptrV->valueless_by_exception());
}

TEST(RNTuple, VariantComplex)
{
   FileRaii fileGuard("test_ntuple_variant_complex.root");

   {
      auto model = RNTupleModel::Create();
      auto ptrVec1 = model->MakeField<std::vector<std::variant<std::optional<int>, float>>>("v1");
      auto ptrVec2 = model->MakeField<std::vector<std::variant<std::variant<int, float>, float>>>("v2");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      for (int i = 0; i < 10; i++) {
         ptrVec1->clear();
         ptrVec2->clear();

         for (int j = 0; j < 5; ++j) {
            std::variant<std::optional<int>, float> var1;
            std::variant<std::variant<int, float>, float> var2;
            if (j % 2 == 0) {
               if (j % 4 == 0) {
                  var1 = std::optional<int>();
                  var2 = std::variant<int, float>(int(1));
               } else {
                  var1 = std::optional<int>(42);
                  var2 = std::variant<int, float>(float(2.0));
               }
            } else {
               var1 = float(1.0);
               var2 = float(3.0);
            }
            ptrVec1->emplace_back(var1);
            ptrVec2->emplace_back(var2);
         }

         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto ptrVec1 =
      reader->GetModel().GetDefaultEntry().GetPtr<std::vector<std::variant<std::optional<int>, float>>>("v1");
   auto ptrVec2 =
      reader->GetModel().GetDefaultEntry().GetPtr<std::vector<std::variant<std::variant<int, float>, float>>>("v2");
   EXPECT_EQ(10u, reader->GetNEntries());
   for (int i = 0; i < 10; i++) {
      reader->LoadEntry(i);
      EXPECT_EQ(5u, ptrVec1->size());
      EXPECT_EQ(5u, ptrVec2->size());
      for (int j = 0; j < 5; ++j) {
         EXPECT_EQ(j % 2, ptrVec1->at(j).index());
         EXPECT_EQ(j % 2, ptrVec2->at(j).index());
         if (j % 2 == 0) {
            if (j % 4 == 0) {
               EXPECT_FALSE(std::get<0>(ptrVec1->at(j)));
               EXPECT_EQ(1, std::get<0>(std::get<0>(ptrVec2->at(j))));
            } else {
               EXPECT_EQ(42, std::get<0>(ptrVec1->at(j)));
               EXPECT_FLOAT_EQ(2.0, std::get<1>(std::get<0>(ptrVec2->at(j))));
            }
         } else {
            EXPECT_FLOAT_EQ(1.0, std::get<1>(ptrVec1->at(j)));
            EXPECT_FLOAT_EQ(3.0, std::get<1>(ptrVec2->at(j)));
         }
      }
   }
}
