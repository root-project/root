#include "ntuple_test.hxx"

#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>

class RNewStringField final : public ROOT::RFieldBase {
private:
   std::uint32_t fFieldVersion = 0;
   std::uint32_t fTypeVersion = 0;

protected:
   std::unique_ptr<ROOT::RFieldBase> CloneImpl(std::string_view newName) const final
   {
      return std::make_unique<RNewStringField>(newName, GetTypeName(), fFieldVersion, fTypeVersion);
   }
   void ConstructValue(void *where) const final { new (where) std::string(); }
   std::unique_ptr<RDeleter> GetDeleter() const final { return std::make_unique<RTypedDeleter<std::string>>(); }

public:
   RNewStringField(std::string_view name, std::string_view type, std::uint32_t fieldVersion, std::uint32_t typeVersion)
      : ROOT::RFieldBase(name, type, ROOT::ENTupleStructure::kPlain, false /* isSimple */),
        fFieldVersion(fieldVersion),
        fTypeVersion(typeVersion)
   {
   }
   RNewStringField(RNewStringField &&other) = default;
   RNewStringField &operator=(RNewStringField &&other) = default;
   ~RNewStringField() override = default;

   size_t GetValueSize() const final { return sizeof(std::string); }
   size_t GetAlignment() const final { return alignof(std::string); }

   std::uint32_t GetFieldVersion() const final { return fFieldVersion; }
   std::uint32_t GetTypeVersion() const final { return fTypeVersion; }
};

TEST(RNTupleEvolution, CheckVersions)
{
   FileRaii fileGuard("test_ntuple_evolution_check_versions.root");
   {
      auto model = ROOT::RNTupleModel::Create();
      model->AddField(std::make_unique<RNewStringField>("f1", "std::string", 137, 0));
      model->AddField(std::make_unique<RNewStringField>("f2", "std::string", 0, 138));
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(0, reader->GetNEntries());

   try {
      reader->GetView<std::string>("f1");
      FAIL() << "non-matching field versions should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("field version 0 vs. 137"));
   }

   try {
      reader->GetView<std::string>("f2");
      FAIL() << "non-matching type versions should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("type version 0 vs. 138"));
   }
}

TEST(RNTupleEvolution, CheckStructure)
{
   FileRaii fileGuard("test_ntuple_evolution_check_structure.root");
   {
      auto model = ROOT::RNTupleModel::Create();
      model->AddField(std::make_unique<ROOT::RVectorField>("f", std::make_unique<RField<char>>("_0")));
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
   }

   auto model = ROOT::RNTupleModel::Create();
   std::vector<std::unique_ptr<RFieldBase>> itemFields;
   itemFields.emplace_back(std::make_unique<RField<char>>("_0"));
   model->AddField(std::make_unique<ROOT::RRecordField>("f", std::move(itemFields)));

   try {
      RNTupleReader::Open(std::move(model), "ntpl", fileGuard.GetPath());
      FAIL() << "non-matching structural role should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("structural role Record vs. Collection"));
   }
}

TEST(RNTupleEvolution, CheckTypeName)
{
   FileRaii fileGuard("test_ntuple_evolution_check_type_name.root");
   {
      auto model = ROOT::RNTupleModel::Create();
      model->AddField(std::make_unique<RNewStringField>("f", "std::fancy_string", 0, 0));
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   try {
      reader->GetView<std::string>("f");
      FAIL() << "non-matching type names should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible on-disk type name std::fancy_string"));
   }
}

TEST(RNTupleEvolution, CheckRepetitionCount)
{
   FileRaii fileGuard("test_ntuple_evolution_check_repetition_count.root");
   {
      auto model = ROOT::RNTupleModel::Create();
      model->MakeField<std::array<char, 2>>("f");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   try {
      reader->GetView<std::array<char, 3>>("f");
      FAIL() << "non-matching type names should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("repetition count 3 vs. 2"));
   }
}

TEST(RNTupleEvolution, CheckVariant)
{
   FileRaii fileGuard("test_ntuple_evolution_check_variant.root");
   {
      auto model = ROOT::RNTupleModel::Create();
      auto v = model->MakeField<std::variant<std::int32_t, std::vector<char>>>("f");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *v = 137;
      writer->Fill();
      *v = std::vector<char>{'R', 'O', 'O', 'T'};
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

   try {
      reader->GetView<std::variant<std::int32_t>>("f");
      FAIL() << "non-matching number of variants should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("number of variants on-disk do not match"));
   }

   try {
      reader->GetView<std::variant<std::int32_t, std::vector<char>, float>>("f");
      FAIL() << "non-matching number of variants should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("number of variants on-disk do not match"));
   }

   auto v = reader->GetView<std::variant<std::int64_t, ROOT::RVec<int>>>("f");
   EXPECT_EQ(137, std::get<std::int64_t>(v(0)));
   EXPECT_EQ('R', std::get<ROOT::RVec<int>>(v(1))[0]);
   EXPECT_EQ('O', std::get<ROOT::RVec<int>>(v(1))[1]);
   EXPECT_EQ('O', std::get<ROOT::RVec<int>>(v(1))[2]);
   EXPECT_EQ('T', std::get<ROOT::RVec<int>>(v(1))[3]);
}

TEST(RNTupleEvolution, CheckPairTuple)
{
   FileRaii fileGuard("test_ntuple_evolution_check_pair_tuple.root");
   {
      auto model = ROOT::RNTupleModel::Create();
      auto p = model->MakeField<std::pair<char, float>>("p");
      auto t2 = model->MakeField<std::tuple<char, float>>("t2");
      model->MakeField<std::tuple<char, float, float>>("t3");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      *p = {1, 2.0};
      *t2 = {3, 4.0};
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

   try {
      reader->GetView<std::tuple<char>>("p");
      FAIL() << "non-matching number of subfields for pair/tuple should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid number of on-disk subfields"));
   }

   try {
      reader->GetView<std::tuple<char, float, float>>("p");
      FAIL() << "non-matching number of subfields for pair/tuple should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid number of on-disk subfields"));
   }

   try {
      reader->GetView<std::pair<char, float>>("t3");
      FAIL() << "non-matching number of subfields for pair/tuple should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid number of on-disk subfields"));
   }

   auto t2 = reader->GetView<std::pair<char, double>>("t2");
   auto p = reader->GetView<std::tuple<int, double>>("p");
   EXPECT_EQ(3, t2(0).first);
   EXPECT_DOUBLE_EQ(4.0, t2(0).second);
   EXPECT_EQ(1, std::get<0>(p(0)));
   EXPECT_DOUBLE_EQ(2.0, std::get<1>(p(0)));
}
