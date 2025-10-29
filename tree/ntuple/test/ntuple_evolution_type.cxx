#include "ntuple_test.hxx"
#include "SimpleCollectionProxy.hxx"
#include "STLContainerEvolution.hxx"

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

TEST(RNTupleEvolution, Enum)
{
   FileRaii fileGuard("test_ntuple_evolution_check_enum.root");
   {
      auto model = ROOT::RNTupleModel::Create();
      auto e1 = model->MakeField<CustomEnumInt8>("e1");
      auto e2 = model->MakeField<CustomEnum>("e2");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      *e1 = static_cast<CustomEnumInt8>(42);
      *e2 = static_cast<CustomEnum>(kCustomEnumVal);

      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto ve1 = reader->GetView<int>("e1");
   auto ve2 = reader->GetView<int>("e2");
   auto ve3 = reader->GetView<RenamedCustomEnum>("e2");
   EXPECT_EQ(42, ve1(0));
   EXPECT_EQ(7, ve2(0));
   EXPECT_EQ(kRenamedCustomEnumVal, ve3(0));
}

TEST(RNTupleEvolution, CheckAtomic)
{
   // TODO(jblomer): enable test with CustomAtomicNotLockFree once linking of libatomic is sorted out.

   FileRaii fileGuard("test_ntuple_evolution_check_atomic.root");
   {
      auto model = ROOT::RNTupleModel::Create();
      auto atomicInt = model->MakeField<std::atomic<std::int32_t>>("atomicInt");
      auto regularInt = model->MakeField<std::int32_t>("regularInt");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      *atomicInt = 7;
      *regularInt = 13;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

   auto v1 = reader->GetView<std::atomic<std::int64_t>>("atomicInt");
   auto v2 = reader->GetView<std::atomic<std::int64_t>>("regularInt");
   auto v3 = reader->GetView<std::int64_t>("atomicInt");

   try {
      reader->GetView<std::atomic<std::byte>>("atomicInt");
      FAIL() << "automatic evolution into an invalid atomic inner type should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible with on-disk field"));
   }

   try {
      reader->GetView<CustomAtomicNotLockFree>("atomicInt");
      FAIL() << "automatic evolution into an invalid non-atomic inner type should fail";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible type name for field"));
   }

   EXPECT_EQ(7, v1(0));
   EXPECT_EQ(13, v2(0));
   EXPECT_EQ(7, v3(0));
}

TEST(RNTupleEvolution, ArrayAsRVec)
{
   FileRaii fileGuard("test_ntuple_evolution_array_as_rvec.root");
   {
      auto model = ROOT::RNTupleModel::Create();
      auto a = model->MakeField<std::array<int, 2>>("a");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      *a = {1, 2};
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

   auto a = reader->GetView<ROOT::RVec<short>>("a");
   const auto &f = a.GetField(); // necessary to silence clang warning
   EXPECT_EQ(typeid(f), typeid(ROOT::RArrayAsRVecField));
   EXPECT_EQ(2u, a(0).size());
   EXPECT_EQ(1, a(0)[0]);
   EXPECT_EQ(2, a(0)[1]);
}

TEST(RNTupleEvolution, ArrayAsVector)
{
   FileRaii fileGuard("test_ntuple_evolution_array_as_vector.root");
   {
      auto model = ROOT::RNTupleModel::Create();
      auto a = model->MakeField<std::array<int, 2>>("a");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      *a = {0, 1};
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

   auto aAsShort = reader->GetView<std::vector<short>>("a");
   const auto &f = aAsShort.GetField(); // necessary to silence clang warning
   EXPECT_EQ(typeid(f), typeid(ROOT::RArrayAsVectorField));
   EXPECT_EQ(2u, aAsShort(0).size());
   EXPECT_EQ(0, aAsShort(0)[0]);
   EXPECT_EQ(1, aAsShort(0)[1]);

   auto aAsBool = reader->GetView<std::vector<bool>>("a");
   EXPECT_EQ(2u, aAsBool(0).size());
   EXPECT_FALSE(aAsBool(0)[0]);
   EXPECT_TRUE(aAsBool(0)[1]);
}

TEST(RNTupleEvolution, CheckNullable)
{
   FileRaii fileGuard("test_ntuple_evolution_check_nullable.root");
   {
      auto model = ROOT::RNTupleModel::Create();
      auto o = model->MakeField<std::optional<std::int32_t>>("o");
      auto u = model->MakeField<std::unique_ptr<std::int32_t>>("u");
      auto i = model->MakeField<std::int32_t>("i");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      *o = 7;
      *u = std::make_unique<std::int32_t>(11);
      *i = 13;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());

   auto v1 = reader->GetView<std::unique_ptr<std::int64_t>>("o");
   auto v2 = reader->GetView<std::optional<std::int64_t>>("u");
   auto v3 = reader->GetView<std::unique_ptr<std::int64_t>>("i");
   auto v4 = reader->GetView<std::optional<std::int64_t>>("i");

   try {
      reader->GetView<std::optional<std::string>>("i");
      FAIL() << "evolution of a nullable field with an invalid inner field should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("of type std::string is incompatible with on-disk field"));
   }

   EXPECT_EQ(7, *v1(0));
   EXPECT_EQ(11, *v2(0));
   EXPECT_EQ(13, *v3(0));
   EXPECT_EQ(13, *v4(0));
}

TEST(RNTupleEvolution, NullableToVector)
{
   FileRaii fileGuard("test_ntuple_evolution_nullable_to_vector.root");
   {
      auto model = ROOT::RNTupleModel::Create();
      auto o = model->MakeField<std::optional<int>>("o");
      auto u = model->MakeField<std::unique_ptr<int>>("u");
      auto writer = ROOT::RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());

      *o = 137;
      *u = std::make_unique<int>(42);
      writer->Fill();
      o->reset();
      u->reset();
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto v1 = reader->GetView<std::vector<short int>>("o");
   auto v2 = reader->GetView<ROOT::RVec<short int>>("o");
   auto v3 = reader->GetView<std::vector<short int>>("u");
   auto v4 = reader->GetView<ROOT::RVec<short int>>("u");
   EXPECT_EQ(137, v1(0)[0]);
   EXPECT_EQ(137, v2(0)[0]);
   EXPECT_EQ(42, v3(0)[0]);
   EXPECT_EQ(42, v4(0)[0]);
   EXPECT_TRUE(v1(1).empty());
   EXPECT_TRUE(v2(1).empty());
   EXPECT_TRUE(v3(1).empty());
   EXPECT_TRUE(v4(1).empty());
}

namespace {
template <typename CollectionT, bool OfPairsT>
void WriteCollection(std::string_view ntplName, TFile &f)
{
   auto model = RNTupleModel::Create();
   auto ptrCollection = model->MakeField<CollectionT>("f");
   auto writer = ROOT::RNTupleWriter::Append(std::move(model), ntplName, f);
   if constexpr (OfPairsT) {
      *ptrCollection = {{1, 2}, {3, 4}, {5, 6}};
   } else {
      *ptrCollection = {1, 2, 3};
   }
   writer->Fill();
   ptrCollection->clear();
   writer->Fill();
   if constexpr (OfPairsT) {
      *ptrCollection = {{7, 8}};
   } else {
      *ptrCollection = {4};
   }
   writer->Fill();
}

template <typename CollectionT, bool OfPairsT>
void ReadCollection(std::string_view ntplName, std::string_view path)
{
   auto reader = RNTupleReader::Open(ntplName, path);
   ASSERT_EQ(3u, reader->GetNEntries());

   auto view = reader->GetView<CollectionT>("f");
   CollectionT exp0;
   CollectionT exp2;
   if constexpr (OfPairsT) {
      exp0 = {{1, 2}, {3, 4}, {5, 6}};
      exp2 = {{7, 8}};
   } else {
      exp0 = {1, 2, 3};
      exp2 = {4};
   }
   EXPECT_EQ(exp0.size(), view(0).size());
   for (const auto &elem : exp0) {
      const auto &ref = view(0);
      EXPECT_TRUE(std::find(ref.begin(), ref.end(), elem) != ref.end());
   }
   EXPECT_TRUE(view(1).empty());
   EXPECT_EQ(exp2.size(), view(2).size());
   for (std::size_t i = 0; i < exp2.size(); ++i)
      EXPECT_EQ(*exp2.begin(), *view(2).begin());
}

template <typename CollectionT, bool OfPairsT>
void ReadCollectionFail(std::string_view ntplName, std::string_view path)
{
   auto reader = RNTupleReader::Open(ntplName, path);
   ASSERT_EQ(3u, reader->GetNEntries());

   try {
      reader->GetView<CollectionT>("f");
      FAIL() << "this case of automatic collection schema evolution should have failed";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible type"));
   }
}
} // anonymous namespace

namespace ROOT {
template <>
struct IsCollectionProxy<CollectionProxy<int>> : std::true_type {};
template <>
struct IsCollectionProxy<CollectionProxy<short int>> : std::true_type {};
template <>
struct IsCollectionProxy<CollectionProxy<std::pair<int, int>>> : std::true_type {};
template <>
struct IsCollectionProxy<CollectionProxy<std::pair<short int, short int>>> : std::true_type {};
} // namespace ROOT

TEST(RNTupleEvolution, Collections)
{
   FileRaii fileGuard("test_ntuple_evolution_collections.root");
   auto f = std::unique_ptr<TFile>(TFile::Open(fileGuard.GetPath().c_str(), "UPDATE"));

   TClass::GetClass("CollectionProxy<int>")->CopyCollectionProxy(SimpleCollectionProxy<CollectionProxy<int>>());
   TClass::GetClass("CollectionProxy<short int>")
      ->CopyCollectionProxy(SimpleCollectionProxy<CollectionProxy<short int>>());
   TClass::GetClass("CollectionProxy<std::pair<int, int>>")
      ->CopyCollectionProxy(SimpleCollectionProxy<CollectionProxy<std::pair<int, int>>>());
   TClass::GetClass("CollectionProxy<std::pair<short int, short int>>")
      ->CopyCollectionProxy(SimpleCollectionProxy<CollectionProxy<std::pair<short int, short int>>>());

   {
      auto model = RNTupleModel::Create();
      model->AddField(ROOT::RVectorField::CreateUntyped("f", std::make_unique<RField<int>>("x")));
      model->Freeze();
      auto v = std::static_pointer_cast<std::vector<int>>(model->GetDefaultEntry().GetPtr<void>("f"));
      auto writer = ROOT::RNTupleWriter::Append(std::move(model), "untyped", *f);
      *v = {1, 2, 3};
      writer->Fill();
      v->clear();
      writer->Fill();
      *v = {4};
      writer->Fill();
   }
   {
      auto model = RNTupleModel::Create();
      model->AddField(ROOT::RVectorField::CreateUntyped("f", std::make_unique<RField<std::pair<int, int>>>("x")));
      model->Freeze();
      auto v = std::static_pointer_cast<std::vector<std::pair<int, int>>>(model->GetDefaultEntry().GetPtr<void>("f"));
      auto writer = ROOT::RNTupleWriter::Append(std::move(model), "untypedOfPairs", *f);
      *v = {{1, 2}, {3, 4}, {5, 6}};
      writer->Fill();
      v->clear();
      writer->Fill();
      *v = {{7, 8}};
      writer->Fill();
   }
   {
      auto model = RNTupleModel::Create();
      auto proxy = model->MakeField<CollectionProxy<int>>("f");
      auto writer = RNTupleWriter::Append(std::move(model), "proxy", *f);
      proxy->v = {1, 2, 3};
      writer->Fill();
      proxy->v.clear();
      writer->Fill();
      proxy->v = {4};
      writer->Fill();
   }
   {
      auto model = RNTupleModel::Create();
      auto proxy = model->MakeField<CollectionProxy<std::pair<int, int>>>("f");
      auto writer = RNTupleWriter::Append(std::move(model), "proxyOfPairs", *f);
      proxy->v = {{1, 2}, {3, 4}, {5, 6}};
      writer->Fill();
      proxy->v.clear();
      writer->Fill();
      proxy->v = {{7, 8}};
      writer->Fill();
   }

   WriteCollection<std::vector<int>, false>("vector", *f);
   WriteCollection<ROOT::RVec<int>, false>("rvec", *f);
   WriteCollection<std::set<int>, false>("set", *f);
   WriteCollection<std::unordered_set<int>, false>("unordered_set", *f);
   WriteCollection<std::multiset<int>, false>("multiset", *f);
   WriteCollection<std::unordered_multiset<int>, false>("unordered_multiset", *f);
   WriteCollection<std::map<int, int>, true>("map", *f);
   WriteCollection<std::unordered_map<int, int>, true>("unordered_map", *f);
   WriteCollection<std::multimap<int, int>, true>("multimap", *f);
   WriteCollection<std::unordered_multimap<int, int>, true>("unordered_multimap", *f);

   WriteCollection<std::vector<std::pair<int, int>>, true>("vectorOfPairs", *f);
   WriteCollection<ROOT::RVec<std::pair<int, int>>, true>("rvecOfPairs", *f);
   WriteCollection<std::set<std::pair<int, int>>, true>("setOfPairs", *f);
   WriteCollection<std::unordered_set<std::pair<int, int>>, true>("unordered_setOfPairs", *f);
   WriteCollection<std::multiset<std::pair<int, int>>, true>("multisetOfPairs", *f);
   WriteCollection<std::unordered_multiset<std::pair<int, int>>, true>("unordered_multisetOfPairs", *f);

   // All variations written out. Now test the collection matrix.

   ReadCollection<std::vector<short int>, false>("untyped", fileGuard.GetPath());
   ReadCollection<std::vector<short int>, false>("proxy", fileGuard.GetPath());
   ReadCollection<std::vector<short int>, false>("rvec", fileGuard.GetPath());
   ReadCollection<std::vector<short int>, false>("set", fileGuard.GetPath());
   ReadCollection<std::vector<short int>, false>("unordered_set", fileGuard.GetPath());
   ReadCollection<std::vector<short int>, false>("multiset", fileGuard.GetPath());
   ReadCollection<std::vector<short int>, false>("unordered_multiset", fileGuard.GetPath());
   ReadCollection<std::vector<std::pair<short int, short int>>, true>("map", fileGuard.GetPath());
   ReadCollection<std::vector<std::pair<short int, short int>>, true>("unordered_map", fileGuard.GetPath());
   ReadCollection<std::vector<std::pair<short int, short int>>, true>("multimap", fileGuard.GetPath());
   ReadCollection<std::vector<std::pair<short int, short int>>, true>("unordered_multimap", fileGuard.GetPath());

   ReadCollection<ROOT::RVec<short int>, false>("untyped", fileGuard.GetPath());
   ReadCollection<ROOT::RVec<short int>, false>("proxy", fileGuard.GetPath());
   ReadCollection<ROOT::RVec<short int>, false>("vector", fileGuard.GetPath());
   ReadCollection<ROOT::RVec<short int>, false>("set", fileGuard.GetPath());
   ReadCollection<ROOT::RVec<short int>, false>("unordered_set", fileGuard.GetPath());
   ReadCollection<ROOT::RVec<short int>, false>("multiset", fileGuard.GetPath());
   ReadCollection<ROOT::RVec<short int>, false>("unordered_multiset", fileGuard.GetPath());
   ReadCollection<ROOT::RVec<std::pair<short int, short int>>, true>("map", fileGuard.GetPath());
   ReadCollection<ROOT::RVec<std::pair<short int, short int>>, true>("unordered_map", fileGuard.GetPath());
   ReadCollection<ROOT::RVec<std::pair<short int, short int>>, true>("multimap", fileGuard.GetPath());
   ReadCollection<ROOT::RVec<std::pair<short int, short int>>, true>("unordered_multimap", fileGuard.GetPath());

   ReadCollectionFail<std::set<short int>, false>("untyped", fileGuard.GetPath());
   ReadCollectionFail<std::set<short int>, false>("proxy", fileGuard.GetPath());
   ReadCollectionFail<std::set<short int>, false>("vector", fileGuard.GetPath());
   ReadCollectionFail<std::set<short int>, false>("rvec", fileGuard.GetPath());
   ReadCollection<std::set<short int>, false>("unordered_set", fileGuard.GetPath());
   ReadCollectionFail<std::set<short int>, false>("multiset", fileGuard.GetPath());
   ReadCollectionFail<std::set<short int>, false>("unordered_multiset", fileGuard.GetPath());
   ReadCollection<std::set<std::pair<short int, short int>>, true>("map", fileGuard.GetPath());
   ReadCollection<std::set<std::pair<short int, short int>>, true>("unordered_map", fileGuard.GetPath());
   ReadCollectionFail<std::set<std::pair<short int, short int>>, true>("multimap", fileGuard.GetPath());
   ReadCollectionFail<std::set<std::pair<short int, short int>>, true>("unordered_multimap", fileGuard.GetPath());

   ReadCollectionFail<std::unordered_set<short int>, false>("untyped", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_set<short int>, false>("proxy", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_set<short int>, false>("vector", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_set<short int>, false>("rvec", fileGuard.GetPath());
   ReadCollection<std::unordered_set<short int>, false>("set", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_set<short int>, false>("multiset", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_set<short int>, false>("unordered_multiset", fileGuard.GetPath());
   ReadCollection<std::unordered_set<std::pair<short int, short int>>, true>("map", fileGuard.GetPath());
   ReadCollection<std::unordered_set<std::pair<short int, short int>>, true>("unordered_map", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_set<std::pair<short int, short int>>, true>("multimap", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_set<std::pair<short int, short int>>, true>("unordered_multimap",
                                                                                 fileGuard.GetPath());

   ReadCollection<std::multiset<short int>, false>("untyped", fileGuard.GetPath());
   ReadCollection<std::multiset<short int>, false>("proxy", fileGuard.GetPath());
   ReadCollection<std::multiset<short int>, false>("vector", fileGuard.GetPath());
   ReadCollection<std::multiset<short int>, false>("rvec", fileGuard.GetPath());
   ReadCollection<std::multiset<short int>, false>("unordered_set", fileGuard.GetPath());
   ReadCollection<std::multiset<short int>, false>("set", fileGuard.GetPath());
   ReadCollection<std::multiset<short int>, false>("unordered_multiset", fileGuard.GetPath());
   ReadCollection<std::multiset<std::pair<short int, short int>>, true>("map", fileGuard.GetPath());
   ReadCollection<std::multiset<std::pair<short int, short int>>, true>("unordered_map", fileGuard.GetPath());
   ReadCollection<std::multiset<std::pair<short int, short int>>, true>("multimap", fileGuard.GetPath());
   ReadCollection<std::multiset<std::pair<short int, short int>>, true>("unordered_multimap", fileGuard.GetPath());

   ReadCollection<std::unordered_multiset<short int>, false>("untyped", fileGuard.GetPath());
   ReadCollection<std::unordered_multiset<short int>, false>("proxy", fileGuard.GetPath());
   ReadCollection<std::unordered_multiset<short int>, false>("vector", fileGuard.GetPath());
   ReadCollection<std::unordered_multiset<short int>, false>("rvec", fileGuard.GetPath());
   ReadCollection<std::unordered_multiset<short int>, false>("unordered_set", fileGuard.GetPath());
   ReadCollection<std::unordered_multiset<short int>, false>("set", fileGuard.GetPath());
   ReadCollection<std::unordered_multiset<short int>, false>("multiset", fileGuard.GetPath());
   ReadCollection<std::unordered_multiset<std::pair<short int, short int>>, true>("map", fileGuard.GetPath());
   ReadCollection<std::unordered_multiset<std::pair<short int, short int>>, true>("unordered_map", fileGuard.GetPath());
   ReadCollection<std::unordered_multiset<std::pair<short int, short int>>, true>("multimap", fileGuard.GetPath());
   ReadCollection<std::unordered_multiset<std::pair<short int, short int>>, true>("unordered_multimap",
                                                                                  fileGuard.GetPath());

   ReadCollectionFail<std::map<short int, short int>, true>("untypedOfPairs", fileGuard.GetPath());
   ReadCollectionFail<std::map<short int, short int>, true>("proxyOfPairs", fileGuard.GetPath());
   ReadCollectionFail<std::map<short int, short int>, true>("vectorOfPairs", fileGuard.GetPath());
   ReadCollectionFail<std::map<short int, short int>, true>("rvecOfPairs", fileGuard.GetPath());
   ReadCollectionFail<std::map<short int, short int>, true>("setOfPairs", fileGuard.GetPath());
   ReadCollectionFail<std::map<short int, short int>, true>("unordered_setOfPairs", fileGuard.GetPath());
   ReadCollectionFail<std::map<short int, short int>, true>("multisetOfPairs", fileGuard.GetPath());
   ReadCollectionFail<std::map<short int, short int>, true>("unordered_multisetOfPairs", fileGuard.GetPath());
   ReadCollection<std::map<short int, short int>, true>("unordered_map", fileGuard.GetPath());
   ReadCollectionFail<std::map<short int, short int>, true>("multimap", fileGuard.GetPath());
   ReadCollectionFail<std::map<short int, short int>, true>("unordered_multimap", fileGuard.GetPath());

   ReadCollectionFail<std::unordered_map<short int, short int>, true>("untypedOfPairs", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_map<short int, short int>, true>("proxyOfPairs", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_map<short int, short int>, true>("vectorOfPairs", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_map<short int, short int>, true>("rvecOfPairs", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_map<short int, short int>, true>("setOfPairs", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_map<short int, short int>, true>("unordered_setOfPairs", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_map<short int, short int>, true>("multisetOfPairs", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_map<short int, short int>, true>("unordered_multisetOfPairs", fileGuard.GetPath());
   ReadCollection<std::unordered_map<short int, short int>, true>("map", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_map<short int, short int>, true>("multimap", fileGuard.GetPath());
   ReadCollectionFail<std::unordered_map<short int, short int>, true>("unordered_multimap", fileGuard.GetPath());

   ReadCollection<std::multimap<short int, short int>, true>("untypedOfPairs", fileGuard.GetPath());
   ReadCollection<std::multimap<short int, short int>, true>("proxyOfPairs", fileGuard.GetPath());
   ReadCollection<std::multimap<short int, short int>, true>("vectorOfPairs", fileGuard.GetPath());
   ReadCollection<std::multimap<short int, short int>, true>("rvecOfPairs", fileGuard.GetPath());
   ReadCollection<std::multimap<short int, short int>, true>("setOfPairs", fileGuard.GetPath());
   ReadCollection<std::multimap<short int, short int>, true>("unordered_setOfPairs", fileGuard.GetPath());
   ReadCollection<std::multimap<short int, short int>, true>("multisetOfPairs", fileGuard.GetPath());
   ReadCollection<std::multimap<short int, short int>, true>("unordered_multisetOfPairs", fileGuard.GetPath());
   ReadCollection<std::multimap<short int, short int>, true>("map", fileGuard.GetPath());
   ReadCollection<std::multimap<short int, short int>, true>("unordered_map", fileGuard.GetPath());
   ReadCollection<std::multimap<short int, short int>, true>("multimap", fileGuard.GetPath());
   ReadCollection<std::multimap<short int, short int>, true>("unordered_multimap", fileGuard.GetPath());

   ReadCollection<std::unordered_multimap<short int, short int>, true>("untypedOfPairs", fileGuard.GetPath());
   ReadCollection<std::unordered_multimap<short int, short int>, true>("proxyOfPairs", fileGuard.GetPath());
   ReadCollection<std::unordered_multimap<short int, short int>, true>("vectorOfPairs", fileGuard.GetPath());
   ReadCollection<std::unordered_multimap<short int, short int>, true>("rvecOfPairs", fileGuard.GetPath());
   ReadCollection<std::unordered_multimap<short int, short int>, true>("setOfPairs", fileGuard.GetPath());
   ReadCollection<std::unordered_multimap<short int, short int>, true>("unordered_setOfPairs", fileGuard.GetPath());
   ReadCollection<std::unordered_multimap<short int, short int>, true>("multisetOfPairs", fileGuard.GetPath());
   ReadCollection<std::unordered_multimap<short int, short int>, true>("unordered_multisetOfPairs",
                                                                       fileGuard.GetPath());
   ReadCollection<std::unordered_multimap<short int, short int>, true>("map", fileGuard.GetPath());
   ReadCollection<std::unordered_multimap<short int, short int>, true>("unordered_map", fileGuard.GetPath());
   ReadCollection<std::unordered_multimap<short int, short int>, true>("multimap", fileGuard.GetPath());
   ReadCollection<std::unordered_multimap<short int, short int>, true>("unordered_multimap", fileGuard.GetPath());
}
