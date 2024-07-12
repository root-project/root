#include "ntuple_test.cxx"
#include "SimpleCollectionProxy.hxx"

template <typename FieldT>
class RHashVisitorValidTypesTest : public testing::Test {};

using HashableFieldTypes =
   ::testing::Types<bool, double, float, char, std::string, std::int8_t, std::int16_t, std::int32_t, std::int64_t,
                    std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t>;

TYPED_TEST_SUITE(RHashVisitorValidTypesTest, HashableFieldTypes);

template <typename T>
void TestGetHash(RNTupleModel &model)
{
   auto fld = model.MakeField<T>("fld", static_cast<T>(42));
   RHashValueVisitor visitor(fld.get());
   model.GetField("fld").AcceptVisitor(visitor);
   EXPECT_EQ(std::hash<T>()(static_cast<T>(42)), visitor.GetHash());
}

template <>
void TestGetHash<std::string>(RNTupleModel &model)
{
   auto fld = model.MakeField<std::string>("fld", "foo");
   RHashValueVisitor visitor(fld.get());
   model.GetField("fld").AcceptVisitor(visitor);
   EXPECT_EQ(std::hash<std::string>()("foo"), visitor.GetHash());
}

TYPED_TEST(RHashVisitorValidTypesTest, GetHash)
{
   auto model = RNTupleModel::Create();
   TestGetHash<TypeParam>(*model);
}

template <typename FieldT>
class RHashVisitorInvalidTypesTest : public testing::Test {};

using NonHashableFieldTypes =
   ::testing::Types<std::byte, std::array<int, 3>, ROOT::RVec<float>, CustomStruct, CustomEnum,
                    std::unique_ptr<std::int64_t>, std::variant<std::string, CustomStruct>, std::bitset<8>,
                    std::vector<bool>>;

TYPED_TEST_SUITE(RHashVisitorInvalidTypesTest, NonHashableFieldTypes);

TYPED_TEST(RHashVisitorInvalidTypesTest, ThrowError)
{
   auto model = RNTupleModel::Create();
   auto fld = model->MakeField<TypeParam>("fld");
   RHashValueVisitor visitor(fld.get());
   try {
      model->GetField("fld").AcceptVisitor(visitor);
      FAIL() << "an exception should be thrown for non-hashable field types";
   } catch (RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("hashing is not supported for fields of type"));
   }
}
