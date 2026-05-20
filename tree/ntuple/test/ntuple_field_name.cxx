#include "ntuple_test.hxx"

namespace {

template <typename FieldT>
void CheckSingleSubfieldName()
{
   try {
      FieldT("f", std::make_unique<ROOT::RField<char>>("x"));
      FAIL() << "invalid subfield name should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("invalid subfield name"));
   }
}

template <typename ContainerT>
void ResetContainer(ContainerT &)
{
   static_assert(!std::is_same_v<ContainerT, ContainerT>, "unsupported container type");
}

template <>
void ResetContainer(std::vector<std::unique_ptr<ROOT::RFieldBase>> &v)
{
   v.resize(2);
}

template <>
void ResetContainer(std::array<std::unique_ptr<ROOT::RFieldBase>, 2> &)
{
}

template <typename FieldT, typename SubfieldCollectionT>
void CheckDoubleSubfieldName(SubfieldCollectionT &&items)
{
   ASSERT_EQ(2u, items.size());

   items[0] = std::make_unique<RField<char>>("_0");
   items[1] = std::make_unique<RField<char>>("x");
   try {
      FieldT("f", std::move(items));
      FAIL() << "invalid subfield name should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("invalid subfield name"));
   }

   ResetContainer(items);

   items[0] = std::make_unique<RField<char>>("x");
   items[1] = std::make_unique<RField<char>>("_1");
   try {
      FieldT("f", std::move(items));
      FAIL() << "invalid subfield name should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("invalid subfield name"));
   }
}

} // anonymous namespace

TEST(RNTuple, SubfieldName)
{
   CheckSingleSubfieldName<ROOT::RAtomicField>();
   CheckSingleSubfieldName<ROOT::RVectorField>();
   CheckSingleSubfieldName<ROOT::RRVecField>();
   CheckSingleSubfieldName<ROOT::ROptionalField>();

   CheckDoubleSubfieldName<ROOT::RVariantField>(std::vector<std::unique_ptr<ROOT::RFieldBase>>(2));
   CheckDoubleSubfieldName<ROOT::RTupleField>(std::vector<std::unique_ptr<ROOT::RFieldBase>>(2));
   CheckDoubleSubfieldName<ROOT::RPairField>(std::array<std::unique_ptr<ROOT::RFieldBase>, 2>());

   try {
      ROOT::RArrayField("f", std::make_unique<ROOT::RField<char>>("x"), 2);
      FAIL() << "invalid subfield name should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("invalid subfield name"));
   }

   try {
      ROOT::RSetField("f", ROOT::RSetField::ESetType::kSet, std::make_unique<ROOT::RField<char>>("x"));
      FAIL() << "invalid subfield name should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("invalid subfield name"));
   }

   std::unique_ptr<ROOT::RPairField> p{
      new ROOT::RPairField("f", {std::make_unique<RField<char>>("_0"), std::make_unique<RField<char>>("_1")})};
   try {
      ROOT::RMapField("f", ROOT::RMapField::EMapType::kMap, std::move(p));
      FAIL() << "invalid subfield name should fail";
   } catch (const ROOT::RException &e) {
      EXPECT_THAT(e.what(), testing::HasSubstr("invalid subfield name"));
   }
}

TEST(RNTuple, FieldNameCollision)
{
   ROOT::RFieldZero zero;
   zero.Attach(std::make_unique<RField<char>>("x"));
   EXPECT_THROW(zero.Attach(std::make_unique<RField<char>>("x")), ROOT::RException);

   std::vector<std::unique_ptr<ROOT::RFieldBase>> items;
   items.emplace_back(std::make_unique<RField<char>>("x"));
   items.emplace_back(std::make_unique<RField<char>>("x"));
   try {
      ROOT::RRecordField("f", std::move(items));
      FAIL() << "duplicate field names should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("duplicate field name"));
   }
}
