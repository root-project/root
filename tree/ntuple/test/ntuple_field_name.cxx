#include "ntuple_test.hxx"

TEST(RNTuple, SubFieldName)
{
   EXPECT_EQ("_0", ROOT::RAtomicField("f", std::make_unique<RField<char>>("x")).begin()->GetFieldName());
   EXPECT_EQ("_0", ROOT::RVectorField("f", std::make_unique<RField<char>>("x")).begin()->GetFieldName());
   EXPECT_EQ("_0", ROOT::RRVecField("f", std::make_unique<RField<char>>("x")).begin()->GetFieldName());
   EXPECT_EQ("_0", ROOT::RArrayField("f", std::make_unique<RField<char>>("x"), 2).begin()->GetFieldName());
   EXPECT_EQ("_0", ROOT::ROptionalField("f", std::make_unique<RField<char>>("x")).begin()->GetFieldName());
   EXPECT_EQ("_0", ROOT::RAtomicField("f", std::make_unique<RField<char>>("x")).begin()->GetFieldName());
   EXPECT_EQ("_0", ROOT::RSetField("f", ROOT::RSetField::ESetType::kSet, std::make_unique<RField<char>>("x"))
                      .begin()
                      ->GetFieldName());

   {
      std::unique_ptr<ROOT::RPairField> p{
         new ROOT::RPairField("x", {std::make_unique<RField<char>>("x"), std::make_unique<RField<char>>("x")})};
      EXPECT_EQ("_0", ROOT::RMapField("f", ROOT::RMapField::EMapType::kMap, std::move(p)).begin()->GetFieldName());
   }

   {
      std::vector<std::unique_ptr<ROOT::RFieldBase>> items;
      items.emplace_back(std::make_unique<RField<char>>("x"));
      items.emplace_back(std::make_unique<RField<int>>("x"));
      ROOT::RVariantField f("f", std::move(items));
      auto itr = f.begin();
      EXPECT_EQ("_0", itr->GetFieldName());
      itr++;
      EXPECT_EQ("_1", itr->GetFieldName());
   }

   {
      std::array<std::unique_ptr<ROOT::RFieldBase>, 2> items;
      items[0] = std::make_unique<RField<char>>("x");
      items[1] = std::make_unique<RField<char>>("x");
      ROOT::RPairField f("f", std::move(items));
      auto itr = f.begin();
      EXPECT_EQ("_0", itr->GetFieldName());
      itr++;
      EXPECT_EQ("_1", itr->GetFieldName());
   }

   {
      std::vector<std::unique_ptr<ROOT::RFieldBase>> items;
      items.emplace_back(std::make_unique<RField<char>>("x"));
      items.emplace_back(std::make_unique<RField<char>>("x"));
      ROOT::RTupleField f("f", std::move(items));
      auto itr = f.begin();
      EXPECT_EQ("_0", itr->GetFieldName());
      itr++;
      EXPECT_EQ("_1", itr->GetFieldName());
   }
}
