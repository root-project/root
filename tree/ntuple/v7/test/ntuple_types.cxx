#include "ntuple_test.hxx"

TEST(RNTuple, TypeName) {
   EXPECT_STREQ("float", ROOT::Experimental::RField<float>::TypeName().c_str());
   EXPECT_STREQ("std::vector<std::string>",
                ROOT::Experimental::RField<std::vector<std::string>>::TypeName().c_str());
   EXPECT_STREQ("CustomStruct",
                ROOT::Experimental::RField<CustomStruct>::TypeName().c_str());
   EXPECT_STREQ("DerivedB",
                ROOT::Experimental::RField<DerivedB>::TypeName().c_str());
}


TEST(RNTuple, CreateField)
{
   auto field = RFieldBase::Create("test", "vector<unsigned int>").Unwrap();
   EXPECT_STREQ("std::vector<std::uint32_t>", field->GetType().c_str());
   auto value = field->GenerateValue();
   field->DestroyValue(value);
}

TEST(RNTuple, Int64_t)
{
   auto field = RField<std::int64_t>("myInt64");
   auto otherField = RFieldBase::Create("test", "std::int64_t").Unwrap();
}

TEST(RNTuple, Char)
{
   auto charField = RField<char>("myChar");
   auto otherField = RFieldBase::Create("test", "char").Unwrap();
   ASSERT_EQ("char", otherField->GetType());

   auto charTField = RField<Char_t>("myChar");
   ASSERT_EQ("char", charTField.GetType());
}

TEST(RNTuple, Int8_t)
{
   auto field = RField<std::int8_t>("myInt8");
   auto otherField = RFieldBase::Create("test", "std::int8_t").Unwrap();
}

TEST(RNTuple, Int16_t)
{
   auto field = RField<std::int16_t>("myInt16");
   auto otherField = RFieldBase::Create("test", "std::int16_t").Unwrap();
   ASSERT_EQ("std::int16_t", RFieldBase::Create("myShort", "Short_t").Unwrap()->GetType());
}

TEST(RNTuple, UInt16_t)
{
   auto field = RField<std::uint16_t>("myUint16");
   auto otherField = RFieldBase::Create("test", "std::uint16_t").Unwrap();
   ASSERT_EQ("std::uint16_t", RFieldBase::Create("myUShort", "UShort_t").Unwrap()->GetType());
}

TEST(RNTuple, UnsupportedStdTypes)
{
   try {
      auto field = RFieldBase::Create("pair_field", "std::pair<int, float>").Unwrap();
      FAIL() << "should not be able to make a std::pair field";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("std::pair<int, float> is not supported"));
   }
   try {
      auto field = RField<std::pair<int, float>>("pair_field");
      FAIL() << "should not be able to make a std::pair field";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("pair<int,float> is not supported"));
   }
   try {
      auto field = RField<std::weak_ptr<int>>("myWeakPtr");
      FAIL() << "should not be able to make a std::weak_ptr field";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("weak_ptr<int> is not supported"));
   }
   try {
      auto field = RField<std::vector<std::weak_ptr<int>>>("weak_ptr_vec");
      FAIL() << "should not be able to make a std::vector<std::weak_ptr> field";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("weak_ptr<int> is not supported"));
   }
}

TEST(RNTuple, Casting)
{
   FileRaii fileGuard("test_ntuple_casting.root");
   auto modelA = RNTupleModel::Create();
   modelA->MakeField<std::int32_t>("myInt", 42);
   {
      auto writer = RNTupleWriter::Recreate(std::move(modelA), "ntuple", fileGuard.GetPath());
      writer->Fill();
   }

   try {
      auto modelB = RNTupleModel::Create();
      auto fieldCast = modelB->MakeField<float>("myInt");
      auto reader = RNTupleReader::Open(std::move(modelB), "ntuple", fileGuard.GetPath());
      FAIL() << "should not be able to cast int to float";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("not convertible to the requested type"));
   }

   auto modelC = RNTupleModel::Create();
   auto fieldCast = modelC->MakeField<std::int64_t>("myInt");
   auto reader = RNTupleReader::Open(std::move(modelC), "ntuple", fileGuard.GetPath());
   reader->LoadEntry(0);
   EXPECT_EQ(42, *fieldCast);
}

TEST(RNTuple, TClass)
{
   FileRaii fileGuard("test_ntuple_tclass.ntuple");
   {
      auto model = RNTupleModel::Create();
      auto fieldKlass = model->MakeField<DerivedB>("klass");
      RNTupleWriteOptions options;
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath(), options);
      for (int i = 0; i < 20000; i++) {
         DerivedB klass;
         klass.a = static_cast<float>(i);
         klass.v1.emplace_back(static_cast<float>(i));
         klass.v2.emplace_back(std::vector<float>(3, static_cast<float>(i)));
         klass.s = "hi" + std::to_string(i);

         klass.a_v.emplace_back(static_cast<float>(i + 1));
         klass.a_s = "bye" + std::to_string(i);

         klass.b_f1 = static_cast<float>(i + 2);
         klass.b_f2 = static_cast<float>(i + 3);
         *fieldKlass = klass;
         ntuple->Fill();
      }
   }

   {
      auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
      EXPECT_EQ(20000U, ntuple->GetNEntries());
      auto viewKlass = ntuple->GetView<DerivedB>("klass");
      for (auto i : ntuple->GetEntryRange()) {
         float fi = static_cast<float>(i);
         EXPECT_EQ(fi, viewKlass(i).a);
         EXPECT_EQ(std::vector<float>{fi}, viewKlass(i).v1);
         EXPECT_EQ((std::vector<float>(3, fi)), viewKlass(i).v2.at(0));
         EXPECT_EQ("hi" + std::to_string(i), viewKlass(i).s);

         EXPECT_EQ(std::vector<float>{fi + 1}, viewKlass(i).a_v);
         EXPECT_EQ("bye" + std::to_string(i), viewKlass(i).a_s);

         EXPECT_EQ((fi + 2), viewKlass(i).b_f1);
         EXPECT_EQ(0.0, viewKlass(i).b_f2);
      }
   }

   {
      auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
      try {
         auto viewKlass = ntuple->GetView<DerivedA>("klass");
         FAIL() << "GetView<a_base_class_of_T> should throw";
      } catch (const RException& err) {
         EXPECT_THAT(err.what(), testing::HasSubstr("Column missing: column #0 for field a"));
      }
   }
}
