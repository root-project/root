#include "ntuple_test.hxx"
#include "TInterpreter.h"

TEST(RNTuple, TypeName) {
   EXPECT_STREQ("float", ROOT::Experimental::RField<float>::TypeName().c_str());
   EXPECT_STREQ("std::vector<std::string>",
                ROOT::Experimental::RField<std::vector<std::string>>::TypeName().c_str());
   EXPECT_STREQ("CustomStruct",
                ROOT::Experimental::RField<CustomStruct>::TypeName().c_str());
   EXPECT_STREQ("DerivedB",
                ROOT::Experimental::RField<DerivedB>::TypeName().c_str());

   auto field = RField<DerivedB>("derived");
   EXPECT_EQ(sizeof(DerivedB), field.GetValueSize());

   EXPECT_STREQ("std::pair<std::pair<float,CustomStruct>,std::int32_t>", (ROOT::Experimental::RField<
                 std::pair<std::pair<float,CustomStruct>,int>>::TypeName().c_str()));
}


TEST(RNTuple, CreateField)
{
   auto field = RFieldBase::Create("test", "vector<unsigned int>").Unwrap();
   EXPECT_STREQ("std::vector<std::uint32_t>", field->GetType().c_str());
   auto value = field->GenerateValue();
   field->DestroyValue(value);
}

TEST(RNTuple, StdPair)
{
   auto field = RField<std::pair<int64_t, float>>("pairField");
   EXPECT_STREQ("std::pair<std::int64_t,float>", field.GetType().c_str());
   auto otherField = RFieldBase::Create("test", "std::pair<int64_t, float>").Unwrap();
   EXPECT_STREQ(field.GetType().c_str(), otherField->GetType().c_str());
   EXPECT_EQ((sizeof(std::pair<int64_t, float>)), field.GetValueSize());
   EXPECT_EQ((sizeof(std::pair<int64_t, float>)), otherField->GetValueSize());
   EXPECT_EQ((alignof(std::pair<int64_t, float>)), field.GetAlignment());
   EXPECT_EQ((alignof(std::pair<int64_t, float>)), otherField->GetAlignment());

   auto pairPairField = RField<std::pair<std::pair<int64_t, float>,
      std::vector<std::pair<CustomStruct, double>>>>("pairPairField");
   EXPECT_STREQ(
      "std::pair<std::pair<std::int64_t,float>,std::vector<std::pair<CustomStruct,double>>>",
      pairPairField.GetType().c_str());

   FileRaii fileGuard("test_ntuple_rfield_stdpair.root");
   {
      auto model = RNTupleModel::Create();
      auto pair_field = model->MakeField<std::pair<double, std::string>>(
         {"myPair", "a very cool field"}
      );
      auto myPair2 = RFieldBase::Create("myPair2", "std::pair<double, std::string>").Unwrap();
      model->AddField(std::move(myPair2));

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "pair_ntuple", fileGuard.GetPath());
      auto pair_field2 = ntuple->GetModel()->GetDefaultEntry()->Get<std::pair<double, std::string>>("myPair2");
      for (int i = 0; i < 2; i++) {
         *pair_field = {static_cast<double>(i), std::to_string(i)};
         *pair_field2 = {static_cast<double>(i + 1), std::to_string(i + 1)};
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open("pair_ntuple", fileGuard.GetPath());
   EXPECT_EQ(2, ntuple->GetNEntries());

   auto viewPair = ntuple->GetView<std::pair<double, std::string>>("myPair");
   auto viewPair2 = ntuple->GetView<std::pair<double, std::string>>("myPair2");
   for (auto i : ntuple->GetEntryRange()) {
      EXPECT_EQ(static_cast<double>(i), viewPair(i).first);
      EXPECT_EQ(std::to_string(i), viewPair(i).second);

      EXPECT_EQ(static_cast<double>(i + 1), viewPair2(i).first);
      EXPECT_EQ(std::to_string(i + 1), viewPair2(i).second);
   }
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

TEST(RNTuple, TClassMultipleInheritance)
{
   FileRaii fileGuard("test_ntuple_tclass_multinheritance.ntuple");
   {
      auto model = RNTupleModel::Create();
      auto fieldKlass = model->MakeField<DerivedC>("klass");
      RNTupleWriteOptions options;
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath(), options);
      for (int i = 0; i < 10000; i++) {
         DerivedC klass;
         klass.DerivedA::a = static_cast<float>(i);
         klass.DerivedA::v1.emplace_back(static_cast<float>(i));
         klass.DerivedA::v2.emplace_back(std::vector<float>(3, static_cast<float>(i)));
         klass.DerivedA::s = "hi" + std::to_string(i);

         klass.a_v.emplace_back(static_cast<float>(i + 1));
         klass.a_s = "bye" + std::to_string(i);
         klass.a2_f = static_cast<float>(i + 2);

         klass.c_i = i;
         klass.c_a2.a2_f = static_cast<float>(i + 3);
         *fieldKlass = klass;
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
   EXPECT_EQ(10000U, ntuple->GetNEntries());
   auto viewKlass = ntuple->GetView<DerivedC>("klass");
   for (auto i : ntuple->GetEntryRange()) {
      float fi = static_cast<float>(i);
      EXPECT_EQ(fi, viewKlass(i).DerivedA::a);
      EXPECT_EQ(std::vector<float>{fi}, viewKlass(i).DerivedA::v1);
      EXPECT_EQ((std::vector<float>(3, fi)), viewKlass(i).DerivedA::v2.at(0));
      EXPECT_EQ("hi" + std::to_string(i), viewKlass(i).DerivedA::s);

      EXPECT_EQ(std::vector<float>{fi + 1}, viewKlass(i).a_v);
      EXPECT_EQ("bye" + std::to_string(i), viewKlass(i).a_s);
      EXPECT_EQ((fi + 2), viewKlass(i).a2_f);

      EXPECT_EQ(i, viewKlass(i).c_i);
      EXPECT_EQ((fi + 3), viewKlass(i).c_a2.a2_f);
   }
}

TEST(RNTuple, TClassEBO)
{
   // Empty base optimization is required for standard layout types (since C++11)
   EXPECT_EQ(sizeof(TestEBO), sizeof(std::uint64_t));
   auto field = RField<TestEBO>("klass");
   EXPECT_EQ(sizeof(TestEBO), field.GetValueSize());

   FileRaii fileGuard("test_ntuple_tclassebo.ntuple");
   {
      auto model = RNTupleModel::Create();
      auto fieldKlass = model->MakeField<TestEBO>("klass");
      RNTupleWriteOptions options;
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath(), options);
      (*fieldKlass).u64 = 42;
      ntuple->Fill();
   }

   {
      auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
      EXPECT_EQ(1U, ntuple->GetNEntries());
      auto idEmptyBase = ntuple->GetDescriptor()->FindFieldId("klass.:_0");
      EXPECT_NE(idEmptyBase, ROOT::Experimental::kInvalidDescriptorId);
      auto viewKlass = ntuple->GetView<TestEBO>("klass");
      EXPECT_EQ(42, viewKlass(0).u64);
   }
}

TEST(RNTuple, TClassTemplatedBase)
{
   // For non-cxxmodules builds, cling needs to parse the header for the `SG::sgkey_t` type to be known
   gInterpreter->ProcessLine("#include \"CustomStruct.hxx\"");

   FileRaii fileGuard("test_ntuple_tclass_templatebase.ntuple");
   {
      auto model = RNTupleModel::Create();
      auto fieldKlass = model->MakeField<PackedContainer<int>>("klass");
      RNTupleWriteOptions options;
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath(), options);
      for (int i = 0; i < 10000; i++) {
         new (fieldKlass.get()) PackedContainer<int>({i + 2, i + 3},
                                                     {/*m_nbits=*/ (uint8_t)i,
                                                      /*m_nmantissa=*/ (uint8_t)i,
                                                      /*m_scale=*/ static_cast<float>(i + 1),
                                                      /*m_flags=*/ 0,
                                                      /*m_sgkey=*/ (uint32_t)(i + 1),
                                                      /*c_uint=*/ (uint8_t)i});
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
   EXPECT_EQ(10000U, ntuple->GetNEntries());
   auto viewKlass = ntuple->GetView<PackedContainer<int>>("klass");
   for (auto i : ntuple->GetEntryRange()) {
      float fi = static_cast<float>(i);
      EXPECT_EQ(((uint8_t)i), viewKlass(i).m_params.m_nbits);
      EXPECT_EQ(((uint8_t)i), viewKlass(i).m_params.m_nmantissa);
      EXPECT_EQ((fi + 1), viewKlass(i).m_params.m_scale);
      EXPECT_EQ(0, viewKlass(i).m_params.m_flags);
      EXPECT_EQ(((uint8_t)i), viewKlass(i).m_params.c_uint);
      EXPECT_EQ(((uint32_t)(i + 1)), viewKlass(i).m_params.m_sgkey);

      EXPECT_EQ((std::vector<int>{static_cast<int>(i + 2),
                                  static_cast<int>(i + 3)}), viewKlass(i));
   }
}

TEST(RNTuple, Enums)
{
   FileRaii fileGuard("test_ntuple_enums.ntuple");

   {
      auto model = RNTupleModel::Create();
      auto fieldKlass = model->MakeField<StructWithEnums>("klass");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
   ASSERT_EQ(1U, ntuple->GetNEntries());
   auto viewKlass = ntuple->GetView<StructWithEnums>("klass");
   EXPECT_EQ(42, viewKlass(0).a);
   EXPECT_EQ(137, viewKlass(0).b);
}
