#include "ntuple_test.hxx"
#include "SimpleCollectionProxy.hxx"
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
   EXPECT_STREQ(
      "std::tuple<std::tuple<char,CustomStruct,char>,std::int32_t>",
      (ROOT::Experimental::RField<std::tuple<std::tuple<char, CustomStruct, char>, int>>::TypeName().c_str()));
}


TEST(RNTuple, CreateField)
{
   auto field = RFieldBase::Create("test", "vector<unsigned int>").Unwrap();
   EXPECT_STREQ("std::vector<std::uint32_t>", field->GetType().c_str());
   auto value = field->GenerateValue();
   field->DestroyValue(value);

   std::vector<std::unique_ptr<RFieldBase>> itemFields;
   itemFields.push_back(std::make_unique<RField<std::uint32_t>>("u32"));
   itemFields.push_back(std::make_unique<RField<std::uint8_t>>("u8"));
   ROOT::Experimental::RRecordField record("test", itemFields);
   EXPECT_EQ(alignof(std::uint32_t), record.GetAlignment());
   // Check that trailing padding is added after `u8` to comply with the alignment requirements of uint32_t
   EXPECT_EQ(sizeof(std::uint32_t) + alignof(std::uint32_t), record.GetValueSize());
}

TEST(RNTuple, ArrayField)
{
   auto field = RFieldBase::Create("test", "int32_t[10]").Unwrap();
   EXPECT_EQ((10 * sizeof(int32_t)), field->GetValueSize());
   EXPECT_EQ(alignof(int32_t[10]), field->GetAlignment());
   auto otherField = RFieldBase::Create("test", "std::vector<float[3]   > [10 ]  ").Unwrap();
   EXPECT_EQ((10 * sizeof(std::vector<float[3]>)), otherField->GetValueSize());

   // Malformed type names
   EXPECT_THROW(RFieldBase::Create("test", "unsigned int[]").Unwrap(), ROOT::Experimental::RException);
   EXPECT_THROW(RFieldBase::Create("test", "unsigned int [[2").Unwrap(), ROOT::Experimental::RException);
   EXPECT_THROW(RFieldBase::Create("test", "unsigned[2] int[10]").Unwrap(), ROOT::Experimental::RException);

   // Multi-dimensional arrays are not currently supported
   EXPECT_THROW(RFieldBase::Create("test", "int32_t[10][11]").Unwrap(), ROOT::Experimental::RException);

   unsigned char charArray[] = {0x00, 0x01, 0x02, 0x03};

   FileRaii fileGuard("test_ntuple_rfield_array.root");
   {
      auto model = RNTupleModel::Create();
      auto struct_field = model->MakeField<StructWithArrays>("struct");
      model->AddField(std::make_unique<RField<float[2]>>("array1"));
      model->AddField(RFieldBase::Create("array2", "unsigned char[4]").Unwrap());

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      auto array1_field = ntuple->GetModel()->GetDefaultEntry()->Get<float[2]>("array1");
      auto array2_field = ntuple->GetModel()->GetDefaultEntry()->Get<unsigned char[4]>("array2");
      for (int i = 0; i < 2; i++) {
         new (struct_field.get()) StructWithArrays({{'n', 't', 'p', 'l'}, {1.0, 42.0}});
         new (array1_field) float[2]{0.0f, static_cast<float>(i)};
         memcpy(array2_field, charArray, sizeof(charArray));
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(2, ntuple->GetNEntries());
   auto viewStruct = ntuple->GetView<StructWithArrays>("struct");
   auto viewArray1 = ntuple->GetView<float[2]>("array1");
   auto viewArray2 = ntuple->GetView<unsigned char[4]>("array2");
   for (auto i : ntuple->GetEntryRange()) {
      EXPECT_EQ(0, memcmp(viewStruct(i).c, "ntpl", 4));
      EXPECT_EQ(1.0f, viewStruct(i).f[0]);
      EXPECT_EQ(42.0f, viewStruct(i).f[1]);

      float fs[] = {0.0f, static_cast<float>(i)};
      EXPECT_EQ(0, memcmp(viewArray1(i), fs, sizeof(fs)));
      EXPECT_EQ(0, memcmp(viewArray2(i), charArray, sizeof(charArray)));
   }
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

TEST(RNTuple, StdTuple)
{
   auto field = RField<std::tuple<char, int64_t, char>>("tupleField");
   EXPECT_STREQ("std::tuple<char,std::int64_t,char>", field.GetType().c_str());
   auto otherField = RFieldBase::Create("test", "std::tuple<char, int64_t, char>").Unwrap();
   EXPECT_STREQ(field.GetType().c_str(), otherField->GetType().c_str());
   EXPECT_EQ((sizeof(std::tuple<char, int64_t, char>)), field.GetValueSize());
   EXPECT_EQ((sizeof(std::tuple<char, int64_t, char>)), otherField->GetValueSize());
   EXPECT_EQ((alignof(std::tuple<char, int64_t, char>)), field.GetAlignment());
   EXPECT_EQ((alignof(std::tuple<char, int64_t, char>)), otherField->GetAlignment());

   auto tupleTupleField =
      RField<std::tuple<std::tuple<int64_t, float, char, float>, std::vector<std::tuple<char, char, char>>>>(
         "tupleTupleField");
   EXPECT_STREQ("std::tuple<std::tuple<std::int64_t,float,char,float>,std::vector<std::tuple<char,char,char>>>",
                tupleTupleField.GetType().c_str());

   FileRaii fileGuard("test_ntuple_rfield_stdtuple.root");
   {
      auto model = RNTupleModel::Create();
      auto tuple_field = model->MakeField<std::tuple<char, float, std::string, char>>({"myTuple", "4-tuple"});
      auto myTuple2 = RFieldBase::Create("myTuple2", "std::tuple<char, float, std::string, char>").Unwrap();
      auto myTuple3 = RFieldBase::Create("myTuple3", "std::tuple<int32_t, std::tuple<std::string, char>>").Unwrap();
      model->AddField(std::move(myTuple2));
      model->AddField(std::move(myTuple3));

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "tuple_ntuple", fileGuard.GetPath());
      auto tuple_field2 =
         ntuple->GetModel()->GetDefaultEntry()->Get<std::tuple<char, float, std::string, char>>("myTuple2");
      auto tuple_field3 =
         ntuple->GetModel()->GetDefaultEntry()->Get<std::tuple<int32_t, std::tuple<std::string, char>>>("myTuple3");
      for (int i = 0; i < 2; i++) {
         *tuple_field = {'A' + i, static_cast<float>(i), std::to_string(i), '0' + i};
         *tuple_field2 = {'B' + i, static_cast<float>(i), std::to_string(i), '1' + i};
         *tuple_field3 = {i, {std::to_string(i), '2' + i}};
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open("tuple_ntuple", fileGuard.GetPath());
   EXPECT_EQ(2, ntuple->GetNEntries());

   auto viewTuple = ntuple->GetView<std::tuple<char, float, std::string, char>>("myTuple");
   auto viewTuple2 = ntuple->GetView<std::tuple<char, float, std::string, char>>("myTuple2");
   auto viewTuple3 = ntuple->GetView<std::tuple<int32_t, std::tuple<std::string, char>>>("myTuple3");
   for (auto i : ntuple->GetEntryRange()) {
      EXPECT_EQ(static_cast<char>('A' + i), std::get<0>(viewTuple(i)));
      EXPECT_EQ(static_cast<double>(i), std::get<1>(viewTuple(i)));
      EXPECT_EQ(std::to_string(i), std::get<2>(viewTuple(i)));
      EXPECT_EQ(static_cast<char>('0' + i), std::get<3>(viewTuple(i)));

      EXPECT_EQ(static_cast<char>('B' + i), std::get<0>(viewTuple2(i)));
      EXPECT_EQ(static_cast<double>(i), std::get<1>(viewTuple2(i)));
      EXPECT_EQ(std::to_string(i), std::get<2>(viewTuple2(i)));
      EXPECT_EQ(static_cast<char>('1' + i), std::get<3>(viewTuple2(i)));

      EXPECT_EQ(static_cast<int32_t>(i), std::get<0>(viewTuple3(i)));
      const auto &nested = std::get<1>(viewTuple3(i));
      EXPECT_EQ(std::to_string(i), std::get<0>(nested));
      EXPECT_EQ(static_cast<char>('2' + i), std::get<1>(nested));
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

TEST(RNTuple, TVirtualCollectionProxy)
{
   SimpleCollectionProxy<StructUsingCollectionProxy<char>> proxyC;
   SimpleCollectionProxy<StructUsingCollectionProxy<float>> proxyF;
   SimpleCollectionProxy<StructUsingCollectionProxy<CustomStruct>> proxyS;
   SimpleCollectionProxy<StructUsingCollectionProxy<StructUsingCollectionProxy<float>>> proxyNested;

   // `RCollectionClassField` instantiated but no collection proxy set (yet)
   EXPECT_THROW(RField<StructUsingCollectionProxy<float>>("hasTraitButNoCollectionProxySet"),
                ROOT::Experimental::RException);

   auto klassC = TClass::GetClass("StructUsingCollectionProxy<char>");
   klassC->CopyCollectionProxy(proxyC);
   auto klassF = TClass::GetClass("StructUsingCollectionProxy<float>");
   klassF->CopyCollectionProxy(proxyF);
   auto klassS = TClass::GetClass("StructUsingCollectionProxy<CustomStruct>");
   klassS->CopyCollectionProxy(proxyS);
   auto klassNested = TClass::GetClass("StructUsingCollectionProxy<StructUsingCollectionProxy<float>>");
   klassNested->CopyCollectionProxy(proxyNested);

   // `RClassField` instantiated (due to intentionally missing `IsCollectionProxy<T>` trait) but a collection proxy is
   // set
   auto klassI = TClass::GetClass("StructUsingCollectionProxy<int>");
   klassI->CopyCollectionProxy(SimpleCollectionProxy<StructUsingCollectionProxy<int>>{});
   EXPECT_THROW(RField<StructUsingCollectionProxy<int>>("noTraitButCollectionProxySet"),
                ROOT::Experimental::RException);

   auto field = RField<StructUsingCollectionProxy<float>>("c");
   EXPECT_EQ(sizeof(StructUsingCollectionProxy<float>), field.GetValueSize());

   StructUsingCollectionProxy<float> proxiedVecF;
   for (unsigned i = 0; i < 10; ++i) {
      proxiedVecF.v.push_back(static_cast<float>(i));
   }

   FileRaii fileGuard("test_ntuple_tvirtualcollectionproxy.ntuple");
   {
      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("C", "StructUsingCollectionProxy<char>").Unwrap());
      model->AddField(RFieldBase::Create("F", "StructUsingCollectionProxy<float>").Unwrap());
      model->AddField(RFieldBase::Create("S", "StructUsingCollectionProxy<CustomStruct>").Unwrap());
      auto fieldVproxyF = model->MakeField<std::vector<StructUsingCollectionProxy<float>>>("VproxyF");
      auto fieldNested = model->MakeField<StructUsingCollectionProxy<StructUsingCollectionProxy<float>>>("nested");

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
      auto fieldC = ntuple->GetModel()->GetDefaultEntry()->Get<StructUsingCollectionProxy<char>>("C");
      auto fieldF = ntuple->GetModel()->GetDefaultEntry()->Get<StructUsingCollectionProxy<float>>("F");
      auto fieldS = ntuple->GetModel()->GetDefaultEntry()->Get<StructUsingCollectionProxy<CustomStruct>>("S");
      for (unsigned i = 0; i < 1000; ++i) {
         if ((i % 100) == 0) {
            fieldC->v.clear();
            fieldF->v.clear();
            fieldS->v.clear();
         }
         fieldC->v.push_back(42);
         fieldF->v.push_back(static_cast<float>(i % 100));

         std::vector<float> v1;
         for (unsigned j = 0, nItems = (i % 10); j < nItems; ++j) {
            v1.push_back(static_cast<float>(j));
         }
         fieldS->v.push_back(CustomStruct{
            /*a=*/static_cast<float>(i % 100),
            /*v1=*/std::move(v1),
            /*v2=*/std::vector<std::vector<float>>{{static_cast<float>(i % 100)}, {static_cast<float>((i % 100) + 1)}},
            /*s=*/"hello" + std::to_string(i % 100)});

         fieldVproxyF->push_back(proxiedVecF);
         fieldNested->v.push_back(proxiedVecF);
         ntuple->Fill();
      }
   }

   {
      auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
      EXPECT_EQ(1000U, ntuple->GetNEntries());
      auto viewC = ntuple->GetView<StructUsingCollectionProxy<char>>("C");
      auto viewF = ntuple->GetView<StructUsingCollectionProxy<float>>("F");
      auto viewS = ntuple->GetView<StructUsingCollectionProxy<CustomStruct>>("S");
      auto viewVproxyF = ntuple->GetView<std::vector<StructUsingCollectionProxy<float>>>("VproxyF");
      auto viewNested = ntuple->GetView<StructUsingCollectionProxy<StructUsingCollectionProxy<float>>>("nested");
      for (auto i : ntuple->GetEntryRange()) {
         auto &collC = viewC(i);
         auto &collF = viewF(i);
         auto &collS = viewS(i);
         auto &collVproxyF = viewVproxyF(i);
         auto &collNested = viewNested(i);

         EXPECT_EQ((i % 100) + 1, collC.v.size());
         for (unsigned j = 0; j < collC.v.size(); ++j) {
            EXPECT_EQ(42, collC.v[j]);
         }
         EXPECT_EQ((i % 100) + 1, collF.v.size());
         for (unsigned j = 0; j < collF.v.size(); ++j) {
            EXPECT_EQ(static_cast<float>(j), collF.v[j]);
         }

         EXPECT_EQ((i % 100) + 1, collS.v.size());
         for (unsigned j = 0; j < collS.v.size(); ++j) {
            const auto &item = collS.v[j];
            EXPECT_EQ(static_cast<float>(j), item.a);
            for (unsigned k = 0; k < item.v1.size(); ++k) {
               EXPECT_EQ(static_cast<float>(k), item.v1[k]);
            }
            EXPECT_EQ(static_cast<float>(j), item.v2[0][0]);
            EXPECT_EQ(static_cast<float>(j + 1), item.v2[1][0]);
            EXPECT_EQ("hello" + std::to_string(j), item.s);
         }

         EXPECT_EQ(i + 1, collVproxyF.size());
         for (unsigned j = 0; j < collVproxyF.size(); ++j) {
            EXPECT_EQ(proxiedVecF.v, collVproxyF[i].v);
         }
         EXPECT_EQ(i + 1, collNested.v.size());
         for (unsigned j = 0; j < collNested.v.size(); ++j) {
            EXPECT_EQ(proxiedVecF.v, collNested.v[i].v);
         }
      }
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

TEST(RNTuple, Traits)
{
   EXPECT_EQ(RFieldBase::kTraitTrivialType | RFieldBase::kTraitMappable, RField<float>("f").GetTraits());
   EXPECT_EQ(RFieldBase::kTraitTrivialType | RFieldBase::kTraitMappable, RField<bool>("f").GetTraits());
   EXPECT_EQ(RFieldBase::kTraitTrivialType | RFieldBase::kTraitMappable, RField<int>("f").GetTraits());
   EXPECT_EQ(0, RField<std::string>("f").GetTraits());
   EXPECT_EQ(0, RField<std::vector<float>>("f").GetTraits());
   EXPECT_EQ(0, RField<std::vector<bool>>("f").GetTraits());
   EXPECT_EQ(0, RField<ROOT::RVec<float>>("f").GetTraits());
   EXPECT_EQ(0, RField<ROOT::RVec<bool>>("f").GetTraits());
   auto f1 = RField<std::pair<float, std::string>>("f");
   EXPECT_EQ(0, f1.GetTraits());
   auto f2 = RField<std::pair<float, int>>("f");
   EXPECT_EQ(RFieldBase::kTraitTrivialType, f2.GetTraits());
   auto f3 = RField<std::variant<float, std::string>>("f");
   EXPECT_EQ(0, f3.GetTraits());
   auto f4 = RField<std::variant<float, int>>("f");
   EXPECT_EQ(RFieldBase::kTraitTriviallyDestructible, f4.GetTraits());
   auto f5 = RField<std::tuple<float, std::string>>("f");
   EXPECT_EQ(0, f5.GetTraits());
   auto f6 = RField<std::tuple<float, int>>("f");
   EXPECT_EQ(RFieldBase::kTraitTrivialType, f6.GetTraits());
   auto f7 = RField<std::tuple<float, std::variant<int, float>>>("f");
   EXPECT_EQ(RFieldBase::kTraitTriviallyDestructible, f7.GetTraits());
   auto f8 = RField<std::array<float, 3>>("f");
   EXPECT_EQ(RFieldBase::kTraitTrivialType, f8.GetTraits());
   auto f9 = RField<std::array<std::string, 3>>("f");
   EXPECT_EQ(0, f9.GetTraits());

   EXPECT_EQ(RFieldBase::kTraitTrivialType, RField<TrivialTraits>("f").GetTraits());
   EXPECT_EQ(0, RField<TransientTraits>("f").GetTraits());
   EXPECT_EQ(RFieldBase::kTraitTriviallyDestructible, RField<VariantTraits>("f").GetTraits());
   EXPECT_EQ(0, RField<StringTraits>("f").GetTraits());
   EXPECT_EQ(RFieldBase::kTraitTriviallyDestructible, RField<ConstructorTraits>("f").GetTraits());
   EXPECT_EQ(RFieldBase::kTraitTriviallyConstructible, RField<DestructorTraits>("f").GetTraits());
}
