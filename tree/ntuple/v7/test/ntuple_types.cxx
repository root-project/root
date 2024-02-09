#include "ntuple_test.hxx"
#include "SimpleCollectionProxy.hxx"
#include "ROOT/TestSupport.hxx"
#include "TInterpreter.h"

#include <bitset>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

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

TEST(RNTuple, EnumBasics)
{
   // Needs fix of TEnum
   // auto stdEnum = RFieldBase::Create("f", "std::byte");
   // EXPECT_FALSE(stdEnum);

   auto f = RFieldBase::Create("f", "CustomEnum").Unwrap();

   auto model = RNTupleModel::Create();
   auto ptrEnum = model->MakeField<CustomEnum>("e");
   auto ptrVecEnum = model->MakeField<std::vector<CustomEnum>>("ve");
   model->MakeField<StructWithEnums>("swe");

   EXPECT_EQ(model->GetField("e").GetTypeName(), f->GetTypeName());

   FileRaii fileGuard("test_ntuple_enum_basics.root");
   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrEnum = kCustomEnumVal;
      ptrVecEnum->emplace_back(kCustomEnumVal);
      ptrVecEnum->emplace_back(kCustomEnumVal);
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(1, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_EQ(kCustomEnumVal, *reader->GetModel().GetDefaultEntry().GetPtr<CustomEnum>("e"));
   auto ptrStructWithEnums = reader->GetModel().GetDefaultEntry().GetPtr<StructWithEnums>("swe");
   EXPECT_EQ(42, ptrStructWithEnums->a);
   EXPECT_EQ(137, ptrStructWithEnums->b);
   EXPECT_EQ(kCustomEnumVal, ptrStructWithEnums->e);
   EXPECT_EQ(2u, reader->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomEnum>>("ve")->size());
   EXPECT_EQ(kCustomEnumVal, reader->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomEnum>>("ve")->at(0));
   EXPECT_EQ(kCustomEnumVal, reader->GetModel().GetDefaultEntry().GetPtr<std::vector<CustomEnum>>("ve")->at(1));
}

using EnumClassInts = ::testing::Types<CustomEnumInt8, CustomEnumUInt8, CustomEnumInt16, CustomEnumUInt16,
                                       CustomEnumInt32, CustomEnumUInt32, CustomEnumInt64, CustomEnumUInt64>;

template <typename EnumT>
class EnumClass : public ::testing::Test {
public:
   using Enum_t = EnumT;
};

TYPED_TEST_SUITE(EnumClass, EnumClassInts);

TYPED_TEST(EnumClass, Widths)
{
   using ThisEnum_t = typename TestFixture::Enum_t;
   using Underlying_t = std::underlying_type_t<ThisEnum_t>;

   auto enumName = RField<ThisEnum_t>::TypeName();

   FileRaii fileGuard("test_ntuple_enum_class_" + enumName + ".root");
   {
      auto model = RNTupleModel::Create();
      auto ptrEnum = model->MakeField<ThisEnum_t>("e");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrEnum = static_cast<ThisEnum_t>(0);
      writer->Fill();
      *ptrEnum = static_cast<ThisEnum_t>(1);
      writer->Fill();
      *ptrEnum = static_cast<ThisEnum_t>(std::numeric_limits<Underlying_t>::max());
      writer->Fill();
      *ptrEnum = static_cast<ThisEnum_t>(std::numeric_limits<Underlying_t>::max() - 1);
      writer->Fill();
      if (std::is_signed_v<Underlying_t>) {
         *ptrEnum = static_cast<ThisEnum_t>(-1);
         writer->Fill();
         *ptrEnum = static_cast<ThisEnum_t>(std::numeric_limits<Underlying_t>::min());
         writer->Fill();
         *ptrEnum = static_cast<ThisEnum_t>(std::numeric_limits<Underlying_t>::min() + 1);
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   auto ptrEnum = reader->GetModel().GetDefaultEntry().GetPtr<ThisEnum_t>("e");
   reader->LoadEntry(0);
   EXPECT_EQ(static_cast<ThisEnum_t>(0), *ptrEnum);
   reader->LoadEntry(1);
   EXPECT_EQ(static_cast<ThisEnum_t>(1), *ptrEnum);
   reader->LoadEntry(2);
   EXPECT_EQ(static_cast<ThisEnum_t>(std::numeric_limits<Underlying_t>::max()), *ptrEnum);
   reader->LoadEntry(3);
   EXPECT_EQ(static_cast<ThisEnum_t>(std::numeric_limits<Underlying_t>::max() - 1), *ptrEnum);

   if (!std::is_signed_v<Underlying_t>)
      return;

   reader->LoadEntry(4);
   EXPECT_EQ(static_cast<ThisEnum_t>(-1), *ptrEnum);
   reader->LoadEntry(5);
   EXPECT_EQ(static_cast<ThisEnum_t>(std::numeric_limits<Underlying_t>::min()), *ptrEnum);
   reader->LoadEntry(6);
   EXPECT_EQ(static_cast<ThisEnum_t>(std::numeric_limits<Underlying_t>::min() + 1), *ptrEnum);
}

TEST(RNTuple, CreateField)
{
   auto field = RFieldBase::Create("test", "vector<unsigned int>").Unwrap();
   EXPECT_STREQ("std::vector<std::uint32_t>", field->GetTypeName().c_str());
   auto value = field->CreateValue();

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
   EXPECT_THROW(RFieldBase::Create("test", "unsigned[2] int[10").Unwrap(), ROOT::Experimental::RException);

   unsigned char charArray[] = {0x00, 0x01, 0x02, 0x03};

   FileRaii fileGuard("test_ntuple_rfield_array.root");
   {
      auto model = RNTupleModel::Create();
      auto struct_field = model->MakeField<StructWithArrays>("struct");
      model->AddField(std::make_unique<RField<float[2]>>("array1"));
      model->AddField(RFieldBase::Create("array2", "unsigned char[4]").Unwrap());

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      auto array1_field = ntuple->GetModel().GetDefaultEntry().GetPtr<float[2]>("array1");
      auto array2_field = ntuple->GetModel().GetDefaultEntry().GetPtr<unsigned char[4]>("array2");
      for (int i = 0; i < 2; i++) {
         new (struct_field.get()) StructWithArrays({{'n', 't', 'p', 'l'}, {1.0, 42.0}, {{2*i}, {2*i + 1}}});
         new (array1_field.get()) float[2]{0.0f, static_cast<float>(i)};
         memcpy(array2_field.get(), charArray, sizeof(charArray));
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
      EXPECT_EQ(2*i, viewStruct(i).i[0][0]);
      EXPECT_EQ(2*i + 1, viewStruct(i).i[1][0]);

      float fs[] = {0.0f, static_cast<float>(i)};
      EXPECT_EQ(0, memcmp(viewArray1(i), fs, sizeof(fs)));
      EXPECT_EQ(0, memcmp(viewArray2(i), charArray, sizeof(charArray)));
   }
}

TEST(RNTuple, NDimArrayField)
{
   auto field = RFieldBase::Create("test", "int32_t[2][3]").Unwrap();
   EXPECT_EQ((6 * sizeof(int32_t)), field->GetValueSize());
   EXPECT_EQ(alignof(int32_t[2][3]), field->GetAlignment());

   FileRaii fileGuard("test_ntuple_rfield_array.root");
   {
      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("dim2", "int[2][3]").Unwrap());
      model->AddField(RFieldBase::Create("dim3", "int[1][2][3]").Unwrap());

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      auto dim2_field = ntuple->GetModel().GetDefaultEntry().GetPtr<int[2][3]>("dim2");
      auto dim3_field = ntuple->GetModel().GetDefaultEntry().GetPtr<int[1][2][3]>("dim3");
      (dim2_field.get())[0][0] = 0;
      (dim2_field.get())[0][1] = 1;
      (dim2_field.get())[0][2] = 2;
      (dim2_field.get())[1][0] = 3;
      (dim2_field.get())[1][1] = 4;
      (dim2_field.get())[1][2] = 5;
      (dim3_field.get())[0][0][0] = 0;
      (dim3_field.get())[0][0][1] = 1;
      (dim3_field.get())[0][0][2] = 2;
      (dim3_field.get())[0][1][0] = 3;
      (dim3_field.get())[0][1][1] = 4;
      (dim3_field.get())[0][1][2] = 5;
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1, ntuple->GetNEntries());
   auto viewDim2 = ntuple->GetView<int[2][3]>("dim2");
   auto viewDim3 = ntuple->GetView<int[1][2][3]>("dim3");
   EXPECT_EQ(0, viewDim2(0)[0][0]);
   EXPECT_EQ(1, viewDim2(0)[0][1]);
   EXPECT_EQ(2, viewDim2(0)[0][2]);
   EXPECT_EQ(3, viewDim2(0)[1][0]);
   EXPECT_EQ(4, viewDim2(0)[1][1]);
   EXPECT_EQ(5, viewDim2(0)[1][2]);
   EXPECT_EQ(0, viewDim3(0)[0][0][0]);
   EXPECT_EQ(1, viewDim3(0)[0][0][1]);
   EXPECT_EQ(2, viewDim3(0)[0][0][2]);
   EXPECT_EQ(3, viewDim3(0)[0][1][0]);
   EXPECT_EQ(4, viewDim3(0)[0][1][1]);
   EXPECT_EQ(5, viewDim3(0)[0][1][2]);
}

TEST(RNTuple, StdPair)
{
   auto field = RField<std::pair<int64_t, float>>("pairField");
   EXPECT_STREQ("std::pair<std::int64_t,float>", field.GetTypeName().c_str());
   auto otherField = RFieldBase::Create("test", "std::pair<int64_t, float>").Unwrap();
   EXPECT_EQ(field.GetTypeName(), otherField->GetTypeName());
   EXPECT_EQ((sizeof(std::pair<int64_t, float>)), field.GetValueSize());
   EXPECT_EQ((sizeof(std::pair<int64_t, float>)), otherField->GetValueSize());
   EXPECT_EQ((alignof(std::pair<int64_t, float>)), field.GetAlignment());
   EXPECT_EQ((alignof(std::pair<int64_t, float>)), otherField->GetAlignment());

   auto pairPairField = RField<std::pair<std::pair<int64_t, float>,
      std::vector<std::pair<CustomStruct, double>>>>("pairPairField");
   EXPECT_STREQ("std::pair<std::pair<std::int64_t,float>,std::vector<std::pair<CustomStruct,double>>>",
                pairPairField.GetTypeName().c_str());

   FileRaii fileGuard("test_ntuple_rfield_stdpair.root");
   {
      auto model = RNTupleModel::Create();
      auto pair_field = model->MakeField<std::pair<double, std::string>>(
         {"myPair", "a very cool field"}
      );
      auto myPair2 = RFieldBase::Create("myPair2", "std::pair<double, std::string>").Unwrap();
      model->AddField(std::move(myPair2));

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "pair_ntuple", fileGuard.GetPath());
      auto pair_field2 = ntuple->GetModel().GetDefaultEntry().GetPtr<std::pair<double, std::string>>("myPair2");
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
   EXPECT_STREQ("std::tuple<char,std::int64_t,char>", field.GetTypeName().c_str());
   auto otherField = RFieldBase::Create("test", "std::tuple<char, int64_t, char>").Unwrap();
   EXPECT_EQ(field.GetTypeName(), otherField->GetTypeName());
   EXPECT_EQ((sizeof(std::tuple<char, int64_t, char>)), field.GetValueSize());
   EXPECT_EQ((sizeof(std::tuple<char, int64_t, char>)), otherField->GetValueSize());
   EXPECT_EQ((alignof(std::tuple<char, int64_t, char>)), field.GetAlignment());
   EXPECT_EQ((alignof(std::tuple<char, int64_t, char>)), otherField->GetAlignment());

   auto tupleTupleField =
      RField<std::tuple<std::tuple<int64_t, float, char, float>, std::vector<std::tuple<char, char, char>>>>(
         "tupleTupleField");
   EXPECT_STREQ("std::tuple<std::tuple<std::int64_t,float,char,float>,std::vector<std::tuple<char,char,char>>>",
                tupleTupleField.GetTypeName().c_str());

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
         ntuple->GetModel().GetDefaultEntry().GetPtr<std::tuple<char, float, std::string, char>>("myTuple2");
      auto tuple_field3 =
         ntuple->GetModel().GetDefaultEntry().GetPtr<std::tuple<int32_t, std::tuple<std::string, char>>>("myTuple3");
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

TEST(RNTuple, StdSet)
{
   auto field = RField<std::set<int64_t>>("setField");
   EXPECT_STREQ("std::set<std::int64_t>", field.GetTypeName().c_str());
   auto otherField = RFieldBase::Create("test", "std::set<int64_t>").Unwrap();
   EXPECT_EQ(field.GetTypeName(), otherField->GetTypeName());
   EXPECT_EQ((sizeof(std::set<int64_t>)), field.GetValueSize());
   EXPECT_EQ((sizeof(std::set<int64_t>)), otherField->GetValueSize());
   EXPECT_EQ((alignof(std::set<int64_t>)), field.GetAlignment());
   // For type-erased set fields, we use `alignof(std::set<std::max_align_t>)` to set the alignment,
   // so the actual alignment may be smaller.
   EXPECT_LE((alignof(std::set<int64_t>)), otherField->GetAlignment());

   auto setSetField = RField<std::set<std::set<CustomStruct>>>("setSetField");
   EXPECT_STREQ("std::set<std::set<CustomStruct>>", setSetField.GetTypeName().c_str());

   FileRaii fileGuard("test_ntuple_rfield_stdset.root");
   {
      auto model = RNTupleModel::Create();
      auto set_field = model->MakeField<std::set<float>>({"mySet", "float set"});
      auto set_field2 = model->MakeField<std::set<std::tuple<int, char, CustomStruct>>>({"mySet2"});

      auto mySet3 = RFieldBase::Create("mySet3", "std::set<std::string>").Unwrap();
      auto mySet4 = RFieldBase::Create("mySet4", "std::set<std::set<char>>").Unwrap();

      model->AddField(std::move(mySet3));
      model->AddField(std::move(mySet4));

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "set_ntuple", fileGuard.GetPath());
      auto set_field3 = ntuple->GetModel().GetDefaultEntry().GetPtr<std::set<std::string>>("mySet3");
      auto set_field4 = ntuple->GetModel().GetDefaultEntry().GetPtr<std::set<std::set<char>>>("mySet4");
      for (int i = 0; i < 2; i++) {
         *set_field = {static_cast<float>(i), 3.14, 0.42};
         *set_field2 = {
            std::make_tuple(i, static_cast<char>(i + 65), CustomStruct{6.f, {7.f, 8.f}, {{9.f}, {10.f}}, "foo"}),
            std::make_tuple(i + 1, static_cast<char>(i + 97), CustomStruct{2.f, {3.f, 4.f}, {{5.f}, {6.f}}, "bar"})};
         *set_field3 = {"Hello", "world!", std::to_string(i)};
         *set_field4 = {{static_cast<char>(i), 'a'}, {'r', 'o', 'o', 't'}, {'h', 'i'}};
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open("set_ntuple", fileGuard.GetPath());
   EXPECT_EQ(2, ntuple->GetNEntries());

   auto viewSet = ntuple->GetView<std::set<float>>("mySet");
   auto viewSet2 = ntuple->GetView<std::set<std::tuple<int, char, CustomStruct>>>("mySet2");
   auto viewSet3 = ntuple->GetView<std::set<std::string>>("mySet3");
   auto viewSet4 = ntuple->GetView<std::set<std::set<char>>>("mySet4");
   for (auto i : ntuple->GetEntryRange()) {
      EXPECT_EQ(std::set<float>({static_cast<float>(i), 3.14, 0.42}), viewSet(i));

      auto tupleSet = std::set<std::tuple<int, char, CustomStruct>>(
         {std::make_tuple(i, static_cast<char>(i + 65), CustomStruct{6.f, {7.f, 8.f}, {{9.f}, {10.f}}, "foo"}),
          std::make_tuple(i + 1, static_cast<char>(i + 97), CustomStruct{2.f, {3.f, 4.f}, {{5.f}, {6.f}}, "bar"})});
      EXPECT_EQ(tupleSet, viewSet2(i));

      EXPECT_EQ(std::set<std::string>({"Hello", "world!", std::to_string(i)}), viewSet3(i));
      EXPECT_EQ(std::set<std::set<char>>({{static_cast<char>(i), 'a'}, {'r', 'o', 'o', 't'}, {'h', 'i'}}), viewSet4(i));
   }

   ntuple->LoadEntry(0);
   auto mySet2 = ntuple->GetModel().GetDefaultEntry().GetPtr<std::set<std::tuple<int, char, CustomStruct>>>("mySet2");
   auto tupleSet = std::set<std::tuple<int, char, CustomStruct>>(
      {std::make_tuple(0, 'A', CustomStruct{6.f, {7.f, 8.f}, {{9.f}, {10.f}}, "foo"}),
       std::make_tuple(1, 'a', CustomStruct{2.f, {3.f, 4.f}, {{5.f}, {6.f}}, "bar"})});
   EXPECT_EQ(tupleSet, *mySet2);
}

TEST(RNTuple, StdUnorderedSet)
{
   auto field = RField<std::unordered_set<int64_t>>("setField");
   EXPECT_STREQ("std::unordered_set<std::int64_t>", field.GetTypeName().c_str());
   auto otherField = RFieldBase::Create("test", "std::unordered_set<int64_t>").Unwrap();
   EXPECT_EQ(field.GetTypeName(), otherField->GetTypeName());
   EXPECT_EQ((sizeof(std::unordered_set<int64_t>)), field.GetValueSize());
   EXPECT_EQ((sizeof(std::unordered_set<int64_t>)), otherField->GetValueSize());
   EXPECT_EQ((alignof(std::unordered_set<int64_t>)), field.GetAlignment());
   // For type-erased set fields, we use `alignof(std::set<std::max_align_t>)` to set the alignment,
   // so the actual alignment may be smaller.
   EXPECT_LE((alignof(std::unordered_set<int64_t>)), otherField->GetAlignment());

   FileRaii fileGuard("test_ntuple_rfield_stdunorderedset.root");
   {
      auto model = RNTupleModel::Create();
      auto set_field = model->MakeField<std::unordered_set<float>>({"mySet", "unordered float set"});
      auto set_field2 = model->MakeField<std::unordered_set<CustomStruct>>({"mySet2"});

      auto mySet3 = RFieldBase::Create("mySet3", "std::unordered_set<std::string>").Unwrap();
      auto mySet4 = RFieldBase::Create("mySet4", "std::unordered_set<std::vector<bool>>").Unwrap();

      model->AddField(std::move(mySet3));
      model->AddField(std::move(mySet4));

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "set_ntuple", fileGuard.GetPath());
      auto set_field3 = ntuple->GetModel().GetDefaultEntry().GetPtr<std::unordered_set<std::string>>("mySet3");
      auto set_field4 = ntuple->GetModel().GetDefaultEntry().GetPtr<std::unordered_set<std::vector<bool>>>("mySet4");
      for (int i = 0; i < 2; i++) {
         *set_field = {static_cast<float>(i), 3.14, 0.42};
         *set_field2 = {CustomStruct{6.f, {7.f, 8.f}, {{9.f}, {10.f}}, "foo"},
                        CustomStruct{2.f, {3.f, 4.f}, {{5.f}, {6.f}}, "bar"}};
         *set_field3 = {"Hello", "world!", std::to_string(i)};
         *set_field4 = {{(i % 2 == 0)}, {}, {false, true}};
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open("set_ntuple", fileGuard.GetPath());
   EXPECT_EQ(2, ntuple->GetNEntries());

   auto viewSet = ntuple->GetView<std::unordered_set<float>>("mySet");
   auto viewSet2 = ntuple->GetView<std::unordered_set<CustomStruct>>("mySet2");
   auto viewSet3 = ntuple->GetView<std::unordered_set<std::string>>("mySet3");
   auto viewSet4 = ntuple->GetView<std::unordered_set<std::vector<bool>>>("mySet4");
   for (auto i : ntuple->GetEntryRange()) {
      EXPECT_EQ(std::unordered_set<float>({static_cast<float>(i), 3.14, 0.42}), viewSet(i));

      auto pairSet = std::unordered_set<CustomStruct>(
         {CustomStruct{6.f, {7.f, 8.f}, {{9.f}, {10.f}}, "foo"}, CustomStruct{2.f, {3.f, 4.f}, {{5.f}, {6.f}}, "bar"}});
      EXPECT_EQ(pairSet, viewSet2(i));

      EXPECT_EQ(std::unordered_set<std::string>({"Hello", "world!", std::to_string(i)}), viewSet3(i));
      EXPECT_EQ(std::unordered_set<std::vector<bool>>({{(i % 2 == 0)}, {}, {false, true}}), viewSet4(i));
   }

   ntuple->LoadEntry(0);
   auto mySet2 = ntuple->GetModel().GetDefaultEntry().GetPtr<std::unordered_set<CustomStruct>>("mySet2");
   auto pairSet = std::unordered_set<CustomStruct>(
      {CustomStruct{6.f, {7.f, 8.f}, {{9.f}, {10.f}}, "foo"}, CustomStruct{2.f, {3.f, 4.f}, {{5.f}, {6.f}}, "bar"}});
   EXPECT_EQ(pairSet, *mySet2);
}

TEST(RNTuple, StdMap)
{
   auto field = RField<std::map<char, int64_t>>("mapField");
   EXPECT_STREQ("std::map<char,std::int64_t>", field.GetTypeName().c_str());
   auto otherField = RFieldBase::Create("test", "std::map<char, int64_t>").Unwrap();
   EXPECT_EQ(field.GetTypeName(), otherField->GetTypeName());
   EXPECT_EQ((sizeof(std::map<char, int64_t>)), field.GetValueSize());
   EXPECT_EQ((sizeof(std::map<char, int64_t>)), otherField->GetValueSize());
   EXPECT_EQ((alignof(std::map<char, int64_t>)), field.GetAlignment());
   // For type-erased map fields, we use `alignof(std::map<std::max_align_t, std::max_align_t>)` to map the alignment,
   // so the actual alignment may be smaller.
   EXPECT_LE((alignof(std::map<char, int64_t>)), otherField->GetAlignment());
   // The assumption is that the alignment of inner items does not matter. If at any point there is a mismatch, this
   // test should fail.
   EXPECT_EQ((alignof(std::map<char, char>)), otherField->GetAlignment());

   auto mapMapField = RField<std::map<char, std::map<int, CustomStruct>>>("mapMapField");
   EXPECT_STREQ("std::map<char,std::map<std::int32_t,CustomStruct>>", mapMapField.GetTypeName().c_str());

   EXPECT_THROW(RFieldBase::Create("myInvalidMap", "std::map<char>").Unwrap(), RException);
   EXPECT_THROW(RFieldBase::Create("myInvalidMap", "std::map<char, std::string, int>").Unwrap(), RException);

   auto invalidInnerField = RFieldBase::Create("someIntField", "int").Unwrap();
   EXPECT_THROW(std::make_unique<ROOT::Experimental::RMapField>("myInvalidMap", "std::map<char, int>",
                                                                std::move(invalidInnerField)),
                RException);

   FileRaii fileGuard("test_ntuple_rfield_stdmap.root");
   {
      auto model = RNTupleModel::Create();
      auto map_field = model->MakeField<std::map<std::string, float>>({"myMap", "string to float map"});
      auto map_field2 = model->MakeField<std::map<int, std::vector<CustomStruct>>>({"myMap2"});

      auto myMap3 = RFieldBase::Create("myMap3", "std::map<char, std::string>").Unwrap();
      auto myMap4 = RFieldBase::Create("myMap4", "std::map<float, std::map<char, std::int32_t>>").Unwrap();

      model->AddField(std::move(myMap3));
      model->AddField(std::move(myMap4));

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "map_ntuple", fileGuard.GetPath());
      auto map_field3 = ntuple->GetModel().GetDefaultEntry().GetPtr<std::map<char, std::string>>("myMap3");
      auto map_field4 =
         ntuple->GetModel().GetDefaultEntry().GetPtr<std::map<float, std::map<char, std::int32_t>>>("myMap4");
      for (int i = 0; i < 2; i++) {
         *map_field = {{"foo", static_cast<float>(i + 0.1)},
                       {"bar", static_cast<float>(i * 0.2)},
                       {"baz", static_cast<float>(i * 0.3)}};
         *map_field2 = {{i,
                         {CustomStruct{6.f, {7.f, 8.f}, {{9.f}, {10.f}}, "foo"},
                          CustomStruct{2.f, {3.f, 4.f}, {{5.f}, {6.f}}, "bar"}}},
                        {i + 1, {CustomStruct{3.f, {4.f, 5.f}, {{1.f}, {2.f}}, "baz"}}}};
         *map_field3 = {{static_cast<char>(i), "Hello"}, {static_cast<char>(i), "world!"}};
         *map_field4 = {{static_cast<float>(i * 3.14), {{'a', static_cast<std::int32_t>(i)}}},
                        {static_cast<float>(i / 10),
                         {{'a', static_cast<std::int32_t>(i)}, {'b', static_cast<std::int32_t>(i * 2)}}}};
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open("map_ntuple", fileGuard.GetPath());
   EXPECT_EQ(2, ntuple->GetNEntries());

   auto viewMap = ntuple->GetView<std::map<std::string, float>>("myMap");
   auto viewMap2 = ntuple->GetView<std::map<int, std::vector<CustomStruct>>>("myMap2");
   auto viewMap3 = ntuple->GetView<std::map<char, std::string>>("myMap3");
   auto viewMap4 = ntuple->GetView<std::map<float, std::map<char, std::int32_t>>>("myMap4");
   for (auto i : ntuple->GetEntryRange()) {
      std::map<std::string, float> map1{{"foo", static_cast<float>(i + 0.1)},
                                        {"bar", static_cast<float>(i * 0.2)},
                                        {"baz", static_cast<float>(i * 0.3)}};
      EXPECT_EQ(map1, viewMap(i));

      std::map<int, std::vector<CustomStruct>> map2{
         {static_cast<int>(i),
          {CustomStruct{6.f, {7.f, 8.f}, {{9.f}, {10.f}}, "foo"},
           CustomStruct{2.f, {3.f, 4.f}, {{5.f}, {6.f}}, "bar"}}},
         {static_cast<int>(i + 1), {CustomStruct{3.f, {4.f, 5.f}, {{1.f}, {2.f}}, "baz"}}}};
      EXPECT_EQ(map2, viewMap2(i));

      std::map<char, std::string> map3{{static_cast<char>(i), "Hello"}, {static_cast<char>(i), "world!"}};
      EXPECT_EQ(map3, viewMap3(i));

      std::map<float, std::map<char, std::int32_t>> map4{
         {static_cast<float>(i * 3.14), {{'a', static_cast<std::int32_t>(i)}}},
         {static_cast<float>(i / 10), {{'a', static_cast<std::int32_t>(i)}, {'b', static_cast<std::int32_t>(i * 2)}}}};
      EXPECT_EQ(map4, viewMap4(i));
   }

   ntuple->LoadEntry(0);
   auto myMap2 = ntuple->GetModel().GetDefaultEntry().GetPtr<std::map<int, std::vector<CustomStruct>>>("myMap2");
   auto vecMap = std::map<int, std::vector<CustomStruct>>(
      {{0,
        {CustomStruct{6.f, {7.f, 8.f}, {{9.f}, {10.f}}, "foo"}, CustomStruct{2.f, {3.f, 4.f}, {{5.f}, {6.f}}, "bar"}}},
       {1, {CustomStruct{3.f, {4.f, 5.f}, {{1.f}, {2.f}}, "baz"}}}});
   EXPECT_EQ(vecMap, *myMap2);
}

TEST(RNTuple, StdUnorderedMap)
{
   auto field = RField<std::unordered_map<char, int64_t>>("mapField");
   EXPECT_STREQ("std::unordered_map<char,std::int64_t>", field.GetTypeName().c_str());
   auto otherField = RFieldBase::Create("test", "std::unordered_map<char, int64_t>").Unwrap();
   EXPECT_STREQ(field.GetTypeName().c_str(), otherField->GetTypeName().c_str());
   EXPECT_EQ((sizeof(std::unordered_map<char, int64_t>)), field.GetValueSize());
   EXPECT_EQ((sizeof(std::unordered_map<char, int64_t>)), otherField->GetValueSize());
   EXPECT_EQ((alignof(std::unordered_map<char, int64_t>)), field.GetAlignment());
   // For type-erased map fields, we use `alignof(std::map<std::max_align_t, std::max_align_t>)` to map the alignment,
   // so the actual alignment may be smaller.
   EXPECT_LE((alignof(std::unordered_map<char, int64_t>)), otherField->GetAlignment());
   // The assumption is that the alignment of inner items does not matter. If at any point there is a mismatch, this
   // test should fail.
   EXPECT_EQ((alignof(std::unordered_map<char, char>)), otherField->GetAlignment());

   EXPECT_THROW(RFieldBase::Create("myInvalidMap", "std::unordered_map<char>").Unwrap(), RException);
   EXPECT_THROW(RFieldBase::Create("myInvalidMap", "std::unordered_map<char, std::string, int>").Unwrap(), RException);

   FileRaii fileGuard("test_ntuple_rfield_stdunorderedmap.root");
   {
      auto model = RNTupleModel::Create();
      auto map_field =
         model->MakeField<std::unordered_map<std::string, float>>({"myMap", "unordered string to float map"});
      auto map_field2 = model->MakeField<std::unordered_map<int, CustomStruct>>({"myMap2"});

      auto myMap3 = RFieldBase::Create("myMap3", "std::unordered_map<char, std::string>").Unwrap();
      auto myMap4 = RFieldBase::Create("myMap4", "std::unordered_map<float, std::vector<bool>>").Unwrap();

      model->AddField(std::move(myMap3));
      model->AddField(std::move(myMap4));

      auto ntuple = RNTupleWriter::Recreate(std::move(model), "map_ntuple", fileGuard.GetPath());
      auto map_field3 = ntuple->GetModel().GetDefaultEntry().GetPtr<std::unordered_map<char, std::string>>("myMap3");
      auto map_field4 =
         ntuple->GetModel().GetDefaultEntry().GetPtr<std::unordered_map<float, std::vector<bool>>>("myMap4");
      for (int i = 0; i < 2; i++) {
         *map_field = {{"foo", static_cast<float>(i + 0.1)},
                       {"bar", static_cast<float>(i * 0.2)},
                       {"baz", static_cast<float>(i * 0.3)}};
         *map_field2 = {{i, CustomStruct{6.f, {7.f, 8.f}, {{9.f}, {10.f}}, "foo"}},
                        {i + 1, CustomStruct{3.f, {4.f, 5.f}, {{1.f}, {2.f}}, "baz"}}};
         *map_field3 = {{static_cast<char>(i), "Hello"}, {static_cast<char>(i), "world!"}};
         *map_field4 = {{static_cast<float>(i * 3.14), {true, (i % 2 == 0), false}},
                        {static_cast<float>(i / 10), {(i % 2 == 1), true}}};
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open("map_ntuple", fileGuard.GetPath());
   EXPECT_EQ(2, ntuple->GetNEntries());

   auto viewMap = ntuple->GetView<std::unordered_map<std::string, float>>("myMap");
   auto viewMap2 = ntuple->GetView<std::unordered_map<int, CustomStruct>>("myMap2");
   auto viewMap3 = ntuple->GetView<std::unordered_map<char, std::string>>("myMap3");
   auto viewMap4 = ntuple->GetView<std::unordered_map<float, std::vector<bool>>>("myMap4");
   for (auto i : ntuple->GetEntryRange()) {
      std::unordered_map<std::string, float> map1{{"foo", static_cast<float>(i + 0.1)},
                                                  {"bar", static_cast<float>(i * 0.2)},
                                                  {"baz", static_cast<float>(i * 0.3)}};
      EXPECT_EQ(map1, viewMap(i));

      std::unordered_map<int, CustomStruct> map2{{i, CustomStruct{6.f, {7.f, 8.f}, {{9.f}, {10.f}}, "foo"}},
                                                 {i + 1, CustomStruct{3.f, {4.f, 5.f}, {{1.f}, {2.f}}, "baz"}}};
      EXPECT_EQ(map2, viewMap2(i));

      std::unordered_map<char, std::string> map3{{static_cast<char>(i), "Hello"}, {static_cast<char>(i), "world!"}};
      EXPECT_EQ(map3, viewMap3(i));

      std::unordered_map<float, std::vector<bool>> map4{{static_cast<float>(i * 3.14), {true, (i % 2 == 0), false}},
                                                        {static_cast<float>(i / 10), {(i % 2 == 1), true}}};
      EXPECT_EQ(map4, viewMap4(i));
   }

   ntuple->LoadEntry(0);
   auto myMap2 = ntuple->GetModel().GetDefaultEntry().GetPtr<std::unordered_map<int, CustomStruct>>("myMap2");
   auto customStructMap =
      std::unordered_map<int, CustomStruct>({{0, CustomStruct{6.f, {7.f, 8.f}, {{9.f}, {10.f}}, "foo"}},
                                             {1, CustomStruct{3.f, {4.f, 5.f}, {{1.f}, {2.f}}, "baz"}}});
   EXPECT_EQ(customStructMap, *myMap2);
}

TEST(RNTuple, Int64)
{
   auto field = RFieldBase::Create("test", "std::int64_t").Unwrap();
   auto otherField = RFieldBase::Create("test", "std::uint64_t").Unwrap();

   FileRaii fileGuard("test_ntuple_int64.root");

   auto model = RNTupleModel::Create();

   auto f1 = std::make_unique<RField<std::int64_t>>("i1");
   f1->SetColumnRepresentative({ROOT::Experimental::EColumnType::kInt64});
   model->AddField(std::move(f1));

   auto f2 = std::make_unique<RField<std::int64_t>>("i2");
   f2->SetColumnRepresentative({ROOT::Experimental::EColumnType::kSplitInt64});
   model->AddField(std::move(f2));

   auto f3 = std::make_unique<RField<std::uint64_t>>("i3");
   f3->SetColumnRepresentative({ROOT::Experimental::EColumnType::kUInt64});
   model->AddField(std::move(f3));

   auto f4 = std::make_unique<RField<std::uint64_t>>("i4");
   f4->SetColumnRepresentative({ROOT::Experimental::EColumnType::kSplitUInt64});
   model->AddField(std::move(f4));

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      auto e = writer->CreateEntry();
      *e->GetPtr<std::int64_t>("i1") = std::numeric_limits<std::int64_t>::max() - 137;
      *e->GetPtr<std::int64_t>("i2") = std::numeric_limits<std::int64_t>::max() - 138;
      *e->GetPtr<std::uint64_t>("i3") = std::numeric_limits<std::uint64_t>::max() - 42;
      *e->GetPtr<std::uint64_t>("i4") = std::numeric_limits<std::uint64_t>::max() - 43;
      writer->Fill(*e);
   }

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   EXPECT_EQ(ROOT::Experimental::EColumnType::kInt64,
             (*desc.GetColumnIterable(desc.FindFieldId("i1")).begin()).GetModel().GetType());
   EXPECT_EQ(ROOT::Experimental::EColumnType::kSplitInt64,
             (*desc.GetColumnIterable(desc.FindFieldId("i2")).begin()).GetModel().GetType());
   EXPECT_EQ(ROOT::Experimental::EColumnType::kUInt64,
             (*desc.GetColumnIterable(desc.FindFieldId("i3")).begin()).GetModel().GetType());
   EXPECT_EQ(ROOT::Experimental::EColumnType::kSplitUInt64,
             (*desc.GetColumnIterable(desc.FindFieldId("i4")).begin()).GetModel().GetType());
   reader->LoadEntry(0);
   EXPECT_EQ(std::numeric_limits<std::int64_t>::max() - 137,
             *reader->GetModel().GetDefaultEntry().GetPtr<std::int64_t>("i1"));
   EXPECT_EQ(std::numeric_limits<std::int64_t>::max() - 138,
             *reader->GetModel().GetDefaultEntry().GetPtr<std::int64_t>("i2"));
   EXPECT_EQ(std::numeric_limits<std::uint64_t>::max() - 42,
             *reader->GetModel().GetDefaultEntry().GetPtr<std::uint64_t>("i3"));
   EXPECT_EQ(std::numeric_limits<std::uint64_t>::max() - 43,
             *reader->GetModel().GetDefaultEntry().GetPtr<std::uint64_t>("i4"));
}

TEST(RNTuple, Int32)
{
   auto field = RFieldBase::Create("test", "std::int32_t").Unwrap();
   auto otherField = RFieldBase::Create("test", "std::uint32_t").Unwrap();

   FileRaii fileGuard("test_ntuple_int32.root");

   auto model = RNTupleModel::Create();

   auto f1 = std::make_unique<RField<std::int32_t>>("i1");
   f1->SetColumnRepresentative({ROOT::Experimental::EColumnType::kInt32});
   model->AddField(std::move(f1));

   auto f2 = std::make_unique<RField<std::int32_t>>("i2");
   f2->SetColumnRepresentative({ROOT::Experimental::EColumnType::kSplitInt32});
   model->AddField(std::move(f2));

   auto f3 = std::make_unique<RField<std::uint32_t>>("i3");
   f3->SetColumnRepresentative({ROOT::Experimental::EColumnType::kUInt32});
   model->AddField(std::move(f3));

   auto f4 = std::make_unique<RField<std::uint32_t>>("i4");
   f4->SetColumnRepresentative({ROOT::Experimental::EColumnType::kSplitUInt32});
   model->AddField(std::move(f4));

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      auto e = writer->CreateEntry();
      *e->GetPtr<std::int32_t>("i1") = std::numeric_limits<std::int32_t>::max() - 137;
      *e->GetPtr<std::int32_t>("i2") = std::numeric_limits<std::int32_t>::max() - 138;
      *e->GetPtr<std::uint32_t>("i3") = std::numeric_limits<std::uint32_t>::max() - 42;
      *e->GetPtr<std::uint32_t>("i4") = std::numeric_limits<std::uint32_t>::max() - 43;
      writer->Fill(*e);
   }

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   EXPECT_EQ(ROOT::Experimental::EColumnType::kInt32,
             (*desc.GetColumnIterable(desc.FindFieldId("i1")).begin()).GetModel().GetType());
   EXPECT_EQ(ROOT::Experimental::EColumnType::kSplitInt32,
             (*desc.GetColumnIterable(desc.FindFieldId("i2")).begin()).GetModel().GetType());
   EXPECT_EQ(ROOT::Experimental::EColumnType::kUInt32,
             (*desc.GetColumnIterable(desc.FindFieldId("i3")).begin()).GetModel().GetType());
   EXPECT_EQ(ROOT::Experimental::EColumnType::kSplitUInt32,
             (*desc.GetColumnIterable(desc.FindFieldId("i4")).begin()).GetModel().GetType());
   reader->LoadEntry(0);
   EXPECT_EQ(std::numeric_limits<std::int32_t>::max() - 137,
             *reader->GetModel().GetDefaultEntry().GetPtr<std::int32_t>("i1"));
   EXPECT_EQ(std::numeric_limits<std::int32_t>::max() - 138,
             *reader->GetModel().GetDefaultEntry().GetPtr<std::int32_t>("i2"));
   EXPECT_EQ(std::numeric_limits<std::uint32_t>::max() - 42,
             *reader->GetModel().GetDefaultEntry().GetPtr<std::uint32_t>("i3"));
   EXPECT_EQ(std::numeric_limits<std::uint32_t>::max() - 43,
             *reader->GetModel().GetDefaultEntry().GetPtr<std::uint32_t>("i4"));
}

TEST(RNTuple, Int16)
{
   auto field = RFieldBase::Create("test", "std::int16_t").Unwrap();
   auto otherField = RFieldBase::Create("test", "std::uint16_t").Unwrap();
   ASSERT_EQ("std::int16_t", RFieldBase::Create("myShort", "Short_t").Unwrap()->GetTypeName());
   ASSERT_EQ("std::uint16_t", RFieldBase::Create("myUShort", "UShort_t").Unwrap()->GetTypeName());

   FileRaii fileGuard("test_ntuple_int16.root");

   auto model = RNTupleModel::Create();

   auto f1 = std::make_unique<RField<std::int16_t>>("i1");
   f1->SetColumnRepresentative({ROOT::Experimental::EColumnType::kInt16});
   model->AddField(std::move(f1));

   auto f2 = std::make_unique<RField<std::int16_t>>("i2");
   f2->SetColumnRepresentative({ROOT::Experimental::EColumnType::kSplitInt16});
   model->AddField(std::move(f2));

   auto f3 = std::make_unique<RField<std::uint16_t>>("i3");
   f3->SetColumnRepresentative({ROOT::Experimental::EColumnType::kUInt16});
   model->AddField(std::move(f3));

   auto f4 = std::make_unique<RField<std::uint16_t>>("i4");
   f4->SetColumnRepresentative({ROOT::Experimental::EColumnType::kSplitUInt16});
   model->AddField(std::move(f4));

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      auto e = writer->CreateEntry();
      *e->GetPtr<std::int16_t>("i1") = std::numeric_limits<std::int16_t>::max() - 137;
      *e->GetPtr<std::int16_t>("i2") = std::numeric_limits<std::int16_t>::max() - 138;
      *e->GetPtr<std::uint16_t>("i3") = std::numeric_limits<std::uint16_t>::max() - 42;
      *e->GetPtr<std::uint16_t>("i4") = std::numeric_limits<std::uint16_t>::max() - 43;
      writer->Fill(*e);
   }

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   EXPECT_EQ(ROOT::Experimental::EColumnType::kInt16,
             (*desc.GetColumnIterable(desc.FindFieldId("i1")).begin()).GetModel().GetType());
   EXPECT_EQ(ROOT::Experimental::EColumnType::kSplitInt16,
             (*desc.GetColumnIterable(desc.FindFieldId("i2")).begin()).GetModel().GetType());
   EXPECT_EQ(ROOT::Experimental::EColumnType::kUInt16,
             (*desc.GetColumnIterable(desc.FindFieldId("i3")).begin()).GetModel().GetType());
   EXPECT_EQ(ROOT::Experimental::EColumnType::kSplitUInt16,
             (*desc.GetColumnIterable(desc.FindFieldId("i4")).begin()).GetModel().GetType());
   reader->LoadEntry(0);
   EXPECT_EQ(std::numeric_limits<std::int16_t>::max() - 137,
             *reader->GetModel().GetDefaultEntry().GetPtr<std::int16_t>("i1"));
   EXPECT_EQ(std::numeric_limits<std::int16_t>::max() - 138,
             *reader->GetModel().GetDefaultEntry().GetPtr<std::int16_t>("i2"));
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max() - 42,
             *reader->GetModel().GetDefaultEntry().GetPtr<std::uint16_t>("i3"));
   EXPECT_EQ(std::numeric_limits<std::uint16_t>::max() - 43,
             *reader->GetModel().GetDefaultEntry().GetPtr<std::uint16_t>("i4"));
}

TEST(RNTuple, Char)
{
   auto charField = RField<char>("myChar");
   auto otherField = RFieldBase::Create("test", "char").Unwrap();
   ASSERT_EQ("char", otherField->GetTypeName());

   auto charTField = RField<Char_t>("myChar");
   ASSERT_EQ("char", charTField.GetTypeName());
}

TEST(RNTuple, Byte)
{
   FileRaii fileGuard("ntuple_test_byte.root");

   auto byteField = RField<std::byte>("myByte");
   auto otherField = RFieldBase::Create("test", "std::byte").Unwrap();
   ASSERT_EQ("std::byte", otherField->GetTypeName());

   {
      auto model = RNTupleModel::Create();
      auto f = model->MakeField<std::byte>("b", std::byte{137});
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(1u, reader->GetNEntries());
   reader->LoadEntry(0);
   EXPECT_EQ(std::byte{137}, *reader->GetModel().GetDefaultEntry().GetPtr<std::byte>("b"));
}

TEST(RNTuple, Int8_t)
{
   auto field = RField<std::int8_t>("myInt8");
   auto otherField = RFieldBase::Create("test", "std::int8_t").Unwrap();
}

TEST(RNTuple, Double)
{
   FileRaii fileGuard("double.root");

   auto model = RNTupleModel::Create();

   auto f1 = std::make_unique<RField<double>>("d1");
   f1->SetColumnRepresentative({ROOT::Experimental::EColumnType::kReal64});
   model->AddField(std::move(f1));

   auto f2 = std::make_unique<RField<double>>("d2");
   f2->SetColumnRepresentative({ROOT::Experimental::EColumnType::kSplitReal64});
   model->AddField(std::move(f2));

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      auto e = writer->CreateEntry();
      *e->GetPtr<double>("d1") = 1.0;
      *e->GetPtr<double>("d2") = 2.0;
      writer->Fill(*e);
   }

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   EXPECT_EQ(ROOT::Experimental::EColumnType::kReal64,
             (*desc.GetColumnIterable(desc.FindFieldId("d1")).begin()).GetModel().GetType());
   EXPECT_EQ(ROOT::Experimental::EColumnType::kSplitReal64,
             (*desc.GetColumnIterable(desc.FindFieldId("d2")).begin()).GetModel().GetType());
   reader->LoadEntry(0);
   EXPECT_DOUBLE_EQ(1.0, *reader->GetModel().GetDefaultEntry().GetPtr<double>("d1"));
   EXPECT_DOUBLE_EQ(2.0, *reader->GetModel().GetDefaultEntry().GetPtr<double>("d2"));
}

TEST(RNTuple, Float)
{
   FileRaii fileGuard("test_ntuple_float.root");

   auto model = RNTupleModel::Create();

   auto f1 = std::make_unique<RField<float>>("f1");
   f1->SetColumnRepresentative({ROOT::Experimental::EColumnType::kReal32});
   model->AddField(std::move(f1));

   auto f2 = std::make_unique<RField<float>>("f2");
   f2->SetColumnRepresentative({ROOT::Experimental::EColumnType::kSplitReal32});
   model->AddField(std::move(f2));

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      auto e = writer->CreateEntry();
      *e->GetPtr<float>("f1") = 1.0;
      *e->GetPtr<float>("f2") = 2.0;
      writer->Fill(*e);
   }

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   const auto &desc = reader->GetDescriptor();
   EXPECT_EQ(ROOT::Experimental::EColumnType::kReal32,
             (*desc.GetColumnIterable(desc.FindFieldId("f1")).begin()).GetModel().GetType());
   EXPECT_EQ(ROOT::Experimental::EColumnType::kSplitReal32,
             (*desc.GetColumnIterable(desc.FindFieldId("f2")).begin()).GetModel().GetType());
   reader->LoadEntry(0);
   EXPECT_FLOAT_EQ(1.0, *reader->GetModel().GetDefaultEntry().GetPtr<float>("f1"));
   EXPECT_FLOAT_EQ(2.0, *reader->GetModel().GetDefaultEntry().GetPtr<float>("f2"));
}

TEST(RNTuple, StdAtomic)
{
   auto field = RField<std::atomic<int64_t>>("atomicField");
   EXPECT_STREQ("std::atomic<std::int64_t>", field.GetTypeName().c_str());
   auto otherField = RFieldBase::Create("test", "std::atomic<int64_t>").Unwrap();
   EXPECT_EQ(field.GetTypeName(), otherField->GetTypeName());
   EXPECT_EQ((sizeof(std::atomic<int64_t>)), field.GetValueSize());
   EXPECT_EQ((alignof(std::atomic<int64_t>)), field.GetAlignment());

   FileRaii fileGuard("test_ntuple_rfield_stdatomic.root");
   {
      auto model = RNTupleModel::Create();
      auto f1 = model->MakeField<std::atomic<bool>>("f1");
      model->AddField(RFieldBase::Create("f2", "std::atomic<float>").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      auto f2 = writer->GetModel().GetDefaultEntry().GetPtr<std::atomic<float>>("f2");
      for (int i = 0; i < 2; i++) {
         *f1 = i % 2 == 0;
         *f2 = static_cast<float>(i);
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(2, reader->GetNEntries());

   auto viewF1 = reader->GetView<std::atomic<bool>>("f1");
   auto viewF2 = reader->GetView<std::atomic<float>>("f2");
   for (auto i : reader->GetEntryRange()) {
      EXPECT_EQ(i % 2 == 0, viewF1(i));
      EXPECT_FLOAT_EQ(static_cast<float>(i), viewF2(i));
   }
}

TEST(RNTuple, Bitset)
{
   FileRaii fileGuard("test_ntuple_bitset.root");

   auto model = RNTupleModel::Create();

   auto f1 = model->MakeField<std::bitset<66>>("f1");
   EXPECT_EQ(std::string("std::bitset<66>"), model->GetField("f1").GetTypeName());
   EXPECT_EQ(sizeof(std::bitset<66>), model->GetField("f1").GetValueSize());
   auto f2 = model->MakeField<std::bitset<8>>("f2", "10101010");

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      writer->Fill();
      f1->set(0);
      f1->set(3);
      f1->set(33);
      f1->set(65);
      f2->flip();
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(std::string("std::bitset<66>"), reader->GetModel().GetField("f1").GetTypeName());
   auto bs1 = reader->GetModel().GetDefaultEntry().GetPtr<std::bitset<66>>("f1");
   auto bs2 = reader->GetModel().GetDefaultEntry().GetPtr<std::bitset<8>>("f2");
   reader->LoadEntry(0);
   EXPECT_EQ("000000000000000000000000000000000000000000000000000000000000000000", bs1->to_string());
   EXPECT_EQ("10101010", bs2->to_string());
   reader->LoadEntry(1);
   EXPECT_EQ("100000000000000000000000000000001000000000000000000000000000001001", bs1->to_string());
   EXPECT_EQ("01010101", bs2->to_string());
}

struct RTagNullableFieldDefault {};
struct RTagNullableFieldSparse {};
struct RTagNullableFieldDense {};
using UniquePtrTags = ::testing::Types<RTagNullableFieldDefault, RTagNullableFieldSparse, RTagNullableFieldDense>;

template <typename TagT>
class UniquePtr : public ::testing::Test {
public:
   using Tag_t = TagT;
};

TYPED_TEST_SUITE(UniquePtr, UniquePtrTags);

template <typename TypeT, typename TagT>
static void AddUniquePtrField(RNTupleModel &model, const std::string &fieldName)
{
   auto fld = std::make_unique<RField<std::unique_ptr<TypeT>>>(fieldName);
   if constexpr (std::is_same_v<TagT, RTagNullableFieldSparse>) {
      fld->SetSparse();
   }
   if constexpr (std::is_same_v<TagT, RTagNullableFieldDense>) {
      fld->SetDense();
   }
   model.AddField(std::move(fld));
}

TYPED_TEST(UniquePtr, Basics)
{
   using RUniquePtrField = ROOT::Experimental::RUniquePtrField;

   FileRaii fileGuard("test_ntuple_unique_ptr.root");

   {
      auto model = RNTupleModel::Create();

      AddUniquePtrField<bool, typename TestFixture::Tag_t>(*model, "PBool");
      AddUniquePtrField<CustomStruct, typename TestFixture::Tag_t>(*model, "PCustomStruct");
      AddUniquePtrField<IOConstructor, typename TestFixture::Tag_t>(*model, "PIOConstructor");
      AddUniquePtrField<std::unique_ptr<std::string>, typename TestFixture::Tag_t>(*model, "PPString");
      AddUniquePtrField<std::array<char, 2>, typename TestFixture::Tag_t>(*model, "PArray");

      EXPECT_EQ("std::unique_ptr<bool>", model->GetField("PBool").GetTypeName());
      EXPECT_EQ(std::string("std::unique_ptr<CustomStruct>"), model->GetField("PCustomStruct").GetTypeName());
      EXPECT_EQ(std::string("std::unique_ptr<IOConstructor>"), model->GetField("PIOConstructor").GetTypeName());
      EXPECT_EQ(std::string("std::unique_ptr<std::unique_ptr<std::string>>"),
                model->GetField("PPString").GetTypeName());
      EXPECT_EQ(std::string("std::unique_ptr<std::array<char,2>>"), model->GetField("PArray").GetTypeName());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());

      if constexpr (std::is_same_v<typename TestFixture::Tag_t, RTagNullableFieldDefault>) {
         EXPECT_TRUE(dynamic_cast<const RUniquePtrField &>(writer->GetModel().GetField("PBool")).IsDense());
         EXPECT_TRUE(dynamic_cast<const RUniquePtrField &>(writer->GetModel().GetField("PCustomStruct")).IsSparse());
         EXPECT_TRUE(dynamic_cast<const RUniquePtrField &>(writer->GetModel().GetField("PArray")).IsDense());
      }
      if constexpr (std::is_same_v<typename TestFixture::Tag_t, RTagNullableFieldSparse>) {
         EXPECT_TRUE(dynamic_cast<const RUniquePtrField &>(writer->GetModel().GetField("PBool")).IsSparse());
         EXPECT_TRUE(dynamic_cast<const RUniquePtrField &>(writer->GetModel().GetField("PCustomStruct")).IsSparse());
         EXPECT_TRUE(dynamic_cast<const RUniquePtrField &>(writer->GetModel().GetField("PIOConstructor")).IsSparse());
         EXPECT_TRUE(dynamic_cast<const RUniquePtrField &>(writer->GetModel().GetField("PPString")).IsSparse());
         EXPECT_TRUE(dynamic_cast<const RUniquePtrField &>(writer->GetModel().GetField("PArray")).IsSparse());
      }
      if constexpr (std::is_same_v<typename TestFixture::Tag_t, RTagNullableFieldDense>) {
         EXPECT_TRUE(dynamic_cast<const RUniquePtrField &>(writer->GetModel().GetField("PBool")).IsDense());
         EXPECT_TRUE(dynamic_cast<const RUniquePtrField &>(writer->GetModel().GetField("PCustomStruct")).IsDense());
         EXPECT_TRUE(dynamic_cast<const RUniquePtrField &>(writer->GetModel().GetField("PIOConstructor")).IsDense());
         EXPECT_TRUE(dynamic_cast<const RUniquePtrField &>(writer->GetModel().GetField("PPString")).IsDense());
         EXPECT_TRUE(dynamic_cast<const RUniquePtrField &>(writer->GetModel().GetField("PArray")).IsDense());
      }

      auto pBool = writer->GetModel().GetDefaultEntry().GetPtr<std::unique_ptr<bool>>("PBool");
      auto pCustomStruct = writer->GetModel().GetDefaultEntry().GetPtr<std::unique_ptr<CustomStruct>>("PCustomStruct");
      auto pIOConstructor =
         writer->GetModel().GetDefaultEntry().GetPtr<std::unique_ptr<IOConstructor>>("PIOConstructor");
      auto ppString =
         writer->GetModel().GetDefaultEntry().GetPtr<std::unique_ptr<std::unique_ptr<std::string>>>("PPString");
      auto pArray = writer->GetModel().GetDefaultEntry().GetPtr<std::unique_ptr<std::array<char, 2>>>("PArray");

      *pBool = std::make_unique<bool>(true);
      EXPECT_EQ(nullptr, pCustomStruct->get());
      EXPECT_EQ(nullptr, pIOConstructor->get());
      EXPECT_EQ(nullptr, ppString->get());
      EXPECT_EQ(nullptr, pArray->get());
      writer->Fill();
      *pBool = nullptr;
      *pCustomStruct = std::make_unique<CustomStruct>();
      *pIOConstructor = std::make_unique<IOConstructor>(nullptr);
      *ppString = std::make_unique<std::unique_ptr<std::string>>(std::make_unique<std::string>());
      *pArray = std::make_unique<std::array<char, 2>>();
      writer->Fill();
      (*pCustomStruct)->a = 42.0;
      (*pIOConstructor)->a = 13;
      (*(*ppString))->assign("abc");
      (*pArray)->at(1) = 'x';
      writer->Fill();
      *pBool = std::make_unique<bool>(false);
      *pCustomStruct = nullptr;
      *pIOConstructor = nullptr;
      *ppString = nullptr;
      *pArray = nullptr;
      writer->Fill();
      writer->CommitCluster();
      *ppString = std::make_unique<std::unique_ptr<std::string>>(std::make_unique<std::string>("de"));
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   const auto &model = reader->GetModel();
   EXPECT_EQ("std::unique_ptr<bool>", model.GetField("PBool").GetTypeName());
   EXPECT_EQ(std::string("std::unique_ptr<CustomStruct>"), model.GetField("PCustomStruct").GetTypeName());
   EXPECT_EQ(std::string("std::unique_ptr<IOConstructor>"), model.GetField("PIOConstructor").GetTypeName());
   EXPECT_EQ(std::string("std::unique_ptr<std::unique_ptr<std::string>>"), model.GetField("PPString").GetTypeName());
   EXPECT_EQ(std::string("std::unique_ptr<std::array<char,2>>"), model.GetField("PArray").GetTypeName());

   const auto &entry = model.GetDefaultEntry();
   auto pBool = entry.GetPtr<std::unique_ptr<bool>>("PBool");
   auto pCustomStruct = entry.GetPtr<std::unique_ptr<CustomStruct>>("PCustomStruct");
   auto pIOConstructor = entry.GetPtr<std::unique_ptr<IOConstructor>>("PIOConstructor");
   auto ppString = entry.GetPtr<std::unique_ptr<std::unique_ptr<std::string>>>("PPString");
   auto pArray = entry.GetPtr<std::unique_ptr<std::array<char, 2>>>("PArray");

   reader->LoadEntry(0);
   EXPECT_TRUE(*(pBool->get()));
   EXPECT_EQ(nullptr, pCustomStruct->get());
   EXPECT_EQ(nullptr, pIOConstructor->get());
   EXPECT_EQ(nullptr, ppString->get());
   EXPECT_EQ(nullptr, pArray->get());

   reader->LoadEntry(1);
   EXPECT_EQ(nullptr, pBool->get());
   EXPECT_FLOAT_EQ(0.0, pCustomStruct->get()->a);
   EXPECT_EQ(7, pIOConstructor->get()->a);
   EXPECT_TRUE(ppString->get()->get()->empty());
   EXPECT_EQ(0, pArray->get()->at(0));
   EXPECT_EQ(0, pArray->get()->at(1));

   reader->LoadEntry(2);
   EXPECT_EQ(nullptr, pBool->get());
   EXPECT_FLOAT_EQ(42.0, pCustomStruct->get()->a);
   EXPECT_EQ(13, pIOConstructor->get()->a);
   EXPECT_EQ("abc", *(ppString->get()->get()));
   EXPECT_EQ(0, pArray->get()->at(0));
   EXPECT_EQ('x', pArray->get()->at(1));

   reader->LoadEntry(3);
   EXPECT_FALSE(*(pBool->get()));
   EXPECT_EQ(nullptr, pCustomStruct->get());
   EXPECT_EQ(nullptr, pIOConstructor->get());
   EXPECT_EQ(nullptr, ppString->get());
   EXPECT_EQ(nullptr, pArray->get());

   reader->LoadEntry(4);
   EXPECT_FALSE(*(pBool->get()));
   EXPECT_EQ(nullptr, pCustomStruct->get());
   EXPECT_EQ(nullptr, pIOConstructor->get());
   EXPECT_EQ("de", *(ppString->get()->get()));
   EXPECT_EQ(nullptr, pArray->get());
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

   auto fldI1 = RFieldBase::Create("i1", "std::int32_t").Unwrap();
   fldI1->SetColumnRepresentative({EColumnType::kInt32});
   auto fldI2 = RFieldBase::Create("i2", "std::int32_t").Unwrap();
   fldI2->SetColumnRepresentative({EColumnType::kSplitInt32});
   auto fldF = ROOT::Experimental::RFieldBase::Create("F", "float").Unwrap();
   fldF->SetColumnRepresentative({EColumnType::kReal32});
   try {
      fldF->SetColumnRepresentative({EColumnType::kBit});
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("invalid column representative"));
   }

   auto modelA = RNTupleModel::Create();
   modelA->AddField(std::move(fldI1));
   modelA->AddField(std::move(fldI2));
   modelA->AddField(std::move(fldF));
   {
      auto writer = RNTupleWriter::Recreate(std::move(modelA), "ntuple", fileGuard.GetPath());
      *writer->GetModel().GetDefaultEntry().GetPtr<std::int32_t>("i1") = 42;
      *writer->GetModel().GetDefaultEntry().GetPtr<std::int32_t>("i2") = 137;
      writer->Fill();
   }

   try {
      auto model = RNTupleModel::Create();
      auto f = ROOT::Experimental::RFieldBase::Create("i1", "std::int32_t").Unwrap();
      f->SetColumnRepresentative({EColumnType::kInt32});
      model->AddField(std::move(f));
      auto reader = RNTupleReader::Open(std::move(model), "ntuple", fileGuard.GetPath());
      FAIL() << "should not be able fix column representation when model is connected to a page source";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(),
                  testing::HasSubstr("fixed column representative only valid when connecting to a page sink"));
   }

   try {
      auto modelB = RNTupleModel::Create();
      auto fieldCast = modelB->MakeField<float>("i1");
      auto reader = RNTupleReader::Open(std::move(modelB), "ntuple", fileGuard.GetPath());
      FAIL() << "should not be able to cast int to float";
   } catch (const RException& err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("On-disk column types"));
      EXPECT_THAT(err.what(), testing::HasSubstr("cannot be matched"));
   }

   auto modelC = RNTupleModel::Create();
   auto fieldCast1 = modelC->MakeField<std::int64_t>("i1");
   auto fieldCast2 = modelC->MakeField<std::int64_t>("i2");
   auto reader = RNTupleReader::Open(std::move(modelC), "ntuple", fileGuard.GetPath());
   reader->LoadEntry(0);
   EXPECT_EQ(42, *fieldCast1);
   EXPECT_EQ(137, *fieldCast2);
}

TEST(RNTuple, HalfPrecisionFloat)
{
   FileRaii fileGuard("test_ntuple_half_precision_float.root");

   // TODO: Add std::float16 tests once available (from C++23)
   auto f1Fld = RFieldBase::Create("f1", "float").Unwrap();
   dynamic_cast<RField<float> *>(f1Fld.get())->SetHalfPrecision();
   EXPECT_EQ(EColumnType::kReal16, f1Fld->GetColumnRepresentative()[0]);
   EXPECT_EQ("float", f1Fld->GetTypeName());

   auto fVecFld = RFieldBase::Create("fVec", "std::vector<float>").Unwrap();
   dynamic_cast<RField<float> *>(fVecFld->GetSubFields()[0])->SetHalfPrecision();
   EXPECT_EQ(EColumnType::kReal16, fVecFld->GetSubFields()[0]->GetColumnRepresentative()[0]);

   auto model = RNTupleModel::Create();
   model->AddField(std::move(f1Fld));
   model->AddField(std::move(fVecFld));

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      auto f1 = writer->GetModel().GetDefaultEntry().GetPtr<float>("f1");
      auto fVec = writer->GetModel().GetDefaultEntry().GetPtr<std::vector<float>>("fVec");
      *f1 = 0.1f;
      *fVec = {0.1f, 0.2f};
      writer->Fill();
      *f1 = 4245.5f;
      *fVec = {std::numeric_limits<float>::max(), std::numeric_limits<float>::min(),
               std::numeric_limits<float>::lowest(), std::numeric_limits<float>::denorm_min()};
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());

   EXPECT_EQ(4, ROOT::Experimental::Internal::RColumnElementBase::Generate(EColumnType::kReal16)->GetSize());

   const auto &desc = reader->GetDescriptor();
   EXPECT_EQ(EColumnType::kReal16, (*desc.GetColumnIterable(desc.FindFieldId("f1")).begin()).GetModel().GetType());

   auto f1 = reader->GetModel().GetDefaultEntry().GetPtr<float>("f1");
   auto fVec = reader->GetModel().GetDefaultEntry().GetPtr<std::vector<float>>("fVec");
   reader->LoadEntry(0);
   EXPECT_FLOAT_EQ(0.0999755859375f, *f1);
   EXPECT_FLOAT_EQ(0.0999755859375f, (*fVec)[0]);
   EXPECT_FLOAT_EQ(0.199951171875f, (*fVec)[1]);
   reader->LoadEntry(1);
   EXPECT_FLOAT_EQ(4244.f, *f1);
   EXPECT_FLOAT_EQ(INFINITY, (*fVec)[0]);
   EXPECT_FLOAT_EQ(0.0f, (*fVec)[1]);
   EXPECT_FLOAT_EQ(-INFINITY, (*fVec)[2]);
   EXPECT_FLOAT_EQ(0.0f, (*fVec)[3]);
}

TEST(RNTuple, Double32)
{
   FileRaii fileGuard("test_ntuple_double32.root");

   auto fldD1 = RFieldBase::Create("d1", "double").Unwrap();
   fldD1->SetColumnRepresentative({EColumnType::kReal32});
   auto fldD2 = RFieldBase::Create("d2", "Double32_t").Unwrap();
   EXPECT_EQ("Double32_t", fldD2->GetTypeAlias());

   auto model = RNTupleModel::Create();
   model->AddField(std::move(fldD1));
   model->AddField(std::move(fldD2));

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      auto d1 = writer->GetModel().GetDefaultEntry().GetPtr<double>("d1");
      auto d2 = writer->GetModel().GetDefaultEntry().GetPtr<double>("d2");
      *d1 = 0.0;
      *d2 = 0.0;
      writer->Fill();
      *d1 = std::numeric_limits<float>::max();
      *d2 = *d1;
      writer->Fill();
      *d1 = std::numeric_limits<float>::min();
      *d2 = *d1;
      writer->Fill();
      *d1 = std::numeric_limits<float>::lowest();
      *d2 = *d1;
      writer->Fill();
      *d1 = std::numeric_limits<float>::infinity();
      *d2 = *d1;
      writer->Fill();
      *d1 = std::numeric_limits<float>::denorm_min();
      *d2 = *d1;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   EXPECT_EQ(EColumnType::kReal32, reader->GetModel().GetField("d1").GetColumnRepresentative()[0]);
   EXPECT_EQ("", reader->GetModel().GetField("d1").GetTypeAlias());
   EXPECT_EQ(EColumnType::kSplitReal32, reader->GetModel().GetField("d2").GetColumnRepresentative()[0]);
   EXPECT_EQ("Double32_t", reader->GetModel().GetField("d2").GetTypeAlias());
   auto d1 = reader->GetModel().GetDefaultEntry().GetPtr<double>("d1");
   auto d2 = reader->GetModel().GetDefaultEntry().GetPtr<double>("d2");
   reader->LoadEntry(0);
   EXPECT_DOUBLE_EQ(0.0, *d1);
   EXPECT_DOUBLE_EQ(*d1, *d2);
   reader->LoadEntry(1);
   EXPECT_DOUBLE_EQ(std::numeric_limits<float>::max(), *d1);
   EXPECT_DOUBLE_EQ(*d1, *d2);
   reader->LoadEntry(2);
   EXPECT_DOUBLE_EQ(std::numeric_limits<float>::min(), *d1);
   EXPECT_DOUBLE_EQ(*d1, *d2);
   reader->LoadEntry(3);
   EXPECT_DOUBLE_EQ(std::numeric_limits<float>::lowest(), *d1);
   EXPECT_DOUBLE_EQ(*d1, *d2);
   reader->LoadEntry(4);
   EXPECT_DOUBLE_EQ(std::numeric_limits<float>::infinity(), *d1);
   EXPECT_DOUBLE_EQ(*d1, *d2);
   reader->LoadEntry(5);
   EXPECT_DOUBLE_EQ(std::numeric_limits<float>::denorm_min(), *d1);
   EXPECT_DOUBLE_EQ(*d1, *d2);

   auto modelFloat = RNTupleModel::Create();
   auto d2Float = modelFloat->MakeField<float>("d2");
   auto readerFloat = RNTupleReader::Open(std::move(modelFloat), "ntuple", fileGuard.GetPath());
   readerFloat->LoadEntry(0);
   EXPECT_FLOAT_EQ(0.0, *d2Float);
   readerFloat->LoadEntry(1);
   EXPECT_DOUBLE_EQ(std::numeric_limits<float>::max(), *d2Float);
   readerFloat->LoadEntry(2);
   EXPECT_DOUBLE_EQ(std::numeric_limits<float>::min(), *d2Float);
   readerFloat->LoadEntry(3);
   EXPECT_DOUBLE_EQ(std::numeric_limits<float>::lowest(), *d2Float);
   readerFloat->LoadEntry(4);
   EXPECT_DOUBLE_EQ(std::numeric_limits<float>::infinity(), *d2Float);
   readerFloat->LoadEntry(5);
   EXPECT_DOUBLE_EQ(std::numeric_limits<float>::denorm_min(), *d2Float);
}

TEST(RNTuple, Double32Extended)
{
   FileRaii fileGuard("test_ntuple_double32_extended.root");

   auto fldObj = RFieldBase::Create("obj", "LowPrecisionFloats").Unwrap();
   auto model = RNTupleModel::Create();
   model->AddField(std::move(fldObj));

   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntuple", fileGuard.GetPath());
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("ntuple", fileGuard.GetPath());
   auto obj = reader->GetModel().GetDefaultEntry().GetPtr<LowPrecisionFloats>("obj");
   EXPECT_EQ("Double32_t", reader->GetModel().GetField("obj").GetSubFields()[1]->GetTypeAlias());
   EXPECT_EQ("Double32_t", reader->GetModel().GetField("obj").GetSubFields()[2]->GetSubFields()[0]->GetTypeAlias());
   EXPECT_DOUBLE_EQ(0.0, obj->a);
   EXPECT_DOUBLE_EQ(1.0, obj->b);
   EXPECT_DOUBLE_EQ(2.0, obj->c[0]);
   EXPECT_DOUBLE_EQ(3.0, obj->c[1]);
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
         EXPECT_THAT(err.what(), testing::HasSubstr("No on-disk column information for field `klass.:_0.a`"));
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
      auto idEmptyStruct = ntuple->GetDescriptor().FindFieldId("klass.:_0");
      EXPECT_NE(idEmptyStruct, ROOT::Experimental::kInvalidDescriptorId);
      auto viewKlass = ntuple->GetView<TestEBO>("klass");
      EXPECT_EQ(42, viewKlass(0).u64);
   }
}

TEST(RNTuple, IOConstructor)
{
   FileRaii fileGuard("test_ntuple_ioconstructor.ntuple");

   auto model = RNTupleModel::Create();
   auto fldObj = RFieldBase::Create("obj", "IOConstructor").Unwrap();
   model->AddField(std::move(fldObj));
   {
      auto writer = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
      writer->Fill();
   }

   auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
   EXPECT_EQ(1U, ntuple->GetNEntries());
   auto obj = ntuple->GetModel().GetDefaultEntry().GetPtr<IOConstructor>("obj");
   EXPECT_EQ(7, obj->a);
}

TEST(RNTuple, TClassTemplateBased)
{
   FileRaii fileGuard("test_ntuple_tclass_templatebased.ntuple");
   {
      auto model = RNTupleModel::Create();
      auto fieldObject = model->MakeField<EdmWrapper<CustomStruct>>("klass");
      auto writer = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
      writer->Fill();
      fieldObject->fMember.a = 42.0;
      fieldObject->fMember.v1.push_back(1.0);
      fieldObject->fMember.s = "x";
      writer->Fill();
      fieldObject->fIsPresent = false;
      writer->Fill();
   }

   auto reader = RNTupleReader::Open("f", fileGuard.GetPath());

   const auto &fieldObject = reader->GetModel().GetField("klass");
   EXPECT_EQ("EdmWrapper<CustomStruct>", fieldObject.GetTypeName());
   auto object = reader->GetModel().GetDefaultEntry().GetPtr<EdmWrapper<CustomStruct>>("klass");
   reader->LoadEntry(0);
   EXPECT_TRUE(object->fIsPresent);
   reader->LoadEntry(1);
   EXPECT_TRUE(object->fIsPresent);
   EXPECT_FLOAT_EQ(42.0, object->fMember.a);
   EXPECT_EQ(1u, object->fMember.v1.size());
   EXPECT_FLOAT_EQ(1.0, object->fMember.v1[0]);
   EXPECT_EQ("x", object->fMember.s);
   reader->LoadEntry(2);
   EXPECT_FALSE(object->fIsPresent);
}

TEST(RNTuple, TClassStlDerived)
{
   FileRaii fileGuard("test_ntuple_tclass_stlderived.ntuple");
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
   // Exposing as a non-vector forces iteration over collection elements in `ReadGlobalImpl()`
   SimpleCollectionProxy<StructUsingCollectionProxy<float>, ROOT::kSTLdeque> proxyF;
   SimpleCollectionProxy<StructUsingCollectionProxy<CustomStruct>> proxyS;
   SimpleCollectionProxy<StructUsingCollectionProxy<StructUsingCollectionProxy<float>>> proxyNested;

   // `RProxiedCollectionField` instantiated but no collection proxy set (yet)
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
      auto fieldC = ntuple->GetModel().GetDefaultEntry().GetPtr<StructUsingCollectionProxy<char>>("C");
      auto fieldF = ntuple->GetModel().GetDefaultEntry().GetPtr<StructUsingCollectionProxy<float>>("F");
      auto fieldS = ntuple->GetModel().GetDefaultEntry().GetPtr<StructUsingCollectionProxy<CustomStruct>>("S");
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

TEST(RNTuple, TClassReadRules)
{
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kWarning, "[ROOT.NTuple]", "ignoring I/O customization rule with non-transient member: a", false);
   diags.optionalDiag(kWarning, "ROOT::Experimental::Detail::RPageSinkFile::RPageSinkFile",
                      "The RNTuple file format will change.", false);
   diags.optionalDiag(kWarning, "[ROOT.NTuple]", "Pre-release format version: RC 2", false);

   FileRaii fileGuard("test_ntuple_tclassrules.ntuple");
   char c[4] = {'R', 'O', 'O', 'T'};
   {
      auto model = RNTupleModel::Create();
      auto fieldKlass = model->MakeField<StructWithIORules>("klass");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
      for (int i = 0; i < 20; i++) {
         *fieldKlass = StructWithIORules{/*a=*/static_cast<float>(i), /*chars=*/c};
         ntuple->Fill();
      }
   }

   auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
   EXPECT_EQ(20U, ntuple->GetNEntries());
   auto viewKlass = ntuple->GetView<StructWithIORules>("klass");
   for (auto i : ntuple->GetEntryRange()) {
      float fi = static_cast<float>(i);
      EXPECT_EQ(fi, viewKlass(i).a);
      EXPECT_TRUE(0 == memcmp(c, viewKlass(i).s.chars, sizeof(c)));

      // The following values are set from a read rule; see CustomStructLinkDef.h
      EXPECT_EQ(fi + 1.0f, viewKlass(i).b);
      EXPECT_EQ(viewKlass(i).a + viewKlass(i).b, viewKlass(i).c);
      EXPECT_EQ("ROOT", viewKlass(i).s.str);
   }
}

TEST(RNTuple, RColumnRepresentations)
{
   using RColumnRepresentations = ROOT::Experimental::RFieldBase::RColumnRepresentations;
   RColumnRepresentations colReps1;
   EXPECT_EQ(RFieldBase::ColumnRepresentation_t(), colReps1.GetSerializationDefault());
   EXPECT_EQ(RColumnRepresentations::TypesList_t{RFieldBase::ColumnRepresentation_t()},
             colReps1.GetDeserializationTypes());

   RColumnRepresentations colReps2({{EColumnType::kReal64}, {EColumnType::kSplitReal64}},
                                   {{EColumnType::kReal32}, {EColumnType::kReal16}});
   EXPECT_EQ(RFieldBase::ColumnRepresentation_t({EColumnType::kReal64}), colReps2.GetSerializationDefault());
   EXPECT_EQ(RColumnRepresentations::TypesList_t(
                {{EColumnType::kReal64}, {EColumnType::kSplitReal64}, {EColumnType::kReal32}, {EColumnType::kReal16}}),
             colReps2.GetDeserializationTypes());
}
