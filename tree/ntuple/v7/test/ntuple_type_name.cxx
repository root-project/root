#include "ntuple_test.hxx"

TEST(RNTuple, TypeNameBasics)
{
   EXPECT_STREQ("float", ROOT::Experimental::RField<float>::TypeName().c_str());
   EXPECT_STREQ("std::vector<std::string>", ROOT::Experimental::RField<std::vector<std::string>>::TypeName().c_str());
   EXPECT_STREQ("CustomStruct", ROOT::Experimental::RField<CustomStruct>::TypeName().c_str());
   EXPECT_STREQ("DerivedB", ROOT::Experimental::RField<DerivedB>::TypeName().c_str());

   auto field = RField<DerivedB>("derived");
   EXPECT_EQ(sizeof(DerivedB), field.GetValueSize());

   EXPECT_STREQ("std::pair<std::pair<float,CustomStruct>,std::int32_t>",
                (ROOT::Experimental::RField<std::pair<std::pair<float, CustomStruct>, int>>::TypeName().c_str()));
   EXPECT_STREQ(
      "std::tuple<std::tuple<char,CustomStruct,char>,std::int32_t>",
      (ROOT::Experimental::RField<std::tuple<std::tuple<char, CustomStruct, char>, int>>::TypeName().c_str()));
}

TEST(RNTuple, TypeNameNormalization)
{
   EXPECT_EQ("CustomStruct", RFieldBase::Create("f", "class CustomStruct").Unwrap()->GetTypeName());
   EXPECT_EQ("", RFieldBase::Create("f", "class CustomStruct").Unwrap()->GetTypeAlias());

   EXPECT_EQ("CustomStruct", RFieldBase::Create("f", "struct CustomStruct").Unwrap()->GetTypeName());
   EXPECT_EQ("", RFieldBase::Create("f", "struct CustomStruct").Unwrap()->GetTypeAlias());

   EXPECT_EQ("CustomEnum", RFieldBase::Create("f", "enum CustomEnum").Unwrap()->GetTypeName());
   EXPECT_EQ("", RFieldBase::Create("f", "enum CustomEnum").Unwrap()->GetTypeAlias());

   EXPECT_EQ("std::int32_t", RFieldBase::Create("f", "signed").Unwrap()->GetTypeName());
   EXPECT_EQ("", RFieldBase::Create("f", "signed").Unwrap()->GetTypeAlias());

   EXPECT_EQ("std::map<std::int32_t,std::int32_t>", RFieldBase::Create("f", "map<int, int>").Unwrap()->GetTypeName());
   EXPECT_EQ("", RFieldBase::Create("f", "map<int, int>").Unwrap()->GetTypeAlias());

   EXPECT_EQ("std::uint32_t", RFieldBase::Create("f", "SG::sgkey_t").Unwrap()->GetTypeName());
   EXPECT_EQ("SG::sgkey_t", RFieldBase::Create("f", "SG::sgkey_t").Unwrap()->GetTypeAlias());

   const std::string innerCV = "class InnerCV<const int, const volatile int, volatile const int, volatile int>";
   const std::string normInnerCV =
      "InnerCV<const std::int32_t,const volatile std::int32_t,const volatile std::int32_t,volatile std::int32_t>";
   EXPECT_EQ(normInnerCV, RFieldBase::Create("f", innerCV).Unwrap()->GetTypeName());
   EXPECT_EQ("", RFieldBase::Create("f", innerCV).Unwrap()->GetTypeAlias());

   const std::string example = "const pair<size_t, array<class CustomStruct, 6>>";
   std::string normExample;
   if (sizeof(std::size_t) == 4) {
      normExample = "std::pair<std::uint32_t,std::array<CustomStruct,6>>";
   } else {
      normExample = "std::pair<std::uint64_t,std::array<CustomStruct,6>>";
   }
   EXPECT_EQ(normExample, RFieldBase::Create("f", example).Unwrap()->GetTypeName());
   EXPECT_EQ("std::pair<size_t,std::array<CustomStruct,6>>", RFieldBase::Create("f", example).Unwrap()->GetTypeAlias());

   EXPECT_EQ("std::vector<CustomStruct>",
             RFieldBase::Create("f", "::std::vector<::CustomStruct>").Unwrap()->GetTypeName());
   EXPECT_EQ("", RFieldBase::Create("f", "::std::vector<::CustomStruct>").Unwrap()->GetTypeAlias());
}

TEST(RNTuple, TClassDefaultTemplateParameter)
{
   FileRaii fileGuard("test_ntuple_default_template_parameter.root");

   {
      auto model = RNTupleModel::Create();
      model->MakeField<DataVector<int>>("f1"); // default second template parameter is double
      model->MakeField<DataVector<int, float>>("f2");
      model->AddField(RFieldBase::Create("f3", "DataVector<int>").Unwrap());
      model->AddField(RFieldBase::Create("f4", "struct DataVector<bool,vector<unsigned>>").Unwrap());
      model->AddField(RFieldBase::Create("f5", "DataVector<Double32_t>").Unwrap());
      model->AddField(RFieldBase::Create("f6", "DataVector<int, double>").Unwrap());
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(0u, reader->GetNEntries());

   const auto &desc = reader->GetDescriptor();
   EXPECT_EQ("DataVector<std::int32_t,double>", desc.GetFieldDescriptor(desc.FindFieldId("f1")).GetTypeName());
   EXPECT_EQ("", desc.GetFieldDescriptor(desc.FindFieldId("f1")).GetTypeAlias());

   EXPECT_EQ("DataVector<std::int32_t,float>", desc.GetFieldDescriptor(desc.FindFieldId("f2")).GetTypeName());
   EXPECT_EQ("", desc.GetFieldDescriptor(desc.FindFieldId("f2")).GetTypeAlias());

   EXPECT_EQ("DataVector<std::int32_t,double>", desc.GetFieldDescriptor(desc.FindFieldId("f3")).GetTypeName());
   EXPECT_EQ("", desc.GetFieldDescriptor(desc.FindFieldId("f3")).GetTypeAlias());

   EXPECT_EQ("DataVector<bool,std::vector<std::uint32_t>>",
             desc.GetFieldDescriptor(desc.FindFieldId("f4")).GetTypeName());
   EXPECT_EQ("", desc.GetFieldDescriptor(desc.FindFieldId("f4")).GetTypeAlias());

   EXPECT_EQ("DataVector<double,double>", desc.GetFieldDescriptor(desc.FindFieldId("f5")).GetTypeName());
   EXPECT_EQ("DataVector<Double32_t,double>", desc.GetFieldDescriptor(desc.FindFieldId("f5")).GetTypeAlias());

   EXPECT_EQ("DataVector<std::int32_t,double>", desc.GetFieldDescriptor(desc.FindFieldId("f6")).GetTypeName());
   EXPECT_EQ("", desc.GetFieldDescriptor(desc.FindFieldId("f6")).GetTypeAlias());

   auto v1 = reader->GetView<DataVector<int>>("f1");
   auto v3 = reader->GetView<DataVector<int>>("f3");
   EXPECT_THROW(reader->GetView<DataVector<int>>("f2"), ROOT::RException);
}

TEST(RNTuple, TemplateArgIntegerNormalization)
{
   EXPECT_EQ("IntegerTemplates<0,0>", RFieldBase::Create("f", "IntegerTemplates<0ll,0ull>").Unwrap()->GetTypeName());
   EXPECT_EQ("", RFieldBase::Create("f", "IntegerTemplates<0ll,0ull>").Unwrap()->GetTypeAlias());

   EXPECT_EQ("IntegerTemplates<-1,1>", RFieldBase::Create("f", "IntegerTemplates<-1LL,1ULL>").Unwrap()->GetTypeName());
   EXPECT_EQ("", RFieldBase::Create("f", "IntegerTemplates<-1LL,1ULL>").Unwrap()->GetTypeAlias());

   EXPECT_EQ("IntegerTemplates<-2147483650,9223372036854775810u>",
             RFieldBase::Create("f", "IntegerTemplates<-2147483650ll,9223372036854775810>").Unwrap()->GetTypeName());
   EXPECT_EQ("",
             RFieldBase::Create("f", "IntegerTemplates<-2147483650ll,9223372036854775810>").Unwrap()->GetTypeAlias());

   EXPECT_THROW(RFieldBase::Create("f", "IntegerTemplates<-1u,0u>").Unwrap(), ROOT::RException);
   EXPECT_THROW(RFieldBase::Create("f", "IntegerTemplates<1u,0x>").Unwrap(), ROOT::RException);
}
