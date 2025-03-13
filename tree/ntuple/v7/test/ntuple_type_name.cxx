#include "ntuple_test.hxx"

TEST(RNTuple, TypeNameBasics)
{
   EXPECT_STREQ("float", ROOT::RField<float>::TypeName().c_str());
   EXPECT_STREQ("std::vector<std::string>", ROOT::RField<std::vector<std::string>>::TypeName().c_str());
   EXPECT_STREQ("CustomStruct", ROOT::RField<CustomStruct>::TypeName().c_str());
   EXPECT_STREQ("DerivedB", ROOT::RField<DerivedB>::TypeName().c_str());

   auto field = RField<DerivedB>("derived");
   EXPECT_EQ(sizeof(DerivedB), field.GetValueSize());

   EXPECT_STREQ("std::pair<std::pair<float,CustomStruct>,std::int32_t>",
                (ROOT::RField<std::pair<std::pair<float, CustomStruct>, int>>::TypeName().c_str()));
   EXPECT_STREQ("std::tuple<std::tuple<char,CustomStruct,char>,std::int32_t>",
                (ROOT::RField<std::tuple<std::tuple<char, CustomStruct, char>, int>>::TypeName().c_str()));
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
   EXPECT_TRUE(RFieldBase::Create("f", "std::int32_t").Unwrap()->GetTypeAlias().empty());

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
      model->MakeField<DataVector<StructUsingCollectionProxy<int>>>("f7");
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

   // Ensure the typed API does not throw an exception
   auto f1 = reader->GetModel().GetDefaultEntry().GetPtr<DataVector<int>>("f1");
   auto f4 = reader->GetModel().GetDefaultEntry().GetPtr<DataVector<bool, std::vector<unsigned int>>>("f4");
   auto f7 = reader->GetModel().GetDefaultEntry().GetPtr<DataVector<StructUsingCollectionProxy<int>>>("f7");
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

TEST(RNTuple, TClassDefaultTemplateParameterInner)
{
   FileRaii fileGuard("test_ntuple_default_template_parameter_inner.root");

   {
      auto model = RNTupleModel::Create();
      model->MakeField<DataVector<int>::Inner>("f1"); // default second template parameters is double
      model->MakeField<DataVector<int, float>::Inner>("f2");
      model->AddField(RFieldBase::Create("f3", "DataVector<int>::Inner").Unwrap());
      model->AddField(RFieldBase::Create("f4", "DataVector<Double32_t>::Inner").Unwrap());
      model->AddField(RFieldBase::Create("f5", "DataVector<int, double>::Inner").Unwrap());
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(0u, reader->GetNEntries());

   const auto &desc = reader->GetDescriptor();
   EXPECT_EQ("DataVector<std::int32_t,double>::Inner", desc.GetFieldDescriptor(desc.FindFieldId("f1")).GetTypeName());
   EXPECT_EQ("", desc.GetFieldDescriptor(desc.FindFieldId("f1")).GetTypeAlias());

   EXPECT_EQ("DataVector<std::int32_t,float>::Inner", desc.GetFieldDescriptor(desc.FindFieldId("f2")).GetTypeName());
   EXPECT_EQ("", desc.GetFieldDescriptor(desc.FindFieldId("f2")).GetTypeAlias());

   EXPECT_EQ("DataVector<std::int32_t,double>::Inner", desc.GetFieldDescriptor(desc.FindFieldId("f3")).GetTypeName());
   EXPECT_EQ("", desc.GetFieldDescriptor(desc.FindFieldId("f3")).GetTypeAlias());

   EXPECT_EQ("DataVector<double,double>::Inner", desc.GetFieldDescriptor(desc.FindFieldId("f4")).GetTypeName());
   EXPECT_EQ("DataVector<Double32_t,double>::Inner", desc.GetFieldDescriptor(desc.FindFieldId("f4")).GetTypeAlias());

   EXPECT_EQ("DataVector<std::int32_t,double>::Inner", desc.GetFieldDescriptor(desc.FindFieldId("f5")).GetTypeName());
   EXPECT_EQ("", desc.GetFieldDescriptor(desc.FindFieldId("f5")).GetTypeAlias());

   auto v1 = reader->GetView<DataVector<int>::Inner>("f1");
   auto v3 = reader->GetView<DataVector<int>::Inner>("f3");
   EXPECT_THROW(reader->GetView<DataVector<int>::Inner>("f2"), ROOT::RException);
}

TEST(RNTuple, TClassDefaultTemplateParameterNested)
{
   FileRaii fileGuard("test_ntuple_default_template_parameter_nested.root");

   {
      auto model = RNTupleModel::Create();
      model->MakeField<DataVector<int>::Nested<int>>("f1"); // default second template parameters are double
      model->MakeField<DataVector<int, float>::Nested<int, float>>("f2");
      model->AddField(RFieldBase::Create("f3", "DataVector<int>::Nested<int>").Unwrap());
      model->AddField(RFieldBase::Create("f4", "DataVector<Double32_t>::Nested<Double32_t>").Unwrap());
      model->AddField(RFieldBase::Create("f5", "DataVector<int, double>::Nested<int, double>").Unwrap());
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   EXPECT_EQ(0u, reader->GetNEntries());

   const auto &desc = reader->GetDescriptor();
   EXPECT_EQ("DataVector<std::int32_t,double>::Nested<std::int32_t,double>",
             desc.GetFieldDescriptor(desc.FindFieldId("f1")).GetTypeName());
   EXPECT_EQ("", desc.GetFieldDescriptor(desc.FindFieldId("f1")).GetTypeAlias());

   EXPECT_EQ("DataVector<std::int32_t,float>::Nested<std::int32_t,float>",
             desc.GetFieldDescriptor(desc.FindFieldId("f2")).GetTypeName());
   EXPECT_EQ("", desc.GetFieldDescriptor(desc.FindFieldId("f2")).GetTypeAlias());

   EXPECT_EQ("DataVector<std::int32_t,double>::Nested<std::int32_t,double>",
             desc.GetFieldDescriptor(desc.FindFieldId("f3")).GetTypeName());
   EXPECT_EQ("", desc.GetFieldDescriptor(desc.FindFieldId("f3")).GetTypeAlias());

   EXPECT_EQ("DataVector<double,double>::Nested<double,double>",
             desc.GetFieldDescriptor(desc.FindFieldId("f4")).GetTypeName());
   EXPECT_EQ("DataVector<Double32_t,double>::Nested<Double32_t,double>",
             desc.GetFieldDescriptor(desc.FindFieldId("f4")).GetTypeAlias());

   EXPECT_EQ("DataVector<std::int32_t,double>::Nested<std::int32_t,double>",
             desc.GetFieldDescriptor(desc.FindFieldId("f5")).GetTypeName());
   EXPECT_EQ("", desc.GetFieldDescriptor(desc.FindFieldId("f5")).GetTypeAlias());

   auto v1 = reader->GetView<DataVector<int>::Nested<int>>("f1");
   auto v3 = reader->GetView<DataVector<int>::Nested<int>>("f3");
   EXPECT_THROW(reader->GetView<DataVector<int>::Nested<int>>("f2"), ROOT::RException);
}

TEST(RNTuple, TypeNameTemplatesNestedAlias)
{
   auto hash = RFieldBase::Create("f", "EdmHash<1>").Unwrap();
   EXPECT_EQ("EdmHash<1>", hash->GetTypeName());
   EXPECT_EQ("", hash->GetTypeAlias());

   const auto hashSubfields = hash->GetConstSubfields();
   ASSERT_EQ(2, hashSubfields.size());
   EXPECT_EQ("fHash", hashSubfields[0]->GetFieldName());
   EXPECT_EQ("std::string", hashSubfields[0]->GetTypeName());
   EXPECT_EQ("EdmHash<1>::value_type", hashSubfields[0]->GetTypeAlias());

   EXPECT_EQ("fHash2", hashSubfields[1]->GetFieldName());
   EXPECT_EQ("std::string", hashSubfields[1]->GetTypeName());
   // FIXME: This should really be EdmHash<1>::value_typeT<EdmHash<1>::value_type>, but this is the value we get from
   // TDataMember::GetFullTypeName right now...
   EXPECT_EQ("value_typeT<EdmHash<1>::value_type>", hashSubfields[1]->GetTypeAlias());
}

TEST(RNTuple, ContextDependentTypeNames)
{
   // Adapted from https://gitlab.cern.ch/amete/rntuple-bug-report-20250219, reproducer of
   // https://github.com/root-project/root/issues/17774

   FileRaii fileGuard("test_ntuple_type_name_contextdep.root");

   {
      auto model = RNTupleModel::Create();
      auto fieldBase = RFieldBase::Create("foo", "DerivedWithTypedef").Unwrap();
      model->AddField(std::move(fieldBase));
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      auto entry = ntuple->GetModel().CreateBareEntry();
      auto ptr = std::make_unique<DerivedWithTypedef>();
      entry->BindRawPtr("foo", ptr.get());
      for (auto i = 0; i < 10; ++i) {
         ptr->m.push_back(i);
         ntuple->Fill(*entry);
         ptr->m.clear();
      }
   }

   {
      auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
      EXPECT_EQ(reader->GetNEntries(), 10);

      const auto &desc = reader->GetDescriptor();
      const auto fooId = desc.FindFieldId("foo");
      const auto baseId = desc.GetFieldDescriptor(fooId).GetLinkIds()[0];
      {
         const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("m", fooId));
         EXPECT_EQ(fdesc.GetTypeName(), "std::vector<std::int32_t>");
         EXPECT_EQ(fdesc.GetTypeAlias(), "MyVec<std::int32_t>");
      }
      {
         const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("a", baseId));
         EXPECT_EQ(fdesc.GetTypeName(), "float");
         EXPECT_EQ(fdesc.GetTypeAlias(), "");
      }
      {
         const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("v1", baseId));
         EXPECT_EQ(fdesc.GetTypeName(), "std::vector<float>");
         EXPECT_EQ(fdesc.GetTypeAlias(), "");
      }
      {
         const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("v2", baseId));
         EXPECT_EQ(fdesc.GetTypeName(), "std::vector<std::vector<float>>");
         EXPECT_EQ(fdesc.GetTypeAlias(), "");
      }
      {
         const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("s", baseId));
         EXPECT_EQ(fdesc.GetTypeName(), "std::string");
         EXPECT_EQ(fdesc.GetTypeAlias(), "");
      }
      {
         const auto &fdesc = desc.GetFieldDescriptor(desc.FindFieldId("b", baseId));
         EXPECT_EQ(fdesc.GetTypeName(), "std::byte");
         EXPECT_EQ(fdesc.GetTypeAlias(), "");
      }
   }
}
