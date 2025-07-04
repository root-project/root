#include "ntuple_test.hxx"
#include "SimpleCollectionProxy.hxx"

TEST(RNTupleShow, Empty)
{
   std::string rootFileName{"test_ntuple_show_empty.root"};
   std::string ntupleName{"EmptyNTuple"};
   FileRaii fileGuard(rootFileName);
   {
      auto model = RNTupleModel::Create();
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, rootFileName);
      ntuple->Fill();
      ntuple->Fill();
   }
   auto model2 = RNTupleModel::Create();
   auto ntuple2 = RNTupleReader::Open(std::move(model2), ntupleName, rootFileName);

   std::ostringstream os;
   ntuple2->Show(0, os);
   std::string fString{"{}\n"};
   EXPECT_EQ(fString, os.str());

   std::ostringstream os1;
   ntuple2->Show(1, os1);
   std::string fString1{"{}\n"};
   EXPECT_EQ(fString1, os1.str());
}

TEST(RNTupleShow, BasicTypes)
{
   std::string rootFileName{"test_ntuple_show_basictypes.root"};
   std::string ntupleName{"SimpleTypesNtuple"};
   FileRaii fileGuard(rootFileName);
   {
      auto model = RNTupleModel::Create();
      auto fieldPt = model->MakeField<float>("pt");
      auto fielddb = model->MakeField<double>("db");
      auto fieldbyte = model->MakeField<std::byte>("byte");
      auto fieldint = model->MakeField<int>("int");
      auto fielduint = model->MakeField<unsigned>("uint");
      auto field64uint = model->MakeField<std::uint64_t>("uint64");
      auto fieldstring = model->MakeField<std::string>("string");
      auto fieldbool = model->MakeField<bool>("boolean");
      auto fieldu8 = model->MakeField<uint8_t>("uint8");
      auto fieldi8 = model->MakeField<int8_t>("int8");
      auto fieldbitset = model->MakeField<std::bitset<65>>("bitset");
      auto fielduniqueptr = model->MakeField<std::unique_ptr<std::string>>("pstring");
      auto fieldatomic = model->MakeField<std::atomic<bool>>("atomic");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, rootFileName);

      *fieldPt = 5.0f;
      *fielddb = 9.99;
      *fieldbyte = std::byte{137};
      *fieldint = -4;
      *fielduint = 3;
      *field64uint = 44444444444ull;
      *fieldstring = "TestString";
      *fieldbool = true;
      *fieldu8 = 97;
      *fieldi8 = 97;
      *fieldbitset = std::bitset<65>("10000000000000000000000000000010000000000000000000000000000010010");
      *fielduniqueptr = std::make_unique<std::string>("abc");
      *fieldatomic = false;
      ntuple->Fill();

      *fieldPt = 8.5f;
      *fielddb = 9.998;
      *fieldbyte = std::byte{42};
      *fieldint = -94;
      *fielduint = -30;
      *field64uint = 2299994967294ull;
      *fieldstring = "TestString2";
      *fieldbool = false;
      *fieldu8 = 98;
      *fieldi8 = 98;
      fieldbitset->flip();
      fielduniqueptr->reset();
      *fieldatomic = true;
      ntuple->Fill();
   }

   auto ntuple2 = RNTupleReader::Open(ntupleName, rootFileName);

   std::ostringstream os;
   ntuple2->Show(0, os);
   // clang-format off
   std::string fString{
R"({
  "pt": 5,
  "db": 9.99,
  "byte": 0x89,
  "int": -4,
  "uint": 3,
  "uint64": 44444444444,
  "string": "TestString",
  "boolean": true,
  "uint8": 97,
  "int8": 97,
  "bitset": "10000000000000000000000000000010000000000000000000000000000010010",
  "pstring": "abc",
  "atomic": false
}
)" };
   // clang-format on
   EXPECT_EQ(fString, os.str());

   std::ostringstream os1;
   ntuple2->Show(1, os1);
   // clang-format off
   std::string fString1{
R"({
  "pt": 8.5,
  "db": 9.998,
  "byte": 0x2a,
  "int": -94,
  "uint": 4294967266,
  "uint64": 2299994967294,
  "string": "TestString2",
  "boolean": false,
  "uint8": 98,
  "int8": 98,
  "bitset": "01111111111111111111111111111101111111111111111111111111111101101",
  "pstring": null,
  "atomic": true
}
)" };
   // clang-format on
   EXPECT_EQ(fString1, os1.str());

   try {
      ntuple2->LoadEntry(2);
      FAIL() << "loading a non-existing entry should throw";
   } catch (const ROOT::RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("entry with index 2 out of bounds"));
   }
}

TEST(RNTupleShow, Vectors)
{
   std::string rootFileName{"test_ntuple_show_vector.root"};
   std::string ntupleName{"VecNTuple"};
   FileRaii fileGuard(rootFileName);
   {
      auto model = RNTupleModel::Create();
      auto fieldIntVec = model->MakeField<std::vector<int>>("intVec");
      auto fieldFloatVecVec = model->MakeField<std::vector<std::vector<float>>>("floatVecVec");
      auto fieldBoolVecVec = model->MakeField<std::vector<std::vector<bool>>>("booleanVecVec");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, rootFileName);

      *fieldIntVec = std::vector<int>{4, 5, 6};
      *fieldFloatVecVec = std::vector<std::vector<float>>{std::vector<float>{0.1, 0.2}, std::vector<float>{1.1, 1.2}};
      *fieldBoolVecVec = std::vector<std::vector<bool>>{
         std::vector<bool>{false, true, false}, std::vector<bool>{false, true}, std::vector<bool>{true, false, false}};
      ntuple->Fill();

      fieldIntVec->emplace_back(7);
      fieldFloatVecVec->emplace_back(std::vector<float>{2.2, 2.3});
      fieldBoolVecVec->emplace_back(std::vector<bool>{false, true});
      ntuple->Fill();
   }
   auto model2 = RNTupleModel::Create();
   auto fieldIntVec = model2->MakeField<std::vector<int>>("intVec");
   auto fieldFloatVecVec = model2->MakeField<std::vector<std::vector<float>>>("floatVecVec");
   auto fieldBoolVecVec = model2->MakeField<std::vector<std::vector<bool>>>("booleanVecVec");
   auto ntuple2 = RNTupleReader::Open(std::move(model2), ntupleName, rootFileName);

   std::ostringstream os;
   ntuple2->Show(0, os);
   // clang-format off
   std::string fString{ 
R"({
  "intVec": [4, 5, 6],
  "floatVecVec": [[0.1, 0.2], [1.1, 1.2]],
  "booleanVecVec": [[false, true, false], [false, true], [true, false, false]]
}
)" };
   // clang-format on
   EXPECT_EQ(fString, os.str());

   std::ostringstream os1;
   ntuple2->Show(1, os1);
   // clang-format off
   std::string fString1{
R"({
  "intVec": [4, 5, 6, 7],
  "floatVecVec": [[0.1, 0.2], [1.1, 1.2], [2.2, 2.3]],
  "booleanVecVec": [[false, true, false], [false, true], [true, false, false], [false, true]]
}
)" };
   // clang-format on
   EXPECT_EQ(fString1, os1.str());
}

TEST(RNTupleShow, Arrays)
{
   std::string rootFileName{"test_ntuple_show_array.root"};
   std::string ntupleName{"Arrays"};
   FileRaii fileGuard(rootFileName);
   {
      auto model = RNTupleModel::Create();
      auto Intarrayfield = model->MakeField<std::array<int, 2>>("IntArray");
      auto Floatarrayfield = model->MakeField<std::array<float, 3>>("FloatArray");
      auto Vecarrayfield = model->MakeField<std::array<std::vector<double>, 4>>("ArrayOfVec");
      auto StringArray = model->MakeField<std::array<std::string, 2>>("stringArray");
      auto arrayOfArray = model->MakeField<std::array<std::array<bool, 2>, 3>>("ArrayOfArray");
      auto arrayVecfield = model->MakeField<std::vector<std::array<float, 2>>>("VecOfArray");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, rootFileName);

      *Intarrayfield = {1, 3};
      *Floatarrayfield = {3.5f, 4.6f, 5.7f};
      *Vecarrayfield = {std::vector<double>{1, 2}, std::vector<double>{4, 5}, std::vector<double>{7, 8, 9},
                        std::vector<double>{11}};
      *StringArray = {"First", "Second"};
      *arrayOfArray = {std::array<bool, 2>{true, false}, std::array<bool, 2>{false, true},
                       std::array<bool, 2>{false, false}};
      *arrayVecfield = {std::array<float, 2>{0, 1}, std::array<float, 2>{2, 3}, std::array<float, 2>{4, 5}};
      ntuple->Fill();

      *Intarrayfield = {2, 5};
      *Floatarrayfield = {2.3f, 5.7f, 11.13f};
      *Vecarrayfield = {std::vector<double>{17, 19}, std::vector<double>{23, 29}, std::vector<double>{31, 37, 41},
                        std::vector<double>{43}};
      *StringArray = {"Third", "Fourth"};
      *arrayOfArray = {std::array<bool, 2>{true, true}, std::array<bool, 2>{false, true},
                       std::array<bool, 2>{true, true}};
      *arrayVecfield = {std::array<float, 2>{6, 7}, std::array<float, 2>{8, 9}};
      ntuple->Fill();
   }
   auto model2 = RNTupleModel::Create();
   auto Intarrayfield = model2->MakeField<std::array<int, 2>>("IntArray");
   auto Floatarrayfield = model2->MakeField<std::array<float, 3>>("FloatArray");
   auto Vecarrayfield = model2->MakeField<std::array<std::vector<double>, 4>>("ArrayOfVec");
   auto StringArray = model2->MakeField<std::array<std::string, 2>>("stringArray");
   auto arrayOfArray = model2->MakeField<std::array<std::array<bool, 2>, 3>>("ArrayOfArray");
   auto arrayVecfield = model2->MakeField<std::vector<std::array<float, 2>>>("VecOfArray");
   auto ntuple2 = RNTupleReader::Open(std::move(model2), ntupleName, rootFileName);

   std::ostringstream os;
   ntuple2->Show(0, os);
   // clang-format off
   std::string fString{
R"({
  "IntArray": [1, 3],
  "FloatArray": [3.5, 4.6, 5.7],
  "ArrayOfVec": [[1, 2], [4, 5], [7, 8, 9], [11]],
  "stringArray": ["First", "Second"],
  "ArrayOfArray": [[true, false], [false, true], [false, false]],
  "VecOfArray": [[0, 1], [2, 3], [4, 5]]
}
)"};
   // clang-format on
   EXPECT_EQ(fString, os.str());

   std::ostringstream os1;
   ntuple2->Show(1, os1);
   // clang-format off
   std::string fString1{
R"({
  "IntArray": [2, 5],
  "FloatArray": [2.3, 5.7, 11.13],
  "ArrayOfVec": [[17, 19], [23, 29], [31, 37, 41], [43]],
  "stringArray": ["Third", "Fourth"],
  "ArrayOfArray": [[true, true], [false, true], [true, true]],
  "VecOfArray": [[6, 7], [8, 9]]
}
)"};
   // clang-format on
   EXPECT_EQ(fString1, os1.str());
}

TEST(RNTupleShow, Objects)
{
   std::string rootFileName{"test_ntuple_show_object.root"};
   std::string ntupleName{"Objects"};
   FileRaii fileGuard(rootFileName);
   {
      auto model = RNTupleModel::Create();
      auto customStructfield = model->MakeField<CustomStruct>("CustomStruct");
      auto customStructVec = model->MakeField<std::vector<CustomStruct>>("CustomStructVec");
      auto customStructArray = model->MakeField<std::array<CustomStruct, 2>>("CustomStructArray");
      auto derivedAfield = model->MakeField<DerivedA>("DerivedA");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), ntupleName, rootFileName);

      *customStructfield = CustomStruct{4.1f, std::vector<float>{0.1f, 0.2f, 0.3f},
                                        std::vector<std::vector<float>>{{1.1f, 1.2f, 1.3f}, {2.1f, 2.2f, 2.3f}},
                                        "Example1String", std::byte{0}};
      *customStructVec = {CustomStruct{4.2f, std::vector<float>{0.1f, 0.2f, 0.3f},
                                       std::vector<std::vector<float>>{{1.1f, 1.3f}, {2.1f, 2.2f, 2.3f}},
                                       "Example2String", std::byte{0}},
                          CustomStruct{4.3f, std::vector<float>{0.1f, 0.2f, 0.3f},
                                       std::vector<std::vector<float>>{{1.1f, 1.2f, 1.3f}, {2.1f, 2.3f}},
                                       "Example3String", std::byte{0}},
                          CustomStruct{4.4f, std::vector<float>{0.1f, 0.3f},
                                       std::vector<std::vector<float>>{{1.1f, 1.2f, 1.3f}, {2.1f, 2.2f, 2.3f}},
                                       "Example4String", std::byte{0}}};
      *customStructArray = {CustomStruct{4.5f, std::vector<float>{0.1f, 0.2f, 0.3f},
                                         std::vector<std::vector<float>>{{1.1f, 1.3f}, {2.1f, 2.2f, 2.3f}},
                                         "AnotherString1", std::byte{0}},
                            CustomStruct{4.6f, std::vector<float>{0.1f, 0.2f, 0.3f},
                                         std::vector<std::vector<float>>{{1.1f, 1.2f, 1.3f}, {2.1f, 2.3f}},
                                         "AnotherString2", std::byte{0}}};
      *derivedAfield = DerivedA();
      ntuple->Fill();
   }
   auto model2 = RNTupleModel::Create();
   auto customStructfield = model2->MakeField<CustomStruct>("CustomStruct");
   auto customStructVec = model2->MakeField<std::vector<CustomStruct>>("CustomStructVec");
   auto customStructArray = model2->MakeField<std::array<CustomStruct, 2>>("CustomStructArray");
   auto derivedAfield = model2->MakeField<DerivedA>("DerivedA");
   auto ntuple2 = RNTupleReader::Open(std::move(model2), ntupleName, rootFileName);

   std::ostringstream os;
   ntuple2->Show(0, os);
   // clang-format off
   std::string fString{ 
R"({
  "CustomStruct": {
    "a": 4.1,
    "v1": [0.1, 0.2, 0.3],
    "v2": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]],
    "s": "Example1String",
    "b": 0x00
  },
  "CustomStructVec": [)"
     R"({"a": 4.2, "v1": [0.1, 0.2, 0.3], "v2": [[1.1, 1.3], [2.1, 2.2, 2.3]], "s": "Example2String", "b": 0x00}, )"
     R"({"a": 4.3, "v1": [0.1, 0.2, 0.3], "v2": [[1.1, 1.2, 1.3], [2.1, 2.3]], "s": "Example3String", "b": 0x00}, )"
     R"({"a": 4.4, "v1": [0.1, 0.3], "v2": [[1.1, 1.2, 1.3], [2.1, 2.2, 2.3]], "s": "Example4String", "b": 0x00}],)" R"(
  "CustomStructArray": [)"
     R"({"a": 4.5, "v1": [0.1, 0.2, 0.3], "v2": [[1.1, 1.3], [2.1, 2.2, 2.3]], "s": "AnotherString1", "b": 0x00}, )"
     R"({"a": 4.6, "v1": [0.1, 0.2, 0.3], "v2": [[1.1, 1.2, 1.3], [2.1, 2.3]], "s": "AnotherString2", "b": 0x00}],)" R"(
  "DerivedA": {
    ":_0": {
      "a": 0,
      "v1": [],
      "v2": [],
      "s": "",
      "b": 0x00
    },
    "a_v": [],
    "a_s": ""
  }
}
)" };
   // clang-format on
   EXPECT_EQ(fString, os.str());
}

TEST(RNTupleShow, Collections)
{
   using ROOT::RRecordField;
   using ROOT::RVectorField;

   std::string rootFileName{"test_ntuple_show_collection.root"};
   std::string ntupleName{"Collections"};
   FileRaii fileGuard(rootFileName);
   {
      struct MyStruct {
         short myShort;
         float myFloat;
      };

      auto model = RNTupleModel::Create();
      std::vector<std::unique_ptr<RFieldBase>> leafFields;
      leafFields.emplace_back(std::make_unique<RField<short>>("myShort"));
      leafFields.emplace_back(std::make_unique<RField<float>>("myFloat"));
      auto recordField = std::make_unique<RRecordField>("_0", std::move(leafFields));
      EXPECT_EQ(offsetof(MyStruct, myShort), recordField->GetOffsets()[0]);
      EXPECT_EQ(offsetof(MyStruct, myFloat), recordField->GetOffsets()[1]);

      auto collectionField = RVectorField::CreateUntyped("myCollection", std::move(recordField));
      model->AddField(std::move(collectionField));
      model->Freeze();

      auto v = std::static_pointer_cast<std::vector<MyStruct>>(model->GetDefaultEntry().GetPtr<void>("myCollection"));
      auto writer = RNTupleWriter::Recreate(std::move(model), ntupleName, rootFileName);

      v->emplace_back(MyStruct({1, 10.0}));
      v->emplace_back(MyStruct({2, 20.0}));
      writer->Fill();
   }

   auto reader = RNTupleReader::Open(ntupleName, rootFileName);
   std::ostringstream osData;
   reader->Show(0, osData);
   // clang-format off
   std::string outputData{
R"({
  "myCollection": [{"myShort": 1, "myFloat": 10}, {"myShort": 2, "myFloat": 20}]
}
)" };
   // clang-format on
   EXPECT_EQ(outputData, osData.str());

   std::ostringstream osFields;
   reader->PrintInfo(ROOT::ENTupleInfo::kSummary, osFields);
   // clang-format off
   std::string outputFields{
      "************************************ NTUPLE ************************************\n"
      "* N-Tuple : Collections                                                        *\n"
      "* Entries : 1                                                                  *\n"
      "********************************************************************************\n"
      "* Field 1           : myCollection                                             *\n"
      "*   Field 1.1       : _0                                                       *\n"
      "*     Field 1.1.1   : myShort (std::int16_t)                                   *\n"
      "*     Field 1.1.2   : myFloat (float)                                          *\n"
      "********************************************************************************\n" };
   // clang-format on
   EXPECT_EQ(outputFields, osFields.str());
}

TEST(RNTupleShow, RVec)
{
   std::string rootFileName{"test_ntuple_show_rvec.root"};
   std::string ntupleName{"r"};
   FileRaii fileGuard(rootFileName);

   auto modelWrite = RNTupleModel::Create();
   auto fieldIntVec = modelWrite->MakeField<ROOT::RVec<int>>("intVec");
   auto fieldFloatVecVec = modelWrite->MakeField<ROOT::RVec<ROOT::RVec<float>>>("floatVecVec");
   auto fieldCustomStructVec = modelWrite->MakeField<ROOT::RVec<CustomStruct>>("customStructVec");

   auto modelRead = modelWrite->Clone();
   {
      auto ntuple = RNTupleWriter::Recreate(std::move(modelWrite), ntupleName, rootFileName);

      *fieldIntVec = ROOT::RVec<int>{1, 2, 3};
      *fieldFloatVecVec = ROOT::RVec<ROOT::RVec<float>>{ROOT::RVec<float>{0.1, 0.2}, ROOT::RVec<float>{1.1, 1.2}};
      *fieldCustomStructVec =
         ROOT::RVec<CustomStruct>{CustomStruct(), {1.f, {2.f, 3.f}, {{4.f}, {5.f}}, "foo", std::byte{0}}};
      ntuple->Fill();

      fieldIntVec->emplace_back(4);
      fieldFloatVecVec->emplace_back(ROOT::RVec<float>{2.1, 2.2});
      fieldCustomStructVec->emplace_back(CustomStruct{6.f, {7.f, 8.f}, {{9.f}, {10.f}}, "bar", std::byte{0}});
      ntuple->Fill();
   }

   auto r = RNTupleReader::Open(std::move(modelRead), ntupleName, rootFileName);

   std::ostringstream os;
   r->Show(0, os);
   // clang-format off
   std::string expected1{
R"({
  "intVec": [1, 2, 3],
  "floatVecVec": [[0.1, 0.2], [1.1, 1.2]],
  "customStructVec": [{"a": 0, "v1": [], "v2": [], "s": "", "b": 0x00}, {"a": 1, "v1": [2, 3], "v2": [[4], [5]], "s": "foo", "b": 0x00}]
}
)"};
   // clang-format on
   EXPECT_EQ(os.str(), expected1);

   std::ostringstream os2;
   r->Show(1, os2);
   // clang-format off
   std::string expected2{
R"({
  "intVec": [1, 2, 3, 4],
  "floatVecVec": [[0.1, 0.2], [1.1, 1.2], [2.1, 2.2]],
  "customStructVec": [)"
     R"({"a": 0, "v1": [], "v2": [], "s": "", "b": 0x00}, )"
     R"({"a": 1, "v1": [2, 3], "v2": [[4], [5]], "s": "foo", "b": 0x00}, )"
     R"({"a": 6, "v1": [7, 8], "v2": [[9], [10]], "s": "bar", "b": 0x00}])" R"(
}
)"};
   // clang-format on
   EXPECT_EQ(os2.str(), expected2);
}

TEST(RNTupleShow, RVecTypeErased)
{
   std::string rootFileName{"test_ntuple_show_rvec_typeerased.root"};
   std::string ntupleName{"r"};
   FileRaii fileGuard(rootFileName);
   {
      auto m = RNTupleModel::Create();

      auto intVecField = RFieldBase::Create("intVec", "ROOT::VecOps::RVec<int>").Unwrap();
      m->AddField(std::move(intVecField));

      auto floatVecVecField =
         RFieldBase::Create("floatVecVec", "ROOT::VecOps::RVec<ROOT::VecOps::RVec<float>>").Unwrap();
      m->AddField(std::move(floatVecVecField));

      auto customStructField = RFieldBase::Create("customStructVec", "ROOT::VecOps::RVec<CustomStruct>").Unwrap();
      m->AddField(std::move(customStructField));

      ROOT::RVec<int> intVec = {1, 2, 3};
      ROOT::RVec<ROOT::RVec<float>> floatVecVec{ROOT::RVec<float>{0.1, 0.2}, ROOT::RVec<float>{1.1, 1.2}};
      ROOT::RVec<CustomStruct> customStructVec{CustomStruct(), {1.f, {2.f, 3.f}, {{4.f}, {5.f}}, "foo", std::byte{0}}};

      m->Freeze();
      m->GetDefaultEntry().BindRawPtr("intVec", &intVec);
      m->GetDefaultEntry().BindRawPtr("floatVecVec", &floatVecVec);
      m->GetDefaultEntry().BindRawPtr("customStructVec", &customStructVec);

      auto ntuple = RNTupleWriter::Recreate(std::move(m), ntupleName, rootFileName);

      ntuple->Fill();

      intVec.emplace_back(4);
      floatVecVec.emplace_back(ROOT::RVec<float>{2.1, 2.2});
      customStructVec.emplace_back(CustomStruct{6.f, {7.f, 8.f}, {{9.f}, {10.f}}, "bar", std::byte{0}});
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open(ntupleName, rootFileName);

   std::ostringstream os;
   ntuple->Show(0, os);
   // clang-format off
   std::string fString{
R"({
  "intVec": [1, 2, 3],
  "floatVecVec": [[0.1, 0.2], [1.1, 1.2]],
  "customStructVec": [{"a": 0, "v1": [], "v2": [], "s": "", "b": 0x00}, {"a": 1, "v1": [2, 3], "v2": [[4], [5]], "s": "foo", "b": 0x00}]
}
)"};
   // clang-format on
   EXPECT_EQ(fString, os.str());

   std::ostringstream os1;
   ntuple->Show(1, os1);
   // clang-format off
   std::string fString1{
R"({
  "intVec": [1, 2, 3, 4],
  "floatVecVec": [[0.1, 0.2], [1.1, 1.2], [2.1, 2.2]],
  "customStructVec": [)"
     R"({"a": 0, "v1": [], "v2": [], "s": "", "b": 0x00}, )"
     R"({"a": 1, "v1": [2, 3], "v2": [[4], [5]], "s": "foo", "b": 0x00}, )"
     R"({"a": 6, "v1": [7, 8], "v2": [[9], [10]], "s": "bar", "b": 0x00}]
}
)"};
   // clang-format off
   EXPECT_EQ(fString1, os1.str());
}

TEST(RNTupleShow, CollectionProxy)
{
   SimpleCollectionProxy<StructUsingCollectionProxy<float>> proxy;
   auto klass = TClass::GetClass("StructUsingCollectionProxy<float>");
   klass->CopyCollectionProxy(proxy);

   FileRaii fileGuard("test_ntuple_show_collectionproxy.ntuple");
   {
      auto model = RNTupleModel::Create();
      auto proxyF = model->MakeField<StructUsingCollectionProxy<float>>("proxyF");
      auto vecProxyF = model->MakeField<std::vector<StructUsingCollectionProxy<float>>>("vecProxyF");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());
      proxyF->v.push_back(42.0f);
      proxyF->v.push_back(24.0f);
      StructUsingCollectionProxy<float> floats;
      floats.v.push_back(1.0f);
      floats.v.push_back(2.0f);
      vecProxyF->push_back(floats);
      vecProxyF->push_back(floats);
      ntuple->Fill();
   }

   {
      auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
      EXPECT_EQ(1U, ntuple->GetNEntries());

      std::ostringstream os;
      ntuple->Show(0, os);
      // clang-format off
      std::string expected{
R"({
  "proxyF": [42, 24],
  "vecProxyF": [[1, 2], [1, 2]]
}
)"};
      // clang-format on
      EXPECT_EQ(os.str(), expected);
   }
}

TEST(RNTupleShow, Map)
{
   FileRaii fileGuard("test_ntuple_show_map.ntuple");
   {
      auto model = RNTupleModel::Create();
      auto mapF = model->MakeField<std::map<std::string, float>>("mapF");
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "f", fileGuard.GetPath());

      *mapF = {{"foo", 3.14}, {"bar", 2.72}};
      ntuple->Fill();
   }

   auto ntuple = RNTupleReader::Open("f", fileGuard.GetPath());
   EXPECT_EQ(1U, ntuple->GetNEntries());

   std::ostringstream os;
   ntuple->Show(0, os);
   // clang-format off
   std::string expected{
R"({
  "mapF": [{"_0": "bar", "_1": 2.72}, {"_0": "foo", "_1": 3.14}]
}
)"};
   // clang-format on
   EXPECT_EQ(os.str(), expected);
}

TEST(RNTupleShow, Enum)
{

   FileRaii fileGuard("test_ntuple_show_enum.ntuple");
   {
      auto model = RNTupleModel::Create();
      auto ptrEnum = model->MakeField<CustomEnum>("enum");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      *ptrEnum = kCustomEnumVal;
      writer->Fill();
      *ptrEnum = static_cast<CustomEnum>(137);
      writer->Fill();
   }

   auto ntuple = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2U, ntuple->GetNEntries());

   std::ostringstream os0;
   ntuple->Show(0, os0);
   // clang-format off
   std::string expected{
R"({
  "enum": 7
}
)"};
   // clang-format on
   EXPECT_EQ(os0.str(), expected);

   std::ostringstream os1;
   ntuple->Show(1, os1);
   // clang-format off
   expected = std::string(
R"({
  "enum": 137
}
)");
   // clang-format on
   EXPECT_EQ(os1.str(), expected);
}

TEST(RNTupleShow, Friends)
{
   FileRaii fileGuard1("test_ntuple_show_friends1.ntuple");
   {
      auto model = RNTupleModel::Create();
      auto foo = model->MakeField<float>("foo");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl1", fileGuard1.GetPath());
      *foo = 3.14;
      writer->Fill();
   }

   FileRaii fileGuard2("test_ntuple_show_friends2.ntuple");
   {
      auto model = RNTupleModel::Create();
      auto bar = model->MakeField<float>("bar");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl2", fileGuard2.GetPath());
      *bar = 2.72;
      writer->Fill();
   }
}
