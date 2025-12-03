#include "ntuple_test.hxx"

TEST(RNtuplePrint, FullString)
{
   FileRaii fileGuard("test_ntuple_print_fullstring.root");
   {
      auto model = RNTupleModel::Create();
      auto fieldPx = model->MakeField<float>("px");
      auto fieldPy = model->MakeField<float>("py");
      auto fieldProj = RFieldBase::Create("proj", "float").Unwrap();
      model->AddProjectedField(std::move(fieldProj), [](const std::string &) { return std::string("px"); });
      RNTupleWriteOptions options;
      options.SetCompression(0);
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath(), options);
      *fieldPx = 1.0;
      *fieldPy = 1.0;
      writer->Fill();
   }
   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   std::ostringstream osSummary;
   reader->PrintInfo(ROOT::ENTupleInfo::kSummary, osSummary);
   std::string reference{std::string("")
       + "************************************ NTUPLE ************************************\n"
       + "* N-Tuple : ntpl                                                               *\n"
       + "* Entries : 1                                                                  *\n"
       + "********************************************************************************\n"
       + "* Field 1   : px (float)                                                       *\n"
       + "* Field 2   : py (float)                                                       *\n"
       + "* Field 3   : proj (float)                                                     *\n"
       + "********************************************************************************\n"};
   EXPECT_EQ(reference, osSummary.str());

   std::ostringstream osDetails;
   reader->PrintInfo(ROOT::ENTupleInfo::kStorageDetails, osDetails);
   reference = "============================================================\n"
               "NTUPLE:      ntpl\n"
               "Compression: 0\n"
               "------------------------------------------------------------\n"
               "  # Entries:        1\n"
               "  # Fields:         4\n"
               "  # Columns:        2\n"
               "  # Alias Columns:  1\n"
               "  # Pages:          2\n"
               "  # Clusters:       1\n"
               "  Size on storage:  8 B\n"
               "  Compression rate: 1.00\n"
               "  Header size:      .* B\n"
               "  Footer size:      .* B\n"
               "  Metadata / data:  .*\n"
               "------------------------------------------------------------\n"
               "CLUSTER DETAILS\n"
               "------------------------------------------------------------\n"
               "  #     0   Entry range:     .0..0.  --  1\n"
               "            # Pages:         2\n"
               "            Size on storage: 8 B\n"
               "            Compression:     1.00\n"
               "------------------------------------------------------------\n"
               "COLUMN DETAILS\n"
               "------------------------------------------------------------\n"
               "  px .#0.  --  Real32                                 .id:0.\n"
               "    # Elements:          1\n"
               "    # Pages:             1\n"
               "    Avg elements / page: 1\n"
               "    Avg page size:       4 B\n"
               "    Size on storage:     4 B\n"
               "    Compression:         1.00\n"
               "............................................................\n"
               "  py .#0.  --  Real32                                 .id:1.\n"
               "    # Elements:          1\n"
               "    # Pages:             1\n"
               "    Avg elements / page: 1\n"
               "    Avg page size:       4 B\n"
               "    Size on storage:     4 B\n"
               "    Compression:         1.00\n"
               "............................................................\n";
   EXPECT_THAT(osDetails.str(), testing::MatchesRegex(reference));
}

TEST(RNtuplePrint, Int)
{
   std::stringstream os;
   RPrintSchemaVisitor visitor(os);
   RField<int> testField("intTest");
   testField.AcceptVisitor(visitor);
   std::string expected{std::string("")
       + "* Field 1   : intTest (std::int32_t)                                           *\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, Float)
{
   std::stringstream os;
   RPrintSchemaVisitor visitor(os, 'a');
   RField<float> testField("floatTest");
   testField.AcceptVisitor(visitor);
   std::string expected{std::string("")
       + "a Field 1   : floatTest (float)                                                a\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, Vector)
{
   std::stringstream os;
   RPrepareVisitor prepVisitor;
   RField<std::vector<float>> testField("floatVecTest");
   testField.AcceptVisitor(prepVisitor);
   RPrintSchemaVisitor visitor(os, '$');
   visitor.SetDeepestLevel(prepVisitor.GetDeepestLevel());
   visitor.SetNumFields(prepVisitor.GetNumFields());
   testField.AcceptVisitor(visitor);
   std::string expected{std::string("")
       + "$ Field 1       : floatVecTest (std::vector<float>)                            $\n"
       + "$   Field 1.1   : _0 (float)                                                   $\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, ArrayAsRVec)
{
   std::stringstream os;
   RPrepareVisitor prepVisitor;
   ROOT::RArrayAsRVecField testField("arrayasrvecfield", std::make_unique<ROOT::RField<float>>("myfloat"), 0);
   testField.AcceptVisitor(prepVisitor);
   RPrintSchemaVisitor visitor(os, '$');
   visitor.SetDeepestLevel(prepVisitor.GetDeepestLevel());
   visitor.SetNumFields(prepVisitor.GetNumFields());
   testField.AcceptVisitor(visitor);
   std::string expected{std::string("") +
                        "$ Field 1       : arrayasrvecfield (ROOT::VecOps::RVec<float>)                 $\n" +
                        "$   Field 1.1   : myfloat (float)                                              $\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, VectorNested)
{
   std::stringstream os;
   RPrepareVisitor prepVisitor;
   RField<std::vector<std::vector<float>>> testField("floatVecVecTest");
   testField.AcceptVisitor(prepVisitor);
   RPrintSchemaVisitor visitor(os, 'x');
   visitor.SetDeepestLevel(prepVisitor.GetDeepestLevel());
   visitor.SetNumFields(prepVisitor.GetNumFields());
   testField.AcceptVisitor(visitor);
   std::string expected{std::string("")
       + "x Field 1           : floatVecVecTest (std::vector<std::vector<float>>)        x\n"
       + "x   Field 1.1       : _0 (std::vector<float>)                                  x\n"
       + "x     Field 1.1.1   : _0 (float)                                               x\n"};
   EXPECT_EQ(expected, os.str());
}

TEST(RNtuplePrint, NarrowManyEntriesVecVecTraverse)
{
   std::stringstream os;
   RPrepareVisitor prepVisitor;
   RField<std::vector<std::vector<float>>> testField("floatVecVecTest");
   testField.AcceptVisitor(prepVisitor);
   RPrintSchemaVisitor visitor(os, ' ', 25);
   visitor.SetDeepestLevel(prepVisitor.GetDeepestLevel());
   visitor.SetNumFields(prepVisitor.GetNumFields());
   testField.AcceptVisitor(visitor);
   std::string expected{std::string("")
       + "  Field 1    : floatV... \n"
       + "    Field... : _0 (st... \n"
       + "      Fie... : _0 (fl... \n"};
   EXPECT_EQ(expected, os.str());
}

/* Currently the width can't be set by PrintInfo(). This test will be enabled when this feature is added.
TEST(RNTuplePrint, TooShort)
{
FileRaii fileGuard("test.root");
{
   auto model = RNTupleModel::Create();
   auto fieldPt = model->MakeField<float>("pt");
   auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff", "test.root");
}
auto ntuple2 = RNTupleReader::Open("Staff", "test.root");
std::ostringstream os;
ntuple2->PrintInfo(ROOT::ENTupleInfo::kSummary, os, '+', 29);
std::string fString{"The width is too small! Should be at least 30.\n"};
EXPECT_EQ(fString, os.str());
}
*/
