#include "ntuple_test.hxx"

TEST(RNTupleBulk, Simple)
{
   FileRaii fileGuard("test_ntuple_bulk_simple.root");
   {
      auto model = RNTupleModel::Create();
      auto fldInt = model->MakeField<int>("int");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      for (int i = 0; i < 10; ++i) {
         *fldInt = i;
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   RFieldBase::RBulkValues bulk = reader->GetModel().CreateBulk("int");

   auto mask = std::make_unique<bool[]>(10);
   std::fill(mask.get(), mask.get() + 10, false /* the optimization for simple fields should ignore the mask */);
   auto intArr5 = static_cast<int *>(bulk.ReadBulk(RNTupleLocalIndex(0, 0), mask.get(), 5));
   for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, intArr5[i]);
   }

   auto intArr1 = static_cast<int *>(bulk.ReadBulk(RNTupleLocalIndex(0, 1), mask.get(), 1));
   EXPECT_EQ(1, intArr1[0]);
   EXPECT_EQ(static_cast<int *>(intArr5) + 1, static_cast<int *>(intArr1));

   auto intArr10 = static_cast<int *>(bulk.ReadBulk(RNTupleLocalIndex(0, 0), mask.get(), 10));
   for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(i, intArr10[i]);
   }
}

TEST(RNTupleBulk, Complex)
{
   FileRaii fileGuard("test_ntuple_bulk_complex.root");
   {
      auto model = RNTupleModel::Create();
      auto fldS = model->MakeField<CustomStruct>("S");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      for (int i = 0; i < 10; ++i) {
         fldS->a = i;
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   RFieldBase::RBulkValues bulk = reader->GetModel().CreateBulk("S");
   auto mask = std::make_unique<bool[]>(10);
   for (unsigned int i = 0; i < 10; ++i)
      mask[i] = (i % 2 == 0);

   auto SArr5 = static_cast<CustomStruct *>(bulk.ReadBulk(RNTupleLocalIndex(0, 0), mask.get(), 5));
   for (int i = 0; i < 5; ++i) {
      EXPECT_FLOAT_EQ((i % 2 == 0) ? float(i) : 0.0, SArr5[i].a);
   }

   auto SArr1 = static_cast<CustomStruct *>(bulk.ReadBulk(RNTupleLocalIndex(0, 1), mask.get() + 1, 1));
   EXPECT_FLOAT_EQ(0.0, SArr1[0].a);
   EXPECT_EQ(static_cast<CustomStruct *>(SArr5) + 1, static_cast<CustomStruct *>(SArr1));

   SArr1 = static_cast<CustomStruct *>(bulk.ReadBulk(RNTupleLocalIndex(0, 1), mask.get(), 1));
   EXPECT_FLOAT_EQ(1.0, SArr1[0].a);
   EXPECT_EQ(static_cast<CustomStruct *>(SArr5) + 1, static_cast<CustomStruct *>(SArr1));

   for (unsigned int i = 0; i < 10; ++i)
      mask[i] = !mask[i];
   auto SArr10 = static_cast<CustomStruct *>(bulk.ReadBulk(RNTupleLocalIndex(0, 0), mask.get(), 10));
   for (int i = 0; i < 10; ++i) {
      EXPECT_FLOAT_EQ((i % 2 == 0) ? 0.0 : float(i), SArr10[i].a);
   }

   auto SArrAll = static_cast<CustomStruct *>(bulk.ReadBulk(RNTupleLocalIndex(0, 0), nullptr, 10));
   for (int i = 0; i < 10; ++i) {
      EXPECT_FLOAT_EQ(float(i), SArrAll[i].a);
   }
}

TEST(RNTupleBulk, CardinalityField)
{
   FileRaii fileGuard("test_ntuple_bulk_cardinality.root");
   {
      auto model = RNTupleModel::Create();
      auto fldVec = model->MakeField<ROOT::RVec<int>>("vint");
      model->AddProjectedField(std::make_unique<RField<ROOT::RNTupleCardinality<std::uint32_t>>>("card32"),
                               [](const std::string &) { return "vint"; });
      model->AddProjectedField(std::make_unique<RField<ROOT::RNTupleCardinality<std::uint64_t>>>("card64"),
                               [](const std::string &) { return "vint"; });
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      for (int i = 0; i < 10; ++i) {
         fldVec->resize(i);
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &model = reader->GetModel();

   RFieldBase::RBulkValues bulk32 = model.CreateBulk("card32");
   RFieldBase::RBulkValues bulk64 = model.CreateBulk("card64");

   auto mask = std::make_unique<bool[]>(10);
   std::fill(mask.get(), mask.get() + 10, false /* the cardinality field optimization should ignore the mask */);

   auto card32Arr = static_cast<std::uint32_t *>(bulk32.ReadBulk(RNTupleLocalIndex(0, 0), mask.get(), 10));
   auto card64Arr = static_cast<std::uint64_t *>(bulk64.ReadBulk(RNTupleLocalIndex(0, 0), mask.get(), 10));
   for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(i, card32Arr[i]);
      EXPECT_EQ(i, card64Arr[i]);
   }
}

TEST(RNTupleBulk, RVec)
{
   FileRaii fileGuard("test_ntuple_bulk_rvec.root");
   {
      auto model = RNTupleModel::Create();
      auto fldVecI = model->MakeField<ROOT::RVecI>("vint");
      auto fldVecS = model->MakeField<ROOT::RVec<CustomStruct>>("vs");
      auto fldVecVI = model->MakeField<ROOT::RVec<ROOT::RVecI>>("vvint");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      for (int i = 0; i < 10; ++i) {
         fldVecI->resize(i);
         fldVecS->resize(i);
         fldVecVI->resize(i);
         for (int j = 0; j < i; ++j) {
            fldVecI->at(j) = j;
            fldVecS->at(j).a = j;
            fldVecVI->at(j).resize(j);
            for (int k = 0; k < j; ++k) {
               fldVecVI->at(j).at(k) = k;
            }
         }
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   const auto &model = reader->GetModel();

   RFieldBase::RBulkValues bulkI = model.CreateBulk("vint");
   RFieldBase::RBulkValues bulkS = model.CreateBulk("vs");
   RFieldBase::RBulkValues bulkVI = model.CreateBulk("vvint");

   auto mask = std::make_unique<bool[]>(10);
   std::fill(mask.get(), mask.get() + 10, true);
   mask[1] = false; // the RVec<simple type> field optimization should ignore the mask

   auto iArr = static_cast<ROOT::RVecI *>(bulkI.ReadBulk(RNTupleLocalIndex(0, 0), mask.get(), 10));
   auto sArr = static_cast<ROOT::RVec<CustomStruct> *>(bulkS.ReadBulk(RNTupleLocalIndex(0, 0), mask.get(), 10));
   auto viArr = static_cast<ROOT::RVec<ROOT::RVecI> *>(bulkVI.ReadBulk(RNTupleLocalIndex(0, 0), mask.get(), 10));
   for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(i, iArr[i].size());
      EXPECT_EQ(i == 1 ? 0 : i, sArr[i].size());
      EXPECT_EQ(i == 1 ? 0 : i, viArr[i].size());
      for (std::size_t j = 0; j < iArr[i].size(); ++j) {
         EXPECT_EQ(j, iArr[i].at(j));
      }
      // RVec<PoD> should have all the vector elements of the bulk stored consecutively in memory
      if (i > 1) {
         EXPECT_EQ(&iArr[i - 1][0] + iArr[i - 1].size(), &iArr[i][0]);
      }
      for (std::size_t j = 0; j < sArr[i].size(); ++j) {
         EXPECT_FLOAT_EQ(j, sArr[i].at(j).a);
      }
      for (std::size_t j = 0; j < viArr[i].size(); ++j) {
         EXPECT_EQ(j, viArr[i].at(j).size());
         for (std::size_t k = 0; k < viArr[i].at(j).size(); ++k) {
            EXPECT_EQ(k, viArr[i][j][k]);
         }
      }
   }
}

TEST(RNTupleBulk, Adopted)
{
   FileRaii fileGuard("test_ntuple_bulk_adopted.root");
   {
      auto model = RNTupleModel::Create();
      auto fldVecI = model->MakeField<ROOT::RVecI>("vint");
      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      for (int i = 0; i < 10; ++i) {
         fldVecI->resize(i);
         for (int j = 0; j < i; ++j) {
            fldVecI->at(j) = j;
         }
         writer->Fill();
      }
   }

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   RFieldBase::RBulkValues bulkI = reader->GetModel().CreateBulk("vint");

   auto mask = std::make_unique<bool[]>(10);
   std::fill(mask.get(), mask.get() + 10, true);

   auto iArr = static_cast<ROOT::RVecI *>(bulkI.ReadBulk(RNTupleLocalIndex(0, 0), mask.get(), 10));
   for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(i, iArr[i].size());
      for (std::size_t j = 0; j < iArr[i].size(); ++j) {
         EXPECT_EQ(j, iArr[i].at(j));
      }
   }

   auto buf1 = std::make_unique<ROOT::RVecI[]>(10);
   bulkI.AdoptBuffer(buf1.get(), 10);
   bulkI.ReadBulk(RNTupleLocalIndex(0, 0), mask.get(), 10);
   for (int i = 0; i < 10; ++i) {
      EXPECT_EQ(i, buf1[i].size());
      for (std::size_t j = 0; j < buf1[i].size(); ++j) {
         EXPECT_EQ(j, buf1[i].at(j));
      }
   }

   auto buf2 = std::make_unique<ROOT::RVecI[]>(10);
   bulkI.AdoptBuffer(buf2.get(), 5);
   EXPECT_THROW(bulkI.ReadBulk(RNTupleLocalIndex(0, 0), mask.get(), 10), ROOT::RException);
   bulkI.ReadBulk(RNTupleLocalIndex(0, 0), mask.get(), 5);
   for (int i = 0; i < 5; ++i) {
      EXPECT_EQ(i, buf2[i].size());
      for (std::size_t j = 0; j < buf2[i].size(); ++j) {
         EXPECT_EQ(j, buf2[i].at(j));
      }
   }
}
