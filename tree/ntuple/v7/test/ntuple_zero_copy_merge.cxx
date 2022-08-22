#include "ntuple_test.hxx"

using namespace std;
// Import classes from experimental namespace for the time being
using RNTupleModel = ROOT::Experimental::RNTupleModel;
using RNTupleReader = ROOT::Experimental::RNTupleReader;
using RNTupleWriter = ROOT::Experimental::RNTupleWriter;

constexpr char const* kNTupleFileName1 = "ntpl001_staff.root";
constexpr char const* kNTupleFileName2 = "ntpl001_staff_new.root";
constexpr char const* kNTupleFileNameMerged = "ntpl001_staff_merged.root";

TEST(RNTuple, ZeroCopyMerge)
{
   int val = 0;
   size_t SZ_TUPLE1 = 3500;
   size_t SZ_TUPLE2 = 3500;
   {
      // We create a unique pointer to an empty data model
      auto model = RNTupleModel::Create();
      auto fldAge = model->MakeField<int>("Age");
      auto fldGrade = model->MakeField<int>("Grade");
      auto fldStep = model->MakeField<int>("Step");

      auto options = ROOT::Experimental::RNTupleWriteOptions();
      options.SetContainerFormat(ENTupleContainerFormat::kBare);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff", kNTupleFileName1, options);

      std::string record;


      for ( size_t i = 0; i < SZ_TUPLE1; i++) {
         std::istringstream iss(record);
         *fldGrade = val;
         *fldAge = val * 10;
         *fldStep = val * 100;
         ntuple->Fill();

         if (val == 1500) {
            // break;
            ntuple->CommitCluster(true);
         }
         val++;
      }
      ntuple->CommitCluster();
   }

   {
      auto model = RNTupleModel::Create();
      auto fldAge = model->MakeField<int>("Age");
      auto fldGrade = model->MakeField<int>("Grade");
      auto fldStep = model->MakeField<int>("Step");

      auto options = ROOT::Experimental::RNTupleWriteOptions();
      options.SetContainerFormat(ENTupleContainerFormat::kBare);
      auto ntuple = RNTupleWriter::Recreate(std::move(model), "Staff_New", kNTupleFileName2, options);

      std::string record;


      for ( size_t i=0; i < SZ_TUPLE2; i++) {
         std::istringstream iss(record);
         *fldGrade = val;
         *fldAge = val * 10;
         *fldStep = val * 100;
         ntuple->Fill();

         if (val == 2500) {
            // break;
            ntuple->CommitCluster(true);
         }
         val++;
      }
   }

   auto options = ROOT::Experimental::RNTupleWriteOptions();

   options.SetContainerFormat(ENTupleContainerFormat::kBare);

   auto model2 = RNTupleModel::Create();
   auto fldAge2 = model2->MakeField<int>("Age");
   auto fldGrade2 = model2->MakeField<int>("Grade");
   auto fldStep2 = model2->MakeField<int>("Step");
   {
      RNTupleWriter ntuple_dst(std::move(model2),
                               std::make_unique<RPageSinkFile>("Staff_Merged", kNTupleFileNameMerged, options));

      ntuple_dst.FastDuplicate(
         "Staff", kNTupleFileName1, 0);
      ntuple_dst.FastDuplicate(
         "Staff_New", kNTupleFileName2, 0);
   };

   {

      auto ntupleMerged = RNTupleReader::Open("Staff_Merged", kNTupleFileNameMerged);
      ntupleMerged->PrintInfo();

      auto ntuple1 = RNTupleReader::Open("Staff", kNTupleFileName1);
      auto ntuple2 = RNTupleReader::Open("Staff_New", kNTupleFileName2);

      auto desc1 = ntuple1->GetDescriptor();
      auto desc2 = ntuple2->GetDescriptor();
      auto descMerged = ntupleMerged->GetDescriptor();

      auto nEntries1 = desc1->GetNEntries();
      auto nEntries2 = desc2->GetNEntries();
      auto nEntriesMerged = descMerged->GetNEntries();
      EXPECT_EQ(nEntries1 + nEntries2, nEntriesMerged);

      auto nClusters1 = desc1->GetNClusters();
      auto nClusters2 = desc2->GetNClusters();
      auto nClustersMerged = descMerged->GetNClusters();
      EXPECT_EQ(nClusters1 + nClusters2, nClustersMerged);

      auto nClusterGroup1 = desc1->GetNClusterGroups();
      auto nClusterGroup2 = desc2->GetNClusterGroups();
      auto nClusterGroupMerged = descMerged->GetNClusterGroups();
      EXPECT_EQ(nClusterGroup1 + nClusterGroup2, nClusterGroupMerged);



      auto viewAgeMerged = ntupleMerged->GetView<int>("Age");
      auto viewGradeMerged = ntupleMerged->GetView<int>("Grade");
      auto viewStepMerged = ntupleMerged->GetView<int>("Step");

      auto viewAge1 = ntuple1->GetView<int>("Age");
      auto viewGrade1 = ntuple1->GetView<int>("Grade");
      auto viewStep1 = ntuple1->GetView<int>("Step");

      auto viewAge2 = ntuple2->GetView<int>("Age");
      auto viewGrade2 = ntuple2->GetView<int>("Grade");
      auto viewStep2 = ntuple2->GetView<int>("Step");


      int globalIdx = 0;
      std::cout << "Age Grade Step" << std::endl;
      for (auto localIdx : ntuple1->GetEntryRange()) {
         //std::cout << viewGradeMerged(globalIdx) << " " << viewAgeMerged(globalIdx) << " " << viewStepMerged(globalIdx) << std::endl;
         EXPECT_EQ(viewAge1(localIdx), viewAgeMerged(globalIdx));
         EXPECT_EQ(viewGrade1(localIdx), viewGradeMerged(globalIdx));
         EXPECT_EQ(viewStep1(localIdx), viewStepMerged(globalIdx));
         globalIdx++;
      }
      for (auto localIdx : ntuple2->GetEntryRange()) {
         //std::cout << viewGradeMerged(globalIdx) << " " << viewAgeMerged(globalIdx) << " " << viewStepMerged(globalIdx) << std::endl;
         EXPECT_EQ(viewAge2(localIdx), viewAgeMerged(globalIdx));
         EXPECT_EQ(viewGrade2(localIdx), viewGradeMerged(globalIdx));
         EXPECT_EQ(viewStep2(localIdx), viewStepMerged(globalIdx));
         globalIdx++;
      }
   }
}