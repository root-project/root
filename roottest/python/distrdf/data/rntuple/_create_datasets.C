#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <vector>
#include <string>
#include <TRandom.h>

void create_check_backend()
{

   auto every_n_entries = 10;
   std::vector<std::string> ntpl_names{"tree_0", "tree_1", "tree_2"};
   std::vector<std::string> filenames{"distrdf_roottest_check_backend_0.root", "distrdf_roottest_check_backend_1.root",
                                      "distrdf_roottest_check_backend_2.root"};

   for (auto i = 0; i < ntpl_names.size(); i++) {
      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<int>("x");
      *fldX = 1;
      auto ntpl = ROOT::RNTupleWriter::Recreate(std::move(model), ntpl_names[i], filenames[i]);
      for (auto j = 0; j < 100; j++) {
         if (j % every_n_entries == 0) {
            ntpl->CommitCluster();
         }
         ntpl->Fill();
      }
   }
}

void create_cloned_actions()
{
   std::vector<std::uint64_t> clusters{66,   976,  1542, 1630, 2477, 3566, 4425, 4980, 5109, 5381,
                                       5863, 6533, 6590, 6906, 8312, 8361, 8900, 8952, 9144, 9676};
   std::string datasetname{"Events"};
   std::string filename{"distrdf_roottest_check_cloned_actions_asnumpy.root"};

   auto model = ROOT::RNTupleModel::Create();
   auto fldEv = model->MakeField<std::int64_t>("event");
   auto ntpl = ROOT::RNTupleWriter::Recreate(std::move(model), datasetname, filename);
   for (auto i = 0; i < 10000; i++) {
      *fldEv = i;
      // Flush a cluster of entries at the defined cluster boundaries
      if (std::find(clusters.begin(), clusters.end(), i) != clusters.end()) {
         ntpl->CommitCluster();
      }
      ntpl->Fill();
   }
}

void create_empty_rntuple()
{
   auto ntpl = ROOT::RNTupleWriter::Recreate(ROOT::RNTupleModel::Create(), "empty", "empty.root");
}

void create_definepersample()
{
   std::vector<std::string> filenames{"distrdf_roottest_definepersample_sample1.root",
                                      "distrdf_roottest_definepersample_sample2.root",
                                      "distrdf_roottest_definepersample_sample3.root"};
   std::string ntpl_name = "Events";
   for (const auto &fn : filenames) {
      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<ULong64_t>("x");
      auto ntpl = ROOT::RNTupleWriter::Recreate(std::move(model), ntpl_name, fn);
      for (ULong64_t entry = 0; entry < 10; entry++) {
         *fldX = entry;
         ntpl->Fill();
      }
   }
}

void create_friend_trees_alignment()
{
   std::vector<std::string> ntpl_names{
      "distrdf_roottest_check_friend_trees_alignment_1", "distrdf_roottest_check_friend_trees_alignment_2",
      "distrdf_roottest_check_friend_trees_alignment_3", "distrdf_roottest_check_friend_trees_alignment_4",
      "distrdf_roottest_check_friend_trees_alignment_5", "distrdf_roottest_check_friend_trees_alignment_6"};
   std::vector<std::string> filenames{
      "distrdf_roottest_check_friend_trees_alignment_1.root", "distrdf_roottest_check_friend_trees_alignment_2.root",
      "distrdf_roottest_check_friend_trees_alignment_3.root", "distrdf_roottest_check_friend_trees_alignment_4.root",
      "distrdf_roottest_check_friend_trees_alignment_5.root", "distrdf_roottest_check_friend_trees_alignment_6.root"};
   std::vector<std::pair<ULong64_t, ULong64_t>> limits{
      {0, 10}, {10, 20}, {20, 30}, {30, 40}, {40, 50}, {50, 60},
   };
   for (auto i = 0; i < limits.size(); i++) {
      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<ULong64_t>("x");
      auto ntpl = ROOT::RNTupleWriter::Recreate(std::move(model), ntpl_names[i], filenames[i]);
      for (ULong64_t entry = limits[i].first; entry < limits[i].second; entry++) {
         *fldX = entry;
         ntpl->CommitCluster();
         ntpl->Fill();
      }
   }
}

void create_friend_trees()
{
   auto create_ntpl = [](const std::string &ntplname, const std::string &filename, int gaus_mean) {
      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<double>("x");
      auto ntpl = ROOT::RNTupleWriter::Recreate(std::move(model), ntplname, filename);
      TRandom r;
      for (auto i = 0; i < 10000; i++) {
         *fldX = r.Gaus(gaus_mean, 1);
         ntpl->Fill();
      }
   };

   std::string main_ntplname = "T";
   std::string friend_ntplname = "TF";
   auto main_mean{10};
   auto friend_mean{20};
   std::string main_filename = "distrdf_roottest_check_friend_trees_main.root";
   std::string friend_filename = "distrdf_roottest_check_friend_trees_friend.root";

   create_ntpl(main_ntplname, main_filename, main_mean);
   create_ntpl(friend_ntplname, friend_filename, friend_mean);

   // 7584
   std::string ntpl_name_rn1 = "randomNumbers";
   std::string ntpl_name_rn2 = "randomNumbersBis";
   std::string filename_7584 = "distrdf_roottest_check_friend_trees_7584.root";
   TFile out_file{filename_7584.c_str(), "recreate"};
   {
      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<double>("x");
      auto ntpl = ROOT::RNTupleWriter::Append(std::move(model), ntpl_name_rn1, out_file);
      TRandom r;
      for (auto i = 0; i < 10000; i++) {
         *fldX = r.Gaus(main_mean, 1);
         ntpl->Fill();
      }
   }
   {
      auto model = ROOT::RNTupleModel::Create();
      auto fldX = model->MakeField<double>("x");
      auto ntpl = ROOT::RNTupleWriter::Append(std::move(model), ntpl_name_rn2, out_file);
      TRandom r;
      for (auto i = 0; i < 10000; i++) {
         *fldX = r.Gaus(friend_mean, 1);
         ntpl->Fill();
      }
   }
}

void create_reducer_merge()
{
   std::string ntpl_name = "tree";
   std::string filename = "distrdf_roottest_check_reducer_merge_1.root";
   auto model = ROOT::RNTupleModel::Create();
   auto fldV = model->MakeField<double>("v");
   auto ntpl = ROOT::RNTupleWriter::Recreate(std::move(model), ntpl_name, filename);
   for (auto i = 0; i < 100; i++) {
      *fldV = static_cast<double>(i);
      ntpl->Fill();
   }
}

void create_rungraphs()
{
   std::string ntpl_name = "tree";
   std::string filename = "distrdf_roottest_check_rungraphs.root";
   auto model = ROOT::RNTupleModel::Create();
   auto fldb1 = model->MakeField<int>("b1");
   *fldb1 = 42;
   auto fldb2 = model->MakeField<int>("b2");
   *fldb2 = 42;
   auto fldb3 = model->MakeField<int>("b3");
   *fldb3 = 42;
   auto ntpl = ROOT::RNTupleWriter::Recreate(std::move(model), ntpl_name, filename);
   auto nentries = 10000;
   auto every_n_entries = 5000;
   for (auto i = 0; i < nentries; i++) {
      if (i % every_n_entries == 0) {
         ntpl->CommitCluster();
      }
      ntpl->Fill();
   }
}

void _create_datasets()
{
   create_check_backend();
   create_cloned_actions();
   create_empty_rntuple();
   create_definepersample();
   create_friend_trees_alignment();
   create_friend_trees();
   create_reducer_merge();
   create_rungraphs();
}
