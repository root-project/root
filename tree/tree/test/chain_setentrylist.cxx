#include <stdexcept>
#include <string>

#include "TEntryList.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"
#include "TChain.h"

#include "gtest/gtest.h"

void FillTree(const char *filename, const char *treeName, int nevents) {
   TFile f{filename, "RECREATE"};
   TTree t{treeName, treeName};

   int b;
   t.Branch("b1", &b);

   for (int i = 0; i < nevents; ++i) {
      b = i;
      t.Fill();
   }

   t.Write();
   f.Close();
}

TEST(TChain, SetEntryListSyncNoSubLists) {

   const auto start1{0};
   const auto end1{5};
   auto treename1{"syncnosublists_tree10entries"};
   auto filename1{"syncnosublists_tree10entries.root"};
   auto fullpath1{"syncnosublists_tree10entries.root/syncnosublists_tree10entries"};
   const auto nentries1{10};

   FillTree(filename1, treename1, nentries1);

   TEntryList elist;

   for(auto entry = start1; entry < end1; entry++){
      elist.Enter(entry);
   }

   TChain c;
   c.Add(fullpath1);

   try{
      c.SetEntryList(&elist, "sync");
   } catch (const std::runtime_error &err){
      std::string msg{"In 'TChain::SetEntryList': "};
      msg += "the input TEntryList doesn't have sub entry lists. Please make sure too add them through ";
      msg += "TEntryList::AddSubList";
      EXPECT_STREQ(msg.c_str(), err.what());
   }

   gSystem->Unlink(filename1);

}

TEST(TChain, SetEntryListSyncWrongFile) {

   const auto start_1{0};
   const auto end_1{20};
   auto treename1{"syncwrongfile_tree10entries"};
   auto filename1{"syncwrongfile_tree10entries.root"};
   auto fullpath1{"syncwrongfile_tree10entries.root/syncwrongfile_tree10entries"};
   const auto nentries1{10};

   const auto start_2{0};
   const auto end_2{10};
   auto treename2{"syncwrongfile_tree20entries"};
   auto filename2{"syncwrongfile_tree20entries.root"};
   const auto nentries2{20};

   FillTree(filename1, treename1, nentries1);
   FillTree(filename2, treename2, nentries2);

   TEntryList elist;
   TEntryList sublist1{"", "", treename1, filename1};
   TEntryList sublist2{"", "", treename2, filename2};

   for(auto entry = start_1; entry < end_1; entry++){
      sublist1.Enter(entry);
   }

   for(auto entry = start_2; entry < end_2; entry++){
      sublist2.Enter(entry);
   }

   elist.AddSubList(&sublist1);
   elist.AddSubList(&sublist2);

   TChain c;
   c.Add(fullpath1);
   c.Add(fullpath1);

   try{
      c.SetEntryList(&elist, "sync");
   } catch (const std::runtime_error &err){
      std::string msg{"In 'TChain::SetEntryList': "};
      msg += "the sub entry list at index 1 doesn't correspond to treename '";
      msg += treename1;
      msg += "' and filename '";
      msg += filename1;
      msg += "': it has treename '";
      msg += treename2;
      msg += "' and filename '";
      msg += filename2;
      msg += "'";
      EXPECT_STREQ(msg.c_str(), err.what());
   }

   gSystem->Unlink(filename1);
   gSystem->Unlink(filename2);
}

TEST(TChain, SetEntryListSyncWrongNumberOfSubLists) {

   const auto start_1{0};
   const auto end_1{20};
   auto treename1{"syncwrongnumberofsublists_tree10entries"};
   auto filename1{"syncwrongnumberofsublists_tree10entries.root"};
   auto fullpath1{"syncwrongnumberofsublists_tree10entries.root/syncwrongnumberofsublists_tree10entries"};
   const auto nentries1{10};

   auto treename2{"syncwrongnumberofsublists_tree20entries"};
   auto filename2{"syncwrongnumberofsublists_tree20entries.root"};
   auto fullpath2{"syncwrongnumberofsublists_tree20entries.root/syncwrongnumberofsublists_tree20entries"};
   const auto nentries2{20};

   FillTree(filename1, treename1, nentries1);
   FillTree(filename2, treename2, nentries2);

   TEntryList elist;
   TEntryList sublist1{"", "", treename1, filename1};

   for(auto entry = start_1; entry < end_1; entry++){
      sublist1.Enter(entry);
   }

   elist.AddSubList(&sublist1);

   TChain c;
   c.Add(fullpath1);
   c.Add(fullpath2);

   try{
      c.SetEntryList(&elist, "sync");
   } catch (const std::runtime_error &err){
      std::string msg{"In 'TChain::SetEntryList': "};
      msg += "the number of sub entry lists in the input TEntryList (1)";
      msg += " is not equal to the number of files in the chain (2)";
      EXPECT_STREQ(msg.c_str(), err.what());
   }


   gSystem->Unlink(filename1);
   gSystem->Unlink(filename2);
}
