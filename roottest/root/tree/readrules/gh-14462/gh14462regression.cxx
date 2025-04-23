#include <TMemFile.h>
#include <TTree.h>

#include <string>
#include <memory>

#include "userClass.hxx"

#include "gtest/gtest.h"

void writeTree(TMemFile &outFile)
{
   TTree myTree("myTree", "");
   userClass obj;

   myTree.Branch("branch1", &obj);
   myTree.Fill();
   outFile.WriteObject(&myTree, myTree.GetName());
}

bool readCustomClassDataMember(TMemFile &inFile)
{
   userClass *obj_to_read{nullptr};
   std::unique_ptr<TTree> iTree{inFile.Get<TTree>("myTree")};
   iTree->SetBranchAddress("branch1", &obj_to_read);
   iTree->GetEntry(0);

   return obj_to_read->transientMember;
}

// Regression test for https://github.com/root-project/root/issues/14462
TEST(ReadRules, CustomReadRuleWithSpace)
{
   TMemFile testFile{"customReadRuleWithSpace.root", "RECREATE"};

   writeTree(testFile);

   auto flag = readCustomClassDataMember(testFile);

   EXPECT_TRUE(flag);
}
