#include <limits>

#include "RtypesCore.h"
#include "TChain.h"
#include "TEntryList.h"
#include "TFile.h"
#include "TSystem.h"
#include "TTree.h"

#include "gtest/gtest.h"

void FillTree(const char *filename, const char *treeName, ULong64_t nentries)
{
    TFile f{filename, "RECREATE"};
    TTree t{treeName, treeName};

    ULong64_t x;
    t.Branch("x", &x);

    for (ULong64_t i = 0; i < nentries; ++i)
    {
        x = i;
        t.Fill();
    }

    t.Write();
    f.Close();
}

/*
Regression test for https://github.com/root-project/root/issues/11026

Creates one tree with exactly (std::numeric_limits<int>::max()/100) entries.
Then builds a TChain with 101 such trees, so that the total number of entries is
higher than the maximum value of int. The chain is connected in lockstep with a
TEntryList which has one sub TEntryList per tree in the chain.

The test finally calls `TEntryList::GetEntryAndTree` for the entry index right
after the maximum int value. This triggered the int overflow bug described in
the linked issue.
*/
TEST(TEntryList, EntryGreaterThanMaxInt)
{
    constexpr int limit = std::numeric_limits<int>::max();
    constexpr Long64_t overlimit = static_cast<Long64_t>(limit) + 1;
    constexpr ULong64_t nentries = limit / 100;

    auto treename{"tentrylist_regression_intoverflow"};
    auto filename{"tentrylist_regression_intoverflow.root"};
    auto fullpath{"tentrylist_regression_intoverflow.root?#tentrylist_regression_intoverflow"};

    FillTree(filename, treename, nentries);

    TChain chain{treename};
    TEntryList elists{};
    for (int i = 0; i < 101; i++)
    {
        chain.Add(fullpath, nentries);
        // Each tree has its own TEntryList, which spans all entries of the
        // tree itself.
        TEntryList sublist{};
        sublist.SetTreeName(treename);
        sublist.SetFileName(filename);
        sublist.EnterRange(0, nentries);
        // Add the current TEntryList to the main one.
        elists.AddSubList(&sublist);
    }
    // This will set the correct tree index in each of the sub-TEntryList
    // objects in the main TEntryList.
    chain.SetEntryList(&elists, "sync");

    int treenum = -1;
    auto eindex = elists.GetEntryAndTree(overlimit, treenum);

    // The following are the expected values if `GetEntryAndTree` is properly
    // able to retrieve the current tree and the correct entry index in that
    // tree. It used to return treenum=-1,eindex=-1 because of the int overflow.
    EXPECT_EQ(treenum, 100);
    EXPECT_EQ(eindex, 48);

    gSystem->Unlink(filename);
}
