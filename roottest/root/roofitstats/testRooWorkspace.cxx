#include <RooWorkspace.h>

#include <TFile.h>

#include <gtest/gtest.h>

// This test covers an issue that was reported after updates to ROOT IO:
//
//     https://github.com/root-project/root/issues/10282
//
// The reproducer workspace was created with ROOT 6.26.00 with the following
// script:
//
// ```C++
// auto f = TFile::Open("toyws/WS-boostedHbb-glob_xs_toy.root");
//
// RooWorkspace ws{"combWS", "__temp__"};
// ws.defineSet("myset", RooArgSet{});
// ws.writeToFile("test_workspace_01.root");
// ```
//
// The original toy workspace was created by ATLAS users and it can be found
// here (some ROOT 6.24 release was used to produce these workspaces):
//
//     https://gitlab.cern.ch/kran/toyws/-/tree/master
//
// The script above aimed for a workspace that is as tiny as possible while
// reproducing the problem. These three conditions were found sufficient to
// create a workspace that is affected by issue 10282:
//
//   1. A file with a broken workspace needs to be opened
//   2. The reproducer workspace needs to have the same name as the broken
//      workspace in that file
//   3. The reproducer workspace must have some RooArgSet defined
TEST(RooWorkspace, Issue_10282) {
   auto f = TFile::Open("test_workspace_01.root");
   auto * ws = f->Get<RooWorkspace>("combWS");

   ASSERT_NE(ws->set("myset"), nullptr);
   ASSERT_EQ(ws->set("myset")->size(), 0);
}
