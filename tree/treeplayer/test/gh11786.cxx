/// Regression test for https://github.com/root-project/root/issues/11786
///
/// TTreePlayer::Scan should honour the printf left-justification flag ('-')
/// in the "col=" option. Before the fix, a '-' in the column specification
/// would abort the parsing of the format list: all columns from the first '-'
/// onwards reverted to the default width and stayed right-justified.

#include <TTree.h>
#include <TTreePlayer.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include <gtest/gtest.h>

namespace {

struct FileRAII {
   const char *fPath;
   FileRAII(const char *name) : fPath(name) {}
   ~FileRAII() { std::remove(fPath); }
};

// Capture the (redirected) output of a TTree::Scan call.
std::string
ScanToString(TTree &tree, const char *varexp, const char *selection, const char *option, const char *redirectPath)
{
   auto *player = static_cast<TTreePlayer *>(tree.GetPlayer());
   FileRAII redirectFile{redirectPath};
   player->SetScanRedirect(true);
   player->SetScanFileName(redirectFile.fPath);
   tree.Scan(varexp, selection, option);
   player->SetScanRedirect(false);

   std::ifstream redirectStream(redirectFile.fPath);
   std::stringstream redirectOutput;
   redirectOutput << redirectStream.rdbuf();
   return redirectOutput.str();
}

TTree *MakeTree()
{
   auto *tree = new TTree("t", "t");
   static char name[32];
   static int val;
   tree->Branch("name", name, "name/C");
   tree->Branch("val", &val, "val/I");
   const char *names[] = {"alpha", "b", "gammalong", "de"};
   const int vals[] = {1, 22, 333, 4};
   for (int i = 0; i < 4; ++i) {
      strcpy(name, names[i]);
      val = vals[i];
      tree->Fill();
   }
   return tree;
}

} // namespace

TEST(TTreePlayerScan, LeftJustifiedColumns)
{
   std::unique_ptr<TTree> tree{MakeTree()};

   const std::string out = ScanToString(*tree, "name:val", "", "col=-20s:-8d", "gh11786_left.txt");

   const std::string expected = "**********************************************\n"
                                "*    Row   * name                 * val      *\n"
                                "**********************************************\n"
                                "*        0 * alpha                * 1        *\n"
                                "*        1 * b                    * 22       *\n"
                                "*        2 * gammalong            * 333      *\n"
                                "*        3 * de                   * 4        *\n"
                                "**********************************************\n";

   EXPECT_EQ(out, expected);
}

TEST(TTreePlayerScan, RightJustifiedColumnsStillWork)
{
   std::unique_ptr<TTree> tree{MakeTree()};

   const std::string out = ScanToString(*tree, "name:val", "", "col=20s:8d", "gh11786_right.txt");

   const std::string expected = "**********************************************\n"
                                "*    Row   *                 name *      val *\n"
                                "**********************************************\n"
                                "*        0 *                alpha *        1 *\n"
                                "*        1 *                    b *       22 *\n"
                                "*        2 *            gammalong *      333 *\n"
                                "*        3 *                   de *        4 *\n"
                                "**********************************************\n";

   EXPECT_EQ(out, expected);
}

// A default (right-justified) column sandwiched between left-justified ones:
// this exercises the case where the '-' used to abort parsing mid-list.
TEST(TTreePlayerScan, MixedJustification)
{
   std::unique_ptr<TTree> tree{MakeTree()};

   const std::string out = ScanToString(*tree, "name:name:val", "", "col=-15s::-6d", "gh11786_mixed.txt");

   const std::string expected = "***************************************************\n"
                                "*    Row   * name            *      name * val    *\n"
                                "***************************************************\n"
                                "*        0 * alpha           *     alpha * 1      *\n"
                                "*        1 * b               *         b * 22     *\n"
                                "*        2 * gammalong       * gammalong * 333    *\n"
                                "*        3 * de              *        de * 4      *\n"
                                "***************************************************\n";

   EXPECT_EQ(out, expected);
}
