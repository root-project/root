#include <TTree.h>

int hadd_check_changed_comp_tree(const char *fname, const char *fname2, int expectedComp) {
  struct DeferRemove {
    const char *fname, *fname2;
    ~DeferRemove() {
      gSystem->Unlink(fname);
      gSystem->Unlink(fname2);
    }
  } defer { fname, fname2 };
  std::unique_ptr<TFile> file { TFile::Open(fname, "READ") };
  TTree *tree = file->Get<TTree>("t");
  if (!tree) return 1;
  TBranch *branch = tree->GetBranch("x");
  if (!branch) return 2;
  return (branch->GetCompressionSettings() == expectedComp) ? 0 : 3;
}
