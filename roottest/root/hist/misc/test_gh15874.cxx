/*
This is a test for https://github.com/root-project/root/issues/15874
The input file was created with ROOT version 6.30.04, so that
https://github.com/root-project/root/pull/8546 was not there yet. Practically
this means that THnSparseL is a typedef to THnSparseT<TArrayL>, whereas
afterwards it became a typedef to THnSparseT<TArrayL64>. The file is created
with the following program:

```cpp
#include <TFile.h>
#include <THnSparse.h>

void write_thnsparse_l()
{
   std::unique_ptr<TFile> file{TFile::Open("hist.root", "RECREATE")};
   int bins[2] = {10, 20};
   double xmin[2] = {0., -5.};
   double xmax[2] = {10., 5.};
   THnSparseL hist{"hist", "hist", 2, bins, xmin, xmax};
   file->WriteObject(&hist, "hist");
}

int main(){
    write_thnsparse_l();
}
```
*/
#include "gtest/gtest.h"

#include "TClass.h"
#include "TFile.h"
#include "THnSparse.h"

TEST(GH15874, Regression) {
  TFile f{"test_gh15874.root"};
  auto *hist = f.Get<THnSparseL>("hist");
  ASSERT_TRUE(hist);

  auto *expectedClass = TClass::GetClass("THnSparseT<TArrayL64>");
  ASSERT_EQ(hist->IsA(), expectedClass);
}
