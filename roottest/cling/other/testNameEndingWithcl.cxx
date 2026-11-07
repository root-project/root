// Regression test for https://github.com/root-project/root/issues/23493
#include <TROOT.h>

int main() {
  // this causes ROOT to be initialized
  gROOT;
  return 0;
}
