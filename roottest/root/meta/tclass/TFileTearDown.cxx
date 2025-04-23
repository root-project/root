#include <TH1.h>
#include <TFile.h>
#include <TRint.h>

int main() {
  TRint *app = new TRint("app", 0, NULL);
  TFile *f=new TFile("foo.root", "RECREATE");
  TH1 *foo = new TH1D("foo","foo",10,0,10);
  (void)app;
  (void)f;
  (void)foo;
  return 0;
}
