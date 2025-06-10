#include <TObject.h>
#include <TFile.h>
#include <TRandom.h>

class testClass : public TObject
{
public:
  testClass() { }
  ~testClass() override { }

  Float16_t fMember[100];

  ClassDefOverride(testClass, 1)
};

void runfloat16()
{
  testClass* t = new testClass;

  for (Int_t i=0; i<100; i++)
    t->fMember[i] = gRandom->Rndm();

  TFile::Open("file.root", "RECREATE");
  t->Write();
  gFile->Close();

  t = nullptr;
  TFile::Open("file.root");
  gFile->GetObject("testClass", t);
}


