#include "TFile.h"
#include "TClass.h"

class MyClass {
public:
   Int_t fValue;
};

void execMissingCheckSum()
{
  TFile *f = new TFile("missingCheckSum.root");
  if (!f->Get("obj")) printf("Error: could not read the object (1)\n");
  ((TObjArray*)TClass::GetClass("MyClass")->GetStreamerInfos())->RemoveAt(1);
  if (!f->Get("obj")) printf("Error: could not read the object (2)\n");
}

