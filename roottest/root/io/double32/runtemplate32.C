#include "runtemplate32.h"
#include "longExample.h"

#ifdef __MAKECINT__
#pragma link C++ class WithDouble+;
#pragma link C++ class MyVector<Double32_t>+;
#pragma link C++ class MyVector<float>+;
// #pragma link C++ class MyVector<double>+;
#endif

#include "TClass.h"
#include "TStreamerInfo.h"
#include "TROOT.h"
#include "TRealData.h"
#include "Riostream.h"
#include "TDataMember.h"
#include "TFile.h"

int runtemplate32 ()
{
   gROOT->ProcessLine(".L other.C+");
   gROOT->GetClass("WithDouble")->GetStreamerInfo()->ls();
   gROOT->GetClass("MyVector<double>")->GetStreamerInfo()->ls();
   gROOT->GetClass("MyVector<Double32_t>")->GetStreamerInfo()->ls();
   gROOT->GetClass("Contains")->GetStreamerInfo()->ls();
   gROOT->GetClass("Contains")->GetListOfRealData()->ls("noaddr");

   TRealData *r = (TRealData*)gROOT->GetClass("Contains")->GetListOfRealData()->At(1);
   cout << "The following should be a Double32_t: " << r->GetDataMember()->GetTypeName() << endl;

   TFile *f = new TFile("double32.root", "RECREATE");
   Contains *c = new Contains;
   f->WriteObject(c,"myobj");
   delete f;

   TClass *cl = TClass::GetClass("m02<Double32_t>");
   cl->GetStreamerInfo()->ls("noaddr");

   return 0;
}
