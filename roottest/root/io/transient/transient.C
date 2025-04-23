#include "TObject.h"
#include "TCollection.h"
#include "TClass.h"
#include "TObjArray.h"

class MyClass : public TObject {
public:
   TObjArray obj;
   TIter *val;
   MyClass() : val(new TIter(&obj)) {}
   ClassDef(MyClass,1);
};

void transient() {
   TClass *cl = gROOT->GetClass("MyClass");
   new TFile("
   cl->GetStreamerInfo();
}
