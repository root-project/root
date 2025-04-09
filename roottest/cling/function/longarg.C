#include "TNamed.h"
#include "TDataMember.h"
#include "TMethodCall.h"
#include "TClass.h"
#include <iostream>
using namespace std;
 
void longarg()
{
   TNamed* name = new TNamed("name", "title");
   cout << name->GetName() << endl;
   TDataMember* thevar = name->IsA()->GetDataMember("fName");
   TMethodCall* setter = thevar->SetterMethod(name->IsA());

   TString param ("\"start \"");
   for(Int_t i = 0; i < 1011; ++i){
      param.Insert(param.Last('\"'), "1");
   }
   param.Insert(param.Last('\"'), " end");

   cout << param.Length() << endl;
   setter->Execute(name, param.Data());
   cout << name->GetName() << endl;

   param.Insert(param.Last('\"'), "2");
   cout << param.Length() << endl;
   setter->Execute(name, param.Data());
   cout << name->GetName() << endl;
}
