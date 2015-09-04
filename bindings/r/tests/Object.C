//script to test TRFunction
#include<TRInterface.h>

ROOT::R::TRInterface &r = ROOT::R::TRInterface::Instance();

void Object()
{
//      r.SetVerbose(kFALSE);
      ROOT::R::TRFunctionImport print("print");
      
      ROOT::R::TRFunctionImport c("c");
      ROOT::R::TRFunctionImport cat("cat");
      ROOT::R::TRFunctionImport require("require");
      ROOT::R::TRFunctionImport head("head");
      ROOT::R::TRFunctionImport attributes("attributes");
       
      ROOT::R::TRObject vector=c(1,2,3,4);
      ROOT::R::TRObject v=vector;
      v.SetAttribute("Size",4);
      v.SetAttribute("Name","TestVector");
//      v.SetAttribute("NameNull",NULL);
      
      int s=v.GetAttribute("Size");
      TString name=v.GetAttribute("Name");
      
      
      print(attributes(v));
      print(cat("ROOT:",s));
      print(cat("ROOT:",name));
}
