//script to test TRFunction
#include<TRInterface.h>

ROOT::R::TRInterface &r = ROOT::R::TRInterface::Instance();

void Object()
{
//      r.SetVerbose(kFALSE);
      ROOT::R::TRFunctionImport print("print");
      
      ROOT::R::TRFunctionImport c("c");
      ROOT::R::TRFunctionImport require("require");
      ROOT::R::TRFunctionImport head("head");
      ROOT::R::TRFunctionImport attributes("attributes");
       
      ROOT::R::TRObject vector=c(1,2,3,4);
      ROOT::R::TRObject v=vector;
//      v.SetAttribute("test","test");
      v.Attr["attvector"]=v;
      v.Attr["str"]="something";
      r["v"]=v;
      
      v=v.Attr["str"];
      r["v"]=v;
      
      print(attributes(v));
      
}
