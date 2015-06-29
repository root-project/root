//script to test TRFunction
#include<TRInterface.h>
#include<TMath.h>
double funv(TVectorD v)
{
   return v[0] * v[1];
}

void funm(TMatrixD m)
{
   m.Print();
}

void funs(TString s)
{
   std::cout << "hello " << s.Data() << std::endl;
}

//this prototype dont work because argument should be
//an object to pass an array.
Double_t fun3(Double_t *x, Double_t *par)
{
   return x[0] * par[0];
}

Double_t fun4(Double_t x)
{
   return x * 3;;
}


void Functions(TString type = "Import")
{
   ROOT::R::TRInterface &r = ROOT::R::TRInterface::Instance();
   r.SetVerbose(kFALSE);
   if (type == "Export") {
      std::cout << "-----------------------" << std::endl;
      std::cout << "Testing FunctionExport" << std::endl;
      std::cout << "-----------------------" << std::endl;
      r["funv"] << ROOT::R::TRFunctionExport(funv);
      r << "print(funv(c(2,3)))";

      r["funm"] << ROOT::R::TRFunctionExport(funm);
      r << "cat(funm(matrix(c(1,2,3,4),2,2)))";

      r["funs"] << ROOT::R::TRFunctionExport(funs);

      r << "cat(funs('ROOTR'))";

      r["DiLog"] << ROOT::R::TRFunctionExport(TMath::DiLog);
      r << "print(DiLog(2))";

      r << "x <- seq(0,10,0.01)";
      r << "y <- NULL ";
      r << "for(i in seq(along=x)) { \
		y <- c(y,DiLog(i)) \
	     }";

      ROOT::R::TRFunctionExport f4;
      f4.SetFunction(fun4);
      r["fun4"] << f4;
      r << "print(fun4(1))";
   } else {
      std::cout << "-----------------------" << std::endl;
      std::cout << "Testing FunctionImport " << std::endl;
      std::cout << "-----------------------" << std::endl;
      ROOT::R::TRFunctionImport print("print");
      print("Hola");
      
      ROOT::R::TRFunctionImport c("c");
      ROOT::R::TRFunctionImport require("require");
      ROOT::R::TRFunctionImport plot("plot");
      ROOT::R::TRFunctionImport devoff("dev.off");
      
      ROOT::R::TRFunctionImport rsin("sin");
      ROOT::R::TRFunctionImport options("options");
      ROOT::R::TRFunctionImport x11("x11");
      options("device='x11'");
      
      ROOT::R::TRObjectProxy vector=c(1,2,3,4);
      x11();
      plot(rsin);
//      devoff();
      print(vector);
      
      require("C50");
   }
}
