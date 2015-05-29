#include<TRInterface.h>
#include<vector>
//script to test RExport a TRobjectProxy
void Proxy()
{
  ROOT::R::TRInterface *gR=ROOT::R::TRInterface::InstancePtr();
  gR->SetVerbose(kTRUE);
   //////////////////////////
   //Getting values from R //
   //////////////////////////
   std::cout << "======Getting values from R ======\n";
   TString  s = gR->ParseEval("'ROOTR'");

   TVectorD v = gR->ParseEval("c(1,2,3,4)");
   std::vector<Double_t> sv=gR->ParseEval("c(1.01,2,3,4)");
   TMatrixD m = gR->ParseEval("matrix(c(1,2,3,4),2,2)");

   std::cout<<"-----------------------------------"<<endl;
   std::cout << s << std::endl;
   std::cout<<"-----------------------------------"<<endl;
   v.Print();
   std::cout<<"-----------------------------------"<<endl;
   for(int i=0;i<sv.size();i++) std::cout<<sv[i]<<" "<<std::endl;
   std::cout<<"-----------------------------------"<<endl;
   m.Print();

   Double_t d = gR->ParseEval("1.1");
   Float_t  f = gR->ParseEval("0.1");
   Int_t    i = gR->ParseEval("1");
   
   std::cout << d << " " << f << " " << i << std::endl;

  ////////////////////////
  //Passing values to R  //
  /////////////////////////

   std::cout << "======Passing values to R ======\n";
   gR->Assign(s, "s");
   gR->Parse("print(s)");
   std::cout<<"-----------------------------------"<<endl;
   
   (*gR)["v"]=v;
   gR->Parse("print(v)");
   std::cout<<"-----------------------------------"<<endl;

   gR->Assign(sv, "sv");
   gR->Parse("print(sv)");
   std::cout<<"-----------------------------------"<<endl;

   gR->Assign(m, "m");
   gR->Parse("print(m)");
   std::cout<<"-----------------------------------"<<endl;

   gR->Assign(d, "d");
   gR->Parse("print(d)");
   std::cout<<"-----------------------------------"<<endl;

   gR->Assign(f, "f");
   gR->Parse("print(f)");
   std::cout<<"-----------------------------------"<<endl;

   gR->Assign(i, "i");
   gR->Parse("print(i)");

}
