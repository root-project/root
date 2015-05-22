#include <cassert>
#include <vector>
#include <algorithm>
#include <stdlib.h>
#include <TROOT.h>
#include <TSystem.h>
#include <TMath.h>
#include <TBenchmark.h>
#include <TSystem.h>
#include <TApplication.h>
#include <TFormula.h>
#include <v5/TFormula.h>
#include <TRandom.h>
#include <iostream>
#include "TFormulaParsingTests.h"

using namespace std;


template<typename F, typename S, typename T>
struct Triple
{
   F first;
   S second;
   T third;
   Triple() {}
   Triple(const F &f, const S &s, const T &t): first(f),second(s),third(t) {}
};
typedef pair<TString,Double_t> SDpair;

TString testFormula; 
SDpair *testVars; 
SDpair *testParams;
Int_t Nparams;
Int_t Nvars;


class TFormulaTests : public TFormula
{
protected:
  
public:
               TFormulaTests(TString name, TString formula): TFormula(name,formula){}
   virtual     ~TFormulaTests(){}
   Bool_t      ParserNew();
   Bool_t      GetVarVal();
   Bool_t      GetParVal();
   Bool_t      AddVar();
   Bool_t      SetVar();
   Bool_t      AddVars();
   Bool_t      SetVars();
   Bool_t      SetPar1();
   Bool_t      SetPar2();
   Bool_t      SetPars1();
   Bool_t      SetPars2();
   Bool_t      Eval();
   Bool_t      Stress(Int_t n = 10000);

   Bool_t      Parser();

   

};

Bool_t TFormulaTests::GetVarVal()
{
   
   Bool_t successful = true;
   for(Int_t i = 0; i < Nvars; ++i)
   {
      SDpair var = testVars[i];
      if(var.second != GetVariable(var.first) || var.second != fVars[var.first].fValue)
      {
         printf("fail:%s\t%lf\n",var.first.Data(), var.second);
         successful = false;
      }
   }
   return successful;
}
Bool_t TFormulaTests::GetParVal()
{
   Bool_t successful = true;
   
   for(Int_t i = 0 ; i < Nparams; ++i)
   {
      SDpair param = testParams[i];
      if(param.second != GetParameter(param.first) )//|| param.second != fParams[param.first].fValue)
      {
         printf("fail:%s\t%lf\n",param.first.Data(), param.second);
         successful = false;
      }
   }

   return successful;
}
Bool_t TFormulaTests::AddVar()
{
   Bool_t successful = true;
   TFormula *test = new TFormula("AddVarTest","var1 + var2 + var3");
   TString vars[] = {"var1","var2","var3"};
   for(Int_t i = 0; i < 3; ++i)
   {
      test->AddVariable(vars[i],i);
   }
   for(Int_t i = 0; i < 3; ++i)
   {
      if(test->GetVariable(vars[i]) != (Double_t)i)
      {
         printf("fail:%s\n",vars[i].Data());
         successful = false;
      }
   }

   return successful;
}
Bool_t TFormulaTests::SetVar()
{
   Bool_t successful = true;
   TFormula *test = new TFormula("SetVarTest","var1 + var2 + var3");
   SDpair *vars = new SDpair[3];
   vars[0] = SDpair("var",1);
   vars[1] = SDpair("var2",2);
   vars[2] = SDpair("var3",3); 
   test->SetVariables(vars, 3);
   for(Int_t i = 0 ; i < 3; ++i)
   {
      SDpair var = vars[i];
      test->SetVariable(var.first,var.second + 100.0);
   }
   for(Int_t i = 0 ; i < 3; ++i)
   {
      SDpair var = vars[i];
      if(test->GetVariable(var.first) != var.second + 100.0)
      {
         printf("fail:%s\t%lf\n",var.first.Data(),var.second);
         successful = false;
      }
   }   

   return successful;
}
Bool_t TFormulaTests::AddVars()
{
   Bool_t successful = true;
   TFormula *test = new TFormula("AddVarsTest","var1 + var2 + var3");
   TString vars[] = {"var1","var2","var3"};
   // SDpair *vars = new SDpair[3];
   // vars[0] = SDpair("var",1);
   // vars[1] = SDpair("var2",2);
   // vars[2] = SDpair("var3",3); 
   test->AddVariables(vars, 3);
   for(Int_t i = 0; i < 3; ++i)
   {
      if(test->GetVariable(vars[i]) != 0.0)
      {
         printf("fail:%s\t%lf\n",vars[i].Data(),test->GetVariable(vars[i]));
         successful = false;
      }
   }   

   return successful;
}
Bool_t TFormulaTests::SetVars()
{
   Bool_t successful = true;
   TFormula *test = new TFormula("SetVarsTest","var1 + var2 + var3");
   SDpair *vars = new SDpair[3];
   vars[0] = SDpair("var",1);
   vars[1] = SDpair("var2",2);
   vars[2] = SDpair("var3",3); 
   test->SetVariables(vars, 3);
   for(Int_t i = 0 ; i < 3; ++i)
   {
      SDpair v = vars[i];
      v.second += 100.0;
   }
   test->SetVariables(vars,3);
   for(Int_t i = 0; i < 3; ++i)
   {
      SDpair v = vars[i];
      if(test->GetVariable(v.first) != v.second)
      {
         printf("fail:%s\t%lf\n",v.first.Data(),v.second);
         successful = false;
      }
   }
   return successful;
}

Bool_t TFormulaTests::SetPar1()
{
   Bool_t successful = true;
   TFormula *test = new TFormula("SetParTest1","[0] + [te_st]");

   SDpair *params = new SDpair[2];
   params[0] = SDpair("0",1);
   params[1] = SDpair("te_st",2);
   for(Int_t i = 0 ; i < 2; ++i)
   {
      SDpair p = params[i];
      test->SetParameter(p.first,p.second);
   }
   for(Int_t i = 0 ; i < 2; ++i)
   {
      SDpair p = params[i];
      if(test->GetParameter(p.first) != p.second)
      {
         printf("fail:%s\t%lf\n",p.first.Data(), p.second);
         successful = false;
      }
   }

   return successful;
}
Bool_t TFormulaTests::SetPar2()
{
   Bool_t successful = true;
   TFormula *test = new TFormula("SetParTest2","[0] + [1]+[2]");
   Double_t params[] = { 123,456,789 };
   for(Int_t i = 0; i < 3; ++i)
   {
      test->SetParameter(i,params[i]);
   }
   for(Int_t i = 0; i < 3; ++i)
   {
      if(test->GetParameter(i) != params[i])
      {
         printf("fail:%d\t%lf\n",i, params[i]);
         successful = false;
      }
   }

   return successful;
}
Bool_t TFormulaTests::SetPars2()
{
   Bool_t successful = true;
   TFormula *test = new TFormula("SetParsTest2","[0] + [1]+[2]");
   Double_t params[] = { 123,456,789};
   test->SetParameters(params);
   for(Int_t i = 0; i < 3; ++i)
   {
      if(test->GetParameter(i) != params[i])
      {
         printf("fail:%d\t%lf\n",i, params[i]);
         successful = false;
      }
   }

   return successful;
}
Bool_t TFormulaTests::SetPars1()
{
   Bool_t successful = true;
   TFormula *test = new TFormula("SetParsTest1","[0] + [te_st]");
   SDpair *params = new SDpair[2];
   params[0] = SDpair("0",1);
   params[1] = SDpair("te_st",2);
   test->SetParameter(params[0].first,params[0].second);
   test->SetParameter(params[1].first,params[1].second);
   for(Int_t i = 0 ; i < 2; ++i)
   {
      SDpair p = params[i];
      if(test->GetParameter(p.first) != p.second)
      {
         printf("fail:%s\t%lf\n",p.first.Data(), p.second);
         successful = false;
      }
   }

   return successful;
}
Bool_t TFormulaTests::Eval()
{
   Bool_t successful = true;
   TFormula *test = new TFormula("EvalTest","y^z^t - sin(cos(x))");
   Double_t x,y,z,t;
   x = TMath::Pi();
   y = 2.0;
   z = 3.0;
   t = 4.0;

   Double_t result = TMath::Power(y,TMath::Power(z,t)) - TMath::Sin(TMath::Cos(x));
   Double_t TFresult = test->Eval(x,y,z,t);

   if(!TMath::AreEqualAbs(result,TFresult,0.00001))
   {
      printf("TF:%lf\tTMath::%lf\n",TFresult,result);
      successful = false;
   }

   return successful;
}

Bool_t TFormulaTests::ParserNew()
{
   //x_1- [test]^(TMath::Sin(pi*var*TMath::DegToRad())) - var1pol2(0) + gausn(0)*ylandau(0)+zexpo(10)

   Bool_t successful = true;
   Triple<TString,TString,Int_t> *funcs = new Triple<TString,TString,Int_t>[24]; //name,body,Nargs
   funcs[0] = Triple<TString,TString,Int_t>("x_1","",0);
   funcs[1] = Triple<TString,TString,Int_t>("var","",0);
   funcs[2] = Triple<TString,TString,Int_t>("var1","",0);
   funcs[3] = Triple<TString,TString,Int_t>("test","",0);
   funcs[4] = Triple<TString,TString,Int_t>("0","",0);
   funcs[5] = Triple<TString,TString,Int_t>("1","",0);
   funcs[6] = Triple<TString,TString,Int_t>("pi","",0);
   funcs[7] = Triple<TString,TString,Int_t>("2","",0);
   funcs[8] = Triple<TString,TString,Int_t>("10","",0);
   funcs[9] = Triple<TString,TString,Int_t>("11","",0);
   funcs[10] = Triple<TString,TString,Int_t>("x","",0);
   funcs[11] = Triple<TString,TString,Int_t>("y","",0);
   funcs[12] = Triple<TString,TString,Int_t>("z","",0);
   funcs[13] = Triple<TString,TString,Int_t>("false","",0);
   funcs[14] = Triple<TString,TString,Int_t>("pow","{var1},2",2);
   funcs[15] = Triple<TString,TString,Int_t>("pow","{var1},1",2);
   funcs[16] = Triple<TString,TString,Int_t>("pow","{[test]},(TMath::Sin({pi}*{var}*TMath::DegToRad()))",2);
   funcs[17] = Triple<TString,TString,Int_t>("exp","{[10]}+{[11]}*{z}",1);
   funcs[18] = Triple<TString,TString,Int_t>("TMath::Landau","{y},{[0]},{[1]},{false}",4);
   funcs[19] = Triple<TString,TString,Int_t>("sqrt","2*{pi}",1);
   funcs[20] = Triple<TString,TString,Int_t>("exp","-0.5*pow((({x}-{[1]})/{[2]}),2)",1);
   funcs[21] = Triple<TString,TString,Int_t>("pow","(({x}-{[1]})/{[2]}),2",2);
   funcs[22] = Triple<TString,TString,Int_t>("TMath::Sin","{pi}*{var}*TMath::DegToRad()",1);
   funcs[23] = Triple<TString,TString,Int_t>("TMath::DegToRad","",0);
   TString vars[] = {"x_1","var","var1"};
   TString params[] = {"test","0","1","2"};
   for(Int_t i = 0; i < 24 ; ++i)
   {
      Triple<TString,TString,Int_t> func = funcs[i];
      TFormulaFunction f(func.first,func.second,func.third);
      if(find(fFuncs.begin(),fFuncs.end(),f) == fFuncs.end())
      {
         printf("fail:%s\t%s\n",func.first.Data(),func.second.Data());
         successful = false;
      }
   }
   for(Int_t i = 0 ; i < 3 ;++i)
   {
      TString var = vars[i];
      if(fVars.find(var) == fVars.end())
      {
         printf("fail:%s\n",var.Data());
         successful = false;
      }
   }
   for(Int_t i = 0; i < 4; ++i)
   {
      TString param = params[i];
      if(fParams.find(param) == fParams.end())
      {   
         printf("fail:%s\n",param.Data());
         successful = false;
      }
   }

   return successful;
}

Bool_t TFormulaTests::Stress(Int_t n)
{
    
#ifdef OLD
   TString formula = "x";
   SDpair *vars = new SDpair[n];
   SDpair *params = new SDpair[n*5];
   gBenchmark->Start("TFormula Stress Total Time");
   for(Int_t i = 0; i < n ; ++i)
   {
      vars[i] = SDpair(TString::Format("x%d",i),i+1);
      params[i] = SDpair(TString::Format("p%d",i),i+1);
      formula.Append(TString::Format("+ %s + %s*gausn(0) - [%d]",vars[i].first.Data(),vars[i].first.Data(),i));
      //cout << formula.Data() << endl;
   }
   for(Int_t i = n; i < n*5; ++i)
   {
      params[i] = SDpair(TString::Format("p%d",i),i+1);
      formula.Append(TString::Format("*[%d]",i));
      //cout << formula.Data() << endl;
   }
#else
   TString formula = "x";
   //SDpair *vars = new SDpair[n];
   SDpair *params = new SDpair[n*5];
   std::vector<double> parv;
   gBenchmark->Start("TFormula Stress Total Time");
   int i0 = 0;
   for(Int_t i = 0; i < n ; i+=5)
   {
      // vars[i] = SDpair(TString::Format("x",i),i+1);
      for (int j = 0; j < 5; ++j) {
         double val = 2.0;
         params[i+j] = SDpair(TString::Format("p%d",i+j),val);
         //cout << "set parameter " << i+j << " value " << j+1 << endl;
         parv.push_back(val);
      }
      formula.Append(TString::Format("*[%d]+ y*[%d]*[%d] + z*gausn(%d)*[%d] - [%d]*t",i,i+1,i+1,i+2,i+3,i+4));
      //formula.Append(TString::Format("*[%d]+ y*[%d]*[%d] + z*gausn(%d) - [%d]*t",i,i+1,i+1,i+2,i+5));
      //cout << formula.Data() << endl;
      i0 = i; 
   }  
   i0 += 5;
   for(Int_t i = i0; i < n*5; ++i)
   {
      double val = i;
      params[i] = SDpair(TString::Format("p%d",i),val);
      //cout << "set parameter " << i << " value " << i+1 << endl;
      parv.push_back(val);
      formula.Append(TString::Format("+[%d]",i));
      //cout << formula.Data() << endl;
   }
#endif
   // new version with only 4 variables but n-parameters

   gBenchmark->Start(TString::Format("TFormula Initialization with %d variables and %d parameters\n",n,n*5));
   TFormula *test = new TFormula("TFStressTest",formula);
   gBenchmark->Show(TString::Format("TFormula Initialization with %d variables and %d parameters\n",n,n*5));
   // gBenchmark->Start(TString::Format("Adding %d variables\n",n));
   // test->AddVariables(vars,n);
   // gBenchmark->Show(TString::Format("Adding %d variables\n",n));
   gBenchmark->Start(TString::Format("Setting %d parameters\n",n*5));
   for (int i = 0; i < n*5; ++i) 
      test->SetParameter(params[i].first, params[i].second);
   gBenchmark->Show(TString::Format("Setting %d parameters\n",n*5));

   int neval = n*1000;
   gBenchmark->Start(TString::Format("%d Evaluations\n",neval));
   double xx[4] = {3,3,3,3};
   TRandom rndm;
   TStopwatch w; w.Start();
   double s = 0; 
   for(Int_t i = 0; i < neval; ++i)
   {
      if (i > 0) rndm.RndmArray(4,xx);
      double f = test->EvalPar(xx,0);
      if (i == 0) printf(" f = %20.16g \n",f);
      if (TMath::Even(i) ) s += f; 
      else s -= f;          
   }
   printf("Evaluation time :\t");
   w.Print();
   std::cout << "result = " <<  s << std::endl; 
   gBenchmark->Show(TString::Format("%d Evaluations\n",neval));
   //test->Print("v");
   gBenchmark->Show("TFormula Stress Total Time");


   std::cout << "\n\n Testing old TFormula \n" << endl;

   ROOT::v5::TFormula::SetMaxima(5000,5000,5000);

   gBenchmark->Start(TString::Format("ROOT::v5::TFormula Initialization with %d variables and %d parameters\n",n,n*5));
   ROOT::v5::TFormula *testOld = new ROOT::v5::TFormula("TFStressTestOld",formula);
   gBenchmark->Show(TString::Format("ROOT::v5::TFormula Initialization with %d variables and %d parameters\n",n,n*5));
   // gBenchmark->Start(TString::Format("Adding %d variables\n",n));
   // test->AddVariables(vars,n);
   // gBenchmark->Show(TString::Format("Adding %d variables\n",n));
   gBenchmark->Start(TString::Format("ROOT::v5::TFormula: Setting %d parameters\n",n*5));
   testOld->SetParameters(&parv[0]);
   gBenchmark->Show( TString::Format("ROOT::v5::TFormula: Setting %d parameters\n",n*5));

   gBenchmark->Start(TString::Format("ROOT::v5::TFormula: %d Evaluations\n",neval));
   TRandom rndm2;
   std::cout << "start evaluatuons  " << std::endl;
   s = 0;
   w.Start();
   double xx2[4] = {3,3,3,3};
   for(Int_t i = 0; i < neval; ++i)
   {
      if (i > 0) rndm2.RndmArray(4,xx2);      
      double f = testOld->EvalPar(xx2,0);
      if (i == 0) printf(" f = %20.16g \n",f);
      if (TMath::Even(i) ) s += f; 
      else s -= f;          
   }
   printf("Evaluation time :\t");
   w.Print();
   std::cout << "result = " <<  s << std::endl; 
   gBenchmark->Show(TString::Format("ROOT::v5::TFormula: %d Evaluations\n",neval));
   //testOld->Print("v");
   gBenchmark->Show("ROOT::v5::TFormula Stress Total Time");

   
   return true;
}


bool TFormulaTests::Parser() {
   std::cout << "Test parsing of expression compatible with old TFormula" << std::endl;
   TFormulaParsingTests t;
   int nfailed = t.runTests();
   if (nfailed != 0) {
      std::cout << "ERROR - Parsing test of TFormula failed - number of failures is " << nfailed << std::endl;
      return false;
   }
   return true; 
}
   



int main(int argc, char **argv)
{
   printf("strting .....\n");

   TApplication theApp("App", &argc, argv);
   gBenchmark = new TBenchmark();
   Int_t n = 200;
   if(argc > 1) 
      n = TString(argv[1]).Atoi();
   printf("************************************************\n");
   printf("================TFormula Tests===============\n");
   printf("************************************************\n");



   testFormula = "x- [test]^(TMath::Sin(pi*var*TMath::DegToRad())) - var1pol2(0) + gausn(0)*landau(0)+expo(10)";
   //testFormula = "x - [test]^(TMath::Sin(pi*y*TMath::DegToRad())) - pol2(0) + gausn(0)*landau(0)+expo(10)";

   printf("creating formula .....\n");
   TFormulaTests * test = new TFormulaTests("TFtests","");
   test->AddVariable("var",0);
   test->AddVariable("var1",0);
   test->Compile(testFormula);
      

#ifdef LATER

   Nparams = 6;
   Nvars = 6;
   testParams = new SDpair[Nparams];
   testVars = new SDpair[Nvars];
   testParams[0] = SDpair("test",123);
   testParams[1] = SDpair("0",456);
   testParams[2] = SDpair("1",789);
   testParams[3] = SDpair("2",123);
   testParams[4] = SDpair("10",123);
   testParams[5] = SDpair("11",456);

   testVars[0] = SDpair("x_1",123);
   testVars[1] = SDpair("var",456);
   testVars[2] = SDpair("var1",789);
   testVars[3] = SDpair("x",123);
   testVars[4] = SDpair("y",456);
   testVars[5] = SDpair("z",123);

   
   test->AddVariables(testVars,Nvars);
   test->SetParameters(testParams);

   printf("Parser test:%s\n",(test->ParserNew() ? "PASSED" : "FAILED"));
   printf("GetVariableValue test:%s\n",(test->GetVarVal() ? "PASSED" : "FAILED")); 
   printf("GetParameterValue test:%s\n",(test->GetParVal() ? "PASSED" : "FAILED"));   
   printf("AddVariable test:%s\n",(test->AddVar() ? "PASSED" : "FAILED"));
   printf("AddVariables test:%s\n",(test->AddVars() ? "PASSED" : "FAILED"));
   printf("SetVariable test:%s\n",(test->SetVar() ? "PASSED" : "FAILED"));
   printf("SetVariables test:%s\n",(test->SetVars() ? "PASSED" : "FAILED"));
   printf("SetParameter1 test:%s\n",(test->SetPar1() ? "PASSED" : "FAILED"));
   printf("SetParameter2 test:%s\n",(test->SetPar2() ? "PASSED" : "FAILED"));
   printf("SetParameters1 test:%s\n",(test->SetPars1() ? "PASSED" : "FAILED"));
   printf("SetParameters2 test:%s\n",(test->SetPars2() ? "PASSED" : "FAILED"));
   printf("Eval test:%s\n",(test->Eval() ? "PASSED" : "FAILED"));
#endif
   printf("Stress test:%s\n",(test->Stress(n) ? "PASSED" : "FAILED"));
   printf("Parsing test:%s\n",(test->Parser() ? "PASSED" : "FAILED"));

   return 0;
}





