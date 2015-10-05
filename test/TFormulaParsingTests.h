#include "TF1.h"
#include "TF2.h"
#include "TF3.h"
#include "TFormula.h"
#include "TGraph.h"
#include "Math/ChebyshevPol.h"

// test of tformula neeeded to be run


class TFormulaParsingTests {

   
bool verbose; 
std::vector<int> failedTests; 
public:

TFormulaParsingTests(bool _verbose = false) : verbose(_verbose) {}

bool test1() {
   // test composition of functions
   TF1 f1("f1","[0]+[1]*x*x");
   TF1 f2("f2","[0]+[1]*f1");

   f2.SetParameters(1,2,3,4);

   return (f2.Eval(2) == 39.);

}

bool test2() {

   TF1 f1("f1","[0]+[1]*x");
   TF1 f2("f2","[0]+[1]*x*f1");

   TF1 f3("f3",f2.GetExpFormula() ); 

   f3.SetParameters(1,2,3,4);

   return (f3.Eval(2) == 45.);

}


bool test3() {

   // still tets composition of functions
   TF1 f1("f1","gaus");
   TF1 f2("f2","[0]+[1]*x+f1");


   f2.SetParameters(10,2,5,2,1);

   f1.SetParameters(5,2,1); 

   return (f2.Eval(2) == (10. + 2*2 + f1.Eval(2)) );

}

bool test4() {

   // similar but with different name (it contains gaus)
   TF1 f1("fgaus","gaus");
   TF1 f2("f2","[0]+[1]*x+fgaus");


   f2.SetParameters(10,2,5,2,1);

   f1.SetParameters(5,2,1); 

   return (f2.Eval(2) == (10. + 2*2 + f1.Eval(2)) );

}
bool test5() {

   // similar but with different name (it contains gaus)
   TF1 f1("gausnfunc","gaus");
   TF1 f2("f2","[0]+[1]*x+gausnfunc");


   f2.SetParameters(10,2,5,2,1);

   f1.SetParameters(5,2,1); 

   return (f2.Eval(2) == (10. + 2*2 + f1.Eval(2)) );

}

bool test1a() {
   // this makes infinite loop
   // why re-using same name
   TF1 f1("f1","[0]+[1]*x*x");
   TF1 f2("f1","[0]+[1]*f1");
   return true;
}


bool test6() {
   // test linear function used in fitting
   bool ok = true; 
   double x[] = {1,2,3,4,5};
   double y[] = {1,4,7,9,10};
   TGraph g(5,x,y);

   int iret = g.Fit("x++1","Q");
   ok &= (iret == 0);
   iret = g.Fit("1++x","Q");
   return iret == 0; 
}

bool test7() {
   // test copying and deleting of linear functions
   TF1 * f1 = new TF1("f1","1++x");
   if (f1->GetNpar() != 2) return false;
   f1->SetParameters(2,3);
   if (f1->Eval(3) != 11) return  false;

   if (verbose) printf("Test7: test linear part1 of function\n"); 
   TFormula * lin1 = (TFormula*) f1->GetLinearPart(1);
   assert (lin1);
   if (lin1->Eval(3) != 3) return false; 

   if (verbose) printf("Test7: test copying linear function\n"); 
   
   TF1 * f2 = new TF1(*f1);
   if (f2->Eval(3) != 11) return  false;

   if (verbose) printf("Test7: test linear part1 of copied function\n");
   if (!f2->IsLinear()) return false; 
   lin1 = (TFormula*) f2->GetLinearPart(1);
   assert (lin1);
   if (lin1->Eval(3) != 3) return false; 

   delete f1;

   if (verbose) printf("Test7: test cloning linear function\n"); 

   TF1 * f3 = (TF1*) f2->Clone("f3");
   if (f3->Eval(3) != 11) return  false;

   if (verbose) printf("Test7: test deleting the copied function\n"); 
   delete f2;

   if (verbose) printf("Test7: test linear part1 of cloned function\n"); 
   if (!f3->IsLinear()) return false; 
   lin1 = (TFormula*) f3->GetLinearPart(1);
   assert (lin1);
   if (verbose) printf("Test7: test evaluating linear part1 of cloned function\n"); 
   if (lin1->Eval(3) != 3) return false; 


   if (verbose) printf("Test7: test deleting the cloned function\n"); 
   delete f3; 
   return true; 
}

bool test8() {
   // test the operator ^
   bool ok = true;
   TFormula * f = 0;
   f = new TFormula("f","x^y");

   ok &= (f->Eval(2,3) == 8);
   delete f; 

   f = new TFormula("f","(x+[0])^y");
   f->SetParameter(0,1);
   ok &= (f->Eval(2,3) == 27);
   delete f; 

   f = new TFormula("f","sqrt(x+[0])^y");
   f->SetParameter(0,2);
   ok &= (f->Eval(2,3) == 8);
   delete f; 

   f = new TFormula("f","[0]/((x+2)^y)");
   f->SetParameter(0,27);
   ok &= (f->Eval(1,3) == 1);
   delete f; 
   
   f = new TFormula("f","[0]/((x+2)^(y+1))");
   f->SetParameter(0,27);
   ok &= (f->Eval(1,2) == 1);
   delete f; 

   // test also nested operators
   f = new TFormula("f","((x+1)^y)^z");
   ok &= (f->Eval(1,3,4) == 4096);
   delete f; 

   f = new TFormula("f","x^((y+1)^z)");
   ok &= (f->Eval(2,1,3) == 256);
   delete f;
   
   return ok;
   
}

bool test9() {
   // test the exponent notations in numbers
   bool ok = true;

   TFormula * f = 0;
   f = new TFormula("f","x+2.0e1");
   ok &= (f->Eval(1) == 21.);
   
   f = new TFormula("f","x*2.e-1");
   ok &= (f->Eval(10) == 2.);

   f = new TFormula("f","x*2.e+1");
   ok &= (f->Eval(0.1) == 2.);

   f = new TFormula("f","x*2E2");
   ok &= (f->Eval(0.01) == 2.);

   delete f;
   return ok;

   
}

bool test10() {
   // test the operator "? : "
   bool ok = true;
   TFormula * f = 0;
   f  = new TFormula("f","(x<0)?-x:x");
   ok &= (f->Eval(-2) == 2); 
   ok &= (f->Eval(2) == 2);

   f = new TFormula("f","(x<0)?x:pol2");
   f->SetParameters(1,2,3);
   ok &= (f->Eval(-2) == -2); 
   ok &= (f->Eval(2) == 1 + 2*2 + 2*2*3); 

   delete f;
   return ok; 
}

bool test11() {
   // test with ::
   bool ok = true;

   TFormula f1("f","ROOT::Math::normal_pdf(x,1,2)");
   TFormula f2("f","[0]+TMath::Gaus(x,2,1,true)");
   f2.SetParameter(0,1);
   ok &= ( (f1.Eval(2) +1. ) == f2.Eval(2) );
   return ok;
}
bool test12() { 
   // test parameters order 
   bool ok = true; 
   TFormula * f = 0; 
   f = new TFormula("f","[2] + [3]*x + [0]*x^2 + [1]*x^3");
   f->SetParameters(1,2,3,4);
   double result = 3+4*2+1*4+2*8;
   ok &= (f->Eval(2) == result); 
   f = new TFormula("f","[b] + [c]*x + [d]*x^2 + [a]*x^3");
   f->SetParameters(1,2,3,4);
   result = 2+3*2+4*4+1*8;
   ok &= (f->Eval(2) == result);
   // change a parameter value
   f->SetParName(2,"par2");
   ok &= (f->Eval(2) == result);
   return ok;
}

bool test13()  {
   // test GetExpFormula
   TFormula f("f","[2] + [0]*x + [1]*x*x");
   f.SetParameters(1,2,3);
   return (f.GetExpFormula() == TString("[p2]+[p0]*x+[p1]*x*x"));
}
bool test14()  {
   // test GetExpFormula
   TFormula f("f","[2] + [0]*x + [1]*x*x");
   f.SetParameters(1,2,3);
   return (f.GetExpFormula("P") == TString("3+1*x+2*x*x"));
}
bool test15()  {
   // test GetExpFormula
   TFormula f("f","[2] + [0]*x + [1]*x*x");
   f.SetParameters(1,2,3);
   return (f.GetExpFormula("CLING") == TString("p[2]+p[0]*x[0]+p[1]*x[0]*x[0] ") ); // need an extra white space
}
bool test16()  {
   // test GetExpFormula
   TFormula f("f","[2] + [0]*x + [1]*x*x");
   f.SetParameters(1,2,3);
   return (f.GetExpFormula("CLING P") == TString("3.000000+1.000000*x[0]+2.000000*x[0]*x[0] ") );
}

bool test17() {
   // test Eval for TF1
   TF1 * f1 = new TF1("f1","[0]*sin([1]*x)");
   f1->SetParameters(2,3);
   TF1 * f0 = new TF1("f0",[](double *x, double *p){ return p[0]*sin(p[1]*x[0]); },0,10,2);
   f0->SetParameters(2,3);
   bool ok = true; 
   ok &= (f1->Eval(1.5) == f0->Eval(1.5) );
   double xx[1] = {2.5};
   ok &= (f1->EvalPar(xx) == f0->Eval(2.5) );
   return ok;
}

bool test18() {
   // test Eval for TF2
   TF2 * f1 = new TF2("f2","[0]*sin([1]*x*y)");
   f1->SetParameters(2,3);
   TF2 * f0 = new TF2("f0",[](double *x, double *p){ return p[0]*sin(p[1]*x[0]*x[1]); },0,10,0,10,2);
   f0->SetParameters(2,3);
   bool ok = true; 
   ok &= (f1->Eval(1.5,2.5) == f0->Eval(1.5,2.5) );
   double par[2] = {3,4};
   double xx[2] = {0.8,1.6};
   ok &= (f1->EvalPar(xx,par) == f0->EvalPar(xx,par) );
   return ok; 
}

bool test19() {
   // test Eval for TF3
   TF3 * f1 = new TF3("f3","[0]*sin([1]*x*y*z)");
   f1->SetParameters(2,3);
   TF3 * f0 = new TF3("f0",[](double *x, double *p){ return p[0]*sin(p[1]*x[0]*x[1]*x[2]); },0,10,0,10,0,10,2);
   f0->SetParameters(2,3);
   bool ok = true; 
   ok &= (f1->Eval(1.5,2.5,3.5) == f0->Eval(1.5,2.5,3.5) );
   double par[2] = {3,4};
   double xx[3] = {0.8,1.6,2.2};
   ok &= (f1->EvalPar(xx,par) == f0->EvalPar(xx,par) );
   return ok; 
}

bool test20() {
   // test parameter order with more than 10 parameters
   TF2 f2("f2","xygaus+xygaus(5)+xygaus(10)+[offset]");
   double params[16] = {1,0,1,1,1, 2,-1,2,0,2, 2,1,3,-1,2, 10};
   f2.SetParameters(params);
   TF2 f0("f2",[](double *x, double *p){ return p[0]*TMath::Gaus(x[0],p[1],p[2])*TMath::Gaus(x[1],p[3],p[4]) +
            p[5]*TMath::Gaus(x[0],p[6],p[7])*TMath::Gaus(x[1],p[8],p[9]) + 
            p[10]*TMath::Gaus(x[0],p[11],p[12])*TMath::Gaus(x[1],p[13],p[14]) + p[15]; },
      -10,10,-10,10,16);
   double xx[2]={1,2};
   //printf(" difference = %f , value %f \n", f2.Eval(1,2) - f0.EvalPar(xx,params), f2.Eval(1,2) );
   return ( f2.Eval(1,2) == f0.EvalPar(xx,params) );
}

bool test21() {
   // test parsing polynomials (bug ROOT-7312)
   TFormula f("f","pol2+gaus(3)");
   f.SetParameters(1,2,3,1,0,1);
   TF1 f0("f0",[](double *x, double *p){ return p[0]+x[0]*p[1]+x[0]*x[0]*p[2]+p[3]*TMath::Gaus(x[0],p[4],p[5]); },0,1,6);
   f0.SetParameters(f.GetParameters() );
   return (f.Eval(2) == f0.Eval(2) );
}

bool test22() {
   // test chebyshev
   TF1 f("f","cheb10+[offset]");
   double p[12] = {1,1,1,1,1,1,1,1,1,1,1,10 };
   f.SetParameters(p);
   return (f.Eval(0.5) == ROOT::Math::ChebyshevN(10, 0.5, p ) + f.GetParameter("offset"));
}

bool test23() {
   // fix function compositions using pre-defined functions
   bool ok = true; 
   TF1 f1("f1","gaus");
   TF1 f2("f2","[0]+f1");
   TF1 f0("f0",[](double *x, double *p){ return p[0]+p[1]*TMath::Gaus(x[0],p[2],p[3]); },-3,3,4 );
   f2.SetParameters(10,1,0,1);
   f0.SetParameters(f2.GetParameters() );
   ok &= (f2.Eval(1) == f0.Eval(1) );

   TF1 f3("f3","f1+[0]");
   // param order should be the same
   f3.SetParameters( f2.GetParameters() );
   ok &= (f2.Eval(1) == f0.Eval(1) );
   return ok;
}

bool test24() {
   // test I/O for parameter ordering
   bool ok = true; 
   TF2 f("f","xygaus");
   f.SetParameters(10,0,1,-1,2);
   TF2 * f2 = (TF2*) f.Clone();
   ok &= ( f.Eval(1,1) == f2->Eval(1,1) );
   // test with copy
   TF2 f3(f);
   ok &= ( f.Eval(1,1) == f3.Eval(1,1) );
   return ok;
}

bool test25() {
   // fix parsing of operator^ (ROOT-7349)
   bool ok = true; 
   TF1 f1("f1","x^-2.5");
   ok &= (f1.Eval(3.) == TMath::Power(3,-2.5) );

   TF1 f2("f2","x^+2.5");
   //TF1 f3("f3","std::pow(x,2.5)");  // this needed to be fixed
   TF1 f3("f3","TMath::Power(x,2.5)");
   ok &= (f2.Eval(3.) == f3.Eval(3) );

   //cms test
   TF1 t1("t1","(x<190)?(-18.7813+(((2.49368+(10.3321/(x^0.881126)))*exp(-((x^-1.66603)/0.074916)))-(-17.5757*exp(-((x^-1464.26)/-7.94004e+06))))):(1.09984+(0.394544*exp(-(x/562.407))))");
   double x = 2;
   double y =(x<190)?(-18.7813+(((2.49368+(10.3321/(std::pow(x,0.881126))))*exp(-((std::pow(x,-1.66603))/0.074916)))-(-17.5757*exp(-((std::pow(x,-1464.26))/-7.94004e+06))))):(1.09984+(0.394544*exp(-(x/562.407))));
   ok &= (t1.Eval(2) == y );

   // tests with scientific notations
   auto ff = new TFormula("ff","x+2.e-2^1.2e-1");
   ok &= ( ff->Eval(1.) == (1. + std::pow(2.e-2,1.2e-1) ) );

   ff = new TFormula("ff","x^-1.2e1");
   ok &= ( ff->Eval(1.5) == std::pow(1.5,-1.2e1) ) ;

   ff = new TFormula("ff","1.5e2^x");
   ok &= ( ff->Eval(2) == std::pow(1.5e2,2) );

   ff = new TFormula("ff","1.5e2^x^-1.1e-2");
   ok &= ( ff->Eval(2.) == std::pow(1.5e2, std::pow(2,-1.1e-2) ) );

   // test same prelacements
   ff = new TFormula("ff","pol10(3)+pol2");
   std::vector<double> p = {1,2,3,4,5,6,7,8,9,10,11,12,13,14};
   ff->SetParameters(p.data() );
   double sum = 0; for (auto &a : p) { sum+= a;} 
   ok &= ( ff->Eval(1.) == sum );

   return ok;   
}

bool test26() {
   // test sign function
   bool ok = true;  
   TF1 f("f","x*sign(1.,x+2.)");
   ok &= (f.Eval(2) == 2);
   ok &= (f.Eval(-1) == -1);
   ok &= (f.Eval(-3) == 3);
   TF1 f2("f2","x*TMath::Sign(1,x+2)");
   ok &= (f2.Eval(2) == 2);
   ok &= (f2.Eval(-1) == -1);
   ok &= (f2.Eval(-3) == 3);
   TF1 f3("f3","TMath::SignBit(x-2)");
   ok &= (f3.Eval(1) == 1);
   ok &= (f3.Eval(3) == 0);
   return ok;
}

bool test27() {
   // test ssq function
   bool ok = true;  
   TF1 f1("f1","x+sq(x+2)+sq(x+[0])");
   TF1 f2("f2","x+(x+2)^2+(x+[0])^2");
   f1.SetParameter(0,3); f2.SetParameter(0,3);
   ok &= (f1.Eval(2) == f2.Eval(2));
   ok &= (f1.Eval(-4) == f2.Eval(-4));
   // test nested expressions and conflict with sqrt
   TF1 f3("f3","sqrt(1.+sq(x))");
   ok &= (f3.Eval(2) == sqrt(5) );
   TF1 f4("f4","sq(1.+std::sqrt(x))");
   ok &= (f4.Eval(2) == TMath::Sq(1.+sqrt(2)) );
   TF1 f5("f5","sqrt(((TMath::Sign(1,[0])*sq([0]/x))+(sq([1])*(x^([3]-1))))+sq([2]))");
   auto func = [](double *x, double *p){ return TMath::Sqrt(((TMath::Sign(1,p[0])*TMath::Sq(p[0]/x[0]))+(TMath::Sq(p[1])*(TMath::Power(x[0],(p[3]-1)))))+TMath::Sq(p[2])); };
   TF1 f6("f6",func,-10,10,4);
   f5.SetParameters(-1,2,3,4); f6.SetParameters(f5.GetParameters());
   ok &= (f5.Eval(2) == f6.Eval(2) );
   return ok;
}

bool test28() {
   bool ok = true; 
   // test composition of two functions
   TF1 fsin("fsin", "[0]*sin(x)", 0., 10.);
   fsin.SetParNames( "sin");
   fsin.SetParameter( 0, 2.1);

   TF1  fcos("fcos", "[0]*cos(x)", 0., 10.);
   fcos.SetParNames( "cos");
   fcos.SetParameter( 0, 1.1);

   TF1 fsincos("fsc", "fsin+fcos");

   // keep same order in evaluation
   TF1 f0("f0",[](double *x, double *p){ return p[1]*sin(x[0]) + p[0]*cos(x[0]);},0.,10.,2);
   f0.SetParameters(1.1,2.1);
   ok &= (fsincos.Eval(2) == f0.Eval(2) );
   return ok;

}

bool test29() {
   // test hexadecimal numbers 
   bool ok = true; 
   TF1 f1("f1","x+[0]*0xaf");
   f1.SetParameter(0,2);
   ok &= (f1.Eval(3) == (3.+2*175.) );

   TF1 f2("f2","0x64^2+x");
   ok &= (f2.Eval(1) == 10001 );

   TF1 f3("f3","x^0x000c+1");
   ok &= (f3.Eval(2) == 4097 );

   return ok; 

}

bool test30() {
// handle -- (++ is in linear expressions)
   bool ok = true;    
   TF1 f1("f1","x--[0]");
   f1.SetParameter(0,2);
   ok &= (f1.Eval(3) == 5. );

   return ok; 

}

bool test31() {
// test whitespaces in par name and cloning
   bool ok = true;    
   TF1 f1("f1","x*[0]");
   f1.SetParameter(0,2);
   f1.SetParName(0,"First Param");
   auto f2 = (TF1*) f1.Clone();
   
   ok &= (f1.Eval(3) == f2->Eval(3) );
   ok &= (TString(f1.GetParName(0) ) == TString(f2->GetParName(0) ) );

   return ok; 

}


   
void PrintError(int itest)  { 
   Error("TFormula test","test%d FAILED ",itest);
   failedTests.push_back(itest);
}
void IncrTest(int & itest) {
   if (itest > 0) std::cout << ".\n";
   itest++;
   std::cout << "Test " << itest << " :        ";
} 

int runTests(bool debug = false) {

   verbose = debug;
         
   int itest = 0;

   
   IncrTest(itest); if (!test1() ) { PrintError(itest); } 
   IncrTest(itest); if (!test2() ) { PrintError(itest); }
   IncrTest(itest); if (!test3() ) { PrintError(itest); }
   IncrTest(itest); if (!test4() ) { PrintError(itest); }
   IncrTest(itest); if (!test5() ) { PrintError(itest); }
   IncrTest(itest); if (!test6() ) { PrintError(itest); }
   IncrTest(itest); if (!test7() ) { PrintError(itest); }
   IncrTest(itest); if (!test8() ) { PrintError(itest); }
   IncrTest(itest); if (!test9() ) { PrintError(itest); }
   IncrTest(itest); if (!test10() ) { PrintError(itest); }
   IncrTest(itest); if (!test11() ) { PrintError(itest); }
   IncrTest(itest); if (!test12() ) { PrintError(itest); }
   IncrTest(itest); if (!test13() ) { PrintError(itest); }
   IncrTest(itest); if (!test14() ) { PrintError(itest); }
   IncrTest(itest); if (!test15() ) { PrintError(itest); }
   IncrTest(itest); if (!test16() ) { PrintError(itest); }
   IncrTest(itest); if (!test17() ) { PrintError(itest); }
   IncrTest(itest); if (!test18() ) { PrintError(itest); }
   IncrTest(itest); if (!test19() ) { PrintError(itest); }
   IncrTest(itest); if (!test20() ) { PrintError(itest); }
   IncrTest(itest); if (!test21() ) { PrintError(itest); }
   IncrTest(itest); if (!test22() ) { PrintError(itest); }
   IncrTest(itest); if (!test23() ) { PrintError(itest); }
   IncrTest(itest); if (!test24() ) { PrintError(itest); }
   IncrTest(itest); if (!test25() ) { PrintError(itest); }
   IncrTest(itest); if (!test26() ) { PrintError(itest); }
   IncrTest(itest); if (!test27() ) { PrintError(itest); }
   IncrTest(itest); if (!test28() ) { PrintError(itest); }
   IncrTest(itest); if (!test29() ) { PrintError(itest); }
   IncrTest(itest); if (!test30() ) { PrintError(itest); }
   IncrTest(itest); if (!test31() ) { PrintError(itest); }

   std::cout << ".\n";
    
   if (failedTests.size() == 0)  
      std::cout << "All TFormula Parsing tests PASSED !" << std::endl;
   else {
      Error("TFORMULA Tests","%d tests failed ",int(failedTests.size()) );
      std::cout << "failed tests are : ";
      for (auto & itest : failedTests) { 
         std::cout << itest << "   ";
      }
      std::cout << std::endl;
   }
   
   return failedTests.size(); 
   
}

};
