#include "TF1.h"
#include "TFormula.h"
#include "TGraph.h"

// test of tformula neeeded to be run


class TFormulaParceTests {

   
bool verbose; 
std::vector<int> failedTests; 
public:

TFormulaParceTests(bool _verbose = false) : verbose(_verbose) {}

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

   f = new TFormula("f","(x+[0])^y");
   f->SetParameter(0,1);
   ok &= (f->Eval(2,3) == 27);

   f = new TFormula("f","sqrt(x+[0])^y");
   f->SetParameter(0,2);
   ok &= (f->Eval(2,3) == 8);

   f = new TFormula("f","[0]/((x+2)^y)");
   f->SetParameter(0,27);
   ok &= (f->Eval(1,3) == 1);
   
   f = new TFormula("f","[0]/((x+2)^(y+1))");
   f->SetParameter(0,27);
   ok &= (f->Eval(1,2) == 1);

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

   std::cout << ".\n";
    
   if (failedTests.size() == 0)  
      std::cout << "All TFormula Parcing tests PASSED !" << std::endl;
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
