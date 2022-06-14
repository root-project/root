#include "TFitEditor.h"

#include "TApplication.h"
#include "TROOT.h"
#include "TBenchmark.h"

#include "TCanvas.h"
#include "TH1.h"

#include "TPluginManager.h"
#include "TError.h"

#include "TGComboBox.h"

#include <iostream>
#include <exception>
#include <stdexcept>
#include <cmath>
using namespace std;

#include "CommonDefs.h"

// Function that compares to doubles up to an error limit
int equals(Double_t n1, Double_t n2, double ERRORLIMIT = 1.E-10)
{
   return fabs( n1 - n2 ) > ERRORLIMIT * fabs(n1);
}

// Selects, given a TGComboBox*, the entry whose title is name.
int SelectEntry(TGComboBox* cb, const char* name)
{
   TGTextLBEntry* findEntry = static_cast<TGTextLBEntry*>( cb->FindEntry(name) );
   cb->Select(findEntry->EntryId());

   return findEntry->EntryId();
}

// Class to make the Unit Testing. It is important than the test
// methods are inside the class as this in particular is defined as a
// friend of the TFitEditor. This way, we can access the private
// methods of TFitEditor to perform several types of tests.
class FitEditorUnitTesting
{
private:
   // Pointer to the current (and only one) TFitEditor opened.
   TFitEditor* f;

   // These two variables are here to redirect the standard output to
   // a file.
   int old_stdout;
   FILE *out;
public:

   // Exception thrown when any of the pointers managed by the
   // FitEditorUnitTesting class are invalid
   class InvalidPointer: public std::exception
   {
   private:
      const char* _exp;
   public:
      InvalidPointer(const char* exp): _exp(exp) {};
      const char* what() { return _exp; };
   };

   // Constructor: Receives the instance of the TFitEditor
   FitEditorUnitTesting() {
      // Redirect the stdout to a file outputUnitTesting.txt
      old_stdout = dup (fileno (stdout));
      (void) freopen ("outputUnitTesting.txt", "w", stdout);
      out = fdopen (old_stdout, "w");

      // Execute the initial script
      gROOT->ProcessLine(".x $ROOTSYS/tutorials/fit/FittingDemo.C+");

      // Get an instance of the TFitEditor
      TCanvas* c1 = static_cast<TCanvas*>( gROOT->FindObject("c1") );
      TH1*      h = static_cast<TH1*>    ( gROOT->FindObject("histo") );

      f = TFitEditor::GetInstance(c1,h);

      if ( f == 0 )
         throw InvalidPointer("In FitEditorUnitTesting constructor");
   }

   // The destructor will close the TFitEditor and terminate the
   // application. Unfortunately, the application must be run from
   // main, otherwise, the test will make a segmentation fault while
   // trying to retrieve the TFitEditor singleton. If the user wants
   // to play a bit with the fitpanel once the tests have finised,
   // then they should comment this method.
   ~FitEditorUnitTesting() {
      f->DoClose();
      gApplication->Terminate();
   }

   // This is a generic method to make the output of all the tests
   // consistent. T is a function pointer to one of the tests
   // function. It has been implemented through templates to permit
   // more test types than the originally designed.
   // @ str : Name of the test
   // @ func : Member function pointer to the real implementation of
   // the test.
   template <typename T>
   int MakeTest(const char* str,  T func )
   {
      fprintf(stdout, "\n***** %s *****\n", str);
      int status = (this->*func)();

      fprintf(stdout, "%s..........", str);
      fprintf(out, "%s..........", str);
      if ( status == 0 ) {
         fprintf(stdout, "OK\n");
         fprintf(out, "OK\n");
      }
      else {
         fprintf(stdout, "FAILED\n");
         fprintf(out, "FAILED\n");
      }
      return status;
   }

   // This is where all the tests are called. If the user wants to add
   // new tests or avoid executing one of the existing ones, it is
   // here where they should do it.
   int UnitTesting() {
      int result = 0;

      fprintf(out, "\n**STARTING TFitEditor Unit Tests**\n\n");

      result += MakeTest("TestHistogramFit...", &FitEditorUnitTesting::TestHistogramFit);

      result += MakeTest("TestGSLFit.........", &FitEditorUnitTesting::TestGSLFit);

      result += MakeTest("TestUpdate.........", &FitEditorUnitTesting::TestUpdate);

      result += MakeTest("TestGraph..........", &FitEditorUnitTesting::TestGraph);

      result += MakeTest("TestGraphError.....", &FitEditorUnitTesting::TestGraphError);

      result += MakeTest("TestGraph2D........", &FitEditorUnitTesting::TestGraph2D);

      result += MakeTest("TestGraph2DError...", &FitEditorUnitTesting::TestGraph2DError);

      result += MakeTest("TestUpdateTree.....", &FitEditorUnitTesting::TestUpdateTree);

      result += MakeTest("TestTree1D.........", &FitEditorUnitTesting::TestTree1D);

      result += MakeTest("TestTree2D.........", &FitEditorUnitTesting::TestTree2D);

      result += MakeTest("TestTreeND.........", &FitEditorUnitTesting::TestTreeND);

      fprintf(out, "\nRemember to also check outputUnitTesting.txt for "
              "more detailed information\n\n");

      return result;
   }

   // This is a debuggin method used to print the parameter values
   // stored in the fitpanel. This is useful when performing a fit, to
   // know against which values the test should be compare to.
   void PrintFuncPars()
   {
      static int counter = 0;
      fprintf(out, "Printing the Func Pars (%d)\n", ++counter);
      for ( unsigned int i = 0; i < f->fFuncPars.size(); ++i ) {
         fprintf(out, "%30.20f %30.20f %30.20f\n", f->fFuncPars[i][0], f->fFuncPars[i][1], f->fFuncPars[i][2]);
      }
   }

   // This function compares the parameters stored in the TFitEditor
   // with the ones passed by the test functions. Normally, if the
   // function return 0, it means all the parameters are equal up to a
   // certain limit, thus the test was successful.
   int CompareFuncPars(std::vector<TFitEditor::FuncParamData_t>& pars)
   {
      int status = 0;
      for ( unsigned int i = 0; i < f->fFuncPars.size(); ++i ) {
         for ( unsigned int j = 0; j < 3; ++j) {
            int internalStatus = equals(pars[i][j], f->fFuncPars[i][j]);
            //fprintf(out, "i: %d, j: %d, e: %d\n", i, j, internalStatus);
            status += internalStatus;
         }
      }

      return status;
   }

   // From here, the implementation of the different tests. The names
   // of the test should be enough to know what they are testing, as
   // these tests are mean to be as simple as possible.

   int TestHistogramFit() {
      f->fTypeFit->Select(kFP_UFUNC, kTRUE);
      f->fFuncList->Select(kFP_ALTFUNC, kTRUE);
      f->DoFit();

      std::vector<TFitEditor::FuncParamData_t> pars(6);
      pars[0][0] = -0.86471376634076801970;  pars[0][1] = pars[0][2] = 0.0;
      pars[1][0] = 45.84337697060870908672;  pars[1][1] = pars[1][2] = 0.0;
      pars[2][0] = -13.32141783912906873866; pars[2][1] = pars[2][2] = 0.0;
      pars[3][0] = 13.80743352672578438955;  pars[3][1] = pars[3][2] = 0.0;
      pars[4][0] = 0.17230936727526752206;   pars[4][1] = pars[4][2] = 0.0;
      pars[5][0] = 0.98728095791845293938;   pars[5][1] = pars[5][2] = 0.0;

      return CompareFuncPars(pars);
   }

   int TestGSLFit() {
      f->fTypeFit->Select(kFP_PREVFIT, kTRUE);
      f->fLibGSL->Toggled(kTRUE);
      f->fMinMethodList->Select(kFP_BFGS2, kTRUE);
      f->DoFit();

      std::vector<TFitEditor::FuncParamData_t> pars(6);
      pars[0][0] = -0.86471376626133966692;  pars[0][1] = pars[0][2] = 0.0;
      pars[1][0] = 45.84337697042452219875;  pars[1][1] = pars[1][2] = 0.0;
      pars[2][0] = -13.32141783972060622432; pars[2][1] = pars[2][2] = 0.0;
      pars[3][0] = 13.80743352667312962012;  pars[3][1] = pars[3][2] = 0.0;
      pars[4][0] = 0.17230936776683797307;   pars[4][1] = pars[4][2] = 0.0;
      pars[5][0] = 0.98728095212777022827;   pars[5][1] = pars[5][2] = 0.0;

      return CompareFuncPars(pars);
   }

   int TestUpdate() {
      gROOT->ProcessLine(".x $ROOTSYS/tutorials/fit/ConfidenceIntervals.C+");
      f->DoUpdate();

      return 0;
   }

   int TestGraph() {
      SelectEntry(f->fDataSet, "TGraph::GraphNoError");

      f->fLibMinuit2->Toggled(kTRUE);
      f->fMinMethodList->Select(kFP_MIGRAD, kTRUE);

      f->fTypeFit->Select(kFP_UFUNC, kTRUE);
      SelectEntry(f->fFuncList, "fpol");
      f->DoFit();

      std::vector<TFitEditor::FuncParamData_t> pars(2);
      pars[0][0] = -1.07569876898511784802;  pars[0][1] = pars[0][2] = 0.0;
      pars[1][0] = 1.83337233651544084800;  pars[1][1] = pars[1][2] = 0.0;

      return CompareFuncPars(pars);
   }

    int TestGraphError() {
      SelectEntry(f->fDataSet, "TGraphErrors::Graph");

      f->fLibMinuit2->Toggled(kTRUE);
      f->fMinMethodList->Select(kFP_MIGRAD, kTRUE);

      f->fTypeFit->Select(kFP_UFUNC, kTRUE);
      SelectEntry(f->fFuncList, "fpol");
      f->DoFit();

      std::vector<TFitEditor::FuncParamData_t> pars(2);
      pars[0][0] = -1.07569876898508010044;  pars[0][1] = pars[0][2] = 0.0;
      pars[1][0] = 1.83337233651530895351;  pars[1][1] = pars[1][2] = 0.0;

      return CompareFuncPars(pars);
   }

   int TestGraph2D() {
      SelectEntry(f->fDataSet, "TGraph2D::Graph2DNoError");

      f->fLibMinuit2->Toggled(kTRUE);
      f->fMinMethodList->Select(kFP_MIGRAD, kTRUE);

      f->fTypeFit->Select(kFP_UFUNC, kTRUE);
      SelectEntry(f->fFuncList, "f2");

      // Set the parameters to the original ones in
      // ConfidenceIntervals.C. Otherwise it will be using those of
      // the last fit with fpol and will make an invalid fit.
      f->fFuncPars[0][0] = 0.5;
      f->fFuncPars[1][0] = 1.5;

      f->DoFit();

      std::vector<TFitEditor::FuncParamData_t> pars(2);
      pars[0][0] = 0.57910401391086918643;  pars[0][1] = pars[0][2] = 0.0;
      pars[1][0] = 1.73731204173242681499;  pars[1][1] = pars[1][2] = 0.0;

      return CompareFuncPars(pars);
   }

   int TestGraph2DError() {
      SelectEntry(f->fDataSet, "TGraph2DErrors::Graph2D");

      f->fLibMinuit2->Toggled(kTRUE);
      f->fMinMethodList->Select(kFP_MIGRAD, kTRUE);

      f->fTypeFit->Select(kFP_UFUNC, kTRUE);
      SelectEntry(f->fFuncList, "f2");

      // Set the parameters to the original ones in
      // ConfidenceIntervals.C. Otherwise it will be using those of
      // the last fit with f2 and the fit will make no sense.
      f->fFuncPars[0][0] = 0.5;
      f->fFuncPars[1][0] = 1.5;

      f->DoFit();

      std::vector<TFitEditor::FuncParamData_t> pars(2);
      pars[0][0] = 0.57911670684083915717;  pars[0][1] = pars[0][2] = 0.0;
      pars[1][0] = 1.73735012087486695442;  pars[1][1] = pars[1][2] = 0.0;

      return CompareFuncPars(pars);
   }

   int TestUpdateTree() {
      gROOT->ProcessLine(".x ~/tmp/fitpanel/createTree.C++");
      f->DoUpdate();
      return 0;
   }

   int TestTree1D() {
      TObject* objSelected = gROOT->FindObject("tree");
      if ( !objSelected )
         throw InvalidPointer("In TestUpdateTree");

      Int_t selected = kFP_NOSEL + 6;

      f->ProcessTreeInput(objSelected, selected, "x", "y>1");
      f->fTypeFit->Select(kFP_PRED1D, kTRUE);
      SelectEntry(f->fFuncList, "gausn");

      f->fFuncPars.resize(3);
      f->fFuncPars[0][0] = f->fFuncPars[0][1] = f->fFuncPars[0][2] = 1;
      f->fFuncPars[1][0] = 0;
      f->fFuncPars[2][0] = 1;

      f->DoFit();

      std::vector<TFitEditor::FuncParamData_t> pars(3);
      pars[0][0] = 1.0;  pars[0][1] = pars[0][2] = 1.0;
      pars[1][0] = 0.57616222565122654498;  pars[1][1] = pars[1][2] = 0.0;
      pars[2][0] = 0.90739764318839521984;  pars[2][1] = pars[2][2] = 0.0;

      return CompareFuncPars(pars);
   }

   int TestTree2D() {
      TObject* objSelected = gROOT->FindObject("tree");
      if ( !objSelected )
         throw InvalidPointer("In TestUpdateTree");

      Int_t selected = kFP_NOSEL + 6;

      f->ProcessTreeInput(objSelected, selected, "x:y", "");
      f->fTypeFit->Select(kFP_UFUNC, kTRUE);
      SelectEntry(f->fFuncList, "gaus2d");

      f->fFuncPars[0][0] = 1; f->fFuncPars[0][1] = f->fFuncPars[0][2] = 0;
      f->fFuncPars[1][0] = 1; f->fFuncPars[1][1] = f->fFuncPars[1][2] = 0;
      f->fFuncPars[2][0] = 0; f->fFuncPars[2][1] = f->fFuncPars[2][2] = 0;

      f->DoFit();

      std::vector<TFitEditor::FuncParamData_t> pars(3);
      pars[0][0] = 1.01009862846512765699;  pars[0][1] = pars[0][2] = 0.0;
      pars[1][0] = 2.00223267618221001385;  pars[1][1] = pars[1][2] = 0.0;
      pars[2][0] = 0.49143171847344568892;  pars[2][1] = pars[2][2] = 0.0;

      return CompareFuncPars(pars);
   }

   int TestTreeND() {
      TObject* objSelected = gROOT->FindObject("tree");
      if ( !objSelected )
         throw InvalidPointer("In TestUpdateTree");

      Int_t selected = kFP_NOSEL + 6;

      f->ProcessTreeInput(objSelected, selected, "x:y:z:u:v:w", "");
      f->fTypeFit->Select(kFP_UFUNC, kTRUE);
      SelectEntry(f->fFuncList, "gausND");

      f->fFuncPars[ 0][0] = 1.0; f->fFuncPars[ 0][1] = f->fFuncPars[ 0][2] = 0;
      f->fFuncPars[ 1][0] = 1.0; f->fFuncPars[ 1][1] = f->fFuncPars[ 1][2] = 0;
      f->fFuncPars[ 2][0] = 0.1; f->fFuncPars[ 2][1] = f->fFuncPars[ 2][2] = 0;
      f->fFuncPars[ 3][0] = 0.0; f->fFuncPars[ 3][1] = f->fFuncPars[ 3][2] = 0;
      f->fFuncPars[ 4][0] = 2.0; f->fFuncPars[ 4][1] = f->fFuncPars[ 4][2] = 0;
      f->fFuncPars[ 5][0] = 0.0; f->fFuncPars[ 5][1] = f->fFuncPars[ 5][2] = 0;
      f->fFuncPars[ 6][0] = 3.0; f->fFuncPars[ 6][1] = f->fFuncPars[ 6][2] = 0;
      f->fFuncPars[ 7][0] = 0.0; f->fFuncPars[ 7][1] = f->fFuncPars[ 7][2] = 0;
      f->fFuncPars[ 8][0] = 4.0; f->fFuncPars[ 8][1] = f->fFuncPars[ 8][2] = 0;
      f->fFuncPars[ 9][0] = 0.0; f->fFuncPars[ 9][1] = f->fFuncPars[ 9][2] = 0;
      f->fFuncPars[10][0] = 9.0; f->fFuncPars[10][1] = f->fFuncPars[10][2] = 0;

      f->DoFit();

      std::vector<TFitEditor::FuncParamData_t> pars(11);
      pars[ 0][0] = 1.01010130092504835098;  pars[ 0][1] = pars[ 0][2] = 0;
      pars[ 1][0] = 2.00223693541403102714;  pars[ 1][1] = pars[ 1][2] = 0;
      pars[ 2][0] = 0.49142981449519324011;  pars[ 2][1] = pars[ 2][2] = 0;
      pars[ 3][0] = 0.03058404503876750724;  pars[ 3][1] = pars[ 3][2] = 0;
      pars[ 4][0] = 2.98217423626109168211;  pars[ 4][1] = pars[ 4][2] = 0;
      pars[ 5][0] = 0.08458881936812148727;  pars[ 5][1] = pars[ 5][2] = 0;
      pars[ 6][0] = 3.97659923278031923743;  pars[ 6][1] = pars[ 6][2] = 0;
      pars[ 7][0] = -0.03584554242634782617; pars[ 7][1] = pars[ 7][2] = 0;
      pars[ 8][0] = 4.96478032328273499729;  pars[ 8][1] = pars[ 8][2] = 0;
      pars[ 9][0] = 0.89557700499129078153;  pars[ 9][1] = pars[ 9][2] = 0;
      pars[10][0] = 9.92938972972320499366;  pars[10][1] = pars[10][2] = 0;


      return CompareFuncPars(pars);
   }
};

// Runs the  basic script  and pops  out the fit  panel. Then  it will
// initialize the  FitEditorUnitTesting class and make it  run all the
// tests
int UnitTesting()
{
   FitEditorUnitTesting fUT;

   return fUT.UnitTesting();
}

// The main function. It is VERY important that it is run using the
// TApplication.
int main(int argc, char** argv)
{
   TApplication* theApp = 0;

   theApp = new TApplication("App",&argc,argv);

   int ret =  UnitTesting();

   theApp->Run();
   delete theApp;
   theApp = 0;

   return ret;
}
