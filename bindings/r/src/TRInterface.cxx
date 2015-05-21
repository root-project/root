/*************************************************************************
 * Copyright (C) 2013-2014, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include<TRInterface.h>
#include"TRCompletion.h"
#include<vector>

extern "C"
{
#include <stdio.h>
#include <stdlib.h>
}
#include<TRint.h>

//______________________________________________________________________________
/* Begin_Html
<center><h2>TRInterface class</h2></center>

</p>
The TRInterface class lets you procces R code from ROOT.<br>
You can call R libraries and their functions, plot results in R or ROOT,<br>
and use the power of ROOT and R at the same time.<br>
It also lets you pass scalars, vectors and matrices from ROOT to R<br>
and from R to ROOT using TRObjectProxy; but you can to use overloaded opetarors [],<< and >> <br>
to work with ROOTR like work with streams of data.<br>

TRInterface class can not be instantiated directly, but you can create objects using the static methods
TRInterface& Instance() and TRInterface* InstancePtr() to create your own objects.<br>
<br>
</p>
Show an example below:
End_Html
Begin_Macro(source)
{

//Create an exponential fit
//The idea is to create a set of numbers x,y with noise from ROOT,
//pass them to R and fit the data to x^3,
//get the fitted coefficient(power) and plot the data,
//the known function and the fitted function.
//Author:: Omar Zapata
   TCanvas *c1 = new TCanvas("c1","Curve Fit",700,500);
   c1->SetGrid();

   // draw a frame for multiples graphs
   TMultiGraph *mg = new TMultiGraph();

   // create the first graph (points with gaussian noise)
   const Int_t n = 24;
   Double_t x[n] ;
   Double_t y[n] ;
   //Generate points along a X^3 with noise
   TRandom rg;
   rg.SetSeed(520);
   for (Int_t i = 0; i < n; i++) {
      x[i] = rg.Uniform(0, 1);
      y[i] = TMath::Power(x[i], 3) + rg.Gaus() * 0.06;
   }

   TGraph *gr1 = new TGraph(n,x,y);
   gr1->SetMarkerColor(kBlue);
   gr1->SetMarkerStyle(8);
   gr1->SetMarkerSize(1);
   mg->Add(gr1);

      // create second graph
   TF1 *f_known=new TF1("f_known","pow(x,3)",0,1);
   TGraph *gr2 = new TGraph(f_known);
   gr2->SetMarkerColor(kRed);
   gr2->SetMarkerStyle(8);
   gr2->SetMarkerSize(1);
   mg->Add(gr2);

   //passing x and y values to R for fitting
   ROOT::R::TRInterface &r=ROOT::R::TRInterface::Instance();
   r["x"]<<TVectorD(n, x);
   r["y"]<<TVectorD(n, y);
   //creating a R data frame
   r<<"ds<-data.frame(x=x,y=y)";
   //fitting x and y to X^power using Nonlinear Least Squares
   r<<"m <- nls(y ~ I(x^power),data = ds, start = list(power = 1),trace = T)";
   //getting the fitted value (power)
   Double_t power;
   r["summary(m)$coefficients[1]"]>>power;

   TF1 *f_fitted=new TF1("f_fitted","pow(x,[0])",0,1);
   f_fitted->SetParameter(0,power);
   //plotting the fitted function
   TGraph *gr3 = new TGraph(f_fitted);
   gr3->SetMarkerColor(kGreen);
   gr3->SetMarkerStyle(8);
   gr3->SetMarkerSize(1);

   mg->Add(gr3);
   mg->Draw("ap");

   //displaying basic results
   TPaveText *pt = new TPaveText(0.1,0.6,0.5,0.9,"brNDC");
   pt->SetFillColor(18);
   pt->SetTextAlign(12);
   pt->AddText("Fitting x^power ");
   pt->AddText(" \"Blue\"   Points with gaussian noise to be fitted");
   pt->AddText(" \"Red\"    Known function x^3");
   TString fmsg;
   fmsg.Form(" \"Green\"  Fitted function with power=%.4lf",power);
   pt->AddText(fmsg);
   pt->Draw();
   c1->Update();
   return c1;
}
End_Macro */

using namespace ROOT::R;
ClassImp(TRInterface)

static ROOT::R::TRInterface *gR = NULL;
static Bool_t statusEventLoop;

//______________________________________________________________________________
TRInterface::TRInterface(const int argc, const char *argv[], const bool loadRcpp, const bool verbose, const bool interactive): TObject()
{
// The command line arguments are by deafult argc=0 and argv=NULL,
// The verbose mode is by default disabled but you can enable it to show procedures information in stdout/stderr
   if (RInside::instancePtr()) throw std::runtime_error("Can only have one TRInterface instance");
   fR = new RInside(argc, argv, loadRcpp, verbose, interactive);

   //Installing the readline callbacks for completion in the
   //method Interactive
   rcompgen_rho = R_FindNamespace(Rf_mkString("utils"));
   RComp_assignBufferSym  = Rf_install(".assignLinebuffer");
   RComp_assignStartSym   = Rf_install(".assignStart");
   RComp_assignEndSym     = Rf_install(".assignEnd");
   RComp_assignTokenSym   = Rf_install(".assignToken");
   RComp_completeTokenSym = Rf_install(".completeToken");
   RComp_getFileCompSym   = Rf_install(".getFileComp");
   RComp_retrieveCompsSym = Rf_install(".retrieveCompletions");
   rl_attempted_completion_function = R_custom_completion;
   statusEventLoop = kFALSE;
}

TRInterface::~TRInterface()
{
   if (th) delete th;
}

void TRInterface::LoadModule(TString name)
{
  //Method to load wrapped ROOT's classes into R's environment
  //e.g: r.LoadModule("Hist") load the current(partially) wrapped classes TF1 and TGraph
   if (name == "Hist") {
      gApplication->ProcessLine("#include<TRF1.h>");
      gApplication->ProcessLine("LOAD_ROOTR_MODULE(ROOTR_TRF1);");
      gR->Execute("TF1       <- .GlobalEnv$.__C__Rcpp_TRF1");

      gApplication->ProcessLine("#include<TRGraph.h>");
      gApplication->ProcessLine("LOAD_ROOTR_MODULE(ROOTR_TRGraph);");
      gR->Execute("TGraph     <- .GlobalEnv$.__C__Rcpp_TRGraph");

   }
   if (name == "Gpad") {
      gApplication->ProcessLine("#include<TRCanvas.h>");
      gApplication->ProcessLine("LOAD_ROOTR_MODULE(ROOTR_TRCanvas);");
      gR->Execute("TCanvas     <- .GlobalEnv$.__C__Rcpp_TRCanvas");
   }
   if (name == "Rint") {
      gApplication->ProcessLine("#include<TRRint.h>");
      gApplication->ProcessLine("LOAD_ROOTR_MODULE(ROOTR_TRRint);");
      gR->Execute("TRint     <- .GlobalEnv$.__C__Rcpp_TRRint");
   }
   if (name == "Core") {
      gApplication->ProcessLine("#include<TRSystem.h>");
      gApplication->ProcessLine("LOAD_ROOTR_MODULE(ROOTR_TRSystem);");
   }
   if (name == "IO") {
      gApplication->ProcessLine("#include<TRFile.h>");
      gApplication->ProcessLine("LOAD_ROOTR_MODULE(ROOTR_TRFile);");
      gR->Execute("TFile     <- .GlobalEnv$.__C__Rcpp_TRFile");
   }

}


//______________________________________________________________________________
Int_t  TRInterface::Eval(const TString &code, TRObjectProxy  &ans)
{
// Parse R code and returns status of execution.
// the RObject's response is saved in  ans
   SEXP fans;
   Int_t rc = 0;
   try{ 
        rc = fR->parseEval(code.Data(), fans);
    } 
   catch(Rcpp::exception& __ex__){
       Error("Eval", "%s",__ex__.what());
       forward_exception_to_r( __ex__ ) ;
   }
   catch(...){Error("Eval", "Can execute the requested code: %s",code.Data());}
   ans = fans;
   ans.SetStatus((rc == 0) ? kTRUE : kFALSE);
   return rc;
}

//______________________________________________________________________________
void TRInterface::Execute(const TString &code)
{
// Execute R code.
  try{ 

        fR->parseEvalQ(code.Data());
    }
   catch(Rcpp::exception& __ex__){
       Error("Execute", "%s",__ex__.what());
       forward_exception_to_r( __ex__ ) ;
   }
   catch(...){Error("Execute", "Can execute the requested code: %s",code.Data());}
}

//______________________________________________________________________________
TRObjectProxy TRInterface::Eval(const TString &code)
{
// Execute R code. 
//The RObject result of execution is returned in TRObjectProxy
  
   SEXP ans;
   int rc = 0;
   try{
   rc = fR->parseEval(code.Data(), ans);
       } 
   catch(Rcpp::exception& __ex__){
       Error("Eval", "%s",__ex__.what());
       forward_exception_to_r( __ex__ ) ;
   }
   catch(...){Error("Eval", "Can execute the requested code: %s",code.Data());}

   return TRObjectProxy(ans , (rc == 0) ? kTRUE : kFALSE);
}


void TRInterface::SetVerbose(Bool_t status)
{
   //verbose mode shows you all the procedures in stdout/stderr
   //very important to debug and to see the results.
   fR->setVerbose(status);
}

//______________________________________________________________________________
TRInterface::Binding TRInterface::operator[](const TString &name)
{
   return Binding(this, name);
}

//______________________________________________________________________________
void TRInterface::Assign(const TRFunction &obj, const TString &name)
{
   //This method lets you pass c++ functions to R environment.
   fR->assign(*obj.f, name.Data());
}

//______________________________________________________________________________
void TRInterface::Interactive()
{
   //This method launches a R command line to run directly R code which you can
   //pass to ROOT calling the apropiate method.

   while (kTRUE) {
      char *line = readline("[r]:");
      if (!line) continue;
      if (std::string(line) == ".q") break;
      Execute(line);
      if (*line) add_history(line);
      free(line);
   }
}


//______________________________________________________________________________
TRInterface *TRInterface::InstancePtr()
{
  //return a pointer to TRInterface.
   if (!gR) {
      const char *R_argv[] = {"rootr", "--gui=none", "--no-save", "--no-readline", "--silent", "--vanilla", "--slave"};
      gR = new TRInterface(7, R_argv, true, false, false);
   }
   gR->ProcessEventsLoop();
   return gR;
}

//______________________________________________________________________________
TRInterface &TRInterface::Instance()
{
  //return a reference object of TRInterface.
   return  *TRInterface::InstancePtr();
}

Bool_t TRInterface::IsInstalled(TString pkg)
{
    TString cmd="is.element('"+pkg+"', installed.packages()[,1])";
    return fR->parseEval(cmd.Data());
}

Bool_t TRInterface::Require(TString pkg)
{
    TString cmd="require('"+pkg+"',quiet=TRUE)";
    return fR->parseEval(cmd.Data());
}

Bool_t TRInterface::Install(TString pkg,TString repos)
{
    TString cmd="install.packages('"+pkg+"',repos='"+repos+"',dependencies=TRUE)";
    fR->parseEval(cmd.Data());
    return IsInstalled(pkg);
}


#undef _POSIX_C_SOURCE
#include <R_ext/eventloop.h>

//______________________________________________________________________________
void TRInterface::ProcessEventsLoop()
{
   //run the R's eventloop to process graphics events
   if (!statusEventLoop) {
      th = new TThread([](void * args) {
         while (kTRUE) {
            fd_set *fd;
            int usec = 10000;
            fd = R_checkActivity(usec, 0);
            R_runHandlers(R_InputHandlers, fd);
            gSystem->Sleep(100);
         }
      });
      th->Run();
      statusEventLoop = kTRUE;
   }
}
