// Author: Omar Zapata  Omar.Zapata@cern.ch   2014

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
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
#include <TRint.h>
#include <TSystem.h>

#if defined(HAS_X11)
#include <X11/Xlib.h>
#include "TROOT.h"
#include "TEnv.h"
#endif
using namespace ROOT::R;

static ROOT::R::TRInterface *gR = nullptr;
static Bool_t statusEventLoop;

TRInterface::TRInterface(const Int_t argc, const Char_t *argv[], const Bool_t loadRcpp, const Bool_t verbose,
                         const Bool_t interactive)
   : TObject()
{
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
   std::string osname = Eval("Sys.info()['sysname']");
   //only for linux/mac windows is not supported by ROOT yet.
#if defined(HAS_X11)
   if (!gROOT->IsBatch()) {
      if (gEnv->GetValue("X11.XInitThread", 1)) {
         // Must be very first call before any X11 call !!
         if (!XInitThreads())
            Warning("OpenDisplay", "system has no X11 thread support");
      }
   }
#endif
   if (osname == "Linux") {
      Execute("options(device='x11')");
   } else {
      Execute("options(device='quartz')");
   }

}

TRInterface::~TRInterface()
{
   statusEventLoop = kFALSE;
   if (th) delete th;
   if (fR) delete fR;
   if (gR == this) gR = nullptr;
}

//______________________________________________________________________________
Int_t  TRInterface::Eval(const TString &code, TRObject  &ans)
{
   SEXP fans;

   Int_t rc = kFALSE;
   try {
      rc = fR->parseEval(code.Data(), fans);
   } catch (Rcpp::exception &__ex__) {
      Error("Eval", "%s", __ex__.what());
      forward_exception_to_r(__ex__) ;
   } catch (...) {
      Error("Eval", "Can execute the requested code: %s", code.Data());
   }
   ans = fans;
   ans.SetStatus((rc == 0) ? kTRUE : kFALSE);
   return rc;
}

//______________________________________________________________________________
void TRInterface::Execute(const TString &code)
{
   try {

      fR->parseEvalQ(code.Data());
   } catch (Rcpp::exception &__ex__) {
      Error("Execute", "%s", __ex__.what());
      forward_exception_to_r(__ex__) ;
   } catch (...) {
      Error("Execute", "Can execute the requested code: %s", code.Data());
   }
}

//______________________________________________________________________________
TRObject TRInterface::Eval(const TString &code)
{
// Execute R code.
//The RObject result of execution is returned in TRObject

   SEXP ans;

   int rc = kFALSE;
   try {
      rc = fR->parseEval(code.Data(), ans);
   } catch (Rcpp::exception &__ex__) {
      Error("Eval", "%s", __ex__.what());
      forward_exception_to_r(__ex__) ;
   } catch (...) {
      Error("Eval", "Can execute the requested code: %s", code.Data());
   }

   return TRObject(ans, (rc == 0) ? kTRUE : kFALSE);
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
void TRInterface::Assign(const TRFunctionExport &obj, const TString &name)
{
   fR->assign(*obj.f, name.Data());
}

//______________________________________________________________________________
void TRInterface::Assign(const TRDataFrame &obj, const TString &name)
{
   //This method lets you pass c++ functions to R environment.
   fR->assign(obj.df, name.Data());
}

//______________________________________________________________________________
void TRInterface::Interactive()
{
   while (kTRUE) {
      Char_t *line = readline("[r]:");
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
   if (!gR) {
      const Char_t *R_argv[] = {"rootr",    "--gui=none", "--no-save", "--no-readline",
                                "--silent", "--vanilla",  "--slave"};
      gR = new TRInterface(7, R_argv, true, false, false);
   }
   gR->ProcessEventsLoop();
   return gR;
}

//______________________________________________________________________________
TRInterface &TRInterface::Instance()
{
   return  *TRInterface::InstancePtr();
}

namespace {

// Per CRAN policy, an R package name starts with a letter, contains only ASCII
// letters, digits, and dots, and does not end with a dot. Restricting to this
// set is a helpful validation step for the user and prevents R-source
// injection via the string concatenation done in IsInstalled / Require /
// Install below.
bool IsValidRPackageName(const TString &pkg)
{
   const Ssiz_t n = pkg.Length();
   if (n == 0)
      return false;
   const char first = pkg[0];
   if (!((first >= 'A' && first <= 'Z') || (first >= 'a' && first <= 'z')))
      return false;
   for (Ssiz_t i = 1; i < n; ++i) {
      const char c = pkg[i];
      const Bool_t ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.';
      if (!ok)
         return false;
   }
   return pkg[n - 1] != '.';
}

// Allow only an http(s)/ftp/file URL with no characters that could break
// out of the R single-quoted literal in Install().
bool IsValidRReposUrl(const TString &repos)
{
   const Ssiz_t n = repos.Length();
   if (n == 0)
      return false;
   const char *prefixes[] = {"http://", "https://", "ftp://", "file://"};
   Bool_t prefixOk = false;
   for (const char *s : prefixes) {
      if (repos.BeginsWith(s)) {
         prefixOk = true;
         break;
      }
   }
   if (!prefixOk)
      return false;
   for (Ssiz_t i = 0; i < n; ++i) {
      const char c = repos[i];
      if (c == '\'' || c == '\\' || c == '`' || c == ';' || c == '\n' || c == '\r')
         return false;
   }
   return true;
}

} // namespace

//______________________________________________________________________________
Bool_t TRInterface::IsInstalled(TString pkg)
{
   if (!IsValidRPackageName(pkg)) {
      Error("IsInstalled", "Invalid R package name: %s", pkg.Data());
      return kFALSE;
   }
   TString cmd = "is.element('" + pkg + "', installed.packages()[,1])";
   return this->Eval(cmd).As<Bool_t>();
}

//______________________________________________________________________________
Bool_t TRInterface::Require(TString pkg)
{
   if (!IsValidRPackageName(pkg)) {
      Error("Require", "Invalid R package name: %s", pkg.Data());
      return kFALSE;
   }
   TString cmd = "require('" + pkg + "',quiet=TRUE)";
   return this->Eval(cmd).As<Bool_t>();
}

//______________________________________________________________________________
Bool_t TRInterface::Install(TString pkg, TString repos)
{
   if (!IsValidRPackageName(pkg)) {
      Error("Install", "Invalid R package name: %s", pkg.Data());
      return kFALSE;
   }
   if (!IsValidRReposUrl(repos)) {
      Error("Install", "Invalid R repository URL: %s", repos.Data());
      return kFALSE;
   }
   TString cmd = "install.packages('" + pkg + "',repos='" + repos + "',dependencies=TRUE)";
   this->Eval(cmd);
   return IsInstalled(pkg);
}


#undef _POSIX_C_SOURCE
#include <R_ext/eventloop.h>

//______________________________________________________________________________
void TRInterface::ProcessEventsLoop()
{
   if (!statusEventLoop) {
      th = new TThread([](void */*args */) {
         while (statusEventLoop) {
            fd_set *fd;
            Int_t usec = 10000;
            fd = R_checkActivity(usec, 0);
            R_runHandlers(R_InputHandlers, fd);
            if (gSystem) gSystem->Sleep(100);
         }
      });
      statusEventLoop = kTRUE;
      th->Run();
   }
}
