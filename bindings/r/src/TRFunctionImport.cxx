/*************************************************************************
 * Copyright (C)      2015, Omar Andres Zapata Mesa                      *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include<TRFunctionImport.h>
#include <TRObject.h>

#include <Rcpp/Function.h>

//______________________________________________________________________________
/* Begin_Html
End_Html
*/


using namespace ROOT::R;
ClassImp(TRFunctionImport)



//______________________________________________________________________________
TRFunctionImport::TRFunctionImport(const TRFunctionImport &fun): TObject(fun)
{
   f = fun.f;
}

//______________________________________________________________________________
TRFunctionImport::TRFunctionImport(const TString &name)
{
   f = new Rcpp::Function(name.Data());
}

//______________________________________________________________________________
TRFunctionImport::TRFunctionImport(const TString &name, const TString &ns)
{
   f = new Rcpp::Function(name.Data(), ns.Data());
}

//______________________________________________________________________________
TRFunctionImport::TRFunctionImport(TRObject &obj): TObject(obj)
{
   f = new Rcpp::Function((SEXP)obj);
}

//______________________________________________________________________________
TRFunctionImport::TRFunctionImport(SEXP obj)
{
   f = new Rcpp::Function(obj);
}
