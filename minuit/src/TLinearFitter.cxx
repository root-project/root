// @(#)root/minuit:$Name:  $:$Id: TLinearFitter.cxx,v 1.8 2005/04/29 16:10:42 brun Exp $
// Author: Anna Kreshuk 04/03/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TLinearFitter.h"
#include "TDecompChol.h"
#include "TGraph.h"
#include "TGraph2D.h"
#include "TMultiGraph.h"


ClassImp(TLinearFitter)

//////////////////////////////////////////////////////////////////////////
//
// The Linear Fitter - fitting functions that are LINEAR IN PARAMETERS
//
// Linear fitter is used to fit a set of data points with a linear
// combination of specified functions. Note, that "linear" in the name
// stands only for the model dependency on parameters, the specified
// functions can be nonlinear.
// The general form of this kind of model is
//
//          y(x) = a[0] + a[1]*f[1](x)+...a[n]*f[n](x)
//
// Functions f are fixed functions of x. For example, fitting with a
// polynomial is linear fitting in this sense.
//
//                         The fitting method
//
// The fit is performed using the Normal Equations method with Cholesky
// decomposition.
//
//                         Why should it be used?
//
// The linear fitter is considerably faster than general non-linear
// fitters and doesn't require to set the initial values of parameters.
//
//                          Using the fitter:
//
// 1.Adding the data points:
//  1.1 To store or not to store the input data?
//      - There are 2 options in the constructor - to store or not
//        store the input data. The advantages of storing the data
//        are that you'll be able to reset the fitting model without
//        adding all the points again, and that for very large sets
//        of points the chisquare is calculated more precisely.
//        The obvious disadvantage is the amount of memory used to
//        keep all the points.
//      - Before you start adding the points, you can change the
//        store/not store option by StoreData() method.
//  1.2 The data can be added:
//      - simply point by point - AddPoint() method
//      - an array of points at once:
//        If the data is already stored in some arrays, this data
//        can be assigned to the linear fitter without physically
//        coping bytes, thanks to the Use() method of
//        TVector and TMatrix classes - AssignData() method
//
// 2.Setting the formula
//  2.1 The linear formula syntax:
//      -Additive parts are separated by 2 plus signes "++"
//       --for example "1 ++ x" - for fitting a straight line
//      -All standard functions, undrestood by TFormula, can be used
//       as additive parts
//       --TMath functions can be used too
//      -Functions, used as additive parts, shouldn't have any parameters,
//       even if those parameters are set.
//       --for example, if normalizing a sum of a gaus(0, 1) and a
//         gaus(0, 2), don't use the built-in "gaus" of TFormula,
//         because it has parameters, take TMath::Gaus(x, 0, 1) instead.
//      -Polynomials can be used like "pol3", .."polN"
//      -If fitting a more than 3-dimensional formula, variables should
//       be numbered as follows:
//       -- x0, x1, x2... For example, to fit  "1 ++ x0 ++ x1 ++ x2 ++ x3*x3"
//  2.2 Setting the formula:
//    2.2.1 If fitting a 1-2-3-dimensional formula, one can create a
//          TF123 based on a linear expression and pass this function
//          to the fitter:
//          --Example:
//            TLinearFitter *lf = new TLinearFitter();
//            TF2 *f2 = new TF2("f2", "x ++ y ++ x*x*y*y", -2, 2, -2, 2);
//            lf->SetFormula(f2);
//          --The results of the fit are then stored in the function,
//            just like when the TH1::Fit or TGraph::Fit is used
//          --A linear function of this kind is by no means different
//            from any other function, it can be drawn, evaluated, etc.
//    2.2.2 There is no need to create the function if you don't want to,
//          the formula can be set by expression:
//          --Example:
//            // 2 is the number of dimensions
//            TLinearFitter *lf = new TLinearFitter(2);
//            lf->SetFormula("x ++ y ++ x*x*y*y");
//          --That's the only way to go, if you want to fit in more
//            than 3 dimensions
//    2.2.3 The fastest functions to compute are polynomials and hyperplanes.
//          --Polynomials are set the usual way: "pol1", "pol2",...
//          --Hyperplanes are set by expression "hyp3", "hyp4", ...
//          ---The "hypN" expressions only work when the linear fitter
//             is used directly, not through TH1::Fit or TGraph::Fit.
//             To fit a graph or a histogram with a hyperplane, define
//             the function as "1++x++y".
//          ---A constant term is assumed for a hyperplane, when using
//             the "hypN" expression, so "hyp3" is in fact fitting with
//             "1++x++y++z" function.
//          --Fitting hyperplanes is much faster than fitting other
//            expressions so if performance is vital, calculate the
//            function values beforehand and give them to the fitter
//            as variables
//          --Example:
//            You want to fit "sin(x)|cos(2*x)" very fast. Calculate
//            sin(x) and cos(2*x) beforehand and store them in array *data.
//            Then:
//            TLinearFitter *lf=new TLinearFitter(2, "hyp2");
//            lf->AssignData(npoint, 2, data, y);
//
//  2.3 Resetting the formula
//    2.3.1 If the input data is stored (or added via AssignData() function),
//          the fitting formula can be reset without re-adding all the points.
//          --Example:
//            TLinearFitter *lf=new TLinearFitter("1++x++x*x");
//            lf->AssignData(n, 1, x, y, e);
//            lf->Eval()
//            //looking at the parameter significance, you see,
//            // that maybe the fit will improve, if you take out
//            // the constant term
//            lf->SetFormula("x++x*x");
//            lf->Eval();
//            ...
//    2.3.2 If the input data is not stored, the fitter will have to be
//          cleared and the data will have to be added again to try a
//          different formula.
//
// 3.Accessing the fit results
//  3.1 There are methods in the fitter to access all relevant information:
//      --GetParameters, GetCovarianceMatrix, etc
//      --the t-values of parameters and their significance can be reached by
//        GetParTValue() and GetParSignificance() methods
//  3.2 If fitting with a pre-defined TF123, the fit results are also
//      written into this function.
//
//////////////////////////////////////////////////////////////////////////



//______________________________________________________________________________
TLinearFitter::TLinearFitter()
{
   //default c-tor, input data is stored
   //If you don't want to store the input data,
   //run the function StoreData(kFALSE) after constructor

   fChisquare=0;
   fNpoints=0;
   fY2=0;
   fNfixed=0;
   fIsSet=kFALSE;
   fFormula=0;
   fFixedParams=0;
   fSpecial=0;
   fInputFunction=0;
   fStoreData=kTRUE;
}

//______________________________________________________________________________
TLinearFitter::TLinearFitter(Int_t ndim)
{
   //The parameter stands for number of dimensions in the fitting formula
   //The input data is stored. If you don't want to store the input data,
   //run the function StoreData(kFALSE) after constructor

   fNdim=ndim;
   fNpoints=0;
   fY2=0;
   fNfixed=0;
   fFixedParams=0;
   fFormula=0;
   fIsSet=kFALSE;
   fChisquare=0;
   fSpecial=0;
   fInputFunction=0;
   fStoreData=kTRUE;
}

//______________________________________________________________________________
TLinearFitter::TLinearFitter(Int_t ndim, const char *formula, Option_t *opt)
{
   //First parameter stands for number of dimensions in the fitting formula
   //Second parameter is the fitting formula: see class description for formula syntax
   //Options:
   //The option is to store or not to store the data
   //If you don't want to store the data, choose "" for the option, or run 
   //StoreData(kFalse) member function after the constructor

   fNdim=ndim;
   fNpoints=0;
   fChisquare=0;
   fY2=0;
   fNfixed=0;
   fFixedParams=0;
   fSpecial=0;
   fInputFunction=0;
   TString option=opt;
   option.ToUpper();
   if (option.Contains("D"))
      fStoreData=kTRUE;
   else
      fStoreData=kFALSE;

   SetFormula(formula);
}

//______________________________________________________________________________
TLinearFitter::TLinearFitter(TFormula *function, Option_t *opt)
{
   //This constructor uses a linear function. How to create it?
   //TFormula now accepts formulas of the following kind:
   //TFormula("f", "x++y++z++x*x"). Other than the look, it's in no
   //way different from the regular formula, it can be evaluated,
   //drawn, etc.
   //The option is to store or not to store the data
   //If you don't want to store the data, choose "" for the option, or run
   //StoreData(kFalse) member function after the constructor

   fNdim=function->GetNdim();
   if (!function->IsLinear()){
      Int_t number=function->GetNumber();
      if (number<299 || number>310){
         Error("TLinearFitter", "Trying to fit with a nonlinear function");
         return;
      }
   }
   fNpoints=0;
   fChisquare=0;
   fY2=0;
   fNfixed=0;
   fFixedParams=0;
   fSpecial=0;
   fFormula = 0;
   TString option=opt;
   option.ToUpper();
   if (option.Contains("D"))
      fStoreData=kTRUE;
   else
      fStoreData=kFALSE;
   fIsSet=kTRUE;
   SetFormula(function);
}

//______________________________________________________________________________
TLinearFitter::~TLinearFitter()
{
   // Linear fitter cleanup.

   if (fFormula)
      delete [] fFormula;
  
   fFormula = 0;
   delete [] fFixedParams;
   fFixedParams = 0;
   fInputFunction = 0;
   fFunctions.Delete();
   //delete fFunctions;

}

//______________________________________________________________________________
void TLinearFitter::AddPoint(Double_t *x, Double_t y, Double_t e)
{
   //Adds 1 point to the fitter.
   //First parameter stands for the coordinates of the point, where the function is measured
   //Second parameter - the value being fitted
   //Third parameter - weight(measurement error) of this point (=1 by default)

   Int_t size;
   fNpoints++;
   if (fStoreData){
      size=fY.GetNoElements();
      if (size<fNpoints){
         fY.ResizeTo(fNpoints+fNpoints/2);
         fE.ResizeTo(fNpoints+fNpoints/2);
         fX.ResizeTo(fNpoints+fNpoints/2, fNdim);
      }

      Int_t j=fNpoints-1;
      fY(j)=y;
      fE(j)=e;
      for (Int_t i=0; i<fNdim; i++)
         fX(j,i)=x[i];
   }
   //add the point to the design matrix, if the formula has been set
   if (!fFunctions.IsEmpty() || fInputFunction || fSpecial>199)
          AddToDesign(x, y, e);
   else if (!fStoreData)
      Error("AddPoint", "Point can't be added, because the formula hasn't been set and data is not stored");
}

//______________________________________________________________________________
void TLinearFitter::AssignData(Int_t npoints, Int_t xncols, Double_t *x, Double_t *y, Double_t *e)
{
   //This function is to use when you already have all the data in arrays
   //and don't want to copy them into the fitter. In this function, the Use() method
   //of TVectorD and TMatrixD is used, so no bytes are physically moved around.
   //First parameter - number of points to fit
   //Second parameter - number of variables in the model
   //Third parameter - the variables of the model, stored in the following way:
   //(x0(0), x1(0), x2(0), x3(0), x0(1), x1(1), x2(1), x3(1),...

   if (npoints<fNpoints){
      Error("AddData", "Those points are already added");
      return;
   }
   Bool_t same=kFALSE;
   if (fX.GetMatrixArray()==x && fY.GetMatrixArray()==y){
      if (e){
         if (fE.GetMatrixArray()==e)
            same=kTRUE;
      }
   }

   fX.Use(npoints, xncols, x);
   fY.Use(npoints, y);
   if (e)
      fE.Use(npoints, e);
   else {
      fE.ResizeTo(npoints);
      fE=1;
   }
   Int_t xfirst;
   if (!fFunctions.IsEmpty() || fInputFunction || fSpecial>199) {
      if (same)
         xfirst=fNpoints;

      else
         xfirst=0;
      for (Int_t i=xfirst; i<npoints; i++)
         AddToDesign(TMatrixDRow(fX, i).GetPtr(), fY(i), fE(i));
   }
   fNpoints=npoints;
}

//______________________________________________________________________________
void TLinearFitter::AddToDesign(Double_t *x, Double_t y, Double_t e)
{
   //Add a point to the AtA matrix and to the Atb vector.

   Int_t i, j, ii;
   y/=e;

   Double_t val[100];

   if ((fSpecial>100)&&(fSpecial<200)){
      //polynomial fitting
      Int_t npar=fSpecial-100;
      val[0]=1;
      for (i=1; i<npar; i++)
         val[i]=val[i-1]*x[0];
      for (i=0; i<npar; i++)
         val[i]/=e;
   } else {
      if (fSpecial>200){
         //Hyperplane fitting. Constant term is added
         Int_t npar=fSpecial-201;
         val[0]=1./e;
         for (i=0; i<npar; i++)
            val[i+1]=x[i]/e;
      } else {
         //general case
         for (ii=0; ii<fNfunctions; ii++){
            if (!fFunctions.IsEmpty()){
               TF1 *f1 = (TF1*)(fFunctions.UncheckedAt(ii));
               val[ii]=f1->EvalPar(0, x)/e;
            } else {
               TFormula *f=(TFormula*)fInputFunction->GetLinearPart(ii);
               val[ii]=f->EvalPar(0, x)/e;
            }
         }

      }
   }
   //additional matrices for numerical stability
   for (i=0; i<fNfunctions; i++){
      for (j=0; j<i; j++)
         fDesignTemp3(j, i)+=val[i]*val[j];
      fDesignTemp3(i, i)+=val[i]*val[i];
      fAtbTemp3(i)+=val[i]*y;

   }
   fY2Temp+=y*y;
   fIsSet=kTRUE;

   if (fNpoints % 100 == 0 && fNpoints>100){
      fDesignTemp2+=fDesignTemp3;
      fDesignTemp3.Zero();
      fAtbTemp2+=fAtbTemp3;
      fAtbTemp3.Zero();   
      if (fNpoints % 10000 == 0 && fNpoints>10000){
         fDesignTemp+=fDesignTemp2;
         fDesignTemp2.Zero();
         fAtbTemp+=fAtbTemp2;
         fAtbTemp2.Zero();
         fY2+=fY2Temp;
         fY2Temp=0;	 
	 if (fNpoints % 1000000 == 0 && fNpoints>1000000){
            fDesign+=fDesignTemp;
            fDesignTemp.Zero();
            fAtb+=fAtbTemp;
            fAtbTemp.Zero();
	 }
      }
   }
}

//______________________________________________________________________________
void TLinearFitter::Clear(Option_t * /*option*/)
{
   //Clears everything. Used in TH1::Fit().

   fParams.Clear();
   fParCovar.Clear();
   fTValues.Clear();
   fParSign.Clear();
   fDesign.Clear();
   fDesignTemp.Clear();
   fDesignTemp2.Clear();
   fDesignTemp3.Clear();
   fAtb.Clear();
   fAtbTemp.Clear();
   fAtbTemp2.Clear();
   fAtbTemp3.Clear();
   fFunctions.Clear();
   fInputFunction=0;
   fY.Clear();
   fX.Clear();
   fE.Clear();

   fNpoints=0;
   fNfunctions=0;
   fFormulaSize=0;
   fNdim=0;
   delete [] fFormula;
   fFormula=0;
   fIsSet=0;
   delete [] fFixedParams;
   fFixedParams=0;

   fChisquare=0;
   fY2=0;
   fSpecial=0;
}

//______________________________________________________________________________
void TLinearFitter::ClearPoints()
{
   //To be used when different sets of points are fitted with the same formula.

   fDesign.Zero();
   fAtb.Zero();
   fDesignTemp.Zero();
   fDesignTemp2.Zero();
   fDesignTemp3.Zero();
   fAtbTemp.Zero();
   fAtbTemp2.Zero();
   fAtbTemp3.Zero();

   fParams.Zero();
   fParCovar.Zero();
   fTValues.Zero();
   fParSign.Zero();

   for (Int_t i=0; i<fNfunctions; i++)
      fFixedParams[i]=0;
   fChisquare=0;
   fNpoints=0;

}

//______________________________________________________________________________
void TLinearFitter::Chisquare()
{
   //Calculates the chisquare.

   Int_t i, j;
   Double_t sumtotal2;
   Double_t temp, temp2;

   if (!fStoreData){
      sumtotal2 = 0;
      for (i=0; i<fNfunctions; i++){
         for (j=0; j<i; j++){
            sumtotal2 += 2*fParams(i)*fParams(j)*fDesign(j, i);
         }
         sumtotal2 += fParams(i)*fParams(i)*fDesign(i, i);
         sumtotal2 -= 2*fParams(i)*fAtb(i);
      }
      sumtotal2 += fY2;
   } else {
      sumtotal2 = 0;
      if (fInputFunction){
         for (i=0; i<fNpoints; i++){
            temp = fInputFunction->EvalPar(TMatrixDRow(fX, i).GetPtr());
            temp2 = (fY(i)-temp)*(fY(i)-temp);
            temp2 /= fE(i)*fE(i);
            sumtotal2 += temp2;
         }
      } else {
         sumtotal2 = 0;
         Double_t val[100];
         for (Int_t point=0; point<fNpoints; point++){
            temp = 0;
            if ((fSpecial>100)&&(fSpecial<200)){
               Int_t npar = fSpecial-100;
               val[0] = 1;
               for (i=1; i<npar; i++)
                  val[i] = val[i-1]*fX(point, 0);
               for (i=0; i<npar; i++)
                  temp += fParams(i)*val[i];
            } else {
               if (fSpecial>200) {
		  //hyperplane case
                  Int_t npar = fSpecial-201;
		  temp+=fParams(0);
                  for (i=0; i<npar; i++)
                     temp += fParams(i+1)*fX(point, i);
               } else {
                  for (j=0; j<fNfunctions; j++) {
                     TF1 *f1 = (TF1*)(fFunctions.UncheckedAt(j));
                     val[j] = f1->EvalPar(0, TMatrixDRow(fX, point).GetPtr());
                     temp += fParams(j)*val[j];
                  }
               }
            }
         temp2 = (fY(point)-temp)*(fY(point)-temp);
         temp2 /= fE(point)*fE(point);
         sumtotal2 += temp2;
         }
      }
   }
   fChisquare = sumtotal2;

}

//______________________________________________________________________________
void TLinearFitter::Eval()
{
   // Evaluate the function.

   Double_t e;
   if (fFunctions.IsEmpty()&&(!fInputFunction)&&(fSpecial<200)){
      Error("TLinearFitter::Eval", "The formula hasn't been set");
      return;
   }
   //
   if (!fIsSet){
      Bool_t update = UpdateMatrix();
      if (!update)
         return;
   }
   //

   fDesignTemp2+=fDesignTemp3;
   fDesignTemp+=fDesignTemp2;
   fDesign+=fDesignTemp;
   fDesignTemp3.Zero();
   fDesignTemp2.Zero();
   fDesignTemp.Zero();
   fAtbTemp2+=fAtbTemp3;
   fAtbTemp+=fAtbTemp2;
   fAtb+=fAtbTemp;
   fAtbTemp3.Zero();
   fAtbTemp2.Zero();
   fAtbTemp.Zero();

   fY2+=fY2Temp;
   fY2Temp=0;
   fParams.ResizeTo(fNfunctions);
   fTValues.ResizeTo(fNfunctions);
   fParSign.ResizeTo(fNfunctions);
   fParCovar.ResizeTo(fNfunctions,fNfunctions);

   fChisquare=0;

   //fixing fixed parameters, if there are any
   Int_t i, ii, j=0;
   if (fNfixed>0){
      for (ii=0; ii<fNfunctions; ii++)
         fDesignTemp(ii, fNfixed) = fAtb(ii);
      for (i=0; i<fNfunctions; i++){
         if (fFixedParams[i]){
            for (ii=0; ii<i; ii++)
               fDesignTemp(ii, j) = fDesign(ii, i);
            for (ii=i; ii<fNfunctions; ii++)
               fDesignTemp(ii, j) = fDesign(i, ii);
            j++;
            for (ii=0; ii<fNfunctions; ii++){
               fAtb(ii)-=fParams(i)*(fDesignTemp(ii, j-1));
            }
         }
      }
      for (i=0; i<fNfunctions; i++){
         if (fFixedParams[i]){
            for (ii=0; ii<fNfunctions; ii++){
               fDesign(ii, i) = 0;
               fDesign(i, ii) = 0;
            }
            fDesign (i, i) = 1;
            fAtb(i) = fParams(i);
         }
      }
   }

   TDecompChol chol(fDesign);
   Bool_t ok;
   TVectorD coef(fNfunctions);
   coef=chol.Solve(fAtb, ok);
   if (!ok){
      fParams.Zero();
      fParCovar.Zero();
      return;
   }
   fParams=coef;
   fParCovar=chol.Invert();

   for (i=0; i<fNfunctions; i++){
     fTValues(i) = fParams(i)/(TMath::Sqrt(fParCovar(i, i)));
     fParSign(i) = 2*(1-TMath::StudentI(TMath::Abs(fTValues(i)),fNpoints-fNfunctions));
   }

   if (fInputFunction){
      fInputFunction->SetParameters(fParams.GetMatrixArray());
      for (i=0; i<fNfunctions; i++){
         e = TMath::Sqrt(fParCovar(i, i));
         ((TF1*)fInputFunction)->SetParError(i, e);
      }
      if (!fObjectFit)
         ((TF1*)fInputFunction)->SetChisquare(GetChisquare());
      ((TF1*)fInputFunction)->SetNDF(fNpoints-fNfunctions+fNfixed);
   }

   //if parameters were fixed, change the design matrix back as it was before fixing
   j = 0;
   if (fNfixed>0){
      for (i=0; i<fNfunctions; i++){
         if (fFixedParams[i]){
            for (ii=0; ii<i; ii++){
               fDesign(ii, i) = fDesignTemp(ii, j);
               fAtb(ii) = fDesignTemp(ii, fNfixed);
            }
            for (ii=i; ii<fNfunctions; ii++){
               fDesign(i, ii) = fDesignTemp(ii, j);
               fAtb(ii) = fDesignTemp(ii, fNfixed);
            }
            j++;
         }
      }
   }
}

//______________________________________________________________________________
void TLinearFitter::FixParameter(Int_t ipar)
{
   //Fixes paramter #ipar at its current value.

   if (fParams.NonZeros()<1){
      Error("FixParameter", "no value available to fix the parameter");
      return;
   }
   if (ipar>fNfunctions || ipar<0){
      Error("FixParameter", "illegal parameter value");
      return;
   }
   if (fNfixed==fNfunctions) {
      Error("FixParameter", "no free parameters left");
      return;
   }
   fFixedParams[ipar] = 1;
   fNfixed++;
}

//______________________________________________________________________________
void TLinearFitter::FixParameter(Int_t ipar, Double_t parvalue)
{
   //Fixes parameter #ipar at value parvalue.

   if (ipar>fNfunctions || ipar<0){
      Error("FixParameter", "illegal parameter value");
      return;
   }
   if (fNfixed==fNfunctions) {
      Error("FixParameter", "no free parameters left");
      return;
   }
   fFixedParams[ipar] = 1;
   fParams(ipar) = parvalue;
   fNfixed++;
}

//______________________________________________________________________________
void TLinearFitter::ReleaseParameter(Int_t ipar)
{
   //Releases parameter #ipar.

    if (ipar>fNfunctions || ipar<0){
      Error("ReleaseParameter", "illegal parameter value");
      return;
   }
    if (!fFixedParams[ipar]){
       Warning("ReleaseParameter","This parameter is not fixed\n");
       return;
    } else {
       fFixedParams[ipar] = 0;
       fNfixed--;
    }
}

//______________________________________________________________________________
Double_t TLinearFitter::GetChisquare()
{
   // Get the Chisquare.

   if (fChisquare > 1e-16)
      return fChisquare;
   else {
      Chisquare();
      return fChisquare;
   }
}

//______________________________________________________________________________
void TLinearFitter::GetCovarianceMatrix(TMatrixD &matr)
{
   if (matr.GetNrows()!=fNfunctions || matr.GetNcols()!=fNfunctions){
      matr.ResizeTo(fNfunctions, fNfunctions);
   }
   matr = fParCovar;
}

//______________________________________________________________________________
void TLinearFitter::GetErrors(TVectorD &vpar)
{
   if (vpar.GetNoElements()!=fNfunctions) {
     vpar.ResizeTo(fNfunctions);
  }
   for (Int_t i=0; i<fNfunctions; i++)
      vpar(i) = TMath::Sqrt(fParCovar(i, i));

}

//______________________________________________________________________________
void TLinearFitter::GetParameters(TVectorD &vpar)
{
   if (vpar.GetNoElements()!=fNfunctions) {
     vpar.ResizeTo(fNfunctions);
  }
  vpar=fParams;
}

//______________________________________________________________________________
Double_t TLinearFitter::GetParError(Int_t ipar) const
{
   if (ipar<0 || ipar>fNfunctions) {
      Error("GetParError", "illegal value of parameter");
      return 0;
   }

   return TMath::Sqrt(fParCovar(ipar, ipar));
}

//______________________________________________________________________________
void TLinearFitter::SetDim(Int_t ndim)
{
   //set the number of dimensions

   fNdim=ndim;
   fY.ResizeTo(ndim+1);
   fX.ResizeTo(ndim+1, ndim);
   fE.ResizeTo(ndim+1);

   fNpoints=0;
   fIsSet=kFALSE;
}

//______________________________________________________________________________
void TLinearFitter::SetFormula(const char *formula)
{
  //Additive parts should be separated by "++".
  //Examples (ai are parameters to fit):
  //1.fitting function: a0*x0 + a1*x1 + a2*x2
  //  input formula "x0++x1++x2"
  //2.TMath functions can be used:
  //  fitting function: a0*TMath::Gaus(x0, 0, 1) + a1*x1
  //  input formula:    "TMath::Gaus(x0, 0, 1)++x1"
  //fills the array of functions

   Int_t size, special = 0;
   Int_t i, j;

   Int_t len = strlen(formula);
   fFormulaSize = len;
   fFormula = new char[len+1];
   strcpy(fFormula, formula);
   fSpecial = 0;
   //in case of a hyperplane:
   char *fstring;
   fstring = (char *)strstr(fFormula, "hyp");
   if (fstring!=NULL){
      fstring+=3;
      sscanf(fstring, "%d", &size);
      //+1 for the constant term
      size++;
      fSpecial=200+size;
   }

   TString sstring(fFormula);
   sstring = sstring.ReplaceAll("++", 2, "|", 1);

   char *copyformula=new char[fFormulaSize+30];
   strcpy(copyformula, sstring.Data());

   //count the number of functions
   fstring=strtok(copyformula, "|");
   j=0;
   while (fstring!=NULL){
      j++;
      fstring=strtok(NULL, "|");
   }

   //change the size of functions array and clear it
   if (!fFunctions.IsEmpty())
      fFunctions.Clear();

   fNfunctions=j;
   fFunctions.Expand(fNfunctions);

   //replace xn by [n]
   char pattern[5];
   char replacement[6]; 

   for (i=0; i<fNdim; i++){
      sprintf(pattern, "x%d", i);
      sprintf(replacement, "[%d]", i);
      sstring = sstring.ReplaceAll(pattern, Int_t(i/10)+2, replacement, Int_t(i/10)+3);
   }
   //replace the regular x, y, z

   sstring = sstring.ReplaceAll("y", 1, "[1]", 3);
   sstring = sstring.ReplaceAll("z", 1, "[2]", 3);
   //check in order not to replace the x in exp
   fstring = (char*)strchr(sstring.Data(), 'x');
   while (fstring){
      Int_t offset = fstring - sstring.Data();
      if (*(fstring-1)!='e' && *(fstring+1)!='p')
         sstring.Replace(fstring - sstring.Data(), 1, "[0]",3);
      else
         offset++;
      fstring = (char*)strchr(sstring.Data()+offset, 'x');
   }


   //fill the array of functions
   j=0;
   if (fSpecial==0){
      //in case it's not a hyperplane
      strcpy(copyformula, sstring.Data());
      fstring=strtok(copyformula, "|");
      while (fstring!=NULL){
         TF1 *f=new TF1("f", fstring, -1, 1);
         special=f->GetNumber();
         if (!f) {
            Error("TLinearFitter", "f not allocated");
            return;
         }
         fFunctions.Add(f);
         fstring=strtok(NULL, "|");
      }

      if ((fNfunctions==1)&&(special>299)&&(special<310)){
         //if fitting a polynomial
         size=special-299;
         fSpecial=100+size;
      } else
         size=fNfunctions;
   }
   fNfunctions=size;
   //change the size of design matrix
   fDesign.ResizeTo(size, size);
   fAtb.ResizeTo(size);
   fDesignTemp.ResizeTo(size, size);
   fDesignTemp2.ResizeTo(size, size);
   fDesignTemp3.ResizeTo(size, size);
   fAtbTemp.ResizeTo(size);
   fAtbTemp2.ResizeTo(size);
   fAtbTemp3.ResizeTo(size);
   //
   if (fFixedParams)
      delete [] fFixedParams;
   fFixedParams=new Bool_t(size);
   fDesign.Zero();
   fAtb.Zero();
   fDesignTemp.Zero();
   fDesignTemp2.Zero();
   fDesignTemp3.Zero();
   fAtbTemp.Zero();
   fAtbTemp2.Zero();
   fAtbTemp3.Zero();
   fY2Temp=0;
   fY2=0;
   for (i=0; i<size; i++)
      fFixedParams[i]=0;
   fIsSet=kFALSE;
   fChisquare=0;

}

//______________________________________________________________________________
void TLinearFitter::SetFormula(TFormula *function)
{
   //Set the fitting function.

   Int_t special, size;
   fInputFunction=function;
   fNfunctions=fInputFunction->GetNpar();
   special=fInputFunction->GetNumber();

   if ((special>299)&&(special<310)){
      //if fitting a polynomial
      size=special-299;
      fSpecial=100+size;
   } else
      size=fNfunctions;

   fNfunctions=size;
   //change the size of design matrix
   fDesign.ResizeTo(size, size);
   fAtb.ResizeTo(size);
   fDesignTemp.ResizeTo(size, size);
   fAtbTemp.ResizeTo(size);

   fDesignTemp2.ResizeTo(size, size);
   fDesignTemp3.ResizeTo(size, size);

   fAtbTemp2.ResizeTo(size);
   fAtbTemp3.ResizeTo(size);
   //
   if (fFixedParams)
      delete [] fFixedParams;
   fFixedParams=new Bool_t[size];
   fDesign.Zero();
   fAtb.Zero();
   fDesignTemp.Zero();
   fAtbTemp.Zero();

   fDesignTemp2.Zero();
   fDesignTemp3.Zero();

   fAtbTemp2.Zero();
   fAtbTemp3.Zero();
   fY2Temp=0;
   fY2=0;
   for (Int_t i=0; i<size; i++)
      fFixedParams[i]=0;
   fIsSet=kFALSE;
   fChisquare=0;

}

//______________________________________________________________________________
Bool_t TLinearFitter::UpdateMatrix()
{

   //Update the design matrix after the formula has been changed.

     if (fStoreData){
        for (Int_t i=0; i<fNpoints; i++){
           AddToDesign(TMatrixDRow(fX, i).GetPtr(), fY(i), fE(i));
        }
        return 1;
     } else {
        Error("UpdateMatrix", "matrix can't be updated - input points not stored");
        return 0;
     }
}

//______________________________________________________________________________
Int_t TLinearFitter::ExecuteCommand(const char *command, Double_t * /*args*/, Int_t /*nargs*/)
{
   //To use in TGraph::Fit and TH1::Fit().

   if (!strcmp(command, "FitGraph"))      GraphLinearFitter();
   if (!strcmp(command, "FitHist"))       HistLinearFitter();
   if (!strcmp(command, "FitGraph2D"))    Graph2DLinearFitter();
   if (!strcmp(command, "FitMultiGraph")) MultiGraphLinearFitter();

   return 0;
}

//______________________________________________________________________________
void TLinearFitter::PrintResults(Int_t level, Double_t /*amin*/) const
{
   // Level = 3 (to be consistent with minuit)  prints parameters and parameter
   // errors.

   if (level==3){
      printf("Fitting results:\nParameters:\nNO.\t\tVALUE\t\tERROR\n");
      for (Int_t i=0; i<fNfunctions; i++){
	printf("%d\t%f\t%f\n", i, fParams(i), TMath::Sqrt(fParCovar(i, i)));
      }
   }
}

//______________________________________________________________________________
void TLinearFitter::GraphLinearFitter()
{
   //Used in TGraph::Fit().

   StoreData(kFALSE);
   TGraph *grr=(TGraph*)GetObjectFit();
   TF1 *f1=(TF1*)GetUserFunc();
   Foption_t Foption=GetFitOption();

   //Int_t np=0;
   Double_t *x=grr->GetX();
   Double_t *y=grr->GetY();
   Double_t e;

   //set the fitting formula
   SetDim(1);
   SetFormula(f1);

   //put the points into the fitter
   Int_t n=grr->GetN();
   for (Int_t i=0; i<n; i++){
      if (!f1->IsInside(&x[i])) continue;
      e=grr->GetErrorY(i);
      if (e<0 || Foption.W1)
         e=1;
      AddPoint(&x[i], y[i], e);
   }

   
   Eval();

   //calculate the precise chisquare
   if (!Foption.Nochisq){
      Double_t temp, temp2, sumtotal=0;
      for (Int_t i=0; i<n; i++){
         if (!f1->IsInside(&x[i])) continue;
         temp=f1->Eval(x[i]);
         temp2=(y[i]-temp)*(y[i]-temp);
         e=grr->GetErrorY(i);
         if (e<0 || Foption.W1)
            e=1;
         temp2/=(e*e);

         sumtotal+=temp2;
      }
      fChisquare=sumtotal;
      f1->SetChisquare(fChisquare);
   }
}

//______________________________________________________________________________
void TLinearFitter::Graph2DLinearFitter()
{
   StoreData(kFALSE);

   TGraph2D *gr=(TGraph2D*)GetObjectFit();
   TF2 *f2=(TF2*)GetUserFunc();

   //TGraph2DErrors *gre=0;
   //if (gr->InheritsFrom(TGraph2DErrors::Class())) gre=(TGraph2DErrors*)gr;


   Foption_t Foption=GetFitOption();
   Int_t n        = gr->GetN();
   Double_t *gx   = gr->GetX();
   Double_t *gy   = gr->GetY();
   Double_t *gz   = gr->GetZ();
   //  Double_t fxmin = f2->GetXmin();
   //Double_t fxmax = f2->GetXmax();
   //Double_t fymin = f2->GetYmin();
   //Double_t fymax = f2->GetYmax();
   Double_t x[2];
   Double_t z, e;

   SetDim(2);
   SetFormula(f2);

   for (Int_t bin=0;bin<n;bin++) {
      x[0] = gx[bin];
      x[1] = gy[bin];
      if (!f2->IsInside(x)) {
         continue;
      }
      z   = gz[bin];
      //if (gre && !Foption.W1) e  = gr->GetErrorZ(bin);
      //else e = 1;
      e=gr->GetErrorZ(bin);
      if (e<0 || Foption.W1)
         e=1;
      AddPoint(x, z, e);
   }

   Eval();

   if (!Foption.Nochisq){
      Double_t temp, temp2, sumtotal=0;
      for (Int_t bin=0; bin<n; bin++){
         x[0] = gx[bin];
         x[1] = gy[bin];
         if (!f2->IsInside(x)) {
            continue;
         }
         z   = gz[bin];

         temp=f2->Eval(x[0], x[1]);
         temp2=(z-temp)*(z-temp);
         //if (gre && !Foption.W1)
         //   e=gr->GetErrorZ(bin);
         //else
         //   e=1;
         e=gr->GetErrorZ(bin);
         if (e<0 || Foption.W1)
            e=1;
         temp2/=(e*e);

         sumtotal+=temp2;
      }
      fChisquare=sumtotal;
      f2->SetChisquare(fChisquare);
   }
}

//______________________________________________________________________________
void TLinearFitter::MultiGraphLinearFitter()
{

   Int_t n, i;
   Double_t *gx, *gy;
   Double_t e;
   TVirtualFitter *grFitter = TVirtualFitter::GetFitter();
   TMultiGraph *mg     = (TMultiGraph*)grFitter->GetObjectFit();
   TF1 *f1   = (TF1*)grFitter->GetUserFunc();
   Foption_t Foption = grFitter->GetFitOption();

   SetDim(1);
   SetFormula(f1);

   TGraph *gr;
   TIter next(mg->GetListOfGraphs());
   while ((gr = (TGraph*) next())) {
      n        = gr->GetN();
      gx   = gr->GetX();
      gy   = gr->GetY();
      for (i=0; i<n; i++){
         if (!f1->IsInside(&gx[i])) continue;
         e=gr->GetErrorY(i);
         if (e<0 || Foption.W1)
            e=1;
         AddPoint(&gx[i], gy[i], e);
      }
   }

   Eval();

   //calculate the chisquare
   if (!Foption.Nochisq){
      Double_t temp, temp2, sumtotal=0;
      next.Reset();
      while((gr = (TGraph*)next())) {
         n        = gr->GetN();
         gx   = gr->GetX();
         gy   = gr->GetY();
         for (i=0; i<n; i++){
            if (!f1->IsInside(&gx[i])) continue;
            temp=f1->Eval(gx[i]);
            temp2=(gy[i]-temp)*(gy[i]-temp);
            e=gr->GetErrorY(i);
            if (e<0 || Foption.W1)
               e=1;
            temp2/=(e*e);

            sumtotal+=temp2;
         }

      }
      fChisquare=sumtotal;
      f1->SetChisquare(fChisquare);
   }
}

//______________________________________________________________________________
void TLinearFitter::HistLinearFitter()
{
   // Minimization function for H1s using a Chisquare method.

   StoreData(kFALSE);
   Double_t cu,eu;
   // Double_t dersum[100], grad[100];
   Double_t x[3];
   Int_t bin,binx,biny,binz;
   //   Axis_t binlow, binup, binsize;

   TH1 *hfit = (TH1*)GetObjectFit();
   TF1 *f1   = (TF1*)GetUserFunc();

   Foption_t Foption = GetFitOption();
   //   printf("%s\n", f1->GetName());
   SetDim(hfit->GetDimension());
   SetFormula(f1);

   Int_t hxfirst = GetXfirst();
   Int_t hxlast  = GetXlast();
   Int_t hyfirst = GetYfirst();
   Int_t hylast  = GetYlast();
   Int_t hzfirst = GetZfirst();
   Int_t hzlast  = GetZlast();
   TAxis *xaxis  = hfit->GetXaxis();
   TAxis *yaxis  = hfit->GetYaxis();
   TAxis *zaxis  = hfit->GetZaxis();

   for (binz=hzfirst;binz<=hzlast;binz++) {
      x[2]  = zaxis->GetBinCenter(binz);
      for (biny=hyfirst;biny<=hylast;biny++) {
         x[1]  = yaxis->GetBinCenter(biny);
         for (binx=hxfirst;binx<=hxlast;binx++) {
            x[0]  = xaxis->GetBinCenter(binx);
            if (!f1->IsInside(x)) continue;
            bin = hfit->GetBin(binx,biny,binz);
            cu  = hfit->GetBinContent(bin);
            if (Foption.W1) {
               eu = 1;
            } else {
               eu  = hfit->GetBinError(bin);
               if (eu <= 0) continue;
            }
            AddPoint(x, cu, eu);

         }
      }
   }

   Eval();

   if (!Foption.Nochisq){
      Double_t temp, temp2, sumtotal=0;
      for (binz=hzfirst;binz<=hzlast;binz++) {
         x[2]  = zaxis->GetBinCenter(binz);
         for (biny=hyfirst;biny<=hylast;biny++) {
            x[1]  = yaxis->GetBinCenter(biny);
            for (binx=hxfirst;binx<=hxlast;binx++) {
               x[0]  = xaxis->GetBinCenter(binx);
               if (!f1->IsInside(x)) continue;
               bin = hfit->GetBin(binx,biny,binz);
               cu  = hfit->GetBinContent(bin);

               if (Foption.W1) {
               eu = 1;
               } else {
                  eu  = hfit->GetBinError(bin);
                  if (eu <= 0) continue;
               }
               temp=f1->EvalPar(x);
               temp2=(cu-temp)*(cu-temp);
               temp2/=(eu*eu);
               sumtotal+=temp2;
            }
         }
      }

      fChisquare=sumtotal;
      f1->SetChisquare(fChisquare);
   }
}
