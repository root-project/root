// @(#)root/minuit:$Name:  $:$Id: TLinearFitter.h,v 1.8 2003/05/15 19:18:31 brun Exp $
// Author: Anna Kreshuk 04/03/2005


/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVectorD.h"
#include "TMatrixD.h"
#include "TFormula.h"
#include "TVirtualFitter.h"

class TLinearFitter: public TVirtualFitter {   
   
 private:
   TVectorD     fParams;         //vector of parameters
   TMatrixDSym  fParCovar;       //matrix of parameters' covariances
   TVectorD     fTValues;        //T-Values of parameters
   TVectorD     fParSign;        //significance levels of parameters
   TMatrixDSym  fDesign;         //matrix AtA
   TMatrixDSym  fDesignTemp;     //temporary matrix, used for num.stability
   TMatrixDSym  fDesignTemp2;
   TMatrixDSym  fDesignTemp3;

   TVectorD     fAtb;            //vector Atb
   TVectorD     fAtbTemp;        //temporary vector, used for num.stability
   TVectorD     fAtbTemp2;
   TVectorD     fAtbTemp3;

   Bool_t       *fFixedParams;   //array of fixed/released params
   
   TObjArray    fFunctions;      //array of basis functions
   TVectorD     fY;              //the values being fit 
   Double_t     fY2;             //sum of square of y, used for chisquare
   Double_t     fY2Temp;         //temporary variable used for num.stability
   TMatrixD     fX;              //values of x
   TVectorD     fE;              //the errors if they are known
   TFormula     *fInputFunction; //the function being fit
   
   Int_t        fNpoints;        //number of points
   Int_t        fNfunctions;     //number of basis functions
   Int_t        fFormulaSize;    //length of the formula
   Int_t        fNdim;           //number of dimensions in the formula
   Int_t        fNfixed;         //number of fixed parameters
   Int_t        fSpecial;        //=100+n if fitting a polynomial of deg.n
                                 //=200+n if fitting an n-dimensional hyperplane 
   char         *fFormula;       //the formula
   Bool_t       fIsSet;          //Has the formula been set?
   Bool_t       fStoreData;      //Is the data stored?
   
   Double_t     fChisquare;      //Chisquare of the fit

   Double_t     fEsum;           //used to check the num.stability of chisquare
   Double_t     fEcorsum;        //used to check the num.stability of chisquare

      
 private:
   void AddToDesign(Double_t *x, Double_t y, Double_t e);
   void GraphLinearFitter();
   void Graph2DLinearFitter();
   void HistLinearFitter();
   void MultiGraphLinearFitter();
   
 public:
   TLinearFitter();
   TLinearFitter(Int_t ndim, const char *formula, Option_t *opt="D");
   TLinearFitter(Int_t ndim);
   TLinearFitter(TFormula *function, Option_t *opt="D");
   virtual ~TLinearFitter();
   
   virtual void       AddPoint(Double_t *x, Double_t y, Double_t e=1);
   virtual void       AssignData(Int_t npoints, Int_t xncols, Double_t *x, Double_t *y, Double_t *e=0);

   virtual void       Clear(Option_t *option="");
   virtual void       ClearPoints();
   virtual void       Chisquare();
   virtual void       Eval();
   virtual Int_t      ExecuteCommand(const char *command, Double_t */*args*/, Int_t /*nargs*/);
   virtual void       FixParameter(Int_t ipar);
   virtual void       FixParameter(Int_t ipar, Double_t parvalue);
   virtual Double_t   GetChisquare();
   virtual Double_t*  GetCovarianceMatrix() const {return 0;}
   virtual void       GetCovarianceMatrix(TMatrixD &matr);
   virtual Double_t   GetCovarianceMatrixElement(Int_t i, Int_t j) const {return fParCovar(i, j);}
   virtual void       GetErrors(TVectorD &vpar);
   virtual Int_t      GetNumberTotalParameters() const {return fNfunctions;}
   virtual Int_t      GetNumberFreeParameters() const {return fNfunctions-fNfixed;}
   virtual void       GetParameters(TVectorD &vpar);
   virtual Double_t   GetParameter(Int_t ipar) const {return fParams(ipar);}
   virtual Double_t   GetParError(Int_t ipar) const;
   virtual Double_t   GetParTValue(Int_t ipar) const {return fTValues(ipar);}
   virtual Double_t   GetParSignificance(Int_t ipar) const {return fParSign(ipar);}
   virtual Bool_t     IsFixed(Int_t ipar) const {return fFixedParams[ipar];}
   virtual void       PrintResults(Int_t level, Double_t amin=0) const;
   virtual void       ReleaseParameter(Int_t ipar);
   virtual void       SetDim(Int_t n);
   virtual void       SetFormula(const char* formula);
   virtual void       SetFormula(TFormula *function);
   virtual void       StoreData(Bool_t store) {fStoreData=store;}
   virtual Bool_t     UpdateMatrix();

   //dummy functions for TVirtualFitter:

   virtual Double_t  Chisquare(Int_t /*npar*/, Double_t */*params*/) const {return 0;}
   virtual Int_t     GetErrors(Int_t /*ipar*/,Double_t & /*eplus*/, Double_t & /*eminus*/, Double_t & /*eparab*/, Double_t & /*globcc*/) const {return 0;}
   virtual Int_t     GetParameter(Int_t /*ipar*/,char* /*name*/,Double_t& /*value*/,Double_t& /*verr*/,Double_t& /*vlow*/, Double_t& /*vhigh*/) const  {return 0;}
   virtual Int_t     GetStats(Double_t& /*amin*/, Double_t& /*edm*/, Double_t& /*errdef*/, Int_t& /*nvpar*/, Int_t& /*nparx*/) const {return 0;}
   virtual Double_t  GetSumLog(Int_t /*i*/) {return 0;}
   virtual void      SetFitMethod(const char */*name*/) {;}
   virtual Int_t     SetParameter(Int_t /*ipar*/,const char */*parname*/,Double_t /*value*/,Double_t /*verr*/,Double_t /*vlow*/, Double_t /*vhigh*/) {return 0;}
   

   ClassDef(TLinearFitter, 1) //fit a set of data points with a linear combination of functions
};

