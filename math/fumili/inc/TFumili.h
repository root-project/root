// @(#)root/fumili:$Id$
// Author: Rene Brun   31/08/99

/////////////////////////////////////////////////////////////////////////
//                                                                     //
//  TFumili                                                            //
//                                                                     //
//  The FUMILI Minimization package                                    //
//                                                                     //
///////////////////////////////////////////////////////////////////////// 

#ifndef ROOT_TFumili
#define ROOT_TFumili

#ifndef ROOT_TVirtualFitter
#include "TVirtualFitter.h"
#endif

class TF1;
 
class TFumili : public  TVirtualFitter {
private:
   Int_t fMaxParam;     //
   Int_t fNlog;         //
   Int_t fNfcn;         // Number of FCN calls;
   Int_t fNED1;         // Number of experimental vectors X=(x1,x2,...xK)
   Int_t fNED2;         // K - Length of vector X plus 2 (for chi2)
   Int_t fNED12;        // fNED1+fNED2
   Int_t fNpar;         //  fNpar - number of parameters
   Int_t fNstepDec;     //  fNstepDec - maximum number of step decreasing counter
   Int_t fNlimMul;      //  fNlimMul - after fNlimMul successful iterations permits four-fold increasing of fPL
   Int_t fNmaxIter;     //  fNmaxIter - maximum number of iterations
   Int_t fLastFixed;    // Last fixed parameter number
   Int_t fENDFLG;       // End flag of fit 
   Int_t fINDFLG[5];    // internal flags;
  
  
   Bool_t fGRAD;        // user calculated gradients  
   Bool_t fWARN;        // warnings
   Bool_t fDEBUG;       // debug info
   Bool_t fLogLike;     // LogLikelihood flag
   Bool_t fNumericDerivatives; //

   Double_t *fZ0;       //[fMaxParam2] Matrix of approximate second derivatives of objective function
                        // This matrix is diagonal and always contain only variable parameter's
                        // derivatives
   Double_t *fZ;        //[fMaxParam2] Invers fZ0 matrix - covariance matrix
   Double_t *fGr;       //[fMaxParam] Gradients of objective function
   Double_t *fParamError; //[fMaxParam] Parameter errors
   Double_t *fSumLog;   //[fNlog]
   Double_t *fEXDA;     //[fNED12] experimental data poInt_ter
  
   //  don't calculate parameter errors - take them from fParamError array
   Double_t *fA;        //[fMaxParam] Fit parameter array
   Double_t *fPL0;      //[fMaxParam] Step initial bounds
   Double_t *fPL;       //[fMaxParam] Limits for parameters step. If <0, then parameter is fixed
 
   // Defines multidimensional parallelepiped with center in param. vector
   Double_t *fDA;       //[fMaxParam] Parameter step
   Double_t *fAMX;      //[fMaxParam] Maximum param value
   Double_t *fAMN;      //[fMaxParam] Minimum param value
   Double_t *fR;        //[fMaxParam] Correlation factors
  
   Double_t *fDF;       //[fMaxParam] // First derivatives of theoretical function 
   Double_t *fCmPar;    //[fMaxParam] parameters of commands

   Double_t fS;         //  fS - objective function value (return)
   Double_t fEPS;       //  fEPS - required precision of parameters. If fEPS<0 then 
   Double_t fRP;        // Precision of fit ( machine zero on CDC 6000) quite old yeh?
   Double_t fAKAPPA;    //
   Double_t fGT;        // Expected function change in next iteration
   TString *fANames;    //[fMaxParam] Parameter names
   TString fCword;      //  Command string


//  TF1 *fTFNF1;         //Pointer to theoretical function
//  void (*fFCN) (Int_t &, Double_t *, Double_t &f, Double_t *, Int_t); //
//  //wrapper function to calculate functional value, gradients and Z-matrix
//  Double_t (*fTFN)(Double_t *, Double_t *, Double_t*); // Wrapper function for TFN

public:

   TFumili(Int_t maxpar=25);
   virtual  ~TFumili();

   void             BuildArrays();
   virtual Double_t Chisquare(Int_t npar, Double_t *params) const;
   virtual void     Clear(Option_t *opt=""); 
   void             DeleteArrays();
   void             Derivatives(Double_t*,Double_t*);
   Int_t            Eval(Int_t& npar, Double_t *grad, Double_t &fval, Double_t *par, Int_t flag); // Evaluate the minimisation function
   Double_t         EvalTFN(Double_t *,Double_t*);
   virtual Int_t    ExecuteCommand(const char *command, Double_t *args, Int_t nargs);
   Int_t            ExecuteSetCommand(Int_t ); 
   virtual void     FitChisquare(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
   virtual void     FitChisquareI(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
   virtual void     FitLikelihood(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
   virtual void     FitLikelihoodI(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
   virtual void     FixParameter(Int_t ipar); 
   virtual Double_t *GetCovarianceMatrix() const;
   virtual Double_t GetCovarianceMatrixElement(Int_t i, Int_t j) const;
   virtual Int_t    GetErrors(Int_t ipar,Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &globcc) const;
   virtual Int_t    GetNumberTotalParameters() const;
   virtual Int_t    GetNumberFreeParameters() const;
   Double_t*        GetPL0() const { return fPL0;} 
   virtual Double_t GetParError(Int_t ipar) const;
   virtual Double_t GetParameter(Int_t ipar) const ;
   virtual Int_t    GetParameter(Int_t ipar,char *name,Double_t &value,Double_t &verr,Double_t &vlow, Double_t &vhigh) const;
   virtual const char *GetParName(Int_t ipar) const;
   virtual Int_t    GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx) const;
   virtual Double_t GetSumLog(Int_t );
   Double_t*        GetZ() const { return fZ;}
   void             InvertZ(Int_t); 
   virtual Bool_t   IsFixed(Int_t ipar) const;
   Int_t            Minimize(); 
   virtual void     PrintResults(Int_t k,Double_t p) const;
   virtual void     ReleaseParameter(Int_t ipar); 
   Int_t            SGZ();
   void             SetData(Double_t *,Int_t,Int_t);
   virtual void     SetFitMethod(const char *name);
   virtual Int_t    SetParameter(Int_t ipar,const char *parname,Double_t value,Double_t verr,Double_t vlow, Double_t vhigh);
   void             SetParNumber(Int_t ParNum) { fNpar = ParNum;};

   ClassDef(TFumili,0) //The FUMILI Minimization package
};

R__EXTERN TFumili * gFumili;
#endif 




