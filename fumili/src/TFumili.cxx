// @(#)root/fumili:$Name:  $:$Id: TFumili.cxx,v 1.5 2003/05/06 08:23:42 rdm Exp $
// Author: Stanislav Nesterov  07/05/2003

//BEGIN_HTML
/*
<H2>FUMILI minimization package</H2>
<p>FUMILI is used to minimize Chi-square function or to search maximum of
likelihood function.

<p>Experimentally measured values $F_i$ are fitted with theoretical
functions $f_i({\vec x}_i,\vec\theta\,\,)$, where ${\vec x}_i$ are
coordinates, and $\vec\theta$ -- vector of parameters.

<p>For better convergence Chi-square function has to be the following form

<p>$$
{\chi^2\over2}={1\over2}\sum^n_{i=1}\left(f_i(\vec
x_i,\vec\theta\,\,)-F_i\over\sigma_i\right)^2 \eqno(1)
$$
<p>where $\sigma_i$ are errors of measured function.

<p>The minimum condition is
<p>$$
{\partial\chi^2\over\partial\theta_i}=\sum^n_{j=1}{1\over\sigma^2_j}\cdot
{\partial f_j\over\partial\theta_i}\left[f_j(\vec
x_j,\vec\theta\,\,)-F_j\right]=0,\qquad i=1\ldots m\eqno(2)
$$
<p>where m is the quantity of parameters.

<p>Expanding left part of (2) over parameter increments and
retaining only linear terms one gets
<p>$$
\left(\partial\chi^2\over\theta_i\right)_{\vec\theta={\vec\theta}^0}
+\sum_k\left(\partial^2\chi^2\over\partial\theta_i\partial\theta_k\right)_{
\vec\theta={\vec\theta}^0}\cdot(\theta_k-\theta_k^0)
= 0\eqno(3)
$$

 <p>Here ${\vec\theta}_0$ is some initial value of parameters. In general
case:
<p>$$
{\partial^2\chi^2\over\partial\theta_i\partial\theta_k}=
\sum^n_{j=1}{1\over\sigma^2_j}{\partial f_j\over\theta_i}
{\partial f_j\over\theta_k} + 
\sum^n_{j=1}{(f_j - F_j)\over\sigma^2_j}\cdot 
{\partial^2f_j\over\partial\theta_i\partial\theta_k}\eqno(4)
$$

<p>In FUMILI algorithm for second derivatives of Chi-square approximate
expression is used when last term in (4) is discarded. It is often
done, not always wittingly, and sometimes causes troubles, for example,
if user wants to limit parameters with positive values by writing down
$\theta_i^2$ instead of $\theta_i$. FUMILI will fail if one tries
minimize $\chi^2 = g^2(\vec\theta)$ where g is arbitrary function.

<p>Approximate value is:
<p>$${\partial^2\chi^2\over\partial\theta_i\partial\theta_k}\approx
Z_{ik}=
\sum^n_{j=1}{1\over\sigma^2_j}{\partial f_j\over\theta_i}
{\partial f_j\over\theta_k}\eqno(5)
$$

<p>Then the equations for parameter increments are
<p>$$\left(\partial\chi^2\over\partial\theta_i\right)_{\vec\theta={\vec\theta}^0}
+\sum_k Z_{ik}\cdot(\theta_k-\theta^0_k) = 0, 
\qquad i=1\ldots m\eqno(6)
$$

<p>Remarkable feature of algorithm is the technique for step
restriction. For an initial value of parameter ${\vec\theta}^0$ a
parallelepiped $P_0$ is built with the center at ${\vec\theta}^0$ and
axes parallel to coordinate axes $\theta_i$. The lengths of
parallelepiped sides along i-th axis is $2b_i$, where $b_i$ is such a
value that the functions $f_j(\vec\theta)$ are quasi-linear all over
the parallelepiped. 

<p>FUMILI takes into account simple linear inequalities in the form:
$$
\theta_i^{\rm min}\le\theta_i\le\theta^{\rm max}_i\eqno(7)
$$

<p>They form parallelepiped $P$ ($P_0$ may be deformed by $P$). 
Very similar step formulae are used in FUMILI for negative logarithm
of the likelihood function with the same idea - linearization of
functional argument.

 */
//END_HTML



#include "TROOT.h"
#include "TFumili.h"
#include "TMath.h"
#include "TF1.h"
#include "TH1.h"
#include "TGraph.h"

#include "Riostream.h"


extern void H1FitChisquareFumili(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void H1FitLikelihoodFumili(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);
extern void GraphFitChisquareFumili(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag);


ClassImp(TFumili);

TFumili *gFumili=0;
// Machine dependent values  FIXME!!
// But don't set min=max=0 if param is unlimited
static const Double_t kMAXDOUBLE=1e300;
static const Double_t kMINDOUBLE=-1e300;


//______________________________________________________________________________
TFumili::TFumili(Int_t maxpar) 
{//----------- FUMILI constructor ---------
  // maxpar is the maximum number of parameters used with TFumili object
  //
  fMaxParam = TMath::Max(maxpar,25);
  if (fMaxParam>200) fMaxParam=25;
  fMaxParam2 *= fMaxParam;
  BuildArrays();
  
  fNumericDerivatives = true;
  fLogLike = false;
  fNpar    = fMaxParam;
  fGRAD    = false;
  fWARN    = true;
  fDEBUG   = false;
  fNlog    = 0;
  fSumLog  = 0;
  fNED1    = 0;
  fNED2    = 0;
  fNED12   = fNED1+fNED2;
  fEXDA    = 0;
  fFCN     = 0;
  fNfcn    = 0;
  fRP      = 1.e-15; //precision
  fS       = 1e10;
  fEPS     =0.01;
  fENDFLG  = 0;
  fNlimMul = 2;
  fNmaxIter= 150;
  fNstepDec= 5;
  fLastFixed = -1;
  
  SetName("Fumili");
  gFumili = this;
  gROOT->GetListOfSpecials()->Add(gFumili);

}

//______________________________________________________________________________
void TFumili::BuildArrays(){
  //
  //   Allocates memory for internal arrays. Called by TFumili::TFumili
  //
  fCmPar      = new Double_t[fMaxParam];
  fA          = new Double_t[fMaxParam];
  fPL0        = new Double_t[fMaxParam];
  fPL         = new Double_t[fMaxParam];
  fParamError = new Double_t[fMaxParam];
  fDA         = new Double_t[fMaxParam];
  fAMX        = new Double_t[fMaxParam];
  fAMN        = new Double_t[fMaxParam];
  fR          = new Double_t[fMaxParam];
  fDF         = new Double_t[fMaxParam];
  fGr         = new Double_t[fMaxParam]; 
  fANames     = new TString[fMaxParam];
  
  //   fX = new Double_t[10];

  Int_t Zsize = fMaxParam*(fMaxParam+1)/2;
  fZ0 = new Double_t[Zsize];
  fZ  = new Double_t[Zsize];

  for (Int_t i=0;i<fMaxParam;i++){
    fA[i] =0.;
    fDF[i]=0.;
    fAMN[i]=kMINDOUBLE;
    fAMX[i]=kMAXDOUBLE;
    fPL0[i]=.1;
    fPL[i] =.1;
    fParamError[i]=0.;
    fANames[i]=Form("%d",i);
  }
}


//______________________________________________________________________________
TFumili::~TFumili() {
  // 
  // TFumili destructor
  //
  DeleteArrays();
  gROOT->GetListOfSpecials()->Remove(this);
  if (gFumili == this) gFumili = 0; 
}

//______________________________________________________________________________
Double_t TFumili::Chisquare(Int_t npar, Double_t *params)
{
   // return a chisquare equivalent
   
   Double_t amin = 0;
   H1FitChisquareFumili(npar,params,amin,params,1);
   return amin;
}


//______________________________________________________________________________
void TFumili::Clear(Option_t *)
{
  //
  // Resets all parameter names, values and errors to zero
  // 
  // Argument opt is ignored
  //
  // NB: this procedure doesn't reset parameter limits 
  //
  for (Int_t i=0;i<fNpar;i++){
    fA[i]   =0.;
    fDF[i]  =0.;
    fPL0[i] =.1;
    fPL[i]  =.1;
    fParamError[i]=0.;
    fANames[i]=Form("%d",i);
  }
}


//______________________________________________________________________________
void TFumili::DeleteArrays(){
  //
  // Deallocates memory. Called from destructor TFumili::~TFumili
  //
  delete[] fCmPar;
  delete[] fANames;
  delete[] fDF;
  // delete[] fX;
  delete[] fZ0;
  delete[] fZ;
  delete[] fGr;
  delete[] fA;
  delete[] fPL0;
  delete[] fPL;
  delete[] fDA;
  delete[] fAMN;
  delete[] fAMX;
  delete[] fParamError;
  delete[] fR;
}


//______________________________________________________________________________
Double_t TFumili::GetSumLog(Int_t n)
{
   // return Sum(log(i) i=0,n
   // used by log likelihood fits

   if (n < 0) return 0;
   if (n > fNlog) {
      if (fSumLog) delete [] fSumLog;
      fNlog = 2*n+1000;
      fSumLog = new Double_t[fNlog+1];
      Double_t fobs = 0;
      for (Int_t j=0;j<=fNlog;j++) {
         if (j > 1) fobs += TMath::Log(j);
         fSumLog[j] = fobs;
      }
   }
   if (fSumLog) return fSumLog[n];
   return 0;
}

//______________________________________________________________________________
void TFumili::PrintResults(Int_t ikode,Double_t p) const
{
  // Prints fit results. 
  //  
  // ikode is the type of printing parameters
  // p is functional value
  //
  //  ikode = 1   - print values, errors and limits
  //  ikode = 2   - print values, errors and steps
  //  ikode = 3   - print values, errors, steps and derivatives
  //  ikode = 4   - print only values and errors
  //
  TString ExitStatus="";
  TString xsexpl="";
  TString colhdu[3],colhdl[3],cx2,cx3;
  switch (fENDFLG) {
  case 1:
    ExitStatus="CONVERGED";
    break;
  case -1:
    ExitStatus="CONST FCN";
    xsexpl="****\n* FUNCTIONAL IS NOT DECREASING OR BAD DERIVATIVES\n****";
    break;
  case -2:
    ExitStatus="ERRORS INF";
    xsexpl="****\n* ESTIMATED ERRORS ARE INFINITE\n****";
    break;
  case -3:
    ExitStatus="MAX ITER.";
    xsexpl="****\n* MAXIMUM NUMBER OF ITERATIONS IS EXCEEDED\n****";
    break;
  case -4:
    ExitStatus="ZERO PROBAB";
    xsexpl="****\n* PROBABILITY OF LIKLIHOOD FUNCTION IS NEGATIVE OR ZERO\n****";
    break;
  default:
    ExitStatus="UNDEFINED";
    xsexpl="****\n* FIT IS IN PROGRESS\n****";
    break;
  }
  if (ikode == 1) {
    colhdu[0] = "              ";
    colhdl[0] = "      ERROR   ";
    colhdu[1] = "      PHYSICAL";
    colhdu[2] = " LIMITS       ";
    colhdl[1] = "    NEGATIVE  ";
    colhdl[2] = "    POSITIVE  ";
  }
  if (ikode == 2) {
    colhdu[0] = "              ";
    colhdl[0] = "      ERROR   ";
    colhdu[1] = "    INTERNAL  ";
    colhdl[1] = "    STEP SIZE ";
    colhdu[2] = "    INTERNAL  ";
    colhdl[2] = "      VALUE   ";
  }
  if (ikode == 3) {
    colhdu[0] = "              ";
    colhdl[0] = "      ERROR   ";
    colhdu[1] = "       STEP   ";
    colhdl[1] = "       SIZE   ";
    colhdu[2] = "       FIRST  ";
    colhdl[2] = "    DERIVATIVE";
  }
  if (ikode == 4) {
    colhdu[0] = "    PARABOLIC ";
    colhdl[0] = "      ERROR   ";
    colhdu[1] = "        MINOS ";
    colhdu[2] = "ERRORS        ";
    colhdl[1] = "   NEGATIVE   ";
    colhdl[2] = "   POSITIVE   ";
  }
  if(fENDFLG<1)Printf((const char*)xsexpl.Data());
  Printf(" FCN=%g FROM FUMILI  STATUS=%-10s %9d CALLS OF FCN",
	 p,ExitStatus.Data(),fNfcn);
  Printf(" EDM=%g ",fGT);
  Printf("  EXT PARAMETER              %-14s%-14s%-14s",
	 (const char*)colhdu[0].Data()
	 ,(const char*)colhdu[1].Data()
	 ,(const char*)colhdu[2].Data());
  Printf("  NO.   NAME          VALUE  %-14s%-14s%-14s",
	 (const char*)colhdl[0].Data()
	 ,(const char*)colhdl[1].Data()
	 ,(const char*)colhdl[2].Data());

  for (Int_t i=0;i<fNpar;i++){ 

    if (ikode==3) { 
      cx2 = Form("%14.5e",fDA[i]);
      cx3 = Form("%14.5e",fGr[i]);

    }
    if (ikode==1) {
      cx2 = Form("%14.5e",fAMN[i]);
      cx3 = Form("%14.5e",fAMX[i]);
    }
    if (ikode==2) {
      cx2 = Form("%14.5e",fDA[i]);
      cx3 = Form("%14.5e",fA[i]);
    }
    if(ikode==4){
      cx2 = " *undefined*  ";
      cx3 = " *undefined*  ";
    }
    if(fPL0[i]<=0.) { cx2="    *fixed*   ";cx3=""; }
    Printf("%4d %-11s%14.5e%14.5e%-14s%-14s",i+1
	   ,fANames[i].Data(),fA[i],fParamError[i]
	   ,cx2.Data(),cx3.Data());
  }
}


//______________________________________________________________________________
Int_t TFumili::Eval(Int_t& npar, Double_t *grad, Double_t &fval, Double_t *par, Int_t flag)
{  
  // Evaluate the minimisation function
  //  Input parameters:
  //    npar:    number of currently variable parameters
  //    par:     array of (constant and variable) parameters
  //    flag:    Indicates what is to be calculated
  //    grad:    array of gradients
  //  Output parameters:
  //    fval:    The calculated function value. 
  //    grad:    The vector of first derivatives.
  // 
  // The meaning of the parameters par is of course defined by the user, 
  // who uses the values of those parameters to calculate his function value. 
  // The starting values must be specified by the user.
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // Inside FCN user has to define Z-matrix by means TFumili::GetZ 
  //  and TFumili::Derivatives,
  // set theoretical function by means of TFumili::SetUserFunc, 
  // but first - pass number of parameters by TFumili::SetParNumber
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // Later values are determined by Fumili as it searches for the minimum 
  // or performs whatever analysis is requested by the user.
  //
  // The default function calls the function specified in SetFCN
  //
   
  if (fFCN) (*fFCN)(npar,grad,fval,par,flag);
  return npar;
}


//______________________________________________________________________________
void TFumili::SetFitMethod(const char *name)
{
   // ret fit method (chisquare or loglikelihood)
   
   if (!strcmp(name,"H1FitChisquare"))    SetFCN(H1FitChisquareFumili);
   if (!strcmp(name,"H1FitLikelihood"))   SetFCN(H1FitLikelihoodFumili);
   if (!strcmp(name,"GraphFitChisquare")) SetFCN(GraphFitChisquareFumili);
}



//______________________________________________________________________________
Int_t TFumili::Minimize()
{// Main minimization procedure
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
//         FUMILI  
//  Based on ideas, proposed by I.N. Silin
//    [See NIM A440, 2000 (p431)]
// conerted from FORTRAN to C  by
//     Sergey Yaschenko <s.yaschenko@fz-juelich.de>
//
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*//
  //
  // This function is called after setting theoretical function 
  // by means of TFumili::SetUserFunc and initializing parameters.
  // Optionally one can set FCN function (see TFumili::SetFCN and TFumili::Eval)
  // If FCN is undefined then user has to provide data arrays by calling
  //  TFumili::SetData procedure.
  //
  // TFumili::Minimize return following values:
  //    0  - fit is converged
  //   -2  - functional is not decreasing (or bad derivatives)
  //   -3  - error estimations are infinite
  //   -4  - maximum number of iterations is exceeded
  //
  Int_t I;
  // Flag3 - is fit is chi2 or likelihood? 0 - chi2, 1 - likelihood
  fINDFLG[2]=0;
  //
  // Are the parameters outside of the boundaries ?
  //
  Int_t parn;

  if(fFCN) {
    Eval(parn,fGr,fS,fA,9); fNfcn++;}
  for( I = 0; I < fNpar; I++)
    {
      if(fA[I] > fAMX[I]) fA[I] = fAMX[I];
      if(fA[I] < fAMN[I]) fA[I] = fAMN[I];
    }

  Int_t NN2, N, FIXFLG,  IFIX1, FI, NN3, NN1, N0;
  Double_t T1;
  NN2=0;

  // Number of parameters;
  N=fNpar;
  FIXFLG=0;

  // Exit flag
  fENDFLG=0;

  // Flag2
  fINDFLG[1] = 0;
  IFIX1=-1;
  FI=0;
  NN3=0;
  
  // Initialize param.step limits
  for( I=0; I < N; I++) {
      fR[I]=0.;
      if ( fEPS > 0.) fParamError[I] = 0.;
      fPL[I] = fPL0[I];
  }

 L3: // Start Iteration
 
  NN1 = 1;
  T1 = 1.;
 
 L4: // New iteration
 
  // fS - objective function value - zero first
  fS = 0.;
  // N0 - number of variable parameters in fit
  N0 = 0;
  for( I = 0; I < N; I++) {
      fGr[I]=0.; // zero gradients
      if (fPL0[I] > .0) {
	  N0=N0+1; 
	  // new iteration - new parallelepiped
	  if (fPL[I] > .0) fPL0[I]=fPL[I];
      }
  }
  Int_t NN0 , NA;
  // Calculate number of fZ-matrix elements as NN0=1+2+..+N0 
  NN0 = N0*(N0+1)/2;
  // if (NN0 >= 1) ????
  // fZ-matrix is initialized
  for( I=0; I < NN0; I++) fZ[I]=0.;
  NA = fNpar;

  // Flag1
  fINDFLG[0] = 0;
  Int_t ijkl=1;

  // Calculate fS - objective function, fGr - gradients, fZ - fZ-matrix
  if(fFCN) {
    Eval(parn,fGr,fS,fA,2);
    fNfcn++;
  } else
    ijkl = SGZ();
  if(!ijkl) return 10; 
  if (ijkl == -1) fINDFLG[0]=1;
  Double_t SP, T, OLDS=0;

  // SP - scaled on fS machine precision
  SP=fRP*TMath::Abs(fS);

  // save fZ-matrix
  for( I=0; I < NN0; I++) fZ0[I] = fZ[I];
  if (NN3 > 0) 
    if (NN1 <= fNstepDec) {
	T=2.*(fS-OLDS-fGT);
	if (fINDFLG[0] == 0) {
	    if (TMath::Abs(fS-OLDS) <= SP && -fGT <= SP) goto L19;
	    if(	0.59*T < -fGT) goto L19;
	    T = -fGT/T;
	    if (T < 0.25 ) T = 0.25;
	}
	else   T = 0.25;
	fGT = fGT*T;
	T1 = T1*T;
	NN2=0;
	for( I = 0; I < N; I++)
	  if (fPL[I] > 0.) {
	      fA[I]=fA[I]-fDA[I];
	      fPL[I]=fPL[I]*T;
	      fDA[I]=fDA[I]*T;
	      fA[I]=fA[I]+fDA[I];
	  }
	NN1=NN1+1;
	goto L4;
      }
 
 L19:
 
  if(fINDFLG[0] != 0) {
      fENDFLG=-4;
      printf("trying to execute an illegal junp at L85\n");
      //goto L85;
  } 

 
  Int_t K1, K2, I1, J, L;
  K1 = 1;
  K2 = 1;
  I1 = 1;
  // In this cycle we removed from fZ contributions from fixed parameters
  // We'll get fixed parameters after boudary check
  for( I = 0; I < N; I++)
    if (fPL0[I] > .0) { 
	// if parameter was fixed - release it
	if (fPL[I] == 0.) fPL[I]=fPL0[I];
	if (fPL[I] > .0) // ??? it is already non-zero
	  {
	    // if derivative is negative and we above maximum
	    // or vice versa then fix parameter again and increment K1 by I1
	    if ((fA[I] >= fAMX[I] && fGr[I] < 0.) ||
		(fA[I] <= fAMN[I] && fGr[I] > 0.))
	      {
		fPL[I] = 0.;
		K1 = K1 + I1; // I1 stands for fZ-matrix row-number multiplier
		///  - skip this row
		//  in case we are fixing parameter number I
	      } else {
		for( J=0; J <= I; J++) // cycle on columns of fZ-matrix
		  if (fPL0[J] > .0) 
		    { // if parameter is not fixed then fZ = fZ0 
		      // Now matrix fZ of other dimension
		      if (fPL[J] > .0) 
			{
			  fZ[K2 -1] = fZ0[K1 -1];
			  K2=K2+1;
			}  
		      K1=K1+1;
		    }
	      }
	  }  
	else K1 = K1 + I1; // In case of negative fPL[i] - after mconvd
	I1=I1+1;  // Next row of fZ0
      }

  // INVERT fZ-matrix (mconvd() procedure)
  I1 = 1;
  L  = 1;
  for( I = 0; I < N; I++) // extract diagonal elements to fR-vector
    if (fPL[I] > .0)
      { 
	fR[I] = fZ[L - 1];
	I1 = I1+1;
	L = L + I1;
      }
  Int_t L1, K, IFIX;
  Double_t BI, AIMAX=0, AMB;
  N0 = I1 - 1;
  InvertZ(N0);

  // fZ matrix now is inversed
  if (fINDFLG[0] != 0) // problems
    { // some PLs now have negative values, try to reduce fZ-matrix again
      fINDFLG[0] = 0;
      fINDFLG[1] = 1; // errors can be infinite
      FIXFLG = FIXFLG + 1;
      FI = 0;
      goto L19;
    }

  // ... CALCULATE THEORETICAL STEP TO MINIMUM
  I1 = 1;
  for( I = 0; I < N; I++)
    {
      fDA[I]=0.; // initial step is zero
      if (fPL[I] > .0)
	{   // for non-fixed parameters
	  L1=1;
	  for( L = 0; L < N; L++)
	    if (fPL[L] > .0)
	      { // Caluclate offset of Z^-1(I1,L1) element in packed matrix
		// because we skip fixed param numbers we need also I,L
		if (I1 <= L1 ) K=L1*(L1-1)/2+I1;
		else K=I1*(I1-1)/2+L1;
		// dA_i = \sum (-Z^{-1}_{il}*grad(fS)_l)
		fDA[I]=fDA[I]-fGr[L]*fZ[K - 1];
		L1=L1+1;
	      }
	  I1=I1+1;
	}
    }
  //	  ... CHECK FOR PARAMETERS ON BOUNDARY
  Double_t AFIX, SIGI, AKAP;

  AFIX=0.;
  IFIX = -1;
  I1 = 1;
  L = I1;
  for( I = 0; I < N; I++)
    if (fPL[I] > .0)
      {
	SIGI = TMath::Sqrt(TMath::Abs(fZ[L - 1])); // calculate \sqrt{Z^{-1}_{ii}} 
	fR[I] = fR[I]*fZ[L - 1];      // Z_ii * Z^-1_ii
	if (fEPS > .0) fParamError[I]=SIGI;
	if ((fA[I] >= fAMX[I] && fDA[I] > 0.) || (fA[I] <= fAMN[I]
					       && fDA[I] < .0))
	  { // if parameter out of bounds and if step is making things worse
      
	    AKAP = TMath::Abs(fDA[I]/SIGI);
	    // let's found maximum of dA/sigi - the worst of parameter steps
	    if (AKAP > AFIX)
	      {
		AFIX=AKAP;
		IFIX=I;
		IFIX1=I;
	      }
	  }
	I1=I1+1;
	L=L+I1;
      }
  if (IFIX != -1)
    { // so the worst parameter is found - fix it and exclude,
      //  reduce fZ-matrix again
      fPL[IFIX] = -1.;
      FIXFLG = FIXFLG + 1;
      FI = 0;
      //.. REPEAT CALCULATION OF THEORETICAL STEP AFTER FIXING EACH PARAMETER
      goto L19;
    }

  //... CALCULATE STEP CORRECTION FACTOR
  Double_t ALAMBD, AL, BM, ABI,ABM;
  ALAMBD = 1.;
  fAKAPPA = 0.;
  Int_t IMAX=0;
  IMAX = -1;


  for( I = 0; I < N; I++)
    if (fPL[I] > .0)
      {
	BM = fAMX[I] - fA[I];  
	ABI = fA[I] + fPL[I]; // upper  parameter limit
	ABM = fAMX[I];
	if (fDA[I] <= .0)
	  {
	    BM = fA[I] - fAMN[I];
	    ABI = fA[I] - fPL[I]; // lower parameter limit
	    ABM = fAMN[I];
	  }
	BI = fPL[I];
	// if parallelepiped boundary is crossing limits
	// then reduce it (deforming)
	if ( BI > BM)
	  {
	    BI = BM;
	    ABI = ABM;
	  }
	// if calculated step is out of bounds
	if ( TMath::Abs(fDA[I]) > BI)
	  {
	    // derease step splitter ALAMBDA if needed
	    AL = TMath::Abs(BI/fDA[I]);
	    if (ALAMBD > AL)
	      {
		IMAX=I;
		AIMAX=ABI;
		ALAMBD=AL;
	      }
	  }
	// fAKAPPA - parameter will be <fEPS if fit is converged
	AKAP = TMath::Abs(fDA[I]/fParamError[I]); 
	if (AKAP > fAKAPPA) fAKAPPA=AKAP;
      }
  //... CALCULATE NEW CORRECTED STEP
  fGT = 0.;
  AMB = 1.e18;
  // ALAMBD - multiplier to split teoretical step dA
  if (ALAMBD > .0) AMB = 0.25/ALAMBD;
  for( I = 0; I < N; I++)
    if (fPL[I] > .0)
      {
	if (NN2 > fNlimMul ) 
	  if (TMath::Abs(fDA[I]/fPL[I]) > AMB )
	    {
	      fPL[I] = 4.*fPL[I]; // increase parallelepiped
	      T1=4.; // flag - that fPL was increased
	    }
	// cut step
	fDA[I] = fDA[I]*ALAMBD;
	// expected functional value change in next iteration
	fGT = fGT + fDA[I]*fGr[I];
      }

  //.. CHECK IF MINIMUM ATTAINED AND SET EXIT MODE
  // if expected fGT smaller than precision
  // and other stuff
  if (-fGT <= SP && T1 < 1. && ALAMBD < 1.)fENDFLG = -1; // function is not decreasing 

  if (fENDFLG >= 0)
    if (fAKAPPA < TMath::Abs(fEPS)) // fit is converged
      {
	if (FIXFLG == 0) 
	  fENDFLG=1; // successful fit
	else
	  {// we have fixed parameters
	    if (fENDFLG == 0)
	      //... CHECK IF FIXING ON BOUND IS CORRECT
	      {
		fENDFLG = 1;
		FIXFLG = 0;
		IFIX1=-1;
		// release fixed parameters
		for( I = 0; I < fNpar; I++) fPL[I] = fPL0[I];
		fINDFLG[1] = 0;
		// and repeat iteration
		goto L19;
	      }
	    else
	      {
		if( IFIX1 >= 0)
		  {
		    FI = FI + 1;
		    fENDFLG = 0;
		  }
	      }
	  }
      }
    else // fit is not converged
      { 
	if( FIXFLG != 0)
	  {
	    if( FI > FIXFLG )
	      {
		//... CHECK IF FIXING ON BOUND IS CORRECT
		fENDFLG = 1;
		FIXFLG = 0;
		IFIX1=-1;
		for( I = 0; I < fNpar; I++) fPL[I] = fPL0[I];
		fINDFLG[1] = 0;
		goto L19;
	      }
	    else
	      {
		FI = FI + 1;
		fENDFLG = 0;
	      }
	  }
	else
	  {
	    FI = FI + 1;
	    fENDFLG = 0;
	  }
      }

// L85:
  // iteration number limit is exceeded
  if(fENDFLG == 0 && NN3 >= fNmaxIter) fENDFLG=-3;
  // fit errors are infinite;
  if(fENDFLG > 0 && fINDFLG[1] > 0) fENDFLG=-2;
  //MONITO (fS,fNpar,NN3,IT,fEPS,fGT,fAKAPPA,ALAMBD);
  if (fENDFLG == 0)
    {// make step
      for ( I = 0; I < N; I++) fA[I] = fA[I] + fDA[I];
      if (IMAX >= 0) fA[IMAX] = AIMAX;
      OLDS=fS;
      NN2=NN2+1;
      NN3=NN3+1;
    }
  else
    { 
      // fill covariant matrix VL
      // fill parameter error matrix up
      Int_t il;
      il = 0;
      for( Int_t ip = 0; ip < fNpar; ip++)
	{ 
	  if( fPL0[ip] > .0)
	    for( Int_t jp = 0; jp <= ip; jp++)
	      if(fPL0[jp] > .0)
		{
		  //	 VL[ind(ip,jp)] = fZ[il];
		  il = il + 1;
		}
	}
      return fENDFLG - 1;
    }
  goto L3;
}

//______________________________________________________________________________
Double_t TFumili::EvalTFN(Double_t * /*df*/, Double_t *X)
{
  // Evaluate theoretical function
  // df: array of partial derivatives
  // X:  vector of theoretical function argument
  //if(fTFN) 
  //  return (*fTFN)(df,X,fA);
  //else if(fTFNF1){

    TF1 *f1 = (TF1*)fUserFunc;
    return f1->EvalPar(X,fA);
  //}
  return 0.;
}


//______________________________________________________________________________
Int_t TFumili::SGZ()
{
  //  Evaluates objective function ( chi-square ), gradients and   
  //  Z-matrix using data provided by user via TFumili::SetData
  //
  fS = 0.;
  Int_t i,j,L,K2=1,K1,KI=0;
  Double_t *X  = new Double_t[fNED2];
  Double_t *df = new Double_t[fNpar];
  Int_t NX = fNED2-2;
  for (L=0;L<fNED1;L++) { // cycle on all exp. points
    K1 = K2;
    if (fLogLike) {
      fNumericDerivatives = kTRUE;
      NX  = fNED2;
      K1 -= 2;
    };
  
    for (i=0;i<NX;i++){
      KI  += 1+i;
      X[i] = fEXDA[KI];
    }
    //  Double_t Y = ARITHM(df,X);
    Double_t Y = EvalTFN(df,X);
    if(fNumericDerivatives) Derivatives(df,X);
    Double_t SIG=1.;
    if(fLogLike) { // Likelihood method
      if(Y>0.) {
	fS = fS - log(Y);
	Y  = -Y;
	SIG= Y;
      } else { // 
	delete [] X;
	delete [] df;
	fS = 1e10;
	return -1; // indflg[0] = 1;
      }
    } else { // Chi2 method
      SIG = fEXDA[K2]; // sigma of experimental point
      Y = Y - fEXDA[K1-1]; // f(x_i) - F_i
      fS = fS + (Y*Y/(SIG*SIG))*.5; // simple chi2/2
    }
    Int_t N = 0;
    for (i=0;i<fNpar;i++) 
      if (fPL0[i]>0){
	df[N]   = df[i]/SIG; // left only non-fixed param derivatives div by Sig
	fGr[i] += df[N]*(Y/SIG);
	N++;
      }
    L = 0;
    for (i=0;i<N;i++)
      for (j=0;j<=i;j++) 
	fZ[L++] += df[i]*df[j];
    K2 += fNED2;
  }
 
  delete[] df;
  delete[] X;
  return 1;
}



//______________________________________________________________________________
void TFumili::InvertZ(Int_t n)
{
  // Inverts packed diagonal matrix Z by square-root method.
  //  Matrix elements corresponding to 
  // fix parameters are removed.
  //
  // n: number of variable parameters
  //
  static Double_t am = 3.4e138;
  static Double_t rp = 5.0e-14;
  Double_t  ap, aps, c, d;
  Double_t *R_1=fR;
  Double_t *PL_1=fPL;
  Double_t *Z_1=fZ;
  Int_t i, k, l, ii, ki, li, kk, ni, ll, nk, nl, ir, lk;
  if (n < 1) {
    return;
  }
  --PL_1;
  --R_1;
  --Z_1;
  aps = am / n;
  aps = sqrt(aps);
  ap = 1.0e0 / (aps * aps);
  ir = 0;
  for (i = 1; i <= n; ++i) {
  L1:
    ++ir;
    if (PL_1[ir] <= 0.0e0) {
      goto L1;
    } else {
      goto L2;
    }
  L2:
    ni = i * (i - 1) / 2;
    ii = ni + i;
    k = n + 1;
    if (Z_1[ii] <= rp * TMath::Abs(R_1[ir]) || Z_1[ii] <= ap) {
      goto L19;
    }
    Z_1[ii] = 1.0e0 / sqrt(Z_1[ii]);
    nl = ii - 1;
  L3:
    if (nl - ni <= 0) {
      goto L5;
    } else {
      goto L4;
    }
  L4:
    Z_1[nl] *= Z_1[ii];
    if (TMath::Abs(Z_1[nl]) >= aps) {
      goto L16;
    }
    --nl;
    goto L3;
  L5:
    if (i - n >= 0) {
      goto L12;
    } else {
      goto L6;
    }
  L6:
    --k;
    nk = k * (k - 1) / 2;
    nl = nk;
    kk = nk + i;
    d = Z_1[kk] * Z_1[ii];
    c = d * Z_1[ii];
    l = k;
  L7:
    ll = nk + l;
    li = nl + i;
    Z_1[ll] -= Z_1[li] * c;
    --l;
    nl -= l;
    if (l - i <= 0) {
      goto L9;
    } else {
      goto L7;
    }
  L8:
    ll = nk + l;
    li = ni + l;
    Z_1[ll] -= Z_1[li] * d;
  L9:
    --l;
    if (l <= 0) {
      goto L10;
    } else {
      goto L8;
    }
  L10:
    Z_1[kk] = -c;
    if (k - i - 1 <= 0) {
      goto L11;
    } else {
      goto L6;
    }
  L11:
    ;
  }
 L12:
  for (i = 1; i <= n; ++i) {
    for (k = i; k <= n; ++k) {
      nl = k * (k - 1) / 2;
      ki = nl + i;
      d = 0.0e0;
      for (l = k; l <= n; ++l) {
	li = nl + i;
	lk = nl + k;
	d += Z_1[li] * Z_1[lk];
	nl += l;
      }
      ki = k * (k - 1) / 2 + i;
      Z_1[ki] = d;
    }
  }
 L15:
  return;
 L16:
  k = i + nl - ii;
  ir = 0;
  for (i = 1; i <= k; ++i) {
  L17:
    ++ir;
    if (PL_1[ir] <= 0.0e0) {
      goto L17;
    }
  }
 L19:
  PL_1[ir] = -2.0e0;
  R_1[ir] = 0.0e0;
  fINDFLG[0] = ir - 1;
  goto L15;
}




//______________________________________________________________________________
void TFumili::Derivatives(Double_t *DF,Double_t *fX){
  //  
  // Calculates partial derivatives of theoretical function
  //
  // Input:
  //    fX  - vector of data point
  // Output:
  //    DF - array of derivatives
  //
  // ARITHM.F 
  // Converted from CERNLIB
  //
  Double_t ff,AI,HI,Y,PI;
  Y = EvalTFN(DF,fX);
  for (Int_t i=0;i<fNpar;i++){
    DF[i]=0;
    if(fPL0[i]>0.){
      AI = fA[i]; // save current parameter value
      HI = 0.01*fPL0[i]; // diff step 
      PI = fRP*TMath::Abs(AI);
      if (HI<PI) HI = PI; // if diff step is less than precision
      fA[i] = AI+HI;
   
      if (fA[i]>fAMX[i]) { // if param is out of limits
	fA[i] = AI-HI;
	HI = -HI;
	if (fA[i]<fAMN[i]) { // again out of bounds
	  fA[i] = fAMX[i];   // set param to high limit
	  HI = fAMX[i]-AI;
	  if (fAMN[i]-AI+HI<0) { // if HI < (AI-fAMN)
	    fA[i]=fAMN[i];
	    HI=fAMN[i]-AI;
	  }
	}
      }
      ff = EvalTFN(DF,fX);
      DF[i] = (ff-Y)/HI;
      fA[i] = AI;
    }
  }
}


//______________________________________________________________________________
void TFumili::SetData(Double_t *exdata,Int_t numpoints,Int_t vecsize){
  // Sets pointer to data array provided by user. 
  // Necessary if SetFCN is not called.
  // 
  // numpoints:    number of experimental points
  // vecsize:      size of data point vector + 2 
  //               (for N-dimensional fit vecsize=N+2)
  // exdata:       data array with following format
  //
  //   exdata[0] = ExpValue_0     - experimental data value number 0
  //   exdata[1] = ExpSigma_0     - error of value number 0
  //   exdata[2] = X_0[0]        
  //   exdata[3] = X_0[1]
  //       .........
  //   exdata[vecsize-1] = X_0[vecsize-3]
  //   exdata[vecsize]   = ExpValue_1
  //   exdata[vecsize+1] = ExpSigma_1
  //   exdata[vecsize+2] = X_1[0]
  //       .........
  //   exdata[vecsize*(numpoints-1)] = ExpValue_(numpoints-1)
  //       .........
  //   exdata[vecsize*numpoints-1] = X_(numpoints-1)[vecsize-3]
  //
  
  if(exdata){
    fNED1 = numpoints;
    fNED2 = vecsize;
    fEXDA = exdata;
  }
}


//______________________________________________________________________________
void TFumili::FixParameter(Int_t ipar) { 
  // Fixes parameter number ipar

  if(ipar>=0 && ipar<fNpar && fPL0[ipar]>0.) {
    fPL0[ipar] = -fPL0[ipar]; 
    fLastFixed = ipar;
  }
}


//______________________________________________________________________________
void TFumili::ReleaseParameter(Int_t ipar) {
  // Releases parameter number ipar

  if(ipar>=0 && ipar<fNpar && fPL0[ipar]<=0.) {
    fPL0[ipar] = -fPL0[ipar]; 
    if (fPL0[ipar] == 0. || fPL0[ipar]>=1.) fPL0[ipar]=.1;
  }
}


//______________________________________________________________________________
Int_t TFumili::SetParameter(Int_t ipar,const char *parname,Double_t value,Double_t verr,Double_t vlow, Double_t vhigh) {
  // Sets for prameter number ipar initial parameter value, 
  // name parname, initial error verr and limits vlow and vhigh
  // If vlow = vhigh but not equil to zero, parameter will be fixed.
  // If vlow = vhigh = 0, parameter is released and its limits are discarded
  //
  if (ipar<0 || ipar>=fNpar) return -1;
  fANames[ipar] = parname;
  fA[ipar] = value; 
  fParamError[ipar] = verr;
  if(vlow<vhigh) {
    fAMN[ipar] = vlow;
    fAMX[ipar] = vhigh;
  } else {
    if(vhigh<vlow) {
       fAMN[ipar] = vhigh;
       fAMX[ipar] = vlow;
    }
    if(vhigh==vlow) {
      if(vhigh==0.) {
	ReleaseParameter(ipar);
	fAMN[ipar] = kMINDOUBLE;
	fAMX[ipar] = kMAXDOUBLE;
      }
      if(vlow!=0) FixParameter(ipar);
    }
  }
  return 0;
}

//______________________________________________________________________________
Int_t TFumili::GetParameter(Int_t ipar,char *cname,Double_t &value,Double_t &verr,Double_t &vlow, Double_t &vhigh) {
  // Get various ipar parameter attributs:
  // 
  // cname:    parameter name
  // value:    parameter value
  // verr:     parameter error
  // vlow:     lower limit
  // vhigh:    upper limit
  //
  if (ipar<0 || ipar>=fNpar) {
    value = 0;
    verr  = 0;
    vlow  = 0;
    vhigh = 0;
    return -1;
  }
  strcpy(cname,fANames[ipar].Data());
  value = fA[ipar];
  verr  = fParamError[ipar];
  vlow  = fAMN[ipar];
  vhigh = fAMX[ipar];
  return 0;
}

//______________________________________________________________________________
Int_t TFumili::GetStats(Double_t &amin, Double_t &edm, Double_t &errdef, Int_t &nvpar, Int_t &nparx)
{
   // return global fit parameters
   //   amin     : chisquare
   //   edm      : estimated distance to minimum
   //   errdef
   //   nvpar    : number of variable parameters
   //   nparx    : total number of parameters
  amin   = 2*fS;
  edm    = fGT; // 
  errdef = 0; // ??
  nparx  = fNpar;
  nvpar  = 0;
  for(Int_t ii=0; ii<fNpar; ii++) {
    if(fPL0[ii]>0.) nvpar++;
  }  
  return 0;
}


//______________________________________________________________________________
Int_t TFumili::GetErrors(Int_t ipar,Double_t &eplus, Double_t &eminus, Double_t &eparab, Double_t &globcc) {
  // Return errors after MINOs
  // not implemented
  eparab = 0;
  globcc = 0;  
  if (ipar<0 || ipar>=fNpar) {
    eplus  = 0;
    eminus = 0;
    return -1;
  }
  eplus=fParamError[ipar];
  eminus=-eplus;
  return 0;
}

//______________________________________________________________________________
Int_t TFumili::ExecuteSetCommand(Int_t nargs){
  //
  // Called from TFumili::ExecuteCommand in case 
  // of "SET xxx" and "SHOW xxx".
  //
  static Int_t nntot = 30;
  static const char *cname[30] = {
    "FCN value ", // 0 .
    "PARameters", // 1 .
    "LIMits    ", // 2 .
    "COVariance", // 3 .
    "CORrelatio", // 4 .
    "PRInt levl", // 5 not implemented yet
    "NOGradient", // 6 .
    "GRAdient  ", // 7 .
    "ERRor def ", // 8 not sure how to implement - by time being ignored
    "INPut file", // 9 not implemented 
    "WIDth page", // 10 not implemented yet
    "LINes page", // 11 not implemented yet
    "NOWarnings", // 12 .
    "WARnings  ", // 13 .
    "RANdom gen", // 14 not implemented
    "TITle     ", // 15 ignored
    "STRategy  ", // 16 ignored
    "EIGenvalue", // 17 not implemented yet 
    "PAGe throw", // 18 ignored
    "MINos errs", // 19 not implemented yet
    "EPSmachine", // 20 .
    "OUTputfile", // 21 not implemented
    "BATch     ", // 22 ignored
    "INTeractiv", // 23 ignored
    "VERsion   ", // 24 .
    "reserve   ", // 25 .
    "NODebug   ", // 26 .
    "DEBug     ", // 27 .
    "SHOw      ", // 28 err
    "SET       "};// 29 err

  TString  cfname, cmode, ckind,  cwarn, copt, ctemp, ctemp2;
  Int_t i, ind;
  Bool_t SETCommand=kFALSE;
  for (ind = 0; ind < nntot; ++ind) {
    ctemp  = cname[ind];
    ckind  = ctemp(0,3);
    ctemp2 = fCword(4,6);
    if (strstr(ctemp2.Data(),ckind.Data())) break;
  }
  ctemp2 = fCword(0,3);
  if(ctemp2.Contains("SET")) SETCommand=true;
  if(ctemp2.Contains("HEL") || ctemp2.Contains("SHO")) SETCommand=false;
  
  if (ind>=nntot) return -3;

  switch(ind){
  case 0: // SET FCN value illegial // SHOw only
    if(!SETCommand) Printf("FCN=%f",fS);
    return 0;
  case 1: // PARameter <parno> <value> 
    {  
      if (nargs<2 && SETCommand) return -1;
      Int_t parnum;
      Double_t val;
      if(SETCommand) {
	parnum = Int_t(fCmPar[0])-1;
	val= fCmPar[1];
	if(parnum<0 || parnum>=fNpar) return -2; //no such parameter
	fA[parnum] = val;
      } else {
	if (nargs>0) {
	  parnum = Int_t(fCmPar[0])-1;
	  if(parnum<0 || parnum>=fNpar) return -2; //no such parameter
	  Printf("Parameter %s = %E",fANames[parnum].Data(),fA[parnum]);
	} else
	  for (i=0;i<fNpar;i++)
	    Printf("Parameter %s = %E",fANames[i].Data(),fA[i]);
	
      }
      return 0;
    }
  case 2: // LIMits [parno] [ <lolim> <uplim> ]
    {
      Int_t parnum;
      Double_t lolim,uplim;
      if (nargs<1) {
	for(i=0;i<fNpar;i++) 
	  if(SETCommand) {
	    fAMN[i] = kMINDOUBLE;
	    fAMX[i] = kMAXDOUBLE;
	  } else 
	    Printf("Limits for param %s: Low=%E, High=%E",
		   fANames[i].Data(),fAMN[i],fAMX[i]);
      } else {
	parnum = Int_t(fCmPar[0])-1;
	if(parnum<0 || parnum>=fNpar)return -1;
	if(SETCommand) {
	  if(nargs>2) {
	    lolim = fCmPar[1];
	    uplim = fCmPar[2];
	    if(uplim==lolim) return -1;
	    if(lolim>uplim) {
	      Double_t tmp = lolim;
	      lolim = uplim;
	      uplim = tmp;
	    }
	  } else {
	    lolim = kMINDOUBLE;
	    uplim = kMAXDOUBLE;
	  }
	  fAMN[parnum] = lolim;
	  fAMX[parnum] = uplim;
	} else 
	  Printf("Limits for param %s Low=%E, High=%E",
		 fANames[parnum].Data(),fAMN[parnum],fAMX[parnum]);
      }
      return 0;   
    }
  case 3:
    {
      if(SETCommand) return 0;
      Printf("\nCovariant matrix ");
      Int_t L = 0,nn=0,nnn=0;
      for (i=0;i<fNpar;i++) if(fPL0[i]>0.) nn++;
      for (i=0;i<nn;i++) {
	for(;fPL0[nnn]<=0.;nnn++);
	printf("%5s: ",fANames[nnn++].Data());
	for (Int_t j=0;j<=i;j++) 
	  printf("%11.2E",fZ[L++]);
	cout<<endl;
      }
      cout<<endl;
      return 0;
    }
  case 4:
    if(SETCommand) return 0;
    Printf("\nGlobal correlation factors (maximum correlation of the parameter\n  with arbitrary linear combination of other parameters)");
    for(i=0;i<fNpar;i++) {
      printf("%5s: ",fANames[i].Data());
      printf("%11.3E\n",TMath::Sqrt(1-1/((fR[i]!=0.)?fR[i]:1.)) );
    }
    cout<<endl;
    return 0;
  case 5:   // PRIntout not implemented
    return -10;
  case 6: // NOGradient
    if(!SETCommand) return 0;
    fGRAD = false;
    return 0;
  case 7: // GRAdient
    if(!SETCommand) return 0;
    fGRAD = true;
    return 0;
  case 8: // ERRordef - now ignored
    return 0;
  case 9: // INPut - not implemented
    return -10;
  case 10: // WIDthpage - not implemented
    return -10;
  case 11: // LINesperpage - not implemented
    return -10;
  case 12: //NOWarnings
    if(!SETCommand) return 0;
    fWARN = false;
    return 0;
  case 13: // WARnings 
    if(!SETCommand) return 0;
    fWARN = true;
    return 0;
  case 14: // RANdomgenerator - not implemented
    return -10;
  case 15: // TITle - ignored
    return 0; 
  case 16: // STRategy - ignored
    return 0;
  case 17: // EIGenvalues - not implemented
    return -10;
  case 18: // PAGethrow - ignored
    return 0;
  case 19: // MINos errors - not implemented
    return -10;
  case 20: //EPSmachine
    if(!SETCommand) {
      Printf("Relative floating point presicion RP=%E",fRP);
    } else 
      if (nargs>0) {
	Double_t pres=fCmPar[0];
	if (pres<1e-5 && pres>1e-34) fRP=pres;
      }
    return 0;
  case 21: // OUTputfile - not implemented
    return -10;
  case 22: // BATch - ignored
    return 0;
  case 23: // INTerative - ignored
    return 0;
  case 24: // VERsion
    if(SETCommand) return 0;
    Printf("FUMILI-ROOT version 0.1");
    return 0;
  case 25: // reserved
    return 0;
  case 26: // NODebug
    if(!SETCommand) return 0;
    fDEBUG = false;
    return 0;
  case 27: // DEBug
    if(!SETCommand) return 0;
    fDEBUG = true;
    return 0;
  case 28:
  case 29:
    return -3;
  default:
    break;
  }
  return -3;
}

//______________________________________________________________________________
Int_t TFumili::ExecuteCommand(const char *command, Double_t *args, Int_t nargs){
  // 
  //  Execute MINUIT commands. MINImize, SIMplex, MIGrad and FUMili all
  //  will call TFumili::Minimize method.
  // 
  //  For full command list see 
  //  MINUIT. Reference Manual. CERN Program Library Long Writeup D506.
  //
  //  Improvement and errors calculation are not yet implemented as well
  //  as Monte-Carlo seeking and minimization. 
  //  Contour commands are also unsupported.
  //
  //  command   : command string
  //  args      : array of arguments
  //  nargs     : number of arguments
  //
  TString comand = command;
  static TString clower = "abcdefghijklmnopqrstuvwxyz";
  static TString cupper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
  const Int_t nntot = 40;
  const char *cname[nntot] = {
    "MINImize  ",    //  0    checked
    "SEEk      ",    //  1    none 
    "SIMplex   ",    //  2    checked same as 0
    "MIGrad    ",    //  3    checked  same as 0
    "MINOs     ",    //  4    none 
    "SET xxx   ",    //  5 lot of stuff
    "SHOw xxx  ",    //  6 -----------
    "TOP of pag",    //  7 .
    "FIX       ",   //   8 . 
    "REStore   ",   //   9 .
    "RELease   ",   //   10 .
    "SCAn      ",   //   11  not yet implemented
    "CONtour   ",   //   12  not yet implemented
    "HESse     ",   //   13  not yet implemented
    "SAVe      ",   //   14  obsolete
    "IMProve   ",   //   15  not yet implemented
    "CALl fcn  ",   //   16 .  
    "STAndard  ",   //   17 .
    "END       ",   //   18 .
    "EXIt      ",   //   19 .
    "RETurn    ",   //   20 .
    "CLEar     ",   //   21 .
    "HELP      ",   //   22 not yet implemented
    "MNContour ",   //   23 not yet implemented
    "STOp      ",   //   24 .
    "JUMp      ",   //   25 not yet implemented
    "          ",   //   
    "          ",   // 
    "FUMili    ",   //    28 checked same as 0
    "          ",   //
    "          ",  //
    "          ",  //
    "          ",  //
    "COVARIANCE",  // 33
    "PRINTOUT  ",  // 34
    "GRADIENT  ",  // 35
    "MATOUT    ",  // 36
    "ERROR DEF ",  // 37
    "LIMITS    ",  // 38
    "PUNCH     "}; // 39

  
  fCword = comand;
  fCword.ToUpper();
  if (nargs<=0) fCmPar[0] = 0;
  Int_t i;
  for(i=0;i<fMaxParam;i++){
    if(i<=nargs) fCmPar[i] = args[i];
  }
  /*
  fNmaxIter = int(fCmPar[0]);
  if (fNmaxIter <= 0) {
     fNmaxIter = fNpar*10 + 20 + fNpar*M*5;
  }
  fEPS = fCmPar[1];
  */
  //*-*-               look for command in list CNAME . . . . . . . . . .
  TString ctemp = fCword(0,3);
  Int_t ind;
  for (ind = 0; ind < nntot; ++ind) {
	if (strncmp(ctemp.Data(),cname[ind],3) == 0) break;
  }
  if (ind==nntot) return -3; // Unknow command - input ignored
  if (fCword(0,4) == "MINO") ind=3;
  switch (ind) {
  case 0:  case 3: case 2: case 28:
    // MINImize [maxcalls] [tolerance]
    // also SIMplex, MIGrad  and  FUMili 
    if(nargs>=1)
      fNmaxIter=TMath::Max(Int_t(fCmPar[0]),fNmaxIter); // FIXME!!
    if(nargs==2) 
      fEPS=fCmPar[1];
    return Minimize();
  case 1:
    // SEEk not implemented in this package
    return -10;

  case 4: // MINos errors analysis not implemented
    return -10;

  case 5: case 6: // SET xxx & SHOW xxx
    return ExecuteSetCommand(nargs);

  case 7: // Obsolete command
    Printf("1");
    return 0;
  case 8: // FIX <parno> ....
    if (nargs<1) return -1; // No parameters specified
    for (i=0;i<nargs;i++) {
      Int_t parnum = Int_t(fCmPar[i])-1;
      FixParameter(parnum);
    }
    return 0;
  case 9: // REStore <code>
    if (nargs<1) return 0;
    if(fCmPar[0]==0.) 
     for (i=0;i<fNpar;i++)
       ReleaseParameter(i);
    else
      if(fCmPar[0]==1.) {
	ReleaseParameter(fLastFixed);
	cout <<fLastFixed<<endl;
      }
    return 0;
  case 10: // RELease <parno> ...
    if (nargs<1) return -1; // No parameters specified
    for (i=0;i<nargs;i++) {
      Int_t parnum = Int_t(fCmPar[i])-1;
      ReleaseParameter(parnum);
    }
    return 0;
  case 11: // SCAn not implemented
    return -10;
  case 12: // CONt not implemented 
    return -10;
  
  case 13: // HESSe not implemented
    return -10;
  case 14: // SAVe
    Printf("SAVe command is obsolete");
    return -10;
  case 15: // IMProve not implemented
    return -10;
  case 16: // CALl fcn <iflag>
    {if(nargs<1) return -1;
    Int_t flag = Int_t(fCmPar[0]);
    Double_t fval;
    Eval(fNpar,fGr,fval,fA,flag);
    return 0;}
  case 17: // STAndard must call function STAND
    return 0;
  case 18:   case 19:
  case 20:  case 24: 
    {
    Double_t fval;
    Int_t flag = 3;
    Eval(fNpar,fGr,fval,fA,flag);
    return 0;
    }
  case 21:
    Clear();
    return 0;
  case 22: //HELp not implemented
  case 23: //MNContour not implemented
  case 25: // JUMp not implemented
    return -10;
  case 26:   case 27:  case 29:  case 30:  case 31:  case 32: 
    return 0; // blank commands
  case 33:   case 34:   case 35:  case 36:   case 37:  case 38: 
  case 39:
    Printf("Obsolete command. Use corresponding SET command instead");
    return -10;
  default:
    break;
  }
  return 0;
}



//______________________________________________________________________________
void H1FitChisquareFumili(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag)
{
   Double_t cu,eu,fu,fsum;
   Double_t x[3];
   Int_t i, bin,binx,biny,binz;
   Axis_t binlow, binup, binsize;
   Double_t *zik=0;
   Double_t *pl0=0;
   Int_t npfits = 0;

   TFumili *hFitter = (TFumili*)TVirtualFitter::GetFitter();
   TH1 *hfit = (TH1*)hFitter->GetObjectFit();
   TF1 *f1   = (TF1*)hFitter->GetUserFunc();
   Foption_t Foption = hFitter->GetFitOption();

   npar = f1->GetNpar();
   hFitter->SetParNumber(npar);
   if(flag == 9) return;
   zik = hFitter->GetZ();
   pl0 = hFitter->GetPL0();

   Double_t *df=new Double_t[npar];
   f1->InitArgs(x,u);
   f = 0;
   Int_t hxfirst = hFitter->GetXfirst(); 
   Int_t hxlast  = hFitter->GetXlast(); 
   Int_t hyfirst = hFitter->GetYfirst(); 
   Int_t hylast  = hFitter->GetYlast(); 
   Int_t hzfirst = hFitter->GetZfirst(); 
   Int_t hzlast  = hFitter->GetZlast(); 
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
            TF1::RejectPoint(kFALSE);
            if (Foption.Integral) {
               binlow  = xaxis->GetBinLowEdge(binx);
               binsize = xaxis->GetBinWidth(binx);
               binup   = binlow + binsize;
               fu      = f1->Integral(binlow,binup,u)/binsize;
            } else {
               fu = f1->EvalPar(x,u);
            }
            if (TF1::RejectedPoint()) continue;
            if (Foption.W1) {
               eu = 1;
            } else {
               eu  = hfit->GetBinError(bin);
               if (eu <= 0) continue;
            }
	    npfits++;
	    hFitter->Derivatives(df,x);
	    Int_t N = 0;
	    fsum = (fu-cu)/eu;
	    for (i=0;i<npar;i++) 
	      if (pl0[i]>0){
		df[N] = df[i]/eu; 
		// left only non-fixed param derivatives / by Sigma
		gin[i] += df[N]*fsum;
		N++;
	      }
	    Int_t L = 0;
	    for (i=0;i<N;i++)
	      for (Int_t j=0;j<=i;j++) 
		zik[L++] += df[i]*df[j];
            f += .5*fsum*fsum;
         }
      }
   }
   f1->SetNumberFitPoints(npfits);
   delete[] df;
} 

//______________________________________________________________________________
void H1FitLikelihoodFumili(Int_t &npar, Double_t *gin, Double_t &f, Double_t *u, Int_t flag)
{
//   -*-*-*-*Minimization function for H1s using a Likelihood method*-*-*-*-*-*
//           =======================================================
//     Basically, it forms the likelihood by determining the Poisson
//     probability that given a number of entries in a particular bin,
//     the fit would predict it's value.  This is then done for each bin,
//     and the sum of the logs is taken as the likelihood.
//     PDF:  P=exp(-f(x_i))/[F_i]!*(f(x_i))^[F_i]
//    where F_i - experimental value, f(x_i) - expected theoretical value
//    [F_i] - integer part of F_i.
//    drawback is that if F_i>Int_t - GetSumLog will fail
//    for big F_i is faster to use Euler's Gamma-function

   Double_t cu,fu,fobs,fsub;
   Double_t x[3];
   Int_t i, bin,binx,biny,binz,icu;
   Axis_t binlow, binup, binsize;

   Int_t npfits = 0;

   TFumili *hFitter = (TFumili*)TVirtualFitter::GetFitter();
   TH1 *hfit = (TH1*)hFitter->GetObjectFit();
   TF1 *f1   = (TF1*)hFitter->GetUserFunc();
   Foption_t Foption = hFitter->GetFitOption();
   npar = f1->GetNpar();

   hFitter->SetParNumber(npar);
   if(flag == 9) return;
   Double_t *zik = hFitter->GetZ();
   Double_t *pl0 = hFitter->GetPL0();

   Double_t *df=new Double_t[npar];

   f1->InitArgs(x,u);
   f = 0;
   Int_t hxfirst = hFitter->GetXfirst(); 
   Int_t hxlast  = hFitter->GetXlast(); 
   Int_t hyfirst = hFitter->GetYfirst(); 
   Int_t hylast  = hFitter->GetYlast(); 
   Int_t hzfirst = hFitter->GetZfirst(); 
   Int_t hzlast  = hFitter->GetZlast(); 
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
            TF1::RejectPoint(kFALSE);
            bin = hfit->GetBin(binx,biny,binz);
            cu  = hfit->GetBinContent(bin);
            if (Foption.Integral) {
               binlow  = xaxis->GetBinLowEdge(binx);
               binsize = xaxis->GetBinWidth(binx);
               binup   = binlow + binsize;
               fu      = f1->Integral(binlow,binup,u)/binsize;
            } else {
               fu = f1->EvalPar(x,u);
            }
            if (TF1::RejectedPoint()) continue;
            npfits++;
	    if (fu < 1.e-9) fu = 1.e-9; 
	    icu   = Int_t(cu);
            fsub  = -fu +icu*TMath::Log(fu);
            fobs  = hFitter->GetSumLog(icu);
	    fsub -= fobs;
    	    hFitter->Derivatives(df,x);
	    int N=0;
	    // Here we need gradients of Log likelihood function
	    // 
	    for (i=0;i<npar;i++) 
	      if (pl0[i]>0){
	    	df[N]   = df[i]*(icu/fu-1); 
	    	gin[i] -= df[N];
		N++;
	      }
	    Int_t L = 0;
	    // Z-matrix here - production of first derivatives  
	    //  of log-likelihood function
	    for (i=0;i<N;i++)
	      for (Int_t j=0;j<=i;j++) 
	    	zik[L++] += df[i]*df[j];
            
            f -= fsub;
         }
      }
   }
   f *=.5;
   f1->SetNumberFitPoints(npfits);
   delete[] df;
}



//______________________________________________________________________________
void GraphFitChisquareFumili(Int_t &npar, Double_t * gin, Double_t &f,
                       Double_t *u, Int_t flag)
{
//*-*-*-*-*-*Minimization function for Graphs using a Chisquare method*-*-*-*-*
//*-*        =========================================================
//
// In case of a TGraphErrors object, ex, the error along x,  is projected
// along the y-direction by calculating the function at the points x-ex and
// x+ex.
//
// The chisquare is computed as the sum of the quantity below at each point:
//
//                     (y - f(x))**2
//         -----------------------------------
//         ey**2 + ((f(x+ex) - f(x-ex))/2)**2
//
// where x and y are the point coordinates

   Double_t cu,eu,ex,ey,eux,fu,fsum,fm,fp;
   Double_t x[1], xx[1];
   Double_t xm,xp;
   Int_t i, bin, npfits=0;

   TFumili *grFitter = (TFumili*)TVirtualFitter::GetFitter();
   TGraph *gr     = (TGraph*)grFitter->GetObjectFit();
   TF1 *f1   = (TF1*)grFitter->GetUserFunc();
   Foption_t Foption = grFitter->GetFitOption();
   
   Int_t n        = gr->GetN();
   Double_t *gx   = gr->GetX();
   Double_t *gy   = gr->GetY();
   Double_t fxmin = f1->GetXmin();
   Double_t fxmax = f1->GetXmax();
   npar           = f1->GetNpar();

   grFitter->SetParNumber(npar);

   if(flag == 9) return;
   Double_t *zik = grFitter->GetZ();
   Double_t *pl0 = grFitter->GetPL0();
   Double_t *df  = new Double_t[npar];


   f1->InitArgs(x,u);
   f      = 0;
   for (bin=0;bin<n;bin++) {
      x[0] = gx[bin];
      if (!f1->IsInside(x)) continue;
      cu   = gy[bin];
      TF1::RejectPoint(kFALSE);
      fu   = f1->EvalPar(x,u);
      if (TF1::RejectedPoint()) continue;
      //      fsum = (cu-fu);
      npfits++;
      Double_t eusq=1.;
      if (Foption.W1) {
	//         f += fsum*fsum;
	//         continue;
	eu = 1.;
      } else {
	ex  = gr->GetErrorX(bin);
	ey  = gr->GetErrorY(bin);
	if (ex < 0) ex = 0;
	if (ey < 0) ey = 0;
	if (ex >= 0) {
	  xm = x[0] - ex; if (xm < fxmin) xm = fxmin;
	  xp = x[0] + ex; if (xp > fxmax) xp = fxmax;
	  xx[0] = xm; fm = f1->EvalPar(xx,u);
	  xx[0] = xp; fp = f1->EvalPar(xx,u);
	  eux = 0.5*(fp-fm);
	} else
	  eux = 0.;
	eu = ey*ey+eux*eux;
	if (eu <= 0) eu = 1;
	eusq = TMath::Sqrt(eu);
      }
      grFitter->Derivatives(df,x);
      Int_t N = 0;
      fsum = (fu-cu)/eusq;
      for (i=0;i<npar;i++) 
	if (pl0[i]>0){
	  df[N] = df[i]/eusq; 
	  // left only non-fixed param derivatives / by Sigma
	  gin[i] += df[N]*fsum;
	  N++;
	}
      Int_t L = 0;
      for (i=0;i<N;i++)
	for (Int_t j=0;j<=i;j++) 
	  zik[L++] += df[i]*df[j];
      f += .5*fsum*fsum;

   }
   f1->SetNumberFitPoints(npfits);
}


