/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_Tools                                                            *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        *
 *      U. of Victoria, Canada,                                                   *
 *      MPI-KP Heidelberg, Germany,                                               *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/license.txt)                                      *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_Tools.cxx,v 1.3 2006/05/08 17:56:50 brun Exp $
 **********************************************************************************/
#include <algorithm>

#include "TMVA_Tools.h"
#include "Riostream.h"
#include "TObjString.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TH1.h"
#include "TSpline.h"
#include "TVector.h"
#include "TMatrixD.h"
#include "TVectorD.h"

#define TMVA_Tools_NAME_ "TMVA_Tools"

using namespace TMVA_Tools;

Double_t TMVA_Tools::NormVariable( Double_t x, Double_t xmin, Double_t xmax )
{
  // output range: [-1, 1]
  return 2*(x - xmin)/(xmax - xmin) - 1.0;
}

void TMVA_Tools::ComputeStat( TTree* theTree, TString theVarName,
			     Double_t& meanS, Double_t& meanB,
			     Double_t& rmsS,  Double_t& rmsB,
			     Double_t& xmin,  Double_t& xmax,
			     Bool_t    norm )
{
  // sanity check
  if (0 == theTree) {
    cout << "---" << TMVA_Tools_NAME_ << ": Error in TMVA_Tools::ComputeStat:"
	 << " tree is zero pointer ==> exit(1)" << endl;
    exit(1);
  }

  // does variable exist in tree?
  if (0 == theTree->FindBranch( theVarName )) {
    cout << "---" << TMVA_Tools_NAME_ << ": Error in TMVA_Tools::ComputeStat: variable: "
	 << theVarName << " is not member of tree ==> exit(1)" << endl;
    exit(1);
  }

  Long64_t entries = theTree->GetEntries();

  // first fill signal and background in arrays before analysis
  Double_t* varVecS  = new Double_t[entries];
  Double_t* varVecB  = new Double_t[entries];
  xmin               = +1e20;
  xmax               = -1e20;
  Long64_t nEventsS  = -1;
  Long64_t nEventsB  = -1;
  Double_t xmin_ = 0, xmax_ = 0;

  if (norm) {
    xmin_ = theTree->GetMinimum( theVarName );
    xmax_ = theTree->GetMaximum( theVarName );
  }

  for (Int_t ievt=0; ievt<entries; ievt++) {

    Double_t theVar = TMVA_Tools::GetValue( theTree, ievt, theVarName );
    if (norm) theVar = __N__( theVar, xmin_, xmax_ );

    if ((Int_t)TMVA_Tools::GetValue( theTree, ievt, "type" ) == 1) // this is signal
      varVecS[++nEventsS] = theVar;
    else  // this is background
      varVecB[++nEventsB] = theVar;

    if (theVar > xmax) xmax = theVar;
    if (theVar < xmin) xmin = theVar;
  }
  ++nEventsS;
  ++nEventsB;

  // basic statistics
  meanS = TMath::Mean( nEventsS, varVecS );
  meanB = TMath::Mean( nEventsB, varVecB );
  rmsS  = TMath::RMS ( nEventsS, varVecS );
  rmsB  = TMath::RMS ( nEventsB, varVecB );

  delete [] varVecS;
  delete [] varVecB;
}

void TMVA_Tools::ComputeStat( const std::vector<TMVA_Event*> eventCollection, Int_t ivar,
			     Double_t& meanS, Double_t& meanB,
			     Double_t& rmsS,  Double_t& rmsB,
			     Double_t& xmin,  Double_t& xmax,
			     Bool_t    norm )
{

  // does variable exist?
  if (ivar > eventCollection[0]->GetEventSize()){
    cout << "---" << TMVA_Tools_NAME_ << ": Error in TMVA_Tools::ComputeStat: variable: "
	 << ivar << " is too big ==> exit(1)" << endl;
    exit(1);
  }

  Int_t entries = eventCollection.size();
  // first fill signal and background in arrays before analysis
  Double_t* varVecS  = new Double_t[entries];
  Double_t* varVecB  = new Double_t[entries];
  xmin               = +1e20;
  xmax               = -1e20;
  Long64_t nEventsS  = -1;
  Long64_t nEventsB  = -1;
  Double_t xmin_ = 0, xmax_ = 0;

  std::vector<Double_t> content;
  if (norm) {
    for (int ie=0; ie<entries; ie++)content.push_back(eventCollection[ie]->GetData(ivar));
    xmax_ = *(std::max_element(content.begin(), content.end()));
    xmin_ = *(std::min_element(content.begin(), content.end()));
  }

  for (Int_t ievt=0; ievt<entries; ievt++) {
    if (norm) content[ievt] = __N__( content[ievt], xmin_, xmax_ );

    if (eventCollection[ievt]->GetType() == 1) // this is signal
      varVecS[++nEventsS] = eventCollection[ievt]->GetData(ivar);
    else  // this is background
      varVecB[++nEventsB] = eventCollection[ievt]->GetData(ivar);

    if (eventCollection[ievt]->GetData(ivar) > xmax) xmax = eventCollection[ievt]->GetData(ivar);
    if (eventCollection[ievt]->GetData(ivar) < xmin) xmin = eventCollection[ievt]->GetData(ivar);
  }
  ++nEventsS;
  ++nEventsB;

  // basic statistics
  meanS = TMath::Mean( nEventsS, varVecS );
  meanB = TMath::Mean( nEventsB, varVecB );
  rmsS  = TMath::RMS ( nEventsS, varVecS );
  rmsB  = TMath::RMS ( nEventsB, varVecB );

  delete [] varVecS;
  delete [] varVecB;
}

void TMVA_Tools::GetCovarianceMatrix( TTree* theTree, TMatrixDBase *theMatrix,
				     vector<TString>* theVars, Int_t theType, Bool_t norm )
{
  Long64_t      entries = theTree->GetEntries();
  const Int_t   nvar    = theVars->size();
  Int_t         ievt, ivar, jvar;
  TVectorD      vec(nvar);
  TMatrixD      mat2(nvar, nvar);
  TVectorD      xmin(nvar), xmax(nvar);

  // init matrices
  for (ivar=0; ivar<nvar; ivar++) {
    vec(ivar) = 0;
    if (norm) {
      xmin(ivar) = theTree->GetMinimum( (*theVars)[ivar] );
      xmax(ivar) = theTree->GetMaximum( (*theVars)[ivar] );
    }
    for (jvar=0; jvar<nvar; jvar++) {
      mat2(ivar, jvar) = 0;
    }
  }

  // event loop
  Int_t ic = 0;
  for (ievt=0; ievt<entries; ievt++) {

    if (Int_t(TMVA_Tools::GetValue( theTree, ievt, "type" )) == theType) {

      ic++; // count used events
      for (ivar=0; ivar<nvar; ivar++) {
	Double_t xi = TMVA_Tools::GetValue( theTree, ievt, (*theVars)[ivar] );
	if (norm) xi = __N__( xi, xmin(ivar), xmax(ivar) );
	vec(ivar) += xi;
	mat2(ivar, ivar) += (xi*xi);

	for (jvar=ivar+1; jvar<nvar; jvar++) {
	  Double_t xj = TMVA_Tools::GetValue( theTree, ievt, (*theVars)[jvar] );
	  if (norm) xj = __N__( xj, xmin(jvar), xmax(jvar) );
	  mat2(ivar, jvar) += (xi*xj);
	  mat2(jvar, ivar) = mat2(ivar, jvar); // symmetric matrix
	}
      }
    }
  }

  // variance-covariance
  Double_t n = (Double_t)ic;
  for (ivar=0; ivar<nvar; ivar++)
    for (jvar=0; jvar<nvar; jvar++)
      (*theMatrix)(ivar, jvar) = mat2(ivar, jvar)/n - vec(ivar)*vec(jvar)/pow(n,2);
}

void TMVA_Tools::GetCorrelationMatrix( TTree* theTree, TMatrixDBase *theMatrix,
				      vector<TString>* theVars, Int_t theType )
{
  TMVA_Tools::GetCovarianceMatrix( theTree, theMatrix, theVars, theType, kTRUE );

  // correlation
  const Int_t nvar = theVars->size();

  for (Int_t ivar=0; ivar<nvar; ivar++) {
    for (Int_t jvar=0; jvar<nvar; jvar++) {
      if (ivar != jvar) {
	Double_t d = (*theMatrix)(ivar, ivar)*(*theMatrix)(jvar, jvar);
	if (d > 0) (*theMatrix)(ivar, jvar) /= sqrt(d);
	else {
	  cout << "---" << TMVA_Tools_NAME_ << ": Warning: zero variances for variables "
	       << "(" << (*theVars)[ivar] << ", " << (*theVars)[jvar] << endl;
	  (*theMatrix)(ivar, jvar) = 0;
	}
      }
    }
  }

  for (Int_t ivar=0; ivar<nvar; ivar++) (*theMatrix)(ivar, ivar) = 1.0;
}

// square-root of symmetric matrix
// of course the resulting sqrtMat is also symmetric, but it's easier to
// treat it as a general matrix
void TMVA_Tools::GetSQRootMatrix( TMatrixDSym* symMat, TMatrixD* sqrtMat )
{
  Int_t n = symMat->GetNrows();

  // sanity check
  if (NULL != sqrtMat)
    if (sqrtMat->GetNrows() != n || sqrtMat->GetNcols() != n) {
      cout << "--- " << TMVA_Tools_NAME_ << ": mismatch in matrices ==> abort: "
	   << n << " " << sqrtMat->GetNrows() << " " << sqrtMat->GetNcols() << endl;
    }

  // compute eigenvectors
  TMatrixDSymEigen* eigen = new TMatrixDSymEigen( *symMat );

  // D = ST C S
  TMatrixD* si = new TMatrixD( eigen->GetEigenVectors() );
  TMatrixD* s  = new TMatrixD( *si ); // copy
  si->Transpose( *si ); // invert (= transpose)

  // diagonal matrices
  TMatrixD* d = new TMatrixD( n, n);
  d->Mult( (*si), (*symMat) ); (*d) *= (*s);

  // sanity check: matrix must be diagonal and positive definit
  Int_t i, j;
  Double_t epsilon = 1.0e-13;
  for (i=0; i<n; i++) {
    for (j=0; j<n; j++) {
      if ((i != j && TMath::Abs((*d)(i,j)) > epsilon) ||
	  (i == j && (*d)(i,i) < 0)) {
	cout << "--- " << TMVA_Tools_NAME_
	     << ": Error in matrix diagonalization; printing S and B ==> abort" << endl;
	d->Print();
	exit(1);
      }
    }
  }

  // make exactly diagonal
  for (i=0; i<n; i++) for (j=0; j<n; j++) if (j != i) (*d)(i,j) = 0;

  // compute the square-root C' of covariance matrix: C = C'*C'
  for (i=0; i<n; i++) (*d)(i,i) = sqrt((*d)(i,i));
  if (NULL == sqrtMat) sqrtMat = new TMatrixD( n, n );
  sqrtMat->Mult( (*s), (*d) );
  (*sqrtMat) *= (*si);

  // invert square-root matrices
  sqrtMat->Invert();

  delete eigen;
  delete s;
  delete si;
  delete d;
}

TH1* TMVA_Tools::projNormTH1F( TTree* theTree, TString theVarName,
			      TString name, Int_t nbins,
			      Double_t xmin, Double_t xmax, TString cut )
{
  TH1* hist = new TH1F( name, name, nbins, xmin, xmax );
  hist->Sumw2(); // enable quadratic errors
  theTree->Project( name, theVarName, cut );
  NormHist( hist );
  return hist;
}

Double_t TMVA_Tools::NormHist( TH1* theHist, Double_t norm )
{
  if (NULL == theHist) {
    cout << "--- " << TMVA_Tools_NAME_ << "::NormHist: null TH1 pointer ==> abort" << endl;
    exit(1);
  }
  TAxis* tx  = theHist->GetXaxis();
  Double_t w = ((theHist->GetEntries() > 0 ? theHist->GetEntries() : 1)
		* (tx->GetXmax() - tx->GetXmin())/tx->GetNbins());
  theHist->Scale( (w > 0) ? norm/w : norm );
  return w;
}

TList* TMVA_Tools::ParseFormatLine( TString formatString )
{
  // Parse the string and cut into labels separated by ":"

  TList*   labelList = new TList();
  TString* label     = new TString();
  Int_t    nLabels   = 0;

  const Int_t n = (Int_t)formatString.Length();
  TObjString** label_obj = new TObjString*[n];  // array of labels

  for (Int_t i=0; i<n; i++) {
    label->Append(formatString(i));
    if (formatString(i)==':') {
      label->Chop();
      label_obj[nLabels] = new TObjString(label->Data());
      labelList->Add(label_obj[nLabels]);
      label->Resize(0);
      nLabels++;
    }
    if (i == n-1) {
      label_obj[nLabels] = new TObjString(label->Data());
      labelList->Add(label_obj[nLabels]);
      label->Resize(0);
      nLabels++;
    }
  }
  delete label;
  delete [] label_obj;
  return labelList;
}

Double_t TMVA_Tools::GetValue( TTree *theTree, Int_t entry, TString varname )
{
  // branch addresses
  static Float_t  f = 0;
  static Double_t d = 0;
  static Int_t    i = 0;

  // sanity check
  if (0 == theTree) {
    cout << "---" << TMVA_Tools_NAME_ << ": fatal error: zero tree pointer ==> exit(1) " << endl;
    exit(1);
  }

  // return value
  Double_t retval = -1;

  TBranch* branch = theTree->GetBranch( varname );
  if (0 != branch) {

    TLeaf *leaf = branch->GetLeaf(branch->GetName());

    if (((TString)leaf->GetTypeName()).Contains("Int_t")) {     
      branch->SetAddress(&i);
      branch->GetEntry(entry);
      retval = (Double_t)i;
    }
    else if (((TString)leaf->GetTypeName()).Contains("Float_t")) {
      branch->SetAddress(&f);
      branch->GetEntry(entry);
      retval = (Double_t)f;
    }
    else if (((TString)leaf->GetTypeName()).Contains("Double_t")) {
      branch->SetAddress(&d);
      branch->GetEntry(entry);
      retval = (Double_t)d;
    }

  } // end of found right branch
  else {
    cout << "---" << TMVA_Tools_NAME_ << ": branch " << varname
	 << " does not exist in tree" << endl;
    cout << "---" << TMVA_Tools_NAME_ << ": candidates are:" << endl;
    TIter next_branch1( theTree->GetListOfBranches() );
    while (TBranch *branch = (TBranch*)next_branch1())
      cout << "---\t" << branch->GetName() << endl;
  }

  return retval;
}

// check quality of splining by comparing splines and histograms in each bin
Bool_t TMVA_Tools::CheckSplines( TH1* theHist, TSpline* theSpline )
{
  const Double_t sanityCrit = 0.01; // relative deviation

  Bool_t retval = kTRUE;
  for (Int_t ibin=1; ibin<=theHist->GetNbinsX(); ibin++) {
    Double_t x  = theHist->GetBinCenter( ibin );
    Double_t yh = theHist->GetBinContent( ibin ); // the histogram output
    Double_t ys = theSpline->Eval( x );           // the spline output

    if (ys + yh > 0) {
      Double_t dev = 0.5*(ys - yh)/(ys + yh);
      if (TMath::Abs(dev) > sanityCrit) {
 	cout << "---" << TMVA_Tools_NAME_ << ": Warning: Spline failed sanity criterion; "
	     << " relative deviation from histogram: " << dev
	     << " in (bin, value): (" << ibin << ", " << x << ")" << endl;
	retval = kFALSE;
      }
    }
  }

  return retval;
}



vector<Double_t> TMVA_Tools::MVADiff(vector<Double_t> & a, vector<Double_t> & b)
{
  if (a.size()!=b.size()) {
    //printf("try to make diff on two vectors with different length a=%d b=%d\n",
    //       a.size(), b.size());
    throw;
  }
  vector<Double_t> result(a.size());
  for (UInt_t i=0; i<a.size();i++) result[i]=a[i]-b[i];
  return result;
}

void TMVA_Tools::Scale( vector<Double_t> &v, Double_t f )
{
  for (UInt_t i=0; i<v.size();i++) v[i]*=f;
}

void TMVA_Tools::Scale( vector<Float_t> &v, Float_t f )
{
  for (UInt_t i=0; i<v.size();i++) v[i]*=f;
}

Int_t TMVA_Tools::LocateMax(vector<Double_t> v)
{
  Int_t position=-1;
  vector<Double_t>::iterator i=max_element(v.begin(),v.end());
  for (vector<Double_t>::iterator it=v.begin(); it!=v.end();it++) {
    position++;
    if (it == i) break;
  }
  return position;
}

Int_t TMVA_Tools::LocateMin(vector<Double_t> v)
{
  Int_t position=-1;
  vector<Double_t>::iterator i=min_element(v.begin(),v.end());
  for (vector<Double_t>::iterator it=v.begin(); it!=v.end();it++) {
    position++;
    if (it == i) break;
  }
  return position;
}

void TMVA_Tools::UsefulSortAscending(vector< vector<Double_t> > &v)
{
  UInt_t nArrays=v.size();
  Double_t temp;
  if (nArrays > 0) {
    UInt_t sizeofarray=v[0].size();
    for (UInt_t i=0; i<sizeofarray; i++) {
      for (UInt_t j=sizeofarray-1; j>i; j--) {
	if (v[0][j-1] > v[0][j]) {
 	  for (UInt_t k=0; k< nArrays; k++) {
 	    temp = v[k][j-1];v[k][j-1] = v[k][j]; v[k][j] = temp;
 	  }
	}
      }
    }
  }
}

void TMVA_Tools::UsefulSortDescending(vector< vector<Double_t> > &v, vector<TString>* vs)
{
  UInt_t nArrays=v.size();
  Double_t temp;
  if (nArrays > 0) {
    UInt_t sizeofarray=v[0].size();
    for (UInt_t i=0; i<sizeofarray; i++) {
      for (UInt_t j=sizeofarray-1; j>i; j--) {
	if (v[0][j-1] < v[0][j]) {
 	  for (UInt_t k=0; k< nArrays; k++) {
 	    temp = v[k][j-1]; v[k][j-1] = v[k][j]; v[k][j] = temp;
	  }
	  if (NULL != vs) {
	    TString temps = (*vs)[j-1]; (*vs)[j-1] = (*vs)[j]; (*vs)[j] = temps;
	  }
	}
      }
    }
  }
}

void TMVA_Tools::UsefulSortDescending(vector<Double_t> &v)
{
  vector< vector<Double_t> > vtemp;
  vtemp.push_back(v);
  UsefulSortDescending(vtemp);
  v = vtemp[0];
}

void TMVA_Tools::UsefulSortAscending(vector<Double_t>  &v)
{
  vector<vector<Double_t> > vtemp;
  vtemp.push_back(v);
  UsefulSortAscending(vtemp);
  v=vtemp[0];
}

int TMVA_Tools::GetIndexMaxElement(vector<Double_t>  &v)
{
  if (v.size()==0) return -1;

  Int_t pos=0; Double_t mx=v[0];
  for (UInt_t i=0; i<v.size(); i++){
    if (v[i] > mx){
      mx=v[i];
      pos=i;
    }
  }
  return pos;
}

int TMVA_Tools::GetIndexMinElement(vector<Double_t>  &v)
{
  if (v.size()==0) return -1;

  Int_t pos=0; Double_t mn=v[0];
  for (UInt_t i=0; i<v.size(); i++){
    if (v[i] < mn){
      mn=v[i];
      pos=i;
    }
  }
  return pos;
}

// ============================================================================
