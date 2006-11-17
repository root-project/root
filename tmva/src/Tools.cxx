// @(#)root/tmva $Id: Tools.cxx,v 1.55 2006/11/16 22:51:59 helgevoss Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Tools                                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        *
 *      U. of Victoria, Canada,                                                   *
 *      MPI-K Heidelberg, Germany ,                                               *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://ttmva.sourceforge.net/LICENSE)                                         *
 **********************************************************************************/

#include <algorithm>
#include "Riostream.h"
#include "TObjString.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TH1.h"
#include "TSpline.h"
#include "TVector.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TTreeFormula.h"

#ifndef TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef TMVA_Event
#include "TMVA/Event.h"
#endif

namespace TMVA {

   namespace Tools {
      const char* Tools_NAME_ = "Tools"; // name to locate output
      static MsgLogger* Tools_Logger = 0;
   }   
}

TMVA::MsgLogger& TMVA::Tools::Logger()
{
   // static access to a common MsgLogger

   return Tools_Logger ? *Tools_Logger : *(Tools_Logger = new MsgLogger( Tools_NAME_ ));
}

Double_t TMVA::Tools::NormVariable( Double_t x, Double_t xmin, Double_t xmax )
{
   // normalise to output range: [-1, 1]
   return 2*(x - xmin)/(xmax - xmin) - 1.0;
}

void TMVA::Tools::ComputeStat( TTree* theTree, const TString& theVarName,
                               Double_t& meanS, Double_t& meanB,
                               Double_t& rmsS,  Double_t& rmsB,
                               Double_t& xmin,  Double_t& xmax,
                               Bool_t    norm )
{
   // sanity check
   if (0 == theTree) Logger() << kFATAL << "<ComputeStat> tree is zero pointer" << Endl;

   // does variable exist in tree?
   if (0 == theTree->FindBranch( theVarName )) 
      Logger() << kFATAL << "<ComputeStat> variable: " << theVarName << " is not member of tree" << Endl;

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

   static Int_t    theType;
   TBranch * br1 = theTree->GetBranch("type" );
   br1->SetAddress( & theType );

   static Double_t theVarD = 0;
   static Float_t  theVarF = 0;
   static Int_t    theVarI = 0;

   TBranch * br2 = theTree->GetBranch( theVarName );
   TString leafType = ((TLeaf*)br2->GetListOfLeaves()->At(0))->GetTypeName();
   Int_t tIdx = -1;
   if (leafType=="Double_t") {
      tIdx=0;
      br2->SetAddress( & theVarD );
   } 
   else if(leafType=="Float_t") {
      tIdx=1;
      br2->SetAddress( & theVarF );
   } 
   else if(leafType=="Int_t") {
      tIdx=2;
      br2->SetAddress( & theVarI );
   } 
   else {
      Logger() << kFATAL << "<ComputeStat> unknown Variable Type " << leafType << Endl;
   }

   for (Int_t ievt=0; ievt<entries; ievt++) {
      br1->GetEntry(ievt);
      br2->GetEntry(ievt);
      Double_t theVar = 0;
      switch(tIdx) {
      case 0: theVar = theVarD; break;
      case 1: theVar = theVarF; break;
      case 2: theVar = theVarI; break;
      }
      if (norm) theVar = __N__( theVar, xmin_, xmax_ );

      if(theType == 1) // this is signal
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

void TMVA::Tools::GetSQRootMatrix( TMatrixDSym* symMat, TMatrixD*& sqrtMat )
{
   // square-root of symmetric matrix
   // of course the resulting sqrtMat is also symmetric, but it's easier to
   // treat it as a general matrix
   Int_t n = symMat->GetNrows();

   // sanity check
   if (NULL != sqrtMat) {
      if (sqrtMat->GetNrows() != n || sqrtMat->GetNcols() != n) {
         Logger() << kFATAL << "<GetSQRootMatrix> mismatch in matrices: "
                  << n << " " << sqrtMat->GetNrows() << " " << sqrtMat->GetNcols() << Endl;
      }
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
   Double_t epsilon = 1.0e-8;
   for (i=0; i<n; i++) {
      for (j=0; j<n; j++) {
         if ((i != j && TMath::Abs((*d)(i,j))/TMath::Sqrt((*d)(i,i)*(*d)(j,j)) > epsilon) ||
             (i == j && (*d)(i,i) < 0)) {
            //d->Print();
            Logger() << kWARNING << "<GetSQRootMatrix> error in matrix diagonalization; printed S and B" << Endl;
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

TH1* TMVA::Tools::projNormTH1F( TTree* theTree, TString theVarName,
                                TString name, Int_t nbins,
                                Double_t xmin, Double_t xmax, TString cut )
{
   // projects variable from tree into normalised histogram
 
   // needed because of ROOT bug (feature) that excludes events that have value == xmax
   xmax += 0.00001; 
   
   TH1* hist = new TH1F( name, name, nbins, xmin, xmax );
   hist->Sumw2(); // enable quadratic errors
   theTree->Project( name, theVarName, cut );
   NormHist( hist );
   return hist;
}

Double_t TMVA::Tools::NormHist( TH1* theHist, Double_t norm )
{
   // normalises histogram
   if (NULL == theHist) Logger() << kFATAL << "<NormHist> null TH1 pointer" << Endl;

   TAxis* tx  = theHist->GetXaxis();
   Double_t w = ((theHist->GetEntries() > 0 ? theHist->GetEntries() : 1)
                 * (tx->GetXmax() - tx->GetXmin())/tx->GetNbins());
   theHist->Scale( (w > 0) ? norm/w : norm );
   return w;
}

TList* TMVA::Tools::ParseFormatLine( TString formatString )
{
   // Parse the string and cut into labels separated by ":"
   TList*   labelList = new TList();
   TString  label;

   const Int_t n = (Int_t)formatString.Length();

   for (Int_t i=0; i<n; i++) {
      label.Append(formatString(i));
      if (formatString(i)==':') {
         label.Chop();
         labelList->Add(new TObjString(label.Data()));
         label.Resize(0);
      }
      if (i == n-1) {
         labelList->Add(new TObjString(label.Data()));
         label.Resize(0);
      }
   }
   return labelList;                                                 
}


vector<Int_t>* TMVA::Tools::ParseANNOptionString( TString theOptions, Int_t nvar,
                                                  vector<Int_t>* nodes )
{
   // parse option string for ANN methods
   // default settings (should be defined in theOption string)
   TList*  list  = TMVA::Tools::ParseFormatLine( theOptions );

   // format and syntax of option string: "3000:N:N+2:N-3:6"
   //
   // where:
   //        3000 - number of training cycles (epochs)
   //        N    - number of nodes in first hidden layer, where N is the number
   //               of discriminating variables used (note that the first ANN
   //               layer necessarily has N nodes, and hence is not given).
   //        N+2  - number of nodes in 2nd hidden layer (2 nodes more than
   //               number of variables)
   //        N-3  - number of nodes in 3rd hidden layer (3 nodes less than
   //               number of variables)
   //        6    - 6 nodes in last (4th) hidden layer (note that the last ANN
   //               layer in MVA has 2 nodes, each one for signal and background
   //               classes)

   // sanity check
   if (list->GetSize() < 1) {
      Logger() << kFATAL << "<ParseANNOptionString> unrecognized option string: " << theOptions << Endl;
   }

   // add number of cycles
   nodes->push_back( atoi( ((TObjString*)list->At(0))->GetString() ) );

   Int_t a;
   if (list->GetSize() > 1) {
      for (Int_t i=1; i<list->GetSize(); i++) {
         TString s = ((TObjString*)list->At(i))->GetString();
         s.ToUpper();
         if (s(0) == 'N')  {
            if (s.Length() > 1) nodes->push_back( nvar + atoi(&s[1]) );
            else                nodes->push_back( nvar );
         }
         else if ((a = atoi( s )) > 0) nodes->push_back( atoi(s ) );
         else {
            Logger() << kFATAL << "<ParseANNOptionString> unrecognized option string: " << theOptions << Endl;
         }
      }
   }

   return nodes;
}

Bool_t TMVA::Tools::CheckSplines( TH1* theHist, TSpline* theSpline )
{
   // check quality of splining by comparing splines and histograms in each bin
   const Double_t sanityCrit = 0.01; // relative deviation

   Bool_t retval = kTRUE;
   for (Int_t ibin=1; ibin<=theHist->GetNbinsX(); ibin++) {
      Double_t x  = theHist->GetBinCenter( ibin );
      Double_t yh = theHist->GetBinContent( ibin ); // the histogram output
      Double_t ys = theSpline->Eval( x );           // the spline output

      if (ys + yh > 0) {
         Double_t dev = 0.5*(ys - yh)/(ys + yh);
         if (TMath::Abs(dev) > sanityCrit) {
            Logger() << kFATAL << "<CheckSplines> Spline failed sanity criterion; "
                     << " relative deviation from histogram: " << dev
                     << " in (bin, value): (" << ibin << ", " << x << ")" << Endl;
            retval = kFALSE;
         }
      }
   }

   return retval;
}

vector<Double_t> TMVA::Tools::MVADiff(vector<Double_t> & a, vector<Double_t> & b)
{
   // computes difference between two vectors
   if (a.size() != b.size()) {
      throw;
   }
   vector<Double_t> result(a.size());
   for (UInt_t i=0; i<a.size();i++) result[i]=a[i]-b[i];
   return result;
}

void TMVA::Tools::Scale( vector<Double_t> &v, Double_t f )
{
   // scales double vector
   for (UInt_t i=0; i<v.size();i++) v[i]*=f;
}

void TMVA::Tools::Scale( vector<Float_t> &v, Float_t f )
{
   // scales float vector
   for (UInt_t i=0; i<v.size();i++) v[i]*=f;
}

void TMVA::Tools::UsefulSortAscending(vector< vector<Double_t> > &v)
{
   // sort 2D vector
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

void TMVA::Tools::UsefulSortDescending(vector< vector<Double_t> > &v, vector<TString>* vs)
{
   // sort 2D vector AND (in parallel) TString vector
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

void TMVA::Tools::UsefulSortDescending(vector<Double_t> &v)
{
   // sort vector
   vector< vector<Double_t> > vtemp;
   vtemp.push_back(v);
   UsefulSortDescending(vtemp);
   v = vtemp[0];
}

void TMVA::Tools::UsefulSortAscending(vector<Double_t>  &v)
{
   // sort vector
   vector<vector<Double_t> > vtemp;
   vtemp.push_back(v);
   UsefulSortAscending(vtemp);
   v=vtemp[0];
}

int TMVA::Tools::GetIndexMaxElement(vector<Double_t>  &v)
{
   // find index of maximum entry in vector
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

int TMVA::Tools::GetIndexMinElement(vector<Double_t>  &v)
{
   // find index of minimum entry in vector
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

// check if regular expression
Bool_t TMVA::Tools::ContainsRegularExpression( const TString& s )  
{
   // helper function to search for "!%^&()'<>?= " in a string

   Bool_t  regular = kFALSE;
   for (Int_t i = 0; i < TMVA::Tools::__regexp__.Length(); i++) 
      if (s.Contains( TMVA::Tools::__regexp__[i] )) { regular = kTRUE; break; }

   return regular;
}

// replace regular expressions
TString TMVA::Tools::ReplaceRegularExpressions( const TString& s, TString r )  
{
   // helper function to remove all occurences "!%^&()'<>?= " from a string
   // and replace all ::,*,/,+,- with _M_,_T_,_D_,_P_,_M_ respectively

   TString snew = s;
   for (Int_t i = 0; i < TMVA::Tools::__regexp__.Length(); i++) 
      snew.ReplaceAll( TMVA::Tools::__regexp__[i], r );

   snew.ReplaceAll( "::", r );
   snew.ReplaceAll( "*", "_T_" );
   snew.ReplaceAll( "/", "_D_" );
   snew.ReplaceAll( "+", "_P_" );
   snew.ReplaceAll( "-", "_M_" );

   return snew;
}

void TMVA::Tools::FormattedOutput( const TMatrixD& M, const std::vector<TString>& V, MsgLogger& logger )
{
   // sanity check: matrix must be quadratic
   UInt_t nvar = V.size();
   if ((UInt_t)M.GetNcols() != nvar || (UInt_t)M.GetNrows() != nvar) {
      logger << kFATAL << "<FormattedOutput> fatal error with dimensions: " 
             << M.GetNcols() << " OR " << M.GetNrows() << " != " << nvar << " ==> abort" << Endl;
   }

   // get length of each variable, and maximum length  
   UInt_t minL = 7;
   UInt_t maxL = minL;
   std::vector<UInt_t> vLengths;
   for (UInt_t ivar=0; ivar<nvar; ivar++) {
      vLengths.push_back(TMath::Max( (UInt_t)V[ivar].Length(), minL ));
      maxL = TMath::Max( vLengths.back(), maxL );
   }
   
   // count column length
   UInt_t clen = maxL+1;
   for (UInt_t icol=0; icol<nvar; icol++) clen += vLengths[icol]+1;

   // bar line
   for (UInt_t i=0; i<clen; i++) logger << "-";
   logger << Endl;

   // title bar   
   logger << setw(maxL+1) << " ";
   for (UInt_t icol=0; icol<nvar; icol++) logger << setw(vLengths[icol]+1) << V[icol];
   logger << Endl;

   // the numbers
   for (UInt_t irow=0; irow<nvar; irow++) {
      logger << setw(maxL) << V[irow] << ":";
      for (UInt_t icol=0; icol<nvar; icol++) {
         logger << setw(vLengths[icol]+1) << Form( "%+1.3f", M(irow,icol) );
      }      
      logger << Endl;
   }

   // bar line
   for (UInt_t i=0; i<clen; i++) logger << "-";
   logger << Endl;
}

const TString BC_blue   = "\033[1;34m" ;
const TString BC_red    = "\033[1;31m" ;
const TString BC__      = "\033[1m"    ;
const TString EC__      = "\033[0m"    ;

void TMVA::Tools::TMVAWelcomeMessage()
{
   // direct output, eg, when starting ROOT session -> no use of Logger here
   cout  << endl
         << BC__ << "T" << "MVA -- Toolkit for Multivariate Analysis"
         << EC__ << endl;
   cout << "        " << "Copyright (C) 2005-2006 CERN, LAPP & MPI-K Heidelberg and Victoria" << endl;
   cout << "        " << "Home page http://tmva.sourceforge.net" << endl;
   cout << "        " << "All rights reserved, please read http://tmva.sf.net/license.txt" << endl << endl;
}

