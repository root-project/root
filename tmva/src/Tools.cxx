// @(#)root/tmva $Id: Tools.cxx,v 1.13 2007/06/19 13:26:21 brun Exp $   
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
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://ttmva.sourceforge.net/LICENSE)                                         *
 **********************************************************************************/

#include <algorithm>
#include "Riostream.h"
#include "TObjString.h"
#include "TMath.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TH1.h"
#include "TList.h"
#include "TSpline.h"
#include "TVector.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TTreeFormula.h"

#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_Config
#include "TMVA/Config.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif
#ifndef ROOT_TMVA_Version
#include "TMVA/Version.h"
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

//_______________________________________________________________________
Double_t TMVA::Tools::GetSeparation( TH1* S, TH1* B ) 
{
   // compute "separation" defined as
   // <s2> = (1/2) Int_-oo..+oo { (S^2(x) - B^2(x))/(S(x) + B(x)) dx }
   Double_t separation = 0;
   
   // sanity checks
   // signal and background histograms must have same number of bins and 
   // same limits
   if ((S->GetNbinsX() != B->GetNbinsX()) || (S->GetNbinsX() <= 0)) {
      Logger() << kFATAL << "<GetSeparation> signal and background"
               << " histograms have different number of bins: " 
               << S->GetNbinsX() << " : " << B->GetNbinsX() << Endl;
   }

   if (S->GetXaxis()->GetXmin() != B->GetXaxis()->GetXmin() || 
       S->GetXaxis()->GetXmax() != B->GetXaxis()->GetXmax() || 
       S->GetXaxis()->GetXmax() <= S->GetXaxis()->GetXmin()) {
      Logger() << kINFO << S->GetXaxis()->GetXmin() << " " << B->GetXaxis()->GetXmin() 
               << " " << S->GetXaxis()->GetXmax() << " " << B->GetXaxis()->GetXmax() 
               << " " << S->GetXaxis()->GetXmax() << " " << S->GetXaxis()->GetXmin() << Endl;
      Logger() << kFATAL << "<GetSeparation> signal and background"
               << " histograms have different or invalid dimensions:" << Endl;
   }

   Int_t    nstep  = S->GetNbinsX();
   Double_t intBin = (S->GetXaxis()->GetXmax() - S->GetXaxis()->GetXmin())/nstep;
   Double_t nS     = S->GetEntries()*intBin;
   Double_t nB     = B->GetEntries()*intBin;
   if (nS > 0 && nB > 0) {
      for (Int_t bin=0; bin<nstep; bin++) {
         Double_t s = S->GetBinContent( bin )/nS;
         Double_t b = B->GetBinContent( bin )/nB;
         // separation
         if (s + b > 0) separation += 0.5*(s - b)*(s - b)/(s + b);
      }
      separation *= intBin;
   }
   else {
      Logger() << kWARNING << "<GetSeparation> histograms with zero entries: " 
               << nS << " : " << nB << " cannot compute separation"
               << Endl;
      separation = 0;
   }

   return separation;
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
   else if (leafType=="Float_t") {
      tIdx=1;
      br2->SetAddress( & theVarF );
   } 
   else if (leafType=="Int_t") {
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
      if (norm) theVar = Tools::NormVariable( theVar, xmin_, xmax_ );

      if (theType == 1) varVecS[++nEventsS] = theVar; // this is signal
      else              varVecB[++nEventsB] = theVar; // this is background

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

TMatrixD* TMVA::Tools::GetSQRootMatrix( TMatrixDSym* symMat )
{
   // square-root of symmetric matrix
   // of course the resulting sqrtMat is also symmetric, but it's easier to
   // treat it as a general matrix
   Int_t n = symMat->GetNrows();

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
   for (i=0; i<n; i++) (*d)(i,i) = TMath::Sqrt((*d)(i,i));

   TMatrixD* sqrtMat = new TMatrixD( n, n );
   sqrtMat->Mult( (*s), (*d) );
   (*sqrtMat) *= (*si);

   // invert square-root matrices
   sqrtMat->Invert();

   delete eigen;
   delete s;
   delete si;
   delete d;

   return sqrtMat;
}

const TMatrixD* TMVA::Tools::GetCorrelationMatrix( const TMatrixD* covMat )
{
   // turns covariance into correlation matrix   
   if (covMat == 0) return 0;

   // sanity check
   Int_t nvar = covMat->GetNrows();
   if (nvar != covMat->GetNcols()) 
      Logger() << kFATAL << "<GetCorrelationMatrix> input matrix not quadratic" << Endl;
   
   TMatrixD* corrMat = new TMatrixD( nvar, nvar );

   for (Int_t ivar=0; ivar<nvar; ivar++) {
      for (Int_t jvar=0; jvar<nvar; jvar++) {
         if (ivar != jvar) {
            Double_t d = (*covMat)(ivar, ivar)*(*covMat)(jvar, jvar);
            if (d > 0) (*corrMat)(ivar, jvar) = (*covMat)(ivar, jvar)/TMath::Sqrt(d);
            else {
               Logger() << kWARNING << "<GetCorrelationMatrix> zero variances for variables "
                       << "(" << ivar << ", " << jvar << ")" << Endl;
               (*corrMat)(ivar, jvar) = 0;
            }
         }
         else (*corrMat)(ivar, ivar) = 1.0;
      }
   }

   return corrMat;
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

TList* TMVA::Tools::ParseFormatLine( TString formatString, const char* sep )
{
   // Parse the string and cut into labels separated by ":"
   TList*   labelList = new TList();
   while (formatString.First(sep)==0) formatString.Remove(0,1); // remove initial separators

   while (formatString.Length()>0) {
      if (formatString.First(sep) == -1) { // no more separator
         labelList->Add(new TObjString(formatString.Data()));
         formatString="";
         break;
      }

      Ssiz_t posSep = formatString.First(sep);
      labelList->Add(new TObjString(TString(formatString(0,posSep)).Data()));
      formatString.Remove(0,posSep+1);
      
      while(formatString.First(sep)==0) formatString.Remove(0,1); // // remove additional separators
      
   }
   return labelList;                                                 
}

vector<Int_t>* TMVA::Tools::ParseANNOptionString( TString theOptions, Int_t nvar,
                                                  vector<Int_t>* nodes )
{
   // parse option string for ANN methods
   // default settings (should be defined in theOption string)
   TList* list  = TMVA::Tools::ParseFormatLine( theOptions, ":" );

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

Bool_t TMVA::Tools::CheckSplines( const TH1* theHist, const TSpline* theSpline )
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

Int_t TMVA::Tools::GetIndexMaxElement( vector<Double_t>  &v )
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

Int_t TMVA::Tools::GetIndexMinElement( vector<Double_t>  &v )
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
   for (Int_t i = 0; i < Tools::__regexp__.Length(); i++) 
      if (s.Contains( Tools::__regexp__[i] )) { regular = kTRUE; break; }

   return regular;
}

// replace regular expressions
TString TMVA::Tools::ReplaceRegularExpressions( const TString& s, const TString& r )  
{
   // helper function to remove all occurences "!%^&()'<>?= " from a string
   // and replace all ::,*,/,+,- with _M_,_T_,_D_,_P_,_M_ respectively

   TString snew = s;
   for (Int_t i = 0; i < Tools::__regexp__.Length(); i++) 
      snew.ReplaceAll( Tools::__regexp__[i], r );

   snew.ReplaceAll( "::", r );
   snew.ReplaceAll( "*", "_T_" );
   snew.ReplaceAll( "/", "_D_" );
   snew.ReplaceAll( "+", "_P_" );
   snew.ReplaceAll( "-", "_M_" );
   snew.ReplaceAll( " ", "_" );
   snew.ReplaceAll( "[", "_" );
   snew.ReplaceAll( "]", "_" );

   return snew;
}

//_______________________________________________________________________
const TString& TMVA::Tools::Color( const TString & c ) 
{
   // human readable color strings
   static TString gClr_none         = "" ;
   static TString gClr_white        = "\033[1;37m";  // white
   static TString gClr_black        = "\033[30m";    // black
   static TString gClr_blue         = "\033[34m";    // blue
   static TString gClr_red          = "\033[1;31m" ; // red
   static TString gClr_yellow       = "\033[1;33m";  // yellow
   static TString gClr_darkred      = "\033[31m";    // dark red
   static TString gClr_darkgreen    = "\033[32m";    // dark green
   static TString gClr_darkyellow   = "\033[33m";    // dark yellow
                                    
   static TString gClr_bold         = "\033[1m"    ; // bold 
   static TString gClr_black_b      = "\033[30m"   ; // bold black
   static TString gClr_lblue_b      = "\033[1;34m" ; // bold light blue
   static TString gClr_lgreen_b     = "\033[1;32m";  // bold light green
                                    
   static TString gClr_blue_bg      = "\033[44m";    // blue background
   static TString gClr_red_bg       = "\033[1;41m";  // white on red background
   static TString gClr_whiteonblue  = "\033[1;44m";  // white on blue background
   static TString gClr_whiteongreen = "\033[1;42m";  // white on green background
   static TString gClr_grey_bg      = "\033[47m";    // grey background

   static TString gClr_reset  = "\033[0m";     // reset

   if (!gConfig().UseColor()) return gClr_none;

   if (c == "white" )         return gClr_white; 
   if (c == "blue"  )         return gClr_blue; 
   if (c == "black"  )        return gClr_black; 
   if (c == "lightblue")      return gClr_lblue_b;
   if (c == "yellow")         return gClr_yellow; 
   if (c == "red"   )         return gClr_red; 
   if (c == "dred"  )         return gClr_darkred; 
   if (c == "dgreen")         return gClr_darkgreen; 
   if (c == "lgreenb")        return gClr_lgreen_b;
   if (c == "dyellow")        return gClr_darkyellow; 

   if (c == "bold")           return gClr_bold; 
   if (c == "bblack")         return gClr_black_b; 

   if (c == "blue_bgd")       return gClr_blue_bg; 
   if (c == "red_bgd" )       return gClr_red_bg; 
 
   if (c == "white_on_blue" ) return gClr_whiteonblue; 
   if (c == "white_on_green") return gClr_whiteongreen; 

   if (c == "reset") return gClr_reset; 

   cout << "Unknown color " << c << endl;
   exit(1);

   return gClr_none;
}

void TMVA::Tools::FormattedOutput( const std::vector<Double_t>& values, const std::vector<TString>& V, 
                                   const TString titleVars, const TString titleValues, MsgLogger& logger,
                                   TString format )
{
   // formatted output of simple table

   // sanity check
   UInt_t nvar = V.size();
   if ((UInt_t)values.size() != nvar) {
      logger << kFATAL << "<FormattedOutput> fatal error with dimensions: " 
             << values.size() << " OR " << " != " << nvar << Endl;
   }

   // find maximum length in V (and column title)
   UInt_t maxL = 7;
   std::vector<UInt_t> vLengths;
   for (UInt_t ivar=0; ivar<nvar; ivar++) maxL = TMath::Max( (UInt_t)V[ivar].Length(), maxL );
   maxL = TMath::Max( (UInt_t)titleVars.Length(), maxL );

   // column length
   UInt_t maxV = 7;
   maxV = TMath::Max( (UInt_t)titleValues.Length() + 1, maxL );

   // full column length
   UInt_t clen = maxL + maxV + 3;

   // bar line
   for (UInt_t i=0; i<clen; i++) logger << "-";
   logger << Endl;

   // title bar   
   logger << setw(maxL) << titleVars << ":";
   logger << setw(maxV+1) << titleValues << ":";
   logger << Endl;
   for (UInt_t i=0; i<clen; i++) logger << "-";
   logger << Endl;

   // the numbers
   for (UInt_t irow=0; irow<nvar; irow++) {
      logger << setw(maxL) << V[irow] << ":";
      logger << setw(maxV+1) << Form( format.Data(), values[irow] );
      logger << Endl;
   }

   // bar line
   for (UInt_t i=0; i<clen; i++) logger << "-";
   logger << Endl;
}

void TMVA::Tools::FormattedOutput( const TMatrixD& M, const std::vector<TString>& V, MsgLogger& logger )
{
   // formatted output of matrix (with labels)

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

void TMVA::Tools::writeFloatArbitraryPrecision(Float_t val, ostream & os) {
   // writes a float value with the available precision to a stream
   os << val << " :: ";
   void * c = &val;
   for(int i=0; i<4; i++) {
      Int_t ic = *((char*)c+i)-'\0';
      if(ic<0) ic+=256;
      os << ic << " ";
   }
   os << ":: ";
}

void TMVA::Tools::readFloatArbitraryPrecision(Float_t & val, istream & is) {
   // reads a float value with the available precision from a stream
   Float_t a = 0;
   is >> a;
   TString dn;
   is >> dn;
   Int_t c[4];
   void * ap = &a;
   for(int i=0; i<4; i++) {
      is >> c[i];
      *((char*)ap+i) = '\0'+c[i];
   }
   is >> dn;
   val = a;
}



void TMVA::Tools::TMVAWelcomeMessage()
{
   // direct output, eg, when starting ROOT session -> no use of Logger here
   cout << endl;
   cout << Color("bold") << "TMVA -- Toolkit for Multivariate Analysis" << Color("reset") << endl;
   cout << "        " << "Copyright (C) 2005-2007 CERN, MPI-K Heidelberg and Victoria U." << endl;
   cout << "        " << "Home page http://tmva.sourceforge.net" << endl;
   cout << "        " << "All rights reserved, please read http://tmva.sf.net/license.txt" << endl << endl;
}

void TMVA::Tools::TMVAVersionMessage( MsgLogger& logger )
{
   // prints the release number and date
   logger << "________________Version " << TMVA_RELEASE << ", " << TMVA_RELEASE_DATE 
          << "" << Endl;
}

void TMVA::Tools::TMVAWelcomeMessage( MsgLogger& logger, EWelcomeMessage msgType )
{
   // various kinds of welcome messages
   // ASCII text generated by this site: http://www.network-science.de/ascii/

   switch (msgType) {

   case kStandardWelcomeMsg:
      logger << Color("white") << "TMVA -- Toolkit for Multivariate Analysis" << Color("reset") << Endl;
      logger << "Copyright (C) 2005-2006 CERN, LAPP & MPI-K Heidelberg and Victoria U." << Endl;
      logger << "Home page http://tmva.sourceforge.net" << Endl;
      logger << "All rights reserved, please read http://tmva.sf.net/license.txt" << Endl << Endl;
      break;

   case kIsometricWelcomeMsg:
      logger << "   ___           ___           ___           ___      " << Endl;
      logger << "  /\\  \\         /\\__\\         /\\__\\         /\\  \\     " << Endl;
      logger << "  \\:\\  \\       /::|  |       /:/  /        /::\\  \\    " << Endl;
      logger << "   \\:\\  \\     /:|:|  |      /:/  /        /:/\\:\\  \\   " << Endl;
      logger << "   /::\\  \\   /:/|:|__|__   /:/__/  ___   /::\\~\\:\\  \\  " << Endl;
      logger << "  /:/\\:\\__\\ /:/ |::::\\__\\  |:|  | /\\__\\ /:/\\:\\ \\:\\__\\ " << Endl;
      logger << " /:/  \\/__/ \\/__/~~/:/  /  |:|  |/:/  / \\/__\\:\\/:/  / " << Endl;
      logger << "/:/  /            /:/  /   |:|__/:/  /       \\::/  /  " << Endl;
      logger << "\\/__/            /:/  /     \\::::/__/        /:/  /   " << Endl;
      logger << "                /:/  /       ~~~~           /:/  /    " << Endl;
      logger << "                \\/__/                       \\/__/     " << Endl << Endl;
      break;

   case kBlockWelcomeMsg:
      logger << Endl;
      logger << "_|_|_|_|_|  _|      _|  _|      _|    _|_|    " << Endl;
      logger << "    _|      _|_|  _|_|  _|      _|  _|    _|  " << Endl;
      logger << "    _|      _|  _|  _|  _|      _|  _|_|_|_|  " << Endl;
      logger << "    _|      _|      _|    _|  _|    _|    _|  " << Endl;
      logger << "    _|      _|      _|      _|      _|    _|  " << Endl << Endl;
      break;

   case kLeanWelcomeMsg:
      logger << Endl;
      logger << "_/_/_/_/_/  _/      _/  _/      _/    _/_/   " << Endl;
      logger << "   _/      _/_/  _/_/  _/      _/  _/    _/  " << Endl;
      logger << "  _/      _/  _/  _/  _/      _/  _/_/_/_/   " << Endl;
      logger << " _/      _/      _/    _/  _/    _/    _/    " << Endl;
      logger << "_/      _/      _/      _/      _/    _/     " << Endl << Endl;
      break;

   case kLogoWelcomeMsg:
      logger << Endl;
      logger << "_/_/_/_/_/ _|      _|  _|      _|    _|_|   " << Endl;
      logger << "   _/      _|_|  _|_|  _|      _|  _|    _| " << Endl;
      logger << "  _/       _|  _|  _|  _|      _|  _|_|_|_| " << Endl;
      logger << " _/        _|      _|    _|  _|    _|    _| " << Endl;
      logger << "_/         _|      _|      _|      _|    _| " << Endl << Endl;
      break;

   case kSmall1WelcomeMsg:
      logger << " _____ __  ____   ___   " << Endl;
      logger << "|_   _|  \\/  \\ \\ / /_\\  " << Endl;
      logger << "  | | | |\\/| |\\ V / _ \\ " << Endl;
      logger << "  |_| |_|  |_| \\_/_/ \\_\\" << Endl << Endl;
      break;

   case kSmall2WelcomeMsg:
      logger << " _____ __  ____     ___     " << Endl;
      logger << "|_   _|  \\/  \\ \\   / / \\    " << Endl;
      logger << "  | | | |\\/| |\\ \\ / / _ \\   " << Endl;
      logger << "  | | | |  | | \\ V / ___ \\  " << Endl;
      logger << "  |_| |_|  |_|  \\_/_/   \\_\\ " << Endl << Endl;
      break;

   case kOriginalWelcomeMsgColor:
      logger << kINFO << "" << Color("red") 
             << "_______________________________________" << Color("reset") << Endl;
      logger << kINFO << "" << Color("blue")
             << Color("red_bgd") << Color("bwhite") << " // " << Color("reset")
             << Color("white") << Color("blue_bgd") 
             << "|\\  /|| \\  //  /\\\\\\\\\\\\\\\\\\\\\\\\ \\ \\ \\ " << Color("reset") << Endl;
      logger << kINFO << ""<< Color("blue")
             << Color("red_bgd") << Color("white") << "//  " << Color("reset")
             << Color("white") << Color("blue_bgd") 
             << "| \\/ ||  \\//  /--\\\\\\\\\\\\\\\\\\\\\\\\ \\ \\ \\" << Color("reset") << Endl;
      break;
      
   case kOriginalWelcomeMsgBW:
      logger << kINFO << "" 
             << "_______________________________________" << Endl;
      logger << kINFO << " // "
             << "|\\  /|| \\  //  /\\\\\\\\\\\\\\\\\\\\\\\\ \\ \\ \\ " << Endl;
      logger << kINFO << "//  " 
             << "| \\/ ||  \\//  /--\\\\\\\\\\\\\\\\\\\\\\\\ \\ \\ \\" << Endl;
      break;
      
   default:
      logger << kFATAL << "unknown message type: " << msgType << Endl;
   }
}

