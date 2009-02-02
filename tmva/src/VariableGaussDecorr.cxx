// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableGaussDecorr                                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Decorrelation of input variables                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <list>
#include "Riostream.h"
#include "TVectorF.h"
#include "TVectorD.h"
#include "TMatrixD.h"
#include "TMatrixDBase.h"
#include "TMath.h"
#include "TCanvas.h"

#include "TMVA/VariableGaussDecorr.h"
#include "TMVA/Tools.h"

ClassImp(TMVA::VariableGaussDecorr)

   //_______________________________________________________________________
   TMVA::VariableGaussDecorr::VariableGaussDecorr( std::vector<VariableInfo>& varinfo )
      : VariableTransformBase( varinfo, Types::kGaussDecorr )
{ 
   // constructor
   SetName("GaussDecorr");
   gROOT->cd();
   for (UInt_t ivar=0; ivar < GetNVariables(); ivar++){
      vector< TH1F* >  tmp;
      tmp.push_back(0); // signal
      tmp.push_back(0); // backgr
      fCumulativeDist.push_back(tmp);
   }
   fDecorrMatrix[0] = fDecorrMatrix[1] = 0;
   fFlatNotGaussD   = kFALSE;

   // can only be applied one after the other when they are created. But in order to
   // determine the decorrelation transformation, I need to first apply the Gauss transformation
   fApplyGaussTransform  = kFALSE;  
   fApplyDecorrTransform = kFALSE; 
}

//_______________________________________________________________________
TMVA::VariableGaussDecorr::~VariableGaussDecorr( void )
{
   // destructor

   for (UInt_t ivar=0; ivar<(UInt_t)GetNVariables(); ivar++) {
      for (UInt_t i=0; i<2; i++) {
         if (0 != fCumulativeDist[ivar][i] ) { 
            delete fCumulativeDist[ivar][i]; 
         }
      }
   }
}

//_______________________________________________________________________
Bool_t TMVA::VariableGaussDecorr::PrepareTransformation( TTree* inputTree )
{
   // calculate the decorrelation matrix and the normalization
   if (!IsEnabled() || IsCreated()) return kTRUE;

   if (inputTree == 0) return kFALSE;

   if (GetNVariables() > 200) { 
      fLogger << kINFO << "----------------------------------------------------------------------------" 
              << Endl;
      fLogger << kINFO 
              << ": More than 200 variables, will not calculate decorrelation matrix "
              << inputTree->GetName() << "!" << Endl;
      fLogger << kINFO << "----------------------------------------------------------------------------" 
              << Endl;
      return kFALSE;
   }   

   GetCumulativeDist( inputTree );

   fApplyGaussTransform = kTRUE;

   if (!fFlatNotGaussD) {
      GetSQRMats( inputTree );

      fApplyDecorrTransform = kTRUE;
   }

   SetCreated( kTRUE );

   CalcNorm( inputTree );

   return kTRUE;
}

//_______________________________________________________________________
std::vector<TString>* TMVA::VariableGaussDecorr::GetTransformationStrings( Types::ESBType type ) const
{
   // creates string with variable transformations applied
   TMatrixD* m = type==Types::kSignal ? fDecorrMatrix[Types::kSignal] : fDecorrMatrix[Types::kBackground];

   const Int_t nvar = GetNVariables();
   std::vector<TString>* strVec = new std::vector<TString>;

   // fill vector
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      TString str( "" );
      for (Int_t jvar=0; jvar<nvar; jvar++) {
         if (jvar > 0) str += ((*m)(ivar,jvar) > 0) ? " + " : " - ";
         str += Form( "%10.5g*%s", 
                      TMath::Abs((*m)(ivar,jvar)), 
                      (TString("[") + Variable(jvar).GetExpression() + "]").Data() );
      }
      strVec->push_back( str );
   }      

   return strVec;
}

//_______________________________________________________________________
void TMVA::VariableGaussDecorr::ApplyTransformation( Types::ESBType type ) const
{
   // apply the decorrelation transformation
   if (! ( IsCreated() || fApplyDecorrTransform || fApplyGaussTransform) ) return;

   // get the variable vector of the current event
   const UInt_t nvar = GetNVariables();
   TVectorD vec( nvar );
   for (UInt_t ivar=0; ivar<nvar; ivar++) vec(ivar) = GetEventRaw().GetVal(ivar);

   if (fApplyGaussTransform) {
      // decide if you want to transform into a gaussian according to the signal or backgr
      // transformation 
      Double_t cumulant, supmin,total;
      Int_t    thebin;
      Double_t xvar, x0, x1, y0, y1;
      UInt_t sb = (type==Types::kSignal) ? 0 : 1;
      for (UInt_t ivar=0; ivar<nvar; ivar++) {
         if (0 != fCumulativeDist[ivar][sb]) { 
            // first make it flat
            // my old code..
            //             Int_t    thebin   = (fCumulativeDist[ivar][sb])->FindBin(vec(ivar));
            //             Double_t cumulant = (fCumulativeDist[ivar][sb])->GetBinContent(thebin);
            // Diegos interpoloation instead..
            xvar = vec(ivar);
            total = fCumulativeDist[ivar][sb]->GetNbinsX()*fElementsPerBin;
            supmin = 0.5/total;
            thebin = (fCumulativeDist[ivar][sb])->FindBin(vec(ivar)); // xvar < xmin ==> thebin = 0 . xvar > xmax ==> thebin = NbinsX() + 1
            //cumulant     = (fCumulativeDist[ivar][sb])->GetBinContent(thebin);

            x0 = fCumulativeDist[ivar][sb]->GetBinLowEdge(TMath::Max(thebin,1)); 
            x1 = fCumulativeDist[ivar][sb]->GetBinLowEdge(TMath::Min(thebin+1,fCumulativeDist[ivar][sb]->GetNbinsX()+1)); 

            // if (thebin <= 1) { y0 = 1./(fElementsPerBin*fCumulative[ivar][sb]->GetNbinsX()); }   TO DO
            y0 = fCumulativeDist[ivar][sb]->GetBinContent(TMath::Max(thebin-1,0)); // Y0 = F(x0); Y0 >= 0
            y1 = fCumulativeDist[ivar][sb]->GetBinContent(TMath::Min(thebin,fCumulativeDist[ivar][sb]->GetNbinsX()+1));  // Y1 = F(x1);  Y1 <= 1

            if (thebin == 0) {
               y0 = supmin;
               y1 = supmin;
            }
            if (thebin == 1) {
               y0 = supmin;
            }
            if (thebin > fCumulativeDist[ivar][sb]->GetNbinsX()) { 
               y0 = 1.-supmin;
               y1 = 1.-supmin; 
            }
            if (thebin == fCumulativeDist[ivar][sb]->GetNbinsX()) { 
               y1 = 1.-supmin; 
            }

            //if (thebin == fCumulativeDist[ivar][sb]->GetNbinsX()) { y1 = 1- 1./(fElementsPerBin*fCumulative[ivar][sb]->GetNbinsX());} TO DO

            //Special cases:
            //1 : x < xmin of the training sample
            //if (xvar < fCumulative[ivar][sb]->GetBinLowEdge(1)){

            //if (xvar==x0){ cumulat = y0; }
            if (x0 == x1) { cumulant = y1;}
            else { cumulant = y0 + (y1-y0)*(xvar-x0)/(x1-x0);} // Interpolation
            // Doesn't exist something like 'elif' in C++ ??
            if (xvar <= fCumulativeDist[ivar][sb]->GetBinLowEdge(1)){
               cumulant = supmin;
            }
            if (xvar >= fCumulativeDist[ivar][sb]->GetBinLowEdge(fCumulativeDist[ivar][sb]->GetNbinsX()+1)){
               cumulant = 1-supmin;
            }


            // now transfor to a gaussian
            // actually, the "sqrt(2) is not really necessary, who cares if it is totally normalised
            //            vec(ivar) =  TMath::Sqrt(2.)*TMath::ErfInverse(2*sum - 1);
            if (fFlatNotGaussD) vec(ivar) = cumulant; 
            else {
               // sanity correction for out-of-range values
               Double_t maxErfInvArgRange = 0.99999999;
               Double_t arg = 2.0*cumulant - 1.0;
               arg = TMath::Min(+maxErfInvArgRange,arg);
               arg = TMath::Max(-maxErfInvArgRange,arg);

               vec(ivar) = 1.414213562*TMath::ErfInverse(arg);
            }
         }         
      }
   }

   if (fApplyDecorrTransform ) {

      // transformation to decorrelate the variables      
      TMatrixD* m = type==Types::kSignal ? fDecorrMatrix[Types::kSignal] : fDecorrMatrix[Types::kBackground];
      if (m == 0) {
         fLogger << kFATAL << "Transformation matrix for " << (Types::kSignal?"signal":"background") 
                 << " is not defined" << Endl;
      }      

      // diagonalise variable vectors
      vec *= *m;
   }  

   for (UInt_t ivar=0; ivar<nvar; ivar++) GetEvent().SetVal(ivar,vec(ivar));
   GetEvent().SetType       ( GetEventRaw().Type() );
   GetEvent().SetWeight     ( GetEventRaw().GetWeight() );
   GetEvent().SetBoostWeight( GetEventRaw().GetBoostWeight() );
}

//_______________________________________________________________________
void TMVA::VariableGaussDecorr::GetCumulativeDist( TTree* tr )
{
   // fill the cumulative distributions
   const UInt_t nvar = GetNVariables();
   ResetBranchAddresses( tr );

   fElementsPerBin= Int_t( tr->GetEntries()/2000.);
   //   fElementsPerBin= 10;
   UInt_t Nbins[2];

   std::list< Float_t >  listsForBinning[nvar][2];
   std::vector< Float_t >   vsForBinning[nvar][2];
   Nbins[0] = 0;  // Nbins[0] = number of bins for signal distributions. It depends on the number of entries, thus it's the same for all the input variables
   Nbins[1] = 0;  // Nbins[1] = "", for bkg

   //   UInt_t nHighBins=100000;

   // if normalisation required, determine min/max
   TVectorF xmin(nvar), xmax(nvar);
   for (Int_t iev=0; iev<tr->GetEntries(); iev++) {
      // fill the event
      ReadEvent(tr, iev, Types::kSignal);

      for (UInt_t ivar=0; ivar<nvar; ivar++) {
         if (iev == 0) {
            xmin(ivar) = GetEventRaw().GetVal(ivar);
            xmax(ivar) = GetEventRaw().GetVal(ivar);
         }
         else {
            xmin(ivar) = TMath::Min( xmin(ivar), GetEventRaw().GetVal(ivar) );
            xmax(ivar) = TMath::Max( xmax(ivar), GetEventRaw().GetVal(ivar) );
         }
      }
   }

   // create histogram for the cumulative distribution.
   std::vector< std::vector< TH1F* > >  tmpCumulator; //! o.k.. what would be the proper name


   for (UInt_t i=0; i<2; i++) {
      //what a funny way to make a loop over "signal (=0) and backgr(=1).. but it
      //seems to be general habit, and the stuff below is then needed to 
      //make sure that "signal" and "backgr" are recognized according to their
      //predefined types...
      Types::ESBType  type;
      type = (i==0) ? Types::kSignal : Types::kBackground;



      // perform event loop

      for (Int_t ievt=0; ievt < tr->GetEntries(); ievt++) {

         // fill the event  (no, it does NOT mean that you want to read a signal
         // event. A bit irritating again, but the "Types::kSignal" only means, that
         // "if it were set to "Types::kTrueType", then this variable would be
         // overwritten in the function call with the type "kSignal" or "kBackgr" 
         // depending on what event type event ievt is
         ReadEvent(tr, ievt, Types::kSignal);

         if (GetEventRaw().IsSignal() == (type==Types::kSignal) ) {
            for (UInt_t ivar=0; ivar<nvar; ivar++) {
               listsForBinning[ivar][i].push_back(GetEventRaw().GetVal(ivar));  
            }  
         }
      }
   }
   // Sorting the lists, getting Nbins ...

   for (UInt_t sb=0; sb< 2; sb++){
      for (UInt_t ivar=0; ivar<nvar; ivar++) {
         listsForBinning[ivar][sb].sort();  

         std::list< Float_t >::iterator it;
         UInt_t ievt=0;
         //UInt_t nbins = 0;
         for (it=listsForBinning[ivar][sb].begin(); it != listsForBinning[ivar][sb].end(); it++){
            if (!(ievt%fElementsPerBin)) {
               vsForBinning[ivar][sb].push_back(*it);
            }
            ievt++;
         }
         vsForBinning[ivar][sb].push_back(listsForBinning[ivar][sb].back()); 
      }
      Nbins[sb] = vsForBinning[0][sb].size();
   }

   // if one exist already.. delete it 
   char SB[2]={'S','B'};

   for (UInt_t ivar=0; ivar<nvar; ivar++) {

      Float_t* binnings_S = new Float_t[Nbins[0]];
      Float_t* binnings_B = new Float_t[Nbins[1]];

      for (UInt_t k =0 ; k < Nbins[0]; k++){
         binnings_S[k] = vsForBinning[ivar][0][k];
      }
      for (UInt_t k =0 ; k < Nbins[1]; k++){
         binnings_B[k] = vsForBinning[ivar][1][k];
      }

      vector< TH1F* > tmp;

      tmp.push_back( new TH1F(Form("tmpCumulator_Var%d_S",ivar),
                              Form("tmpCumulator_Var%d_S",ivar),
                              Nbins[0] -1,
                              binnings_S));; // signal
      tmp.push_back( new TH1F(Form("tmpCumulator_Var%d_B",ivar),
                              Form("tmpCumulator_Var%d_B",ivar),
                              Nbins[1] -1,
                              binnings_B)); // backgr
      tmpCumulator.push_back(tmp);

      for (UInt_t i=0; i<2; i++) {
         if (0 != fCumulativeDist[ivar][i] ) { 
            delete fCumulativeDist[ivar][i]; 
         }
      }
      fCumulativeDist[ivar][0] = new TH1F(Form("Cumulative_Var%d_%c",ivar,SB[0]),
                                          Form("Cumulative_Var%d_%c",ivar,SB[0]),
                                          Nbins[0]-1,
                                          binnings_S);
      fCumulativeDist[ivar][1] = new TH1F(Form("Cumulative_Var%d_%c",ivar,SB[1]),
                                          Form("Cumulative_Var%d_%c",ivar,SB[1]),
                                          Nbins[1]-1,
                                          binnings_B);

      delete [] binnings_S;
      delete [] binnings_B;
   }

   for (UInt_t i=0; i<2; i++) {
      //what a funny way to make a loop over "signal (=0) and backgr(=1).. but it
      //seems to be general habit, and the stuff below is then needed to 
      //make sure that "signal" and "backgr" are recognized according to their
      //predefined types...
      Types::ESBType  type;
      type = (i==0) ? Types::kSignal : Types::kBackground;

      // perform event loop
      Int_t ic = 0;
      for (Int_t ievt=0; ievt<tr->GetEntries(); ievt++) {

         // fill the event  (no, it does NOT mean that you want to read a signal
         // event. A bit irritating again, but the "Types::kSignal" only means, that
         // "if it were set to "Types::kTrueType", then this variable would be
         // overwritten in the function call with the type "kSignal" or "kBackgr" 
         // depending on what event type event ievt is
         ReadEvent(tr, ievt, Types::kSignal);

         if (GetEventRaw().IsSignal() == (type==Types::kSignal) ) {
            ic++; // count used events
            for (UInt_t ivar=0; ivar<nvar; ivar++) {
               tmpCumulator[ivar][i]->Fill(GetEventRaw().GetVal(ivar));;               
            }
         }
      }
   }

   // now sum up in order to get the real cumulative distribution   
   Double_t  sum    = 0, total, cte, supmin;
   for (UInt_t sb=0; sb<2; sb++) {
      for (UInt_t ivar=0; ivar<nvar; ivar++) {
         sum = 0;
         total = (tmpCumulator[ivar][sb])->GetSumOfWeights();
         supmin = 0.5/total;//TODO: Check the exact value of supmin !!!
         cte = (1.-2*supmin)/(fCumulativeDist[ivar][sb]->GetNbinsX());
         for (Int_t ibin=1; ibin <=fCumulativeDist[ivar][sb]->GetNbinsX() ; ibin++){
            //         for (UInt_t ibin=1; ibin <= nHighBins; ibin++){
            sum += (tmpCumulator[ivar][sb])->GetBinContent(ibin);
            Double_t binCenter = (tmpCumulator[ivar][sb])->GetXaxis()->GetBinCenter(ibin);

            Int_t currentBin = (fCumulativeDist[ivar][sb])->FindBin(binCenter);
            //            (fCumulativeDist[ivar][sb])->SetBinContent(currentBin,sum/total);
            (fCumulativeDist[ivar][sb])->SetBinContent(currentBin,supmin + currentBin*cte);
         }
         fCumulativeDist[ivar][sb]->SetBinContent(fCumulativeDist[ivar][sb]->GetNbinsX()+1, 1-supmin);
         fCumulativeDist[ivar][sb]->SetBinContent(0,supmin);
         delete tmpCumulator[ivar][sb];
      }
   }
}

//_______________________________________________________________________
void TMVA::VariableGaussDecorr::GetSQRMats( TTree* tr )
{
   // compute square-root matrices for signal and background
   for (UInt_t i=0; i<2; i++) {
      if (0 != fDecorrMatrix[i] ) { delete fDecorrMatrix[i]; fDecorrMatrix[i]=0; }

      Int_t nvar = GetNVariables();
      TMatrixDSym* covMat = new TMatrixDSym( nvar );

      GetCovarianceMatrix( tr, (i==Types::kSignal),  covMat );

      fDecorrMatrix[i] = gTools().GetSQRootMatrix( covMat );
      if (fDecorrMatrix[i] == 0) 
         fLogger << kFATAL << "<GetSQRMats> Zero pointer returned for SQR matrix" << Endl;
   }
}

//_______________________________________________________________________
void TMVA::VariableGaussDecorr::GetCovarianceMatrix( TTree* tr, Bool_t isSignal, TMatrixDBase* mat )
{
   // compute covariance matrix

   UInt_t nvar = GetNVariables(), ivar = 0, jvar = 0;

   // init matrices
   TVectorD vec(nvar);
   TMatrixD mat2(nvar, nvar);      
   for (ivar=0; ivar<nvar; ivar++) {
      vec(ivar) = 0;
      for (jvar=0; jvar<nvar; jvar++) {
         mat2(ivar, jvar) = 0;
      }
   }

   ResetBranchAddresses( tr );

   // if normalisation required, determine min/max
   TVectorF xmin(nvar), xmax(nvar);
   if (IsNormalised()) {
      for (Int_t i=0; i<tr->GetEntries(); i++) {
         // fill the event
         ReadEvent(tr, i, Types::kSignal);

         for (ivar=0; ivar<nvar; ivar++) {
            if (i == 0) {
               xmin(ivar) = GetEventRaw().GetVal(ivar);
               xmax(ivar) = GetEventRaw().GetVal(ivar);
            }
            else {
               xmin(ivar) = TMath::Min( xmin(ivar), GetEventRaw().GetVal(ivar) );
               xmax(ivar) = TMath::Max( xmax(ivar), GetEventRaw().GetVal(ivar) );
            }
         }
      }
   }

   // perform event loop
   Double_t ic = 0;
   for (Int_t i=0; i<tr->GetEntries(); i++) {

      // fill the event
      ReadEvent(tr, i, Types::kSignal);

      if (GetEventRaw().IsSignal() == isSignal) {
         Double_t weight = GetEventRaw().GetWeight();
         ic += weight; // count used events
         for (ivar=0; ivar<nvar; ivar++) {

            Double_t xi = ( (IsNormalised()) ? gTools().NormVariable( GetEventRaw().GetVal(ivar), xmin(ivar), xmax(ivar) )
                            : GetEventRaw().GetVal(ivar) );
            vec(ivar) += xi*weight;
            mat2(ivar, ivar) += (xi*xi*weight);

            for (jvar=ivar+1; jvar<nvar; jvar++) {
               Double_t xj =  ( (IsNormalised()) ? gTools().NormVariable( GetEventRaw().GetVal(jvar), xmin(ivar), xmax(ivar) )
                                : GetEventRaw().GetVal(jvar) );
               mat2(ivar, jvar) += (xi*xj*weight);
               mat2(jvar, ivar) = mat2(ivar, jvar); // symmetric matrix
            }
         }
      }
   }

   // variance-covariance
   for (ivar=0; ivar<nvar; ivar++) {
      for (jvar=0; jvar<nvar; jvar++) {
         (*mat)(ivar, jvar) = mat2(ivar, jvar)/ic - vec(ivar)*vec(jvar)/(ic*ic);
      }
   }

   tr->ResetBranchAddresses();
}

//_______________________________________________________________________
void TMVA::VariableGaussDecorr::WriteTransformationToStream( std::ostream& o ) const
{
   // write the cumulative distributions to the weight file
   for (UInt_t ivar=0; ivar< GetNVariables(); ivar++){
      if ( fCumulativeDist[ivar][0]==0 || fCumulativeDist[ivar][1]==0 )
         fLogger << kFATAL << "Cumulative histograms for variable " << ivar << " don't exist, can't write it to weight file" << Endl;

      for (UInt_t i=0; i<2; i++){
         TH1 * histToWrite = fCumulativeDist[ivar][i];
         const Int_t nBins = histToWrite->GetNbinsX();
         o << "CumulativeHistogram" 
           << "   " << i  // signal =0, backgr = 1  
           << "   " << ivar 
           << "   " << histToWrite->GetName() 
           << "   " << nBins                                 // nbins
           << "   " << setprecision(12) << fElementsPerBin 
           << endl;

         // write the smoothed hist
         o << "BinBoundaries" << endl;
         o << std::setprecision(8);
         for (Int_t ibin=0; ibin<nBins; ibin++) {
            o << std::setw(15) << std::left << histToWrite->GetBinLowEdge(ibin+1) << " ";
            if ((ibin+1)%5==0) o << endl;
         }
         o << std::setw(15) << std::left << histToWrite->GetXaxis()->GetXmax() << " ";

         // write the smoothed hist
         o << "BinContent" << endl;
         o << std::setprecision(8);
         for (Int_t ibin=0; ibin<nBins; ibin++) {
            o << std::setw(15) << std::left << histToWrite->GetBinContent(ibin+1) << " ";
            if ((ibin+1)%5==0) o << endl;
         }
      }
   }
   o << "#"<<endl;
   o << "TransformToFlatInsetadOfGaussD= " << fFlatNotGaussD <<endl;
   o << "#"<<endl;

   // write the decorrelation matrix to the stream
   for (Int_t matType=0; matType<2; matType++) {
      o << "# correlation matrix " << endl;
      TMatrixD* mat = fDecorrMatrix[matType];
      o << (matType==0?"signal":"background") << " " << mat->GetNrows() << " x " << mat->GetNcols() << endl;
      for (Int_t row = 0; row<mat->GetNrows(); row++) {
         for (Int_t col = 0; col<mat->GetNcols(); col++) {
            o << setprecision(12) << setw(20) << (*mat)[row][col] << " ";
         }
         o << endl;
      }
   }
   o << "##" << endl;
}

//_______________________________________________________________________
void TMVA::VariableGaussDecorr::ReadTransformationFromStream( std::istream& istr )
{
   // Read the decorellation matrix from an input stream
   Bool_t addDirStatus = TH1::AddDirectoryStatus();
   TH1::AddDirectory(0); // this avoids the binding of the hists in TMVA::PDF to the current ROOT file
   char buf[512];
   istr.getline(buf,512);

   TString strvar, dummy;

   Int_t type;
   Int_t ivar;

   Int_t nrows(0), ncols(0);

   while (!(buf[0]=='#'&& buf[1]=='#')) { // if line starts with ## return
      char* p = buf;
      while (*p==' ' || *p=='\t') p++; // 'remove' leading whitespace
      if (*p=='#' || *p=='\0') {
         istr.getline(buf,512);
         continue; // if comment or empty line, read the next line
      }
      std::stringstream sstr(buf);
      sstr >> strvar;

      if (strvar=="CumulativeHistogram") {
         TString devnullS;
         Int_t   nbins;
         TString hname="";

         sstr  >> type >> ivar >> hname >> nbins >> fElementsPerBin; 

         Float_t *Binnings = new Float_t[nbins+1];
         Float_t val;
         istr >> devnullS; // read the line "BinBoundaries" ..
         for (Int_t ibin=0; ibin<nbins+1; ibin++) {
            istr >> val;
            Binnings[ibin]=val;
         }

         TH1F * histToRead = fCumulativeDist[ivar][type];
         if ( histToRead !=0 ) delete histToRead;
         // recreate the cumulative histogram to be filled with the values read
         histToRead = new TH1F( hname, hname, nbins, Binnings );
         delete [] Binnings;
         histToRead->SetDirectory(0);
         fCumulativeDist[ivar][type]=histToRead;

         istr >> devnullS; // read the line "BinContent" .. 
         for (Int_t ibin=0; ibin<nbins; ibin++) {
            istr >> val;
            histToRead->SetBinContent(ibin+1,val);
         }
      } 
      if (strvar=="TransformToFlatInsetadOfGaussD=") { sstr >> fFlatNotGaussD; }
      //now read the decorrelation matrix
      if (strvar=="signal" || strvar=="background") {
         sstr >> nrows >> dummy >> ncols;
         Int_t matType = (strvar=="signal"?0:1);
         if (fDecorrMatrix[matType] != 0) delete fDecorrMatrix[matType];
         TMatrixD* mat = fDecorrMatrix[matType] = new TMatrixD(nrows,ncols);
         // now read all matrix parameters
         for (Int_t row = 0; row<mat->GetNrows(); row++) {
            for (Int_t col = 0; col<mat->GetNcols(); col++) {
               istr >> (*mat)[row][col];
            }
         }
      } // done reading a matrix

      istr.getline(buf,512); // reading the next line   
   }
   TH1::AddDirectory(addDirStatus);

   fApplyGaussTransform = kTRUE;  
   if (fFlatNotGaussD) fApplyDecorrTransform = kFALSE; 
   else fApplyDecorrTransform = kTRUE; 

   SetCreated();

}

//_______________________________________________________________________
void TMVA::VariableGaussDecorr::PrintTransformation( ostream& ) 
{
   // prints the transformation matrix
   fLogger << kINFO << "Transformation matrix signal:" << Endl;
   fDecorrMatrix[0]->Print();
   fLogger << kINFO << "Transformation matrix background:" << Endl;
   fDecorrMatrix[1]->Print();
}

//_______________________________________________________________________
void TMVA::VariableGaussDecorr::MakeFunction( std::ostream& fout, const TString& fcncName, Int_t part ) 
{
   // creates a decorrelation function

   fout << " std::cout << \"ERROR, the Gauss Transfomration is not implemented yet\" << std::endl; exit(1)"<<std::endl; 
   TMatrixD* mat = fDecorrMatrix[0];
   if (part == 1) {
      fout << std::endl;
      fout << "   double fSigTF["<<mat->GetNrows()<<"]["<<mat->GetNcols()<<"];" << std::endl;
      fout << "   double fBgdTF["<<mat->GetNrows()<<"]["<<mat->GetNcols()<<"];" << std::endl;
      fout << std::endl;
   }

   if (part == 2) {
      fout << "inline void " << fcncName << "::InitTransform()" << std::endl;
      fout << "{" << std::endl;
      for (int i=0; i<mat->GetNrows(); i++) {
         for (int j=0; j<mat->GetNcols(); j++) {
            fout << "   fSigTF["<<i<<"]["<<j<<"] = " << std::setprecision(12) << (*mat)[i][j] << ";" << std::endl;
            fout << "   fBgdTF["<<i<<"]["<<j<<"] = " << std::setprecision(12) << (*fDecorrMatrix[1])[i][j] << ";" << std::endl;
         }
      }
      fout << "}" << std::endl;
      fout << std::endl;
      fout << "inline void " << fcncName << "::Transform( std::vector<double>& iv, int sigOrBgd ) const" << std::endl;
      fout << "{" << std::endl;
      fout << "   std::vector<double> tv;" << std::endl;
      fout << "   for (int i=0; i<"<<mat->GetNrows()<<";i++) {" << std::endl;
      fout << "      double v = 0;" << std::endl;
      fout << "      for (int j=0; j<"<<mat->GetNcols()<<"; j++)" << std::endl;
      fout << "         v += iv[j] * (sigOrBgd==0 ? fSigTF[i][j] : fBgdTF[i][j]);" << std::endl;
      fout << "      tv.push_back(v);" << std::endl;
      fout << "   }" << std::endl;
      fout << "   for (int i=0; i<"<<mat->GetNrows()<<";i++) iv[i] = tv[i];" << std::endl;
      fout << "}" << std::endl;
   }
}
