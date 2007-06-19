// @(#)root/tmva $Id: VariableDecorrTransform.cxx,v 1.25 2007/06/04 22:52:00 stelzer Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableDecorrTransform                                               *
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

#include "Riostream.h"
#include "TVectorF.h"
#include "TVectorD.h"
#include "TMatrixD.h"
#include "TMatrixDBase.h"

#include "TMVA/VariableDecorrTransform.h"
#include "TMVA/Tools.h"

ClassImp(TMVA::VariableDecorrTransform)

//_______________________________________________________________________
TMVA::VariableDecorrTransform::VariableDecorrTransform( std::vector<VariableInfo>& varinfo )
   : VariableTransformBase( varinfo, Types::kDecorrelated )
{ 
   // constructor
   SetName("DecorrTransform");
   fDecorrMatrix[0] = fDecorrMatrix[1] = 0;
}

//_______________________________________________________________________
Bool_t TMVA::VariableDecorrTransform::PrepareTransformation( TTree* inputTree )
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

   GetSQRMats( inputTree );

   SetCreated( kTRUE );

   CalcNorm( inputTree );

   return kTRUE;
}

//_______________________________________________________________________
void TMVA::VariableDecorrTransform::ApplyTransformation( Types::ESBType type ) const
{
   // apply the decorrelation transformation
   if (!IsCreated()) return;

   TMatrixD* m = type==Types::kSignal ? fDecorrMatrix[Types::kSignal] : fDecorrMatrix[Types::kBackground];
   if (m == 0)
      fLogger << kFATAL << "Transformation matrix for " << (Types::kSignal?"signal":"background") << " is not defined" 
              << Endl;
   
   // transformation to decorrelate the variables
   const Int_t nvar = GetNVariables();
   TVectorD vec( nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) vec(ivar) = GetEventRaw().GetVal(ivar);

   // diagonalise variable vectors
   vec *= *m;
      
   for (Int_t ivar=0; ivar<nvar; ivar++) GetEvent().SetVal(ivar,vec(ivar));
   GetEvent().SetType       ( GetEventRaw().Type() );
   GetEvent().SetWeight     ( GetEventRaw().GetWeight() );
   GetEvent().SetBoostWeight( GetEventRaw().GetBoostWeight() );
}

//_______________________________________________________________________
void TMVA::VariableDecorrTransform::GetSQRMats( TTree* tr )
{
   // compute square-root matrices for signal and background
   for (UInt_t i=0; i<2; i++) {
      if (0 != fDecorrMatrix[i] ) { delete fDecorrMatrix[i]; fDecorrMatrix[i]=0; }

      Int_t nvar = GetNVariables();
      TMatrixDSym* covMat = new TMatrixDSym( nvar );

      GetCovarianceMatrix( tr, (i==0),  covMat );

      fDecorrMatrix[i] = Tools::GetSQRootMatrix( covMat );
      if (fDecorrMatrix[i] == 0) 
         fLogger << kFATAL << "<GetSQRMats> Zero pointer returned for SQR matrix" << Endl;
   }
}

//_______________________________________________________________________
void TMVA::VariableDecorrTransform::GetCovarianceMatrix( TTree* tr, Bool_t isSignal, TMatrixDBase* mat )
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
   Int_t ic = 0;
   for (Int_t i=0; i<tr->GetEntries(); i++) {

      // fill the event
      ReadEvent(tr, i, Types::kSignal);

      if (GetEventRaw().IsSignal() == isSignal) {
         ic++; // count used events
         for (ivar=0; ivar<nvar; ivar++) {

            Double_t xi = ( (IsNormalised()) ? Tools::NormVariable( GetEventRaw().GetVal(ivar), xmin(ivar), xmax(ivar) )
                            : GetEventRaw().GetVal(ivar) );
            vec(ivar) += xi;
            mat2(ivar, ivar) += (xi*xi);

            for (jvar=ivar+1; jvar<nvar; jvar++) {
               Double_t xj =  ( (IsNormalised()) ? Tools::NormVariable( GetEventRaw().GetVal(jvar), xmin(ivar), xmax(ivar) )
                                : GetEventRaw().GetVal(jvar) );
               mat2(ivar, jvar) += (xi*xj);
               mat2(jvar, ivar) = mat2(ivar, jvar); // symmetric matrix
            }
         }
      }
   }

   // variance-covariance
   Double_t n = (Double_t)ic;
   for (ivar=0; ivar<nvar; ivar++) {
      for (jvar=0; jvar<nvar; jvar++) {
         (*mat)(ivar, jvar) = mat2(ivar, jvar)/n - vec(ivar)*vec(jvar)/(n*n);
      }
   }

   tr->ResetBranchAddresses();
}

//_______________________________________________________________________
void TMVA::VariableDecorrTransform::WriteTransformationToStream( std::ostream& o ) const
{
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
void TMVA::VariableDecorrTransform::ReadTransformationFromStream( std::istream& istr )
{
   // Read the decorellation matrix from an input stream

   char buf[512];
   istr.getline(buf,512);
   TString strvar, dummy;
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
   SetCreated();
}

//_______________________________________________________________________
void TMVA::VariableDecorrTransform::PrintTransformation( ostream& ) 
{
   // prints the transformation matrix
   fLogger << kINFO << "Transformation matrix signal:" << Endl;
   fDecorrMatrix[0]->Print();
   fLogger << kINFO << "Transformation matrix background:" << Endl;
   fDecorrMatrix[1]->Print();
}

//_______________________________________________________________________
void TMVA::VariableDecorrTransform::MakeFunction( std::ostream& fout, const TString& fcncName, Int_t part ) 
{
   // creates a decorrelation function
   TMatrixD* mat = fDecorrMatrix[0];
   if (part==1) {
      fout << std::endl;
      fout << "   double fSigTF["<<mat->GetNrows()<<"]["<<mat->GetNcols()<<"];" << std::endl;
      fout << "   double fBgdTF["<<mat->GetNrows()<<"]["<<mat->GetNcols()<<"];" << std::endl;
      fout << std::endl;
   }

   if (part==2) {
      fout << "inline void " << fcncName << "::InitTransform()" << std::endl;
      fout << "{" << std::endl;
      TMatrixD* mat = fDecorrMatrix[0];
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
