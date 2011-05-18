// @(#)root/tmva $Id$
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
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch> - CERN, Switzerland           *
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

#include <iostream>
#include <iomanip>

#include "TVectorF.h"
#include "TVectorD.h"
#include "TMatrixD.h"
#include "TMatrixDBase.h"

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMVA_VariableDecorrTransform
#include "TMVA/VariableDecorrTransform.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_DataSet
#include "TMVA/DataSet.h"
#endif

ClassImp(TMVA::VariableDecorrTransform)

//_______________________________________________________________________
TMVA::VariableDecorrTransform::VariableDecorrTransform( DataSetInfo& dsi )
  : VariableTransformBase( dsi, Types::kDecorrelated, "Deco" )
{ 
   // constructor
}

//_______________________________________________________________________
TMVA::VariableDecorrTransform::~VariableDecorrTransform()
{
   // destructor
   for (std::vector<TMatrixD*>::iterator it = fDecorrMatrices.begin(); it != fDecorrMatrices.end(); it++) {
      if ((*it) != 0) delete (*it);
   }
}

//_______________________________________________________________________
void TMVA::VariableDecorrTransform::Initialize()
{
   // initialisation
}

//_______________________________________________________________________
Bool_t TMVA::VariableDecorrTransform::PrepareTransformation( const std::vector<Event*>& events )
{
   // calculate the decorrelation matrix and the normalization
   Initialize();

   if (!IsEnabled() || IsCreated()) return kTRUE;

   Log() << kINFO << "Preparing the Decorrelation transformation..." << Endl;

   SetNVariables(events[0]->GetNVariables());

   if (GetNVariables() > 200) { 
      Log() << kINFO << "----------------------------------------------------------------------------" 
            << Endl;
      Log() << kINFO 
            << ": More than 200 variables, will not calculate decorrelation matrix "
            << "!" << Endl;
      Log() << kINFO << "----------------------------------------------------------------------------" 
            << Endl;
      return kFALSE;
   }   

   CalcSQRMats( events, GetNClasses() );

   SetCreated( kTRUE );

   return kTRUE;
}

//_______________________________________________________________________
std::vector<TString>* TMVA::VariableDecorrTransform::GetTransformationStrings( Int_t cls ) const
{
   // creates string with variable transformations applied

   Int_t whichMatrix = cls;
   // if cls (the class chosen by the user) not existing, assume that user wants to 
   // have the matrix for all classes together. 
   
   if (cls < 0 || cls > GetNClasses()) whichMatrix = GetNClasses();
   
   TMatrixD* m = fDecorrMatrices.at(whichMatrix);
   if (m == 0) {
      if (whichMatrix == GetNClasses() )
         Log() << kFATAL << "Transformation matrix all classes is not defined" 
               << Endl;
      else
         Log() << kFATAL << "Transformation matrix for class " << whichMatrix << " is not defined" 
               << Endl;
   }

   const Int_t nvar = GetNVariables();
   std::vector<TString>* strVec = new std::vector<TString>;

   // fill vector
   for (Int_t ivar=0; ivar<nvar; ivar++) {
      TString str( "" );
      for (Int_t jvar=0; jvar<nvar; jvar++) {
         str += ((*m)(ivar,jvar) > 0) ? " + " : " - ";
         str += Form( "%10.5g*[%s]", TMath::Abs((*m)(ivar,jvar)), Variables()[jvar].GetLabel().Data() );
      }
      strVec->push_back( str );
   }      

   return strVec;
}

//_______________________________________________________________________
const TMVA::Event* TMVA::VariableDecorrTransform::Transform( const TMVA::Event* const ev, Int_t cls ) const
{
   // apply the decorrelation transformation
   if (!IsCreated())
      Log() << kFATAL << "Transformation matrix not yet created" 
            << Endl;

   Int_t whichMatrix = cls;
   // if cls (the class chosen by the user) not existing, assume that he wants to have the matrix for all classes together. 
   // EVT this is a workaround to address the reader problem with transforma and EvaluateMVA(std::vector<float/double> ,...) 
   if (cls < 0 || cls >= (int) fDecorrMatrices.size()) whichMatrix = fDecorrMatrices.size()-1;
   //EVT workaround end
   //if (cls < 0 || cls > GetNClasses()) {
   //   whichMatrix = GetNClasses();
   //   if (GetNClasses() == 1 ) whichMatrix = (fDecorrMatrices.size()==1?0:2);
   //}

   TMatrixD* m = fDecorrMatrices.at(whichMatrix);
   if (m == 0) {
      if (whichMatrix == GetNClasses() )
         Log() << kFATAL << "Transformation matrix all classes is not defined" 
               << Endl;
      else
         Log() << kFATAL << "Transformation matrix for class " << whichMatrix << " is not defined" 
               << Endl;
   }

   // transformation to decorrelate the variables
   const Int_t nvar = GetNVariables();
   TVectorD vec( nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) vec(ivar) = ev->GetValue(ivar);
   // diagonalise variable vectors
   vec *= *m;

   if (fTransformedEvent==0 || fTransformedEvent->GetNVariables()!=ev->GetNVariables()) {
      if (fTransformedEvent!=0) { delete fTransformedEvent; fTransformedEvent = 0; }
      fTransformedEvent = new Event();
   }

   for (UInt_t itgt=0; itgt<ev->GetNTargets(); itgt++) fTransformedEvent->SetTarget( itgt, ev->GetTarget(itgt) );
   for (Int_t  ivar=0; ivar<nvar;              ivar++) fTransformedEvent->SetVal   ( ivar,vec(ivar) );

   fTransformedEvent->SetWeight     ( ev->GetWeight() );
   fTransformedEvent->SetBoostWeight( ev->GetBoostWeight() );
   fTransformedEvent->SetClass      ( ev->GetClass() );
   return fTransformedEvent;
}

//_______________________________________________________________________
const TMVA::Event* TMVA::VariableDecorrTransform::InverseTransform( const TMVA::Event* const ev, Int_t cls ) const
{
   // apply the inverse decorrelation transformation ... 
   // TODO : this is only a copy of the transform method... build the inverse transformation
   Log() << kFATAL << "Inverse transformation for decorrelation transformation not yet implemented. Hence, this transformation cannot be applied together with regression. Please contact the authors if necessary." << Endl;

   if (!IsCreated())
      Log() << kFATAL << "Transformation matrix not yet created" 
            << Endl;

   Int_t whichMatrix = cls;
   // if cls (the class chosen by the user) not existing, assume that he wants to have the matrix for all classes together. 
   if (cls < 0 || cls > GetNClasses()) {
      whichMatrix = GetNClasses();
   }

   TMatrixD* m = fDecorrMatrices.at(whichMatrix);
   if (m == 0) {
      if (whichMatrix == GetNClasses() )
         Log() << kFATAL << "Transformation matrix all classes is not defined" 
               << Endl;
      else
         Log() << kFATAL << "Transformation matrix for class " << whichMatrix << " is not defined" 
               << Endl;
   }

   // transformation to decorrelate the variables
   const Int_t nvar = GetNVariables();
   TVectorD vec( nvar );
   for (Int_t ivar=0; ivar<nvar; ivar++) vec(ivar) = ev->GetValue(ivar);

   // diagonalise variable vectors
   vec *= *m;

   if (fBackTransformedEvent==0 || fBackTransformedEvent->GetNVariables()!=ev->GetNVariables()) {
      if (fBackTransformedEvent!=0) { delete fBackTransformedEvent; fBackTransformedEvent = 0; }
      fBackTransformedEvent = new Event( *ev );
   }
   for (UInt_t itgt = 0; itgt < ev->GetNTargets(); itgt++ ) fBackTransformedEvent->SetTarget( itgt, ev->GetTarget(itgt) );
   for ( Int_t ivar=0; ivar<nvar; ivar++) fBackTransformedEvent->SetVal(ivar,vec(ivar));
   fBackTransformedEvent->SetWeight     ( ev->GetWeight() );
   fBackTransformedEvent->SetBoostWeight( ev->GetBoostWeight() );
   fBackTransformedEvent->SetClass      ( ev->GetClass() );

   return fBackTransformedEvent;
}


//_______________________________________________________________________
void TMVA::VariableDecorrTransform::CalcSQRMats( const std::vector<Event*>& events, Int_t maxCls )
{
   // compute square-root matrices for signal and background

   // delete old matrices if any
   for (std::vector<TMatrixD*>::iterator it = fDecorrMatrices.begin(); 
        it != fDecorrMatrices.end(); it++)
      if (0 != (*it) ) { delete (*it); *it=0; }


   // if more than one classes, then produce one matrix for all events as well (beside the matrices for each class)
   const UInt_t matNum = (maxCls<=1)?maxCls:maxCls+1;
   fDecorrMatrices.resize( matNum, (TMatrixD*) 0 );
      
   std::vector<TMatrixDSym*>* covMat = gTools().CalcCovarianceMatrices( events, maxCls );
   
   
   for (UInt_t cls=0; cls<matNum; cls++) {
      TMatrixD* sqrMat = gTools().GetSQRootMatrix( covMat->at(cls) );
      if ( sqrMat==0 ) 
         Log() << kFATAL << "<GetSQRMats> Zero pointer returned for SQR matrix" << Endl;
      fDecorrMatrices[cls] = sqrMat;
      delete (*covMat)[cls];
   }
   delete covMat;
}

//_______________________________________________________________________
void TMVA::VariableDecorrTransform::WriteTransformationToStream( std::ostream& o ) const
{
   // write the decorrelation matrix to the stream
   Int_t cls = 0;
   Int_t dp = o.precision();
   for (std::vector<TMatrixD*>::const_iterator itm = fDecorrMatrices.begin(); itm != fDecorrMatrices.end(); itm++) {
      o << "# correlation matrix " << std::endl;
      TMatrixD* mat = (*itm);
      o << cls << " " << mat->GetNrows() << " x " << mat->GetNcols() << std::endl;
      for (Int_t row = 0; row<mat->GetNrows(); row++) {
         for (Int_t col = 0; col<mat->GetNcols(); col++) {
            o << std::setprecision(12) << std::setw(20) << (*mat)[row][col] << " ";
         }
         o << std::endl;
      }
      cls++;
   }
   o << "##" << std::endl;
   o << std::setprecision(dp);
}

//_______________________________________________________________________
void TMVA::VariableDecorrTransform::AttachXMLTo(void* parent) 
{
   // node attachment to parent
   void* trf = gTools().AddChild(parent, "Transform");
   gTools().AddAttr(trf,"Name", "Decorrelation");

   for (std::vector<TMatrixD*>::const_iterator itm = fDecorrMatrices.begin(); itm != fDecorrMatrices.end(); itm++) {
      TMatrixD* mat = (*itm);
      /*void* decmat = gTools().xmlengine().NewChild(trf, 0, "Matrix");
        gTools().xmlengine().NewAttr(decmat,0,"Rows", gTools().StringFromInt(mat->GetNrows()) );
        gTools().xmlengine().NewAttr(decmat,0,"Columns", gTools().StringFromInt(mat->GetNcols()) );

        std::stringstream s;
        for (Int_t row = 0; row<mat->GetNrows(); row++) {
        for (Int_t col = 0; col<mat->GetNcols(); col++) {
        s << (*mat)[row][col] << " ";
        }
        }
        gTools().xmlengine().AddRawLine( decmat, s.str().c_str() );*/
      gTools().WriteTMatrixDToXML(trf,"Matrix",mat);
   }
}

//_______________________________________________________________________
void TMVA::VariableDecorrTransform::ReadFromXML( void* trfnode ) 
{
   // Read the transformation matrices from the xml node

   // first delete the old matrices
   for( std::vector<TMatrixD*>::iterator it = fDecorrMatrices.begin(); it != fDecorrMatrices.end(); it++ )
      if( (*it) != 0 ) delete (*it);
   fDecorrMatrices.clear();

   void* ch = gTools().GetChild(trfnode);
   while(ch!=0) {
      Int_t nrows, ncols;
      gTools().ReadAttr(ch, "Rows", nrows);
      gTools().ReadAttr(ch, "Columns", ncols);
      TMatrixD* mat = new TMatrixD(nrows,ncols);
      const char* content = gTools().GetContent(ch);
      std::stringstream s(content);
      for (Int_t row = 0; row<nrows; row++) {
         for (Int_t col = 0; col<ncols; col++) {
            s >> (*mat)[row][col];
         }
      }
      fDecorrMatrices.push_back(mat);
      ch = gTools().GetNextChild(ch);
   }
   SetCreated();
}


//_______________________________________________________________________
void TMVA::VariableDecorrTransform::ReadTransformationFromStream( std::istream& istr, const TString& classname )
{
   // Read the decorellation matrix from an input stream

   char buf[512];
   istr.getline(buf,512);
   TString strvar, dummy;
   Int_t nrows(0), ncols(0);
   UInt_t classIdx=0;
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
         UInt_t cls=0;
         if(strvar=="background") cls=1;
         if(strvar==classname) classIdx = cls;
         // coverity[tainted_data_argument]
         sstr >> nrows >> dummy >> ncols;
         if (fDecorrMatrices.size() <= cls ) fDecorrMatrices.resize(cls+1);
         if (fDecorrMatrices.at(cls) != 0) delete fDecorrMatrices.at(cls);
         TMatrixD* mat = fDecorrMatrices.at(cls) = new TMatrixD(nrows,ncols);
         // now read all matrix parameters
         for (Int_t row = 0; row<mat->GetNrows(); row++) {
            for (Int_t col = 0; col<mat->GetNcols(); col++) {
               istr >> (*mat)[row][col];
            }
         }
      } // done reading a matrix
      istr.getline(buf,512); // reading the next line
   }

   fDecorrMatrices.push_back( new TMatrixD(*fDecorrMatrices[classIdx]) );

   SetCreated();
}

//_______________________________________________________________________
void TMVA::VariableDecorrTransform::PrintTransformation( ostream& ) 
{
   // prints the transformation matrix
   Int_t cls = 0;
   for (std::vector<TMatrixD*>::iterator itm = fDecorrMatrices.begin(); itm != fDecorrMatrices.end(); itm++) {
      Log() << kINFO << "Transformation matrix "<< cls <<":" << Endl;
      (*itm)->Print();
   }
}

//_______________________________________________________________________
void TMVA::VariableDecorrTransform::MakeFunction( std::ostream& fout, const TString& fcncName, Int_t part, UInt_t trCounter, Int_t )
{
   // creates C++ code fragment of the decorrelation transform for inclusion in standalone C++ class

   Int_t dp = fout.precision();

   UInt_t numC = fDecorrMatrices.size();
   // creates a decorrelation function
   if (part==1) {
      TMatrixD* mat = fDecorrMatrices.at(0); // ToDo check if all Decorr matrices have identical dimensions
      fout << std::endl;
      fout << "   double fDecTF_"<<trCounter<<"["<<numC<<"]["<<mat->GetNrows()<<"]["<<mat->GetNcols()<<"];" << std::endl;
   }

   if (part==2) {
      fout << std::endl;
      fout << "//_______________________________________________________________________" << std::endl;
      fout << "inline void " << fcncName << "::InitTransform_"<<trCounter<<"()" << std::endl;
      fout << "{" << std::endl;
      for (UInt_t icls = 0; icls < numC; icls++){
         TMatrixD* matx = fDecorrMatrices.at(icls); 
         for (int i=0; i<matx->GetNrows(); i++) {
            for (int j=0; j<matx->GetNcols(); j++) {
               fout << "   fDecTF_"<<trCounter<<"["<<icls<<"]["<<i<<"]["<<j<<"] = " << std::setprecision(12) << (*matx)[i][j] << ";" << std::endl;
            }
         }
      }
      fout << "}" << std::endl;
      fout << std::endl;
      TMatrixD* matx = fDecorrMatrices.at(0); // ToDo check if all Decorr matrices have identicla dimensions
      fout << "//_______________________________________________________________________" << std::endl;
      fout << "inline void " << fcncName << "::Transform_"<<trCounter<<"( std::vector<double>& iv, int cls) const" << std::endl;
      fout << "{" << std::endl;
      fout << "   if (cls < 0 || cls > "<<GetNClasses()<<") {"<< std::endl;
      fout << "       if ("<<GetNClasses()<<" > 1 ) cls = "<<GetNClasses()<<";"<< std::endl;
      fout << "       else cls = "<<(fDecorrMatrices.size()==1?0:2)<<";"<< std::endl;
      fout << "   }"<< std::endl;
      fout << "   std::vector<double> tv;" << std::endl;
      fout << "   for (int i=0; i<"<<matx->GetNrows()<<";i++) {" << std::endl;
      fout << "      double v = 0;" << std::endl;
      fout << "      for (int j=0; j<"<<matx->GetNcols()<<"; j++)" << std::endl;
      fout << "         v += iv[j] * fDecTF_"<<trCounter<<"[cls][i][j];" << std::endl;
      fout << "      tv.push_back(v);" << std::endl;
      fout << "   }" << std::endl;
      fout << "   for (int i=0; i<"<<matx->GetNrows()<<";i++) iv[i] = tv[i];" << std::endl;
      fout << "}" << std::endl;
   }

   fout << std::setprecision(dp);
}
