// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariablePCATransform                                                  *
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

#include <iostream>
#include <iomanip>

#include "TVectorF.h"
#include "TVectorD.h"
#include "TMatrixD.h"
#include "TMatrixDBase.h"

#include "TMVA/VariablePCATransform.h"

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#include "TMVA/DataSet.h"
#include "TMVA/Tools.h"

ClassImp(TMVA::VariablePCATransform)

//_______________________________________________________________________
TMVA::VariablePCATransform::VariablePCATransform( DataSetInfo& dsi )
: VariableTransformBase( dsi, Types::kPCA, "PCA" )
{
   // constructor
}

//_______________________________________________________________________
TMVA::VariablePCATransform::~VariablePCATransform()
{
   // destructor
   for (UInt_t i=0; i<fMeanValues.size(); i++) {
      if (fMeanValues.at(i)   != 0) delete fMeanValues.at(i);
      if (fEigenVectors.at(i) != 0) delete fEigenVectors.at(i);
   }
}

//_______________________________________________________________________
void TMVA::VariablePCATransform::Initialize()
{
   // initialization of the transformation.
   // Has to be called in the preparation and not in the constructor,
   // since the number of classes it not known at construction, but
   // only after the creation of the DataSet which might be later.
}

//_______________________________________________________________________
Bool_t TMVA::VariablePCATransform::PrepareTransformation( const std::vector<Event*>& events )
{
   // calculate the principal components using the ROOT class TPrincipal
   // and the normalization
   Initialize();

   if (!IsEnabled() || IsCreated()) return kTRUE;

   Log() << kINFO << "Preparing the Principle Component (PCA) transformation..." << Endl;

   SetNVariables(events[0]->GetNVariables());

   // TPrincipal doesn't support PCA transformation for 1 or less variables
   if (GetNVariables() <= 1) {
      Log() << kINFO << "Cannot perform PCA transformation for " << GetNVariables() << " variable only" << Endl;
      return kFALSE;
   }

   if (GetNVariables() > 200) {
      Log() << kINFO << "----------------------------------------------------------------------------"
            << Endl;
      Log() << kINFO
            << ": More than 200 variables, will not calculate PCA!" << Endl;
      Log() << kINFO << "----------------------------------------------------------------------------"
            << Endl;
      return kFALSE;
   }

   CalculatePrincipalComponents( events );

   SetCreated( kTRUE );

   return kTRUE;
}

//_______________________________________________________________________
const TMVA::Event* TMVA::VariablePCATransform::Transform( const Event* const ev, Int_t cls ) const
{
   // apply the principal component analysis
   if (!IsCreated()) return 0;

   const Int_t nvar = ev->GetNVariables();
   // if we have more than one class, take the last PCA analysis where all classes are combined if
   // the cls parameter is outside the defined classes
   // If there is only one class, then no extra class for all events of all classes has to be created

   //if (cls < 0 || cls > GetNClasses()) cls = (fMeanValues.size()==1?0:2);//( GetNClasses() == 1 ? 0 : 1 );  ;
   // EVT this is a workaround to address the reader problem with transforma and EvaluateMVA(std::vector<float/double> ,...) 
   if (cls < 0 || cls >= (int) fMeanValues.size()) cls = fMeanValues.size()-1;
   // EVT workaround end

   // Perform PCA and put it into PCAed events tree

   if (fTransformedEvent==0 || fTransformedEvent->GetNVariables()!=ev->GetNVariables()) {
      if(fTransformedEvent!=0) delete fTransformedEvent;
      fTransformedEvent = new Event();
   }


   // set the variable values
   const std::vector<UInt_t>* varArrange = ev->GetVariableArrangement();

   if(!varArrange) {
      std::vector<Float_t> rv = X2P( ev->GetValues(), cls );
      for (Int_t ivar=0; ivar<nvar; ++ivar)
         fTransformedEvent->SetVal(ivar, rv[ivar]);
   } else {
      std::vector<Float_t> rv(nvar);
      for (Int_t ivar=0; ivar<nvar; ++ivar)
         rv[ivar] = ev->GetValue(ivar);
      rv = X2P( rv, cls );
      for (Int_t ivar=0; ivar<nvar; ++ivar)
         fTransformedEvent->SetVal(ivar, rv[ivar]);
   }
   // set the targets
   for (UInt_t itgt=0; itgt<ev->GetNTargets(); itgt++)
      fTransformedEvent->SetTarget( itgt, ev->GetTarget(itgt) );
   // and the rest
   fTransformedEvent->SetWeight     ( ev->GetWeight() );
   fTransformedEvent->SetBoostWeight( ev->GetBoostWeight() );
   fTransformedEvent->SetClass      ( ev->GetClass() );

   return fTransformedEvent;
}

//_______________________________________________________________________
const TMVA::Event* TMVA::VariablePCATransform::InverseTransform( const Event* const ev, Int_t cls ) const
{
   // apply the principal component analysis
   // TODO: implementation of inverse transformation
   Log() << kFATAL << "Inverse transformation for PCA transformation not yet implemented. Hence, this transformation cannot be applied together with regression. Please contact the authors if necessary." << Endl;

   if (!IsCreated()) return 0;
   const Int_t nvar = ev->GetNVariables();

   // if we have more than one class, take the last PCA analysis where all classes are combined if
   // the cls parameter is outside the defined classes
   // If there is only one class, then no extra class for all events of all classes has to be created
   if (cls < 0 || cls > GetNClasses()) cls = ( GetNClasses() == 1 ? 0 : 1 );


   // Perform PCA and put it into PCAed events tree
   std::vector<Float_t> rv = X2P( ev->GetValues(), cls );

   if (fBackTransformedEvent==0 || fBackTransformedEvent->GetNVariables()!=ev->GetNVariables()) {
      if(fBackTransformedEvent!=0) delete fBackTransformedEvent;
      fBackTransformedEvent = new Event( *ev );
   }
   for (Int_t ivar=0; ivar<nvar; ivar++) fBackTransformedEvent->SetVal(ivar, rv[ivar]);
   fBackTransformedEvent->SetClass      ( ev->GetClass() );
   fBackTransformedEvent->SetWeight     ( ev->GetWeight() );
   fBackTransformedEvent->SetBoostWeight( ev->GetBoostWeight() );

   return fBackTransformedEvent;
}

//_______________________________________________________________________
void TMVA::VariablePCATransform::CalculatePrincipalComponents( const std::vector<Event*>& events )
{
   // calculate the principal components for the signal and the background data
   // it uses the MakePrincipal method of ROOT's TPrincipal class

   const Int_t nvar = GetNVariables();

   // if we have more than one class, add another PCA analysis which combines all classes
   const UInt_t maxPCA = (GetNClasses()<=1) ? GetNClasses() : GetNClasses()+1;

   // PCA [signal/background/class x/class y/... /all classes]
   std::vector<TPrincipal*> pca(maxPCA);
   for (UInt_t i=0; i<maxPCA; i++) pca[i] = new TPrincipal(nvar,"");

   // !! Not normalizing and not storing input data, for performance reasons. Should perhaps restore normalization.

   Long64_t ievt, entries = events.size();
   Double_t *dvec = new Double_t[nvar];

   for (ievt=0; ievt<entries; ievt++) {
      Event* ev = events[ievt];
      for (Int_t i = 0; i < nvar; i++) dvec[i] = (Double_t) ev->GetValue(i);
      pca.at(ev->GetClass())->AddRow( dvec );
      if (GetNClasses() > 1) pca.at(maxPCA-1)->AddRow( dvec );
   }

   // delete possible leftovers
   for (UInt_t i=0; i<fMeanValues.size(); i++)   if (fMeanValues[i]   != 0) delete fMeanValues[i];
   for (UInt_t i=0; i<fEigenVectors.size(); i++) if (fEigenVectors[i] != 0) delete fEigenVectors[i];
   fMeanValues.resize(maxPCA,0);
   fEigenVectors.resize(maxPCA,0);

   for (UInt_t i=0; i<maxPCA; i++ ) {
      pca.at(i)->MakePrincipals();

      // retrieve mean values, eigenvectors and sigmas
      fMeanValues[i]   = new TVectorD( *(pca.at(i)->GetMeanValues()) ); // need to copy since we want to own
      fEigenVectors[i] = new TMatrixD( *(pca.at(i)->GetEigenVectors()) );
   }

   for (UInt_t i=0; i<maxPCA; i++) delete pca.at(i);
   delete [] dvec;
}

//_______________________________________________________________________
std::vector<Float_t> TMVA::VariablePCATransform::X2P( const std::vector<Float_t>& x, Int_t cls ) const
{
   // Calculate the principal components from the original data vector
   // x, and return it in p (function extracted from TPrincipal::X2P)
   // It's the users responsibility to make sure that both x and p are
   // of the right size (i.e., memory must be allocated for p)
   const Int_t nvar = x.size();
   std::vector<Float_t> p(nvar,0);

   for (Int_t i = 0; i < nvar; i++) {
      Double_t pv = 0;
      for (Int_t j = 0; j < nvar; j++)
         pv += (((Double_t)x.at(j)) - (*fMeanValues.at(cls))(j)) * (*fEigenVectors.at(cls))(j,i);
      p[i] = pv;
   }
   return p;
}

//_______________________________________________________________________
void TMVA::VariablePCATransform::WriteTransformationToStream( std::ostream& o ) const
{
   // write mean values to stream
   for (Int_t sbType=0; sbType<2; sbType++) {
      o << "# PCA mean values " << std::endl;
      const TVectorD* means = fMeanValues[sbType];
      o << (sbType==0 ? "Signal" : "Background") << " " << means->GetNrows() << std::endl;
      for (Int_t row = 0; row<means->GetNrows(); row++) {
         o << std::setprecision(12) << std::setw(20) << (*means)[row];
      }
      o << std::endl;
   }
   o << "##" << std::endl;

   // write eigenvectors to stream
   for (Int_t sbType=0; sbType<2; sbType++) {
      o << "# PCA eigenvectors " << std::endl;
      const TMatrixD* mat = fEigenVectors[sbType];
      o << (sbType==0 ? "Signal" : "Background") << " " << mat->GetNrows() << " x " << mat->GetNcols() << std::endl;
      for (Int_t row = 0; row<mat->GetNrows(); row++) {
         for (Int_t col = 0; col<mat->GetNcols(); col++) {
            o << std::setprecision(12) << std::setw(20) << (*mat)[row][col] << " ";
         }
         o << std::endl;
      }
   }
   o << "##" << std::endl;
}

//_______________________________________________________________________
void TMVA::VariablePCATransform::AttachXMLTo(void* parent) {
   // create XML description of PCA transformation

   void* trfxml = gTools().AddChild(parent, "Transform");
   gTools().AddAttr(trfxml, "Name", "PCA");

   // write mean values to stream
   for (UInt_t sbType=0; sbType<fMeanValues.size(); sbType++) {
      void* meanxml = gTools().AddChild( trfxml, "Statistics");
      const TVectorD* means = fMeanValues[sbType];
      gTools().AddAttr( meanxml, "Class",     (sbType==0 ? "Signal" :(sbType==1 ? "Background":"Combined")) );
      gTools().AddAttr( meanxml, "ClassIndex", sbType );
      gTools().AddAttr( meanxml, "NRows",      means->GetNrows() );
      TString meansdef = "";
      for (Int_t row = 0; row<means->GetNrows(); row++)
         meansdef += gTools().StringFromDouble((*means)[row]) + " ";
      gTools().AddRawLine( meanxml, meansdef );
   }

   // write eigenvectors to stream
   for (UInt_t sbType=0; sbType<fEigenVectors.size(); sbType++) {
      void* evxml = gTools().AddChild( trfxml, "Eigenvectors");
      const TMatrixD* mat = fEigenVectors[sbType];
      gTools().AddAttr( evxml, "Class",      (sbType==0 ? "Signal" :(sbType==1 ? "Background":"Combined") ) );
      gTools().AddAttr( evxml, "ClassIndex", sbType );
      gTools().AddAttr( evxml, "NRows",      mat->GetNrows() );
      gTools().AddAttr( evxml, "NCols",      mat->GetNcols() );
      TString evdef = "";
      for (Int_t row = 0; row<mat->GetNrows(); row++)
         for (Int_t col = 0; col<mat->GetNcols(); col++)
            evdef += gTools().StringFromDouble((*mat)[row][col]) + " ";
      gTools().AddRawLine( evxml, evdef );
   }
}

//_______________________________________________________________________
void TMVA::VariablePCATransform::ReadFromXML( void* trfnode )
{
   // Read the transformation matrices from the xml node

   Int_t nrows, ncols;
   UInt_t clsIdx;
   TString classtype;
   TString nodeName;

   void* ch = gTools().GetChild(trfnode);
   while (ch) {
      nodeName = gTools().GetName(ch);
      if (nodeName == "Statistics") {
         // read mean values
         gTools().ReadAttr(ch, "Class",      classtype);
         gTools().ReadAttr(ch, "ClassIndex", clsIdx);
         gTools().ReadAttr(ch, "NRows",      nrows);

         // set the correct size
         if (fMeanValues.size()<=clsIdx) fMeanValues.resize(clsIdx+1,0);
         if (fMeanValues[clsIdx]==0) fMeanValues[clsIdx] = new TVectorD( nrows );
         fMeanValues[clsIdx]->ResizeTo( nrows );

         // now read vector entries
         std::stringstream s(gTools().GetContent(ch));
         for (Int_t row = 0; row<nrows; row++) s >> (*fMeanValues[clsIdx])(row);
      }
      else if ( nodeName == "Eigenvectors" ) {
         // Read eigenvectors
         gTools().ReadAttr(ch, "Class",      classtype);
         gTools().ReadAttr(ch, "ClassIndex", clsIdx);
         gTools().ReadAttr(ch, "NRows",      nrows);
         gTools().ReadAttr(ch, "NCols",      ncols);

         if (fEigenVectors.size()<=clsIdx) fEigenVectors.resize(clsIdx+1,0);
         if (fEigenVectors[clsIdx]==0) fEigenVectors[clsIdx] = new TMatrixD( nrows, ncols );
         fEigenVectors[clsIdx]->ResizeTo( nrows, ncols );

         // now read matrix entries
         std::stringstream s(gTools().GetContent(ch));
         for (Int_t row = 0; row<nrows; row++)
            for (Int_t col = 0; col<ncols; col++)
               s >> (*fEigenVectors[clsIdx])[row][col];
      } // done reading eigenvectors
      ch = gTools().GetNextChild(ch);
   }

   SetCreated();
}

//_______________________________________________________________________
void TMVA::VariablePCATransform::ReadTransformationFromStream( std::istream& istr, const TString& classname )
{
   // Read mean values from input stream
   char buf[512];
   istr.getline(buf,512);
   TString strvar, dummy;
   Int_t nrows(0), ncols(0);
   UInt_t classIdx=(classname=="signal"?0:1);

   for (UInt_t i=0; i<fMeanValues.size(); i++) {
      if (fMeanValues.at(i)   != 0) delete fMeanValues.at(i);
      if (fEigenVectors.at(i) != 0) delete fEigenVectors.at(i);
   }
   fMeanValues.resize(3);
   fEigenVectors.resize(3);

   Log() << kINFO << "VariablePCATransform::ReadTransformationFromStream(): " << Endl;

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

         sstr >> nrows;
         Int_t sbType = (strvar=="signal" ? 0 : 1);

         if (fMeanValues[sbType] == 0) fMeanValues[sbType] = new TVectorD( nrows );
         else                          fMeanValues[sbType]->ResizeTo( nrows );

         // now read vector entries
         for (Int_t row = 0; row<nrows; row++) istr >> (*fMeanValues[sbType])(row);

      } // done reading vector

      istr.getline(buf,512); // reading the next line
   }

   // Read eigenvectors from input stream
   istr.getline(buf,512);
   while (!(buf[0]=='#'&& buf[1]=='#')) { // if line starts with ## return
      char* p = buf;
      while(*p==' ' || *p=='\t') p++; // 'remove' leading whitespace
      if (*p=='#' || *p=='\0') {
         istr.getline(buf,512);
         continue; // if comment or empty line, read the next line
      }
      std::stringstream sstr(buf);
      sstr >> strvar;
      if (strvar=="signal" || strvar=="background") {

         // coverity[tainted_data_argument]
         sstr >> nrows >> dummy >> ncols;
         Int_t sbType = (strvar=="signal" ? 0 : 1);

         if (fEigenVectors[sbType] == 0) fEigenVectors[sbType] = new TMatrixD( nrows, ncols );
         else                            fEigenVectors[sbType]->ResizeTo( nrows, ncols );

         // now read matrix entries
         for (Int_t row = 0; row<fEigenVectors[sbType]->GetNrows(); row++) {
            for (Int_t col = 0; col<fEigenVectors[sbType]->GetNcols(); col++) {
               istr >> (*fEigenVectors[sbType])[row][col];
            }
         }

      } // done reading matrix
      istr.getline(buf,512); // reading the next line
   }
   fMeanValues[2] = new TVectorD( *fMeanValues[classIdx] );
   fEigenVectors[2] = new TMatrixD( *fEigenVectors[classIdx] );

   SetCreated();
}

//_______________________________________________________________________
void TMVA::VariablePCATransform::MakeFunction( std::ostream& fout, const TString& fcncName,
                                               Int_t part, UInt_t trCounter, Int_t )
{
   // creates C++ code fragment of the PCA transform for inclusion in standalone C++ class

   UInt_t nvar = fEigenVectors[0]->GetNrows();

   // creates a PCA transformation function
   UInt_t numC = fMeanValues.size();
   if (part==1) {
      fout << std::endl;
      fout << "   void X2P_"<<trCounter<<"( const double*, double*, int ) const;" << std::endl;
      fout << "   double fMeanValues_"<<trCounter<<"["<<numC<<"]["
           << fMeanValues[0]->GetNrows()   << "];" << std::endl;   // mean values
      fout << "   double fEigenVectors_"<<trCounter<<"["<<numC<<"]["
           << fEigenVectors[0]->GetNrows() << "]["
           << fEigenVectors[0]->GetNcols() <<"];" << std::endl;   // eigenvectors
      fout << std::endl;
   }

   // sanity check
   if (numC>1){
      if (fMeanValues[0]->GetNrows()   != fMeanValues[1]->GetNrows() ||
          fEigenVectors[0]->GetNrows() != fEigenVectors[1]->GetNrows() ||
          fEigenVectors[0]->GetNcols() != fEigenVectors[1]->GetNcols()) {
         Log() << kFATAL << "<MakeFunction> Mismatch in vector/matrix dimensions" << Endl;
      }
   }

   if (part==2) {

      fout << std::endl;
      fout << "//_______________________________________________________________________" << std::endl;
      fout << "inline void " << fcncName << "::X2P_"<<trCounter<<"( const double* x, double* p, int index ) const" << std::endl;
      fout << "{" << std::endl;
      fout << "   // Calculate the principal components from the original data vector" << std::endl;
      fout << "   // x, and return it in p (function extracted from TPrincipal::X2P)" << std::endl;
      fout << "   // It's the users responsibility to make sure that both x and p are" << std::endl;
      fout << "   // of the right size (i.e., memory must be allocated for p)." << std::endl;
      fout << "   const int nvar = " << nvar << ";" << std::endl;
      fout << std::endl;
      fout << "   for (int i = 0; i < nvar; i++) {" << std::endl;
      fout << "      p[i] = 0;" << std::endl;
      fout << "      for (int j = 0; j < nvar; j++) p[i] += (x[j] - fMeanValues_"<<trCounter<<"[index][j]) * fEigenVectors_"<<trCounter<<"[index][j][i];" << std::endl;
      fout << "   }" << std::endl;
      fout << "}" << std::endl;
      fout << std::endl;
      fout << "//_______________________________________________________________________" << std::endl;
      fout << "inline void " << fcncName << "::InitTransform_"<<trCounter<<"()" << std::endl;
      fout << "{" << std::endl;

      // fill vector of mean values
      fout << "   // initialise vector of mean values" << std::endl;
      std::streamsize dp = fout.precision();
      for (UInt_t index=0; index<numC; index++) {
         for (int i=0; i<fMeanValues[index]->GetNrows(); i++) {
            fout << "   fMeanValues_"<<trCounter<<"["<<index<<"]["<<i<<"] = " << std::setprecision(12)
                 << (*fMeanValues[index])(i) << ";" << std::endl;
         }
      }

      // fill matrix of eigenvectors
      fout << std::endl;
      fout << "   // initialise matrix of eigenvectors" << std::endl;
      for (UInt_t index=0; index<numC; index++) {
         for (int i=0; i<fEigenVectors[index]->GetNrows(); i++) {
            for (int j=0; j<fEigenVectors[index]->GetNcols(); j++) {
               fout << "   fEigenVectors_"<<trCounter<<"["<<index<<"]["<<i<<"]["<<j<<"] = " << std::setprecision(12)
                    << (*fEigenVectors[index])(i,j) << ";" << std::endl;
            }
         }
      }
      fout << std::setprecision(dp);
      fout << "}" << std::endl;
      fout << std::endl;
      fout << "//_______________________________________________________________________" << std::endl;
      fout << "inline void " << fcncName << "::Transform_"<<trCounter<<"( std::vector<double>& iv, int cls ) const" << std::endl;
      fout << "{" << std::endl;
      fout << "   const int nvar = " << nvar << ";" << std::endl;
      fout << "   double *dv = new double[nvar];" << std::endl;
      fout << "   double *rv = new double[nvar];" << std::endl;
      fout << "   if (cls < 0 || cls > "<<GetNClasses()<<") {"<< std::endl;
      fout << "       if ("<<GetNClasses()<<" > 1 ) cls = "<<GetNClasses()<<";"<< std::endl;
      fout << "       else cls = "<<(numC==1?0:2)<<";"<< std::endl;
      fout << "   }"<< std::endl;
      fout << "   for (int ivar=0; ivar<nvar; ivar++) dv[ivar] = iv[ivar];" << std::endl;
      fout << std::endl;
      fout << "   // Perform PCA and put it into PCAed events tree" << std::endl;
      fout << "   this->X2P_"<<trCounter<<"( dv, rv, cls );" << std::endl;
      fout << "   for (int ivar=0; ivar<nvar; ivar++) iv[ivar] = rv[ivar];" << std::endl;
      fout << std::endl;
      fout << "   delete [] dv;" << std::endl;
      fout << "   delete [] rv;" << std::endl;
      fout << "}" << std::endl;
   }
}
