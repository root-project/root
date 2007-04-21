// @(#)root/tmva $Id: MethodSVM.cxx,v 1.8 2007/04/19 10:32:04 brun Exp $    
// Author: Marcin Wolter, Andrzej Zemla

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodSVM                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Marcin Wolter  <Marcin.Wolter@cern.ch> - IFJ PAN, Krakow, Poland          *
 *      Andrzej Zemla  <a_zemla@o2.pl>         - IFJ PAN, Krakow, Poland          *
 *      (IFJ PAN: Henryk Niewodniczanski Inst. Nucl. Physics, Krakow, Poland)     *   
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      PAN, Krakow, Poland                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// SMO Platt's SVM classifier with Keerthi & Shavade improvements   
//_______________________________________________________________________

#include "Riostream.h"
#include "TMath.h"
#include "TFile.h"

#ifndef ROOT_TMVA_MethodSVM
#include "TMVA/MethodSVM.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_Timer
#include "TMVA/Timer.h"
#endif

const int basketsize__ = 1280000;

ClassImp(TMVA::MethodSVM)

//_______________________________________________________________________
   TMVA::MethodSVM::MethodSVM( TString jobName, TString methodTitle, DataSet& theData, 
                               TString theOption, TDirectory* theTargetDir )
      : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor
   InitSVM();

   DeclareOptions();

   ParseOptions();

   ProcessOptions();

   SetKernel();

   PrepareDataToTrain();
}

//_______________________________________________________________________
TMVA::MethodSVM::MethodSVM( DataSet& theData, 
                            TString theWeightFile,  
                            TDirectory* theTargetDir )
   : TMVA::MethodBase( theData, theWeightFile, theTargetDir ) 
{
   // constructor from weight file
   InitSVM();

   DeclareOptions();
}

//_______________________________________________________________________
TMVA::MethodSVM::~MethodSVM( void )
{  
   // destructor

   if (fAlphas     != 0) delete fAlphas;  
   if (fErrorCache != 0) delete fErrorCache;
   if (fVariables  != 0) {
      for (Int_t i = 0; i < GetNvar(); i++) delete (*fVariables)[i];
      delete fVariables;
   }
   if (fNormVar    != 0) delete fNormVar; 
   if (fTypesVec   != 0) delete fTypesVec;
   if (fI          != 0) delete fI;
   if (fKernelDiag != 0) delete fKernelDiag;
}

//_______________________________________________________________________
void TMVA::MethodSVM::InitSVM( void )
{
   // default initialisation
   SetMethodName( "SVM" );
   SetMethodType( TMVA::Types::kSVM );
   SetTestvarName();

   fAlphas     = new vector< Float_t >( Data().GetNEvtTrain());
   fErrorCache = new vector< Float_t >( Data().GetNEvtTrain());
   fVariables  = new vector< Float_t* >( GetNvar() );
   for (Int_t i = 0; i < GetNvar(); i++)
      (*fVariables)[i] = new Float_t[ Data().GetNEvtTrain() ];

   fNormVar    = new vector< Float_t> ( Data().GetNEvtTrain() );
   fTypesVec   = new vector< Int_t >  ( Data().GetNEvtTrain() );
   fI          = new vector< Short_t >( Data().GetNEvtTrain() );
   fKernelDiag = new vector< Float_t> ( Data().GetNEvtTrain() );
}

void TMVA::MethodSVM::DeclareOptions() 
{
   // declare options available for this method
   DeclareOptionRef( fC         = 1.,   "C",       "C parameter" );
   DeclareOptionRef( fTolerance = 0.01, "Tol",     "Tolerance parameter" );
   DeclareOptionRef( fMaxIter   = 1000, "MaxIter", "Max number of training loops" );

   // for gaussian kernel parameter(s)
   DeclareOptionRef( fDoubleSigmaSquered = 2., "Sigma", "Kernel parameter: Sigma");
  
   // for polynomiarl kernel parameter(s)
   DeclareOptionRef( fOrder = 3, "Order", "Polynomial Kernel parameter: polynomial order");

   // for sigmoid kernel parameters
   DeclareOptionRef( fTheta = 1., "Theta", "Sigmoid Kernel parameter: Theta");
   DeclareOptionRef( fKappa = 1., "Kappa", "Sigmoid Kernel parameter: Kappa");
  
   DeclareOptionRef( fTheKernel = "Gauss", "Kernel", "Uses kernel function");
   AddPreDefVal( TString("Linear")     );
   AddPreDefVal( TString("Gauss")   );
   AddPreDefVal( TString("Polynomial") );
   AddPreDefVal( TString("Sigmoid")    );
}

void TMVA::MethodSVM::ProcessOptions() 
{
   // evaluate options 
   MethodBase::ProcessOptions();

   if      (fTheKernel == "Linear"    ) fKernelType = kLinear;
   else if (fTheKernel == "Gauss"     ) fKernelType = kRBF;
   else if (fTheKernel == "Polynomial") fKernelType = kPolynomial;
   else if (fTheKernel == "Sigmoid"   ) fKernelType = kSigmoidal;
   else {
      fLogger << kWARNING <<"unknown kernel function! Choose Linear" << Endl;
      fKernelType = kLinear;
   }
}

//_______________________________________________________________________
void TMVA::MethodSVM::Train( void )
{
   // train the SVM

   Int_t numChanged  = 0;
   Int_t examineAll  = 1;
  
   fB_low =  1;
   fB_up  = -1;
  
   fI_low = Data().GetNEvtTrain()-1;
   fI_up = 0;
  
   (*fErrorCache)[fI_up]  = -1;
   (*fErrorCache)[fI_low] =  1;

   // timing
   TMVA::Timer timer( GetName() );
   fLogger << kINFO << "Sorry, no computing time forecast available for SVM, please wait ..." << Endl;

   Int_t numit = 0;
   while ((numChanged > 0) || (examineAll > 0)) {
      numChanged = 0;
    
      if (examineAll) {
         for (Int_t k =0; k < Data().GetNEvtTrain(); k++)
            numChanged += this->ExamineExample(k);
      }
      else {
         for (Int_t k =0; k < Data().GetNEvtTrain(); k++) {
            if ((*fI)[k] == 0) {
               numChanged += this->ExamineExample(k);
               if (fB_up > fB_low - 2*fTolerance) {
                  numChanged = 0;
                  break;
               }
            }
         }
      }

      if      (examineAll == 1) examineAll = 0;
      else if (numChanged == 0) examineAll = 1;
      
      // indicate unordered progress
      if (fB_up > fB_low - 2*fTolerance) 
         timer.DrawProgressBar( Form( "number-changed/examine-all/delta/counter: (%i, %i, %g, %i)", 
                                      numChanged, examineAll, fB_up - fB_low + 2*fTolerance, ++numit) );
      if (numit >= fMaxIter) {
         fLogger << kWARNING << "<Train> Max number of iterations exceeded. "
                 << " Training may not be completed. Try use less C parameter" << Endl;
         break;
      }
   }

   fLogger << kINFO << "<Train> elapsed time: " << timer.GetElapsedTime()    
           << "                                          " << Endl;
    
   fLogger << kINFO << "<Train> number of iterations: " << numit << Endl;
 
   fBparm = 0.5*( fB_low + fB_up );
  
   delete fI; fI = 0;
   delete fErrorCache; fErrorCache = 0;

   this->Results();
  
   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;
}

//_______________________________________________________________________
void  TMVA::MethodSVM::WriteWeightsToStream( ostream& o ) const
{
   // write configuration to output stream
   if(TxtWeightsOnly()) {
      Int_t evtCounter=0;
      o << fBparm << endl;

      for (Int_t ievt = 0; ievt < Data().GetNEvtTrain(); ievt++)
         if ((*fAlphas)[ievt] != 0) evtCounter++;
      o << evtCounter << endl;

      evtCounter=0;
      for (Int_t ievt = 0; ievt < Data().GetNEvtTrain(); ievt++) {
         if ((*fAlphas)[ievt] != 0) {
            o << evtCounter++ << "    " << (Double_t)(*fAlphas)[ievt] * (*fTypesVec)[ievt];

            for (Int_t ivar = 0; ivar < GetNvar(); ivar++) o << " " << (*fVariables)[ivar][ievt];
            o << endl;
         }
      }
      // write max data values
      for (Int_t ivar = 0; ivar < GetNvar(); ivar++) o << GetXmax( ivar ) << " ";
      o << endl;

      for (Int_t ivar = 0; ivar < GetNvar(); ivar++) o << GetXmin( ivar ) << " ";
      o << endl;

   } 
   else {
      TString rfname( GetWeightFileName() ); rfname.ReplaceAll( ".txt", ".root" );
      o << "# weights stored in root i/o file: " << rfname << endl;  
      o << fBparm << endl;
   }
}

//_______________________________________________________________________
void TMVA::MethodSVM::WriteWeightsToStream( TFile& ) const
{
   // write training sample (TTree) to file
    
   TTree *suppVecTree = new TTree("SuppVecTree", "Support Vector tree");
    
   UInt_t nvar = 0;
    
   Float_t* sVVar = new Float_t[ GetNvar() ];
   vector< Double_t > *alpha_t = new vector< Double_t >;
    
   // create tree branches
    
   for (UInt_t ivar=0; ivar < Data().GetNVariables(); ivar++) {
      // add Branch to Support Vector Tree
      const char* myVar = Data().GetInternalVarName(ivar).Data();
      char vt = Data().VarType(ivar);   // the variable type, 'F' 
      if (vt=='F') { 
         suppVecTree->Branch( myVar,&sVVar[nvar], Form("%s/%c", myVar, vt), basketsize__ );
         nvar++;
      }
      
      else {
         fLogger << kFATAL << "<WriteWeightsToStream> unknown variable type '" 
                 << vt << "' encountered; allowed are: 'F'"
                 << Endl;
      }
   } // end of loop over input variables
    
   for (Int_t ievt = 0; ievt < Data().GetNEvtTrain(); ievt++) {
      if ((*fAlphas)[ievt] != 0) {
	
         for (Int_t ivar = 0; ivar < GetNvar(); ivar++) {
            sVVar[ivar] = (*fVariables)[ivar][ievt];
         }
         alpha_t->push_back((Double_t)(*fAlphas)[ievt] * (*fTypesVec)[ievt]);
	
         suppVecTree->Fill();
      }
   }
    
   TVectorD alphaVec(alpha_t->size());
   for (UInt_t i = 0; i < alpha_t->size(); i++) alphaVec[i] = (*alpha_t)[i];

   alphaVec.Write("AlphasVector");

   // Write min, max values
   TVectorD maxVars( GetNvar() );
   TVectorD minVars( GetNvar() );
  
   for( Int_t ivar = 0; ivar < GetNvar(); ivar ++){
      maxVars[ivar] = GetXmax( ivar );
      minVars[ivar] = GetXmin( ivar );
   }
   maxVars.Write("MaxVars");
   minVars.Write("MinVars");

   delete alpha_t; alpha_t = 0;
    
   delete [] sVVar; sVVar = 0;
} 

  
//_______________________________________________________________________
void  TMVA::MethodSVM::ReadWeightsFromStream( istream& istr )
{
   // read configuration from input stream
   if(TxtWeightsOnly()) {
      istr >> fBparm;
   
      istr >> fNsupv;
      if(fAlphas!=0) delete fAlphas;
      fAlphas = new vector< Float_t >(fNsupv+1);

      if(fVariables!=0) {
         for(vector< Float_t* >::iterator it = fVariables->begin(); it!=fVariables->end(); it++)
            delete[] *it;
         delete fVariables;
      }
      fVariables = new vector< Float_t* >(GetNvar());
      for (Int_t i = 0; i < GetNvar(); i++) 
         (*fVariables)[i] = new Float_t[fNsupv + 1];

      if(fNormVar!=0) delete fNormVar;
      fNormVar = new vector< Float_t >(fNsupv + 1);

      fMaxVars = new TVectorD( GetNvar() );
      fMinVars = new TVectorD( GetNvar() );

      Double_t readTmp;
      Int_t IEvt;
      for (Int_t ievt = 0; ievt < fNsupv; ievt++) {
         istr >> IEvt >> (*fAlphas)[ievt];

         (*fNormVar)[ievt] = 0;
         for (Int_t ivar = 0; ivar < GetNvar(); ivar++) {
            istr >> readTmp;
            (*fVariables)[ivar][ievt] = readTmp;
            (*fNormVar)[ievt] += readTmp * readTmp;
         }
      }

      for (Int_t ivar = 0; ivar < GetNvar(); ivar++)
         istr >> (*fMaxVars)[ivar];
     
      for (Int_t ivar = 0; ivar < GetNvar(); ivar++)
         istr >> (*fMinVars)[ivar];
       
      SetKernel();
   } else {
      istr >> fBparm;
   }
}

//_______________________________________________________________________
void TMVA::MethodSVM::ReadWeightsFromStream( TFile& fFin )
{
   // read training sample from file
   TTree *suppVecTree = (TTree*)fFin.Get( "SuppVecTree" );
  
   // reading support vectors from tree to vectors
   Int_t  nevt = (Int_t)suppVecTree->GetEntries();
   fNsupv = nevt;
  
   Int_t nvar = suppVecTree->GetNbranches(); 
  
   Float_t *var = new Float_t[nvar];
   Int_t i = 0; 

   TIter next_branch1( suppVecTree->GetListOfBranches() );
   while (TBranch *branch = (TBranch*)next_branch1())
      suppVecTree->SetBranchAddress( branch->GetName(), &var[i++]);
   
   TVectorD *alphaVec = (TVectorD*)fFin.Get( "AlphasVector" );

   fMaxVars = new TVectorD( GetNvar() );
   fMinVars = new TVectorD( GetNvar() );
      
   fMaxVars  = (TVectorD*)fFin.Get( "MaxVars");
   fMinVars  = (TVectorD*)fFin.Get( "MinVars");
   
   fAlphas = new vector< Float_t >( nevt + 1 );  
   
   fVariables = new vector< Float_t* >(nvar);
   fAlphas = new vector< Float_t >( nevt + 1 );  
   
   fVariables = new vector< Float_t* >(nvar);
   
   for (Int_t i = 0; i < nvar; i++) (*fVariables)[i] = new Float_t[nevt + 1];

   fNormVar = new vector< Float_t >(nevt + 1);
     
   for (Int_t ievt = 0; ievt < nevt; ievt++) {      
      suppVecTree->GetEntry(ievt);
      (*fNormVar)[ievt] = 0;
      for (Int_t ivar = 0; ivar < nvar; ivar++) { // optymalizacja
         (*fVariables)[ivar][ievt] = var[ivar];
         (*fNormVar)[ievt] += (*fVariables)[ivar][ievt] * (*fVariables)[ivar][ievt];
      }
      (*fAlphas)[ievt] = (Float_t)(*alphaVec)(ievt);
   }
   SetKernel();
   delete [] var;
}

//_______________________________________________________________________
Double_t TMVA::MethodSVM::GetMvaValue()
{
   // returns MVA value for given event
   Double_t myMVA = 0;
   (*fNormVar)[fNsupv] = 0; 

   for (Int_t ivar = 0; ivar < GetNvar(); ivar++) {
      (*fVariables)[ivar][fNsupv] = TMVA::Tools::NormVariable( GetEventVal( ivar ), (*fMinVars)[ivar], (*fMaxVars)[ivar]);
      (*fNormVar)[fNsupv] += (*fVariables)[ivar][fNsupv] * (*fVariables)[ivar][fNsupv]; 
   }

   for (Int_t ievt = 0; ievt < fNsupv - 1; ievt++) {
      myMVA = myMVA + (*fAlphas)[ievt] * (this->*fKernelFunc)(fNsupv, ievt);
   }

   myMVA -=fBparm;

   return 1.0/(1.0 + TMath::Exp(-myMVA));
}

//_______________________________________________________________________
void TMVA::MethodSVM::PrepareDataToTrain()
{
   // puts all events in std::vectors 
   //normalized data
   for (Int_t ievt = 0; ievt < Data().GetNEvtTrain(); ievt++) {   

      ReadTrainingEvent( ievt );  
      (*fNormVar)[ievt] = 0.;

      for (Int_t ivar = 0; ivar < GetNvar(); ivar++) {
         (*fVariables)[ivar][ievt] = GetEvent().GetValueNormalized( ivar );
         (*fNormVar)[ievt] +=  (*fVariables)[ivar][ievt]* (*fVariables)[ivar][ievt];
      }
      if (GetEvent().IsSignal()) {
         (*fTypesVec)[ievt] = 1;
         (*fI)[ievt] = 1;
      }
      else {
         (*fTypesVec)[ievt] = -1;
         (*fI)[ievt] = -1;
      }
   } 
   for (Int_t ievt = 0; ievt < Data().GetNEvtTrain(); ievt++) {
      switch ( fKernelType ){

      case kLinear:
         (*fKernelDiag)[ievt] = (*fNormVar)[ievt];
         break;

      case kRBF:
         (*fKernelDiag)[ievt] = 1;
         break;

      default:
         (*fKernelDiag)[ievt] = (this->*fKernelFunc)(ievt, ievt);
         break;
      }
   }
}

//________________________________________________________________________
void TMVA::MethodSVM::SetKernel()
{
   // set the Kernel according to the fKernelType variable

   switch( fKernelType ) {
    
   case kLinear:
      fKernelFunc = &MethodSVM::LinearKernel;
      fWeightVector = new vector<Float_t>( GetNvar() );
      break;
    
   case kRBF:
      fKernelFunc = &MethodSVM::RBFKernel;
      if (fDoubleSigmaSquered <= 0.) {
         fDoubleSigmaSquered = 1.;
         fLogger <<kWARNING << "wrong Sigma value, uses default ::"<<fDoubleSigmaSquered<<endl;
      }
      break;
     
   case kPolynomial:
      fKernelFunc = &MethodSVM::PolynomialKernel;
      if (fOrder < 2) {
         fOrder = 2;
         fLogger << kWARNING << "wrong polynomial order! Choose Order = "<< fOrder <<Endl;
      }
      break;
    
   case kSigmoidal:
      fKernelFunc = &MethodSVM::SigmoidalKernel;
      break;
     
   }
}

//________________________________________________________________________
Int_t TMVA::MethodSVM::ExamineExample( Int_t jevt )
{
   // Examine example

   Int_t ievt = 0;
   Float_t fType_J;
   Float_t fErrorC_J;
   Float_t fAlpha_J;
  
  
   fType_J = (*fTypesVec)[jevt];
   fAlpha_J = (*fAlphas)[jevt];
  
   if ((*fI)[jevt] == 0) fErrorC_J = (*fErrorCache)[jevt]; 
   else {
      fErrorC_J = LearnFunc( jevt ) - fType_J;
      (*fErrorCache)[jevt] = fErrorC_J;
        
      // update (fB_low, fI_low)
      if (((*fI)[jevt] == 1)&&(fErrorC_J < fB_up )) {
         fB_up = fErrorC_J;
         fI_up = jevt;
      }
      else
         if (((*fI)[jevt] == -1)&&(fErrorC_J > fB_low)) {
            fB_low = fErrorC_J;
            fI_low = jevt;
         }
   }
   Bool_t converged = kTRUE;
  
   if ((*fI)[jevt]>= 0) {
      if (fB_low - fErrorC_J > 2*fTolerance) {
         converged = kFALSE;
         ievt = fI_low; 
      }
   }
  
   if ((*fI)[jevt]<= 0) {
      if ((fErrorC_J - fB_up) > 2*fTolerance) {
         converged = kFALSE;
         ievt = fI_up;
      }
   }
  
   if (converged) return 0;
  
   if ((*fI)[jevt] == 0) {
      if (fB_low - fErrorC_J > fErrorC_J - fB_up) ievt = fI_low;
      else                                        ievt = fI_up;
   }
  
   if (TakeStep(ievt, jevt)) return 1;
   else                      return 0;
}

//________________________________________________________________________
Int_t TMVA::MethodSVM::TakeStep( Int_t ievt , Int_t jevt)
{ 
   // take step

   if (ievt == jevt) return 0;
   const Float_t epsilon = 1e-12;

   Float_t type_I,  type_J;
   Float_t errorC_I,  errorC_J;
   Float_t alpha_I, alpha_J;
    
   Float_t newAlpha_I, newAlpha_J;
   Int_t   s;  

   Float_t l, h, lobj = 0, hobj = 0;
   Float_t eta;

   type_I   = (*fTypesVec)[ievt];
   alpha_I  = (*fAlphas)[ievt];
   errorC_I = (*fErrorCache)[ievt];

   type_J   = (*fTypesVec)[jevt];
   alpha_J  = (*fAlphas)[jevt];
   errorC_J = (*fErrorCache)[jevt];
    
   s = Int_t( type_I * type_J );
  
   // compute l, h

   if (type_I == type_J) {
      Float_t gamma = alpha_I + alpha_J;
      if (gamma > fC) {
         l = gamma -fC;
         h = fC;
      }
      else {
         l = 0;
         h = gamma;
      }
   }
   else {
      Float_t gamma = alpha_I - alpha_J;
      if (gamma > 0) {
         l = 0;
         h = fC - gamma;
      }
      else {
         l = -gamma;
         h = fC;
      }
   }
  
   if (l == h)  return 0;

   Float_t kernel_II, kernel_IJ, kernel_JJ;

   kernel_II = (*fKernelDiag)[ievt];
   kernel_IJ = (this->*fKernelFunc)( ievt, jevt );
   kernel_JJ =  (*fKernelDiag)[jevt];

   eta = 2*kernel_IJ - kernel_II - kernel_JJ; 
 
   if (eta < 0) {
      newAlpha_J = alpha_J + (type_J*( errorC_J - errorC_I ))/eta;
    
      if      (newAlpha_J < l) newAlpha_J = l;
      else if (newAlpha_J > h) newAlpha_J = h;
   }

   else {

      Float_t c_I = eta/2;
      Float_t c_J = type_J*( errorC_I - errorC_J ) - eta * alpha_J;
      lobj = c_I * l * l + c_J * l;
      hobj = c_I * h * h + c_J * h;

      if      (lobj > hobj + epsilon)  newAlpha_J = l;
      else if (lobj < hobj - epsilon)  newAlpha_J = h; 
      else                              newAlpha_J = alpha_J;
   }

   if (TMath::Abs( newAlpha_J - alpha_J ) < ( epsilon * ( newAlpha_J + alpha_J+ epsilon )))
      return 0;

   newAlpha_I = alpha_I - s*( newAlpha_J - alpha_J );

   if (newAlpha_I < 0) {
      newAlpha_J += s* newAlpha_I;
      newAlpha_I = 0;
   }
   else if (newAlpha_I > fC) {
      Float_t temp = newAlpha_I -fC;
      newAlpha_J += s * temp;
      newAlpha_I = fC;
   }
  
   Float_t dL_I = type_I * ( newAlpha_I - alpha_I );
   Float_t dL_J = type_J * ( newAlpha_J - alpha_J );  

   if (fKernelType == kLinear) {
      for (Int_t ivar = 0; ivar < GetNvar(); ivar++)
         (*fWeightVector)[ivar] = ( (*fWeightVector)[ivar] +  
                                    dL_I * (*fVariables)[ivar][ievt] + 
                                    dL_J*(*fVariables)[ivar][jevt] );
   }
  
   // update error cache
   for (Int_t i = 0; i < Data().GetNEvtTrain(); i++ ) {

      if ((*fI)[i] == 0)
         (*fErrorCache)[i] += ( dL_I * (this->*fKernelFunc)( ievt, i ) + 
                                dL_J * (this->*fKernelFunc)( jevt, i ) );
   }

   // store new alphas
   (*fAlphas)[ievt] = newAlpha_I;
   (*fAlphas)[jevt] = newAlpha_J;

   // set new indexes
   SetIndex(ievt);
   SetIndex(jevt);

   // update error cache
   (*fErrorCache)[ievt] = errorC_I + dL_I*kernel_II + dL_J*kernel_IJ;
   (*fErrorCache)[jevt] = errorC_J + dL_I*kernel_IJ + dL_J*kernel_JJ;

   // compute fI_low, fB_low

   fB_low = -1*1e30;
   fB_up = 1e30;
  
   for (Int_t i = 0; i < Data().GetNEvtTrain(); i++) {
      if ((*fI)[i] == 0) {
         if ((*fErrorCache)[i]> fB_low) {
            fB_low = (*fErrorCache)[i];
            fI_low = i;
         }
      
         if ((*fErrorCache)[i]< fB_up) {
            fB_up = (*fErrorCache)[i];
            fI_up = i;
         }
      }
   }

   // for optimized alfa's
   if (fB_low < TMath::Max((*fErrorCache)[ievt], (*fErrorCache)[jevt])) {
      if ((*fErrorCache)[ievt]> fB_low) {
         fB_low = (*fErrorCache)[ievt];
         fI_low = ievt;
      }
      else {
         fB_low = (*fErrorCache)[jevt];
         fI_low = jevt;
      }
   }
  
   if (fB_up > TMath::Max((*fErrorCache)[ievt], (*fErrorCache)[jevt])) {
      if ((*fErrorCache)[ievt]< fB_low) {
         fB_up = (*fErrorCache)[ievt];
         fI_up = ievt;
      }
      else {
         fB_up = (*fErrorCache)[jevt];
         fI_up = jevt;
      }
   }  
   return 1;
}
//_____________________________________________________________
Float_t TMVA::MethodSVM::LinearKernel( Int_t ievt, Int_t jevt ) const 
{
   // linear kernel
   Float_t val = 0.; 

   for (Int_t ivar = 0; ivar < GetNvar(); ivar++)
      val += (*fVariables)[ivar][ievt] * (*fVariables)[ivar][jevt];
   return val;
}

//_____________________________________________________________
Float_t TMVA::MethodSVM::RBFKernel( Int_t ievt, Int_t jevt ) const
{
   // radial basis function kernel
   
   Float_t val = TMVA::MethodSVM::LinearKernel( ievt, jevt );
   val *= (-2);
   val += (*fNormVar)[ievt] + (*fNormVar)[jevt];
  
   return TMath::Exp( -val/fDoubleSigmaSquered );
}

//_____________________________________________________________
Float_t TMVA::MethodSVM::PolynomialKernel( Int_t ievt, Int_t jevt ) const
{
   // polynomial kernel

   Float_t val = TMVA::MethodSVM::LinearKernel( ievt, jevt );
   val += fTheta;
   Float_t valResult = 1.;
   for (Int_t i = fOrder; i > 0; i /= 2){
      if (i%2) valResult = val; 
      val *= val; 
   } 
   
   return valResult;
}

//_____________________________________________________________
Float_t TMVA::MethodSVM::SigmoidalKernel( Int_t ievt, Int_t jevt ) const 
{
   // sigmoid Kernel

   Float_t val = fKappa*TMVA::MethodSVM::LinearKernel( ievt, jevt );
   val += fTheta;

   return TMath::TanH( val );
}

//______________________________________________________________
Float_t TMVA::MethodSVM::LearnFunc( Int_t kevt)
{
   // learn function

   Float_t s = 0.;
   for (Int_t ievt = 0; ievt < Data().GetNEvtTrain(); ievt++) {
      if ((*fAlphas)[ievt]>0)
         s+=(*fAlphas)[ievt] * (Float_t)(*fTypesVec)[ievt] * (this->*fKernelFunc)( ievt, kevt );
   }

   return s;
}

//_______________________________________________________________
void TMVA::MethodSVM::SetIndex( Int_t ievt )
{
   // set the index

   if ((0<(*fAlphas)[ievt]) && ((*fAlphas)[ievt])<fC) (*fI)[ievt]=0; // I0
  
   if ((*fTypesVec)[ievt] == 1) {
      if      ((*fAlphas)[ievt] == 0)  (*fI)[ievt] =  1;     // I1
      else if ((*fAlphas)[ievt] == fC) (*fI)[ievt] = -1;     // I3
   }
  
   if ((*fTypesVec)[ievt] == -1) {
      if      ((*fAlphas)[ievt] == 0)  (*fI)[ievt] = -1;     // I4
      else if ((*fAlphas)[ievt] == fC) (*fI)[ievt] =  1;     // I2
   }
}

//_______________________________________________________________
void TMVA::MethodSVM::Results()
{
   // results
   Int_t nvec = 0;
   for (Int_t i = 0; i < Data().GetNEvtTrain(); i++) if ((*fAlphas)[i] == 0) nvec++;

   fLogger << kINFO << "Results:" << Endl;
   fLogger << kINFO << "- number of support vectors: " << nvec 
           << " (" << 100*nvec/Data().GetNEvtTrain() << "%)" << Endl;
   fLogger << kINFO << "- b: " << fBparm << Endl;
}
