// ver. 09.05.2007 wprowadzone wagi ( ale bez konkretnych poprawek )
// @(#)root/tmva $Id: MethodSVM.cxx,v 1.59 2007/06/15 22:01:33 andreas.hoecker Exp $    
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
 *      CERN, Switzerland                                                         * 
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

Bool_t wbug = kTRUE;
ClassImp(TMVA::MethodSVM)

//_______________________________________________________________________
TMVA::MethodSVM::MethodSVM( TString jobName, TString methodTitle, DataSet& theData, 
                            TString theOption, TDirectory* theTargetDir )
   : TMVA::MethodBase( jobName, methodTitle, theData, theOption, theTargetDir )
{
   // standard constructor
   InitSVM();

   // interpretation of configuration option string
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
TMVA::MethodSVM::~MethodSVM()
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

   if (fSupportVectors!=0) {
      for (vector<Float_t*>::iterator it = fSupportVectors->begin(); it!=fSupportVectors->end(); it++)
         delete[] *it;
      delete fSupportVectors;
   }
}

//_______________________________________________________________________
void TMVA::MethodSVM::InitSVM()
{
   // default initialisation
   SetMethodName( "SVM" );
   SetMethodType( TMVA::Types::kSVM );
   SetTestvarName();

   // SVM always uses normalised input variables
   SetNormalised( kTRUE );

   fAlphas     = new vector< Float_t >( Data().GetNEvtTrain());
   fErrorCache = new vector< Float_t >( Data().GetNEvtTrain());
   fVariables  = new vector< Float_t* >( GetNvar() );
   for (Int_t i = 0; i < GetNvar(); i++)
      (*fVariables)[i] = new Float_t[ Data().GetNEvtTrain() ];

   fNormVar  = new vector< Float_t> ( Data().GetNEvtTrain() );
   fTypesVec = new vector< Int_t >  ( Data().GetNEvtTrain() );
   fI        = new vector< Short_t >( Data().GetNEvtTrain() );
   fKernelDiag = new vector< Float_t> ( Data().GetNEvtTrain() );
}

void TMVA::MethodSVM::DeclareOptions() 
{
   // declare options available for this method
   DeclareOptionRef( fC         = 1.,   "C",       "C parameter" );
   DeclareOptionRef( fTolerance = 0.01, "Tol",     "Tolerance parameter" );
   DeclareOptionRef( fMaxIter   = 1000, "MaxIter", "Maximum number of training loops" );

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
void TMVA::MethodSVM::Train()
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
   Timer timer( GetName() );
   fLogger << kINFO << "Sorry, no computing time forecast available for SVM, please wait ..." << Endl;

   Float_t numChangedOld = 0;
   Int_t deltaChanges = 0;
   Int_t numit    = 0;
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
               if ((fB_up > fB_low - 2*fTolerance) ) {
                  numChanged = 0;
                  break;
               }
            }
         }
      }

      if      (examineAll == 1) examineAll = 0;
      else if (numChanged == 0 || numChanged < 10 || deltaChanges > 3 ) examineAll = 1;

      if (numChanged == numChangedOld) deltaChanges++;
      else                             deltaChanges = 0;
      numChangedOld = numChanged;
      ++numit;

      // indicate unordered progress
      if (fB_up > fB_low - 2*fTolerance) 
         timer.DrawProgressBar( Form( "number-changed/examine-all/delta/counter: (%i, %i, %g, %i)", 
                                      numChanged, examineAll, fB_up - fB_low + 2*fTolerance, numit) );
      if ( numit >= fMaxIter){
         fLogger << kWARNING << "<Train> Max number of iterations exceeded. "
                 << "Training may not be completed. Try use less C parameter" << Endl;
         break;
      }
   }

   fLogger << kINFO << "<Train> elapsed time: " << timer.GetElapsedTime()    
           << "                                          " << Endl;
    
   fLogger << kINFO << "<Train> number of iterations: " << numit << Endl;
 
   fBparm = 0.5*( fB_low + fB_up );
  
   delete fI; fI = 0;
   delete fErrorCache; fErrorCache = 0;

   Results();
   StoreSupportVectors();

   // default sanity checks
   if (!CheckSanity()) fLogger << kFATAL << "<Train> sanity check failed" << Endl;
}

//_______________________________________________________________________
void  TMVA::MethodSVM::WriteWeightsToStream( ostream& o ) const
{
   // write configuration to output stream
   if (TxtWeightsOnly()) {
      o << fBparm << endl;
      o << fNsupv << endl;


      for (Int_t isv = 0; isv < fNsupv; isv++ ) {
         o << isv;
         for (Int_t ivar = 0; ivar <= GetNvar(); ivar++)	  o << " " << (*fSupportVectors)[ivar][isv];
         o << endl;
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
      suppVecTree->Branch( myVar,&sVVar[nvar], Form("%s/F", myVar), basketsize__ );
      nvar++;
   }
    
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
  
   for ( Int_t ivar = 0; ivar < GetNvar(); ivar ++) {
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
   if (TxtWeightsOnly()) {
      istr >> fBparm;
   
      istr >> fNsupv;
      if (fAlphas!=0) delete fAlphas;
      fAlphas = new vector< Float_t >(fNsupv+1);

      if (fVariables!=0) {
         for (vector< Float_t* >::iterator it = fVariables->begin(); it!=fVariables->end(); it++)
            delete[] *it;
         delete fVariables;
      }
      fVariables = new vector< Float_t* >(GetNvar());
      for (Int_t i = 0; i < GetNvar(); i++) 
         (*fVariables)[i] = new Float_t[fNsupv + 1];

      if (fNormVar!=0) delete fNormVar;
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
   } 
   else {
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
   
   fAlphas    = new vector<Float_t >(nevt + 1);     
   fVariables = new vector<Float_t*>(nvar);

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
      (*fVariables)[ivar][fNsupv] = GetEventVal( ivar );
      (*fNormVar)[fNsupv] += (*fVariables)[ivar][fNsupv] * (*fVariables)[ivar][fNsupv]; 
   }

   for (Int_t ievt = 0; ievt < fNsupv ; ievt++) {
      myMVA += (*fAlphas)[ievt] * (this->*fKernelFunc)(fNsupv, ievt);
   }

   myMVA -= fBparm;
   return 1.0/(1.0 + TMath::Exp(-myMVA));
}

//_______________________________________________________________________
void TMVA::MethodSVM::PrepareDataToTrain()
{
   // puts all events in std::vectors 
   //normalized data
   Float_t weightavg = 0;
   for (Int_t ievt = 0; ievt < Data().GetNEvtTrain(); ievt++) {   

      ReadTrainingEvent( ievt );  
      weightavg += GetEventWeight();
      (*fNormVar)[ievt] = 0.;

      for (Int_t ivar = 0; ivar < GetNvar(); ivar++) {
            
         (*fVariables)[ivar][ievt] = GetEventVal(ivar);
         (*fNormVar)[ievt] +=  (*fVariables)[ivar][ievt]* (*fVariables)[ivar][ievt];
      }
      if (IsSignalEvent()) {
         (*fTypesVec)[ievt] = 1;
         (*fI)[ievt] = 1;
      }
      else {
         (*fTypesVec)[ievt] = -1;
         (*fI)[ievt] = -1;
      }
   } 
   for (Int_t ievt = 0; ievt < Data().GetNEvtTrain(); ievt++) {
      switch ( fKernelType ) {

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
   fC = ( fC * Data().GetNEvtTrain()) / weightavg;
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
         fLogger << kWARNING << "wrong Sigma value, uses default ::" << fDoubleSigmaSquered << Endl;
      }
      break;
     
   case kPolynomial:
      fKernelFunc = &MethodSVM::PolynomialKernel;
      if (fOrder < 2) {
         fOrder = 2;
         fLogger << kWARNING << "wrong polynomial order! Choose Order = "<< fOrder << Endl;
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

   ReadTrainingEvent( ievt );
   Float_t c_i = fC * GetEvent().GetWeight();
   
   ReadTrainingEvent( jevt );
   Float_t c_j = fC * GetEvent().GetWeight();  

   if (!wbug) cout << c_i <<"\t"<< c_j<<"\t\t ";

   // *************************************

   // compute l, h

   if (type_I == type_J) {
      Float_t gamma = alpha_I + alpha_J;
      
      if ( c_i > c_j ) {
         if ( gamma < c_j ) {
            l = 0;
            h = gamma;
         }
         else{
            h = c_j;
            if ( gamma < c_i )
               l = 0;
            else
               l = gamma - c_i;
         }
      }           
      else {
         if ( gamma < c_i ){
            l = 0;
            h = gamma;
         }
         else {
            l = gamma - c_i;
            if ( gamma < c_j )
               h = gamma;
            else
               h = c_j;
         }
      }
   }
   else {
      Float_t gamma = alpha_I - alpha_J;
      if (gamma > 0) {
         l = 0;
         if ( gamma >= (c_i - c_j) ) 
            h = c_i - gamma;
         else
            h = c_j;
      }
      else {
         l = -gamma;
         if ( (c_i - c_j) >= gamma)
            h = c_j;
         else 
            h = c_i - gamma;
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
      if (newAlpha_J > c_j) fLogger << kWARNING << "Unbound Alpha J!!" << Endl;
      newAlpha_I = 0;
   }
   else if (newAlpha_I > c_i) {
      Float_t temp = newAlpha_I - c_i;
      newAlpha_J += s * temp;
      newAlpha_I = c_i;
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
   for (Int_t i = fOrder; i > 0; i /= 2) {
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
   ReadTrainingEvent( ievt );
   Float_t c_temp = fC * GetEvent().GetWeight();

   if ((0<(*fAlphas)[ievt]) && ((*fAlphas)[ievt]) < c_temp ) 
      (*fI)[ievt]=0;                                          // I0
  
   if ((*fTypesVec)[ievt] == 1) {
      if      ((*fAlphas)[ievt] == 0)  
         (*fI)[ievt] =  1;                                    // I1
      else if ((*fAlphas)[ievt] == c_temp ) 
         (*fI)[ievt] = -1;                                    // I3
   }
  
   if ((*fTypesVec)[ievt] == -1) {
      if      ((*fAlphas)[ievt] == 0 )  
         (*fI)[ievt] = -1;                                    // I4
      else if ((*fAlphas)[ievt] == c_temp ) 
         (*fI)[ievt] =  1;                                    // I2
   }
}

//_______________________________________________________________
void TMVA::MethodSVM::Results()
{
   // results
   Int_t nvec = 0;
   for (Int_t i = 0; i < Data().GetNEvtTrain(); i++) if ((*fAlphas)[i] != 0) nvec++;

   fLogger << kINFO << "Results:" << Endl;
   fLogger << kINFO << "- number of support vectors: " << nvec 
           << " (" << 100*nvec/Data().GetNEvtTrain() << "%)" << Endl;
   fLogger << kINFO << "- b: " << fBparm << Endl;
}

//_______________________________________________________________
void TMVA::MethodSVM::StoreSupportVectors()
{
   UInt_t evtCounter = 0;

   for (Int_t ievt = 0; ievt < Data().GetNEvtTrain(); ievt++)
      if ((*fAlphas)[ievt] != 0) evtCounter++;

   fNsupv = evtCounter;

   fSupportVectors = new vector< Float_t* >( GetNvar()+1 );
   for (Int_t i = 0; i <= GetNvar(); i++) 
      (*fSupportVectors)[i] = new Float_t[fNsupv];

   evtCounter=0;
   for (Int_t ievt = 0; ievt < Data().GetNEvtTrain(); ievt++) {
      if ((*fAlphas)[ievt] != 0) {
         (*fSupportVectors)[0][evtCounter] = (Float_t)((*fAlphas)[ievt] * (*fTypesVec)[ievt]);
         for (Int_t ivar = 0; ivar < GetNvar(); ivar++) 
            (*fSupportVectors)[ivar+1][evtCounter] = (*fVariables)[ivar][ievt];
         evtCounter++;
      }
   }
   fLogger << kINFO << "All support vectors stored properly" << Endl;
}

//_______________________________________________________________________
void TMVA::MethodSVM::MakeClassSpecific( std::ostream& fout, const TString& className ) const
{
   // write specific classifier response
   fout << "   // not implemented for class: \"" << className << "\"" << endl;
   fout << "   float        fBparameter;" << endl;
   fout << "   int          fNOfSuppVec;" << endl;
   fout << "   static float fAllSuppVectors[][" << fNsupv << "];" << endl;
   fout << "   static float fAlphaTypeCoef[" << fNsupv << "];" << endl;
   fout << endl;
   fout << "   // Kernel parameter(s) " << endl;
   if (fTheKernel == "Gauss"     ) 
      fout << "   float fSigmaParm;" << endl;
   else if (fTheKernel == "Polynomial") {
      fout << "   float fThetaParm;" << endl;
      fout << "   int   fOrderParm;" << endl;
   }
   else if (fTheKernel == "Sigmoid"   ) {
      fout << "   float fThetaParm;" << endl;
      fout << "   float fKappaParm;" << endl;
   }
   fout << "};" << endl;
   fout << "" << endl;

   //Initialize function definition
   fout << "inline void " << className << "::Initialize() " << endl;
   fout << "{" << endl;
   fout << "   fBparameter = " << fBparm << ";" << endl;
   fout << "   fNOfSuppVec = " << fNsupv << ";" << endl;
   fout << "" << endl;

   fout << "   // Kernel parameter(s) " << endl;
   if (fTheKernel == "Gauss"     ) 
      fout << "   fSigmaParm  = " << -1./fDoubleSigmaSquered << ";" << endl;
   else if (fTheKernel == "Polynomial") {
      fout << "   fThetaParm  = " << fTheta << ";" << endl;
      fout << "   fOrderParm  = " << fOrder << ";" << endl;
   }
   else if (fTheKernel == "Sigmoid"   ) {
      fout << "   fThetaParm = " << fTheta << ";" << endl;
      fout << "   fKappaParm = " << fKappa << ";" << endl;
   }
   fout << "}" << endl;
   fout << endl;

   // GetMvaValue__ function defninition
   fout << "inline double " << className << "::GetMvaValue__(const std::vector<double>& inputValues ) const" << endl;
   fout << "{" << endl;
   fout << "   double mvaval = 0; " << endl;
   fout << "   double temp = 0; " << endl;
   fout << endl;
   fout << "   for (int ievt = 0; ievt < fNOfSuppVec; ievt++ ){" << endl;
   fout << "      temp = 0;" << endl;
   fout << "      for ( unsigned int ivar = 0; ivar < GetNvar(); ivar++ ) {" << endl;

   if (fTheKernel == "Gauss"     ) {
      fout << "         temp += (fAllSuppVectors[ivar][ievt] - inputValues[ivar])  " << endl;
      fout << "               * (fAllSuppVectors[ivar][ievt] - inputValues[ivar]); " << endl;
      fout << "      }" << endl;  
      fout << "      mvaval += fAlphaTypeCoef[ievt] * exp( fSigmaParm * temp ); " << endl;
   }   
   else if (fTheKernel == "Polynomial") {
      fout << "         temp += fAllSuppVectors[ivar][ievt] * inputValues[ivar]; " << endl;
      fout << "      }" << endl;  
      fout << "      temp += fThetaParm;" << endl;
      fout << "      double val_temp = 1; " << endl;
      fout << "      for (int i = fOrderParm; i > 0; i /= 2) {" << endl;
      fout << "         if (i%2) val_temp = temp;" << endl; 
      fout << "         temp *= temp;" << endl;
      fout << "      }" << endl;
      fout << "      mvaval += fAlphaTypeCoef[ievt] * val_temp; " << endl;
   }
   else if (fTheKernel == "Sigmoid"   ) {
      fout << "         temp += fAllSuppVectors[ivar][ievt] * inputValues[ivar]; " << endl;
      fout << "      }" << endl;
      fout << "      temp *= fKappaParm;" << endl;
      fout << "      temp += fThetaParm;" << endl;
      fout << "      mvaval += fAlphaTypeCoef[ievt] * tanh( temp );" << endl;
   }
   else{
      // for linear case
      fout << "         temp += fAllSuppVectors[ivar][ievt] * inputValues[ivar]; " << endl;
      fout << "      }" << endl;  
      fout << "      mvaval += fAlphaTypeCoef[ievt] * temp;" << endl;
   }

   fout << "   }" << endl;
   fout << "   mvaval -= fBparameter;" << endl;
   fout << "   return 1./(1. + exp( -mvaval));" << endl;
   fout << "}" << endl;
   fout << "// Clean up" << endl;
   fout << "inline void " << className << "::Clear() " << endl;
   fout << "{" << endl;
   fout << "   // nothing to clear " << endl;
   fout << "}" << endl;
   fout << "" << endl;   

   // define support vectors
   fout << "float " << className << "::fAlphaTypeCoef[] =" << endl;
   fout << "{ ";   
   for (Int_t isv = 0; isv < fNsupv; isv++) {
      fout << (*fSupportVectors)[0][isv];
      if (isv < fNsupv-1) fout << ", ";
   }
   fout << " };" << endl << endl;

   fout << "float " << className << "::fAllSuppVectors[][" << fNsupv << "] =" << endl;
   fout << "{";   
   for (Int_t ivar = 0; ivar < GetNvar(); ivar++) {
      fout << endl;
      fout << "   { ";
      for (Int_t isv = 0; isv < fNsupv; isv++){
         fout << (*fSupportVectors)[ivar+1][isv];
         if (isv < fNsupv-1) fout << ", ";
      }
      fout << " }";
      if (ivar < GetNvar()-1) fout << ", " << endl;
      else                    fout << endl;
   }   
   fout << "};" << endl;
}

//_______________________________________________________________________
void TMVA::MethodSVM::GetHelpMessage() const
{
   // get help message text
   //
   // typical length of text line: 
   //         "|--------------------------------------------------------------|"
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Short description:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "The Support Vector Machine (SVM) builds a hyperplance separating" << Endl;
   fLogger << "signal and background events (vectors) using the minimal subset of " << Endl;
   fLogger << "all vectors used for training (support vectors). The extension to" << Endl;
   fLogger << "the non-linear case is performed by mapping input vectors into a " << Endl;
   fLogger << "higher-dimensional feature space in which linear separation is " << Endl;
   fLogger << "possible. The use of the kernel functions thereby eliminates the " << Endl;
   fLogger << "explicit transformation to the feature space. The implemented SVM " << Endl;
   fLogger << "algorithm performs the classification tasks using linear, polynomial, " << Endl;
   fLogger << "Gaussian and sigmoidal kernel functions. The Gaussian kernel allows " << Endl;
   fLogger << "to apply any discriminant shape in the input space." << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance optimisation:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "SVM is a general purpose non-linear classification method, which " << Endl;
   fLogger << "does not require data preprocessing like decorrelation or Principal " << Endl;
   fLogger << "Component Analysis. It generalises quite well and can handle analyses " << Endl;
   fLogger << "with large numbers of input variables." << Endl;
   fLogger << Endl;
   fLogger << Tools::Color("bold") << "--- Performance tuning via configuration options:" << Tools::Color("reset") << Endl;
   fLogger << Endl;
   fLogger << "Optimal performance requires primarily a proper choice of the kernel " << Endl;
   fLogger << "parameters (the width \"Sigma\" in case of Gaussian kernel) and the" << Endl;
   fLogger << "cost parameter \"C\". The user must optimise them empirically by running" << Endl;
   fLogger << "SVM several times with different parameter sets. The time needed for " << Endl;
   fLogger << "each evaluation scales like the square of the number of training " << Endl;
   fLogger << "events so that a coarse preliminary tuning should be performed on " << Endl;
   fLogger << "reduced data sets." << Endl;
}
