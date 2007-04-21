// @(#)root/tmva $Id: MethodSVM.h,v 1.7 2007/04/19 06:53:01 brun Exp $    
// Author: Marcin Wolter, Andrzej Zemla

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodSVM                                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Support Vector Machine                                                    *
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

#ifndef ROOT_TMVA_MethodSVM
#define ROOT_TMVA_MethodSVM

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodSVM                                                            //
//                                                                      //
// SMO Platt's SVM classifier with Keerthi & Shavade improvements       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_TMatrixD
#include "TMatrixD.h"
#endif
#ifndef ROOT_TMVA_TVectorD
#include "TVectorD.h"
#endif

namespace TMVA {

   class MethodSVM : public MethodBase {

   public:

      MethodSVM( TString jobName, 
                 TString methodTitle, 
                 DataSet& theData,
                 TString theOption = "",
                 TDirectory* theTargetDir = 0 );
      
      MethodSVM( DataSet& theData, 
                 TString theWeightFile,  
                 TDirectory* theTargetDir = NULL );

      virtual ~MethodSVM( void );
    
      // training method
      virtual void Train( void );

      using MethodBase::WriteWeightsToStream;
      using MethodBase::ReadWeightsFromStream;

      // write weights to file
      virtual void WriteWeightsToStream( std::ostream& o ) const;
      virtual void WriteWeightsToStream( TFile& fout ) const;

      // read weights from file
      virtual void ReadWeightsFromStream( std::istream& istr );
      virtual void ReadWeightsFromStream( TFile& fFin );

      // calculate the MVA value
      virtual Double_t GetMvaValue();

      void InitSVM( void );

      // ranking of input variables
      const Ranking* CreateRanking() { return 0; } 

      enum EKernelType { kLinear , kRBF, kPolynomial, kSigmoidal };

   private:

      // the option handling methods
      virtual void DeclareOptions();
      virtual void ProcessOptions();

      TString     fTheKernel;           // kernel name

      EKernelType fKernelType;          // to be defined
      Float_t     fC;                   // to be defined 
      Float_t     fTolerance;           // treshold parameter
      Int_t       fMaxIter;             // max number of training loops
      
      // Kernel parameters
      Float_t     fDoubleSigmaSquered;  // for RBF Kernel
      Int_t       fOrder;               // for Polynomial Kernel ( polynomial order )
      Float_t     fTheta;               // for Sigmoidal Kernel
      Float_t     fKappa;               // for Sigmoidal Kernel
      
      Float_t     fBparm;               // to be defined
      Float_t     fB_up;                // to be defined
      Float_t     fB_low;               // to be defined
      Int_t       fI_up;                // to be defined
      Int_t       fI_low;               // to be defined
      Int_t       fNsupv;               // to be defined

      Int_t   ExamineExample( Int_t  );
      Int_t   TakeStep( Int_t , Int_t );
      
      Float_t LearnFunc( Int_t );
      Float_t (MethodSVM::*fKernelFunc)( Int_t, Int_t ) const;
   
      // kernel functions
      Float_t LinearKernel    ( Int_t, Int_t ) const;
      Float_t RBFKernel       ( Int_t, Int_t ) const;         
      Float_t PolynomialKernel( Int_t, Int_t ) const;
      Float_t SigmoidalKernel ( Int_t, Int_t ) const; 
     
      vector< Float_t >*  fAlphas;       // to be defined
      vector< Float_t >*  fErrorCache;   // to be defined
      vector< Float_t >*  fWeightVector; // weight vector for linear SVM
      vector< Float_t* >* fVariables;    // data vectors
      vector< Float_t >*  fNormVar;      // norm
      vector< Int_t >*    fTypesVec;     // type vector
      vector< Short_t >*  fI;            // to be defined
      vector < Float_t >* fKernelDiag;   // to be defined

      TVectorD* fMaxVars;                // to be defined
      TVectorD* fMinVars;                // to be defined

      void SetIndex( Int_t );
      void PrepareDataToTrain();
      void SetKernel();
      void Results();

      ClassDef(MethodSVM,0)  // Support Vector Machine
   };

} // namespace TMVA

#endif // MethodSVM_H
