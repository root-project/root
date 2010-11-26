// @(#)root/tmva $Id$    
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
 *      Andrzej Zemla  <azemla@cern.ch>         - IFJ PAN, Krakow, Poland         *
 *      (IFJ PAN: Henryk Niewodniczanski Inst. Nucl. Physics, Krakow, Poland)     *   
 *                                                                                *
 * Introduction of regression by:                                                 *
 *      Krzysztof Danielowski <danielow@cern.ch> - IFJ PAN & AGH, Krakow, Poland  *
 *      Kamil Kraszewski      <kalq@cern.ch>     - IFJ PAN & UJ, Krakow, Poland   *
 *      Maciej Kruk           <mkruk@cern.ch>    - IFJ PAN & AGH, Krakow, Poland  *
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
#ifndef ROOT_TMatrixDfwd
#include "TMatrixDfwd.h"
#endif
#endif
#ifndef ROOT_TMVA_TVectorD
#ifndef ROOT_TVectorD
#include "TVectorD.h"
#endif
#endif

namespace TMVA 
{
   class SVWorkingSet;
   class SVEvent;
   class SVKernelFunction;
   
   class MethodSVM : public MethodBase {

   public:

      MethodSVM( const TString& jobName, const TString& methodTitle, DataSetInfo& theData,
                 const TString& theOption = "", TDirectory* theTargetDir = 0 );
      
      MethodSVM( DataSetInfo& theData, const TString& theWeightFile, TDirectory* theTargetDir = NULL );

      virtual ~MethodSVM( void );
    
      virtual Bool_t HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets );

      // training method
      void Train( void );

      using MethodBase::ReadWeightsFromStream;

      // write weights to file
      void WriteWeightsToStream( TFile& fout   ) const;
      void AddWeightsXMLTo     ( void*  parent ) const;

      // read weights from file
      void ReadWeightsFromStream( std::istream& istr );
      void ReadWeightsFromStream( TFile& fFin     );
      void ReadWeightsFromXML   ( void*  wghtnode );
      // calculate the MVA value

      Double_t GetMvaValue( Double_t* err = 0, Double_t* errUpper = 0 );
      const std::vector<Float_t>& GetRegressionValues();
      
      void Init( void );

      // ranking of input variables
      const Ranking* CreateRanking() { return 0; } 

   protected:

      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      void MakeClassSpecific( std::ostream&, const TString& ) const;

      // get help message text
      void GetHelpMessage() const;

   private:

      // the option handling methods
      void DeclareOptions();
      void DeclareCompatibilityOptions();
      void ProcessOptions();
      
      Float_t                 fCost; // cost value
      Float_t                 fTolerance;       // tolerance parameter
      UInt_t                  fMaxIter; // max number of iteration
      UShort_t                fNSubSets; // nr of subsets, default 1
      Float_t                 fBparm;           // free plane coefficient 
      Float_t                 fGamma;           // RBF Kernel parameter
      SVWorkingSet*           fWgSet;           // svm working set 
      std::vector<TMVA::SVEvent*>*  fInputData;       // vector of training data in SVM format
      std::vector<TMVA::SVEvent*>*  fSupportVectors; // contains support vectors
      SVKernelFunction*       fSVKernelFunction;  // kernel function

      TVectorD*               fMinVars;         // for normalization //is it still needed?? 
      TVectorD*               fMaxVars;         // for normalization //is it still needed?? 

      // for backward compatibility
      TString     fTheKernel;           // kernel name
      Float_t     fDoubleSigmaSquared;  // for RBF Kernel
      Int_t       fOrder;               // for Polynomial Kernel ( polynomial order )
      Float_t     fTheta;               // for Sigmoidal Kernel
      Float_t     fKappa;               // for Sigmoidal Kernel
      
      ClassDef(MethodSVM,0)  // Support Vector Machine
   };

} // namespace TMVA

#endif // MethodSVM_H
