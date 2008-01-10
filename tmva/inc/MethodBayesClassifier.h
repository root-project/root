// @(#)root/tmva $Id$    
// Author: Abhishek Narain

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodBayesClassifier                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Bayesian Classifier                                                       *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Abhishek Narain, <narainabhi@gmail.com> - University of Houston           *
 *                                                                                *
 * Copyright (c) 2005-2006:                                                       *
 *      University of Houston,                                                    *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodBayesClassifier
#define ROOT_TMVA_MethodBayesClassifier

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodBayesClassifier                                                //
//                                                                      //
// Description...                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif

namespace TMVA {

   class MethodBayesClassifier : public MethodBase {

   public:

      MethodBayesClassifier( const TString& jobName, 
                             const TString& methodTitle, 
                             DataSet& theData,
                             const TString& theOption = "",
                             TDirectory* theTargetDir = 0 );
      
      MethodBayesClassifier( DataSet& theData, 
                             const TString& theWeightFile,  
                             TDirectory* theTargetDir = NULL );
      
      virtual ~MethodBayesClassifier( void );
    
      // training method
      void Train( void );

      using MethodBase::WriteWeightsToStream;
      using MethodBase::ReadWeightsFromStream;

      // write weights to file
      void WriteWeightsToStream( ostream& o ) const;

      // read weights from file
      void ReadWeightsFromStream( istream& istr );

      // calculate the MVA value
      Double_t GetMvaValue();

      void InitBayesClassifier( void );

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
      void ProcessOptions();

      ClassDef(MethodBayesClassifier,0)  // Friedman's BayesClassifier method 
   };

} // namespace TMVA

#endif // MethodBayesClassifier_H
