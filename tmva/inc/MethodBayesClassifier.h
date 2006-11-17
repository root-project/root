// @(#)root/tmva $Id: MethodBayesClassifier.h,v 1.3 2006/11/16 22:51:58 helgevoss Exp $    
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
 *      CERN, Switzerland,                                                        *
 *      U. of Victoria, Canada,                                                   *
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

      MethodBayesClassifier( TString jobName, 
                             TString methodTitle, 
                             DataSet& theData,
                             TString theOption = "",
                             TDirectory* theTargetDir = 0 );
      
      MethodBayesClassifier( DataSet& theData, 
                             TString theWeightFile,  
                             TDirectory* theTargetDir = NULL );
      
      virtual ~MethodBayesClassifier( void );
    
      // training method
      virtual void Train( void );

      // write weights to file
      virtual void WriteWeightsToStream( ostream& o ) const;

      // read weights from file
      virtual void ReadWeightsFromStream( istream& istr );

      // calculate the MVA value
      virtual Double_t GetMvaValue();

      void InitBayesClassifier( void );

      // ranking of input variables
      const Ranking* CreateRanking() { return 0; }

   private:

      // the option handling methods
      virtual void DeclareOptions();
      virtual void ProcessOptions();

      ClassDef(MethodBayesClassifier,0)  // Friedman's BayesClassifier method 
         ;
   };

} // namespace TMVA

#endif // MethodBayesClassifier_H
