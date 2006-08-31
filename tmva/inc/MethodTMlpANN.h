// @(#)root/tmva $Id: MethodTMlpANN.h,v 1.2 2006/05/23 13:03:15 brun Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodTMlpANN                                                         *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation of interface for Root-integrated artificial neural         *
 *      network: TMultiLayerPerceptron, author: Christophe.Delaere@cern.ch        *
 *      for a manual, see                                                         *
 *      http://root.cern.ch/root/html/TMultiLayerPerceptron.html                  *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        *
 *      U. of Victoria, Canada,                                                   *
 *      MPI-KP Heidelberg, Germany,                                               *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: MethodTMlpANN.h,v 1.2 2006/05/23 13:03:15 brun Exp $
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodTMlpANN
#define ROOT_TMVA_MethodTMlpANN

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodTMlpANN                                                        //
//                                                                      //
// Implementation of interface for Root-integrated artificial neural    //
// network: TMultiLayerPerceptron                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodANNBase
#include "TMVA/MethodANNBase.h"
#endif
#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif

namespace TMVA {

   class MethodTMlpANN : public MethodBase, MethodANNBase {

   public:

      MethodTMlpANN( TString jobName,
                     std::vector<TString>* theVariables,
                     TTree* theTree = 0,
                     TString theOption = "3000:N-1:N-2",
                     TDirectory* theTargetDir = 0 );

      MethodTMlpANN( std::vector<TString> *theVariables,
                     TString theWeightFile,
                     TDirectory* theTargetDir = NULL );

      virtual ~MethodTMlpANN( void );

      // training method
      virtual void Train( void );

      // write weights to file
      virtual void WriteWeightsToFile( void );

      // read weights from file
      virtual void ReadWeightsFromFile( void );

      // evaluate method
      virtual void PrepareEvaluationTree( TTree* testTree );

      // calculate the MVA value ...
      // - here it is just a dummy, as it is done in the overwritten
      // - PrepareEvaluationtree... ugly but necessary due to the strucure
      //   of TMultiLayerPercepton in ROOT grr... :-(
      virtual Double_t GetMvaValue( Event * /*e*/ ) { return 0; }

      // write method specific histos to target file
      virtual void WriteHistosToFile( void );

      void SetTestTree( TTree* testTree );

      void SetHiddenLayer(TString hiddenlayer = "" ) { fHiddenLayer=hiddenlayer; }

   protected:

   private:

      void CreateMLPOptions( void );

      TString fHiddenLayer; // string containig the hidden layer structure
      Int_t   fNcycles;     // number of training cylcles
      TTree*  fTestTree;    // TestTree

      void InitTMlpANN( void );

      ClassDef(MethodTMlpANN,0) // Implementation of interface for TMultiLayerPerceptron
   };

} // namespace TMVA

#endif
