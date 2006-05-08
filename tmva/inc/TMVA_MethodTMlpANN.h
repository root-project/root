// @(#)root/tmva $Id: TMVA_MethodTMlpANN.h,v 1.8 2006/05/02 23:27:40 helgevoss Exp $ 
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodTMlpANN                                                    *
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
 * $Id: TMVA_MethodTMlpANN.h,v 1.8 2006/05/02 23:27:40 helgevoss Exp $ 
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodTMlpANN
#define ROOT_TMVA_MethodTMlpANN

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_MethodTMlpANN                                                   //
//                                                                      //
// Implementation of interface for Root-integrated artificial neural    //
// network: TMultiLayerPerceptron                                       //  
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodANNBase
#include "TMVA_MethodANNBase.h"
#endif
#ifndef ROOT_TMVA_MethodBase
#include "TMVA_MethodBase.h"
#endif

class TMVA_MethodTMlpANN : public TMVA_MethodBase, TMVA_MethodANNBase {
  
 public:
  
  TMVA_MethodTMlpANN( TString jobName, 
		     vector<TString>* theVariables,  
		     TTree* theTree = 0, 
		     TString theOption = "3000:N-1:N-2", 
		     TDirectory* theTargetDir = 0 );

  TMVA_MethodTMlpANN( vector<TString> *theVariables, 
		      TString theWeightFile,  
		      TDirectory* theTargetDir = NULL );

  virtual ~TMVA_MethodTMlpANN( void );
    
  // training method
  virtual void Train( void );

  // write weights to file
  virtual void WriteWeightsToFile( void );
  
  // read weights from file
  virtual void ReadWeightsFromFile( void );

  // evaluate method
  virtual void PrepareEvaluationTree( TTree* testTree );

  // calculate the MVA value !!!! here it is just a dummy, as it is done in the overwritten
  // PrepareEvaluationtree... ugly but necessary due to the strucure of TMultiLayerPercepton 
  // in ROOT grr...
  virtual Double_t GetMvaValue( TMVA_Event * /*e*/ ) { return 0; }

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

 ClassDef(TMVA_MethodTMlpANN,0) //Implementation of interface for TMultiLayerPerceptron
};

#endif
