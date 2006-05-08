// @(#)root/tmva $Id: TMVA_MethodCFMlpANN.h,v 1.2 2006/05/08 21:33:46 brun Exp $    
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodCFMlpANN                                                   *
 *                                                                                *
 * Description:                                                                   *
 *      Interface for Clermond-Ferrand artificial neural network.                 *
 *      The ANN code has been translated from FORTRAN77 (f2c);			  *
 *      see files: TMVA_MethodCFMlpANN_f2c_mlpl3.cpp 				  *
 *                 TMVA_MethodCFMlpANN_f2c_datacc.cpp				  *
 *                                                                                *
 *      --------------------------------------------------------------------	  *
 *      Reference for the original FORTRAN version:				  *
 *           Authors  : J. Proriol and contributions from ALEPH-Clermont-Fd 	  *
 *                      Team members. Contact : gaypas@afal11.cern.ch   	  *
 *										  *
 *           Copyright: Laboratoire Physique Corpusculaire 			  *
 *                      Universite de Blaise Pascal, IN2P3/CNRS			  *
 *      --------------------------------------------------------------------	  *
 *                                                                                *
 * Usage: options are given through TMVA_Factory: 				  *
 *            factory->BookMethod( "MethodFisher", OptionsString );		  *
 *										  *
 *        where: 								  *
 *            TString OptionsString = "n_training_cycles:n_hidden_layers"	  *
 *										  *
 *        default is:  n_training_cycles = 5000, n_layers = 4			  *
 *        note that the number of hidden layers in the NN is 			  *
 *										  *
 *            n_hidden_layers = n_layers - 2					  *
 *										  *
 *        since there is one input and one output layer. The number of 		  *
 *        nodes (neurons) is predefined to be 					  *
 *										  *
 *           n_nodes[i] = nvars + 1 - i (where i=1..n_layers)			  *
 *										  *
 *        with nvars being the number of variables used in the NN. 		  *
 *        Hence, the default case is: n_neurons(layer 1 (input)) : nvars	  *
 *                                    n_neurons(layer 2 (hidden)): nvars-1	  *
 *                                    n_neurons(layer 3 (hidden)): nvars-1	  *
 *                                    n_neurons(layer 4 (out))   : 2		  *
 *                                                                                *
 *        This artificial neural network usually needs a relatively large         *
 *        number of cycles to converge (8000 and more). Overtraining can          *
 *        be efficienctly tested by comparing the signal and background           *
 *        output of the NN for the events that were used for training and         *
 *        an independent data sample (with equal properties). If the separation   *
 *        performance is significantly better for the training sample, the        *
 *        NN interprets statistical effects, and is hence overtrained. In         * 
 *        this case, the number of cycles should be reduced, or the size          *
 *        of the training sample increased.                                       *
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
 * $Id: TMVA_MethodCFMlpANN.h,v 1.2 2006/05/08 21:33:46 brun Exp $    
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodCFMlpANN
#define ROOT_TMVA_MethodCFMlpANN

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_MethodCFMlpANN                                                  //
//                                                                      //
// Interface for Clermond-Ferrand artificial neural network             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA_MethodBase.h"
#endif
#ifndef ROOT_TMVA_MethodANNBase
#include "TMVA_MethodANNBase.h"
#endif
#ifndef ROOT_TMatrix
#include "TMatrix.h"
#endif

// forward definitions, needed for rootcint
extern "C" int TMVA_MethodCFMlpANN_dataInterface( Double_t*, Double_t*, 
						  Int_t*, Int_t*, Int_t*, Int_t*,
						  Double_t*, Int_t*, Int_t* );

extern "C" void TMVA_MethodCFMlpANN_writeWeightsToFile( Int_t, Int_t, Double_t*, Double_t*, 
							Int_t, Int_t*, 
							Double_t*, Double_t*, Double_t* );

class TMVA_MethodCFMlpANN : public TMVA_MethodBase, TMVA_MethodANNBase {

 public:

  TMVA_MethodCFMlpANN( TString jobName,
		   vector<TString>* theVariables, 
		   TTree* theTree = 0, 
		   TString theOption = "3000:N-1:N-2",
		   TDirectory* theTargetDir = 0 );

  TMVA_MethodCFMlpANN( vector<TString> *theVariables, 
		       TString theWeightFile,  
		       TDirectory* theTargetDir = NULL );

  virtual ~TMVA_MethodCFMlpANN( void );
    
  // training method
  virtual void Train( void );

  // write weights to file
  virtual void WriteWeightsToFile( void );
  
  // read weights from file
  virtual void ReadWeightsFromFile( void );

  // calculate the MVA value
  virtual Double_t GetMvaValue( TMVA_Event *e );

  // write method specific histos to target file
  virtual void WriteHistosToFile( void ) ;

  // data accessors for external functions
  Double_t GetData ( Int_t isel, Int_t ivar ) const { return (*fData)(isel, ivar); }
  Int_t    GetClass( Int_t ivar             ) const { return (*fClass)[ivar];      }

  // static pointer to this object (required for external functions
  static TMVA_MethodCFMlpANN* This( void ) { return fThis; }  

 protected:

 private:

  // this carrier
  static TMVA_MethodCFMlpANN* fThis;

  // LUTs
  TMatrix       *fData ; // the (data,var) string
  vector<Int_t> *fClass; // the event class (1=signal, 2=background)

  // number of events (tot, signal, background)
  Int_t         fNevt;
  Int_t         fNsig;
  Int_t         fNbgd;

  Int_t         fNlayers;
  Int_t         fNcycles;
  Int_t*        fNodes;

  // additional member variables for the independent NN::Evaluation phase
  Double_t*     fXmaxNN;   
  Double_t*     fXminNN;   
  Int_t         fLayermNN;
  Int_t*        fNeuronNN;
  Double_t***   fWNN;    
  Double_t**    fWwNN;  
  Double_t**    fYNN;   
  Double_t*     fTempNN;

  // auxiliary member functions
  Double_t evalANN( vector<Double_t>*, Bool_t& isOK );
  void     nn_ava ( Double_t* );
  Double_t nn_fonc( Int_t, Double_t ) const;

  void InitCFMlpANN( void );

  ClassDef(TMVA_MethodCFMlpANN,0)  //Interface for Clermond-Ferrand artificial neural network
};

#endif
