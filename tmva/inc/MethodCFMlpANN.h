// @(#)root/tmva $Id: MethodCFMlpANN.h,v 1.2 2006/05/23 13:03:15 brun Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodCFMlpANN                                                        *
 *                                                                                *
 * Description:                                                                   *
 *      Interface for Clermond-Ferrand artificial neural network.                 *
 *      The ANN code has been translated from FORTRAN77 (f2c);                    *
 *      see files: MethodCFMlpANN_f2c_mlpl3.cpp                                   *
 *                 MethodCFMlpANN_f2c_datacc.cpp                                  *
 *                                                                                *
 *      --------------------------------------------------------------------      *
 *      Reference for the original FORTRAN version:                               *
 *           Authors  : J. Proriol and contributions from ALEPH-Clermont-Fd       *
 *                      Team members. Contact : gaypas@afal11.cern.ch             *
 *                                                                                *
 *           Copyright: Laboratoire Physique Corpusculaire                        *
 *                      Universite de Blaise Pascal, IN2P3/CNRS                   *
 *      --------------------------------------------------------------------      *
 *                                                                                *
 * Usage: options are given through Factory:                                      *
 *            factory->BookMethod( "MethodFisher", OptionsString );               *
 *                                                                                *
 *        where:                                                                  *
 *            TString OptionsString = "n_training_cycles:n_hidden_layers"         *
 *                                                                                *
 *        default is:  n_training_cycles = 5000, n_layers = 4                     *
 *        note that the number of hidden layers in the NN is                      *
 *                                                                                *
 *            n_hidden_layers = n_layers - 2                                      *
 *                                                                                *
 *        since there is one input and one output layer. The number of            *
 *        nodes (neurons) is predefined to be                                     *
 *                                                                                *
 *           n_nodes[i] = nvars + 1 - i (where i=1..n_layers)                     *
 *                                                                                *
 *        with nvars being the number of variables used in the NN.                *
 *        Hence, the default case is: n_neurons(layer 1 (input)) : nvars          *
 *                                    n_neurons(layer 2 (hidden)): nvars-1        *
 *                                    n_neurons(layer 3 (hidden)): nvars-1        *
 *                                    n_neurons(layer 4 (out))   : 2              *
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
 * $Id: MethodCFMlpANN.h,v 1.2 2006/05/23 13:03:15 brun Exp $    
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodCFMlpANN
#define ROOT_TMVA_MethodCFMlpANN

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodCFMlpANN                                                       //
//                                                                      //
// Interface for Clermond-Ferrand artificial neural network             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef ROOT_TMVA_MethodANNBase
#include "TMVA/MethodANNBase.h"
#endif
#ifndef ROOT_TMVA_MethodCFMlpANN_Utils
#include "TMVA/MethodCFMlpANN_Utils.h"
#endif
#ifndef ROOT_TMVA_TMatrix
#include "TMatrix.h"
#endif

namespace TMVA {

   class MethodCFMlpANN : public MethodBase, MethodANNBase, MethodCFMlpANN_Utils {

   public:

      MethodCFMlpANN( TString jobName,
                      vector<TString>* theVariables, 
                      TTree* theTree = 0, 
                      TString theOption = "3000:N-1:N-2",
                      TDirectory* theTargetDir = 0 );

      MethodCFMlpANN( vector<TString> *theVariables, 
                      TString theWeightFile,  
                      TDirectory* theTargetDir = NULL );

      virtual ~MethodCFMlpANN( void );
    
      // training method
      virtual void Train( void );

      // write weights to file
      virtual void WriteWeightsToFile( void );
  
      // read weights from file
      virtual void ReadWeightsFromFile( void );

      // calculate the MVA value
      virtual Double_t GetMvaValue( Event *e );

      // write method specific histos to target file
      virtual void WriteHistosToFile( void ) ;

      // data accessors for external functions
      Double_t GetData ( Int_t isel, Int_t ivar ) const { return (*fData)(isel, ivar); }
      Int_t    GetClass( Int_t ivar             ) const { return (*fClass)[ivar];      }

      // static pointer to this object (required for external functions
      static MethodCFMlpANN* This( void ) { return fgThis; }  

   protected:

      Int_t DataInterface( Double_t*, Double_t*, Int_t*, Int_t*, Int_t*, Int_t*,
                           Double_t*, Int_t*, Int_t* );
  
      void WriteNNWeightsToFile( Int_t, Int_t, Double_t*, Double_t*, 
                                 Int_t, Int_t*, Double_t*, Double_t*, Double_t* );

   private:

      // this carrier
      static MethodCFMlpANN* fgThis;

      // LUTs
      TMatrix       *fData ;   // the (data,var) string
      vector<Int_t> *fClass;   // the event class (1=signal, 2=background)

      Int_t         fNevt;     // number of training events
      Int_t         fNsig;     // number of signal events
      Int_t         fNbgd;     // number of background

      Int_t         fNlayers;  // number of layers (including input and output layers)
      Int_t         fNcycles;  // number of training cycles
      Int_t*        fNodes;    // number of nodes per layer

      // additional member variables for the independent NN::Evaluation phase
      Double_t*     fXmaxNN;   // maximum values of input variables
      Double_t*     fXminNN;   // minimum values of input variables
      Int_t         fLayermNN; // number of layers (including input and output layers)
      Int_t*        fNeuronNN; // nodes per layer
      Double_t***   fWNN;      // weights
      Double_t**    fWwNN;     // weights
      Double_t**    fYNN;      // weights
      Double_t*     fTempNN;   // temperature (used in activation function)

      // auxiliary member functions
      Double_t EvalANN( vector<Double_t>*, Bool_t& isOK );
      void     NN_ava ( Double_t* );
      Double_t NN_fonc( Int_t, Double_t ) const;

      // default initialisation 
      void InitCFMlpANN( void );

      ClassDef(MethodCFMlpANN,0)  // Interface for Clermond-Ferrand artificial neural network
         };

} // namespace TMVA

#endif
