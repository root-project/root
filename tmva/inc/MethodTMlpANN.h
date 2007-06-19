// @(#)root/tmva $Id: MethodTMlpANN.h,v 1.11 2007/04/19 06:53:01 brun Exp $ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodTMlpANN                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
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


#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif

class TMultiLayerPerceptron;

namespace TMVA {

   class MethodTMlpANN : public MethodBase {
  
   public:

      MethodTMlpANN( TString jobName, 
                     TString methodTitle, 
                     DataSet& theData,
                     TString theOption = "3000:N-1:N-2", 
                     TDirectory* theTargetDir = 0 );

      MethodTMlpANN( DataSet& theData, 
                     TString theWeightFile,  
                     TDirectory* theTargetDir = NULL );

      virtual ~MethodTMlpANN( void );
    
      // training method
      virtual void Train( void );

      using MethodBase::WriteWeightsToStream;
      using MethodBase::ReadWeightsFromStream;

      // write weights to file
      virtual void WriteWeightsToStream( ostream& o ) const;

      // read weights from file
      virtual void ReadWeightsFromStream( istream& istr );

      // calculate the MVA value ...
      // - here it is just a dummy, as it is done in the overwritten
      // - PrepareEvaluationtree... ugly but necessary due to the strucure 
      //   of TMultiLayerPercepton in ROOT grr... :-(
      virtual Double_t GetMvaValue();

      void SetHiddenLayer(TString hiddenlayer = "" ) { fHiddenLayer=hiddenlayer; }

      // ranking of input variables
      const Ranking* CreateRanking() { return 0; }

   protected:

      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      virtual void MakeClassSpecific( std::ostream&, const TString& ) const;

      // get help message text
      void GetHelpMessage() const;

   private:

      // the option handling methods
      virtual void DeclareOptions();
      virtual void ProcessOptions();

      void CreateMLPOptions( TString );

      // option string
      TString fLayerSpec;        // Layer specification option

      TMultiLayerPerceptron* fMLP; // the TMLP
     
      TString fHiddenLayer;      // string containig the hidden layer structure
      Int_t   fNcycles;          // number of training cylcles
      TString fMLPBuildOptions;  // option string to build the mlp

      void InitTMlpANN( void );

      ClassDef(MethodTMlpANN,0) // Implementation of interface for TMultiLayerPerceptron
   };

} // namespace TMVA

#endif
