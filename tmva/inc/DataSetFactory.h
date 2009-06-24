// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : DataSetFactory                                                        *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Contains all the data information                                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2006:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_DataSetFactory
#define ROOT_TMVA_DataSetFactory

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// DataSetFactory                                                       //
//                                                                      //
// Class that contains all the data information                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include <stdlib.h>

#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TTree
#include "TTree.h"
#endif
#ifndef ROOT_TCut
#include "TCut.h"
#endif
#ifndef ROOT_TTreeFormula
#include "TTreeFormula.h"
#endif
#ifndef ROOT_TMatrixDfwd
#include "TMatrixDfwd.h"
#endif
#ifndef ROOT_TPrincipal
#include "TPrincipal.h"
#endif

#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_VariableInfo
#include "TMVA/VariableInfo.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif

namespace TMVA {
   
   class DataSet;
   class DataSetInfo;
   class DataInputHandler;
   class TreeInfo;
   class MsgLogger;

   class DataSetFactory {

   public:

      // singleton class
      static DataSetFactory& Instance() { if (!fgInstance) fgInstance = new DataSetFactory(); return *fgInstance; } 
      static void destroyInstance() { if (fgInstance) { delete fgInstance; fgInstance=0; } }

      DataSet* CreateDataSet( DataSetInfo &, DataInputHandler& );

   private:

      ~DataSetFactory();
      
      DataSetFactory();
      static DataSetFactory *fgInstance;

      DataSet*  BuildInitialDataSet( DataSetInfo&, TMVA::DataInputHandler& );
      DataSet*  BuildDynamicDataSet( DataSetInfo& );
      void      BuildEventVector   ( DataSetInfo& dsi, 
                                     DataInputHandler& dataInput, 
                                     std::vector< std::vector< Event* > >& tmpEventVector, 
                                     std::vector<Double_t>& sumOfWeights, 
                                     std::vector<Double_t>& nTempEvents, 
                                     std::vector<Double_t>& renormFactor,
				     std::vector< std::vector< std::pair< Long64_t, Types::ETreeType > > >& userDefinedEventTypes );
      
      DataSet*  MixEvents          ( DataSetInfo& dsi, 
                                     std::vector< std::vector< Event* > >& tmpEventVector, 
                                     std::vector< std::pair< Int_t, Int_t > >& nTrainTestEvents, 
                                     const TString& splitMode, UInt_t splitSeed, 
                                     std::vector<Double_t>& renormFactor,
				     std::vector< std::vector< std::pair< Long64_t, Types::ETreeType > > >& userDefinedEventTypes );
      
      void      InitOptions        ( DataSetInfo& dsi, 
                                     std::vector< std::pair< Int_t, Int_t > >& nTrainTestEvents, 
                                     TString& normMode, UInt_t& splitSeed, TString& splitMode );
      
      // auxiliary functions to compute correlations
      TMatrixD* CalcCorrelationMatrix( DataSet*, const UInt_t classNumber );
      TMatrixD* CalcCovarianceMatrix ( DataSet*, const UInt_t classNumber );
      void      CalcMinMax           ( DataSet*, DataSetInfo& dsi );

      // resets branch addresses to current event
      void ResetBranchAndEventAddresses( TTree* );
      void ResetCurrentTree() { fCurrentTree = 0; }
      void ChangeToNewTree( TreeInfo&, const DataSetInfo & );

      // verbosity
      Bool_t Verbose() { return fVerbose; }

      // data members

      // verbosity
      Bool_t                     fVerbose;           //! Verbosity
      TString                    fVerboseLevel;      //! VerboseLevel

      // the event 
      mutable TTree*             fCurrentTree;       //! the tree, events are currently read from
      mutable UInt_t             fCurrentEvtIdx;     //! the current event (to avoid reading of the same event)

      // the formulas for reading the original tree
      std::vector<TTreeFormula*> fInputFormulas;   //! input variables
      std::vector<TTreeFormula*> fTargetFormulas;  //! targets
      std::vector<TTreeFormula*> fCutFormulas;     //! cuts
      std::vector<TTreeFormula*> fWeightFormula;   //! weights
      std::vector<TTreeFormula*> fSpectatorFormulas; //! spectators

      mutable MsgLogger*         fLogger;          //! message logger
      MsgLogger& Log() const { return *fLogger; }
   };
}

#endif
