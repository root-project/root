// // @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : DataSetInfo                                                           *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Contains all the data information                                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer <speckmay@mail.cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - DESY, Germany                  *
 *                                                                                *
 * Copyright (c) 2008:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *      DESY Hamburg, Germany                                                     *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_DataSetInfo
#define ROOT_TMVA_DataSetInfo

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// DataSetInfo                                                          //
//                                                                      //
// Class that contains all the data information                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iosfwd>

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TTree
#include "TTree.h"
#endif
#ifndef ROOT_TCut
#include "TCut.h"
#endif
#ifndef ROOT_TMatrixDfwd
#include "TMatrixDfwd.h"
#endif

#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_VariableInfo
#include "TMVA/VariableInfo.h"
#endif
#ifndef ROOT_TMVA_ClassInfo
#include "TMVA/ClassInfo.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif

class TH2;

namespace TMVA {

   class DataSet;
   class VariableTransformBase;
   class MsgLogger;
   class DataSetManager;

   class DataSetInfo : public TObject {

   public:

      DataSetInfo(const TString& name = "Default");
      virtual ~DataSetInfo();

      virtual const char* GetName() const { return fName.Data(); }

      // the data set
      void        ClearDataSet() const;
      DataSet*    GetDataSet() const;

      // ---
      // the variable data
      // ---
      VariableInfo&     AddVariable( const TString& expression, const TString& title = "", const TString& unit = "", 
                                     Double_t min = 0, Double_t max = 0, char varType='F', 
                                     Bool_t normalized = kTRUE, void* external = 0 );
      VariableInfo&     AddVariable( const VariableInfo& varInfo );

      VariableInfo&     AddTarget  ( const TString& expression, const TString& title, const TString& unit, 
                                     Double_t min, Double_t max, Bool_t normalized = kTRUE, void* external = 0 );
      VariableInfo&     AddTarget  ( const VariableInfo& varInfo );

      VariableInfo&     AddSpectator ( const TString& expression, const TString& title, const TString& unit, 
                                       Double_t min, Double_t max, char type = 'F', Bool_t normalized = kTRUE, void* external = 0 );
      VariableInfo&     AddSpectator ( const VariableInfo& varInfo );

      ClassInfo*        AddClass   ( const TString& className );

      // accessors

      // general
      std::vector<VariableInfo>&       GetVariableInfos()         { return fVariables; }
      const std::vector<VariableInfo>& GetVariableInfos() const   { return fVariables; }
      VariableInfo&                    GetVariableInfo( Int_t i ) { return fVariables.at(i); }
      const VariableInfo&              GetVariableInfo( Int_t i ) const { return fVariables.at(i); }

      std::vector<VariableInfo>&       GetTargetInfos()         { return fTargets; }
      const std::vector<VariableInfo>& GetTargetInfos() const   { return fTargets; }
      VariableInfo&                    GetTargetInfo( Int_t i ) { return fTargets.at(i); }
      const VariableInfo&              GetTargetInfo( Int_t i ) const { return fTargets.at(i); }

      std::vector<VariableInfo>&       GetSpectatorInfos()         { return fSpectators; }
      const std::vector<VariableInfo>& GetSpectatorInfos() const   { return fSpectators; }
      VariableInfo&                    GetSpectatorInfo( Int_t i ) { return fSpectators.at(i); }
      const VariableInfo&              GetSpectatorInfo( Int_t i ) const { return fSpectators.at(i); }


      UInt_t                           GetNVariables()    const { return fVariables.size(); }
      UInt_t                           GetNTargets()      const { return fTargets.size(); }
      UInt_t                           GetNSpectators(bool all=kTRUE)   const;

      const TString&                   GetNormalization() const { return fNormalization; }
      void                             SetNormalization( const TString& norm )   { fNormalization = norm; }

      // classification information
      Int_t              GetClassNameMaxLength() const;
      ClassInfo*         GetClassInfo( Int_t clNum ) const;
      ClassInfo*         GetClassInfo( const TString& name ) const;
      void               PrintClasses() const;
      UInt_t             GetNClasses() const { return fClasses.size(); }
      Bool_t             IsSignal( const Event* ev ) const;
      std::vector<Float_t>* GetTargetsForMulticlass( const Event* ev );

      // by variable
      Int_t              FindVarIndex( const TString& )      const;

      // weights
      const TString      GetWeightExpression(Int_t i)      const { return GetClassInfo(i)->GetWeight(); }
      void               SetWeightExpression( const TString& exp, const TString& className = "" );

      // cuts
      const TCut&        GetCut (Int_t i)                         const { return GetClassInfo(i)->GetCut(); }
      const TCut&        GetCut ( const TString& className )      const { return GetClassInfo(className)->GetCut(); }
      void               SetCut ( const TCut& cut, const TString& className );
      void               AddCut ( const TCut& cut, const TString& className );
      Bool_t             HasCuts() const;

      std::vector<TString> GetListOfVariables() const;

      // correlation matrix 
      const TMatrixD*    CorrelationMatrix     ( const TString& className ) const;
      void               SetCorrelationMatrix  ( const TString& className, TMatrixD* matrix );
      void               PrintCorrelationMatrix( const TString& className );
      TH2*               CreateCorrelationMatrixHist( const TMatrixD* m,
                                                      const TString& hName,
                                                      const TString& hTitle ) const;

      // options
      void               SetSplitOptions(const TString& so) { fSplitOptions = so; fNeedsRebuilding = kTRUE; }
      const TString&     GetSplitOptions() const { return fSplitOptions; }

      // root dir
      void               SetRootDir(TDirectory* d) { fOwnRootDir = d; }
      TDirectory*        GetRootDir() const { return fOwnRootDir; }

      void               SetMsgType( EMsgType t ) const;

   private:

      TMVA::DataSetManager*            fDataSetManager; // DSMTEST
      void                       SetDataSetManager( DataSetManager* dsm ) { fDataSetManager = dsm; } // DSMTEST
      friend class DataSetManager;  // DSMTEST (datasetmanager test)

      DataSetInfo( const DataSetInfo& ) : TObject() {}

      void PrintCorrelationMatrix( TTree* theTree );

      TString                    fName;              //! name of the dataset info object

      mutable DataSet*           fDataSet;           //! dataset, owned by this datasetinfo object
      mutable Bool_t             fNeedsRebuilding;   //! flag if rebuilding of dataset is needed (after change of cuts, vars, etc.)

      // expressions/formulas
      std::vector<VariableInfo>  fVariables;         //! list of variable expressions/internal names
      std::vector<VariableInfo>  fTargets;           //! list of targets expressions/internal names
      std::vector<VariableInfo>  fSpectators;          //! list of spectators expressions/internal names

      // the classes
      mutable std::vector<ClassInfo*> fClasses;      //! name and other infos of the classes

      TString                    fNormalization;     //!
      TString                    fSplitOptions;      //!
      
      TDirectory*                fOwnRootDir;        //! ROOT output dir
      Bool_t                     fVerbose;           //! Verbosity

      UInt_t                     fSignalClass;       //! index of the class with the name signal

      std::vector<Float_t>*      fTargetsForMulticlass;       //! all targets 0 except the one with index==classNumber
      
      mutable MsgLogger*         fLogger;            //! message logger
      MsgLogger& Log() const { return *fLogger; }


   };
}

#endif
