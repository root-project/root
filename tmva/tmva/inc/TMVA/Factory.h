// @(#)root/tmva $Id$   
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss, Kai Voss, Eckhard von Toerne, Jan Therhaag
// Updated by: Omar Zapata, Lorenzo Moneta, Sergei Gleyzer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Factory                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      This is the main MVA steering class: it creates (books) all MVA methods,  *
 *      and guides them through the training, testing and evaluation phases.      *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <stelzer@cern.ch>        - DESY, Germany                  *
 *      Peter Speckmayer <peter.speckmayer@cern.ch> - CERN, Switzerland           *
 *      Jan Therhaag          <Jan.Therhaag@cern.ch>   - U of Bonn, Germany       *
 *      Eckhard v. Toerne     <evt@uni-bonn.de>        - U of Bonn, Germany       *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *      Omar Zapata     <Omar.Zapata@cern.ch>    - UdeA/ITM Colombia              *
 *      Lorenzo Moneta  <Lorenzo.Moneta@cern.ch> - CERN, Switzerland              *
 *      Sergei Gleyzer  <Sergei.Gleyzer@cern.ch> - U of Florida & CERN            *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *      UdeA/ITM, Colombia                                                        *
 *      U. of Florida, USA                                                        *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_Factory
#define ROOT_TMVA_Factory

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Factory                                                              //
//                                                                      //
// This is the main MVA steering class: it creates all MVA methods,     //
// and guides them through the training, testing and evaluation         //
// phases                                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string>
#include <vector>
#include <map>
#ifndef ROOT_TCut
#include "TCut.h"
#endif

#ifndef ROOT_TMVA_Configurable
#include "TMVA/Configurable.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif
#ifndef ROOT_TMVA_DataSet
#include "TMVA/DataSet.h"
#endif

class TFile;
class TTree;
class TDirectory;

namespace TMVA {

   class IMethod;
   class MethodBase;
   class DataInputHandler;
   class DataSetInfo;
   class DataSetManager;
   class DataLoader;
   class VariableTransformBase;
   

   class Factory : public Configurable {
   public:

      typedef std::vector<IMethod*> MVector;
      std::map<TString,MVector*>  fMethodsMap;//all methods for every dataset with the same name

      // no default  constructor
      Factory( TString theJobName, TFile* theTargetFile, TString theOption = "" );

      // default destructor
      virtual ~Factory();

      virtual const char*  GetName() const { return "Factory"; }


      MethodBase* BookMethod( DataLoader *loader, TString theMethodName, TString methodTitle, TString theOption = "" );
      MethodBase* BookMethod( DataLoader *loader, Types::EMVA theMethod,  TString methodTitle, TString theOption = "" );
      MethodBase* BookMethod( DataLoader *, TMVA::Types::EMVA /*theMethod*/, 
                              TString /*methodTitle*/, 
                              TString /*methodOption*/, 
                              TMVA::Types::EMVA /*theComposite*/, 
                              TString /*compositeOption = ""*/ ) { return 0; } 

      // optimize all booked methods (well, if desired by the method)
      void OptimizeAllMethods                 (TString fomType="ROCIntegral", TString fitType="FitGA");
      void OptimizeAllMethodsForClassification(TString fomType="ROCIntegral", TString fitType="FitGA") { OptimizeAllMethods(fomType,fitType); }
      void OptimizeAllMethodsForRegression    (TString fomType="ROCIntegral", TString fitType="FitGA") { OptimizeAllMethods(fomType,fitType); }

      // training for all booked methods
      void TrainAllMethods                 ();
      void TrainAllMethodsForClassification( void ) { TrainAllMethods(); }
      void TrainAllMethodsForRegression    ( void ) { TrainAllMethods(); }

      // testing
      void TestAllMethods();

      // performance evaluation
      void EvaluateAllMethods( void );
      void EvaluateAllVariables(DataLoader *loader, TString options = "" ); 
  
      // delete all methods and reset the method vector
      void DeleteAllMethods( void );

      // accessors
      IMethod* GetMethod( const TString& datasetname, const TString& title )const;

      Bool_t Verbose( void ) const { return fVerbose; }
      void SetVerbose( Bool_t v=kTRUE );

      // make ROOT-independent C++ class for classifier response 
      // (classifier-specific implementation)
      // If no classifier name is given, help messages for all booked 
      // classifiers are printed
      virtual void MakeClass(const TString& datasetname , const TString& methodTitle = "" ) const;

      // prints classifier-specific hepl messages, dedicated to 
      // help with the optimisation and configuration options tuning.
      // If no classifier name is given, help messages for all booked 
      // classifiers are printed
      void PrintHelpMessage(const TString& datasetname , const TString& methodTitle = "" ) const;

      static TDirectory* RootBaseDir() { return (TDirectory*)fgTargetFile; }

      static Bool_t IsSilentFile();
      
      Double_t GetROCIntegral(DataLoader *loader,TString theMethodName);
      Double_t GetROCIntegral(TString  datasetname,TString theMethodName);

   private:

      // the beautiful greeting message
      void Greetings();

      void WriteDataInformation(DataSetInfo&     fDataSetInfo);

      void                     SetInputTreesFromEventAssignTrees();

   private:

      // data members

      static TFile*                             fgTargetFile;     //! ROOT output file


      std::vector<TMVA::VariableTransformBase*> fDefaultTrfs;     //! list of transformations on default DataSet

      // cd to local directory
      TString                                   fOptions;         //! option string given by construction (presently only "V")
      TString                                   fTransformations; //! List of transformations to test
      Bool_t                                    fVerbose;         //! verbose mode
      Bool_t					fCorrelations;    //! enable to calculate corelations
      Bool_t					fROC;             //! enable to calculate ROC values
      static Bool_t				fSilentFile;      //! enable to reduce the output file

      TString                                   fJobName;         //! jobname, used as extension in weight file names

      // flag determining the way training and test data are assigned to Factory
      enum DataAssignType { kUndefined = 0, 
                            kAssignTrees,
                            kAssignEvents };
      DataAssignType                            fDataAssignType;  //! flags for data assigning
      std::vector<TTree*>                       fTrainAssignTree; //! for each class: tmp tree if user wants to assign the events directly
      std::vector<TTree*>                       fTestAssignTree;  //! for each class: tmp tree if user wants to assign the events directly

      Int_t                                     fATreeType;          // type of event (=classIndex)
      Float_t                                   fATreeWeight;        // weight of the event
      Float_t*                                  fATreeEvent;         // event variables

      Types::EAnalysisType                      fAnalysisType;    //! the training type

   protected:

      ClassDef(Factory,0)  // The factory creates all MVA methods, and performs their training and testing
   };

} // namespace TMVA

#endif

