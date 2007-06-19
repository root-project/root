// @(#)root/tmva $Id: VariableTransformBase.h,v 1.20 2007/06/08 10:27:25 stelzer Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : VariableTransformBase                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Pre-transformation of input variables (base class)                        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_VariableTransformBase
#define ROOT_TMVA_VariableTransformBase

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// VariableTransformBase                                                //
//                                                                      //
// Linear interpolation class                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include "TTree.h"
#include "TH1.h"
#include "TDirectory.h"

#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif
#ifndef ROOT_TMVA_VariableInfo
#include "TMVA/VariableInfo.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

namespace TMVA {

   class Ranking;

   class VariableTransformBase : public TObject {

   public:
  
      VariableTransformBase( std::vector<TMVA::VariableInfo>&, Types::EVariableTransform tf );
      virtual ~VariableTransformBase( void );

      virtual void   ApplyTransformation( Types::ESBType type = Types::kMaxSBType ) const = 0;
      virtual Bool_t PrepareTransformation( TTree* inputTree ) = 0;
      void PlotVariables( TTree* theTree );

      // accessors
      void   SetEnabled  ( Bool_t e ) { fEnabled = e; }
      void   SetNormalise( Bool_t n ) { fNormalise = n; }
      Bool_t IsEnabled()    const { return fEnabled; }
      Bool_t IsCreated()    const { return fCreated; }
      Bool_t IsNormalised() const { return fNormalise; }

      void CreateEvent() const;

      virtual TMVA::Event& GetEvent() const { 
         if (fEvent==0) CreateEvent();
         return *fEvent; 
      }
      TMVA::Event& GetEventRaw()      const { 
         if (fEventRaw==0) fEventRaw = new TMVA::Event(fVariables); return *fEventRaw; 
      }

      void SetUseSignalTransform( Bool_t e=kTRUE) { fUseSignalTransform = e; }

      virtual const char* GetName() const { return fTransformName; }

      UInt_t GetNVariables() const { return fVariables.size(); }

      void SetRootOutputBaseDir(TDirectory * dir) { fOutputBaseDir = dir; }

      Bool_t ReadEvent( TTree* tr, UInt_t evidx, Types::ESBType type ) const;

      const std::vector<TMVA::VariableInfo>& Variables() const { return fVariables; }

      const VariableInfo& Variable(Int_t ivar) const { return fVariables[ivar]; }
      VariableInfo&       Variable(Int_t ivar) { return fVariables[ivar]; }

      const VariableInfo& Variable(const TString& var) const { return fVariables[FindVar(var)]; }
      VariableInfo&       Variable(const TString& var) { return fVariables[FindVar(var)]; }
      Int_t               FindVar(const TString& var) const { // don't use in loops, it is slow
         for (UInt_t ivar=0; ivar<GetNVariables(); ivar++) 
            if (var == fVariables[ivar].GetInternalVarName()) return ivar;
         return -1;
      }

      void         WriteVarsToStream           ( std::ostream& o, const TString& prefix = "" ) const;
      void         ReadVarsFromStream          ( std::istream& istr );
      virtual void WriteTransformationToStream ( std::ostream& o ) const = 0;
      virtual void ReadTransformationFromStream( std::istream& istr ) = 0;

      // variable ranking
      Ranking* GetVariableRanking()   const { return fRanking; }
      void     PrintVariableRanking() const;

      Types::EVariableTransform GetVariableTransform() const { return fVariableTransform; }

      virtual void PrintTransformation(ostream &) {};

      // writer of function code
      virtual void MakeFunction(std::ostream& fout, const TString& fncName, Int_t part) = 0;


   protected:

      void CalcNorm( TTree * );

      void SetCreated( Bool_t c = kTRUE ) { fCreated = c; }
      void SetName( const TString& c )    { fTransformName = c; }

      void ResetBranchAddresses( TTree* tree ) const;

      Bool_t                  fUseSignalTransform; // true if transformation bases on signal data
      mutable TMVA::Event*    fEvent;     // this is the event
      mutable TMVA::Event*    fEventRaw;  // this is the untransformed event

      TDirectory*             GetOutputBaseDir() const { return fOutputBaseDir; }

   private:

      Types::EVariableTransform fVariableTransform;  // Decorrelation, PCA, etc.

      void           UpdateNorm( Int_t ivar, Double_t x );

      Bool_t         fEnabled;            // has been enabled
      Bool_t         fCreated;            // has been created
      Bool_t         fNormalise;          // normalise input variables

      TString        fTransformName;
      
      std::vector<TMVA::VariableInfo>  fVariables; // event variables [saved to weight file]
      
      mutable TTree* fCurrentTree;        // pointer to tree
      mutable Size_t fCurrentEvtIdx;      // current event index
      TDirectory*    fOutputBaseDir;      // directory

      Ranking*       fRanking;            // ranking object

   protected:

      mutable MsgLogger  fLogger;         // message logger

      ClassDef(VariableTransformBase,0)   //  Base class for variable transformations
   };

} // namespace TMVA

#endif 


