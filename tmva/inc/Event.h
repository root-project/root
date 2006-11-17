// @(#)root/tmva $Id: Event.h,v 1.22 2006/11/16 22:51:58 helgevoss Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Event                                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Event container                                                           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany ,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 **********************************************************************************/

#ifndef TMVA_ROOT_Event
#define TMVA_ROOT_Event

#include <vector>
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMVA_VariableInfo
#include "TMVA/VariableInfo.h"
#endif

class TTree;
class TBranch;

namespace TMVA {

   class Event;

   ostream& operator<<( ostream& os, const Event& event );
   ostream& operator<<( ostream& os, const Event* event );

   class Event {

      friend ostream& operator<<( ostream& os, const Event& event );
      friend ostream& operator<<( ostream& os, const Event* event );

   public:

      Event( const std::vector<TMVA::VariableInfo>& );
      Event( const Event& );
      ~Event() { fgCount--; }
      
      void SetBranchAddresses(TTree* tr);
      std::vector<TBranch*>& Branches() { return fBranches; }

      Bool_t  IsSignal()       const { return (fType==1); }
      Float_t GetWeight()      const { return fWeight*fBoostWeight; }
      Float_t GetBoostWeight() const { return fBoostWeight; }
      Int_t   Type()           const { return fType; }
      void    SetWeight(Float_t w)      { fWeight=w; }
      void    SetBoostWeight(Float_t w) { fBoostWeight=w; }
      void    SetType(Int_t t)          { fType=t; }
      void    SetVal(UInt_t ivar, Float_t val);
      void    CopyVarValues( const Event& other );

      Char_t  VarType(Int_t ivar)     const { return fVariables[ivar].VarType(); }
      Bool_t  IsInt(Int_t ivar)       const { return (fVariables[ivar].VarType()=='I'); }
      Bool_t  IsFloat(Int_t ivar)     const { return (fVariables[ivar].VarType()=='F'); }
      Float_t GetVal(Int_t ivar)      const { return *((Float_t*)fVarPtr[ivar]); }
      Float_t GetValFloat(Int_t ivar) const { return *((Float_t*)fVarPtr[ivar]); }
      Int_t   GetValInt(Int_t ivar)   const { return *((Int_t*)fVarPtr[ivar]); }
      UInt_t  GetNVars()              const { return fVariables.size(); }
      Float_t GetValueNormalized(Int_t ivar) const;

      void Print(std::ostream & o) const;

   private:

      void InitPointers(bool AllowExternalLink = kTRUE);
   
      const std::vector<TMVA::VariableInfo>& fVariables; // the variables
      void **   fVarPtr;          // array containing values
      Int_t *   fVarPtrI;         // integer value
      Float_t*  fVarPtrF;         // float value
      Int_t     fType;            // signal or background type
      Float_t   fWeight;          // event weight (product of global and individual weights)
      Float_t   fBoostWeight;     // internal weight to be set by boosting algorithm
      UInt_t    fCountI;          // the number of Integer variables
      UInt_t    fCountF;          // the number of Float variables

      std::vector<TBranch*> fBranches; // TTree branches

      mutable MsgLogger fLogger;  // message logger

      static Int_t fgCount;       // count instances of Event

   };

}

#endif
