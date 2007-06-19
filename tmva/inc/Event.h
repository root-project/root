// @(#)root/tmva $Id: Event.h,v 1.12 2007/04/19 06:53:01 brun Exp $   
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
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 **********************************************************************************/

#ifndef ROOT_TMVA_Event
#define ROOT_TMVA_Event

#include <vector>
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
// #ifndef ROOT_TMVA_MsgLogger
// #include "TMVA/MsgLogger.h"
// #endif
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

      Event( const std::vector<TMVA:: VariableInfo>&, Bool_t AllowExternalLinks = kTRUE );
      Event( const Event& );
      ~Event();
      
      void SetBranchAddresses(TTree* tr);
      std::vector<TBranch*>& Branches() { return fBranches; }

      Bool_t  IsSignal()       const { return (fType==1); }
      Float_t GetWeight()      const { return fWeight*fBoostWeight; }
      Float_t GetBoostWeight() const { return fBoostWeight; }
      Int_t   Type()           const { return fType; }
      void    SetWeight(Float_t w)      { fWeight=w; }
      void    SetBoostWeight(Float_t w) { fBoostWeight=w; }
      void    SetType(Int_t t)          { fType=t; }
      void    SetType(Types::ESBType t) { fType=(t==Types::kSignal)?1:0; }
      void    SetVal(UInt_t ivar, Float_t val);
      void    SetValFloatNoCheck(UInt_t ivar, Float_t val) { *((Float_t*)fVarPtr[ivar]) = val; }


      void    CopyVarValues( const Event& other );

      Char_t  GetVarType (UInt_t ivar)        const { return fVariables[ivar].GetVarType(); }
      Bool_t  IsInt      (UInt_t ivar)        const { return (fVariables[ivar].GetVarType()=='I'); }
      Bool_t  IsFloat    (UInt_t ivar)        const { return (fVariables[ivar].GetVarType()=='F'); }
      Float_t GetVal     (UInt_t ivar)        const;
      Float_t GetValFloat(UInt_t ivar)        const { return *((Float_t*)fVarPtr[ivar]); }
      UInt_t  GetNVars()                      const { return fVariables.size(); }
      Float_t GetValueNormalized(UInt_t ivar) const;
      void*   GetExternalLink(UInt_t ivar)    const { return fVariables[ivar].GetExternalLink(); }

      void Print(std::ostream & o) const;

      Int_t GetMemSize() const { 
         Int_t size = sizeof(*this);
         size += GetNVars() * (sizeof(void*)+sizeof(Int_t)+sizeof(Float_t));
         return size;
      }

   private:

      void InitPointers(bool AllowExternalLink = kTRUE);
   
      const std::vector<TMVA::VariableInfo>& fVariables; // the variables
      void **   fVarPtr;          // array containing values
      //    Int_t *   fVarPtrI;         // integer value
      Float_t*  fVarPtrF;         // float value
      Int_t     fType;            // signal or background type
      Float_t   fWeight;          // event weight (product of global and individual weights)
      Float_t   fBoostWeight;     // internal weight to be set by boosting algorithm
      UInt_t    fCountI;          // the number of Integer variables
      UInt_t    fCountF;          // the number of Float variables

      std::vector<TBranch*> fBranches; // TTree branches

      static Int_t fgCount;       // count instances of Event

   };

}

#endif
