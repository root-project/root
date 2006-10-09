// @(#)root/tmva $Id: Event.h,v 1.16 2006/09/29 23:27:15 andreas.hoecker Exp $   
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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
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
 **********************************************************************************/

#ifndef TMVA_ROOT_Event
#define TMVA_ROOT_Event

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include "TMVA/VariableInfo.h"

#include <vector>

class TTree;
class TBranch;

namespace TMVA {

   class Event;

   ostream& operator<< (ostream& os, const Event& event);
   ostream& operator<< (ostream& os, const Event* event);

   class Event {

      friend ostream& operator<< (ostream& os, const Event& event);
      friend ostream& operator<< (ostream& os, const Event* event);

   public:

      Event(const std::vector<TMVA::VariableInfo>&);
      Event(const Event& );
      ~Event(){};
      
      void SetBranchAddresses(TTree* tr);
      std::vector<TBranch*>& Branches() { return fBranches; }

      Bool_t  IsSignal()  const { return (fType==1); }
      Float_t GetWeight() const { return fWeight*fBoostWeight; }
      Float_t GetBoostWeight() const { return fBoostWeight; }
      Int_t   Type()      const { return fType; }
      void    SetWeight(Float_t w) { fWeight=w; }
      void    SetBoostWeight(Float_t w) { fBoostWeight=w; }
      void    SetType(Int_t t) { fType=t; }
      void    SetVal(UInt_t ivar, Float_t val);
      void    CopyVarValues( const Event& other );

      Char_t  VarType(Int_t ivar)     const { return fVariables[ivar].VarType(); }
      Bool_t  IsInt(Int_t ivar)       const { return (fVariables[ivar].VarType()=='I'); }
      Bool_t  IsFloat(Int_t ivar)     const { return (fVariables[ivar].VarType()=='F'); }
      Float_t GetVal(Int_t ivar)      const { return *((Float_t*)fVarPtr[ivar]); }
      Float_t GetValFloat(Int_t ivar) const { return *((Float_t*)fVarPtr[ivar]); }
      Int_t   GetValInt(Int_t ivar)   const { return *((Int_t*)fVarPtr[ivar]); }
      Float_t GetValueNormalized(Int_t ivar) const;
      UInt_t  GetNVars()              const { return fVariables.size(); }

      void Print(std::ostream & o) const;

   private:

      void InitPointers(bool AllowExternalLink = kTRUE);
   
      const std::vector<TMVA::VariableInfo>& fVariables;
      void **   fVarPtr;
      Int_t *   fVarPtrI;
      Float_t*  fVarPtrF;
      Int_t     fType;
      Float_t   fWeight;
      Float_t   fBoostWeight;
      UInt_t    fCountI;  //the number of Integer variables
      UInt_t    fCountF;  //the number of Float variables

      std::vector<TBranch*>  fBranches;

   };

}

#endif
