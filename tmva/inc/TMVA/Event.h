// @(#)root/tmva $Id$   
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss, Jan Therhaag

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
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch>  - CERN, Switzerland          *
 *      Jan Therhaag       <Jan.Therhaag@cern.ch>     - U of Bonn, Germany        *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 **********************************************************************************/

#ifndef ROOT_TMVA_Event
#define ROOT_TMVA_Event

#include <iosfwd>
#include <vector>

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_ThreadLocalStorage
#include "ThreadLocalStorage.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif



class TCut;

namespace TMVA {

   class Event;

   std::ostream& operator<<( std::ostream& os, const Event& event );

   class Event {

      friend std::ostream& operator<<( std::ostream& os, const Event& event );

   public:

      // constructors
      Event();
      Event( const Event& );
      explicit Event( const std::vector<Float_t>& values, 
                      const std::vector<Float_t>& targetValues, 
                      const std::vector<Float_t>& spectatorValues, 
                      UInt_t theClass = 0, Double_t weight = 1.0, Double_t boostweight = 1.0 );
      explicit Event( const std::vector<Float_t>& values, 
                      const std::vector<Float_t>& targetValues, 
                      UInt_t theClass = 0, Double_t weight = 1.0, Double_t boostweight = 1.0 );
      explicit Event( const std::vector<Float_t>&, 
                      UInt_t theClass, Double_t weight = 1.0, Double_t boostweight = 1.0 );
      explicit Event( const std::vector<Float_t*>*&, UInt_t nvar );

      ~Event();

      // accessors
      Bool_t  IsDynamic()         const {return fDynamic; }

      //      Double_t GetWeight()         const { return fWeight*fBoostWeight; }
      Double_t GetWeight()         const;
      Double_t GetOriginalWeight() const { return fWeight; }
      Double_t GetBoostWeight()    const { return TMath::Max(Double_t(0.0001),fBoostWeight); }
      UInt_t   GetClass()          const { return fClass; }  

      UInt_t   GetNVariables()        const;
      UInt_t   GetNTargets()          const;
      UInt_t   GetNSpectators()       const;

      Float_t  GetValue( UInt_t ivar) const;
      std::vector<Float_t>& GetValues() 
      {
	  //For a detailed explanation, please see the heading "Avoid Duplication in const and Non-const Member Function," on p. 23, in Item 3 "Use const whenever possible," in Effective C++, 3d ed by Scott Meyers, ISBN-13: 9780321334879.
	  // http://stackoverflow.com/questions/123758/how-do-i-remove-code-duplication-between-similar-const-and-non-const-member-func
	  return const_cast<std::vector<Float_t>&>( static_cast<const Event&>(*this).GetValues() );
      }
      const std::vector<Float_t>& GetValues() const;

      Float_t  GetTarget( UInt_t itgt ) const { return fTargets.at(itgt); }
      std::vector<Float_t>& GetTargets()  { return fTargets; }
      const std::vector<Float_t>& GetTargets() const { return fTargets; }

      Float_t  GetSpectator( UInt_t ivar) const;
      std::vector<Float_t>& GetSpectators()  { return fSpectators; }
      const std::vector<Float_t>& GetSpectators() const { return fSpectators; }

      void     SetWeight             ( Double_t w ) { fWeight=w; }
      void     SetBoostWeight        ( Double_t w ) const { fDoNotBoost ? fDoNotBoost = kFALSE : fBoostWeight=w; }
      void     ScaleBoostWeight      ( Double_t s ) const { fDoNotBoost ? fDoNotBoost = kFALSE : fBoostWeight *= s; }
      void     SetClass              ( UInt_t t )  { fClass=t; }
      void     SetVal                ( UInt_t ivar, Float_t val );
      void     SetTarget             ( UInt_t itgt, Float_t value );
      void     SetSpectator          ( UInt_t ivar, Float_t value );
      void     SetVariableArrangement( std::vector<UInt_t>* const m ) const;

      void     SetDoNotBoost         () const  { fDoNotBoost = kTRUE; }
      static void ClearDynamicVariables() {}

      void     CopyVarValues( const Event& other );
      void     Print        ( std::ostream & o ) const;

      static   void SetIsTraining(Bool_t);
      static   void SetIgnoreNegWeightsInTraining(Bool_t);
   private:

      static   Bool_t          fgIsTraining;    // mark if we are in an actual training or "evaluation/testing" phase --> ignoreNegWeights only in actual training !
      static   Bool_t          fgIgnoreNegWeightsInTraining;


      mutable std::vector<Float_t>   fValues;          // the event values ; mutable, to be able to copy the dynamic values in there

      mutable std::vector<Float_t>   fValuesRearranged;   // the event values ; mutable, to be able to copy the dynamic values in there
      mutable std::vector<Float_t*>* fValuesDynamic;   // the event values
      std::vector<Float_t>   fTargets;         // target values for regression
      mutable std::vector<Float_t>   fSpectators;      // "visisting" variables not used in MVAs ; mutable, to be able to copy the dynamic values in there
      mutable std::vector<UInt_t>*   fVariableArrangement;  // needed for MethodCategories, where we can train on other than the main variables

      UInt_t                         fClass;           // class number
      Double_t                       fWeight;          // event weight (product of global and individual weights)
      mutable Double_t               fBoostWeight;     // internal weight to be set by boosting algorithm
      Bool_t                         fDynamic;         // is set when the dynamic values are taken
      mutable Bool_t                 fDoNotBoost;       // mark event as not to be boosted (used to compensate for events with negative event weights
   };
}

#endif
