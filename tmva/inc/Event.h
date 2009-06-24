// @(#)root/tmva $Id$   
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

#include <iosfwd>
#include <vector>

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_TMVA_Types
#include "TMVA/Types.h"
#endif

namespace TMVA {

   class Event;

   std::ostream& operator<<( std::ostream& os, const Event& event );
   std::ostream& operator<<( std::ostream& os, const Event* event );

   class Event {

      friend std::ostream& operator<<( std::ostream& os, const Event& event );
      friend std::ostream& operator<<( std::ostream& os, const Event* event );

   public:

      // constructors
      Event();
      Event( const Event& );
      explicit Event( const std::vector<Float_t>&, 
                      const std::vector<Float_t>& targetValues, 
                      const std::vector<Float_t>& spectatorValues, 
                      UInt_t theClass = 0, Float_t weight = 1.0, Float_t boostweight = 1.0 );
      explicit Event( const std::vector<Float_t>&, 
                      const std::vector<Float_t>& targetValues, 
                      UInt_t theClass = 0, Float_t weight = 1.0, Float_t boostweight = 1.0 );
      explicit Event( const std::vector<Float_t>&, 
                      UInt_t theClass, Float_t weight = 1.0, Float_t boostweight = 1.0 );
      explicit Event( const std::vector<Float_t*>*& );

      ~Event();

      // accessors
      Bool_t  IsSignal()          const { return (fClass==fSignalClass); } // deprecated: use <DataSetInfo>.IsSignal( Event* )

      Float_t GetWeight()         const { return fWeight*fBoostWeight; }
      Float_t GetOriginalWeight() const { return fWeight; }
      Float_t GetBoostWeight()    const { return TMath::Max(Float_t(0.0001),fBoostWeight); }
      UInt_t  GetType()           const { return GetClass(); }  // better use GetClass()
      UInt_t  GetClass()          const { return fClass; }  
      UInt_t  Type()              const { return fClass; }  // backward compatible -> to be removed

      UInt_t  GetNVariables()     const { return fValues.size(); }
      UInt_t  GetNTargets()       const { return fTargets.size(); }
      UInt_t  GetNSpectators()    const { return fSpectators.size(); }
      UInt_t  GetNVars()          const { return fValues.size(); }  // backward compatible -> to be removed

      Float_t GetVal(UInt_t ivar) const;
      Int_t   GetSignalClass()    const { return fSignalClass; } // intermediate solution to keep IsSignal() of Event working
      
      const std::vector<Float_t>& GetValues() const;

      Float_t GetValue          ( UInt_t ivar) const { return GetVal( ivar ); }

      void    ScaleWeight       ( Float_t s ) { fWeight*=s; }
      void    SetWeight         ( Float_t w ) { fWeight=w; }
      void    SetBoostWeight    ( Float_t w ) { fBoostWeight=w; }
      void    ScaleBoostWeight  ( Float_t s ) { fBoostWeight *= s; }
      void    SetType           ( Int_t t  )  { SetClass(t); }
      void    SetClass          ( UInt_t t )  { fClass=t; }
      void    SetType           ( Types::ESBType t ) { fClass=(t==Types::kSignal) ? 1 : 0; }
      void    SetVal            ( UInt_t ivar, Float_t val );
      void    SetValFloatNoCheck( UInt_t ivar, Float_t val ) { fValues[ivar] = val; }

      void    SetSignalClass    ( UInt_t cls ){ fSignalClass = cls; } // intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event

      void    SetTarget( UInt_t itgt, Float_t value ) { 
         if (fTargets.size() <= itgt) fTargets.resize( itgt+1 );
         fTargets.at(itgt) = value;
      }
      std::vector<Float_t>& GetTargets()             const { return fTargets; }
      Float_t               GetTarget( UInt_t itgt ) const { return fTargets.at(itgt); }

      void    SetSpectator( UInt_t ivar, Float_t value ) { 
         if (fSpectators.size() <= ivar) fSpectators.resize( ivar+1 );
         fSpectators.at(ivar) = value;
      }
      std::vector<Float_t>& GetSpectators()            const { return fSpectators; }
      Float_t               GetSpectator( UInt_t ivar) const { return fSpectators.at(ivar); }

      static void ClearDynamicVariables();

      void    CopyVarValues( const Event& other );
      void    Print        ( std::ostream & o ) const;

   private:

      mutable std::vector<Float_t>   fValues;          // the event values
      static  std::vector<Float_t*>* fgValuesDynamic;  // the event values
      mutable std::vector<Float_t>   fTargets;         // target values for regression

      mutable std::vector<Float_t>   fSpectators;        // "visisting" variables which are never used for any calculation


      UInt_t                         fClass;           // signal or background type: signal=1, background=0
      Float_t                        fWeight;          // event weight (product of global and individual weights)
      Float_t                        fBoostWeight;     // internal weight to be set by boosting algorithm
      Bool_t                         fDynamic;         // is set when the dynamic values are taken

      UInt_t                         fSignalClass;     // intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event
      
      static Int_t                   fgCount;          // count instances of Event
   };
}

#endif
