
/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamDistr                                                          *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Class PDEFoamDistr is an Abstract class representing                      *
 *      n-dimensional real positive integrand function                            *
 *      The main function is Density() which provides the event density at a      *
 *      given point during the foam build-up (sampling).                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Tancredi Carli   - CERN, Switzerland                                      *
 *      Dominik Dannheim - CERN, Switzerland                                      *
 *      S. Jadach        - Institute of Nuclear Physics, Cracow, Poland           *
 *      Alexander Voigt  - CERN, Switzerland                                      *
 *      Peter Speckmayer - CERN, Switzerland                                      *
 *                                                                                *
 * Copyright (c) 2008:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_PDEFoamDistr
#define ROOT_TMVA_PDEFoamDistr

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TH1D
#include "TH1D.h"
#endif
#ifndef ROOT_TH2D
#include "TH2D.h"
#endif

#ifndef ROOT_TMVA_BinarySearchTree
#include "TMVA/BinarySearchTree.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif
#ifndef ROOT_TMVA_PDEFoamCell
#include "TMVA/PDEFoamCell.h"
#endif

namespace TMVA {
   enum EFoamType { kSeparate, kDiscr, kMonoTarget, kMultiTarget };

   // options for filling density (used in Density() to build up foam)
   // kEVENT_DENSITY : use event density for foam buildup
   // kDISCRIMINATOR : use N_sig/(N_sig + N_bg) for foam buildup
   // kTARGET        : use GetTarget(0) for foam build up
   enum TDensityCalc { kEVENT_DENSITY, kDISCRIMINATOR, kTARGET };
}

namespace TMVA {

   // class definition of underlying density
   class PDEFoamDistr : public ::TObject  {

   private:
      Int_t fDim;               // number of dimensions
      Float_t *fXmin;           //[fDim] minimal value of phase space in all dimension
      Float_t *fXmax;           //[fDim] maximal value of phase space in all dimension
      Float_t fVolFrac;         // volume fraction (with respect to total phase space
      BinarySearchTree *fBst;   // Binary tree to find events within a volume
      TDensityCalc fDensityCalc;// method of density calculation

   protected:
      Int_t fSignalClass;      // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event
      Int_t fBackgroundClass;  // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event

      mutable MsgLogger* fLogger;                     //! message logger
      MsgLogger& Log() const { return *fLogger; }

   public:
      PDEFoamDistr();
      PDEFoamDistr(const PDEFoamDistr&);
      virtual ~PDEFoamDistr();

      // Getter and setter for VolFrac option
      void SetVolumeFraction(Float_t vfr){fVolFrac=vfr; return;}
      Float_t GetVolumeFraction(){return fVolFrac;}

      // set foam dimension (mandatory before foam build-up!)
      void SetDim(Int_t idim);

      // set foam boundaries
      void SetXmin(Int_t idim,Float_t wmin){fXmin[idim]=wmin; return;}
      void SetXmax(Int_t idim,Float_t wmax){fXmax[idim]=wmax; return;}

      // transformation functions for event variable into foam boundaries
      // reason: foam allways has boundaries [0, 1]
      Float_t VarTransform(Int_t idim, Float_t x){        // transform [xmin, xmax] --> [0, 1]
         Float_t b=fXmax[idim]-fXmin[idim];
         return (x-fXmin[idim])/b;
      }
      Float_t VarTransformInvers(Int_t idim, Float_t x){  // transform [0, 1] --> [xmin, xmax]
         Float_t b=fXmax[idim]-fXmin[idim];
         return x*b + fXmin[idim];
      }

      // density build-up functions
      void Initialize(Int_t ndim = 2);
      void FillBinarySearchTree( const Event* ev, EFoamType ft, Bool_t NoNegWeights=kFALSE );

      // main function used by PDEFoam
      // returns density at a given point by range searching in BST
      Double_t Density(Double_t *Xarg, Double_t &event_density);

      // Getters and setters for foam filling method
      void SetDensityCalc( TDensityCalc dc ){ fDensityCalc = dc; };
      Bool_t FillDiscriminator(){ return fDensityCalc == kDISCRIMINATOR; }
      Bool_t FillTarget0()      { return fDensityCalc == kTARGET;        }
      Bool_t FillEventDensity() { return fDensityCalc == kEVENT_DENSITY; }

      void SetSignalClass( Int_t cls )     { fSignalClass = cls;  } // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event
      void SetBackgroundClass( Int_t cls ) { fBackgroundClass = cls;  } // TODO: intermediate solution to keep IsSignal() of Event working. TODO: remove IsSignal() from Event

      ClassDef(PDEFoamDistr,2) //Class for Event density
   };  //end of PDEFoamDistr

}  // namespace TMVA

#endif
