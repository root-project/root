
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
#ifndef ROOT_TMVA_PDEFoam
#include "TMVA/PDEFoam.h"
#endif
#ifndef ROOT_TMVA_PDEFoamCell
#include "TMVA/PDEFoamCell.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

namespace TMVA {
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
      const PDEFoam *fPDEFoam;  // PDEFoam to refer to
      BinarySearchTree *fBst;   // Binary tree to find events within a volume
      TDensityCalc fDensityCalc;// method of density calculation

   protected:
      mutable MsgLogger* fLogger;                     //! message logger
      MsgLogger& Log() const { return *fLogger; }

   public:
      PDEFoamDistr();
      PDEFoamDistr(const PDEFoamDistr&);
      virtual ~PDEFoamDistr();

      // density build-up functions
      void Initialize(); // create and initialize binary search tree
      void FillBinarySearchTree( const Event* ev, EFoamType ft, Bool_t NoNegWeights=kFALSE );

      // main function used by PDEFoam
      // returns density at a given point by range searching in BST
      Double_t Density(Double_t *Xarg, Double_t &event_density);

      // Return fDim histograms with signal and bg events
      void FillHist(PDEFoamCell* cell, std::vector<TH1F*>&, std::vector<TH1F*>&, 
		    std::vector<TH1F*>&, std::vector<TH1F*>&);

      // Getter and setter for the fPDEFoam pointer
      void SetPDEFoam(const PDEFoam *foam){ fPDEFoam = foam; }
      const PDEFoam* GetPDEFoam() const { return fPDEFoam; };

      // Getters and setters for foam filling method
      void SetDensityCalc( TDensityCalc dc ){ fDensityCalc = dc; };
      Bool_t FillDiscriminator(){ return fDensityCalc == kDISCRIMINATOR; }
      Bool_t FillTarget0()      { return fDensityCalc == kTARGET;        }
      Bool_t FillEventDensity() { return fDensityCalc == kEVENT_DENSITY; }

      ClassDef(PDEFoamDistr,3) //Class for Event density
   };  //end of PDEFoamDistr

}  // namespace TMVA

#endif
