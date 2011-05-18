
/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Classes: PDEFoamCell                                                           *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Objects of this class are hyperrectangular cells organized in             *
 *      the binary tree. Special algoritm for encoding relalive                   *
 *      positioning of the cells saves total memory allocation needed             *
 *      for the system of cells.                                                  *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      S. Jadach        - Institute of Nuclear Physics, Cracow, Poland           *
 *      Tancredi Carli   - CERN, Switzerland                                      *
 *      Dominik Dannheim - CERN, Switzerland                                      *
 *      Alexander Voigt  - CERN, Switzerland                                      *
 *                                                                                *
 * Copyright (c) 2008:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_PDEFoamCell
#define ROOT_TMVA_PDEFoamCell

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TRef
#include "TRef.h"
#endif

#ifndef ROOT_TMVA_PDEFoamVect
#include "TMVA/PDEFoamVect.h"
#endif

namespace TMVA {

   class PDEFoamCell : public TObject {

      //   static, the same for all cells!
   private:
      Short_t  fDim;                   // Dimension of the vector space
      //   MEMBERS

   private:
      //--- linked tree organization ---
      Int_t    fSerial;                // Serial number
      Int_t    fStatus;                // Status (active, inactive)
      TRef     fParent;                // Pointer to parent cell
      TRef     fDaught0;               // Pointer to daughter 1
      TRef     fDaught1;               // Pointer to daughter 2
      //--- M.C. sampling and choice of the best edge ---

   private:
      Double_t fXdiv;                  // Factor for division
      Int_t    fBest;                  // Best Edge for division
      //--- Integrals of all kinds ---
      Double_t fVolume;                // Cartesian Volume of cell
      Double_t fIntegral;              // Integral over cell (estimate from exploration)
      Double_t fDrive;                 // Driver  integral, only for cell build-up
      //----------  working space for the user --------------
      TObject *fElement;               // may set by the user to save some data in this cell

      //////////////////////////////////////////////////////////////////////////////////////
      //                           METHODS                                                //
      //////////////////////////////////////////////////////////////////////////////////////
   public:
      PDEFoamCell();                          // Default Constructor for ROOT streamers
      PDEFoamCell(Int_t);                     // User Constructor
      PDEFoamCell(const PDEFoamCell&);        // Copy constructor
      virtual ~PDEFoamCell();                 // Destructor
      void  Fill(Int_t, PDEFoamCell*, PDEFoamCell*, PDEFoamCell*);    // Assigns values of attributes
      //--------------- Geometry ----------------------------------
      Double_t  GetXdiv() const { return fXdiv;}          // Pointer to Xdiv
      Int_t     GetBest() const { return fBest;}          // Pointer to Best
      void      SetBest(Int_t    Best){ fBest =Best;}     // Set Best edge candidate
      void      SetXdiv(Double_t Xdiv){ fXdiv =Xdiv;}     // Set x-division for best edge cand.
      void      GetHcub(  PDEFoamVect&, PDEFoamVect&) const;  // Get position and size vectors (h-cubical subspace)
      void      GetHSize( PDEFoamVect& ) const;             // Get size only of cell vector  (h-cubical subspace)
      //--------------- Integrals/Volumes -------------------------
      void      CalcVolume();                             // Calculates volume of cell
      Double_t  GetVolume() const { return fVolume;}      // Volume of cell
      Double_t  GetIntg() const { return fIntegral;}      // Get Integral
      Double_t  GetDriv() const { return fDrive;}         // Get Drive
      void      SetIntg(Double_t Intg){ fIntegral=Intg;}  // Set true integral
      void      SetDriv(Double_t Driv){ fDrive   =Driv;}  // Set driver integral
      //--------------- linked tree organization ------------------
      Int_t     GetStat() const { return fStatus;}        // Get Status
      void      SetStat(Int_t Stat){ fStatus=Stat;}       // Set Status
      PDEFoamCell* GetPare() const { return (PDEFoamCell*) fParent.GetObject(); }  // Get Pointer to parent cell
      PDEFoamCell* GetDau0() const { return (PDEFoamCell*) fDaught0.GetObject(); } // Get Pointer to 1-st daughter vertex
      PDEFoamCell* GetDau1() const { return (PDEFoamCell*) fDaught1.GetObject(); } // Get Pointer to 2-nd daughter vertex
      void      SetDau0(PDEFoamCell* Daug){ fDaught0 = Daug;}  // Set pointer to 1-st daughter
      void      SetDau1(PDEFoamCell* Daug){ fDaught1 = Daug;}  // Set pointer to 2-nd daughter
      void      SetPare(PDEFoamCell* Pare){ fParent  = Pare;}  // Set pointer to parent
      void      SetSerial(Int_t Serial){ fSerial=Serial;}    // Set serial number
      Int_t     GetSerial() const { return fSerial;}         // Get serial number
      UInt_t    GetDepth();                                  // Get depth in binary tree
      //--- other ---
      void Print(Option_t *option) const ;                   // Prints cell content
      //--- getter and setter for user variable ---
      void SetElement(TObject* fobj){ fElement = fobj; }     // Set user variable
      TObject* GetElement() const { return fElement; }       // Get pointer to user varibale
      ////////////////////////////////////////////////////////////////////////////
      ClassDef(PDEFoamCell,1)  //Single cell of FOAM
   }; // end of PDEFoamCell
} // namespace TMVA

#endif
