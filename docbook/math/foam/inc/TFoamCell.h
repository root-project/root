// @(#)root/foam:$Id$
// Author: S. Jadach <mailto:Stanislaw.jadach@ifj.edu.pl>, P.Sawicki <mailto:Pawel.Sawicki@ifj.edu.pl>

#ifndef ROOT_TFoamCell
#define ROOT_TFoamCell

////////////////////////////////////////////////////////////////////////////////////
// Class TFoamCell  used in TFoam                                                 //
//                                                                                //
// Objects of this class are hyperrectangular cells organized in the binary tree. //
// Special algoritm for encoding relalive positioning of the cells                //
// saves total memory allocation needed for the system of cells.                  //
////////////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TRef
#include "TRef.h"
#endif

class TFoamVect;


class TFoamCell : public TObject {
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
   Double_t fPrimary;               // Primary integral, only for MC generation
   //////////////////////////////////////////////////////////////////////////////////////
   //                           METHODS                                                //
   //////////////////////////////////////////////////////////////////////////////////////
public:
   TFoamCell();                          // Default Constructor for ROOT streamers
   TFoamCell(Int_t);                     // User Constructor
   TFoamCell(TFoamCell &);               // Copy Constructor
   virtual ~TFoamCell();                 // Destructor
   void  Fill(Int_t, TFoamCell*, TFoamCell*, TFoamCell*);    // Assigns values of attributes
   TFoamCell&  operator=(const TFoamCell&);       // Substitution operator (never used)
   //--------------- Geometry ----------------------------------
   Double_t  GetXdiv() const { return fXdiv;}          // Pointer to Xdiv
   Int_t     GetBest() const { return fBest;}          // Pointer to Best
   void      SetBest(Int_t    Best){ fBest =Best;}     // Set Best edge candidate
   void      SetXdiv(Double_t Xdiv){ fXdiv =Xdiv;}     // Set x-division for best edge cand.
   void      GetHcub(  TFoamVect&, TFoamVect&) const;  // Get position and size vectors (h-cubical subspace)
   void      GetHSize( TFoamVect& ) const;             // Get size only of cell vector  (h-cubical subspace)
   //--------------- Integrals/Volumes -------------------------
   void      CalcVolume();                             // Calculates volume of cell
   Double_t  GetVolume() const { return fVolume;}      // Volume of cell
   Double_t  GetIntg() const { return fIntegral;}      // Get Integral
   Double_t  GetDriv() const { return fDrive;}         // Get Drive
   Double_t  GetPrim() const { return fPrimary;}       // Get Primary
   void      SetIntg(Double_t Intg){ fIntegral=Intg;}  // Set true integral
   void      SetDriv(Double_t Driv){ fDrive   =Driv;}  // Set driver integral
   void      SetPrim(Double_t Prim){ fPrimary =Prim;}  // Set primary integral
   //--------------- linked tree organization ------------------
   Int_t     GetStat() const { return fStatus;}        // Get Status
   void      SetStat(Int_t Stat){ fStatus=Stat;}       // Set Status
   TFoamCell* GetPare() const { return (TFoamCell*) fParent.GetObject(); }  // Get Pointer to parent cell
   TFoamCell* GetDau0() const { return (TFoamCell*) fDaught0.GetObject(); } // Get Pointer to 1-st daughter vertex
   TFoamCell* GetDau1() const { return (TFoamCell*) fDaught1.GetObject(); } // Get Pointer to 2-nd daughter vertex
   void      SetDau0(TFoamCell* Daug){ fDaught0 = Daug;}  // Set pointer to 1-st daughter
   void      SetDau1(TFoamCell* Daug){ fDaught1 = Daug;}  // Set pointer to 2-nd daughter
   void      SetSerial(Int_t Serial){ fSerial=Serial;}    // Set serial number
   Int_t     GetSerial() const { return fSerial;}         // Get serial number
   //--- other ---
   void Print(Option_t *option) const ;                   // Prints cell content
   ////////////////////////////////////////////////////////////////////////////
   ClassDef(TFoamCell,1)  //Single cell of FOAM
};
/////////////////////////////////////////////////////////////////////////////
#endif
