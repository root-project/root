// $Id: TFoamCell.h,v 1.2 2005/04/04 10:59:34 psawicki Exp $

#ifndef ROOT_TFoamCell
#define ROOT_TFoamCell

////////////////////////////////////////////////////////////////////////////////////
// Class TFoamCell  used in TFoam                                                    //
//                                                                                //
// Objects of this class are hyperrectangular cells organized in the binary tree. //
// Special algoritm for encoding relalive positioning of the cells                //
// saves total memory allocaction needed for the system of cells.                 //
////////////////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TRef.h"

#include "TFoamVect.h"

class TFoamCell : public TObject {
  //   static, the same for all cells!
 private:
  Short_t  fkDim;               // Dimension of the vector space
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
  TFoamCell&  operator=(TFoamCell&);       // Substitution operator (never used)
  //--------------- Geometry ----------------------------------
  Double_t  GetXdiv(void){  return fXdiv;};           // Pointer to Xdiv
  Int_t     GetBest(void){  return fBest;};           // Pointer to Best
  void      SetBest(Int_t    Best){ fBest =Best;};    // Set Best edge candidate
  void      SetXdiv(Double_t Xdiv){ fXdiv =Xdiv;};    // Set x-division for best edge cand.
  void      GetHcub(  TFoamVect&, TFoamVect&);        // Get position and size vectors (h-cubical subspace)
  void      GetHSize( TFoamVect& );                   // Get size only of cell vector  (h-cubical subspace)
  //--------------- Integrals/Volumes -------------------------
  void      CalcVolume(void);                         // Calculates volume of cell
  Double_t  GetVolume(void){ return fVolume;};        // Volume of cell
  Double_t  GetIntg(void){  return fIntegral;};       // Get Integral
  Double_t  GetDriv(void){  return fDrive;};          // Get Drive
  Double_t  GetPrim(void){  return fPrimary;};        // Get Primary
  void      SetIntg(Double_t Intg){ fIntegral=Intg;}; // Set true integral
  void      SetDriv(Double_t Driv){ fDrive   =Driv;}; // Set driver integral
  void      SetPrim(Double_t Prim){ fPrimary =Prim;}; // Set primary integral
  //--------------- linked tree organization ------------------
  Int_t     GetStat(void){ return fStatus;};          // Get Status
  void      SetStat(Int_t Stat){ fStatus=Stat;};      // Set Status
  TFoamCell* GetPare(void){ return  (TFoamCell*) fParent.GetObject();  } // Get Pointer to parent cell
  TFoamCell* GetDau0(void){ return  (TFoamCell*) fDaught0.GetObject(); } // Get Pointer to 1-st daughter vertex
  TFoamCell* GetDau1(void){ return (TFoamCell*)fDaught1.GetObject(); }   // Get Pointer to 2-nd daughter vertex
  void      SetDau0(TFoamCell* Daug){ fDaught0 = Daug;}; // Set pointer to 1-st daughter
  void      SetDau1(TFoamCell* Daug){ fDaught1 = Daug;}; // Set pointer to 2-nd daughter
  void      SetSerial(Int_t Serial){ fSerial=Serial;};   // Set serial number
  Int_t     GetSerial(void){ return fSerial;};           // Get serial number
  //--- other ---
  void PrintContent();                                   // Prints cell content
////////////////////////////////////////////////////////////////////////////
 ClassDef(TFoamCell,1); //Single cell of FOAM
};
/////////////////////////////////////////////////////////////////////////////
#endif
