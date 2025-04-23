// $Id$
// Main authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/**************************************************************************
 * Copyright(c) 1998-2008, ALICE Experiment at CERN, all rights reserved. *
 * See http://aliceinfo.cern.ch/Offline/AliRoot/License.html for          *
 * full copyright notice.                                                 *
 **************************************************************************/

#ifndef AliEveITSDigitsInfo_H
#define AliEveITSDigitsInfo_H

// #include <TEveUtil.h>
#include <TObject.h>

#include <map>
#include <vector>

class TClonesArray;
/* #include <TClonesArray.h> */
class TTree;

class AliITSsegmentationSPD;
class AliITSsegmentationSDD;
class AliITSsegmentationSSD;
class AliITSDDLModuleMapSDD;

class AliRawReader;

/******************************************************************************/
// AliEveITSModuleSelection
/******************************************************************************/

class AliEveITSModuleSelection
{
public:
  AliEveITSModuleSelection();
  virtual ~AliEveITSModuleSelection() {}

  Int_t   GetType() const        { return fType;     }
  void    SetType(Int_t x)       { fType = x;        }
  Int_t   GetLayer() const       { return fLayer;    }
  void    SetLayer(Int_t x)      { fLayer = x;       }
  Float_t GetMinPhi() const      { return fMinPhi;   }
  void    SetMinPhi(Float_t x)   { fMinPhi = x;      }
  Float_t GetMaxPhi() const      { return fMaxPhi;   }
  void    SetMaxPhi(Float_t x)   { fMaxPhi = x;      }
  Float_t GetMinTheta() const    { return fMinTheta; }
  void    SetMinTheta(Float_t x) { fMinTheta = x;    }
  Float_t GetMaxTheta() const    { return fMaxTheta; }
  void    SetMaxTheta(Float_t x) { fMaxTheta = x;    }

  void    SetPhiRange  (Float_t x, Float_t y) { fMinPhi   = x; fMaxPhi   = y; }
  void    SetThetaRange(Float_t x, Float_t y) { fMinTheta = x; fMaxTheta = y; }

protected:
  Int_t    fType;      // Type of modules: 0 - SPD, 1 - SDD, 2 - SSD.
  Int_t    fLayer;     // Layer, 0 - inner SPD, 5 - outer SSD.
  Float_t  fMinPhi;    // Min phi.
  Float_t  fMaxPhi;    // Max phi.
  Float_t  fMinTheta;  // Min theta.
  Float_t  fMaxTheta;  // Max theta.

  ClassDef(AliEveITSModuleSelection, 0); // Helper for selecting a range of ITS modules by type, layer, phi and theta.
};


/******************************************************************************/
// AliEveITSDigitsInfo
/******************************************************************************/

class AliEveITSDigitsInfo : public TObject // , public TEveRefCnt
{
public:
  TTree*                   fTree;         // Tree from which the digits are read.

  AliITSsegmentationSPD*   fSegSPD;       // Segmentation of SPD.
  AliITSsegmentationSDD*   fSegSDD;       // Segmentation of SDD.
  AliITSsegmentationSSD*   fSegSSD;       // Segmentation of SSD.

  Int_t                    fSPDMinVal;    // Default low  limit for display of SPD digits.
  Int_t                    fSSDMinVal;    // Default low  limit for display of SSD digits.
  Int_t                    fSDDMinVal;    // Default low  limit for display of SDD digits.
  Int_t                    fSPDMaxVal;    // Default high limit for display of SPD digits.
  Int_t                    fSSDMaxVal;    // Default high limit for display of SSD digits.
  Int_t                    fSDDMaxVal;    // Default high limit for display of SDD digits.

  Int_t                    fSPDHighLim;   // Maximum value of SPD digits.
  Int_t                    fSDDHighLim;   // Maximum value of SDD digits.
  Int_t                    fSSDHighLim;   // Maximum value of SSD digits.

  Int_t                    fSPDScaleX[5]; // SPD cell-sizes along X for display of scaled-digits.
  Int_t                    fSPDScaleZ[5]; // SPD cell-sizes along Z for display of scaled-digits.
  Int_t                    fSDDScaleX[5]; // SDD cell-sizes along X for display of scaled-digits.
  Int_t                    fSDDScaleZ[5]; // SDD cell-sizes along Z for display of scaled-digits.
  Int_t                    fSSDScale [5]; // SSD cell-sizes for display of scaled-digits.

  static AliITSDDLModuleMapSDD *fgDDLMapSDD;  // Mapping DDL/module to SDD-module number.

  static TObjArray             *fgDeadModSPD; // Dead modules of SPD.
  static TObjArray             *fgDeadModSDD; // Dead modules of SDD.
  static TObjArray             *fgDeadModSSD; // Dead modules of SSD.

  AliEveITSDigitsInfo();
  virtual ~AliEveITSDigitsInfo();

  void SetTree(TTree* tree);
  void ReadRaw(AliRawReader* raw, Int_t mode);

  TClonesArray* GetDigits(Int_t moduleID, Int_t detector) const;

  void GetSPDLocalZ(Int_t j, Float_t& z) const;

  void GetModuleIDs(AliEveITSModuleSelection* sel, std::vector<UInt_t>& ids) const;

  Bool_t HasData(Int_t module, Int_t det_id) const;
  Bool_t IsDead (Int_t module, Int_t det_id) const;

  virtual void Print(Option_t* opt="") const;

protected:
  mutable std::map<Int_t,  TClonesArray*> fSPDmap; // Map from module-id to SPD data.
  mutable std::map<Int_t,  TClonesArray*> fSDDmap; // Map from module-id to SDD data.
  mutable std::map<Int_t,  TClonesArray*> fSSDmap; // Map from module-id to SSD data.

  void SetITSSegmentation();

private:
  Float_t fSPDZCoord[192]; // Precalculated z-coordinates for positions of digits.

  void InitInternals();

  AliEveITSDigitsInfo(const AliEveITSDigitsInfo&);            // Not implemented
  AliEveITSDigitsInfo& operator=(const AliEveITSDigitsInfo&); // Not implemented

  ClassDef(AliEveITSDigitsInfo, 0); // Stores ITS geometry information and event-data in format suitable for visualization.
};


inline void AliEveITSDigitsInfo::GetSPDLocalZ(Int_t j, Float_t& z) const
{
  z = fSPDZCoord[j];
}

#endif
