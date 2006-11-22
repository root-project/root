// @(#)root/g4root:$Name:  $:$Id: TG4RootDetectorConstruction.h,v 1.1 2006/11/22 17:14:01 brun Exp $
// Author: Andrei Gheata   07/08/06

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TG4RootDetectorConstruction
#define ROOT_TG4RootDetectorConstruction


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TG4RootDetectorConstruction                                          //
//                                                                      //
// Builder creating a pseudo G4 geometry starting from a TGeo geometry. //
//                                                                      //
// To invoke the method Construct() the ROOT geometry must be in memory //
// The G4 objects created are:                                          //
//    TGeoElement               ---> G4Element                          //
//    TGeoMaterial/TGeoMixture  ---> G4Material                         //
//    TGeoMatrix                ---> G4RotationMatrix                   //
//    TGeoVolume                ---> G4LogicalVolume                    //
//    TGeoShape           ---> TG4RootSolid  : public G4Solid           //
//    TGeoNode            ---> G4PVPlacement : public G4VPhysicalVolume //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef G4VUserDetectorConstruction_h
#include "G4VUserDetectorConstruction.hh"
#endif

#ifndef G4ROTATIONMATRIX_HH
#include "G4RotationMatrix.hh"
#endif

#ifndef ROOT_TGeoManager
#include "TGeoManager.h"
#endif

#ifndef ROOT_TGeoNode
#include "TGeoNode.h"
#endif

#include <map>

class TObjArray;
class TGeoManager;
class TGeoMaterial;
class TGeoShape;
class TGeoVolume;
class TGeoMatrix;

class G4FieldManager;
class G4VSolid;
class G4LogicalVolume;
class G4VPhysicalVolume;
class G4Material;
class TVirtualUserPostDetConstruction;

class TG4RootDetectorConstruction : public G4VUserDetectorConstruction {

private:
   typedef std::map<const TGeoMaterial *, G4Material *>   G4MaterialMap_t;
   typedef G4MaterialMap_t::const_iterator            G4MaterialIt_t;
   typedef G4MaterialMap_t::value_type                G4MaterialVal_t;
   G4MaterialMap_t       fG4MaterialMap; //! map of G4 materials

   typedef std::map<const TGeoVolume *, G4LogicalVolume *> G4VolumeMap_t;
   typedef G4VolumeMap_t::const_iterator              G4VolumeIt_t;
   typedef G4VolumeMap_t::value_type                  G4VolumeVal_t;
   G4VolumeMap_t         fG4VolumeMap; //! map of G4 volumes
   typedef std::map<const G4LogicalVolume *, TGeoVolume *> VolumeMap_t;
   typedef VolumeMap_t::const_iterator                VolumeIt_t;
   typedef VolumeMap_t::value_type                    VolumeVal_t;
   VolumeMap_t           fVolumeMap; //! map of TGeo volumes

   typedef std::map<const TGeoNode *, G4VPhysicalVolume *> G4PVolumeMap_t;
   typedef G4PVolumeMap_t::const_iterator             G4PVolumeIt_t;
   typedef G4PVolumeMap_t::value_type                 G4PVolumeVal_t;
   G4PVolumeMap_t        fG4PVolumeMap; //! map of G4 physical volumes
   typedef std::map<const G4VPhysicalVolume *, TGeoNode *> PVolumeMap_t;
   typedef PVolumeMap_t::const_iterator               PVolumeIt_t;
   typedef PVolumeMap_t::value_type                   PVolumeVal_t;
   PVolumeMap_t          fPVolumeMap; //! map of TGeo volumes
protected:
   Bool_t                fIsConstructed;   // flag Construct() called
   TGeoManager          *fGeometry;        // TGeo geometry manager
   G4VPhysicalVolume    *fTopPV;           // World G4 physical volume
   TVirtualUserPostDetConstruction        *fSDInit;          // Sensitive detector hook
   // Geometry creators
   void                  CreateG4LogicalVolumes();
   void                  CreateG4Materials();
   void                  CreateG4Elements();
   void                  CreateG4PhysicalVolumes();
   // Converters TGeo->G4 for basic types
   G4VSolid             *CreateG4Solid(TGeoShape *shape);
   G4LogicalVolume      *CreateG4LogicalVolume(TGeoVolume *vol);
   G4VPhysicalVolume    *CreateG4PhysicalVolume(TGeoNode *node);
   G4Material           *CreateG4Material(const TGeoMaterial *mat);
   G4RotationMatrix     *CreateG4Rotation(const TGeoMatrix *matrix);

public:
   TG4RootDetectorConstruction();
   TG4RootDetectorConstruction(TGeoManager *geom);
   virtual ~TG4RootDetectorConstruction();
   
   virtual G4VPhysicalVolume *Construct();

   // Getters
   TGeoManager          *GetGeometryManager() const {return fGeometry;}
   G4VPhysicalVolume    *GetTopPV()           const {return fTopPV;}
   G4Material           *GetG4Material(const TGeoMaterial *mat)  const;
   G4LogicalVolume      *GetG4Volume(const TGeoVolume *vol)      const;
   TGeoVolume           *GetVolume(const G4LogicalVolume *g4vol) const;
   G4VPhysicalVolume    *GetG4VPhysicalVolume(const TGeoNode *node) const;
   TGeoNode             *GetNode(const G4VPhysicalVolume *g4vol) const;
   TVirtualUserPostDetConstruction        *GetSDInit() const {return fSDInit;}
   Bool_t                IsConstructed() const {return fIsConstructed;}

   void                  Initialize(TVirtualUserPostDetConstruction *sdinit=0);         

//   ClassDef(TG4RootDetectorConstruction,0)  // Class creating a G4 gometry based on ROOT geometry
};

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualUserPostDetConstruction  - abstract class for defining links to G4 geometry,   //
//   like sensitive detectors, G4 material properties, user cuts,...    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
 
class TVirtualUserPostDetConstruction {
public:
   TVirtualUserPostDetConstruction() {}
   virtual ~TVirtualUserPostDetConstruction() {}
   
   virtual void          Initialize(TG4RootDetectorConstruction *dc) = 0;
};   
#endif
