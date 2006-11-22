// @(#):$Name:  $:$Id: Exp $
// Author: Andrei Gheata   07/08/06

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TG4RootNavMgr
#define ROOT_TG4RootNavMgr


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TG4RootNavMgr                                                        //
//                                                                      //
// Manager class creating a G4Navigator based on a ROOT geometry.       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TGeoManager;
class TG4RootNavigator;
class TG4RootDetectorConstruction;
class TVirtualUserPostDetConstruction;

class TG4RootNavMgr : public TObject {

protected:
   TGeoManager          *fGeometry;   // Pointer to TGeo geometry
   TG4RootNavigator     *fNavigator;  // G4 navigator working with TGeo
   TG4RootDetectorConstruction *fDetConstruction; // G4 geometry built based on ROOT one
   Bool_t                fConnected;  // Flags connection to G4

   TG4RootNavMgr();
   TG4RootNavMgr(TGeoManager *geom);   

private:
   static TG4RootNavMgr *fRootNavMgr; // Static pointer to singleton

public:
   static TG4RootNavMgr *GetInstance(TGeoManager *geom=0);
   virtual ~TG4RootNavMgr();
   
   Bool_t                ConnectToG4();
   void                  Initialize(TVirtualUserPostDetConstruction *sdinit=0);
   void                  LocateGlobalPointAndSetup(Double_t *pt, Double_t *dir=0);

   //Test utilities
   void                  PrintG4State() const;
   void                  SetVerboseLevel(Int_t level);

   void                  SetNavigator(TG4RootNavigator *nav);
   TG4RootNavigator     *GetNavigator() const {return fNavigator;}
   TG4RootDetectorConstruction *GetDetConstruction() const {return fDetConstruction;}

   ClassDef(TG4RootNavMgr,0)  // Class crreating a G4Navigator based on ROOT geometry
};
#endif
