/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Author: Andrei Gheata   30/06/14

#ifndef ROOT_TGeoParallelWorld
#define ROOT_TGeoParallelWorld


#ifndef ROOT_TGeoVolume
#include "TGeoVolume.h"
#endif

// forward declarations
class TGeoManager;
class TGeoPhysicalNode;
class TGeoVolume;

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TGeoParallelWorld - base class for a flat world that can be navigated  //
//   in parallel                                                          //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

class TGeoParallelWorld : public TNamed
{
protected :
   TGeoManager       *fGeoManager;     //! base geometry
   TObjArray         *fPhysical;       // array of physical nodes
   TGeoVolume        *fVolume;         // helper volume
   Bool_t             fIsClosed;       // Closed flag
   Bool_t             fUseOverlaps;    // Activated if user defined overlapping candidates

   TGeoParallelWorld(const TGeoParallelWorld&); 
   TGeoParallelWorld& operator=(const TGeoParallelWorld&);

public:
   // constructors
   TGeoParallelWorld() : TNamed(),fGeoManager(0),fPhysical(0),fVolume(0),fIsClosed(kFALSE),fUseOverlaps(kFALSE) {}
   TGeoParallelWorld(const char *name, TGeoManager *mgr);

   // destructor
   virtual ~TGeoParallelWorld();
   // API for adding components nodes
//   void              AddNode(TGeoVolume *vol, TGeoMatrix *matrix=0);
   void              AddNode(TGeoPhysicalNode *pnode);
   // Adding overlap candidates can improve performance
   void              AddOverlap(TGeoVolume *vol);
   
   
   // Closing a parallel geometry is mandatory
   Bool_t            CloseGeometry();
   // Refresh structures in case of re-alignment
   void              RefreshPhysicalNodes();

   // Navigation interface
   TGeoPhysicalNode *FindNode(Double_t point[3]);
   TGeoPhysicalNode *FindNextBoundary(Double_t point[3], Double_t dir[3], Double_t &step, Double_t stepmax=1.E30);
   Double_t          Safety(Double_t point[3], Double_t safmax=1.E30);

   // Getters
   TGeoManager      *GetGeometry() const {return fGeoManager;}
   Bool_t            IsClosed() const    {return fIsClosed;}
   TGeoVolume       *GetVolume() const   {return fVolume;}
   
   // Utilities
   void              CheckOverlaps(Double_t ovlp=0.001); // default 10 microns
   void              Draw(Option_t *option);

   ClassDef(TGeoParallelWorld, 1)     // parallel world base clas
};

#endif

