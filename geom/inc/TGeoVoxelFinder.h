/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author : Andrei Gheata           date : Tue 30 Oct 2001 11:57:07 AM CET

#ifndef ROOT_TGeoVoxelFinder
#define ROOT_TGeoVoxelFinder

#ifndef ROOT_TGeoFinder
#include "TGeoFinder.h"
#endif
/*
#ifndef ROOT_TGeoMatrix
#include "TGeoMatrix.h"
#endif

// forward declarations
class TGeoVolume;
class TGeoVolumeDiv;
class TGeoNode;
class TGeoShape;
class TGeoMatrix;
*/
/*************************************************************************
 * TGeoVoxelFinder - finder class handling voxels 
 *  
 *************************************************************************/

class TGeoVoxelFinder : public TGeoFinder
{
private:
   TGeoVolume      *fVolume;          // volume to which applies

   Double_t         *fBoxes;          // list of bounding boxes
   Double_t         *fXb;             // ordered array of X box boundaries
   Double_t         *fYb;             // ordered array of Y box boundaries
   Double_t         *fZb;             // ordered array of Z box boundaries
   Int_t            *fOBx;            // offsets of daughter indices for slices X
   Int_t            *fOBy;            // offsets of daughter indices for slices Y
   Int_t            *fOBz;            // offsets of daughter indices for slices Z
   Int_t            *fIndX;           // indices of daughters inside boundaries X
   Int_t            *fIndY;           // indices of daughters inside boundaries Y
   Int_t            *fIndZ;           // indices of daughters inside boundaries Z
   Int_t            *fCheckList;      // list of candidates
   Int_t             fNcandidates;    // number of candidates
   Int_t             fCurrentVoxel;   // index of current voxel in sorted list
   Int_t             fIb[3];          // number of different boundaries on each axis
   Int_t             fPriority[3];    // priority for each axis
   Int_t             fSlices[3];      // slice indices for current voxel
public :
   TGeoVoxelFinder();
   TGeoVoxelFinder(TGeoVolume *vol);
   virtual ~TGeoVoxelFinder();
   void                BuildBoundingBoxes();
   virtual void        cd(Int_t idiv) {;}
   void                DaughterToMother(Int_t id, Double_t *local, Double_t *master);
   Int_t              *GetCheckList(Double_t *point, Int_t &nelem);
   Bool_t              GetIndices(Double_t *point);
   Bool_t              GetNextIndices(Double_t *point, Double_t *dir);
   virtual TGeoMatrix *GetMatrix()              {return 0;}
   Int_t               GetPriority(Int_t iaxis) {return fPriority[iaxis];}
   Int_t               GetNcandidates()         {return fNcandidates;}
   Int_t              *GetNextVoxel(Double_t *point, Double_t *dir, Int_t &ncheck);
   virtual TGeoNode   *FindNode(Double_t *point);
   void                FindOverlaps(Int_t inode);
   void                Print(Option_t *option="") const;
   void                PrintVoxelLimits(Double_t *point);
   Bool_t              Intersect(Int_t n1, Int_t *array1, Int_t n2, Int_t *array2,
                             Int_t n3, Int_t *array3, Int_t &nf, Int_t *result); 
   void                IntersectAndStore(Int_t n1, Int_t *array1, Int_t n2, Int_t *array2,
                             Int_t n3, Int_t *array3);
   static Bool_t       SearchFast(Int_t value, Int_t n, Int_t *array);
   virtual void        SetBasicVolume(TGeoVolume *vol) {} 
   void                SortBoxes(Option_t *option="");
   void                SortCrossedVoxels(Double_t *point, Double_t *dir);
   Bool_t              Union(Int_t n1, Int_t *array1, Int_t n2, Int_t *array2,
                             Int_t n3, Int_t *array3);
   void                Voxelize(Option_t *option="");

  ClassDef(TGeoVoxelFinder, 1)                // voxel finder class
};

#endif
