// @(#)root/geom:$Name:  $:$Id: TGeoVoxelFinder.h,v 1.4 2002/09/27 16:16:06 brun Exp $
// Author: Andrei Gheata   04/02/02

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoVoxelFinder
#define ROOT_TGeoVoxelFinder

#ifndef ROOT_TObject
#include "TObject.h"
#endif

/*************************************************************************
 * TGeoVoxelFinder - finder class handling voxels 
 *  
 *************************************************************************/

class TGeoVoxelFinder : public TObject
{
protected:
   TGeoVolume      *fVolume;          // volume to which applies

   Int_t             fNcandidates;    // ! number of candidates
   Int_t             fCurrentVoxel;   // ! index of current voxel in sorted list
   Int_t             fIbx;            // number of different boundaries on X axis
   Int_t             fIby;            // number of different boundaries on Y axis
   Int_t             fIbz;            // number of different boundaries on Z axis
   Int_t             fNboxes;         // length of boxes array
   Int_t             fNox;            // length of array of X offsets
   Int_t             fNoy;            // length of array of Y offsets
   Int_t             fNoz;            // length of array of Z offsets
   Int_t             fNx;             // length of array of X voxels
   Int_t             fNy;             // length of array of Y voxels
   Int_t             fNz;             // length of array of Z voxels
   Int_t             fPriority[3];    // priority for each axis
   Int_t             fSlices[3];      // ! slice indices for current voxel
   Double_t         *fBoxes;          //[fNboxes] list of bounding boxes
   Double_t         *fXb;             //[fIbx] ordered array of X box boundaries
   Double_t         *fYb;             //[fIby] ordered array of Y box boundaries
   Double_t         *fZb;             //[fIbz] ordered array of Z box boundaries
   Int_t            *fOBx;            //[fNox] offsets of daughter indices for slices X
   Int_t            *fOBy;            //[fNoy] offsets of daughter indices for slices Y
   Int_t            *fOBz;            //[fNoz] offsets of daughter indices for slices Z
   Int_t            *fIndX;           //[fNx] indices of daughters inside boundaries X
   Int_t            *fIndY;           //[fNy] indices of daughters inside boundaries Y
   Int_t            *fIndZ;           //[fNz] indices of daughters inside boundaries Z
   Int_t            *fCheckList;      //! list of candidates
public :
   TGeoVoxelFinder();
   TGeoVoxelFinder(TGeoVolume *vol);
   virtual ~TGeoVoxelFinder();
   virtual void        BuildVoxelLimits();
   void                CreateCheckList();
   void                DaughterToMother(Int_t id, Double_t *local, Double_t *master) const;
   virtual Double_t    Efficiency();
   virtual Int_t      *GetCheckList(Double_t *point, Int_t &nelem);
   Bool_t              GetIndices(Double_t *point);
   virtual Bool_t      GetNextIndices(Double_t *point, Double_t *dir);
   Int_t               GetPriority(Int_t iaxis) const {return fPriority[iaxis];}
   Int_t               GetNcandidates() const         {return fNcandidates;}
   virtual Int_t      *GetNextVoxel(Double_t *point, Double_t *dir, Int_t &ncheck);
   virtual void        FindOverlaps(Int_t inode) const;
   virtual void        Print(Option_t *option="") const;
   void                PrintVoxelLimits(Double_t *point) const;
   Bool_t              Intersect(Int_t n1, Int_t *array1, Int_t n2, Int_t *array2,
                             Int_t n3, Int_t *array3, Int_t &nf, Int_t *result); 
   void                IntersectAndStore(Int_t n1, Int_t *array1, Int_t n2, Int_t *array2,
                             Int_t n3, Int_t *array3);
   virtual void        SortAll(Option_t *option="");
   void                SortCrossedVoxels(Double_t *point, Double_t *dir);
   Bool_t              Union(Int_t n1, Int_t *array1, Int_t n2, Int_t *array2,
                             Int_t n3, Int_t *array3);
   virtual void        Voxelize(Option_t *option="");

  ClassDef(TGeoVoxelFinder, 1)                // voxel finder class
};

/*************************************************************************
 * TGeoCylVoxels - Cylindrical voxels class 
 *  
 *************************************************************************/

class TGeoCylVoxels : public TGeoVoxelFinder
{
private:

public:
   TGeoCylVoxels();
   TGeoCylVoxels(TGeoVolume *vol);
   virtual ~TGeoCylVoxels();
   
   virtual void        BuildVoxelLimits();
   virtual Double_t    Efficiency();
   virtual void        FindOverlaps(Int_t inode) const;
   virtual Int_t      *GetCheckList(Double_t *point, Int_t &nelem);
   virtual Bool_t      GetNextIndices(Double_t *point, Double_t *dir);
   virtual Int_t      *GetNextVoxel(Double_t *point, Double_t *dir, Int_t &ncheck);
   Int_t               IntersectIntervals(Double_t vox1, Double_t vox2, Double_t phi1, Double_t phi2) const;
   virtual void        Print(Option_t *option="") const;
   virtual void        SortAll(Option_t *option="");
   virtual void        Voxelize(Option_t *option);

  ClassDef(TGeoCylVoxels, 1)                // cylindrical voxel class
};

#endif
