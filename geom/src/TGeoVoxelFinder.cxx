/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// Author :   date

////////////////////////////////////////////////////////////////////////////////
// Full description with examples and pictures
//
//
//
//
//Begin_Html
/*
<img src="gif/t_finder.jpg">
<img src="gif/t_voxelfind.jpg">
<img src="gif/t_voxtree.jpg">
*/
//End_Html
#include "TObject.h"
#include "TGeoMatrix.h"
#include "TGeoPara.h"
#include "TGeoArb8.h"
#include "TGeoNode.h"
#include "TGeoManager.h"

#include "TGeoVoxelFinder.h"

/*************************************************************************
 * TGeoVoxelFinder - finder class handling voxels 
 *  
 *************************************************************************/

ClassImp(TGeoVoxelFinder)


//-----------------------------------------------------------------------------
TGeoVoxelFinder::TGeoVoxelFinder()
{
// Default constructor
   fVolume  = 0;
   fBoxes    = 0;
   fXb       = 0;
   fYb       = 0;
   fZb       = 0;
   fOBx      = 0;
   fOBy      = 0;
   fOBz      = 0;
   fIndX      = 0;
   fIndY      = 0;
   fIndZ      = 0;
   fCheckList = 0;
   fNcandidates  = 0;
   fCurrentVoxel = 0;
}
//-----------------------------------------------------------------------------
TGeoVoxelFinder::TGeoVoxelFinder(TGeoVolume *vol)
{
// Default constructor
   if (!vol) return;
   fVolume  = vol;
   fBoxes    = 0;
   fXb       = 0;
   fYb       = 0;
   fZb       = 0;
   fOBx      = 0;
   fOBy      = 0;
   fOBz      = 0;
   fIndX      = 0;
   fIndY      = 0;
   fIndZ      = 0;
   fCheckList = 0;
   fNcandidates  = 0;
   fCurrentVoxel = 0;
}
//-----------------------------------------------------------------------------
TGeoVoxelFinder::~TGeoVoxelFinder()
{
// Destructor
//   printf("deleting finder of %s\n", fVolume->GetName());
   if (fOBx) delete [] fOBx;
   if (fOBy) delete [] fOBy;
   if (fOBz) delete [] fOBz;
//   printf("OBx OBy OBz...\n");
   if (fBoxes) delete fBoxes;
//   printf("boxes...\n");
   if (fXb) delete [] fXb;
   if (fYb) delete [] fYb;
   if (fZb) delete [] fZb;
//   printf("Xb Yb Zb...\n");
   if (fIndX) delete [] fIndX;
   if (fIndY) delete [] fIndY;
   if (fIndZ) delete [] fIndZ;
//   printf("IndX IndY IndZ...\n");
   if (fCheckList) delete [] fCheckList;
//   printf("checklist...\n");
}
//-----------------------------------------------------------------------------
void TGeoVoxelFinder::BuildBoundingBoxes()
{
// build the array of bounding boxes of the nodes inside
   Int_t nd = fVolume->GetNdaughters();
   if (!nd) return;
//   printf("building boxes for %s  nd=%i\n", fVolume->GetName(), nd);
   Int_t id;
   TGeoNode *node;
   if (fBoxes) delete fBoxes;
   fBoxes = new Double_t[6*nd];
   if (fCheckList) delete fCheckList;
   fCheckList = new Int_t[nd];
   Double_t vert[24];
   Double_t pt[3];
   Double_t xyz[6];
//   printf("boundaries for %s :\n", GetName());
   TGeoBBox *box = 0;
   for (id=0; id<nd; id++) {
      node = fVolume->GetNode(id);
//      if (!strcmp(node->ClassName(), "TGeoNodeOffset") continue;
      box = (TGeoBBox*)node->GetVolume()->GetShape();
      box->SetBoxPoints(&vert[0]);
      for (Int_t point=0; point<8; point++) {
         DaughterToMother(id, &vert[3*point], &pt[0]);
         if (!point) {
            xyz[0] = xyz[1] = pt[0];
            xyz[2] = xyz[3] = pt[1];
            xyz[4] = xyz[5] = pt[2];
            continue;
         }
         for (Int_t j=0; j<3; j++) {
            if (pt[j] < xyz[2*j]) xyz[2*j]=pt[j];
            if (pt[j] > xyz[2*j+1]) xyz[2*j+1]=pt[j];
         }
      }
      fBoxes[6*id] = (xyz[1]-xyz[0])/2.;   // dX
      fBoxes[6*id+1] = (xyz[3]-xyz[2])/2.; // dY
      fBoxes[6*id+2] = (xyz[5]-xyz[4])/2.; // dZ
      fBoxes[6*id+3] = (xyz[0]+xyz[1])/2.; // Ox
      fBoxes[6*id+4] = (xyz[2]+xyz[3])/2.; // Oy
      fBoxes[6*id+5] = (xyz[4]+xyz[5])/2.; // Oz
   }
}
//-----------------------------------------------------------------------------
void TGeoVoxelFinder::DaughterToMother(Int_t id, Double_t *local, Double_t *master)
{
// convert a point from the local reference system of node id to reference
// system of mother volume
   TGeoMatrix *mat = fVolume->GetNode(id)->GetMatrix();
   mat->LocalToMaster(local, master);
}
//-----------------------------------------------------------------------------
TGeoNode *TGeoVoxelFinder::FindNode(Double_t *point)
{
// find the node containing the query point
   return 0;
}
//-----------------------------------------------------------------------------
void TGeoVoxelFinder::FindOverlaps(Int_t inode)
{
// create the list of nodes for which the bboxes overlap with inode's bbox
   if (!fBoxes) return;
   Double_t xmin, xmax, ymin, ymax, zmin, zmax;
   Double_t xmin1, xmax1, ymin1, ymax1, zmin1, zmax1;
   Double_t ddx1, ddx2;
   Int_t nd = fVolume->GetNdaughters();
   Int_t *ovlps = 0;
   Int_t *otmp = new Int_t[nd-1]; 
   Int_t novlp = 0;
   TGeoNode *node = fVolume->GetNode(inode);
   xmin = fBoxes[6*inode+3] - fBoxes[6*inode];
   xmax = fBoxes[6*inode+3] + fBoxes[6*inode];
   ymin = fBoxes[6*inode+4] - fBoxes[6*inode+1];
   ymax = fBoxes[6*inode+4] + fBoxes[6*inode+1];
   zmin = fBoxes[6*inode+5] - fBoxes[6*inode+2];
   zmax = fBoxes[6*inode+5] + fBoxes[6*inode+2];
//   printf("overlaps for MANY node %s\n", node->GetName());

//   printf("xmin=%g  xmax=%g\n", xmin, xmax);
//   printf("ymin=%g  ymax=%g\n", ymin, ymax);
//   printf("zmin=%g  zmax=%g\n", zmin, zmax);
   Bool_t in = kFALSE;
   //TGeoNode *node1;
   // loop on brothers
   for (Int_t ib=0; ib<nd; ib++) {
      if (ib == inode) continue; // everyone overlaps with itself
      in = kFALSE;
      //node1 = fVolume->GetNode(ib);
      xmin1 = fBoxes[6*ib+3] - fBoxes[6*ib];
      xmax1 = fBoxes[6*ib+3] + fBoxes[6*ib];
      ymin1 = fBoxes[6*ib+4] - fBoxes[6*ib+1];
      ymax1 = fBoxes[6*ib+4] + fBoxes[6*ib+1];
      zmin1 = fBoxes[6*ib+5] - fBoxes[6*ib+2];
      zmax1 = fBoxes[6*ib+5] + fBoxes[6*ib+2];
//      printf(" node %s\n", node1->GetName());
//      printf("  xmin1=%g  xmax1=%g\n", xmin1, xmax1);
//      printf("  ymin1=%g  ymax1=%g\n", ymin1, ymax1);
//      printf("  zmin1=%g  zmax1=%g\n", zmin1, zmax1);


      ddx1 = TMath::Abs(xmin1-xmax);
      ddx2 = TMath::Abs(xmax1-xmin);
         if ((ddx1<1E-12)||(ddx2<1E-12)) continue;
//         if ((xmin1==xmin)||(xmax1==xmax)) in = kTRUE;
         if (((xmin1>xmin)&&(xmin1<xmax))||((xmax1>xmin)&&(xmax1<xmax)) ||
             ((xmin>xmin1)&&(xmin<xmax1))||((xmax>xmin1)&&(xmax<xmax1)))
                in = kTRUE;
      if (!in) continue;
//      printf("x overlap...\n");
      in = kFALSE;

      ddx1 = TMath::Abs(ymin1-ymax);
      ddx2 = TMath::Abs(ymax1-ymin);
         if ((ddx1<1E-12)||(ddx2<1E-12)) continue;
//         if ((ymin1==ymin)||(ymax1==ymax)) in = kTRUE;
         if (((ymin1>ymin)&&(ymin1<ymax))||((ymax1>ymin)&&(ymax1<ymax)) ||
             ((ymin>ymin1)&&(ymin<ymax1))||((ymax>ymin1)&&(ymax<ymax1)))
                in = kTRUE;
//      if (!in) continue;
//      printf("y overlap...\n");
      in = kFALSE;

      ddx1 = TMath::Abs(zmin1-zmax);
      ddx2 = TMath::Abs(zmax1-zmin);
         if ((ddx1<1E-12)||(ddx2<1E-12)) continue;
//         if ((zmin1==zmin)||(zmax1==zmax)) in = kTRUE;
         if (((zmin1>zmin)&&(zmin1<zmax))||((zmax1>zmin)&&(zmax1<zmax)) ||
             ((zmin>zmin1)&&(zmin<zmax1))||((zmax>zmin1)&&(zmax<zmax1)))
                in = kTRUE;
      if (!in) continue;
//      printf("Overlapping %i\n", ib);
      otmp[novlp++] = ib;
   }
   if (!novlp) {
//      printf("---no overlaps for MANY node %s\n", node->GetName());
      node->SetOverlaps(ovlps, 1);
      return;
   }
   ovlps = new Int_t[novlp];
   memcpy(ovlps, otmp, novlp*sizeof(Int_t));
   delete otmp;
   node->SetOverlaps(ovlps, novlp);
//   printf("Overlaps for MANY node %s : %i\n", node->GetName(), novlp);
}
//-----------------------------------------------------------------------------
Bool_t TGeoVoxelFinder::GetIndices(Double_t *point)
{
// Getindices for current slices on x, y, z
   fSlices[0] = -2; // -2 means 'all daughters in slice'
   fSlices[1] = -2;
   fSlices[2] = -2;
   Bool_t flag=kTRUE;
   if (fPriority[0]) {
      fSlices[0] = TMath::BinarySearch(fIb[0], fXb, point[0]);
      if ((fSlices[0]<0) || (fSlices[0]>=fIb[0]-1)) {
         flag=kFALSE;
      } else {   
         if (fPriority[0]==2) 
            if (!fIndX[fOBx[fSlices[0]]]) flag = kFALSE;
      }
   }   
   if (fPriority[1]) {
      fSlices[1] = TMath::BinarySearch(fIb[1], fYb, point[1]);
      if ((fSlices[1]<0) || (fSlices[1]>=fIb[1]-1)) {
         flag=kFALSE;
      } else {   
         if (fPriority[1]==2) 
            if (!fIndY[fOBy[fSlices[1]]]) flag = kFALSE;
      }
   }   
   if (fPriority[2]) {
      fSlices[2] = TMath::BinarySearch(fIb[2], fZb, point[2]);
      if ((fSlices[2]<0) || (fSlices[2]>=fIb[2]-1)) return kFALSE;
      if (fPriority[2]==2) {
         if (!fIndZ[fOBz[fSlices[2]]]) return kFALSE;
      }
   }       
   return flag;
}
//-----------------------------------------------------------------------------
Bool_t TGeoVoxelFinder::GetNextIndices(Double_t *point, Double_t *dir)
{
// Get indices for next voxel
   Int_t dind[3];
   memcpy(&dind[0], &fSlices[0], 3*sizeof(Int_t));
   Double_t dmin[3];
//   printf("GetNextIndices current slices : %i %i %i\n", fSlices[0], fSlices[1], fSlices[2]);
   dmin[0] = dmin[1] = dmin[2] = TGeoShape::kBig;
   TGeoBBox *box = (TGeoBBox*)fVolume->GetShape();
   Double_t limit = TGeoShape::kBig;
   Bool_t isXlimit=kFALSE, isYlimit=kFALSE, isZlimit=kFALSE;
   Double_t dmstep = gGeoManager->GetStep();   
//   printf("dmstep=%f\n", dmstep);
   if (dir[0]!=0) {
      if (fSlices[0]!=-2) {
      // if there are slices on this axis, get distance to next slice.
         dind[0]+=(dir[0]<0)?-1:1;
         if (dind[0]<-1) return kFALSE;
         if (dind[0]>fIb[0]-1) return kFALSE;
//         printf("next slicex=%i : x= %f  point[0]=%f dir[0]=%f\n",dind[0],fXb[dind[0]+((dind[0]>fSlices[0])?0:1)], point[0], dir[0]); 
         dmin[0] = (fXb[dind[0]+((dind[0]>fSlices[0])?0:1)]-point[0])/dir[0];
//         printf("dx=%f\n", dmin[0]);
         isXlimit = (dmin[0]>dmstep)?kTRUE:kFALSE;
      } else {
      // if no slicing on this axis, get distance to mother limit
         limit = (box->GetOrigin())[0] + box->GetDX()*((dind[0]<0)?-1:1);
         dmin[0] = (limit-point[0])/dir[0];
         isXlimit = kTRUE;
      }
   }      
   if (dir[1]!=0) {
      if (fSlices[1]!=-2) {
      // if there are slices on this axis, get distance to next slice.
         dind[1]+=(dir[1]<0)?-1:1;
         if (dind[1]<-1) return kFALSE;
         if (dind[1]>fIb[1]-1) return kFALSE;
//         printf("next slicey=%i : y= %f  point[1]=%f dir[1]=%f\n",dind[1], fYb[dind[1]+((dind[1]>fSlices[1])?0:1)], point[1], dir[1]); 
         dmin[1] = (fYb[dind[1]+((dind[1]>fSlices[1])?0:1)]-point[1])/dir[1];
//         printf("dy=%f\n", dmin[1]);
         isYlimit = (dmin[1]>dmstep)?kTRUE:kFALSE;
      } else {
      // if no slicing on this axis, get distance to mother limit
         limit = (box->GetOrigin())[1] + box->GetDY()*((dind[1]<0)?-1:1);
         dmin[1] = (limit-point[1])/dir[1];
         isYlimit = kTRUE;
      }
   }      
   if (dir[2]!=0) {
      if (fSlices[2]!=-2) {
      // if there are slices on this axis, get distance to next slice.
         dind[2]+=(dir[2]<0)?-1:1;
         if (dind[2]<-1) return kFALSE;
         if (dind[2]>fIb[2]-1) return kFALSE;
//         printf("next slicez=%i : z= %f  point[2]=%f dir[2]=%f\n",dind[2],fZb[dind[2]+((dind[2]>fSlices[2])?0:1)], point[2], dir[2]); 
         dmin[2] = (fZb[dind[2]+((dind[2]>fSlices[2])?0:1)]-point[2])/dir[2];
//         printf("dz=%f\n", dmin[2]);
         isZlimit = (dmin[2]>dmstep)?kTRUE:kFALSE;
      } else {
      // if no slicing on this axis, get distance to mother limit
         limit = (box->GetOrigin())[2] + box->GetDZ()*((dind[2]<0)?-1:1);
         dmin[2] = (limit-point[2])/dir[2];
         isZlimit = kTRUE;
      }
   }      
   // now find minimum distance
//   printf("dmin : %f %f %f\n", dmin[0], dmin[1], dmin[2]);
   if (dmin[0]<dmin[1]) {
      if (dmin[0]<dmin[2]) {
      // next X
//         printf("next slicex : %i isXlimit=%i\n", dind[0],(Int_t)isXlimit);
         if (isXlimit) return kFALSE;
         if (dind[0]<0) return kFALSE;
         if (dind[0]>fIb[0]-2) return kFALSE;
         fSlices[0] = dind[0];
         if (fSlices[1]<0) return GetNextIndices(point, dir);
         if (fSlices[1]>fIb[1]-2) return GetNextIndices(point, dir);
         if (fSlices[2]<0) return GetNextIndices(point, dir);
         if (fSlices[2]>fIb[2]-2) return GetNextIndices(point, dir);
         if (fIndX[fOBx[fSlices[0]]]>0) return kTRUE;
         return GetNextIndices(point, dir);
      } else {
      // next Z
//         printf("next slicez : %i isZlimit=%i\n", dind[2],(Int_t)isZlimit);
         if (isZlimit) return kFALSE;
         if (dind[2]<0) return kFALSE;
         if (dind[2]>fIb[2]-2) return kFALSE;
         fSlices[2] = dind[2];
         if (fSlices[0]<0) return GetNextIndices(point, dir);
         if (fSlices[0]>fIb[0]-2) return GetNextIndices(point, dir);
         if (fSlices[1]<0) return GetNextIndices(point, dir);
         if (fSlices[1]>fIb[1]-2) return GetNextIndices(point, dir);
         if (fIndZ[fOBz[fSlices[2]]]>0) return kTRUE;
         return GetNextIndices(point, dir);
      }
   } else {
      if (dmin[1]<dmin[2]) {   
      // next Y
//         printf("next slicey : %i isYlimit=%i\n", dind[1], (Int_t)isYlimit);
         if (isYlimit) return kFALSE;
         if (dind[1]<0) return kFALSE;
         if (dind[1]>fIb[1]-2) return kFALSE;
         fSlices[1] = dind[1];
         if (fSlices[0]<0) return GetNextIndices(point, dir);
         if (fSlices[0]>fIb[0]-2) return GetNextIndices(point, dir);
         if (fSlices[2]<0) return GetNextIndices(point, dir);
         if (fSlices[2]>fIb[2]-2) return GetNextIndices(point, dir);
         if (fIndY[fOBy[fSlices[1]]]>0) return kTRUE;
         return GetNextIndices(point, dir);
      } else {
      // next Z
//         printf("next slicez : %i isZlimit=%i\n", dind[2], (Int_t)isZlimit);
         if (isZlimit) return kFALSE;
         if (dind[2]<0) return kFALSE;
         if (dind[2]>fIb[2]-2) return kFALSE;
         fSlices[2] = dind[2];
         if (fSlices[0]<0) return GetNextIndices(point, dir);
         if (fSlices[0]>fIb[0]-2) return GetNextIndices(point, dir);
         if (fSlices[1]<0) return GetNextIndices(point, dir);
         if (fSlices[1]>fIb[1]-2) return GetNextIndices(point, dir);
         if (fIndZ[fOBz[fSlices[2]]]>0) return kTRUE;
         return GetNextIndices(point, dir);
      }
   }
   return kFALSE;  // never happens
}

//-----------------------------------------------------------------------------
void TGeoVoxelFinder::SortCrossedVoxels(Double_t *point, Double_t *dir)
{
// get the list in the next voxel crossed by a ray
   fCurrentVoxel = 0;
//   printf("###Sort crossed voxels for %s\n", fVolume->GetName());
   if (!GetIndices(point)) {
      fNcandidates = 0;
      UInt_t  loc = 2+((UInt_t)fVolume->GetNdaughters())/8;
      UChar_t *bits = gGeoManager->GetBits() + loc;
      memset(bits, 0, (loc+1)*sizeof(UChar_t));
//      printf("   no candidates in first voxel\n");
      return;
   }
//   printf("   current slices : %i   %i  %i\n", fSlices[0], fSlices[1], fSlices[2]);
   Int_t nd[3];
   memset(&nd[0], 0, 3*sizeof(Int_t));
   Int_t *slicex = 0;
   if (fSlices[0]!=-2) {
      nd[0] = fIndX[fOBx[fSlices[0]]];
      slicex=&fIndX[fOBx[fSlices[0]]+1];
   }   
   Int_t *slicey = 0;
   if (fSlices[1]!=-2) {
      nd[1] = fIndY[fOBy[fSlices[1]]];
      slicey=&fIndY[fOBy[fSlices[1]]+1];
   }   
   Int_t *slicez = 0;
   if (fSlices[2]!=-2) {
      nd[2] = fIndZ[fOBz[fSlices[2]]];
      slicez=&fIndZ[fOBz[fSlices[2]]+1];
   } 
//   printf("Ndaughters in first voxel : %i %i %i\n", nd[0], nd[1], nd[2]);
   IntersectAndStore(nd[0], slicex, nd[1], slicey, nd[2], slicez);  
//   printf("   candidates for first voxel :\n");
//   for (Int_t i=0; i<fNcandidates; i++) printf("    %i\n", fCheckList[i]);
}   
//-----------------------------------------------------------------------------
Int_t *TGeoVoxelFinder::GetCheckList(Double_t *point, Int_t &nelem)
{
// get the list of daughter indices for which point is inside their bbox
   if (!fBoxes) return 0;
   Bool_t one_daughter = kFALSE;
   if (fVolume->GetNdaughters() == 1) {
      fCheckList[0] = 0;
      nelem = 1;
      one_daughter = kTRUE;
   }
   Int_t *slicex = 0;
   Int_t *slicey = 0; 
   Int_t *slicez = 0;
   Int_t nd[3];
   memset(&nd[0], 0, 3*sizeof(Int_t));
   Int_t im;
   if (fPriority[0]) {
      im = TMath::BinarySearch(fIb[0], fXb, point[0]);
      if ((im==-1) || (im==fIb[0]-1)) return 0;
      if (fPriority[0]==2) {
         nd[0] = fIndX[fOBx[im]];
         if (!nd[0]) return 0;
         slicex = &fIndX[fOBx[im]+1];
      }   
   }

   if (fPriority[1]) {
      im = TMath::BinarySearch(fIb[1], fYb, point[1]);
      if ((im==-1) || (im==fIb[1]-1)) return 0;
      if (fPriority[1]==2) {
         nd[1] = fIndY[fOBy[im]];
         if (!nd[1]) return 0;
         slicey = &fIndY[fOBy[im]+1];
      }   
   }

   if (fPriority[2]) {
      im = TMath::BinarySearch(fIb[2], fZb, point[2]);
      if ((im==-1) || (im==fIb[2]-1)) return 0;
      if (fPriority[2]==2) {
         nd[2] = fIndZ[fOBz[im]];
         if (!nd[2]) return 0;
         slicez = &fIndZ[fOBz[im]+1];
      }   
   }
   if (one_daughter) return fCheckList;
   nelem = 0;
//   Int_t i = 0;

   if (Intersect(nd[0], slicex, nd[1], slicey, nd[2], slicez, nelem, fCheckList)) 
      return fCheckList;
   return 0;   
}
//-----------------------------------------------------------------------------
Int_t *TGeoVoxelFinder::GetNextVoxel(Double_t *point, Double_t *dir, Int_t &ncheck)
{
// get the list of new candidates for the next voxel crossed by current ray
//   printf("### GetNextVoxel\n");
   Int_t *list = fCheckList;
   ncheck = fNcandidates;
   if (fCurrentVoxel==0) {
      fCurrentVoxel++;
      return fCheckList;
   }
   fCurrentVoxel++;
   // Get slices for next voxel
   if (!GetNextIndices(point, dir)) {
//      printf("exit\n");
      ncheck = 0;
      return 0;
   }   
//   printf("   next slices : %i   %i  %i\n", fSlices[0], fSlices[1], fSlices[2]);
   Int_t nd[3];
   memset(&nd[0], 0, 3*sizeof(Int_t));
   Int_t *slicex = 0;
   if (fSlices[0]!=-2) {
      nd[0] = fIndX[fOBx[fSlices[0]]];
      slicex=&fIndX[fOBx[fSlices[0]]+1];
//      printf("x: %i %x\n", nd[0], (UInt_t)slicex);
   }   
   Int_t *slicey = 0;
   if (fSlices[1]!=-2) {
      nd[1] = fIndY[fOBy[fSlices[1]]];
      slicey=&fIndY[fOBy[fSlices[1]]+1];
//      printf("y: %i %x\n", nd[1], (UInt_t)slicey);
   }   
   Int_t *slicez = 0;
   if (fSlices[2]!=-2) {
      nd[2] = fIndZ[fOBz[fSlices[2]]];
      slicez=&fIndZ[fOBz[fSlices[2]]+1];
//      printf("z: %i %x\n", nd[2], (UInt_t)slicez);
   } 
   
   if (Union(nd[0], slicex, nd[1], slicey, nd[2], slicez)) {
      list += ncheck; 
//      printf("   new candidates : %i-%i\n", fNcandidates, ncheck);
      ncheck = fNcandidates - ncheck;
//      for (Int_t i=0; i<ncheck; i++) printf("    %i\n", list[i]);
      return list;
   }
//   printf("No new candidates\n");
   ncheck = 0;
   return list;
}   
//-----------------------------------------------------------------------------
Bool_t TGeoVoxelFinder::Union(Int_t n1, Int_t *array1, 
                              Int_t n2, Int_t *array2, 
                              Int_t n3, Int_t *array3)
{
// make the union of fCheckList with the result of the intersection of the 3 arrays
   // first reset bits
/*
   if (array1) {
      printf("   X slice :\n");
      for (Int_t i=0; i<n1; i++) printf("   %i\n", array1[i]);
   }   
   if (array2) {
      printf("   Y slice :\n");
      for (Int_t i=0; i<n2; i++) printf("   %i\n", array2[i]);
   }   
   if (array3) {
      printf("   Z slice :\n");
      for (Int_t i=0; i<n3; i++) printf("   %i\n", array3[i]);
   }   
*/
   Int_t nd = fVolume->GetNdaughters();
//   printf("Nd=%i, nx=%i ny=%i nz=%i\n", nd, n1,n2,n3); 
   Bool_t is_eff[3];
   memset(&is_eff[0], 0, 3*sizeof(Bool_t));
   Int_t ni[3];
   ni[0] = n1;
   ni[1] = n2;
   ni[2] = n3;
   
   Int_t i;
//   if (nd>10) {
      Int_t mins=nd;
      if (array1) {
         mins = n1;
         is_eff[0] = kTRUE;
      }   
      if (array2) {
         if (n2<mins) mins=n2;
         is_eff[1] = kTRUE;
      }   
      if (array3) {
         if (n3<mins) mins=n3;
         is_eff[2] = kTRUE;
      }   
      mins*=10;
      for (i=0; i<3; i++) {
         if (is_eff[i]) {
            if (ni[i]>mins) is_eff[i]=kFALSE;      
         }
      }
//   } else {
//      is_eff[0] = (array1!=0);
//      is_eff[1] = (array2!=0);
//      is_eff[2] = (array3!=0);
//   }         
//   if ((nd<n1) || (nd<n2) || (nd<n3)) {printf("Woops %s nd=%i\n", fVolume->GetName(), nd);
//      printf("%s\n", gGeoManager->GetPath()); exit(0);}
   UInt_t  loc = 2+((UInt_t)nd)/8;
   UChar_t *bits = gGeoManager->GetBits();
   UChar_t *pbits = bits+loc;
   UChar_t *tbits = bits+2*loc;
//   memset(pbits, 0, (loc+1)*sizeof(UChar_t));
   UChar_t bit = 0;
   UInt_t bitnumber = 0;
   UChar_t value = 0;
   Int_t *a1, *a2;
   Int_t nn1=0;
   Int_t nn2=0;
   Int_t maxval = -1;
   Int_t candidates = fNcandidates;
   Bool_t last = kFALSE;
   if (!is_eff[0]) {
      if (!is_eff[1]) {
         if (!is_eff[2]) return kFALSE;
         // test 3-rd array against old bits 
         for (i=0; i<n3; i++) {
            bitnumber = (UInt_t) (array3[i]);
            loc = bitnumber/8;
            value = pbits[loc];
            bit = bitnumber%8;
            // if not set, add new bit
            if ((value & (1<<bit)) == 0) {
               fCheckList[fNcandidates++] = array3[i];
               pbits[loc] |= (1<<bit);   
            }   
         }
         return kTRUE;
      }
      if (!is_eff[2]) {
         for (i=0; i<n2; i++) {
            bitnumber = (UInt_t) (array2[i]);
            loc = bitnumber/8;
            value = pbits[loc];
            bit = bitnumber%8;
            // if not set, add new bit
            if ((value & (1<<bit)) == 0) {
               fCheckList[fNcandidates++] = array2[i];
               pbits[loc] |= (1<<bit);   
            }   
         }
         return kTRUE;
      }   
      a1 = array2;
      nn1 = n2;
      a2 = array3;
      nn2 = n3;
      maxval = TMath::Max(array2[n2-1], array3[n3-1]);
      last = kTRUE;
   } else {
      maxval = array1[n1-1];   
      if (!is_eff[1]) {
         if (!is_eff[2]) {
            for (i=0; i<n1; i++) {
               bitnumber = (UInt_t) (array1[i]);
               loc = bitnumber/8;
               value = pbits[loc];
               bit = bitnumber%8;
               // if not set, add new bit
               if ((value & (1<<bit)) == 0) {
                  fCheckList[fNcandidates++] = array1[i];
                  pbits[loc] |= (1<<bit);   
               }   
            }
            return kTRUE;
         }
         a1 = array1;
         nn1 = n1;
         a2 = array3;
         nn2 = n3;
         maxval = TMath::Max(maxval, array3[n3-1]);
         last = kTRUE;
      } else {
         a1 = array1;
         nn1 = n1;
         a2 = array2;
         nn2 = n2;
         maxval = TMath::Max(array1[n1-1], array2[n2-1]);
         if (is_eff[2]) 
            maxval = TMath::Max(maxval, array3[n3-1]);
         else
            last = kTRUE;   
      }   
   }          
   if (maxval<0) return kFALSE;
   // reset bits
   loc = ((UInt_t)maxval)/8;
   memset(bits, 0, (loc+1)*sizeof(UChar_t));
   if (!last)
      memset(tbits, 0, (loc+1)*sizeof(UChar_t));
   for (i=0; i<nn1; i++) {
   // set bits according to first array
      bitnumber = (UInt_t) (a1[i]);
      loc = bitnumber/8;
      bit = bitnumber%8;
      bits[loc] |= (1<<bit);   
   }
   // test elements of second array
   UChar_t cbit=0;
   for (i=0; i<nn2; i++) {
      bitnumber = (UInt_t) (a2[i]);
      // test if bit is set
      loc = bitnumber/8;
      value = bits[loc];
      bit = bitnumber%8;
      cbit = 1<<bit;
      if ((value & cbit) != 0) {
      // element of second array was also in first array.
      // now check if it is already in the union list
         if ((pbits[loc] & cbit) == 0) {
//            printf("possible candidate : %i\n", a2[i]);
         // this is a new possible candidate (if also in third list) - mark it
            if (last) {
//               printf("is OK\n");
               pbits[loc] |= cbit;
               fCheckList[fNcandidates++] = a2[i];
            } else {
//               printf("set tbits\n");
               tbits[loc] |= cbit;
            }   
         }   
      }   
   }
//   if (fNcandidates==candidates) return kFALSE;
   if (last) return kTRUE;
   loc = ((UInt_t)maxval)/8;
   // test elements of third array
   for (i=0; i<n3; i++) {
      bitnumber = (UInt_t) (array3[i]);
      // test if bit is set
      loc = bitnumber/8;
      value = tbits[loc];
      bit = bitnumber%8;
      cbit = 1<<bit;
      if ((value & cbit) != 0) {
         fCheckList[fNcandidates++] = array3[i];
         pbits[loc] |= cbit;
      }   
   }
   if (fNcandidates==candidates) return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
void TGeoVoxelFinder::IntersectAndStore(Int_t n1, Int_t *array1, 
                                        Int_t n2, Int_t *array2, 
                                        Int_t n3, Int_t *array3)
{
   // first reset bits
   fNcandidates = 0;
   Int_t nd = fVolume->GetNdaughters();
//   printf("Nd=%i, nx=%i ny=%i nz=%i\n", nd, n1,n2,n3); 
   Bool_t is_eff[3];
   memset(&is_eff[0], 0, 3*sizeof(Bool_t));
   Int_t ni[3];
   ni[0] = n1;
   ni[1] = n2;
   ni[2] = n3;
   
   Int_t i;
//   if (nd>10) {
      Int_t mins=nd;
      if (array1) {
         mins = n1;
         is_eff[0] = kTRUE;
      }   
      if (array2) {
         if (n2<mins) mins=n2;
         is_eff[1] = kTRUE;
      }   
      if (array3) {
         if (n3<mins) mins=n3;
         is_eff[2] = kTRUE;
      }   
      mins*=10;
      for (i=0; i<3; i++) {
         if (is_eff[i]) {
            if (ni[i]>mins) is_eff[i]=kFALSE;      
         }
      }
   UInt_t  loc = 2+((UInt_t)nd)/8;
   UChar_t *bits = gGeoManager->GetBits();
   UChar_t *pbits = bits+loc;
   memset(pbits, 0, (loc+1)*sizeof(UChar_t));
   UChar_t bit = 0;
   UInt_t bitnumber = 0;
   UChar_t value = 0;
   Int_t *a1, *a2;
   Int_t nn1=0;
   Int_t nn2=0;
   Int_t maxval = -1;
//   Int_t i;
   Bool_t last = kFALSE;
   if (!is_eff[0]) {
      if (!is_eff[1]) {
         if (!is_eff[2]) return; 
         memcpy(fCheckList, array3, n3*sizeof(Int_t));
         for (i=0; i<n3; i++) {
            bitnumber = (UInt_t) (array3[i]);
            loc = bitnumber/8;
            bit = bitnumber%8;
            pbits[loc] |= (1<<bit);   
         }
         fNcandidates = n3;
         return;
      }
      if (!is_eff[2]) {
         memcpy(fCheckList, array2, n2*sizeof(Int_t));
         for (i=0; i<n2; i++) {
            bitnumber = (UInt_t) (array2[i]);
            loc = bitnumber/8;
            bit = bitnumber%8;
            pbits[loc] |= (1<<bit);   
         }
         fNcandidates = n2;
         return;
      }   
      a1 = array2;
      nn1 = n2;
      a2 = array3;
      nn2 = n3;
      maxval = TMath::Max(array2[n2-1], array3[n3-1]);
      last = kTRUE;
   } else {
      maxval = array1[n1-1];   
      if (!is_eff[1]) {
         if (!is_eff[2]) {
            memcpy(fCheckList, array1, n1*sizeof(Int_t));
            for (i=0; i<n1; i++) {
               bitnumber = (UInt_t) (array1[i]);
               loc = bitnumber/8;
               bit = bitnumber%8;
               pbits[loc] |= (1<<bit);   
            }
            fNcandidates = n1;
            return;
         }
         a1 = array1;
         nn1 = n1;
         a2 = array3;
         nn2 = n3;
         maxval = TMath::Max(maxval, array3[n3-1]);
         last = kTRUE;
      } else {
         a1 = array1;
         nn1 = n1;
         a2 = array2;
         nn2 = n2;
         maxval = TMath::Max(array1[n1-1], array2[n2-1]);
         if (is_eff[2]) maxval = TMath::Max(maxval, array3[n3-1]);
         else last=kTRUE;
      }   
   }          
   if (maxval<0) return;
   // reset bits
   loc = ((UInt_t)maxval)/8;
   memset(bits, 0, (loc+1)*sizeof(UChar_t));
   for (i=0; i<nn1; i++) {
   // set bits according to first array
      bitnumber = (UInt_t) (a1[i]);
      loc = bitnumber/8;
      bit = bitnumber%8;
      bits[loc] |= (1<<bit);   
   }
   // test elements of second array
   for (i=0; i<nn2; i++) {
      bitnumber = (UInt_t) (a2[i]);
      // test if bit is set
      loc = bitnumber/8;
      value = bits[loc];
      bit = bitnumber%8;
      if ((value & (1<<bit)) != 0) {
         fCheckList[fNcandidates++] = a2[i];
         if (last || (!array3)) pbits[loc] |= (1<<bit);
      }   
   }
   if (!fNcandidates) return;
   if (last) return;

   loc = ((UInt_t)maxval)/8;
   memset(bits, 0, (loc+1)*sizeof(UChar_t));
   for (i=0; i<fNcandidates; i++) {
   // set bits according to first result
      bitnumber = (UInt_t) (fCheckList[i]);
      loc = bitnumber/8;
      bit = bitnumber%8;
      bits[loc] |= (1<<bit);   
   }
   // reset result
   fNcandidates = 0;
   // test elements of third array
   for (i=0; i<n3; i++) {
      bitnumber = (UInt_t) (array3[i]);
      // test if bit is set
      loc = bitnumber/8;
      value = bits[loc];
      bit = bitnumber%8;
      if ((value & (1<<bit)) != 0) {
         fCheckList[fNcandidates++] = array3[i];
         pbits[loc] |= (1<<bit);
      }   
   }
   return;
}
//-----------------------------------------------------------------------------
Bool_t TGeoVoxelFinder::Intersect(Int_t n1, Int_t *array1, 
                                  Int_t n2, Int_t *array2, 
                                  Int_t n3, Int_t *array3, Int_t &nf, Int_t *result)
{
// return the intersection of three ordered lists
   Int_t nd = fVolume->GetNdaughters();
//   printf("Nd=%i, nx=%i ny=%i nz=%i\n", nd, n1,n2,n3); 
   Bool_t is_eff[3];
   memset(&is_eff[0], 0, 3*sizeof(Bool_t));
   Int_t ni[3];
   ni[0] = n1;
   ni[1] = n2;
   ni[2] = n3;
   
   Int_t i;
//   if (nd>10) {

      Int_t mins=nd;
      if (array1) {
         mins = n1;
         is_eff[0] = kTRUE;
      }   
      if (array2) {
         if (n2<mins) mins=n2;
         is_eff[1] = kTRUE;
      }   
      if (array3) {
         if (n3<mins) mins=n3;
         is_eff[2] = kTRUE;
      }   
      mins*=10;
      for (i=0; i<3; i++) {
         if (is_eff[i]) {
            if (ni[i]>mins) is_eff[i]=kFALSE;      
         }
      }

//   } else {
//      is_eff[0] = (array1!=0);
//      is_eff[1] = (array2!=0);
//      is_eff[2] = (array3!=0);
//   }         


   Int_t *a1, *a2;
   Int_t nn1=0;
   Int_t nn2=0;
   Int_t maxval = -1;
   Bool_t last = kFALSE;
   if (!is_eff[0]) {
      if (!is_eff[1]) {
         if (!is_eff[2]) return kTRUE; 
         memcpy(result, array3, n3*sizeof(Int_t));
         nf = n3;
         return kTRUE;
      }
      if (!is_eff[2]) {
         memcpy(result, array2, n2*sizeof(Int_t));
         nf = n2;
         return kTRUE;
      }   
      a1 = array2;
      nn1 = n2;
      a2 = array3;
      nn2 = n3;
      maxval = TMath::Max(array2[n2-1], array3[n3-1]);
      last = kTRUE;
   } else {
      maxval = array1[n1-1];   
      if (!is_eff[1]) {
         if (!is_eff[2]) {
            memcpy(result, array1, n1*sizeof(Int_t));
            nf = n1;
            return kTRUE;
         }
         a1 = array1;
         nn1 = n1;
         a2 = array3;
         nn2 = n3;
         maxval = TMath::Max(maxval, array3[n3-1]);
         last = kTRUE;
      } else {
         a1 = array1;
         nn1 = n1;
         a2 = array2;
         nn2 = n2;
         maxval = TMath::Max(array1[n1-1], array2[n2-1]);
         if (is_eff[2]) {
            maxval = TMath::Max(maxval, array3[n3-1]);
         } else last=kTRUE;   
      }   
   }          
   if (maxval<0) return kFALSE;
      
   UChar_t *bits = gGeoManager->GetBits();
   UInt_t  loc = 0;
   UChar_t bit = 0;
   UInt_t bitnumber = 0;
   UChar_t value = 0;
   // reset bits
   loc = ((UInt_t)maxval)/8;
   memset(bits, 0, (loc+1)*sizeof(UChar_t));
//   printf("%s intersecting %i with %i\n", fVolume->GetName(),nn1, nn2);
   for (i=0; i<nn1; i++) {
   // set bits according to first array
      bitnumber = (UInt_t) (a1[i]);
      loc = bitnumber/8;
      bit = bitnumber%8;
      bits[loc] |= (1<<bit);   
   }
   // test elements of second array
   for (i=0; i<nn2; i++) {
      bitnumber = (UInt_t) (a2[i]);
      // test if bit is set
      loc = bitnumber/8;
      value = bits[loc];
      bit = bitnumber%8;
      if ((value & (1<<bit)) != 0)
         result[nf++] = a2[i];
   }
//   printf("   result : %i\n", nf);
   if (!nf) return kFALSE;
   if (last) return kTRUE;

//   printf("    with : n3=%i\n", n3);
   loc = ((UInt_t)maxval)/8;
   memset(bits, 0, (loc+1)*sizeof(UChar_t));
   for (i=0; i<nf; i++) {
   // set bits according to first result
      bitnumber = (UInt_t) (result[i]);
      loc = bitnumber/8;
      bit = bitnumber%8;
      bits[loc] |= (1<<bit);   
   }
   // reset result
   nf = 0;
   // test elements of third array
   for (i=0; i<n3; i++) {
      bitnumber = (UInt_t) (array3[i]);
      // test if bit is set
      loc = bitnumber/8;
      value = bits[loc];
      bit = bitnumber%8;
      if ((value & (1<<bit)) != 0)
         result[nf++] = array3[i];
   }
//   printf("   result : %i\n", nf);
   if (!nf) return kFALSE;
   return kTRUE;
}
//-----------------------------------------------------------------------------
void TGeoVoxelFinder::SortBoxes(Option_t *option)
{
// order bounding boxes along x, y, z
   Int_t nd = fVolume->GetNdaughters();
   if (!nd) return;
//   printf("sorting boxes for %s  nd=%i\n", fVolume->GetName(), nd);
   Double_t *boundaries = new Double_t[6*nd];
   Double_t xmin, xmax, ymin, ymax, zmin, zmax;
   TGeoBBox *box = (TGeoBBox*)fVolume->GetShape();
   // compute ranges on X, Y, Z according to volume bounding box
   xmin = (box->GetOrigin())[0] - box->GetDX();
   xmax = (box->GetOrigin())[0] + box->GetDX();
   ymin = (box->GetOrigin())[1] - box->GetDY();
   ymax = (box->GetOrigin())[1] + box->GetDY();
   zmin = (box->GetOrigin())[2] - box->GetDZ();
   zmax = (box->GetOrigin())[2] + box->GetDZ();
   if ((xmin>=xmax) || (ymin>=ymax) || (zmin>=zmax)) {
      Error("SortBoxes", "wrong bounding box");
      printf("### volume was : %s\n", fVolume->GetName());
      return;
   }   
   Int_t id;
   // compute boundaries coordinates on X,Y,Z
   for (id=0; id<nd; id++) {
      // x boundaries
      boundaries[2*id] = fBoxes[6*id+3]-fBoxes[6*id];
      boundaries[2*id+1] = fBoxes[6*id+3]+fBoxes[6*id];
      // y boundaries
      boundaries[2*id+2*nd] = fBoxes[6*id+4]-fBoxes[6*id+1];
      boundaries[2*id+2*nd+1] = fBoxes[6*id+4]+fBoxes[6*id+1];
      // z boundaries
      boundaries[2*id+4*nd] = fBoxes[6*id+5]-fBoxes[6*id+2];
      boundaries[2*id+4*nd+1] = fBoxes[6*id+5]+fBoxes[6*id+2];
   }
   Int_t *index = new Int_t[2*nd];
   Int_t *ind = new Int_t[(nd+1)*(nd+1)]; // ind[fOBx[i]] = ndghts in slice fInd[i]--fInd[i+1]
   Double_t *temp = new Double_t[2*nd];
   Int_t current = 0;
   Double_t xxmin, xxmax, xbmin, xbmax, ddx1, ddx2;
   // sort x boundaries
   Int_t ib = 0;
   TMath::Sort(2*nd, &boundaries[0], &index[0], kFALSE);
   // compact common boundaries
   for (id=0; id<2*nd; id++) {
      if (!ib) {temp[ib++] = boundaries[index[id]]; continue;}
      if (TMath::Abs(temp[ib-1]-boundaries[index[id]])>1E-10)
         temp[ib++] = boundaries[index[id]];
   }
   // now find priority
   if (ib < 2) {
      Error("SortBoxes", "less than 2 boundaries on X !");
      printf("### volume was : %s\n", fVolume->GetName());
      return;
   }   
   if (ib == 2) {
   // check range
      if (((temp[0]-xmin)<1E-10) && ((temp[1]-xmax)>-1E-10)) {
      // ordering on this axis makes no sense. Clear all arrays.
         fPriority[0] = 0;
         if (fIndX) delete[] fIndX; 
         fIndX = 0;
         if (fXb) delete[] fXb;   
         fXb = 0;
         if (fOBx) delete[] fOBx;  
         fOBx = 0;
      } else {
         fPriority[0] = 1; // all in one slice
      }
   } else {
      fPriority[0] = 2;    // check all
   }
   // store compacted boundaries
   if (fPriority[0]) {
      if (fXb) delete[] fXb;
      fXb = new Double_t[ib];
      memcpy(fXb, &temp[0], ib*sizeof(Double_t));
      fIb[0] = ib;   

      //now build the lists of nodes in each slice
      memset(ind, 0, (nd+1)*(nd+1)*sizeof(Int_t));
                     // ind[fOBx[i]+k] = index of dght k (k<ndghts)
      if (fOBx) delete fOBx;
      fOBx = 0;
      fOBx = new Int_t[fIb[0]-1]; // offsets in ind
      for (id=0; id<fIb[0]-1; id++) {
         fOBx[id] = current; // offset of dght list
         ind[current] = 0; // ndght in this slice
         xxmin = fXb[id];
         xxmax = fXb[id+1];
         for (Int_t ic=0; ic<nd; ic++) {
            xbmin = fBoxes[6*ic+3]-fBoxes[6*ic];   
            xbmax = fBoxes[6*ic+3]+fBoxes[6*ic];
            ddx1 = TMath::Abs(xbmin-xxmax);
            ddx2 = TMath::Abs(xbmax-xxmin);
            if ((xbmin==xxmin)||(xbmax==xxmax)) {
               ind[current]++;
               ind[current+ind[current]] = ic;
               continue;
            }
            if ((ddx1<1E-12)||(ddx2<1E-12)) continue;
            if (((xbmin>xxmin)&&(xbmin<xxmax))||((xbmax>xxmin)&&(xbmax<xxmax)) ||
                ((xxmin>xbmin)&&(xxmin<xbmax))||((xxmax>xbmin)&&(xxmax<xbmax))) {
               // daughter ic in interval
               ind[current]++;
               ind[current+ind[current]] = ic;
            }
         }
         current += ind[current]+1;
      }
      if (fIndX) delete fIndX;
      fIndX = new Int_t[current];
      memcpy(fIndX, &ind[0], current*sizeof(Int_t));
   }   

   // sort y boundaries
   ib = 0;
   TMath::Sort(2*nd, &boundaries[2*nd], &index[0], kFALSE);
   // compact common boundaries
   for (id=0; id<2*nd; id++) {
      if (!ib) {temp[ib++] = boundaries[2*nd+index[id]]; continue;}
      if (TMath::Abs(temp[ib-1]-boundaries[2*nd+index[id]])>1E-10) 
         temp[ib++]=boundaries[2*nd+index[id]];
   }
   // now find priority on Y
   if (ib < 2) {
      Error("SortBoxes", "less than 2 boundaries on Y !");
      printf("### volume was : %s\n", fVolume->GetName());
      return;
   }   
   if (ib == 2) {
   // check range
      if (((temp[0]-ymin)<1E-10) && ((temp[1]-ymax)>-1E-10)) {
      // ordering on this axis makes no sense. Clear all arrays.
         fPriority[1] = 0;
         if (fIndY) delete[] fIndY; 
         fIndY = 0;
         if (fYb) delete[] fYb;   
         fYb = 0;
         if (fOBy) delete[] fOBy;  
         fOBy = 0;
      } else {
         fPriority[1] = 1; // all in one slice
      }
   } else {
      fPriority[1] = 2;    // check all
   }
   if (fPriority[1]) {
      // store compacted boundaries
      if (fYb) delete fYb;
      fYb = new Double_t[ib];
      memcpy(fYb, &temp[0], ib*sizeof(Double_t));
      fIb[1] = ib;

      memset(ind, 0, (nd+1)*(nd+1)*sizeof(Int_t));
      current = 0;
      if (fOBy) delete fOBy;
      fOBy = 0;
      fOBy = new Int_t[fIb[1]-1]; // offsets in ind
      for (id=0; id<fIb[1]-1; id++) {
         fOBy[id] = current; // offset of dght list
         ind[current] = 0; // ndght in this slice
         xxmin = fYb[id];
         xxmax = fYb[id+1];
         for (Int_t ic=0; ic<nd; ic++) {
            xbmin = fBoxes[6*ic+4]-fBoxes[6*ic+1];   
            xbmax = fBoxes[6*ic+4]+fBoxes[6*ic+1];   
            ddx1 = TMath::Abs(xbmin-xxmax);
            ddx2 = TMath::Abs(xbmax-xxmin);
            if ((xbmin==xxmin)||(xbmax==xxmax)) {
               ind[current]++;
               ind[current+ind[current]] = ic;
               continue;
            }
            if ((ddx1<1E-12)||(ddx2<1E-12)) continue;  
            if (((xbmin>xxmin)&&(xbmin<xxmax))||((xbmax>xxmin)&&(xbmax<xxmax)) ||
                ((xxmin>xbmin)&&(xxmin<xbmax))||((xxmax>xbmin)&&(xxmax<xbmax))) {
               // daughter ic in interval
               ind[current]++;
               ind[current+ind[current]] = ic;
            }
         }
         current += ind[current]+1;
      }
      if (fIndY) delete fIndY;
      fIndY = new Int_t[current];
      memcpy(fIndY, &ind[0], current*sizeof(Int_t));
   }
   
   // sort z boundaries
   ib = 0;
   TMath::Sort(2*nd, &boundaries[4*nd], &index[0], kFALSE);
   // compact common boundaries
   for (id=0; id<2*nd; id++) {
      if (!ib) {temp[ib++] = boundaries[4*nd+index[id]]; continue;}
      if ((TMath::Abs(temp[ib-1]-boundaries[4*nd+index[id]]))>1E-10) 
          temp[ib++]=boundaries[4*nd+index[id]];
   }      
   // now find priority on Z
   if (ib < 2) {
      Error("SortBoxes", "less than 2 boundaries on Z !");
      printf("### volume was : %s\n", fVolume->GetName());
      return;
   }   
   if (ib == 2) {
   // check range
      if (((temp[0]-zmin)<1E-10) && ((temp[1]-zmax)>-1E-10)) {
      // ordering on this axis makes no sense. Clear all arrays.
         fPriority[2] = 0;
         if (fIndZ) delete[] fIndZ; 
         fIndZ = 0;
         if (fZb) delete[] fZb;   
         fZb = 0;
         if (fOBz) delete[] fOBz;  
         fOBz = 0;
      } else {
         fPriority[2] = 1; // all in one slice
      }
   } else {
      fPriority[2] = 2;    // check all
   }

   if (fPriority[2]) {
      // store compacted boundaries
      if (fZb) delete fZb;
      fZb = new Double_t[ib];
      memcpy(fZb, &temp[0], (ib)*sizeof(Double_t));
      fIb[2] = ib;
      
      memset(ind, 0, (nd+1)*(nd+1)*sizeof(Int_t));
      current = 0;
      if (fOBz) delete fOBz;
      fOBz = 0;
      fOBz = new Int_t[fIb[2]-1]; // offsets in ind
      for (id=0; id<fIb[2]-1; id++) {
         fOBz[id] = current; // offset of dght list
         ind[current] = 0; // ndght in this slice
         xxmin = fZb[id];
         xxmax = fZb[id+1];
         for (Int_t ic=0; ic<nd; ic++) {
            xbmin = fBoxes[6*ic+5]-fBoxes[6*ic+2];   
            xbmax = fBoxes[6*ic+5]+fBoxes[6*ic+2];   
            ddx1 = TMath::Abs(xbmin-xxmax);
            ddx2 = TMath::Abs(xbmax-xxmin);
            if ((xbmin==xxmin)||(xbmax==xxmax)) {
               ind[current]++;
               ind[current+ind[current]] = ic;
               continue;
            }
            if ((ddx1<1E-12)||(ddx2<1E-12)) continue;
            if (((xbmin>xxmin)&&(xbmin<xxmax))||((xbmax>xxmin)&&(xbmax<xxmax)) ||
                ((xxmin>xbmin)&&(xxmin<xbmax))||((xxmax>xbmin)&&(xxmax<xbmax))) {
               // daughter ic in interval
               ind[current]++;
               ind[current+ind[current]] = ic;
            }
         }
         current += ind[current]+1;
      }
      if (fIndZ) delete fIndZ;
      fIndZ = new Int_t[current];
      memcpy(fIndZ, &ind[0], current*sizeof(Int_t));
   }   

   delete boundaries; boundaries=0;   
   delete index; index=0;
   delete temp; temp=0;
   delete ind;

//   Print();
   if ((!fPriority[0]) && (!fPriority[1]) && (!fPriority[2])) {
      fVolume->SetVoxelFinder(0);
      delete this;
   }
}
//-----------------------------------------------------------------------------
void TGeoVoxelFinder::Print(Option_t *) const
{
   Int_t id;
   printf("Voxels for volume %s (nd=%i)\n", fVolume->GetName(), fVolume->GetNdaughters());
   printf("priority : x=%i y=%i z=%i\n", fPriority[0], fPriority[1], fPriority[2]);
//   return;

   printf("XXX\n");
   if (fPriority[0]) {
      for (id=0; id<fIb[0]; id++) {
//         printf("%15.10f\n",fXb[id]);
         if (id == (fIb[0]-1)) break;
         printf("slice %i : %i\n", id, fIndX[fOBx[id]]);
/*
         for (Int_t j=0;j<fIndX[fOBx[id]]; j++) {
            printf("%s  low, high:  %15.10f --- %15.10f \n", 
            fVolume->GetNode(fIndX[fOBx[id]+j+1])->GetName(),
            fBoxes[6*fIndX[fOBx[id]+j+1]+3]-fBoxes[6*fIndX[fOBx[id]+j+1]],
            fBoxes[6*fIndX[fOBx[id]+j+1]+3]+fBoxes[6*fIndX[fOBx[id]+j+1]]);
         }
*/
      }
   }
   printf("YYY\n"); 
   if (fPriority[1]) { 
      for (id=0; id<fIb[1]; id++) {
//         printf("%15.10f\n", fYb[id]);
         if (id == (fIb[1]-1)) break;
         printf("slice %i : %i\n", id, fIndY[fOBy[id]]);
/*
         for (Int_t j=0;j<fIndY[fOBy[id]]; j++) {
            printf("%s  low, high:  %15.10f --- %15.10f \n", 
            fVolume->GetNode(fIndY[fOBy[id]+j+1])->GetName(),
            fBoxes[6*fIndY[fOBy[id]+j+1]+4]-fBoxes[6*fIndY[fOBy[id]+j+1]+1],
            fBoxes[6*fIndY[fOBy[id]+j+1]+4]+fBoxes[6*fIndY[fOBy[id]+j+1]+1]);
         }
*/
      }
   }
   
   printf("ZZZ\n"); 
   if (fPriority[2]) { 
      for (id=0; id<fIb[2]; id++) {
//         printf("%15.10f\n", fZb[id]);
         if (id == (fIb[2]-1)) break;
         printf("slice %i : %i\n", id, fIndZ[fOBz[id]]);
/*
         for (Int_t j=0;j<fIndZ[fOBz[id]]; j++) {
            printf("%s  low, high:  %15.10f --- %15.10f \n", 
            fVolume->GetNode(fIndZ[fOBz[id]+j+1])->GetName(),
            fBoxes[6*fIndZ[fOBz[id]+j+1]+5]-fBoxes[6*fIndZ[fOBz[id]+j+1]+2],
            fBoxes[6*fIndZ[fOBz[id]+j+1]+5]+fBoxes[6*fIndZ[fOBz[id]+j+1]+2]);
         }
*/
      }
   }
}
//-----------------------------------------------------------------------------
void TGeoVoxelFinder::PrintVoxelLimits(Double_t *point)
{
// print the voxel containing point
   Int_t im=0;
   if (fPriority[0]) {
      im = TMath::BinarySearch(fIb[0], fXb, point[0]);
      if ((im==-1) || (im==fIb[0]-1)) {
         printf("Voxel X limits: OUT\n");
      } else {
         printf("Voxel X limits: %g  %g\n", fXb[im], fXb[im+1]);
      }
   }
   if (fPriority[1]) {
      im = TMath::BinarySearch(fIb[1], fYb, point[1]);
      if ((im==-1) || (im==fIb[1]-1)) {
         printf("Voxel Y limits: OUT\n");
      } else {
         printf("Voxel Y limits: %g  %g\n", fYb[im], fYb[im+1]);
      }
   }
   if (fPriority[2]) {
      im = TMath::BinarySearch(fIb[2], fZb, point[2]);
      if ((im==-1) || (im==fIb[2]-1)) {
         printf("Voxel Z limits: OUT\n");
      } else {
         printf("Voxel Z limits: %g  %g\n", fZb[im], fZb[im+1]);
      }
   }
}
//-----------------------------------------------------------------------------
void TGeoVoxelFinder::Voxelize(Option_t *option)
{
// Voxelize attached volume according to option
   BuildBoundingBoxes();
   SortBoxes();
}

