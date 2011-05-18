// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  06/01/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <algorithm>
#include <cmath>

#include "TError.h"
#include "TF3.h"

#include "TGLMarchingCubes.h"

/*
Implementation of "marching cubes" algortihm for GL module. Used by 
TF3GLPainter, TGLIsoPainter, TGL5DPainter. 
Good and clear algorithm explanation can be found here: 
http://local.wasp.uwa.edu.au/~pbourke/geometry/polygonise/
*/

namespace Rgl {
namespace Mc {

/*
Some routines, values and tables for marching cube method.
*/
extern const UInt_t  eInt[256];
extern const Float_t vOff[8][3];
extern const UChar_t eConn[12][2];
extern const Float_t eDir[12][3];
extern const Int_t   conTbl[256][16];

namespace {

enum ECubeBitMasks {
   k0  = 0x1,
   k1  = 0x2,
   k2  = 0x4,
   k3  = 0x8,
   k4  = 0x10,
   k5  = 0x20,
   k6  = 0x40,
   k7  = 0x80,
   k8  = 0x100,
   k9  = 0x200,
   k10 = 0x400,
   k11 = 0x800,
   //
   k1_5            = k1 | k5,
   k2_6            = k2 | k6,
   k3_7            = k3 | k7,
   k4_5_6_7        = k4 | k5 | k6 | k7,
   k5_6            = k5 | k6,
   k0_1_2_3_7_8_11 = k0 | k1 | k2 | k3 | k7 | k8 | k11,
   k6_7            = k6 | k7
};

//______________________________________________________________________
template<class E, class V>
void ConnectTriangles(TCell<E> &cell, TIsoMesh<V> *mesh, V eps)
{
   UInt_t t[3];
   for (UInt_t i = 0; i < 5; ++i) {
      if (conTbl[cell.fType][3 * i] < 0)
         break;
      for (Int_t j = 2; j >= 0; --j)
         t[j] = cell.fIds[conTbl[cell.fType][3 * i + j]];

      const V *v0 = &mesh->fVerts[t[0] * 3];
      const V *v1 = &mesh->fVerts[t[1] * 3];
      const V *v2 = &mesh->fVerts[t[2] * 3];

      if (std::abs(v0[0] - v1[0]) < eps && 
          std::abs(v0[1] - v1[1]) < eps &&
          std::abs(v0[2] - v1[2]) < eps)
         continue;

      if (std::abs(v2[0] - v1[0]) < eps && 
          std::abs(v2[1] - v1[1]) < eps &&
          std::abs(v2[2] - v1[2]) < eps)
         continue;

      if (std::abs(v0[0] - v2[0]) < eps && 
          std::abs(v0[1] - v2[1]) < eps &&
          std::abs(v0[2] - v2[2]) < eps)
         continue;

      mesh->AddTriangle(t);
   }
}

}//unnamed namespace.

/*
TF3Adapter.
*/
//______________________________________________________________________
void TF3Adapter::SetDataSource(const TF3 *f3)
{
   fTF3 = f3;
   fW = f3->GetXaxis()->GetNbins();//f3->GetNpx();
   fH = f3->GetYaxis()->GetNbins();//f3->GetNpy();
   fD = f3->GetZaxis()->GetNbins();//f3->GetNpz();
}

//______________________________________________________________________
Double_t TF3Adapter::GetData(UInt_t i, UInt_t j, UInt_t k)const
{
   return fTF3->Eval(fMinX * fXScaleInverted + i * fStepX * fXScaleInverted, 
                     fMinY * fYScaleInverted + j * fStepY * fYScaleInverted, 
                     fMinZ * fZScaleInverted + k * fStepZ * fZScaleInverted);
}

/*
TF3 split edge implementation.
*/
//______________________________________________________________________
void TF3EdgeSplitter::SplitEdge(TCell<Double_t> & cell, TIsoMesh<Double_t> * mesh, UInt_t i,
                                Double_t x, Double_t y, Double_t z, Double_t iso)const
{
   //Split the edge and find normal in a new vertex.
   Double_t v[3] = {};
   const Double_t ofst = GetOffset(cell.fVals[eConn[i][0]], cell.fVals[eConn[i][1]], iso);
   v[0] = x + (vOff[eConn[i][0]][0] + ofst * eDir[i][0]) * fStepX;
   v[1] = y + (vOff[eConn[i][0]][1] + ofst * eDir[i][1]) * fStepY;
   v[2] = z + (vOff[eConn[i][0]][2] + ofst * eDir[i][2]) * fStepZ;
   cell.fIds[i] = mesh->AddVertex(v);

   const Double_t stepXU = fStepX * fXScaleInverted;
   const Double_t xU     = x * fXScaleInverted;
   const Double_t stepYU = fStepY * fYScaleInverted;
   const Double_t yU     = y * fYScaleInverted;
   const Double_t stepZU = fStepZ * fZScaleInverted;
   const Double_t zU     = z * fZScaleInverted;

   Double_t vU[3] = {};//U - unscaled.
   vU[0] = xU + (vOff[eConn[i][0]][0] + ofst * eDir[i][0]) * stepXU;
   vU[1] = yU + (vOff[eConn[i][0]][1] + ofst * eDir[i][1]) * stepYU;
   vU[2] = zU + (vOff[eConn[i][0]][2] + ofst * eDir[i][2]) * stepZU;
   //Find normals.
   Double_t n[3];
   n[0] = fTF3->Eval(vU[0] - 0.1 * stepXU, vU[1], vU[2]) -
          fTF3->Eval(vU[0] + 0.1 * stepXU, vU[1], vU[2]);
   n[1] = fTF3->Eval(vU[0], vU[1] - 0.1 * stepYU, vU[2]) -
          fTF3->Eval(vU[0], vU[1] + 0.1 * stepYU, vU[2]);
   n[2] = fTF3->Eval(vU[0], vU[1], vU[2] - 0.1 * stepZU) -
          fTF3->Eval(vU[0], vU[1], vU[2] + 0.1 * stepZU);

   const Double_t len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
   if (len > 1e-7) {
      n[0] /= len;
      n[1] /= len;
      n[2] /= len;
   }

   mesh->AddNormal(n);
}
/*
TMeshBuilder's implementation.
"this->" is used with type-dependant names
in templates.
*/
//______________________________________________________________________
template<class D, class V>
void TMeshBuilder<D, V>::BuildMesh(const D *s, const TGridGeometry<V> &g,
                                   MeshType_t *m, V iso)
{
   //Build iso-mesh using marching cubes.
   static_cast<TGridGeometry<V> &>(*this) = g;

   this->SetDataSource(s);

   if (GetW() < 2 || GetH() < 2 || GetD() < 2) {
      Error("TMeshBuilder::BuildMesh", 
            "Bad grid size, one of dimensions is less than 2");
      return;
   }

   fSlices[0].ResizeSlice(GetW() - 1, GetH() - 1);
   fSlices[1].ResizeSlice(GetW() - 1, GetH() - 1);

   this->SetNormalEvaluator(s);

   fMesh = m;
   fIso  = iso;

   SliceType_t *slice1 = fSlices;
   SliceType_t *slice2 = fSlices + 1;

   this->FetchDensities();
   NextStep(0, 0, slice1);

   for (UInt_t i = 1, e = GetD(); i < e - 1; ++i) {
      NextStep(i, slice1, slice2);
      std::swap(slice1, slice2);
   }

   if(fAvgNormals)
      BuildNormals();
}

//______________________________________________________________________
template<class D, class V>
void TMeshBuilder<D, V>::NextStep(UInt_t depth, const SliceType_t *prevSlice, 
                                  SliceType_t *curr)const
{
   //Fill slice with vertices and triangles.

   if (!prevSlice) {
      //The first slice in mc grid.
      BuildFirstCube(curr);
      BuildRow(curr);
      BuildCol(curr);
      BuildSlice(curr);
   } else {
      BuildFirstCube(depth, prevSlice, curr);
      BuildRow(depth, prevSlice, curr);
      BuildCol(depth, prevSlice, curr);
      BuildSlice(depth, prevSlice, curr);
   }
}

//______________________________________________________________________
template<class D, class V>
void TMeshBuilder<D, V>::BuildFirstCube(SliceType_t *s)const
{
   //The first cube in a grid. nx == 0, ny == 0, nz ==0.
   CellType_t & cell = s->fCells[0];
   cell.fVals[0] = GetData(0, 0, 0);
   cell.fVals[1] = GetData(1, 0, 0);
   cell.fVals[2] = GetData(1, 1, 0);
   cell.fVals[3] = GetData(0, 1, 0);
   cell.fVals[4] = GetData(0, 0, 1);
   cell.fVals[5] = GetData(1, 0, 1);
   cell.fVals[6] = GetData(1, 1, 1);
   cell.fVals[7] = GetData(0, 1, 1);

   cell.fType = 0;
   for (UInt_t i = 0; i < 8; ++i) {
      if (cell.fVals[i] <= fIso)
         cell.fType |= 1 << i;
   }

   for (UInt_t i = 0, edges = eInt[cell.fType]; i < 12; ++i) {
      if (edges & (1 << i))
         SplitEdge(cell, fMesh, i, this->fMinX, this->fMinY, this->fMinZ, fIso);
   }

   ConnectTriangles(cell, fMesh, fEpsilon);
}

//______________________________________________________________________
template<class D, class V>
void TMeshBuilder<D, V>::BuildRow(SliceType_t *s)const
{
   //The first row (along x) in the first slice:
   //ny == 0, nz == 0, nx : [1, W - 1].
   //Each cube has previous cube.
   //Values 0, 3, 4, 7 are taken from the previous cube.
   //Edges 3, 7, 8, 11 are taken from the previous cube.
   for (UInt_t i = 1, e = GetW() - 1; i < e; ++i) {
      const CellType_t &prev = s->fCells[i - 1];
      CellType_t &cell = s->fCells[i];
      cell.fType = 0;

      cell.fVals[0] = prev.fVals[1], cell.fVals[4] = prev.fVals[5];
      cell.fVals[7] = prev.fVals[6], cell.fVals[3] = prev.fVals[2];
      cell.fType |= (prev.fType & k1_5) >> 1;
      cell.fType |= (prev.fType & k2_6) << 1;

      if ((cell.fVals[1] = GetData(i + 1, 0, 0)) <= fIso)
         cell.fType |= k1;
      if ((cell.fVals[2] = GetData(i + 1, 1, 0)) <= fIso)
         cell.fType |= k2;
      if ((cell.fVals[5] = GetData(i + 1, 0, 1)) <= fIso)
         cell.fType |= k5;
      if ((cell.fVals[6] = GetData(i + 1, 1, 1)) <= fIso)
         cell.fType |= k6;

      const UInt_t edges = eInt[cell.fType];
      if (!edges)
         continue;
      //1. Take edges 3, 7, 8, 11 from the previous cube.
      if (edges & k3)
         cell.fIds[3]  = prev.fIds[1];
      if (edges & k7)
         cell.fIds[7]  = prev.fIds[5];
      if (edges & k8)
         cell.fIds[8]  = prev.fIds[9];
      if (edges & k11)
         cell.fIds[11] = prev.fIds[10];
      //2. Intersect edges 0, 1, 2, 4, 5, 6, 9, 10.
      const V x = this->fMinX + i * this->fStepX;
      if (edges & k0)
         SplitEdge(cell, fMesh, 0, x, this->fMinY, this->fMinZ, fIso);
      if (edges & k1)
         SplitEdge(cell, fMesh, 1, x, this->fMinY, this->fMinZ, fIso);
      if (edges & k2)
         SplitEdge(cell, fMesh, 2, x, this->fMinY, this->fMinZ, fIso);
      if (edges & k4)
         SplitEdge(cell, fMesh, 4, x, this->fMinY, this->fMinZ, fIso);
      if (edges & k5)
         SplitEdge(cell, fMesh, 5, x, this->fMinY, this->fMinZ, fIso);
      if (edges & k6)
         SplitEdge(cell, fMesh, 6, x, this->fMinY, this->fMinZ, fIso);
      if (edges & k9)
         SplitEdge(cell, fMesh, 9, x, this->fMinY, this->fMinZ, fIso);
      if (edges & k10)
         SplitEdge(cell, fMesh, 10, x, this->fMinY, this->fMinZ, fIso);
      //3. Connect new triangles.
      ConnectTriangles(cell, fMesh, fEpsilon);
   }
}

//______________________________________________________________________
template<class D, class V>
void TMeshBuilder<D, V>::BuildCol(SliceType_t *s)const
{
   //"Col" (column) consists of cubes along y axis
   //on the first slice (nx == 0, nz == 0).
   //Each cube has a previous cube and shares values:
   //0, 1, 4, 5 (in prev.: 3, 2, 7, 6); and edges:
   //0, 4, 8, 9 (in prev.: 2, 6, 10, 11).
   const UInt_t w = GetW();
   const UInt_t h = GetH();

   for (UInt_t i = 1; i < h - 1; ++i) {
      const CellType_t &prev = s->fCells[(i - 1) * (w - 1)];
      CellType_t &cell = s->fCells[i * (w - 1)];
      cell.fType = 0;
      //Take values 0, 1, 4, 5 from the prev. cube.
      cell.fVals[0] = prev.fVals[3], cell.fVals[1] = prev.fVals[2];
      cell.fVals[4] = prev.fVals[7], cell.fVals[5] = prev.fVals[6];
      cell.fType |= (prev.fType & k2_6) >> 1;
      cell.fType |= (prev.fType & k3_7) >> 3;
      //Calculate values 2, 3, 6, 7.
      if((cell.fVals[2] = GetData(1, i + 1, 0)) <= fIso)
         cell.fType |= k2;
      if((cell.fVals[3] = GetData(0, i + 1, 0)) <= fIso)
         cell.fType |= k3;
      if((cell.fVals[6] = GetData(1, i + 1, 1)) <= fIso)
         cell.fType |= k6;
      if((cell.fVals[7] = GetData(0, i + 1, 1)) <= fIso)
         cell.fType |= k7;

      const UInt_t edges = eInt[cell.fType];
      if(!edges)
         continue;
      //Take edges from the previous cube.
      if (edges & k0)
         cell.fIds[0] = prev.fIds[2];
      if (edges & k4)
         cell.fIds[4] = prev.fIds[6];
      if (edges & k9)
         cell.fIds[9] = prev.fIds[10];
      if (edges & k8)
         cell.fIds[8] = prev.fIds[11];
      //Find the remaining edges.
      const V y = this->fMinY + i * this->fStepY;

      if (edges & k1)
         SplitEdge(cell, fMesh, 1, this->fMinX, y, this->fMinZ, fIso);
      if (edges & k2)
         SplitEdge(cell, fMesh, 2, this->fMinX, y, this->fMinZ, fIso);
      if (edges & k3)
         SplitEdge(cell, fMesh, 3, this->fMinX, y, this->fMinZ, fIso);
      if (edges & k5)
         SplitEdge(cell, fMesh, 5, this->fMinX, y, this->fMinZ, fIso);
      if (edges & k6)
         SplitEdge(cell, fMesh, 6, this->fMinX, y, this->fMinZ, fIso);
      if (edges & k7)
         SplitEdge(cell, fMesh, 7, this->fMinX, y, this->fMinZ, fIso);
      if (edges & k10)
         SplitEdge(cell, fMesh, 10, this->fMinX, y, this->fMinZ, fIso);
      if (edges & k11)
         SplitEdge(cell, fMesh, 11, this->fMinX, y, this->fMinZ, fIso);

      ConnectTriangles(cell, fMesh, fEpsilon);
   }
}

//______________________________________________________________________
template<class D, class V>
void TMeshBuilder<D, V>::BuildSlice(SliceType_t *s)const
{
   //Slice with nz == 0.
   //nx : [1, W - 1], ny : [1, H - 1].
   //nx increased inside inner loop, ny - enclosing loop.
   //Each cube has two neighbours: ny - 1 => "left",
   //nx - 1 => "right".
   const UInt_t w = GetW();
   const UInt_t h = GetH();

   for (UInt_t i = 1; i < h - 1; ++i) {
      const V y = this->fMinY + i * this->fStepY;

      for (UInt_t j = 1; j < w - 1; ++j) {
         const CellType_t &left  = s->fCells[(i - 1) * (w - 1) + j];
         const CellType_t &right = s->fCells[i * (w - 1) + j - 1];
         CellType_t &cell = s->fCells[i * (w - 1) + j];
         cell.fType = 0;
         //Take values 0, 1, 4, 5 from left cube.
         cell.fVals[1] = left.fVals[2];
         cell.fVals[0] = left.fVals[3];
         cell.fVals[5] = left.fVals[6];
         cell.fVals[4] = left.fVals[7];
         cell.fType |= (left.fType & k2_6) >> 1;
         cell.fType |= (left.fType & k3_7) >> 3;
         //3, 7 from right cube.
         cell.fVals[3] = right.fVals[2];
         cell.fVals[7] = right.fVals[6];
         cell.fType |= (right.fType & k2_6) << 1;
         //Calculate values 2, 6.
         if((cell.fVals[2] = GetData(j + 1, i + 1, 0)) <= fIso)
            cell.fType |= k2;
         if((cell.fVals[6] = GetData(j + 1, i + 1, 1)) <= fIso)
            cell.fType |= k6;

         const UInt_t edges = eInt[cell.fType];
         if(!edges)
            continue;
         //Take edges 0, 4, 8, 9 from the "left" cube.
         //In left cube their indices are 2, 6, 11, 10.
         if(edges & k0)
            cell.fIds[0] = left.fIds[2];
         if(edges & k4)
            cell.fIds[4] = left.fIds[6];
         if(edges & k8)
            cell.fIds[8] = left.fIds[11];
         if(edges & k9)
            cell.fIds[9] = left.fIds[10];
         //Take edges 3, 7, 11 from the "right" cube.
         //Their "right" indices are 1, 5, 10.
         if(edges & k3)
            cell.fIds[3]  = right.fIds[1];
         if(edges & k7)
            cell.fIds[7]  = right.fIds[5];
         if(edges & k11)
            cell.fIds[11] = right.fIds[10];
         //Calculate the remaining intersections: edges
         //1, 2, 5, 6, 10.
         const V x = this->fMinX + j * this->fStepX;
         if (edges & k1)
            SplitEdge(cell, fMesh, 1, x, y, this->fMinZ, fIso);
         if (edges & k2)
            SplitEdge(cell, fMesh, 2, x, y, this->fMinZ, fIso);
         if (edges & k5)
            SplitEdge(cell, fMesh, 5, x, y, this->fMinZ, fIso);
         if (edges & k6)
            SplitEdge(cell, fMesh, 6, x, y, this->fMinZ, fIso);
         if (edges & k10)
            SplitEdge(cell, fMesh, 10, x, y, this->fMinZ, fIso);

         ConnectTriangles(cell, fMesh, fEpsilon);
      }
   }
}

//______________________________________________________________________
template<class D, class V>
void TMeshBuilder<D, V>::BuildFirstCube(UInt_t depth, const SliceType_t *prevSlice,
                                        SliceType_t *slice)const
{
   //The first cube in a slice with nz == depth.
   //Neighbour is the first cube in the previous slice.
   //Four values and four edges come from the previous cube.
   const CellType_t &prevCell = prevSlice->fCells[0];
   CellType_t &cell = slice->fCells[0];
   cell.fType = 0;
   //Values 0, 1, 2, 3 are 4, 5, 6, 7
   //in the previous cube.
   cell.fVals[0] = prevCell.fVals[4];
   cell.fVals[1] = prevCell.fVals[5];
   cell.fVals[2] = prevCell.fVals[6];
   cell.fVals[3] = prevCell.fVals[7];
   cell.fType |= (prevCell.fType & k4_5_6_7) >> 4;
   //Calculate 4, 5, 6, 7.
   if((cell.fVals[4] = GetData(0, 0, depth + 1)) <= fIso)
      cell.fType |= k4;
   if((cell.fVals[5] = GetData(1, 0, depth + 1)) <= fIso)
      cell.fType |= k5;
   if((cell.fVals[6] = GetData(1, 1, depth + 1)) <= fIso)
      cell.fType |= k6;
   if((cell.fVals[7] = GetData(0, 1, depth + 1)) <= fIso)
      cell.fType |= k7;

   const UInt_t edges = eInt[cell.fType];
   if(!edges)
      return;

   //Edges 0, 1, 2, 3 taken from the prev. cube -
   //they have indices 4, 5, 6, 7 there.
   if(edges & k0)
      cell.fIds[0] = prevCell.fIds[4];
   if(edges & k1)
      cell.fIds[1] = prevCell.fIds[5];
   if(edges & k2)
      cell.fIds[2] = prevCell.fIds[6];
   if(edges & k3)
      cell.fIds[3] = prevCell.fIds[7];

   const V z = this->fMinZ + depth * this->fStepZ;

   if(edges & k4)
      SplitEdge(cell, fMesh, 4,  this->fMinX, this->fMinY, z, fIso);
   if(edges & k5)
      SplitEdge(cell, fMesh, 5,  this->fMinX, this->fMinY, z, fIso);
   if(edges & k6)
      SplitEdge(cell, fMesh, 6,  this->fMinX, this->fMinY, z, fIso);
   if(edges & k7)
      SplitEdge(cell, fMesh, 7,  this->fMinX, this->fMinY, z, fIso);
   if(edges & k8)
      SplitEdge(cell, fMesh, 8,  this->fMinX, this->fMinY, z, fIso);
   if(edges & k9)
      SplitEdge(cell, fMesh, 9,  this->fMinX, this->fMinY, z, fIso);
   if(edges & k10)
      SplitEdge(cell, fMesh, 10, this->fMinX, this->fMinY, z, fIso);
   if(edges & k11)
      SplitEdge(cell, fMesh, 11, this->fMinX, this->fMinY, z, fIso);

   ConnectTriangles(cell, fMesh, fEpsilon);
}

//______________________________________________________________________
template<class D, class V>
void TMeshBuilder<D, V>::BuildRow(UInt_t depth, const SliceType_t *prevSlice,
                                  SliceType_t *slice)const
{
   //Row with ny == 0 and nz == depth, nx : [1, W - 1].
   //Two neighbours: one from previous slice (called bottom cube here),
   //the second is the previous cube in a row.
   const V z = this->fMinZ + depth * this->fStepZ;
   const UInt_t w = GetW();

   for (UInt_t i = 1; i < w - 1; ++i) {
      const CellType_t &prevCell = slice->fCells[i - 1];
      const CellType_t &bottCell = prevSlice->fCells[i];
      CellType_t &cell = slice->fCells[i];
      cell.fType = 0;
      //Value 0 is not required,
      //only bit number 0 in fType is interesting.
      //3, 4, 7 come from the previous box (2, 5, 6)
      cell.fVals[3] = prevCell.fVals[2];
      cell.fVals[4] = prevCell.fVals[5];
      cell.fVals[7] = prevCell.fVals[6];
      cell.fType |= (prevCell.fType & k1_5) >> 1;
      cell.fType |= (prevCell.fType & k2_6) << 1;
      //1, 2 can be taken from the bottom cube (5, 6).
      cell.fVals[1] = bottCell.fVals[5];
      cell.fVals[2] = bottCell.fVals[6];
      cell.fType |= (bottCell.fType & k5_6) >> 4;
      //5, 6 must be calculated.
      if((cell.fVals[5] = GetData(i + 1, 0, depth + 1)) <= fIso)
         cell.fType |= k5;
      if((cell.fVals[6] = GetData(i + 1, 1, depth + 1)) <= fIso)
         cell.fType |= k6;

      UInt_t edges = eInt[cell.fType];

      if(!edges)
         continue;
      //Take edges 3, 7, 8, 11 from the previous cube (1, 5, 9, 10).
      if(edges & k3)
         cell.fIds[3] = prevCell.fIds[1];
      if(edges & k7)
         cell.fIds[7] = prevCell.fIds[5];
      if(edges & k8)
         cell.fIds[8] = prevCell.fIds[9];
      if(edges & k11)
         cell.fIds[11] = prevCell.fIds[10];
      //Take edges 0, 1, 2 from the bottom cube (4, 5, 6).
      if(edges & k0)
         cell.fIds[0] = bottCell.fIds[4];
      if(edges & k1)
         cell.fIds[1] = bottCell.fIds[5];
      if(edges & k2)
         cell.fIds[2] = bottCell.fIds[6];

      edges &= ~k0_1_2_3_7_8_11;

      if (edges) {
         const V x = this->fMinX + i * this->fStepX;

         if(edges & k4)
            SplitEdge(cell, fMesh, 4,  x, this->fMinY, z, fIso);
         if(edges & k5)
            SplitEdge(cell, fMesh, 5,  x, this->fMinY, z, fIso);
         if(edges & k6)
            SplitEdge(cell, fMesh, 6,  x, this->fMinY, z, fIso);
         if(edges & k9)
            SplitEdge(cell, fMesh, 9,  x, this->fMinY, z, fIso);
         if(edges & k10)
            SplitEdge(cell, fMesh, 10, x, this->fMinY, z, fIso);
      }

      ConnectTriangles(cell, fMesh, fEpsilon);
   }
}

//______________________________________________________________________
template<class D, class V>
void TMeshBuilder<D, V>::BuildCol(UInt_t depth, const SliceType_t *prevSlice,
                                  SliceType_t *slice)const
{
   //nz == depth, nx == 0, ny : [1, H - 1].
   //Two neighbours - from previous slice ("bottom" cube)
   //and previous cube in a column.
   const V z = this->fMinZ + depth * this->fStepZ;
   const UInt_t w = GetW();
   const UInt_t h = GetH();

   for (UInt_t i = 1; i < h - 1; ++i) {
      const CellType_t &left = slice->fCells[(i - 1) * (w - 1)];
      const CellType_t &bott = prevSlice->fCells[i * (w - 1)];
      CellType_t &cell = slice->fCells[i * (w - 1)];
      cell.fType = 0;
      //Value 0 is not required, only bit.
      //Take 1, 4, 5 from left cube.
      cell.fVals[1] = left.fVals[2];
      cell.fVals[4] = left.fVals[7];
      cell.fVals[5] = left.fVals[6];
      cell.fType |= (left.fType & k2_6) >> 1;
      cell.fType |= (left.fType & k3_7) >> 3;
      //2, 3 from bottom.
      cell.fVals[2] = bott.fVals[6];
      cell.fVals[3] = bott.fVals[7];
      cell.fType |= (bott.fType & k6_7) >> 4;
      //Calculate 6, 7.
      if((cell.fVals[6] = GetData(1, i + 1, depth + 1)) <= fIso)
         cell.fType |= k6;
      if((cell.fVals[7] = GetData(0, i + 1, depth + 1)) <= fIso)
         cell.fType |= k7;

      const UInt_t edges = eInt[cell.fType];
      if(!edges)
         continue;

      if(edges & k0)
         cell.fIds[0] = left.fIds[2];
      if(edges & k4)
         cell.fIds[4] = left.fIds[6];
      if(edges & k8)
         cell.fIds[8] = left.fIds[11];
      if(edges & k9)
         cell.fIds[9] = left.fIds[10];

      if(edges & k1)
         cell.fIds[1] = bott.fIds[5];
      if(edges & k2)
         cell.fIds[2] = bott.fIds[6];
      if(edges & k3)
         cell.fIds[3] = bott.fIds[7];

      const V y = this->fMinY + i * this->fStepY;
      
      if(edges & k5)
         SplitEdge(cell, fMesh, 5,  this->fMinX, y, z, fIso);
      if(edges & k6)
         SplitEdge(cell, fMesh, 6,  this->fMinX, y, z, fIso);
      if(edges & k7)
         SplitEdge(cell, fMesh, 7,  this->fMinX, y, z, fIso);
      if(edges & k10)
         SplitEdge(cell, fMesh, 10, this->fMinX, y, z, fIso);
      if(edges & k11)
         SplitEdge(cell, fMesh, 11, this->fMinX, y, z, fIso);

      ConnectTriangles(cell, fMesh, fEpsilon);
   }
}


//______________________________________________________________________
template<class D, class V>
void TMeshBuilder<D, V>::BuildSlice(UInt_t depth, const SliceType_t *prevSlice,
                                    SliceType_t *slice)const
{
   //nz == depth, nx : [1, W - 1], ny : [1, H - 1].
   //Each cube has 3 neighbours, "bottom" cube from
   //the previous slice, "left" and "right" from the
   //current slice.
   const V z = this->fMinZ + depth * this->fStepZ;
   const UInt_t h = GetH();
   const UInt_t w = GetW();

   for (UInt_t i = 1; i < h - 1; ++i) {
      const V y = this->fMinY + i * this->fStepY;
      for (UInt_t j = 1; j < w - 1; ++j) {
         const CellType_t &left = slice->fCells[(i - 1) * (w - 1) + j];
         const CellType_t &right = slice->fCells[i * (w - 1) + j - 1];
         const CellType_t &bott = prevSlice->fCells[i * (w - 1) + j];
         CellType_t &cell = slice->fCells[i * (w - 1) + j];
         cell.fType = 0;

         cell.fVals[1] = left.fVals[2];
         cell.fVals[4] = left.fVals[7];
         cell.fVals[5] = left.fVals[6];
         cell.fType |= (left.fType & k2_6) >> 1;
         cell.fType |= (left.fType & k3_7) >> 3;

         cell.fVals[2] = bott.fVals[6];
         cell.fVals[3] = bott.fVals[7];
         cell.fType |= (bott.fType & k6_7) >> 4;

         cell.fVals[7] = right.fVals[6];
         cell.fType |= (right.fType & k6) << 1;

         if ((cell.fVals[6] = GetData(j + 1, i + 1, depth + 1)) <= fIso)
            cell.fType |= k6;

         const UInt_t edges = eInt[cell.fType];
         if (!edges)
            continue;

         if(edges & k0)
            cell.fIds[0] = left.fIds[2];
         if(edges & k4)
            cell.fIds[4] = left.fIds[6];
         if(edges & k8)
            cell.fIds[8] = left.fIds[11];
         if(edges & k9)
            cell.fIds[9] = left.fIds[10];

         if(edges & k3)
            cell.fIds[3] = right.fIds[1];
         if(edges & k7)
            cell.fIds[7] = right.fIds[5];
         if(edges & k11)
            cell.fIds[11] = right.fIds[10];

         if(edges & k1)
            cell.fIds[1] = bott.fIds[5];
         if(edges & k2)
            cell.fIds[2] = bott.fIds[6];

         const V x = this->fMinX + j * this->fStepX;
         if(edges & k5)
            SplitEdge(cell, fMesh, 5,  x, y, z, fIso);
         if(edges & k6)
            SplitEdge(cell, fMesh, 6,  x, y, z, fIso);
         if(edges & k10)
            SplitEdge(cell, fMesh, 10, x, y, z, fIso);

         ConnectTriangles(cell, fMesh, fEpsilon);
      }
   }
}

//______________________________________________________________________
template<class D, class V>
void TMeshBuilder<D, V>::BuildNormals()const
{
   //Build averaged normals using vertices and
   //trinagles.
   typedef std::vector<UInt_t>::size_type size_type;
   const UInt_t *t;
   V *p1, *p2, *p3;
   V v1[3], v2[3], n[3];
   
   fMesh->fNorms.assign(fMesh->fVerts.size(), V());

   for (size_type i = 0, e = fMesh->fTris.size() / 3; i < e; ++i) {
      t  = &fMesh->fTris[i * 3];
      p1 = &fMesh->fVerts[t[0] * 3];
      p2 = &fMesh->fVerts[t[1] * 3];
      p3 = &fMesh->fVerts[t[2] * 3];
      v1[0] = p2[0] - p1[0];
      v1[1] = p2[1] - p1[1];
      v1[2] = p2[2] - p1[2];
      v2[0] = p3[0] - p1[0];
      v2[1] = p3[1] - p1[1];
      v2[2] = p3[2] - p1[2];
      n[0] = v1[1] * v2[2] - v1[2] * v2[1];
      n[1] = v1[2] * v2[0] - v1[0] * v2[2];
      n[2] = v1[0] * v2[1] - v1[1] * v2[0];

      const V len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);

      if (len < fEpsilon)//degenerated triangle
         continue;

      n[0] /= len;
      n[1] /= len;
      n[2] /= len;
      UInt_t ind = t[0] * 3;
      fMesh->fNorms[ind]     += n[0];
      fMesh->fNorms[ind + 1] += n[1];
      fMesh->fNorms[ind + 2] += n[2];
      ind = t[1] * 3;
      fMesh->fNorms[ind]     += n[0];
      fMesh->fNorms[ind + 1] += n[1];
      fMesh->fNorms[ind + 2] += n[2];
      ind = t[2] * 3;
      fMesh->fNorms[ind]     += n[0];
      fMesh->fNorms[ind + 1] += n[1];
      fMesh->fNorms[ind + 2] += n[2];
   }

   for (size_type i = 0, e = fMesh->fNorms.size() / 3; i < e; ++i) {
      V * nn = &fMesh->fNorms[i * 3];
      const V len = std::sqrt(nn[0] * nn[0] + nn[1] * nn[1] + nn[2] * nn[2]);
      if (len < fEpsilon)
         continue;
      fMesh->fNorms[i * 3]     /= len;
      fMesh->fNorms[i * 3 + 1] /= len;
      fMesh->fNorms[i * 3 + 2] /= len;
   }
}

/////////////////////////////////////////////////////////////////////////
//****************************TABLES***********************************//
/////////////////////////////////////////////////////////////////////////

const UInt_t eInt[256] = 
{
   0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 
   0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
   0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 
   0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
   0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 
   0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
   0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 
   0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
   0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 
   0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
   0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 
   0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
   0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 
   0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
   0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 
   0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
   0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 
   0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
   0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 
   0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
   0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 
   0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
   0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 
   0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
   0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 
   0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
   0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 
   0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
   0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 
   0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
   0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 
   0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
};

const Float_t vOff[8][3] =
{
   {0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f},
   {0.f, 1.f, 0.f}, {0.f, 0.f, 1.f}, {1.f, 0.f, 1.f},
   {1.f, 1.f, 1.f}, {0.f, 1.f, 1.f}
};

const UChar_t eConn[12][2] =
{
   {0, 1}, {1, 2}, {2, 3}, {3, 0},
   {4, 5}, {5, 6}, {6, 7}, {7, 4},
   {0, 4}, {1, 5}, {2, 6}, {3, 7}
};
   
const Float_t eDir[12][3] =
{
   { 1.f,  0.f, 0.f}, {0.f,  1.f, 0.f}, {-1.f, 0.f, 0.f},
   { 0.f, -1.f, 0.f}, {1.f,  0.f, 0.f}, { 0.f, 1.f, 0.f},
   {-1.f,  0.f, 0.f}, {0.f, -1.f, 0.f}, { 0.f, 0.f, 1.f},
   { 0.f,  0.f, 1.f}, {0.f,  0.f, 1.f}, { 0.f, 0.f, 1.f}
};


const Int_t conTbl[256][16] = 
{
   {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
   {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
   {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
   {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
   {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
   {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
   {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
   {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
   {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
   {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
   {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
   {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
   {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
   {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
   {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
   {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
   {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
   {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
   {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
   {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
   {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
   {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
   {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
   {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
   {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
   {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
   {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
   {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
   {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
   {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
   {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
   {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
   {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
   {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
   {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
   {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
   {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
   {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
   {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
   {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
   {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
   {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
   {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
   {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
   {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
   {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
   {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
   {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
   {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
   {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
   {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
   {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
   {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
   {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
   {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
   {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
   {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
   {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
   {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
   {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
   {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
   {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
   {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
   {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
   {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
   {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
   {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
   {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
   {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
   {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
   {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
   {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
   {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
   {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
   {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
   {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
   {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
   {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
   {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
   {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
   {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
   {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
   {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
   {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
   {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
   {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
   {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
   {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
   {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
   {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
   {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
   {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
   {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
   {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
   {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
   {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
   {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
   {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
   {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
   {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
   {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
   {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
   {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
   {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
   {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
   {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
   {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
   {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
   {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
   {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
   {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
   {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
   {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
   {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
   {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
   {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
   {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
   {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
   {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
   {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
   {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
   {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
   {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
   {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
   {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
   {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
   {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
   {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
   {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
   {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
   {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
   {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
   {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
   {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
   {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
   {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
   {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
   {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
   {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
   {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
   {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
   {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
   {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
   {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
   {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
   {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
   {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
   {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
   {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
   {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
   {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
   {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
   {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
   {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
   {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
   {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
   {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
   {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
   {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
   {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
   {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
   {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
   {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
   {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
   {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
   {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
   {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
   {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
   {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
   {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
   {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
   {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
   {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
   {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
   {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
   {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
   {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
   {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
   {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
   {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
   {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
   {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
   {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
   {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
   {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
   {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
   {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
   {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
   {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
   {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
};

template class TMeshBuilder<TH3C, Float_t>;
template class TMeshBuilder<TH3S, Float_t>;
template class TMeshBuilder<TH3I, Float_t>;
template class TMeshBuilder<TH3F, Float_t>;
template class TMeshBuilder<TH3D, Float_t>;
template class TMeshBuilder<TF3, Double_t>;
//TMeshBuilder does not need any detail from TKDEFGT.
//TKDEFGT only helps to select correct implementation.
//Forward class declaration is enough for TKDEFGT.
template class TMeshBuilder<TKDEFGT, Float_t>;

}//namespace Mc
}//namespace Rgl
