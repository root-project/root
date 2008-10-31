// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveTriangleSet.h"
#include "TEveRGBAPalette.h"
#include "TEveManager.h"

#include "TMath.h"
#include "TVector3.h"
#include "TRandom3.h"
#include "TVirtualPad.h"
#include "TVirtualViewer3D.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

//______________________________________________________________________________
//
// Made from a list of vertices and a list of triangles (triplets of
// vertex indices).
//
// If input is composed from triangles with direct vertex coordinates
// one should consider finding all occurences of the same vertex
// and specifying it only once.
//

ClassImp(TEveTriangleSet);

//______________________________________________________________________________
TEveTriangleSet::TEveTriangleSet(Int_t nv, Int_t nt, Bool_t norms, Bool_t cols) :
   TEveElement(fColor),
   TNamed("TEveTriangleSet", 0),
   fNVerts  (nv), fVerts(0),
   fNTrings (nt), fTrings(0), fTringNorms(0), fTringCols(0),
   fColor   (2),  fTransp(0)
{
   // Constructor.

   InitMainTrans();

   fVerts  = new Float_t[3*fNVerts];
   fTrings = new Int_t  [3*fNTrings];
   fTringNorms = (norms) ? new Float_t[3*fNTrings] : 0;
   fTringCols  = (cols)  ? new UChar_t[3*fNTrings] : 0;
}

//______________________________________________________________________________
TEveTriangleSet::~TEveTriangleSet()
{
   // Destructor.

   delete [] fVerts;
   delete [] fTrings;
   delete [] fTringNorms;
   delete [] fTringCols;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTriangleSet::GenerateTriangleNormals()
{
   // Generate triangle normals via cross product of triangle edges.

   if (fTringNorms == 0)  fTringNorms = new Float_t[3*fNTrings];

   TVector3 e1, e2, n;
   Float_t *norm = fTringNorms;
   Int_t   *tring  = fTrings;
   for(Int_t t=0; t<fNTrings; ++t, norm+=3, tring+=3)
   {
      Float_t* v0 = Vertex(tring[0]);
      Float_t* v1 = Vertex(tring[1]);
      Float_t* v2 = Vertex(tring[2]);
      e1.SetXYZ(v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]);
      e2.SetXYZ(v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]);
      n = e1.Cross(e2);
      n.SetMag(1);
      n.GetXYZ(norm);
   }
}

//______________________________________________________________________________
void TEveTriangleSet::GenerateRandomColors()
{
   // Assign random colors to all triangles.

   if (fTringCols == 0)  fTringCols = new UChar_t[3*fNTrings];

   TRandom r;
   r.SetSeed(0);
   UChar_t *col = fTringCols;
   for(Int_t t=0; t<fNTrings; ++t, col+=3)
   {
      col[0] = (UChar_t) r.Uniform(60, 255);
      col[1] = (UChar_t) r.Uniform(60, 255);
      col[2] = (UChar_t) r.Uniform(60, 255);
   }
}

//______________________________________________________________________________
void TEveTriangleSet::GenerateZNormalColors(Float_t fac, Int_t min, Int_t max,
                                            Bool_t interp, Bool_t wrap)
{
   // Generate triangle colors by the z-component of the normal.
   // Current palette is taken from gStyle.

   if (fTringCols  == 0)  fTringCols = new UChar_t[3*fNTrings];
   if (fTringNorms == 0)  GenerateTriangleNormals();

   TEveRGBAPalette pal(min, max, interp, wrap);
   UChar_t *col = fTringCols;
   Float_t *norm = fTringNorms;
   for(Int_t t=0; t<fNTrings; ++t, col+=3, norm+=3)
   {
      Int_t v = TMath::Nint(fac * norm[2]);
      pal.ColorFromValue(v, col, kFALSE);
   }
   gEve->Redraw3D();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTriangleSet::ComputeBBox()
{
   // Compute bounding box.
   // Virtual from TAttBBox.

   if (fNVerts <= 0) {
      BBoxZero();
      return;
   }

   BBoxInit();
   Float_t* v = fVerts;
   for (Int_t i=0; i<fNVerts; ++i, v += 3)
      BBoxCheckPoint(v);
}

//______________________________________________________________________________
void TEveTriangleSet::Paint(Option_t* )
{
   // Paint the object.

   TBuffer3D buffer(TBuffer3DTypes::kGeneric);

   // Section kCore
   buffer.fID           = this;
   buffer.fColor        = GetMainColor();
   buffer.fTransparency = GetMainTransparency();
   RefMainTrans().SetBuffer3D(buffer);
   buffer.SetSectionsValid(TBuffer3D::kCore);

   // We fill kCore on first pass and try with viewer
   Int_t reqSections = gPad->GetViewer3D()->AddObject(buffer);
   if (reqSections == TBuffer3D::kNone) {
      return;
   }

   Error("TEveTriangleSet::Paint", "only direct OpenGL rendering supported.");
}

/******************************************************************************/

//______________________________________________________________________________
TEveTriangleSet* TEveTriangleSet::ReadTrivialFile(const char* file)
{
   // Read a simple ascii input file describing vertices and triangles.

   FILE* f = fopen(file, "r");
   if (f == 0) {
      ::Error("TEveTriangleSet::ReadTrivialFile", Form("file '%s' not found.", file));
      return 0;
   }

   Int_t nv, nt;
   fscanf(f, "%d %d", &nv, &nt);

   TEveTriangleSet* ts = new TEveTriangleSet(nv, nt);

   Float_t *vtx = ts->Vertex(0);
   for (Int_t i=0; i<nv; ++i, vtx+=3) {
      fscanf(f, "%f %f %f", &vtx[0], &vtx[1], &vtx[2]);
   }

   Int_t *tngl = ts->Triangle(0);
   for (Int_t i=0; i<nt; ++i, tngl+=3) {
      fscanf(f, "%d %d %d", &tngl[0], &tngl[1], &tngl[2]);
   }

   fclose(f);

   return ts;
}

