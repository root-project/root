// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  03/08/2004
// NOTE: This code moved from obsoleted TGLSceneObject.h / .cxx - see these
// attic files for previous CVS history

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLCylinder.h"
#include "TGLRnrCtx.h"
#include "TGLIncludes.h"

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TMath.h"

// For debug tracing
#include "TClass.h"
#include "TError.h"

TGLVector3 gLowNormalDefault(0., 0., -1.);
TGLVector3 gHighNormalDefault(0., 0., 1.);

class TGLMesh
{
protected:
   // active LOD (level of detail) - quality
   UInt_t     fLOD;

   Double_t fRmin1, fRmax1, fRmin2, fRmax2;
   Double_t fDz;

   //normals for top and bottom (for cuts)
   TGLVector3 fNlow;
   TGLVector3 fNhigh;

   void GetNormal(const TGLVertex3 &vertex, TGLVector3 &normal)const;
   Double_t GetZcoord(Double_t x, Double_t y, Double_t z)const;
   const TGLVertex3 &MakeVertex(Double_t x, Double_t y, Double_t z)const;

public:
   TGLMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
           const TGLVector3 &l = gLowNormalDefault, const TGLVector3 &h = gHighNormalDefault);
   virtual ~TGLMesh() { }
   virtual void Draw() const = 0;
};

//segment contains 3 quad strips:
//one for inner and outer sides, two for top and bottom
class TubeSegMesh : public TGLMesh {
private:
   // Allocate space for highest quality (LOD) meshes
   TGLVertex3 fMesh[(TGLRnrCtx::kLODHigh + 1) * 8 + 8];
   TGLVector3 fNorm[(TGLRnrCtx::kLODHigh + 1) * 8 + 8];

public:
   TubeSegMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
               Double_t phi1, Double_t phi2, const TGLVector3 &l = gLowNormalDefault,
               const TGLVector3 &h = gHighNormalDefault);

   void Draw() const;
};

//four quad strips:
//outer, inner, top, bottom
class TubeMesh : public TGLMesh
{
private:
   // Allocate space for highest quality (LOD) meshes
   TGLVertex3 fMesh[(TGLRnrCtx::kLODHigh + 1) * 8];
   TGLVector3 fNorm[(TGLRnrCtx::kLODHigh + 1) * 8];

public:
   TubeMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
            const TGLVector3 &l = gLowNormalDefault, const TGLVector3 &h = gHighNormalDefault);

   void Draw() const;
};

//One quad mesh and 2 triangle funs
class TCylinderMesh : public TGLMesh {
private:
   // Allocate space for highest quality (LOD) meshes
   TGLVertex3 fMesh[(TGLRnrCtx::kLODHigh + 1) * 4 + 2];
   TGLVector3 fNorm[(TGLRnrCtx::kLODHigh + 1) * 4 + 2];

public:
   TCylinderMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t dz,
                 const TGLVector3 &l = gLowNormalDefault, const TGLVector3 &h = gHighNormalDefault);

   void Draw() const;
};

//One quad mesh and 2 triangle fans
class TCylinderSegMesh : public TGLMesh
{
private:
   // Allocate space for highest quality (LOD) meshes
   TGLVertex3 fMesh[(TGLRnrCtx::kLODHigh + 1) * 4 + 10];
   TGLVector3 fNorm[(TGLRnrCtx::kLODHigh + 1) * 4 + 10];

public:
   TCylinderSegMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t dz, Double_t phi1, Double_t phi2,
                    const TGLVector3 &l = gLowNormalDefault, const TGLVector3 &h = gHighNormalDefault);
   void Draw() const;
};

TGLMesh::TGLMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
                 const TGLVector3 &l, const TGLVector3 &h) :
   fLOD(LOD),
   fRmin1(r1), fRmax1(r2), fRmin2(r3), fRmax2(r4),
   fDz(dz), fNlow(l), fNhigh(h)
{
   // constructor
}

////////////////////////////////////////////////////////////////////////////////
/// get normal

void TGLMesh::GetNormal(const TGLVertex3 &v, TGLVector3 &n)const
{
   if( fDz < 1.e-10 ) {
      n[0] = 0.;
      n[1] = 0.;
      n[2] = 1.;
   }
   Double_t z = (fRmax1 - fRmax2) / (2 * fDz);
   Double_t mag = TMath::Sqrt(v[0] * v[0] + v[1] * v[1] + z * z);
   if( mag > 1.e-10 ) {
      n[0] = v[0] / mag;
      n[1] = v[1] / mag;
      n[2] = z / mag;
   } else {
      n[0] = v[0];
      n[1] = v[1];
      n[2] = z;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// get Z coordinate

Double_t TGLMesh::GetZcoord(Double_t x, Double_t y, Double_t z)const
{
   Double_t newz = 0;
   if (z < 0) newz = -fDz - (x * fNlow[0] + y * fNlow[1]) / fNlow[2];
   else newz = fDz - (x * fNhigh[0] + y * fNhigh[1]) / fNhigh[2];

   return newz;
}

////////////////////////////////////////////////////////////////////////////////
/// make vertex

const TGLVertex3 &TGLMesh::MakeVertex(Double_t x, Double_t y, Double_t z)const
{
   static TGLVertex3 vert(0., 0., 0.);
   vert[0] = x;
   vert[1] = y;
   vert[2] = GetZcoord(x, y, z);

   return vert;
}

////////////////////////////////////////////////////////////////////////////////

TubeSegMesh::TubeSegMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t dz,
                         Double_t phi1, Double_t phi2,
                         const TGLVector3 &l, const TGLVector3 &h)
                 :TGLMesh(LOD, r1, r2, r3, r4, dz, l, h), fMesh(), fNorm()

{
   // constructor
   const Double_t delta = (phi2 - phi1) / LOD;
   Double_t currAngle = phi1;

   Bool_t even = kTRUE;
   Double_t c = TMath::Cos(currAngle);
   Double_t s = TMath::Sin(currAngle);
   const Int_t topShift = (fLOD + 1) * 4 + 8;
   const Int_t botShift = (fLOD + 1) * 6 + 8;
   Int_t j = 4 * (fLOD + 1) + 2;

   //defining all three strips here, first strip is non-closed here
   for (Int_t i = 0, e = (fLOD + 1) * 2; i < e; ++i) {
      if (even) {
         fMesh[i] = MakeVertex(fRmax2 * c, fRmax2 * s, fDz);
         fMesh[j] = MakeVertex(fRmin2 * c, fRmin2 * s, fDz);
         fMesh[i + topShift] = MakeVertex(fRmin2 * c, fRmin2 * s, fDz);
         fMesh[i + botShift] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         GetNormal(fMesh[j], fNorm[j]);
         fNorm[j].Negate();
         even = kFALSE;
      } else {
         fMesh[i] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         fMesh[j + 1] = MakeVertex(fRmin1 * c, fRmin1 * s, -fDz);
         fMesh[i + topShift] = MakeVertex(fRmax2 * c, fRmax2 * s, fDz);
         fMesh[i + botShift] = MakeVertex(fRmin1 * c, fRmin1 * s, - fDz);
         GetNormal(fMesh[j + 1], fNorm[j + 1]);
         fNorm[j + 1].Negate();
         even = kTRUE;
         currAngle += delta;
         c = TMath::Cos(currAngle);
         s = TMath::Sin(currAngle);
         j -= 2;
      }

      GetNormal(fMesh[i], fNorm[i]);
      fNorm[i + topShift] = fNhigh;
      fNorm[i + botShift] = fNlow;
   }

   //closing first strip
   Int_t ind = 2 * (fLOD + 1);
   TGLVector3 norm(0., 0., 0.);

   fMesh[ind] = fMesh[ind - 2];
   fMesh[ind + 1] = fMesh[ind - 1];
   fMesh[ind + 2] = fMesh[ind + 4];
   fMesh[ind + 3] = fMesh[ind + 5];
   TMath::Normal2Plane(fMesh[ind].CArr(), fMesh[ind + 1].CArr(), fMesh[ind + 2].CArr(),
                       norm.Arr());
   fNorm[ind] = norm;
   fNorm[ind + 1] = norm;
   fNorm[ind + 2] = norm;
   fNorm[ind + 3] = norm;

   ind = topShift - 4;
   fMesh[ind] = fMesh[ind - 2];
   fMesh[ind + 1] = fMesh[ind - 1];
   fMesh[ind + 2] = fMesh[0];
   fMesh[ind + 3] = fMesh[1];
   TMath::Normal2Plane(fMesh[ind].CArr(), fMesh[ind + 1].CArr(), fMesh[ind + 2].CArr(),
                       norm.Arr());
   fNorm[ind] = norm;
   fNorm[ind + 1] = norm;
   fNorm[ind + 2] = norm;
   fNorm[ind + 3] = norm;
}

////////////////////////////////////////////////////////////////////////////////
///Tube segment is drawn as three quad strips
///1. enabling vertex arrays

void TubeSegMesh::Draw() const
{
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);
   //2. setting arrays
   glVertexPointer(3, GL_DOUBLE, sizeof(TGLVertex3), fMesh[0].CArr());
   glNormalPointer(GL_DOUBLE, sizeof(TGLVector3), fNorm[0].CArr());
   //3. draw first strip
   glDrawArrays(GL_QUAD_STRIP, 0, 4 * (fLOD + 1) + 8);
   //4. draw top and bottom strips
   glDrawArrays(GL_QUAD_STRIP, 4 * (fLOD + 1) + 8, 2 * (fLOD + 1));
   glDrawArrays(GL_QUAD_STRIP, 6 * (fLOD + 1) + 8, 2 * (fLOD + 1));

   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_NORMAL_ARRAY);
}

////////////////////////////////////////////////////////////////////////////////
/// constructor

TubeMesh::TubeMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t r3, Double_t r4, Double_t z,
                   const TGLVector3 &l, const TGLVector3 &h)
             :TGLMesh(LOD, r1, r2, r3, r4, z, l, h), fMesh(), fNorm()
{
   const Double_t delta = TMath::TwoPi() / fLOD;
   Double_t currAngle = 0.;

   Bool_t even = kTRUE;
   Double_t c = TMath::Cos(currAngle);
   Double_t s = TMath::Sin(currAngle);

   const Int_t topShift = (fLOD + 1) * 4;
   const Int_t botShift = (fLOD + 1) * 6;
   Int_t j = 4 * (fLOD + 1) - 2;

   //defining all four strips here
   for (Int_t i = 0, e = (fLOD + 1) * 2; i < e; ++i) {
      if (even) {
         fMesh[i] = MakeVertex(fRmax2 * c, fRmax2 * s, fDz);
         fMesh[j] = MakeVertex(fRmin2 * c, fRmin2 * s, fDz);
         fMesh[i + topShift] = MakeVertex(fRmin2 * c, fRmin2 * s, fDz);
         fMesh[i + botShift] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         GetNormal(fMesh[j], fNorm[j]);
         fNorm[j].Negate();
         even = kFALSE;
      } else {
         fMesh[i] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         fMesh[j + 1] = MakeVertex(fRmin1 * c, fRmin1 * s, -fDz);
         fMesh[i + topShift] = MakeVertex(fRmax2 * c, fRmax2 * s, fDz);
         fMesh[i + botShift] = MakeVertex(fRmin1 * c, fRmin1 * s, - fDz);
         GetNormal(fMesh[j + 1], fNorm[j + 1]);
         fNorm[j + 1].Negate();
         even = kTRUE;
         currAngle += delta;
         c = TMath::Cos(currAngle);
         s = TMath::Sin(currAngle);
         j -= 2;
      }

      GetNormal(fMesh[i], fNorm[i]);
      fNorm[i + topShift] = fNhigh;
      fNorm[i + botShift] = fNlow;
   }
}

////////////////////////////////////////////////////////////////////////////////
///Tube is drawn as four quad strips

void TubeMesh::Draw() const
{
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);

   glVertexPointer(3, GL_DOUBLE, sizeof(TGLVertex3), fMesh[0].CArr());
   glNormalPointer(GL_DOUBLE, sizeof(TGLVector3), fNorm[0].CArr());
   //draw outer and inner strips
   glDrawArrays(GL_QUAD_STRIP, 0, 2 * (fLOD + 1));
   glDrawArrays(GL_QUAD_STRIP, 2 * (fLOD + 1), 2 * (fLOD + 1));
   //draw top and bottom strips
   glDrawArrays(GL_QUAD_STRIP, 4 * (fLOD + 1), 2 * (fLOD + 1));
   glDrawArrays(GL_QUAD_STRIP, 6 * (fLOD + 1), 2 * (fLOD + 1));
   //5. disabling vertex arrays
   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_NORMAL_ARRAY);
}

////////////////////////////////////////////////////////////////////////////////
/// constructor

TCylinderMesh::TCylinderMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t dz,
                             const TGLVector3 &l, const TGLVector3 &h)
                 :TGLMesh(LOD, 0., r1, 0., r2, dz, l, h), fMesh(), fNorm()
{
   const Double_t delta = TMath::TwoPi() / fLOD;
   Double_t currAngle = 0.;

   Bool_t even = kTRUE;
   Double_t c = TMath::Cos(currAngle);
   Double_t s = TMath::Sin(currAngle);

   //central point of top fan
   Int_t topShift = (fLOD + 1) * 2;
   fMesh[topShift][0] = fMesh[topShift][1] = 0., fMesh[topShift][2] = fDz;
   fNorm[topShift] = fNhigh;
   ++topShift;

   //central point of bottom fun
   Int_t botShift = topShift + 2 * (fLOD + 1);
   fMesh[botShift][0] = fMesh[botShift][1] = 0., fMesh[botShift][2] = -fDz;
   fNorm[botShift] = fNlow;
   ++botShift;

   //defining 1 strip and 2 fans
   for (Int_t i = 0, e = (fLOD + 1) * 2, j = 0; i < e; ++i) {
      if (even) {
         fMesh[i] = MakeVertex(fRmax2 * c, fRmax2 * s, fDz);
         fMesh[j + topShift] = MakeVertex(fRmin2 * c, fRmin2 * s, fDz);
         fMesh[j + botShift] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         even = kFALSE;
      } else {
         fMesh[i] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         even = kTRUE;
         currAngle += delta;
         c = TMath::Cos(currAngle);
         s = TMath::Sin(currAngle);
         ++j;
      }

      GetNormal(fMesh[i], fNorm[i]);
      fNorm[i + topShift] = fNhigh;
      fNorm[i + botShift] = fNlow;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// draw cylinder mesh

void TCylinderMesh::Draw() const
{
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);

   glVertexPointer(3, GL_DOUBLE, sizeof(TGLVertex3), fMesh[0].CArr());
   glNormalPointer(GL_DOUBLE, sizeof(TGLVector3), fNorm[0].CArr());

   //draw quad strip
   glDrawArrays(GL_QUAD_STRIP, 0, 2 * (fLOD + 1));
   //draw top and bottom funs
   glDrawArrays(GL_TRIANGLE_FAN, 2 * (fLOD + 1), fLOD + 2);
   glDrawArrays(GL_TRIANGLE_FAN, 3 * (fLOD + 1) + 1, fLOD + 2);

   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_NORMAL_ARRAY);
}

////////////////////////////////////////////////////////////////////////////////
///One quad mesh and two fans

TCylinderSegMesh::TCylinderSegMesh(UInt_t LOD, Double_t r1, Double_t r2, Double_t dz, Double_t phi1,
                                    Double_t phi2, const TGLVector3 &l,
                                    const TGLVector3 &h)
                     :TGLMesh(LOD, 0., r1, 0., r2, dz, l, h), fMesh(), fNorm()
{
   Double_t delta = (phi2 - phi1) / fLOD;
   Double_t currAngle = phi1;

   Bool_t even = kTRUE;
   Double_t c = TMath::Cos(currAngle);
   Double_t s = TMath::Sin(currAngle);

   const TGLVertex3 vTop(0., 0., fDz);
   const TGLVertex3 vBot(0., 0., - fDz);

   //center of top fan
   Int_t topShift = (fLOD + 1) * 2 + 8;
   fMesh[topShift] = vTop;
   fNorm[topShift] = fNhigh;
   ++topShift;

   //center of bottom fan
   Int_t botShift = topShift + fLOD + 1;
   fMesh[botShift] = vBot;
   fNorm[botShift] = fNlow;
   ++botShift;

   //defining strip and two fans
   //strip is not closed here
   Int_t i = 0;
   for (Int_t e = (fLOD + 1) * 2, j = 0; i < e; ++i) {
      if (even) {
         fMesh[i] = MakeVertex(fRmax2 * c, fRmax2 * s, fDz);
         fMesh[j + topShift] = MakeVertex(fRmax2 * c, fRmax2 * s, fDz);
         fMesh[j + botShift] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         even = kFALSE;
         fNorm[j + topShift] = fNhigh;
         fNorm[j + botShift] = fNlow;
      } else {
         fMesh[i] = MakeVertex(fRmax1 * c, fRmax1 * s, - fDz);
         even = kTRUE;
         currAngle += delta;
         c = TMath::Cos(currAngle);
         s = TMath::Sin(currAngle);
         ++j;
      }

      GetNormal(fMesh[i], fNorm[i]);
   }

   //closing first strip
   Int_t ind = 2 * (fLOD + 1);
   TGLVector3 norm(0., 0., 0.);

   fMesh[ind] = fMesh[ind - 2];
   fMesh[ind + 1] = fMesh[ind - 1];
   fMesh[ind + 2] = vTop;
   fMesh[ind + 3] = vBot;
   TMath::Normal2Plane(fMesh[ind].CArr(), fMesh[ind + 1].CArr(), fMesh[ind + 2].CArr(),
                          norm.Arr());
   fNorm[ind] = norm;
   fNorm[ind + 1] = norm;
   fNorm[ind + 2] = norm;
   fNorm[ind + 3] = norm;

   ind += 4;
   fMesh[ind] = vTop;
   fMesh[ind + 1] = vBot;
   fMesh[ind + 2] = fMesh[0];
   fMesh[ind + 3] = fMesh[1];
   TMath::Normal2Plane(fMesh[ind].CArr(), fMesh[ind + 1].CArr(), fMesh[ind + 2].CArr(),
                       norm.Arr());
   fNorm[ind] = norm;
   fNorm[ind + 1] = norm;
   fNorm[ind + 2] = norm;
   fNorm[ind + 3] = norm;
}

////////////////////////////////////////////////////////////////////////////////
///Cylinder segment is drawn as one quad strip and
///two triangle fans
///1. enabling vertex arrays

void TCylinderSegMesh::Draw() const
{
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);
   //2. setting arrays
   glVertexPointer(3, GL_DOUBLE, sizeof(TGLVertex3), fMesh[0].CArr());
   glNormalPointer(GL_DOUBLE, sizeof(TGLVector3), fNorm[0].CArr());
   //3. draw quad strip
   glDrawArrays(GL_QUAD_STRIP, 0, 2 * (fLOD + 1) + 8);
   //4. draw top and bottom funs
   glDrawArrays(GL_TRIANGLE_FAN, 2 * (fLOD + 1) + 8, fLOD + 2);
   //      glDrawArrays(GL_TRIANGLE_FAN, 3 * (fLOD + 1) + 9, fLOD + 2);
   //5. disabling vertex arrays
   glDisableClientState(GL_VERTEX_ARRAY);
   glDisableClientState(GL_NORMAL_ARRAY);
}


/** \class TGLCylinder
\ingroup opengl
Implements a native ROOT-GL cylinder that can be rendered at
different levels of detail.
*/

ClassImp(TGLCylinder);

////////////////////////////////////////////////////////////////////////////////
/// Copy out relevant parts of buffer - we create and delete mesh
/// parts on demand in DirectDraw() and they are DL cached

TGLCylinder::TGLCylinder(const TBuffer3DTube &buffer) :
   TGLLogicalShape(buffer)
{
   fDLSize = 14;

   fR1 = buffer.fRadiusInner;
   fR2 = buffer.fRadiusOuter;
   fR3 = buffer.fRadiusInner;
   fR4 = buffer.fRadiusOuter;
   fDz = buffer.fHalfLength;

   fLowPlaneNorm = gLowNormalDefault;
   fHighPlaneNorm = gHighNormalDefault;

   switch (buffer.Type())
   {
      default:
      case TBuffer3DTypes::kTube:
      {
         fSegMesh = kFALSE;
         fPhi1 = 0;
         fPhi2 = 360;
         break;
      }

      case TBuffer3DTypes::kTubeSeg:
      case TBuffer3DTypes::kCutTube:
      {
         fSegMesh = kTRUE;

         const TBuffer3DTubeSeg * segBuffer = dynamic_cast<const TBuffer3DTubeSeg *>(&buffer);
         if (!segBuffer) {
            Error("TGLCylinder::TGLCylinder", "cannot cast TBuffer3D");
            return;
         }

         fPhi1 = segBuffer->fPhiMin;
         fPhi2 = segBuffer->fPhiMax;
         if (fPhi2 < fPhi1) fPhi2 += 360.;
         fPhi1 *= TMath::DegToRad();
         fPhi2 *= TMath::DegToRad();

         if (buffer.Type() == TBuffer3DTypes::kCutTube) {
            const TBuffer3DCutTube * cutBuffer = dynamic_cast<const TBuffer3DCutTube *>(&buffer);
            if (!cutBuffer) {
               Error("TGLCylinder::TGLCylinder", "cannot cast TBuffer3D");
               return;
            }

            for (UInt_t i =0; i < 3; i++) {
               fLowPlaneNorm[i] = cutBuffer->fLowPlaneNorm[i];
               fHighPlaneNorm[i] = cutBuffer->fHighPlaneNorm[i];
            }
         }
         break;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
///destructor

TGLCylinder::~TGLCylinder()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Return display-list offset for given LOD.
/// Calculation based on what is done in virtual QuantizeShapeLOD below.

UInt_t TGLCylinder::DLOffset(Short_t lod) const
{
   UInt_t  off = 0;
   if      (lod >= 100) off = 0;
   else if (lod <  10)  off = lod / 2;
   else                 off = lod / 10 + 4;
   return off;
}

////////////////////////////////////////////////////////////////////////////////
/// Factor in scene/viewer LOD and quantize.

Short_t TGLCylinder::QuantizeShapeLOD(Short_t shapeLOD, Short_t combiLOD) const
{
   Int_t lod = ((Int_t)shapeLOD * (Int_t)combiLOD) / 100;

   if (lod >= 100)
   {
      lod = 100;
   }
   else if (lod > 10)
   {  // Round LOD above 10 to nearest 10
      Double_t quant = 0.1 * ((static_cast<Double_t>(lod)) + 0.5);
      lod            = 10  *   static_cast<Int_t>(quant);
   }
   else
   {  // Round LOD below 10 to nearest 2
      Double_t quant = 0.5 * ((static_cast<Double_t>(lod)) + 0.5);
      lod            = 2   *   static_cast<Int_t>(quant);
   }
   return static_cast<Short_t>(lod);
}

////////////////////////////////////////////////////////////////////////////////
/// Debug tracing

void TGLCylinder::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   if (gDebug > 4) {
      Info("TGLCylinder::DirectDraw", "this %zd (class %s) LOD %d",
           (size_t)this, IsA()->GetName(), rnrCtx.ShapeLOD());
   }

   // As we are now support display list caching we can create, draw and
   // delete mesh parts of suitable LOD (quality) here - it will be cached
   // into a display list by base-class TGLLogicalShape::Draw(),
   // against our id and the LOD value. So this will only occur once
   // for a certain cylinder/LOD combination
   std::vector<TGLMesh *> meshParts;

   // Create mesh parts
   if (!fSegMesh) {
      meshParts.push_back(new TubeMesh   (rnrCtx.ShapeLOD(), fR1, fR2, fR3, fR4,
                                          fDz, fLowPlaneNorm, fHighPlaneNorm));
   } else {
      meshParts.push_back(new TubeSegMesh(rnrCtx.ShapeLOD(), fR1, fR2, fR3, fR4,
                                          fDz, fPhi1, fPhi2,
                                          fLowPlaneNorm, fHighPlaneNorm));
   }

   // Draw mesh parts
   for (UInt_t i = 0; i < meshParts.size(); ++i) meshParts[i]->Draw();

   // Delete mesh parts
   for (UInt_t i = 0; i < meshParts.size(); ++i) {
      delete meshParts[i];
      meshParts[i] = 0;//not to have invalid pointer for pseudo-destructor call :)
   }
}
