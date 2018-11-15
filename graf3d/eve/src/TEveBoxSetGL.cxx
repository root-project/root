// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveBoxSetGL.h"
#include "TEveBoxSet.h"

#include "TGLIncludes.h"
#include "TGLRnrCtx.h"
#include "TGLSelectRecord.h"
#include "TGLQuadric.h"

/** \class TEveBoxSetGL
\ingroup TEve
A GL rendering class for TEveBoxSet.
*/

ClassImp(TEveBoxSetGL);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

TEveBoxSetGL::TEveBoxSetGL() : TEveDigitSetGL(), fM(0), fBoxDL(0)
{
   fDLCache = kFALSE; // Disable display list, used internally for boxes, cones.
   fMultiColor = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TEveBoxSetGL::~TEveBoxSetGL()
{
   DLCachePurge();
}

////////////////////////////////////////////////////////////////////////////////
/// Return GL primitive used to render the boxes, based on the
/// render-mode specified in the model object.

Int_t TEveBoxSetGL::PrimitiveType() const
{
   return (fM->fRenderMode != TEveDigitSet::kRM_Line) ? GL_QUADS : GL_LINE_LOOP;
}

////////////////////////////////////////////////////////////////////////////////
/// Fill array p to represent a box (0,0,0) - (dx,dy,dz).

void TEveBoxSetGL::MakeOriginBox(Float_t p[8][3], Float_t dx, Float_t dy, Float_t dz) const
{
   // bottom
   p[0][0] = 0;  p[0][1] = dy; p[0][2] = 0;
   p[1][0] = dx; p[1][1] = dy; p[1][2] = 0;
   p[2][0] = dx; p[2][1] = 0;  p[2][2] = 0;
   p[3][0] = 0;  p[3][1] = 0;  p[3][2] = 0;
   // top
   p[4][0] = 0;  p[4][1] = dy; p[4][2] = dz;
   p[5][0] = dx; p[5][1] = dy; p[5][2] = dz;
   p[6][0] = dx; p[6][1] = 0;  p[6][2] = dz;
   p[7][0] = 0;  p[7][1] = 0;  p[7][2] = dz;
}

////////////////////////////////////////////////////////////////////////////////
/// Render a box specified by points in array p with standard
/// axis-aligned normals.

inline void TEveBoxSetGL::RenderBoxStdNorm(const Float_t p[8][3]) const
{
   // bottom: 0123
   glNormal3f(0, 0, -1);
   glVertex3fv(p[0]);  glVertex3fv(p[1]);
   glVertex3fv(p[2]);  glVertex3fv(p[3]);
   // top:    7654
   glNormal3f(0, 0, 1);
   glVertex3fv(p[7]); glVertex3fv(p[6]);
   glVertex3fv(p[5]); glVertex3fv(p[4]);
   // back:  0451
   glNormal3f(0, 1, 0);
   glVertex3fv(p[0]); glVertex3fv(p[4]);
   glVertex3fv(p[5]); glVertex3fv(p[1]);
   // front:   3267
   glNormal3f(0, -1, 0);
   glVertex3fv(p[3]);  glVertex3fv(p[2]);
   glVertex3fv(p[6]);  glVertex3fv(p[7]);
   // left:    0374
   glNormal3f(-1, 0, 0);
   glVertex3fv(p[0]);  glVertex3fv(p[3]);
   glVertex3fv(p[7]);  glVertex3fv(p[4]);
   // right:   1562
   glNormal3f(1, 0, 0);
   glVertex3fv(p[1]);  glVertex3fv(p[5]);
   glVertex3fv(p[6]);  glVertex3fv(p[2]);
}

namespace
{
   void subtract_and_normalize(const Float_t a[3], const Float_t b[3],
                               Float_t o[3])
   {
      // Calculate a - b and normalize the result.
      o[0] = a[0] - b[0];
      o[1] = a[1] - b[1];
      o[2] = a[2] - b[2];
      Float_t d = sqrtf(o[0]*o[0] + o[1]*o[1] + o[2]*o[2]);
      if (d != 0)
      {
         d = 1.0f / d;
         o[0] *= d;
         o[1] *= d;
         o[2] *= d;
      }
   }
}
////////////////////////////////////////////////////////////////////////////////
/// Render box, calculate normals on the fly from first three points.

void TEveBoxSetGL::RenderBoxAutoNorm(const Float_t p[8][3]) const
{
   Float_t e[6][3], n[3];
   subtract_and_normalize(p[1], p[0], e[0]);
   subtract_and_normalize(p[3], p[0], e[1]);
   subtract_and_normalize(p[4], p[0], e[2]);
   subtract_and_normalize(p[5], p[6], e[3]);
   subtract_and_normalize(p[7], p[6], e[4]);
   subtract_and_normalize(p[2], p[6], e[5]);

   // bottom: 0123
   glNormal3fv(TMath::Cross(e[0], e[1], n));
   glVertex3fv(p[0]); glVertex3fv(p[1]);
   glVertex3fv(p[2]); glVertex3fv(p[3]);
   // top:    7654
   glNormal3fv(TMath::Cross(e[3], e[4], n));
   glVertex3fv(p[7]); glVertex3fv(p[6]);
   glVertex3fv(p[5]); glVertex3fv(p[4]);
   // back:  0451
   glNormal3fv(TMath::Cross(e[2], e[0], n));
   glVertex3fv(p[0]); glVertex3fv(p[4]);
   glVertex3fv(p[5]); glVertex3fv(p[1]);
   // front:   3267
   glNormal3fv(TMath::Cross(e[4], e[5], n));
   glVertex3fv(p[3]); glVertex3fv(p[2]);
   glVertex3fv(p[6]); glVertex3fv(p[7]);
   // left:    0374
   glNormal3fv(TMath::Cross(e[1], e[2], n));
   glVertex3fv(p[0]); glVertex3fv(p[3]);
   glVertex3fv(p[7]); glVertex3fv(p[4]);
   // right:   1562
   glNormal3fv(TMath::Cross(e[5], e[3], n));
   glVertex3fv(p[1]); glVertex3fv(p[5]);
   glVertex3fv(p[6]); glVertex3fv(p[2]);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a display-list for rendering a single box, based on the
/// current box-type.
/// Some box-types don't benefit from the display-list rendering and
/// so display-list is not created.

void TEveBoxSetGL::MakeDisplayList() const
{
   if (fM->fBoxType == TEveBoxSet::kBT_AABox         ||
       fM->fBoxType == TEveBoxSet::kBT_AABoxFixedDim ||
       fM->fBoxType == TEveBoxSet::kBT_Cone          ||
       fM->fBoxType == TEveBoxSet::kBT_EllipticCone  ||
       fM->fBoxType == TEveBoxSet::kBT_Hex)
   {
      if (fBoxDL == 0)
         fBoxDL = glGenLists(1);

      glNewList(fBoxDL, GL_COMPILE);

      if (fM->fBoxType < TEveBoxSet::kBT_Cone)
      {
         glBegin(PrimitiveType());
         Float_t p[8][3];
         if (fM->fBoxType == TEveBoxSet::kBT_AABox)
            MakeOriginBox(p, 1.0f, 1.0f, 1.0f);
         else
            MakeOriginBox(p, fM->fDefWidth, fM->fDefHeight, fM->fDefDepth);
         RenderBoxStdNorm(p);
         glEnd();
      }
      else if (fM->fBoxType < TEveBoxSet::kBT_Hex)
      {
         static TGLQuadric quad;
         Int_t nt = 15; // number of corners
         gluCylinder(quad.Get(), 0, 1, 1, nt, 1);

         if (fM->fDrawConeCap)
         {
            glPushMatrix();
            glTranslatef(0, 0, 1);
            gluDisk(quad.Get(), 0, 1, nt, 1);
            glPopMatrix();
         }
      }
      else // Hexagons
      {
         static TGLQuadric quad;
         Int_t nt = 6; // number of corners
         gluCylinder(quad.Get(), 1, 1, 1, nt, 1);

         gluQuadricOrientation(quad.Get(), GLU_INSIDE);
         gluDisk(quad.Get(), 0, 1, nt, 1);
         gluQuadricOrientation(quad.Get(), GLU_OUTSIDE);

         glPushMatrix();
         glTranslatef(0, 0, 1);
         gluDisk(quad.Get(), 0, 1, nt, 1);
         glPopMatrix();
      }

      glEndList();

      TGLUtil::CheckError("TEveBoxSetGL::MakeDisplayList");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Determines if display-list will be used for rendering.
/// Virtual from TGLLogicalShape.

Bool_t TEveBoxSetGL::ShouldDLCache(const TGLRnrCtx& rnrCtx) const
{
   return TEveDigitSetGL::ShouldDLCache(rnrCtx);
}

////////////////////////////////////////////////////////////////////////////////
/// Called when display lists have been destroyed externally and the
/// internal display-list data needs to be cleare.
/// Virtual from TGLLogicalShape.

void TEveBoxSetGL::DLCacheDrop()
{
   fBoxDL = 0;
   TGLObject::DLCacheDrop();
}

////////////////////////////////////////////////////////////////////////////////
/// Called when display-lists need to be returned to the system.
/// Virtual from TGLLogicalShape.

void TEveBoxSetGL::DLCachePurge()
{
   if (fBoxDL != 0)
   {
      PurgeDLRange(fBoxDL, 1);
      fBoxDL = 0;
   }
   TGLObject::DLCachePurge();
}

////////////////////////////////////////////////////////////////////////////////
/// Set model object.
/// Virtual from TGLObject.

Bool_t TEveBoxSetGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   fM = SetModelDynCast<TEveBoxSet>(obj);
   return kTRUE;
}

namespace
{
  inline void AntiFlick(Float_t x, Float_t y, Float_t z)
  {
     // Render anti-flickering point.
     glBegin(GL_POINTS);
     glVertex3f(x, y, z);
     glEnd();
  }
}

////////////////////////////////////////////////////////////////////////////////
/// GL rendering for all box-types.

void TEveBoxSetGL::RenderBoxes(TGLRnrCtx& rnrCtx) const
{
   static const TEveException eH("TEveBoxSetGL::RenderBoxes ");

   if (rnrCtx.SecSelection()) glPushName(0);

   Int_t boxSkip = 0;
   if (fM->fBoxSkip > 0 && rnrCtx.CombiLOD() < TGLRnrCtx::kLODHigh &&
       !rnrCtx.SecSelection())
   {
      boxSkip = TMath::Nint(TMath::Power(fM->fBoxSkip, 2.0 - 0.02*rnrCtx.CombiLOD()));
   }

   TEveChunkManager::iterator bi(fM->fPlex);
   if (rnrCtx.Highlight() && fHighlightSet)
      bi.fSelection = fHighlightSet;

   switch (fM->fBoxType)
   {

      case TEveBoxSet::kBT_FreeBox:
      {
         GLenum primitiveType = PrimitiveType();
         while (bi.next())
         {
            TEveBoxSet::BFreeBox_t& b = * (TEveBoxSet::BFreeBox_t*) bi();
            if (SetupColor(b))
            {
               if (rnrCtx.SecSelection()) glLoadName(bi.index());
               glBegin(primitiveType);
               RenderBoxAutoNorm(b.fVertices);
               glEnd();
               if (fM->fAntiFlick)
                  AntiFlick(0.5f*(b.fVertices[0][0] + b.fVertices[6][0]),
                            0.5f*(b.fVertices[0][1] + b.fVertices[6][1]),
                            0.5f*(b.fVertices[0][2] + b.fVertices[6][2]));
            }
            if (boxSkip) { Int_t s = boxSkip; while (s--) bi.next(); }
         }
         break;
      } // end case free-box

      case TEveBoxSet::kBT_AABox:
      {
         glEnable(GL_NORMALIZE);
         while (bi.next())
         {
            TEveBoxSet::BAABox_t& b = * (TEveBoxSet::BAABox_t*) bi();
            if (SetupColor(b))
            {
               if (rnrCtx.SecSelection()) glLoadName(bi.index());
               glPushMatrix();
               glTranslatef(b.fA, b.fB, b.fC);
               glScalef    (b.fW, b.fH, b.fD);
               glCallList(fBoxDL);
               if (fM->fAntiFlick)
                  AntiFlick(0.5f, 0.5f, 0.5f);
               glPopMatrix();
            }
            if (boxSkip) { Int_t s = boxSkip; while (s--) bi.next(); }
         }
         break;
      }

      case TEveBoxSet::kBT_AABoxFixedDim:
      {
         while (bi.next())
         {
            TEveBoxSet::BAABoxFixedDim_t& b = * (TEveBoxSet::BAABoxFixedDim_t*) bi();
            if (SetupColor(b))
            {
               if (rnrCtx.SecSelection()) glLoadName(bi.index());
               glTranslatef(b.fA, b.fB, b.fC);
               glCallList(fBoxDL);
               if (fM->fAntiFlick)
                  AntiFlick(0.5f*fM->fDefWidth, 0.5f*fM->fDefHeight, 0.5f*fM->fDefDepth);
               glTranslatef(-b.fA, -b.fB, -b.fC);
            }
            if (boxSkip) { Int_t s = boxSkip; while (s--) bi.next(); }
         }
         break;
      }

      case TEveBoxSet::kBT_Cone:
      {
         using namespace TMath;

         glEnable(GL_NORMALIZE);
         Float_t theta=0, phi=0, h=0;
         while (bi.next())
         {
            TEveBoxSet::BCone_t& b = * (TEveBoxSet::BCone_t*) bi();
            if (SetupColor(b))
            {
               if (rnrCtx.SecSelection()) glLoadName(bi.index());
               h     = b.fDir.Mag();
               phi   = ATan2(b.fDir.fY, b.fDir.fX)*RadToDeg();
               theta = ATan (b.fDir.fZ / Sqrt(b.fDir.fX*b.fDir.fX + b.fDir.fY*b.fDir.fY))*RadToDeg();
               glPushMatrix();
               glTranslatef(b.fPos.fX, b.fPos.fY, b.fPos.fZ);
               glRotatef(phi,        0, 0, 1);
               glRotatef(90 - theta, 0, 1, 0);
               glScalef (b.fR, b.fR, h);
               glCallList(fBoxDL);
               if (fM->fAntiFlick)
                  AntiFlick(0.0f, 0.0f, 0.5f);
               glPopMatrix();
            }
            if (boxSkip) { Int_t s = boxSkip; while (s--) bi.next(); }
         }
         break;
      }

      case TEveBoxSet::kBT_EllipticCone:
      {
         using namespace TMath;

         glEnable(GL_NORMALIZE);
         Float_t theta=0, phi=0, h=0;
         while (bi.next())
         {
            TEveBoxSet::BEllipticCone_t& b = * (TEveBoxSet::BEllipticCone_t*) bi();
            if (SetupColor(b))
            {
               if (rnrCtx.SecSelection()) glLoadName(bi.index());
               h     = b.fDir.Mag();
               phi   = ATan2(b.fDir.fY, b.fDir.fX)*RadToDeg();
               theta = ATan (b.fDir.fZ / Sqrt(b.fDir.fX*b.fDir.fX + b.fDir.fY*b.fDir.fY))*RadToDeg();
               glPushMatrix();
               glTranslatef(b.fPos.fX, b.fPos.fY, b.fPos.fZ);
               glRotatef(phi,        0, 0, 1);
               glRotatef(90 - theta, 0, 1, 0);
               glRotatef(b.fAngle,   0, 0, 1);
               glScalef (b.fR, b.fR2, h);
               glCallList(fBoxDL);
               if (fM->fAntiFlick)
                  AntiFlick(0.0f, 0.0f, 0.5f);
               glPopMatrix();
            }
            if (boxSkip) { Int_t s = boxSkip; while (s--) bi.next(); }
         }
         break;
      }

      case TEveBoxSet::kBT_Hex:
      {
         using namespace TMath;

         glEnable(GL_NORMALIZE);
         while (bi.next())
         {
            TEveBoxSet::BHex_t& h = * (TEveBoxSet::BHex_t*) bi();
            if (SetupColor(h))
            {
               if (rnrCtx.SecSelection()) glLoadName(bi.index());
               glPushMatrix();
               glTranslatef(h.fPos.fX, h.fPos.fY, h.fPos.fZ);
               glRotatef(h.fAngle, 0, 0, 1);
               glScalef (h.fR, h.fR, h.fDepth);
               glCallList(fBoxDL);
               if (fM->fAntiFlick)
                  AntiFlick(0.0f, 0.0f, 0.5f);
               glPopMatrix();
            }
            if (boxSkip) { Int_t s = boxSkip; while (s--) bi.next(); }
         }
         break;
      }

      default:
      {
         throw eH + "unsupported box-type.";
      }

   } // end switch box-type

   if (rnrCtx.SecSelection()) glPopName();
}

////////////////////////////////////////////////////////////////////////////////
/// Actual rendering code.
/// Virtual from TGLLogicalShape.

void TEveBoxSetGL::DirectDraw(TGLRnrCtx& rnrCtx) const
{
   TEveBoxSet& mB = * fM;
   // printf("TEveBoxSetGL::DirectDraw N boxes %d\n", mB.fPlex.Size());

   if (mB.fPlex.Size() > 0)
   {
      MakeDisplayList();

      if (! mB.fSingleColor && ! mB.fValueIsColor && mB.fPalette == 0)
      {
         mB.AssertPalette();
      }

      glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT);

      if ( ! rnrCtx.IsDrawPassOutlineLine())
      {
         if (mB.fRenderMode == TEveDigitSet::kRM_Fill)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
         else if (mB.fRenderMode == TEveDigitSet::kRM_Line)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      }

      if (mB.fBoxType == TEveBoxSet::kBT_Cone ||
          mB.fBoxType == TEveBoxSet::kBT_EllipticCone)
      {
         glDisable(GL_CULL_FACE);
      }

      if (mB.fDisableLighting) glDisable(GL_LIGHTING);

      RenderBoxes(rnrCtx);

      glPopAttrib();
   }

   DrawFrameIfNeeded(rnrCtx);
}

////////////////////////////////////////////////////////////////////////////////
/// Interface for direct rendering from classes that include TEveBoxSet
/// as a member.

void TEveBoxSetGL::Render(TGLRnrCtx& rnrCtx)
{
   DirectDraw(rnrCtx);
}
