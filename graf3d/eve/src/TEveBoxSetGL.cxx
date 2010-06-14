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
#include "TEveFrameBoxGL.h"

#include "TGLIncludes.h"
#include "TGLRnrCtx.h"
#include "TGLSelectRecord.h"
#include "TGLQuadric.h"

//==============================================================================
//==============================================================================
// TEveBoxSetGL
//==============================================================================

//______________________________________________________________________________
//
// A GL rendering class for TEveBoxSet.
//

ClassImp(TEveBoxSetGL);

//______________________________________________________________________________
TEveBoxSetGL::TEveBoxSetGL() : TEveDigitSetGL(), fM(0), fBoxDL(0)
{
   // Default constructor.

   // fDLCache = false; // Disable display list.
   fMultiColor = kTRUE;
}

//______________________________________________________________________________
TEveBoxSetGL::~TEveBoxSetGL()
{
   // Destructor.

   DLCachePurge();
}

/******************************************************************************/
// Protected methods
/******************************************************************************/

//______________________________________________________________________________
Int_t TEveBoxSetGL::PrimitiveType() const
{
   // Return GL primitive used to render the boxes, based on the
   // render-mode specified in the model object.

   return (fM->fRenderMode != TEveDigitSet::kRM_Line) ? GL_QUADS : GL_LINE_LOOP;
}

//______________________________________________________________________________
void TEveBoxSetGL::MakeOriginBox(Float_t p[24], Float_t dx, Float_t dy, Float_t dz) const
{
   // Fill array p to represent a box (0,0,0) - (dx,dy,dz).

   // bottom
   p[0] = 0;  p[1] = dy; p[2] = 0;  p += 3;
   p[0] = dx; p[1] = dy; p[2] = 0;  p += 3;
   p[0] = dx; p[1] = 0;  p[2] = 0;  p += 3;
   p[0] = 0;  p[1] = 0;  p[2] = 0;  p += 3;
   // top
   p[0] = 0;  p[1] = dy; p[2] = dz; p += 3;
   p[0] = dx; p[1] = dy; p[2] = dz; p += 3;
   p[0] = dx; p[1] = 0;  p[2] = dz; p += 3;
   p[0] = 0;  p[1] = 0;  p[2] = dz;
}

//______________________________________________________________________________
inline void TEveBoxSetGL::RenderBox(const Float_t p[24]) const
{
   // Render a box specified by points in array p.

   // bottom: 0123
   glNormal3f(0, 0, -1);
   glVertex3fv(p);      glVertex3fv(p + 3);
   glVertex3fv(p + 6);  glVertex3fv(p + 9);
   // top:    7654
   glNormal3f(0, 0, 1);
   glVertex3fv(p + 21); glVertex3fv(p + 18);
   glVertex3fv(p + 15); glVertex3fv(p + 12);
   // back:  0451
   glNormal3f(0, 1, 0);
   glVertex3fv(p);      glVertex3fv(p + 12);
   glVertex3fv(p + 15); glVertex3fv(p + 3);
   // front:   3267
   glNormal3f(0, -1, 0);
   glVertex3fv(p + 9);   glVertex3fv(p + 6);
   glVertex3fv(p + 18);  glVertex3fv(p + 21);
   // left:    0374
   glNormal3f(-1, 0, 0);
   glVertex3fv(p);       glVertex3fv(p + 9);
   glVertex3fv(p + 21);  glVertex3fv(p + 12);
   // right:   1562
   glNormal3f(1, 0, 0);
   glVertex3fv(p + 3);   glVertex3fv(p + 15);
   glVertex3fv(p + 18);  glVertex3fv(p + 6);
}

//______________________________________________________________________________
void TEveBoxSetGL::MakeDisplayList() const
{
   // Create a display-list for rendering a single box, based on the
   // current box-type.
   // Some box-types don't benefit from the display-list rendering and
   // so display-list is not created.

   if (fM->fBoxType == TEveBoxSet::kBT_AABox         ||
       fM->fBoxType == TEveBoxSet::kBT_AABoxFixedDim ||
       fM->fBoxType == TEveBoxSet::kBT_Cone          ||
       fM->fBoxType == TEveBoxSet::kBT_EllipticCone)
   {
      if (fBoxDL == 0)
         fBoxDL = glGenLists(1);

      glNewList(fBoxDL, GL_COMPILE);

      if (fM->fBoxType < TEveBoxSet::kBT_Cone)
      {
         glBegin(PrimitiveType());
         Float_t p[24];
         if (fM->fBoxType == TEveBoxSet::kBT_AABox)
            MakeOriginBox(p, 1.0f, 1.0f, 1.0f);
         else
            MakeOriginBox(p, fM->fDefWidth, fM->fDefHeight, fM->fDefDepth);
         RenderBox(p);
         glEnd();
      }
      else
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

      glEndList();
   }
}

/******************************************************************************/
// Virtuals from base-classes
/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveBoxSetGL::ShouldDLCache(const TGLRnrCtx& rnrCtx) const
{
   // Determines if display-list will be used for rendering.
   // Virtual from TGLLogicalShape.

   MakeDisplayList();

   return TEveDigitSetGL::ShouldDLCache(rnrCtx);
}

//______________________________________________________________________________
void TEveBoxSetGL::DLCacheDrop()
{
   // Called when display lists have been destroyed externally and the
   // internal display-list data needs to be cleare.
   // Virtual from TGLLogicalShape.

   fBoxDL = 0;
   TGLObject::DLCacheDrop();
}

//______________________________________________________________________________
void TEveBoxSetGL::DLCachePurge()
{
   // Called when display-lists need to be returned to the system.
   // Virtual from TGLLogicalShape.

   if (fBoxDL != 0)
   {
      PurgeDLRange(fBoxDL, 1);
      fBoxDL = 0;
   }
   TGLObject::DLCachePurge();
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveBoxSetGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.
   // Virtual from TGLObject.

   if (SetModelCheckClass(obj, TEveBoxSet::Class())) {
      fM = dynamic_cast<TEveBoxSet*>(obj);
      return kTRUE;
   }
   return kFALSE;
}

/******************************************************************************/

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

/******************************************************************************/

//______________________________________________________________________________
void TEveBoxSetGL::RenderBoxes(TGLRnrCtx& rnrCtx) const
{
   // GL rendering for all box-types.

   static const TEveException eH("TEveBoxSetGL::RenderBoxes ");

   if (rnrCtx.SecSelection()) glPushName(0);

   Int_t boxSkip = 0;
   if (rnrCtx.ShapeLOD() < 50)
      boxSkip = 6 - (rnrCtx.ShapeLOD()+1)/10;

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
               RenderBox(b.fVertices);
               glEnd();
               if (fM->fAntiFlick)
                  AntiFlick(0.5f*(b.fVertices[0] + b.fVertices[18]),
                            0.5f*(b.fVertices[1] + b.fVertices[19]),
                            0.5f*(b.fVertices[2] + b.fVertices[20]));
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

      default:
      {
         throw(eH + "unsupported box-type.");
      }

   } // end switch box-type

   if (rnrCtx.SecSelection()) glPopName();
}

//______________________________________________________________________________
void TEveBoxSetGL::DirectDraw(TGLRnrCtx& rnrCtx) const
{
   // Actual rendering code.
   // Virtual from TGLLogicalShape.

   TEveBoxSet& mB = * fM;
   // printf("TEveBoxSetGL::DirectDraw N boxes %d\n", mB.fPlex.Size());

   if (mB.fPlex.Size() > 0)
   {
      if (! mB.fSingleColor && ! mB.fValueIsColor && mB.fPalette == 0)
      {
         mB.AssertPalette();
      }

      glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT);

      if (mB.fRenderMode == TEveDigitSet::kRM_Fill)
         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
      else if (mB.fRenderMode == TEveDigitSet::kRM_Line)
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

      if (mB.fBoxType == TEveBoxSet::kBT_Cone ||
          mB.fBoxType == TEveBoxSet::kBT_EllipticCone)
      {
         glDisable(GL_CULL_FACE);
      }

      if (mB.fDisableLigting) glDisable(GL_LIGHTING);

      RenderBoxes(rnrCtx);

      glPopAttrib();
   }

   if (mB.fFrame != 0 && ! rnrCtx.SecSelection() &&
       ! (rnrCtx.Highlight() && AlwaysSecondarySelect()))
   {
      TEveFrameBoxGL::Render(mB.fFrame);
   }
}

//______________________________________________________________________________
void TEveBoxSetGL::Render(TGLRnrCtx& rnrCtx)
{
   // Interface for direct rendering from classes that include TEveBoxSet
   // as a member.

   MakeDisplayList();
   DirectDraw(rnrCtx);
   glDeleteLists(fBoxDL, 1);
   fBoxDL = 0;
}
