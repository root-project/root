// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMath.h"

#include "TEveQuadSetGL.h"
#include "TEveFrameBoxGL.h"

#include "TGLRnrCtx.h"
#include "TGLIncludes.h"

//==============================================================================
// TEveQuadSetGL
//==============================================================================

//______________________________________________________________________________
//
// GL-renderer for TEveQuadSet class.

ClassImp(TEveQuadSetGL);

/******************************************************************************/

//______________________________________________________________________________
TEveQuadSetGL::TEveQuadSetGL() : TEveDigitSetGL(), fM(0)
{
   // Constructor.

   // fDLCache = false; // Disable DL.
   fMultiColor = kTRUE;
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveQuadSetGL::SetModel(TObject* obj, const Option_t* /*opt*/)
{
   // Set model object.

   fM = SetModelDynCast<TEveQuadSet>(obj);
   return kTRUE;
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

//______________________________________________________________________________
void TEveQuadSetGL::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // Draw quad-set with GL.

   static const TEveException eH("TEveQuadSetGL::DirectDraw ");

   // printf("QuadSetGLRenderer::DirectDraw Style %d, LOD %d\n", rnrCtx.Style(), rnrCtx.LOD());

   TEveQuadSet& mQ = * fM;

   if (mQ.fPlex.Size() > 0)
   {
      if (! mQ.fSingleColor && ! mQ.fValueIsColor && mQ.fPalette == 0)
      {
         mQ.AssertPalette();
      }

      glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT);
      glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
      glEnable(GL_COLOR_MATERIAL);
      glDisable(GL_CULL_FACE);

      if ( ! rnrCtx.IsDrawPassOutlineLine())
      {
         if (mQ.fRenderMode == TEveDigitSet::kRM_Fill)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
         else if (mQ.fRenderMode == TEveDigitSet::kRM_Line)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      }

      if (mQ.fDisableLighting)  glDisable(GL_LIGHTING);

      if (mQ.fQuadType < TEveQuadSet::kQT_Rectangle_End)    RenderQuads(rnrCtx);
      else if (mQ.fQuadType < TEveQuadSet::kQT_Line_End)    RenderLines(rnrCtx);
      else if (mQ.fQuadType < TEveQuadSet::kQT_Hexagon_End) RenderHexagons(rnrCtx);

      glPopAttrib();
   }

   if (mQ.fFrame != 0 && ! rnrCtx.SecSelection() && 
       ! (rnrCtx.Highlight() && AlwaysSecondarySelect()))
   {
      TEveFrameBoxGL::Render(mQ.fFrame);
   }
}

//______________________________________________________________________________
void TEveQuadSetGL::RenderQuads(TGLRnrCtx& rnrCtx) const
{
   // GL rendering for free-quads and rectangles.

   static const TEveException eH("TEveQuadSetGL::RenderQuads ");

   TEveQuadSet& mQ = * fM;

   GLenum primitiveType;
   if (mQ.fRenderMode != TEveDigitSet::kRM_Line)
   {
      primitiveType = GL_QUADS;
      if (mQ.fQuadType == TEveQuadSet::kQT_FreeQuad)
         glEnable(GL_NORMALIZE);
      else
         glNormal3f(0, 0, 1);
   } else {
      primitiveType = GL_LINE_LOOP;
   }

   TEveChunkManager::iterator qi(mQ.fPlex);
   if (rnrCtx.Highlight() && fHighlightSet)
      qi.fSelection = fHighlightSet;

   if (rnrCtx.SecSelection()) glPushName(0);

   switch (mQ.fQuadType)
   {

      case TEveQuadSet::kQT_FreeQuad:
      {
         Float_t e1[3], e2[3], normal[3];
         while (qi.next()) {
            TEveQuadSet::QFreeQuad_t& q = * (TEveQuadSet::QFreeQuad_t*) qi();
            if (SetupColor(q))
            {
               Float_t* p = q.fVertices;
               e1[0] = p[3] - p[0]; e1[1] = p[4] - p[1]; e1[2] = p[5] - p[2];
               e2[0] = p[6] - p[0]; e2[1] = p[7] - p[1]; e2[2] = p[8] - p[2];
               TMath::Cross(e1, e2, normal);
               if (rnrCtx.SecSelection()) glLoadName(qi.index());
               glBegin(primitiveType);
               glNormal3fv(normal);
               glVertex3fv(p);
               glVertex3fv(p + 3);
               glVertex3fv(p + 6);
               glVertex3fv(p + 9);
               glEnd();
               if (mQ.fAntiFlick)
                  AntiFlick(0.5f*(p[0]+p[6]), 0.5f*(p[1]+p[7]), 0.5f*(p[2]+p[8]));
            }
         }
         break;
      }

      case TEveQuadSet::kQT_RectangleXY:
      {
         while (qi.next()) {
            TEveQuadSet::QRect_t& q = * (TEveQuadSet::QRect_t*) qi();
            if (SetupColor(q))
            {
               if (rnrCtx.SecSelection()) glLoadName(qi.index());
               glBegin(primitiveType);
               glVertex3f(q.fA,        q.fB,        q.fC);
               glVertex3f(q.fA + q.fW, q.fB,        q.fC);
               glVertex3f(q.fA + q.fW, q.fB + q.fH, q.fC);
               glVertex3f(q.fA,        q.fB + q.fH, q.fC);
               glEnd();
               if (mQ.fAntiFlick)
                  AntiFlick(q.fA + 0.5f*q.fW, q.fB + 0.5f*q.fH, q.fC);
            }
         }
         break;
      }

      case TEveQuadSet::kQT_RectangleXZ:
      {
         while (qi.next()) {
            TEveQuadSet::QRect_t& q = * (TEveQuadSet::QRect_t*) qi();
            if (SetupColor(q))
            {
               if (rnrCtx.SecSelection()) glLoadName(qi.index());
               glBegin(primitiveType);
               glVertex3f(q.fA,        q.fC, q.fB);
               glVertex3f(q.fA + q.fW, q.fC, q.fB);
               glVertex3f(q.fA + q.fW, q.fC, q.fB + q.fH);
               glVertex3f(q.fA,        q.fC, q.fB + q.fH);
               glEnd();
               if (mQ.fAntiFlick)
                  AntiFlick(q.fA + 0.5f*q.fW, q.fC, q.fB + 0.5f*q.fH);
            }
         }
         break;
      }

      case TEveQuadSet::kQT_RectangleYZ:
      {
         while (qi.next()) {
            TEveQuadSet::QRect_t& q = * (TEveQuadSet::QRect_t*) qi();
            if (SetupColor(q))
            {
               if (rnrCtx.SecSelection()) glLoadName(qi.index());
               glBegin(primitiveType);
               glVertex3f(q.fC, q.fA,        q.fB);
               glVertex3f(q.fC, q.fA + q.fW, q.fB);
               glVertex3f(q.fC, q.fA + q.fW, q.fB + q.fH);
               glVertex3f(q.fC, q.fA,        q.fB + q.fH);
               glEnd();
               if (mQ.fAntiFlick)
                  AntiFlick(q.fC, q.fA + 0.5f*q.fW, q.fB + 0.5f*q.fH);
            }
         }
         break;
      }

      case TEveQuadSet::kQT_RectangleXYFixedDim:
      {
         const Float_t& w = mQ.fDefWidth;
         const Float_t& h = mQ.fDefHeight;
         while (qi.next()) {
            TEveQuadSet::QRectFixDim_t& q = * (TEveQuadSet::QRectFixDim_t*) qi();
            if (SetupColor(q))
            {
               if (rnrCtx.SecSelection()) glLoadName(qi.index());
               glBegin(primitiveType);
               glVertex3f(q.fA,     q.fB,     q.fC);
               glVertex3f(q.fA + w, q.fB,     q.fC);
               glVertex3f(q.fA + w, q.fB + h, q.fC);
               glVertex3f(q.fA,     q.fB + h, q.fC);
               glEnd();
               glEnd();
               if (mQ.fAntiFlick)
                  AntiFlick(q.fA + 0.5f*w, q.fB + 0.5f*h, q.fC);
            }
         }
         break;
      }

      case TEveQuadSet::kQT_RectangleXYFixedZ:
      {
         const Float_t& z = mQ.fDefCoord;
         while (qi.next()) {
            TEveQuadSet::QRectFixC_t& q = * (TEveQuadSet::QRectFixC_t*) qi();
            if (SetupColor(q))
            {
               if (rnrCtx.SecSelection()) glLoadName(qi.index());
               glBegin(primitiveType);
               glVertex3f(q.fA,        q.fB,        z);
               glVertex3f(q.fA + q.fW, q.fB,        z);
               glVertex3f(q.fA + q.fW, q.fB + q.fH, z);
               glVertex3f(q.fA,        q.fB + q.fH, z);
               glEnd();
               if (mQ.fAntiFlick)
                  AntiFlick(q.fA + 0.5f*q.fW, q.fB + 0.5f*q.fH, z);
            }
         }
         break;
      }

      case TEveQuadSet::kQT_RectangleXZFixedY:
      {
         const Float_t& y = mQ.fDefCoord;
         while (qi.next()) {
            TEveQuadSet::QRectFixC_t& q = * (TEveQuadSet::QRectFixC_t*) qi();
            if (SetupColor(q))
            {
               if (rnrCtx.SecSelection()) glLoadName(qi.index());
               glBegin(primitiveType);
               glVertex3f(q.fA,        y, q.fB);
               glVertex3f(q.fA + q.fW, y, q.fB);
               glVertex3f(q.fA + q.fW, y, q.fB + q.fH);
               glVertex3f(q.fA,        y, q.fB + q.fH);
               glEnd();
               if (mQ.fAntiFlick)
                  AntiFlick(q.fA + 0.5f*q.fW, y, q.fB + 0.5f*q.fH);
            }
         }
         break;
      }

      case TEveQuadSet::kQT_RectangleYZFixedX:
      {
         const Float_t& x = mQ.fDefCoord;
         while (qi.next()) {
            TEveQuadSet::QRectFixC_t& q = * (TEveQuadSet::QRectFixC_t*) qi();
            if (SetupColor(q))
            {
               if (rnrCtx.SecSelection()) glLoadName(qi.index());
               glBegin(primitiveType);
               glVertex3f(x, q.fA,        q.fB);
               glVertex3f(x, q.fA + q.fW, q.fB);
               glVertex3f(x, q.fA + q.fW, q.fB + q.fH);
               glVertex3f(x, q.fA,        q.fB + q.fH);
               glEnd();
               if (mQ.fAntiFlick)
                  AntiFlick(x, q.fA + 0.5f*q.fW, q.fB + 0.5f*q.fH);
            }
         }
         break;
      }

      case TEveQuadSet::kQT_RectangleXYFixedDimZ:
      {
         const Float_t& z = mQ.fDefCoord;
         const Float_t& w = mQ.fDefWidth;
         const Float_t& h = mQ.fDefHeight;
         while (qi.next()) {
            TEveQuadSet::QRectFixDimC_t& q = * (TEveQuadSet::QRectFixDimC_t*) qi();
            if (SetupColor(q))
            {
               if (rnrCtx.SecSelection()) glLoadName(qi.index());
               glBegin(primitiveType);
               glVertex3f(q.fA,     q.fB,     z);
               glVertex3f(q.fA + w, q.fB,     z);
               glVertex3f(q.fA + w, q.fB + h, z);
               glVertex3f(q.fA,     q.fB + h, z);
               glEnd();
               if (mQ.fAntiFlick)
                  AntiFlick(q.fA + 0.5f*w, q.fB + 0.5f*h, z);
            }
         }
         break;
      }

      case TEveQuadSet::kQT_RectangleXZFixedDimY:
      {
         const Float_t& y = mQ.fDefCoord;
         const Float_t& w = mQ.fDefWidth;
         const Float_t& h = mQ.fDefHeight;
         while (qi.next()) {
            TEveQuadSet::QRectFixDimC_t& q = * (TEveQuadSet::QRectFixDimC_t*) qi();
            if (SetupColor(q))
            {
               if (rnrCtx.SecSelection()) glLoadName(qi.index());
               glBegin(primitiveType);
               glVertex3f(q.fA,     y, q.fB);
               glVertex3f(q.fA + w, y, q.fB);
               glVertex3f(q.fA + w, y, q.fB + h);
               glVertex3f(q.fA,     y, q.fB + h);
               glEnd();
               if (mQ.fAntiFlick)
                  AntiFlick(q.fA + 0.5f*w, y, q.fB + 0.5f*h);
            }
         }
         break;
      }

      case TEveQuadSet::kQT_RectangleYZFixedDimX:
      {
         const Float_t& x = mQ.fDefCoord;
         const Float_t& w = mQ.fDefWidth;
         const Float_t& h = mQ.fDefHeight;
         while (qi.next()) {
            TEveQuadSet::QRectFixDimC_t& q = * (TEveQuadSet::QRectFixDimC_t*) qi();
            if (SetupColor(q))
            {
               if (rnrCtx.SecSelection()) glLoadName(qi.index());
               glBegin(primitiveType);
               glVertex3f(x, q.fA,     q.fB);
               glVertex3f(x, q.fA + w, q.fB);
               glVertex3f(x, q.fA + w, q.fB + h);
               glVertex3f(x, q.fA,     q.fB + h);
               glEnd();
               if (mQ.fAntiFlick)
                  AntiFlick(x, q.fA + 0.5f*w, q.fB + 0.5f*h);
            }
         }
         break;
      }

      default:
         throw(eH + "unsupported quad-type.");

   } // end switch quad-type

   if (rnrCtx.SecSelection()) glPopName();
}

//______________________________________________________________________________
void TEveQuadSetGL::RenderLines(TGLRnrCtx & rnrCtx) const
{
   // GL rendering for line-types.

   static const TEveException eH("TEveQuadSetGL::RenderLines ");

   TEveQuadSet& mQ = * fM;

   TEveChunkManager::iterator qi(mQ.fPlex);
   if (rnrCtx.Highlight() && fHighlightSet)
      qi.fSelection = fHighlightSet;

   if (rnrCtx.SecSelection()) glPushName(0);

   switch (mQ.fQuadType)
   {

      case TEveQuadSet::kQT_LineXYFixedZ:
      {
         const Float_t& z = mQ.fDefCoord;
         while (qi.next()) {
            TEveQuadSet::QLineFixC_t& q = * (TEveQuadSet::QLineFixC_t*) qi();
            if (SetupColor(q))
            {
               if (rnrCtx.SecSelection()) glLoadName(qi.index());
               glBegin(GL_LINES);
               glVertex3f(q.fA,         q.fB,         z);
               glVertex3f(q.fA + q.fDx, q.fB + q.fDy, z);
               glEnd();
            }
         }
         break;
      }

      case TEveQuadSet::kQT_LineXZFixedY:
      {
         const Float_t& z = mQ.fDefCoord;
         while (qi.next()) {
            TEveQuadSet::QLineFixC_t& q = * (TEveQuadSet::QLineFixC_t*) qi();
            if (SetupColor(q))
            {
               if (rnrCtx.SecSelection()) glLoadName(qi.index());
               glBegin(GL_LINES);
               glVertex3f(q.fA,         z, q.fB);
               glVertex3f(q.fA + q.fDx, z, q.fB + q.fDy);
               glEnd();
            }
         }
         break;
      }

      default:
         throw(eH + "unsupported quad-type.");

   }

   if (rnrCtx.SecSelection()) glPopName();
}

//______________________________________________________________________________
void TEveQuadSetGL::RenderHexagons(TGLRnrCtx & rnrCtx) const
{
   // GL rendering for hexagons.

   static const TEveException eH("TEveQuadSetGL::RenderHexagons ");

   const Float_t sqr3hf = 0.5*TMath::Sqrt(3);

   TEveQuadSet& mQ = * fM;

   GLenum primitveType = (mQ.fRenderMode != TEveDigitSet::kRM_Line) ?
      GL_POLYGON : GL_LINE_LOOP;

   glNormal3f(0, 0, 1);

   TEveChunkManager::iterator qi(mQ.fPlex);
   if (rnrCtx.Highlight() && fHighlightSet)
      qi.fSelection = fHighlightSet;

   if (rnrCtx.SecSelection()) glPushName(0);

   switch (mQ.fQuadType)
   {

      case TEveQuadSet::kQT_HexagonXY:
      {
         while (qi.next()) {
            TEveQuadSet::QHex_t& q = * (TEveQuadSet::QHex_t*) qi();
            if (SetupColor(q))
            {
               const Float_t rh = q.fR * 0.5;
               const Float_t rs = q.fR * sqr3hf;
               if (rnrCtx.SecSelection()) glLoadName(qi.index());
               glBegin(primitveType);
               glVertex3f( q.fR + q.fA,       q.fB, q.fC);
               glVertex3f(   rh + q.fA,  rs + q.fB, q.fC);
               glVertex3f(  -rh + q.fA,  rs + q.fB, q.fC);
               glVertex3f(-q.fR + q.fA,       q.fB, q.fC);
               glVertex3f(  -rh + q.fA, -rs + q.fB, q.fC);
               glVertex3f(   rh + q.fA, -rs + q.fB, q.fC);
               glEnd();
               if (mQ.fAntiFlick)
                  AntiFlick(q.fA, q.fB, q.fC);
            }
         }
         break;
      }

      case TEveQuadSet::kQT_HexagonYX:
      {
         while (qi.next()) {
            TEveQuadSet::QHex_t& q = * (TEveQuadSet::QHex_t*) qi();
            if (SetupColor(q))
            {
               const Float_t rh = q.fR * 0.5;
               const Float_t rs = q.fR * sqr3hf;
               if (rnrCtx.SecSelection()) glLoadName(qi.index());
               glBegin(primitveType);
               glVertex3f( rs + q.fA,    rh + q.fB, q.fC);
               glVertex3f(      q.fA,  q.fR + q.fB, q.fC);
               glVertex3f(-rs + q.fA,    rh + q.fB, q.fC);
               glVertex3f(-rs + q.fA,   -rh + q.fB, q.fC);
               glVertex3f(      q.fA, -q.fR + q.fB, q.fC);
               glVertex3f( rs + q.fA,   -rh + q.fB, q.fC);
               glEnd();
               if (mQ.fAntiFlick)
                  AntiFlick(q.fA, q.fB, q.fC);
            }
         }
         break;
      }

      default:
         throw(eH + "unsupported quad-type.");

   } // end switch quad-type

   if (rnrCtx.SecSelection()) glPopName();
}
