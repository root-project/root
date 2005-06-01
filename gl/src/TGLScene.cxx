// @(#)root/gl:$Name:  $:$Id: TGLScene.cxx,v 1.6 2005/06/01 14:07:14 brun Exp $
// Author:  Richard Maunder  25/05/2005
// Parts taken from original TGLRender by Timur Pocheptsov

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// TODO: Function descriptions
// TODO: Class def - same as header

#include "TGLScene.h"
#include "TGLCamera.h"
#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TGLStopwatch.h"
#include "TGLIncludes.h"
#include "TError.h"
#include "TString.h"

#include <vector>
#include <Riostream.h>

ClassImp(TGLScene)

//______________________________________________________________________________
TGLScene::TGLScene() :
   fBoundingBox(), fBoundingBoxValid(kFALSE), 
   fLastDrawLOD(kHigh), fCanCullLowLOD(kFALSE),
   fSelectedPhysical(0)
{
}

//______________________________________________________________________________
TGLScene::~TGLScene()
{
   DestroyAllPhysicals();
   DestroyAllLogicals();
}

//TODO: Inline
//______________________________________________________________________________
void TGLScene::AdoptLogical(TGLLogicalShape & shape)
{
   assert(!fLogicalShapes.count(shape.ID()));
   fLogicalShapes.insert(LogicalShapeMapValueType_t(shape.ID(), &shape));
}

//______________________________________________________________________________
Bool_t TGLScene::DestroyLogical(ULong_t ID)
{
   TGLLogicalShape * logical = FindLogical(ID);
   if (logical) {
      if (logical->Ref() == 0) {
         delete logical;
         return true;
      }
   }

   return kFALSE;
}

//______________________________________________________________________________
UInt_t TGLScene::DestroyAllLogicals()
{
   UInt_t count = 0;
   LogicalShapeMapIt_t logicalShapeIt = fLogicalShapes.begin();
   const TGLLogicalShape * logicalShape;
   while (logicalShapeIt != fLogicalShapes.end()) {
      logicalShape = logicalShapeIt->second;
      if (logicalShape) {
         if (logicalShape->Ref() == 0) {
            fLogicalShapes.erase(logicalShapeIt++);
            delete logicalShape;
            ++count;
            continue;
         }
      } else {
         assert(kFALSE);
      }
      ++logicalShapeIt;
   }

   if (count > 0) {
      fBoundingBoxValid = kFALSE;
   }

   return count;
}

//TODO: Inline
//______________________________________________________________________________
TGLLogicalShape * TGLScene::FindLogical(ULong_t ID) const
{
   LogicalShapeMapCIt_t it = fLogicalShapes.find(ID);
   if (it != fLogicalShapes.end()) {
      return it->second;
   } else {
      return 0;
   }
}

//TODO: Inline
//______________________________________________________________________________
void TGLScene::AdoptPhysical(TGLPhysicalShape & shape)
{
   assert(!fPhysicalShapes.count(shape.ID()));
   fPhysicalShapes.insert(PhysicalShapeMapValueType_t(shape.ID(), &shape));
   fBoundingBoxValid = kFALSE;
}

//______________________________________________________________________________
Bool_t TGLScene::DestroyPhysical(ULong_t ID)
{
   TGLPhysicalShape * physical = FindPhysical(ID);
   if (physical) {
      if (fSelectedPhysical == physical) {
         fSelectedPhysical = 0;
      }

      delete physical;
      fBoundingBoxValid = kFALSE;
      fCanCullLowLOD = kFALSE;
      return true;
   }

   return kFALSE;
}

//______________________________________________________________________________
UInt_t TGLScene::DestroyPhysicals(const TGLCamera & camera)
{
   UInt_t count = 0;
   PhysicalShapeMapIt_t physicalShapeIt = fPhysicalShapes.begin();
   const TGLPhysicalShape * physical;
   while (physicalShapeIt != fPhysicalShapes.end()) {
      physical = physicalShapeIt->second;
      if (physical) {
         // Destroy any physical shape no longer of interest to camera
         if (!camera.OfInterest(physical->BoundingBox())) {
            fPhysicalShapes.erase(physicalShapeIt++);
            delete physical;
            if (fSelectedPhysical == physical) {
               fSelectedPhysical = 0;
            }
            ++count;
            continue;
         }
      } else {
         assert(kFALSE);
      }
      ++physicalShapeIt;
   }

   if (count > 0) {
      fBoundingBoxValid = kFALSE;
      fCanCullLowLOD = kFALSE;
   }

   return count;
}

//______________________________________________________________________________
UInt_t TGLScene::DestroyAllPhysicals()
{
   UInt_t count = 0;
   PhysicalShapeMapIt_t physicalShapeIt = fPhysicalShapes.begin();
   const TGLPhysicalShape * physical;
   while (physicalShapeIt != fPhysicalShapes.end()) {
      physical = physicalShapeIt->second;
      if (physical) {
         fPhysicalShapes.erase(physicalShapeIt++);
         delete physical;
         ++count;
         continue;
      } else {
         assert(kFALSE);
      }
      ++physicalShapeIt;
   }

   if (fSelectedPhysical) {
      fSelectedPhysical = 0;
   }
   if (count > 0) {
      fBoundingBoxValid = kFALSE;
      fCanCullLowLOD = kFALSE;
   }


   return count;
}

//TODO: Inline
//______________________________________________________________________________
TGLPhysicalShape * TGLScene::FindPhysical(ULong_t ID) const
{
   PhysicalShapeMapCIt_t it = fPhysicalShapes.find(ID);
   if (it != fPhysicalShapes.end()) {
      return it->second;
   } else {
      return 0;
   }
}

//______________________________________________________________________________
void TGLScene::SetColorByLogical(ULong_t logicalID, const Float_t rgba[4])
{
   PhysicalShapeMapIt_t physicalShapeIt = fPhysicalShapes.begin();
   TGLPhysicalShape * physical;
   while (physicalShapeIt != fPhysicalShapes.end()) {
      physical = physicalShapeIt->second;
      if (physical) {
         if (physical->GetLogical().ID() == logicalID) {
            physical->SetColor(rgba);
         }
      } else {
         assert(kFALSE);
      }
      ++physicalShapeIt;
   }
}

//______________________________________________________________________________
//TODO: Merge axes flag and LOD into general draw flag
void TGLScene::Draw(const TGLCamera & camera, UInt_t sceneLOD, Double_t timeout) const
{
   Bool_t  run = kTRUE;
   void (TGLPhysicalShape::*drawPtr)(UInt_t)const = &TGLPhysicalShape::Draw;

   if (fDrawMode)
      fDrawMode == kWireFrame ? drawPtr = &TGLPhysicalShape::DrawWireFrame :
                                drawPtr = &TGLPhysicalShape::DrawOutline;

   TGLStopwatch stopwatch;
   stopwatch.Start();

   // If the scene bounding box is inside the camera frustum then
   // no need to check individual shapes - everything is visible
   Bool_t doFrustumCheck = (camera.FrustumOverlap(BoundingBox()) != kInside);

   // Loop through all placed shapes in scene
   PhysicalShapeMapCIt_t physicalShapeIt = fPhysicalShapes.begin();
   const TGLPhysicalShape * physicalShape;

   while (physicalShapeIt != fPhysicalShapes.end() && run)
   {
      physicalShape = physicalShapeIt->second;
      if (!physicalShape)
      {
         assert(kFALSE);
         continue;
      }

      EOverlap frustumOverlap = kInside;
      if (doFrustumCheck)
      {
         frustumOverlap = camera.FrustumOverlap(physicalShape->BoundingBox());
      }

      if (frustumOverlap == kInside || frustumOverlap == kPartial)
      {
         // Get the shape draw quality
         UInt_t shapeLOD = CalcPhysicalLOD(*physicalShape, camera, sceneLOD);

         // Skip drawing low (i.e. small projected) LOD shapes on non-100% passes
         // if previously we failed to complete in time.
         if (sceneLOD < kHigh && fCanCullLowLOD && shapeLOD < 10) {
               ++physicalShapeIt;
               continue;
         }
         
         //Draw, DrawWireFrame, DrawOutline
         (physicalShape->*drawPtr)(shapeLOD);
      }
      ++physicalShapeIt;

      // Terminate the draw is over timeout
      // TODO: Really need front/back sorting before this can
      // be useful
      if (timeout > 0.0 && stopwatch.Lap() > timeout) {
         run = kFALSE;
      }
   }

   if (fSelectedPhysical) {
      // Ensure the selected object is always draw regardless of above terminations
      // This could result in it being drawn twice - but is (probably) cheaper than testing 
      // in loop and recording if it was done
      // TODO: Sort selected to front of draw list
      UInt_t shapeLOD = CalcPhysicalLOD(*fSelectedPhysical, camera, sceneLOD);

      (fSelectedPhysical->*drawPtr)(shapeLOD);
      
      // Draw selected object bounding box - for some reason this gets obscurred if done 
      // in TGLPhysicalShape::Draw
      if (fDrawMode == kFill || fDrawMode == kOutline) {
         glDisable(GL_LIGHTING);
      }

      //BBOX is white for wireframe mode and fill,
      //red for outlines
      switch (fDrawMode) {
      case kFill:
      case kWireFrame :
         glColor3d(1., 1., 1.);

         break;
      case kOutline:
         glColor3d(1., 0., 0.);
         
         break;
      }

      glDisable(GL_DEPTH_TEST);
      fSelectedPhysical->BoundingBox().Draw();
      glEnable(GL_DEPTH_TEST);

      if (fDrawMode == kFill || fDrawMode == kOutline) {
         glEnable(GL_LIGHTING);
      }
   }

   // Failed to complete in time? Record flag to cull low LODs next time
   if (timeout > 0.0 && stopwatch.End() > timeout) {
      fCanCullLowLOD = kTRUE;
   }

   // Record this so that any Select() draw can be redone at same quality and ensure
   // accuracy of picking
   // TODO: Also record timeout?
   fLastDrawLOD = sceneLOD;
}

//______________________________________________________________________________
void TGLScene::DrawAxes() const
{
   // Draw out the scene axes
   // Taken directly from TGLRender by Timur Pocheptsov
   static const UChar_t xyz[][8] = {{0x44, 0x44, 0x28, 0x10, 0x10, 0x28, 0x44, 0x44},
                                     {0x10, 0x10, 0x10, 0x10, 0x10, 0x28, 0x44, 0x44},
                                     {0x7c, 0x20, 0x10, 0x10, 0x08, 0x08, 0x04, 0x7c}};

   /*if (!fPxs) {
      fPxs = kTRUE;
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   }*/
   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glPushAttrib(GL_DEPTH_BUFFER_BIT);
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_LIGHTING);

   const Double_t axeColors[][3] = {{1., 0., 0.}, // X axis red
                                    {0., 1., 0.}, // Y axis green
                                    {0., 0., 1.}};// Z axis blue
   //glColor3dv(axeColors[3]);

   const TGLBoundingBox & box = BoundingBox();
   Double_t xmin = box.XMin(), xmax = box.XMax();
   Double_t ymin = box.YMin(), ymax = box.YMax();
   Double_t zmin = box.ZMin(), zmax = box.ZMax();

   glBegin(GL_LINES);
   glColor3dv(axeColors[0]);
   glVertex3d(xmin, ymin, zmin);
   glVertex3d(xmax, ymin, zmin);
   glColor3dv(axeColors[1]);
   glVertex3d(xmin, ymin, zmin);
   glVertex3d(xmin, ymax, zmin);
   glColor3dv(axeColors[2]);
   glVertex3d(xmin, ymin, zmin);
   glVertex3d(xmin, ymin, zmax);
   glEnd();

   // X label
   glColor3dv(axeColors[0]);
   glRasterPos3d(xmax, ymin + 12, zmin);
   glBitmap(8, 8, 0., 0., 0., 0., xyz[0]);
   DrawNumber(xmax, xmax, ymin, zmin, 9.);
   DrawNumber(xmin, xmin, ymin, zmin, 0.);

   // Y label
   glColor3dv(axeColors[1]);
   glRasterPos3d(xmin, ymax + 12, zmin);
   glBitmap(8, 8, 0, 0, 12., 0, xyz[1]);
   DrawNumber(ymax, xmin, ymax, zmin, 9.);
   DrawNumber(ymin, xmin, ymin, zmin, 9.);

   // Z label
   glColor3dv(axeColors[2]);
   glRasterPos3d(xmin, ymin, zmax);
   glBitmap(8, 8, 0, 0, 0., 0, xyz[2]);
   DrawNumber(zmax, xmin, ymin, zmax, 9.);
   DrawNumber(zmin, xmin, ymin, zmin, -9.);

   glEnable(GL_LIGHTING);
   glEnable(GL_DEPTH_TEST);
   glPopAttrib();
}

//______________________________________________________________________________
void TGLScene::DrawNumber(Double_t num, Double_t x, Double_t y, Double_t z, Double_t yorig) const
{
   static const UChar_t
      digits[][8] = {{0x38, 0x44, 0x44, 0x44, 0x44, 0x44, 0x44, 0x38},//0
                     {0x10, 0x10, 0x10, 0x10, 0x10, 0x70, 0x10, 0x10},//1
                     {0x7c, 0x44, 0x20, 0x18, 0x04, 0x04, 0x44, 0x38},//2
                     {0x38, 0x44, 0x04, 0x04, 0x18, 0x04, 0x44, 0x38},//3
                     {0x04, 0x04, 0x04, 0x04, 0x7c, 0x44, 0x44, 0x44},//4
                     {0x7c, 0x44, 0x04, 0x04, 0x7c, 0x40, 0x40, 0x7c},//5
                     {0x7c, 0x44, 0x44, 0x44, 0x7c, 0x40, 0x40, 0x7c},//6
                     {0x20, 0x20, 0x20, 0x10, 0x08, 0x04, 0x44, 0x7c},//7
                     {0x38, 0x44, 0x44, 0x44, 0x38, 0x44, 0x44, 0x38},//8
                     {0x7c, 0x44, 0x04, 0x04, 0x7c, 0x44, 0x44, 0x7c},//9
                     {0x18, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},//.
                     {0x00, 0x00, 0x00, 0x00, 0x7c, 0x00, 0x00, 0x00}};//-

   TString str;
   str+=Long_t(num);

   glRasterPos3i(Int_t(x), Int_t(y), Int_t(z));
   for (Ssiz_t i = 0, e = str.Length(); i < e; ++i) {
      if (str[i] == '.') {
         glBitmap(8, 8, 0., yorig, 7., 0., digits[10]);
         if (i + 1 < e)
            glBitmap(8, 8, 0., yorig, 7., 0., digits[str[i + 1] - '0']);
         break;
      } else if (str[i] == '-') {
         glBitmap(8, 8, 0., yorig, 7., 0., digits[11]);
      } else {
         glBitmap(8, 8, 0., yorig, 7., 0., digits[str[i] - '0']);
      }
   }
}

//______________________________________________________________________________
UInt_t TGLScene:: CalcPhysicalLOD(const TGLPhysicalShape & shape, const TGLCamera & camera,
                                 UInt_t sceneLOD) const
{
   // Find diagonal pixel size of projected drawable BB, using camera
   Double_t diagonal = static_cast<Double_t>(camera.ViewportSize(shape.BoundingBox()).Diagonal());

   // TODO: Get real screen size - assuming 2000 pixel screen at present
   // Calculate a non-linear sizing hint for this shape. Needs more experimenting with...
   UInt_t sizeLOD = static_cast<UInt_t>(pow(diagonal,0.4) * 100.0 / pow(2000.0,0.4));

   // Factor in scene quality
   UInt_t shapeLOD = (sceneLOD * sizeLOD) / 100;

   if (shapeLOD > 10) {
      Double_t quant = ((static_cast<Double_t>(shapeLOD)) + 0.3) / 10;
      shapeLOD = static_cast<UInt_t>(quant)*10;
   } else {
      Double_t quant = ((static_cast<Double_t>(shapeLOD)) + 0.3) / 3;
      shapeLOD = static_cast<UInt_t>(quant)*3;
   }

   if (shapeLOD > 100) {
      shapeLOD = 100;
   }

   return shapeLOD;
}

//______________________________________________________________________________
Bool_t TGLScene::Select(const TGLCamera & camera)
{
   Bool_t redrawReq = kFALSE;

   // Create the select buffer. This will work as we have a flat set of physical shapes.
   // We only ever load a single name in TGLPhysicalShape::DirectDraw so any hit record always
   // has same 4 GLuint format
   static std::vector<GLuint> selectBuffer(fPhysicalShapes.size()*4);
   glSelectBuffer(selectBuffer.size(), &selectBuffer[0]);

   // Enter picking mode
   glRenderMode(GL_SELECT);
   glInitNames();
   glPushName(0);

   // Draw out scene at last visible quality
   Draw(camera, kHigh, kFALSE); // No axes

   // Retrieve the hit count and return to render
   GLint hits = glRenderMode(GL_RENDER);

   if (hits < 0) {
      Error("TGLScene::Select", "selection buffer overflow");
   } else if (hits > 0) {
      // Every hit record has format (GLuint per item) - see above for selectBuffer
      // for reason. Format is:
      //
      // no of names in name block (1 always)
      // minDepth
      // maxDepth
      // name(s) (1 always)
      assert(selectBuffer[0] == 1);
      UInt_t minDepth = selectBuffer[1];
      UInt_t minDepthName = selectBuffer[3];

      // Find the nearest picked object
      // TODO: Put back transparency picking stuff
      for (Int_t i = 1; i < hits; ++i) {
         assert(selectBuffer[i*4] == 1); // Single name per record
         if (selectBuffer[i*4 + 1] < minDepth) {
            minDepth = selectBuffer[i*4 + 1];
            minDepthName = selectBuffer[i*4 + 3];
         }
      }

      TGLPhysicalShape * selected = FindPhysical(minDepthName);
      if (!selected) {
         assert(kFALSE);
         return kFALSE;
      }

      // Swap any selection
      if (selected != fSelectedPhysical) {
         if (fSelectedPhysical) {
            fSelectedPhysical->Select(kFALSE);
         }
         fSelectedPhysical = selected;
         fSelectedPhysical->Select(true);
         redrawReq = true;
      }
   } else { // 0 hits
      if (fSelectedPhysical) {
         fSelectedPhysical->Select(kFALSE);
         fSelectedPhysical = 0;
         redrawReq = true;
      }
   }

   return redrawReq;
}

//______________________________________________________________________________
void TGLScene::SelectedModified() 
{
   // The selected object was modified external - our bounding box
   // is potentially invalid
   if (fSelectedPhysical && !BoundingBox().AlignedContains(fSelectedPhysical->BoundingBox())) {
      fBoundingBoxValid = kFALSE;
   }
}

//______________________________________________________________________________
const TGLBoundingBox & TGLScene::BoundingBox() const
{
   if (!fBoundingBoxValid) {
      Double_t xMin, xMax, yMin, yMax, zMin, zMax;
      xMin = xMax = yMin = yMax = zMin = zMax = 0.0;
      PhysicalShapeMapCIt_t physicalShapeIt = fPhysicalShapes.begin();
      const TGLPhysicalShape * physicalShape;
      while (physicalShapeIt != fPhysicalShapes.end())
      {
         physicalShape = physicalShapeIt->second;
         if (!physicalShape)
         {
            assert(kFALSE);
            continue;
         }
         TGLBoundingBox box = physicalShape->BoundingBox();
         if (physicalShapeIt == fPhysicalShapes.begin()) {
            xMin = box.XMin(); xMax = box.XMax();
            yMin = box.YMin(); yMax = box.YMax();
            zMin = box.ZMin(); zMax = box.ZMax();
         } else {
            if (box.XMin() < xMin) { xMin = box.XMin(); }
            if (box.XMax() > xMax) { xMax = box.XMax(); }
            if (box.YMin() < yMin) { yMin = box.YMin(); }
            if (box.YMax() > yMax) { yMax = box.YMax(); }
            if (box.ZMin() < zMin) { zMin = box.ZMin(); }
            if (box.ZMax() > zMax) { zMax = box.ZMax(); }
         }
         ++physicalShapeIt;
      }
      fBoundingBox.SetAligned(TGLVertex3(xMin,yMin,zMin), TGLVertex3(xMax,yMax,zMax));
      fBoundingBoxValid = true;
   }
   return fBoundingBox;
}

void TGLScene::Dump() const
{
   std::cout << "Scene: " << fLogicalShapes.size() << " Logicals / " << fPhysicalShapes.size() << " Physicals " << std::endl;
}

//______________________________________________________________________________
UInt_t TGLScene::SizeOf() const
{
   UInt_t size = sizeof(this);

   std::cout << "Size: Scene Only " << size << std::endl;

   LogicalShapeMapCIt_t logicalShapeIt = fLogicalShapes.begin();
   const TGLLogicalShape * logicalShape;
   while (logicalShapeIt != fLogicalShapes.end()) {
      logicalShape = logicalShapeIt->second;
      size += sizeof(*logicalShape);
      ++logicalShapeIt;
   }

   std::cout << "Size: Scene + Shapes " << size << std::endl;

   PhysicalShapeMapCIt_t physicalShapeIt = fPhysicalShapes.begin();
   const TGLPhysicalShape * physicalShape;
   while (physicalShapeIt != fPhysicalShapes.end()) {
      physicalShape = physicalShapeIt->second;
      size += sizeof(*physicalShape);
      ++physicalShapeIt;
   }

   std::cout << "Size: Scene + Shapes + Placed Shapes " << size << std::endl;

   return size;
}
