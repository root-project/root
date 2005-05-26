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

#include <vector>
#include <Riostream.h>

ClassImp(TGLScene)

//______________________________________________________________________________
TGLScene::TGLScene() :
   fBoundingBoxValid(kFALSE),
   fCanCullLowLOD(kFALSE),
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
Bool_t TGLScene::DestroyLogical(UInt_t ID)
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
TGLLogicalShape * TGLScene::FindLogical(UInt_t ID) const
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
Bool_t TGLScene::DestroyPhysical(UInt_t ID)
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
TGLPhysicalShape * TGLScene::FindPhysical(UInt_t ID) const
{
   PhysicalShapeMapCIt_t it = fPhysicalShapes.find(ID);
   if (it != fPhysicalShapes.end()) {
      return it->second;
   } else {
      return 0;
   }
}

//______________________________________________________________________________
void TGLScene::Draw(const TGLCamera & camera, UInt_t sceneLOD, Double_t timeout) const
{
   Bool_t  run = kTRUE;

   TGLStopwatch stopwatch;
   stopwatch.Start();

   // If the scene bounding box is inside the camera frustum then 
   // no need to check individual shapes - everything is visible
   Bool_t doFrustumCheck = doFrustumCheck = camera.FrustumOverlap(BoundingBox()) != kInside;

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
         // if previously we failed to complete in time
         if (sceneLOD < kHigh && fCanCullLowLOD && shapeLOD < 10) {
               ++physicalShapeIt;
               continue;
         }
         physicalShape->Draw(shapeLOD);

      }
      ++physicalShapeIt;

      // Terminate the draw is over timeout
      // TODO: Really need front/back sorting before this can 
      // be useful
      if (timeout > 0.0 && stopwatch.Lap() > timeout) {
         run = kFALSE;
      }
   }
   
   // For some reason this gets obscurred if done in TGLPhysicalShape::Draw
   if (fSelectedPhysical) {
      glDisable(GL_DEPTH_TEST);
      fSelectedPhysical->BoundingBox().Draw();
      glEnable(GL_DEPTH_TEST);
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
   Draw(camera,kHigh);
    
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
