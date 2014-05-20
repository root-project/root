// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLScenePad.h"

#include "TGLViewer.h"
#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TGLObject.h"
#include "TGLStopwatch.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TPolyMarker3D.h"
#include "TColor.h"
#include "TROOT.h"
#include "TH3.h"

#include "TGLFaceSet.h"
#include "TGLPolyLine.h"
#include "TGLPolyMarker.h"
#include "TGLCylinder.h"
#include "TGLSphere.h"

#include "TVirtualPad.h"
#include "TAtt3D.h"
#include "TClass.h"
#include "TList.h"
#include "TMath.h"

#include "TGLPlot3D.h"


//______________________________________________________________________________
// TGLScenePad
//
// Implements VirtualViewer3D interface and fills the base-class
// visualization structures from pad contents.
//

ClassImp(TGLScenePad)


//______________________________________________________________________________
TGLScenePad::TGLScenePad(TVirtualPad* pad) :
   TVirtualViewer3D(),
   TGLScene(),

   fPad               (pad),
   fInternalPIDs      (kFALSE),
   fNextInternalPID   (1), // 0 reserved
   fLastPID           (0), // 0 reserved
   fAcceptedPhysicals (0),
   fComposite         (0),
   fCSLevel           (0),
   fSmartRefresh      (kFALSE)
{
   // Constructor.
}


/******************************************************************************/
// Histo import and Sub-pad traversal
/******************************************************************************/

//______________________________________________________________________________
void TGLScenePad::AddHistoPhysical(TGLLogicalShape* log, const Float_t *histoColor)
{
   // Scale and rotate a histo object to mimic placement in canvas.

   Double_t how = ((Double_t) gPad->GetWh()) / gPad->GetWw();

   Double_t lw = gPad->GetAbsWNDC();
   Double_t lh = gPad->GetAbsHNDC() * how;
   Double_t lm = TMath::Min(lw, lh);

   const TGLBoundingBox& bb = log->BoundingBox();

   // Timur always packs histos in a square: let's just take x-diff.
   Double_t size  = TMath::Sqrt(3) * (bb.XMax() - bb.XMin());
   Double_t scale = lm / size;
   TGLVector3 scaleVec(scale, scale, scale);

   Double_t tx = gPad->GetAbsXlowNDC() + lw;
   Double_t ty = gPad->GetAbsYlowNDC() * how + lh;
   TGLVector3 transVec(0, ty, tx); // For viewer convention (starts looking along -x).

   // XXXX plots no longer centered at 0. Or they never were?
   // Impossible to translate and scale them as they should be, it
   // seems. This requers further investigation, eventually.
   //
   // bb.Dump();
   // printf("lm=%f, size=%f, scale=%f, tx=%f, ty=%f\n",
   //        lm, size, scale, tx, ty);
   //
   // TGLVector3 c(bb.Center().Arr());
   // c.Negate();
   // c.Dump();
   // mat.Translate(c);

   TGLMatrix mat;
   mat.Scale(scaleVec);
   mat.Translate(transVec);
   mat.RotateLF(3, 2, TMath::PiOver2());
   mat.RotateLF(1, 3, TMath::DegToRad()*gPad->GetTheta());
   mat.RotateLF(1, 2, TMath::DegToRad()*(gPad->GetPhi() - 90));
   Float_t rgba[4] = {1.f, 1.f, 1.f, 1.f};
   if (histoColor) {
      rgba[0] = histoColor[0];
      rgba[1] = histoColor[1];
      rgba[2] = histoColor[2];
      rgba[3] = histoColor[3];
   }
   TGLPhysicalShape* phys = new TGLPhysicalShape(fNextInternalPID++, *log, mat, false, rgba);
   AdoptPhysical(*phys);

   // Part of XXXX above.
   // phys->BoundingBox().Dump();
}

namespace {

//______________________________________________________________________________
Bool_t HasPolymarkerAndFrame(const TList *lst)
{
   //TTree::Draw can create polymarker + empty TH3 (to draw as a frame around marker).
   //Unfortunately, this is not good for GL - this will be two unrelated
   //objects in two unrelated coordinate systems.
   //So, this function checks list contents, and if it founds empty TH3 and polymarker,
   //the must be combined as one object.
   //Later we'll reconsider the design.
   Bool_t gotEmptyTH3 = kFALSE;
   Bool_t gotMarker = kFALSE;

   TObjOptLink *lnk = lst ? (TObjOptLink*)lst->FirstLink() : 0;
   for (; lnk; lnk = (TObjOptLink*)lnk->Next()) {
      const TObject *obj = lnk->GetObject();
      if (const TH3 *th3 = dynamic_cast<const TH3*>(obj)) {
         if(!th3->GetEntries())
            gotEmptyTH3 = kTRUE;
      } else if (dynamic_cast<const TPolyMarker3D *>(obj))
         gotMarker = kTRUE;
   }

   return gotMarker && gotEmptyTH3;
}

}



//______________________________________________________________________________
void TGLScenePad::SubPadPaint(TVirtualPad* pad)
{
   // Iterate over pad-primitves and import them.

   TVirtualPad      *padsav  = gPad;
   TVirtualViewer3D *vv3dsav = pad->GetViewer3D();
   gPad = pad;
   pad->SetViewer3D(this);

   TList       *prims = pad->GetListOfPrimitives();

   if (HasPolymarkerAndFrame(prims)) {
      ComposePolymarker(prims);
   } else {
      TObjOptLink *lnk   = (prims) ? (TObjOptLink*)prims->FirstLink() : 0;
      for (; lnk; lnk = (TObjOptLink*)lnk->Next())
         ObjectPaint(lnk->GetObject(), lnk->GetOption());
   }

   pad->SetViewer3D(vv3dsav);
   gPad = padsav;
}


//______________________________________________________________________________
void TGLScenePad::ObjectPaint(TObject* obj, Option_t* opt)
{
   // Override of virtual TVirtualViewer3D::ObjectPaint().
   // Special handling of 2D/3D histograms to activate Timur's
   // histo-painters.

   TGLPlot3D* log = TGLPlot3D::CreatePlot(obj, opt, gPad);
   if (log)
   {
      AdoptLogical(*log);
      AddHistoPhysical(log);
   }
   else if (obj->InheritsFrom(TAtt3D::Class()))
   {
      // Handle 3D primitives here.
      obj->Paint(opt);
   }
   else if (obj->InheritsFrom(TVirtualPad::Class()))
   {
      SubPadPaint(dynamic_cast<TVirtualPad*>(obj));
   }
   else
   {
      // Handle 2D primitives here.
      obj->Paint(opt);
   }
}

//______________________________________________________________________________
void TGLScenePad::PadPaintFromViewer(TGLViewer* viewer)
{
   // Entry point for requesting update of scene's contents from
   // gl-viewer.

   Bool_t sr = fSmartRefresh;
   fSmartRefresh = viewer->GetSmartRefresh();

   PadPaint(fPad);

   fSmartRefresh = sr;
}

//______________________________________________________________________________
void TGLScenePad::PadPaint(TVirtualPad* pad)
{
   // Entry point for updating scene contents via VirtualViewer3D
   // interface.
   // For now this is handled by TGLViewer as it remains
   // the 'Viewer3D' of given pad.

   if (pad != fPad)
   {
      Error("TGLScenePad::PadPaint", "Mismatch between pad argument and data-member!");
      return;
   }

   BeginScene();
   SubPadPaint(fPad);
   EndScene();
}


//==============================================================================
// VV3D
//==============================================================================

//______________________________________________________________________________
void TGLScenePad::BeginScene()
{
   // Start building of the scene.
   // Old contents is dropped, unless smart-refresh is in active. Then
   // the object supporting it are kept in a cache and possibly reused.
   //
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture.

   if (gDebug>2) {
      Info("TGLScenePad::BeginScene", "entering.");
   }

   if ( ! BeginUpdate()) {
      Error("TGLScenePad::BeginScene", "could not take scene lock.");
      return;
   }

   UInt_t destroyedLogicals  = 0;
   UInt_t destroyedPhysicals = 0;

   TGLStopwatch stopwatch;
   if (gDebug > 2) {
      stopwatch.Start();
   }

   // Rebuilds can potentially invalidate all logical and
   // physical shapes.
   // Physicals must be removed first.
   destroyedPhysicals = DestroyPhysicals();
   if (fSmartRefresh) {
      destroyedLogicals = BeginSmartRefresh();
   } else {
      destroyedLogicals = DestroyLogicals();
   }

   // Potentially using external physical IDs
   fInternalPIDs = kFALSE;

   // Reset internal physical ID counter
   fNextInternalPID = 1;
   fLastPID         = 0;

   // Reset tracing info
   fAcceptedPhysicals = 0;

   if (gDebug > 2) {
      Info("TGLScenePad::BeginScene", "destroyed %d physicals %d logicals in %f msec",
            destroyedPhysicals, destroyedLogicals, stopwatch.End());
      DumpMapSizes();
   }
}

//______________________________________________________________________________
void TGLScenePad::EndScene()
{
   // End building of the scene.
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture

   if (fSmartRefresh) {
      EndSmartRefresh();
   }

   EndUpdate();

   if (gDebug > 2) {
      Info("TGLScenePad::EndScene", "Accepted %d physicals", fAcceptedPhysicals);
      DumpMapSizes();
   }
}

//______________________________________________________________________________
Int_t TGLScenePad::AddObject(const TBuffer3D& buffer, Bool_t* addChildren)
{
   // Add an object to the viewer, using internal physical IDs
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture

   // If this is called we are generating internal physical IDs
   fInternalPIDs = kTRUE;
   Int_t sections = AddObject(fNextInternalPID, buffer, addChildren);
   return sections;
}

//______________________________________________________________________________
Int_t TGLScenePad::AddObject(UInt_t physicalID, const TBuffer3D& buffer, Bool_t* addChildren)
{
   // Add an object to the scene, using an external physical ID
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture

   // TODO: Break this up and make easier to understand. This is
   // pretty convoluted due to the large number of cases it has to
   // deal with:
   // i)   exisiting physical and/or logical;
   // ii)  external provider may or may not supply bounding box;
   // iii) local/global reference frame;
   // iv)  deferred filling of some sections of the buffer;
   // v)   internal or external physical IDs;
   // vi)  composite components as special case.
   //
   // The buffer filling means the function is re-entrant which adds
   // to complication.

   if (physicalID == 0) {
      Error("TGLScenePad::AddObject", "0 physical ID reserved");
      return TBuffer3D::kNone;
   }

   // Internal and external physical IDs cannot be mixed in a scene build
   if (fInternalPIDs && physicalID != fNextInternalPID) {
      Error("TGLScenePad::AddObject", "invalid next physical ID - mix of internal + external IDs?");
      return TBuffer3D::kNone;
   }

   // We always take all children ... interest is viewer dependent.
   if (addChildren)
      *addChildren = kTRUE;

   // Scene should be modify locked
   if (CurrentLock() != kModifyLock) {
      Error("TGLScenePad::AddObject", "expected scene to be modify-locked.");
      return TBuffer3D::kNone;
   }

   // Note that 'object' here is really a physical/logical pair described
   // in buffer + physical ID.

   // If adding component to a current partial composite do this now
   if (fComposite) {
      RootCsg::TBaseMesh *newMesh = RootCsg::ConvertToMesh(buffer);
      // Solaris CC can't create stl pair with enumerate type
      fCSTokens.push_back(std::make_pair(static_cast<UInt_t>(TBuffer3D::kCSNoOp), newMesh));
      return TBuffer3D::kNone;
   }

   // TODO: Could be a data member - save possible double lookup?
   TGLPhysicalShape *physical = FindPhysical(physicalID);
   TGLLogicalShape  *logical  = 0;

   // If we have a valid (non-zero) ID, see if the logical is already cached.
   // If it is not, try to create a direct renderer object.
   if (buffer.fID)
   {
      logical = FindLogical(buffer.fID);
      if (!logical)
         logical = AttemptDirectRenderer(buffer.fID);
   }

   // First attempt to add this physical.
   if (physicalID != fLastPID)
   {
      // Existing physical.
      // MT comment: I don't think this should ever happen.
      if (physical)
      {
         // If we have physical we should have logical cached, too.
         if (!logical) {
            Error("TGLScenePad::AddObject", "cached physical with no assocaited cached logical");
         }

         // Since we already have logical no need for further checks.
         // Done ... prepare for next object.
         if (fInternalPIDs)
            ++fNextInternalPID;

         return TBuffer3D::kNone;
      }

      // Need any extra sections in buffer?
      Bool_t includeRaw    = (logical == 0);
      Int_t  extraSections = ValidateObjectBuffer(buffer, includeRaw);
      if (extraSections != TBuffer3D::kNone)
         return extraSections;

      fLastPID = physicalID;
   }

   if (fLastPID != physicalID) {
      Error("TGLScenePad::AddObject", "internal physical ID tracking error?");
   }

   // Being here means we need to add a physical, maybe logical as well.
   if (physical) {
      Error("TGLScenePad::AddObject", "expecting to require physical");
      return TBuffer3D::kNone;
   }

   // Create logical if required.
   if (!logical)
   {
      logical = CreateNewLogical(buffer);
      if (!logical) {
         Error("TGLScenePad::AddObject", "failed to create logical");
         return TBuffer3D::kNone;
      }
      // Add logical to scene
      AdoptLogical(*logical);
   }

   // Create the physical, bind it to the logical and add it to the scene.
   physical = CreateNewPhysical(physicalID, buffer, *logical);

   if (physical)
   {
      AdoptPhysical(*physical);
      buffer.fPhysicalID = physicalID;
      ++fAcceptedPhysicals;
      if (gDebug>3 && fAcceptedPhysicals%1000 == 0) {
         Info("TGLScenePad::AddObject", "added %d physicals", fAcceptedPhysicals);
      }
   }
   else
   {
      Error("TGLScenePad::AddObject", "failed to create physical");
   }

   // Done ... prepare for next object.
   if (fInternalPIDs)
      fNextInternalPID++;

   return TBuffer3D::kNone;
}

//______________________________________________________________________________
Bool_t TGLScenePad::OpenComposite(const TBuffer3D& buffer, Bool_t* addChildren)
{
   // Open new composite container.
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture.

   if (fComposite) {
      Error("TGLScenePad::OpenComposite", "composite already open");
      return kFALSE;
   }
   UInt_t extraSections = AddObject(buffer, addChildren);
   if (extraSections != TBuffer3D::kNone) {
      Error("TGLScenePad::OpenComposite", "expected top level composite to not require extra buffer sections");
   }

   // If composite was created it is of interest - we want the rest of the
   // child components
   if (fComposite) {
      return kTRUE;
   } else {
      return kFALSE;
   }
}

//______________________________________________________________________________
void TGLScenePad::CloseComposite()
{
   // Close composite container
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture

   // If we have a partially complete composite build it now
   if (fComposite) {
      // TODO: Why is this member and here - only used in BuildComposite()
      fCSLevel = 0;

      RootCsg::TBaseMesh *resultMesh = BuildComposite();
      fComposite->SetFromMesh(resultMesh);
      delete resultMesh;
      for (UInt_t i = 0; i < fCSTokens.size(); ++i) delete fCSTokens[i].second;
      fCSTokens.clear();
      fComposite = 0;
   }
}

//______________________________________________________________________________
void TGLScenePad::AddCompositeOp(UInt_t operation)
{
   // Add composite operation used to combine objects added via AddObject
   // TVirtualViewer3D interface overload - see base/src/TVirtualViewer3D.cxx
   // for description of viewer architecture

   fCSTokens.push_back(std::make_pair(operation, (RootCsg::TBaseMesh *)0));
}


// Protected methods

//______________________________________________________________________________
Int_t TGLScenePad::ValidateObjectBuffer(const TBuffer3D& buffer, Bool_t includeRaw) const
{
   // Validate if the passed 'buffer' contains all sections we require to add object.
   // Returns Int_t combination of TBuffer::ESection flags still required - or
   // TBuffer3D::kNone if buffer is valid.
   // If 'includeRaw' is kTRUE check for kRaw/kRawSizes - skip otherwise.
   // See base/src/TVirtualViewer3D.cxx for description of viewer architecture

   // kCore: Should always be filled
   if (!buffer.SectionsValid(TBuffer3D::kCore)) {
      Error("TGLScenePad::ValidateObjectBuffer", "kCore section of buffer should be filled always");
      return TBuffer3D::kNone;
   }

   // Need to check raw (kRaw/kRawSizes)?
   if (!includeRaw) {
      return TBuffer3D::kNone;
   }

   // kRawSizes / kRaw: These are on demand based on shape type
   Bool_t needRaw = kFALSE;

   // We need raw tesselation in these cases:
   //
   // 1. Shape type is NOT kSphere / kTube / kTubeSeg / kCutTube / kComposite
   if (buffer.Type() != TBuffer3DTypes::kSphere  &&
       buffer.Type() != TBuffer3DTypes::kTube    &&
       buffer.Type() != TBuffer3DTypes::kTubeSeg &&
       buffer.Type() != TBuffer3DTypes::kCutTube &&
       buffer.Type() != TBuffer3DTypes::kComposite)
   {
      needRaw = kTRUE;
   }
   // 2. Sphere type is kSPHE, but the sphere is hollow and/or cut - we
   //    do not support native drawing of these currently
   else if (buffer.Type() == TBuffer3DTypes::kSphere)
   {
      const TBuffer3DSphere * sphereBuffer = dynamic_cast<const TBuffer3DSphere *>(&buffer);
      if (sphereBuffer) {
         if (!sphereBuffer->IsSolidUncut()) {
            needRaw = kTRUE;
         }
      } else {
         Error("TGLScenePad::ValidateObjectBuffer", "failed to cast buffer of type 'kSphere' to TBuffer3DSphere");
         return TBuffer3D::kNone;
      }
   }
   // 3. kBoundingBox is not filled - we generate a bounding box from
   else if (!buffer.SectionsValid(TBuffer3D::kBoundingBox))
   {
      needRaw = kTRUE;
   }
   // 4. kShapeSpecific is not filled - except in case of top level composite
   else if (!buffer.SectionsValid(TBuffer3D::kShapeSpecific) &&
             buffer.Type() != TBuffer3DTypes::kComposite)
   {
      needRaw = kTRUE;
   }
   // 5. We are a component (not the top level) of a composite shape
   else if (fComposite)
   {
      needRaw = kTRUE;
   }

   if (needRaw && !buffer.SectionsValid(TBuffer3D::kRawSizes|TBuffer3D::kRaw)) {
      return TBuffer3D::kRawSizes|TBuffer3D::kRaw;
   } else {
      return TBuffer3D::kNone;
   }
}

//______________________________________________________________________________
TGLLogicalShape* TGLScenePad::CreateNewLogical(const TBuffer3D& buffer) const
{
   // Create and return a new TGLLogicalShape from the supplied buffer
   TGLLogicalShape * newLogical = 0;

   if (buffer.fColor == 1) // black -> light-brown; std behaviour for geom
      const_cast<TBuffer3D&>(buffer).fColor = 42;

   switch (buffer.Type())
   {
      case TBuffer3DTypes::kLine:
         newLogical = new TGLPolyLine(buffer);
         break;
      case TBuffer3DTypes::kMarker:
         newLogical = new TGLPolyMarker(buffer);
         break;
      case TBuffer3DTypes::kSphere:
      {
         const TBuffer3DSphere * sphereBuffer = dynamic_cast<const TBuffer3DSphere *>(&buffer);
         if (sphereBuffer)
         {
            // We can only draw solid uncut spheres natively at present.
            // If somebody already passed the raw buffer, they probably want us to use it.
            if (sphereBuffer->IsSolidUncut() && !buffer.SectionsValid(TBuffer3D::kRawSizes|TBuffer3D::kRaw))
            {
               newLogical = new TGLSphere(*sphereBuffer);
            } else {
               newLogical = new TGLFaceSet(buffer);
            }
         } else {
            Error("TGLScenePad::CreateNewLogical", "failed to cast buffer of type 'kSphere' to TBuffer3DSphere");
         }
         break;
      }
      case TBuffer3DTypes::kTube:
      case TBuffer3DTypes::kTubeSeg:
      case TBuffer3DTypes::kCutTube:
      {
         const TBuffer3DTube * tubeBuffer = dynamic_cast<const TBuffer3DTube *>(&buffer);
         if (tubeBuffer)
         {
            // If somebody already passed the raw buffer, they probably want us to use it.
            if (!buffer.SectionsValid(TBuffer3D::kRawSizes|TBuffer3D::kRaw)) {
               newLogical = new TGLCylinder(*tubeBuffer);
            } else {
               newLogical = new TGLFaceSet(buffer);
            }
         } else {
            Error("TGLScenePad::CreateNewLogical", "failed to cast buffer of type 'kTube/kTubeSeg/kCutTube' to TBuffer3DTube");
         }
         break;
      }
      case TBuffer3DTypes::kComposite:
      {
         // Create empty faceset and record partial complete composite object
         // Will be populated with mesh in CloseComposite()
         if (fComposite)
         {
            Error("TGLScenePad::CreateNewLogical", "composite already open");
         }
         fComposite = new TGLFaceSet(buffer);
         newLogical = fComposite;
         break;
      }
      default:
         newLogical = new TGLFaceSet(buffer);
         break;
   }

   return newLogical;
}

//______________________________________________________________________________
TGLPhysicalShape*
TGLScenePad::CreateNewPhysical(UInt_t ID, const TBuffer3D& buffer,
                               const TGLLogicalShape& logical) const
{
   // Create and return a new TGLPhysicalShape with id 'ID', using
   // 'buffer' placement information (translation etc), and bound to
   // suppled 'logical'

   // Extract indexed color from buffer
   // TODO: Still required? Better use proper color triplet in buffer?
   Int_t colorIndex = buffer.fColor;
   if (colorIndex < 0) colorIndex = 42;
   Float_t rgba[4];
   TGLScene::RGBAFromColorIdx(rgba, colorIndex, buffer.fTransparency);
   return new TGLPhysicalShape(ID, logical, buffer.fLocalMaster,
                               buffer.fReflection, rgba);
}


//______________________________________________________________________________
void TGLScenePad::ComposePolymarker(const TList *lst)
{
   TPolyMarker3D *pm = 0;
   TH3 *th3 = 0;
   TObjOptLink *lnk = (TObjOptLink*)lst->FirstLink();
   for (; lnk; lnk = (TObjOptLink*)lnk->Next()) {
      TObject *obj = lnk->GetObject();
      if (TPolyMarker3D *dPm = dynamic_cast<TPolyMarker3D*>(obj)) {
         if(!pm)
            pm = dPm;
      } else if (TH3 *dTH3 = dynamic_cast<TH3*>(obj)) {
         if(!th3 && !dTH3->GetEntries())
            th3 = dTH3;
      } else
         ObjectPaint(obj, lnk->GetOption());

      if (pm && th3) {
         //Create a new TH3 plot, containing polymarker.
         TGLPlot3D* log = TGLPlot3D::CreatePlot(th3, pm);
         AdoptLogical(*log);
         //Try to extract polymarker's color and
         //create a physical shape with correct color.
         const Color_t cInd = pm->GetMarkerColor();
         if (TColor *c = gROOT->GetColor(cInd)) {
            Float_t rgba[4] = {0.f, 0.f, 0.f, 1.};
            c->GetRGB(rgba[0], rgba[1], rgba[2]);
            AddHistoPhysical(log, rgba);
         } else
            AddHistoPhysical(log);

         //Composition was added into gl-viewer.
         pm = 0;
         th3 = 0;
      }
   }
}

//______________________________________________________________________________
RootCsg::TBaseMesh* TGLScenePad::BuildComposite()
{
   // Build and return composite shape mesh
   const CSPart_t &currToken = fCSTokens[fCSLevel];
   UInt_t opCode = currToken.first;

   if (opCode != TBuffer3D::kCSNoOp) {
      ++fCSLevel;
      RootCsg::TBaseMesh *left = BuildComposite();
      RootCsg::TBaseMesh *right = BuildComposite();
      //RootCsg::TBaseMesh *result = 0;
      switch (opCode) {
      case TBuffer3D::kCSUnion:
         return RootCsg::BuildUnion(left, right);
      case TBuffer3D::kCSIntersection:
         return RootCsg::BuildIntersection(left, right);
      case TBuffer3D::kCSDifference:
         return RootCsg::BuildDifference(left, right);
      default:
         Error("BuildComposite", "Wrong operation code %d\n", opCode);
         return 0;
      }
   } else return fCSTokens[fCSLevel++].second;
}

//______________________________________________________________________________
TGLLogicalShape* TGLScenePad::AttemptDirectRenderer(TObject* id)
{
   // Try to construct an appropriate logical-shape sub-class based
   // on id'class, following convention that SomeClassGL is a suitable
   // renderer for class SomeClass.

   TClass* cls = TGLObject::GetGLRenderer(id->IsA());
   if (cls == 0)
      return 0;

   TGLObject* rnr = reinterpret_cast<TGLObject*>(cls->New());
   if (rnr) {
      Bool_t status;
      try
      {
         status = rnr->SetModel(id);
      }
      catch (std::exception&)
      {
         status = kFALSE;
      }
      if (!status)
      {
         Warning("TGLScenePad::AttemptDirectRenderer", "failed initializing direct rendering.");
         delete rnr;
         return 0;
      }
      rnr->SetBBox();
      AdoptLogical(*rnr);
   }
   return rnr;
}
