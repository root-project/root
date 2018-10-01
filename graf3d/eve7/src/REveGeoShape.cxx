// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/REveGeoShape.hxx"
#include "ROOT/REveTrans.hxx"
#include "ROOT/REveManager.hxx"
#include "ROOT/REvePolygonSetProjected.hxx"
#include "ROOT/REveProjections.hxx"
#include "ROOT/REveProjectionManager.hxx"

#include "ROOT/REveGeoShapeExtract.hxx"
#include "ROOT/REveGeoPolyShape.hxx"
#include "ROOT/REveRenderData.hxx"

#include "TROOT.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TVirtualViewer3D.h"
#include "TColor.h"
#include "TFile.h"

#include "TGeoShape.h"
#include "TGeoVolume.h"
#include "TGeoNode.h"
#include "TGeoShapeAssembly.h"
#include "TGeoCompositeShape.h"
#include "TGeoBoolNode.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TVirtualGeoPainter.h"

#include "json.hpp"


namespace
{
   TGeoManager* init_geo_mangeur()
   {
      // Create a phony geo manager that can be used for storing free
      // shapes. Otherwise shapes register themselves to current
      // geo-manager (or even create one).

      TGeoManager  *old    = gGeoManager;
      TGeoIdentity *old_id = gGeoIdentity;
      gGeoManager = 0;
      TGeoManager* mgr = new TGeoManager();
      mgr->SetNameTitle("REveGeoShape::fgGeoMangeur",
                        "Static geo manager used for wrapped TGeoShapes.");
      gGeoIdentity = new TGeoIdentity("Identity");
      gGeoManager  = old;
      gGeoIdentity = old_id;
      return mgr;
   }

  TGeoHMatrix localGeoHMatrixIdentity;
}

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

/** \class REveGeoShape
\ingroup REve
Wrapper for TGeoShape with absolute positioning and color
attributes allowing display of extracted TGeoShape's (without an
active TGeoManager) and simplified geometries (needed for non-linear
projections).

TGeoCompositeShapes and TGeoAssemblies are supported.

If fNSegments data-member is < 2 (0 by default), the default number of
segments is used for tesselation and special GL objects are
instantiated for selected shapes (spheres, tubes). If fNSegments is > 2,
it gets forwarded to geo-manager and this tesselation detail is
used when creating the buffer passed to GL.
*/

TGeoManager* REX::REveGeoShape::fgGeoMangeur = init_geo_mangeur();

////////////////////////////////////////////////////////////////////////////////
/// Return static geo-manager that is used internally to make shapes
/// lead a happy life.
/// Set gGeoManager to this object when creating TGeoShapes to be
/// passed into EveGeoShapes.

TGeoManager* REveGeoShape::GetGeoMangeur()
{
   return fgGeoMangeur;
}

////////////////////////////////////////////////////////////////////////////////
/// Return static identity matrix in homogeneous representation.
/// This is needed because TGeoCompositeShape::PaintComposite()
/// assumes TGeoShape::fgTransform is a TGeoHMatrix and we need to pass in
/// an identity matrix when painting a composite shape.

TGeoHMatrix* REveGeoShape::GetGeoHMatrixIdentity()
{
   return &localGeoHMatrixIdentity;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveGeoShape::REveGeoShape(const char* name, const char* title) :
   REveShape       (name, title),
   fNSegments      (0),
   fShape          (0),
   fCompositeShape (0)
{
   InitMainTrans();
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

REveGeoShape::~REveGeoShape()
{
   SetShape(0);
}

////////////////////////////////////////////////////////////////////////////////
/// Create derived REveGeoShape form a TGeoCompositeShape.

TGeoShape* REveGeoShape::MakePolyShape()
{
   return REveGeoPolyShape::Construct(fCompositeShape, fNSegments);
}

////////////////////////////////////////////////////////////////////////////////
/// Fill core part of JSON representation.

Int_t REveGeoShape::WriteCoreJson(nlohmann::json& j, Int_t rnr_offset)
{
   Int_t ret = REveShape::WriteCoreJson(j, rnr_offset);

   // XXXXX Apaprently don't need this one ...

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Crates 3D point array for rendering.

void REveGeoShape::BuildRenderData()
{
   if (fShape == 0) return;

   REveGeoPolyShape *egps = nullptr;

   if (fCompositeShape)
   {
      egps = dynamic_cast<REveGeoPolyShape*>(fShape);
   }
   else
   {
      REveGeoManagerHolder gmgr(fgGeoMangeur, fNSegments);

      TBuffer3D *b3d = fShape->MakeBuffer3D();
      egps = new REveGeoPolyShape();
      egps->SetFromBuff3D(*b3d);
      delete b3d;
   }

   fRenderData = std::make_unique<REveRenderData>("makeEveGeoShape");

   egps->FillRenderData(*fRenderData);

   if (fCompositeShape == 0)
   {
      delete egps;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set number of segments.

void REveGeoShape::SetNSegments(Int_t s)
{
   if (s != fNSegments && fCompositeShape != 0)
   {
      delete fShape;
      fShape = MakePolyShape();
   }
   fNSegments = s;
}

////////////////////////////////////////////////////////////////////////////////
/// Set TGeoShape shown by this object.
///
/// The shape is owned by REveGeoShape but TGeoShape::fUniqueID is
/// used for reference counting so you can pass the same shape to
/// several EveGeoShapes.
///
/// If it if is taken from an existing TGeoManager, manually
/// increase the fUniqueID before passing it to REveGeoShape.

void REveGeoShape::SetShape(TGeoShape* s)
{
   REveGeoManagerHolder gmgr(fgGeoMangeur);

   if (fCompositeShape)
   {
      delete fShape;
      fShape = fCompositeShape;
   }
   if (fShape)
   {
      fShape->SetUniqueID(fShape->GetUniqueID() - 1);
      if (fShape->GetUniqueID() == 0)
      {
         delete fShape;
      }
   }
   fShape = s;
   if (fShape)
   {
      fShape->SetUniqueID(fShape->GetUniqueID() + 1);
      fCompositeShape = dynamic_cast<TGeoCompositeShape*>(fShape);
      if (fCompositeShape)
      {
         fShape = MakePolyShape();
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Compute bounding-box.

void REveGeoShape::ComputeBBox()
{
   TGeoBBox *bb = dynamic_cast<TGeoBBox*>(fShape);
   if (bb)
   {
      BBoxInit();
      const Double_t *o = bb->GetOrigin();
      BBoxCheckPoint(o[0] - bb->GetDX(), o[0] - bb->GetDY(), o[0] - bb->GetDZ());
      BBoxCheckPoint(o[0] + bb->GetDX(), o[0] + bb->GetDY(), o[0] + bb->GetDZ());
   }
   else
   {
      BBoxZero();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Paint object.

void REveGeoShape::Paint(Option_t* /*option*/)
{
   static const REveException eh("REveGeoShape::Paint ");

   if (fShape == 0)
      return;

   REveGeoManagerHolder gmgr(fgGeoMangeur, fNSegments);

   if (fCompositeShape)
   {
      Double_t halfLengths[3] = { fCompositeShape->GetDX(), fCompositeShape->GetDY(), fCompositeShape->GetDZ() };

      TBuffer3D buff(TBuffer3DTypes::kComposite);
      buff.fID           = this;
      buff.fColor        = GetMainColor();
      buff.fTransparency = GetMainTransparency();
      RefMainTrans().SetBuffer3D(buff);
      buff.fLocalFrame   = kTRUE; // Always enforce local frame (no geo manager).
      buff.SetAABoundingBox(fCompositeShape->GetOrigin(), halfLengths);
      buff.SetSectionsValid(TBuffer3D::kCore|TBuffer3D::kBoundingBox);

      Bool_t paintComponents = kTRUE;

      // Start a composite shape, identified by this buffer
      if (TBuffer3D::GetCSLevel() == 0)
         paintComponents = gPad->GetViewer3D()->OpenComposite(buff);

      TBuffer3D::IncCSLevel();

      // Paint the boolean node - will add more buffers to viewer
      TGeoMatrix *gst = TGeoShape::GetTransform();
      TGeoShape::SetTransform(REveGeoShape::GetGeoHMatrixIdentity());
      if (paintComponents) fCompositeShape->GetBoolNode()->Paint("");
      TGeoShape::SetTransform(gst);
      // Close the composite shape
      if (TBuffer3D::DecCSLevel() == 0)
         gPad->GetViewer3D()->CloseComposite();
   }
   else
   {
      TBuffer3D& buff = (TBuffer3D&) fShape->GetBuffer3D
         (TBuffer3D::kCore, kFALSE);

      buff.fID           = this;
      buff.fColor        = GetMainColor();
      buff.fTransparency = GetMainTransparency();
      RefMainTrans().SetBuffer3D(buff);
      buff.fLocalFrame   = kTRUE; // Always enforce local frame (no geo manager).

      Int_t sections = TBuffer3D::kBoundingBox | TBuffer3D::kShapeSpecific;
      if (fNSegments > 2)
         sections |= TBuffer3D::kRawSizes | TBuffer3D::kRaw;
      fShape->GetBuffer3D(sections, kTRUE);

      Int_t reqSec = gPad->GetViewer3D()->AddObject(buff);

      if (reqSec != TBuffer3D::kNone) {
         // This shouldn't happen, but I suspect it does sometimes.
         if (reqSec & TBuffer3D::kCore)
            Warning(eh, "Core section required again for shape='%s'. This shouldn't happen.", GetName());
         fShape->GetBuffer3D(reqSec, kTRUE);
         reqSec = gPad->GetViewer3D()->AddObject(buff);
      }

      if (reqSec != TBuffer3D::kNone)
         Warning(eh, "Extra section required: reqSec=%d, shape=%s.", reqSec, GetName());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Save the shape tree as REveGeoShapeExtract.
/// File is always recreated.

void REveGeoShape::SaveExtract(const char* file, const char* name)
{
   REveGeoShapeExtract* gse = DumpShapeTree(this, 0);

   TFile f(file, "RECREATE");
   gse->Write(name);
   f.Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Write the shape tree as REveGeoShapeExtract to current directory.

void REveGeoShape::WriteExtract(const char* name)
{
   REveGeoShapeExtract* gse = DumpShapeTree(this, 0);
   gse->Write(name);
}

////////////////////////////////////////////////////////////////////////////////
/// Export this shape and its descendants into a geoshape-extract.

REveGeoShapeExtract* REveGeoShape::DumpShapeTree(REveGeoShape* gsre,
                                                 REveGeoShapeExtract* parent)
{
   REveGeoShapeExtract* she = new REveGeoShapeExtract(gsre->GetName(), gsre->GetTitle());
   she->SetTrans(gsre->RefMainTrans().Array());
   {
      Int_t   ci = gsre->GetFillColor();
      TColor *c  = gROOT->GetColor(ci);
      Float_t rgba[4] = { 1, 0, 0, Float_t(1 - gsre->GetMainTransparency()/100.) };
      if (c)
      {
         rgba[0] = c->GetRed();
         rgba[1] = c->GetGreen();
         rgba[2] = c->GetBlue();
      }
      she->SetRGBA(rgba);
   }
   {
      Int_t   ci = gsre->GetLineColor();
      TColor *c  = gROOT->GetColor(ci);
      Float_t rgba[4] = { 1, 0, 0, 1 };
      if (c)
      {
         rgba[0] = c->GetRed();
         rgba[1] = c->GetGreen();
         rgba[2] = c->GetBlue();
      }
      she->SetRGBALine(rgba);
   }
   she->SetRnrSelf(gsre->GetRnrSelf());
   she->SetRnrElements(gsre->GetRnrChildren());
   she->SetRnrFrame(gsre->GetDrawFrame());
   she->SetMiniFrame(gsre->GetMiniFrame());
   she->SetShape(gsre->GetShape());
   if (gsre->HasChildren())
   {
      TList* ele = new TList();
      she->SetElements(ele);
      she->GetElements()->SetOwner(true);
      REveElement::List_i i = gsre->BeginChildren();
      while (i != gsre->EndChildren()) {
         REveGeoShape* l = dynamic_cast<REveGeoShape*>(*i);
         DumpShapeTree(l, she);
         i++;
      }
   }
   if (parent)
      parent->GetElements()->Add(she);

   return she;
}

////////////////////////////////////////////////////////////////////////////////
/// Import a shape extract 'gse' under element 'parent'.

REveGeoShape* REveGeoShape::ImportShapeExtract(REveGeoShapeExtract* gse,
                                               REveElement*         parent)
{
   REveGeoManagerHolder gmgr(fgGeoMangeur);
   REveManager::RRedrawDisabler redrawOff(REX::gEve);
   REveGeoShape* gsre = SubImportShapeExtract(gse, parent);
   gsre->ElementChanged();
   return gsre;
}

////////////////////////////////////////////////////////////////////////////////
/// Recursive version for importing a shape extract tree.

REveGeoShape* REveGeoShape::SubImportShapeExtract(REveGeoShapeExtract* gse,
                                                  REveElement*         parent)
{
   REveGeoShape* gsre = new REveGeoShape(gse->GetName(), gse->GetTitle());
   gsre->RefMainTrans().SetFromArray(gse->GetTrans());
   const Float_t* rgba = gse->GetRGBA();
   gsre->SetMainColorRGB(rgba[0], rgba[1], rgba[2]);
   gsre->SetMainAlpha(rgba[3]);
   rgba = gse->GetRGBALine();
   gsre->SetLineColor(TColor::GetColor(rgba[0], rgba[1], rgba[2]));
   gsre->SetRnrSelf(gse->GetRnrSelf());
   gsre->SetRnrChildren(gse->GetRnrElements());
   gsre->SetDrawFrame(gse->GetRnrFrame());
   gsre->SetMiniFrame(gse->GetMiniFrame());
   gsre->SetShape(gse->GetShape());

   if (parent)
      parent->AddElement(gsre);

   if (gse->HasElements())
   {
      TIter next(gse->GetElements());
      REveGeoShapeExtract* chld;
      while ((chld = (REveGeoShapeExtract*) next()) != 0)
         SubImportShapeExtract(chld, gsre);
   }

   return gsre;
}

////////////////////////////////////////////////////////////////////////////////
/// Return class for projected objects:
///  - 2D projections: REvePolygonSetProjected,
///  - 3D projections: REveGeoShapeProjected.
/// Virtual from REveProjectable.

TClass* REveGeoShape::ProjectedClass(const REveProjection* p) const
{
   if (p->Is2D())
      return REvePolygonSetProjected::Class();
   else
      return REveGeoShapeProjected::Class();
}

////////////////////////////////////////////////////////////////////////////////
/// Create a TBuffer3D suitable for presentation of the shape.
/// Transformation matrix is also applied.

TBuffer3D* REveGeoShape::MakeBuffer3D()
{
   if (fShape == 0) return 0;

   if (dynamic_cast<TGeoShapeAssembly*>(fShape)) {
      // TGeoShapeAssembly makes a bad TBuffer3D.
      return 0;
   }

   REveGeoManagerHolder gmgr(fgGeoMangeur, fNSegments);

   TBuffer3D* buff  = fShape->MakeBuffer3D();
   REveTrans& mx    = RefMainTrans();
   if (mx.GetUseTrans())
   {
      Int_t n = buff->NbPnts();
      Double_t* pnts = buff->fPnts;
      for(Int_t k = 0; k < n; ++k)
      {
         mx.MultiplyIP(&pnts[3*k]);
      }
   }
   return buff;
}


//==============================================================================
// REveGeoShapeProjected
//==============================================================================

/** \class REveGeoShapeProjected
\ingroup REve
A 3D projected REveGeoShape.
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveGeoShapeProjected::REveGeoShapeProjected() :
   REveShape("REveGeoShapeProjected"),
   fBuff(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// This should never be called as this class is only used for 3D
/// projections.
/// The implementation is required as this metod is abstract.
/// Just emits a warning if called.

void REveGeoShapeProjected::SetDepthLocal(Float_t /*d*/)
{
   Warning("SetDepthLocal", "This function only exists to fulfill an abstract interface.");
}

////////////////////////////////////////////////////////////////////////////////
/// This is virtual method from base-class REveProjected.

void REveGeoShapeProjected::SetProjection(REveProjectionManager* mng,
                                          REveProjectable* model)
{
   REveProjected::SetProjection(mng, model);

   REveGeoShape* gre = dynamic_cast<REveGeoShape*>(fProjectable);
   CopyVizParams(gre);
}

////////////////////////////////////////////////////////////////////////////////
/// This is virtual method from base-class REveProjected.

void REveGeoShapeProjected::UpdateProjection()
{
   REveGeoShape   *gre = dynamic_cast<REveGeoShape*>(fProjectable);
   REveProjection *prj = fManager->GetProjection();

   delete fBuff;
   fBuff = gre->MakeBuffer3D();

   if (fBuff)
   {
      fBuff->SetSectionsValid(TBuffer3D::kCore | TBuffer3D::kRawSizes | TBuffer3D::kRaw);

      Double_t *p = fBuff->fPnts;
      for (UInt_t i = 0; i < fBuff->NbPnts(); ++i, p+=3)
      {
         prj->ProjectPointdv(p, 0);
      }
   }

   ResetBBox();
}

////////////////////////////////////////////////////////////////////////////////
/// Override of virtual method from TAttBBox.

void REveGeoShapeProjected::ComputeBBox()
{
   if (fBuff && fBuff->NbPnts() > 0)
   {
      BBoxInit();

      Double_t *p = fBuff->fPnts;
      for (UInt_t i = 0; i < fBuff->NbPnts(); ++i, p+=3)
      {
         BBoxCheckPoint(p[0], p[1], p[2]);
      }
   }
   else
   {
      BBoxZero();
   }
}
