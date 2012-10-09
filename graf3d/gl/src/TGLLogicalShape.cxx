// @(#)root/gl:$Id$
// Author:  Richard Maunder  25/05/2005

#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TGLRnrCtx.h"
#include "TGLScene.h"
#include "TGLCamera.h"
#include "TGLSelectRecord.h"
#include "TGLContext.h"
#include "TGLIncludes.h"

#include "TBuffer3D.h"
#include "TClass.h"
#include "TContextMenu.h"


//==============================================================================
// TGLLogicalShape
//==============================================================================

//______________________________________________________________________________
//
// Abstract logical shape - a GL 'drawable' - base for all shapes -
// faceset sphere etc. Logical shapes are a unique piece of geometry,
// described in it's local frame - e.g if we have three spheres in :
// Sphere A - Radius r1, center v1
// Sphere B - Radius r2, center v2
// Sphere C - Radius r1, center v3
//
// Spheres A and C can share a common logical sphere of radius r1 - and
// place them with two physicals with translations of v1 & v2.  Sphere B
// requires a different logical (radius r2), placed with physical with
// translation v2.
//
// Physical shapes know about and can share logicals. Logicals do not
// about (aside from reference counting) physicals or share them.
//
// This sharing of logical shapes greatly reduces memory consumption and
// scene (re)build times in typical detector geometries which have many
// repeated objects placements.
//
// TGLLogicalShapes have reference counting, performed by the client
// physical shapes which are using it.
//
// Display list information is also stored here, possibly per LOD
// level. Most classes do not support LOD (only sphere and tube) and
// therefore reasonable defaults are encoded in the following virtual
// functions:
//
// * ELODAxes SupportedLODAxes()  { return kLODAxesNone; }
// * Int_t    DLCacheSize()       { return 1; }
// * UInt_t   DLOffset(lod);      // Transform lod into DL offset.
// * Short_t  QuantizeShapeLOD(); // Quantize lod.
//
// Classes that have per-LOD display-lists than override these functions.
// 'UShort_t fDLValid' is used as a bit-field determining validity of
// each quantized LOD-level; hopefully one will not have more than 16
// LOD levels per class.
// See also: TGLPhysicalShape::CalculateShapeLOD() where LOD is calculated.
//
// See base/src/TVirtualViewer3D for description of common external 3D
// viewer architecture and how external viewer clients use it.
//

ClassImp(TGLLogicalShape);

Bool_t TGLLogicalShape::fgIgnoreSizeForCameraInterest = kFALSE;

//______________________________________________________________________________
TGLLogicalShape::TGLLogicalShape() :
   fRef           (0),
   fFirstPhysical (0),
   fExternalObj   (0),
   fScene         (0),
   fDLBase        (0),
   fDLSize        (1),
   fDLValid       (0),
   fDLCache       (kTRUE),
   fRefStrong     (kFALSE),
   fOwnExtObj     (kFALSE)
{
   // Constructor.
}

//______________________________________________________________________________
TGLLogicalShape::TGLLogicalShape(TObject* obj) :
   fRef           (0),
   fFirstPhysical (0),
   fExternalObj   (obj),
   fScene         (0),
   fDLBase        (0),
   fDLSize        (1),
   fDLValid       (0),
   fDLCache       (kTRUE),
   fRefStrong     (kFALSE),
   fOwnExtObj     (kFALSE)
{
   // Constructor with external object.
}

//______________________________________________________________________________
TGLLogicalShape::TGLLogicalShape(const TBuffer3D & buffer) :
   fRef           (0),
   fFirstPhysical (0),
   fExternalObj   (buffer.fID),
   fScene         (0),
   fDLBase        (0),
   fDLSize        (1),
   fDLValid       (0),
   fDLCache       (kTRUE),
   fRefStrong     (kFALSE),
   fOwnExtObj     (kFALSE)
{
   // Constructor from TBuffer3D.

   // Use the bounding box in buffer if valid
   if (buffer.SectionsValid(TBuffer3D::kBoundingBox)) {
      fBoundingBox.Set(buffer.fBBVertex);
   } else if (buffer.SectionsValid(TBuffer3D::kRaw)) {
   // otherwise use the raw points to generate one
      fBoundingBox.SetAligned(buffer.NbPnts(), buffer.fPnts);
   }

   // If the logical is created without an external object reference,
   // we create a generic  here and delete it during the destruction.
   if (fExternalObj == 0)
   {
      fExternalObj = new TNamed("Generic object", "Internal object created for bookkeeping.");
      fOwnExtObj = kTRUE;
   }
}

//______________________________________________________________________________
TGLLogicalShape::~TGLLogicalShape()
{
   // Destroy logical shape.

   // Physicals should have been cleared elsewhere as they are managed
   // by the scene. But this could change.
   if (fRef > 0) {
      Warning("TGLLogicalShape::~TGLLogicalShape", "some physicals still lurking around.");
      DestroyPhysicals();
   }
   DLCachePurge();
   if (fOwnExtObj)
   {
      delete fExternalObj;
   }
}


/**************************************************************************/
// Physical shape ref-counting, replica management
/**************************************************************************/

//______________________________________________________________________________
void TGLLogicalShape::AddRef(TGLPhysicalShape* phys) const
{
   // Add reference to given physical shape.

   phys->fNextPhysical = fFirstPhysical;
   fFirstPhysical = phys;
   ++fRef;
}

//______________________________________________________________________________
void TGLLogicalShape::SubRef(TGLPhysicalShape* phys) const
{
   // Remove reference to given physical shape, potentially deleting
   // *this* object when hitting zero ref-count (if fRefStrong is
   // true).

   assert(phys != 0);

   Bool_t found = kFALSE;
   if (fFirstPhysical == phys) {
      fFirstPhysical = phys->fNextPhysical;
      found = kTRUE;
   } else {
      TGLPhysicalShape *shp1 = fFirstPhysical, *shp2;
      while ((shp2 = shp1->fNextPhysical) != 0) {
         if (shp2 == phys) {
            shp1->fNextPhysical = shp2->fNextPhysical;
            found = kTRUE;
            break;
         }
         shp1 = shp2;
      }
   }
   if (found == kFALSE) {
      Error("TGLLogicalShape::SubRef", "Attempt to un-ref an unregistered physical.");
      return;
   }

   if (--fRef == 0 && fRefStrong)
      delete this;
}

//______________________________________________________________________________
void TGLLogicalShape::DestroyPhysicals()
{
   // Destroy all physicals attached to this logical.

   TGLPhysicalShape *curr = fFirstPhysical, *next;
   while (curr)
   {
      next = curr->fNextPhysical;
      curr->fLogicalShape = 0;
      --fRef;
      delete curr;
      curr = next;
   }
   assert (fRef == 0);
   fFirstPhysical = 0;
}

//______________________________________________________________________________
UInt_t TGLLogicalShape::UnrefFirstPhysical()
{
   // Unreference first physical in the list, returning its id and
   // making it fit for destruction somewhere else.
   // Returns 0 if there are no replicas attached.

   if (fFirstPhysical == 0) return 0;

   TGLPhysicalShape *phys = fFirstPhysical;
   UInt_t            phid = phys->ID();
   fFirstPhysical = phys->fNextPhysical;
   phys->fLogicalShape = 0;
   --fRef;
   return phid;
}


/**************************************************************************/
// Bounding-boxes
/**************************************************************************/

//______________________________________________________________________________
void TGLLogicalShape::UpdateBoundingBoxesOfPhysicals()
{
   // Update bounding-boxed of all dependent physicals.

   TGLPhysicalShape* pshp = fFirstPhysical;
   while (pshp)
   {
      pshp->UpdateBoundingBox();
      pshp = pshp->fNextPhysical;
   }
}


/**************************************************************************/
// Display-list cache
/**************************************************************************/

//______________________________________________________________________________
Bool_t TGLLogicalShape::SetDLCache(Bool_t cache)
{
   // Modify capture of draws into display list cache kTRUE - capture,
   // kFALSE direct draw. Return kTRUE is state changed, kFALSE if not.

   if (cache == fDLCache)
      return kFALSE;

   if (fDLCache)
      DLCachePurge();
   fDLCache = cache;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLLogicalShape::ShouldDLCache(const TGLRnrCtx& rnrCtx) const
{
   // Returns kTRUE if draws should be display list cached
   // kFALSE otherwise.
   //
   // Here we check that:
   // a) fScene is set (Scene manages link to GL-context);
   // b) secondary selection is not in progress as different
   //    render-path is usually taken in this case.
   //
   // Otherwise we return internal bool.
   //
   // Override this in sub-class if different behaviour is required.

   if (!fDLCache || !fScene   ||
       (rnrCtx.SecSelection() && SupportsSecondarySelect()))
   {
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGLLogicalShape::DLCacheClear()
{
   // Clear all entries for all LODs for this drawable from the
   // display list cache but keeping the reserved ids from GL context.

   fDLValid = 0;
}

//______________________________________________________________________________
void TGLLogicalShape::DLCacheDrop()
{
   // Drop all entries for all LODs for this drawable from the display
   // list cache, WITHOUT returning the reserved ids to GL context.
   //
   // This is called by scene if it realized that the GL context was
   // destroyed.

   fDLBase  = 0;
   fDLValid = 0;
}

//______________________________________________________________________________
void TGLLogicalShape::DLCachePurge()
{
   // Purge all entries for all LODs for this drawable from the
   // display list cache, returning the reserved ids to GL context.
   //
   // If you override this function:
   // 1. call the base-class version from it;
   // 2. call it from the destructor of the derived class!

   if (fDLBase != 0)
   {
      PurgeDLRange(fDLBase, fDLSize);
      fDLBase  = 0;
      fDLValid = 0;
   }
}

//______________________________________________________________________________
void TGLLogicalShape::PurgeDLRange(UInt_t base, Int_t size) const
{
   // Purge given display-list range.
   // Utility function.

   if (fScene)
   {
      fScene->GetGLCtxIdentity()->RegisterDLNameRangeToWipe(base, size);
   }
   else
   {
      Warning("TGLLogicalShape::PurgeDLRange", "Scene unknown, attempting direct deletion.");
      glDeleteLists(base, size);
   }
}

//______________________________________________________________________________
Short_t TGLLogicalShape::QuantizeShapeLOD(Short_t shapeLOD,
                                          Short_t /*combiLOD*/) const
{
   // Logical shapes usually support only discreet LOD values,
   // especially in view of display-list caching.
   // This function should be overriden to perform the desired quantization.
   // See TGLSphere.

   return shapeLOD;
}

//______________________________________________________________________________
void TGLLogicalShape::Draw(TGLRnrCtx& rnrCtx) const
{
   // Draw the GL drawable, using draw flags. If DL caching is enabled
   // (see SetDLCache) then attempt to draw from the cache, if not found
   // attempt to capture the draw - done by DirectDraw() - into a new cache entry.
   // If not cached just call DirectDraw() for normal non DL cached drawing.

   // Debug tracing
   if (gDebug > 4) {
      Info("TGLLogicalShape::Draw", "this %ld (class %s) LOD %d", (Long_t)this, IsA()->GetName(), rnrCtx.ShapeLOD());
   }

entry_point:
   // If shape is not cached, or a capture to cache is already in
   // progress perform a direct draw DL can be nested, but not created
   // in nested fashion. As we only build DL on draw demands have to
   // protected against this here.
   // MT: I can't see how this could happen right now ... with
   // rendering from a flat drawable-list.

   if (!ShouldDLCache(rnrCtx) || rnrCtx.IsDLCaptureOpen())
   {
      DirectDraw(rnrCtx);
      return;
   }

   if (fDLBase == 0)
   {
      fDLBase = glGenLists(fDLSize);
      if (fDLBase == 0)
      {
         Warning("TGLLogicalShape::Draw", "display-list registration failed.");
         fDLCache = kFALSE;
         goto entry_point;
      }
   }

   Short_t lod = rnrCtx.ShapeLOD();
   UInt_t  off = DLOffset(lod);
   if ((1<<off) & fDLValid)
   {
      glCallList(fDLBase + off);
   }
   else
   {
      rnrCtx.OpenDLCapture();
      glNewList(fDLBase + off, GL_COMPILE_AND_EXECUTE);
      DirectDraw(rnrCtx);
      glEndList();
      rnrCtx.CloseDLCapture();
      fDLValid |= (1<<off);
   }
}

//______________________________________________________________________________
void TGLLogicalShape::DrawHighlight(TGLRnrCtx& rnrCtx, const TGLPhysicalShape* pshp, Int_t lvl) const
{
   // Draw the logical shape in highlight mode.
   // If lvl argument is less than 0 (-1 by default), the index into color-set
   // is taken from the physical shape itself.

   if (lvl < 0) lvl = pshp->GetSelected();

   glColor4ubv(rnrCtx.ColorSet().Selection(lvl).CArr());
   TGLUtil::LockColor();
   Draw(rnrCtx);
   TGLUtil::UnlockColor();
}

//______________________________________________________________________________
void TGLLogicalShape::ProcessSelection(TGLRnrCtx& /*rnrCtx*/, TGLSelectRecord& rec)
{
   // Virtual method called-back after a secondary selection hit
   // is recorded (see TGLViewer::HandleButton(), Ctrl-Button1).
   // The ptr argument holds the GL pick-record of the closest hit.
   //
   // This base-class implementation simply prints out the result.

   printf("TGLLogicalShape::ProcessSelection %d names on the stack (z1=%g, z2=%g).\n",
          rec.GetN(), rec.GetMinZ(), rec.GetMaxZ());
   printf("  Names: ");
   for (Int_t j=0; j<rec.GetN(); ++j) printf ("%u ", rec.GetItem(j));
   printf("\n");
}

//______________________________________________________________________________
void TGLLogicalShape::InvokeContextMenu(TContextMenu& menu, UInt_t x, UInt_t y) const
{
   // Invoke popup menu or our bound external TObject (if any), using passed
   // 'menu' object, at location 'x' 'y'
   if (fExternalObj) {
      menu.Popup(x, y, fExternalObj);
   }
}

//______________________________________________________________________________
Bool_t TGLLogicalShape::IgnoreSizeForOfInterest() const
{
   // Return true if size of this shape should be ignored when determining if
   // the object should be drawn. In this base-class we simply return state of
   // static flag fgIgnoreSizeForCameraInterest.
   //
   // Several sub-classes override this virtual function.

   return fgIgnoreSizeForCameraInterest;
}

//______________________________________________________________________________
Bool_t TGLLogicalShape::GetIgnoreSizeForCameraInterest()
{
   // Get state of static fgIgnoreSizeForCameraInterest flag.
   // When this is true all objects, also very small, will be drawn by GL.

   return fgIgnoreSizeForCameraInterest;
}

//______________________________________________________________________________
void TGLLogicalShape::SetIgnoreSizeForCameraInterest(Bool_t isfci)
{
   // Set state of static fgIgnoreSizeForCameraInterest flag.

   fgIgnoreSizeForCameraInterest = isfci;
}
