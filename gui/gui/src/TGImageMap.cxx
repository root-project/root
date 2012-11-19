// @(#)root/gui:$Id$
// Author: Valeriy Onuchin & Fons Rademakers   18/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGImageMap (with TGRegion and TGRegionWithId help classes)           //
//                                                                      //
// A TGImageMap provides the functionality like a clickable image in    //
// a web browser with sensitive regions (MAP HTML tag).                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGImageMap.h"
#include "TRefCnt.h"
#include "TGMenu.h"
#include "TGToolTip.h"
#include "TList.h"
#include "TArrayS.h"


ClassImp(TGRegion)
ClassImp(TGRegionWithId)
ClassImpQ(TGImageMap)


TGRegionWithId  *gCurrentRegion; // current region

static TGRegion *gEmptyRegion = 0;
static Int_t gPointerX;  // current X mouse position
static Int_t gPointerY;  // current Y mouse position



class TGRegionData : public TRefCnt {

friend class TGRegion;

private:
   Region_t   fRgn;     // region handle
   Bool_t     fIsNull;  // true if null region

public:
   TGRegionData() { fRgn = 0; fIsNull = kTRUE; AddReference(); }
   ~TGRegionData() { }
   TGRegionData &operator=(const TGRegionData &r);
};

//______________________________________________________________________________
TGRegionData &TGRegionData::operator=(const TGRegionData &r)
{
   // Assignemnt of region data object.

   if (this != &r) {
      fRefs   = r.fRefs;
      fRgn    = r.fRgn;
      fIsNull = r.fIsNull;
   }
   return *this;
}


////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________
TGRegion::TGRegion()
{
   // Create a region object.

   if (!gEmptyRegion)                      // avoid too many allocs
      gEmptyRegion = new TGRegion(kTRUE);

   fData = gEmptyRegion->fData;
   fData->AddReference();
}

//______________________________________________________________________________
TGRegion::TGRegion(Bool_t is_null)
{
   // Create empty region.

   fData          = new TGRegionData;
   fData->fRgn    = gVirtualX->CreateRegion();
   fData->fIsNull = is_null;
}

//______________________________________________________________________________
TGRegion::TGRegion(Int_t x, Int_t y, UInt_t w, UInt_t h, ERegionType)
{
   // Create and initialize a region with a rectangle.

   fData          = new TGRegionData;
   fData->fRgn    = gVirtualX->CreateRegion();
   fData->fIsNull = kFALSE;

   Rectangle_t xr;
   xr.fX      = (Short_t) x;
   xr.fY      = (Short_t) y;
   xr.fWidth  = (UShort_t) w;
   xr.fHeight = (UShort_t) h;
   gVirtualX->UnionRectWithRegion(&xr, fData->fRgn, fData->fRgn);
}

//______________________________________________________________________________
TGRegion::TGRegion(Int_t n, TPoint *points, Bool_t winding)
{
   // Create and intialize a region with a polygon.

   fData            = new TGRegionData;
   fData->fIsNull   = kFALSE;
   Point_t *gpoints = new Point_t[n];

   for (int i = 0; i < n; i++) {
      gpoints[i].fX = (Short_t) points[i].GetX();
      gpoints[i].fY = (Short_t) points[i].GetY();
   }

   fData->fRgn = gVirtualX->PolygonRegion(gpoints, n, winding);
}

//______________________________________________________________________________
TGRegion::TGRegion(const TArrayS &x, const TArrayS &y, Bool_t winding)
{
   // Create and initialize a region with an X and a Y array of points.

   fData          = new TGRegionData;
   fData->fIsNull = kFALSE;

   Int_t n = x.GetSize();
   if (n != y.GetSize()) {
      Error("TGRegion", "x and y arrays must have same length");
      return;
   }
   Point_t *gpoints = new Point_t[n];

   for (int i = 0; i < n; i++) {
      gpoints[i].fX = x.GetArray()[i];
      gpoints[i].fY = y.GetArray()[i];
   }

   fData->fRgn = gVirtualX->PolygonRegion(gpoints, n, winding);
}

//_____________________________________________________________________________
TGRegion::TGRegion(Int_t n, Int_t *x, Int_t *y, Bool_t winding)
{
   // Create and initialize a region with an X and Y array of points.

   fData          = new TGRegionData;
   fData->fIsNull = kFALSE;
   Point_t *gpoints = new Point_t[n];

   for (int i = 0; i < n; i++) {
      gpoints[i].fX = x[i];
      gpoints[i].fY = y[i];
   }

   fData->fRgn = gVirtualX->PolygonRegion(gpoints, n, winding);
}

//______________________________________________________________________________
TGRegion::TGRegion(const TGRegion &r) : TObject(r)
{
   // Region copy constructor.

   fData = r.fData;
   fData->AddReference();
}

//______________________________________________________________________________
TGRegion::~TGRegion()
{
   // Delete a region.

   if (fData->RemoveReference() <= 0) {
      gVirtualX->DestroyRegion(fData->fRgn);
      delete fData;
   }
}

//______________________________________________________________________________
TGRegion &TGRegion::operator=(const TGRegion &r)
{
   // Region assignment operator.

   if (this != &r) {
      TObject::operator=(r);
      r.fData->AddReference();

      if (fData->RemoveReference() <= 0) {
         gVirtualX->DestroyRegion(fData->fRgn);
         delete fData;
      }
      fData = r.fData;
   }
   return *this;
}

//______________________________________________________________________________
TGRegion TGRegion::CopyRegion() const
{
   // Copy a region.

   TGRegion r(fData->fIsNull);
   gVirtualX->UnionRegion(fData->fRgn, r.fData->fRgn, r.fData->fRgn);
   return r;
}

//______________________________________________________________________________
Bool_t TGRegion::IsNull() const
{
   // Return true if region is not set.

   return fData->fIsNull;
}

//______________________________________________________________________________
Bool_t TGRegion::IsEmpty() const
{
   // Return true if region is empty.

   return fData->fIsNull || gVirtualX->EmptyRegion(fData->fRgn);
}

//______________________________________________________________________________
Bool_t TGRegion::Contains(const TPoint &p) const
{
   // Return true if point p is contained in the region.

   return gVirtualX->PointInRegion((Int_t)p.GetX(), (Int_t)p.GetY(), fData->fRgn);
}

//______________________________________________________________________________
Bool_t TGRegion::Contains(Int_t x, Int_t y) const
{
   // Return true if point (x,y) is contained in the region.

   return gVirtualX->PointInRegion(x, y, fData->fRgn);
}

//______________________________________________________________________________
TGRegion TGRegion::Unite(const TGRegion &r) const
{
   // Return the union of this region with r.

   TGRegion result(kFALSE);
   gVirtualX->UnionRegion(fData->fRgn, r.fData->fRgn, result.fData->fRgn);
   return result;
}

//______________________________________________________________________________
TGRegion TGRegion::Intersect(const TGRegion &r) const
{
   // Returns a region which is the intersection of this region and r.

   TGRegion result(kFALSE);
   gVirtualX->IntersectRegion(fData->fRgn, r.fData->fRgn, result.fData->fRgn);
   return result;
}

//______________________________________________________________________________
TGRegion TGRegion::Subtract(const TGRegion &r) const
{
   // Returns a region which is r subtracted from this region.

   TGRegion result(kFALSE);
   gVirtualX->SubtractRegion(fData->fRgn, r.fData->fRgn, result.fData->fRgn);
   return result;
}

//______________________________________________________________________________
TGRegion TGRegion::Eor(const TGRegion &r) const
{
   // Returns a region which is the difference between the union and
   // intersection this region and r.

   TGRegion result(kFALSE);
   gVirtualX->XorRegion(fData->fRgn, r.fData->fRgn, result.fData->fRgn);
   return result;
}

//______________________________________________________________________________
TGDimension TGRegion::GetDimension() const
{
   // Return dimension of region (widht, height).

   Rectangle_t r = { 0, 0, 0, 0 };
   gVirtualX->GetRegionBox(fData->fRgn, &r);
   return TGDimension(r.fWidth, r.fHeight);
}

//______________________________________________________________________________
TGPosition TGRegion::GetPosition() const
{
   // Return position of region (x, y).

   Rectangle_t r = { 0, 0, 0, 0 };
   gVirtualX->GetRegionBox(fData->fRgn, &r);
   return TGPosition(r.fX, r.fY);
}

//______________________________________________________________________________
Bool_t TGRegion::operator==(const TGRegion &r) const
{
   // Region == operator.

   return fData == r.fData ?
             kTRUE : gVirtualX->EqualRegion(fData->fRgn, r.fData->fRgn);
}


////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________
TGRegionWithId::TGRegionWithId() : TGRegion()
{
   // Create GUI region (with id and possible tooltip).

   fId    = 0;
   fTip   = 0;
   fPopup = 0;
}

//______________________________________________________________________________
TGRegionWithId::TGRegionWithId(Int_t id, Int_t x, Int_t y,
                               UInt_t w, UInt_t h, ERegionType type) :
   TGRegion(x, y, w, h, type)
{
   // Create GUI region (with id and possible tooltip).

   fId    = id;
   fTip   = 0;
   fPopup = 0;
}

//______________________________________________________________________________
TGRegionWithId::TGRegionWithId(Int_t id, Int_t n, TPoint *points,
                               Bool_t winding) :
   TGRegion(n, points, winding)
{
   // Create GUI region (with id and possible tooltip).

   fId    = id;
   fTip   = 0;
   fPopup = 0;
}

//______________________________________________________________________________
TGRegionWithId::TGRegionWithId(const TGRegionWithId &reg) : TGRegion(reg)
{
   // Copy constructor.

   fId    = reg.GetId();
   fTip   = 0;
   fPopup = 0;
}

//______________________________________________________________________________
TGRegionWithId::TGRegionWithId(const TGRegion &reg, Int_t id) :
   TGRegion(reg)
{
   // Copy ctor which allows setting of new id.

   fId    = id;
   fTip   = 0;
   fPopup = 0;
}

//______________________________________________________________________________
TGRegionWithId::~TGRegionWithId()
{
   // Cleanup.

   delete fTip;
}

//______________________________________________________________________________
void TGRegionWithId::DisplayPopup()
{
   // Display popup menu associated with this region.

   if (fPopup) fPopup->PlaceMenu(gPointerX, gPointerY, kTRUE, kTRUE);
}

//______________________________________________________________________________
void TGRegionWithId::SetToolTipText(const char *text, Long_t delayms,
                                    const TGFrame *frame)
{
   // Set tool tip text associated with this region. The delay is in
   // milliseconds (minimum 250). To remove tool tip call method with
   // text = 0.

   if (fTip) {
      delete fTip;
      fTip = 0;
   }

   if (text && strlen(text))
      fTip = new TGToolTip(gClient->GetDefaultRoot(), frame, text, delayms);
}

////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________
TGImageMap::TGImageMap(const TGWindow *p, const TGPicture *pic) :
   TGPictureButton(p, pic)
{
   // Create an image map widget.

   fCursorMouseOut  = kPointer;
   fCursorMouseOver = kHand;
   fListOfRegions   = new TList;
   fTrash           = new TList;
   fMainTip         = 0;
   fNavMode = kNavRegions;

   SetDisabledPicture(fPic);
   SetState(kButtonDisabled);

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask |
                         kPointerMotionMask, kNone, kNone);

   AddInput(kKeyPressMask | kKeyReleaseMask | kPointerMotionMask |
            kStructureNotifyMask | kLeaveWindowMask);
   SetWindowName();
}

//______________________________________________________________________________
TGImageMap::TGImageMap(const TGWindow *p, const TString &pic) :
   TGPictureButton(p, pic.Data())
{
   // Create an image map widget.

   fCursorMouseOut  = kPointer;
   fCursorMouseOver = kHand;
   fListOfRegions   = new TList;
   fTrash           = new TList;
   fMainTip         = 0;
   fNavMode = kNavRegions;

   SetDisabledPicture(fPic);
   SetState(kButtonDisabled);

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask |
                         kPointerMotionMask, kNone, kNone);

   AddInput(kKeyPressMask | kKeyReleaseMask | kPointerMotionMask |
            kStructureNotifyMask | kLeaveWindowMask);
   SetWindowName();
}

//______________________________________________________________________________
TGImageMap::~TGImageMap()
{
   // Cleanup image map widget.

   delete fMainTip;
   fTrash->Delete();
   delete fTrash;
   fListOfRegions->Delete();
   delete fListOfRegions;
}

//______________________________________________________________________________
void TGImageMap::AddRegion(const TGRegion &region, Int_t id)
{
   // Add a region to the image map.

   fListOfRegions->Add(new TGRegionWithId(region, id));
}

//______________________________________________________________________________
TGPopupMenu *TGImageMap::CreatePopup(Int_t id)
{
   // Create popoup menu or returns existing for regions with specified id.

   TIter next(fListOfRegions);
   TGRegionWithId *region;
   TGPopupMenu    *popup = 0;
   TGPopupMenu    *newpopup = 0;

   while ((region = (TGRegionWithId*)next())) {
      if (id == region->GetId()) {
         popup = region->GetPopup();
         if (!popup && !newpopup) {
            newpopup = new TGPopupMenu(this);
            fTrash->Add(newpopup);
         }
         if (newpopup) region->SetPopup(newpopup);
      }
   }
   return newpopup ? newpopup : popup;
}

//______________________________________________________________________________
TGPopupMenu *TGImageMap::GetPopup(Int_t id)
{
   // Return popup for regions with specified id.

   TIter next(fListOfRegions);
   TGRegionWithId *region;

   while ((region = (TGRegionWithId*)next())) {
      if (id == region->GetId()) return region->GetPopup();
   }
   return 0;
}

//______________________________________________________________________________
Bool_t TGImageMap::HandleMotion(Event_t *event)
{
   // Handle mouse motion events.

   TIter next(fListOfRegions);
   TGRegionWithId *region;

   if (fNavMode != kNavRegions) return kTRUE;
   gPointerX = event->fX;
   gPointerY = event->fY;

   while ((region = (TGRegionWithId*)next())) {
      if (region->Contains(gPointerX, gPointerY)) {
         if (fLastVisited == region->GetId()) return kTRUE;
         if (fLastVisited) OnMouseOut(fLastVisited);
         fLastVisited = region->GetId();
         fTip = region->GetToolTipText();
         gCurrentRegion = region;
         OnMouseOver(fLastVisited);
         return kTRUE;
      }
   }

   if (fLastVisited) {
      OnMouseOut(fLastVisited);
      fTip = fMainTip;
   }
   fLastVisited = 0;  // main
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGImageMap::HandleDoubleClick(Event_t *event)
{
   // Handle double click events.

   TIter next(fListOfRegions);
   TGRegionWithId *region;

   if (fTip) fTip->Hide();
   if (event->fCode != kButton1 ) return kTRUE;
   if (fNavMode != kNavRegions) return kTRUE;

   gPointerX = event->fX;
   gPointerY = event->fY;

   while ((region = (TGRegionWithId*)next())) {
      if (region->Contains(gPointerX, gPointerY)) {
         DoubleClicked(region->GetId());
         gCurrentRegion = region;
         return kTRUE;
      }
   }
   DoubleClicked();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGImageMap::HandleButton(Event_t *event)
{
   // Handle button events.

   TIter next(fListOfRegions);
   TGRegionWithId *region;
   TGPopupMenu *pop;

   if (fTip) fTip->Hide();
   if (fNavMode != kNavRegions) return kTRUE;

   gPointerX = event->fX;
   gPointerY = event->fY;

   while ((region = (TGRegionWithId*)next())) {
      if (region->Contains(gPointerX, gPointerY)) {
         gCurrentRegion = region;
         if (event->fType == kButtonPress) {
            if (event->fCode == kButton1 )
               RegionClicked(region->GetId());
            else if (event->fCode == kButton3 ) {
               pop = region->GetPopup();
               if (pop) pop->PlaceMenu(gPointerX, gPointerY, kTRUE, kTRUE);
            }
         }
         return kTRUE;
      }
   }
   if (event->fType == kButtonPress)
      Clicked();
   return kTRUE;
}

//______________________________________________________________________________
void TGImageMap::SetToolTipText(const char *text, Long_t delayms)
{
   // Set tooltip text for main region.

   if (fMainTip) delete fMainTip;
   fMainTip = 0;

   if (text && strlen(text))
      fMainTip = new TGToolTip(fClient->GetDefaultRoot(), this, text, delayms);
}

//______________________________________________________________________________
void TGImageMap::SetToolTipText(Int_t id, const char *text, Long_t delayms)
{
   // Set tooltip text for regions with specified id.

   TIter next(fListOfRegions);
   TGRegionWithId *region;

   while ((region = (TGRegionWithId*)next())) {
      if (id == region->GetId())
         region->SetToolTipText(text, delayms, this);
   }
}

//______________________________________________________________________________
void TGImageMap::OnMouseOver(Int_t id)
{
   // Handle when mouse moves over region id. Emits signal
   // OnMouseOver(Int_t).

   if (fTip) fTip->Reset();
   if (fMainTip) fMainTip->Hide();
   gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(fCursorMouseOver));
   Emit("OnMouseOver(Int_t)", id);
}

//______________________________________________________________________________
void TGImageMap::OnMouseOut(Int_t id)
{
   // Handle when mouse moves from region id. Emits signal
   // OnMouseOut(Int_t).

   if(fTip) fTip->Hide();
   if(fMainTip) fMainTip->Reset();
   gVirtualX->SetCursor(fId,gVirtualX->CreateCursor(fCursorMouseOut));
   Emit("OnMouseOut(Int_t)",id);
}

//______________________________________________________________________________
void TGImageMap::RegionClicked(Int_t id)
{
   // Handle when mouse was clicked on region id. Emits signal
   // RegionClicked(Int_t).

   Emit("RegionClicked(Int_t)",id);
}

//______________________________________________________________________________
void TGImageMap::DoubleClicked()
{
   // Handle when mouse is double clicked on main map. Emits signal
   // DoubleClicked().

   Emit("DoubleClicked()");
}

//______________________________________________________________________________
void TGImageMap::DoubleClicked(Int_t id)
{
   // Handle when mouse is double clicked on region id. Emits signal
   // DoubleClicked(Int_t).

   Emit("DoubleClicked(Int_t)",id);
}
