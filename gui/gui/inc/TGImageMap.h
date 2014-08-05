// @(#)root/gui:$Id$
// Author: Valeriy Onuchin & Fons Rademakers   18/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGImageMap
#define ROOT_TGImageMap

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGImageMap (with TGRegion and TGRegionWithId help classes)           //
//                                                                      //
// A TGImageMap provides the functionality like a clickable image in    //
// a web browser with sensitive regions (MAP HTML tag).                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGButton.h"
#include "TPoint.h"
#include "TGDimension.h"


class TGRegionData;
class TGPopupMenu;
class TGToolTip;
class TArrayS;


class TGRegion : public TObject {

protected:
   TGRegionData   *fData;  // data describing region

   TGRegion(Bool_t);
   TGRegion CopyRegion() const;

public:
   enum ERegionType { kRectangle, kEllipse };

   TGRegion();
   TGRegion(Int_t x, Int_t y, UInt_t w, UInt_t h, ERegionType = kRectangle);
   TGRegion(Int_t n, TPoint *points, Bool_t winding = kFALSE);
   TGRegion(Int_t n, Int_t *x, Int_t *y, Bool_t winding = kFALSE);
   TGRegion(const TArrayS &x, const TArrayS &y, Bool_t winding = kFALSE);
   TGRegion(const TGRegion &reg);
   virtual ~TGRegion();

   Bool_t      Contains(const TPoint &p) const;
   Bool_t      Contains(Int_t x, Int_t y) const;
   TGRegion    Unite(const TGRegion &r) const;
   TGRegion    Intersect(const TGRegion &r) const;
   TGRegion    Subtract(const TGRegion &r) const;
   TGRegion    Eor(const TGRegion &r) const;
   TGDimension GetDimension() const;
   TGPosition  GetPosition() const;
   Bool_t      IsNull() const;
   Bool_t      IsEmpty() const;

   TGRegion operator|(const TGRegion &r) const { return Unite(r); }
   TGRegion operator+(const TGRegion &r) const { return Unite(r); }
   TGRegion operator&(const TGRegion &r) const { return Intersect(r); }
   TGRegion operator-(const TGRegion &r) const { return Subtract(r); }
   TGRegion operator^(const TGRegion &r) const { return Eor(r); }
   TGRegion& operator|=(const TGRegion &r) { return *this = *this | r; }
   TGRegion& operator+=(const TGRegion &r) { return *this = *this + r; }
   TGRegion& operator&=(const TGRegion &r) { return *this = *this & r; }
   TGRegion& operator-=(const TGRegion &r) { return *this = *this - r; }
   TGRegion& operator^=(const TGRegion &r) { return *this = *this ^ r; }
   Bool_t operator==(const TGRegion &r)  const;
   Bool_t operator!=(const TGRegion &r) const { return !(operator==(r)); }
   TGRegion &operator=(const TGRegion &r);

   ClassDef(TGRegion,0) // Describes a region
};


class TGRegionWithId : public TGRegion {

private:

   TGRegionWithId& operator=(const TGRegionWithId&); // Not implemented

protected:
   Int_t         fId;      // region id
   TGToolTip    *fTip;     // tooltip
   TGPopupMenu  *fPopup;   // popup menu

public:
   TGRegionWithId();
   TGRegionWithId(Int_t id, Int_t x, Int_t y, UInt_t w, UInt_t h,
                  ERegionType = kRectangle);
   TGRegionWithId(Int_t id, Int_t n, TPoint *points, Bool_t winding = kFALSE);
   TGRegionWithId(const TGRegionWithId &reg);
   TGRegionWithId(const TGRegion &reg, Int_t id);
   virtual ~TGRegionWithId();

   Int_t        GetId() const { return fId; }
   TGToolTip   *GetToolTipText() const { return fTip; }
   void         SetToolTipText(const char *text, Long_t delayms,
                               const TGFrame *frame);
   TGPopupMenu *GetPopup() const { return fPopup; }
   void         SetPopup(TGPopupMenu *popup) { fPopup = popup; }
   void         DisplayPopup();

   ClassDef(TGRegionWithId,0) // Region with id, tooltip text and popup menu
};


class TGImageMap : public TGPictureButton {

private:

   TGImageMap(const TGImageMap&); // Not implemented
   TGImageMap& operator=(const TGImageMap&); // Not implemented

public:
   enum ENavMode { kNavRegions, kNavGrid };

protected:
   TList      *fListOfRegions;   // list of regions
   ENavMode    fNavMode;         // navigation mode
   ECursor     fCursorMouseOver; // cursor shape in regions
   ECursor     fCursorMouseOut;  // cursor shape out of regions
   Int_t       fLastVisited;     // id of the last visited region
   TGToolTip  *fMainTip;         // tooltip text for main region
   TList      *fTrash;           // collect all objects that need to be cleaned up

public:
   TGImageMap(const TGWindow *p = 0, const TGPicture *pic = 0);
   TGImageMap(const TGWindow *p, const TString &pic);
   virtual ~TGImageMap();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleDoubleClick(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);

   ENavMode       GetNavMode() { return fNavMode; }
   void           AddRegion(const TGRegion &region, Int_t id);
   TGPopupMenu   *CreatePopup(Int_t id);
   TGPopupMenu   *GetPopup(Int_t id);

   void SetToolTipText(const char *text, Long_t delayms = 300);
   void SetToolTipText(Int_t id, const char *text, Long_t delayms = 300);
   void SetCursor(ECursor cursor = kHand) { fCursorMouseOver = cursor; }
   void SetPicture(const TGPicture * /*new_pic*/) { } // disabled

   virtual void RegionClicked(Int_t id); // *SIGNAL*
   virtual void DoubleClicked(Int_t id); // *SIGNAL*
   virtual void DoubleClicked();         // *SIGNAL*
   virtual void OnMouseOver(Int_t id);   // *SIGNAL*
   virtual void OnMouseOut(Int_t id);    // *SIGNAL*

   ClassDef(TGImageMap,0)  // Clickable image (like MAP in HTML)
};

R__EXTERN TGRegionWithId *gCurrentRegion;

#endif
