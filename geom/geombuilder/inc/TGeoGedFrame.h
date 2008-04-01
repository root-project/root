// @(#)root/geombuilder:$Id$
// Author: Matevz Tadel   25/09/2006

#ifndef ROOT_TGeoGedFrame
#define ROOT_TGeoGedFrame

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGeoGedFrame                                                        //
//                                                                      //
//  Common base class for geombuilder editors.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TGTab;
class TGeoTabManager;
class TVirtualPad;

class TGeoGedFrame : public TGedFrame {

protected:
   TGTab          *fTab;           //tab of the ged-editor
   TGeoTabManager *fTabMgr;        //tab manager corresponding to ged-editor
   TVirtualPad    *fPad;           //selected pad, if exists

public:
   TGeoGedFrame(const TGWindow *p = 0,
                Int_t width = 140, Int_t height = 30,
                UInt_t options = kChildFrame,
                Pixel_t back = GetDefaultFrameBackground());

   virtual void SetActive(Bool_t active = kTRUE);
   virtual void Update();

   ClassDef(TGeoGedFrame, 0) // Common base-class for geombuilder editors.
};

#endif
