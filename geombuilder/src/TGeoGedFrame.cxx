// @(#)root/geombuilder:$Id$
// Author: Matevz Tadel   25/09/2006

//______________________________________________________________________________
//                                                                      //
//  TGeoGedFrame                                                        //
//                                                                      //
//  Common base class for geombuilder editors.                          //
//                                                                      //
//______________________________________________________________________________

#include "TGeoGedFrame.h"
#include "TGeoTabManager.h"
#include "TGedEditor.h"
#include "TGTab.h"
#include "TVirtualPad.h"

ClassImp(TGeoGedFrame)

//______________________________________________________________________________
TGeoGedFrame::TGeoGedFrame(const TGWindow *p, Int_t width, Int_t height,
                           UInt_t options, Pixel_t back) :
  TGedFrame(p, width, height, options, back),
  fTab(0), fTabMgr(0), fPad(0)
{
// Constructor.
   fTab = fGedEditor->GetTab();
   fPad = fGedEditor->GetPad();
   fTabMgr = TGeoTabManager::GetMakeTabManager(fGedEditor);
}

//______________________________________________________________________________
void TGeoGedFrame::SetActive(Bool_t active)
{
   // Set active GUI attribute frames related to the selected object.

   if (active)
      ((TGCompositeFrame*)GetParent())->ShowFrame(this);
   else
      ((TGCompositeFrame*)GetParent())->HideFrame(this);

// no need to call for every single editor Layout of TGMainFrame
//   ((TGMainFrame*)GetMainFrame())->Layout();

   // to avoid that the user changes options on a deactivated Tab
   if (fTab->IsEnabled(fTab->GetCurrent()))
      fTab->SetTab(fTab->GetCurrent());
   else
      fTab->SetTab(0);
}

//______________________________________________________________________________
void TGeoGedFrame::Update()
{
   // Override Update from TGedFrame as fGedEditor can be null.

   if (fGedEditor) {
      fGedEditor->Update();
   } else {
      fPad->Modified();
      fPad->Update();
   }
}
