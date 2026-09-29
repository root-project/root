// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveProjectionAxis
#define ROOT7_REveProjectionAxis

#include <ROOT/REveText.hxx>

#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {

class REveProjectionManager;


class REveProjectionAxis : public REveText
{
public:
   /// Where the labelled values come from.
   ///   kValue    round numbers in original space, unevenly spaced on screen
   ///   kPosition evenly spaced on screen, irregular label values
   enum ELabMode_e { kValue = 0, kPosition };

   /// Which edges of the viewport carry an axis.
   enum EAxesMode_e { kHorizontal = 0, kVertical, kAll };

   /// One tick: where it lands in projected space, what to write, and whether it
   /// is a labelled (major) tick or a bare subdivision.
   struct Tick_t {
      Float_t     fPos{0};      ///< position along the axis, in projected coordinates
      std::string fLabel;       ///< empty for a minor tick
      Bool_t      fMajor{true};
   };

private:
   REveProjectionAxis(const REveProjectionAxis &) = delete;
   REveProjectionAxis &operator=(const REveProjectionAxis &) = delete;

protected:
   REveProjectionManager *fManager{nullptr}; ///<! held as an aunt, not owned

   ELabMode_e  fLabMode{kValue};
   EAxesMode_e fAxesMode{kAll};

   /// Primary divisions * 100 + secondary divisions, decoded as
   /// TEveProjectionAxes does, the reverse of TAttAxis. Finer than usual
   /// because the client drops labels that do not fit.
   Int_t   fNdivisions{1010};
   Float_t fRangeFactor{2.0};  ///< tick range as a multiple of the manager's bbox extent, about its centre
   /// Take the viewer's foreground colour instead of fTextColor/fLineColor, so
   /// the axis stays legible when the background changes. On by default.
   Bool_t  fUseFgColor{kTRUE};
   Bool_t  fDrawCenter{kFALSE};
   Bool_t  fDrawOrigin{kFALSE};

   std::vector<Tick_t> fTicks[2]; ///<! computed ticks, [0] horizontal, [1] vertical

   REveText *fDistLabel{nullptr}; ///<! optional read-out of the distortion, not owned

   void BuildTicks(Int_t ax);

public:
   REveProjectionAxis(REveProjectionManager *m, const Text_t *n = "REveProjectionAxis",
                      const Text_t *t = "");
   ~REveProjectionAxis() override;

   REveProjectionManager *GetManager() const { return fManager; }

   ELabMode_e GetLabMode() const { return fLabMode; }
   void SetLabMode(ELabMode_e m) { fLabMode = m; StampObjProps(); }

   EAxesMode_e GetAxesMode() const { return fAxesMode; }
   void SetAxesMode(EAxesMode_e m) { fAxesMode = m; StampObjProps(); }

   Int_t GetNdivisions() const { return fNdivisions; }
   void SetNdivisions(Int_t n) { fNdivisions = n; StampObjProps(); }

   Float_t GetRangeFactor() const { return fRangeFactor; }
   void SetRangeFactor(Float_t f) { fRangeFactor = f; StampObjProps(); }

   Bool_t GetUseFgColor() const { return fUseFgColor; }
   void SetUseFgColor(Bool_t x) { fUseFgColor = x; StampObjProps(); }

   Bool_t GetDrawCenter() const { return fDrawCenter; }
   void SetDrawCenter(Bool_t x) { fDrawCenter = x; StampObjProps(); }

   Bool_t GetDrawOrigin() const { return fDrawOrigin; }
   void SetDrawOrigin(Bool_t x) { fDrawOrigin = x; StampObjProps(); }

   void UpdateTicks();

   void SetDistortionLabel(REveText *t);
   void RemoveAunt(REveAunt *au) override;
   void UpdateDistortionLabel();

   Int_t WriteCoreJson(nlohmann::json &j, Int_t rnr_offset) override;
   void BuildRenderData() override;
};

} // namespace Experimental
} // namespace ROOT

#endif
