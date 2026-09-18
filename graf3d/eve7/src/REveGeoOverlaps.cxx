
#include <ROOT/REveGeoOverlaps.hxx>
#include <ROOT/REveGeoShape.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REvePointSet.hxx>
#include <ROOT/REveStraightLineSet.hxx>

#include "TGeoBBox.h"
#include "TGeoManager.h"
#include "TGeoNode.h"
#include "TGeoOverlap.h"
#include "TGeoShape.h"
#include "TGeoVolume.h"
#include "TObjArray.h"
#include "TPolyMarker3D.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <utility>

using namespace ROOT::Experimental;

namespace {

/// Volume plus matrix is what ties a TGeoOverlap to a daughter, since it never
/// names the node itself.
bool SameMatrix(const TGeoMatrix *a, const TGeoMatrix *b)
{
   constexpr double tol = 1e-9;
   if (!a || !b)
      return false;
   const Double_t *ta = a->GetTranslation(), *tb = b->GetTranslation();
   for (int i = 0; i < 3; ++i)
      if (std::fabs(ta[i] - tb[i]) > tol)
         return false;
   const Double_t *ra = a->GetRotationMatrix(), *rb = b->GetRotationMatrix();
   for (int i = 0; i < 9; ++i)
      if (std::fabs(ra[i] - rb[i]) > tol)
         return false;
   return true;
}

bool IsDaughter(TGeoVolume *mom, TGeoVolume *vol, const TGeoMatrix *matr)
{
   if (!mom || !vol || !matr)
      return false;
   for (int d = 0; d < mom->GetNdaughters(); ++d) {
      TGeoNode *node = mom->GetNode(d);
      if (node->GetVolume() == vol && SameMatrix(node->GetMatrix(), matr))
         return true;
   }
   return false;
}

bool IsFinitePoint(const Float_t *p)
{
   return std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]);
}

bool InsideWorld(const Float_t *p, double limit)
{
   if (limit <= 0.)
      return true;
   return std::fabs(p[0]) <= limit && std::fabs(p[1]) <= limit && std::fabs(p[2]) <= limit;
}

/// A representative size for a volume's shape, for scaling a marker to fit
/// next to it. TGeoBBox is the common ancestor of nearly every TGeoShape
/// subclass -- tubes, cones, trapezoids all derive from it and fill in its
/// dx/dy/dz via their own ComputeBBox() -- so this isn't limited to literal
/// boxes. The *smallest* half-extent is what matters: a thin plate a metre
/// wide and a millimetre thick is a millimetre-scale feature, not a metre-scale
/// one, and a cross sized off the wrong dimension would stick out past it.
double ShapeSize(TGeoVolume *vol)
{
   if (!vol || !vol->GetShape())
      return 0.;
   if (auto *bbox = dynamic_cast<TGeoBBox *>(vol->GetShape()))
      return std::min({bbox->GetDX(), bbox->GetDY(), bbox->GetDZ()});
   return 0.;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////

REveGeoOverlapTable::REveGeoOverlapTable(const Text_t *n, const Text_t *t) : REveElement(n, t)
{
   fHolder = new REveElement("OverlapViz", "Shapes of the selected overlap");
}

////////////////////////////////////////////////////////////////////////////////
/// Find every overlap and extrusion in the geometry and work out where each one is.
///
/// One pass over the top volume. CheckOverlaps() is recursive, so checking volumes
/// one at a time would repeat the same work and cost seconds each time.

void REveGeoOverlapTable::ScanOverlaps(TGeoManager *mgr, double precision)
{
   fMgr = mgr;
   fEntries.clear();
   fSelected = -1;
   fPrecision = precision;
   ++fScanGen;

   if (!mgr || !mgr->GetTopVolume())
      return;

   // CheckOverlaps() needs a navigator for whichever thread calls it -- its very
   // first line is SetCheckingOverlaps(), which crashes on a null navigator. The
   // initial scan runs on the main thread, which already has one from loading the
   // geometry; a rescan from SetPrecision() runs on REveManager's MIR-exec thread,
   // which does not.
   if (!mgr->GetCurrentNavigator())
      mgr->AddNavigator();

   mgr->ClearOverlaps();
   mgr->GetTopVolume()->CheckOverlaps(precision);

   TObjArray *lst = mgr->GetListOfOverlaps();
   if (!lst)
      return;

   // Mother candidates per volume. Only genuine overlaps need this -- there both
   // sides are daughters of some common mother; an extrusion names its mother itself.
   std::map<TGeoVolume *, std::vector<TGeoVolume *>> mothers;
   bool mothersBuilt = false;
   auto buildMothers = [&mothers, mgr]() {
      TIter next(mgr->GetListOfVolumes());
      while (auto *vol = static_cast<TGeoVolume *>(next()))
         for (int d = 0; d < vol->GetNdaughters(); ++d)
            mothers[vol->GetNode(d)->GetVolume()].emplace_back(vol);
   };

   // Every TGeoVolume* actually placed somewhere under top. A candidate mother
   // can satisfy IsDaughter() below (right daughters, right matrices) and still
   // be the wrong pick: some geometries (ALICE's does) carry two structurally
   // identical TGeoVolume objects sharing one name, only one of which the real
   // tree ever places -- the other is a registered-but-orphaned duplicate.
   // IsDaughter() alone can't tell them apart since it only ever looks at a
   // candidate's own local daughter list, never at whether the candidate itself
   // is reachable, so that has to be checked separately.
   std::set<TGeoVolume *> reachable;
   bool reachableBuilt = false;
   auto buildReachable = [&reachable, mgr]() {
      TGeoVolume *top = mgr->GetTopVolume();
      reachable.insert(top);
      TGeoIterator iter(top);
      TGeoNode *node = nullptr;
      while ((node = iter.Next()))
         reachable.insert(node->GetVolume());
   };

   // A coarse fuse against numeric garbage. TGeoOverlap's points are in the
   // mother's own (small) local frame -- see LocalizeMarkers() -- so anything
   // legitimate is nowhere near the size of the world; this only exists to catch
   // outright nonsense (CMS has points out at 1e16) before it ever reaches a
   // TGeoShape::Contains()/Safety() call. LocalizeMarkers() does the real,
   // targeted filtering once the mother/daughters are known.
   double worldLimit = 0.;
   if (auto *wbox = dynamic_cast<TGeoBBox *>(mgr->GetTopVolume()->GetShape()))
      worldLimit = 2. * std::max(wbox->GetDX(), std::max(wbox->GetDY(), wbox->GetDZ())) + 100.;

   std::vector<std::vector<float>> rawMarkers; // mother-local points, kept until the mother is known

   for (int i = 0; i < lst->GetEntriesFast(); ++i) {
      auto *ovl = dynamic_cast<TGeoOverlap *>(lst->At(i));
      if (!ovl)
         continue;

      Entry ent;
      ent.fName = ovl->GetName() ? ovl->GetName() : "";
      ent.fTitle = ovl->GetTitle() ? ovl->GetTitle() : "";
      ent.fExtrusion = ovl->IsExtrusion();
      ent.fValue = ovl->GetOverlap();
      ent.fV1 = ovl->GetFirstVolume();
      ent.fV2 = ovl->GetSecondVolume();
      ent.fVol1 = ent.fV1 ? ent.fV1->GetName() : "";
      ent.fVol2 = ent.fV2 ? ent.fV2->GetName() : "";
      if (ovl->GetFirstMatrix())
         ent.fM1 = *ovl->GetFirstMatrix();
      if (ovl->GetSecondMatrix())
         ent.fM2 = *ovl->GetSecondMatrix();

      if (ent.fExtrusion) {
         ent.fMother = ent.fV1;
      } else {
         if (!mothersBuilt) {
            buildMothers();
            mothersBuilt = true;
         }
         if (!reachableBuilt) {
            buildReachable();
            reachableBuilt = true;
         }
         auto it = mothers.find(ent.fV1);
         if (it != mothers.end())
            for (auto *cand : it->second)
               if (reachable.count(cand) && IsDaughter(cand, ent.fV1, ovl->GetFirstMatrix()) &&
                   IsDaughter(cand, ent.fV2, ovl->GetSecondMatrix())) {
                  ent.fMother = cand;
                  break;
               }
      }

      // Without a mother there is no frame to place it in. Drawing it anyway would
      // put it somewhere arbitrary, which is worse than leaving it out.
      if (!ent.fMother)
         continue;

      std::vector<float> pts;
      if (TPolyMarker3D *pm = ovl->GetPolyMarker()) {
         const Float_t *p = pm->GetP();
         ent.fNumRawMarkers = pm->GetN();
         pts.reserve(3 * pm->GetN());
         for (int k = 0; k < pm->GetN(); ++k)
            if (IsFinitePoint(p + 3 * k) && InsideWorld(p + 3 * k, worldLimit)) {
               pts.push_back(p[3 * k]);
               pts.push_back(p[3 * k + 1]);
               pts.push_back(p[3 * k + 2]);
            }
      }

      fEntries.emplace_back(std::move(ent));
      rawMarkers.emplace_back(std::move(pts));
   }

   CollectPlacements(mgr);

   for (size_t i = 0; i < fEntries.size(); ++i)
      LocalizeMarkers(fEntries[i], rawMarkers[i]);

   BuildGroups();

   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Global matrix of every physical placement of each mother volume.
///
/// One walk of the geometry serves all entries; the set of mothers is small (four
/// volumes for the CMS geometry) but finding them needs the full traversal anyway.

void REveGeoOverlapTable::CollectPlacements(TGeoManager *mgr)
{
   std::set<TGeoVolume *> wanted;
   for (auto &ent : fEntries)
      wanted.insert(ent.fMother);

   std::map<TGeoVolume *, std::vector<TGeoHMatrix>> found;

   TGeoVolume *top = mgr->GetTopVolume();
   if (wanted.count(top))
      found[top].emplace_back(TGeoHMatrix()); // identity: the top volume is the world

   TGeoIterator iter(top);
   TGeoNode *node = nullptr;
   while ((node = iter.Next())) {
      TGeoVolume *vol = node->GetVolume();
      if (!wanted.count(vol))
         continue;
      const TGeoMatrix *cur = iter.GetCurrentMatrix();
      found[vol].emplace_back(cur ? TGeoHMatrix(*cur) : TGeoHMatrix());
   }

   for (auto &ent : fEntries) {
      auto it = found.find(ent.fMother);
      if (it != found.end())
         ent.fGlobals = it->second;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Filter the marker points down to the ones that plausibly belong to this flaw.
///
/// TGeoOverlap's TPolyMarker3D is filled in the frame of whichever node
/// CheckOverlaps() was examining -- i.e. already the mother's own local frame,
/// the same one fM1/fM2 are expressed in. There is no placement to guess and no
/// conversion to do: the points already are what they need to be. (An earlier
/// version of this function assumed they were global and ran them through a
/// placement's matrix "to make them local" -- verified directly against a real
/// geometry, that extra step takes a point that sits exactly on a daughter's
/// surface and moves it hundreds of cm away, which is a worse bug than anything
/// it was trying to fix.)
///
/// What *is* still worth doing is dropping outliers: TGeoOverlap's own points
/// can be finite and plausible-looking while sitting nowhere near either
/// daughter, and for a flaw a fraction of a millimetre across that's glaring --
/// the crude "inside the world" fuse applied before this function is far too
/// loose to catch it (it exists only to stop outright numeric garbage, not to
/// judge plausibility). So each point is tested against the two daughter
/// shapes it should be between and dropped if it's near neither.
///
/// The shapes involved can be anything the geometry uses -- a tube, a
/// composite, not just a box -- so the test goes through
/// TGeoShape::Contains()/Safety() rather than assuming a TGeoBBox.

void REveGeoOverlapTable::LocalizeMarkers(Entry &ent, const std::vector<float> &localPts)
{
   ent.fLocalMarkers.clear();
   if (localPts.empty())
      return;

   const size_t npt = localPts.size() / 3;

   // a little slack: the points sit on the offending surface, not strictly inside
   const double margin = 1.0;
   const double farAway = 1.e6; // stands in for "no daughter to test against"

   auto daughterDistance = [](TGeoVolume *vol, const TGeoHMatrix &local, const Double_t *lp) {
      if (!vol || !vol->GetShape())
         return -1.; // no shape to test: not a vote against, just not informative
      Double_t dl[3];
      local.MasterToLocal(lp, dl); // lp is mother-local; bring it into the daughter's own frame
      if (vol->GetShape()->Contains(dl))
         return 0.;
      return vol->GetShape()->Safety(dl, kFALSE);
   };

   // For an extrusion, fV1 *is* the mother -- testing a point against the whole
   // mother would accept almost anything inside it, so only genuine daughters
   // (not the mother itself) are tested.
   bool testV1 = ent.fV1 && ent.fV1 != ent.fMother;
   bool testV2 = ent.fV2 && ent.fV2 != ent.fMother;

   // Distance from lp to the nearer of the two daughters, or 0 if neither is
   // testable (nothing to judge the point against, so don't disqualify it).
   auto distance = [&](const Double_t *lp) {
      if (!testV1 && !testV2)
         return 0.;
      double d1 = testV1 ? daughterDistance(ent.fV1, ent.fM1, lp) : farAway;
      double d2 = testV2 ? daughterDistance(ent.fV2, ent.fM2, lp) : farAway;
      return std::min(d1 < 0. ? farAway : d1, d2 < 0. ? farAway : d2);
   };

   ent.fLocalMarkers.reserve(localPts.size());
   for (size_t k = 0; k < npt; ++k) {
      Double_t lp[3] = {localPts[3 * k], localPts[3 * k + 1], localPts[3 * k + 2]};
      if (distance(lp) > margin)
         continue;
      ent.fLocalMarkers.push_back(localPts[3 * k]);
      ent.fLocalMarkers.push_back(localPts[3 * k + 1]);
      ent.fLocalMarkers.push_back(localPts[3 * k + 2]);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Text shown when the mouse is over one of the drawn pieces.
///
/// The client puts this straight into innerHTML, so <br> works and one element can
/// carry the whole story: which flaw, how big, which two volumes, and -- the part
/// that is otherwise impossible to tell by looking -- which placement out of how
/// many this particular piece is.
///
/// \param which 0 = first volume, 1 = second volume, 2 = the marker points

std::string REveGeoOverlapTable::MakeTip(const Entry &ent, int which, int placement, int inst, int ninst) const
{
   TString t;
   t += TString::Format("<b>%s</b> &middot; %s &middot; <b>%.5f cm</b><br>", ent.fName.c_str(),
                        ent.fExtrusion ? "extrusion" : "overlap", ent.fValue);

   // mark the volume this piece actually is
   t += TString::Format("%sA %s%s<br>", which == 0 ? "<b>" : "", ent.fVol1.c_str(), which == 0 ? "</b>" : "");
   t += TString::Format("%sB %s%s<br>", which == 1 ? "<b>" : "", ent.fVol2.c_str(), which == 1 ? "</b>" : "");

   if (ent.fMother)
      t += TString::Format("in %s<br>", ent.fMother->GetName());

   if (ent.fGlobals.size() > 1)
      t += TString::Format("placement %d of %d<br>", placement + 1, (int)ent.fGlobals.size());

   if (ninst > 1)
      t += TString::Format("instance %d of %d<br>", inst + 1, ninst);

   if (which == 2) {
      int kept = (int)(ent.fLocalMarkers.size() / 3);
      int dropped = ent.fNumRawMarkers - kept;
      if (dropped > 0)
         t += TString::Format("%d points (%d dropped as out of range)", kept, dropped);
      else
         t += TString::Format("%d points", kept);
   }

   return std::string(t.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Draw one placement of the selected overlap: the two shapes and the points.

void REveGeoOverlapTable::AddPlacement(const Entry &ent, const TGeoHMatrix &global, int placement,
                                       std::set<std::string> *seen, int inst, int ninst)
{
   auto addShape = [&](TGeoVolume *vol, const TGeoHMatrix &local, Color_t col, const char *tag, int which) {
      if (!vol || !vol->GetShape())
         return;
      TGeoHMatrix m = global;
      m.Multiply(&local);

      // Drawing a whole group would otherwise stack 36 identical copies of the
      // shared mother on top of each other -- same volume, same matrix, no extra
      // information and a much heavier scene.
      if (seen) {
         const Double_t *t = m.GetTranslation(), *r = m.GetRotationMatrix();
         char key[512];
         int n = snprintf(key, sizeof(key), "%p|%.4f,%.4f,%.4f", (void *)vol, t[0], t[1], t[2]);
         for (int i = 0; i < 9 && n < (int)sizeof(key); ++i)
            n += snprintf(key + n, sizeof(key) - n, ",%.6f", r[i]);
         if (!seen->insert(key).second)
            return;
      }
      // REveGeoShape uses TGeoShape::UniqueID as a reference count and deletes the
      // shape when it drops to zero. The shapes here belong to the TGeoManager and
      // do not take part in that scheme, so handing one over directly means the
      // geometry's own shape is deleted as soon as these elements are destroyed --
      // which happens on the very next selection. Give each element a private clone.
      auto *geoShape = static_cast<TGeoShape *>(vol->GetShape()->Clone());
      if (!geoShape)
         return;
      geoShape->SetUniqueID(0);

      auto *shape = new REveGeoShape(TString::Format("%s_%s_%s", ent.fName.c_str(), tag, vol->GetName()).Data(),
                                     MakeTip(ent, which, placement, inst, ninst).c_str());
      shape->SetShape(geoShape);
      shape->SetTransMatrix(m);
      shape->SetMainColor(col);
      shape->SetMainTransparency(50);
      // fPickable defaults to false, and without it the element never enters the
      // pick pass at all -- no highlight and no tooltip, with nothing to say why.
      shape->SetPickable(kTRUE);
      fHolder->AddElement(shape);
   };

   addShape(ent.fV1, ent.fM1, kRed, "A", 0);
   addShape(ent.fV2, ent.fM2, kYellow, "B", 1);

   if (ent.fLocalMarkers.empty())
      return;

   const size_t npt = ent.fLocalMarkers.size() / 3;
   auto *ps = new REvePointSet(TString::Format("%s_points", ent.fName.c_str()).Data(),
                               MakeTip(ent, 2, placement, inst, ninst).c_str(), npt);
   ps->SetMarkerColor(kCyan);
   ps->SetMarkerSize(fMarkerSize);
   ps->SetPickable(kTRUE);
   ps->SetMarkerStyle(4);

   // A GL point can vanish behind a transparent volume depending on draw order,
   // so give every marker a second, harder-to-hide representation: a 3D cross,
   // three lines through the point along x/y/z.
   auto *cross = new REveStraightLineSet(TString::Format("%s_cross", ent.fName.c_str()).Data(),
                                         MakeTip(ent, 2, placement, inst, ninst).c_str());
   cross->SetMainColor(kCyan);
   cross->SetLineWidth(2);
   cross->SetPickable(kTRUE);

   // Scaled to the smaller of the two daughters, not a flat constant: entries in
   // this table range from a 33 cm overlap down to a 0.02 cm one, and a fixed
   // arm length is either invisible against the big one or many times the size
   // of the small one. fCrossSize remains the cap, in cm, it always was --
   // still tunable via SetCrossSize() -- but the arm only uses that much of it
   // when the smaller shape is actually big enough to warrant it; otherwise it
   // scales down to a fraction of that shape instead of overshooting it.
   constexpr double kCrossFraction = 0.3; // arm half-length as a fraction of the smaller daughter's thinnest dimension
   double s1 = ShapeSize(ent.fV1), s2 = ShapeSize(ent.fV2);
   double smallest = (s1 > 0. && s2 > 0.) ? std::min(s1, s2) : std::max(s1, s2);
   double proportional = smallest > 0. ? kCrossFraction * smallest : (double)fCrossSize;
   const float h = (float)std::min(proportional, (double)fCrossSize);

   for (size_t k = 0; k < npt; ++k) {
      Double_t lp[3] = {ent.fLocalMarkers[3 * k], ent.fLocalMarkers[3 * k + 1], ent.fLocalMarkers[3 * k + 2]}, gp[3];
      global.LocalToMaster(lp, gp);
      ps->SetNextPoint((float)gp[0], (float)gp[1], (float)gp[2]);

      float x = (float)gp[0], y = (float)gp[1], z = (float)gp[2];
      cross->AddLine(x - h, y, z, x + h, y, z);
      cross->AddLine(x, y - h, z, x, y + h, z);
      cross->AddLine(x, y, z - h, x, y, z + h);
   }
   fHolder->AddElement(ps);
   fHolder->AddElement(cross);
}

////////////////////////////////////////////////////////////////////////////////
/// Show one overlap and nothing else. Called from the client as a MIR when a row
/// is picked; a negative index just clears the view.

void REveGeoOverlapTable::SelectOverlap(int idx)
{
   fHolder->DestroyElements();

   fSelected = ((idx >= 0) && (idx < (int)fEntries.size())) ? idx : -1;
   fSelectedGroup = -1;

   if (fSelected >= 0) {
      const Entry &ent = fEntries[fSelected];
      for (size_t p = 0; p < ent.fGlobals.size(); ++p)
         AddPlacement(ent, ent.fGlobals[p], (int)p);
   }

   fHolder->StampObjProps();
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Collapse entries that are the same flaw repeated.
///
/// Two entries belong together when they are the same kind of problem between the
/// same pair of logical volumes. The value is the same across a group in practice,
/// but the largest is kept rather than assumed.

void REveGeoOverlapTable::BuildGroups()
{
   fGroups.clear();
   std::map<std::string, int> index;

   for (size_t i = 0; i < fEntries.size(); ++i) {
      const Entry &e = fEntries[i];
      std::string key = e.fVol1 + "|" + e.fVol2 + "|" + (e.fExtrusion ? "e" : "o");
      auto it = index.find(key);
      if (it == index.end()) {
         index[key] = (int)fGroups.size();
         Group g;
         g.fVol1 = e.fVol1;
         g.fVol2 = e.fVol2;
         g.fExtrusion = e.fExtrusion;
         g.fValue = e.fValue;
         g.fMembers.emplace_back((int)i);
         fGroups.emplace_back(std::move(g));
      } else {
         Group &g = fGroups[it->second];
         g.fMembers.emplace_back((int)i);
         if (e.fValue > g.fValue)
            g.fValue = e.fValue;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Show every instance of one kind of flaw at once.
///
/// Seeing all 36 supermodule extrusions together is the point of the grouped view:
/// it says at a glance whether the problem is systematic or confined to a few
/// placements. Shapes are de-duplicated, so the shared mother is drawn once.

void REveGeoOverlapTable::SelectUniqueOverlap(int gidx)
{
   fHolder->DestroyElements();
   fSelected = -1;
   fSelectedGroup = ((gidx >= 0) && (gidx < (int)fGroups.size())) ? gidx : -1;

   if (fSelectedGroup >= 0) {
      std::set<std::string> seen;
      const auto &mem = fGroups[fSelectedGroup].fMembers;
      for (size_t k = 0; k < mem.size(); ++k) {
         const Entry &ent = fEntries[mem[k]];
         for (size_t p = 0; p < ent.fGlobals.size(); ++p)
            AddPlacement(ent, ent.fGlobals[p], (int)p, &seen, (int)k, (int)mem.size());
      }
   }

   fHolder->StampObjProps();
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Re-run whichever selection is current, so a change of appearance takes effect
/// without the client having to re-pick the row.

void REveGeoOverlapTable::Redraw()
{
   if (fSelectedGroup >= 0)
      SelectUniqueOverlap(fSelectedGroup);
   else if (fSelected >= 0)
      SelectOverlap(fSelected);
}

////////////////////////////////////////////////////////////////////////////////
/// Marker size for the overlap points. The points are the small thing in a scene
/// full of large volumes, so this wants to be generous.

void REveGeoOverlapTable::SetMarkerSize(float size)
{
   if (size <= 0.f)
      return;
   fMarkerSize = size;
   Redraw();
}

////////////////////////////////////////////////////////////////////////////////
/// Cap, in cm, on the half-length of each marker's 3D-cross arm. AddPlacement()
/// scales the actual arm to a fraction of the smaller daughter's size, so this
/// only bites when that would otherwise exceed it.

void REveGeoOverlapTable::SetCrossSize(float size)
{
   if (size <= 0.f)
      return;
   fCrossSize = size;
   Redraw();
}

////////////////////////////////////////////////////////////////////////////////
/// Re-run the overlap check at a new precision and rebuild the table from
/// scratch. Whatever was selected no longer means anything -- ScanOverlaps()
/// rebuilds fEntries/fGroups in a new order with new indices -- so the 3D view
/// is cleared rather than left showing a stale selection.

void REveGeoOverlapTable::SetPrecision(double precision)
{
   if (precision <= 0. || !fMgr)
      return;

   fHolder->DestroyElements();
   fSelectedGroup = -1;
   ScanOverlaps(fMgr, precision); // clears fSelected, resets fPrecision, stamps object props
}

////////////////////////////////////////////////////////////////////////////////
/// Print one overlap's details to the server console: name, volumes, mother,
/// TGeoOverlap's own title (usually the pair of node paths), every placement's
/// global translation, and how many marker points survived filtering. For
/// debugging from the client's row context menu -- this doesn't touch the 3D
/// view or the selection.

void REveGeoOverlapTable::PrintOverlap(int idx)
{
 
   if ((idx < 0) || (idx >= (int)fEntries.size())) {
      printf("REveGeoOverlapTable::PrintOverlap: index %d out of range (%d entries)\n", idx, (int)fEntries.size());
      fflush(stdout);
      return;
   }

   const Entry &e = fEntries[idx];
   printf("=== %s : %s : %.6f cm ===\n", e.fName.c_str(), e.fExtrusion ? "extrusion" : "overlap", e.fValue);
   printf("  A: %s\n", e.fVol1.c_str());
   printf("  B: %s\n", e.fVol2.c_str());
   printf("  mother: %s\n", e.fMother ? e.fMother->GetName() : "(none)");
   if (!e.fTitle.empty())
      printf("  title: %s\n", e.fTitle.c_str());
   printf("  placements: %d\n", (int)e.fGlobals.size());
   for (size_t p = 0; p < e.fGlobals.size(); ++p) {
      const Double_t *t = e.fGlobals[p].GetTranslation();
      printf("    [%d] translation = (%.4f, %.4f, %.4f)\n", (int)p, t[0], t[1], t[2]);
   }
   int kept = (int)(e.fLocalMarkers.size() / 3);
   printf("  points: %d raw, %d kept (%d dropped)\n", e.fNumRawMarkers, kept, e.fNumRawMarkers - kept);
   // stdout is fully buffered when it isn't a tty (the normal case for a
   // long-running web server process), so without this the output just sits in
   // the buffer indefinitely instead of reaching the terminal.
   fflush(stdout);
}

////////////////////////////////////////////////////////////////////////////////
/// Print a whole group -- every entry sharing this (vol1, vol2, extrusion) --
/// to the server console, as one PrintOverlap() block per member.

void REveGeoOverlapTable::PrintUniqueOverlap(int gidx)
{
   if ((gidx < 0) || (gidx >= (int)fGroups.size())) {
      printf("REveGeoOverlapTable::PrintUniqueOverlap: index %d out of range (%d groups)\n", gidx, (int)fGroups.size());
      fflush(stdout);
      return;
   }

   const Group &g = fGroups[gidx];
   printf("=== group: %s | %s | %s (%d members) ===\n", g.fVol1.c_str(), g.fVol2.c_str(),
          g.fExtrusion ? "extrusion" : "overlap", (int)g.fMembers.size());
   for (int ei : g.fMembers)
      PrintOverlap(ei);
}

////////////////////////////////////////////////////////////////////////////////

Int_t REveGeoOverlapTable::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   Int_t ret = REveElement::WriteCoreJson(j, rnr_offset);

   j["fPrecision"] = fPrecision;
   j["fScanGen"] = fScanGen;
   j["fSelected"] = fSelected;

   nlohmann::json arr = nlohmann::json::array();
   for (size_t i = 0; i < fEntries.size(); ++i) {
      const Entry &e = fEntries[i];
      nlohmann::json o = nlohmann::json::object();
      o["idx"] = (int)i;
      o["name"] = e.fName;
      o["kind"] = e.fExtrusion ? "extrusion" : "overlap";
      o["value"] = e.fValue;
      o["vol1"] = e.fVol1;
      o["vol2"] = e.fVol2;
      o["mother"] = e.fMother ? e.fMother->GetName() : "";
      o["nplaced"] = (int)e.fGlobals.size();
      o["npoints"] = (int)(e.fLocalMarkers.size() / 3);
      o["ndropped"] = e.fNumRawMarkers - (int)(e.fLocalMarkers.size() / 3);
      o["title"] = e.fTitle;
      arr.emplace_back(std::move(o));
   }
   j["fOverlaps"] = std::move(arr);

   nlohmann::json garr = nlohmann::json::array();
   for (size_t i = 0; i < fGroups.size(); ++i) {
      const Group &g = fGroups[i];
      int npts = 0;
      for (int ei : g.fMembers)
         npts += (int)(fEntries[ei].fLocalMarkers.size() / 3);
      nlohmann::json o = nlohmann::json::object();
      o["idx"] = (int)i;
      o["kind"] = g.fExtrusion ? "extrusion" : "overlap";
      o["value"] = g.fValue;
      o["vol1"] = g.fVol1;
      o["vol2"] = g.fVol2;
      o["ninst"] = (int)g.fMembers.size();
      o["npoints"] = npts;
      o["name"] = g.fMembers.empty() ? "" : fEntries[g.fMembers.front()].fName;
      garr.emplace_back(std::move(o));
   }
   j["fGroups"] = std::move(garr);
   j["fSelectedGroup"] = fSelectedGroup;
   j["fMarkerSize"] = fMarkerSize;
   j["fCrossSize"] = fCrossSize;

   return ret;
}
