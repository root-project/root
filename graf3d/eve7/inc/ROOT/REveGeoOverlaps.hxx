
#ifndef ROOT7_REveGeoOverlaps
#define ROOT7_REveGeoOverlaps

#include <ROOT/REveElement.hxx>

#include "TGeoMatrix.h"

#include <set>
#include <string>
#include <vector>

class TGeoManager;
class TGeoVolume;

namespace ROOT {
namespace Experimental {

////////////////////////////////////////////////////////////////////////////////
/// \class REveGeoOverlapTable
/// \brief The overlaps and extrusions of a geometry, as a flat list plus a 3D view.
///
/// Server side of the overlap browser. ScanOverlaps() runs TGeoVolume::CheckOverlaps()
/// once over the whole geometry and turns the resulting TGeoOverlap objects into
/// something a client can display; the list is streamed with the element and the
/// client picks one row, which comes back as a SelectOverlap() MIR.
///
/// Three things TGeoOverlap does not hand over and that this class works out:
///
///  - **which placement an overlap belongs to.** TGeoOverlap names two volumes and
///    gives their matrices but never a TGeoNode. The matrices are in the frame of
///    the mother volume whose daughters were checked, so (volume, matrix) pins the
///    daughter down exactly.
///  - **where that is in the detector.** An overlap is a property of a *logical*
///    volume, so it exists at every physical placement of that volume -- possibly
///    several. Each entry therefore carries the global matrix of every placement,
///    and the 3D view draws the pair at all of them.
///  - **which points are usable.** The TPolyMarker3D that TGeoOverlap fills is in
///    the mother's own local frame -- the same one the matrices above use, not
///    global coordinates, despite how it looks at first glance -- and some points
///    come out non-finite, or finite but nowhere near either daughter. Left in,
///    the bad ones give the point set an unbounded bbox (the camera has nothing
///    to frame) or just draw markers in the wrong place; both are handled here.

class REveGeoOverlapTable : public REveElement {
public:
   /// One kind of flaw, with every placement of it that the geometry contains.
   /// The CMS geometry reports the EBAR/ESPM extrusion 36 times, once per
   /// supermodule; grouping turns 71 rows into 36 distinct problems.
   struct Group {
      std::string fVol1, fVol2;
      bool fExtrusion{true};
      double fValue{0.};              ///< largest value among the members
      std::vector<int> fMembers;      ///< indices into the entry list
   };

   struct Entry {
      std::string fName;                    ///< TGeoOverlap name, e.g. ov00035
      std::string fTitle;                   ///< human-readable description
      std::string fVol1, fVol2;             ///< names of the two volumes
      bool fExtrusion{true};                ///< extrusion, as opposed to a real overlap
      double fValue{0.};                    ///< overlap/extrusion size, in cm

      TGeoVolume *fV1{nullptr};             ///<! first volume; the mother, for an extrusion
      TGeoVolume *fV2{nullptr};             ///<! second volume
      TGeoVolume *fMother{nullptr};         ///<! volume both matrices are expressed in
      TGeoHMatrix fM1, fM2;                 ///<! placements, in the mother's local frame

      std::vector<TGeoHMatrix> fGlobals;    ///<! global matrix of every placement of the mother
      std::vector<float> fLocalMarkers;     ///<! marker points, in the mother's frame, finite only
      int fNumRawMarkers{0};                ///< points the overlap carried, before filtering
   };

private:
   REveGeoOverlapTable(const REveGeoOverlapTable &) = delete;
   REveGeoOverlapTable &operator=(const REveGeoOverlapTable &) = delete;

   std::vector<Entry> fEntries;
   std::vector<Group> fGroups;
   REveElement *fHolder{nullptr};   ///<! where the selected overlap is drawn; put this in a 3D scene
   TGeoManager *fMgr{nullptr};      ///<! kept from ScanOverlaps() so SetPrecision() can re-run it
   int fSelected{-1};
   int fSelectedGroup{-1};
   double fPrecision{0.001};
   int fScanGen{0};                 ///< bumped by ScanOverlaps(); lets the client tell a rescan from an ordinary selection stamp
   float fMarkerSize{8.f};
   float fCrossSize{15.f}; ///< cap, in cm, on the half-length of each marker's 3D-cross arm; AddPlacement()
                           ///< scales the actual arm to the smaller daughter's size and never exceeds this

   void CollectPlacements(TGeoManager *mgr);
   void LocalizeMarkers(Entry &ent, const std::vector<float> &localPts);
   void AddPlacement(const Entry &ent, const TGeoHMatrix &global, int placement, std::set<std::string> *seen = nullptr,
                     int inst = -1, int ninst = 0);
   std::string MakeTip(const Entry &ent, int which, int placement, int inst, int ninst) const;
   void BuildGroups();
   void Redraw();

public:
   REveGeoOverlapTable(const Text_t *n = "Overlaps", const Text_t *t = "");
   ~REveGeoOverlapTable() override {}

   void ScanOverlaps(TGeoManager *mgr, double precision = 0.001);

   /** Element carrying the shapes of the selected overlap. Add it to a 3D scene. */
   REveElement *GetVizHolder() { return fHolder; }

   int GetNumOverlaps() const { return (int)fEntries.size(); }
   int GetNumGroups() const { return (int)fGroups.size(); }
   int GetSelected() const { return fSelected; }
   double GetPrecision() const { return fPrecision; }
   float GetMarkerSize() const { return fMarkerSize; }
   float GetCrossSize() const { return fCrossSize; }

   // reachable from the client as MIRs
   void SelectOverlap(int idx);
   void SelectUniqueOverlap(int gidx);
   void SetMarkerSize(float size);
   void SetCrossSize(float size);
   void SetPrecision(double precision);
   void PrintOverlap(int idx);
   void PrintUniqueOverlap(int gidx);

   Int_t WriteCoreJson(nlohmann::json &j, Int_t rnr_offset) override;
};

} // namespace Experimental
} // namespace ROOT

#endif
