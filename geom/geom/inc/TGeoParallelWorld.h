/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Authors: Andrei Gheata   30/06/14
//          Sandro Wenzel   01/09/24

#ifndef ROOT_TGeoParallelWorld
#define ROOT_TGeoParallelWorld

#include "TNamed.h"
#include "TGeoVoxelGrid.h"

// forward declarations
class TGeoManager;
class TGeoPhysicalNode;
class TGeoVolume;

class TGeoParallelWorld : public TNamed {

public:
   // internal enum letting choose between
   // VoxelFinder or BVH-based algorithms
   enum class AccelerationMode { kVoxelFinder, kBVH };

   // a structure for safety evaluation (caching) purpose
   // to be stored per 3D grid voxel
   struct SafetyVoxelInfo {
      float min_safety{-1.f}; // the minimum safety from the mid-point of this voxel to any leaf bounding box
      int idx_start{-1}; // the index into an external storage, where candidate bounding boxes to search for this voxel
                         // are stored (if -1) --> VoxelInfo not yet initialized
      unsigned int num_candidates{0}; // the number of candidate bounding boxes to search
      bool isInitialized() const { return idx_start >= 0; }
   };

protected:
   TGeoManager *fGeoManager;     // base geometry
   TObjArray *fPaths;            // array of paths
   Bool_t fUseOverlaps;          // Activated if user defined overlapping candidates
   Bool_t fIsClosed;             //! Closed flag
   TGeoVolume *fVolume;          //! helper volume
   TGeoPhysicalNode *fLastState; //! Last PN touched
   TObjArray *fPhysical;         //! array of physical nodes

   void *fBVH = nullptr; //! BVH helper structure for safety and navigation
   TGeoVoxelGrid<SafetyVoxelInfo> *fSafetyVoxelCache =
      nullptr;                                        //! A regular 3D cache layer for fast point-based safety lookups
   std::vector<unsigned int> fSafetyCandidateStore{}; //! stores bounding boxes serving a quick safety candidates (to be
                                                      //! used with the VoxelGrid and SafetyVoxelInfo)
   void *fBoundingBoxes = nullptr;                    //! to keep the vector of primitive axis aligned bounding boxes
   AccelerationMode fAccMode = AccelerationMode::kVoxelFinder; //! switch between different algorithm implementations

   TGeoParallelWorld(const TGeoParallelWorld &) = delete;
   TGeoParallelWorld &operator=(const TGeoParallelWorld &) = delete;

public:
   // constructors
   TGeoParallelWorld()
      : TNamed(),
        fGeoManager(nullptr),
        fPaths(nullptr),
        fUseOverlaps(kFALSE),
        fIsClosed(kFALSE),
        fVolume(nullptr),
        fLastState(nullptr),
        fPhysical(nullptr)
   {
   }
   TGeoParallelWorld(const char *name, TGeoManager *mgr);

   // destructor
   ~TGeoParallelWorld() override;
   // API for adding components nodes
   void AddNode(const char *path);
   // Activate/deactivate  overlap usage
   void SetUseOverlaps(Bool_t flag) { fUseOverlaps = flag; }
   Bool_t IsUsingOverlaps() const { return fUseOverlaps; }
   void ResetOverlaps() const;
   // Adding overlap candidates can highly improve performance.
   void AddOverlap(TGeoVolume *vol, Bool_t activate = kTRUE);
   void AddOverlap(const char *volname, Bool_t activate = kTRUE);
   // The normal PW mode (without declaring overlaps) does detect them
   Int_t PrintDetectedOverlaps() const;

   // Closing a parallel geometry is mandatory
   Bool_t CloseGeometry();
   // Refresh structures in case of re-alignment
   void RefreshPhysicalNodes();

   // ability to choose algorithm implementation; should be called before CloseGeometry
   void SetAccelerationMode(AccelerationMode const &mode) { fAccMode = mode; }
   AccelerationMode const &GetAccelerationMode() const { return fAccMode; }

   // BVH related functions for building, printing, checking
   void BuildBVH();
   void PrintBVH() const;
   bool CheckBVH(void *, size_t) const;

   // --- main navigation interfaces ----

   // FindNode
   TGeoPhysicalNode *FindNode(Double_t point[3])
   {
      switch (fAccMode) {
      case AccelerationMode::kVoxelFinder: return FindNodeOrig(point);
      case AccelerationMode::kBVH:
         return FindNodeBVH(point);
         // case AccelerationMode::kLoop : return FindNodeLoop(point);
      }
      return nullptr;
   }

   // main interface for Safety
   Double_t Safety(Double_t point[3], Double_t safmax = 1.E30)
   {
      switch (fAccMode) {
      case AccelerationMode::kVoxelFinder: return SafetyOrig(point, safmax);
      case AccelerationMode::kBVH:
         return VoxelSafety(point, safmax);
         // case AccelerationMode::kLoop : return SafetyLoop(point, safmax);
      }
      return 0;
   }

   // main interface for FindNextBoundary
   TGeoPhysicalNode *FindNextBoundary(Double_t point[3], Double_t dir[3], Double_t &step, Double_t stepmax = 1.E30)
   {
      switch (fAccMode) {
      case AccelerationMode::kVoxelFinder: return FindNextBoundaryOrig(point, dir, step, stepmax);
      case AccelerationMode::kBVH:
         return FindNextBoundaryBVH(point, dir, step, stepmax);
         // case AccelerationMode::kLoop : return FindNextBoundaryLoop(point, dir, step, stepmax);
      }
      return nullptr;
   }

   // Getters
   TGeoManager *GetGeometry() const { return fGeoManager; }
   Bool_t IsClosed() const { return fIsClosed; }
   TGeoVolume *GetVolume() const { return fVolume; }

   // Utilities
   void CheckOverlaps(Double_t ovlp = 0.001); // default 10 microns
   void Draw(Option_t *option) override;

private:
   // various implementations for FindNextBoundary
   TGeoPhysicalNode *FindNextBoundaryLoop(Double_t point[3], Double_t dir[3], Double_t &step, Double_t stepmax = 1.E30);
   TGeoPhysicalNode *FindNextBoundaryOrig(Double_t point[3], Double_t dir[3], Double_t &step, Double_t stepmax = 1.E30);
   TGeoPhysicalNode *FindNextBoundaryBVH(Double_t point[3], Double_t dir[3], Double_t &step, Double_t stepmax = 1.E30);

   // various implementations for FindNode
   TGeoPhysicalNode *FindNodeLoop(Double_t point[3]);
   TGeoPhysicalNode *FindNodeOrig(Double_t point[3]);
   TGeoPhysicalNode *FindNodeBVH(Double_t point[3]);

   // various implementations for Safety
   Double_t SafetyLoop(Double_t point[3], Double_t safmax = 1.E30);
   Double_t SafetyBVH(Double_t point[3], Double_t safmax = 1.E30);
   Double_t SafetyOrig(Double_t point[3], Double_t safmax = 1.E30);
   Double_t VoxelSafety(Double_t point[3], Double_t safmax = 1.E30);

   // helper functions related to local safety caching (3D voxel grid)

   // Given a 3D point, returns the
   // a) the min radius R such that there is at least one leaf bounding box fully enclosed
   //    in the sphere of radius R around point
   // b) the set of leaf bounding boxes who partially lie within radius + margin

   // ... using BVH
   std::pair<double, double>
   GetBVHSafetyCandidates(double point[3], std::vector<int> &candidates, double margin = 0.) const;
   // .... same with a simpler, slower algorithm
   std::pair<double, double>
   GetLoopSafetyCandidates(double point[3], std::vector<int> &candidates, double margin = 0.) const;

   void InitSafetyVoxel(TGeoVoxelGridIndex const &);
   void TestVoxelGrid(); // a debug method to play with the voxel grid

   ClassDefOverride(TGeoParallelWorld, 3) // parallel world base class
};

#endif
