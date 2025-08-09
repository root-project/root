/// \author Sandro Wenzel <sandro.wenzel@cern.ch>
/// \date 2024-02-22

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoVoxelGrid
#define ROOT_TGeoVoxelGrid

#include <array>
#include <cmath>
#include <limits>

// a simple structure to encode voxel indices, to address
// individual voxels in the 3D grid.
struct TGeoVoxelGridIndex {
   int ix{-1};
   int iy{-1};
   int iz{-1};
   size_t idx{std::numeric_limits<size_t>::max()};
   bool isValid() const { return idx != std::numeric_limits<size_t>::max(); }
};

/// A finite 3D grid structure, mapping/binning arbitrary 3D cartesian points
/// onto discrete "voxels". Each such voxel can store an object of type T.
/// The precision of the voxel binning is done with S (float or double).
template <typename T, typename S = float>
class TGeoVoxelGrid {
public:
   TGeoVoxelGrid(S xmin, S ymin, S zmin, S xmax, S ymax, S zmax, S Lx_, S Ly_, S Lz_)
      : fMinBound{xmin, ymin, zmin}, fMaxBound{xmax, ymax, zmax}, fLx(Lx_), fLy(Ly_), fLz(Lz_)
   {

      // Calculate the number of voxels in each dimension
      fNx = static_cast<int>((fMaxBound[0] - fMinBound[0]) / fLx);
      fNy = static_cast<int>((fMaxBound[1] - fMinBound[1]) / fLy);
      fNz = static_cast<int>((fMaxBound[2] - fMinBound[2]) / fLz);

      finvLx = 1. / fLx;
      finvLy = 1. / fLy;
      finvLz = 1. / fLz;

      fHalfDiag = std::sqrt(fLx / 2. * fLx / 2. + fLy / 2. * fLy / 2. + fLz / 2. * fLz / 2.);

      // Resize the grid to hold the voxels
      fGrid.resize(fNx * fNy * fNz);
   }

   T &at(int i, int j, int k) { return fGrid[index(i, j, k)]; }

   // check if point is covered by voxel structure
   bool inside(std::array<S, 3> const &p) const
   {
      for (int i = 0; i < 3; ++i) {
         if (p[i] < fMinBound[i] || p[i] > fMaxBound[i]) {
            return false;
         }
      }
      return true;
   }

   // Access a voxel given a 3D point P
   T &at(std::array<S, 3> const &P)
   {
      int i, j, k;
      pointToVoxelIndex(P, i, j, k); // Convert point to voxel index
      return fGrid[index(i, j, k)];  // Return reference to voxel's data
   }

   T *at(TGeoVoxelGridIndex const &vi)
   {
      if (!vi.isValid()) {
         return nullptr;
      }
      return &fGrid[vi.idx];
   }

   // Set the data of a voxel at point P
   void set(std::array<S, 3> const &p, const T &value)
   {
      int i, j, k;
      pointToVoxelIndex(p, i, j, k); // Convert point to voxel index
      fGrid[index(i, j, k)] = value; // Set the value at the voxel
   }

   // Set the data of a voxel at point P
   void set(int i, int j, int k, const T &value)
   {
      fGrid[index(i, j, k)] = value; // Set the value at the voxel
   }

   void set(TGeoVoxelGridIndex const &vi, const T &value) { fGrid[vi.idx] = value; }

   // Get voxel dimensions
   int getVoxelCountX() const { return fNx; }
   int getVoxelCountY() const { return fNy; }
   int getVoxelCountZ() const { return fNz; }

   // returns the cartesian mid-point coordinates of a voxel given by a VoxelIndex
   std::array<S, 3> getVoxelMidpoint(TGeoVoxelGridIndex const &vi) const
   {
      const S midX = fMinBound[0] + (vi.ix + 0.5) * fLx;
      const S midY = fMinBound[1] + (vi.iy + 0.5) * fLy;
      const S midZ = fMinBound[2] + (vi.iz + 0.5) * fLz;

      return {midX, midY, midZ};
   }

   S getDiagonalLength() const { return fHalfDiag; }

   // Convert a point p(x, y, z) to voxel indices (i, j, k)
   // if point is outside set indices i,j,k to -1
   void pointToVoxelIndex(std::array<S, 3> const &p, int &i, int &j, int &k) const
   {
      if (!inside(p)) {
         i = -1;
         j = -1;
         k = -1;
      }

      i = static_cast<int>((p[0] - fMinBound[0]) * finvLx);
      j = static_cast<int>((p[1] - fMinBound[1]) * finvLy);
      k = static_cast<int>((p[2] - fMinBound[2]) * finvLz);

      // Clamp the indices to valid ranges
      i = std::min(i, fNx - 1);
      j = std::min(j, fNy - 1);
      k = std::min(k, fNz - 1);
   }

   // Convert a point p(x, y, z) to voxel index object
   // if outside, an invalid index object will be returned
   TGeoVoxelGridIndex pointToVoxelIndex(std::array<S, 3> const &p) const
   {
      if (!inside(p)) {
         return TGeoVoxelGridIndex(); // invalid voxel index
      }

      int i = static_cast<int>((p[0] - fMinBound[0]) * finvLx);
      int j = static_cast<int>((p[1] - fMinBound[1]) * finvLy);
      int k = static_cast<int>((p[2] - fMinBound[2]) * finvLz);

      // Clamp the indices to valid ranges
      i = std::min(i, fNz - 1);
      j = std::min(j, fNy - 1);
      k = std::min(k, fNz - 1);

      return TGeoVoxelGridIndex{i, j, k, index(i, j, k)};
   }

   TGeoVoxelGridIndex pointToVoxelIndex(S x, S y, S z) const { return pointToVoxelIndex(std::array<S, 3>{x, y, z}); }

   // Convert voxel indices (i, j, k) to a linear index in the grid array
   size_t index(int i, int j, int k) const { return i + fNx * (j + fNy * k); }

   void indexToIndices(size_t idx, int &i, int &j, int &k) const
   {
      k = idx % fNz;
      j = (idx / fNz) % fNy;
      i = idx / (fNy * fNz);
   }

   // member data

   std::array<S, 3> fMinBound;
   std::array<S, 3> fMaxBound; // 3D bounds for grid structure
   S fLx, fLy, fLz;            // Voxel dimensions
   S finvLx, finvLy, finvLz;   // inverse voxel dimensions
   S fHalfDiag;                // cached value for voxel half diagonal length

   int fNx, fNy, fNz;    // Number of voxels in each dimension
   std::vector<T> fGrid; // The actual voxel grid data container
};

#endif
