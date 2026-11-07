/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @file
 * Helper function to verify if triangulation has no inconsistencies
 */

#ifndef CDT_Zahj4kpHLwFgkKtcOI1i
#define CDT_Zahj4kpHLwFgkKtcOI1i

#include <CDT.h>

#include <algorithm>

namespace CDT
{

/**
 * Verify that triangulation topology is consistent.
 *
 * Checks:
 *  - for each vertex adjacent triangles contain the vertex
 *  - each triangle's neighbor in turn has triangle as its neighbor
 *  - each of triangle's vertices has triangle as adjacent
 *
 * @tparam T type of vertex coordinates (e.g., float, double)
 * @tparam TNearPointLocator class providing locating near point for efficiently
 */
template <typename T, typename TNearPointLocator>
inline bool verifyTopology(const CDT::Triangulation<T, TNearPointLocator>& cdt)
{
    // Check if vertices' adjacent triangles contain vertex
    const VerticesTriangles vertTris = calculateTrianglesByVertex(
        cdt.triangles, static_cast<VertInd>(cdt.vertices.size()));
    for(VertInd iV(0); iV < VertInd(cdt.vertices.size()); ++iV)
    {
        const TriIndVec& vTris = vertTris[iV];
        typedef TriIndVec::const_iterator TriIndCit;
        for(TriIndCit it = vTris.begin(); it != vTris.end(); ++it)
        {
            const array<VertInd, 3>& vv = cdt.triangles[*it].vertices;
            if(std::find(vv.begin(), vv.end(), iV) == vv.end())
                return false;
        }
    }
    // Check if triangle neighbor links are fine
    for(TriInd iT(0); iT < TriInd(cdt.triangles.size()); ++iT)
    {
        const Triangle& t = cdt.triangles[iT];
        typedef NeighborsArr3::const_iterator NCit;
        for(NCit it = t.neighbors.begin(); it != t.neighbors.end(); ++it)
        {
            if(*it == noNeighbor)
                continue;
            const array<TriInd, 3>& nn = cdt.triangles[*it].neighbors;
            if(std::find(nn.begin(), nn.end(), iT) == nn.end())
                return false;
        }
    }
    // Check if triangle's vertices have triangle as adjacent
    for(TriInd iT(0); iT < TriInd(cdt.triangles.size()); ++iT)
    {
        const Triangle& t = cdt.triangles[iT];
        typedef VerticesArr3::const_iterator VCit;
        for(VCit it = t.vertices.begin(); it != t.vertices.end(); ++it)
        {
            const TriIndVec& tt = vertTris[*it];
            if(std::find(tt.begin(), tt.end(), iT) == tt.end())
                return false;
        }
    }
    return true;
}

/// Check that every triangle is wound counter-clockwise: degenerate and
/// inverted triangles are topologically consistent, so #verifyTopology misses
/// them
template <typename T, typename TNearPointLocator>
inline bool verifyWinding(const CDT::Triangulation<T, TNearPointLocator>& cdt)
{
    typedef TriangleVec::const_iterator Cit;
    for(Cit t = cdt.triangles.begin(); t != cdt.triangles.end(); ++t)
    {
        if(orient2D(
               cdt.vertices[t->vertices[2]],
               cdt.vertices[t->vertices[0]],
               cdt.vertices[t->vertices[1]]) <= T(0))
        {
            return false;
        }
    }
    return true;
}

/// Check that each vertex has a neighbor triangle
template <typename T, typename TNearPointLocator>
inline bool eachVertexHasNeighborTriangle(
    const CDT::Triangulation<T, TNearPointLocator>& cdt)
{
    const TriIndVec& vertTris = cdt.VertTrisInternal();
    typedef TriIndVec::const_iterator TriIndCit;
    for(TriIndCit it = vertTris.begin(); it != vertTris.end(); ++it)
        if(*it == noNeighbor)
            return false;
    return true;
}

} // namespace CDT

#endif
