/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @file
 * Public API
 */

#ifndef CDT_lNrmUayWQaIR5fxnsg9B
#define CDT_lNrmUayWQaIR5fxnsg9B

#include "CDTUtils.h"
#include "Triangulation.h"

#include "remove_at.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iterator>
#include <memory>
#include <stack>
#include <vector>

/// Namespace containing triangulation functionality
namespace CDT
{

/** @defgroup API Public API
 *  Contains API for constrained and conforming Delaunay triangulations
 */
/// @{

/**
 * Type used for storing layer depths for triangles
 * @note LayerDepth should support 60K+ layers, which could be to much or
 * too little for some use cases. Feel free to re-define this typedef.
 */
typedef unsigned short LayerDepth;
typedef LayerDepth BoundaryOverlapCount;

/// Triangles by vertex index
typedef std::vector<TriIndVec> VerticesTriangles;

/** @defgroup helpers Helpers
 *  Helpers for working with CDT::Triangulation.
 */
/// @{

/**
 * Calculate triangles adjacent to vertices (triangles by vertex index)
 * @param triangles triangulation
 * @param verticesSize total number of vertices to pre-allocate the output
 * @return triangles by vertex index
 */
CDT_EXPORT VerticesTriangles
calculateTrianglesByVertex(const TriangleVec& triangles, VertInd verticesSize);

/**
 * Information about removed duplicated vertices.
 *
 * Contains mapping information and removed duplicates indices.
 * @note vertices {0,1,2,3,4} where 0 and 3 are the same will produce mapping
 *       {0,1,2,0,3} (to new vertices {0,1,2,3}) and duplicates {3}
 */
struct CDT_EXPORT DuplicatesInfo
{
    std::vector<std::size_t> mapping;    ///< vertex index mapping
    std::vector<std::size_t> duplicates; ///< duplicates' indices
};

/**
 * Find duplicates in given custom point-type range
 * @note duplicates are points with exactly same X and Y coordinates
 * @tparam TVertexIter iterator that dereferences to custom point type
 * @tparam TGetVertexCoordX function object getting x coordinate from vertex.
 * Getter signature: const TVertexIter::value_type& -> T
 * @tparam TGetVertexCoordY function object getting y coordinate from vertex.
 * Getter signature: const TVertexIter::value_type& -> T
 * @param first beginning of the range of vertices
 * @param last end of the range of vertices
 * @param getX getter of X-coordinate
 * @param getY getter of Y-coordinate
 * @returns information about vertex duplicates
 */
template <
    typename T,
    typename TVertexIter,
    typename TGetVertexCoordX,
    typename TGetVertexCoordY>
DuplicatesInfo FindDuplicates(
    TVertexIter first,
    TVertexIter last,
    TGetVertexCoordX getX,
    TGetVertexCoordY getY);

/**
 * Remove duplicates in-place from vector of custom points
 * @tparam TVertex vertex type
 * @tparam TAllocator allocator used by input vector of vertices
 * @param vertices vertices to remove duplicates from
 * @param duplicates information about duplicates
 */
template <typename TVertex, typename TAllocator>
void RemoveDuplicates(
    std::vector<TVertex, TAllocator>& vertices,
    const std::vector<std::size_t>& duplicates);

/**
 * Remove duplicated points in-place
 *
 * @tparam T type of vertex coordinates (e.g., float, double)
 * @param[in, out] vertices collection of vertices to remove duplicates from
 * @returns information about duplicated vertices that were removed.
 */
template <typename T>
CDT_EXPORT DuplicatesInfo RemoveDuplicates(std::vector<V2d<T> >& vertices);

/**
 * Remap vertex indices in edges (in-place) using given vertex-index mapping.
 * @tparam TEdgeIter iterator that dereferences to custom edge type
 * @tparam TGetEdgeVertexStart function object getting start vertex index
 * from an edge.
 * Getter signature: const TEdgeIter::value_type& -> CDT::VertInd
 * @tparam TGetEdgeVertexEnd function object getting end vertex index from
 * an edge. Getter signature: const TEdgeIter::value_type& -> CDT::VertInd
 * @tparam TMakeEdgeFromStartAndEnd function object that makes new edge from
 * start and end vertices
 * @param first beginning of the range of edges
 * @param last end of the range of edges
 * @param mapping vertex-index mapping
 * @param getStart getter of edge start vertex index
 * @param getEnd getter of edge end vertex index
 * @param makeEdge factory for making edge from vetices
 */
template <
    typename TEdgeIter,
    typename TGetEdgeVertexStart,
    typename TGetEdgeVertexEnd,
    typename TMakeEdgeFromStartAndEnd>
CDT_EXPORT void RemapEdges(
    TEdgeIter first,
    TEdgeIter last,
    const std::vector<std::size_t>& mapping,
    TGetEdgeVertexStart getStart,
    TGetEdgeVertexEnd getEnd,
    TMakeEdgeFromStartAndEnd makeEdge);

/**
 * Remap vertex indices in edges (in-place) using given vertex-index mapping.
 *
 * @note Mapping can be a result of RemoveDuplicates function
 * @param[in,out] edges collection of edges to remap
 * @param mapping vertex-index mapping
 */
CDT_EXPORT void
RemapEdges(std::vector<Edge>& edges, const std::vector<std::size_t>& mapping);

/**
 * Find point duplicates, remove them from vector (in-place) and remap edges
 * (in-place)
 * @note Same as a chained call of CDT::FindDuplicates, CDT::RemoveDuplicates,
 * and CDT::RemapEdges
 * @tparam T type of vertex coordinates (e.g., float, double)
 * @tparam TVertex type of vertex
 * @tparam TGetVertexCoordX function object getting x coordinate from vertex.
 * Getter signature: const TVertexIter::value_type& -> T
 * @tparam TGetVertexCoordY function object getting y coordinate from vertex.
 * Getter signature: const TVertexIter::value_type& -> T
 * @tparam TEdgeIter iterator that dereferences to custom edge type
 * @tparam TGetEdgeVertexStart function object getting start vertex index
 * from an edge.
 * Getter signature: const TEdgeIter::value_type& -> CDT::VertInd
 * @tparam TGetEdgeVertexEnd function object getting end vertex index from
 * an edge. Getter signature: const TEdgeIter::value_type& -> CDT::VertInd
 * @tparam TMakeEdgeFromStartAndEnd function object that makes new edge from
 * start and end vertices
 * @param[in, out] vertices vertices to remove duplicates from
 * @param[in, out] edges collection of edges connecting vertices
 * @param getX getter of X-coordinate
 * @param getY getter of Y-coordinate
 * @param edgesFirst beginning of the range of edges
 * @param edgesLast end of the range of edges
 * @param getStart getter of edge start vertex index
 * @param getEnd getter of edge end vertex index
 * @param makeEdge factory for making edge from vetices
 * @returns information about vertex duplicates
 */
template <
    typename T,
    typename TVertex,
    typename TGetVertexCoordX,
    typename TGetVertexCoordY,
    typename TVertexAllocator,
    typename TEdgeIter,
    typename TGetEdgeVertexStart,
    typename TGetEdgeVertexEnd,
    typename TMakeEdgeFromStartAndEnd>
DuplicatesInfo RemoveDuplicatesAndRemapEdges(
    std::vector<TVertex, TVertexAllocator>& vertices,
    TGetVertexCoordX getX,
    TGetVertexCoordY getY,
    TEdgeIter edgesFirst,
    TEdgeIter edgesLast,
    TGetEdgeVertexStart getStart,
    TGetEdgeVertexEnd getEnd,
    TMakeEdgeFromStartAndEnd makeEdge);

/**
 * Same as a chained call of CDT::RemoveDuplicates + CDT::RemapEdges
 *
 * @tparam T type of vertex coordinates (e.g., float, double)
 * @param[in, out] vertices collection of vertices to remove duplicates from
 * @param[in,out] edges collection of edges to remap
 */
template <typename T>
CDT_EXPORT DuplicatesInfo RemoveDuplicatesAndRemapEdges(
    std::vector<V2d<T> >& vertices,
    std::vector<Edge>& edges);

/**
 * Extract all edges of triangles
 *
 * @param triangles triangles used to extract edges
 * @return an unordered set of all edges of triangulation
 */
CDT_EXPORT EdgeUSet extractEdgesFromTriangles(const TriangleVec& triangles);

/*!
 * Converts piece->original_edges mapping to original_edge->pieces
 * @param pieceToOriginals maps pieces to original edges
 * @return mapping of original edges to pieces
 */
CDT_EXPORT unordered_map<Edge, EdgeVec>
EdgeToPiecesMapping(const unordered_map<Edge, EdgeVec>& pieceToOriginals);

/*!
 * Convert edge-to-pieces mapping into edge-to-split-vertices mapping
 * @tparam T type of vertex coordinates (e.g., float, double)
 * @param edgeToPieces edge-to-pieces mapping
 * @param vertices vertex buffer
 * @return mapping of edge-to-split-points.
 * Split points are sorted from edge's start (v1) to end (v2)
 */
template <typename T>
CDT_EXPORT unordered_map<Edge, std::vector<VertInd> > EdgeToSplitVertices(
    const unordered_map<Edge, EdgeVec>& edgeToPieces,
    const std::vector<V2d<T> >& vertices);

/// @}

/// @}

} // namespace CDT

//*****************************************************************************
// Implementations of template functionlity
//*****************************************************************************
// hash for CDT::V2d<T>
#ifdef CDT_CXX11_IS_SUPPORTED
namespace std
#else
namespace boost
#endif
{
template <typename T>
struct hash<CDT::V2d<T> >
{
    size_t operator()(const CDT::V2d<T>& xy) const
    {
#ifdef CDT_CXX11_IS_SUPPORTED
        typedef std::hash<T> Hasher;
#else
        typedef boost::hash<T> Hasher;
#endif
        return Hasher()(xy.x) ^ Hasher()(xy.y);
    }
};
} // namespace std

namespace CDT
{

//-----
// API
//-----
template <
    typename T,
    typename TVertexIter,
    typename TGetVertexCoordX,
    typename TGetVertexCoordY>
DuplicatesInfo FindDuplicates(
    TVertexIter first,
    TVertexIter last,
    TGetVertexCoordX getX,
    TGetVertexCoordY getY)
{
    typedef unordered_map<V2d<T>, std::size_t> PosToIndex;
    PosToIndex uniqueVerts;
    const std::size_t verticesSize = std::distance(first, last);
    DuplicatesInfo di = {
        std::vector<std::size_t>(verticesSize), std::vector<std::size_t>()};
    for(std::size_t iIn = 0, iOut = iIn; iIn < verticesSize; ++iIn, ++first)
    {
        typename PosToIndex::const_iterator it;
        bool isUnique;
        tie(it, isUnique) = uniqueVerts.insert(
            std::make_pair(V2d<T>::make(getX(*first), getY(*first)), iOut));
        if(isUnique)
        {
            di.mapping[iIn] = iOut++;
            continue;
        }
        di.mapping[iIn] = it->second; // found a duplicate
        di.duplicates.push_back(iIn);
    }
    return di;
}

template <typename TVertex, typename TAllocator>
void RemoveDuplicates(
    std::vector<TVertex, TAllocator>& vertices,
    const std::vector<std::size_t>& duplicates)
{
    vertices.erase(
        remove_at(
            vertices.begin(),
            vertices.end(),
            duplicates.begin(),
            duplicates.end()),
        vertices.end());
}

template <
    typename TEdgeIter,
    typename TGetEdgeVertexStart,
    typename TGetEdgeVertexEnd,
    typename TMakeEdgeFromStartAndEnd>
void RemapEdges(
    TEdgeIter first,
    const TEdgeIter last,
    const std::vector<std::size_t>& mapping,
    TGetEdgeVertexStart getStart,
    TGetEdgeVertexEnd getEnd,
    TMakeEdgeFromStartAndEnd makeEdge)
{
    for(; first != last; ++first)
    {
        *first = makeEdge(
            static_cast<VertInd>(mapping[getStart(*first)]),
            static_cast<VertInd>(mapping[getEnd(*first)]));
    }
}

template <
    typename T,
    typename TVertex,
    typename TGetVertexCoordX,
    typename TGetVertexCoordY,
    typename TVertexAllocator,
    typename TEdgeIter,
    typename TGetEdgeVertexStart,
    typename TGetEdgeVertexEnd,
    typename TMakeEdgeFromStartAndEnd>
DuplicatesInfo RemoveDuplicatesAndRemapEdges(
    std::vector<TVertex, TVertexAllocator>& vertices,
    TGetVertexCoordX getX,
    TGetVertexCoordY getY,
    const TEdgeIter edgesFirst,
    const TEdgeIter edgesLast,
    TGetEdgeVertexStart getStart,
    TGetEdgeVertexEnd getEnd,
    TMakeEdgeFromStartAndEnd makeEdge)
{
    const DuplicatesInfo di =
        FindDuplicates<T>(vertices.begin(), vertices.end(), getX, getY);
    RemoveDuplicates(vertices, di.duplicates);
    RemapEdges(edgesFirst, edgesLast, di.mapping, getStart, getEnd, makeEdge);
    return di;
}

template <typename T>
unordered_map<Edge, std::vector<VertInd> > EdgeToSplitVertices(
    const unordered_map<Edge, EdgeVec>& edgeToPieces,
    const std::vector<V2d<T> >& vertices)
{
    typedef std::pair<VertInd, T> VertCoordPair;
    struct ComparePred
    {
        bool operator()(const VertCoordPair& a, const VertCoordPair& b) const
        {
            return a.second < b.second;
        }
    } comparePred;

    unordered_map<Edge, std::vector<VertInd> > edgeToSplitVerts;
    typedef unordered_map<Edge, EdgeVec>::const_iterator It;
    for(It e2pIt = edgeToPieces.begin(); e2pIt != edgeToPieces.end(); ++e2pIt)
    {
        const Edge& e = e2pIt->first;
        const T dX = vertices[e.v2()].x - vertices[e.v1()].x;
        const T dY = vertices[e.v2()].y - vertices[e.v1()].y;
        const bool isX = std::abs(dX) >= std::abs(dY); // X-coord longer
        const bool isAscending =
            isX ? dX >= 0 : dY >= 0; // Longer coordinate ascends
        const EdgeVec& pieces = e2pIt->second;
        std::vector<VertCoordPair> splitVerts;
        // size is:  2[ends] + (pieces - 1)[split vertices] = pieces + 1
        splitVerts.reserve(pieces.size() + 1);
        typedef EdgeVec::const_iterator EIt;
        for(EIt pieceIt = pieces.begin(); pieceIt != pieces.end(); ++pieceIt)
        {
            const array<VertInd, 2> vv = {pieceIt->v1(), pieceIt->v2()};
            typedef array<VertInd, 2>::const_iterator VIt;
            for(VIt v = vv.begin(); v != vv.end(); ++v)
            {
                const T c = isX ? vertices[*v].x : vertices[*v].y;
                splitVerts.push_back(std::make_pair(*v, isAscending ? c : -c));
            }
        }
        // sort by longest coordinate
        std::sort(splitVerts.begin(), splitVerts.end(), comparePred);
        // remove duplicates
        splitVerts.erase(
            std::unique(splitVerts.begin(), splitVerts.end()),
            splitVerts.end());
        assert(splitVerts.size() > 2); // 2 end points with split vertices
        std::pair<Edge, std::vector<VertInd> > val =
            std::make_pair(e, std::vector<VertInd>());
        val.second.reserve(splitVerts.size());
        typedef typename std::vector<VertCoordPair>::const_iterator SEIt;
        for(SEIt it = splitVerts.begin() + 1; it != splitVerts.end() - 1; ++it)
        {
            val.second.push_back(it->first);
        }
        edgeToSplitVerts.insert(val);
    }
    return edgeToSplitVerts;
}

} // namespace CDT

#ifndef CDT_USE_AS_COMPILED_LIBRARY
#include "CDT.hpp"
#endif

#endif // header-guard
