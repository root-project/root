/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @file
 * Triangulation class - implementation
 */

#include "Triangulation.h"
#include "portable_nth_element.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <stdexcept>

namespace CDT
{

typedef std::deque<TriInd> TriDeque;

namespace detail
{

/// Needed for c++03 compatibility (no uniform initialization available)
template <typename T>
array<T, 3> arr3(const T& v0, const T& v1, const T& v2)
{
    const array<T, 3> out = {v0, v1, v2};
    return out;
}

namespace defaults
{

const std::size_t nTargetVerts = 0;
const SuperGeometryType::Enum superGeomType = SuperGeometryType::SuperTriangle;
const VertexInsertionOrder::Enum vertexInsertionOrder =
    VertexInsertionOrder::Auto;
const IntersectingConstraintEdges::Enum intersectingEdgesStrategy =
    IntersectingConstraintEdges::Ignore;
const float minDistToConstraintEdge(0);

} // namespace defaults

} // namespace detail

template <typename T, typename TNearPointLocator>
Triangulation<T, TNearPointLocator>::Triangulation()
    : m_nTargetVerts(detail::defaults::nTargetVerts)
    , m_superGeomType(detail::defaults::superGeomType)
    , m_vertexInsertionOrder(detail::defaults::vertexInsertionOrder)
    , m_intersectingEdgesStrategy(detail::defaults::intersectingEdgesStrategy)
    , m_minDistToConstraintEdge(detail::defaults::minDistToConstraintEdge)
{}

template <typename T, typename TNearPointLocator>
Triangulation<T, TNearPointLocator>::Triangulation(
    const VertexInsertionOrder::Enum vertexInsertionOrder)
    : m_nTargetVerts(detail::defaults::nTargetVerts)
    , m_superGeomType(detail::defaults::superGeomType)
    , m_vertexInsertionOrder(vertexInsertionOrder)
    , m_intersectingEdgesStrategy(detail::defaults::intersectingEdgesStrategy)
    , m_minDistToConstraintEdge(detail::defaults::minDistToConstraintEdge)
{}

template <typename T, typename TNearPointLocator>
Triangulation<T, TNearPointLocator>::Triangulation(
    const VertexInsertionOrder::Enum vertexInsertionOrder,
    const IntersectingConstraintEdges::Enum intersectingEdgesStrategy,
    const T minDistToConstraintEdge)
    : m_nTargetVerts(detail::defaults::nTargetVerts)
    , m_superGeomType(detail::defaults::superGeomType)
    , m_vertexInsertionOrder(vertexInsertionOrder)
    , m_intersectingEdgesStrategy(intersectingEdgesStrategy)
    , m_minDistToConstraintEdge(minDistToConstraintEdge)
{}

template <typename T, typename TNearPointLocator>
Triangulation<T, TNearPointLocator>::Triangulation(
    const VertexInsertionOrder::Enum vertexInsertionOrder,
    const TNearPointLocator& nearPtLocator,
    const IntersectingConstraintEdges::Enum intersectingEdgesStrategy,
    const T minDistToConstraintEdge)
    : m_nearPtLocator(nearPtLocator)
    , m_nTargetVerts(detail::defaults::nTargetVerts)
    , m_superGeomType(detail::defaults::superGeomType)
    , m_vertexInsertionOrder(vertexInsertionOrder)
    , m_intersectingEdgesStrategy(intersectingEdgesStrategy)
    , m_minDistToConstraintEdge(minDistToConstraintEdge)
{}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::eraseDummies()
{
    if(m_dummyTris.empty())
        return;
    const TriIndUSet dummySet(m_dummyTris.begin(), m_dummyTris.end());
    TriIndUMap triIndMap;
    triIndMap[noNeighbor] = noNeighbor;
    for(TriInd iT(0), iTnew(0); iT < TriInd(triangles.size()); ++iT)
    {
        if(dummySet.count(iT))
            continue;
        triIndMap[iT] = iTnew;
        triangles[iTnew] = triangles[iT];
        iTnew++;
    }
    triangles.erase(triangles.end() - dummySet.size(), triangles.end());

    // remap adjacent triangle indices for vertices
    for(TriIndVec::iterator iT = m_vertTris.begin(); iT != m_vertTris.end();
        ++iT)
    {
        *iT = triIndMap[*iT];
    }
    // remap neighbor indices for triangles
    for(TriangleVec::iterator t = triangles.begin(); t != triangles.end(); ++t)
    {
        NeighborsArr3& nn = t->neighbors;
        for(NeighborsArr3::iterator iN = nn.begin(); iN != nn.end(); ++iN)
            *iN = triIndMap[*iN];
    }
    // clear dummy triangles
    m_dummyTris = std::vector<TriInd>();
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::eraseSuperTriangle()
{
    if(m_superGeomType != SuperGeometryType::SuperTriangle)
        return;
    // find triangles adjacent to super-triangle's vertices
    TriIndUSet toErase;
    for(TriInd iT(0); iT < TriInd(triangles.size()); ++iT)
    {
        Triangle& t = triangles[iT];
        if(t.vertices[0] < 3 || t.vertices[1] < 3 || t.vertices[2] < 3)
            toErase.insert(iT);
    }
    finalizeTriangulation(toErase);
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::eraseOuterTriangles()
{
    // make dummy triangles adjacent to super-triangle's vertices
    assert(m_vertTris[0] != noNeighbor);
    const std::stack<TriInd> seed(std::deque<TriInd>(1, m_vertTris[0]));
    const TriIndUSet toErase = growToBoundary(seed);
    finalizeTriangulation(toErase);
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::eraseOuterTrianglesAndHoles()
{
    const std::vector<LayerDepth> triDepths = calculateTriangleDepths();
    TriIndUSet toErase;
    toErase.reserve(triangles.size());
    for(std::size_t iT = 0; iT != triangles.size(); ++iT)
    {
        if(triDepths[iT] % 2 == 0)
            toErase.insert(static_cast<TriInd>(iT));
    }
    finalizeTriangulation(toErase);
}

/// Remap removing super-triangle: subtract 3 from vertices
inline Edge RemapNoSuperTriangle(const Edge& e)
{
    return Edge(VertInd(e.v1() - 3), VertInd(e.v2() - 3));
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::removeTriangles(
    const TriIndUSet& removedTriangles)
{
    if(removedTriangles.empty())
        return;
    // remove triangles and calculate triangle index mapping
    TriIndUMap triIndMap;
    for(TriInd iT(0), iTnew(0); iT < TriInd(triangles.size()); ++iT)
    {
        if(removedTriangles.count(iT))
            continue;
        triIndMap[iT] = iTnew;
        triangles[iTnew] = triangles[iT];
        iTnew++;
    }
    triangles.erase(triangles.end() - removedTriangles.size(), triangles.end());
    // adjust triangles' neighbors
    for(TriInd iT(0); iT < triangles.size(); ++iT)
    {
        Triangle& t = triangles[iT];
        // update neighbors to account for removed triangles
        NeighborsArr3& nn = t.neighbors;
        for(NeighborsArr3::iterator n = nn.begin(); n != nn.end(); ++n)
        {
            if(removedTriangles.count(*n))
            {
                *n = noNeighbor;
            }
            else if(*n != noNeighbor)
            {
                *n = triIndMap[*n];
            }
        }
    }
}
template <typename T, typename TNearPointLocator>
TriIndVec& Triangulation<T, TNearPointLocator>::VertTrisInternal()
{
    return m_vertTris;
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::finalizeTriangulation(
    const TriIndUSet& removedTriangles)
{
    eraseDummies();
    m_vertTris = TriIndVec();
    // remove super-triangle
    if(m_superGeomType == SuperGeometryType::SuperTriangle)
    {
        vertices.erase(vertices.begin(), vertices.begin() + 3);
        // Edge re-mapping
        { // fixed edges
            EdgeUSet updatedFixedEdges;
            typedef CDT::EdgeUSet::const_iterator It;
            for(It e = fixedEdges.begin(); e != fixedEdges.end(); ++e)
            {
                updatedFixedEdges.insert(RemapNoSuperTriangle(*e));
            }
            fixedEdges = updatedFixedEdges;
        }
        { // overlap count
            unordered_map<Edge, BoundaryOverlapCount> updatedOverlapCount;
            typedef unordered_map<Edge, BoundaryOverlapCount>::const_iterator
                It;
            for(It it = overlapCount.begin(); it != overlapCount.end(); ++it)
            {
                updatedOverlapCount.insert(std::make_pair(
                    RemapNoSuperTriangle(it->first), it->second));
            }
            overlapCount = updatedOverlapCount;
        }
        { // split edges mapping
            unordered_map<Edge, EdgeVec> updatedPieceToOriginals;
            typedef unordered_map<Edge, EdgeVec>::const_iterator It;
            for(It it = pieceToOriginals.begin(); it != pieceToOriginals.end();
                ++it)
            {
                EdgeVec ee = it->second;
                for(EdgeVec::iterator eeIt = ee.begin(); eeIt != ee.end();
                    ++eeIt)
                {
                    *eeIt = RemapNoSuperTriangle(*eeIt);
                }
                updatedPieceToOriginals.insert(
                    std::make_pair(RemapNoSuperTriangle(it->first), ee));
            }
            pieceToOriginals = updatedPieceToOriginals;
        }
    }
    // remove other triangles
    removeTriangles(removedTriangles);
    // adjust triangle vertices: account for removed super-triangle
    if(m_superGeomType == SuperGeometryType::SuperTriangle)
    {
        for(TriangleVec::iterator t = triangles.begin(); t != triangles.end();
            ++t)
        {
            VerticesArr3& vv = t->vertices;
            for(VerticesArr3::iterator v = vv.begin(); v != vv.end(); ++v)
            {
                *v -= 3;
            }
        }
    }
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::initializedWithCustomSuperGeometry()
{
    m_nearPtLocator.initialize(vertices);
    m_nTargetVerts = vertices.size();
    m_superGeomType = SuperGeometryType::Custom;
}

template <typename T, typename TNearPointLocator>
TriIndUSet Triangulation<T, TNearPointLocator>::growToBoundary(
    std::stack<TriInd> seeds) const
{
    TriIndUSet traversed;
    while(!seeds.empty())
    {
        const TriInd iT = seeds.top();
        seeds.pop();
        traversed.insert(iT);
        const Triangle& t = triangles[iT];
        for(Index i(0); i < Index(3); ++i)
        {
            const Edge opEdge(t.vertices[ccw(i)], t.vertices[cw(i)]);
            if(fixedEdges.count(opEdge))
                continue;
            const TriInd iN = t.neighbors[opoNbr(i)];
            if(iN != noNeighbor && traversed.count(iN) == 0)
                seeds.push(iN);
        }
    }
    return traversed;
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::makeDummy(const TriInd iT)
{
    m_dummyTris.push_back(iT);
}

template <typename T, typename TNearPointLocator>
TriInd Triangulation<T, TNearPointLocator>::addTriangle(const Triangle& t)
{
    if(m_dummyTris.empty())
    {
        triangles.push_back(t);
        return TriInd(triangles.size() - 1);
    }
    const TriInd nxtDummy = m_dummyTris.back();
    m_dummyTris.pop_back();
    triangles[nxtDummy] = t;
    return nxtDummy;
}

template <typename T, typename TNearPointLocator>
TriInd Triangulation<T, TNearPointLocator>::addTriangle()
{
    if(m_dummyTris.empty())
    {
        const Triangle dummy = {
            {noVertex, noVertex, noVertex},
            {noNeighbor, noNeighbor, noNeighbor}};
        triangles.push_back(dummy);
        return TriInd(triangles.size() - 1);
    }
    const TriInd nxtDummy = m_dummyTris.back();
    m_dummyTris.pop_back();
    return nxtDummy;
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::insertEdges(
    const std::vector<Edge>& edges)
{
    insertEdges(edges.begin(), edges.end(), edge_get_v1, edge_get_v2);
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::conformToEdges(
    const std::vector<Edge>& edges)
{
    conformToEdges(edges.begin(), edges.end(), edge_get_v1, edge_get_v2);
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::fixEdge(const Edge& edge)
{
    if(!fixedEdges.insert(edge).second)
    {
        ++overlapCount[edge]; // if edge is already fixed increment the counter
    }
}

namespace detail
{

// add element to 'to' if not already in 'to'
template <typename T, typename Allocator1>
void insert_unique(std::vector<T, Allocator1>& to, const T& elem)
{
    if(std::find(to.begin(), to.end(), elem) == to.end())
    {
        to.push_back(elem);
    }
}

// add elements of 'from' that are not present in 'to' to 'to'
template <typename T, typename Allocator1, typename Allocator2>
void insert_unique(
    std::vector<T, Allocator1>& to,
    const std::vector<T, Allocator2>& from)
{
    typedef typename std::vector<T, Allocator2>::const_iterator Cit;
    to.reserve(to.size() + from.size());
    for(Cit cit = from.begin(); cit != from.end(); ++cit)
    {
        insert_unique(to, *cit);
    }
}

} // namespace detail

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::fixEdge(
    const Edge& edge,
    const Edge& originalEdge)
{
    fixEdge(edge);
    if(edge != originalEdge)
        detail::insert_unique(pieceToOriginals[edge], originalEdge);
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::fixEdge(
    const Edge& edge,
    const BoundaryOverlapCount overlaps)
{
    fixedEdges.insert(edge);
    overlapCount[edge] = overlaps; // override overlap counter
}

namespace detail
{

template <typename T>
T lerp(const T& a, const T& b, const T t)
{
    return (T(1) - t) * a + t * b;
}

// Precondition: ab and cd intersect normally
template <typename T>
V2d<T> intersectionPosition(
    const V2d<T>& a,
    const V2d<T>& b,
    const V2d<T>& c,
    const V2d<T>& d)
{
    using namespace predicates::adaptive;

    // note: for better accuracy we interpolate x and y separately
    // on a segment with the shortest x/y-projection correspondingly
    const T a_cd = orient2d(c.x, c.y, d.x, d.y, a.x, a.y);
    const T b_cd = orient2d(c.x, c.y, d.x, d.y, b.x, b.y);
    const T t_ab = a_cd / (a_cd - b_cd);

    const T c_ab = orient2d(a.x, a.y, b.x, b.y, c.x, c.y);
    const T d_ab = orient2d(a.x, a.y, b.x, b.y, d.x, d.y);
    const T t_cd = c_ab / (c_ab - d_ab);

    return V2d<T>::make(
        std::fabs(a.x - b.x) < std::fabs(c.x - d.x) ? lerp(a.x, b.x, t_ab)
                                                    : lerp(c.x, d.x, t_cd),
        std::fabs(a.y - b.y) < std::fabs(c.y - d.y) ? lerp(a.y, b.y, t_ab)
                                                    : lerp(c.y, d.y, t_cd));
}

} // namespace detail

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::insertEdgeIteration(
    const Edge edge,
    const Edge originalEdge,
    EdgeVec& remaining,
    std::vector<TriangulatePseudopolygonTask>& tppIterations)
{
    const VertInd iA = edge.v1();
    VertInd iB = edge.v2();
    if(iA == iB) // edge connects a vertex to itself
        return;

    if(hasEdge(iA, iB))
    {
        fixEdge(edge, originalEdge);
        return;
    }

    const V2d<T>& a = vertices[iA];
    const V2d<T>& b = vertices[iB];
    const T distanceTolerance =
        m_minDistToConstraintEdge == T(0)
            ? T(0)
            : m_minDistToConstraintEdge * distance(a, b);

    TriInd iT;
    // Note: 'L' is left and 'R' is right of the inserted constraint edge
    VertInd iVL, iVR;
    tie(iT, iVL, iVR) = intersectedTriangle(iA, a, b, distanceTolerance);
    // if one of the triangle vertices is on the edge, move edge start
    if(iT == noNeighbor)
    {
        const Edge edgePart(iA, iVL);
        fixEdge(edgePart, originalEdge);
        remaining.push_back(Edge(iVL, iB));
        return;
    }
    Triangle t = triangles[iT];
    std::vector<TriInd> intersected(1, iT);
    std::vector<VertInd> polyL, polyR;
    std::vector<TriInd> outerTrisL, outerTrisR;
    polyL.reserve(2);
    polyL.push_back(iA);
    polyL.push_back(iVL);
    outerTrisL.push_back(edgeNeighbor(t, iA, iVL));
    polyR.reserve(2);
    polyR.push_back(iA);
    polyR.push_back(iVR);
    outerTrisR.push_back(edgeNeighbor(t, iA, iVR));
    VertInd iV = iA;

    while(!t.containsVertex(iB))
    {
        const TriInd iTopo = opposedTriangle(t, iV);
        const Triangle& tOpo = triangles[iTopo];
        const VertInd iVopo = opposedVertex(tOpo, iT);

        // Resolve intersection between two constraint edges if needed
        if(m_intersectingEdgesStrategy ==
               IntersectingConstraintEdges::Resolve &&
           fixedEdges.count(Edge(iVL, iVR)))
        {
            const VertInd iNewVert = static_cast<VertInd>(vertices.size());

            // split constraint edge that already exists in triangulation
            const Edge splitEdge(iVL, iVR);
            const Edge half1(iVL, iNewVert);
            const Edge half2(iNewVert, iVR);
            const BoundaryOverlapCount overlaps = overlapCount[splitEdge];
            // remove the edge that will be split
            fixedEdges.erase(splitEdge);
            overlapCount.erase(splitEdge);
            // add split edge's halves
            fixEdge(half1, overlaps);
            fixEdge(half2, overlaps);
            // maintain piece-to-original mapping
            EdgeVec newOriginals(1, splitEdge);
            const unordered_map<Edge, EdgeVec>::const_iterator originalsIt =
                pieceToOriginals.find(splitEdge);
            if(originalsIt != pieceToOriginals.end())
            { // edge being split was split before: pass-through originals
                newOriginals = originalsIt->second;
                pieceToOriginals.erase(originalsIt);
            }
            detail::insert_unique(pieceToOriginals[half1], newOriginals);
            detail::insert_unique(pieceToOriginals[half2], newOriginals);
            // add a new point at the intersection of two constraint edges
            const V2d<T> newV = detail::intersectionPosition(
                vertices[iA], vertices[iB], vertices[iVL], vertices[iVR]);
            addNewVertex(newV, noNeighbor);
            std::stack<TriInd> triStack =
                insertVertexOnEdge(iNewVert, iT, iTopo);
            tryAddVertexToLocator(iNewVert);
            ensureDelaunayByEdgeFlips(newV, iNewVert, triStack);
            // TODO: is it's possible to re-use pseudo-polygons
            //  for inserting [iA, iNewVert] edge half?
            remaining.push_back(Edge(iA, iNewVert));
            remaining.push_back(Edge(iNewVert, iB));
            return;
        }

        const PtLineLocation::Enum loc =
            locatePointLine(vertices[iVopo], a, b, distanceTolerance);
        if(loc == PtLineLocation::Left)
        {
            // hanging edge check
            // previous entry of the vertex in poly if edge is hanging
            const IndexSizeType prev = polyL.size() - 2;
            if(iVopo == polyL[prev])
            { // hanging edge
                outerTrisL[prev] = noNeighbor;
                outerTrisL.push_back(noNeighbor);
            }
            else
            { // normal case
                outerTrisL.push_back(edgeNeighbor(tOpo, polyL.back(), iVopo));
            }
            polyL.push_back(iVopo);
            iV = iVL;
            iVL = iVopo;
        }
        else if(loc == PtLineLocation::Right)
        {
            // hanging edge check
            // previous entry of the vertex in poly if edge is hanging
            const IndexSizeType prev = polyR.size() - 2;
            if(iVopo == polyR[prev])
            { // hanging edge
                outerTrisR[prev] = noNeighbor;
                outerTrisR.push_back(noNeighbor);
            }
            else
            { // normal case
                outerTrisR.push_back(edgeNeighbor(tOpo, polyR.back(), iVopo));
            }
            polyR.push_back(iVopo);
            iV = iVR;
            iVR = iVopo;
        }
        else // encountered point on the edge
            iB = iVopo;

        intersected.push_back(iTopo);
        iT = iTopo;
        t = triangles[iT];
    }
    outerTrisL.push_back(edgeNeighbor(t, polyL.back(), iB));
    outerTrisR.push_back(edgeNeighbor(t, polyR.back(), iB));
    polyL.push_back(iB);
    polyR.push_back(iB);

    assert(!intersected.empty());
    // make sure start/end vertices have a valid adjacent triangle
    // that is not intersected by an edge
    if(m_vertTris[iA] == intersected.front())
        pivotVertexTriangleCW(iA);
    if(m_vertTris[iB] == intersected.back())
        pivotVertexTriangleCW(iB);
    // Remove intersected triangles
    typedef std::vector<TriInd>::const_iterator TriIndCit;
    for(TriIndCit it = intersected.begin(); it != intersected.end(); ++it)
        makeDummy(*it);
    // Triangulate pseudo-polygons on both sides
    std::reverse(polyR.begin(), polyR.end());
    std::reverse(outerTrisR.begin(), outerTrisR.end());
    const TriInd iTL = addTriangle();
    const TriInd iTR = addTriangle();
    triangulatePseudopolygon(polyL, outerTrisL, iTL, iTR, tppIterations);
    triangulatePseudopolygon(polyR, outerTrisR, iTR, iTL, tppIterations);

    if(iB != edge.v2()) // encountered point on the edge
    {
        // fix edge part
        const Edge edgePart(iA, iB);
        fixEdge(edgePart, originalEdge);
        remaining.push_back(Edge(iB, edge.v2()));
        return;
    }
    else
    {
        fixEdge(edge, originalEdge);
    }
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::insertEdge(
    Edge edge,
    const Edge originalEdge,
    EdgeVec& remaining,
    std::vector<TriangulatePseudopolygonTask>& tppIterations)
{
    // use iteration over recursion to avoid stack overflows
    remaining.clear();
    remaining.push_back(edge);
    while(!remaining.empty())
    {
        edge = remaining.back();
        remaining.pop_back();
        insertEdgeIteration(edge, originalEdge, remaining, tppIterations);
    }
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::conformToEdgeIteration(
    Edge edge,
    const EdgeVec& originals,
    BoundaryOverlapCount overlaps,
    std::vector<ConformToEdgeTask>& remaining)
{
    const VertInd iA = edge.v1();
    VertInd iB = edge.v2();
    if(iA == iB) // edge connects a vertex to itself
        return;

    if(hasEdge(iA, iB))
    {
        overlaps > 0 ? fixEdge(edge, overlaps) : fixEdge(edge);
        // avoid marking edge as a part of itself
        if(!originals.empty() && edge != originals.front())
        {
            detail::insert_unique(pieceToOriginals[edge], originals);
        }
        return;
    }

    const V2d<T>& a = vertices[iA];
    const V2d<T>& b = vertices[iB];
    const T distanceTolerance =
        m_minDistToConstraintEdge == T(0)
            ? T(0)
            : m_minDistToConstraintEdge * distance(a, b);
    TriInd iT;
    VertInd iVleft, iVright;
    tie(iT, iVleft, iVright) = intersectedTriangle(iA, a, b, distanceTolerance);
    // if one of the triangle vertices is on the edge, move edge start
    if(iT == noNeighbor)
    {
        const Edge edgePart(iA, iVleft);
        overlaps > 0 ? fixEdge(edgePart, overlaps) : fixEdge(edgePart);
        detail::insert_unique(pieceToOriginals[edgePart], originals);
#ifdef CDT_CXX11_IS_SUPPORTED
        remaining.emplace_back(Edge(iVleft, iB), originals, overlaps);
#else
        remaining.push_back(make_tuple(Edge(iVleft, iB), originals, overlaps));
#endif
        return;
    }

    VertInd iV = iA;
    Triangle t = triangles[iT];
    while(std::find(t.vertices.begin(), t.vertices.end(), iB) ==
          t.vertices.end())
    {
        const TriInd iTopo = opposedTriangle(t, iV);
        const Triangle& tOpo = triangles[iTopo];
        const VertInd iVopo = opposedVertex(tOpo, iT);
        const V2d<T> vOpo = vertices[iVopo];

        // Resolve intersection between two constraint edges if needed
        if(m_intersectingEdgesStrategy ==
               IntersectingConstraintEdges::Resolve &&
           fixedEdges.count(Edge(iVleft, iVright)))
        {
            const VertInd iNewVert = static_cast<VertInd>(vertices.size());

            // split constraint edge that already exists in triangulation
            const Edge splitEdge(iVleft, iVright);
            const Edge half1(iVleft, iNewVert);
            const Edge half2(iNewVert, iVright);

            const unordered_map<Edge, BoundaryOverlapCount>::const_iterator
                splitEdgeOverlapsIt = overlapCount.find(splitEdge);
            const BoundaryOverlapCount splitEdgeOverlaps =
                splitEdgeOverlapsIt != overlapCount.end()
                    ? splitEdgeOverlapsIt->second
                    : 0;
            // remove the edge that will be split and add split edge's
            // halves
            fixedEdges.erase(splitEdge);
            if(splitEdgeOverlaps > 0)
            {
                overlapCount.erase(splitEdgeOverlapsIt);
                fixEdge(half1, splitEdgeOverlaps);
                fixEdge(half2, splitEdgeOverlaps);
            }
            else
            {
                fixEdge(half1);
                fixEdge(half2);
            }
            // maintain piece-to-original mapping
            EdgeVec newOriginals(1, splitEdge);
            const unordered_map<Edge, EdgeVec>::const_iterator originalsIt =
                pieceToOriginals.find(splitEdge);
            if(originalsIt != pieceToOriginals.end())
            { // edge being split was split before: pass-through originals
                newOriginals = originalsIt->second;
                pieceToOriginals.erase(originalsIt);
            }
            detail::insert_unique(pieceToOriginals[half1], newOriginals);
            detail::insert_unique(pieceToOriginals[half2], newOriginals);

            // add a new point at the intersection of two constraint edges
            const V2d<T> newV = detail::intersectionPosition(
                vertices[iA],
                vertices[iB],
                vertices[iVleft],
                vertices[iVright]);
            addNewVertex(newV, noNeighbor);
            std::stack<TriInd> triStack =
                insertVertexOnEdge(iNewVert, iT, iTopo);
            tryAddVertexToLocator(iNewVert);
            ensureDelaunayByEdgeFlips(newV, iNewVert, triStack);
#ifdef CDT_CXX11_IS_SUPPORTED
            remaining.emplace_back(Edge(iNewVert, iB), originals, overlaps);
            remaining.emplace_back(Edge(iA, iNewVert), originals, overlaps);
#else
            remaining.push_back(
                make_tuple(Edge(iNewVert, iB), originals, overlaps));
            remaining.push_back(
                make_tuple(Edge(iA, iNewVert), originals, overlaps));
#endif
            return;
        }

        iT = iTopo;
        t = triangles[iT];

        const PtLineLocation::Enum loc =
            locatePointLine(vOpo, a, b, distanceTolerance);
        if(loc == PtLineLocation::Left)
        {
            iV = iVleft;
            iVleft = iVopo;
        }
        else if(loc == PtLineLocation::Right)
        {
            iV = iVright;
            iVright = iVopo;
        }
        else // encountered point on the edge
            iB = iVopo;
    }

    // encountered one or more points on the edge: add remaining edge part
    if(iB != edge.v2())
    {
#ifdef CDT_CXX11_IS_SUPPORTED
        remaining.emplace_back(Edge(iB, edge.v2()), originals, overlaps);
#else
        remaining.push_back(
            make_tuple(Edge(iB, edge.v2()), originals, overlaps));
#endif
    }

    // add mid-point to triangulation
    const VertInd iMid = static_cast<VertInd>(vertices.size());
    const V2d<T>& start = vertices[iA];
    const V2d<T>& end = vertices[iB];
    addNewVertex(
        V2d<T>::make((start.x + end.x) / T(2), (start.y + end.y) / T(2)),
        noNeighbor);
    const std::vector<Edge> flippedFixedEdges =
        insertVertex_FlipFixedEdges(iMid);

#ifdef CDT_CXX11_IS_SUPPORTED
    remaining.emplace_back(Edge(iMid, iB), originals, overlaps);
    remaining.emplace_back(Edge(iA, iMid), originals, overlaps);
#else
    remaining.push_back(make_tuple(Edge(iMid, iB), originals, overlaps));
    remaining.push_back(make_tuple(Edge(iA, iMid), originals, overlaps));
#endif

    // re-introduce fixed edges that were flipped
    // and make sure overlap count is preserved
    for(std::vector<Edge>::const_iterator it = flippedFixedEdges.begin();
        it != flippedFixedEdges.end();
        ++it)
    {
        const Edge& flippedFixedEdge = *it;
        fixedEdges.erase(flippedFixedEdge);

        BoundaryOverlapCount prevOverlaps = 0;
        const unordered_map<Edge, BoundaryOverlapCount>::const_iterator
            overlapsIt = overlapCount.find(flippedFixedEdge);
        if(overlapsIt != overlapCount.end())
        {
            prevOverlaps = overlapsIt->second;
            overlapCount.erase(overlapsIt);
        }
        // override overlapping boundaries count when re-inserting an edge
        EdgeVec prevOriginals(1, flippedFixedEdge);
        const unordered_map<Edge, EdgeVec>::const_iterator originalsIt =
            pieceToOriginals.find(flippedFixedEdge);
        if(originalsIt != pieceToOriginals.end())
        {
            prevOriginals = originalsIt->second;
        }
#ifdef CDT_CXX11_IS_SUPPORTED
        remaining.emplace_back(flippedFixedEdge, prevOriginals, prevOverlaps);
#else
        remaining.push_back(
            make_tuple(flippedFixedEdge, prevOriginals, prevOverlaps));
#endif
    }
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::conformToEdge(
    Edge edge,
    EdgeVec originals,
    BoundaryOverlapCount overlaps,
    std::vector<ConformToEdgeTask>& remaining)
{
    // use iteration over recursion to avoid stack overflows
    remaining.clear();
#ifdef CDT_CXX11_IS_SUPPORTED
    remaining.emplace_back(edge, originals, overlaps);
#else
    remaining.push_back(make_tuple(edge, originals, overlaps));
#endif
    while(!remaining.empty())
    {
        tie(edge, originals, overlaps) = remaining.back();
        remaining.pop_back();
        conformToEdgeIteration(edge, originals, overlaps, remaining);
    }
}

/*!
 * Returns:
 *  - intersected triangle index
 *  - index of point on the left of the line
 *  - index of point on the right of the line
 * If left point is right on the line: no triangle is intersected:
 *  - triangle index is no-neighbor (invalid)
 *  - index of point on the line
 *  - index of point on the right of the line
 */
template <typename T, typename TNearPointLocator>
tuple<TriInd, VertInd, VertInd>
Triangulation<T, TNearPointLocator>::intersectedTriangle(
    const VertInd iA,
    const V2d<T>& a,
    const V2d<T>& b,
    const T orientationTolerance) const
{
    const TriInd startTri = m_vertTris[iA];
    TriInd iT = startTri;
    do
    {
        const Triangle t = triangles[iT];
        const Index i = vertexInd(t.vertices, iA);
        const VertInd iP2 = t.vertices[ccw(i)];
        const T orientP2 = orient2D(vertices[iP2], a, b);
        const PtLineLocation::Enum locP2 = classifyOrientation(orientP2);
        if(locP2 == PtLineLocation::Right)
        {
            const VertInd iP1 = t.vertices[cw(i)];
            const T orientP1 = orient2D(vertices[iP1], a, b);
            const PtLineLocation::Enum locP1 = classifyOrientation(orientP1);
            if(locP1 == PtLineLocation::OnLine)
            {
                return make_tuple(noNeighbor, iP1, iP1);
            }
            if(locP1 == PtLineLocation::Left)
            {
                if(orientationTolerance)
                {
                    T closestOrient;
                    VertInd iClosestP;
                    if(std::abs(orientP1) <= std::abs(orientP2))
                    {
                        closestOrient = orientP1;
                        iClosestP = iP1;
                    }
                    else
                    {
                        closestOrient = orientP2;
                        iClosestP = iP2;
                    }
                    if(classifyOrientation(
                           closestOrient, orientationTolerance) ==
                       PtLineLocation::OnLine)
                    {
                        return make_tuple(noNeighbor, iClosestP, iClosestP);
                    }
                }
                return make_tuple(iT, iP1, iP2);
            }
        }
        iT = t.next(iA).first;
    } while(iT != startTri);
    throw std::runtime_error("Could not find vertex triangle intersected by "
                             "edge. Note: can be caused by duplicate points.");
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::addSuperTriangle(const Box2d<T>& box)
{
    m_nTargetVerts = 3;
    m_superGeomType = SuperGeometryType::SuperTriangle;

    const V2d<T> center = {
        (box.min.x + box.max.x) / T(2), (box.min.y + box.max.y) / T(2)};
    const T w = box.max.x - box.min.x;
    const T h = box.max.y - box.min.y;
    T r = std::sqrt(w * w + h * h) / T(2); // incircle radius
    r *= T(1.1);
    const T R = T(2) * r;                        // excircle radius
    const T shiftX = R * std::sqrt(T(3)) / T(2); // R * cos(30 deg)
    const V2d<T> posV1 = {center.x - shiftX, center.y - r};
    const V2d<T> posV2 = {center.x + shiftX, center.y - r};
    const V2d<T> posV3 = {center.x, center.y + R};
    addNewVertex(posV1, TriInd(0));
    addNewVertex(posV2, TriInd(0));
    addNewVertex(posV3, TriInd(0));
    const Triangle superTri = {
        {VertInd(0), VertInd(1), VertInd(2)},
        {noNeighbor, noNeighbor, noNeighbor}};
    addTriangle(superTri);
    if(m_vertexInsertionOrder != VertexInsertionOrder::Auto)
    {
        m_nearPtLocator.initialize(vertices);
    }
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::addNewVertex(
    const V2d<T>& pos,
    const TriInd iT)
{
    vertices.push_back(pos);
    m_vertTris.push_back(iT);
}

template <typename T, typename TNearPointLocator>
std::vector<Edge>
Triangulation<T, TNearPointLocator>::insertVertex_FlipFixedEdges(
    const VertInd iV1)
{
    std::vector<Edge> flippedFixedEdges;

    const V2d<T>& v1 = vertices[iV1];
    const VertInd startVertex = m_nearPtLocator.nearPoint(v1, vertices);
    array<TriInd, 2> trisAt = walkingSearchTrianglesAt(v1, startVertex);
    std::stack<TriInd> triStack =
        trisAt[1] == noNeighbor ? insertVertexInsideTriangle(iV1, trisAt[0])
                                : insertVertexOnEdge(iV1, trisAt[0], trisAt[1]);

    TriInd iTopo, n1, n2, n3, n4;
    VertInd iV2, iV3, iV4;
    while(!triStack.empty())
    {
        const TriInd iT = triStack.top();
        triStack.pop();

        edgeFlipInfo(iT, iV1, iTopo, iV2, iV3, iV4, n1, n2, n3, n4);
        if(iTopo != noNeighbor && isFlipNeeded(v1, iV1, iV2, iV3, iV4))
        {
            // if flipped edge is fixed, remember it
            const Edge flippedEdge(iV2, iV4);
            if(!fixedEdges.empty() &&
               fixedEdges.find(flippedEdge) != fixedEdges.end())
            {
                flippedFixedEdges.push_back(flippedEdge);
            }

            flipEdge(iT, iTopo, iV1, iV2, iV3, iV4, n1, n2, n3, n4);
            triStack.push(iT);
            triStack.push(iTopo);
        }
    }

    tryAddVertexToLocator(iV1);
    return flippedFixedEdges;
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::insertVertex(
    const VertInd iVert,
    const VertInd walkStart)
{
    const V2d<T>& v = vertices[iVert];
    const array<TriInd, 2> trisAt = walkingSearchTrianglesAt(v, walkStart);
    std::stack<TriInd> triStack =
        trisAt[1] == noNeighbor
            ? insertVertexInsideTriangle(iVert, trisAt[0])
            : insertVertexOnEdge(iVert, trisAt[0], trisAt[1]);
    ensureDelaunayByEdgeFlips(v, iVert, triStack);
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::insertVertex(const VertInd iVert)
{
    const V2d<T>& v = vertices[iVert];
    const VertInd walkStart = m_nearPtLocator.nearPoint(v, vertices);
    insertVertex(iVert, walkStart);
    tryAddVertexToLocator(iVert);
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::ensureDelaunayByEdgeFlips(
    const V2d<T>& v1,
    const VertInd iV1,
    std::stack<TriInd>& triStack)
{
    TriInd iTopo, n1, n2, n3, n4;
    VertInd iV2, iV3, iV4;
    while(!triStack.empty())
    {
        const TriInd iT = triStack.top();
        triStack.pop();

        edgeFlipInfo(iT, iV1, iTopo, iV2, iV3, iV4, n1, n2, n3, n4);
        if(iTopo != noNeighbor && isFlipNeeded(v1, iV1, iV2, iV3, iV4))
        {
            flipEdge(iT, iTopo, iV1, iV2, iV3, iV4, n1, n2, n3, n4);
            triStack.push(iT);
            triStack.push(iTopo);
        }
    }
}

/*
 *                       v4         original edge: (v1, v3)
 *                      /|\   flip-candidate edge: (v,  v2)
 *                    /  |  \
 *              n3  /    |    \  n4
 *                /      |      \
 * new vertex--> v1    T | Topo  v3
 *                \      |      /
 *              n1  \    |    /  n2
 *                    \  |  /
 *                      \|/
 *                       v2
 */
template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::edgeFlipInfo(
    const TriInd iT,
    const VertInd iV1,
    TriInd& iTopo,
    VertInd& iV2,
    VertInd& iV3,
    VertInd& iV4,
    TriInd& n1,
    TriInd& n2,
    TriInd& n3,
    TriInd& n4)
{
    /*     v[2]
           / \
      n[2]/   \n[1]
         /_____\
    v[0]  n[0]  v[1]  */
    const Triangle& t = triangles[iT];
    if(t.vertices[0] == iV1)
    {
        iV2 = t.vertices[1];
        iV4 = t.vertices[2];
        n1 = t.neighbors[0];
        n3 = t.neighbors[2];
        iTopo = t.neighbors[1];
    }
    else if(t.vertices[1] == iV1)
    {
        iV2 = t.vertices[2];
        iV4 = t.vertices[0];
        n1 = t.neighbors[1];
        n3 = t.neighbors[0];
        iTopo = t.neighbors[2];
    }
    else
    {
        iV2 = t.vertices[0];
        iV4 = t.vertices[1];
        n1 = t.neighbors[2];
        n3 = t.neighbors[1];
        iTopo = t.neighbors[0];
    }
    if(iTopo == noNeighbor)
        return;
    const Triangle& tOpo = triangles[iTopo];
    if(tOpo.neighbors[0] == iT)
    {
        iV3 = tOpo.vertices[2];
        n2 = tOpo.neighbors[1];
        n4 = tOpo.neighbors[2];
    }
    else if(tOpo.neighbors[1] == iT)
    {
        iV3 = tOpo.vertices[0];
        n2 = tOpo.neighbors[2];
        n4 = tOpo.neighbors[0];
    }
    else
    {
        iV3 = tOpo.vertices[1];
        n2 = tOpo.neighbors[0];
        n4 = tOpo.neighbors[1];
    }
}

/*!
 * Handles super-triangle vertices.
 * Super-tri points are not infinitely far and influence the input points
 * Three cases are possible:
 *  1.  If one of the opposed vertices is super-tri: no flip needed
 *  2.  One of the shared vertices is super-tri:
 *      check if on point is same side of line formed by non-super-tri
 * vertices as the non-super-tri shared vertex
 *  3.  None of the vertices are super-tri: normal circumcircle test
 */
/*
 *                       v4         original edge: (v2, v4)
 *                      /|\   flip-candidate edge: (v1, v3)
 *                    /  |  \
 *                  /    |    \
 *                /      |      \
 * new vertex--> v1      |       v3
 *                \      |      /
 *                  \    |    /
 *                    \  |  /
 *                      \|/
 *                       v2
 */
template <typename T, typename TNearPointLocator>
bool Triangulation<T, TNearPointLocator>::isFlipNeeded(
    const V2d<T>& v,
    const VertInd iV1,
    const VertInd iV2,
    const VertInd iV3,
    const VertInd iV4) const
{
    if(fixedEdges.count(Edge(iV2, iV4)))
        return false; // flip not needed if the original edge is fixed
    const V2d<T>& v2 = vertices[iV2];
    const V2d<T>& v3 = vertices[iV3];
    const V2d<T>& v4 = vertices[iV4];
    if(m_superGeomType == SuperGeometryType::SuperTriangle)
    {
        // If flip-candidate edge touches super-triangle in-circumference
        // test has to be replaced with orient2d test against the line
        // formed by two non-artificial vertices (that don't belong to
        // super-triangle)
        if(iV1 < 3) // flip-candidate edge touches super-triangle
        {
            // does original edge also touch super-triangle?
            if(iV2 < 3)
                return locatePointLine(v2, v3, v4) ==
                       locatePointLine(v, v3, v4);
            if(iV4 < 3)
                return locatePointLine(v4, v2, v3) ==
                       locatePointLine(v, v2, v3);
            return false; // original edge does not touch super-triangle
        }
        if(iV3 < 3) // flip-candidate edge touches super-triangle
        {
            // does original edge also touch super-triangle?
            if(iV2 < 3)
                return locatePointLine(v2, v, v4) == locatePointLine(v3, v, v4);
            if(iV4 < 3)
                return locatePointLine(v4, v2, v) == locatePointLine(v3, v2, v);
            return false; // original edge does not touch super-triangle
        }
        // flip-candidate edge does not touch super-triangle
        if(iV2 < 3)
            return locatePointLine(v2, v3, v4) == locatePointLine(v, v3, v4);
        if(iV4 < 3)
            return locatePointLine(v4, v2, v3) == locatePointLine(v, v2, v3);
    }
    return isInCircumcircle(v, v2, v3, v4);
}

/* Flip edge between T and Topo:
 *
 *                v4         | - old edge
 *               /|\         ~ - new edge
 *              / | \
 *          n3 /  T' \ n4
 *            /   |   \
 *           /    |    \
 *     T -> v1 ~~~~~~~~ v3 <- Topo
 *           \    |    /
 *            \   |   /
 *          n1 \Topo'/ n2
 *              \ | /
 *               \|/
 *                v2
 */
template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::flipEdge(
    const TriInd iT,
    const TriInd iTopo,
    const VertInd v1,
    const VertInd v2,
    const VertInd v3,
    const VertInd v4,
    const TriInd n1,
    const TriInd n2,
    const TriInd n3,
    const TriInd n4)
{
    // change vertices and neighbors
    using detail::arr3;
    triangles[iT] = Triangle::make(arr3(v4, v1, v3), arr3(n3, iTopo, n4));
    triangles[iTopo] = Triangle::make(arr3(v2, v3, v1), arr3(n2, iT, n1));
    // adjust neighboring triangles and vertices
    changeNeighbor(n1, iT, iTopo);
    changeNeighbor(n4, iTopo, iT);
    // only adjust adjacent triangles if triangulation is not finalized:
    // can happen when called from outside on an already finalized
    // triangulation
    if(!isFinalized())
    {
        setAdjacentTriangle(v4, iT);
        setAdjacentTriangle(v2, iTopo);
    }
}

/* Insert point into triangle: split into 3 triangles:
 *  - create 2 new triangles
 *  - re-use old triangle for the 3rd
 *                      v3
 *                    / | \
 *                   /  |  \ <-- original triangle (t)
 *                  /   |   \
 *              n3 /    |    \ n2
 *                /newT2|newT1\
 *               /      v      \
 *              /    __/ \__    \
 *             /  __/       \__  \
 *            / _/      t'     \_ \
 *          v1 ___________________ v2
 *                     n1
 */
template <typename T, typename TNearPointLocator>
std::stack<TriInd>
Triangulation<T, TNearPointLocator>::insertVertexInsideTriangle(
    VertInd v,
    TriInd iT)
{
    const TriInd iNewT1 = addTriangle();
    const TriInd iNewT2 = addTriangle();

    Triangle& t = triangles[iT];
    const array<VertInd, 3> vv = t.vertices;
    const array<TriInd, 3> nn = t.neighbors;
    const VertInd v1 = vv[0], v2 = vv[1], v3 = vv[2];
    const TriInd n1 = nn[0], n2 = nn[1], n3 = nn[2];
    // make two new triangles and convert current triangle to 3rd new
    // triangle
    using detail::arr3;
    triangles[iNewT1] = Triangle::make(arr3(v2, v3, v), arr3(n2, iNewT2, iT));
    triangles[iNewT2] = Triangle::make(arr3(v3, v1, v), arr3(n3, iT, iNewT1));
    t = Triangle::make(arr3(v1, v2, v), arr3(n1, iNewT1, iNewT2));
    // adjust adjacent triangles
    setAdjacentTriangle(v, iT);
    setAdjacentTriangle(v3, iNewT1);
    // change triangle neighbor's neighbors to new triangles
    changeNeighbor(n2, iT, iNewT1);
    changeNeighbor(n3, iT, iNewT2);
    // return newly added triangles
    std::stack<TriInd> newTriangles;
    newTriangles.push(iT);
    newTriangles.push(iNewT1);
    newTriangles.push(iNewT2);
    return newTriangles;
}

/* Inserting a point on the edge between two triangles
 *    T1 (top)        v1
 *                   /|\
 *              n1 /  |  \ n4
 *               /    |    \
 *             /  T1' | Tnew1\
 *           v2-------v-------v4
 *             \  T2' | Tnew2/
 *               \    |    /
 *              n2 \  |  / n3
 *                   \|/
 *   T2 (bottom)      v3
 */
template <typename T, typename TNearPointLocator>
std::stack<TriInd> Triangulation<T, TNearPointLocator>::insertVertexOnEdge(
    VertInd v,
    TriInd iT1,
    TriInd iT2)
{
    const TriInd iTnew1 = addTriangle();
    const TriInd iTnew2 = addTriangle();

    Triangle& t1 = triangles[iT1];
    Triangle& t2 = triangles[iT2];
    Index i = opposedVertexInd(t1.neighbors, iT2);
    const VertInd v1 = t1.vertices[i];
    const VertInd v2 = t1.vertices[ccw(i)];
    const TriInd n1 = t1.neighbors[i];
    const TriInd n4 = t1.neighbors[cw(i)];
    i = opposedVertexInd(t2.neighbors, iT1);
    const VertInd v3 = t2.vertices[i];
    const VertInd v4 = t2.vertices[ccw(i)];
    const TriInd n3 = t2.neighbors[i];
    const TriInd n2 = t2.neighbors[cw(i)];
    // add new triangles and change existing ones
    using detail::arr3;
    t1 = Triangle::make(arr3(v, v1, v2), arr3(iTnew1, n1, iT2));
    t2 = Triangle::make(arr3(v, v2, v3), arr3(iT1, n2, iTnew2));
    triangles[iTnew1] = Triangle::make(arr3(v, v4, v1), arr3(iTnew2, n4, iT1));
    triangles[iTnew2] = Triangle::make(arr3(v, v3, v4), arr3(iT2, n3, iTnew1));
    // adjust adjacent triangles
    setAdjacentTriangle(v, iT1);
    setAdjacentTriangle(v4, iTnew1);
    // adjust neighboring triangles and vertices
    changeNeighbor(n4, iT1, iTnew1);
    changeNeighbor(n3, iT2, iTnew2);
    // return newly added triangles
    std::stack<TriInd> newTriangles;
    newTriangles.push(iT1);
    newTriangles.push(iTnew2);
    newTriangles.push(iT2);
    newTriangles.push(iTnew1);
    return newTriangles;
}

template <typename T, typename TNearPointLocator>
array<TriInd, 2>
Triangulation<T, TNearPointLocator>::trianglesAt(const V2d<T>& pos) const
{
    array<TriInd, 2> out = {noNeighbor, noNeighbor};
    for(TriInd i = TriInd(0); i < TriInd(triangles.size()); ++i)
    {
        const Triangle& t = triangles[i];
        const V2d<T>& v1 = vertices[t.vertices[0]];
        const V2d<T>& v2 = vertices[t.vertices[1]];
        const V2d<T>& v3 = vertices[t.vertices[2]];
        const PtTriLocation::Enum loc = locatePointTriangle(pos, v1, v2, v3);
        if(loc == PtTriLocation::Outside)
            continue;
        out[0] = i;
        if(isOnEdge(loc))
            out[1] = t.neighbors[edgeNeighbor(loc)];
        return out;
    }
    throw std::runtime_error("No triangle was found at position");
}

template <typename T, typename TNearPointLocator>
TriInd Triangulation<T, TNearPointLocator>::walkTriangles(
    const VertInd startVertex,
    const V2d<T>& pos) const
{
    // begin walk in search of triangle at pos
    TriInd currTri = m_vertTris[startVertex];
    bool found = false;
    detail::SplitMix64RandGen prng;
    while(!found)
    {
        const Triangle& t = triangles[currTri];
        found = true;
        // stochastic offset to randomize which edge we check first
        const Index offset(prng() % 3);
        for(Index i_(0); i_ < Index(3); ++i_)
        {
            const Index i((i_ + offset) % 3);
            const V2d<T>& vStart = vertices[t.vertices[i]];
            const V2d<T>& vEnd = vertices[t.vertices[ccw(i)]];
            const PtLineLocation::Enum edgeCheck =
                locatePointLine(pos, vStart, vEnd);
            const TriInd iN = t.neighbors[i];
            if(edgeCheck == PtLineLocation::Right && iN != noNeighbor)
            {
                found = false;
                currTri = t.neighbors[i];
                break;
            }
        }
    }
    return currTri;
}

template <typename T, typename TNearPointLocator>
array<TriInd, 2> Triangulation<T, TNearPointLocator>::walkingSearchTrianglesAt(
    const V2d<T>& pos,
    const VertInd startVertex) const
{
    array<TriInd, 2> out = {noNeighbor, noNeighbor};
    const TriInd iT = walkTriangles(startVertex, pos);
    // Finished walk, locate point in current triangle
    const Triangle& t = triangles[iT];
    const V2d<T>& v1 = vertices[t.vertices[0]];
    const V2d<T>& v2 = vertices[t.vertices[1]];
    const V2d<T>& v3 = vertices[t.vertices[2]];
    const PtTriLocation::Enum loc = locatePointTriangle(pos, v1, v2, v3);
    if(loc == PtTriLocation::Outside)
        throw std::runtime_error("No triangle was found at position");
    out[0] = iT;
    if(isOnEdge(loc))
        out[1] = t.neighbors[edgeNeighbor(loc)];
    return out;
}

/* Flip edge between T and Topo:
 *
 *                v4         | - old edge
 *               /|\         ~ - new edge
 *              / | \
 *          n3 /  T' \ n4
 *            /   |   \
 *           /    |    \
 *     T -> v1~~~~~~~~~v3 <- Topo
 *           \    |    /
 *            \   |   /
 *          n1 \Topo'/ n2
 *              \ | /
 *               \|/
 *                v2
 */
template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::flipEdge(
    const TriInd iT,
    const TriInd iTopo)
{
    Triangle& t = triangles[iT];
    Triangle& tOpo = triangles[iTopo];
    const array<TriInd, 3>& triNs = t.neighbors;
    const array<TriInd, 3>& triOpoNs = tOpo.neighbors;
    const array<VertInd, 3>& triVs = t.vertices;
    const array<VertInd, 3>& triOpoVs = tOpo.vertices;
    // find vertices and neighbors
    Index i = opposedVertexInd(t.neighbors, iTopo);
    const VertInd v1 = triVs[i];
    const VertInd v2 = triVs[ccw(i)];
    const TriInd n1 = triNs[i];
    const TriInd n3 = triNs[cw(i)];
    i = opposedVertexInd(tOpo.neighbors, iT);
    const VertInd v3 = triOpoVs[i];
    const VertInd v4 = triOpoVs[ccw(i)];
    const TriInd n4 = triOpoNs[i];
    const TriInd n2 = triOpoNs[cw(i)];
    // change vertices and neighbors
    using detail::arr3;
    t = Triangle::make(arr3(v4, v1, v3), arr3(n3, iTopo, n4));
    tOpo = Triangle::make(arr3(v2, v3, v1), arr3(n2, iT, n1));
    // adjust neighboring triangles and vertices
    changeNeighbor(n1, iT, iTopo);
    changeNeighbor(n4, iTopo, iT);
    // only adjust adjacent triangles if triangulation is not finalized:
    // can happen when called from outside on an already finalized
    // triangulation
    if(!isFinalized())
    {
        setAdjacentTriangle(v4, iT);
        setAdjacentTriangle(v2, iTopo);
    }
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::changeNeighbor(
    const TriInd iT,
    const TriInd oldNeighbor,
    const TriInd newNeighbor)
{
    if(iT == noNeighbor)
        return;
    NeighborsArr3& nn = triangles[iT].neighbors;
    assert(
        nn[0] == oldNeighbor || nn[1] == oldNeighbor || nn[2] == oldNeighbor);
    if(nn[0] == oldNeighbor)
        nn[0] = newNeighbor;
    else if(nn[1] == oldNeighbor)
        nn[1] = newNeighbor;
    else
        nn[2] = newNeighbor;
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::changeNeighbor(
    const TriInd iT,
    const VertInd iVedge1,
    const VertInd iVedge2,
    const TriInd newNeighbor)
{
    assert(iT != noNeighbor);
    Triangle& t = triangles[iT];
    t.neighbors[edgeNeighborInd(t.vertices, iVedge1, iVedge2)] = newNeighbor;
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::triangulatePseudopolygon(
    const std::vector<VertInd>& poly,
    const std::vector<TriInd>& outerTris,
    const TriInd iT,
    const TriInd iN,
    std::vector<TriangulatePseudopolygonTask>& iterations)
{
    assert(poly.size() > 2);
    // note: uses interation instead of recursion to avoid stack overflows
    iterations.clear();
    iterations.push_back(make_tuple(
        IndexSizeType(0),
        static_cast<IndexSizeType>(poly.size() - 1),
        iT,
        iN,
        Index(0)));
    while(!iterations.empty())
    {
        triangulatePseudopolygonIteration(poly, outerTris, iterations);
    }
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::triangulatePseudopolygonIteration(
    const std::vector<VertInd>& poly,
    const std::vector<TriInd>& outerTris,
    std::vector<TriangulatePseudopolygonTask>& iterations)
{
    IndexSizeType iA, iB;
    TriInd iT, iParent;
    Index iInParent;
    assert(!iterations.empty());
    tie(iA, iB, iT, iParent, iInParent) = iterations.back();
    iterations.pop_back();
    assert(iB - iA > 1 && iT != noNeighbor && iParent != noNeighbor);
    Triangle& t = triangles[iT];
    // find Delaunay point
    const IndexSizeType iC = findDelaunayPoint(poly, iA, iB);

    const VertInd a = poly[iA];
    const VertInd b = poly[iB];
    const VertInd c = poly[iC];
    // split pseudo-polygon in two parts and triangulate them
    //
    // note: first part needs to be pushed on stack last
    // in order to be processed first
    //
    // second part: points after the Delaunay point
    if(iB - iC > 1)
    {
        const TriInd iNext = addTriangle();
        iterations.push_back(make_tuple(iC, iB, iNext, iT, Index(1)));
    }
    else // pseudo-poly is reduced to a single outer edge
    {
        const TriInd outerTri = outerTris[iC];
        if(outerTri != noNeighbor)
        {
            assert(outerTri != iT);
            t.neighbors[1] = outerTri;
            changeNeighbor(outerTri, c, b, iT);
        }
    }
    // first part: points before the Delaunay point
    if(iC - iA > 1)
    { // add next triangle and add another iteration
        const TriInd iNext = addTriangle();
        iterations.push_back(make_tuple(iA, iC, iNext, iT, Index(2)));
    }
    else
    { // pseudo-poly is reduced to a single outer edge
        const TriInd outerTri =
            outerTris[iA] != noNeighbor ? outerTris[iA] : m_vertTris[c];
        if(outerTri != noNeighbor)
        {
            assert(outerTri != iT);
            t.neighbors[2] = outerTri;
            changeNeighbor(outerTri, c, a, iT);
        }
    }
    // Finalize triangle
    // note: only when triangle is finalized to we add it as a neighbor to
    // parent to maintain triangulation topology consistency
    triangles[iParent].neighbors[iInParent] = iT;
    t.neighbors[0] = iParent;
    t.vertices = detail::arr3(a, b, c);
    // needs to be done at the end not to affect finding edge triangles
    setAdjacentTriangle(c, iT);
}

template <typename T, typename TNearPointLocator>
IndexSizeType Triangulation<T, TNearPointLocator>::findDelaunayPoint(
    const std::vector<VertInd>& poly,
    const IndexSizeType iA,
    const IndexSizeType iB) const
{
    assert(iB - iA > 1);
    const V2d<T>& a = vertices[poly[iA]];
    const V2d<T>& b = vertices[poly[iB]];
    IndexSizeType out = iA + 1;
    const V2d<T>* c = &vertices[poly[out]]; // caching for better performance
    for(IndexSizeType i = iA + 1; i < iB; ++i)
    {
        const V2d<T>& v = vertices[poly[i]];
        if(isInCircumcircle(v, a, b, *c))
        {
            out = i;
            c = &v;
        }
    }
    assert(out > iA && out < iB); // point is between ends
    return out;
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::insertVertices(
    const std::vector<V2d<T> >& newVertices)
{
    return insertVertices(
        newVertices.begin(), newVertices.end(), getX_V2d<T>, getY_V2d<T>);
}

template <typename T, typename TNearPointLocator>
bool Triangulation<T, TNearPointLocator>::isFinalized() const
{
    return m_vertTris.empty() && !vertices.empty();
}

template <typename T, typename TNearPointLocator>
unordered_map<TriInd, LayerDepth>
Triangulation<T, TNearPointLocator>::peelLayer(
    std::stack<TriInd> seeds,
    const LayerDepth layerDepth,
    std::vector<LayerDepth>& triDepths) const
{
    unordered_map<TriInd, LayerDepth> behindBoundary;
    while(!seeds.empty())
    {
        const TriInd iT = seeds.top();
        seeds.pop();
        triDepths[iT] = std::min(triDepths[iT], layerDepth);
        behindBoundary.erase(iT);
        const Triangle& t = triangles[iT];
        for(Index i(0); i < Index(3); ++i)
        {
            const Edge opEdge(t.vertices[ccw(i)], t.vertices[cw(i)]);
            const TriInd iN = t.neighbors[opoNbr(i)];
            if(iN == noNeighbor || triDepths[iN] <= layerDepth)
                continue;
            if(fixedEdges.count(opEdge))
            {
                const unordered_map<Edge, LayerDepth>::const_iterator cit =
                    overlapCount.find(opEdge);
                const LayerDepth triDepth = cit == overlapCount.end()
                                                ? layerDepth + 1
                                                : layerDepth + cit->second + 1;
                behindBoundary[iN] = triDepth;
                continue;
            }
            seeds.push(iN);
        }
    }
    return behindBoundary;
}

template <typename T, typename TNearPointLocator>
std::vector<LayerDepth>
Triangulation<T, TNearPointLocator>::calculateTriangleDepths() const
{
    std::vector<LayerDepth> triDepths(
        triangles.size(), std::numeric_limits<LayerDepth>::max());
    std::stack<TriInd> seeds(TriDeque(1, m_vertTris[0]));
    LayerDepth layerDepth = 0;
    LayerDepth deepestSeedDepth = 0;

    unordered_map<LayerDepth, TriIndUSet> seedsByDepth;
    do
    {
        const unordered_map<TriInd, LayerDepth>& newSeeds =
            peelLayer(seeds, layerDepth, triDepths);

        seedsByDepth.erase(layerDepth);
        typedef unordered_map<TriInd, LayerDepth>::const_iterator Iter;
        for(Iter it = newSeeds.begin(); it != newSeeds.end(); ++it)
        {
            deepestSeedDepth = std::max(deepestSeedDepth, it->second);
            seedsByDepth[it->second].insert(it->first);
        }
        const TriIndUSet& nextLayerSeeds = seedsByDepth[layerDepth + 1];
        seeds = std::stack<TriInd>(
            TriDeque(nextLayerSeeds.begin(), nextLayerSeeds.end()));
        ++layerDepth;
    } while(!seeds.empty() || deepestSeedDepth > layerDepth);

    return triDepths;
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::insertVertices_AsProvided(
    VertInd superGeomVertCount)
{
    for(VertInd iV = superGeomVertCount; iV < vertices.size(); ++iV)
    {
        insertVertex(iV);
    }
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::insertVertices_Randomized(
    VertInd superGeomVertCount)
{
    std::size_t vertexCount = vertices.size() - superGeomVertCount;
    std::vector<VertInd> ii(vertexCount);
    detail::iota(ii.begin(), ii.end(), superGeomVertCount);
    detail::random_shuffle(ii.begin(), ii.end());
    for(std::vector<VertInd>::iterator it = ii.begin(); it != ii.end(); ++it)
    {
        insertVertex(*it);
    }
}

namespace detail
{

// log2 implementation backwards compatible with pre c++11
template <typename T>
inline double log2_bc(T x)
{
#ifdef CDT_CXX11_IS_SUPPORTED
    return std::log2(x);
#else
    static double log2_constant = std::log(2.0);
    return std::log(static_cast<double>(x)) / log2_constant;
#endif
}

/// Since KD-tree bulk load builds a balanced tree the maximum length of a
/// queue can be pre-calculated: it is calculated as size of a completely
/// filled tree layer plus the number of the nodes on a completely filled
/// layer that have two children.
inline std::size_t maxQueueLengthBFSKDTree(const std::size_t vertexCount)
{
    const int filledLayerPow2 =
        static_cast<int>(std::floor(log2_bc(vertexCount)) - 1);
    const std::size_t nodesInFilledTree =
        static_cast<std::size_t>(std::pow(2., filledLayerPow2 + 1) - 1);
    const std::size_t nodesInLastFilledLayer =
        static_cast<std::size_t>(std::pow(2., filledLayerPow2));
    const std::size_t nodesInLastLayer = vertexCount - nodesInFilledTree;
    return nodesInLastLayer >= nodesInLastFilledLayer
               ? nodesInLastFilledLayer + nodesInLastLayer -
                     nodesInLastFilledLayer
               : nodesInLastFilledLayer;
}

template <typename T>
class FixedCapacityQueue
{
public:
    FixedCapacityQueue(const std::size_t capacity)
        : m_vec(capacity)
        , m_front(m_vec.begin())
        , m_back(m_vec.begin())
        , m_size(0)
    {}
    bool empty() const
    {
        return m_size == 0;
    }
    const T& front() const
    {
        return *m_front;
    }
    void pop()
    {
        assert(m_size > 0);
        ++m_front;
        if(m_front == m_vec.end())
            m_front = m_vec.begin();
        --m_size;
    }
    void push(const T& t)
    {
        assert(m_size < m_vec.size());
        *m_back = t;
        ++m_back;
        if(m_back == m_vec.end())
            m_back = m_vec.begin();
        ++m_size;
    }
#ifdef CDT_CXX11_IS_SUPPORTED
    void push(const T&& t)
    {
        assert(m_size < m_vec.size());
        *m_back = t;
        ++m_back;
        if(m_back == m_vec.end())
            m_back = m_vec.begin();
        ++m_size;
    }
#endif
private:
    std::vector<T> m_vec;
    typename std::vector<T>::iterator m_front;
    typename std::vector<T>::iterator m_back;
    std::size_t m_size;
};

template <typename T>
struct less_than_x
{
    less_than_x(const std::vector<V2d<T> >& vertices)
        : vertices(vertices)
    {}
    bool operator()(const VertInd a, const VertInd b) const
    {
        return vertices[a].x < vertices[b].x;
    }
    const std::vector<V2d<T> >& vertices;
};

template <typename T>
struct less_than_y
{
    less_than_y(const std::vector<V2d<T> >& vertices)
        : vertices(vertices)
    {}
    bool operator()(const VertInd a, const VertInd b) const
    {
        return vertices[a].y < vertices[b].y;
    }
    const std::vector<V2d<T> >& vertices;
};

} // namespace detail

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::insertVertices_KDTreeBFS(
    VertInd superGeomVertCount,
    V2d<T> boxMin,
    V2d<T> boxMax)
{
    // calculate original indices
    const VertInd vertexCount = vertices.size() - superGeomVertCount;
    if(vertexCount <= 0)
        return;
    std::vector<VertInd> ii(vertexCount);
    detail::iota(ii.begin(), ii.end(), superGeomVertCount);

    typedef std::vector<VertInd>::iterator It;
    detail::FixedCapacityQueue<tuple<It, It, V2d<T>, V2d<T>, VertInd> > queue(
        detail::maxQueueLengthBFSKDTree(vertexCount));
    queue.push(make_tuple(ii.begin(), ii.end(), boxMin, boxMax, VertInd(0)));

    It first, last;
    V2d<T> newBoxMin, newBoxMax;
    VertInd parent, mid;

    const detail::less_than_x<T> cmpX(vertices);
    const detail::less_than_y<T> cmpY(vertices);

    while(!queue.empty())
    {
        tie(first, last, boxMin, boxMax, parent) = queue.front();
        queue.pop();
        assert(first != last);

        const std::ptrdiff_t len = std::distance(first, last);
        if(len == 1)
        {
            insertVertex(*first, parent);
            continue;
        }
        const It midIt = first + len / 2;
        if(boxMax.x - boxMin.x >= boxMax.y - boxMin.y)
        {
            detail::portable_nth_element(first, midIt, last, cmpX);
            mid = *midIt;
            const T split = vertices[mid].x;
            newBoxMin.x = split;
            newBoxMin.y = boxMin.y;
            newBoxMax.x = split;
            newBoxMax.y = boxMax.y;
        }
        else
        {
            detail::portable_nth_element(first, midIt, last, cmpY);
            mid = *midIt;
            const T split = vertices[mid].y;
            newBoxMin.x = boxMin.x;
            newBoxMin.y = split;
            newBoxMax.x = boxMax.x;
            newBoxMax.y = split;
        }
        insertVertex(mid, parent);
        if(first != midIt)
        {
            queue.push(make_tuple(first, midIt, boxMin, newBoxMax, mid));
        }
        if(midIt + 1 != last)
        {
            queue.push(make_tuple(midIt + 1, last, newBoxMin, boxMax, mid));
        }
    }
}

template <typename T, typename TNearPointLocator>
bool Triangulation<T, TNearPointLocator>::hasEdge(
    const VertInd a,
    const VertInd b) const
{
    const TriInd triStart = m_vertTris[a];
    assert(triStart != noNeighbor);
    TriInd iT = triStart;
    VertInd iV = noVertex;
    do
    {
        const Triangle& t = triangles[iT];
        tie(iT, iV) = t.next(a);
        assert(iT != noNeighbor);
        if(iV == b)
            return true;
    } while(iT != triStart);
    return false;
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::setAdjacentTriangle(
    const VertInd v,
    const TriInd t)
{
    assert(t != noNeighbor);
    m_vertTris[v] = t;
    assert(
        triangles[t].vertices[0] == v || triangles[t].vertices[1] == v ||
        triangles[t].vertices[2] == v);
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::pivotVertexTriangleCW(const VertInd v)
{
    assert(m_vertTris[v] != noNeighbor);
    m_vertTris[v] = triangles[m_vertTris[v]].next(v).first;
    assert(m_vertTris[v] != noNeighbor);
    assert(
        triangles[m_vertTris[v]].vertices[0] == v ||
        triangles[m_vertTris[v]].vertices[1] == v ||
        triangles[m_vertTris[v]].vertices[2] == v);
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::removeAdjacentTriangle(
    const VertInd v)
{
    m_vertTris[v] = noNeighbor;
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::tryAddVertexToLocator(const VertInd v)
{
    if(!m_nearPtLocator.empty()) // only if locator is initialized already
        m_nearPtLocator.addPoint(v, vertices);
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::tryInitNearestPointLocator()
{
    if(!vertices.empty() && m_nearPtLocator.empty())
    {
        m_nearPtLocator.initialize(vertices);
    }
}

} // namespace CDT
