/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @file
 * Triangulation class - implementation
 */
#ifndef CDT_pDqrlveWIOrIWeUCkPqX
#define CDT_pDqrlveWIOrIWeUCkPqX

#include "Triangulation.h"
#include "portable_nth_element.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <stdexcept>

CDT_ENSURE_PRECISE_MATH_FOR_CONSTRUCTIONS

namespace CDT
{

typedef std::deque<TriInd> TriDeque;

namespace detail
{

/// Turn a vector into a queue, preserving order
inline EdgeQueue toQueue(const EdgeVec& edges)
{
    return EdgeQueue(EdgeQueue::container_type(edges.begin(), edges.end()));
}

/// Sort a vector and remove its duplicate elements
template <typename TVec>
void sortUnique(TVec& v)
{
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
}

namespace defaults
{

const VertexInsertionOrder::Enum vertexInsertionOrder =
    VertexInsertionOrder::Auto;
const IntersectingConstraintEdges::Enum intersectingEdgesStrategy =
    IntersectingConstraintEdges::NotAllowed;
const float minDistToConstraintEdge(0);

} // namespace defaults

} // namespace detail

CDT_INLINE_IF_HEADER_ONLY Unrefined::Unrefined()
    : shortEdgeTriangles(0)
    , circumcenterOutside(0)
    , circumcenterOnVertex(0)
    , sharpFixedCorner(0)
    , shortEdges(0)
    , splitVertexInvalid(0)
{}

template <typename T, typename TNearPointLocator>
Triangulation<T, TNearPointLocator>::Triangulation()
    : m_vertexInsertionOrder(detail::defaults::vertexInsertionOrder)
    , m_intersectingEdgesStrategy(detail::defaults::intersectingEdgesStrategy)
    , m_minDistToConstraintEdge(detail::defaults::minDistToConstraintEdge)
#ifdef CDT_ENABLE_CALLBACK_HANDLER
    , m_callbackHandler(NULL)
#endif
{}

template <typename T, typename TNearPointLocator>
Triangulation<T, TNearPointLocator>::Triangulation(
    const VertexInsertionOrder::Enum vertexInsertionOrder)
    : m_vertexInsertionOrder(vertexInsertionOrder)
    , m_intersectingEdgesStrategy(detail::defaults::intersectingEdgesStrategy)
    , m_minDistToConstraintEdge(detail::defaults::minDistToConstraintEdge)
#ifdef CDT_ENABLE_CALLBACK_HANDLER
    , m_callbackHandler(NULL)
#endif
{}

template <typename T, typename TNearPointLocator>
Triangulation<T, TNearPointLocator>::Triangulation(
    const VertexInsertionOrder::Enum vertexInsertionOrder,
    const IntersectingConstraintEdges::Enum intersectingEdgesStrategy,
    const T minDistToConstraintEdge)
    : m_vertexInsertionOrder(vertexInsertionOrder)
    , m_intersectingEdgesStrategy(intersectingEdgesStrategy)
    , m_minDistToConstraintEdge(minDistToConstraintEdge)
#ifdef CDT_ENABLE_CALLBACK_HANDLER
    , m_callbackHandler(NULL)
#endif
{}

template <typename T, typename TNearPointLocator>
Triangulation<T, TNearPointLocator>::Triangulation(
    const VertexInsertionOrder::Enum vertexInsertionOrder,
    const TNearPointLocator& nearPtLocator,
    const IntersectingConstraintEdges::Enum intersectingEdgesStrategy,
    const T minDistToConstraintEdge)
    : m_nearPtLocator(nearPtLocator)
    , m_vertexInsertionOrder(vertexInsertionOrder)
    , m_intersectingEdgesStrategy(intersectingEdgesStrategy)
    , m_minDistToConstraintEdge(minDistToConstraintEdge)
#ifdef CDT_ENABLE_CALLBACK_HANDLER
    , m_callbackHandler(NULL)
#endif
{}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::eraseSuperTriangle()
{
    finalizeTriangulation(collectSuperTriangle());
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::eraseOuterTriangles()
{
    finalizeTriangulation(collectOuterTriangles());
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::eraseOuterTrianglesAndHoles()
{
    finalizeTriangulation(collectOuterTrianglesAndHoles());
}

template <typename T, typename TNearPointLocator>
TriIndUSet Triangulation<T, TNearPointLocator>::collectSuperTriangle() const
{
    if(isFinalized())
        handleException(FinalizedError(CDT_SOURCE_LOCATION));

    // find triangles adjacent to super-triangle's vertices
    TriIndUSet toErase;
    for(TriInd iT(0); iT < TriInd(triangles.size()); ++iT)
    {
        if(touchesSuperTriangle(triangles[iT]))
            toErase.insert(iT);
    }
    return toErase;
}

template <typename T, typename TNearPointLocator>
TriIndUSet Triangulation<T, TNearPointLocator>::collectOuterTriangles() const
{
    if(isFinalized())
        handleException(FinalizedError(CDT_SOURCE_LOCATION));

    assert(m_vertTris[0] != noNeighbor);
    const std::stack<TriInd> seed(std::deque<TriInd>(1, m_vertTris[0]));
    return growToBoundary(seed);
}

template <typename T, typename TNearPointLocator>
TriIndUSet
Triangulation<T, TNearPointLocator>::collectOuterTrianglesAndHoles() const
{
    if(isFinalized())
        handleException(FinalizedError(CDT_SOURCE_LOCATION));

    const std::vector<LayerDepth> triDepths = calculateTriangleDepths();
    TriIndUSet toErase;
    toErase.reserve(triangles.size());
    for(std::size_t iT = 0; iT != triangles.size(); ++iT)
    {
        if(triDepths[iT] % 2 == 0)
            toErase.insert(static_cast<TriInd>(iT));
    }
    return toErase;
}

/// Remap removing super-triangle: subtract 3 from vertices
inline Edge RemapNoSuperTriangle(const Edge& e)
{
    return Edge(
        VertInd(e.v1() - nSuperTriVerts), VertInd(e.v2() - nSuperTriVerts));
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
const TriIndVec& Triangulation<T, TNearPointLocator>::VertTrisInternal() const
{
    return m_vertTris;
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::finalizeTriangulation(
    const TriIndUSet& removedTriangles)
{
    if(isFinalized())
        handleException(FinalizedError(CDT_SOURCE_LOCATION));

    m_vertTris = TriIndVec();
    // remove super-triangle
    vertices.erase(vertices.begin(), vertices.begin() + nSuperTriVerts);
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
        typedef unordered_map<Edge, BoundaryOverlapCount>::const_iterator It;
        for(It it = overlapCount.begin(); it != overlapCount.end(); ++it)
        {
            updatedOverlapCount.insert(
                std::make_pair(RemapNoSuperTriangle(it->first), it->second));
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
            for(EdgeVec::iterator eeIt = ee.begin(); eeIt != ee.end(); ++eeIt)
            {
                *eeIt = RemapNoSuperTriangle(*eeIt);
            }
            updatedPieceToOriginals.insert(
                std::make_pair(RemapNoSuperTriangle(it->first), ee));
        }
        pieceToOriginals = updatedPieceToOriginals;
    }
    // remove other triangles
    removeTriangles(removedTriangles);
    // adjust triangle vertices: account for removed super-triangle
    for(TriangleVec::iterator t = triangles.begin(); t != triangles.end(); ++t)
    {
        VerticesArr3& vv = t->vertices;
        for(VerticesArr3::iterator v = vv.begin(); v != vv.end(); ++v)
        {
            *v -= nSuperTriVerts;
        }
    }
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
TriInd Triangulation<T, TNearPointLocator>::addTriangle(const Triangle& t)
{
    const TriInd iT = trianglesCount();
    triangles.push_back(t);
    return iT;
}

template <typename T, typename TNearPointLocator>
TriInd Triangulation<T, TNearPointLocator>::addTriangle()
{
    return addTriangle(Triangle());
}

template <typename T, typename TNearPointLocator>
VertInd Triangulation<T, TNearPointLocator>::verticesCount() const
{
    return static_cast<VertInd>(vertices.size());
}

template <typename T, typename TNearPointLocator>
TriInd Triangulation<T, TNearPointLocator>::trianglesCount() const
{
    return static_cast<TriInd>(triangles.size());
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
void Triangulation<T, TNearPointLocator>::splitFixedEdge(
    const Edge& edge,
    const VertInd iSplitVert)
{
    // split constraint (fixed) edge that already exists in triangulation
    const Edge half1(edge.v1(), iSplitVert);
    const Edge half2(iSplitVert, edge.v2());
    // remove the edge that and add its halves
    fixedEdges.erase(edge);
    fixEdge(half1);
    fixEdge(half2);
    // maintain overlaps
    typedef unordered_map<Edge, BoundaryOverlapCount>::const_iterator It;
    const It overlapIt = overlapCount.find(edge);
    if(overlapIt != overlapCount.end())
    {
        overlapCount[half1] += overlapIt->second;
        overlapCount[half2] += overlapIt->second;
        overlapCount.erase(overlapIt);
    }
    // maintain piece-to-original mapping
    EdgeVec newOriginals(1, edge);
    const unordered_map<Edge, EdgeVec>::const_iterator originalsIt =
        pieceToOriginals.find(edge);
    if(originalsIt != pieceToOriginals.end())
    { // edge being split was split before: pass-through originals
        newOriginals = originalsIt->second;
        pieceToOriginals.erase(originalsIt);
    }
    detail::insert_unique(pieceToOriginals[half1], newOriginals);
    detail::insert_unique(pieceToOriginals[half2], newOriginals);
}

template <typename T, typename TNearPointLocator>
VertInd Triangulation<T, TNearPointLocator>::addSplitEdgeVertex(
    const Edge& edge,
    const V2d<T>& splitVert,
    const TriInd iT,
    const TriInd iTopo,
    const AddVertexType::Enum vertexType)
{
    // add a new point on the edge that splits an edge in two
    const VertInd iSplitVert = verticesCount();
    addNewVertex(splitVert, noNeighbor);

#ifdef CDT_ENABLE_CALLBACK_HANDLER
    if(m_callbackHandler)
    {
        m_callbackHandler->onAddVertexStart(iSplitVert, vertexType);
    }
#else
    (void)vertexType;
#endif

    std::stack<TriInd> triStack = insertVertexOnEdge(iSplitVert, iT, iTopo);
    // before the flips: isFlipNeeded needs the halves to be fixed
    splitFixedEdge(edge, iSplitVert);
    tryAddVertexToLocator(iSplitVert);
    ensureDelaunayByEdgeFlips(iSplitVert, triStack);
    return iSplitVert;
}

template <typename T, typename TNearPointLocator>
OptionalVertInd Triangulation<T, TNearPointLocator>::splitFixedEdgeAt(
    const Edge& edge,
    const V2d<T>& splitVert,
    const TriInd iT,
    const TriInd iTopo,
    const AddVertexType::Enum vertexType)
{
    if(!isEdgeSplitVertexValid(splitVert, iT, iTopo))
        return OptionalVertInd(noVertex);
    const VertInd iSplitVert =
        addSplitEdgeVertex(edge, splitVert, iT, iTopo, vertexType);
    return OptionalVertInd(iSplitVert);
}

template <typename T, typename TNearPointLocator>
bool Triangulation<T, TNearPointLocator>::isEdgeSplitVertexValid(
    const V2d<T>& splitVert,
    const TriInd iT,
    const TriInd iTopo) const
{
    // quadrilateral of the two triangles, CCW: v2 and v4 end the split edge
    const Triangle& t1 = triangles[iT];
    const Index i = opposedVertexInd(t1.neighbors, iTopo);
    const V2d<T>& v1 = vertices[t1.vertices[i]];
    const V2d<T>& v2 = vertices[t1.vertices[ccw(i)]];
    const V2d<T>& v4 = vertices[t1.vertices[cw(i)]];
    const Triangle& t2 = triangles[iTopo];
    const V2d<T>& v3 =
        vertices[t2.vertices[opposedVertexInd(t2.neighbors, iT)]];
    return locatePointLine(splitVert, v1, v2) == PtLineLocation::Left &&
           locatePointLine(splitVert, v2, v3) == PtLineLocation::Left &&
           locatePointLine(splitVert, v3, v4) == PtLineLocation::Left &&
           locatePointLine(splitVert, v4, v1) == PtLineLocation::Left;
}

template <typename T, typename TNearPointLocator>
Edge Triangulation<T, TNearPointLocator>::originalInputEdge(const Edge& e) const
{
    const Edge orig =
        pieceToOriginals.count(e) ? pieceToOriginals.at(e).front() : e;
    return Edge(
        VertInd(orig.v1() - nSuperTriVerts),
        VertInd(orig.v2() - nSuperTriVerts));
}

template <typename T, typename TNearPointLocator>
const Triangle&
Triangulation<T, TNearPointLocator>::triangleAt(const TriInd iT) const
{
    if(iT >= TriInd(triangles.size()))
        handleException(Error(
            iT == noNeighbor
                ? "Attempted reading no-neighbor sentinel value triangle"
                : "Triangle index " + CDT::to_string(iT) + " out of range " +
                      CDT::to_string(triangles.size()),
            CDT_SOURCE_LOCATION));
    return triangles[iT];
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::fixEdge(
    const Edge& edge,
    const Edge& originalEdge)
{
    fixEdge(edge);
    if(edge != originalEdge)
        detail::insert_unique(pieceToOriginals[edge], originalEdge);
}

namespace detail
{

template <typename T>
T lerp(const T& a, const T& b, const T t)
{
    return (T(1) - t) * a + t * b;
}

/// Whether the angle at apex between apex->a and apex->b is smaller than 60°
template <typename T>
bool isAngleAtApexSmall(const V2d<T>& apex, const V2d<T>& a, const V2d<T>& b)
{
    const T ux = a.x - apex.x, uy = a.y - apex.y;
    const T wx = b.x - apex.x, wy = b.y - apex.y;
    const T dot = ux * wx + uy * wy;
    // cos(60°) = 0.5
    return dot > T(0) && T(2) * dot > distance(apex, a) * distance(apex, b);
}

// Precondition: ab and cd intersect normally
template <typename T>
V2d<T> intersectionPosition(
    const V2d<T>& a,
    const V2d<T>& b,
    const V2d<T>& c,
    const V2d<T>& d)
{
    // note: for better accuracy we interpolate x and y separately
    // on a segment with the shortest x/y-projection correspondingly
    const T a_cd = predicates::orient2d(c.x, c.y, d.x, d.y, a.x, a.y);
    const T b_cd = predicates::orient2d(c.x, c.y, d.x, d.y, b.x, b.y);
    const T t_ab = a_cd / (a_cd - b_cd);

    const T c_ab = predicates::orient2d(a.x, a.y, b.x, b.y, c.x, c.y);
    const T d_ab = predicates::orient2d(a.x, a.y, b.x, b.y, d.x, d.y);
    const T t_cd = c_ab / (c_ab - d_ab);

    return V2d<T>(
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
    std::vector<TriangulatePseudoPolygonTask>& tppIterations)
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
    polyL.reserve(2);
    polyL.push_back(iA);
    polyL.push_back(iVL);
    polyR.reserve(2);
    polyR.push_back(iA);
    polyR.push_back(iVR);
    unordered_map<Edge, TriInd> outerTris;
    outerTris[Edge(iA, iVL)] = edgeNeighbor(t, iA, iVL);
    outerTris[Edge(iA, iVR)] = edgeNeighbor(t, iA, iVR);
    VertInd iV = iA;

    while(!t.containsVertex(iB))
    {
        const TriInd iTopo = opposedTriangle(t, iV);
        const Triangle& tOpo = triangleAt(iTopo);
        const VertInd iVopo = opposedVertex(tOpo, iT);

        switch(m_intersectingEdgesStrategy)
        {
        case IntersectingConstraintEdges::NotAllowed:
            if(fixedEdges.count(Edge(iVL, iVR)))
                handleException(IntersectingConstraintsError(
                    originalInputEdge(originalEdge),
                    originalInputEdge(Edge(iVL, iVR)),
                    CDT_SOURCE_LOCATION));
            break;
        case IntersectingConstraintEdges::TryResolve:
        {
            if(!fixedEdges.count(Edge(iVL, iVR)))
                break;
            // split edge at the intersection of two constraint edges
            const V2d<T> newV = detail::intersectionPosition(
                vertices[iA], vertices[iB], vertices[iVL], vertices[iVR]);
            const OptionalVertInd splitVert = splitFixedEdgeAt(
                Edge(iVL, iVR),
                newV,
                iT,
                iTopo,
                AddVertexType::FixedEdgesIntersection);
            if(!splitVert.hasValue())
                handleException(InvalidEdgeSplitVertex(
                    originalInputEdge(originalEdge),
                    originalInputEdge(Edge(iVL, iVR)),
                    CDT_SOURCE_LOCATION));
            const VertInd iNewVert = splitVert.value();
            // TODO: is it's possible to re-use pseudo-polygons
            //  for inserting [iA, iNewVert] edge half?
            remaining.push_back(Edge(iA, iNewVert));
            remaining.push_back(Edge(iNewVert, iB));
            return;
        }
        case IntersectingConstraintEdges::DontCheck:
            assert(!fixedEdges.count(Edge(iVL, iVR)));
            break;
        }

        const PtLineLocation::Enum loc =
            locatePointLine(vertices[iVopo], a, b, distanceTolerance);
        if(loc == PtLineLocation::Left)
        {
            const Edge e(polyL.back(), iVopo);
            const TriInd outer = edgeNeighbor(tOpo, e.v1(), e.v2());
            if(!outerTris.insert(std::make_pair(e, outer)).second)
                outerTris.at(e) = noNeighbor; // hanging edge detected
            polyL.push_back(iVopo);
            iV = iVL;
            iVL = iVopo;
        }
        else if(loc == PtLineLocation::Right)
        {
            const Edge e(polyR.back(), iVopo);
            const TriInd outer = edgeNeighbor(tOpo, e.v1(), e.v2());
            if(!outerTris.insert(std::make_pair(e, outer)).second)
                outerTris.at(e) = noNeighbor; // hanging edge detected
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
    outerTris[Edge(polyL.back(), iB)] = edgeNeighbor(t, polyL.back(), iB);
    outerTris[Edge(polyR.back(), iB)] = edgeNeighbor(t, polyR.back(), iB);
    polyL.push_back(iB);
    polyR.push_back(iB);

    assert(!intersected.empty());
    // make sure start/end vertices have a valid adjacent triangle
    // that is not intersected by an edge
    if(m_vertTris[iA] == intersected.front())
        pivotVertexTriangleCW(iA);
    if(m_vertTris[iB] == intersected.back())
        pivotVertexTriangleCW(iB);

    {
#ifdef CDT_ENABLE_CALLBACK_HANDLER
        if(m_callbackHandler)
        {
            m_callbackHandler->onReTriangulatePolygon(intersected);
        }
#endif

        // Triangulate pseudo-polygons on both sides
        std::reverse(polyR.begin(), polyR.end());

        // note: intersected triangles are re-used for new triangles
        // every triangulation of an n-gon has n − 2 triangles
        // even if outer polygon has hanging edges it holds
        assert(intersected.size() >= 2);
        const TriInd iTL = intersected.back();
        intersected.pop_back();
        const TriInd iTR = intersected.back();
        intersected.pop_back();

        triangulatePseudoPolygon(
            polyL, outerTris, iTL, iTR, intersected, tppIterations);
        triangulatePseudoPolygon(
            polyR, outerTris, iTR, iTL, intersected, tppIterations);
        assert(intersected.empty());
    }

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
    std::vector<TriangulatePseudoPolygonTask>& tppIterations)
{
#ifdef CDT_ENABLE_CALLBACK_HANDLER
    if(m_callbackHandler)
    {
        m_callbackHandler->onAddEdgeStart(edge);
    }
#endif

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
        fixEdge(edge);
        if(overlaps > 0)
            overlapCount[edge] = overlaps;
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
        fixEdge(edgePart);
        if(overlaps > 0)
            overlapCount[edgePart] = overlaps;
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
        const Triangle& tOpo = triangleAt(iTopo);
        const VertInd iVopo = opposedVertex(tOpo, iT);
        const V2d<T> vOpo = vertices[iVopo];

        switch(m_intersectingEdgesStrategy)
        {
        case IntersectingConstraintEdges::NotAllowed:
            if(fixedEdges.count(Edge(iVleft, iVright)))
                handleException(IntersectingConstraintsError(
                    originalInputEdge(edge),
                    originalInputEdge(Edge(iVleft, iVright)),
                    CDT_SOURCE_LOCATION));
            break;
        case IntersectingConstraintEdges::TryResolve:
        {
            if(!fixedEdges.count(Edge(iVleft, iVright)))
                break;
            // split edge at the intersection of two constraint edges
            const V2d<T> newV = detail::intersectionPosition(
                vertices[iA],
                vertices[iB],
                vertices[iVleft],
                vertices[iVright]);
            const OptionalVertInd splitVert = splitFixedEdgeAt(
                Edge(iVleft, iVright),
                newV,
                iT,
                iTopo,
                AddVertexType::FixedEdgesIntersection);
            if(!splitVert.hasValue())
                handleException(InvalidEdgeSplitVertex(
                    originalInputEdge(edge),
                    originalInputEdge(Edge(iVleft, iVright)),
                    CDT_SOURCE_LOCATION));
            const VertInd iNewVert = splitVert.value();
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
        case IntersectingConstraintEdges::DontCheck:
            assert(!fixedEdges.count(Edge(iVleft, iVright)));
            break;
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
    const VertInd iMid = verticesCount();
    const V2d<T>& start = vertices[iA];
    const V2d<T>& end = vertices[iB];
    addNewVertex(
        V2d<T>((start.x + end.x) / T(2), (start.y + end.y) / T(2)), noNeighbor);

#ifdef CDT_ENABLE_CALLBACK_HANDLER
    if(m_callbackHandler)
    {
        m_callbackHandler->onAddVertexStart(
            iMid, AddVertexType::FixedEdgeMidpoint);
    }
#endif

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
#ifdef CDT_ENABLE_CALLBACK_HANDLER
    if(m_callbackHandler)
    {
        m_callbackHandler->onAddEdgeStart(edge);
    }
#endif

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

    handleException(Error(
        "Could not find vertex triangle intersected by an edge.",
        CDT_SOURCE_LOCATION));
    return make_tuple(noNeighbor, noVertex, noVertex);
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::addSuperTriangle(const Box2d<T>& box)
{
    const V2d<T> center(
        (box.min.x + box.max.x) / T(2), (box.min.y + box.max.y) / T(2));
    const T w = box.max.x - box.min.x;
    const T h = box.max.y - box.min.y;
    T r = std::max(w, h); // incircle radius upper bound

    // Note: make sure radius is big enough. Constants chosen experimentally.
    // - for tiny bounding boxes: use 1.0 as the smallest radius
    // - multiply radius by 2.0 for extra safety margin
    r = std::max(T(2) * r, T(1));

    // Note: for very large floating point numbers rounding can lead to wrong
    // super-triangle coordinates. This is a very rare corner-case so the
    // handling is very primitive.
    { // note: '<=' means '==' but avoids the warning
        while(center.y <= center.y - r)
            r *= T(2);
    }

    const T R = T(2) * r;                       // excircle radius
    const T cos_30_deg = T(0.8660254037844386); // note: (std::sqrt(3.0) / 2.0)
    const T shiftX = R * cos_30_deg;
    const V2d<T> posV1(center.x - shiftX, center.y - r);
    const V2d<T> posV2(center.x + shiftX, center.y - r);
    const V2d<T> posV3(center.x, center.y + R);
    addNewVertex(posV1, TriInd(0));
    addNewVertex(posV2, TriInd(0));
    addNewVertex(posV3, TriInd(0));

#ifdef CDT_ENABLE_CALLBACK_HANDLER
    if(m_callbackHandler)
    {
        m_callbackHandler->onAddSuperTriangle();
    }
#endif

    addTriangle(
        Triangle(arr3(VertInd(0), VertInd(1), VertInd(2)), arr3(noNeighbor)));

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
    array<TriInd, 2> trisAt = walkingSearchTrianglesAt(iV1, startVertex);
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
        if(iTopo != noNeighbor && isFlipNeeded(iV1, iV2, iV3, iV4, true))
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
#ifdef CDT_ENABLE_CALLBACK_HANDLER
    if(m_callbackHandler)
    {
        m_callbackHandler->onAddVertexStart(iVert, AddVertexType::UserInput);
    }
#endif

    const array<TriInd, 2> trisAt = walkingSearchTrianglesAt(iVert, walkStart);
    std::stack<TriInd> triStack =
        trisAt[1] == noNeighbor
            ? insertVertexInsideTriangle(iVert, trisAt[0])
            : insertVertexOnEdge(iVert, trisAt[0], trisAt[1], true);
    ensureDelaunayByEdgeFlips(iVert, triStack);
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
        if(iTopo != noNeighbor && isFlipNeeded(iV1, iV2, iV3, iV4))
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

template <typename T, typename TNearPointLocator>
bool Triangulation<T, TNearPointLocator>::isSameOriginalEdge(
    const Edge& e1,
    const Edge& e2) const
{
    if(!fixedEdges.count(e1) || !fixedEdges.count(e2))
        return false;
    typedef unordered_map<Edge, EdgeVec>::const_iterator It;
    const It it1 = pieceToOriginals.find(e1);
    const It it2 = pieceToOriginals.find(e2);
    return (it1 == pieceToOriginals.end() ? e1 : it1->second.front()) ==
           (it2 == pieceToOriginals.end() ? e2 : it2->second.front());
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
    const VertInd iV1,
    const VertInd iV2,
    const VertInd iV3,
    const VertInd iV4,
    const bool doFlipFixedEdges) const
{
    if(!doFlipFixedEdges && fixedEdges.count(Edge(iV2, iV4)))
        return false; // flip not needed if the original edge is fixed
    // the flip would make a triangle out of two pieces of one input edge
    if(!fixedEdges.empty() &&
       (isSameOriginalEdge(Edge(iV1, iV2), Edge(iV2, iV3)) ||
        isSameOriginalEdge(Edge(iV3, iV4), Edge(iV4, iV1))))
        return false;
    const V2d<T>& v1 = vertices[iV1];
    const V2d<T>& v2 = vertices[iV2];
    const V2d<T>& v3 = vertices[iV3];
    const V2d<T>& v4 = vertices[iV4];
    // If flip-candidate edge touches super-triangle in-circumference
    // test has to be replaced with orient2d test against the line
    // formed by two non-artificial vertices (that don't belong to
    // super-triangle)
    if(iV1 < nSuperTriVerts) // flip-candidate edge touches super-triangle
    {
        // does original edge also touch super-triangle?
        if(iV2 < nSuperTriVerts)
            return locatePointLine(v2, v3, v4) == locatePointLine(v1, v3, v4);
        if(iV4 < nSuperTriVerts)
            return locatePointLine(v4, v2, v3) == locatePointLine(v1, v2, v3);
        return false; // original edge does not touch super-triangle
    }
    if(iV3 < nSuperTriVerts) // flip-candidate edge touches super-triangle
    {
        // does original edge also touch super-triangle?
        if(iV2 < nSuperTriVerts)
        {
            return locatePointLine(v2, v1, v4) == locatePointLine(v3, v1, v4);
        }
        if(iV4 < nSuperTriVerts)
        {
            return locatePointLine(v4, v2, v1) == locatePointLine(v3, v2, v1);
        }
        return false; // original edge does not touch super-triangle
    }
    // flip-candidate edge does not touch super-triangle
    if(iV2 < nSuperTriVerts)
        return locatePointLine(v2, v3, v4) == locatePointLine(v1, v3, v4);
    if(iV4 < nSuperTriVerts)
        return locatePointLine(v4, v2, v3) == locatePointLine(v1, v2, v3);
    return isInCircumcircle(v1, v2, v3, v4);
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
#ifdef CDT_ENABLE_CALLBACK_HANDLER
    if(m_callbackHandler)
    {
        m_callbackHandler->onFlipEdge(iT, iTopo);
    }
#endif

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
    t = Triangle(arr3(v4, v1, v3), arr3(n3, iTopo, n4));
    tOpo = Triangle(arr3(v2, v3, v1), arr3(n2, iT, n1));
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
bool Triangulation<T, TNearPointLocator>::isRefinementNeeded(
    const Triangle& tri,
    const RefinementCriterion::Enum refinementCriterion,
    const T refinementThreshold) const
{
    const V2d<T>& a = vertices[tri.vertices[0]];
    const V2d<T>& b = vertices[tri.vertices[1]];
    const V2d<T>& c = vertices[tri.vertices[2]];
    switch(refinementCriterion)
    {
    case RefinementCriterion::SmallestAngle:
        return smallestAngle(a, b, c) < refinementThreshold;
    case RefinementCriterion::LargestArea:
        return area(a, b, c) > refinementThreshold;
    }
    assert(false); // unreachable code
    return false;
}

template <typename T, typename TNearPointLocator>
bool Triangulation<T, TNearPointLocator>::isSmallestAngleFixed(
    const Triangle& tri) const
{
    Index iApex(0);
    T shortestSqLen = distanceSquared(
        vertices[tri.vertices[ccw(Index(0))]],
        vertices[tri.vertices[cw(Index(0))]]);
    for(Index i(1); i < Index(3); ++i)
    {
        const T sqLen = distanceSquared(
            vertices[tri.vertices[ccw(i)]], vertices[tri.vertices[cw(i)]]);
        if(sqLen < shortestSqLen)
        {
            shortestSqLen = sqLen;
            iApex = i;
        }
    }
    const VertInd apex = tri.vertices[iApex];
    return fixedEdges.count(Edge(apex, tri.vertices[ccw(iApex)])) &&
           fixedEdges.count(Edge(apex, tri.vertices[cw(iApex)]));
}

template <typename T, typename TNearPointLocator>
bool Triangulation<T, TNearPointLocator>::isEdgeEncroached(
    const Edge& edge) const
{
    TriInd iT, iTopo;
    tie(iT, iTopo) = edgeTriangles(edge.v1(), edge.v2());
    assert(iT != noNeighbor && iTopo != noNeighbor);
    const VertInd v1 = opposedVertex(triangles[iT], iTopo);
    const VertInd v2 = opposedVertex(triangles[iTopo], iT);
    const V2d<T>& edgeStart = vertices[edge.v1()];
    const V2d<T>& edgeEnd = vertices[edge.v2()];
    return detail::isEncroachingOnEdge(vertices[v1], edgeStart, edgeEnd) ||
           detail::isEncroachingOnEdge(vertices[v2], edgeStart, edgeEnd);
}

template <typename T, typename TNearPointLocator>
bool Triangulation<T, TNearPointLocator>::isEdgeEncroachedBy(
    const Edge& edge,
    const V2d<T>& v) const
{
    return detail::isEncroachingOnEdge(
        v, vertices[edge.v1()], vertices[edge.v2()]);
}

template <typename T, typename TNearPointLocator>
EdgeVec Triangulation<T, TNearPointLocator>::findEncroachedFixedEdges() const
{
    if(isFinalized())
        handleException(FinalizedError(CDT_SOURCE_LOCATION));

    // Search in all fixed edges to find encroached edges
    EdgeVec encroachedEdges;
    typedef EdgeUSet::const_iterator Iter;
    for(Iter it = fixedEdges.begin(); it != fixedEdges.end(); ++it)
    {
        const Edge edge = *it;
        if(isEdgeEncroached(edge))
        {
            encroachedEdges.push_back(edge);
        }
    }
    // fixedEdges is a hash set: its iteration order is platform-dependent,
    // so sort to keep refinement output deterministic
    std::sort(encroachedEdges.begin(), encroachedEdges.end());
    return encroachedEdges;
}

template <typename T, typename TNearPointLocator>
TriIndVec Triangulation<T, TNearPointLocator>::findUnrefinedTriangles(
    const RefinementCriterion::Enum refinementCriterion,
    const T refinementThreshold) const
{
    const bool hasSuperTriangle = !isFinalized();
    TriIndVec unrefinedTriangles;
    for(TriInd iT(0), n(triangles.size()); iT < n; ++iT)
    {
        const Triangle& t = triangles[iT];
        if(hasSuperTriangle && touchesSuperTriangle(t))
            continue;
        if(isRefinementNeeded(t, refinementCriterion, refinementThreshold))
            unrefinedTriangles.push_back(iT);
    }
    return unrefinedTriangles;
}

template <typename T, typename TNearPointLocator>
EdgeVec Triangulation<T, TNearPointLocator>::edgesEncroachedBy(
    const V2d<T>& v,
    const TriInd iT) const
{
    /*
     * A fixed edge not yet encroached by an existing vertex can only be
     * encroached by v if v is inside the circumscribed circle of the edge's
     * triangle on v's side: the circumscribed circles of an edge's two
     * triangles cover the edge's diametral circle. Such triangles are visible
     * from v, so growing from the triangle at v without crossing fixed edges
     * reaches all of them.
     */
    EdgeVec encroachedEdges;
    TriIndUSet traversed;
    std::stack<TriInd> toTraverse;
    toTraverse.push(iT);
    traversed.insert(iT);
    while(!toTraverse.empty())
    {
        const Triangle& t = triangles[toTraverse.top()];
        toTraverse.pop();
        for(Index i(0); i < Index(3); ++i)
        {
            const Edge opEdge(t.vertices[ccw(i)], t.vertices[cw(i)]);
            if(fixedEdges.count(opEdge))
            {
                // both edge's triangles can be reached: avoid duplicate edges
                if(isEdgeEncroachedBy(opEdge, v))
                    detail::insert_unique(encroachedEdges, opEdge);
                continue;
            }
            const TriInd iN = t.neighbors[opoNbr(i)];
            if(iN == noNeighbor || traversed.count(iN))
                continue;
            const Triangle& n = triangles[iN];
            if(!isInCircumcircle(
                   v,
                   vertices[n.vertices[0]],
                   vertices[n.vertices[1]],
                   vertices[n.vertices[2]]))
                continue;
            traversed.insert(iN);
            toTraverse.push(iN);
        }
    }
    return encroachedEdges;
}

template <typename T, typename TNearPointLocator>
TriIndVec Triangulation<T, TNearPointLocator>::resolveEncroachedEdges(
    EdgeQueue encroachedEdges,
    VertInd& remainingVertexBudget,
    const VertInd steinerVerticesOffset,
    const V2d<T>* const circumcenterOrNull,
    const RefinementCriterion::Enum refinementCriterion,
    const T badTriangleThreshold,
    TriIndUSet* const toEraseOrNull,
    const T minEdgeLength,
    Unrefined& unrefined)
{
    std::vector<TriInd> badTriangles;

    while(!encroachedEdges.empty() && remainingVertexBudget > 0)
    {
        const Edge edge = encroachedEdges.front();
        encroachedEdges.pop();
        if(fixedEdges.find(edge) == fixedEdges.end())
        {
            continue;
        }
        // give up on already-too-short edges rather than split forever
        if(distance(vertices[edge.v1()], vertices[edge.v2()]) <= minEdgeLength)
        {
            ++unrefined.shortEdges;
            continue;
        }
        // split encroached edge
        const OptionalVertInd splitVert = splitEncroachedEdge(
            edge, steinerVerticesOffset, toEraseOrNull, unrefined);
        if(!splitVert.hasValue())
            continue;
        const VertInd iSplitVert = splitVert.value();
        --remainingVertexBudget;

        const TriInd start = m_vertTris[iSplitVert];
        TriInd iT = start;
        do
        {
            const Triangle& t = triangles[iT];
            const bool isMarkedForErasure =
                toEraseOrNull && toEraseOrNull->count(iT);
            if(circumcenterOrNull && !isMarkedForErasure &&
               !touchesSuperTriangle(t) &&
               isRefinementNeeded(t, refinementCriterion, badTriangleThreshold))
            {
                badTriangles.push_back(iT);
            }
            for(Index i(0); i < Index(3); ++i)
            {
                const Edge triEdge(t.vertices[i], t.vertices[cw(i)]);
                if(fixedEdges.find(triEdge) == fixedEdges.end())
                    continue;
                if(isEdgeEncroached(triEdge) ||
                   (circumcenterOrNull &&
                    isEdgeEncroachedBy(triEdge, *circumcenterOrNull)))
                {
                    encroachedEdges.push(triEdge);
                }
            }
            iT = t.next(iSplitVert).first;
        } while(iT != start);
    }
    return badTriangles;
}

template <typename T, typename TNearPointLocator>
OptionalVertInd Triangulation<T, TNearPointLocator>::splitEncroachedEdge(
    const Edge edge,
    const VertInd steinerVerticesOffset,
    TriIndUSet* const toEraseOrNull,
    Unrefined& unrefined)
{
    const V2d<T>& start = vertices[edge.v1()];
    const V2d<T>& end = vertices[edge.v2()];

    TriInd iT, iTopo;
    tie(iT, iTopo) = edgeTriangles(edge.v1(), edge.v2());
    assert(iT != noNeighbor && iTopo != noNeighbor);

    T split = T(0.5);
    // Edge sorts its vertices and Steiner ones are appended after the input
    // ones, so only v1 can be an input vertex here
    if(edge.v1() < steinerVerticesOffset &&
       edge.v2() >= steinerVerticesOffset &&
       hasAnotherFixedEdgeAtSmallAngle(edge.v1(), edge))
    {
        // In Ruppert's paper, he used D(0.01) factor to divide edge length, but
        // that introduces FP rounding errors, so it's avoided.
        const T len = distance(start, end);
        const T d = len / T(2);
        // Find the splitting distance
        T nearestPowerOfTwo = T(1);
        while(d > nearestPowerOfTwo)
        {
            nearestPowerOfTwo *= T(2);
        }
        while(d < T(0.75) * nearestPowerOfTwo)
        {
            nearestPowerOfTwo *= T(0.5);
        }
        split = nearestPowerOfTwo / len;
    }

    const V2d<T> mid = V2d<T>(
        detail::lerp(start.x, end.x, split),
        detail::lerp(start.y, end.y, split));

    const OptionalVertInd iMid = splitFixedEdgeAt(
        edge, mid, iT, iTopo, AddVertexType::RefinementEdgeSplit);
    if(!iMid.hasValue())
    {
        ++unrefined.splitVertexInvalid;
    }
    else if(toEraseOrNull)
    {
        // splitting reuses iT/iTopo for two of the four resulting triangles and
        // appends the other two: propagate erasure marks to the new triangles
        if(toEraseOrNull->count(iT))
            toEraseOrNull->insert(TriInd(triangles.size() - 2));
        if(toEraseOrNull->count(iTopo))
            toEraseOrNull->insert(TriInd(triangles.size() - 1));
    }
    return iMid;
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
#ifdef CDT_ENABLE_CALLBACK_HANDLER
    if(m_callbackHandler)
    {
        m_callbackHandler->onFlipEdge(iT, iTopo);
    }
#endif

    // change vertices and neighbors
    triangles[iT] = Triangle(arr3(v4, v1, v3), arr3(n3, iTopo, n4));
    triangles[iTopo] = Triangle(arr3(v2, v3, v1), arr3(n2, iT, n1));
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

#ifdef CDT_ENABLE_CALLBACK_HANDLER
    if(m_callbackHandler)
    {
        m_callbackHandler->onInsertVertexInsideTriangle(iT, iNewT1, iNewT2);
    }
#endif

    Triangle& t = triangles[iT];
    const array<VertInd, 3> vv = t.vertices;
    const array<TriInd, 3> nn = t.neighbors;
    const VertInd v1 = vv[0], v2 = vv[1], v3 = vv[2];
    const TriInd n1 = nn[0], n2 = nn[1], n3 = nn[2];
    // make two new triangles and convert current triangle to 3rd new
    // triangle
    triangles[iNewT1] = Triangle(arr3(v2, v3, v), arr3(n2, iNewT2, iT));
    triangles[iNewT2] = Triangle(arr3(v3, v1, v), arr3(n3, iT, iNewT1));
    t = Triangle(arr3(v1, v2, v), arr3(n1, iNewT1, iNewT2));
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
    TriInd iT2,
    const bool doHandleFixedSplitEdge)
{
    const TriInd iTnew1 = addTriangle();
    const TriInd iTnew2 = addTriangle();

#ifdef CDT_ENABLE_CALLBACK_HANDLER
    if(m_callbackHandler)
    {
        m_callbackHandler->onInsertVertexOnEdge(iT1, iT2, iTnew1, iTnew2);
    }
#endif

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
    t1 = Triangle(arr3(v, v1, v2), arr3(iTnew1, n1, iT2));
    t2 = Triangle(arr3(v, v2, v3), arr3(iT1, n2, iTnew2));
    triangles[iTnew1] = Triangle(arr3(v, v4, v1), arr3(iTnew2, n4, iT1));
    triangles[iTnew2] = Triangle(arr3(v, v3, v4), arr3(iT2, n3, iTnew1));
    // adjust adjacent triangles
    setAdjacentTriangle(v, iT1);
    setAdjacentTriangle(v4, iTnew1);
    // adjust neighboring triangles and vertices
    changeNeighbor(n4, iT1, iTnew1);
    changeNeighbor(n3, iT2, iTnew2);
    // properly handle the case when the split edge is a fixed edge
    if(doHandleFixedSplitEdge)
    {
        const Edge sharedEdge(v2, v4);
        if(fixedEdges.count(sharedEdge))
            splitFixedEdge(sharedEdge, v);
    }
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
    handleException(
        Error("No triangle was found at position", CDT_SOURCE_LOCATION));
    return out;
}

template <typename T, typename TNearPointLocator>
OptionalTriInd Triangulation<T, TNearPointLocator>::walkTriangles(
    const VertInd startVertex,
    const V2d<T>& pos) const
{
    // begin walk in search of triangle at pos
    TriInd currTri = m_vertTris[startVertex];
    bool found = false;
    bool isOutside = false;
    detail::SplitMix64RandGen prng;
    while(!found)
    {
        const Triangle& t = triangles[currTri];
        found = true;
        isOutside = false;
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
            if(edgeCheck == PtLineLocation::Right)
            {
                if(iN != noNeighbor)
                {
                    found = false;
                    currTri = iN;
                    break;
                }
                // crossing this edge would leave the triangulated area:
                // other edges can still lead to a triangle containing pos
                isOutside = true;
            }
        }
    }
    return isOutside ? OptionalTriInd(noNeighbor) : OptionalTriInd(currTri);
}

template <typename T, typename TNearPointLocator>
array<TriInd, 2> Triangulation<T, TNearPointLocator>::walkingSearchTrianglesAt(
    const VertInd iV,
    const VertInd startVertex) const
{
    const V2d<T> v = vertices[iV];
    array<TriInd, 2> out = {noNeighbor, noNeighbor};
    const OptionalTriInd walkResult = walkTriangles(startVertex, v);
    if(!walkResult.hasValue())
    {
        handleException(
            Error("No triangle was found at position", CDT_SOURCE_LOCATION));
    }
    const TriInd iT = walkResult.value();
    // Finished walk, locate point in current triangle
    const Triangle& t = triangles[iT];
    const V2d<T>& v1 = vertices[t.vertices[0]];
    const V2d<T>& v2 = vertices[t.vertices[1]];
    const V2d<T>& v3 = vertices[t.vertices[2]];
    const PtTriLocation::Enum loc = locatePointTriangle(v, v1, v2, v3);

    if(loc == PtTriLocation::Outside)
    {
        handleException(
            Error("No triangle was found at position", CDT_SOURCE_LOCATION));
    }
    if(loc == PtTriLocation::OnVertex)
    {
        const VertInd iDupe = v1 == v   ? t.vertices[0]
                              : v2 == v ? t.vertices[1]
                                        : t.vertices[2];
        handleException(DuplicateVertexError(
            VertInd(iV - nSuperTriVerts),
            VertInd(iDupe - nSuperTriVerts),
            CDT_SOURCE_LOCATION));
    }

    out[0] = iT;
    if(isOnEdge(loc))
        out[1] = t.neighbors[edgeNeighbor(loc)];
    return out;
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
void Triangulation<T, TNearPointLocator>::triangulatePseudoPolygon(
    const std::vector<VertInd>& poly,
    unordered_map<Edge, TriInd>& outerTris,
    TriInd iT,
    TriInd iN,
    std::vector<TriInd>& trianglesToReuse,
    std::vector<TriangulatePseudoPolygonTask>& iterations)
{
    assert(poly.size() > 2);
    // note: uses iteration instead of recursion to avoid stack overflows
    iterations.clear();
    iterations.push_back(make_tuple(
        IndexSizeType(0),
        static_cast<IndexSizeType>(poly.size() - 1),
        iT,
        iN,
        Index(0)));
    while(!iterations.empty())
    {
        triangulatePseudoPolygonIteration(
            poly, outerTris, trianglesToReuse, iterations);
    }
}

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::triangulatePseudoPolygonIteration(
    const std::vector<VertInd>& poly,
    unordered_map<Edge, TriInd>& outerTris,
    std::vector<TriInd>& trianglesToReuse,
    std::vector<TriangulatePseudoPolygonTask>& iterations)
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
    // note: second part needs to be pushed on stack first to be processed first

    // second part: points after the Delaunay point
    if(iB - iC > 1)
    {
        assert(!trianglesToReuse.empty());
        const TriInd iNext = trianglesToReuse.back();
        trianglesToReuse.pop_back();
        iterations.push_back(make_tuple(iC, iB, iNext, iT, Index(1)));
    }
    else // pseudo-poly is reduced to a single outer edge
    {
        const Edge outerEdge(b, c);
        const TriInd outerTri = outerTris.at(outerEdge);
        if(outerTri != noNeighbor)
        {
            assert(outerTri != iT);
            t.neighbors[1] = outerTri;
            changeNeighbor(outerTri, c, b, iT);
        }
        else
            outerTris.at(outerEdge) = iT;
    }
    // first part: points before the Delaunay point
    if(iC - iA > 1)
    { // add next triangle and add another iteration
        assert(!trianglesToReuse.empty());
        const TriInd iNext = trianglesToReuse.back();
        trianglesToReuse.pop_back();
        iterations.push_back(make_tuple(iA, iC, iNext, iT, Index(2)));
    }
    else
    { // pseudo-poly is reduced to a single outer edge
        const Edge outerEdge(c, a);
        const TriInd outerTri = outerTris.at(outerEdge);
        if(outerTri != noNeighbor)
        {
            assert(outerTri != iT);
            t.neighbors[2] = outerTri;
            changeNeighbor(outerTri, c, a, iT);
        }
        else
            outerTris.at(outerEdge) = iT;
    }
    // Finalize triangle
    // note: only when triangle is finalized to we add it as a neighbor to
    // parent to maintain triangulation topology consistency
    triangles[iParent].neighbors[iInParent] = iT;
    t.neighbors[0] = iParent;
    t.vertices = arr3(a, b, c);
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

#ifdef CDT_ENABLE_CALLBACK_HANDLER
template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::setCallbackHandler(
    ICallbackHandler* callbackHandler)
{
    m_callbackHandler = callbackHandler;
}
#endif

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::insertVertices_AsProvided(
    VertInd superGeomVertCount)
{
    for(VertInd iV = superGeomVertCount; iV < vertices.size(); ++iV)
    {
#ifdef CDT_ENABLE_CALLBACK_HANDLER
        if(m_callbackHandler && m_callbackHandler->isAbortCalculation())
        {
            return;
        }
#endif
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
#ifdef CDT_ENABLE_CALLBACK_HANDLER
        if(m_callbackHandler && m_callbackHandler->isAbortCalculation())
        {
            return;
        }
#endif
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
class less_than_x
{
    const std::vector<V2d<T> >& m_vertices;

public:
    less_than_x(const std::vector<V2d<T> >& vertices)
        : m_vertices(vertices)
    {}
    bool operator()(const VertInd a, const VertInd b) const
    {
        return m_vertices[a].x < m_vertices[b].x;
    }
};

template <typename T>
class less_than_y
{
    const std::vector<V2d<T> >& m_vertices;

public:
    less_than_y(const std::vector<V2d<T> >& vertices)
        : m_vertices(vertices)
    {}
    bool operator()(const VertInd a, const VertInd b) const
    {
        return m_vertices[a].y < m_vertices[b].y;
    }
};

} // namespace detail

template <typename T, typename TNearPointLocator>
void Triangulation<T, TNearPointLocator>::insertVertices_KDTreeBFS(
    VertInd superGeomVertCount,
    Box2d<T> box)
{
    // calculate original indices
    const VertInd vertexCount(verticesCount() - superGeomVertCount);
    if(vertexCount <= VertInd(0))
        return;
    std::vector<VertInd> ii(vertexCount);
    detail::iota(ii.begin(), ii.end(), superGeomVertCount);

    typedef std::vector<VertInd>::iterator It;
    detail::FixedCapacityQueue<tuple<It, It, V2d<T>, V2d<T>, VertInd> > queue(
        detail::maxQueueLengthBFSKDTree(vertexCount));
    queue.push(make_tuple(ii.begin(), ii.end(), box.min, box.max, VertInd(0)));

    It first, last;
    V2d<T> newBoxMin, newBoxMax;
    VertInd parent, mid;

    const detail::less_than_x<T> cmpX(vertices);
    const detail::less_than_y<T> cmpY(vertices);

    while(!queue.empty())
    {
#ifdef CDT_ENABLE_CALLBACK_HANDLER
        if(m_callbackHandler && m_callbackHandler->isAbortCalculation())
        {
            return;
        }
#endif
        tie(first, last, box.min, box.max, parent) = queue.front();
        queue.pop();
        assert(first != last);

        const std::ptrdiff_t len = std::distance(first, last);
        if(len == 1)
        {
            insertVertex(*first, parent);
            continue;
        }
        const It midIt = first + len / 2;
        if(box.max.x - box.min.x >= box.max.y - box.min.y)
        {
            detail::portable_nth_element(first, midIt, last, cmpX);
            mid = *midIt;
            const T split = vertices[mid].x;
            newBoxMin.x = split;
            newBoxMin.y = box.min.y;
            newBoxMax.x = split;
            newBoxMax.y = box.max.y;
        }
        else
        {
            detail::portable_nth_element(first, midIt, last, cmpY);
            mid = *midIt;
            const T split = vertices[mid].y;
            newBoxMin.x = box.min.x;
            newBoxMin.y = split;
            newBoxMax.x = box.max.x;
            newBoxMax.y = split;
        }
        insertVertex(mid, parent);
        if(first != midIt)
        {
            queue.push(make_tuple(first, midIt, box.min, newBoxMax, mid));
        }
        if(midIt + 1 != last)
        {
            queue.push(make_tuple(midIt + 1, last, newBoxMin, box.max, mid));
        }
    }
}

template <typename T, typename TNearPointLocator>
std::pair<TriInd, TriInd> Triangulation<T, TNearPointLocator>::edgeTriangles(
    const VertInd a,
    const VertInd b) const
{
    const TriInd triStart = m_vertTris[a];
    assert(triStart != noNeighbor);
    TriInd iT = triStart, iTNext = triStart;
    VertInd iV = noVertex;
    do
    {
        const Triangle& t = triangles[iT];
        tie(iTNext, iV) = t.next(a);
        assert(iTNext != noNeighbor);
        if(iV == b)
        {
            return std::make_pair(iT, iTNext);
        }
        iT = iTNext;
    } while(iT != triStart);
    return std::make_pair(noNeighbor, noNeighbor);
}

template <typename T, typename TNearPointLocator>
bool Triangulation<T, TNearPointLocator>::hasEdge(
    const VertInd a,
    const VertInd b) const
{
    return edgeTriangles(a, b).first != invalidIndexSizeType;
}

/// Checks whether vertex v has a fixed edge, other than excludeEdge, incident
/// to it at a small angle: recognizes a subsegment endpoint as a shared corner
/// even after v's segments have already been split.
template <typename T, typename TNearPointLocator>
bool Triangulation<T, TNearPointLocator>::hasAnotherFixedEdgeAtSmallAngle(
    const VertInd v,
    const Edge& excludeEdge) const
{
    // splits stay on the original segment: its direction from v is unchanged
    const VertInd iVOther =
        excludeEdge.v1() == v ? excludeEdge.v2() : excludeEdge.v1();
    const TriInd triStart = m_vertTris[v];
    assert(triStart != noNeighbor);
    TriInd iT = triStart;
    do
    {
        const Triangle& t = triangles[iT];
        TriInd iTNext;
        VertInd iV;
        tie(iTNext, iV) = t.next(v);
        const Edge candidate(v, iV);
        if(candidate != excludeEdge && fixedEdges.count(candidate) &&
           detail::isAngleAtApexSmall(
               vertices[v], vertices[iVOther], vertices[iV]))
        {
            return true;
        }
        iT = iTNext;
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

template <typename T, typename TNearPointLocator>
Unrefined Triangulation<T, TNearPointLocator>::refineTriangles(
    const VertInd maxVerticesToInsert,
    const RefinementCriterion::Enum refinementCriterion,
    const T refinementThreshold,
    TriIndUSet* const toEraseOrNull,
    const T minEdgeLength)
{
    if(isFinalized())
        handleException(FinalizedError(CDT_SOURCE_LOCATION));

    tryInitNearestPointLocator();

    Unrefined unrefined;
    VertInd remainingVertexBudget = maxVerticesToInsert;
    const VertInd steinerVerticesOffset = VertInd(vertices.size());

    // split all the encroached constrained (fixed) edges
    resolveEncroachedEdges(
        detail::toQueue(findEncroachedFixedEdges()),
        remainingVertexBudget,
        steinerVerticesOffset,
        NULL, // no circumcenter yet: only edge-vs-edge encroachment matters
        refinementCriterion,
        refinementThreshold,
        toEraseOrNull,
        minEdgeLength,
        unrefined);

    // refine triangulation by inserting bad-quality triangles' circumcenters
    TriIndQueue badTriangles;
    for(TriInd iT(0), n(triangles.size()); iT < n; ++iT)
    {
        const Triangle& t = triangles[iT];
        if(!touchesSuperTriangle(t) &&
           !(toEraseOrNull && toEraseOrNull->count(iT)) &&
           isRefinementNeeded(t, refinementCriterion, refinementThreshold))
        {
            badTriangles.push(iT);
        }
    }

    while(!badTriangles.empty() && remainingVertexBudget > 0)
    {
        const TriInd iT = badTriangles.front();
        badTriangles.pop();
        if(toEraseOrNull && toEraseOrNull->count(iT))
            continue;
        if(!isRefinementNeeded(
               triangles[iT], refinementCriterion, refinementThreshold) ||
           (refinementCriterion == RefinementCriterion::SmallestAngle &&
            isSmallestAngleFixed(triangles[iT])))
        {
            // fixed-edge sharp angle is impossible to refine
            continue;
        }
        // copy: resolveEncroachedEdges below can re-allocate 'triangles'
        const VerticesArr3 badTVerts = triangles[iT].vertices;
        const V2d<T>& v0 = vertices[badTVerts[0]];
        const V2d<T>& v1 = vertices[badTVerts[1]];
        const V2d<T>& v2 = vertices[badTVerts[2]];
        const T shortestEdge = std::min(
            distance(v0, v1), std::min(distance(v1, v2), distance(v2, v0)));
        if(shortestEdge <= minEdgeLength)
        {
            // same minEdgeLength give-up
            ++unrefined.shortEdgeTriangles;
            continue;
        }
        const V2d<T> circumcenterPos = circumcenter(v0, v1, v2);
        const OptionalTriInd triAtCircumcenter = walkTriangles(
            m_nearPtLocator.nearPoint(circumcenterPos, vertices),
            circumcenterPos);
        if(!triAtCircumcenter.hasValue())
        {
            // circumcenter falls outside triangulated area
            ++unrefined.circumcenterOutside;
            continue;
        }

        const VertInd budgetBeforeSplits = remainingVertexBudget;
        const EdgeVec encroachedEdges =
            edgesEncroachedBy(circumcenterPos, triAtCircumcenter.value());
        const TriIndVec badTris = resolveEncroachedEdges(
            detail::toQueue(encroachedEdges),
            remainingVertexBudget,
            steinerVerticesOffset,
            &circumcenterPos,
            refinementCriterion,
            refinementThreshold,
            toEraseOrNull,
            minEdgeLength,
            unrefined);
        if(!remainingVertexBudget)
            break;

        // a circumcenter encroaching on fixed edges:
        // split edges instead and re-visit the triangle later
        if(remainingVertexBudget != budgetBeforeSplits || !badTris.empty())
        {
            typedef TriIndVec::const_iterator It;
            for(It it = badTris.begin(); it != badTris.end(); ++it)
            {
                badTriangles.push(*it);
            }
            badTriangles.push(iT);
            continue;
        }
        // splitting was given up on: inserting would encroach anyway
        if(!encroachedEdges.empty())
            continue;

        const TriInd iCircumcenterTri = triAtCircumcenter.value();
        const Triangle& circumcenterTri = triangles[iCircumcenterTri];
        const PtTriLocation::Enum loc = locatePointTriangle(
            circumcenterPos,
            vertices[circumcenterTri.vertices[0]],
            vertices[circumcenterTri.vertices[1]],
            vertices[circumcenterTri.vertices[2]]);
        if(loc == PtTriLocation::OnVertex)
        {
            // circumcenter coincides with an existing vertex
            ++unrefined.circumcenterOnVertex;
            continue;
        }
        const array<TriInd, 2> trisAt = {
            iCircumcenterTri,
            isOnEdge(loc) ? circumcenterTri.neighbors[edgeNeighbor(loc)]
                          : noNeighbor};
        // Skip adding a Steiner point if the triangle will be removed anyway.
        if(toEraseOrNull &&
           (toEraseOrNull->count(trisAt[0]) || toEraseOrNull->count(trisAt[1])))
        {
            // circumcenter falls into a triangle that will be removed
            ++unrefined.circumcenterOutside;
            continue;
        }

        --remainingVertexBudget;
        const VertInd iVert = static_cast<VertInd>(vertices.size());
        addNewVertex(circumcenterPos, noNeighbor);
#ifdef CDT_ENABLE_CALLBACK_HANDLER
        if(m_callbackHandler)
        {
            m_callbackHandler->onAddVertexStart(
                iVert, AddVertexType::RefinementCircumcenter);
        }
#endif
        std::stack<TriInd> triStack =
            trisAt[1] == noNeighbor
                ? insertVertexInsideTriangle(iVert, trisAt[0])
                : insertVertexOnEdge(iVert, trisAt[0], trisAt[1], true);
        ensureDelaunayByEdgeFlips(iVert, triStack);
        tryAddVertexToLocator(iVert);

        TriInd start = m_vertTris[iVert];
        TriInd currTri = start;
        do
        {
            const Triangle& t = triangles[currTri];
            if(!(toEraseOrNull && toEraseOrNull->count(currTri)) &&
               !touchesSuperTriangle(t) &&
               isRefinementNeeded(t, refinementCriterion, refinementThreshold))
            {
                badTriangles.push(currTri);
            }
            currTri = t.next(iVert).first;
        } while(currTri != start);
    }

    // sharp corners that come from the input are impossible to refine: report
    // the triangles that are left too sharp because of them
    if(refinementCriterion == RefinementCriterion::SmallestAngle)
    {
        for(TriInd iT(0), n(triangles.size()); iT < n; ++iT)
        {
            const Triangle& t = triangles[iT];
            if(touchesSuperTriangle(t) ||
               (toEraseOrNull && toEraseOrNull->count(iT)))
            {
                continue;
            }
            if(isRefinementNeeded(
                   t, refinementCriterion, refinementThreshold) &&
               isSmallestAngleFixed(t))
            {
                ++unrefined.sharpFixedCorner;
            }
        }
    }
    return unrefined;
}

} // namespace CDT

CDT_RESTORE_MATH_SETTINGS_FOR_CONSTRUCTIONS

#endif // header-guard
