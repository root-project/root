#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace mapbox {

namespace util {

template <std::size_t I, typename T>
struct nth {
    inline static typename std::tuple_element<I, T>::type get(const T& t) { return std::get<I>(t); };
};

} // namespace util

namespace detail {

template <typename N = uint32_t>
class Earcut {
public:
    std::vector<N> indices;
    std::size_t vertices = 0;

    template <typename Polygon>
    void operator()(const Polygon& points);

private:
    struct Node {
        // i is a (bits(N)-1)-wide field packed alongside the 1-bit steiner flag; mask index to that
        // width so it fits without a narrowing warning (a no-op for any real vertex index).
        Node(N index, double x_, double y_)
            : x(x_), y(y_), i(index & ((N(1) << (sizeof(N) * 8 - 1)) - 1)), steiner(0) {}
        Node(const Node&) = delete;
        Node& operator=(const Node&) = delete;
        Node(Node&&) = delete;
        Node& operator=(Node&&) = delete;

        const double x;
        const double y;

        // previous and next vertice nodes in a polygon ring
        Node* prev = nullptr;
        Node* next = nullptr;

        // z-order curve value
        int32_t z = 0;

        // original index in polygon
        const N i : (sizeof(N) * 8 - 1);

        // indicates whether this is a steiner point
        N steiner : 1;

        // previous and next nodes in z-order
        Node* prevZ = nullptr;
        Node* nextZ = nullptr;
    };

    // Cache-optimized Triangle structure for repeated geometric tests
    struct Triangle {
        const double ax, ay;
        const double bx, by;
        const double cx, cy;
        // triangle bounding box, used to cheaply reject most candidate points before the
        // full point-in-triangle test (which is 6 multiplies)
        const double minX, minY, maxX, maxY;

        Triangle(const Node* a, const Node* b, const Node* c)
            : ax(a->x),
              ay(a->y),
              bx(b->x),
              by(b->y),
              cx(c->x),
              cy(c->y),
              minX(std::min<double>(ax, std::min<double>(bx, cx))),
              minY(std::min<double>(ay, std::min<double>(by, cy))),
              maxX(std::max<double>(ax, std::max<double>(bx, cx))),
              maxY(std::max<double>(ay, std::max<double>(by, cy))) {}

        inline double area() const { return (by - ay) * (cx - bx) - (bx - ax) * (cy - by); }

        inline bool inBBox(double px, double py) const { return px >= minX && px <= maxX && py >= minY && py <= maxY; }

        inline bool containsPoint(double px, double py) const {
            return (cx - px) * (ay - py) >= (ax - px) * (cy - py) && (ax - px) * (by - py) >= (bx - px) * (ay - py) &&
                   (bx - px) * (cy - py) >= (cx - px) * (by - py);
        }

        // as containsPoint, but false when the point coincides with the triangle's first vertex (a)
        inline bool containsPointExceptFirst(double px, double py) const {
            return !(ax == px && ay == py) && containsPoint(px, py);
        }
    };

    template <typename Ring>
    Node* linkedList(const Ring& points, const bool clockwise);
    Node* filterPoints(Node* start, Node* end = nullptr);
    void earcutLinked(Node* ear);
    bool isEar(Node* ear);
    bool isEarHashed(Node* ear);
    Node* cureLocalIntersections(Node* start);
    void splitEarcut(Node* start);
    template <typename Polygon>
    Node* eliminateHoles(const Polygon& points, Node* outerNode);
    Node* eliminateHole(Node* hole, Node* outerNode);
    Node* findHoleBridge(Node* hole, Node* outerNode);
    void buildBlockIndex(std::size_t maxNodes, std::size_t numHoles);
    void indexSegment(Node* head, Node* stop);
    void growBlock(Node* head, Node* tail);
    Node* liveBlockHead(std::size_t b);
    Node* liveBlockStop(std::size_t b);
    bool sectorContainsSector(const Node* m, const Node* p);
    void indexCurve(Node* start);
    Node* sortLinked(Node* list);
    int32_t zOrder(const double x_, const double y_);
    Node* getLeftmost(Node* start);
    bool pointInTriangle(double ax, double ay, double bx, double by, double cx, double cy, double px, double py) const;
    bool isValidDiagonal(Node* a, Node* b);
    double area(const Node* p, const Node* q, const Node* r) const;
    bool equals(const Node* p1, const Node* p2);
    bool intersects(const Node* p1, const Node* q1, const Node* p2, const Node* q2, bool includeBoundary = true);
    bool onSegment(const Node* p, const Node* q, const Node* r);
    bool intersectsPolygon(const Node* a, const Node* b);
    bool locallyInside(const Node* a, const Node* b);
    bool middleInside(const Node* a, const Node* b);
    Node* splitPolygon(Node* a, Node* b);
    template <typename Point>
    Node* insertNode(std::size_t i, const Point& p, Node* last);
    void removeNode(Node* p);

    bool hashing;
    // set by filterPoints whenever it removes at least one node; read by earcutLinked's stall
    // handler to decide whether another clip pass is worth attempting before the costlier stages
    bool filteredOut = false;
    double minX, maxX;
    double minY, maxY;
    double inv_size = 0;

    template <typename T, typename Alloc = std::allocator<T>>
    class ObjectPool {
    public:
        ObjectPool(std::size_t blockSize_) : baseBlockSize(std::max<std::size_t>(blockSize_, 256)) {
            allocateNewBlock();
        }
        ~ObjectPool() { clear(); }
        template <typename... Args>
        T* construct(Args&&... args) {
            // If current block is full, move to next block or allocate new one
            if (currentIndex >= baseBlockSize) {
                currentBlockIndex++;
                if (currentBlockIndex < memoryBlocks.size()) {
                    // Reuse existing block
                    currentIndex = 0;
                } else {
                    // Allocate a new one
                    allocateNewBlock();
                }
            }

            T* object = memoryBlocks[currentBlockIndex].get() + currentIndex;
            alloc_traits::construct(alloc, object, std::forward<Args>(args)...);
            totalObjects++;
            currentIndex++;
            return object;
        }
        void clear() {
            // Destroy all objects, but keep blocks allocated for reuse
            std::size_t objectsDestroyed = 0;
            for (std::size_t blockIdx = 0; blockIdx < memoryBlocks.size() && objectsDestroyed < totalObjects;
                 ++blockIdx) {
                // check if we are in the last block
                std::size_t objectsInThisBlock = std::min(baseBlockSize, totalObjects - objectsDestroyed);
                for (std::size_t i = 0; i < objectsInThisBlock; ++i) {
                    T* object = memoryBlocks[blockIdx].get() + i;
                    alloc_traits::destroy(alloc, object);
                }
                objectsDestroyed += objectsInThisBlock;
            }
            // Reset to start from first block again
            currentBlockIndex = 0;
            currentIndex = 0;
            totalObjects = 0;
        }

    private:
        Alloc alloc;
        typedef typename std::allocator_traits<Alloc> alloc_traits;

        // Custom deleter that uses the allocator
        struct AllocDeleter {
            Alloc alloc;
            std::size_t capacity;
            void operator()(T* ptr) { alloc_traits::deallocate(alloc, ptr, capacity); }
        };

        std::vector<std::unique_ptr<T[], AllocDeleter>> memoryBlocks;
        std::size_t currentBlockIndex = 0;
        std::size_t currentIndex = 0;
        std::size_t totalObjects = 0;
        const std::size_t baseBlockSize;

        void allocateNewBlock() {
            T* rawMemory = alloc_traits::allocate(alloc, baseBlockSize);
            auto newBlock = std::unique_ptr<T[], AllocDeleter>(rawMemory, AllocDeleter{alloc, baseBlockSize});
            memoryBlocks.push_back(std::move(newBlock));
            currentBlockIndex = memoryBlocks.size() - 1;
            currentIndex = 0;
        }
    };

    std::unique_ptr<ObjectPool<Node>> nodes;
    std::vector<Node*> holeQueue;
    // reused scratch buffer for sortLinked: materialize the z-linked ring, std::sort, relink
    std::vector<Node*> sortBuffer;

    // Block-bbox index for findHoleBridge (issue #183): one [minX,minY,maxX,maxY] bbox per K
    // consecutive ring edges, so the leftward-ray scan can skip whole blocks in O(1) instead of
    // walking the whole merged ring. Grown append-only — the outer ring seeds it, then each merged
    // hole appends a segment (head node, stop node, K-blocks over head..stop); independent segments,
    // not a ring tiling, since splices land mid-ring. Buffers reused/grown across calls.
    //
    // filterPoints only drops collinear/coincident points, so a stale bbox stays a conservative
    // superset of its live edges (never a false skip); the scan skips dead nodes (p->prev->next != p)
    // and lazily advances a dead head/stop. Blocks are scanned in append (not ring) order, so the
    // chosen bridge can differ from the un-indexed code — a different but equally valid result.
    static constexpr int32_t K = 16; // edges per block
    std::vector<double> blockBBox;   // [minX,minY,maxX,maxY] per block
    std::vector<Node*> blockHead;    // first node of each block's segment
    std::vector<Node*> blockStop;    // node just past each block's segment (exclusive walk bound)
    std::size_t numBlocks = 0;
    // true only while eliminateHoles merges holes, so removeNode keeps the block index live (growBlock)
    bool indexActive = false;
};

template <typename N>
template <typename Polygon>
void Earcut<N>::operator()(const Polygon& points) {
    // reset
    indices.clear();
    vertices = 0;

    if (points.empty()) return;

    double x;
    double y;
    int threshold = 80;
    std::size_t len = 0;

    const std::size_t numRings = static_cast<std::size_t>(points.size());
    for (std::size_t i = 0; threshold >= 0 && i < numRings; i++) {
        threshold -= static_cast<int>(points[i].size());
        len += static_cast<std::size_t>(points[i].size());
    }

    // estimate size of nodes and indices
    if (!nodes) {
        std::size_t estimatedNodes = len * 3 / 2;
        nodes = std::make_unique<ObjectPool<Node>>(std::max<std::size_t>(estimatedNodes, 256));
    }
    indices.reserve(len + static_cast<std::size_t>(points[0].size()));

    Node* outerNode = linkedList(points[0], true);
    if (!outerNode || outerNode->prev == outerNode->next) return;

    if (points.size() > 1) outerNode = eliminateHoles(points, outerNode);

    // if the shape is not too simple, we'll use z-order curve hash later; calculate polygon bbox
    hashing = threshold < 0;
    if (hashing) {
        Node* p = outerNode->next;
        minX = maxX = outerNode->x;
        minY = maxY = outerNode->y;
        do {
            x = p->x;
            y = p->y;
            minX = std::min<double>(minX, x);
            minY = std::min<double>(minY, y);
            maxX = std::max<double>(maxX, x);
            maxY = std::max<double>(maxY, y);
            p = p->next;
        } while (p != outerNode);

        // minX, minY and inv_size are later used to transform coords into integers for z-order calculation
        inv_size = std::max<double>(maxX - minX, maxY - minY);
        inv_size = inv_size != .0 ? (32767. / inv_size) : .0;
    }

    earcutLinked(outerNode);

    nodes->clear();
    holeQueue.clear();
}

// create a circular doubly linked list from polygon points in the specified winding order
template <typename N>
template <typename Ring>
typename Earcut<N>::Node* Earcut<N>::linkedList(const Ring& points, const bool clockwise) {
    using Point = typename Ring::value_type;
    double sum = 0;
    const std::size_t len = static_cast<std::size_t>(points.size());
    std::size_t i, j;
    Node* last = nullptr;

    // calculate original winding order of a polygon ring
    for (i = 0, j = len > 0 ? len - 1 : 0; i < len; j = i++) {
        const auto& p1 = points[i];
        const auto& p2 = points[j];
        const double p20 = util::nth<0, Point>::get(p2);
        const double p10 = util::nth<0, Point>::get(p1);
        const double p11 = util::nth<1, Point>::get(p1);
        const double p21 = util::nth<1, Point>::get(p2);
        sum += (p20 - p10) * (p11 + p21);
    }

    // link points into circular doubly-linked list in the specified winding order
    if (clockwise == (sum > 0)) {
        for (i = 0; i < len; i++) last = insertNode(vertices + i, points[i], last);
    } else {
        for (i = len; i-- > 0;) last = insertNode(vertices + i, points[i], last);
    }

    if (last && equals(last, last->next)) {
        removeNode(last);
        last = last->next;
    }

    vertices += len;

    return last;
}

// Remove collinear or coincident points; removability depends only on a node's immediate
// neighbors, so we sweep forward and re-check the predecessor after each removal. With no `end`
// we sweep the whole ring, lapping until nothing is removable (the fixpoint the clipper needs).
// With an explicit `end` we heal only the dirty window around a bridge/diagonal cut, stopping at
// `end` rather than lapping — O(window) instead of O(ring).
template <typename N>
typename Earcut<N>::Node* Earcut<N>::filterPoints(Node* start, Node* end) {
    if (!start) return start;
    const bool full = !end;
    if (full) end = start;

    Node* p = start;
    bool again;
    do {
        again = false;
        if (p != p->next && !p->steiner && (equals(p, p->next) || area(p->prev, p, p->next) == 0)) {
            if (full || p == end) end = p->prev; // pull the stop bound back past the removal
            filteredOut = true;
            removeNode(p);
            p = p->prev; // re-check the predecessor
            again = true;
        } else if (full || p != end) {
            p = p->next;
            again = !full; // local heal: keep looping until the sweep reaches end
        }
    } while (again || p != end);

    return end;
}

// main ear slicing loop which triangulates a polygon (given as a linked list)
template <typename N>
void Earcut<N>::earcutLinked(Node* ear) {
    if (!ear) return;

    // interlink polygon nodes in z-order
    if (hashing) indexCurve(ear);

    Node* stop = ear;
    Node* prev;
    Node* next;
    bool cured = false;

    // iterate through ears, slicing them one by one
    while (ear->prev != ear->next) {
        prev = ear->prev;
        next = ear->next;

        // reflex check is hoisted here to avoid constructing the Triangle for reflex corners
        if (area(prev, ear, next) < 0 && (hashing ? isEarHashed(ear) : isEar(ear))) {
            // cut off the triangle
            indices.emplace_back(prev->i);
            indices.emplace_back(ear->i);
            indices.emplace_back(next->i);

            removeNode(ear);

            ear = next;
            stop = next;

            continue;
        }

        ear = next;

        // if we looped through the whole remaining polygon and can't find any more ears
        if (ear == stop) {
            // try filtering collinear/coincident points and slicing again — repeat as long as
            // filtering actually removes nodes, since each removal can expose new ears
            filteredOut = false;
            ear = filterPoints(ear);
            if (filteredOut) {
                stop = ear;
                continue;
            }

            // filtering is exhausted: cure small local self-intersections once, then retry
            if (!cured) {
                ear = cureLocalIntersections(ear);
                stop = ear;
                cured = true;
                continue;
            }

            // as a last resort, try splitting the remaining polygon into two
            splitEarcut(ear);
            break;
        }
    }
}

// check whether a polygon node forms a valid ear with adjacent nodes
template <typename N>
bool Earcut<N>::isEar(Node* ear) {
    const Node* a = ear->prev;
    const Node* b = ear;
    const Node* c = ear->next;

    // reflex check is hoisted into the earcutLinked caller
    const Triangle tri(a, b, c);

    // now make sure we don't have other points inside the potential ear
    Node* p = ear->next->next;

    while (p != ear->prev) {
        if (tri.inBBox(p->x, p->y) && tri.containsPointExceptFirst(p->x, p->y) && area(p->prev, p, p->next) >= 0)
            return false;
        p = p->next;
    }

    return true;
}

template <typename N>
bool Earcut<N>::isEarHashed(Node* ear) {
    const Node* a = ear->prev;
    const Node* b = ear;
    const Node* c = ear->next;

    // reflex check is hoisted into the earcutLinked caller
    const Triangle tri(a, b, c);

    // z-order range for the current triangle bbox;
    const int32_t minZ = zOrder(tri.minX, tri.minY);
    const int32_t maxZ = zOrder(tri.maxX, tri.maxY);

    // first look for points inside the triangle in increasing z-order
    Node* p = ear->nextZ;

    while (p && p->z <= maxZ) {
        if (p != ear->next && tri.inBBox(p->x, p->y) && tri.containsPointExceptFirst(p->x, p->y) &&
            area(p->prev, p, p->next) >= 0)
            return false;
        p = p->nextZ;
    }

    // then look for points in decreasing z-order
    p = ear->prevZ;

    while (p && p->z >= minZ) {
        if (p != ear->next && tri.inBBox(p->x, p->y) && tri.containsPointExceptFirst(p->x, p->y) &&
            area(p->prev, p, p->next) >= 0)
            return false;
        p = p->prevZ;
    }

    return true;
}

// go through all polygon nodes and cure small local self-intersections
template <typename N>
typename Earcut<N>::Node* Earcut<N>::cureLocalIntersections(Node* start) {
    Node* p = start;
    bool cured = false;
    do {
        Node* a = p->prev;
        Node* b = p->next->next;

        // a self-intersection where edge (v[i-1],v[i]) intersects (v[i+1],v[i+2]);
        // includeBoundary=false so a mere collinear touch isn't treated as a crossing
        if (intersects(a, p, p->next, b, false) && locallyInside(a, b) && locallyInside(b, a)) {
            indices.emplace_back(a->i);
            indices.emplace_back(p->i);
            indices.emplace_back(b->i);

            // remove two nodes involved
            removeNode(p);
            removeNode(p->next);

            p = start = b;
            cured = true;
        }
        p = p->next;
    } while (p != start);

    return cured ? filterPoints(p) : p;
}

// try splitting polygon into two and triangulate them independently
template <typename N>
void Earcut<N>::splitEarcut(Node* start) {
    // look for a valid diagonal that divides the polygon into two
    Node* a = start;
    do {
        Node* b = a->next->next;
        while (b != a->prev) {
            if (a->i != b->i && isValidDiagonal(a, b)) {
                // split the polygon in two by the diagonal
                Node* c = splitPolygon(a, b);

                // filter colinear points around the cuts
                a = filterPoints(a, a->next);
                c = filterPoints(c, c->next);

                // run earcut on each half
                earcutLinked(a);
                earcutLinked(c);
                return;
            }
            b = b->next;
        }
        a = a->next;
    } while (a != start);
}

// link every hole into the outer loop, producing a single-ring polygon without holes
template <typename N>
template <typename Polygon>
typename Earcut<N>::Node* Earcut<N>::eliminateHoles(const Polygon& points, Node* outerNode) {
    const size_t len = static_cast<std::size_t>(points.size());

    holeQueue.clear();
    for (size_t i = 1; i < len; i++) {
        Node* list = linkedList(points[i], false);
        if (list) {
            if (list == list->next) list->steiner = true;
            holeQueue.push_back(getLeftmost(list));
        }
    }
    // compareXYSlope: sort by x, then y, then slope. When two holes' leftmost points coincide, the
    // slope tiebreak makes the bridge land on the shared vertex instead of bridging the wrong hole.
    std::sort(holeQueue.begin(), holeQueue.end(), [](const Node* a, const Node* b) {
        if (a->x != b->x) return a->x < b->x;
        if (a->y != b->y) return a->y < b->y;
        const double adx = a->next->x - a->x, ady = a->next->y - a->y;
        const double bdx = b->next->x - b->x, bdy = b->next->y - b->y;
        const bool aDegenerate = adx == 0 && ady == 0;
        const bool bDegenerate = bdx == 0 && bdy == 0;
        if (aDegenerate != bDegenerate) return aDegenerate;
        return ady * bdx < bdy * adx;
    });

    // block-bbox index for findHoleBridge, grown append-only as holes merge. Seed it with the
    // outer ring, then append each merged hole.
    buildBlockIndex(vertices, holeQueue.size());
    indexSegment(outerNode, outerNode);

    // process holes from left to right; indexActive lets removeNode keep block bboxes live as
    // filterPoints heals edges during merges (see growBlock)
    indexActive = true;
    for (size_t i = 0; i < holeQueue.size(); i++) {
        outerNode = eliminateHole(holeQueue[i], outerNode);
    }
    indexActive = false;

    // collapse collinear/coincident points across the whole merged ring once before clipping
    return filterPoints(outerNode);
}

// find a bridge between vertices that connects hole with an outer ring and and link it
template <typename N>
typename Earcut<N>::Node* Earcut<N>::eliminateHole(Node* hole, Node* outerNode) {
    Node* bridge = findHoleBridge(hole, outerNode);
    if (!bridge) {
        return outerNode;
    }

    Node* bridgeReverse = splitPolygon(bridge, hole);

    // index the merged-in segment before filtering: in ring order the splice runs
    // bridge -> hole -> bridgeReverse -> bridge2 -> (bridge's old next), covering the hole's edges
    // and both new slit edges. filterPoints below only drops collinear/coincident points, so these
    // bboxes stay valid (conservative) supersets.
    Node* bridge2 = bridgeReverse->next;
    indexSegment(bridge, bridge2->next);

    // heal collinear/coincident points around the two new slit edges
    filterPoints(bridgeReverse, bridgeReverse->next);
    return filterPoints(bridge, bridge->next);
}

// David Eberly's algorithm for finding a bridge between hole and outer polygon
template <typename N>
typename Earcut<N>::Node* Earcut<N>::findHoleBridge(Node* hole, Node* outerNode) {
    Node* p = outerNode;
    double hx = hole->x;
    double hy = hole->y;
    double qx = -std::numeric_limits<double>::max();
    Node* m = nullptr;

    // find a segment intersected by a ray from the hole's leftmost Vertex to the left;
    // segment's endpoint with lesser x will be potential connection Vertex,
    // unless they intersect at a vertex, then choose the vertex
    if (equals(hole, p)) return p;

    // scan blocks; skip any whose bbox can't hold a crossing that beats qx and lies left of hx
    // (the prune Morton order can't express — explicit per-axis [minY,maxY]/[minX,maxX])
    for (std::size_t b = 0, g = 0; b < numBlocks; b++, g += 4) {
        if (hy < blockBBox[g + 1] || hy > blockBBox[g + 3] || blockBBox[g] > hx || blockBBox[g + 2] <= qx) continue;

        // ensure the walk's exclusive bound is live so we don't overrun into other blocks
        const Node* stop = liveBlockStop(b);
        p = liveBlockHead(b);
        do {
            if (p->prev->next == p) { // skip nodes removed by filterPoints (stale in the index)
                if (equals(hole, p->next))
                    return p->next;
                else if (hy <= p->y && hy >= p->next->y && p->next->y != p->y) {
                    double x = p->x + (hy - p->y) * (p->next->x - p->x) / (p->next->y - p->y);
                    if (x <= hx && x > qx) {
                        qx = x;
                        m = p->x < p->next->x ? p : p->next;
                        if (x == hx) return m; // hole touches outer segment; pick leftmost endpoint
                    }
                }
            }
            p = p->next;
        } while (p != stop);
    }

    if (!m) return 0;

    // look for points inside the triangle of hole Vertex, segment intersection and endpoint;
    // if there are no points found, we have a valid connection;
    // otherwise choose the Vertex of the minimum angle with the ray as connection Vertex

    const double mx = m->x;
    const double my = m->y;
    const double tminY = std::min<double>(hy, my); // the triangle's y span; x span is [mx, hx]
    const double tmaxY = std::max<double>(hy, my);
    double tanMin = std::numeric_limits<double>::max();

    // scan the same blocks; skip any whose bbox can't overlap the triangle's [mx,hx]x[tminY,tmaxY] box
    for (std::size_t b = 0, g = 0; b < numBlocks; b++, g += 4) {
        if (blockBBox[g + 2] < mx || blockBBox[g] > hx || blockBBox[g + 3] < tminY || blockBBox[g + 1] > tmaxY)
            continue;

        const Node* stop = liveBlockStop(b);
        p = liveBlockHead(b);
        do {
            if (p->prev->next == p && hx >= p->x && p->x >= mx && hx != p->x && // skip dead nodes
                pointInTriangle(hy < my ? hx : qx, hy, mx, my, hy < my ? qx : hx, hy, p->x, p->y)) {
                const double tanCur = std::abs(hy - p->y) / (hx - p->x); // tangential

                // if hole point sits on p's horizontal edge (T-junction touch): the bridge runs
                // along that edge — locallyInside rejects it as collinear, but it's valid
                if ((locallyInside(p, hole) || (p->y == hy && p->next->y == hy && p->next->x > hx)) &&
                    (tanCur < tanMin ||
                     (tanCur == tanMin && (p->x > m->x || (p->x == m->x && sectorContainsSector(m, p)))))) {
                    m = p;
                    tanMin = tanCur;
                }
            }
            p = p->next;
        } while (p != stop);
    }

    return m;
}

// Block-bbox index buffers: size once from the input upper bound and reuse across calls.
template <typename N>
void Earcut<N>::buildBlockIndex(std::size_t maxNodes, std::size_t numHoles) {
    // upper bound: every input node indexed once, +2 bridge nodes per hole, plus a partial
    // trailing block per appended segment (outer ring + one per hole)
    const std::size_t maxBlocks = (maxNodes + 2 * numHoles + K - 1) / K + numHoles + 2;
    if (blockBBox.size() < maxBlocks * 4) blockBBox.resize(maxBlocks * 4);
    if (blockHead.size() < maxBlocks) {
        blockHead.resize(maxBlocks);
        blockStop.resize(maxBlocks);
    }
    numBlocks = 0;
}

// index the ring run head..stop (exclusive) as ceil(len / K) blocks; head == stop means the whole
// ring. each block's bbox covers both endpoints of every edge it owns.
template <typename N>
void Earcut<N>::indexSegment(Node* head, Node* stop) {
    Node* p = head;
    do {
        const std::size_t b = numBlocks++;
        blockHead[b] = p;
        double bMinX = std::numeric_limits<double>::max();
        double bMinY = std::numeric_limits<double>::max();
        double bMaxX = std::numeric_limits<double>::lowest();
        double bMaxY = std::numeric_limits<double>::lowest();
        int32_t k = 0;
        do {
            Node* c = p->next;              // edge p->c; bbox must bound both endpoints
            p->z = static_cast<int32_t>(b); // reuse z as the owning block during eliminateHoles (see growBlock)
            if (p->x < bMinX) bMinX = p->x;
            if (p->x > bMaxX) bMaxX = p->x;
            if (p->y < bMinY) bMinY = p->y;
            if (p->y > bMaxY) bMaxY = p->y;
            if (c->x < bMinX) bMinX = c->x;
            if (c->x > bMaxX) bMaxX = c->x;
            if (c->y < bMinY) bMinY = c->y;
            if (c->y > bMaxY) bMaxY = c->y;
            p = c;
        } while (++k < K && p != stop);
        blockStop[b] = p;
        const std::size_t g = b * 4;
        blockBBox[g] = bMinX;
        blockBBox[g + 1] = bMinY;
        blockBBox[g + 2] = bMaxX;
        blockBBox[g + 3] = bMaxY;
    } while (p != stop);
}

// when filterPoints heals an edge head->tail (removing the collinear node between them), the healed
// edge can extend past head's frozen block bbox if its old far endpoint lived in another block; grow
// head's block bbox to cover tail so the leftward-ray prune can't false-skip it.
template <typename N>
void Earcut<N>::growBlock(Node* head, Node* tail) {
    const std::size_t g = static_cast<std::size_t>(head->z) * 4;
    if (tail->x < blockBBox[g]) blockBBox[g] = tail->x;
    if (tail->y < blockBBox[g + 1]) blockBBox[g + 1] = tail->y;
    if (tail->x > blockBBox[g + 2]) blockBBox[g + 2] = tail->x;
    if (tail->y > blockBBox[g + 3]) blockBBox[g + 3] = tail->y;
}

// the block's head node can be removed by filterPoints during merges; advance it to the next live
// node so the walk doesn't start on (and immediately terminate at) a dead node. For the single
// full-ring seed block (head == stop) the same forward advance keeps them equal, so the do-while
// still laps the whole ring instead of collapsing to an empty walk.
template <typename N>
typename Earcut<N>::Node* Earcut<N>::liveBlockHead(std::size_t b) {
    Node* head = blockHead[b];
    while (head->prev->next != head) head = head->next;
    blockHead[b] = head;
    return head;
}

template <typename N>
typename Earcut<N>::Node* Earcut<N>::liveBlockStop(std::size_t b) {
    Node* stop = blockStop[b];
    while (stop->prev->next != stop) stop = stop->next;
    blockStop[b] = stop;
    return stop;
}

// whether sector in vertex m contains sector in vertex p in the same coordinates
template <typename N>
bool Earcut<N>::sectorContainsSector(const Node* m, const Node* p) {
    return area(m->prev, m, p->prev) < 0 && area(p->next, m, m->next) < 0;
}

// interlink polygon nodes in z-order
template <typename N>
void Earcut<N>::indexCurve(Node* start) {
    assert(start);
    Node* p = start;

    do {
        // always (re)compute: z may still hold a block index left over from eliminateHoles
        p->z = zOrder(p->x, p->y);
        p->prevZ = p->prev;
        p->nextZ = p->next;
        p = p->next;
    } while (p != start);

    p->prevZ->nextZ = nullptr;
    p->prevZ = nullptr;

    sortLinked(p);
}

// Sort the z-linked ring by z-order. Upstream earcut replaced its linked merge sort with an
// array sort (materialize node refs → sort → relink); in C++ std::sort over a contiguous
// Node* buffer inlines the comparator fully and beats both a linked merge sort and a hand radix
// (measured on the MVT tiles fixture) — JS's rejection of native Array.sort does not transfer.
template <typename N>
typename Earcut<N>::Node* Earcut<N>::sortLinked(Node* list) {
    assert(list);
    // list is a null-terminated nextZ chain (see indexCurve); walk it into the scratch buffer
    sortBuffer.clear();
    for (Node* p = list; p; p = p->nextZ) sortBuffer.push_back(p);

    std::sort(sortBuffer.begin(), sortBuffer.end(), [](const Node* a, const Node* b) { return a->z < b->z; });

    // relink in sorted order
    Node* prev = nullptr;
    for (Node* p : sortBuffer) {
        p->prevZ = prev;
        if (prev) prev->nextZ = p;
        prev = p;
    }
    prev->nextZ = nullptr;
    return sortBuffer.front();
}

// z-order of a Vertex given coords and size of the data bounding box
template <typename N>
int32_t Earcut<N>::zOrder(const double x_, const double y_) {
    // coords are transformed into non-negative 15-bit integer range
    int32_t x = static_cast<int32_t>((x_ - minX) * inv_size);
    int32_t y = static_cast<int32_t>((y_ - minY) * inv_size);

    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;

    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;

    return x | (y << 1);
}

// find the leftmost node of a polygon ring
template <typename N>
typename Earcut<N>::Node* Earcut<N>::getLeftmost(Node* start) {
    Node* p = start;
    Node* leftmost = start;
    do {
        if (p->x < leftmost->x || (p->x == leftmost->x && p->y < leftmost->y)) leftmost = p;
        p = p->next;
    } while (p != start);

    return leftmost;
}

// check if a point lies within a convex triangle
template <typename N>
bool Earcut<N>::pointInTriangle(
    double ax, double ay, double bx, double by, double cx, double cy, double px, double py) const {
    return (cx - px) * (ay - py) >= (ax - px) * (cy - py) && (ax - px) * (by - py) >= (bx - px) * (ay - py) &&
           (bx - px) * (cy - py) >= (cx - px) * (by - py);
}

// check if a diagonal between two polygon nodes is valid (lies in polygon interior)
template <typename N>
bool Earcut<N>::isValidDiagonal(Node* a, Node* b) {
    // degenerate zero-length case
    const bool zeroLength = equals(a, b) && area(a->prev, a, a->next) > 0 && area(b->prev, b, b->next) > 0;
    return a->next->i != b->i &&
           (zeroLength ||
            (locallyInside(a, b) && locallyInside(b, a) &&                         // locally visible
             (area(a->prev, a, b->prev) != 0.0 || area(a, b->prev, b) != 0.0))) && // no opposite-facing sectors
           !intersectsPolygon(a, b) &&                                             // doesn't intersect other edges
           (zeroLength || middleInside(a, b));                                     // diagonal inside polygon
}

// signed area of a triangle
template <typename N>
double Earcut<N>::area(const Node* p, const Node* q, const Node* r) const {
    return (q->y - p->y) * (r->x - q->x) - (q->x - p->x) * (r->y - q->y);
}

// check if two points are equal
template <typename N>
bool Earcut<N>::equals(const Node* p1, const Node* p2) {
    return p1->x == p2->x && p1->y == p2->y;
}

// check if two segments intersect; by default includes collinear boundary touches
template <typename N>
bool Earcut<N>::intersects(const Node* p1, const Node* q1, const Node* p2, const Node* q2, bool includeBoundary) {
    const double o1 = area(p1, q1, p2);
    const double o2 = area(p1, q1, q2);
    const double o3 = area(p2, q2, p1);
    const double o4 = area(p2, q2, q1);

    // general case: the two segments straddle each other (proper crossing)
    if (((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) && ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0))) return true;

    if (!includeBoundary) return false;

    if (o1 == 0 && onSegment(p1, p2, q1)) return true; // p1, q1 and p2 are collinear and p2 lies on p1q1
    if (o2 == 0 && onSegment(p1, q2, q1)) return true; // p1, q1 and q2 are collinear and q2 lies on p1q1
    if (o3 == 0 && onSegment(p2, p1, q2)) return true; // p2, q2 and p1 are collinear and p1 lies on p2q2
    if (o4 == 0 && onSegment(p2, q1, q2)) return true; // p2, q2 and q1 are collinear and q1 lies on p2q2

    return false;
}

// for collinear points p, q, r, check if point q lies on segment pr
template <typename N>
bool Earcut<N>::onSegment(const Node* p, const Node* q, const Node* r) {
    return q->x <= std::max<double>(p->x, r->x) && q->x >= std::min<double>(p->x, r->x) &&
           q->y <= std::max<double>(p->y, r->y) && q->y >= std::min<double>(p->y, r->y);
}

// check if a polygon diagonal intersects any polygon segments
template <typename N>
bool Earcut<N>::intersectsPolygon(const Node* a, const Node* b) {
    // diagonal bbox; an edge whose bbox can't overlap it can't intersect it, so
    // skip the orientation test for those (the common case — the diagonal is short)
    const double diagMinX = std::min<double>(a->x, b->x);
    const double diagMaxX = std::max<double>(a->x, b->x);
    const double diagMinY = std::min<double>(a->y, b->y);
    const double diagMaxY = std::max<double>(a->y, b->y);

    const Node* p = a;
    do {
        const Node* n = p->next;
        if ((p->x > diagMaxX && n->x > diagMaxX) || (p->x < diagMinX && n->x < diagMinX) ||
            (p->y > diagMaxY && n->y > diagMaxY) || (p->y < diagMinY && n->y < diagMinY)) {
            p = n;
            continue;
        }
        if (p->i != a->i && n->i != a->i && p->i != b->i && n->i != b->i && intersects(p, n, a, b)) return true;
        p = n;
    } while (p != a);

    return false;
}

// check if a polygon diagonal is locally inside the polygon
template <typename N>
bool Earcut<N>::locallyInside(const Node* a, const Node* b) {
    return area(a->prev, a, a->next) < 0 ? area(a, b, a->next) >= 0 && area(a, a->prev, b) >= 0
                                         : area(a, b, a->prev) < 0 || area(a, a->next, b) < 0;
}

// check if the middle Vertex of a polygon diagonal is inside the polygon
template <typename N>
bool Earcut<N>::middleInside(const Node* a, const Node* b) {
    const Node* p = a;
    bool inside = false;
    double px = (a->x + b->x) / 2;
    double py = (a->y + b->y) / 2;
    do {
        const Node* n = p->next;
        if (((p->y > py) != (n->y > py)) && (px < (n->x - p->x) * (py - p->y) / (n->y - p->y) + p->x)) inside = !inside;
        p = n;
    } while (p != a);

    return inside;
}

// link two polygon vertices with a bridge; if the vertices belong to the same ring, it splits
// polygon into two; if one belongs to the outer ring and another to a hole, it merges it into a
// single ring
template <typename N>
typename Earcut<N>::Node* Earcut<N>::splitPolygon(Node* a, Node* b) {
    Node* a2 = nodes->construct(a->i, a->x, a->y);
    Node* b2 = nodes->construct(b->i, b->x, b->y);
    Node* an = a->next;
    Node* bp = b->prev;

    a->next = b;
    b->prev = a;

    a2->next = an;
    an->prev = a2;

    b2->next = a2;
    a2->prev = b2;

    bp->next = b2;
    b2->prev = bp;

    return b2;
}

// create a node and util::optionally link it with previous one (in a circular doubly linked list)
template <typename N>
template <typename Point>
typename Earcut<N>::Node* Earcut<N>::insertNode(std::size_t i, const Point& pt, Node* last) {
    Node* p = nodes->construct(static_cast<N>(i), util::nth<0, Point>::get(pt), util::nth<1, Point>::get(pt));

    if (!last) {
        p->prev = p;
        p->next = p;

    } else {
        assert(last);
        p->next = last->next;
        p->prev = last;
        last->next->prev = p;
        last->next = p;
    }
    return p;
}

template <typename N>
void Earcut<N>::removeNode(Node* p) {
    p->next->prev = p->prev;
    p->prev->next = p->next;

    if (p->prevZ) p->prevZ->nextZ = p->nextZ;
    if (p->nextZ) p->nextZ->prevZ = p->prevZ;

    // keep the hole-bridge index's block bboxes covering the healed prev->next edge
    if (indexActive) growBlock(p->prev, p->next);
}
} // namespace detail

template <typename N = uint32_t, typename Polygon>
std::vector<N> earcut(const Polygon& poly) {
    mapbox::detail::Earcut<N> earcut;
    earcut(poly);
    return std::move(earcut.indices);
}

namespace detail {

// Refine a triangulation toward the constrained Delaunay triangulation by legalizing every interior
// edge in place with Lawson flips — maximizing the minimum angle and removing most slivers. Adapted
// from delaunator's edge legalization. Uses non-robust predicates: float input is fine, and the
// worst case is a not-quite-Delaunay edge, never an invalid mesh. Ported from earcut v3.2.3.
template <typename N>
class Refiner {
public:
    // triangles: triangle indices as returned by earcut, mutated in place.
    // coords: random-access container of points, indexed by vertex index (coords[i] -> point i),
    // read through the same util::nth<0>/<1> accessors as earcut's input.
    template <typename Coords>
    void operator()(std::vector<N>& triangles, const Coords& coords) {
        using Point = typename std::decay<decltype(coords[0])>::type;
        const int n = static_cast<int>(triangles.size());
        if (n < 6) return;
        ensureScratch(static_cast<std::size_t>(n));
        gen++; // bumping the generation logically empties the hash (no clearing)
        std::fill(heVec.begin(), heVec.begin() + n, -1);

        // Raw pointers into the scratch: indexed by the int/uint half-edge and hash indices below,
        // where operator[]'s size_type would trip -Wsign-conversion on every subscript.
        N* t = triangles.data();
        int32_t* he = heVec.data();
        int32_t* edgeStack = edgeStackVec.data();
        int32_t* hTable = hTableVec.data();
        uint32_t* hStamp = hStampVec.data();
        uint8_t* edgeStamp = edgeStampVec.data();

        auto X = [&](N p) -> double { return static_cast<double>(util::nth<0, Point>::get(coords[p])); };
        auto Y = [&](N p) -> double { return static_cast<double>(util::nth<1, Point>::get(coords[p])); };

        // Build half-edge twins with an undirected-edge hash; consumed slots mark linked pairs. As
        // each pair is linked we seed the stack with one representative (s, the earlier-inserted
        // edge) — this fuses the initial "push every interior edge" pass into the build, saving a
        // full O(n) scan. edgeStamp is all-zero here (balanced push/pop leaves it clean) and each
        // pair links once, so the seed write needs no dedup guard.
        int i = 0;
        for (int e = 0; e < n; e++) {
            const N a = t[e], b = t[nextHE(e)];
            const N lo = a < b ? a : b, hi = a < b ? b : a;
            uint32_t h = (uint32_t(lo) * 0x9e3779b1u ^ uint32_t(hi) * 0x85ebca6bu) & hMask;
            while (hStamp[h] == gen) {
                const int32_t s = hTable[h];
                // s == -1 marks a consumed slot (a pair already linked) — skip past it
                if (s != -1) {
                    const N sa = t[s], sb = t[nextHE(s)];
                    if ((sa == lo && sb == hi) || (sa == hi && sb == lo)) {
                        he[e] = s;
                        he[s] = e;
                        hTable[h] = -1; // link, then consume the slot
                        edgeStamp[s] = 1;
                        edgeStack[i++] = s; // seed the interior edge for the cascade
                        break;
                    }
                }
                h = (h + 1) & hMask;
            }
            if (hStamp[h] != gen) {
                hTable[h] = e;
                hStamp[h] = gen;
            } // first occurrence: insert
        }

        while (i > 0) {
            const int a = edgeStack[--i];
            edgeStamp[a] = 0;
            const int b = he[a];
            if (b == -1) continue;

            const int a0 = a - a % 3;
            const int b0 = b - b % 3;
            const int ar = a0 + (a + 2) % 3;
            const int al = a0 + (a + 1) % 3;
            const int bl = b0 + (b + 2) % 3;
            const int br = b0 + (b + 1) % 3;
            const N p0 = t[ar], pr = t[a], pl = t[al], p1 = t[bl];

            const double x0 = X(p0), y0 = Y(p0);
            const double xr = X(pr), yr = Y(pr);
            const double xl = X(pl), yl = Y(pl);
            const double x1 = X(p1), y1 = Y(p1);

            // Test inCircle first: most interior edges are already Delaunay (inCircle true → no
            // flip), so this short-circuits before the two convexity orients on the common path. The
            // quad must also be convex (both new triangles CCW) — flipping a reflex quad would push
            // a triangle outside the polygon. Boundary/hole edges self-protect via he == -1.
            if (!inCircle(x0, y0, xr, yr, xl, yl, x1, y1) && orient(x0, y0, xr, yr, x1, y1) > 0 &&
                orient(x0, y0, x1, y1, xl, yl) > 0) {
                t[a] = p1;
                t[b] = p0;
                const int32_t hbl = he[bl], har = he[ar];
                he[a] = hbl;
                if (hbl != -1) he[hbl] = a;
                he[b] = har;
                if (har != -1) he[har] = b;
                he[ar] = bl;
                he[bl] = ar;

                // re-check the quad's four outer edges; skip boundary edges (he == -1) and any
                // already queued (edgeStamp), which also keeps the stack bounded by n.
                if (hbl != -1 && edgeStamp[a] == 0) {
                    edgeStamp[a] = 1;
                    edgeStack[i++] = a;
                }
                if (har != -1 && edgeStamp[b] == 0) {
                    edgeStamp[b] = 1;
                    edgeStack[i++] = b;
                }
                if (he[al] != -1 && edgeStamp[al] == 0) {
                    edgeStamp[al] = 1;
                    edgeStack[i++] = al;
                }
                if (he[br] != -1 && edgeStamp[br] == 0) {
                    edgeStamp[br] = 1;
                    edgeStack[i++] = br;
                }
            }
        }
    }

private:
    // Reusable scratch, grown on demand like earcut's z-order arrays and reused across calls:
    //   he      = twin half-edge of each edge, or -1 on the polygon boundary
    //   hTable  = open-addressing hash, slot -> half-edge index, valid iff hStamp[slot] == gen
    //   edgeStamp = pending-in-stack flag, cleared when the edge is popped
    std::vector<int32_t> heVec, edgeStackVec, hTableVec;
    std::vector<uint32_t> hStampVec;
    std::vector<uint8_t> edgeStampVec;
    uint32_t hMask = 0, gen = 0;

    static int nextHE(int e) { return e - e % 3 + (e + 1) % 3; } // next half-edge in same triangle

    static double orient(double ax, double ay, double bx, double by, double cx, double cy) {
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    }

    // Whether p is inside or exactly on the circumcircle of triangle (a, b, c). Sign is negated vs
    // the usual predicate to match earcut's CCW winding — the standard sign builds the anti-Delaunay
    // mesh. Cocircular quads are legal ties, so refine only flips when this returns false.
    static bool inCircle(double ax, double ay, double bx, double by, double cx, double cy, double px, double py) {
        const double dx = ax - px, dy = ay - py, ex = bx - px, ey = by - py, fx = cx - px, fy = cy - py;
        const double ap = dx * dx + dy * dy, bp = ex * ex + ey * ey, cp = fx * fx + fy * fy;
        // A near-cocircular quad is a legal Delaunay tie, but roundoff can flag both an edge and its
        // flip as illegal, cascading into an endless flip loop (#205) — so treat a determinant
        // within a small margin of zero as a tie. The determinant's worst-case roundoff error is
        // provably below 9e-16·(ap+bp+cp)² (Shewchuk-style bound), so the margin guarantees every
        // executed flip is illegal in exact arithmetic, and Lawson flipping always terminates.
        const double s = ap + bp + cp;
        return dx * (ey * cp - bp * fy) - dy * (ex * cp - bp * fx) + ap * (ex * fy - ey * fx) <= 1e-13 * s * s;
    }

    void ensureScratch(std::size_t n) {
        // edgeStack holds at most one entry per half-edge (edgeStamp dedups), so n is a safe cap.
        if (edgeStackVec.size() < n) edgeStackVec.resize(n);
        if (heVec.size() < n) heVec.resize(n);
        if (edgeStampVec.size() < n) edgeStampVec.resize(n, 0);
        std::size_t size = 1;
        while (size < n * 4) size <<= 1; // power-of-two table, load factor <= 0.25
        if (hTableVec.size() < size) {
            hTableVec.resize(size);
            hStampVec.resize(size, 0);
        }
        hMask = uint32_t(size) - 1;
    }
};

} // namespace detail

// Opt-in Delaunay-refinement post-pass for earcut() output (or any manifold triangle-index array).
// Legalizes every interior edge in place with Lawson flips. See detail::Refiner. `coords` is a
// random-access container of points indexed by vertex index; `triangles` is mutated in place.
template <typename N, typename Coords>
void refine(std::vector<N>& triangles, const Coords& coords) {
    static thread_local mapbox::detail::Refiner<N> refiner;
    refiner(triangles, coords);
}

} // namespace mapbox
