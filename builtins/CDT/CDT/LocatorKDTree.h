/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

/**
 * @file
 * Adapter between for KDTree and CDT
 */

#ifndef CDT_POINTKDTREE_H
#define CDT_POINTKDTREE_H

#include "CDTUtils.h"
#include "KDTree.h"

namespace CDT
{

/// KD-tree holding points
template <
    typename TCoordType,
    size_t NumVerticesInLeaf = 32,
    size_t InitialStackDepth = 32,
    size_t StackDepthIncrement = 32>
class LocatorKDTree
{
public:
    /// Initialize KD-tree with points
    void initialize(const std::vector<V2d<TCoordType> >& points)
    {
        typedef V2d<TCoordType> V2d_t;
        V2d_t min = points.front();
        V2d_t max = min;
        typedef typename std::vector<V2d_t>::const_iterator Cit;
        for(Cit it = points.begin(); it != points.end(); ++it)
        {
            min = V2d_t::make(std::min(min.x, it->x), std::min(min.y, it->y));
            max = V2d_t::make(std::max(max.x, it->x), std::max(max.y, it->y));
        }
        m_kdTree = KDTree_t(min, max);
        for(VertInd i(0); i < points.size(); ++i)
        {
            m_kdTree.insert(i, points);
        }
    }
    /// Add point to KD-tree
    void addPoint(const VertInd i, const std::vector<V2d<TCoordType> >& points)
    {
        m_kdTree.insert(i, points);
    }
    /// Find nearest point using R-tree
    VertInd nearPoint(
        const V2d<TCoordType>& pos,
        const std::vector<V2d<TCoordType> >& points) const
    {
        return m_kdTree.nearest(pos, points).second;
    }

    CDT::VertInd size() const
    {
        return m_kdTree.size();
    }

    bool empty() const
    {
        return !size();
    }

private:
    typedef KDTree::KDTree<
        TCoordType,
        NumVerticesInLeaf,
        InitialStackDepth,
        StackDepthIncrement>
        KDTree_t;
    KDTree_t m_kdTree;
};

} // namespace CDT

#endif
