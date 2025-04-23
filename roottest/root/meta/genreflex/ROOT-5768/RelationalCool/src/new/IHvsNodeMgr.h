// $Id: IHvsNodeMgr.h,v 1.4 2009-12-17 18:38:54 avalassi Exp $
#ifndef COOLKERNEL_IHVSNODEMGR_H
#define COOLKERNEL_IHVSNODEMGR_H

// Include files
#include <string>
#include "CoolKernel/IHvsNodeRecordMgr.h"

namespace cool {

  // Forward declarations
  class IHvsNode;

  /** @class IHvsNodeMgr IHvsNodeMgr.h
   *
   *  Abstract interface for the manager of one HVS node tree.
   *
   *  The HVS node manager handles a single tree of HVS nodes.
   *  The tree is a connected (single root), directed (relations are
   *  between parent and child), acyclic graph where each vertex has
   *  indegree 0 or 1 (all nodes have 1 parent, but the root has none).
   *
   *  HVS nodes can be uniquely identified by their "full path" names
   *  (following the same syntax used for UNIX files and directories).
   *
   *  Each node is also assigned a unique integer id by the system.
   *
   *  When users create a node, they must specify whether it can have
   *  children or not. Nodes that cannot have children are called leaf
   *  nodes. Nodes that can have children (even if they currently have
   *  none) are called inner nodes or internal nodes.
   *
   *  @author Andrea Valassi
   *  @date   2006-03-02
   */

  class IHvsNodeMgr : public IHvsNodeRecordMgr {

  public:

    /// Destructor
    virtual ~IHvsNodeMgr() {};

    /// Find a node by (full path) name.
    /// Throws NodeNotFound if the node does not exist.
    virtual const IHvsNode& findNode( const std::string& fullPath ) const = 0;

    /// Find a node by id.
    /// Throws NodeNotFound if the node does not exist.
    virtual const IHvsNode& findNode( const uInt32 id ) const = 0;

    /// Drop an existing node (leaf node or inner node).
    /// Return true if the node and all its structures are dropped as expected.
    /// Return false (without throwing any exception) if the node and
    /// all its structures do not exist any more on exit from this method,
    /// but the node or some of its structures did not exist to start with.
    /// Throw an exception if the node or one of its structures
    /// cannot be dropped (i.e. continues to exist on exit from this method):
    /// Throw an exception if the node is a non-empty inner node.
    /// Also deletes any tags associated to the node
    /// (throw an exception if such tags cannot be deleted).
    virtual const bool dropNode( const std::string& fullPath ) = 0;

  };

}

#endif // COOLKERNEL_IHVSNODEMGR_H
