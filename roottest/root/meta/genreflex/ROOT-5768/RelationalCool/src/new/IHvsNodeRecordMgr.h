// $Id: IHvsNodeRecordMgr.h,v 1.3 2009-12-17 18:38:54 avalassi Exp $
#ifndef COOLKERNEL_IHVSNODERECORDMGR_H
#define COOLKERNEL_IHVSNODERECORDMGR_H

// Include files
#include <string>

namespace cool {

  // Forward declarations
  class IHvsNodeRecord;

  /** @class IHvsNodeRecordMgr IHvsNodeRecordMgr.h
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

  class IHvsNodeRecordMgr {

  public:

    /// Destructor
    virtual ~IHvsNodeRecordMgr() {};

    /// Find a node record by (full path) name.
    /// Throws NodeNotFound if the node does not exist.
    virtual const IHvsNodeRecord&
    findNodeRecord( const std::string& fullPath ) const = 0;

    /// Find a node record by id.
    /// Throws NodeNotFound if the node does not exist.
    virtual const IHvsNodeRecord&
    findNodeRecord( const uInt32 id ) const = 0;

    /// Does a node with this (full path) name exist?
    virtual const bool existsNode( const std::string& nodeName ) const = 0;

  };

}

#endif // COOLKERNEL_IHVSNODERECORDMGR_H
