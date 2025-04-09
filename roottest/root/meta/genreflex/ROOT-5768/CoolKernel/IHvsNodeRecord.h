// $Id: IHvsNodeRecord.h,v 1.17 2012-07-08 20:02:33 avalassi Exp $
#ifndef COOLKERNEL_IHVSNODERECORD_H
#define COOLKERNEL_IHVSNODERECORD_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include <cctype>
#include <string>
#include "CoolKernel/IRecord.h"
#include "CoolKernel/types.h"

namespace cool
{

  // Forward declarations
  class ITime;

  /** @class IHvsNodeRecord IHvsNodeRecord.h
   *
   *  Read-only abstract interface to one node in an HVS node tree.
   *
   *  The system handles a single tree of HVS nodes.
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
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2004-12-10
   */

  class IHvsNodeRecord
  {

  public:

    /// Destructor.
    virtual ~IHvsNodeRecord() {}

    /// Node full path name in the HVS hierarchy
    /// (this is always unique within a database).
    virtual const std::string& fullPath() const = 0;

    /// Node description.
    virtual const std::string& description() const = 0;

    /// Is this a leaf node?
    virtual bool isLeaf() const = 0;

    /// Has this node been stored into the database?
    virtual bool isStored() const = 0;

    /// Insertion time into the database.
    /// Throws an exception if the node has not been stored yet.
    virtual const ITime& insertionTime() const = 0;

    /// System-assigned node ID.
    /// Throws an exception if the node has not been stored yet.
    virtual UInt32 id() const = 0;

    /// System-assigned ID of the parent node.
    /// Throws an exception if the node has not been stored yet.
    /// Convention: parentId() = id() if the node has no parent (root node).
    virtual UInt32 parentId() const = 0;

  protected:

    /// Return the 'attributes' of the HVS node
    /// (implementation-specific properties not exposed in the API).
    virtual const IRecord& nodeAttributes() const = 0;

#ifdef COOL290CO
  private:

    /// Assignment operator is private (see bug #95823)
    IHvsNodeRecord& operator=( const IHvsNodeRecord& rhs );
#endif

  };

}
#endif // COOLKERNEL_IHVSNODERECORD_H
