// $Id: IHvsTagRecord.h,v 1.7 2009-12-16 17:17:37 avalassi Exp $
#ifndef COOLKERNEL_IHVSTAGRECORD_H
#define COOLKERNEL_IHVSTAGRECORD_H

// Include files
#include <string>
#include "CoolKernel/IHvsNode.h"
#include "CoolKernel/Time.h"

namespace cool {

  /** @class IHvsTagRecord IHvsTagRecord.h
   *
   *  Read-only abstract interface to one tag in an HVS tag tree.
   *
   *  An HVS tag represents a tagged version of one HVS node.
   *  It can be uniquely identified by the tag name and the node name or ID.
   *  Each tag is also assigned a unique integer ID by the system.
   *
   *  The terms "tag" and "tag name" are NOT equivalent.
   *  A given tag name can be used either for only one inner node or for any
   *  number of leaf nodes: the same tag name cannot be used for more than
   *  one inner node, or for one inner node and one or more leaf nodes.
   *  Users can reserve the use of a given tag name for either type of node.
   *
   *  TEMPORARY! A tag name cannot be used YET for more than one leaf node!
   *
   *  For inner nodes, an HVS tag represents a collection
   *  of versions of (some of its) children nodes.
   *  For leaf nodes, an HVS tag typically represents a collection
   *  of versions of (some of the) data associated to the leaf node,
   *  such as a collection of IOVs in a conditions data folder.
   *
   *  When different tags with the same name exist in different leaf nodes,
   *  they are assigned different integer IDs so that they can be renamed
   *  independently while keeping their distinct ID and properties.
   *
   *  @author Andrea Valassi
   *  @date   2006-03-03
   */

  class IHvsTagRecord {

  public:

    /// System-assigned tag ID.
    virtual UInt32 id() const = 0;

    /// Tag scope: node where the tag is defined.
    virtual UInt32 nodeId() const = 0;

    /// Tag name.
    virtual const std::string& name() const = 0;

    /// Tag description.
    virtual const std::string& description() const = 0;

    /// Tag insertion time into the database (creation time).
    virtual const ITime& insertionTime() const = 0;

  protected:

    /// Destructor.
    virtual ~IHvsTagRecord() {}

  };

}

#endif // COOLKERNEL_IHVSTAGRECORD_H
