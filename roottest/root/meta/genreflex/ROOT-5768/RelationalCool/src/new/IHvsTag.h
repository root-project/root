// $Id: IHvsTag.h,v 1.3 2009-12-17 15:46:15 avalassi Exp $
#ifndef COOLKERNEL_IHVSTAG_H
#define COOLKERNEL_IHVSTAG_H

// Include files
#include "CoolKernel/IHvsTagRecord.h"

namespace cool {

  /** @class IHvsTag IHvsTag.h
   *
   *  Abstract interface for one tag stored in an HVS tag tree
   *  (read-only interface to access the properties of existing tags).
   *
   *  HVS tags can be uniquely identified by their names.
   *  The terms "tag" and "tag name" are thus equivalent.
   *
   *  Each tag is also assigned a unique integer id by the system.
   *
   *  When users create a tag, they must specify whether it will be used
   *  for inner nodes or leaf nodes. A given tag can be used only for one
   *  inner node or for many leaf nodes: the same tag cannot be used for
   *  more than one inner node, or for one inner node and one leaf node.
   *
   *  @author Andrea Valassi
   *  @date   2006-03-03
   */

  class IHvsTag : public IHvsTagRecord {

  public:

    /// Destructor.
    virtual ~IHvsTag() {}

    /// Change the tag description stored in the database.
    virtual void setDescription( const std::string& description ) = 0;

  };

}

#endif // COOLKERNEL_IHVSTAG_H
