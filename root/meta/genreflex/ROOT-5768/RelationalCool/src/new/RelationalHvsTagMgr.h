// $Id: RelationalHvsTagMgr.h,v 1.3 2009-12-17 18:38:54 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALHVSTAGMGR_H
#define RELATIONALCOOL_RELATIONALHVSTAGMGR_H

// Include files
#include "CoolKernel/IHvsTagMgr.h"

namespace cool {

  /** @class RelationalHvsTagMgr RelationalHvsTagMgr.h
   *
   *  Abstract interface for the manager of one HVS tag tree.
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
   *  @date   2006-03-02
   */

  class RelationalHvsTagMgr {

  public:

    /// Destructor
    virtual ~RelationalHvsTagMgr() {};

    /// Get a tag record by name.
    /// Throws TagNotFound if the tag does not exist.
    virtual const IHvsTagRecord&
    findTagRecord( const std::string& tagName ) const;

    /*
    /// Get a tag by name.
    /// Throws TagNotFound if the tag does not exist.
    virtual const IHvsTag& findTag( const std::string& tagName );
    */

    /// Does a tag with this name exist?
    /// Tag names, except for "HEAD", are case sensitive.
    /// Returns true for the reserved tags "" and "HEAD".
    virtual const bool existsTag( const std::string& tagName ) const;

    /*
    /// Return the type of node (inner/leaf) where the tag can be defined.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Throws TagNotFound if the tag does not exist.
    virtual const IHvsNode::Type tagScope( const std::string& tagName ) const;
    */

    /// Return the names of the node where the tag is defined.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Throws TagNotFound if the tag does not exist.
    /// Throws ReservedHeadTag for the HEAD tag (defined in all folders).
    virtual const std::string taggedNode( const std::string& tagName ) const;

    /// Return the names of the nodes where the tag is defined.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Throws TagNotFound if the tag does not exist.
    /// Throws ReservedHeadTag for the HEAD tag (defined in all folders).
    virtual const
    std::vector<std::string> taggedNodes( const std::string& tagName ) const;

    /*
    /// Main HVS method: determine the tag associated to a node, given
    /// a tag associated to an ancestor of that descendant node.
    /// The corresponding ancestor node is also internally determined.
    /// Throw an exception if the tag is not associated to any ancestor node.
    /// The input 'ancestor' tag is returned if it is directly associated to
    /// the node itself, rather than to an ancestor of that node.
    virtual const
    std::string resolveTag( const std::string& descendantNode,
                            const std::string& ancestorTag );

    /// Main HVS method: determine the tag associated to a node, given
    /// an ancestor of that descendant node and a tag associated to it.
    /// Throw an exception if the tag is not associated to the ancestor node.
    /// Throw an exception if the two nodes are not ancestor and descendant.
    virtual const
    std::string resolveTag( const std::string& descendantNode,
                            const std::string& ancestorTag,
                            const std::string& ancestorNode );
    */

  private:

    /// Handle to the IHvsNodeMgr
    const IHvsNodeMgr& hvsNodeMgr() const;

  private:

    /// Handle to the IHvsNodeMgr
    IHvsNodeMgr* m_hvsNodeMgr;

  };

}

#endif // RELATIONALCOOL_RELATIONALHVSTAGMGR_H
