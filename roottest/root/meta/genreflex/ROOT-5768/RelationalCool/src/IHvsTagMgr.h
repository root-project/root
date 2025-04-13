// $Id: IHvsTagMgr.h,v 1.10 2009-12-17 18:38:53 avalassi Exp $
#ifndef COOLKERNEL_IHVSTAGMGR_H
#define COOLKERNEL_IHVSTAGMGR_H

// Include files
#include <string>
#include <vector>
#include "CoolKernel/IHvsNode.h"

namespace cool {

  // Forward declarations
  class HvsTagRecord;
  //class IHvsTag;
  //class IHvsTagRecord;

  /** @class IHvsTagMgr IHvsTagMgr.h
   *
   *  Abstract interface for the manager of one HVS tag tree.
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
   *  @date   2006-03-02
   */

  class IHvsTagMgr {

  public:

    /// Destructor
    virtual ~IHvsTagMgr() {};

    /// This method does not handle transactions.
    /// Create a tag in an HVS node.
    /// Throws TagExists if the tag already exists (in this or another node).
    virtual const HvsTagRecord
    createTag( UInt32 nodeId,
               const std::string& tagName,
               const std::string& description = "" ) const = 0;

    /// This method does not handle transactions.
    /// Delete a tag in an HVS node.
    virtual void deleteTag( UInt32 nodeId,
                            const std::string& tagName ) const = 0;

    /*
    /// Has this tag name been reserved for a node type?
    /// [NB: this may be true even if it is not used in any node.]
    /// Tag names, except for "HEAD", are case sensitive.
    /// Returns true for the reserved tags "" and "HEAD".
    virtual bool existsTagName( const std::string& tagName ) const = 0;
    */

    /// Return the type of node (inner/leaf) where this tag name can be used.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Throws TagNameNotFound if the tag name does not exist.
    virtual IHvsNode::Type tagNameScope( const std::string& tagName ) const = 0;

    /// Does a tag with this name exist (in any node)?
    /// Tag names, except for "HEAD", are case sensitive.
    /// Returns true for the reserved tags "" and "HEAD".
    virtual bool existsTag( const std::string& tagName ) const = 0;

    /*
    /// TEMPORARY! A tag name cannot be used YET for more than one leaf node!
    /// Return the ID of the node where the tag is defined.
    /// Throws ReservedHeadTag for the HEAD tag (defined in all folders).
    /// Throws TagNotFound if the tag does not exist.
    virtual UInt32 taggedNodeId( const std::string& tagName ) const = 0;

    /// TEMPORARY! A tag name cannot be used YET for more than one leaf node!
    /// Return the name of the node where the tag is defined.
    /// Throws ReservedHeadTag for the HEAD tag (defined in all folders).
    /// Throws TagNotFound if the tag does not exist.
    virtual const std::string taggedNode( const std::string& tagName ) const = 0;

    /// Return the IDs of the nodes where the tag is defined.
    /// Throws ReservedHeadTag for the HEAD tag (defined in all folders).
    /// Throws TagNotFound if the tag does not exist.
    virtual const std::vector<UInt32>
    taggedNodeIds( const std::string& tagName ) const = 0;
    */

    /// Return the names of the nodes where the tag is defined.
    /// Throws ReservedHeadTag for the HEAD tag (defined in all folders).
    /// Throws TagNotFound if the tag does not exist.
    virtual const std::vector<std::string>
    taggedNodes( const std::string& tagName ) const = 0;

    /*
    /// Does a tag with this name exist in the given node?
    /// Tag names, except for "HEAD", are case sensitive.
    /// Returns true for the reserved tags "" and "HEAD".
    virtual bool existsTag( const std::string& tagName,
                            UInt32 nodeId ) = 0;
    */

    /// Find a tag record by nodeId and tag name.
    /// Throws ReservedHeadTag for the HEAD tag (defined in all folders).
    /// Throws TagNotFound if the tag does not exist.
    virtual const HvsTagRecord findTagRecord( UInt32 nodeId,
                                              const std::string& tagName ) const = 0;

    /// Find a tag record by nodeId and tagId.
    /// Throws ReservedHeadTag for the HEAD tag (defined in all folders).
    /// Throws TagNotFound if the tag does not exist.
    virtual const HvsTagRecord findTagRecord( UInt32 nodeId,
                                              UInt32 tagId ) const = 0;

    /*
    /// Does a tag with this name exist in the given node?
    /// Tag names, except for "HEAD", are case sensitive.
    /// Returns true for the reserved tags "" and "HEAD".
    virtual bool existsTag( const std::string& tagName,
                                  const std::string& nodeName ) = 0;

    /// Find a tag record by name and node name.
    /// Throws ReservedHeadTag for the HEAD tag (defined in all folders).
    /// Throws TagNotFound if the tag does not exist in the given node.
    virtual const IHvsTagRecord
    findTagRecord( const std::string& tagName,
                   const std::string& nodeName ) const = 0;
    */

    /*
    /// Find a tag by name and nodeID.
    /// Throws TagNotFound if the tag does not exist.
    virtual const IHvsTag& findTag( const std::string& tagName,
                                    UInt32 nodeId ) const = 0;

    /// Find a tag by name and node name.
    /// Throws TagNotFound if the tag does not exist.
    virtual const IHvsTag& findTag( const std::string& tagName,
                                    const std::string& nodeName ) const = 0;
    */

    /// Create a relation between tags for a pair of parent/child nodes.
    /// Create the parent tag in the parent node if not defined yet.
    /// Create the child tag in the child node if not defined yet.
    /// Throws ReservedHeadTag if one of the two tags is a HEAD tag.
    /// Throws NodeIsSingleVersion if either node does not support versioning.
    /// Throws NodeRelationNotFound if the nodes are not parent and child.
    /// Throws TagExists if either tag is already used in another node.
    /// Throws TagRelationExists if a relation to a child tag already exists.
    virtual void createTagRelation( UInt32 parentNodeId,
                                    const std::string& parentTagName,
                                    UInt32 childNodeId,
                                    const std::string& childTagName ) const = 0;

    /// Delete the relation between a parent tag node and a child tag.
    /// Delete the parent tag if not related to another parent/child tag.
    /// Delete the child tag if not related to another tag or IOVs.
    /// Throws ReservedHeadTag if the parent tag is a HEAD tag.
    /// Throws TagNotFound if the parent tag does not exist in the parent node.
    /// Throws TagRelationNotFound if the parent tag has no related child tag.
    /// TEMPORARY? - returns the tagId of the deleted related child tag
    virtual UInt32 deleteTagRelation( UInt32 parentNodeId,
                                      const std::string& parentTagName,
                                      UInt32 childNodeId ) const = 0;

    /// Find the child node tag associated to the given parent node tag.
    /// Throws ReservedHeadTag if the parent tag is a HEAD tag.
    /// Throws TagNotFound if the parent tag does not exist in the parent node.
    /// Throws TagRelationNotFound if the parent tag has no related child tag.
    virtual UInt32 findTagRelation( UInt32 parentNodeId,
                                    const std::string& parentTagName,
                                    UInt32 childNodeId ) const = 0;

    /// Main HVS method: determine the descendant node tag that is related to
    /// the given ancestor tag (assumed to be defined in an ancestor node).
    /// The corresponding ancestor node is also internally determined.
    /// The ancestor tag is returned if defined directly in the descendant.
    /// Throws ReservedHeadTag if the ancestor tag is a HEAD tag.
    /// Throws TagNotFound if the tag does not exist in any inner node.
    /// Throws NodeRelationNotFound if the inner node where the ancestor tag
    /// is defined is not an ancestor of the descendant node.
    /// Throws TagRelationNotFound if no hierarchical tag relation exists.
    virtual UInt32 resolveTag( const std::string& ancestorTagName,
                               UInt32 descendantNodeId ) const = 0;

  };

}

#endif // COOLKERNEL_IHVSTAGMGR_H
