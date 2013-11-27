// $Id: IHvsNode.h,v 1.45 2013-04-18 15:12:25 avalassi Exp $
#ifndef COOLKERNEL_IHVSNODE_H
#define COOLKERNEL_IHVSNODE_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include <algorithm>
#include <vector>
#include "CoolKernel/IHvsNodeRecord.h"
#include "CoolKernel/HvsTagLock.h"

namespace cool
{

  // Forward declarations
  class Time;

  /** @class IHvsNode IHvsNode.h
   *
   *  Abstract interface for one node in an HVS node tree.
   *
   *  Extends the IHvsNodeRecord with non-const methods which may
   *  require read and/or write access to the persistent storage.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2004-12-10
   */

  class IHvsNode : virtual public IHvsNodeRecord
  {

  public:

    /// HVS node type: inner node (node that can have children, even if it
    /// currently has none) or leaf node (node that cannot have children).
    typedef enum { INNER_NODE=0, LEAF_NODE } Type;

  public:

    /// Destructor.
    virtual ~IHvsNode() {}

    /// Change the node description stored in the database.
    virtual void setDescription( const std::string& description ) = 0;

    /// Lists all tags defined for this node (ordered alphabetically).
    /// Tag names, except for "HEAD", are case sensitive.
    /// The reserved tags "" and "HEAD" are NOT included in the list.
    /// Returns an empty list for folder sets and single version folders.
    virtual const std::vector<std::string> listTags() const = 0;

    /// Insertion time of a tag defined for this node
    /// (i.e. the time when the tag was first assigned to this node).
    /// Tag names, except for "HEAD", are case sensitive.
    /// Node creation time is returned for "" and "HEAD".
    /// For all other tag names, throws TagNotFound if tag does not exist
    /// (or node is a folder set or a single version folder).
    virtual const Time tagInsertionTime( const std::string& tagName ) const = 0;

    /// Description of a tag defined for this node.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Default description "HEAD tag" is returned for "" and "HEAD".
    /// For all other tag names, throws TagNotFound if tag does not exist
    /// (or node is a folder set or a single version folder).
    virtual const std::string tagDescription( const std::string& tagName ) const = 0;

    /// Set the persistent lock status of a tag defined for this node.
    virtual void setTagLockStatus( const std::string& tagName,
                                   HvsTagLock::Status tagLockStatus ) = 0;

    /// Get the persistent lock status of a tag defined for this node.
    virtual HvsTagLock::Status tagLockStatus( const std::string& tagName ) const = 0;

    /// Is this tag name a reserved HEAD tag?
    /// Returns true for "" and any case-insensitive variant of "HEAD".
    static bool isHeadTag( const std::string& tagName )
    {
      std::string ucTagName = tagName;
      std::transform( ucTagName.begin(),
                      ucTagName.end(),
                      ucTagName.begin(),
                      ::toupper ); // fix bug #101246
      if ( ucTagName == "" || ucTagName == "HEAD" ) return true;
      else return false;
    }

    /// Most common tag name of the reserved HEAD tag: "HEAD"
    static const std::string headTag()
    {
      return std::string( "HEAD" );
    }

    /// Create a relation between a parent node tag and a tag in this node.
    /// Create the parent node tag if not defined yet.
    /// Create the tag in this node if not defined yet.
    /// Throws ReservedHeadTag if one of the two tags is a HEAD tag.
    /// Throws NodeIsSingleVersion if either node does not support versioning.
    /// Throws TagExists if either tag is already used in another node.
    /// Throws TagRelationExists if a relation to a child tag already exists.
    /// Throws TagIsLocked if the parent tag is locked.
    virtual void createTagRelation( const std::string& parentTagName,
                                    const std::string& tagName ) const = 0;

    /// Delete the relation between a parent tag node and a tag in this node.
    /// Delete the parent tag if not related to another parent/child tag.
    /// Delete the tag in this node if not related to another tag or IOVs.
    /// Throws TagNotFound if the parent tag does not exist in the parent node.
    /// Throws TagRelationNotFound if the parent tag has no related child tag.
    /// Throws TagIsLocked if the parent tag is locked.
    /// Throws TagIsLocked if the child tag is locked and would be deleted.
    virtual void deleteTagRelation( const std::string& parentTagName ) const = 0;

    /// Show the tag in this node associated to the given parent node tag.
    /// Throws ReservedHeadTag if the parent tag is a HEAD tag.
    /// Throws TagNotFound if the parent tag does not exist in the parent node.
    /// Throws TagRelationNotFound if the parent tag has no related child tag.
    virtual const std::string
    findTagRelation( const std::string& parentTagName ) const = 0;

    /// Main HVS method: determine the tag in this node that is related to the
    /// given ancestor tag (assumed to be defined in an ancestor of this node).
    /// The corresponding ancestor node is also internally determined.
    /// The ancestor tag is returned if defined directly in the descendant.
    /// Throws ReservedHeadTag if the ancestor tag is a HEAD tag.
    /// Throws TagNotFound if the tag does not exist in any inner node.
    /// Throws NodeRelationNotFound if the inner node where the ancestor tag
    /// is defined is not an ancestor of the descendant node.
    /// Throws TagRelationNotFound if no hierarchical tag relation exists.
    virtual const std::string
    resolveTag( const std::string& ancestorTagName ) const = 0;

    /*
    /// Does this tag exist in this node (independently of whether
    /// it references a parent tag or is referenced by any children)?
    /// Throws ReservedHeadTag if this is the HEAD tag.
    virtual bool existsTag( const std::string& tagName ) const = 0;

    /// Does this tag defined in this node have any relation (i.e. does
    /// it reference a parent tag or is it referenced by any children)?
    /// Throws ReservedHeadTag if this is the HEAD tag.
    /// Throws TagNotFound if this tag does not exist in this node.
    virtual bool isTagUsed( const std::string& tagName ) const = 0;
    */

    /// Set the description of a tag.
    /// Throws TagNotFound the tag does not exist.
    /// Throws an Exception if the description is longer than 255 characters.
    virtual void setTagDescription( const std::string& tagName,
                                    const std::string& description ) = 0;

#ifdef COOL290CO
  private:

    /// Assignment operator is private (see bug #95823)
    IHvsNode& operator=( const IHvsNode& rhs );
#endif

  };

}
#endif // COOLKERNEL_IHVSNODE_H
