// $Id: RelationalHvsNode.h,v 1.31 2009-12-16 17:17:37 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALHVSNODE_H
#define RELATIONALCOOL_RELATIONALHVSNODE_H

// Disable warning C4250 on Windows (inheritance via dominance)
// Copied from SEAL (Dictionary/Reflection/src/Tools.h)
#ifdef WIN32
#pragma warning ( disable : 4250 )
#endif

// Include files
#include "CoolKernel/IHvsNode.h"

// Local include files
#include "RelationalDatabasePtr.h"
#include "RelationalHvsNodeRecord.h"

namespace cool {

  /** @class RelationalHvsNode RelationalHvsNode.h
   *
   *  Relational implementation of one node in an HVS node tree.
   *  Multiple virtual inheritance from IHvsNode and RelationalHvsNodeRecord
   *  (diamond virtual inheritance of IHvsNodeRecord abstract interface).
   *
   *  @author Andrea Valassi and Sven A. Schmidt
   *  @date   2004-12-10
   */

  class RelationalHvsNode : virtual public IHvsNode,
                            virtual public RelationalHvsNodeRecord {

  public:

    /// Destructor
    virtual ~RelationalHvsNode();

    /// Set the node description
    void setDescription( const std::string& description );

    /// Lists all tags defined for this node (ordered alphabetically)
    const std::vector<std::string> listTags() const;

    /// Insertion time of a tag defined for this node
    /// (i.e. the time when the tag was first assigned to this node)
    const Time tagInsertionTime( const std::string& tagName ) const;

    /// Set the description of a tag.
    /// Throws TagNotFound the tag does not exist.
    /// Throws an Exception if the description is longer than 255 characters.
    void setTagDescription( const std::string& tagName,
                            const std::string& description );

    /// Description of a tag defined for this node
    const std::string tagDescription( const std::string& tagName ) const;

    /// Set the persistent lock status of a tag defined for this node.
    void setTagLockStatus( const std::string& tagName,
                           HvsTagLock::Status tagLockStatus );

    /// Get the persistent lock status of a tag defined for this node.
    HvsTagLock::Status tagLockStatus( const std::string& tagName ) const;

    /// Create a relation between a parent node tag and a tag in this node.
    /// Create the parent node tag if not defined yet.
    /// Create the tag in this node if not defined yet.
    /// Throws ReservedHeadTag if one of the two tags is a HEAD tag.
    /// Throws NodeIsSingleVersion if either node does not support versioning.
    /// Throws TagExists if either tag is already used in another node.
    /// Throws TagRelationExists if a relation to a child tag already exists.
    void createTagRelation( const std::string& parentTagName,
                            const std::string& tagName ) const;

    /// Delete the relation between a parent tag node and a tag in this node.
    /// Delete the parent tag if not related to another parent/child tag.
    /// Delete the tag in this node if not related to another tag or IOVs.
    /// Throws TagNotFound if the parent tag does not exist in the parent node.
    /// Throws TagRelationNotFound if the parent tag has no related child tag.
    void deleteTagRelation( const std::string& parentTagName ) const;

    /// Show the tag in this node associated to the given parent node tag.
    /// Throws ReservedHeadTag if the parent tag is a HEAD tag.
    /// Throws TagNotFound if the parent tag does not exist in the parent node.
    /// Throws TagRelationNotFound if the parent tag has no related child tag.
    const std::string
    findTagRelation( const std::string& parentTagName ) const;

    /// Main HVS method: determine the tag in this node that is related to the
    /// given ancestor tag (assumed to be defined in an ancestor of this node).
    /// The corresponding ancestor node is also internally determined.
    /// The ancestor tag is returned if defined directly in the descendant.
    /// Throws ReservedHeadTag if the ancestor tag is a HEAD tag.
    /// Throws TagNotFound if the tag does not exist in any inner node.
    /// Throws NodeRelationNotFound if the inner node where the ancestor tag
    /// is defined is not an ancestor of the descendant node.
    /// Throws TagRelationNotFound if no hierarchical tag relation exists.
    const std::string resolveTag( const std::string& ancestorTagName ) const;

    /*
    /// Does this tag exist in this node (independently of whether
    /// it references a parent tag or is referenced by any children)?
    /// Throws ReservedHeadTag if this is the HEAD tag.
    bool existsTag( const std::string& tagName ) const;

    /// Does this tag defined in this node have any relation (i.e. does
    /// it reference a parent tag or is it referenced by any children)?
    /// Throws ReservedHeadTag if this is the HEAD tag.
    /// Throws TagNotFound if this tag does not exist in this node.
    bool isTagUsed( const std::string& tagName ) const;
    */

    /*
    /// Does this tag defined in this node have any relation (i.e. does
    /// it reference a parent tag or is it referenced by any children)?
    /// Returns false if this tag does not exist in this node.
    bool isTagUsed( UInt32 tagId ) const;
    */

    /// Get a RelationalDatabase reference
    const RelationalDatabase& db() const { return *m_db; }

  protected:

    /// Constructor from a relational row retrieved from persistent storage
    /// This constructor is protected, hence no need to inline with base class
    /// non-standard constructors to prevent Windows compilation error C2248
    RelationalHvsNode( const RelationalDatabasePtr& db,
                       const coral::AttributeList& nodeTableRow );

    /*
    /// Constructor from a RelationalHvsNodeRecord
    RelationalHvsNode( const RelationalDatabasePtr& db,
                       const RelationalHvsNodeRecord& nodeRecord );
    */

  private:

    /// Standard constructor is private
    RelationalHvsNode();

    /// Copy constructor is private
    RelationalHvsNode( const RelationalHvsNode& rhs );

    /// Assignment operator is private
    RelationalHvsNode& operator=( const RelationalHvsNode& rhs );

  private:

    /// Backward pointer to the parent RelationalDatabase
    RelationalDatabasePtr m_db;

  };

}

#endif // RELATIONALCOOL_RELATIONALHVSNODE_H
