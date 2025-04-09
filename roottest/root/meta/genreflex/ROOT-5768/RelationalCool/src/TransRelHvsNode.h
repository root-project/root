// $Id: TransRelHvsNode.h,v 1.2 2012-06-29 15:25:40 avalassi Exp $
#ifndef RELATIONALCOOL_TRANSRELHVSNODE_H
#define RELATIONALCOOL_TRANSRELHVSNODE_H

// Include files
#include "CoolKernel/IHvsNode.h"
#include "CoolKernel/InternalErrorException.h"

// Local include files
#include "RelationalException.h"
#include "RelationalFolder.h"
#include "RelationalFolderSet.h"
#include "RelationalHvsNode.h"

namespace cool {

  /** @class TransRelHvsNode TransRelHvsNode.h
   *
   *  Transaction aware wrapper around RelationalHvsNode 
   *
   *  Wrapps the IHvsNode part of a RelationalFolder or RelationalFolderSet
   *
   *  @author Martin Wache
   *  @date   2010-11-3
   */

  class TransRelHvsNode : virtual public IHvsNode
  {

    // -- IHvsNode interface 
    /// Change the node description stored in the database.
    virtual void setDescription( const std::string& description );

    /// Lists all tags defined for this node (ordered alphabetically).
    /// Tag names, except for "HEAD", are case sensitive.
    /// The reserved tags "" and "HEAD" are NOT included in the list.
    /// Returns an empty list for folder sets and single version folders.
    virtual const std::vector<std::string> listTags() const;

    /// Insertion time of a tag defined for this node
    /// (i.e. the time when the tag was first assigned to this node).
    /// Tag names, except for "HEAD", are case sensitive.
    /// Node creation time is returned for "" and "HEAD".
    /// For all other tag names, throws TagNotFound if tag does not exist
    /// (or node is a folder set or a single version folder).
    virtual const Time tagInsertionTime( const std::string& tagName ) const;

    /// Description of a tag defined for this node.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Default description "HEAD tag" is returned for "" and "HEAD".
    /// For all other tag names, throws TagNotFound if tag does not exist
    /// (or node is a folder set or a single version folder).
    virtual const std::string tagDescription( const std::string& tagName ) const;

    /// Set the persistent lock status of a tag defined for this node.
    virtual void setTagLockStatus( const std::string& tagName,
                                   HvsTagLock::Status tagLockStatus );

    /// Get the persistent lock status of a tag defined for this node.
    virtual HvsTagLock::Status tagLockStatus( const std::string& tagName ) const;

    /// Create a relation between a parent node tag and a tag in this node.
    /// Create the parent node tag if not defined yet.
    /// Create the tag in this node if not defined yet.
    /// Throws ReservedHeadTag if one of the two tags is a HEAD tag.
    /// Throws NodeIsSingleVersion if either node does not support versioning.
    /// Throws TagExists if either tag is already used in another node.
    /// Throws TagRelationExists if a relation to a child tag already exists.
    /// Throws TagIsLocked if the parent tag is locked.
    virtual void createTagRelation( const std::string& parentTagName,
                                    const std::string& tagName ) const;

    /// Delete the relation between a parent tag node and a tag in this node.
    /// Delete the parent tag if not related to another parent/child tag.
    /// Delete the tag in this node if not related to another tag or IOVs.
    /// Throws TagNotFound if the parent tag does not exist in the parent node.
    /// Throws TagRelationNotFound if the parent tag has no related child tag.
    /// Throws TagIsLocked if the parent tag is locked.
    /// Throws TagIsLocked if the child tag is locked and would be deleted.
    virtual void deleteTagRelation( const std::string& parentTagName ) const;

    /// Show the tag in this node associated to the given parent node tag.
    /// Throws ReservedHeadTag if the parent tag is a HEAD tag.
    /// Throws TagNotFound if the parent tag does not exist in the parent node.
    /// Throws TagRelationNotFound if the parent tag has no related child tag.
    virtual const std::string
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
    virtual const std::string
    resolveTag( const std::string& ancestorTagName ) const;

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
                                    const std::string& description );

    // -- IHvsNodeRecord interface
    // no access to database, pass on to RelationalFolder

    /// Node full path name in the HVS hierarchy
    /// (this is always unique within a database)
    const std::string& fullPath() const
    {
      return m_hvsNode->fullPath();
    };

    /// Node description
    const std::string& description() const
    {
      return m_hvsNode->description();
    };

    /// Is this a leaf node?
    bool isLeaf() const
    {
      return m_hvsNode->isLeaf();
    };

    /// Has this node been stored into the database?
    bool isStored() const 
    { 
      return m_hvsNode->isStored(); 
    };

    /// Insertion time into the database
    /// Throws an exception if the node has not been stored yet
    const ITime& insertionTime() const
    {
      return m_hvsNode->insertionTime();
    };

    /// System-assigned node ID
    /// Throws an exception if the node has not been stored yet
    unsigned int id() const
    {
      return m_hvsNode->id();
    };

    /// System-assigned ID of the parent node
    /// Throws an exception if the node has not been stored yet
    /// Convention: parentId() = id() if the node has no parent (root node)
    unsigned int parentId() const
    {
      return m_hvsNode->parentId();
    };
  
  protected:

    /// Return the 'attributes' of the HVS node
    /// (implementation-specific properties not exposed in the API).
    /// FIXME can't call the implementation of the wrapped
    /// class, since it is protected.
    virtual const IRecord& nodeAttributes() const
    {
      return m_nodeAttributes;
    };

  protected:

    const RelationalDatabase& db() const 
    { 
      RelationalFolder* rf = dynamic_cast<RelationalFolder*>(m_hvsNode);
      RelationalFolderSet* rfs = dynamic_cast<RelationalFolderSet*>(m_hvsNode);
      if ( rf ) return rf->db(); // Fix Coverity FORWARD_NULL
      if ( rfs ) return rfs->db(); // Fix Coverity FORWARD_NULL
      throw InternalErrorException( "Could not dynamic cast hvsNode",
                                    "TransRelHvsNode::db" );
    }

  public:


    TransRelHvsNode( IHvsNode *hvsNode )
      : m_hvsNode( hvsNode )
    {
    };

    /// Destructor.
    virtual ~TransRelHvsNode()
    {
    };

  private:

    /// Standard constructor is private
    TransRelHvsNode();

    /// Copy constructor is private
    TransRelHvsNode( const TransRelHvsNode& rhs );

    /// Assignment operator is private
    TransRelHvsNode& operator=( const TransRelHvsNode& rhs );

  private:

    /// the wrapped RelationalFolder
    IHvsNode *m_hvsNode;

    /// FIXME? cant call nodeAttributes because it is protected.
    Record m_nodeAttributes;

  };

}

#endif
