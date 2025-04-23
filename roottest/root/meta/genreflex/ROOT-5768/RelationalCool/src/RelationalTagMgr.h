// $Id: RelationalTagMgr.h,v 1.72 2011-04-08 16:08:10 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALTAGMGR_H
#define RELATIONALCOOL_RELATIONALTAGMGR_H

// Include files
#include <memory>
//#include "CoolKernel/IHvsTagMgr.h"
#include "IHvsTagMgr.h"
#include "CoralBase/MessageStream.h"

namespace cool {

  // Forward declarations
  class RelationalDatabase;
  class RelationalNodeMgr;
  class RelationalQueryMgr;
  class RelationalTableRow;

  // TEMPORARY
  class RelationalHvsNode;

  /** @class RelationalTagMgr RelationalTagMgr.h
   *
   *  Abstract base class for a manager of a hierarchy
   *  of conditions database tags stored in a relational database.
   *
   *  @author Andrea Valassi and Marco Clemencic
   *  @date   2006-03-02
   */

  class RelationalTagMgr : public IHvsTagMgr {

  public:

    /// Destructor
    virtual ~RelationalTagMgr();

    /// Constructor from a relational database
    RelationalTagMgr( const RelationalDatabase& db );

    /// Return the type of node (inner/leaf) where this tag name can be used.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Throws TagNameNotFound if the tag name does not exist.
    IHvsNode::Type tagNameScope( const std::string& tagName ) const;

    /// Does a tag with this name exist (in any node)?
    /// Tag names, except for "HEAD", are case sensitive.
    /// Returns true for the reserved tags "" and "HEAD".
    bool existsTag( const std::string& tagName ) const;

    /// Return the names of the nodes where the tag is defined.
    /// Throws ReservedHeadTag for the HEAD tag (defined in all folders).
    /// Throws TagNotFound if the tag does not exist.
    const std::vector<std::string>
    taggedNodes( const std::string& tagName ) const;

    /// Find a tag record by nodeId and tag name.
    /// Throws ReservedHeadTag for the HEAD tag (defined in all folders).
    /// Throws TagNotFound if the tag does not exist.
    /// Starts a transaction.
    const HvsTagRecord findTagRecord( UInt32 nodeId,
                                      const std::string& tagName ) const;

    /// Find a tag record by nodeId and tagId.
    /// Throws ReservedHeadTag for the HEAD tag (defined in all folders).
    /// Throws TagNotFound if the tag does not exist.
    /// Starts a transaction.
    const HvsTagRecord findTagRecord( UInt32 nodeId,
                                      UInt32 tagId ) const;

    /// Create a relation between tags for a pair of parent/child nodes.
    /// Create the parent tag in the parent node if not defined yet.
    /// Create the child tag in the child node if not defined yet.
    /// Throws ReservedHeadTag if one of the two tags is a HEAD tag.
    /// Throws NodeIsSingleVersion if either node does not support versioning.
    /// Throws NodeRelationNotFound if the nodes are not parent and child.
    /// Throws TagExists if either tag is already used in another node.
    /// Throws TagRelationExists if a relation to a child tag already exists.
    void createTagRelation( UInt32 parentNodeId,
                            const std::string& parentTagName,
                            UInt32 childNodeId,
                            const std::string& childTagName ) const;

    /// Delete the relation between a parent tag node and a child tag.
    /// Delete the parent tag if not related to another parent/child tag.
    /// - NOT Delete the child tag if not related to another tag or IOVs.
    /// Throws ReservedHeadTag if the parent tag is a HEAD tag.
    /// Throws TagNotFound if the parent tag does not exist in the parent node.
    /// Throws TagRelationNotFound if the parent tag has no related child tag.
    /// - NEW returns the id of the deleted childTagId
    //void
    UInt32 deleteTagRelation( UInt32 parentNodeId,
                              const std::string& parentTagName,
                              UInt32 childNodeId ) const;

    /// Delete the relation between a parent tag node and a child tag.
    /// Delete the parent tag if not related to another parent/child tag.
    /// Delete the child tag if not related to another tag or IOVs.
    /// Throws ReservedHeadTag if the parent tag is a HEAD tag.
    /// Throws TagNotFound if the parent tag does not exist in the parent node.
    /// Throws TagRelationNotFound if the parent tag has no related child tag.
    void deleteTagRelation( const RelationalHvsNode& childNode,
                            const std::string& parentTagName ) const;

    /// Find the child node tag associated to the given parent node tag.
    /// Throws ReservedHeadTag if the parent tag is a HEAD tag.
    /// Throws TagNotFound if the parent tag does not exist in the parent node.
    /// Throws TagRelationNotFound if the parent tag has no related child tag.
    UInt32 findTagRelation( UInt32 parentNodeId,
                            const std::string& parentTagName,
                            UInt32 childNodeId ) const;

    /// Main HVS method: determine the descendant node tag that is related to
    /// the given ancestor tag (assumed to be defined in an ancestor node).
    /// The corresponding ancestor node is also internally determined.
    /// The ancestor tag is returned if defined directly in the descendant.
    /// Throws ReservedHeadTag if the ancestor tag is a HEAD tag.
    /// Throws TagNotFound if the tag does not exist in any inner node.
    /// Throws NodeRelationNotFound if the inner node where the ancestor tag
    /// is defined is not an ancestor of the descendant node.
    /// Throws TagRelationNotFound if no hierarchical tag relation exists.
    UInt32 resolveTag( const std::string& ancestorTagName,
                       UInt32 descendantNodeId ) const;

    /// Fetch the global tag table row for the given tagName.
    /// Throws TagNotFound if the tag does not exist in any node.
    const RelationalTableRow
    fetchGlobalTagTableRowForNode( const std::string& tagName ) const;

    /// Fetch all global tag table rows for the given nodeId.
    const std::vector<RelationalTableRow>
    fetchGlobalTagTableRows( UInt32 nodeId ) const;

    /// Fetch all global tag table rows for the given tagName.
    /// Throws TagNotFound if the tag does not exist in any node.
    const std::vector<RelationalTableRow>
    fetchGlobalTagTableRows( const std::string& tagName ) const;

    /// Fetch the global tag table row for the given nodeId and tagName
    const RelationalTableRow
    fetchGlobalTagTableRow( UInt32 nodeId,
                            const std::string& tagName ) const;

    /// Fetch the global tag table row for the given nodeId and tagId
    const RelationalTableRow
    fetchGlobalTagTableRow( UInt32 nodeId,
                            UInt32 tagId ) const;

    /// Insertion time of a tag defined for the given node
    /// (i.e. the time when the tag was first assigned to the node).
    const Time tagInsertionTime( const IHvsNode* node,
                                 const std::string& tagName ) const;

    /// Description of a tag defined for the given node.
    const std::string tagDescription( const IHvsNode* node,
                                      const std::string& tagName ) const;

    /// Lock status of a tag defined for the given node.
    void setTagLockStatus( const IHvsNode* node,
                           const std::string& tagName,
                           HvsTagLock::Status tagLockStatus );

    /// Lock status of a tag defined for the given node.
    HvsTagLock::Status tagLockStatus( const IHvsNode* node,
                                      const std::string& tagName ) const;

    /// This method does not handle transactions.
    /// Find a tag in an HVS node, else create it if it does not exist yet.
    const HvsTagRecord findOrCreateTag( UInt32 nodeId,
                                        const std::string& tagName ) const;

    /// TEMPORARY! Also fill the local tag table until it is removed.
    /// This method does not handle transactions.
    /// Create a tag in an HVS node.
    /// Throws TagExists if the tag already exists (in this or another node).
    const HvsTagRecord createTag( UInt32 nodeId,
                                  const std::string& tagName,
                                  const std::string& description = "" ) const;

    /// TEMPORARY! Remove this method when the local tag table is removed.
    /// Internal method.
    const HvsTagRecord createTagAndLocalTag
    ( UInt32 nodeId,
      const std::string& tagName,
      const std::string& description,
      const std::string& localTagTableName ) const;

    /// This method does not handle transactions.
    /// Delete a tag in an HVS node.
    void deleteTag( UInt32 nodeId,
                    const std::string& tagName ) const;

    /// Create a new entry in the global tag table.
    void insertGlobalTagTableRow( UInt32 nodeId,
                                  UInt32 tagId,
                                  const std::string& tagName,
                                  HvsTagLock::Status tagLockStatus,
                                  const std::string& tagDescription,
                                  const std::string& insertionTime ) const;

    /// Delete the row indicated by nodeId and tagId from the global tag table.
    void deleteGlobalTagTableRow( UInt32 nodeId,
                                  UInt32 tagId ) const;

    /// Delete all rows for nodeId from the global tag table.
    /// Returns the number of deleted rows.
    UInt32 deleteGlobalTagTableRowsForNode( const UInt32 nodeId ) const;

    /// TEMPORARY! The local tag table must be removed!
    /// Creates a new entry in the given local tag table
    /// Returns the tag id of the new entry
    void insertTagTableRow( const std::string& tagTableName,
                            UInt32 tagId,
                            const std::string& tagName,
                            const std::string& description,
                            const std::string& insertionTime ) const;

    /// TEMPORARY! The local tag table must be removed!
    /// Delete the row indicated by tagId from the given local tag table.
    void deleteTagTableRow( const std::string& tagTableName,
                            UInt32 tagId ) const;

    /// Does this user tag exist in the tag2tag table?
    bool existsTagInTag2TagTable( UInt32 nodeId,
                                  UInt32 tagId ) const;

    /*
    /// Does this tag defined in this node have any relation (i.e. does
    /// it reference a parent tag or is it referenced by any children)?
    /// Returns false if this tag does not exist in this node.
    bool isTagUsed( UInt32 nodeId,
                          UInt32 tagId ) const;
    */

    /// Delete all rows for the given node from the tag2tag table.
    UInt32 deleteTag2TagTableRowsForNode( const UInt32 nodeId ) const;

    /// Set the description of a tag.
    /// Throws TagNotFound the tag does not exist.
    /// Throws an Exception if the description is longer than 255 characters.
    void setTagDescription( const IHvsNode* node,
                            const std::string& tagName,
                            const std::string& description );


  protected:

    /// Get the RelationalDatabase reference
    const RelationalDatabase& db() const { return m_db; }

    /// Get a CORAL MessageStream
    coral::MessageStream& log() const { return *m_log; }

    /// Get a relational query manager
    RelationalQueryMgr& queryMgr() const;

    /// Get a relational node manager
    RelationalNodeMgr& nodeMgr() const;

  private:

    /// Standard constructor is private
    RelationalTagMgr();

    /// Copy constructor is private
    RelationalTagMgr( const RelationalTagMgr& rhs );

    /// Assignment operator is private
    RelationalTagMgr& operator=( const RelationalTagMgr& rhs );

    /// Create a new row in the tag2tag table.
    void insertTag2TagTableRow( UInt32 parentNodeId,
                                UInt32 parentTagId,
                                UInt32 childNodeId,
                                UInt32 childTagId,
                                const std::string& insertionTime ) const;

    /// Fetch the given row from the tag2tag table.
    /// Throw RowNotFound if no row was found.
    const RelationalTableRow
    fetchTag2TagTableRow( UInt32 parentNodeId,
                          UInt32 parentTagId,
                          UInt32 childNodeId ) const;

    /// Delete the given row row in the tag2tag table.
    /// Throw RowNotDeleted if no row was deleted.
    void deleteTag2TagTableRow( UInt32 parentNodeId,
                                UInt32 parentTagId,
                                UInt32 childNodeId ) const;

  protected:

    /// Reference to the RelationalDatabase
    const RelationalDatabase& m_db;

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

  };

}

#endif // RELATIONALCOOL_RELATIONALTAGMGR_H
