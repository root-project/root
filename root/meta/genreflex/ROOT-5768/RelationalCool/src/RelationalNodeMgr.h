// $Id: RelationalNodeMgr.h,v 1.25 2010-03-29 14:26:55 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALNODEMGR_H
#define RELATIONALCOOL_RELATIONALNODEMGR_H

// Include files
#include <memory>
#include "CoolKernel/types.h"
#include "CoralBase/AttributeList.h"
#include "CoralBase/MessageStream.h"

namespace cool {

  // Forward declarations
  class RelationalDatabase;
  class RelationalQueryMgr;
  class RelationalTableRow;

  /** @class RelationalNodeMgr RelationalNodeMgr.h
   *
   *  Abstract base class for a manager of a hierarchy
   *  of conditions database nodes stored in a relational database.
   *
   *  @author Andrea Valassi and Marco Clemencic
   *  @date   2006-03-02
   */

  class RelationalNodeMgr {

  public:

    /// Destructor
    virtual ~RelationalNodeMgr();

    /// Constructor from a RelationalDatabase reference
    RelationalNodeMgr( const RelationalDatabase& db );

    /// Does this node exist?
    bool existsNode( const std::string& fullPath );

    /// Does this folder set exist?
    bool existsFolderSet( const std::string& folderSetName );

    /// Does this folder exist?
    bool existsFolder( const std::string& folderSetName );

    /// Return the list of existing nodes (ordered alphabetically
    /// ascending/descending)
    const std::vector<std::string> listAllNodes( bool ascending = true );

    /// Return the list of nodes inside the given nodeId with the attribute
    /// isLeaf as specified (ordered by name asc/desc)
    const std::vector<std::string> listNodes( unsigned int nodeId,
                                              bool isLeaf,
                                              bool ascending = true );

    /// Fetch all node rows
    const std::vector<RelationalTableRow>
    fetchAllNodeTableRows( bool ascending = true ) const;

    /// Fetch one node row (lookup by 1 node fullPath)
    const RelationalTableRow
    fetchNodeTableRow( const std::string& fullPath ) const;

    /// Fetch one node row (lookup by 1 nodeId)
    const RelationalTableRow
    fetchNodeTableRow( unsigned int nodeId ) const;

    /// Fetch one node row (lookup with given WHERE clause and bind variables)
    const RelationalTableRow 
    fetchNodeTableRow( const std::string& whereClause,
                       const Record& whereData ) const;

    /// List all nodes in the hierarchy line between ancestor and descendant
    /// (ordered from the ancestor's child to the descendant itself included)
    const std::vector<UInt32>
    resolveNodeHierarchy( UInt32 ancestorNodeId,
                          UInt32 descendantNodeId ) const;

  protected:

    /// Get the RelationalDatabase reference
    const RelationalDatabase& db() const { return m_db; }

    /// Get a CORAL MessageStream
    coral::MessageStream& log() const { return *m_log; }

    /// Get a relational query manager
    RelationalQueryMgr& queryMgr() const;

  private:

    /// Standard constructor is private
    RelationalNodeMgr();

    /// Copy constructor is private
    RelationalNodeMgr( const RelationalNodeMgr& rhs );

    /// Assignment operator is private
    RelationalNodeMgr& operator=( const RelationalNodeMgr& rhs );

  protected:

    /// Reference to the RelationalDatabase
    const RelationalDatabase& m_db;

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

  };

}

#endif // RELATIONALCOOL_RELATIONALNODEMGR_H
