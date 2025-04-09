// $Id: RelationalHvsNodeRecord.h,v 1.8 2009-12-16 17:17:37 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALHVSNODERECORD_H
#define RELATIONALCOOL_RELATIONALHVSNODERECORD_H

// Include files
#include "CoolKernel/Record.h"
#include "CoolKernel/Time.h"
#include "CoolKernel/IHvsNodeRecord.h"

// Local include files
#include "VersionNumber.h"

namespace cool {

  /** @class RelationalHvsNodeRecord RelationalHvsNodeRecord.h
   *
   *  Relational read-only implementation of one node in an HVS node tree.
   *
   *  @author Andrea Valassi and Sven A. Schmidt
   *  @date   2004-12-10
   */

  class RelationalHvsNodeRecord : virtual public IHvsNodeRecord {

  public:

    /// Destructor
    virtual ~RelationalHvsNodeRecord() {}

    /// Node full path name in the HVS hierarchy
    /// (this is always unique within a database)
    const std::string& fullPath() const;

    /// Node description
    const std::string& description() const;

    /// Is this a leaf node?
    bool isLeaf() const;

    /// Has this node been stored into the database?
    bool isStored() const { return true; }

    /// Schema version for this node
    const VersionNumber& schemaVersion() const;

    /// Insertion time into the database
    /// Throws an exception if the node has not been stored yet
    const ITime& insertionTime() const;

    /// System-assigned node ID
    /// Throws an exception if the node has not been stored yet
    unsigned int id() const;

    /// System-assigned ID of the parent node
    /// Throws an exception if the node has not been stored yet
    /// Convention: parentId() = id() if the node has no parent (root node)
    unsigned int parentId() const;

  protected:

    /// Return additional 'attributes' of the HVS node
    /// (implementation-specific attributes not exposed in the API)
    const IRecord& nodeAttributes() const;

    /// Constructor from a relational row retrieved from persistent storage
    RelationalHvsNodeRecord( const coral::AttributeList& nodeTableRow );

    /// Change the node description
    void setDescription( const std::string& description )
    {
      m_description = description;
    }

  private:

    /// Standard constructor is private
    RelationalHvsNodeRecord();

    /// Copy constructor is private
    RelationalHvsNodeRecord( const RelationalHvsNodeRecord& rhs );

    /// Assignment operator is private
    RelationalHvsNodeRecord& operator=( const RelationalHvsNodeRecord& rhs );

  private:

    /// System-assigned node ID
    unsigned int m_id;

    /// System-assigned ID of the parent node
    unsigned int m_parentId;

    /// Node full path name in the HVS hierarchy
    std::string m_fullPath;

    /// Node description
    std::string m_description;

    /// Is this a leaf node?
    bool m_isLeaf;

    /// Schema version for this node
    VersionNumber m_schemaVersion;

    /// Insertion time into the database
    Time m_insertionTime;

    /// Additional implementation-specific 'attributes' of the HVS node
    Record m_nodeAttributes;

  };

}

#endif // RELATIONALCOOL_RELATIONALHVSNODERECORD_H
