// $Id: RelationalObjectTableRow.h,v 1.37 2010-08-24 19:39:29 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALOBJECTTABLEROW_H
#define RELATIONALCOOL_RELATIONALOBJECTTABLEROW_H

// Include files
#include "CoolKernel/IObject.h"
#include "CoolKernel/Time.h"
#include "CoolKernel/ValidityKey.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeListException.h"

// Local include files
#include "RelationalObjectTable.h"
#include "RelationalTableRow.h"
#include "RelationalTableRowBase.h"

namespace cool {

  /** @class RelationalObjectTableRow RelationalObjectTableRow.h
   *
   *  Representation of a RelationalObject as a row in the object table.
   *  Used internally to read/write data from/into persistent storage.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2005-02-09
   */

  class RelationalObjectTableRow : public RelationalTableRowBase
  {

  public:

    /// Destructor
    virtual ~RelationalObjectTableRow();

    /// Constructor from an AttributeList.
    /// Performs a deep copy of the AttributeList values.
    explicit RelationalObjectTableRow( const coral::AttributeList& data );

    /// Constructor from an IObjectPtr.
    /// For performance reasons, only copies references to the object payload:
    /// a row is expected to never outlive the object it originated from.
    explicit RelationalObjectTableRow( const IObjectPtr& object, 
                                       PayloadMode::Mode mode = PayloadMode::INLINEPAYLOAD );

    /// Constructor from a RelationalTableRow.
    /// This constructor is used to 'reinterpret' a generic table row as
    /// obtained from fetch methods as a RelationalObjectTableRow.
    explicit RelationalObjectTableRow( const RelationalTableRow& row );

    /// Copy constructor
    /// Performs a deep copy of the AttributeList values.
    RelationalObjectTableRow( const RelationalObjectTableRow& aRow );

    /// Assignment operator.
    /// Performs a deep copy of the AttributeList values.
    RelationalObjectTableRow& operator=( const RelationalObjectTableRow& rhs );

    /// Comparison operator -- only takes into account object_id
    bool operator==( const RelationalObjectTableRow& rhs )
    {
      return objectId() == rhs.objectId();
    }

    /// Returns true if the given point in time lies in the row's IOV
    bool contains( const ValidityKey& pointInTime )
    {
      return since() <= pointInTime && pointInTime < until();
    }

    /// Returns the object id
    unsigned int objectId() const;

    /// Returns the beginning of the IOV
    ValidityKey since() const;

    /// Returns the beginning of the IOV
    ValidityKey until() const;

    /// Returns the channel id
    ChannelId channelId() const;

    /// Returns the user tag id
    unsigned int userTagId() const;

    /// Returns the original id
    unsigned int originalId() const;

    /// Returns the new head id
    unsigned int newHeadId() const;

    /// Returns the insertion time
    Time insertionTime() const;

    /// Returns the last modification date
    Time lastModDate() const;

    /// Returns the payload id
    unsigned int payloadId() const;

    /// Returns the payload set id
    unsigned int payloadSetId() const;

    /// Returns the number of payloads
    unsigned int payloadSize() const;

    /// Sets the object id
    void setObjectId( unsigned int objectId );

    /// Sets the payload id
    void setPayloadId( unsigned int payloadId );

    /// Sets the payload set id
    void setPayloadSetId( unsigned int payloadSetId );

    /// Sets the number of payloads (vector folders)
    void setPayloadNItems( unsigned int payloadSize);

    /// Sets the beginning of the IOV
    void setSince( const ValidityKey& since );

    /// Sets the end of the IOV
    void setUntil( const ValidityKey& until );

    /// Sets the user tag id
    void setUserTagId( unsigned int userTagId );

    /// Sets the original id
    void setOriginalId( unsigned int value );

    /// Sets the new head id
    void setNewHeadId( unsigned int value );

    /// Data payload value for a specific payload item - returned as true type
    template<class T> const T& payloadValue( const std::string& name ) const
    {
      // AV 19.06.2006 Old version that works for Linux gcc323
      //return m_data[name].data();
      // SAS 19.06.2006 changed for linux gcc403 but fails for Linux gcc323
      //return m_data[name].data<T>();
      // MCl 28.03.2006 this one works on gcc 4.x and 3.2
      return m_data[name].template data<T>();
    }

  private:

    /// Standard constructor is private.
    RelationalObjectTableRow();

  };

  /// Streamer for RelationalObjectTableRow objects
  inline std::ostream &operator<<
    ( std::ostream& s, const RelationalObjectTableRow& r )
  {
    s << r.objectId() << " [" << r.since() << "," << r.until() << "] ";
    s << "[";
    bool first = true;
    for ( coral::AttributeList::const_iterator attr = r.begin();
          attr != r.end(); ++attr ) {
      if ( first ) {
        first = false;
      } else {
        s << "|";
      }
      attr->toOutputStream( s );
    }
    s << "] ";
    try {
      s << r.originalId() << " " << r.newHeadId();
    } catch ( coral::AttributeListException& ) { /* ignored */ }
    return s;
  }

  /// Comparison functor to compare RelationalObjectTableRow by their objectId
  struct eq_objectId :
    public std::binary_function<RelationalObjectTableRow, unsigned int, bool>
  {
    bool operator()( const RelationalObjectTableRow& r,
                     unsigned int objectId ) const {
      return r.objectId() == objectId;
    }
  };

  /// Less than comparison functor to compare RelationalObjectTableRow
  /// by their objectId
  struct lt_objectId :
    public std::binary_function< RelationalObjectTableRow,
    RelationalObjectTableRow, bool>
  {
    bool operator()( const RelationalObjectTableRow& lhs,
                     const RelationalObjectTableRow& rhs ) const {
      return lhs.objectId() < rhs.objectId();
    }
  };

}

#endif // RELATIONALCOOL_RELATIONALOBJECTTABLEROW_H
