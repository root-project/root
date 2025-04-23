// $Id: RelationalPayloadTableRow.h,v 1.1 2010-08-05 11:53:25 mwache Exp $
#ifndef RELATIONALCOOL_RELATIONALPAYLOADTABLEROW_H
#define RELATIONALCOOL_RELATIONALPAYLOADTABLEROW_H

// Include files
#include "CoolKernel/IObject.h"
#include "CoolKernel/Time.h"
#include "CoolKernel/ValidityKey.h"
#include "CoralBase/Attribute.h"
#include "CoralBase/AttributeListException.h"

// Local include files
#include "RelationalTableRow.h"
#include "RelationalTableRowBase.h"

namespace cool {

  /** @class RelationalPayloadTableRow RelationalPayloadTableRow.h
   *
   *  Representation of a RelationalObject as a row in the object table.
   *  Used internally to read/write data from/into persistent storage.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2005-02-09
   */

  class RelationalPayloadTableRow : public RelationalTableRowBase
  {

  public:

    /// Destructor
    virtual ~RelationalPayloadTableRow();

    /// Constructor from an AttributeList.
    /// Performs a deep copy of the AttributeList values.
    explicit RelationalPayloadTableRow( const coral::AttributeList& data );

    /// Constructor from an IObjectPtr.
    /// For performance reasons, only copies references to the object payload:
    /// a row is expected to never outlive the object it originated from.
    explicit RelationalPayloadTableRow( const IObjectPtr& object,
                                        PayloadMode::Mode pMode );

    /// Constructor from an IRecordPtr.
    /// For performance reasons, only copies references to the object payload:
    /// a row is expected to never outlive the object it originated from.
    explicit RelationalPayloadTableRow( const IRecord& Record,
                                        PayloadMode::Mode pMode );

    /// Constructor from a RelationalPayloadTableRow.
    /// This constructor is used to 'reinterpret' a generic table row as
    /// obtained from fetch methods as a RelationalPayloadTableRow.
    explicit RelationalPayloadTableRow( const RelationalTableRow& row );

    /// Copy constructor
    /// Performs a deep copy of the AttributeList values.
    RelationalPayloadTableRow( const RelationalPayloadTableRow& aRow );

    /// Assignment operator.
    /// Performs a deep copy of the AttributeList values.
    RelationalPayloadTableRow& operator=( const RelationalPayloadTableRow& rhs );


    /// Returns the payload id
    unsigned int payloadId() const;

    /// Sets the payload id
    void setPayloadId( unsigned int payloadId );

    /// Returns the payload set id
    unsigned int payloadSetId() const;

    /// Sets the payload set id
    void setPayloadSetId( unsigned int payloadSetId );

    /// Returns the payload item id
    unsigned int payloadItemId() const;

    /// Sets the payload item id
    void setPayloadItemId( unsigned int payloadItemId );

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
    RelationalPayloadTableRow();

  };

  /// Streamer for RelationalPayloadTableRow objects
  inline std::ostream &operator<<
    ( std::ostream& s, const RelationalPayloadTableRow& r )
  {
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
    return s;
  }

}

#endif // RELATIONALCOOL_RELATIONALPAYLOADTABLEROW_H
