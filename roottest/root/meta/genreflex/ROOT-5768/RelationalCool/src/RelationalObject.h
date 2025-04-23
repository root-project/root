// $Id: RelationalObject.h,v 1.47 2012-06-29 13:36:43 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALOBJECT_H
#define RELATIONALCOOL_RELATIONALOBJECT_H

// Include files
#include <vector>
#include "CoolKernel/Exception.h"
#include "CoolKernel/IObject.h"
#include "CoolKernel/IRecordSpecification.h"
#include "CoolKernel/Record.h"
#include "CoolKernel/Time.h"

// Local include files
#include "RecordVectorIterator.h"
#include "PayloadMode.h"

namespace cool {

  // Forward declarations
  class RelationalObjectTableRow;

  /** @class RelationalObject RelationalObject.h
   *
   *  Generic relational implementation of a COOL condition database "object"
   *
   *  @author Sven A. Schmidt, Andrea Valassi and Marco Clemencic
   *  @date   2004-11-30
   */

  class RelationalObject : public IObject {

  public:

    /// Constructor of an object scheduled to be stored persistently
    RelationalObject( const ValidityKey& since,
                      const ValidityKey& until,
                      const IRecord& payload,
                      const ChannelId& channelId,
                      const std::string& userTagName = "" );

    /// Constructor of an object scheduled to be stored persistently
    /// to be used when the payload is already stored in a separate table,
    /// and an other IOV wants to use it.
    RelationalObject( const ValidityKey& since,
                      const ValidityKey& until,
                      const unsigned int payloadId,
                      const ChannelId& channelId,
                      const std::string& userTagName = "" );

    /// Constructor of an object scheduled to be stored persistently
    /// to be used when the payload is already stored in a separate table,
    /// and an other IOV wants to use it.
    RelationalObject( const ValidityKey& since,
                      const ValidityKey& until,
                      const unsigned int payloadSetId,
                      const unsigned int payloadSize,
                      const ChannelId& channelId,
                      const std::string& userTagName = "" );

    /// Constructor of an object scheduled to be stored persistently
    /// to be used when the payload is already stored in a separate table,
    /// and an other IOV wants to use it.
    RelationalObject( const ValidityKey& since,
                      const ValidityKey& until,
                      const std::vector<IRecordPtr>& payload,
                      const ChannelId& channelId,
                      const std::string& userTagName = "" );

    /*
    /// Constructor of an object retrieved from persistent storage
    /// Note: the row can not be const, because the attributes are shared
    /// TEMPORARY? A simple ALS is enough in input: an extended ALS would
    /// be needed only if COOL had to check that data read back from the
    /// database passes the size checks. For the moment we choose to only
    /// implement size checks when WRITING the data into the database
    /// (and if data is only inserted through the C++ API, no checks are
    /// needed in reading back anyway...)
    RelationalObject
    ( RelationalObjectTableRow& row,
      const IRecordSpecification& payloadSpec );
    */

    /// Constructor of an object retrieved from persistent storage
    /// AV 2007.03.26 Use a coral::AttributeList to make this faster.
    /// Observed query time reduction from 0.31s to 0.27s on slc3_ia32_gcc323
    /// for the benchmark Atlas prompt reconstruction query (100MB, 100k rows).
    RelationalObject
    ( const coral::AttributeList& aList,
      const IRecordSpecification& payloadSpec,
      PayloadMode::Mode pMode );

    /// Destructor
    virtual ~RelationalObject() {}

    /// deep copy clone
    virtual IObject* clone() const;

    /// Channel identifier
    const ChannelId& channelId() const;

    /// Channel name
    //const std::string& channelName() const;

    /// Start of validity interval
    /// For stored objects this refers to the visible validity interval
    const ValidityKey& since() const;

    /// End of validity interval
    /// For stored objects this refers to the visible validity interval
    const ValidityKey& until() const;

    /// Data payload
    const IRecord& payload() const;

    /// Vector data payload
    virtual IRecordIterator& payloadIterator() const;

    /// The user tag id
    unsigned int userTagId() const;

    /// The user tag name this objects has been assigned to
    const std::string& userTagName() const;

    /// Has the object been stored into the database?
    bool isStored() const;

    /// System-assigned object ID
    /// Throws an exception if the object has not been stored yet
    unsigned int objectId() const;
    void setObjectId( unsigned int oId )
    {
      m_objectId = oId;
    };

    /// System-assigned payload ID
    /// Throws an exception if the object has not been stored yet
    /// Returns 0 if the payload is not stored in a separate table
    unsigned int payloadId() const;
    void setPayloadId( unsigned int pId )
    {
      m_payloadId = pId;
    };

    /// System-assigned payload ID
    /// Throws an exception if the object has not been stored yet
    /// Returns 0 if the payload is not stored in a separate table
    unsigned int payloadSetId() const;
    void setPayloadSetId( unsigned int pId )
    {
      m_payloadSetId = pId;
    };
    /// number of payloads (vector folders)
    unsigned int payloadSize() const;
    void setPayloadNItems( unsigned int pId )
    {
      m_payloadSize = pId;
    };

    /// Insertion time into the database
    /// Throws an exception if the object has not been stored yet
    const ITime& insertionTime() const;
    void setInsertionTime( const ITime& time )
    {
      m_insertionTime = time;
    };

    /// Start of original validity interval
    /// Throws an exception if the object has not been stored yet
    //const ValidityKey& sinceOriginal() const;

    /// End of original validity interval
    /// Throws an exception if the object has not been stored yet
    //const ValidityKey& untilOriginal() const;

    /// Insertion time of the original object into the database
    /// Throws an exception if the object has not been stored yet
    //const ITime& insertionTimeOriginal() const;

    /// Pretty print to an output stream
    std::ostream& print( std::ostream& s ) const;

  private:

    RelationalObject();
    RelationalObject( const RelationalObject& rhs );
    RelationalObject& operator=( const RelationalObject& rhs );

    /// Beginning of the interval of validity
    ValidityKey m_since;

    /// End of the interval of validity
    ValidityKey m_until;

    /// Object payload pointer- this is always owned by the RelationalObject
    IRecordPtr m_payloadPtr;
 
    /// vector payload - this is always owned by the RelationalObject -- DEEP COPY!
    std::vector<IRecordPtr> m_payloadVector;

    /// the payload vector iterator
    mutable RecordVectorIterator m_payloadIterator;

    /// Channel id
    ChannelId m_channelId;

    /// User tag name
    std::string m_userTagName;

    /// Insertion time
    Time m_insertionTime;

    /// Object id
    unsigned int m_objectId;

    /// payload id
    unsigned int m_payloadId;

    /// payload set id
    unsigned int m_payloadSetId;

    /// number of payloads (vector folders)
    unsigned int m_payloadSize;

    /// User tag id
    unsigned int m_userTagId;

    /// Beginning of the original interval of validity
    //ValidityKey m_sinceOriginal;

    /// End of the original interval of validity
    //ValidityKey m_untilOriginal;

    /// Original insertion time
    //Time m_insertionTimeOriginal;

  };

}

#endif
