// $Id: ConstRelationalObjectAdapter.h,v 1.19 2010-08-26 16:44:09 avalassi Exp $
#ifndef RELATIONALCOOL_CONSTRELATIONALOBJECTADAPTER_H
#define RELATIONALCOOL_CONSTRELATIONALOBJECTADAPTER_H

// Include files
#include "CoolKernel/ConstRecordAdapter.h"
#include "CoolKernel/Exception.h"
#include "CoolKernel/IObject.h"
#include "CoolKernel/IRecordSpecification.h"
//#include "CoolKernel/PayloadMode.h"
#include "PayloadMode.h" // TEMPORARY
#include "CoolKernel/Record.h"
#include "CoolKernel/Time.h"

// Local include files
#include "ConstTimeAdapter.h"
#include "ConstRecordIterator.h"

namespace cool {

  // Forward declarations
  class RelationalObjectTableRow;
  class RelationalObjectIterator;

  /** @class ConstRelationalObjectAdapter ConstRelationalObjectAdapter.h
   *
   *  Read-only wrapper of a constant coral::AttributeList reference,
   *  implementing the cool::IObject interface. The adapter can only be
   *  used as long as the AttributeList is alive. The adapter creates
   *  its own RecordSpecification from one specified at construction time.
   *
   *  @author Andrea Valassi
   *  @date   2007-03-09
   */
  class ConstRelationalObjectAdapter : public IObject {

  public:

    /// Constructor from a record spec and a _const_ AttributeList reference.
    ConstRelationalObjectAdapter( RelationalObjectIterator& iterator,
                                  const coral::AttributeList& aList,
                                  const IRecordSpecification& payloadSpec,
                                  PayloadMode::Mode pMode );

    /// Destructor
    virtual ~ConstRelationalObjectAdapter();

    /// Clone this IObject by performing a deep copy
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

    /// Has the object been stored into the database?
    bool isStored() const;

    /// System-assigned object ID
    /// Throws an exception if the object has not been stored yet
    unsigned int objectId() const;

    /// System-assigned payload ID
    /// Throws an exception if the object has not been stored yet
    /// Returns 0 if the payload is not stored in a separate table
    unsigned int payloadId() const;

    /// System-assigned payload set ID
    /// Throws an exception if the object has not been stored yet
    /// Returns 0 if the payload is not stored in a separate table
    unsigned int payloadSetId() const;
 
    /// Number of payloads for this IOV (vector folders)
    unsigned int payloadSize() const;

    /// Insertion time into the database
    /// Throws an exception if the object has not been stored yet
    const ITime& insertionTime() const;

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

    ConstRelationalObjectAdapter();

    ConstRelationalObjectAdapter( const ConstRelationalObjectAdapter& rhs );

    ConstRelationalObjectAdapter&
    operator=( const ConstRelationalObjectAdapter& rhs );

    /// Is timing active? (hack to activate it at the beginning of the ctor)
    bool isTimingActive() const;

  private:

    /// The underlying iterator (needed for vector folders)
    RelationalObjectIterator& m_iterator;

    /// Is timing active? (hack to activate it at the beginning of the ctor)
    bool m_isTimingActive;

    /// The input coral::AttributeList const reference
    const coral::AttributeList& m_aList;

    /// The input record specification const reference
    const IRecordSpecification& m_payloadSpec;

    /// Beginning of the interval of validity
    const ValidityKey& m_since;

    /// End of the interval of validity
    const ValidityKey& m_until;

    /// Object payload
    const ConstRecordAdapter m_payload;

    /// Vector payload iterator
    mutable ConstRecordIterator m_payloadIterator;

    /// Channel id
    const ChannelId& m_channelId;

    /// Insertion time
    const ConstTimeAdapter m_insertionTime;

    /// Object id
    const unsigned int& m_objectId;

    /// Null payload id (no payload table)
    const unsigned int m_payloadId0;

    /// Payload id
    const unsigned int& m_payloadId;

    /// Payload set id
    const unsigned int& m_payloadSetId;

    /// Default number of payloads
    const unsigned int m_payloadSize1;

    /// Number of payloads (vector folders)
    const unsigned int& m_payloadSize;

    /// Payload mode
    const PayloadMode::Mode m_payloadMode;

    /// Beginning of the original interval of validity
    //const ValidityKey& m_sinceOriginal;

    /// End of the original interval of validity
    //const ValidityKey& m_untilOriginal;

    /// Original insertion time
    //const Time& m_insertionTimeOriginal;

  };

}

#endif
