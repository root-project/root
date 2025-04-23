// $Id: ConstRecordIterator.h,v 1.4 2010-08-26 16:44:09 avalassi Exp $
#ifndef RELATIONALCOOL_CONSTRECORDITERATOR_H
#define RELATIONALCOOL_CONSTRECORDITERATOR_H

// Include files
#include "CoolKernel/ConstRecordAdapter.h"
#include "CoolKernel/Exception.h"
//#include "CoolKernel/IRecordIterator.h"
//#include "CoolKernel/PayloadMode.h"
#include "IRecordIterator.h" // TEMPORARY
#include "PayloadMode.h" // TEMPORARY
#include "CoolKernel/Record.h"
#include "CoolKernel/Time.h"

namespace cool 
{

  // Forward declarations
  class RelationalObjectIterator;

  /** @class ConstRecordIterator
   *
   *  Read-only wrapper of a constant coral::AttributeList reference,
   *  implementing the cool::IRecordIterator interface. The adapter can only be
   *  used as long as the AttributeList is alive. The adapter creates
   *  its own RecordSpecification from one specified at construction time.
   *
   *  @author Martin Wache
   *  @date   2010-05-20
   */

  class ConstRecordIterator : public IRecordIterator 
  {

  public:

    ConstRecordIterator( RelationalObjectIterator& iterator,
                         const coral::AttributeList& aList,
                         const ConstRecordAdapter& record );

    /// Destructor
    virtual ~ConstRecordIterator() {};

    /// Does the iterator have zero objects in the loop?
    virtual bool isEmpty();


    /// Fetch the next record in the iterator loop.
    /// Return false if there is no next record.
    virtual bool goToNext();

    /// Retrieve a reference to the current object in the iterator loop.
    /// NB The reference is only valid until goToNext() is called!
    /// Throw an exception if there is no current object (because the iterator
    /// is empty or is positioned before the first object in the loop).
    virtual const IRecord& currentRef();

    /*
    /// Returns the 'length' of the iterator.
    /// This method might be deprecated in IObjectIterator and has not been 
    /// added to the IRecordIterator API. Use fetchAllAsVector() instead.
    ///virtual unsigned int size() = 0;
    */

    /// Returns all records in the iterator as a vector.
    /// Throws an exception if goToNext() has already retrieved one object:
    /// this method can only be called INSTEAD of the loop using goToNext().
    virtual const IRecordVectorPtr fetchAllAsVector();

    /// Close the iterator and release any associated server resources.
    /// The iterator cannot be used any more after this method is called.
    virtual void close();

    /// places the iterator on the next payload set.
    /// Returns false if no next payload set exists
    bool nextPayloadSet();
    
  private:

    ConstRecordIterator();

    ConstRecordIterator( const ConstRecordIterator& rhs );

    ConstRecordIterator& operator=( const ConstRecordIterator& rhs );

  private:

    /// underlying iterator over the iovs
    RelationalObjectIterator& m_iterator;

    /// the payload's attribute list
    const coral::AttributeList& m_aList;

    /// the payload adaptor
    const ConstRecordAdapter& m_payload;

    /// the payload set id of this payload vector
    unsigned int m_payloadSetId;

    /// the payload count of this payload vector
    unsigned int m_payloadSize;

    /// current object count
    unsigned int m_currentObject;

    /// State of the iterator (active, end of rows, closed; or countonly)
    enum { ACTIVE, END_OF_ROWS, CLOSED } m_state;

  };

}
#endif // RELATIONALCOOL_CONSTRECORDITERATOR_H
