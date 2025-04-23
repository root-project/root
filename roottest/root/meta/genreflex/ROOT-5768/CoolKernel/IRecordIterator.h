// $Id: IRecordIterator.h,v 1.8 2012-07-08 20:02:33 avalassi Exp $
#ifndef COOLKERNEL_IRECORDITERATOR_H
#define COOLKERNEL_IRECORDITERATOR_H 1

// First of all, enable or disable the COOL290 API extensions (bug #92204)
#include "CoolKernel/VersionInfo.h"

#ifdef COOL290VP

// Include files
#include "CoolKernel/pointers.h"

namespace cool
{

  /** @class IRecordIterator IRecordIterator.h
   *
   *  Abstract interface to a conditions database record iterator.
   *  Based on the IObjectIterator interface
   *
   *  Only provides forward iteration, which maps naturally to RDBMS
   *  cursors. Backward iteration is disabled because it would force
   *  the implementation to retrieve the full result set as a vector.
   *
   *  Concrete classes implementing this API are required to position the
   *  iterator BEFORE the first record in the loop in their constructor to
   *  honour this interface. From a user point of view, this means that
   *  instances of an IRecordIterator are always retrieved in this state.
   *
   *  Usage pattern:
   *  - after retrieving the iterator, this is guaranteed to be positioned
   *    BEFORE the first record in the loop
   *  - use goToNext() to retrieve the next row in currentRef(); do this while
   *    goToNext() returns true, indicating that the next row is available
   *  - process the current row using currentRef(); if you need a copy for
   *    later processing, use currentRef().clone()
   *  - repeat the last two steps until goToNext() returns false
   *
   *  In practice:
   *    IRecordIterator& records = ... ;
   *    while ( records->goToNext() ) {
   *      const IRecord& rec = records.currentRef();
   *      ... ;
   *    }
   *
   *  If the close() method is called (any time during the iterator lifetime),
   *  the iterator cannot be reused any more (and it is best to delete it).
   *
   *  @author A. Valassi, S. A. Schmidt, M. Clemencic, M. Wache
   *  @date   2004-12-13
   */

  class IRecordIterator 
  {

  public:

    /// Destructor
    virtual ~IRecordIterator() {}

    /// Does the iterator have zero records in the loop?
    virtual bool isEmpty() = 0;


    /// Fetch the next record in the iterator loop.
    /// Return false if there is no next record.
    virtual bool goToNext() = 0;

    /// Retrieve a reference to the current record in the iterator loop.
    /// NB The reference is only valid until goToNext() is called!
    /// Throw an exception if there is no current record (because the iterator
    /// is empty or is positioned before the first record in the loop).
    virtual const IRecord& currentRef() = 0;

    /// Returns the 'length' of the iterator.
    /// This method might be deprecated in IObjectIterator and has not been 
    /// added to the IRecordIterator API. Use fetchAllAsVector() instead.
    ///virtual unsigned int size() = 0;

    /// Returns all records in the iterator as a vector.
    /// Throws an exception if goToNext() has already retrieved one record:
    /// this method can only be called INSTEAD of the loop using goToNext().
    virtual const IRecordVectorPtr fetchAllAsVector() = 0;

    /// Close the iterator and release any associated server resources.
    /// The iterator cannot be used any more after this method is called.
    virtual void close() = 0;

  private:

    /// Assignment operator is private (see bug #95823)
    IRecordIterator& operator=( const IRecordIterator& rhs );

  };

}
#endif // COOL290VP

#endif // COOLKERNEL_IRECORDITERATOR_H
