// $Id: IObjectIterator.h,v 1.35 2012-07-08 20:02:33 avalassi Exp $
#ifndef COOLKERNEL_IOBJECTITERATOR_H
#define COOLKERNEL_IOBJECTITERATOR_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include "CoolKernel/pointers.h"

namespace cool
{

  /** @class IObjectIterator IObjectIterator.h
   *
   *  Abstract interface to a conditions database object iterator.
   *  Based on the IDataIterator interface from the original CondDB API.
   *
   *  Only provides forward iteration, which maps naturally to RDBMS
   *  cursors. Backward iteration is disabled because it would force
   *  the implementation to retrieve the full result set as a vector.
   *
   *  Concrete classes implementing this API are required to position the
   *  iterator BEFORE the first object in the loop in their constructor to
   *  honour this interface. From a user point of view, this means that
   *  instances of an IObjectIterator are always retrieved in this state.
   *  Before COOL 1.3.0, it was recommended to call the goToStart() method
   *  to initialize the iterator: presently, this method has been obsoleted.
   *
   *  A new API (goToNext/currentRef) has been defined in COOL 2.2.0.
   *  The previous API (next/hasNext/current) has been removed in COOL 2.6.0.
   *
   *  Usage pattern:
   *  - after retrieving the iterator, this is guaranteed to be positioned
   *    BEFORE the first object in the loop
   *  - use goToNext() to retrieve the next row in currentRef(); do this while
   *    goToNext() returns true, indicating that the next row is available
   *  - process the current row using currentRef(); if you need a copy for
   *    later processing, use currentRef().clone()
   *  - repeat the last two steps until goToNext() returns false
   *
   *  In practice:
   *    IObjectIteratorPtr objects = ... ;
   *    while ( objects->goToNext() ) {
   *      const IObject& obj = objects->currentRef();
   *      ... ;
   *    }
   *
   *  If the close() method is called (any time during the iterator lifetime),
   *  the iterator cannot be reused any more (and it is best to delete it).
   *
   *  @author A. Valassi, S. A. Schmidt, M. Clemencic, M. Wache
   *  @date   2004-12-13
   */

  class IObjectIterator
  {

  public:

    /// Destructor
    virtual ~IObjectIterator() {}

    /// Does the iterator have zero objects in the loop?
    virtual bool isEmpty() = 0;

    /*
    /// OBSOLETE! REMOVED IN COOL 2.6.0!
    /// Does the iterator have any objects after the current one in the loop?
    virtual bool hasNext() = 0;

    /// OBSOLETE! REMOVED IN COOL 2.6.0!
    /// Retrieve a shared pointer to the current object in the iterator loop.
    /// Throw an exception if there is no current object (because the iterator
    /// is empty or is positioned before the first object in the loop).
    virtual const IObjectPtr current() = 0;

    /// OBSOLETE! REMOVED IN COOL 2.6.0!
    /// Retrieve a shared pointer to the next object in the iterator loop.
    /// Throw an exception if there is no next object (because the iterator
    /// is empty or is positioned after the last object in the loop).
    virtual const IObjectPtr next() = 0;
    */

    /// NEW API AS OF COOL 2.2.0
    /// Fetch the next object in the iterator loop.
    /// Return false if there is no next object.
    virtual bool goToNext() = 0;

    /// NEW API AS OF COOL 2.2.0
    /// Retrieve a reference to the current object in the iterator loop.
    /// NB The reference is only valid until next() or goToNext() is called!
    /// Throw an exception if there is no current object (because the iterator
    /// is empty or is positioned before the first object in the loop).
    virtual const IObject& currentRef() = 0;

    /*
    /// OBSOLETE! NOT NEEDED SINCE COOL 1.3.0! REMOVED IN COOL 2.6.0!
    /// Position the iterator BEFORE the first object in the loop.
    /// The first object in the loop can be retrieved by calling next().
    /// Throw an exception if goToStart() called after next() has already
    /// retrieved one or more objects: iterators can only be used ONCE.
    virtual void goToStart() = 0;
    */

    /// Returns the 'length' of the iterator.
    virtual unsigned int size() = 0;

    /// Returns all objects in the iterator as a vector.
    /// Throws an exception if next() has already retrieved one object:
    /// this method can only be called INSTEAD of the loop using next().
    virtual const IObjectVectorPtr fetchAllAsVector() = 0;

    /// Close the iterator and release any associated server resources.
    /// The iterator cannot be used any more after this method is called.
    virtual void close() = 0;

#ifdef COOL290CO
  private:

    /// Assignment operator is private (see bug #95823)
    IObjectIterator& operator=( const IObjectIterator& rhs );
#endif

  };

}
#endif // COOLKERNEL_IOBJECTITERATOR_H
