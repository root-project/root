// $Id: RecordVectorIterator.h,v 1.2 2010-08-26 16:44:09 avalassi Exp $
#ifndef RELATIONALCOOL_RECORDVECTORITERATOR_H
#define RELATIONALCOOL_RECORDVECTORITERATOR_H

// Include files
#include <boost/shared_ptr.hpp>
#include <vector>
//#include "CoolKernel/IRecordIterator.h"
#include "IRecordIterator.h" // TEMPORARY
#include "CoolKernel/pointers.h"

namespace cool {

  /** @class RecordVectorIterator RecordVectorIterator.h
   *
   *  Vector implementation of a COOL condition database object iterator.
   *
   *  Useful for the simplest implementation of object retrieval from
   *  the database: retrieving an iterator over the objects in a folder
   *  immediately retrieves the full vector of objects.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Martin Wache
   *  @date   2004-12-13
   */

  class RecordVectorIterator : public IRecordIterator {

  public:
  //  RecordVectorIterator();

    /// Constructor from a shared pointer to a vector of Records.
    /// The iterator is positioned BEFORE its first element.
    RecordVectorIterator( const IRecordVector& Records );
    /// Destructor.
    virtual ~RecordVectorIterator();

    /// Does the iterator have zero objects in the loop?
    bool isEmpty();

    /// Fetch the next object in the iterator loop.
    /// Return false if there is no next object.
    bool goToNext();

    /// Retrieve a reference to the current object in the iterator loop.
    /// NB The reference is only valid until next() or goToNext() is called!
    /// Throw an exception if there is no current object (because the iterator
    /// is empty or is positioned before the first object in the loop).
    const IRecord& currentRef();

    /// Returns the 'length' of the iterator.
    unsigned int size();

    /// Returns all objects in the iterator as a vector.
    const IRecordVectorPtr fetchAllAsVector();

    /// Close the iterator and release any associated server resources.
    /// The iterator cannot be used any more after this method is called.
    void close();

  private:

 
    /// Copy constructor is private
    RecordVectorIterator( const RecordVectorIterator& rhs );

    /// Assignment operator is private
    RecordVectorIterator& operator=( const RecordVectorIterator& rhs );

  private:
    /// Vector of objects
    const IRecordVector& m_objects;

    /// Current position in the loop
    unsigned int m_current1toN;

    /// Has the close() method been called?
    bool m_isClosed;
  };

}

#endif //RELATIONALCOOL_RECORDVECTORITERATOR_H
