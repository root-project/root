// $Id: ObjectVectorIterator.h,v 1.22 2009-12-16 17:17:37 avalassi Exp $
#ifndef RELATIONALCOOL_OBJECTVECTORITERATOR_H
#define RELATIONALCOOL_OBJECTVECTORITERATOR_H

// Include files
#include "CoolKernel/IObjectIterator.h"

namespace cool {

  /** @class ObjectVectorIterator ObjectVectorIterator.h
   *
   *  Vector implementation of a COOL condition database object iterator.
   *
   *  Useful for the simplest implementation of object retrieval from
   *  the database: retrieving an iterator over the objects in a folder
   *  immediately retrieves the full vector of objects.
   *
   *  @author Andrea Valassi and Sven A. Schmidt
   *  @date   2004-12-13
   */

  class ObjectVectorIterator : public IObjectIterator {

  public:

    /// Constructor from a shared pointer to a vector of objects.
    /// The iterator is positioned BEFORE its first element.
    ObjectVectorIterator( const IObjectVectorPtr& objects );

    /// Destructor.
    virtual ~ObjectVectorIterator();

    /// Does the iterator have zero objects in the loop?
    bool isEmpty();

    /// Fetch the next object in the iterator loop.
    /// Return false if there is no next object.
    bool goToNext();

    /// Retrieve a reference to the current object in the iterator loop.
    /// NB The reference is only valid until next() or goToNext() is called!
    /// Throw an exception if there is no current object (because the iterator
    /// is empty or is positioned before the first object in the loop).
    const IObject& currentRef();

    /// Returns the 'length' of the iterator.
    unsigned int size();

    /// Returns all objects in the iterator as a vector.
    const IObjectVectorPtr fetchAllAsVector();

    /// Close the iterator and release any associated server resources.
    /// The iterator cannot be used any more after this method is called.
    void close();

  private:

    /// Standard constructor is private.
    ObjectVectorIterator();

    /// Copy constructor is private
    ObjectVectorIterator( const ObjectVectorIterator& rhs );

    /// Assignment operator is private
    ObjectVectorIterator& operator=( const ObjectVectorIterator& rhs );

  private:

    /// Vector of objects
    IObjectVectorPtr m_objects;

    /// Size of the vector of objects
    unsigned int m_size;

    /// Current position in the loop
    unsigned int m_current1toN;

    /// Has the close() method been called?
    bool m_isClosed;

  };

}

#endif //RELATIONALCOOL_OBJECTVECTORITERATOR_H
