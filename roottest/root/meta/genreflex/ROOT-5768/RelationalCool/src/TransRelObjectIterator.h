// $Id: TransRelObjectIterator.h,v 1.1 2011-04-08 16:08:10 avalassi Exp $
#ifndef RELATIONALCOOL_TRANSRELOBJECTITERATOR_H
#define RELATIONALCOOL_TRANSRELOBJECTITERATOR_H

// Include files
#include "RelationalObjectIterator.h"

namespace cool
{

  /** @class TransRelObjectIterator TransRelObjectIterator.h
   *
   *  Transaction aware wrapper around a RelationalObjectIterator
   *
   *  TransRelObjectIterator takes the ownership of a RelationalTransaction
   *  which is committed when the RelationalObjectIterator is closed. 
   *
   *  @author Martin Wache
   *  @date   2010-11-04
   */

  class TransRelObjectIterator: public IObjectIterator {

  public:

    TransRelObjectIterator( IObjectIteratorPtr itPtr,
                            std::auto_ptr<RelationalTransaction> trans )
      : m_it( itPtr )
      , m_trans( trans )
    {
    };

    /// Destructor
    virtual ~TransRelObjectIterator()
    {
      // make sure the iterator and the transaction
      // are closed
      close();
    }

    /// Does the iterator have zero objects in the loop?
    virtual bool isEmpty()
    {
      return m_it->isEmpty();
    };

    /// NEW API AS OF COOL 2.2.0
    /// Fetch the next object in the iterator loop.
    /// Return false if there is no next object.
    virtual bool goToNext()
    {
      return m_it->goToNext();
    };

    /// NEW API AS OF COOL 2.2.0
    /// Retrieve a reference to the current object in the iterator loop.
    /// NB The reference is only valid until next() or goToNext() is called!
    /// Throw an exception if there is no current object (because the iterator
    /// is empty or is positioned before the first object in the loop).
    virtual const IObject& currentRef()
    {
      return m_it->currentRef();
    };

    /// Returns the 'length' of the iterator.
    virtual unsigned int size()
    {
      return m_it->size();
    };

    /// Returns all objects in the iterator as a vector.
    /// Throws an exception if next() has already retrieved one object:
    /// this method can only be called INSTEAD of the loop using next().
    virtual const IObjectVectorPtr fetchAllAsVector()
    {
      return m_it->fetchAllAsVector();
    };

    /// Close the iterator and release any associated server resources.
    /// The iterator cannot be used any more after this method is called.
    virtual void close();

    /// only to be used from tests
    IObjectIteratorPtr getIt()
    {
      return m_it;
    };

  protected:

    /// the wrapped iterator 
    IObjectIteratorPtr m_it;

    /// the active transaction
    std::auto_ptr<RelationalTransaction> m_trans;

  };

}

#endif
