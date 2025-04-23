// $Id: ObjectIteratorCounter.h,v 1.10 2011-04-18 15:09:00 avalassi Exp $
#ifndef OBJECTITERATORCOUNTER_H
#define OBJECTITERATORCOUNTER_H 1

// Include files
#include <map>
#include <vector>
#include "CoralBase/boost_thread_headers.h"

namespace cool
{

  // Forward declarations
  class IRelationalTransactionMgr;
  class IObjectIterator;

  typedef
  std::vector< const IObjectIterator* > ObjectIteratorVector;

  typedef
  std::map< const IRelationalTransactionMgr*,
            ObjectIteratorVector > ObjectIteratorMap;

  /** @class ObjectIteratorCounter ObjectIteratorCounter.h
   *
   *  Static counter of open 'live' iterators.
   *
   *  @author Andrea Valassi
   *  @date   2007-03-30
   */

  class ObjectIteratorCounter {

  public:
    
    // Test if an Iterator is active for TransactionMgr
    static bool testIteratorActive( const IRelationalTransactionMgr* trMgr );

    // Register an iterator
    static void registerIterator( const IObjectIterator* it,
                                  const IRelationalTransactionMgr* trMgr );

    // Unregister an iterator
    static void unregisterIterator( const IObjectIterator* it,
                                    const IRelationalTransactionMgr* trMgr );

  private:

    virtual ~ObjectIteratorCounter();
    ObjectIteratorCounter();
    ObjectIteratorCounter( const ObjectIteratorCounter& rhs );
    ObjectIteratorCounter& operator=( const ObjectIteratorCounter& rhs );

  private:

    /// List of active 'live' iterators (there should be only one!)
    static ObjectIteratorMap& openIterators();

    /// Mutex to protect the open iterators map
    static boost::mutex& openIteratorsMutex();

  };

}
#endif // OBJECTITERATORCOUNTER_H
