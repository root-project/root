// $Id: IteratorException.h,v 1.7 2009-12-16 17:17:37 avalassi Exp $
#ifndef RELATIONALCOOL_ITERATOREXCEPTION_H
#define RELATIONALCOOL_ITERATOREXCEPTION_H 1

// Include files
#include "CoolKernel/Exception.h"

namespace cool
{
  //--------------------------------------------------------------------------

  /** @class IteratorHasNotStarted
   *
   *  Exception thrown when an invalid operation is attempted on an iterator
   *  that has not been started yet.
   *
   */

  class IteratorHasNotStarted : public Exception {

  public:

    /// Constructor
    explicit IteratorHasNotStarted( const std::string& methodName )
      : Exception( "The iterator has not been started yet", methodName ) {}

    /// Destructor
    virtual ~IteratorHasNotStarted() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class IteratorIsActive
   *
   *  Exception thrown when an invalid operation is attempted on an iterator
   *  that has already retrieved some rows.
   *
   */

  class IteratorIsActive : public Exception {

  public:

    /// Constructor
    explicit IteratorIsActive( const std::string& methodName )
      : Exception
    ( "The iterator has already retrieved some rows and cannot be reused",
      methodName ) {}

    /// Destructor
    virtual ~IteratorIsActive() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class IteratorIsClosed
   *
   *  Exception thrown when an invalid operation is attempted on an iterator
   *  that has already ben closed.
   *
   */

  class IteratorIsClosed : public Exception {

  public:

    /// Constructor
    explicit IteratorIsClosed( const std::string& methodName )
      : Exception( "The iterator has already been closed and cannot be reused",
                   methodName ) {}

    /// Destructor
    virtual ~IteratorIsClosed() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class IteratorHasNoCurrentItem
   *
   *  Exception thrown when attempting to retrieve the current item from an
   *  iterator with no current item.
   *
   */

  class IteratorHasNoCurrentItem : public Exception {

  public:

    /// Constructor
    explicit IteratorHasNoCurrentItem( const std::string& methodName )
      : Exception( "The iterator has no current item", methodName ) {}

    /// Destructor
    virtual ~IteratorHasNoCurrentItem() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class IteratorHasNoNextItem
   *
   *  Exception thrown when attempting to retrieve the next item from an
   *  iterator with no next item.
   *
   */

  class IteratorHasNoNextItem : public Exception {

  public:

    /// Constructor
    explicit IteratorHasNoNextItem( const std::string& methodName )
      : Exception( "The iterator has no next item", methodName ) {}

    /// Destructor
    virtual ~IteratorHasNoNextItem() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class TooManyIterators
   *
   *  Exception thrown when attempting to use two 'live' iterators
   *  at the same time (this would keep two open server cursors).
   *
   */

  class TooManyIterators : public Exception {

  public:

    /// Constructor
    explicit TooManyIterators( const std::string& methodName )
      : Exception( "An iterator is already open in this IDatabase",
                   methodName ) {}

    /// Destructor
    virtual ~TooManyIterators() throw() {}

  };

  //--------------------------------------------------------------------------

}

#endif // RELATIONALCOOL_ITERATOREXCEPTION_H
