#ifndef RELATIONALCOOL_IRELATIONALCURSOR_H
#define RELATIONALCOOL_IRELATIONALCURSOR_H

// Include files
#include "CoralBase/AttributeList.h"

namespace cool
{

  /**
   * Class IRelationalCursor.
   *
   * Abstract interface for a query cursor.
   */

  class IRelationalCursor
  {

  public:

    /// Destructor
    virtual ~IRelationalCursor() {}

    /// Positions the cursor to the next available row in the result set.
    /// If there are no more rows in the result set false is returned.
    virtual bool next() = 0;

    /// Returns a reference to the output buffer holding the last row fetched.
    virtual const coral::AttributeList& currentRow() const = 0;

    /// Explicitly closes the cursor, releasing the resources on the server.
    virtual void close() = 0;

  };

}
#endif
