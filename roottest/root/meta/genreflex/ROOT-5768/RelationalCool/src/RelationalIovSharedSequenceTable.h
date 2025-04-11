// $Id: RelationalIovSharedSequenceTable.h,v 1.2 2009-12-16 17:17:37 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALIOVSHAREDSEQUENCETABLE_H
#define RELATIONALCOOL_RELATIONALIOVSHAREDSEQUENCETABLE_H

// Local include files
#include "uppercaseString.h"

namespace cool {

  /** @namespace cool::RelationalIovSharedSequenceTable RelationalIovSharedSequenceTable.h
   *
   *  Relational schema of the shared 'sequence table' for IOV PKs.
   *
   *  @author Andrea Valassi
   *  @date   2007-01-31
   */

  namespace RelationalIovSharedSequenceTable
  {

    inline const std::string defaultTableName( const std::string& prefix )
    {
      // Note: presently the input prefix is uppercase anyway...
      return uppercaseString(prefix) + "IOVS_SEQ";
    }

  }

}
#endif // RELATIONALCOOL_RELATIONALIOVSHAREDSEQUENCETABLE_H
