// $Id: RelationalTagSharedSequenceTable.h,v 1.2 2009-12-16 17:17:38 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALTAGSHAREDSEQUENCETABLE_H
#define RELATIONALCOOL_RELATIONALTAGSHAREDSEQUENCETABLE_H

// Local include files
#include "uppercaseString.h"

namespace cool {

  /** @namespace cool::RelationalTagSharedSequenceTable RelationalTagSharedSequenceTable.h
   *
   *  Relational schema of the shared 'sequence table' for tag PKs.
   *
   *  @author Andrea Valassi
   *  @date   2007-01-31
   */

  namespace RelationalTagSharedSequenceTable
  {

    inline const std::string defaultTableName( const std::string& prefix )
    {
      // Note: presently the input prefix is uppercase anyway...
      return uppercaseString(prefix) + "TAGS_SEQ";
    }

  }

}
#endif // RELATIONALCOOL_RELATIONALTAGSHAREDSEQUENCETABLE_H
