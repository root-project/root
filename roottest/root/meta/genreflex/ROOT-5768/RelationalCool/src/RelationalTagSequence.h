// $Id: RelationalTagSequence.h,v 1.3 2009-12-16 17:17:38 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALTAGSEQUENCE_H
#define RELATIONALCOOL_RELATIONALTAGSEQUENCE_H

// Local include files
#include "RelationalTagTable.h"

namespace cool {

  /** @namespace cool::RelationalTagSequence RelationalTagSequence.h
   *
   *  Properties of the COOL "sequence" for local tag ID's in each node.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2006-03-28
   */

  namespace RelationalTagSequence {

    inline const std::string sequenceName( const std::string& prefix,
                                           unsigned nodeId )
    {
      return RelationalTagTable::defaultTableName( prefix, nodeId ) + "_SEQ";
    }

  }

}

#endif // RELATIONALCOOL_RELATIONALTAGSEQUENCE_H
