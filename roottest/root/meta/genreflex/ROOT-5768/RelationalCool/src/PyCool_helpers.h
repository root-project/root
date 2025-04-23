// $Id: PyCool_helpers.h,v 1.9 2009-12-17 18:38:53 avalassi Exp $
#ifndef RELATIONALCOOL_PYCOOLHELPERS_H
#define RELATIONALCOOL_PYCOOLHELPERS_H 1

// Include files
#include "CoolKernel/IDatabase.h"
#include "CoolKernel/IDatabaseSvc.h"

namespace cool
{
  namespace PyCool
  {
    namespace Helpers
    {

      // Helper to call RelationalDatabase::refreshDatabase()
      void refreshDatabaseFromDbSvc( cool::IDatabaseSvc* dbSvc,
                                     const DatabaseId& dbId,
                                     bool keepNodes = false );

      // Helper to call RelationalDatabase::refreshDatabase()
      void refreshDatabaseFromDb( cool::IDatabase* db,
                                  bool keepNodes = false );

    }
  }
}

#endif // RELATIONALCOOL_PYCOOLHELPERS_H
