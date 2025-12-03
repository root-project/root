// $Id: RelationalDatabasePtr.h,v 1.3 2009-12-17 18:38:53 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALDATABASEPTR_H
#define RELATIONALCOOL_RELATIONALDATABASEPTR_H

// Include files
#ifdef COOL_HAS_CPP11
#include <memory>
#else
#include <boost/shared_ptr.hpp>
#endif

namespace cool
{

  // Forward declarations
  class RelationalDatabase;

  /// Shared pointer to a RelationalObject
#ifdef COOL_HAS_CPP11
  typedef std::shared_ptr<RelationalDatabase> RelationalDatabasePtr;
#else
  typedef boost::shared_ptr<RelationalDatabase> RelationalDatabasePtr;
#endif

}

#endif
