// $Id: RelationalObjectPtr.h,v 1.3 2009-12-17 18:38:53 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALOBJECTPTR_H
#define RELATIONALCOOL_RELATIONALOBJECTPTR_H 1

// Include files
#ifdef COOL_HAS_CPP11
#include <memory>
#else
#include <boost/shared_ptr.hpp>
#endif

namespace cool
{

  // Forward declarations
  class RelationalObject;

  /// Shared pointer to a RelationalObject
#ifdef COOL_HAS_CPP11
  typedef std::shared_ptr<RelationalObject> RelationalObjectPtr;
#else
  typedef boost::shared_ptr<RelationalObject> RelationalObjectPtr;
#endif

}

#endif
