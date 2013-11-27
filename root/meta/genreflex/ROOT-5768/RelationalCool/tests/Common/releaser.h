// $Id: releaser.h,v 1.2 2009-12-16 17:27:41 avalassi Exp $
#ifndef RELATIONALCOOL_RELEASER_H
#define RELATIONALCOOL_RELEASER_H 1

// Include files
#include "CoralBase/AttributeListSpecification.h"

namespace cool
{

  class releaser {
  public:
    void operator()( coral::AttributeListSpecification* p ) { p->release(); }
  };

}

#endif // RELEASER_H
