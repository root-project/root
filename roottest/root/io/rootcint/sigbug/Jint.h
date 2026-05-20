#ifndef JINT_H
#define JINT_H

#include "sigc++/signal.h"
#include "sigc++/sigc++.h"

class Jint : public SigC::Object
{

public:
    Jint();
    virtual ~Jint();

    SigC::Signal0<void> mom_modified, job_modified;

};                              // end of class Jint

#endif  // JINT_H
