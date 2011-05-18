#ifndef PYROOT_TRFLXCALLBACK_H
#define PYROOT_TRFLXCALLBACK_H

#ifdef PYROOT_USE_REFLEX

// ROOT
#include "Reflex/Callback.h"

// Standard
#include <memory>


namespace PyROOT {

class TRflxCallback : public ROOT::Reflex::ICallback {
public:
   TRflxCallback();
   ~TRflxCallback();

public:
   virtual void operator() ( const ROOT::Reflex::Type& t );
   virtual void operator() ( const ROOT::Reflex::Member& m );

public:
   static PyObject* Enable();
   static PyObject* Disable();

private:
   static std::auto_ptr< ROOT::Reflex::ICallback > gCallback;
};

} // namespace PyROOT

#endif // PYROOT_USE_REFLEX

#endif // !PYROOT_TRFLXCALLBACK_H
