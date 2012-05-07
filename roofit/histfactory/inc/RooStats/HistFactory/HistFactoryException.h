
#ifndef HISTFACTORY_EXCEPTION
#define HISTFACTORY_EXCEPTION


#include <iostream>
#include <exception>
using namespace std;

class hf_exc: public exception
{
  virtual const char* what() const throw()
  {
    return "HistFactory - Exception";
  }
};

static hf_exc bad_hf;

#endif
