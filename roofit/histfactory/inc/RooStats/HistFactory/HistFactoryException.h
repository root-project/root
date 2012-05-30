
#ifndef HISTFACTORY_EXCEPTION
#define HISTFACTORY_EXCEPTION

#include <iostream>
#include <exception>


namespace RooStats{
  namespace HistFactory{

    class hf_exc: public std::exception
    {
      virtual const char* what() const throw()
      {
	return "HistFactory - Exception";
      }
    };

  }
}

//static hf_exc bad_hf;

#endif
