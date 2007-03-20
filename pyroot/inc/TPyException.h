// @(#)root/pyroot:$Name:  $:$Id: TPyException.h,v 1.4 2006/03/09 09:07:02 brun Exp $
// Author: Scott Snyder, Apr 2004

#ifndef ROOT_TPyException
#define ROOT_TPyExecption

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// TPyException                                                             //
//                                                                          //
// Purpose: A C++ exception class for throwing python exceptions            //
//          through C++ code.                                               //
// Created: Apr, 2004, sss, from the version in D0's python_util.           //
//                                                                          //
// The situation is:                                                        //
//   - We're calling ROOT C++ code from python.                             //
//   - The C++ code can call back to python.                                //
//   - What to do then if the python callback throws an exception?          //
//                                                                          //
// We need to get the control flow back to where PyROOT makes the ROOT call.//
// To do that we throw a TPyException.                                      //
// We can then catch this exception when we do the ROOT call.               //
//                                                                          //
// Note that we don't need to save any state in the exception -- it's       //
// already in the python error info variables.                              //
// (??? Actually, if the program is multithreaded, this is dangerous        //
// if the code has released and reacquired the lock along the call chain.   //
// Punt on this for now, though.)                                           //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

// ROOT
#ifndef ROOT_DllImport
#include "DllImport.h"
#endif
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

// Standard
#include <exception>


namespace PyROOT {

R__EXTERN void* TPyExceptionMagic;

class TPyException : public std::exception {
public:
// default constructor
   TPyException();

// destructor
   virtual ~TPyException() throw();

// give reason for raised exception
   virtual const char* what() const throw();

   ClassDef(TPyException,0)   //C++ exception for throwing python exceptions
};

} // namespace PyROOT

#if defined(G__DICTIONARY) && defined(R__SOLARIS)
#define exception std::exception
#endif
#endif
