// lib/dll_stl/eh.h

#include <exception>
#ifndef __hpux
using namespace std;
#endif
#ifdef __SUNPRO_CC
#define exception std::exception
#endif

#ifdef __MAKECINT__
#ifndef G__EXCEPTION_DLL
#define G__EXCEPTION_DLL
#endif

#pragma link C++ global G__EXCEPTION_DLL;
#pragma link C++ class exception;
#pragma link C++ class bad_exception;
#pragma link C++ function set_unexpected;
#pragma link C++ function unexpected;
#pragma link C++ function set_terminate;
#pragma link C++ function terminate;
#pragma link C++ function uncaught_exception;
#pragma link C++ typedef unexpected_handler;
#pragma link C++ typedef terminate_handler;

#endif

