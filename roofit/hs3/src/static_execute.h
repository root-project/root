/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN, Dec 2021
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef static_execute_h
#define static_execute_h

// Execute code on library loading by running code in the constructor of a
// class that is a static class member of another class.
#define STATIC_EXECUTE(MY_CODE)   \
   struct StaticExecutorWrapper { \
      struct Executor {           \
         template <class Func>    \
         Executor(Func func)      \
         {                        \
            func();               \
         }                        \
      };                          \
      static Executor executor;   \
   };                             \
                                  \
   StaticExecutorWrapper::Executor StaticExecutorWrapper::executor{[]() { MY_CODE }};

#endif
