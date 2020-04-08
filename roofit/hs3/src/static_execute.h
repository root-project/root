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
