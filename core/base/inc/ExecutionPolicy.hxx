#ifndef ROOT_Internal_ExecutionPolicy
#define ROOT_Internal_ExecutionPolicy
namespace ROOT {
   namespace Internal {
      enum class ExecutionPolicy { kSerial, kMultithread, kMultiprocess };
   }
}

#endif
