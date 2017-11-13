#ifndef ROOT_Fit_FitExecutionPolicy
#define ROOT_Fit_FitExecutionPolicy
namespace ROOT {
   namespace Internal {
      enum class ExecutionPolicy { kSequential, kMultithread, kMultiprocess };
    }
}

#endif
