#ifndef ROOT_TSpinMutex
#define ROOT_TSpinMutex

#include <atomic>

namespace ROOT {
   /// A spin mutex class which respects the STL interface.
   class TSpinMutex {
   public:
      TSpinMutex() = default;
      TSpinMutex(const TSpinMutex&) = delete;
      ~TSpinMutex(){}
      TSpinMutex& operator=(const TSpinMutex&) = delete;
      void lock(){while (fAFlag.test_and_set(std::memory_order_acquire));}
      void unlock(){fAFlag.clear(std::memory_order_release);}
      bool try_lock(){return !fAFlag.test_and_set(std::memory_order_acquire);}

   private:
      std::atomic_flag fAFlag = ATOMIC_FLAG_INIT;
   };
}

#endif
