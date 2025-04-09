#ifndef CORALKERNEL_MONITOROBJECT_H
#define CORALKERNEL_MONITOROBJECT_H 1

// First of all, enable or disable the CORAL240 API extensions (bug #89707)
#include "CoralBase/VersionInfo.h"

// Simplify PropertyManager (task #30840)
#ifndef CORAL240PM

#include "CoralBase/boost_thread_headers.h"

namespace coral
{

  /*
   * Author: Zsolt Molnar
   */
  struct ThreadTraits
  {
    typedef boost::mutex MUTEX;
    typedef boost::condition CONDITION;
    typedef boost::mutex::scoped_lock LOCK;
    typedef boost::thread_group THREAD_GROUP;
    typedef boost::thread THREAD;
    static boost::thread::id get_id() { return boost::this_thread::get_id(); }
  };

  /** Class MonitorObject
   * The class implements the Monitor pattern that enables concurrent access
   * for the method of an object. The classes that need concurrent access
   * must derive from MonitorObject, and protect the method bodies with the
   * constructs provided by this class. Only those methods must be protected that
   * need synchronization. The best practice if the protection covers the entire
   * method body like the example:
   *
   * int MyClass::mySynchronizedMethod ()
   * {
   * MONITOR_START_CRITICAL
   *     // Do something
   *     return 0;
   * MONITOR_END_CRITICAL
   * }
   *
   * Author: Zsolt Molnar
   */

  class MonitorObject
  {
  public:
    MonitorObject() {};
    virtual ~MonitorObject() {};

    typedef ThreadTraits synch_traits;

  protected:
    mutable synch_traits::MUTEX _monitor_lock;

    static synch_traits::MUTEX& _static_monitor_lock() {
      static synch_traits::MUTEX m;
      return m;
    }
  };

  /// The name of the lock variable taht is set by MONITOR_START_CRITICAL
#define MONITOR_LOCK __lock__
  /// Opens a critical (protected) section. It must be closed by a corresponding MONITOR_END_CRITICAL.
  /// If you fail to indicate the end of the critical section, you will get compilation error.
  /// In runtime, the lock is always released at the end of the scope of the critical section.
#define MONITOR_START_CRITICAL { ThreadTraits::LOCK MONITOR_LOCK(this->_monitor_lock);
  /// Opens a "static" critical section (locks all the objects of a given class).
#define MONITOR_START_STATIC_CRITICAL { ThreadTraits::LOCK MONITOR_LOCK(_static_monitor_lock());
  /// Closes a critical (protected) section.
#define MONITOR_END_CRITICAL }

}

#endif

#endif // CORALKERNEL_MONITOROBJECT_H
