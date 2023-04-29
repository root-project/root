#ifndef CORALKERNEL_PROPERTY_H
#define CORALKERNEL_PROPERTY_H 1

// First of all, enable or disable the CORAL240 API extensions (bug #89707)
#include "CoralBase/VersionInfo.h"

// Simplify PropertyManager (task #30840)
#include <map>
#include <boost/function.hpp>
#ifndef CORAL240PM
#include "CoralKernel/MonitorObject.h"
#else
#include "CoralBase/boost_thread_headers.h"
#endif

#include "CoralKernel/IProperty.h"

namespace coral
{
  /**
   * Class Property
   * Implementation of the IProperty interface.
   * Property objects own a mutex to provide thread safety.
   * The user of the property may register callbacks that are
   * executed whenever the set method is called.
   * The execution order of the callbacks are undefined.
   * The callbacks may be unregistered as well, and the corresponding
   * callbacks must be unregistered when the receiving object is deleted.
   *
   * The callbacks are identified by unique callback ID's.
   * The registering object must remember the callback ID
   * as that ID is used for unregistering the callback.
   *
   * The callbacks must be boost functors with a string input parameter that
   * will hold the new value of the property. There is no check for functor
   * equality, if a callback object is registered twice, it will be called
   * twice, there will be two corresponding callback ID's, and both ID's
   * must be unregistered.
   */
  class Property
    : public IProperty
#ifndef CORAL240PM
    , public MonitorObject
#endif
  {
  public:

    Property();
    ~Property() override;
    bool set(const std::string&) override;
    const std::string& get() const override;

    /**
     * Callback functor type.
     */
    typedef boost::function1<void, std::string> Callback;

    /**
     * Unique ID of the callback.
     */
    typedef long int CallbackID;

    /**
     * Register a callback, and return the corresponding callback ID.
     */
    CallbackID registerCallback(Callback&);

    /**
     * Unregister the callback corresponding to the callback ID. Return true
     * if the callback has been found and unregistered successfully else false.
     */
    bool unregisterCallback(CallbackID&);

  private:

    std::string m_value;
#ifndef CORAL240PM
    typedef std::map<CallbackID, Callback> M_Callbacks;
    M_Callbacks m_callbacks;
    CallbackID m_actualID;
#else
    std::map<CallbackID, Callback> m_callbacks;
    CallbackID m_lastID;
    mutable boost::mutex m_mutex;
#endif

  };

}

#endif // CORALKERNEL_PROPERTY_H
