#ifndef IPROPERTYMANAGER_H_
#define IPROPERTYMANAGER_H_ 1

// First of all, enable or disable the CORAL240 API extensions (bug #89707)
#include "CoralBase/VersionInfo.h"

// Include files
#include <string>
#include "CoralKernel/IProperty.h"

// Temporary fix for bug #63198 on WIN32 (struct was first seen as class).
// The API is internally inconsistent: Context.h forward declares a class:
// eventually struct IPropertyManager will become a class.
#ifdef WIN32
#ifndef CORAL240PR
#pragma warning ( disable : 4099 )
#endif
#endif

namespace coral
{

  /**
   * @class IPropertyManager
   *
   * Interface for an object managing a set of properties. The object must
   * store a fixed property set, properties cannot be added/deleted by this
   * interface. The properties are identified by strings (names).
   *
   * @author Zsolt Molnar
   * @date   2008-05-21
   */
#if defined(CORAL240PR) || defined(__clang__)
  // Fix WIN32 bug #63198 and clang bug #79151 (struct was first seen as class).
  // The API was internally inconsistent: Context.h forward declares a class.
  class IPropertyManager
#else
  struct IPropertyManager
#endif
  {

  public:

    virtual ~IPropertyManager() {}

    /**
     * Return the property identified by the given string or NULL if the
     * property does not exist.
     */
    virtual IProperty* property(const std::string&) = 0;

  };

}
#endif /*IPROPERTYMANAGER_H_*/
