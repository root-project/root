#ifndef IPROPERTY_H_
#define IPROPERTY_H_

#include <string>

namespace coral {
  /**
   * Class IProperty
   * A property descriptor interface. A property has as string value that
   * may be modified in runtime.
   */
  struct IProperty
  {
    virtual ~IProperty() {}
    /**
     * Set the value of the property. Return false if the value is unacceptable.
     */
    virtual bool set(const std::string&) = 0;
    /**
     * The actual value of the propery.
     */
    virtual const std::string& get() const = 0;
  };

} // namespace coral

#endif /*IPROPERTY_H_*/
