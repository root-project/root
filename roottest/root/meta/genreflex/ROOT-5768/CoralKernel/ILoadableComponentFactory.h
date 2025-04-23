#ifndef ILOADABLE_COMPONENT_FACTORY_H
#define ILOADABLE_COMPONENT_FACTORY_H

#include <string>

namespace coral {

  class ILoadableComponent;

  class ILoadableComponentFactory {
  public:
    virtual ILoadableComponent* component() const = 0;
    virtual std::string name() const = 0;
  protected:
    virtual ~ILoadableComponentFactory() {}
  };

}


#endif
