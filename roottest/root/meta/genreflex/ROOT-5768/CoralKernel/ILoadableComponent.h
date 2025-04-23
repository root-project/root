#ifndef CORALKERNEL_ILOADABLE_COMPONENT_H
#define CORALKERNEL_ILOADABLE_COMPONENT_H

#include "RefCounted.h"
#include <string>

namespace coral {

  /// An interface for a loadable component
  class ILoadableComponent : public RefCounted
  {
  public:
    /// Returns the name of the component
    std::string name() const { return m_name; }

  protected:
    /// Constructor
    explicit ILoadableComponent( const std::string& name ) : RefCounted(), m_name( name ) {}

    /// Destructor
    ~ILoadableComponent() override {}

  private:
    /// No copy constructor
    ILoadableComponent( const ILoadableComponent& );
    /// No assignment operator
    ILoadableComponent& operator=( const ILoadableComponent& );

  private:
    std::string m_name;

  };

}

#endif
