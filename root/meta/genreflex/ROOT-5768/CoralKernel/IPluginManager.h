// -*-C++-*-
#ifndef CORALBASE_IPLUGINMANAGER_H
#define CORALBASE_IPLUGINMANAGER_H

#include <string>
#include <set>

namespace coral
{
  class ILoadableComponent;

  /// Interface for a plugin manager
  class IPluginManager {
  public:
    /// Creates a new ILoadableComponent object given its name
    virtual ILoadableComponent* newComponent( const std::string& componentName ) = 0;
    /// Returns the list of known components
    virtual std::set<std::string> knownPlugins() const = 0;
  protected:
    virtual ~IPluginManager() {}
  };
}

#endif
