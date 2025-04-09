#ifndef CORALKERNEL_CONTEXT_H
#define CORALKERNEL_CONTEXT_H 1

// First of all, enable or disable the CORAL240 API extensions (bug #89707)
#include "CoralBase/VersionInfo.h"

#include "CoralBase/boost_thread_headers.h"
#include "CoralKernel/ILoadableComponent.h"
#include "CoralKernel/IHandle.h"

#include <boost/scoped_ptr.hpp>

#include <string>
#include <map>
#include <set>

namespace coral
{

  class IPluginManager;
  class IPropertyManager;
  class PluginManager;

  /// The context singleton class.
  class Context
  {
  public:
    /// Returns the instance of the singleton
    static Context& instance( coral::IPluginManager* pluginManager = 0 );

    /**
       Loads a component given its name and a pointer to the plugin manager.
       Throws an exception in case of a problem
    */
    void loadComponent( const std::string& componentName,
                        coral::IPluginManager* pluginManager = 0 );

    /// Returns a handle for a specified interface
    template< typename T > IHandle<T> query()
    {
      boost::mutex::scoped_lock lock( m_mutex );

      for ( std::map<std::string, ILoadableComponent*>::iterator
              iComponent = m_components.begin(); iComponent != m_components.end(); ++iComponent )
      {
        T* component = dynamic_cast<T*>( iComponent->second );
        if ( component != 0 )
        {
          iComponent->second->addReference();
          return IHandle<T>( component );
        }
      }

      // Return an empty handle
      return IHandle<T>();
    }

    /// Returns a handle for a specified interface given a name
    template< typename T > IHandle<T> query( const std::string& name )
    {
      boost::mutex::scoped_lock lock( m_mutex );

      std::map<std::string, ILoadableComponent*>::iterator iComponent = m_components.find( name );

      if ( iComponent == m_components.end() )
        return IHandle<T>();

      T* component = dynamic_cast<T*>( iComponent->second );

      if ( component != 0 )
      {
        iComponent->second->addReference();
        return IHandle<T>( component );
      }
      else
      {
        return IHandle<T>();
      }
    }

    /// Returns the CORAL property manager component
    /// This was requested by CMS to avoid environment variables (task #6857)
#ifdef CORAL240PM
    /// Change the name from upper to lowercase for consistency (task #30840)
    IPropertyManager& propertyManager();
#else
    IPropertyManager& PropertyManager();
#endif

    /// Returns the list of the components known bt the internal CORAL
    /// plugin manager
    std::set<std::string> knownComponents() const;

    /// Returns the list of plugins loaded
    std::set<std::string> loadedComponents() const;

  private:
    /// Constructor
    Context( coral::IPluginManager* pluginManager = 0 );

    /// Copy constructor or assignment operator. Forbidden!
    Context( const Context& );
    Context& operator=(const Context& );

    /// Destructor
    ~Context();

    /// Checks the existence of a component
    bool existsComponent( const std::string& componentName );

  private:

    /// The mutex lock for the component map
    boost::mutex m_mutex;

    /// The loaded components
    typedef std::map<std::string, ILoadableComponent*> m_components_type;
    m_components_type m_components;

    /// The internal plugin manager
    mutable coral::PluginManager*  m_nativePluginManager;

    /// The plugin manager proxy
    mutable coral::IPluginManager* m_pluginManager;

    /// The property manager
    boost::scoped_ptr<coral::IPropertyManager> m_propertyManager;

  };

}
#endif
