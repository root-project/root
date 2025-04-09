#ifndef CORALKERNEL_CORALPLUGINDEF_H
#define CORALKERNEL_CORALPLUGINDEF_H 1

// First of all, enable or disable the CORAL240 API extensions (bug #89707)
#include "CoralBase/VersionInfo.h"

// Include files
#include "CoralKernel/ILoadableComponent.h"
#include "CoralKernel/ILoadableComponentFactory.h"

namespace coral {
  template<typename T> class CoralPluginFactory : virtual public coral::ILoadableComponentFactory
  {
  public:
    CoralPluginFactory( const std::string& name ) : m_name( name ) {}
    ~CoralPluginFactory() override {}
    coral::ILoadableComponent* component() const override
    { return static_cast< coral::ILoadableComponent* >( new T( m_name ) ); }
    std::string name() const override { return m_name; }
  private:
    std::string m_name;
  };
}


#if defined(_WIN32) || ( !defined(__clang__) && !defined(CORAL240PL) )
// OLD IMPLEMENTATION: inside shared library, locate static factory ptr
// Used always for Windows; never for clang; else if !defined(CORAL24PL)
// [NB Replace factory C++ ref by a factory ptr in C external linkage; see http://www.velocityreviews.com/forums/t739407-c-specific-types-in-extern-c.html]
#define CORAL_PLUGIN_MODULE(NAME,PLUGINCLASS) \
  static coral::CoralPluginFactory< PLUGINCLASS > theFactory( std::string( NAME ) ); \
  extern "C" { coral::ILoadableComponentFactory* coral_component_factory = &theFactory; }
#define CORAL_PLUGIN2_MODULE(NAME,PLUGINCLASS) \
  static coral::CoralPluginFactory< PLUGINCLASS > theFactory2( std::string( NAME ) ); \
  extern "C" { coral::ILoadableComponentFactory* coral_component_factory2 = &theFactory2; }
#else
// NEW IMPLEMENTATION: inside shared library, locate ptr to method
// that instantiates a static factory (fix clang bug #92167)
// [see also http://tldp.org/HOWTO/C++-dlopen/thesolution.html]
// Used never for Windows; always for clang; else if defined(CORAL24PL)
#define CORAL_PLUGIN_MODULE(NAME,PLUGINCLASS) \
extern "C" { coral::ILoadableComponentFactory* coral_component_factory() { \
  static coral::CoralPluginFactory< PLUGINCLASS > theFactory( std::string( NAME ) ); return &theFactory; } }
#define CORAL_PLUGIN2_MODULE(NAME,PLUGINCLASS) \
extern "C" { coral::ILoadableComponentFactory* coral_component_factory2() { \
  static coral::CoralPluginFactory< PLUGINCLASS > theFactory2( std::string( NAME ) ); return &theFactory2; } }
#endif

#endif
