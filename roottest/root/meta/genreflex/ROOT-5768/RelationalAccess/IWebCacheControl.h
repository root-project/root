#ifndef CORAL_RELATIONALACCESS_IWEBCACHECONTROL_H
#define CORAL_RELATIONALACCESS_IWEBCACHECONTROL_H

#include <string>
#include <vector>

namespace coral {

  // forward declarations
  class IWebCacheInfo;

  /**
     @class IWebCacheControl IWebCacheControl.h RelationalAccess/IWebCacheControl.h
     Interface for controlling the behaviour of web caches. By default data residing
     on a web cache are not refreshed unless it is set otherwise through this interface.
  */
  class IWebCacheControl
  {
  public:
    /**
       Instructs the RDBMS backend that all the tables within the schema specified
       by the physical or logical connection should be refreshed, in case they are accessed.
    */
    virtual void refreshSchemaInfo( const std::string& connection ) = 0;

    /**
       Instructs the RDBMS backend that the specified table within the schema specified
       by the physical or logical connection should be refreshed in case it is accessed.
    */
    virtual void refreshTable( const std::string& connection,
                               const std::string& tableName ) = 0;

    /**
       Returns the web cache information for a schema given the corresponding physical or
       logical connection.
    */
    virtual const IWebCacheInfo& webCacheInfo( const std::string& connection ) const = 0;

    /**
       Returns the previous compression level
    */
    virtual int compressionLevel() = 0;

    /**
       Sets the compression level for data transfer, 0 - off, 1 - fast, 5 - default, 9 - maximum
    */
    virtual void setCompressionLevel( int level ) = 0;

    /**
       Sets the list of the web cache proxies for the fail-over mechanism
    */
    virtual void setProxyList( const std::vector<std::string>& proxyList ) = 0;

  protected:
    /// Protected empty destructor
    virtual ~IWebCacheControl() {}
  };
}

#endif
