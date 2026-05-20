// $Id: RelationalDatabaseId.h,v 1.20 2009-12-16 17:17:37 avalassi Exp $
#ifndef RELATIONALCOOL_RELATIONALALDATABASEID_H
#define RELATIONALCOOL_RELATIONALALDATABASEID_H 1

// Include files
#include <string>
#include <map>

namespace cool {

  /** @class RelationalDatabaseId RelationalDatabaseId.h
   *
   *  Lookup information for a relational COOL conditions database
   *  implemented using the RelationalAccess layer (CORAL).
   *
   *  A RelationalDatabaseID can be constructed from a string URL.
   *  The URL is parsed assuming the following syntax
   *  - "alias[(role)]/dbname"
   *  - "technology://server;schema=xx;dbname=xx[;user=xx][;password=xx]"
   *    (<i>deprecated</i>)
   *
   *  The first format uses CORAL ConnectionService. "alias" is the
   *  CORAL logical database name and the optional parameter "role" is
   *  the CORAL role.
   *
   *  Options "schema", "dbname", "user" and "password" can be entered
   *  in any given order. Any other options are presently ignored.
   *
   *  Parameter "technology": required.
   *  Specifies the relational backend technology.
   *  Supported values: "oracle", "mysql", "sqlite" and "frontier".
   *
   *  Parameter "server": required.
   *  Specifies the Internet address of the relational database server.
   *  Supported values:
   *  - oracle: TNS ("devdb10") or EasyConnect ("oradev10[.cern.ch]:10520/D10")
   *  - mysql: host[:port] ("pcitdb59[.cern.ch][:3306]")
   *  - sqlite: none ("none"), sqlite uses files on the local client
   *  - frontier: oracle EasyConnect ("frontier3d[.cern.ch]:8080/Frontier")
   *
   *  Parameter "schema": required.
   *  Specifies the namespace to be used as a prefix for accessing COOL
   *  tables, according to the SQL syntax "select * from namespace.table".
   *  Supported values:
   *  - oracle: user name (name of table owner)
   *  - mysql: 'database' name
   *  - sqlite: 'database' name (see http://www.sqlite.org/lang_attach.html)
   *  - frontier: oracle user name (name of table owner)
   *
   *  Parameter "dbname": required.
   *  Restrictions: "dbname" must be uppercase and 1 to 8 characters long;
   *  it may contain letters, numbers or '_', but it must start with a letter.
   *  Specifies a unique identifier of a COOL "database" within a given schema
   *  (for instance, it may indicate the name of a top-level management table,
   *  or a primary key entry in a top-level management table of predefined
   *  name, or a string prefixed to predefined table names).
   *  Presently, a COOL database "dbname" can be bootstrapped from the single
   *  table "dbname"_DB_ATTRIBUTES, which contains the names of all other
   *  tables or table prefixes for that database; actually, all tables are
   *  created by default with a "dbname_" prefix, but this is not assumed
   *  at lookup time.
   *
   *  Parameter "password": optional.
   *  Specifies the password to connect to the given server as the given user
   *  (an exception is thrown if "password" is specified but "user" is not).
   *  If "password" is absent, a database connection using an authentication
   *  service is attempted (presently, the password is looked up in an XML
   *  file, eventually grid certificates or Kerberos tokens may be used).
   *
   *  Parameter "user": optional.
   *  Specifies the user name to connect to the given server (the password may
   *  be either passed in the URL or looked up via the authentication service).
   *  If "user" is absent, the default user and password for the given server
   *  and schema [and dbname] are looked up via the authentication service.
   *  For mysql and sqlite, connections with no user/password are possible.
   *
   *  Examples:
   *    "oracle://devdb10;schema=aSch;dbname=aDb;user=aUser;password=aPswd"
   *    "oracle://oradev10:10520/D10;schema=aSc;dbname=aDb;user=aUs"
   *    "mysql://pcitdb59;schema=aSch;dbname=aDb;user=aUser;password=aPswd"
   *    "sqlite://none;schema=myFile.db;dbname=aDb"
   *    "frontier://frontier3d.cern.ch:8080/Frontier;schema=aSch;dbname=aDb"
   *
   *  *** WARNING!!! THIS MAY CHANGE IN LATER VERSIONS OF THE CODE!!! ***
   *  New (not for production!): an optional prefix "coral[...]://host:port&"
   *  indicates that COOL should connect to a CORAL server. Any prefix that
   *  begins with "coral" will be interpreted this way (eg "coral", "corals").
   *  Both explicit URLs and aliases are supported, e.g.
   *  "coral://host:port&oracle://server;schema=x;dbname=y;user=w;password=z"
   *  "coral://host:port&alias/dbname"
   *  *** WARNING!!! THIS MAY CHANGE IN LATER VERSIONS OF THE CODE!!! ***
   *
   *  @author Sven A. Schmidt, Andrea Valassi and Marco Clemencic
   *  @date   2004-08-23
   */

  class RelationalDatabaseId {

  public:

    /// Construct a RelationalDatabaseId from a string URL
    RelationalDatabaseId( const std::string& url );

    /// Construct a RelationalDatabaseId from explicit parameters
    RelationalDatabaseId( const std::string& technology,
                          const std::string& server,
                          const std::string& schema,
                          const std::string& dbName,
                          const std::string& user = "",
                          const std::string& password = "" );

    /// Construct a RelationalDatabaseId from explicit parameters
    RelationalDatabaseId( const std::string& alias,
                          const std::string& dbName,
                          const std::string& dbRole = "" );

    /// Destructor
    virtual ~RelationalDatabaseId(){}

    const std::string& middleTier() const { return m_middleTier; }

    const std::string& technology() const { return m_technology; }
    const std::string& server() const { return m_server; }
    const std::string& schema() const { return m_schema; }
    const std::string& dbName() const { return m_dbName; }
    const std::string& user() const { return m_user; }
    const std::string& password() const { return m_password; }

    const std::string& alias() const { return m_alias; }
    const std::string& role() const { return m_role; }

    const std::string& url() const { return m_url; }
    const std::string& urlHidePswd() const { return m_urlHidePswd; }
    const std::string& urlNoDbname() const { return m_urlNoDbname; }

    // Strip the leading 'coral://server:port&' middle-tier prefix
    // (modify the input url and return the stripped prefix).
    static const std::string stripMiddleTier( std::string& url );

  private:

    /// Standard constructor is private
    RelationalDatabaseId();

    /// Validate the parameters specified in the constructor
    void i_validate();

    /// Parse a URL and set the connection parameters accordingly
    void i_parseUrl( const std::string& url );

    /// Extract from a given string containing a sub-string
    /// like "option=*****;" the part between '=' and ';'
    /// (or the end of the string if ';' is not present).
    static const std::string
    extractOption( const std::string& url, const std::string& option );

  private:

    std::string m_middleTier; // Example: "coral[...]://host:port&"

    std::string m_technology;
    std::string m_server;
    std::string m_schema;
    std::string m_dbName;
    std::string m_user;
    std::string m_password;

    std::string m_alias;
    std::string m_role;

    std::string m_url;
    std::string m_urlHidePswd;
    std::string m_urlNoDbname;

  };

}

#endif // RELATIONALCOOL_RELATIONALALDATABASEID_H
