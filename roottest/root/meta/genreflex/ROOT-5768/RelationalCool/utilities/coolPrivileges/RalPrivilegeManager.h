// $Id: RalPrivilegeManager.h,v 1.5 2011-06-27 11:54:06 avalassi Exp $
#ifndef RELATIONALCOOL_RALPRIVILEGEMANAGER_H
#define RELATIONALCOOL_RALPRIVILEGEMANAGER_H 1

// Include files
#include <memory>
#include "CoralBase/MessageStream.h"
#include "RelationalAccess/ITablePrivilegeManager.h"

namespace cool
{

  // Forward declarations
  class RalDatabase;

  /** @class RalPrivilegeManager RalPrivilegeManager.h
   *
   *  Private utility class to grant/revoke table-level object privileges
   *  on COOL schema objects to other users in the same server.
   *
   *  For the moment, this does not implement any public API: the
   *  functionality is offered to users only via pre-built executables.
   *  This class uses a bare RalDatabase which means we are using the
   *  manual transaction mode.
   *
   *  This class can only be used with "oracle" database technology.
   *
   *  @author Andrea Valassi and Sven A. Schmidt
   *  @date   2005-07-05
   */

  class RalPrivilegeManager
  {

    /// Adopt the same Privilege list as RAL (SELECT, UPDATE, INSERT, DELETE)
    typedef enum { SELECT, INSERT, UPDATE, DELETE } Privilege;

  public:

    /// Constructor from a RalDatabase pointer
    RalPrivilegeManager( RalDatabase* db );

    /// Destructor
    virtual ~RalPrivilegeManager();

    /// Grant privileges to read all data in the COOL database
    void grantReaderPrivileges( const std::string& user );

    /// Revoke privileges to read all data in the COOL database
    void revokeReaderPrivileges( const std::string& user );

    /// Grant privileges to insert new IOVs into the COOL database
    /// [NB: reader privileges are also needed and must be granted explicitly]
    void grantWriterPrivileges( const std::string& user );

    /// Revoke privileges to insert new IOVs into the COOL database
    void revokeWriterPrivileges( const std::string& user );

    /// Grant privileges to tag (optionally retag) IOVs in the COOL database
    /// [NB: reader privileges are also needed and must be granted explicitly]
    void grantTaggerPrivileges( const std::string& user,
                                const bool retag );

    /// Revoke privileges to tag and/or retag IOVs in the COOL database
    void revokeTaggerPrivileges( const std::string& user );

  protected:

    /// Get a CORAL MessageStream
    coral::MessageStream& log();

    /// List all tables that a READER must be allowed to have privileges on
    /// [For SELECT privileges these are ALL the tables in the COOL database]
    std::vector< std::string > i_listReaderTables( const Privilege& priv );

    /// List all tables that a WRITER must be allowed to have privileges on
    std::vector< std::string > i_listWriterTables( const Privilege& priv );

    /// List all tables that a TAGGER must be allowed to have privileges on
    std::vector< std::string > i_listTaggerTables( const Privilege& priv,
                                                   const bool retag );

    /// Get the coral::ITablePrivilegeManager for a given table
    coral::ITablePrivilegeManager&
    i_tablePrivMgr( const std::string& table );

    /// Grant a given privilege on a given table to a given user
    void i_grantPrivilege( const Privilege& priv,
                           const std::string& table,
                           const std::string& user );

    /// Revoke a given privilege on a given table to a given user
    void i_revokePrivilege( const Privilege& priv,
                            const std::string& table,
                            const std::string& user );

  private:

    /// Standard constructor is private
    RalPrivilegeManager();

    /// Copy constructor is private
    RalPrivilegeManager( const RalPrivilegeManager& rhs );

    /// Assignment operator is private
    RalPrivilegeManager& operator=( const RalPrivilegeManager& rhs );

  private:

    /// RalDatabase pointer
    RalDatabase* m_db;

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

    /// Vector of four basic privileges
    std::vector<Privilege> m_fourPrivs;

  };

}

#endif // RELATIONALCOOL_RALPRIVILEGEMANAGER_H
