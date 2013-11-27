#ifndef RELATIONALACCESS_ITABLEPRIVILEGEMANAGER_H
#define RELATIONALACCESS_ITABLEPRIVILEGEMANAGER_H

#include <string>

namespace coral {

  /**
   * Class ITablePrivilegeManager
   * Interface for managing the privileges in a table.
   */
  class ITablePrivilegeManager {
  public:
    typedef enum { Select, Update, Insert, Delete } Privilege;
  public:
    /**
     * Grants an access right to a specific user.
     */
    virtual void grantToUser( const std::string& userName,
                              Privilege right ) = 0;

    /**
     * Revokes a right from the specified user.
     */
    virtual void revokeFromUser( const std::string& userName,
                                 Privilege right ) = 0;

    /**
     * Grants the specified right to all users.
     */
    virtual void grantToPublic( Privilege right ) = 0;

    /**
     * Revokes the specified right from all users.
     */
    virtual void revokeFromPublic( Privilege right ) = 0;

  protected:
    /// Protected empty destructor
    virtual ~ITablePrivilegeManager() {}
  };

}

#endif
