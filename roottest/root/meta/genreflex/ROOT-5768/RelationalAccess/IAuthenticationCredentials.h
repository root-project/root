#ifndef RELATIONALACCESS_IAUTHENTICATIONCREDENTIALS_H
#define RELATIONALACCESS_IAUTHENTICATIONCREDENTIALS_H

#include <string>

namespace coral {

  /**
   * Class IAuthenticationCredentials
   * Interface holding the authentication credentials for a specific database connection
   */
  class IAuthenticationCredentials {
  public:
    /**
     * Name of the user item
     */
    static std::string userItem();

    /**
     * Name of the password item
     */
    static std::string passwordItem();

    /**
     * Name of the default role
     */
    static std::string defaultRole();

  public:
    /**
     * Returns the value for a given credential item. If the item name is not valid an InvalidCredentialItemException is thrown.
     */
    virtual std::string valueForItem( const std::string& itemName ) const = 0;

    /**
     * Returns the number of credential items
     */
    virtual int numberOfItems() const = 0;

    /**
     * Returns the name of a crential item for a given index. If the index is out of range a CredentialItemIndexOutOfRangeException is thrown.
     */
    virtual std::string itemName( int itemIndex ) const = 0;

  protected:
    /// Protected empty destructor
    virtual ~IAuthenticationCredentials() {}
  };

}

#endif
