#ifndef RELATIONALACCESS_AUTHENTICATIONCREDENTIALS_H
#define RELATIONALACCESS_AUTHENTICATIONCREDENTIALS_H

#include "RelationalAccess/IAuthenticationCredentials.h"

#include <vector>
#include <map>

namespace coral {

  /**
   * Simple implementation of the IAuthenticationCredentials interface
   */
  class AuthenticationCredentials : virtual public IAuthenticationCredentials
  {
  public:
    /// Constructor
    explicit AuthenticationCredentials( const std::string& implementationName );

    /// Destructor
    ~AuthenticationCredentials() override;

    /// Registers a new item name-value pair. Returns false if an item with the same name already exists.
    bool registerItem( const std::string& name,
                       const std::string& value );

    /**
     * Returns the value for a given credential item. If the item name is not valid an InvalidCredentialItemException is thrown.
     */
    std::string valueForItem( const std::string& itemName ) const override;

    /**
     * Returns the number of credential items
     */
    int numberOfItems() const override;

    /**
     * Returns the name of a crential item for a given index. If the index is out of range a CredentialItemIndexOutOfRangeException is thrown.
     */
    std::string itemName( int itemIndex ) const override;

  private:
    /// The authentication service implementation name
    std::string m_implementationName;

    /// A map holding the item names
    std::map< std::string, std::string > m_itemValues;

    /// The item values
    std::vector< std::string > m_itemNames;
  };

}

#endif
