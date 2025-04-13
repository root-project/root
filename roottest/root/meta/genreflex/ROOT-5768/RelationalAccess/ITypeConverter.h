#ifndef RELATIONALACCESS_ITYPECONVERTER_H
#define RELATIONALACCESS_ITYPECONVERTER_H

#include <string>
#include <set>

namespace coral {

  /**
   * Class ITypeConverter
   * Abstract interface for the registry and the conversion of C++ to SQL types and vice-versa.
   */
  class ITypeConverter {
  public:
    /**
     * Returns the SQL types supported by the particular database server.
     */
    virtual std::set<std::string> supportedSqlTypes() const = 0;

    /**
     * Returns the C++ types supported by the particular implementation.
     */
    virtual std::set<std::string> supportedCppTypes() const = 0;

    /**
     * Returns the default C++ type name for the given SQL type.
     * If an invalid SQL type name is specified, an UnSupportedSqlTypeException is thrown.
     *
     */
    virtual std::string defaultCppTypeForSqlType( const std::string& sqlType ) const = 0;

    /**
     * Returns the currently registered C++ type name for the given SQL type.
     * If an invalid SQL type name is specified, an UnSupportedSqlTypeException is thrown.
     *
     */
    virtual std::string cppTypeForSqlType( const std::string& sqlType) const = 0;

    /**
     * Registers a C++ type name for the given SQL type overriding the existing mapping.
     * If any of the types specified is not supported the relevant TypeConverterException is thrown.
     */
    virtual void setCppTypeForSqlType( const std::string& cppType,
                                       const std::string& sqlType ) = 0;

    /**
     * Returns the default SQL type name for the given C++ type.
     * If an invalid C++ type name is specified, an UnSupportedCppTypeException is thrown.
     *
     */
    virtual std::string defaultSqlTypeForCppType( const std::string& cppType ) const = 0;

    /**
     * Returns the currently registered SQL type name for the given C++ type.
     * If an invalid C++ type name is specified, an UnSupportedCppTypeException is thrown.
     */
    virtual std::string sqlTypeForCppType( const std::string& cppType ) const = 0;

    /**
     * Registers an SQL type name for the given C++ type overriding the existing mapping.
     * If any of the types specified is not supported the relevant TypeConverterException is thrown.
     */
    virtual void setSqlTypeForCppType( const std::string& sqlType,
                                       const std::string& cppType ) = 0;

  protected:
    /// Protected empty destructor
    virtual ~ITypeConverter() {}
  };

}

#endif
