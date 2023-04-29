#ifndef RELATIONALACCESS_IQUERY_H
#define RELATIONALACCESS_IQUERY_H

#include "IQueryDefinition.h"

namespace coral {

  class AttributeListSpecification;
  class AttributeList;
  class ICursor;

  /**
   * Class IQuery
   *
   * Interface for an executable query.
   * Once the execute() method is called no other method can be called.
   * Otherwise a QueryExecutedException will be thrown.
   *
   */
  class IQuery : virtual public IQueryDefinition {
  public:
    /// Destructor
    ~IQuery() override {}

    /**
     * Instructs the server to lock the rows involved in the result set.
     */
    virtual void setForUpdate() = 0;

    /**
     * Defines the client cache size in rows
     */
    virtual void setRowCacheSize( int numberOfCachedRows ) = 0;

    /**
     * Defines the client cache size in MB.
     */
    virtual void setMemoryCacheSize( int sizeInMB ) = 0;

    /**
     * Defines the output types of a given variable in the result set.
     */
    virtual void defineOutputType( const std::string& outputIdentifier,
                                   const std::string& cppTypeName ) = 0;

    /**
     * Defines the output data buffer for the result set.
     */
    virtual void defineOutput( AttributeList& outputDataBuffer ) = 0;

    /**
     * Executes the query and returns a reference to the undelying ICursor object
     * in order for the user to loop over the result set.
     */
    virtual ICursor& execute() = 0;
  };

}

#endif
