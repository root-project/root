// $Id: RalSchemaValidation.h,v 1.10 2009-12-16 17:27:49 avalassi Exp $
#ifndef RELATIONALCOOL_RALSCHEMAVALIDATION_H
#define RELATIONALCOOL_RALSCHEMAVALIDATION_H 1

// Include files
#include <memory>
#include "CoolKernel/DatabaseId.h"
#include "CoolKernel/RecordSpecification.h"
#include "CoralBase/MessageStream.h"

namespace cool
{

  // Forward declarations
  class RalDatabase;
  class RelationalObjectTableRow;
  class RelationalTableRow;

  /** @class RalSchemaValidation RalSchemaValidation.h
   *
   *  Private utility class to implement COOL relational schema validation.
   *
   *  @author Andrea Valassi
   *  @date   2007-10-22
   */

  class RalSchemaValidation
  {

  public:

    /// Constructor from a RalDatabase pointer (see RalPrivilegeManager)
    RalSchemaValidation( RalDatabase* db );

    /// Destructor
    virtual ~RalSchemaValidation();

    /// Validate the schema of the full database.
    void validateDatabase();

  protected:

    /// Get a CORAL MessageStream
    coral::MessageStream& log();

    /// Get the RalDatabase reference
    const RalDatabase& db() const { return *m_db; }

    /// --- SCHEMA VALIDATION FOR COOL_2_8_x

    /// Validate the schema of the full database (28x).
    void validateDatabase_28x();

    /// Check SQL types of columns in all tables (220).
    //void i_checkSqlTypes_220();

    /// Check that channels tables exist for all folders (220).
    /// Check that tagSequence tables exist for all nodes but SV folders (220).
    void i_checkNodeTables_220();

    /// Check the names of all nodes (222).
    void i_checkNodeNames_222();

    /// Check that for all nodes the inline payload mode is not null (28x).
    /// Check that for all folder sets inlineMode is 0 and extRef == "" (28x).
    /// Check that for all folders inlineMode is 0/1 and extRef ==/!= "" (28x).
    /// Check that for all nodes the value of the external ref payload (28x).
    /// Check that all folders with inlineMode 0/1 are version 201/280 (28x).
    /// Check the schema of the channels tables of all folders (220).
    void i_checkNodes_28x();

    /// Check the schema of the channels table of a folder (220).
    void i_checkChannelsTable_220( const RelationalTableRow& row );

  private:

    /// Standard constructor is private
    RalSchemaValidation();

    /// Copy constructor is private
    RalSchemaValidation( const RalSchemaValidation& rhs );

    /// Assignment operator is private
    RalSchemaValidation& operator=( const RalSchemaValidation& rhs );

  private:

    /// RalDatabase pointer
    RalDatabase* m_db;

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

    /// Fatal errors received during schema validation
    bool m_fatal;

    /// Errors received during schema validation
    std::vector<std::string> m_errors;

    /// Warnings received during schema validation
    std::vector<std::string> m_warnings;

  };

}

#endif // RELATIONALCOOL_RALSCHEMAVALIDATION_H
