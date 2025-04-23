// $Id: RalSchemaEvolution.h,v 1.34 2009-12-16 17:27:48 avalassi Exp $
#ifndef RELATIONALCOOL_RALSCHEMAEVOLUTION_H
#define RELATIONALCOOL_RALSCHEMAEVOLUTION_H 1

// Include files
#include <boost/shared_ptr.hpp>
#include "CoolKernel/DatabaseId.h"
#include "CoolKernel/RecordSpecification.h"
#include "CoralBase/MessageStream.h"
#include "RelationalAccess/IConnectionService.h"

// Local include files
#include "../../src/CoralConnectionServiceProxy.h"

namespace cool
{

  // Forward declarations
  class RalDatabase;
  class RelationalObjectTableRow;
  class RelationalTableRow;

  /** @class RalSchemaEvolution RalSchemaEvolution.h
   *
   *  Private utility class to implement COOL relational schema evolution.
   *
   *  @author Andrea Valassi
   *  @date   2006-03-15
   */

  class RalSchemaEvolution
  {

  public:

    /// Constructor
    RalSchemaEvolution( CoralConnectionServiceProxyPtr ppConnSvc,
                        const DatabaseId& dbId );

    /// Destructor
    virtual ~RalSchemaEvolution();

    /// Evolve the schema of the full database to version 220.
    /// Encapsulate schema evolution in a single transaction.
    void evolveDatabase();

  protected:

    /// Get a CORAL MessageStream
    coral::MessageStream& log();

    /// Get the RalDatabase reference
    const RalDatabase& db() const { return *m_db; }

    /// --- SCHEMA EVOLUTION FROM COOL_1_3_0 TO COOL_2_0_0

    /// Evolve the schema of the full database (130->200).
    void evolveDatabase_130_to_200();

    /// Evolve the schema of the node table (130->200).
    void i_evolveNodeTable_130_200();

    /// Evolve the schema of the global tag table (130->200).
    void i_evolveGlobalTagTable_130_200();

    /// Evolve the schema of the tag2tag table (130->200).
    void i_evolveTag2TagTable_130_200();

    /// Create the tag shared sequence (130->200).
    void i_createTagSharedSequence_130_200();

    /// Create the IOV shared sequence (130->200).
    void i_createIovSharedSequence_130_200();

    /// Evolve the IOV table schema for a given folder (130->200).
    void i_evolveIovTable_130_to_200( const RelationalTableRow& row );

    /// Create the channels table for a given folder (130->200).
    void i_createChannelsTable_130_to_200( const RelationalTableRow& row );

    /// Fill the channels tables for a given SV folder (130->200).
    void i_fillChannelsTableSV_130_to_200( const RelationalTableRow& row );

    /// Update node table with updated values for a given folder (130->200).
    void i_updateNodeTable_130_to_200( const RelationalTableRow& row );

    /// Evolve the schema and contents of the main table (130->200).
    void i_evolveMainTable_130_to_200();

    /// Alter selected SQL types in all tables (130->200, MySQL/sqlite only).
    void i_alterSqlTypes_130_to_200();

    /// Decode a 200 RecordSpecification from a 130 encoded 'extended' EALS
    static const RecordSpecification
    decodeRecordSpecification130( const std::string& encodedEALS );

    /// Decode a 200 StorageType from a 130 encoded type specification
    static const StorageType&
    storageType130( const std::string& encodedType );

    const std::vector<ChannelId>
    listChannels( const std::string& objectTableName ) const;

    RelationalObjectTableRow
    fetchLastRow( const std::string& objectTableName,
                  const ChannelId& channelId );

    const RelationalObjectTableRow
    fetchRowForId( const std::string& objectTableName,
                   const unsigned int objectId ) const;

    void insertChannelTableRow( const std::string& channelTableName,
                                const ChannelId& channelId,
                                const unsigned int lastObjectId,
                                const bool hasNewData,
                                const std::string& channelName,
                                const std::string& description ) const;

    /// --- SCHEMA EVOLUTION FROM COOL_2_0_0 TO COOL_2_2_0

    /// Evolve SQL types and the schema of individual nodes (200->220).
    void evolveDatabase_200_to_220();

    /// Alter selected SQL types in all tables (200->220, MySQL only).
    void i_alterSqlTypes_200_to_220();

    /// Evolve the schema of individual nodes (200->220).
    void i_evolveNodes_200_to_220();

    /// Fill the channels tables for a given MV folder (200->220).
    void i_fillChannelsTableMV_200_to_220( const RelationalTableRow& row );

    /// Evolve the IOV table schema for a given folder (200->220).
    void i_evolveIovTable_200_to_220( const RelationalTableRow& row );

    /// Update node table with updated values for a given folder (200->220).
    void i_updateNodeTable_200_to_220( const RelationalTableRow& row );

    /// --- SCHEMA EVOLUTION FROM COOL_2_2_0 TO COOL_2_8_1

    /// Evolve the node table (220->281).
    void evolveDatabase_220_to_281();

    /// Update node table by replacing NULL with 0 or "" (220->281).
    void i_updateNodeTable_220_to_281();

  private:

    /// Standard constructor is private
    RalSchemaEvolution();

    /// Copy constructor is private
    RalSchemaEvolution( const RalSchemaEvolution& rhs );

    /// Assignment operator is private
    RalSchemaEvolution& operator=( const RalSchemaEvolution& rhs );

  private:

    /// RalDatabase pointer
    RalDatabase* m_db;

    /// CORAL MessageStream
    std::auto_ptr<coral::MessageStream> m_log;

  };

}

#endif // RELATIONALCOOL_RALSCHEMAEVOLUTION_H
