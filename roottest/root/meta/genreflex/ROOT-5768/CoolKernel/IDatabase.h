// $Id: IDatabase.h,v 1.92 2012-07-08 20:02:33 avalassi Exp $
#ifndef COOLKERNEL_IDATABASE_H
#define COOLKERNEL_IDATABASE_H 1

// First of all, enable or disable the COOL290 API extensions (see bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include "CoolKernel/DatabaseId.h"
#include "CoolKernel/FolderVersioning.h"
#include "CoolKernel/IHvsNode.h"
#include "CoolKernel/pointers.h"
#ifdef COOL300
#include "CoolKernel/ITransaction.h"
#endif

namespace cool
{

  class IFolderSpecification;
  class IRecord;
  class IRecordSpecification;

  /** @class IDatabase IDatabase.h
   *
   *  Abstract interface to one COOL "condition database" instance of a
   *  given technology (deployed using a specific physical infrastructure).
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2004-11-09
   */

  class IDatabase
  {

  public:

    /// Destructor.
    virtual ~IDatabase() {}

    /// Return the global identifier of the database
    /// [WARNING: any visible passwords are masked out].
    virtual const DatabaseId& databaseId() const = 0;

    /// Return the 'attributes' of the database
    /// (implementation-specific properties not exposed in the API).
    /// Throws DatabaseNotOpen if the database is not open.
    virtual const IRecord& databaseAttributes() const = 0;

    /*
    /// Does the database support this payload specification?
    virtual bool isValidPayloadSpecification
    ( const IRecordSpecification& spec ) = 0;
    */

    /*
    /// Does the database support this channel specification?
    virtual bool isValidChannelSpecification
    ( const IRecordSpecification& spec ) = 0;
    */

    /// Create a new folder set and return the corresponding manager.
    /// The ownership of the folderset manager instance is shared.
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    /// Throws NodeExists if a folder[set] with the same path already exists.
    /// Throws an Exception if the max# of folder[set]s (9999) is exceeded.
    /// Throws an Exception if an invalid versioning mode has been specified.
    /// Throws an Exception if the user does not have writer privileges.
    virtual IFolderSetPtr createFolderSet
    ( const std::string& fullPath,
      const std::string& description = "",
      bool createParents = false ) = 0;

    /// Does this folder set exist?
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    virtual bool existsFolderSet( const std::string& folderSetName ) = 0;

    /// Retrieve an existing folderset and return the corresponding manager.
    /// The ownership of the folderset manager instance is shared.
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    /// Throws FolderSetNotFound if the folderset does not exist.
    virtual IFolderSetPtr getFolderSet( const std::string& fullPath ) = 0;

    /// Create a new folder and return the corresponding manager.
    /// The ownership of the folder manager instance is shared.
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    /// Throws InvalidPayloadSpecification if the payload specification is
    /// invalid: there can be at most 900 fields, including up to 10 BLOB
    /// fields and up to 200 String255 fields; field names must be between
    /// 1 and 30 characters (including only letters, digits or '_'), must
    /// start with a letter and cannot start with the "COOL_" prefix (in any
    /// combination of lowercase and uppercase letters).
    /// Throws NodeExists if a folder[set] with the same path already exists.
    /// Throws an Exception if the max# of folder[set]s (9999) is exceeded.
    /// Throws an Exception if the user does not have writer privileges.
    virtual IFolderPtr createFolder
    ( const std::string& fullPath,
      const IFolderSpecification& folderSpec,
      const std::string& description = "",
      bool createParents = false ) = 0;

    /// DEPRECATED: use IFolderSpecification instead of IRecordSpecification!
    /// This is similar to the COOL1.3.3 API (with IRecordSpecification
    /// instead of ExtendedAttributeListSpecification), for easier porting of
    /// user code, but it is likely to be removed in a future COOL release.
    virtual IFolderPtr createFolder
    ( const std::string& fullPath,
      const IRecordSpecification& payloadSpec,
      const std::string& description = "",
      FolderVersioning::Mode mode = FolderVersioning::SINGLE_VERSION,
      bool createParents = false ) = 0;

    /// Does this folder exist?
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    virtual bool existsFolder( const std::string& fullPath ) = 0;

    /// Retrieve an existing folder and return the corresponding manager.
    /// The ownership of the folder manager instance is shared.
    /// Throws DatabaseNotOpen if the database is not open.
    /// Throws HvsPathHandlerException if the given path has an invalid format.
    /// Throws FolderNotFound if the folder does not exist.
    virtual IFolderPtr getFolder( const std::string& fullPath ) = 0;

    /// Return the list of existing nodes
    /// (in ascending/descending alphabetical order).
    virtual const std::vector<std::string> listAllNodes( bool ascending = true ) = 0;

    /// Drop an existing node (folder or folder set).
    /// Also delete any tags associated to the node.
    /// Return true if the node and all its structures are dropped as expected.
    /// Return false (without throwing any exception) if the node and
    /// all its structures do not exist any more on exit from this method,
    /// but the node or some of its structures did not exist to start with.
    /// Throw an Exception if the node schema version is more recent than
    /// the schema version supported by the current COOL software release.
    /// Throw an Exception if the node or one of its structures cannot
    /// be dropped (i.e. continue to exist on exit from this method).
    /// Throw an Exception if the node is a non-empty folder set.
    /// Throw an Exception if any associated tags cannot be deleted.
    virtual bool dropNode( const std::string& fullPath ) = 0;

    /// HVS: does this tag exist?
    /// Tag names, except for "HEAD", are case sensitive.
    /// Returns true for the reserved tags "" and "HEAD".
    virtual bool existsTag( const std::string& tagName ) const = 0;

    /// HVS: return the node type (inner/leaf) where this tag name can be used.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Throws TagNotFound if the tag does not exist.
    virtual IHvsNode::Type tagNameScope( const std::string& tagName ) const = 0;

    /// HVS: return the names of the nodes where this tag is defined.
    /// Tag names, except for "HEAD", are case sensitive.
    /// Throws TagNotFound if the tag does not exist.
    /// Throws ReservedHeadTag for the HEAD tag (defined in all folders).
    virtual const std::vector<std::string>
    taggedNodes( const std::string& tagName ) const = 0;

    /// Is the database 'open'?
    virtual bool isOpen() const = 0;

    /// (Re)opens the database.
    virtual void openDatabase() = 0;

    /// Closes the database.
    virtual void closeDatabase() = 0;

    /// Return the "COOL database name".
    virtual const std::string& databaseName() const = 0;

#ifdef COOL300
    /// Start a new transaction and enter manual transaction mode
    virtual ITransactionPtr startTransaction() = 0;
#endif

#ifdef COOL290CO
  private:

    /// Assignment operator is private (see bug #95823)
    IDatabase& operator=( const IDatabase& rhs );
#endif

  };

}
#endif // COOLKERNEL_IDATABASE_H
