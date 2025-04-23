// $Id: Exception.h,v 1.58 2012-07-08 20:02:33 avalassi Exp $
#ifndef COOLKERNEL_EXCEPTION_H
#define COOLKERNEL_EXCEPTION_H 1

// First of all, enable or disable the COOL290 API extensions (bug #92204)
#include "CoolKernel/VersionInfo.h"

// Include files
#include <sstream>
#include <exception>
#include "CoolKernel/ChannelId.h"
#include "CoolKernel/ValidityKey.h"

namespace cool {

  //--------------------------------------------------------------------------

  /** @class Exception Exception.h
   *
   *  Top-level base exception class of the COOL package.
   *  Derived from std::exception.
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2004-10-27
   */

  class Exception : public std::exception {

  public:

    /// Constructor
    explicit Exception( const std::string& message,
                        const std::string& domain )
      : m_message( message )
      , m_domain( domain ) {}

    /// Destructor
    virtual ~Exception() throw() {}

    /// Error reporting method
    virtual const char* what() const throw()
    {
      return m_message.c_str();
    }

    /// Return the error domain
    virtual const std::string& domain() const
    {
      return m_domain;
    }

  protected:

    /// Set the error message
    virtual void setMessage( const std::string& message )
    {
      m_message = message;
    }

  private:

    std::string m_message;
    std::string m_domain;

  };

  //--------------------------------------------------------------------------

  /** @class DatabaseNotOpen
   *
   *  Exception thrown when a database is not open.
   */

  class DatabaseNotOpen : public Exception {

  public:

    /// Constructor
    explicit DatabaseNotOpen( const std::string& domain )
      : Exception( "The database is not open", domain ) {}

    /// Destructor
    virtual ~DatabaseNotOpen() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class DatabaseDoesNotExist
   *
   *  Exception thrown when a database does not exist.
   */

  class DatabaseDoesNotExist : public Exception {

  public:

    /// Constructor
    explicit DatabaseDoesNotExist( const std::string& domain )
      : Exception( "The database does not exist", domain ) {}

    /// Destructor
    virtual ~DatabaseDoesNotExist() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class DatabaseOpenInReadOnlyMode
   *
   *  Exception thrown when attempting to update a database
   *  that is open in read-only mode.
   */

  class DatabaseOpenInReadOnlyMode : public Exception {

  public:

    /// Constructor
    explicit DatabaseOpenInReadOnlyMode( const std::string& domain )
      : Exception
    ( "The database is open in read-only mode and cannot be updated",
      domain ) {}

    /// Destructor
    virtual ~DatabaseOpenInReadOnlyMode() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class NodeExists
   *
   *  Exception thrown during folder or folderset creation when
   *  a folder or folder set with that name already exists.
   */

  class NodeExists : public Exception {

  public:

    /// Constructor
    explicit NodeExists( const std::string& fullPath,
                         const std::string& domain )
      : Exception( "Node " + fullPath + " already exists", domain ) {}

    /// Destructor
    virtual ~NodeExists() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class NodeNotFound
   *
   *  Exception thrown when a node with a given name, id and/or other
   *  properties cannot be found (e.g. a node with a given id does not exist,
   *  or a node with a given full name is a folder and not a folder set).
   */

  class NodeNotFound : public Exception {

  public:

    /// Constructor
    explicit NodeNotFound( const std::string& message,
                           const std::string& domain )
      : Exception( message, domain ) {}

    /// Constructor
    explicit NodeNotFound( UInt32 nodeId,
                           const std::string& domain )
      : Exception( "", domain )
    {
      std::ostringstream msg;
      msg << "Node #" << nodeId << " not found";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~NodeNotFound() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class FolderNotFound
   *
   *  Exception thrown when a folder with a given name cannot be found
   *  (either because a node with that name does not exist or because
   *  the name indicates a folder set).
   */

  class FolderNotFound : public NodeNotFound {

  public:

    /// Constructor
    explicit FolderNotFound( const std::string& fullPath,
                             const std::string& domain,
                             bool isFolderSet = false )
      : NodeNotFound( "Folder " + fullPath + " not found", domain )
      , m_isFolderSet( isFolderSet ) {}

    /// Destructor
    virtual ~FolderNotFound() throw() {}

    /// Does the full path indicate a folder set instead of a folder?
    bool isFolderSet() const { return m_isFolderSet; }

  private:

    /// Does the full path indicate a folder set instead of a folder?
    bool m_isFolderSet;

  };

  //--------------------------------------------------------------------------

  /** @class FolderSetNotFound
   *
   *  Exception thrown when a folder set with a given name cannot be found
   *  (either because a node with that name does not exist or because the
   *  name indicates a folder).
   */

  class FolderSetNotFound : public NodeNotFound {

  public:

    /// Constructor
    explicit FolderSetNotFound( const std::string& fullPath,
                                const std::string& domain,
                                bool isFolder = false )
      : NodeNotFound( "Folder set " + fullPath + " not found", domain )
      , m_isFolder( isFolder ) {}

    /// Destructor
    virtual ~FolderSetNotFound() throw() {}

    /// Does the full path indicate a folder instead of a folder set?
    bool isFolder() const { return m_isFolder; }

  private:

    /// Does the full path indicate a folder instead of a folder set?
    bool m_isFolder;

  };

  //--------------------------------------------------------------------------

  /** @class NodeIsSingleVersion
   *
   *  Exception thrown when an HVS node does not support HVS tagging because
   *  it exists in a single version.
   */

  class NodeIsSingleVersion : public Exception {

  public:

    /// Constructor
    explicit NodeIsSingleVersion( const std::string& message,
                                  const std::string& domain )
      : Exception( message, domain ) {}

    /// Destructor
    virtual ~NodeIsSingleVersion() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class FolderIsSingleVersion
   *
   *  Exception thrown when tagging operations are attempted within
   *  a folder that does not support versioning (single-version folder).
   */

  class FolderIsSingleVersion : public NodeIsSingleVersion {

  public:

    /// Constructor
    explicit FolderIsSingleVersion( const std::string& fullPath,
                                    const std::string& domain )
      : NodeIsSingleVersion( "", domain )
    {
      std::ostringstream msg;
      msg << "Folder " << fullPath
          << " does not support IOV versioning or HVS tagging"
          << " (single version folder)";
      setMessage( msg.str() );
    }

    /// Constructor
    explicit FolderIsSingleVersion( const std::string& fullPath,
                                    const std::string& message,
                                    const std::string& domain )
      : NodeIsSingleVersion( "", domain )
    {
      std::ostringstream msg;
      msg << "Folder " << fullPath
          << " does not support IOV versioning or HVS tagging"
          << " (single version folder) - " << message;
      setMessage( msg.str() );
    }

    /// Constructor
    explicit FolderIsSingleVersion( UInt32 nodeId,
                                    const std::string& domain )
      : NodeIsSingleVersion( "", domain )
    {
      std::ostringstream msg;
      msg << "Folder #" << nodeId
          << " does not support IOV versioning or HVS tagging"
          << " (single version folder)";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~FolderIsSingleVersion() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class NodeRelationNotFound
   *
   *  Exception thrown when the expected ancestor-descendant relation
   *  between two given nodes cannot be found (either they are not related
   *  in any way, if the flag is false; or they are not parent and child,
   *  if the flag is true, but they still may be related more distantly).
   */

  class NodeRelationNotFound : public Exception {

  public:

    /// Constructor
    explicit NodeRelationNotFound( UInt32 ancestorNodeId,
                                   UInt32 descendantNodeId,
                                   const std::string& domain,
                                   bool parentOnly = false )
      : Exception( "", domain )
      , m_parentOnly( parentOnly )
    {
      std::ostringstream msg;
#ifndef COOL290EX
      if ( parentOnly )
        msg << "Node #" << descendantNodeId
            << " is not a descendant of node #" << ancestorNodeId;
      else
        msg << "Node #" << descendantNodeId
            << " is not a child of node #" << ancestorNodeId;
#else
      if ( parentOnly )
        msg << "Node #" << descendantNodeId
            << " is not a child of node #" << ancestorNodeId;
      else
        msg << "Node #" << descendantNodeId
            << " is not a descendant of node #" << ancestorNodeId;
#endif
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~NodeRelationNotFound() throw() {}

    /// Was a parent-child relation the only tested relation?
    bool parentOnly() const { return m_parentOnly; }

  private:

    /// Was a parent-child relation the only tested relation?
    bool m_parentOnly;

  };

  //--------------------------------------------------------------------------

  /** @class ObjectNotFound
   *
   *  Exception thrown when an object does not exist.
   */

  class ObjectNotFound : public Exception {

  public:

    /// Constructor
    explicit ObjectNotFound( const std::string& objectIdentity,
                             const std::string& objectContainer,
                             const std::string& domain )
      : Exception ( "Object at '" + objectIdentity
                    + "' not found in " + objectContainer,
                    domain ) {}

    /// Constructor
    explicit ObjectNotFound( const std::string& message,
                             const std::string& domain )
      : Exception ( "Object not found - " + message, domain ) {}

    /// Destructor
    virtual ~ObjectNotFound() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class TagExists
   *
   *  Exception thrown during tag creation when a tag already exists.
   *  Tag names, except for "HEAD", are case sensitive.
   */

  class TagExists : public Exception {

  public:

    /// Constructor
    explicit TagExists( const std::string& tagName,
                        const std::string& domain )
      : Exception( "Tag '" + tagName + "' already exists", domain ) {}

    /// Destructor
    virtual ~TagExists() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class TagNotFound
   *
   *  Exception thrown when a tag does not exist.
   *  Tag names, except for "HEAD", are case sensitive.
   */

  class TagNotFound : public Exception {

  public:

    /// Constructor
    explicit TagNotFound( const std::string& message,
                          const std::string& domain )
      : Exception( message, domain ) {}

    /// Constructor
    explicit TagNotFound( const std::string& tagName,
                          const std::string& nodeName,
                          const std::string& domain )
      : Exception( "", domain )
    {
      std::ostringstream msg;
      msg << "Tag '" << tagName << "' not found in node '" << nodeName << "'";
      setMessage( msg.str() );
    }

    /// Constructor
    explicit TagNotFound( const std::string& tagName,
                          UInt32 nodeId,
                          const std::string& domain )
      : Exception( "", domain )
    {
      std::ostringstream msg;
      msg << "Tag '" << tagName << "' not found in node #" << nodeId;
      setMessage( msg.str() );
    }

    /// Constructor
    explicit TagNotFound( UInt32 tagId,
                          UInt32 nodeId,
                          const std::string& domain )
      : Exception( "", domain )
    {
      std::ostringstream msg;
      msg << "Tag #" << tagId << " not found in node #" << nodeId;
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~TagNotFound() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class ReservedHeadTag
   *
   *  Exception thrown when trying to create or delete the HEAD tag
   *  ("", "HEAD" or a case-insensitive variant such as "head" or "Head").
   *  Also thrown when trying to retrieve the name of the node where
   *  the HEAD tag is defined (it is defined in all folders).
   */

  class ReservedHeadTag : public Exception {

  public:

    /// Constructor
    explicit ReservedHeadTag( const std::string& tagName,
                              const std::string& domain )
      : Exception( "Tag '" + tagName +
                   "' is a reserved HEAD tag: it is defined for all folders"
                   + " and cannot be created or deleted", domain ) {}

    /// Destructor
    virtual ~ReservedHeadTag() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class TagIsLocked
   *
   *  Exception thrown when attempting to modify a locked tag.
   *
   */

  class TagIsLocked : public Exception {

  public:

    /// Constructor
    explicit TagIsLocked( const std::string& message,
                          const std::string& domain )
      : Exception( message, domain ) {}

    /// Destructor
    virtual ~TagIsLocked() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class InvalidTagRelation
   *
   *  Exception thrown when attempting to create an invalid tag relation,
   *  e.g. between the root node and its not-existing parent node
   *  or between two tags with the same name in different nodes.
   */

  class InvalidTagRelation : public Exception {

  public:

    /// Constructor
    explicit InvalidTagRelation( const std::string& parentTagName,
                                 const std::string& childTagName,
                                 UInt32 childNodeId,
                                 const std::string& details,
                                 const std::string& domain )
      : Exception( "", domain )
    {
      std::ostringstream msg;
      msg << "A tag relation cannot be created" 
          << " between tag '" << childTagName << "' in node #" << childNodeId
          << " and parent tag '" << parentTagName << "' in the parent node: " 
          << details;
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~InvalidTagRelation() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class TagRelationExists
   *
   *  Exception thrown when a child tag related to the given parent tag
   *  already exists in the given child of a given parent node.
   */

  class TagRelationExists : public Exception {

  public:

    /// Constructor
    explicit TagRelationExists( UInt32 parentNodeId,
                                UInt32 parentTagId,
                                UInt32 childNodeId,
                                const std::string& domain )
      : Exception( "", domain )
    {
      std::ostringstream msg;
      msg << "A child tag already exists in node #" << childNodeId
          << " related to tag #" << parentTagId
          << " in parent node #" << parentNodeId;
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~TagRelationExists() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class TagRelationNotFound
   *
   *  Exception thrown when no child tag related to the given parent tag
   *  can be found in the given child of a given parent node.
   */

  class TagRelationNotFound : public Exception {

  public:

    /// Constructor
    explicit TagRelationNotFound( UInt32 parentNodeId,
                                  UInt32 parentTagId,
                                  UInt32 childNodeId,
                                  const std::string& domain )
      : Exception( "", domain )
    {
      std::ostringstream msg;
      msg << "No child tag can be found in node #" << childNodeId
          << " related to tag #" << parentTagId
          << " in parent node #" << parentNodeId;
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~TagRelationNotFound() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class ValidityKeyException
   *
   *  Base exception class for validity key exceptions in COOL.
   *  Derived from COOL Exception.
   *
   *  @author Andrea Valassi and Sven A. Schmidt
   *  @date   2004-12-06
   */

  class ValidityKeyException : public Exception {

  public:

    /// Constructor
    explicit ValidityKeyException( const std::string& message,
                                   const std::string& domain )
      : Exception( message, domain ) {}

    /// Destructor
    virtual ~ValidityKeyException() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class ValidityKeyOutOfBoundaries
   *
   *  Exception thrown when validity key is too low or too high.
   */

  class ValidityKeyOutOfBoundaries : public ValidityKeyException {

  public:

    /// Constructor
    explicit ValidityKeyOutOfBoundaries( const ValidityKey& key,
                                         const std::string& domain )
      : ValidityKeyException( "", domain )
    {
      std::ostringstream msg;
      msg << "Validity key out of boundaries: key=" << key
          << ", min=" << ValidityKeyMin
          << ", max=" << ValidityKeyMax;
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~ValidityKeyOutOfBoundaries() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class ValidityIntervalBackwards
   *
   *  Exception thrown when since > until in a validity interval.
   */

  class ValidityIntervalBackwards : public ValidityKeyException {

  public:

    /// Constructor
    explicit ValidityIntervalBackwards( const ValidityKey& since,
                                        const ValidityKey& until,
                                        const std::string& domain )
      : ValidityKeyException( "", domain )
    {
      std::ostringstream msg;
      msg << "Validity interval has since>=until: since=" << since
          << ", until=" << until;
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~ValidityIntervalBackwards() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class InvalidChannelName
   *
   *  Exception thrown when attempting to create a new channel or to
   *  select an existing channel in a folder using an invalid name.
   *
   *  Channel names must be between 1 and 255 characters long; they must
   *  start with a letter and must contain only letters, numbers or the '_'
   *  character (these constraints may be relaxed in a future COOL release).
   *  Channel names are unique: a ChannelNameExists exception is thrown when
   *  attempting to create a channel with a name that is already used.
   *  By default, channels are created with no name, i.e. with an empty string
   *  "" as their name: an InvalidChannelName is thrown when trying to select
   *  a channel using "" as its name, because several such channels exist.
   *
   *  @author Andrea Valassi
   *  @date   2006-12-16
   */

  class InvalidChannelName : public Exception {

  public:

    /// Constructor
    explicit InvalidChannelName( size_t size,
                                 const std::string& domain )
      : Exception( "", domain )
    {
      std::ostringstream msg;
      if ( size == 0 )
        msg << "Empty string '' cannot be used to uniquely select channels";
      else
        msg << "Channel name is too long (size=" << size << ", maxSize=255)";
      setMessage( msg.str() );
    }

    /// Constructor
    explicit InvalidChannelName( const std::string& domain )
      : Exception( "", domain )
    {
      std::ostringstream msg;
      msg << "Invalid channel name: channel names must start with a letter"
          << " and contain only letters, numbers or the '_' character";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~InvalidChannelName() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class ChannelNotFound
   *
   *  Exception thrown when a channel with the given name or id does not exist.
   *
   *  @author Sven A. Schmidt and Andrea Valassi
   *  @date   2006-05-28
   */

  class ChannelNotFound : public Exception {

  public:

    /// Constructor
    explicit ChannelNotFound( const std::string& channelName,
                              const std::string& domain )
      : Exception( "", domain )
    {
      std::ostringstream msg;
      msg << "No channel exists with name '" << channelName << "'";
      setMessage( msg.str() );
    }

    /// Constructor
    explicit ChannelNotFound( const ChannelId& channelId,
                              const std::string& domain )
      : Exception( "", domain )
    {
      std::ostringstream msg;
      msg << "No channel exists with id=" << channelId;
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~ChannelNotFound() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class ChannelExists
   *
   *  Exception thrown when attempting to create a channel with a name
   *  or id that is already used by annother exsting channel.
   *
   *  @author Andrea Valassi
   *  @date   2006-12-16
   */

  class ChannelExists : public Exception {

  public:

    /// Constructor
    explicit ChannelExists( const std::string& folderName,
                            const std::string& channelName,
                            const std::string& domain )
      : Exception( "", domain )
    {
      std::ostringstream msg;
      msg << "A channel with name '" << channelName << "' already exists"
          << " in folder '" << folderName << "'";
      setMessage( msg.str() );
    }

    /// Constructor
    explicit ChannelExists( const std::string& folderName,
                            const ChannelId& channelId,
                            const std::string& domain )
      : Exception( "", domain )
    {
      std::ostringstream msg;
      msg << "A channel with id=" << channelId << " already exists"
          << " in folder '" << folderName << "'";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~ChannelExists() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class InvalidChannelRange
   *
   *  Exception thrown when first > last channel in a ChannelSelection.
   *
   *  @author Sven A. Schmidt
   *  @date   2005-08-11
   */

  class InvalidChannelRange : public Exception {

  public:

    /// Constructor
    explicit InvalidChannelRange( const ChannelId& firstChannel,
                                  const ChannelId& lastChannel,
                                  const std::string& domain )
      : Exception( "", domain )
    {
      std::ostringstream msg;
      msg << "ChannelSelection has first channel > last channel: first="
          << firstChannel << ", last=" << lastChannel;
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~InvalidChannelRange() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class InvalidPayloadSpecification.
   *
   *  Exception thrown when attempting to create a new folder using an
   *  invalid payload specification.
   *
   *  Payload specifications can have at most 900 fields, including up to 10
   *  BLOB fields and up to 200 String255 fields; field names must be between
   *  1 and 30 characters long (including only letters, digits or '_'), must
   *  start with a letter and cannot start with the "COOL_" prefix (in any
   *  combination of lowercase and uppercase letters).
   *
   *  @author Andrea Valassi
   *  @date   2007-01-08
   */

  class InvalidPayloadSpecification : public Exception {

  public:

    /// Constructor
    explicit InvalidPayloadSpecification( const std::string& message,
                                          const std::string& domain )
      : Exception( message, domain ) {}

    /// Destructor
    virtual ~InvalidPayloadSpecification() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class PayloadSpecificationTooManyFields.
   *
   *  Exception thrown when attempting to create a new folder using a
   *  payload specification with more than 900 fields.
   *
   *  @author Andrea Valassi
   *  @date   2007-01-08
   */

  class PayloadSpecificationTooManyFields
    : public InvalidPayloadSpecification {

  public:

    /// Constructor
    explicit PayloadSpecificationTooManyFields( UInt32 nFields,
                                                const std::string& domain )
      : InvalidPayloadSpecification( "", domain )
    {
      std::ostringstream msg;
      msg << "Payload specification has too many fields: #fields="
          << nFields << ", max=900";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~PayloadSpecificationTooManyFields() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class PayloadSpecificationTooManyBlobFields.
   *
   *  Exception thrown when attempting to create a new folder using a
   *  payload specification with more than 10 BLOB fields.
   *
   *  @author Andrea Valassi
   *  @date   2007-01-08
   */

  class PayloadSpecificationTooManyBlobFields
    : public InvalidPayloadSpecification {

  public:

    /// Constructor
    explicit PayloadSpecificationTooManyBlobFields( UInt32 nBlobFields,
                                                    const std::string& domain )
      : InvalidPayloadSpecification( "", domain )
    {
      std::ostringstream msg;
      msg << "Payload specification has too many BLOB fields: #blobFields="
          << nBlobFields << ", max=10";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~PayloadSpecificationTooManyBlobFields() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class PayloadSpecificationTooManyString255Fields.
   *
   *  Exception thrown when attempting to create a new folder using a
   *  payload specification with more than 200 String255 fields.
   *
   *  @author Andrea Valassi
   *  @date   2007-01-10
   */

  class PayloadSpecificationTooManyString255Fields
    : public InvalidPayloadSpecification {

  public:

    /// Constructor
    explicit
    PayloadSpecificationTooManyString255Fields( UInt32 nString255Fields,
                                                const std::string& domain )
      : InvalidPayloadSpecification( "", domain )
    {
      std::ostringstream msg;
      msg << "Payload specification has too many String255 fields:"
          << " #string255Fields=" << nString255Fields << ", max=200";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~PayloadSpecificationTooManyString255Fields() throw() {}

  };

  //--------------------------------------------------------------------------

  /** @class PayloadSpecificationInvalidFieldName.
   *
   *  Exception thrown when attempting to create a new folder using a
   *  payload specification with one or more invalid field names.
   *
   *  Names of payload fields must have between 1 and 30 characters (including
   *  only letters, digits or '_'), must start with a letter and cannot start
   *  with the "COOL_" prefix (in any lowercase/uppercase combination).
   *
   *  @author Andrea Valassi
   *  @date   2007-01-08
   */

  class PayloadSpecificationInvalidFieldName
    : public InvalidPayloadSpecification {

  public:

    /// Constructor
    explicit PayloadSpecificationInvalidFieldName( const std::string& name,
                                                   const std::string& domain )
      : InvalidPayloadSpecification( "", domain )
    {
      std::ostringstream msg;
      msg << "Payload field has invalid name '" << name << "'";
      setMessage( msg.str() );
    }

    /// Destructor
    virtual ~PayloadSpecificationInvalidFieldName() throw() {}

  };

  //--------------------------------------------------------------------------

}
#endif
