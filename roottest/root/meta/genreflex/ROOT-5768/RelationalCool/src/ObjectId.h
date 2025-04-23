// $Id: ObjectId.h,v 1.10 2009-12-16 17:17:37 avalassi Exp $
#ifndef RELATIONALCOOL_OBJECTID_H
#define RELATIONALCOOL_OBJECTID_H 1

// Include files
#include "CoolKernel/Exception.h"

namespace cool
{

  /** @class ObjectId ObjectId.h
   *
   *  This file summarises the objectId numbering scheme assumptions.
   *
   *  Presently ObjectId is typedefed to an unsigned int.
   *  Eventually ObjectId may be changed to a pair of unsigned int or
   *  something else.
   *
   *  Presently this is only used in the RalDatabase "tagAsOfObjectId"
   *  implementation. Eventually this should be used as the unique source
   *  for the numbering schema.
   *
   *  Presently even the typedef is only used in the RalDatabase
   *  "tagAsOfObjectId". Eventually all code manipulating objectId's should
   *  use the typedef.
   *
   *  @author Andrea Valassi and Sven A. Schmidt
   *  @date   2005-04-05
   */

  typedef unsigned int ObjectId;

  const unsigned int ObjectIdIncrement = 6;

  class ObjectIdTest;

  namespace ObjectIdHandler {

    /// Returns true if the given id belongs to a user object
    bool isUserObject( const ObjectId& id );

    /// Returns true if the given id belongs to a "left inserted"
    /// system object
    bool isLSysObject( const ObjectId& id );

    /// Returns true if the given id belongs to a "right inserted"
    /// system object
    bool isRSysObject( const ObjectId& id );

    /// Returns true if the given id belongs to a system object
    bool isSysObject( const ObjectId& id );

    /// Transforms a given id into a user object id
    ObjectId userObject( const ObjectId& id );

    /// Transforms a given id into a "left inserted" system object id
    /// Throws an ObjectIdExecption if the given id is out of bounds.
    ObjectId lSysObject( const ObjectId& id );

    /// Transforms a given id into a "right inserted" system object id
    /// Throws an ObjectIdExecption if the given id is out of bounds.
    ObjectId rSysObject( const ObjectId& id );

    /// Returns a given id's previous user object id
    /// Throws an ObjectIdExecption if the given id is out of bounds.
    ObjectId prevUserObject( const ObjectId& id );

    /// Returns a given id's next user object id
    /// Throws an ObjectIdExecption if the given id is out of bounds.
    ObjectId nextUserObject( const ObjectId& id );

  }


  /** @class ObjectIdException
   *
   *  Exception thrown by the ObjectIdHandler.
   */

  class ObjectIdException : public Exception {

  public:

    /// Constructor
    ObjectIdException( const std::string& message )
      : Exception( message, "ObjectId" ) {}

    /// Destructor
    virtual ~ObjectIdException() throw() {}

  };

}

#endif // RELATIONALCOOL_OBJECTID_H
