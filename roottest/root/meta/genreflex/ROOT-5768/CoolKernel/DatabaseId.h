// $Id: DatabaseId.h,v 1.6 2006-04-03 17:02:19 avalassi Exp $
#ifndef COOLKERNEL_DATABASEID_H
#define COOLKERNEL_DATABASEID_H 1

#include <string>

namespace cool
{

  /** @class DatabaseId DatabaseId.h
   *
   *  Globally unique identifier of one "conditions database" instance.
   *
   *  Recommended syntax without CORAL alias:
   *    "oracle://aTnsHost;schema=aUserName;dbname=aDbname";
   *    "mysql://aTcpHost;schema=aDatabase;dbname=aDbname";
   *    "sqlite://;schema=aFileName.db;dbname=aDbname";
   *
   *  Recommended syntax with CORAL alias:
   *    "anAlias/aDbname";
   *
   *  @author Andrea Valassi, Sven A. Schmidt and Marco Clemencic
   *  @date   2004-11-09
   */

  typedef std::string DatabaseId;

}

#endif // COOLKERNEL_DATABASEID_H
