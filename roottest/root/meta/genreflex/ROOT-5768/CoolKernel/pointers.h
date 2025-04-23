// $Id: pointers.h,v 1.10 2010-08-26 21:40:09 avalassi Exp $
#ifndef COOLKERNEL_POINTERS_H
#define COOLKERNEL_POINTERS_H 1

// First of all, set/unset COOL290, COOL300 and COOL_HAS_CPP11 macros
#include "CoolKernel/VersionInfo.h"

// Include files
#include <memory>
#include <vector>

namespace cool 
{

  /** @file pointers.h
   *
   * This file collects the boost shared pointer type definitions referenced
   * in the CoolKernel API. With the exception of IObjectPtr, these are used
   * in a single header file and could be moved there, but are all kept here
   * for consistency. There is no need to directly include this file in user
   * code. It is automatically picked up wherever required.
   *
   * @author Sven A. Schmidt and Andrea Valassi
   * @date 2004-11-05
   */

  // Forward declarations
  class IDatabase;
  class IFolder;
  class IFolderSet;
  class IObject;
  class IObjectIterator;
  class IRecord;
  class IRecordIterator;
#ifdef COOL300
  class ITransaction;
#endif
  // STD shared pointers
  typedef std::shared_ptr<IDatabase> IDatabasePtr;
  typedef std::shared_ptr<IFolder> IFolderPtr;
  typedef std::shared_ptr<IFolderSet> IFolderSetPtr;
  typedef std::shared_ptr<IObject> IObjectPtr;
  typedef std::shared_ptr<IObjectIterator> IObjectIteratorPtr;
  typedef std::vector<IObjectPtr> IObjectVector;
  typedef std::shared_ptr<IObjectVector> IObjectVectorPtr;
  typedef std::shared_ptr<IRecord> IRecordPtr;
  typedef std::vector<IRecordPtr> IRecordVector;
  typedef std::shared_ptr<IRecordVector> IRecordVectorPtr;
#ifdef COOL300
  typedef std::shared_ptr<ITransaction> ITransactionPtr;
#endif

}

#endif
