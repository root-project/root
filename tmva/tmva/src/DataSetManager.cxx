// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : DataSetManager                                                        *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2006:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::DataSetManager
\ingroup TMVA

Class that contains all the data information.

*/

#include <vector>
#include <iostream>
using std::endl;

#include "TMVA/DataSetManager.h"
#include "TMVA/DataSetFactory.h"
#include "TMVA/DataSet.h"
#include "TMVA/DataSetInfo.h"
#include "TMVA/MsgLogger.h"

#include "TMVA/Types.h"

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::DataSetManager::DataSetManager( DataInputHandler& dataInput )
   : fDatasetFactory(0),
     fDataInput(&dataInput),
     fDataSetInfoCollection(),
     fLogger( new MsgLogger("DataSetManager", kINFO) )
{
}

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::DataSetManager::DataSetManager( )
: fDatasetFactory(0),
fDataInput(0),
fDataSetInfoCollection(),
fLogger( new MsgLogger("DataSetManager", kINFO) )
{
}


////////////////////////////////////////////////////////////////////////////////
/// destructor
///   fDataSetInfoCollection.SetOwner(); // DSMTEST --> created a segfault because the DataSetInfo-objects got deleted twice

TMVA::DataSetManager::~DataSetManager()
{
   if(fDatasetFactory) delete fDatasetFactory;

   delete fLogger;
}

////////////////////////////////////////////////////////////////////////////////
/// Creates the singleton dataset

TMVA::DataSet* TMVA::DataSetManager::CreateDataSet( const TString& dsiName )
{
   DataSetInfo* dsi = GetDataSetInfo( dsiName );
   if (!dsi) Log() << kFATAL << "DataSetInfo object '" << dsiName << "' not found" << Endl;
   if (!fDataInput ) Log() << kFATAL << "DataInputHandler object 'fDataInput' not found" << Endl;

   // factory to create dataset from datasetinfo and datainput
   if(!fDatasetFactory) { fDatasetFactory =new  DataSetFactory(); }
   return fDatasetFactory->CreateDataSet( *dsi, *fDataInput );
}

////////////////////////////////////////////////////////////////////////////////
/// returns datasetinfo object for given name

TMVA::DataSetInfo* TMVA::DataSetManager::GetDataSetInfo(const TString& dsiName)
{
   return (DataSetInfo*)fDataSetInfoCollection.FindObject( dsiName );
}

////////////////////////////////////////////////////////////////////////////////
/// stores a copy of the dataset info object

TMVA::DataSetInfo& TMVA::DataSetManager::AddDataSetInfo(DataSetInfo& dsi)
{
   dsi.SetDataSetManager( this ); // DSMTEST

   DataSetInfo * dsiInList = GetDataSetInfo(dsi.GetName());
   if (dsiInList!=0) return *dsiInList;
   fDataSetInfoCollection.Add( const_cast<DataSetInfo*>(&dsi) );
   return dsi;
}
