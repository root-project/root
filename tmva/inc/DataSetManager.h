// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : DataSetManager                                                        *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Singleton class for dataset management                                    *
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

#ifndef ROOT_TMVA_DataSetManager
#define ROOT_TMVA_DataSetManager

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// DataSetManager                                                       //
//                                                                      //
// Class that contains all the data information                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

namespace TMVA {

   class DataSet;
   class DataSetInfo;
   class DataInputHandler;
   class MsgLogger;

   class DataSetManager {
      
   public:

      // singleton class
      static DataSetManager& Instance();
      static void            CreateInstance( DataInputHandler& dataInput );
      static void            DestroyInstance();

      // ownership stays with this handler
      DataSet*     CreateDataSet ( const TString& dsiName );
      DataSetInfo* GetDataSetInfo( const TString& dsiName );

      // makes a local copy of the dataset info object
      DataSetInfo& AddDataSetInfo( DataSetInfo& dsi );

   private:

      ~DataSetManager();

      static DataSetManager* fgDSManager;

      // private default constructor
      DataSetManager();
      DataSetManager( DataInputHandler& dataInput );

      // access to input data
      DataInputHandler& DataInput() { return fDataInput; }

      DataInputHandler&          fDataInput;             //! source of input data
      TList                      fDataSetInfoCollection; //! all registered dataset definitions
      mutable MsgLogger*         fLogger;   // message logger
      MsgLogger& Log() const { return *fLogger; }    
   };
}

#endif
