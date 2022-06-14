// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer, Joerg Stelzer, Helge Voss

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
 *      Peter Speckmayer <Peter.Speckmayer@cern.ch>  - CERN, Switzerland          *
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

#include "TList.h"
#include "TString.h"

namespace TMVA {

   class DataSet;
   class DataSetInfo;
   class DataInputHandler;
   class DataSetFactory; // DSMTEST
   class MsgLogger;
   class Factory;
   class Envelop;
   class DataSetManager:public TObject {
      friend class Factory;
      friend class Envelop;

   public:

      // private default constructor
      DataSetManager(); // DSMTEST
      DataSetManager( DataInputHandler& dataInput ); //DSMTEST
      ~DataSetManager(); // DSMTEST


      // ownership stays with this handler
      DataSet*     CreateDataSet ( const TString& dsiName );
      DataSetInfo* GetDataSetInfo( const TString& dsiName );

      // makes a local copy of the dataset info object
      DataSetInfo& AddDataSetInfo( DataSetInfo& dsi );

   private:


      TMVA::DataSetFactory* fDatasetFactory;

      // access to input data
      DataInputHandler& DataInput() { return *fDataInput; }

      DataInputHandler           *fDataInput;            ///< source of input data
      TList                      fDataSetInfoCollection; ///< all registered dataset definitions
      MsgLogger*                 fLogger;                ///<! message logger
      MsgLogger& Log() const { return *fLogger; }
   public:

       ClassDef(DataSetManager,1);

   };
}

#endif
