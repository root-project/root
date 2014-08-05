// @(#)geom/geocad:$Id$
// Author: Cinzia Luzzi   5/5/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// TGeoToStep Class                                                           //
// --------------------                                                       //
//                                                                            //
// This class is an interface to convert ROOT's geometry file                 //
// to STEP file. The TGeoToStep Class takes a gGeoManager pointer and gives   //
// back a STEP file. gGeoManager is the instance of TGeoManager class         //
// containing tree of geometries creating resulting geometry.                 //
// Standard for the Exchange of Product model data (STEP) is an international //
// standard for the exchange of industrial product data. It is typically used //
// to exchange data between various CAD, CAM and CAE applications.            //
// TGeoToStep Class is using RootOCC class to translate the root geometry     //
// in the corresponding OpenCascade geometry and  and TOCCToStep to write the //
// OpenCascade geometry to the step File.                                     //
// OpenCascade Technology (OCC) is a software development platform freely     //
// available in open source. It includes C++ components for 3D surface and    //
// solid modeling,visualization, data exchange and rapid application          //
// development. For more information about OCC see http://www.opencascade.org //
// Each object in ROOT is represented by an OCC TopoDS_Shape                  //
//                                                                            //
//   This class is needed to be instanciated and can be used calling the      //
//   CreateGeometry method:                                                   //
//   TGeoToStep * mygeom= new TGeoToStep(gGeoManager);                        //
//   mygeom->CreateGeometry();                                                //
//                                                                            //
// The resuling STEP file will be saved in the current directory and called   //
// geometry.stp                                                               //
// To compile the TGeoCad module on ROOT, OpenCascade must be installed!      //
////////////////////////////////////////////////////////////////////////////////

#include "TGeoManager.h"
#include "TOCCToStep.h"
#include "TGeoToStep.h"
#include "TString.h"
#include "TClass.h"

ClassImp(TGeoToStep)

TGeoToStep::TGeoToStep():TObject(), fGeometry(0)
{

}

TGeoToStep::TGeoToStep(TGeoManager *geom):TObject(), fGeometry(geom)
{

}

TGeoToStep::~TGeoToStep()
{
   if (fGeometry) delete fGeometry;
}

void * TGeoToStep::CreateGeometry()
{
   //ROOT CAD CONVERSION
   fCreate = new TOCCToStep();
   fCreate->OCCShapeCreation(fGeometry);
   fCreate->OCCTreeCreation(fGeometry);
   fCreate->OCCWriteStep("geometry.stp");
   //fCreate->PrintAssembly();
   delete(fCreate);
   return NULL;
}
