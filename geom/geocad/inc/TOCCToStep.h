// @(#)geom/geocad:$Id$
// Author: Cinzia Luzzi   5/5/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TOCCToStep
#define ROOT_TOCCToStep

#include "TGeoNode.h"
#include "TGeoMatrix.h"
#include "TGeoToOCC.h"

#include <TDF_Label.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <TDocStd_Document.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <TDF_Label.hxx>
#include <TopoDS_Shape.hxx>


class TOCCToStep {

private:
   typedef std::map <TGeoVolume *, TDF_Label> LabelMap_t;

   STEPCAFControl_Writer    fWriter; //the step file pointer
   Handle(TDocStd_Document) fDoc;    //the step document element

   // The following probably shouldn't be data members. 
   LabelMap_t               fTree;   //tree of Label's volumes
   TDF_Label                fLabel;  //label of the OCC shape element
   TGeoToOCC                  fRootShape;
   TopoDS_Shape             fShape;  //OCC shape (translated root shape)

   void            OCCDocCreation();
   TopoDS_Shape    AssemblyShape(TGeoVolume *vol, TGeoHMatrix m);
   TGeoVolume     *GetVolumeOfLabel(TDF_Label fLabel);
   TDF_Label       GetLabelOfVolume(TGeoVolume * v);
   void            AddChildLabel(TDF_Label mother, TDF_Label child, TopLoc_Location loc);
   TopLoc_Location CalcLocation(TGeoHMatrix matrix);

   void FillOCCWithNode(TGeoManager* m, TGeoNode* currentNode, TGeoIterator& nextNode, int level, int max_level, int level1_skipped);

public:
   TOCCToStep();
   void      PrintAssembly();
   TDF_Label OCCShapeCreation(TGeoManager *m);
   void      OCCTreeCreation(TGeoManager *m, int max_level = -1);
   bool      OCCPartialTreeCreation(TGeoManager *m, const char* node_name, int max_level = -1);
   bool      OCCPartialTreeCreation(TGeoManager *m, std::map<std::string,int> part_name_levels);


   void      OCCWriteStep(const char *fname);
};

#endif
