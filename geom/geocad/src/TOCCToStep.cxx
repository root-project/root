// @(#)geom/geocad:$Id$
// Author: Cinzia Luzzi   5/5/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
// TOCCToStep Class                                                     //
//                                                                      //
// This class contains implementation of writing OpenCascade's          //
// geometry shapes to the STEP file reproducing the originary ROOT      //
// geometry tree. The TRootStep Class takes a gGeoManager pointer and   //
// gives back a STEP file.                                              //
// The OCCShapeCreation(TGeoManager *m) method starting from            //
// the top of the ROOT geometry tree translates each ROOT shape in the  //
// OCC one. A fLabel is created for each OCC shape and the              //
// correspondance bewteen the the fLabel and the shape is saved         //
// in a map. The OCCTreeCreation(TGeoManager *m) method starting from   //
// the top of the ROOT geometry and using the fLabel-shape map          //
// reproduce the ROOT tree that will be written to the STEP file using  //
// the OCCWriteStep(const char * fname ) method.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TOCCToStep.h"
#include "TGeoToOCC.h"

#include "TGeoVolume.h"
#include "TString.h"
#include "TClass.h"
#include "TGeoManager.h"
#include "TError.h"

#include <Interface_Static.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <TDataStd_Name.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <Standard.hxx>
#include <stdlib.h>
#include <XCAFApp_Application.hxx>

using namespace std;


//______________________________________________________________________________
TOCCToStep::TOCCToStep()
{
   OCCDocCreation();
}

//______________________________________________________________________________
void TOCCToStep::OCCDocCreation()
{
   Handle (XCAFApp_Application)A = XCAFApp_Application::GetApplication();
   if (!A.IsNull()) {
      A->NewDocument ("MDTV-XCAF", fDoc);
   }
   else
      ::Error("TOCCToStep::OCCDocCreation", "creating OCC application");
}

//______________________________________________________________________________
TDF_Label TOCCToStep::OCCShapeCreation(TGeoManager *m)
{
   // Logical fTree creation.

   TDF_Label motherLabel;
   TGeoVolume * currentVolume;
   TGeoVolume * motherVol;
   TGeoVolume * Top;
   TString path;
   Int_t num = 0;
   Int_t level = 0;
   TIter next(m->GetListOfVolumes());
   fLabel = XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->NewShape();
   fShape = fRootShape.OCC_SimpleShape(m->GetTopVolume()->GetShape());
   XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->SetShape(fLabel, fShape);
   TDataStd_Name::Set(fLabel, m->GetTopVolume()->GetName());
   XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->UpdateAssembly(fDoc->Main());
   Top = m->GetTopVolume();
   fTree[Top] = fLabel;
   while ((currentVolume = (TGeoVolume *)next())) {
      if (GetLabelOfVolume(currentVolume).IsNull()) {
         num = currentVolume->GetNdaughters();
         if ((GetLabelOfVolume(currentVolume).IsNull())) {
            if (currentVolume->GetShape()->IsA()==TGeoCompositeShape::Class()) {
               fShape = fRootShape.OCC_CompositeShape((TGeoCompositeShape*)currentVolume->GetShape(), TGeoIdentity());
            } else {
               fShape = fRootShape.OCC_SimpleShape(currentVolume->GetShape());
            }
         }
         TGeoNode *current;
         TGeoIterator nextNode(m->GetTopVolume());
         while ((current = nextNode())) {
            if ((current->GetVolume() == currentVolume) && (GetLabelOfVolume(current->GetVolume()).IsNull())) {
               level = nextNode.GetLevel();
               nextNode.GetPath(path);
               if (level == 1)
                  motherVol = m->GetTopVolume();
               else {
                  TGeoNode * mother = nextNode.GetNode(--level);
                  motherVol = mother->GetVolume();
               }
               motherLabel = GetLabelOfVolume(motherVol);
               if (!motherLabel.IsNull()) {
                  fLabel = TDF_TagSource::NewChild(motherLabel);
                  break;
               } else {
                  TGeoNode * grandMother = nextNode.GetNode(level);
                  motherVol = grandMother->GetVolume();
                  TopoDS_Shape Mothershape;
                  if (motherVol->GetShape()->IsA()==TGeoCompositeShape::Class()) {
                     Mothershape = fRootShape.OCC_CompositeShape((TGeoCompositeShape*)motherVol->GetShape(), TGeoIdentity());
                  } else {
                     Mothershape = fRootShape.OCC_SimpleShape(motherVol->GetShape());
                  }
                  motherLabel = TDF_TagSource::NewChild(GetLabelOfVolume(Top));
                  XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->SetShape(motherLabel, Mothershape);
                  TDataStd_Name::Set(motherLabel, motherVol->GetName());
                  XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->UpdateAssembly(fDoc->Main());
                  fTree[motherVol] = motherLabel;
                  fLabel = TDF_TagSource::NewChild(motherLabel);
                  break;
               }
            }
         }
         XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->SetShape(fLabel, fShape);
         TDataStd_Name::Set(fLabel, currentVolume->GetName());
         XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->UpdateAssembly(fDoc->Main());
         fTree[currentVolume] = fLabel;
      }
   }
   return fLabel;
}

//______________________________________________________________________________
void TOCCToStep::OCCWriteStep(const char *fname)
{
   STEPControl_StepModelType mode = STEPControl_AsIs;
   fWriter.SetNameMode(Standard_True);
   if (!Interface_Static::SetIVal("write.step.assembly", 1)) { //assembly mode
      Error("TOCCToStep::OCCWriteStep", "failed to set assembly mode for step data");
   }
   if (!fWriter.Transfer(fDoc, mode)) {
      ::Error("TOCCToStep::OCCWriteStep", "error translating document");
   }
   fWriter.Write(fname);
}

//______________________________________________________________________________
TDF_Label TOCCToStep::GetLabelOfVolume(TGeoVolume * v)
{
   TDF_Label null;
   if (fTree.find(v) != fTree.end())
      return fTree[v];
   else
      return null;
}

//______________________________________________________________________________
TGeoVolume * TOCCToStep::GetVolumeOfLabel(TDF_Label fLabel)
{
   map <TGeoVolume *,TDF_Label>::iterator it;
   for(it = fTree.begin(); it != fTree.end(); it++)
      if (it->second.IsEqual(fLabel))
         return it->first;
   return 0;
}

//______________________________________________________________________________
void TOCCToStep::AddChildLabel(TDF_Label mother, TDF_Label child, TopLoc_Location loc)
{
   XCAFDoc_DocumentTool::ShapeTool(mother)->AddComponent(mother, child,loc);
   XCAFDoc_DocumentTool::ShapeTool(mother)->UpdateAssembly(mother);
}

//______________________________________________________________________________
TopLoc_Location TOCCToStep::CalcLocation (TGeoHMatrix matrix)
{
   gp_Trsf TR,TR1;
   TopLoc_Location locA;
   Double_t const *t=matrix.GetTranslation();
   Double_t const *r=matrix.GetRotationMatrix();
   TR1.SetTranslation(gp_Vec(t[0],t[1],t[2]));
   TR.SetValues(r[0],r[1],r[2],0,
                r[3],r[4],r[5],0,
                r[6],r[7],r[8],0
#if OCC_VERSION_MAJOR == 6 && OCC_VERSION_MINOR < 8
                ,0,1
#endif
                );
   TR1.Multiply(TR);
   locA = TopLoc_Location (TR1);
   return locA;
}

//______________________________________________________________________________
void TOCCToStep::OCCTreeCreation(TGeoManager * m)
{
   TGeoIterator nextNode(m->GetTopVolume());
   TGeoNode *currentNode = 0;
   TGeoNode *motherNode = 0;
   //TGeoNode *gmotherNode = 0;
   Int_t level;
   TDF_Label labelMother;
   TopLoc_Location loc;
   Int_t nd;

   while ((currentNode = nextNode())) {
      level = nextNode.GetLevel();
      nd = currentNode->GetNdaughters();
      if (!nd) {
         for (int i = level; i > 0; i--) {
            if (i == 1) {
               motherNode = m->GetTopNode();
            } else
               motherNode = nextNode.GetNode(--level);
            labelMother = GetLabelOfVolume(motherNode->GetVolume());
            Int_t ndMother = motherNode->GetNdaughters();
            fLabel = GetLabelOfVolume(currentNode->GetVolume());
            loc = CalcLocation((*(currentNode->GetMatrix())));
            if ((XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->NbComponents(labelMother) < ndMother) && (!nd)) {
               AddChildLabel(labelMother, fLabel, loc);
            } else if ((XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->NbComponents(fLabel) == nd) && (XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->NbComponents(labelMother) == motherNode->GetVolume()->GetIndex(currentNode))) {
               AddChildLabel(labelMother, fLabel, loc);
            }
            currentNode = motherNode;
            fLabel = labelMother;
            nd = currentNode->GetNdaughters();
         }
      }
   }
}

//______________________________________________________________________________
void TOCCToStep::PrintAssembly()
{
#if OCC_VERSION_MAJOR == 6 && OCC_VERSION_MINOR < 8
   XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->Dump();
#else
   XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->Dump(std::cout);
#endif
}


