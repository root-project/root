// @(#)geom/geocad:$Id$
// Author: Cinzia Luzzi   5/5/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TOCCToStep
\ingroup Geometry_cad

This class contains implementation of writing OpenCascade's
geometry shapes to the STEP file reproducing the original ROOT
geometry tree. The TRootStep Class takes a gGeoManager pointer and
gives back a STEP file.
The OCCShapeCreation(TGeoManager *m) method starting from
the top of the ROOT geometry tree translates each ROOT shape in the
OCC one. A fLabel is created for each OCC shape and the
correspondance between the the fLabel and the shape is saved
in a map. The OCCTreeCreation(TGeoManager *m) method starting from
the top of the ROOT geometry and using the fLabel-shape map
reproduce the ROOT tree that will be written to the STEP file using
the OCCWriteStep(const char * fname ) method.

*/
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


////////////////////////////////////////////////////////////////////////////////

TOCCToStep::TOCCToStep()
{
   OCCDocCreation();
}

////////////////////////////////////////////////////////////////////////////////

void TOCCToStep::OCCDocCreation()
{
   Handle (XCAFApp_Application)A = XCAFApp_Application::GetApplication();
   if (!A.IsNull()) {
      A->NewDocument ("MDTV-XCAF", fDoc);
   }
   else
      ::Error("TOCCToStep::OCCDocCreation", "creating OCC application");
}

////////////////////////////////////////////////////////////////////////////////
/// Logical fTree creation.

TDF_Label TOCCToStep::OCCShapeCreation(TGeoManager *m)
{
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
         //num = currentVolume->GetNdaughters();
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

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////

TDF_Label TOCCToStep::GetLabelOfVolume(TGeoVolume * v)
{
   TDF_Label null;
   if (fTree.find(v) != fTree.end())
      return fTree[v];
   else
      return null;
}

////////////////////////////////////////////////////////////////////////////////

TGeoVolume * TOCCToStep::GetVolumeOfLabel(TDF_Label fLabel)
{
   map <TGeoVolume *,TDF_Label>::iterator it;
   for(it = fTree.begin(); it != fTree.end(); ++it)
      if (it->second.IsEqual(fLabel))
         return it->first;
   return 0;
}

////////////////////////////////////////////////////////////////////////////////

void TOCCToStep::AddChildLabel(TDF_Label mother, TDF_Label child, TopLoc_Location loc)
{
   XCAFDoc_DocumentTool::ShapeTool(mother)->AddComponent(mother, child,loc);
   XCAFDoc_DocumentTool::ShapeTool(mother)->UpdateAssembly(mother);
}

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////

void TOCCToStep::OCCTreeCreation(TGeoManager * m, int max_level)
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
      if( level > max_level ){
        continue;
      }
      // This loop looks for nodes which are the end of line (ancestrally) then navigates
      // back up the family tree.  As it does so, the OCC tree is constructed. 
      // It is not clear why it must be done this way, but it could be an idiosyncracy
      // in OCC (which I am not too familar with at the moment).
      nd = currentNode->GetNdaughters();
      if (!nd) {
         for (int i = level; i > 0; i--) {
            if (i == 1) {
               motherNode = m->GetTopNode();
            } else {
               motherNode = nextNode.GetNode(--level);
            }
            labelMother    = GetLabelOfVolume(motherNode->GetVolume());
            Int_t ndMother = motherNode->GetNdaughters();
            fLabel         = GetLabelOfVolume(currentNode->GetVolume());
            loc            = CalcLocation((*(currentNode->GetMatrix())));
            if ((XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->NbComponents(labelMother) < ndMother) && (!nd)) {
               AddChildLabel(labelMother, fLabel, loc);
            } else if ((XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->NbComponents(fLabel) == nd) && 
                       (XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->NbComponents(labelMother) == motherNode->GetVolume()->GetIndex(currentNode))) {
               AddChildLabel(labelMother, fLabel, loc);
            }
            currentNode = motherNode;
            fLabel      = labelMother;
            nd          = currentNode->GetNdaughters();
         }
      }
   }
}
    //______________________________________________________________________________

bool TOCCToStep::OCCPartialTreeCreation(TGeoManager * m, const char* node_name, int max_level)
{
   TGeoIterator nextNode(m->GetTopVolume());
   std::string  search_n         = node_name;
   bool         found_once       = false;
   bool         found_in_level_1 = false;
   auto         volume           = m->GetVolume(node_name);
   int          level1_skipped   = 0;
   TGeoNode*    currentNode      = 0;

   nextNode.SetType(0);
   while ((currentNode = nextNode())) {
     nextNode.SetType(0);
     int level = nextNode.GetLevel();
     if( level > max_level ){
       continue;
     }
     if(level == 1) {
       found_in_level_1 = false;
       if( volume == currentNode->GetVolume() ) {
         found_in_level_1 = true;
         found_once = true;
       }
     }
     if(!found_in_level_1) {
       if(level == 1) {
         level1_skipped++;
       }
       nextNode.SetType(1);
       continue;
     }
     FillOCCWithNode(m, currentNode, nextNode, level, max_level, level1_skipped);
   }
   return found_once;
}
    //______________________________________________________________________________

bool TOCCToStep::OCCPartialTreeCreation(TGeoManager * m, std::map<std::string,int> part_name_levels)
{
   bool         found_once       = false;
   bool         found_in_level_1 = false;
   int          level1_skipped   = 0;

   std::map<TGeoVolume*,std::string> part_name_vols;
   std::vector<TGeoVolume*>  vols;

   for(const auto& pl : part_name_levels) {
     TGeoVolume* avol     = m->GetVolume(pl.first.c_str());
     part_name_vols[avol] = pl.first;
     vols.push_back(avol);
   }

   TGeoIterator nextNode(m->GetTopVolume());
   TGeoNode*    currentNode      = nullptr;
   TGeoVolume*  matched_vol      = nullptr;

   nextNode.SetType(0);
   while ((currentNode = nextNode())) {
     nextNode.SetType(0);
     int level = nextNode.GetLevel();

     // Currently we only isolate level 1 node/volumes.
     // In the future this could be generalized.
     if(level == 1) {
       found_in_level_1 = false;
       for(auto v: vols) {
         if( v == currentNode->GetVolume() ) {
           // could there be more than one?
           matched_vol = v;
           found_in_level_1 = true;
           found_once = true;
         }
       }
     }
     if(!found_in_level_1) {
       if(level == 1) {
         level1_skipped++;
       }
       // switch the iterator type to go directly to sibling nodes  
       nextNode.SetType(1);
       continue;
     }
     int max_level = part_name_levels[ part_name_vols[matched_vol]];
     if( level > max_level  ){
       continue;
     }

     FillOCCWithNode(m, currentNode, nextNode, level, max_level, level1_skipped);
   }
   return found_once;
}
    //______________________________________________________________________________


void TOCCToStep::FillOCCWithNode(TGeoManager* m, TGeoNode* currentNode, TGeoIterator& nextNode, int level, int max_level, int level1_skipped)
{
  // This loop looks for nodes which are the end of line (ancestrally) then navigates
  // back up the family tree.  As it does so, the OCC tree is constructed. 
  // It is not clear why it must be done this way, but it could be an idiosyncracy
  // in OCC (which I am not too familar with at the moment).
  int nd = currentNode->GetNdaughters();
  if(level == max_level) {
    nd = 0;
  }
  if( nd == 0 ) {
    int level_start = std::min(level,max_level);
    for (int i = level_start; i > 0; i--) {
      TGeoNode* motherNode = 0;
      TDF_Label labelMother;
      TopLoc_Location loc;

      if (i == 1) {
        motherNode = m->GetTopNode();
      } else {
        motherNode = nextNode.GetNode(i-1);
      }
      labelMother    = GetLabelOfVolume(motherNode->GetVolume());
      Int_t ndMother = motherNode->GetNdaughters();
      // Why are we using a data member here?
      fLabel         = GetLabelOfVolume(currentNode->GetVolume());
      loc            = CalcLocation((*(currentNode->GetMatrix())));
      // Need to account for the missing daughters from those nodes skipped in level 1
      int skipped_this_level = 0; 
      if(i == 1 ) skipped_this_level = level1_skipped;
      if ((XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->NbComponents(labelMother) < ndMother) && (!nd)) {

        AddChildLabel(labelMother, fLabel, loc);
      } else if ((XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->NbComponents(fLabel) == currentNode->GetNdaughters()) && 
                 (XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->NbComponents(labelMother)+skipped_this_level  == motherNode->GetVolume()->GetIndex(currentNode))) {
        AddChildLabel(labelMother, fLabel, loc);
      }
      currentNode = motherNode;
      fLabel      = labelMother; // again, why a data member?
      nd          = currentNode->GetNdaughters();
    }
  }
}
////////////////////////////////////////////////////////////////////////////////

void TOCCToStep::PrintAssembly()
{
#if OCC_VERSION_MAJOR == 6 && OCC_VERSION_MINOR < 8
   XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->Dump();
#else
   XCAFDoc_DocumentTool::ShapeTool(fDoc->Main())->Dump(std::cout);
#endif
}


