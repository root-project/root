#define HAVE_LIMITS_H
#define HAVE_IOSTREAM 
#define HAVE_IOMANIP

//////////////////////////////////////////////////////////////////////////////////
// OCCStep Class                                                                //
// --------------------                                                         //
//                                                                              //
// This class contains implementation of writing OpenCascade's                  //
// geometry shapes to the STEP file reproducing the originary ROOT geometry     //
// tree.  The TRootStep Class takes a gGeoManager pointer and gives             //
// back a STEP file.                                                            //
// The OCCShapeCreation(TGeoManager *m) method starting from                    //
// the top of the ROOT geometry tree translates each ROOT shape in the OCC one. //
// A fLabel is created for each OCC shape and the corrispondance bewteen the     //
// the fLabel and the shape is saved in a map.                                   //
// The OCCTreeCreation(TGeoManager *m) method starting from the top of the ROOT //
// geometry and using the fLabel-shape map reproduce the ROOT tree that will be  //
// written to the STEP file using the OCCWriteStep(const char * fname ) method.       //
//                                                                              //
//                                                                              //  
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////


#include "OCCStep.h"
#include <RootOCC.h>
//Root
#include <TGeoVolume.h>
#include <TString.h>
#include <TClass.h>
#include <TGeoManager.h>
//Cascade
#include <Interface_Static.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <TDataStd_Name.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <Standard.hxx>
#include <stdlib.h>
#include <XCAFApp_Application.hxx>

using namespace std;

OCCStep::OCCStep() 
{	
   OCCDocCreation();
}

void OCCStep::OCCDocCreation()
{	
   cout<<"creo documento"<<endl;
   Handle (XCAFApp_Application)A = XCAFApp_Application::GetApplication();
   if (!A.IsNull()) {
      A->NewDocument ("MDTV-XCAF",aDoc);
      cout<<"Document created"<<endl;
   }
   else
      cout<<"Error creating application"<<endl;
}

//logical fTree creation
TDF_Label OCCStep::OCCShapeCreation(TGeoManager *m)
{ 	
   
   TDF_Label motherLabel;
   TGeoVolume * currentVolume;
   TGeoVolume * motherVol;
   TGeoVolume * Top;
   TString type;
   TString path;
   Int_t num=0;
   Int_t level=0;
   out.open("/tmp/TGeoCad.log",ios::app);
   TIter next(m->GetListOfVolumes()); 
   fLabel=XCAFDoc_DocumentTool::ShapeTool(aDoc->Main())->NewShape();
   type=m->GetTopVolume()->GetShape()->IsA()->GetName();
   out<<"Translate top: "<<m->GetTopVolume()->GetName()<<endl;
   fShape=RootShape.OCC_SimpleShape(m->GetTopVolume()->GetShape());
   XCAFDoc_DocumentTool::ShapeTool(aDoc->Main())->SetShape(fLabel,fShape);
   TDataStd_Name::Set(fLabel,m->GetTopVolume()->GetName());
   XCAFDoc_DocumentTool::ShapeTool(aDoc->Main())->UpdateAssembly(aDoc->Main());
   Top=m->GetTopVolume();
   fTree[Top]=fLabel;
   while ((currentVolume=(TGeoVolume *)next())) {
      if (GetLabelOfVolume(currentVolume).IsNull()) {
         out<<"Translate : "<<currentVolume->GetName()<<endl;
         type=currentVolume->GetShape()->IsA()->GetName();
         num=currentVolume->GetNdaughters();
         if ((GetLabelOfVolume(currentVolume).IsNull())){
            if( type=="TGeoCompositeShape"){
               out<<"is a TGeoCompositeShape "<<endl;
               fShape=RootShape.OCC_CompositeShape((TGeoCompositeShape*)currentVolume->GetShape(), TGeoIdentity()); 
            } else {  
               out<<"is a simple Shape "<<endl;
               fShape = RootShape.OCC_SimpleShape(currentVolume->GetShape());
            }
         }
         TGeoNode * current;
         TGeoIterator nextNode(m->GetTopVolume());
         while ((current=nextNode())){
            if ((current->GetVolume()==currentVolume)&&(GetLabelOfVolume(current->GetVolume()).IsNull())) {
               level=nextNode.GetLevel();
               nextNode.GetPath(path);
               if (level==1)
                  motherVol=m->GetTopVolume();
               else {
                  TGeoNode * mother= nextNode.GetNode(--level);
                  //out<<"mother"<<mother->GetName()<<endl;
                  motherVol=mother->GetVolume();
               }
               motherLabel=GetLabelOfVolume(motherVol);
               if (!motherLabel.IsNull()) {
                  fLabel=TDF_TagSource::NewChild(motherLabel);
                  break;
               } else {
                  TGeoNode * grandMother= nextNode.GetNode(level);
                  //cout<<"nome mother not translated"<<grandMother->GetName()<<"level"<<level<<endl;
                  motherVol=grandMother->GetVolume();
                  TString type2=motherVol->GetShape()->IsA()->GetName();	
                  TopoDS_Shape Mothershape;
                  if( type2=="TGeoCompositeShape"){
                     Mothershape=RootShape.OCC_CompositeShape((TGeoCompositeShape*)motherVol->GetShape(), TGeoIdentity()); 
                  } else {
                     Mothershape = RootShape.OCC_SimpleShape(motherVol->GetShape());
                  }
                  motherLabel=TDF_TagSource::NewChild(GetLabelOfVolume(Top));
                  XCAFDoc_DocumentTool::ShapeTool(aDoc->Main())->SetShape(motherLabel,Mothershape);
                  TDataStd_Name::Set(motherLabel,motherVol->GetName());
                  XCAFDoc_DocumentTool::ShapeTool(aDoc->Main())->UpdateAssembly(aDoc->Main());
                  fTree[motherVol]=motherLabel;
                  fLabel=TDF_TagSource::NewChild(motherLabel);
                  break;
               }
            }  
         }
         if (fShape.IsNull()) out<<"the fShape is null"<<endl;	
         XCAFDoc_DocumentTool::ShapeTool(aDoc->Main())->SetShape(fLabel,fShape);
         TDataStd_Name::Set(fLabel,currentVolume->GetName());
         XCAFDoc_DocumentTool::ShapeTool(aDoc->Main())->UpdateAssembly(aDoc->Main());
         fTree[currentVolume]=fLabel;
         //cout<<"fTree["<<currentVolume->GetName()<<"]="<<fTree[currentVolume]<<endl;
      }
   }
   out.close();
   return fLabel;
}

void OCCStep::OCCWriteStep(const char *fname)
{	
   STEPControl_StepModelType mode = STEPControl_AsIs;
   fWriter.SetNameMode(Standard_True);
   if (!Interface_Static::SetIVal("write.step.assembly",1)) { //assembly mode
      cout<< "failed to set assembly mode for step data"<<endl;
   }
   if (!fWriter.Transfer(aDoc, mode)) {
      cerr<<"Error translating document" <<endl;
   }
   IFSelect_ReturnStatus stat = fWriter.Write(fname);	
}

TDF_Label OCCStep::GetLabelOfVolume(TGeoVolume * v)
{
   TDF_Label null;
   if(fTree.find(v)!=fTree.end())
      return fTree[v];
   else
      return null;
}

TGeoVolume * OCCStep::GetVolumeOfLabel(TDF_Label fLabel)
{
   map <TGeoVolume *,TDF_Label>::iterator it;
   for(it = fTree.begin(); it != fTree.end(); it++) 
      if (it->second.IsEqual(fLabel))
         return it->first;
}

void OCCStep::AddChildLabel(TDF_Label mother, TDF_Label child, TopLoc_Location loc)
{
   TDF_Label newL=XCAFDoc_DocumentTool::ShapeTool(mother)->AddComponent(mother, child,loc);
   XCAFDoc_DocumentTool::ShapeTool(mother)->UpdateAssembly(mother);
}

TopLoc_Location OCCStep::CalcLocation (TGeoHMatrix matrix)
{ 
   gp_Trsf TR,TR1;
   TopLoc_Location locA;
   Double_t const *t=matrix.GetTranslation();
   Double_t const *r=matrix.GetRotationMatrix();
   TR1.SetTranslation(gp_Vec(t[0],t[1],t[2]));
   TR.SetValues(r[0],r[1],r[2],0,
                r[3],r[4],r[5],0,
                r[6],r[7],r[8],0,
                0, 1);
   TR1.Multiply(TR);
   locA = TopLoc_Location (TR1);
   return locA;
}

void OCCStep::OCCTreeCreation(TGeoManager * m)
{
   TGeoIterator nextNode(m->GetTopVolume());
   TGeoNode *currentNode = 0;
   TGeoNode *motherNode = 0;
   TGeoNode *gmotherNode = 0;
   Int_t level;
   TDF_Label labelMother;
   TopLoc_Location loc;
   Int_t nd;
   out.open("/tmp/TGeoCad.log",ios::app);
   out<<"Start OCC Tree creation"<<endl;
   out.close();
   while ((currentNode=nextNode())){
      level = nextNode.GetLevel();
      nd = currentNode->GetNdaughters();
      //out <<"the current node is"<<currentNode->GetName()<<" ed ha "<<nd<<"figli"<<endl;	
      //out<<"level"<<level<<endl;
      if (!nd){ 
         for (int i=level; i>0;i--){ 
            if (i==1) {
               motherNode = m->GetTopNode();
            } else
               motherNode = nextNode.GetNode(--level);
            labelMother=GetLabelOfVolume(motherNode->GetVolume());
            Int_t ndMother = motherNode->GetNdaughters();
            //out<<"nd mother is"<<ndMother<<endl;
            fLabel = GetLabelOfVolume(currentNode->GetVolume());
            //out <<"il nodo corrente è"<<currentNode->GetName()<<"fLabel correte e'"<<fLabel<<"mother node is"<<motherNode->GetName()<<"fLabelmother e'"<<labelMother<<endl;
            loc=CalcLocation((*(currentNode->GetMatrix())));
            if ((XCAFDoc_DocumentTool::ShapeTool(aDoc->Main())->NbComponents(labelMother)<ndMother)&&(!nd)){ //out<<"primo if, nd vale "<<nd<<endl;
               //out<<"aggiungo--> "<<currentNode->GetName()<<"al mother volume-->"<<motherNode->GetName()<<endl;
               AddChildLabel(labelMother, fLabel,loc);
            }
            else if ((XCAFDoc_DocumentTool::ShapeTool(aDoc->Main())->NbComponents(fLabel)==nd)&&(XCAFDoc_DocumentTool::ShapeTool(aDoc->Main())->NbComponents(labelMother)==motherNode->GetVolume()->GetIndex(currentNode))){
               AddChildLabel(labelMother, fLabel,loc);
               //out<<"aDD--> "<<currentNode->GetName()<<"at mother volume-->"<<motherNode->GetName()<<endl;
            }
            //out<<"component label mother"<<XCAFDoc_DocumentTool::ShapeTool(aDoc->Main())->NbComponents(labelMother)<<"component volume"<<motherNode->GetVolume()->GetIndex(currentNode)<<endl;
            currentNode=motherNode;
            fLabel=labelMother;
            nd=currentNode->GetNdaughters();
         }
      }  
   }    
}


void OCCStep::PrintAssembly()
{
   XCAFDoc_DocumentTool::ShapeTool(aDoc->Main())->Dump();
}


