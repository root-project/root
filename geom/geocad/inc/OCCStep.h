#ifndef OCCStep_H
#define OCCStep_H 1

#include <TGeoNode.h>
#include <TGeoMatrix.h>
#include <TDF_Label.hxx>
#include <RootOCC.h>
#include <XCAFDoc_ShapeTool.hxx>
#include <TDocStd_Document.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <TDF_Label.hxx>
#include <TopoDS_Shape.hxx>


class OCCStep 
{
private:
   void OCCDocCreation();
   TopoDS_Shape AssemblyShape(TGeoVolume *vol, TGeoHMatrix m);
   TGeoVolume * GetVolumeOfLabel(TDF_Label fLabel);
   TDF_Label GetLabelOfVolume(TGeoVolume * v);
   void AddChildLabel(TDF_Label mother, TDF_Label child, TopLoc_Location loc);
   TopLoc_Location CalcLocation (TGeoHMatrix matrix);
   STEPCAFControl_Writer fWriter; //the step file pointer
   Handle(TDocStd_Document) aDoc; //the step document element
   ofstream out;
   typedef std::map < TGeoVolume *, TDF_Label> LabelMap_t;
   LabelMap_t fTree; //tree of Label's volumes
   TDF_Label fLabel; //label of the OCC shape elemet
   RootOCC RootShape;
   TopoDS_Shape fShape; //OCC shape (translated root shape)

public:
   OCCStep();
   void PrintAssembly();
   TDF_Label OCCShapeCreation(TGeoManager *m);
   void OCCTreeCreation(TGeoManager * m);
   void OCCWriteStep(const char *fname);
};
#endif
