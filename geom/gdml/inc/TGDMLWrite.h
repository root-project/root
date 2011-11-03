/*
 * TGDMLWrite.h
 *
 *  Created on: Sep 15, 2011
 *      Author: apytel
 */

#ifndef ROOT_TGDMLWRITE
#define ROOT_TGDMLWRITE

#ifndef ROOT_TGeoMatrix
#include "TGeoMatrix.h"
#endif

#ifndef ROOT_TXMLEngine
#include "TXMLEngine.h"
#endif

#ifndef ROOT_TGeoVolume
#include "TGeoVolume.h"
#endif

#ifndef ROOT_TGeoParaboloid
#include "TGeoParaboloid.h"
#endif

#ifndef ROOT_TGeoSphere
#include "TGeoSphere.h"
#endif

#ifndef ROOT_TGeoArb8
#include "TGeoArb8.h"
#endif

#ifndef ROOT_TGeoCone
#include "TGeoCone.h"
#endif

#ifndef ROOT_TGeoPara
#include "TGeoPara.h"
#endif

#ifndef ROOT_TGeoTrd1
#include "TGeoTrd1.h"
#endif

#ifndef ROOT_TGeoTrd2
#include "TGeoTrd2.h"
#endif

#ifndef ROOT_TGeoTube
#include "TGeoTube.h"
#endif

#ifndef ROOT_TGeoPcon
#include "TGeoPcon.h"
#endif

#ifndef ROOT_TGeoTorus
#include "TGeoTorus.h"
#endif

#ifndef ROOT_TGeoPgon
#include "TGeoPgon.h"
#endif

#ifndef ROOT_TGeoXtru
#include "TGeoXtru.h"
#endif

#ifndef ROOT_TGeoPgon
#include "TGeoPgon.h"
#endif

#ifndef ROOT_TGeoEltu
#include "TGeoEltu.h"
#endif

#ifndef ROOT_TGeoHype
#include "TGeoHype.h"
#endif

#ifndef ROOT_TGeoBoolNode
#include "TGeoBoolNode.h"
#endif

#ifndef ROOT_TGeoCompositeShape
#include "TGeoCompositeShape.h"
#endif

#include <map>
#include <vector>
#include <iostream>

class TGDMLWrite : public TObject {
public:
   TGDMLWrite();
   virtual ~TGDMLWrite();
   //wrapper of all main methods for extraction
   void WriteGDMLfile(TGeoManager * geomanager, const char* filename = "test.gdml", TString option = "");
   enum ENamingType {
      kelegantButSlow = 0,
      kwithoutSufixNotUniq = 1,
      kfastButUglySufix = 2
   };
   void SetNamingSpeed(ENamingType naming);
   void SetG4Compatibility(Bool_t G4Compatible) {
      fgG4Compatibility = G4Compatible;
   };

private:
   struct Xyz {
      Double_t x;
      Double_t y;
      Double_t z;
   };

   typedef  std::map<TString, Bool_t> NameList;
   typedef  std::map<TString, TString> NameListS;
   typedef  std::map<TString, Int_t> NameListI;
   typedef  std::map<TString, Float_t> NameListF;
   struct StructLst {
      NameList fLst;
   };     //to store pointers
   struct NameLst {
      NameListS fLst;        //to map pointers with names
      NameListI fLstIter;    //to store all the iterators for repeating names
   };

   //General lists
   StructLst *fIsotopeList;   //list of isotopes
   StructLst *fElementList;   //list of elements
   StructLst *fMaterialList;  //list of materials
   StructLst *fShapeList;     //list of solids
   StructLst *fVolumeList;    //list of volumes
   StructLst *fAccPatt;       //list of accepted patterns for division
   StructLst *fRejShape;      //list of rejected shapes

   NameLst *fNameList; //list of names (pointer mapped)

   //Data members
   static TGDMLWrite *fgGDMLWrite;                         //pointer to gdml writer
   Int_t  fgNamingSpeed;
   Bool_t fgG4Compatibility;
   XMLDocPointer_t  fGdmlFile;
   TString fTopVolumeName;
   TXMLEngine *fGdmlE;
   XMLNodePointer_t fDefineNode;                           //main <define> node...
   XMLNodePointer_t fMaterialsNode;                        //main <materials> node...
   XMLNodePointer_t fSolidsNode;                           //main <solids> node...
   XMLNodePointer_t fStructureNode;                        //main <structure> node...
   Int_t        fVolCnt;                                   //count of volumes
   Int_t        fPhysVolCnt;                               //count of physical volumes
   UInt_t       fActNameErr;                               //count of name errors
   UInt_t       fSolCnt;                                   //count of name solids

   static const UInt_t fgkProcBit    = BIT(14);    //14th bit is set when solid is processed
   static const UInt_t fgkProcBitVol = BIT(19);    //19th bit is set when volume is processed
   static const UInt_t fgkMaxNameErr = 5;          //maximum number of errors for naming

   //I. Methods returning pointer to the created node
   //1. Main methods to extract everything from ROOT gGeoManager


   XMLNodePointer_t ExtractMaterials(TList* materialsLst); //result <materials>...
   void             ExtractSolids(TObjArray* shapesLst);   //result <solids>...
   void             ExtractVolumes(TGeoVolume* volume);    //result <volume> node...


   //1.1 Materials sub methods - creating Nodes
   XMLNodePointer_t CreateAtomN(Double_t atom, const char * unit = "g/mole");
   XMLNodePointer_t CreateDN(Double_t density, const char * unit = "g/cm3");
   XMLNodePointer_t CreateFractionN(Double_t percentage, const char * refName);

   XMLNodePointer_t CreateIsotopN(TGeoIsotope * isotope, const char * name);
   XMLNodePointer_t CreateElementN(TGeoElement * element, XMLNodePointer_t materials, const char * name);
   XMLNodePointer_t CreateMixtureN(TGeoMixture * mixture, XMLNodePointer_t materials, TString mname);
   XMLNodePointer_t CreateMaterialN(TGeoMaterial * material, TString mname);


   //1.2 Solids sub methods
   XMLNodePointer_t ChooseObject(TGeoShape *geoShape);
   XMLNodePointer_t CreateZplaneN(Double_t z, Double_t rmin, Double_t rmax);

   XMLNodePointer_t CreateBoxN(TGeoBBox * geoShape);
   XMLNodePointer_t CreateParaboloidN(TGeoParaboloid * geoShape);
   XMLNodePointer_t CreateSphereN(TGeoSphere * geoShape);
   XMLNodePointer_t CreateArb8N(TGeoArb8 * geoShape);
   XMLNodePointer_t CreateConeN(TGeoConeSeg * geoShape);
   XMLNodePointer_t CreateConeN(TGeoCone * geoShape);
   XMLNodePointer_t CreateParaN(TGeoPara * geoShape);
   XMLNodePointer_t CreateTrapN(TGeoTrap * geoShape);
   XMLNodePointer_t CreateTwistedTrapN(TGeoGtra * geoShape);
   XMLNodePointer_t CreateTrdN(TGeoTrd1 * geoShape);
   XMLNodePointer_t CreateTrdN(TGeoTrd2 * geoShape);
   XMLNodePointer_t CreateTubeN(TGeoTubeSeg * geoShape);
   XMLNodePointer_t CreateCutTubeN(TGeoCtub * geoShape);
   XMLNodePointer_t CreateTubeN(TGeoTube * geoShape);
   XMLNodePointer_t CreatePolyconeN(TGeoPcon * geoShape);
   XMLNodePointer_t CreateTorusN(TGeoTorus * geoShape);
   XMLNodePointer_t CreatePolyhedraN(TGeoPgon * geoShape);
   XMLNodePointer_t CreateEltubeN(TGeoEltu * geoShape);
   XMLNodePointer_t CreateHypeN(TGeoHype * geoShape);
   XMLNodePointer_t CreateXtrusionN(TGeoXtru * geoShape);

   XMLNodePointer_t CreateCommonBoolN(TGeoCompositeShape *geoShape);

   //1.3 Volume sub methods
   XMLNodePointer_t CreatePhysVolN(const char * volref, const char * posref, const char * rotref, XMLNodePointer_t scaleN);
   XMLNodePointer_t CreateDivisionN(Double_t offset, Double_t width, Int_t number, const char * axis, const char * unit, const char * volref);

   XMLNodePointer_t CreateSetupN(const char * topVolName , const char * name = "default", const char * version = "1.0");
   XMLNodePointer_t StartVolumeN(const char * name, const char * solid, const char * material);
   XMLNodePointer_t StartAssemblyN(const char * name);


   //II. Utility methods
   Xyz GetXYZangles(const Double_t * rotationMatrix);
   //nodes to create position, rotation and similar types first-position/rotation...
   XMLNodePointer_t CreatePositionN(const char * name, const Double_t *position, const char * type = "position", const char * unit = "cm");
   XMLNodePointer_t CreateRotationN(const char * name, Xyz rotation, const char * type = "rotation", const char * unit = "deg");
   TGeoCompositeShape* CreateFakeCtub(TGeoCtub * geoShape);  //create fake cut tube as intersection

   //check name (2nd parameter) whether it is in the list (1st parameter)
   Bool_t IsInList(NameList list, TString name2check);
   TString GenName(TString oldname);
   TString GenName(TString oldname, TString objPointer);
   Bool_t CanProcess(TObject *pointer);
   TString GetPattAxis(Int_t divAxis, const char * pattName, TString& unit);

   ClassDef(TGDMLWrite, 0)    //imports GDML using DOM and binds it to ROOT
};

#endif /* ROOT_TGDMLWRITE */
