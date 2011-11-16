// @(#)root/gdml:$Id$
// Author: Anton Pytel 15/9/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGeoManager.h"
#include "TGeoMaterial.h"
#include "TGeoMatrix.h"
#include "TXMLEngine.h"
#include "TGeoVolume.h"
#include "TGeoBBox.h"
#include "TGeoParaboloid.h"
#include "TGeoArb8.h"
#include "TGeoTube.h"
#include "TGeoCone.h"
#include "TGeoTrd1.h"
#include "TGeoTrd2.h"
#include "TGeoPcon.h"
#include "TGeoPgon.h"
#include "TGeoSphere.h"
#include "TGeoTorus.h"
#include "TGeoPara.h"
#include "TGeoHype.h"
#include "TGeoEltu.h"
#include "TGeoXtru.h"
#include "TGeoScaledShape.h"
#include "TGeoVolume.h"
#include "TROOT.h"
#include "TMath.h"
#include "TGeoMaterial.h"
#include "TGeoBoolNode.h"
#include "TGeoMedium.h"
#include "TGeoElement.h"
#include "TGeoShape.h"
#include "TGeoCompositeShape.h"
#include "TGDMLWrite.h"
#include <stdlib.h>
#include <string>
#include <map>
#include <ctime>

ClassImp(TGDMLWrite)

TGDMLWrite *TGDMLWrite::fgGDMLWrite = 0;

//______________________________________________________________________________
TGDMLWrite::TGDMLWrite()
   : TObject(),
     fIsotopeList(0),
     fElementList(0),
     fMaterialList(0),
     fShapeList(0),
     fVolumeList(0),
     fAccPatt(0),
     fRejShape(0),
     fNameList(0),
     fgNamingSpeed(0),
     fgG4Compatibility(0),
     fGdmlFile(0),
     fTopVolumeName(0),
     fGdmlE(0),
     fDefineNode(0),
     fMaterialsNode(0),
     fSolidsNode(0),
     fStructureNode(0),
     fVolCnt(0),
     fPhysVolCnt(0),
     fActNameErr(0),
     fSolCnt(0)
{
// Default constructor.
   if (fgGDMLWrite) delete fgGDMLWrite;
   fgGDMLWrite = this;
}

//______________________________________________________________________________
TGDMLWrite::~TGDMLWrite()
{
// Destructor.
   delete fIsotopeList;
   delete fElementList;
   delete fMaterialList;
   delete fShapeList;
   delete fVolumeList;
   delete fAccPatt;
   delete fRejShape;
   delete fNameList;
   fgGDMLWrite = 0;
}

//______________________________________________________________________________
void TGDMLWrite::SetNamingSpeed(ENamingType naming)
{
   fgNamingSpeed = naming;
}

//______________________________________________________________________________
void TGDMLWrite::WriteGDMLfile(TGeoManager * geomanager, const char* filename, TString option)
{
// Wrapper of all exporting methods
// Creates blank GDML file and fills it with gGeoManager structure converted
// to GDML structure xml nodes

   //option process
   option.ToLower();
   if (option.Contains("g")) {
      SetG4Compatibility(kTRUE);
      Info("WriteGDMLfile", "Geant4 compatibility mode set");
   } else {
      SetG4Compatibility(kFALSE);
   }
   if (option.Contains("f")) {
      SetNamingSpeed(kfastButUglySufix);
      Info("WriteGDMLfile", "Fast naming convetion with pointer suffix set");
   } else if (option.Contains("n")) {
      SetNamingSpeed(kwithoutSufixNotUniq);
      Info("WriteGDMLfile", "Naming without prefix set - be careful uniqness of name is not ensured");
   } else {
      SetNamingSpeed(kelegantButSlow);
      Info("WriteGDMLfile", "Potentially slow with incremental suffix naming convention set");
   }

   //local variables
   Int_t outputLayout = 1;
   const char * krootNodeName = "gdml";
   const char * knsRefGeneral = "http://www.w3.org/2001/XMLSchema-instance";
   const char * knsNameGeneral = "xsi";
   const char * knsRefGdml = "http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd";
   const char * knsNameGdml = "xsi:noNamespaceSchemaLocation";

   // First create engine
   fGdmlE = new TXMLEngine;
   fGdmlE->SetSkipComments(kTRUE);

   //create blank GDML file
   fGdmlFile = fGdmlE->NewDoc();

   //create root node and add it to blank GDML file
   XMLNodePointer_t rootNode = fGdmlE->NewChild(0, 0, krootNodeName, 0);
   fGdmlE->DocSetRootElement(fGdmlFile, rootNode);

   //add namespaces to root node
   fGdmlE->NewNS(rootNode, knsRefGeneral, knsNameGeneral);
   fGdmlE->NewAttr(rootNode, 0, knsNameGdml, knsRefGdml);

   //initialize general lists and <define>, <structure> nodes
   fIsotopeList  = new StructLst;
   fElementList  = new StructLst;
   fMaterialList = new StructLst;
   fShapeList    = new StructLst;
   fVolumeList   = new StructLst;

   fNameList     = new NameLst;

   fDefineNode = fGdmlE->NewChild(0, 0, "define", 0);
   fStructureNode = fGdmlE->NewChild(0, 0, "structure", 0);
   //========================

   //initialize list of accepted patterns for divisions (in ExtractVolumes)
   fAccPatt   = new StructLst;
   fAccPatt->fLst["TGeoPatternX"] = kTRUE;
   fAccPatt->fLst["TGeoPatternY"] = kTRUE;
   fAccPatt->fLst["TGeoPatternZ"] = kTRUE;
   fAccPatt->fLst["TGeoPatternCylR"] = kTRUE;
   fAccPatt->fLst["TGeoPatternCylPhi"] = kTRUE;
   //========================

   //initialize list of rejected shapes for divisions (in ExtractVolumes)
   fRejShape     = new StructLst;
   //this shapes are rejected because, it is not possible to divide trd2
   //in Y axis and while only trd2 object is imported from GDML
   //it causes a problem when TGeoTrd1 is divided in Y axis
   fRejShape->fLst["TGeoTrd1"] = kTRUE;
   fRejShape->fLst["TGeoTrd2"] = kTRUE;
   //=========================

   //Initialize global counters
   fActNameErr = 0;
   fVolCnt = 0;
   fPhysVolCnt = 0;
   fSolCnt = 0;
   fTopVolumeName = "";

   //calling main extraction functions (with measuring time)
   time_t startT, endT;
   startT = time(NULL);
   fMaterialsNode = ExtractMaterials(geomanager->GetListOfMaterials());
   ExtractSolids(geomanager->GetListOfShapes());
   Info("WriteGDMLfile", "Extracting volumes");
   if (geomanager->GetTopVolume()) {
      ExtractVolumes(geomanager->GetTopVolume());
   } else {
      Info("WriteGDMLfile", "Top volume does not exist!");
      return;
   }
   Info("WriteGDMLfile", "%i volumes added", fVolCnt);
   Info("WriteGDMLfile", "%i physvolumes added", fPhysVolCnt);
   endT = time(NULL);
   //<gdml>
   fGdmlE->AddChild(rootNode, fDefineNode);                 //  <define>...</define>
   fGdmlE->AddChild(rootNode, fMaterialsNode);              //  <materials>...</materials>
   fGdmlE->AddChild(rootNode, fSolidsNode);                 //  <solids>...</solids>
   fGdmlE->AddChild(rootNode, fStructureNode);              //  <structure>...</structure>
   fGdmlE->AddChild(rootNode, CreateSetupN(fTopVolumeName.Data())); //  <setup>...</setup>
   //</gdml>
   Double_t tdiffI = difftime(endT, startT);
   TString tdiffS = (tdiffI == 0 ? TString("< 1 s") : TString::Format("%.0lf s", tdiffI));
   Info("WriteGDMLfile", "Exporting time: %s", tdiffS.Data());
   //=========================

   //Saving document
   fGdmlE->SaveDoc(fGdmlFile, filename, outputLayout);
   Info("WriteGDMLfile", "File %s saved", filename);
}


//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::ExtractMaterials(TList* materialsLst)
{
   Info("ExtractMaterials", "Extracting materials");
   //crate main <materials> node
   XMLNodePointer_t materialsN = fGdmlE->NewChild(0, 0, "materials", 0);
   Int_t matcnt = 0;

   //go through materials  - iterator and object declaration
   TIter next(materialsLst);
   TGeoMaterial *lmaterial;

   while ((lmaterial = (TGeoMaterial *)next())) {
      //generate uniq name
      TString lname = GenName(lmaterial->GetName(), TString::Format("%p", lmaterial));

      if (lmaterial->IsMixture()) {
         TGeoMixture  *lmixture = (TGeoMixture *)lmaterial;
         XMLNodePointer_t mixtureN = CreateMixtureN(lmixture, materialsN, lname);
         fGdmlE->AddChild(materialsN, mixtureN);
      } else {
         XMLNodePointer_t materialN = CreateMaterialN(lmaterial, lname);
         fGdmlE->AddChild(materialsN, materialN);
      }
      matcnt++;
   }
   Info("ExtractMaterials", "%i materials added", matcnt);
   return materialsN;
}

//______________________________________________________________________________
void TGDMLWrite::ExtractSolids(TObjArray* shapesLst)
{
   Info("ExtractSolids", "Extracting solids (%i found).", shapesLst->GetEntries());

   //crate main <solids> node and define node for each solid
   fSolidsNode = fGdmlE->NewChild(0, 0, "solids", 0);
   XMLNodePointer_t solidN;

   //go through shapes - iterator and object declaration
   TIter next(shapesLst);
   TGeoShape *geoshp;

   while ((geoshp = (TGeoShape *)next())) {

      solidN = ChooseObject(geoshp);

      //add solid to solids node
      fGdmlE->AddChild(fSolidsNode, solidN);
      if (solidN != NULL) fSolCnt++;
   }
   Info("ExtractSolids", "%i solids added.", fSolCnt);

}

//______________________________________________________________________________
void TGDMLWrite::ExtractVolumes(TGeoVolume* volume)
{
   XMLNodePointer_t volumeN, childN;
   TString volname, matname, solname, pattClsName, nodeVolNameBak;
   TGeoPatternFinder *pattFinder = 0;
   Bool_t isPattern = kFALSE;

   //create the name for volume/assebmly
   if (volume->IsTopVolume()) {
      //not needed a special function for generating name
      volname = volume->GetName();
      fTopVolumeName = volname;
      //register name to the pointer
      fNameList->fLst[TString::Format("%p", volume)] = volname;
   } else {
      volname = GenName(volume->GetName(), TString::Format("%p", volume));
   }

   //start to create main volume/assembly node
   if (volume->IsAssembly()) {
      volumeN = StartAssemblyN(volname);
   } else {
      //get reference material and solid name
      matname = fNameList->fLst[TString::Format("%p", volume->GetMaterial())];
      solname = fNameList->fLst[TString::Format("%p", volume->GetShape())];
      volumeN = StartVolumeN(volname, solname, matname);


      //divisionvol can't be in assembly
      pattFinder = volume->GetFinder();
      //if found pattern
      if (pattFinder) {
         pattClsName = TString::Format("%s", pattFinder->ClassName());
         TString shapeCls = TString::Format("%s", volume->GetShape()->ClassName());
         //if pattern in accepted pattern list and not in shape rejected list
         if ((fAccPatt->fLst[pattClsName] == kTRUE) &&
             (fRejShape->fLst[shapeCls] != kTRUE)) {
            isPattern = kTRUE;
         }
      }
   }
   //get all nodes in volume
   TObjArray *nodeLst = volume->GetNodes();
   TIter next(nodeLst);
   TGeoNode *geoNode;
   Int_t nCnt = 0;
   //loop through all nodes
   while ((geoNode = (TGeoNode *) next())) {
      //get volume of current node and if not processed then process it
      TGeoVolume * subvol = geoNode->GetVolume();
      if (subvol->TestAttBit(fgkProcBitVol) == kFALSE) {
         subvol->SetAttBit(fgkProcBitVol);
         ExtractVolumes(subvol);
      }

      //volume of this node has to exist because it was processed recursively
      TString nodevolname = fNameList->fLst[TString::Format("%p", geoNode->GetVolume())];
      if (nCnt == 0) { //save name of the first node for divisionvol
         nodeVolNameBak = nodevolname;
      }

      if (isPattern == kFALSE) {
         //create name for node
         TString nodename, posname, rotname;
         nodename = GenName(geoNode->GetName(), TString::Format("%p", geoNode));
         nodename = nodename + "in" + volname;

         //create name for position and clear rotation
         posname = nodename + "pos";
         rotname = "";

         //position
         const Double_t * pos = geoNode->GetMatrix()->GetTranslation();
         childN = CreatePositionN(posname.Data(), pos);
         fGdmlE->AddChild(fDefineNode, childN); //adding node to <define> node

         //Deal with reflection
         XMLNodePointer_t scaleN = NULL;
         Double_t lx, ly, lz;
         Double_t xangle = 0;
         Double_t zangle = 0;
         lx = geoNode->GetMatrix()->GetRotationMatrix()[0];
         ly = geoNode->GetMatrix()->GetRotationMatrix()[4];
         lz = geoNode->GetMatrix()->GetRotationMatrix()[8];
         if (geoNode->GetMatrix()->IsReflection()
             && TMath::Abs(lx) == 1 &&  TMath::Abs(ly) == 1 && TMath::Abs(lz) == 1) {
            scaleN = fGdmlE->NewChild(0, 0, "scale", 0);
            fGdmlE->NewAttr(scaleN, 0, "name", (nodename+"scl").Data());
            fGdmlE->NewAttr(scaleN, 0, "x", TString::Format("%f", lx));
            fGdmlE->NewAttr(scaleN, 0, "y", TString::Format("%f", ly));
            fGdmlE->NewAttr(scaleN, 0, "z", TString::Format("%f", lz));
            //experimentally found out, that rotation should be ta
            if (lx == -1) {
               zangle = 180;
            }
            if (lz == -1) {
               xangle = 180;
            }
         }

         //rotation
         //geoNode->
         TGDMLWrite::Xyz lxyz = GetXYZangles(geoNode->GetMatrix()->GetRotationMatrix());
         lxyz.x -= xangle;
         lxyz.z -= zangle;
         if ((lxyz.x != 0.0) || (lxyz.y != 0.0) || (lxyz.z != 0.0)) {
            rotname = nodename + "rot";
            childN = CreateRotationN(rotname.Data(), lxyz);
            fGdmlE->AddChild(fDefineNode, childN); //adding node to <define> node
         }

         //create physvol for main volume/assembly node
         childN = CreatePhysVolN(nodevolname.Data(), posname.Data(), rotname.Data(), scaleN);
         fGdmlE->AddChild(volumeN, childN);
      }
      nCnt++;
   }
   //create only one divisionvol node
   if (isPattern && pattFinder) {
      //retrieve attributes of division
      Int_t ndiv, divaxis;
      Double_t offset, width, xlo, xhi;
      TString axis, unit;

      ndiv = pattFinder->GetNdiv();
      width = pattFinder->GetStep();

      divaxis = pattFinder->GetDivAxis();
      volume->GetShape()->GetAxisRange(divaxis, xlo, xhi);

      //compute relative start (not positional)
      offset = pattFinder->GetStart() - xlo;
      axis = GetPattAxis(divaxis, pattClsName, unit);

      //create division node
      childN = CreateDivisionN(offset, width, ndiv, axis.Data(), unit.Data(), nodeVolNameBak.Data());
      fGdmlE->AddChild(volumeN, childN);
   }

   fVolCnt++;
   //add volume/assembly node into the <structure> node
   fGdmlE->AddChild(fStructureNode, volumeN);

}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateAtomN(Double_t atom, const char * unit)
{
// Creates "atom" node for GDML

   XMLNodePointer_t atomN = fGdmlE->NewChild(0, 0, "atom", 0);
   fGdmlE->NewAttr(atomN, 0, "unit", unit);
   fGdmlE->NewAttr(atomN, 0, "value", TString::Format("%f", atom));
   return atomN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateDN(Double_t density, const char * unit)
{
// Creates "D" density node for GDML

   XMLNodePointer_t densN = fGdmlE->NewChild(0, 0, "D", 0);
   fGdmlE->NewAttr(densN, 0, "unit", unit);
   fGdmlE->NewAttr(densN, 0, "value", TString::Format("%f", density));
   return densN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateFractionN(Double_t percentage, const char * refName)
{
// Creates "fraction" node for GDML

   XMLNodePointer_t fractN = fGdmlE->NewChild(0, 0, "fraction", 0);
   fGdmlE->NewAttr(fractN, 0, "n", TString::Format("%f", percentage));
   fGdmlE->NewAttr(fractN, 0, "ref", refName);
   return fractN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateIsotopN(TGeoIsotope * isotope, const char * name)
{
// Creates "isotope" node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "isotope", 0);
   fGdmlE->NewAttr(mainN, 0, "name", name);
   fGdmlE->NewAttr(mainN, 0, "N", TString::Format("%i", isotope->GetN()));
   fGdmlE->NewAttr(mainN, 0, "Z", TString::Format("%i", isotope->GetZ()));
   fGdmlE->AddChild(mainN, CreateAtomN(isotope->GetA()));
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateElementN(TGeoElement * element, XMLNodePointer_t materials, const char * name)
{
// Creates "element" node for GDML
   //element node and attribute
   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "element", 0);
   fGdmlE->NewAttr(mainN, 0, "name", name);
   //local associative arrays for saving isotopes and their weight
   //inside element
   NameListF wPercentage;
   NameListI wCounter;

   if (element->HasIsotopes()) {
      Int_t nOfIso = element->GetNisotopes();
      //go through isotopes
      for (Int_t idx = 0; idx < nOfIso; idx++) {
         TGeoIsotope *myIsotope = element->GetIsotope(idx);

         //Get name of the Isotope (
         TString lname = myIsotope->GetName();
         //_iso suffix is added to to avoid problems with same names
         //for material, element and isotopes
         lname = TString::Format("%s_iso", lname.Data());

         //cumulates abudance, in case 2 isotopes with same names
         //within one element
         wPercentage[lname] += element->GetRelativeAbundance(idx);
         wCounter[lname]++;

         //check whether isotope name is not in list of isotopes
         if (IsInList(fIsotopeList->fLst, lname)) {
            continue;
         }
         //add isotope to list of isotopes and to main <materials> node
         fIsotopeList->fLst[lname] = kTRUE;
         XMLNodePointer_t isoNode = CreateIsotopN(myIsotope, lname);
         fGdmlE->AddChild(materials, isoNode);
      }
      //loop through asoc array of isotopes
      for (NameListI::iterator itr = wCounter.begin(); itr != wCounter.end(); itr++) {
         if (itr->second > 1) {
            Info("CreateMixtureN", "Warning: 2 equal isotopes in one element. Check: %s isotope of %s element",
                 itr->first.Data(), name);
         }
         //add fraction child to element with reference to isotope
         fGdmlE->AddChild(mainN, CreateFractionN(wPercentage[itr->first], itr->first.Data()));
      }
   } else {
      fGdmlE->NewAttr(mainN, 0, "formula", element->GetName());
      Int_t valZ = element->Z();
      // Z can't be <1 in Geant4 and Z is optional parameter
      if (valZ >= 1) {
         fGdmlE->NewAttr(mainN, 0, "Z", TString::Format("%i", valZ));
      }
      fGdmlE->AddChild(mainN, CreateAtomN(element->A()));
   }
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateMixtureN(TGeoMixture * mixture, XMLNodePointer_t materials, TString mname)
{
// Creates "material" node for GDML with references to other sub elements

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "material", 0);
   fGdmlE->NewAttr(mainN, 0, "name", mname);
   fGdmlE->AddChild(mainN, CreateDN(mixture->GetDensity()));
   //local associative arrays for saving elements and ther weight
   //inside mixture
   NameListF wPercentage;
   NameListI wCounter;

   Int_t nOfElm = mixture->GetNelements();
   //go through elements
   for (Int_t idx = 0; idx < nOfElm; idx++) {
      TGeoElement *myElement = mixture->GetElement(idx);

      //Get name of the element
      //NOTE: that for element - GetTitle() returns the "name" tag
      //and GetName() returns "formula" tag (see createElementN)
      TString lname = myElement->GetTitle();
      //_elm suffix is added to to avoid problems with same names
      //for material and element
      lname = TString::Format("%s_elm", lname.Data());

      //cumulates percentage, in case 2 elements with same names within one mixture
      wPercentage[lname] += mixture->GetWmixt()[idx];
      wCounter[lname]++;

      //check whether element name is not in list of elements already created
      if (IsInList(fElementList->fLst, lname)) {
         continue;
      }

      //add element to list of elements and to main <materials> node
      fElementList->fLst[lname] = kTRUE;
      XMLNodePointer_t elmNode = CreateElementN(myElement, materials, lname);
      fGdmlE->AddChild(materials, elmNode);
   }
   //loop through asoc array
   for (NameListI::iterator itr = wCounter.begin(); itr != wCounter.end(); itr++) {
      if (itr->second > 1) {
         Info("CreateMixtureN", "WARNING! 2 equal elements in one material. Check: %s element of %s material",
              itr->first.Data(), mname.Data());
      }
      //add fraction child to material with reference to element
      fGdmlE->AddChild(mainN, CreateFractionN(wPercentage[itr->first], itr->first.Data()));
   }

   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateMaterialN(TGeoMaterial * material, TString mname)
{
// Creates "material" node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "material", 0);
   fGdmlE->NewAttr(mainN, 0, "name", mname);
   Double_t valZ = material->GetZ();
   //Z can't be zero in Geant4 so this is workaround for vacuum
   TString tmpname = mname;
   tmpname.ToLower();
   if (valZ < 1) {
      if (tmpname == "vacuum") {
         valZ = 1;
      } else {
         Info("CreateMaterialN", "WARNING! value of Z can't be < 1 in %s material ", mname.Data());
      }
   }
   fGdmlE->NewAttr(mainN, 0, "Z", TString::Format("%f", valZ)); //material->GetZ()));
   fGdmlE->AddChild(mainN, CreateDN(material->GetDensity()));
   fGdmlE->AddChild(mainN, CreateAtomN(material->GetA()));
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateBoxN(TGeoBBox * geoShape)
{
// Creates "box" node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "box", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));
   fGdmlE->NewAttr(mainN, 0, "x", TString::Format("%f", 2 * geoShape->GetDX()));
   fGdmlE->NewAttr(mainN, 0, "y", TString::Format("%f", 2 * geoShape->GetDY()));
   fGdmlE->NewAttr(mainN, 0, "z", TString::Format("%f", 2 * geoShape->GetDZ()));

   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateParaboloidN(TGeoParaboloid * geoShape)
{
// Creates "paraboloid" node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "paraboloid", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));
   fGdmlE->NewAttr(mainN, 0, "rlo", TString::Format("%f", geoShape->GetRlo()));
   fGdmlE->NewAttr(mainN, 0, "rhi", TString::Format("%f", geoShape->GetRhi()));
   fGdmlE->NewAttr(mainN, 0, "dz", TString::Format("%f", geoShape->GetDz()));

   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateSphereN(TGeoSphere * geoShape)
{
// Creates "sphere" node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "sphere", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "rmin", TString::Format("%f", geoShape->GetRmin()));
   fGdmlE->NewAttr(mainN, 0, "rmax", TString::Format("%f", geoShape->GetRmax()));
   fGdmlE->NewAttr(mainN, 0, "startphi", TString::Format("%f", geoShape->GetPhi1()));
   fGdmlE->NewAttr(mainN, 0, "deltaphi", TString::Format("%f", geoShape->GetPhi2() - geoShape->GetPhi1()));
   fGdmlE->NewAttr(mainN, 0, "starttheta", TString::Format("%f", geoShape->GetTheta1()));
   fGdmlE->NewAttr(mainN, 0, "deltatheta", TString::Format("%f", geoShape->GetTheta2() - geoShape->GetTheta1()));

   fGdmlE->NewAttr(mainN, 0, "aunit", "deg");
   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateArb8N(TGeoArb8 * geoShape)
{
// Creates "arb8" node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "arb8", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "v1x", TString::Format("%f", geoShape->GetVertices()[0]));
   fGdmlE->NewAttr(mainN, 0, "v1y", TString::Format("%f", geoShape->GetVertices()[1]));
   fGdmlE->NewAttr(mainN, 0, "v2x", TString::Format("%f", geoShape->GetVertices()[2]));
   fGdmlE->NewAttr(mainN, 0, "v2y", TString::Format("%f", geoShape->GetVertices()[3]));
   fGdmlE->NewAttr(mainN, 0, "v3x", TString::Format("%f", geoShape->GetVertices()[4]));
   fGdmlE->NewAttr(mainN, 0, "v3y", TString::Format("%f", geoShape->GetVertices()[5]));
   fGdmlE->NewAttr(mainN, 0, "v4x", TString::Format("%f", geoShape->GetVertices()[6]));
   fGdmlE->NewAttr(mainN, 0, "v4y", TString::Format("%f", geoShape->GetVertices()[7]));
   fGdmlE->NewAttr(mainN, 0, "v5x", TString::Format("%f", geoShape->GetVertices()[8]));
   fGdmlE->NewAttr(mainN, 0, "v5y", TString::Format("%f", geoShape->GetVertices()[9]));
   fGdmlE->NewAttr(mainN, 0, "v6x", TString::Format("%f", geoShape->GetVertices()[10]));
   fGdmlE->NewAttr(mainN, 0, "v6y", TString::Format("%f", geoShape->GetVertices()[11]));
   fGdmlE->NewAttr(mainN, 0, "v7x", TString::Format("%f", geoShape->GetVertices()[12]));
   fGdmlE->NewAttr(mainN, 0, "v7y", TString::Format("%f", geoShape->GetVertices()[13]));
   fGdmlE->NewAttr(mainN, 0, "v8x", TString::Format("%f", geoShape->GetVertices()[14]));
   fGdmlE->NewAttr(mainN, 0, "v8y", TString::Format("%f", geoShape->GetVertices()[15]));
   fGdmlE->NewAttr(mainN, 0, "dz", TString::Format("%f", geoShape->GetDz()));

   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateConeN(TGeoConeSeg * geoShape)
{
// Creates "cone" node for GDML from TGeoConeSeg object

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "cone", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "z", TString::Format("%f", 2 * geoShape->GetDz()));
   fGdmlE->NewAttr(mainN, 0, "rmin1", TString::Format("%f", geoShape->GetRmin1()));
   fGdmlE->NewAttr(mainN, 0, "rmin2", TString::Format("%f", geoShape->GetRmin2()));
   fGdmlE->NewAttr(mainN, 0, "rmax1", TString::Format("%f", geoShape->GetRmax1()));
   fGdmlE->NewAttr(mainN, 0, "rmax2", TString::Format("%f", geoShape->GetRmax2()));
   fGdmlE->NewAttr(mainN, 0, "startphi", TString::Format("%f", geoShape->GetPhi1()));
   fGdmlE->NewAttr(mainN, 0, "deltaphi", TString::Format("%f", geoShape->GetPhi2() - geoShape->GetPhi1()));

   fGdmlE->NewAttr(mainN, 0, "aunit", "deg");
   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateConeN(TGeoCone * geoShape)
{
// Creates "cone" node for GDML from TGeoCone object

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "cone", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "z", TString::Format("%f", 2 * geoShape->GetDz()));
   fGdmlE->NewAttr(mainN, 0, "rmin1", TString::Format("%f", geoShape->GetRmin1()));
   fGdmlE->NewAttr(mainN, 0, "rmin2", TString::Format("%f", geoShape->GetRmin2()));
   fGdmlE->NewAttr(mainN, 0, "rmax1", TString::Format("%f", geoShape->GetRmax1()));
   fGdmlE->NewAttr(mainN, 0, "rmax2", TString::Format("%f", geoShape->GetRmax2()));
   fGdmlE->NewAttr(mainN, 0, "startphi", TString::Format("%i", 0));
   fGdmlE->NewAttr(mainN, 0, "deltaphi", TString::Format("%i", 360));

   fGdmlE->NewAttr(mainN, 0, "aunit", "deg");
   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateParaN(TGeoPara * geoShape)
{
// Creates "para" node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "para", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "x", TString::Format("%f", geoShape->GetX()));
   fGdmlE->NewAttr(mainN, 0, "y", TString::Format("%f", geoShape->GetY()));
   fGdmlE->NewAttr(mainN, 0, "z", TString::Format("%f", geoShape->GetZ()));
   fGdmlE->NewAttr(mainN, 0, "alpha", TString::Format("%f", geoShape->GetAlpha()));
   fGdmlE->NewAttr(mainN, 0, "theta", TString::Format("%f", geoShape->GetTheta()));
   fGdmlE->NewAttr(mainN, 0, "phi", TString::Format("%f", geoShape->GetPhi()));

   fGdmlE->NewAttr(mainN, 0, "aunit", "deg");
   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateTrapN(TGeoTrap * geoShape)
{
// Creates "trap" node for GDML

   XMLNodePointer_t mainN;

   //if one base equals 0 create Arb8 instead of trap
   if ((geoShape->GetBl1() == 0 || geoShape->GetTl1() == 0 || geoShape->GetH1() == 0) ||
       (geoShape->GetBl2() == 0 || geoShape->GetTl2() == 0 || geoShape->GetH2() == 0)) {
      mainN = CreateArb8N(geoShape);
      return mainN;
   }

   mainN = fGdmlE->NewChild(0, 0, "trap", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "z", TString::Format("%f", 2 * geoShape->GetDz()));
   fGdmlE->NewAttr(mainN, 0, "theta", TString::Format("%f", geoShape->GetTheta()));
   fGdmlE->NewAttr(mainN, 0, "phi", TString::Format("%f", geoShape->GetPhi()));
   fGdmlE->NewAttr(mainN, 0, "x1", TString::Format("%f", 2 * geoShape->GetBl1()));
   fGdmlE->NewAttr(mainN, 0, "x2", TString::Format("%f", 2 * geoShape->GetTl1()));
   fGdmlE->NewAttr(mainN, 0, "x3", TString::Format("%f", 2 * geoShape->GetBl2()));
   fGdmlE->NewAttr(mainN, 0, "x4", TString::Format("%f", 2 * geoShape->GetTl2()));
   fGdmlE->NewAttr(mainN, 0, "y1", TString::Format("%f", 2 * geoShape->GetH1()));
   fGdmlE->NewAttr(mainN, 0, "y2", TString::Format("%f", 2 * geoShape->GetH2()));

   fGdmlE->NewAttr(mainN, 0, "alpha1", TString::Format("%f", geoShape->GetAlpha1()));
   fGdmlE->NewAttr(mainN, 0, "alpha2", TString::Format("%f", geoShape->GetAlpha2()));

   fGdmlE->NewAttr(mainN, 0, "aunit", "deg");
   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateTwistedTrapN(TGeoGtra * geoShape)
{
// Creates "twistedtrap" node for GDML

   XMLNodePointer_t mainN;
   TString lname;

   //if one base equals 0 create Arb8 instead of twisted trap
   if ((geoShape->GetBl1() == 0 && geoShape->GetTl1() == 0 && geoShape->GetH1() == 0) ||
       (geoShape->GetBl2() == 0 && geoShape->GetTl2() == 0 && geoShape->GetH2() == 0)) {
      mainN = CreateArb8N((TGeoArb8 *) geoShape);
      return mainN;
   }

   //if parameter twistAngle (PhiTwist) equals zero create trap node
   if (geoShape->GetTwistAngle() == 0) {
      mainN = CreateTrapN((TGeoTrap *) geoShape);
      return mainN;
   }

   lname = GenName(geoShape->GetName(), TString::Format("%p", geoShape));
   mainN = fGdmlE->NewChild(0, 0, "twistedtrap", 0);
   fGdmlE->NewAttr(mainN, 0, "name", lname);


   fGdmlE->NewAttr(mainN, 0, "z", TString::Format("%f", 2 * geoShape->GetDz()));
   fGdmlE->NewAttr(mainN, 0, "Theta", TString::Format("%f", geoShape->GetTheta()));
   fGdmlE->NewAttr(mainN, 0, "Phi", TString::Format("%f", geoShape->GetPhi()));
   fGdmlE->NewAttr(mainN, 0, "x1", TString::Format("%f", 2 * geoShape->GetBl1()));
   fGdmlE->NewAttr(mainN, 0, "x2", TString::Format("%f", 2 * geoShape->GetTl1()));
   fGdmlE->NewAttr(mainN, 0, "x3", TString::Format("%f", 2 * geoShape->GetBl2()));
   fGdmlE->NewAttr(mainN, 0, "x4", TString::Format("%f", 2 * geoShape->GetTl2()));
   fGdmlE->NewAttr(mainN, 0, "y1", TString::Format("%f", 2 * geoShape->GetH1()));
   fGdmlE->NewAttr(mainN, 0, "y2", TString::Format("%f", 2 * geoShape->GetH2()));

   fGdmlE->NewAttr(mainN, 0, "Alph", TString::Format("%f", geoShape->GetAlpha1()));

   //check if alpha1 equals to alpha2 (converting to string - to avoid problems with floats)
   if (TString::Format("%f", geoShape->GetAlpha1()) != TString::Format("%f", geoShape->GetAlpha2())) {
      Info("CreateTwistedTrapN",
           "ERROR! Object %s is not exported correctly because parameter Alpha2 is not declared in GDML schema",
           lname.Data());
   }
   //fGdmlE->NewAttr(mainN,0, "alpha2", TString::Format("%f", geoShape->GetAlpha2()));
   fGdmlE->NewAttr(mainN, 0, "PhiTwist", TString::Format("%f", geoShape->GetTwistAngle()));

   fGdmlE->NewAttr(mainN, 0, "aunit", "deg");
   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateTrdN(TGeoTrd1 * geoShape)
{
// Creates "trd" node for GDML from object TGeoTrd1

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "trd", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "x1", TString::Format("%f", 2 * geoShape->GetDx1()));
   fGdmlE->NewAttr(mainN, 0, "x2", TString::Format("%f", 2 * geoShape->GetDx2()));
   fGdmlE->NewAttr(mainN, 0, "y1", TString::Format("%f", 2 * geoShape->GetDy()));
   fGdmlE->NewAttr(mainN, 0, "y2", TString::Format("%f", 2 * geoShape->GetDy()));
   fGdmlE->NewAttr(mainN, 0, "z", TString::Format("%f", 2 * geoShape->GetDz()));

   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateTrdN(TGeoTrd2 * geoShape)
{
// Creates "trd" node for GDML from object TGeoTrd2

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "trd", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "x1", TString::Format("%f", 2 * geoShape->GetDx1()));
   fGdmlE->NewAttr(mainN, 0, "x2", TString::Format("%f", 2 * geoShape->GetDx2()));
   fGdmlE->NewAttr(mainN, 0, "y1", TString::Format("%f", 2 * geoShape->GetDy1()));
   fGdmlE->NewAttr(mainN, 0, "y2", TString::Format("%f", 2 * geoShape->GetDy2()));
   fGdmlE->NewAttr(mainN, 0, "z", TString::Format("%f", 2 * geoShape->GetDz()));

   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateTubeN(TGeoTubeSeg * geoShape)
{
// Creates "tube" node for GDML  from  object TGeoTubeSeg

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "tube", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "rmin", TString::Format("%f", geoShape->GetRmin()));
   fGdmlE->NewAttr(mainN, 0, "rmax", TString::Format("%f", geoShape->GetRmax()));
   fGdmlE->NewAttr(mainN, 0, "z", TString::Format("%f", 2 * geoShape->GetDz()));
   fGdmlE->NewAttr(mainN, 0, "startphi", TString::Format("%f", geoShape->GetPhi1()));
   fGdmlE->NewAttr(mainN, 0, "deltaphi", TString::Format("%f", geoShape->GetPhi2() - geoShape->GetPhi1()));

   fGdmlE->NewAttr(mainN, 0, "aunit", "deg");
   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateCutTubeN(TGeoCtub * geoShape)
{
// Creates "cutTube" node for GDML

   XMLNodePointer_t mainN;

   if (fgG4Compatibility == kTRUE) {
      TGeoShape * fakeCtub = CreateFakeCtub(geoShape);
      mainN = ChooseObject(fakeCtub);

      //register name for cuttube shape (so it will be found during volume export)
      TString lname = fNameList->fLst[TString::Format("%p", fakeCtub)];
      fNameList->fLst[TString::Format("%p", geoShape)] = lname;
      Info("CreateCutTubeN", "WARNING! %s - CutTube was replaced by intersection of TGeoTubSeg and two TGeoBBoxes",
           lname.Data());
   } else {
      Info("CreateCutTubeN", "WARNING! CutTube is not recognized by Geant4");
      mainN = fGdmlE->NewChild(0, 0, "cutTube", 0);
      fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

      fGdmlE->NewAttr(mainN, 0, "rmin", TString::Format("%f", geoShape->GetRmin()));
      fGdmlE->NewAttr(mainN, 0, "rmax", TString::Format("%f", geoShape->GetRmax()));
      fGdmlE->NewAttr(mainN, 0, "z", TString::Format("%f", 2 * geoShape->GetDz()));
      fGdmlE->NewAttr(mainN, 0, "startphi", TString::Format("%f", geoShape->GetPhi1()));
      fGdmlE->NewAttr(mainN, 0, "deltaphi", TString::Format("%f", geoShape->GetPhi2() - geoShape->GetPhi1()));
      fGdmlE->NewAttr(mainN, 0, "lowX", TString::Format("%f", geoShape->GetNlow()[0]));
      fGdmlE->NewAttr(mainN, 0, "lowY", TString::Format("%f", geoShape->GetNlow()[1]));
      fGdmlE->NewAttr(mainN, 0, "lowZ", TString::Format("%f", geoShape->GetNlow()[2]));
      fGdmlE->NewAttr(mainN, 0, "highX", TString::Format("%f", geoShape->GetNhigh()[0]));
      fGdmlE->NewAttr(mainN, 0, "highY", TString::Format("%f", geoShape->GetNhigh()[1]));
      fGdmlE->NewAttr(mainN, 0, "highZ", TString::Format("%f", geoShape->GetNhigh()[2]));

      fGdmlE->NewAttr(mainN, 0, "aunit", "deg");
      fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   }
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateTubeN(TGeoTube * geoShape)
{
// Creates "tube" node for GDML from  object TGeoTube

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "tube", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "rmin", TString::Format("%f", geoShape->GetRmin()));
   fGdmlE->NewAttr(mainN, 0, "rmax", TString::Format("%f", geoShape->GetRmax()));
   fGdmlE->NewAttr(mainN, 0, "z", TString::Format("%f", 2 * geoShape->GetDz()));
   fGdmlE->NewAttr(mainN, 0, "startphi", TString::Format("%i", 0));
   fGdmlE->NewAttr(mainN, 0, "deltaphi", TString::Format("%i", 360));

   fGdmlE->NewAttr(mainN, 0, "aunit", "deg");
   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateZplaneN(Double_t z, Double_t rmin, Double_t rmax)
{
// Creates "zplane" node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "zplane", 0);

   fGdmlE->NewAttr(mainN, 0, "z", TString::Format("%f", z));
   fGdmlE->NewAttr(mainN, 0, "rmin", TString::Format("%f", rmin));
   fGdmlE->NewAttr(mainN, 0, "rmax", TString::Format("%f", rmax));

   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreatePolyconeN(TGeoPcon * geoShape)
{
// Creates "polycone" node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "polycone", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "startphi", TString::Format("%f", geoShape->GetPhi1()));
   fGdmlE->NewAttr(mainN, 0, "deltaphi", TString::Format("%f", geoShape->GetDphi()));

   fGdmlE->NewAttr(mainN, 0, "aunit", "deg");
   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   for (Int_t it = 0; it < geoShape->GetNz(); it++) {
      //add zplane child node
      fGdmlE->AddChild(mainN, CreateZplaneN(geoShape->GetZ(it), geoShape->GetRmin(it), geoShape->GetRmax(it)));
   }
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateTorusN(TGeoTorus * geoShape)
{
// Creates "torus" node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "torus", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "rtor", TString::Format("%f", geoShape->GetR()));
   fGdmlE->NewAttr(mainN, 0, "rmin", TString::Format("%f", geoShape->GetRmin()));
   fGdmlE->NewAttr(mainN, 0, "rmax", TString::Format("%f", geoShape->GetRmax()));
   fGdmlE->NewAttr(mainN, 0, "startphi", TString::Format("%f", geoShape->GetPhi1()));
   fGdmlE->NewAttr(mainN, 0, "deltaphi", TString::Format("%f", geoShape->GetDphi()));

   fGdmlE->NewAttr(mainN, 0, "aunit", "deg");
   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");

   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreatePolyhedraN(TGeoPgon * geoShape)
{
// Creates "polyhedra" node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "polyhedra", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "startphi", TString::Format("%f", geoShape->GetPhi1()));
   fGdmlE->NewAttr(mainN, 0, "deltaphi", TString::Format("%f", geoShape->GetDphi()));
   fGdmlE->NewAttr(mainN, 0, "numsides", TString::Format("%i", geoShape->GetNedges()));

   fGdmlE->NewAttr(mainN, 0, "aunit", "deg");
   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   for (Int_t it = 0; it < geoShape->GetNz(); it++) {
      //add zplane child node
      fGdmlE->AddChild(mainN, CreateZplaneN(geoShape->GetZ(it), geoShape->GetRmin(it), geoShape->GetRmax(it)));
   }
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateEltubeN(TGeoEltu * geoShape)
{
// Creates "eltube" node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "eltube", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "dx", TString::Format("%f", geoShape->GetA()));
   fGdmlE->NewAttr(mainN, 0, "dy", TString::Format("%f", geoShape->GetB()));
   fGdmlE->NewAttr(mainN, 0, "dz", TString::Format("%f", geoShape->GetDz()));

   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");

   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateHypeN(TGeoHype * geoShape)
{
// Creates "hype" node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "hype", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "rmin", TString::Format("%f", geoShape->GetRmin()));
   fGdmlE->NewAttr(mainN, 0, "rmax", TString::Format("%f", geoShape->GetRmax()));
   fGdmlE->NewAttr(mainN, 0, "inst", TString::Format("%f", geoShape->GetStIn()));
   fGdmlE->NewAttr(mainN, 0, "outst", TString::Format("%f", geoShape->GetStOut()));
   fGdmlE->NewAttr(mainN, 0, "z", TString::Format("%f", 2 * geoShape->GetDz()));

   fGdmlE->NewAttr(mainN, 0, "aunit", "deg");
   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");

   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateXtrusionN(TGeoXtru * geoShape)
{
// Creates "xtru" node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "xtru", 0);
   fGdmlE->NewAttr(mainN, 0, "name", GenName(geoShape->GetName(), TString::Format("%p", geoShape)));

   fGdmlE->NewAttr(mainN, 0, "lunit", "cm");
   XMLNodePointer_t childN;
   for (Int_t it = 0; it < geoShape->GetNvert(); it++) {
      //add twoDimVertex child node
      childN = fGdmlE->NewChild(0, 0, "twoDimVertex", 0);
      fGdmlE->NewAttr(childN, 0, "x", TString::Format("%f", geoShape->GetX(it)));
      fGdmlE->NewAttr(childN, 0, "y", TString::Format("%f", geoShape->GetY(it)));
      fGdmlE->AddChild(mainN, childN);
   }
   for (Int_t it = 0; it < geoShape->GetNz(); it++) {
      //add section child node
      childN = fGdmlE->NewChild(0, 0, "section", 0);
      fGdmlE->NewAttr(childN, 0, "zOrder", TString::Format("%i", it));
      fGdmlE->NewAttr(childN, 0, "zPosition", TString::Format("%f", geoShape->GetZ(it)));
      fGdmlE->NewAttr(childN, 0, "xOffset", TString::Format("%f", geoShape->GetXOffset(it)));
      fGdmlE->NewAttr(childN, 0, "yOffset", TString::Format("%f", geoShape->GetYOffset(it)));
      fGdmlE->NewAttr(childN, 0, "scalingFactor", TString::Format("%f", geoShape->GetScale(it)));
      fGdmlE->AddChild(mainN, childN);
   }
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateCommonBoolN(TGeoCompositeShape *geoShape)
{
// Creates common part of union intersection and subtraction nodes
   XMLNodePointer_t mainN, ndR, ndL, childN;
   TString nodeName = GenName(geoShape->GetName(), TString::Format("%p", geoShape));
   TString lboolType;
   TGeoBoolNode::EGeoBoolType boolType = geoShape->GetBoolNode()->GetBooleanOperator();
   switch (boolType) {
      case TGeoBoolNode::kGeoUnion:
         lboolType = "union";
         break;
      case TGeoBoolNode::kGeoSubtraction:
         lboolType = "subtraction";
         break;
      case TGeoBoolNode::kGeoIntersection:
         lboolType = "intersection";
         break;
   }

   TGDMLWrite::Xyz lrot = GetXYZangles(geoShape->GetBoolNode()->GetLeftMatrix()->Inverse().GetRotationMatrix());
   const Double_t  *ltr = geoShape->GetBoolNode()->GetLeftMatrix()->GetTranslation();
   TGDMLWrite::Xyz rrot = GetXYZangles(geoShape->GetBoolNode()->GetRightMatrix()->Inverse().GetRotationMatrix());
   const Double_t  *rtr = geoShape->GetBoolNode()->GetRightMatrix()->GetTranslation();

   //left and right nodes are created here also their names are created
   ndL = ChooseObject(geoShape->GetBoolNode()->GetLeftShape());
   ndR = ChooseObject(geoShape->GetBoolNode()->GetRightShape());

   //left and right nodes appended to main structure of nodes (if they are not already there)
   if (ndL != NULL) {
      fGdmlE->AddChild(fSolidsNode, ndL);
      fSolCnt++;
   }
   if (ndR != NULL) {
      fGdmlE->AddChild(fSolidsNode, ndR);
      fSolCnt++;
   }

   //retrieve left and right node names by their pointer to make reference
   TString lname = fNameList->fLst[TString::Format("%p", geoShape->GetBoolNode()->GetLeftShape())];
   TString rname = fNameList->fLst[TString::Format("%p", geoShape->GetBoolNode()->GetRightShape())];

   //create union node and its child nodes (or intersection or subtraction)
   /* <union name="...">
    *   <first ref="left name" />
    *   <second ref="right name" />
    *   <firstposition .../>
    *   <firstrotation .../>
    *   <position .../>
    *   <rotation .../>
    * </union>
   */
   mainN = fGdmlE->NewChild(0, 0, lboolType.Data(), 0);
   fGdmlE->NewAttr(mainN, 0, "name", nodeName);

   //<first> (left)
   childN = fGdmlE->NewChild(0, 0, "first", 0);
   fGdmlE->NewAttr(childN, 0, "ref", lname);
   fGdmlE->AddChild(mainN, childN);

   //<second> (right)
   childN = fGdmlE->NewChild(0, 0, "second", 0);
   fGdmlE->NewAttr(childN, 0, "ref", rname);
   fGdmlE->AddChild(mainN, childN);

   //<firstposition> (left)
   if ((ltr[0] != 0.0) || (ltr[1] != 0.0) || (ltr[2] != 0.0)) {
      childN = CreatePositionN((nodeName + lname + "pos").Data(), ltr, "firstposition");
      fGdmlE->AddChild(mainN, childN);
   }
   //<firstrotation> (left)
   if ((lrot.x != 0.0) || (lrot.y != 0.0) || (lrot.z != 0.0)) {
      childN = CreateRotationN((nodeName + lname + "rot").Data(), lrot, "firstrotation");
      fGdmlE->AddChild(mainN, childN);
   }
   //<position> (right)
   if ((rtr[0] != 0.0) || (rtr[1] != 0.0) || (rtr[2] != 0.0)) {
      childN = CreatePositionN((nodeName + rname + "pos").Data(), rtr, "position");
      fGdmlE->AddChild(mainN, childN);
   }
   //<rotation> (right)
   if ((rrot.x != 0.0) || (rrot.y != 0.0) || (rrot.z != 0.0)) {
      childN = CreateRotationN((nodeName + rname + "rot").Data(), rrot, "rotation");
      fGdmlE->AddChild(mainN, childN);
   }

   return mainN;
}


//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreatePositionN(const char * name, const Double_t *position, const char * type, const char * unit)
{
// Creates "position" kind of node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, type, 0);
   fGdmlE->NewAttr(mainN, 0, "name", name);
   fGdmlE->NewAttr(mainN, 0, "x", TString::Format("%f", position[0]));
   fGdmlE->NewAttr(mainN, 0, "y", TString::Format("%f", position[1]));
   fGdmlE->NewAttr(mainN, 0, "z", TString::Format("%f", position[2]));
   fGdmlE->NewAttr(mainN, 0, "unit", unit);
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateRotationN(const char * name, Xyz rotation, const char * type, const char * unit)
{
// Creates "rotation" kind of node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, type, 0);
   fGdmlE->NewAttr(mainN, 0, "name", name);
   fGdmlE->NewAttr(mainN, 0, "x", TString::Format("%f", rotation.x));
   fGdmlE->NewAttr(mainN, 0, "y", TString::Format("%f", rotation.y));
   fGdmlE->NewAttr(mainN, 0, "z", TString::Format("%f", rotation.z));
   fGdmlE->NewAttr(mainN, 0, "unit", unit);
   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateSetupN(const char * topVolName, const char * name, const char * version)
{
// Creates "setup" node for GDML

   XMLNodePointer_t setupN = fGdmlE->NewChild(0, 0, "setup", 0);
   fGdmlE->NewAttr(setupN, 0, "name", name);
   fGdmlE->NewAttr(setupN, 0, "version", version);
   XMLNodePointer_t fworldN = fGdmlE->NewChild(setupN, 0, "world", 0);
   fGdmlE->NewAttr(fworldN, 0, "ref", topVolName);
   return setupN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::StartVolumeN(const char * name, const char * solid, const char * material)
{
// Creates "volume" node for GDML

   XMLNodePointer_t childN;
   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "volume", 0);
   fGdmlE->NewAttr(mainN, 0, "name", name);

   childN = fGdmlE->NewChild(0, 0, "materialref", 0);
   fGdmlE->NewAttr(childN, 0, "ref", material);
   fGdmlE->AddChild(mainN, childN);

   childN = fGdmlE->NewChild(0, 0, "solidref", 0);
   fGdmlE->NewAttr(childN, 0, "ref", solid);
   fGdmlE->AddChild(mainN, childN);

   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::StartAssemblyN(const char * name)
{
// Creates "assembly" node for GDML

   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "assembly", 0);
   fGdmlE->NewAttr(mainN, 0, "name", name);

   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreatePhysVolN(const char * volref, const char * posref, const char * rotref, XMLNodePointer_t scaleN)
{
// Creates "physvol" node for GDML
   fPhysVolCnt++;
   XMLNodePointer_t childN;
   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "physvol", 0);

   childN = fGdmlE->NewChild(0, 0, "volumeref", 0);
   fGdmlE->NewAttr(childN, 0, "ref", volref);
   fGdmlE->AddChild(mainN, childN);

   childN = fGdmlE->NewChild(0, 0, "positionref", 0);
   fGdmlE->NewAttr(childN, 0, "ref", posref);
   fGdmlE->AddChild(mainN, childN);

   //if is not empty string add this node
   if (strcmp(rotref, "") != 0) {
      childN = fGdmlE->NewChild(0, 0, "rotationref", 0);
      fGdmlE->NewAttr(childN, 0, "ref", rotref);
      fGdmlE->AddChild(mainN, childN);
   }
   if (scaleN != NULL) {
      fGdmlE->AddChild(mainN, scaleN);
   }

   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::CreateDivisionN(Double_t offset, Double_t width, Int_t number, const char * axis, const char * unit, const char * volref)
{
// Creates "divisionvol" node for GDML

   XMLNodePointer_t childN;
   XMLNodePointer_t mainN = fGdmlE->NewChild(0, 0, "divisionvol", 0);
   fGdmlE->NewAttr(mainN, 0, "axis", axis);
   fGdmlE->NewAttr(mainN, 0, "number", TString::Format("%i", number));
   fGdmlE->NewAttr(mainN, 0, "width", TString::Format("%f", width));
   fGdmlE->NewAttr(mainN, 0, "offset", TString::Format("%f", offset));
   fGdmlE->NewAttr(mainN, 0, "unit", unit);

   childN = fGdmlE->NewChild(0, 0, "volumeref", 0);
   fGdmlE->NewAttr(childN, 0, "ref", volref);
   fGdmlE->AddChild(mainN, childN);


   return mainN;
}

//______________________________________________________________________________
XMLNodePointer_t TGDMLWrite::ChooseObject(TGeoShape *geoShape)
{
// Chooses the object and method that should be used for processing
   const char * clsname = geoShape->ClassName();
   XMLNodePointer_t solidN;

   if (CanProcess((TObject *)geoShape) == kFALSE)
      return NULL;


   //process different shapes
   if (strcmp(clsname, "TGeoBBox") == 0) {
      solidN = CreateBoxN((TGeoBBox*) geoShape);
   } else if (strcmp(clsname, "TGeoParaboloid") == 0) {
      solidN = CreateParaboloidN((TGeoParaboloid*) geoShape);
   } else if (strcmp(clsname, "TGeoSphere") == 0) {
      solidN = CreateSphereN((TGeoSphere*) geoShape);
   } else if (strcmp(clsname, "TGeoArb8") == 0) {
      solidN = CreateArb8N((TGeoArb8*) geoShape);
   } else if (strcmp(clsname, "TGeoConeSeg") == 0) {
      solidN = CreateConeN((TGeoConeSeg*) geoShape);
   } else if (strcmp(clsname, "TGeoCone") == 0) {
      solidN = CreateConeN((TGeoCone*) geoShape);
   } else if (strcmp(clsname, "TGeoPara") == 0) {
      solidN = CreateParaN((TGeoPara*) geoShape);
   } else if (strcmp(clsname, "TGeoTrap") == 0) {
      solidN = CreateTrapN((TGeoTrap*) geoShape);
   } else if (strcmp(clsname, "TGeoGtra") == 0) {
      solidN = CreateTwistedTrapN((TGeoGtra*) geoShape);
   } else if (strcmp(clsname, "TGeoTrd1") == 0) {
      solidN = CreateTrdN((TGeoTrd1*) geoShape);
   } else if (strcmp(clsname, "TGeoTrd2") == 0) {
      solidN = CreateTrdN((TGeoTrd2*) geoShape);
   } else if (strcmp(clsname, "TGeoTubeSeg") == 0) {
      solidN = CreateTubeN((TGeoTubeSeg*) geoShape);
   } else if (strcmp(clsname, "TGeoCtub") == 0) {
      solidN = CreateCutTubeN((TGeoCtub*) geoShape);
   } else if (strcmp(clsname, "TGeoTube") == 0) {
      solidN = CreateTubeN((TGeoTube*) geoShape);
   } else if (strcmp(clsname, "TGeoPcon") == 0) {
      solidN = CreatePolyconeN((TGeoPcon*) geoShape);
   } else if (strcmp(clsname, "TGeoTorus") == 0) {
      solidN = CreateTorusN((TGeoTorus*) geoShape);
   } else if (strcmp(clsname, "TGeoPgon") == 0) {
      solidN = CreatePolyhedraN((TGeoPgon*) geoShape);
   } else if (strcmp(clsname, "TGeoEltu") == 0) {
      solidN = CreateEltubeN((TGeoEltu*) geoShape);
   } else if (strcmp(clsname, "TGeoHype") == 0) {
      solidN = CreateHypeN((TGeoHype*) geoShape);
   } else if (strcmp(clsname, "TGeoXtru") == 0) {
      solidN = CreateXtrusionN((TGeoXtru*) geoShape);
   } else if (strcmp(clsname, "TGeoScaledShape") == 0) {
      TGeoScaledShape * geoscale = (TGeoScaledShape *) geoShape;
      Info("ChooseObject",
           "ERROR! TGeoScaledShape object is not possible to process correctly. %s object is processed without scale",
           geoscale->GetShape()->ClassName());
      solidN = ChooseObject(geoscale->GetShape());
      //Name has to be propagated to geoscale level pointer
      fNameList->fLst[TString::Format("%p", geoscale)] =
         fNameList->fLst[TString::Format("%p", geoscale->GetShape())];
   } else if (strcmp(clsname, "TGeoCompositeShape") == 0) {
      solidN = CreateCommonBoolN((TGeoCompositeShape*) geoShape);
   } else if (strcmp(clsname, "TGeoUnion") == 0) {
      solidN = CreateCommonBoolN((TGeoCompositeShape*) geoShape);
   } else if (strcmp(clsname, "TGeoIntersection") == 0) {
      solidN = CreateCommonBoolN((TGeoCompositeShape*) geoShape);
   } else if (strcmp(clsname, "TGeoSubtraction") == 0) {
      solidN = CreateCommonBoolN((TGeoCompositeShape*) geoShape);
   } else {
      Info("ChooseObject", "WARNING! %s shape is not in the list to process", clsname);
      solidN = NULL;
   }
   return solidN;
}

//______________________________________________________________________________
TGDMLWrite::Xyz TGDMLWrite::GetXYZangles(const Double_t * rotationMatrix)
{
   TGDMLWrite::Xyz lxyz;
   Double_t a, b, c;
   Double_t rad = 180.0 / TMath::ACos(-1.0);
   const Double_t *r = rotationMatrix;
   Double_t cosb = sqrt(r[0] * r[0] + r[1] * r[1]);
   if (cosb > 0.00001) {
      a = atan2(r[5], r[8]) * rad;
      b = atan2(-r[2], cosb) * rad;
      c = atan2(r[1], r[0]) * rad;
   } else {
      a = atan2(-r[7], r[4]) * rad;
      b = atan2(-r[2], cosb) * rad;
      c = 0;
   }
   lxyz.x = a;
   lxyz.y = b;
   lxyz.z = c;
   return lxyz;
}

//______________________________________________________________________________
TGeoCompositeShape* TGDMLWrite::CreateFakeCtub(TGeoCtub* geoShape)
{

   Double_t rmin = geoShape->GetRmin();
   Double_t rmax = geoShape->GetRmax();
   Double_t z = geoShape->GetDz();
   Double_t startphi = geoShape->GetPhi1();
   Double_t deltaphi = geoShape->GetPhi2();
   Double_t x1 = geoShape->GetNlow()[0];
   Double_t y1 = geoShape->GetNlow()[1];
   Double_t z1 = geoShape->GetNlow()[2];
   Double_t x2 = geoShape->GetNhigh()[0];
   Double_t y2 = geoShape->GetNhigh()[1];
   Double_t z2 = geoShape->GetNhigh()[2];
   TString xname = geoShape->GetName();


   Double_t h0 = 2.*((TGeoBBox*)geoShape)->GetDZ();
   Double_t h1 = 2 * z;
   Double_t h2 = 2 * z;
   Double_t boxdx = 1E8 * (2 * rmax) + (2 * z);

   TGeoTubeSeg *T = new TGeoTubeSeg((xname + "T").Data(), rmin, rmax, h0, startphi, deltaphi);
   TGeoBBox *B1 = new TGeoBBox((xname + "B1").Data(), boxdx, boxdx, h1);
   TGeoBBox *B2 = new TGeoBBox((xname + "B2").Data(), boxdx, boxdx, h2);


   //first box position parameters
   Double_t phi1 = 360 - TMath::ATan2(x1, y1) * TMath::RadToDeg();
   Double_t theta1 = 360 - TMath::ATan2(sqrt(x1 * x1 + y1 * y1), z1) * TMath::RadToDeg();

   Double_t phi11 = TMath::ATan2(y1, x1) * TMath::RadToDeg() ;
   Double_t theta11 = TMath::ATan2(z1, sqrt(x1 * x1 + y1 * y1)) * TMath::RadToDeg() ;

   Double_t xpos1 = h1 * TMath::Cos((theta11) * TMath::DegToRad()) * TMath::Cos((phi11) * TMath::DegToRad()) * (-1);
   Double_t ypos1 = h1 * TMath::Cos((theta11) * TMath::DegToRad()) * TMath::Sin((phi11) * TMath::DegToRad()) * (-1);
   Double_t zpos1 = h1 * TMath::Sin((theta11) * TMath::DegToRad()) * (-1);

   //second box position parameters
   Double_t phi2 = 360 - TMath::ATan2(x2, y2) * TMath::RadToDeg();
   Double_t theta2 = 360 - TMath::ATan2(sqrt(x2 * x2 + y2 * y2), z2) * TMath::RadToDeg();

   Double_t phi21 = TMath::ATan2(y2, x2) * TMath::RadToDeg() ;
   Double_t theta21 = TMath::ATan2(z2, sqrt(x2 * x2 + y2 * y2)) * TMath::RadToDeg() ;

   Double_t xpos2 = h2 * TMath::Cos((theta21) * TMath::DegToRad()) * TMath::Cos((phi21) * TMath::DegToRad()) * (-1);
   Double_t ypos2 = h2 * TMath::Cos((theta21) * TMath::DegToRad()) * TMath::Sin((phi21) * TMath::DegToRad()) * (-1);
   Double_t zpos2 = h2 * TMath::Sin((theta21) * TMath::DegToRad()) * (-1);


   //positioning
   TGeoTranslation *t0 = new TGeoTranslation(0, 0, 0);
   TGeoTranslation *t1 = new TGeoTranslation(0 + xpos1, 0 + ypos1, 0 + (zpos1 - z));
   TGeoTranslation *t2 = new TGeoTranslation(0 + xpos2, 0 + ypos2, 0 + (zpos2 + z));
   TGeoRotation *r0 = new TGeoRotation((xname + "r0").Data());
   TGeoRotation *r1 = new TGeoRotation((xname + "r1").Data());
   TGeoRotation *r2 = new TGeoRotation((xname + "r2").Data());

   r1->SetAngles(phi1, theta1, 0);
   r2->SetAngles(phi2, theta2, 0);

   TGeoMatrix* m0 = new TGeoCombiTrans(*t0, *r0);
   TGeoMatrix* m1 = new TGeoCombiTrans(*t1, *r1);
   TGeoMatrix* m2 = new TGeoCombiTrans(*t2, *r2);

   TGeoCompositeShape *CS1 = new TGeoCompositeShape((xname + "CS1").Data(), new TGeoIntersection(T, B1, m0, m1));
   TGeoCompositeShape *cs = new TGeoCompositeShape((xname + "CS").Data(), new TGeoIntersection(CS1, B2, m0, m2));
   return cs;
}

//______________________________________________________________________________
Bool_t TGDMLWrite::IsInList(NameList list, TString name2check)
{
// Checks whether name2check is in (NameList) list

   Bool_t isIN = list[name2check];
   return isIN;
}

//______________________________________________________________________________
TString TGDMLWrite::GenName(TString oldname)
{
//NCNAME basic restrictions
//Replace "$" character with empty character

   TString newname = oldname.ReplaceAll("$", "");
   newname = newname.ReplaceAll(" ", "_");
   // :, @, $, %, &, /, +, ,, ;, whitespace characters or different parenthesis
   newname = newname.ReplaceAll(":", "");
   newname = newname.ReplaceAll("@", "");
   newname = newname.ReplaceAll("%", "");
   newname = newname.ReplaceAll("&", "");
   newname = newname.ReplaceAll("/", "");
   newname = newname.ReplaceAll("+", "");
   newname = newname.ReplaceAll(";", "");
   newname = newname.ReplaceAll("{", "");
   newname = newname.ReplaceAll("}", "");
   newname = newname.ReplaceAll("(", "");
   newname = newname.ReplaceAll(")", "");
   newname = newname.ReplaceAll("[", "");
   newname = newname.ReplaceAll("]", "");
   return newname;
}

//______________________________________________________________________________
TString TGDMLWrite::GenName(TString oldname, TString objPointer)
{
   TString newname = GenName(oldname);
   if (newname != oldname) {
      if (fgkMaxNameErr > fActNameErr) {
         Info("GenName",
              "WARNING! Name of the object was changed because it failed to comply with NCNAME xml datatype restrictions.");
      } else if ((fgkMaxNameErr == fActNameErr)) {
         Info("GenName",
              "WARNING! Probably more names are going to be changed to comply with NCNAME xml datatype restriction, but it will not be displayed on the screen.");
      }
      fActNameErr++;
   }
   TString nameIter;
   Int_t iter = 0;
   switch (fgNamingSpeed) {
      case kfastButUglySufix:
         newname = newname + "0x" + objPointer;
         break;
      case kelegantButSlow:
         //0 means not in the list
         iter = fNameList->fLstIter[newname];
         if (iter == 0) {
            nameIter = "";
         } else {
            nameIter = TString::Format("0x%i", iter);
         }
         fNameList->fLstIter[newname]++;
         newname = newname + nameIter;
         break;
      case kwithoutSufixNotUniq:
         //no change
         newname = newname;
         break;
   }
   //store the name (mapped to pointer)
   fNameList->fLst[objPointer] = newname;
   return newname;
}


//______________________________________________________________________________
Bool_t TGDMLWrite::CanProcess(TObject *pointer)
{
   Bool_t isProcessed = kFALSE;
   isProcessed = pointer->TestBit(fgkProcBit);
   pointer->SetBit(fgkProcBit, kTRUE);
   return !(isProcessed);
}

//______________________________________________________________________________
TString TGDMLWrite::GetPattAxis(Int_t divAxis, const char * pattName, TString& unit)
{
   TString resaxis;
   unit = "cm";
   switch (divAxis) {
      case 1:
         if (strcmp(pattName, "TGeoPatternX") == 0) {
            return "kXAxis";
         } else if (strcmp(pattName, "TGeoPatternCylR") == 0) {
            return "kRho";
         }
         break;
      case 2:
         if (strcmp(pattName, "TGeoPatternY") == 0) {
            return "kYAxis";
         } else if (strcmp(pattName, "TGeoPatternCylPhi") == 0) {
            unit = "deg";
            return "kPhi";
         }
         break;
      case 3:
         if (strcmp(pattName, "TGeoPatternZ") == 0) {
            return "kZAxis";
         }
         break;
      default:
         return "kUndefined";
         break;
   }
   return "kUndefined";
}

