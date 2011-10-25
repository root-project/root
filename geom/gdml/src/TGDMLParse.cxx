/* @(#)root/gdml:$Id$ */
// Author: Ben Lloyd 09/11/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


/*************************************************************************

____________________________________________________________________

TGDMLParse Class

--------------------------------------------------------------------

 This class contains the implementation of the GDML  parser associated to
 all the supported GDML elements. User should never need to explicitely 
 instaciate this class. It is internally used by the TGeoManager.

 Each element process has a 'Binding' to ROOT. The 'binding' is specific 
 mapping of GDML elements (materials, solids, etc) to specific objects which 
 should be instanciated by the converted. In the present case (ROOT) the 
 binding is implemented at the near the end of each process function. Most 
 bindings follow similar format, dependent on what is being added to the 
 geometry.

 This file also contains the implementation of the TGDMLRefl class. This is 
 just a small helper class used internally by the 'reflection' method (for 
 reflected solids).

 The presently supported list of TGeo classes is the following:

 Materials:
 TGeoElement
 TGeoMaterial
 TGeoMixture

 Solids:
 TGeoBBox
 TGeoArb8
 TGeoTubeSeg
 TGeoConeSeg
 TGeoCtub
 TGeoPcon
 TGeoTrap
 TGeoGtra
 TGeoTrd2
 TGeoSphere
 TGeoPara
 TGeoTorus
 TGeoHype
 TGeoPgon
 TGeoXtru
 TGeoEltu
 TGeoParaboloid
 TGeoCompositeShape (subtraction, union, intersection)

 Approximated Solids:
 Ellipsoid (approximated to a TGeoBBox)

 Geometry:
 TGeoVolume
 TGeoVolumeAssembly
 divisions
 reflection

When most solids or volumes are added to the geometry they 


 Whenever a new element is added to GDML schema, this class needs to be extended.
 The appropriate method (process) needs to be implemented, as well as the new
 element process then needs to be linked thru the function TGDMLParse

 For any question or remarks concerning this code, please send an email to
 ben.lloyd@cern.ch

****************************************************************************/

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TXMLEngine.h"
#include "TGeoVolume.h"
#include "TGeoBBox.h"
#include "TGeoParaboloid.h"
#include "TGeoArb8.h"
#include "TGeoTube.h"
#include "TGeoCone.h"
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
#include "TGDMLParse.h"
#include <stdlib.h>
#include <string>

ClassImp(TGDMLParse)

//_________________________________________________________________
TGeoVolume* TGDMLParse::GDMLReadFile(const char* filename)
{
  //creates the new instance of the XMLEngine called 'gdml', using the filename >>
  //then parses the file and creates the DOM tree. Then passes the DOM to the 
  //next function to translate it. 

   // First create engine
   TXMLEngine* gdml = new TXMLEngine;
   gdml->SetSkipComments(kTRUE);
   
   // Now try to parse xml file
   XMLDocPointer_t gdmldoc = gdml->ParseFile(filename);
   if (gdmldoc==0) {
      delete gdml;
      return 0;  
   }
   else {

      // take access to main node   
      XMLNodePointer_t mainnode = gdml->DocGetRootElement(gdmldoc);
   
      fFileEngine[fFILENO] = gdml;
      fStartFile = filename;
      fCurrentFile = filename;

      // display recursively all nodes and subnodes
      ParseGDML(gdml, mainnode);

      // Release memory before exit
      gdml->FreeDoc(gdmldoc);
      delete gdml;
   
   }
   return fWorld;
   
}

//________________________________________________________________
const char* TGDMLParse::ParseGDML(TXMLEngine* gdml, XMLNodePointer_t node) 
{
  //this function recursively moves thru the DOM tree of the GDML file. It checks for
  //key words along the way and if a key word is found it calls the corresponding
  //function to interpret the node.
    
   XMLAttrPointer_t attr = gdml->GetFirstAttr(node);  
   const char* name = gdml->GetNodeName(node);
   XMLNodePointer_t parentn = gdml->GetParent(node);
   const char* parent = gdml->GetNodeName(parentn);
   
   const char* posistr = "position";
   const char* setustr = "setup";
   const char* consstr = "constant";
   const char* varistr = "variable";
   const char* rotastr = "rotation";
   const char* scalstr = "scale";
   const char* elemstr = "element";
   const char* istpstr = "isotope";
   const char* matestr = "material";
   const char* volustr = "volume";
   const char* assestr = "assembly";
   const char* twtrstr = "twistedtrap"; //name changed according to schema
   const char* cutTstr = "cutTube";
   const char* bboxstr = "box";
   const char* xtrustr = "xtru";
   const char* arb8str = "arb8";
   const char* tubestr = "tube";
   const char* conestr = "cone";
   const char* polystr = "polycone";
   const char* hypestr = "hype";
   const char* trapstr = "trap";
   const char* trdstr = "trd";
   const char* sphestr = "sphere";
   const char* orbstr = "orb";
   const char* parastr = "para";
   const char* torustr = "torus";
   const char* hedrstr = "polyhedra";
   const char* eltustr = "eltube";
   const char* subtstr = "subtraction";
   const char* uniostr = "union";
   const char* parbstr = "paraboloid";
   const char* intestr = "intersection";
   const char* reflstr = "reflectedSolid";
   const char* ellistr = "ellipsoid";
   Bool_t hasIsotopes;
   
   if ((strcmp(name, posistr)) == 0){ 
      node = PosProcess(gdml, node, attr);
   } else if ((strcmp(name, rotastr)) == 0){ 
      node = RotProcess(gdml, node, attr);
   } else if ((strcmp(name, scalstr)) == 0){ 
      node = SclProcess(gdml, node, attr);
   } else if ((strcmp(name, setustr)) == 0){ 
      node = TopProcess(gdml, node);
   } else if ((strcmp(name, consstr)) == 0){ 
      node = ConProcess(gdml, node, attr);
   } else if ((strcmp(name, varistr)) == 0){ 
      node = ConProcess(gdml, node, attr);
   } else if ((strcmp(name,elemstr)==0) && !gdml->HasAttr(node, "Z")) {
      hasIsotopes = kTRUE;
      node = EleProcess(gdml, node, parentn, hasIsotopes);   
   } else if (((strcmp(name, "atom")) == 0) && ((strcmp(parent, elemstr)) == 0)){
      hasIsotopes = kFALSE; 
      node = EleProcess(gdml, node, parentn, hasIsotopes);
   } else if (((strcmp(name, "atom")) == 0) && ((strcmp(parent, istpstr)) == 0)){ 
      node = IsoProcess(gdml, node, parentn);
   } else if ((strcmp(name, matestr)) == 0){ 
      if(gdml->HasAttr(node, "Z")) {
         int z = 1;
         node = MatProcess(gdml, node, attr, z);
      } else {
         int z = 0;
         node = MatProcess(gdml, node, attr, z);
      }
   } else if ((strcmp(name, volustr)) == 0){ 
      node = VolProcess(gdml, node);
   } else if ((strcmp(name, bboxstr)) == 0){ 
      node = Box(gdml, node, attr);
   } else if ((strcmp(name, ellistr)) == 0){
      node = Ellipsoid(gdml, node, attr);
   } else if ((strcmp(name, cutTstr)) == 0){ 
      node = CutTube(gdml, node, attr);
   } else if ((strcmp(name, arb8str)) == 0){ 
      node = Arb8(gdml, node, attr);
   } else if ((strcmp(name, tubestr)) == 0){ 
      node = Tube(gdml, node, attr);
   } else if ((strcmp(name, conestr)) == 0){ 
      node = Cone(gdml, node, attr);
   } else if ((strcmp(name, polystr)) == 0){ 
      node = Polycone(gdml, node, attr);
   } else if ((strcmp(name, trapstr)) == 0){ 
      node = Trap(gdml, node, attr);
   } else if ((strcmp(name, trdstr)) == 0){ 
      node = Trd(gdml, node, attr);
   } else if ((strcmp(name, sphestr)) == 0){ 
      node = Sphere(gdml, node, attr);
   } else if ((strcmp(name, xtrustr)) == 0){ 
      node = Xtru(gdml, node, attr);
   } else if ((strcmp(name, twtrstr)) == 0){ 
      node = TwistTrap(gdml, node, attr);
   } else if ((strcmp(name, hypestr)) == 0){ 
      node = Hype(gdml, node, attr);
   } else if ((strcmp(name, orbstr)) == 0){ 
      node = Orb(gdml, node, attr);
   } else if ((strcmp(name, parastr)) == 0){ 
      node = Para(gdml, node, attr);
   } else if ((strcmp(name, torustr)) == 0){ 
      node = Torus(gdml, node, attr);
   } else if ((strcmp(name, eltustr)) == 0){ 
      node = ElTube(gdml, node, attr);
   } else if ((strcmp(name, hedrstr)) == 0){ 
      node = Polyhedra(gdml, node, attr);
   } else if ((strcmp(name, parbstr)) == 0){ 
      node = Paraboloid(gdml, node, attr); 
   } else if ((strcmp(name, subtstr)) == 0){ 
      node = BooSolid(gdml, node, attr, 1);
   } else if ((strcmp(name, intestr)) == 0){ 
      node = BooSolid(gdml, node, attr, 2);
   } else if ((strcmp(name, uniostr)) == 0){ 
      node = BooSolid(gdml, node, attr, 3);
   } else if ((strcmp(name, reflstr)) == 0){ 
      node = Reflection(gdml, node, attr);
   } else if ((strcmp(name, assestr)) == 0){ 
      node = AssProcess(gdml, node);
   //CHECK FOR TAGS NOT SUPPORTED
   } else if (((strcmp(name, "gdml")) != 0) && ((strcmp(name, "define")) != 0) && 
      ((strcmp(name, "element")) != 0) && ((strcmp(name, "materials")) != 0) &&  
      ((strcmp(name, "solids")) != 0) &&  ((strcmp(name, "structure")) != 0) &&  
      ((strcmp(name, "zplane")) != 0) &&  ((strcmp(name, "first")) != 0) &&
      ((strcmp(name, "second")) != 0) &&  ((strcmp(name, "twoDimVertex")) != 0) &&
      ((strcmp(name, "firstposition")) != 0) &&  ((strcmp(name, "firstpositionref")) != 0) &&
      ((strcmp(name, "firstrotation")) != 0) &&  ((strcmp(name, "firstrotationref")) != 0) &&
      ((strcmp(name, "section")) != 0) &&  ((strcmp(name, "world")) != 0) &&
      ((strcmp(name, "isotope")) != 0)){
      std::cout << "Error: Unsupported GDML Tag Used :" << name << ". Please Check Geometry/Schema." << std::endl;
   }
   
   // Check for Child node - if present call this funct. recursively until no more
   
   XMLNodePointer_t child = gdml->GetChild(node);
   while (child!=0) {
      ParseGDML(gdml, child); 
      child = gdml->GetNext(child);
   }
   
   return fWorldName;

}

//____________________________________________________________
double TGDMLParse::Evaluate(const char* evalline) {

   //takes a string containing a mathematical expression and returns the value of the expression

   return TFormula("TFormula",evalline).Eval(0);
}

//____________________________________________________________
Int_t TGDMLParse::SetAxis(const char* axisString)
{
   //When using the 'divide' process in the geometry this function
   //sets the variable 'axis' depending on what is specified.

   Int_t axis = 0;

   if((strcmp(axisString, "kXAxis")) == 0){
      axis = 1;
   } else if((strcmp(axisString, "kYAxis")) == 0){
      axis = 2;
   } else if((strcmp(axisString, "kZAxis")) == 0){
      axis = 3;
   } else if((strcmp(axisString, "kRho")) == 0){
      axis = 1;
   } else if((strcmp(axisString, "kPhi")) == 0){
      axis = 2;
   }
   
   return axis;
}

//____________________________________________________________
const char* TGDMLParse::NameShort(const char* name)
{
   //this function looks thru a string for the chars '0x' next to
   //each other, when it finds this, it calls another function to strip
   //the hex address.   It does this recursively until the end of the 
   //string is reached, returning a string without any hex addresses.
   static TString stripped;
   stripped = name;
   Int_t index = -1;
   while ((index = stripped.Index("0x")) >= 0) {
      stripped = stripped(0,index)+stripped(index+10, stripped.Length());
   }   
   return stripped.Data();   
}

//________________________________________________________
XMLNodePointer_t TGDMLParse::ConProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{
   //In the define section of the GDML file, constants can be declared.
   //when the constant keyword is found, this function is called, and the
   //name and value of the constant is stored in the "fformvec" vector as
   //a TFormula class, representing a constant function

   TString name = "";
   TString value = "";
   TString tempattr;
   
   while (attr!=0) {
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      if(tempattr == "value") { 
         value = gdml->GetAttrValue(attr);
      }      
      attr = gdml->GetNextAttr(attr);   
   }
      
   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   fformvec.push_back(new TFormula(name,value));

   return node;
}
//__________________________________________________________
TString TGDMLParse::GetScale(const char* unit)
{
   //Throughout the GDML file, a unit can de specified.   Whether it be
   //angular or linear, values can be used as well as abbreviations such as
   // 'mm' or 'deg'. This function is passed the specified unit and if it is 
   //found, replaces it with the appropriate value.
   
   TString retunit = "";
   
   if(strcmp(unit, "mm") == 0){
      retunit = "0.1";
   }
   else if(strcmp(unit, "milimeter") == 0){
      retunit = "0.1";
   }
   else if(strcmp(unit, "cm") == 0){
      retunit = "1.0";
   }
   else if(strcmp(unit, "centimeter") == 0){
      retunit = "1.0";
   }
   else if(strcmp(unit, "m") == 0){
      retunit = "100.0";
   }
   else if(strcmp(unit, "meter") == 0){
      retunit = "100.0";
   }
   else if(strcmp(unit, "km") == 0){
      retunit = "100000.0";
   }
   else if(strcmp(unit, "kilometer") == 0){
      retunit = "100000.0";
   }
   else if(strcmp(unit, "rad") == 0){
      retunit = TString::Format("%f", TMath::RadToDeg());
   }
   else if(strcmp(unit, "radian") == 0){
      retunit = TString::Format("%f", TMath::RadToDeg());
   }
   else if(strcmp(unit, "deg") == 0){
      retunit = "1.0";
   }
   else if(strcmp(unit, "degree") == 0){
      retunit = "1.0";
   }
   else if(strcmp(unit, "pi") == 0){
      retunit = "pi";
   }
   else if(strcmp(unit, "avogadro") == 0){
      retunit = TString::Format("%f", TMath::Na());
   }
   else{
      retunit = "0";
   }
   return retunit;
   
}

//____________________________________________________________
XMLNodePointer_t TGDMLParse::PosProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{
   //In the define section of the GDML file, positions can be declared.
   //when the position keyword is found, this function is called, and the
   //name and values of the position are converted into type TGeoPosition 
   //and stored in fposmap map using the name as its key. This function 
   //can also be called when declaring solids.
   
   TString lunit = "mm"; 
   TString xpos = "0"; 
   TString ypos = "0"; 
   TString zpos = "0"; 
   TString name = "0";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "x") { 
         xpos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "y") {
         ypos = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "z") {
         zpos = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "unit") {
         lunit = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString xline = "";
   TString yline = "";
   TString zline = "";
   TString retunit;
   
   retunit = GetScale(lunit);
   
   xline = TString::Format("%s*%s", xpos.Data(), retunit.Data());
   yline = TString::Format("%s*%s", ypos.Data(), retunit.Data());
   zline = TString::Format("%s*%s", zpos.Data(), retunit.Data());
   
   TGeoTranslation* pos = new TGeoTranslation(Evaluate(xline),
                              Evaluate(yline),
                              Evaluate(zline));
   
   fposmap[name.Data()] = pos;
   
   return node;
   
}

//___________________________________________________________________
XMLNodePointer_t TGDMLParse::RotProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{
   //In the define section of the GDML file, rotations can be declared.
   //when the rotation keyword is found, this function is called, and the
   //name and values of the rotation are converted into type TGeoRotation 
   //and stored in frotmap map using the name as its key. This function 
   //can also be called when declaring solids.

   TString aunit = "rad"; 
   TString xpos = "0"; 
   TString ypos = "0"; 
   TString zpos = "0"; 
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "x") { 
         xpos = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "y") {
         ypos = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "z") {
         zpos = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "unit") {
         aunit = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString xline = "";
   TString yline = "";
   TString zline = "";
   TString retunit;
   
   retunit = GetScale(aunit);
   
   xline = TString::Format("%s*%s", xpos.Data(), retunit.Data());
   yline = TString::Format("%s*%s", ypos.Data(), retunit.Data());
   zline = TString::Format("%s*%s", zpos.Data(), retunit.Data());
   
   TGeoRotation* rot = new TGeoRotation();
   
   rot->RotateZ(-(Evaluate(zline)));
   rot->RotateY(-(Evaluate(yline)));
   rot->RotateX(-(Evaluate(xline)));
   
   frotmap[name.Data()] = rot;
   
   return node;
   
}

//___________________________________________________________________
XMLNodePointer_t TGDMLParse::SclProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{
   //In the define section of the GDML file, rotations can be declared.
   //when the scale keyword is found, this function is called, and the
   //name and values of the scale are converted into type TGeoScale
   //and stored in fsclmap map using the name as its key. This function 
   //can also be called when declaring solids.

   TString xpos = "0"; 
   TString ypos = "0"; 
   TString zpos = "0"; 
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "x") { 
         xpos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "y") {
         ypos = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "z"){
         zpos = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TGeoScale* scl = new TGeoScale(Evaluate(xpos),Evaluate(ypos),Evaluate(zpos));
   
   fsclmap[name.Data()] = scl;
   
   return node;
}

//___________________________________________________________________
XMLNodePointer_t TGDMLParse::IsoProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t parentn)
{   
   //In the material section of the GDML file, an isotope may be declared. 
   //when the isotope keyword is found, this function is called, and the 
   //required parameters are taken and stored, these are then bound and
   //converted to type TGeoIsotope and stored in fisomap map using the name 
   //as its key.
   TString z = "0";
   TString name = "";
   TString n = "0";
   TString atom = "0";
   TString tempattr;
   
   //obtain attributes for the element
   
   XMLAttrPointer_t attr = gdml->GetFirstAttr(parentn);
   
   while (attr!=0){    
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "z") { 
         z = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "n") {
         n = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }
   
   //get the atom value for the element
   
   attr = gdml->GetFirstAttr(node);
   
   while (attr!=0){      
      
      tempattr = gdml->GetAttrName(attr);
      
      if (tempattr == "value") { 
         atom = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }
   
   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   Int_t z2 = (Int_t)Evaluate(z);
   Int_t n2 = (Int_t)Evaluate(n);
   Double_t atom2 = Evaluate(atom);
   
   TGeoIsotope* iso = new TGeoIsotope(NameShort(name), z2 , n2, atom2);   
   fisomap[name.Data()] = iso;
   
   return node;
  
}

//___________________________________________________________
XMLNodePointer_t TGDMLParse::EleProcess(TXMLEngine* gdml, XMLNodePointer_t node,   XMLNodePointer_t parentn, Bool_t hasIsotopes)
{   
   //In the materials section of the GDML file, elements can be declared.
   //when the element keyword is found, this function is called, and the
   //name and values of the element are converted into type TGeoElement and
   //stored in felemap map using the name as its key.

   TString z = "0";
   TString name = "";
   TString formula = "";
   TString atom = "0";
   TString tempattr;
   Int_t   ncompo = 0;
   typedef FracMap::iterator fractions;
   FracMap fracmap;
   
   XMLNodePointer_t child = 0;
   
   //obtain attributes for the element
   
   XMLAttrPointer_t attr = gdml->GetFirstAttr(node);
   
   if (hasIsotopes) {
      // Get the name of the element
      while (attr!=0){   
         tempattr = gdml->GetAttrName(attr);
         if(tempattr == "name") { 
            name = gdml->GetAttrValue(attr);
            if((strcmp(fCurrentFile,fStartFile)) != 0){
               name = TString::Format("%s_%s", name.Data(), fCurrentFile);
            }
            break;
         }
         attr = gdml->GetNextAttr(attr);
      }   
      // Get component isotopes. Loop all children.
      child = gdml->GetChild(node);
      while (child!=0) {
         // Check for fraction node name
         if((strcmp(gdml->GetNodeName(child), "fraction")) == 0){
            Double_t n = 0;
            TString ref = ""; 
            ncompo = ncompo + 1;
            attr = gdml->GetFirstAttr(child);
            while (attr!=0){   
               tempattr = gdml->GetAttrName(attr);
               tempattr.ToLower();
               if(tempattr == "n") { 
                  n = Evaluate(gdml->GetAttrValue(attr));
               }
               else if (tempattr == "ref") { 
                  ref = gdml->GetAttrValue(attr);
                  if((strcmp(fCurrentFile,fStartFile)) != 0){
                     ref = TString::Format("%s_%s", ref.Data(), fCurrentFile);
                  }
               }
               attr = gdml->GetNextAttr(attr);
            } // loop on child attributes
            fracmap[ref.Data()] = n; 
         }
         child = gdml->GetNext(child);
      } // loop on childs
      // Create TGeoElement
      TGeoElement *ele = new TGeoElement(NameShort(name), "", ncompo);
      for(fractions f = fracmap.begin(); f != fracmap.end(); f++){
         if(fisomap.find(f->first) != fisomap.end()){
            ele->AddIsotope((TGeoIsotope*)fisomap[f->first], f->second);
         }
      }
      felemap[name.Data()] = ele;
      return child;
   }
   
   attr = gdml->GetFirstAttr(parentn);
   while (attr!=0){    
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "z") { 
         z = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "formula") {
         formula = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }
   
   //get the atom value for the element
   
   attr = gdml->GetFirstAttr(node);
   
   while (attr!=0){      
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "value") { 
         atom = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }
   
   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   Int_t z2 = (Int_t)Evaluate(z);
   Double_t atom2 = Evaluate(atom);
   
   TGeoElement* ele = new TGeoElement(formula, NameShort(name), z2 , atom2);
   
   felemap[name.Data()] = ele;
   
   return node;
}

//_________________________________________________________________________
XMLNodePointer_t TGDMLParse::MatProcess(TXMLEngine* gdml, XMLNodePointer_t node,   XMLAttrPointer_t attr, int z)
{   
   //In the materials section of the GDML file, materials can be declared.
   //when the material keyword is found, this function is called, and the
   //name and values of the material are converted into type TGeoMaterial 
   //and stored in fmatmap map using the name as its key. Mixtures can also
   // be declared, and they are converted to TGeoMixture and stored in
   //fmixmap.   These mixtures and materials are then all converted into one
   //common type - TGeoMedium.   The map fmedmap is then built up of all the 
   //mixtures and materials.

 //!Map to hold fractions while being processed
   typedef FracMap::iterator fractions;
   FracMap fracmap;
   
   static int medid = 0;
   XMLNodePointer_t child = gdml->GetChild(node);
   TString tempattr = "";
   Int_t ncompo = 0, mixflag = 2;
   Double_t density = 0;
   TString name = "";
   TGeoMixture* mix = 0; 
   TGeoMaterial* mat = 0;
   TString tempconst = "";
   TString matname;
   Bool_t composite = kFALSE;
   
   if (z == 1){
      Double_t a = 0;
      Double_t d = 0;
      
      while (child!=0) {
         attr = gdml->GetFirstAttr(child);
         
         if((strcmp(gdml->GetNodeName(child), "atom")) == 0){
            while (attr!=0){   
               tempattr = gdml->GetAttrName(attr);
               tempattr.ToLower();
       
               if(tempattr == "value") { 
                  a = Evaluate(gdml->GetAttrValue(attr));
               }       
               attr = gdml->GetNextAttr(attr);
            }          
         }
         
         if((strcmp(gdml->GetNodeName(child), "D")) == 0){
            while (attr!=0){   
               tempattr = gdml->GetAttrName(attr);
               tempattr.ToLower();
          
               if(tempattr == "value") { 
                  d = Evaluate(gdml->GetAttrValue(attr));   
               }       
               attr = gdml->GetNextAttr(attr);
            } 
         }         
         child = gdml->GetNext(child);
      }
      
      //still in the is Z else...but not in the while..
      
      name = gdml->GetAttr(node, "name");

      if((strcmp(fCurrentFile,fStartFile)) != 0){
         name = TString::Format("%s_%s", name.Data(), fCurrentFile);
      }
      
      //CHECK FOR CONSTANTS    
      tempconst = gdml->GetAttr(node, "Z");
      
      mat = new TGeoMaterial(NameShort(name), a, Evaluate(tempconst), d);      
      mixflag = 0;
      TGeoElement* mat_ele = new TGeoElement(NameShort(name), "", atoi(tempconst), a);
      felemap[name.Data()] = mat_ele;
   }
   
   else if (z == 0){
      while (child!=0) {
         attr = gdml->GetFirstAttr(child);
         if((strcmp(gdml->GetNodeName(child), "fraction")) == 0){
            Double_t n = 0;
            TString ref = ""; 
            ncompo = ncompo + 1;
       
            while (attr!=0){   
               tempattr = gdml->GetAttrName(attr);
               tempattr.ToLower();
          
               if(tempattr == "n") { 
                  n = Evaluate(gdml->GetAttrValue(attr));
               }
               else if(tempattr == "ref") { 
                  ref = gdml->GetAttrValue(attr);
                  if((strcmp(fCurrentFile,fStartFile)) != 0){
                     ref = TString::Format("%s_%s", ref.Data(), fCurrentFile);
                  }

               }
          
               attr = gdml->GetNextAttr(attr);
            }
       
            fracmap[ref.Data()] = n; 
         }
         
         else if((strcmp(gdml->GetNodeName(child), "composite")) == 0){
            composite = kTRUE;
            Double_t n = 0;
            TString ref = ""; 
            ncompo = ncompo + 1;
       
            while (attr!=0){   
               tempattr = gdml->GetAttrName(attr);
               tempattr.ToLower();
          
               if(tempattr == "n") { 
                  n = Evaluate(gdml->GetAttrValue(attr));
               }
               else if(tempattr == "ref") { 
                  ref = gdml->GetAttrValue(attr);
               }
          
               attr = gdml->GetNextAttr(attr);
            }
       
            fracmap[ref.Data()] = n;
         }
         
         else if((strcmp(gdml->GetNodeName(child), "D")) == 0){
            while (attr!=0){         
               tempattr = gdml->GetAttrName(attr);
               tempattr.ToLower();
          
               if(tempattr == "value") { 
                  density = Evaluate(gdml->GetAttrValue(attr));
               }
          
               attr = gdml->GetNextAttr(attr);
            }
         }
         
         child = gdml->GetNext(child);       
      }
      
      //still in the not Z else...but not in the while..
      
      name = gdml->GetAttr(node, "name");
      if((strcmp(fCurrentFile,fStartFile)) != 0){
         name = TString::Format("%s_%s", name.Data(), fCurrentFile);
      }
      mix = new TGeoMixture(NameShort(name), 0 /*ncompo*/, density);
      mixflag = 1;
      Int_t natoms;
      Double_t weight;
      
      for(fractions f = fracmap.begin(); f != fracmap.end(); f++){
         matname = f->first;
         matname = NameShort(matname);
         TGeoMaterial *mattmp = (TGeoMaterial*)gGeoManager->GetListOfMaterials()->FindObject(matname);
         if(mattmp || (felemap.find(f->first) != felemap.end())){
            if (composite) {
               natoms = (Int_t)f->second;
               mix->AddElement(felemap[f->first], natoms);
            } else {
               weight = f->second;
               if (mattmp) mix->AddElement(mattmp, weight);
               else        mix->AddElement(felemap[f->first], weight); 
            }   
         } 
         else {
       // mix->DefineElement(i, fmixmap[f->first], f->second); BUG IN PYTHON???
         }
      }
      
   }//end of not Z else
   
   medid = medid + 1;
   
   TGeoMedium* med = 0;
   
   if(mixflag == 1) {
      fmixmap[name.Data()] = mix;
      med = new TGeoMedium(NameShort(name), medid, mix);
   } 
   else if (mixflag == 0) {
      fmatmap[name.Data()] = mat;
      med = new TGeoMedium(NameShort(name), medid, mat);
   }
   
   fmedmap[name.Data()] = med;
   
   return child;
}

//____________________________________________________________
XMLNodePointer_t TGDMLParse::VolProcess(TXMLEngine* gdml, XMLNodePointer_t node)
{   
   //In the structure section of the GDML file, volumes can be declared.
   //when the volume keyword is found, this function is called, and the
   //name and values of the volume are converted into type TGeoVolume and
   //stored in fvolmap map using the name as its key. Volumes reference to 
   //a solid declared higher up in the solids section of the GDML file. 
   //Some volumes reference to other physical volumes to contain inside 
   //that volume, declaring positions and rotations within that volume. 
   //When each 'physvol' is declared, a matrix for its rotation and 
   //translation is built and the 'physvol node' is added to the original 
   //volume using TGeoVolume->AddNode.
   //volume division is also declared within the volume node, and once the
   //values for the division have been collected, using TGeoVolume->divide,
   //the division can be applied.

   XMLAttrPointer_t attr;
   XMLNodePointer_t subchild;
   XMLNodePointer_t child = gdml->GetChild(node);
   TString name;
   TString solidname = "";
   TString tempattr = "";
   TGeoShape* solid = 0;
   TGeoMedium* medium = 0;
   TGeoVolume* vol = 0; 
   TGeoVolume* lv = 0;
   TGeoShape* reflex = 0;
   const Double_t* parentrot = 0;
   int yesrefl = 0;
   TString reftemp = "";
   
   while (child!=0) {
      if((strcmp(gdml->GetNodeName(child), "solidref")) == 0){

         reftemp = gdml->GetAttr(child, "ref");
         if((strcmp(fCurrentFile,fStartFile)) != 0){
            reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
         }
         if(fsolmap.find(reftemp.Data()) != fsolmap.end()){ 
            solid = fsolmap[reftemp.Data()];
         } 
         else if(freflectmap.find(reftemp.Data()) != freflectmap.end()){
            solidname = reftemp;
            reflex = fsolmap[freflectmap[reftemp.Data()]];
         } 
         else {
            printf("Solid: %s, Not Yet Defined!\n", reftemp.Data());
         }
      }
      
      if((strcmp(gdml->GetNodeName(child), "materialref")) == 0){
         reftemp = gdml->GetAttr(child, "ref");
         if((strcmp(fCurrentFile,fStartFile)) != 0){
            reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
         }
         if(fmedmap.find(reftemp.Data()) != fmedmap.end()){ 
            medium = fmedmap[reftemp.Data()];
         } 
         else {
            printf("Medium: %s, Not Yet Defined!\n", gdml->GetAttr(child, "ref"));
         }
      }
      
      child = gdml->GetNext(child);
   }
   
   name = gdml->GetAttr(node, "name");
      
   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }
   
   if(reflex == 0){
      vol = new TGeoVolume(NameShort(name), solid, medium);
   } 
   else {
      vol = new TGeoVolume(NameShort(name), reflex, medium);
      freflvolmap[name.Data()] = solidname;
      TGDMLRefl* parentrefl = freflsolidmap[solidname.Data()];
      parentrot = parentrefl->GetMatrix()->GetRotationMatrix();
      yesrefl = 1;
   }
   
   fvolmap[name.Data()] = vol;
   
   //PHYSVOL - run through child nodes of VOLUME again..
   
   child = gdml->GetChild(node);
   
   while (child!=0) {
      if((strcmp(gdml->GetNodeName(child), "physvol")) == 0){

         TString volref = "";

         TGeoTranslation* pos = 0;
         TGeoRotation* rot = 0;
         TGeoScale* scl = 0;
	
         subchild = gdml->GetChild(child);

         while (subchild!=0){
            tempattr = gdml->GetNodeName(subchild);
            tempattr.ToLower();
    
            if(tempattr == "volumeref"){
               reftemp = gdml->GetAttr(subchild, "ref");
               if((strcmp(fCurrentFile,fStartFile)) != 0){
                  reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
               }
               lv = fvolmap[reftemp.Data()];
               volref = reftemp;
            } 
            else if(tempattr == "file"){

               const char* filevol;
               const char* prevfile = fCurrentFile;

               fCurrentFile = gdml->GetAttr(subchild, "name");
               filevol = gdml->GetAttr(subchild, "volname");
                 
               TXMLEngine* gdml2 = new TXMLEngine;
               gdml2->SetSkipComments(kTRUE);

               XMLDocPointer_t filedoc1 = gdml2->ParseFile(fCurrentFile);
               if (filedoc1==0) {
                  Fatal("VolProcess", "Bad filename given %s", fCurrentFile);
               } 
               // take access to main node   
               XMLNodePointer_t mainnode2 = gdml2->DocGetRootElement(filedoc1);
               //increase depth counter + add DOM pointer
               fFILENO = fFILENO + 1;
               fFileEngine[fFILENO] = gdml2;

               if(ffilemap.find(fCurrentFile) != ffilemap.end()){ 
                  volref = ffilemap[fCurrentFile];
               } 
               else {
                  volref = ParseGDML(gdml2, mainnode2);
                  ffilemap[fCurrentFile] = volref;
               }

               if(filevol){
                  volref = filevol;
                  if((strcmp(fCurrentFile,fStartFile)) != 0){
                     volref = TString::Format("%s_%s", volref.Data(), fCurrentFile);
                  }
               }

               fFILENO = fFILENO - 1;
               gdml = fFileEngine[fFILENO];
               fCurrentFile = prevfile;
               lv = fvolmap[volref.Data()];
               //File tree complete - Release memory before exit
               gdml->FreeDoc(filedoc1);
               delete gdml2;
            }
            else if(tempattr == "position"){
               attr = gdml->GetFirstAttr(subchild);
               PosProcess(gdml, subchild, attr);
               reftemp = gdml->GetAttr(subchild, "name");
               if((strcmp(fCurrentFile,fStartFile)) != 0){
                  reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
               }
               pos = fposmap[reftemp.Data()];
            }
            else if(tempattr == "positionref"){
               reftemp = gdml->GetAttr(subchild, "ref");
               if((strcmp(fCurrentFile,fStartFile)) != 0){
                  reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
               }
               if(fposmap.find(reftemp.Data()) != fposmap.end()) pos = fposmap[reftemp.Data()];
               else std::cout << "ERROR! Physvol's position " << reftemp << " not found!" << std::endl;
            } 
            else if(tempattr == "rotation") {
               attr = gdml->GetFirstAttr(subchild);
               RotProcess(gdml, subchild, attr);
               reftemp = gdml->GetAttr(subchild, "name");
               if((strcmp(fCurrentFile,fStartFile)) != 0){
                  reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
               }
               rot = frotmap[reftemp.Data()];
            }
            else if(tempattr == "rotationref"){
               reftemp = gdml->GetAttr(subchild, "ref");
               if((strcmp(fCurrentFile,fStartFile)) != 0){
                  reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
               }
               if (frotmap.find(reftemp.Data()) != frotmap.end()) rot = frotmap[reftemp.Data()];
               else std::cout << "ERROR! Physvol's rotation " << reftemp << " not found!" << std::endl;
            }
            else if(tempattr =="scale") {
               attr = gdml->GetFirstAttr(subchild);
               SclProcess(gdml, subchild, attr);
               reftemp = gdml->GetAttr(subchild, "name");
               if((strcmp(fCurrentFile,fStartFile)) != 0){
                  reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
               }
               scl = fsclmap[reftemp.Data()];
            }
            else if(tempattr == "scaleref"){
               reftemp = gdml->GetAttr(subchild, "ref");
               if((strcmp(fCurrentFile,fStartFile)) != 0){
                  reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
               }
               if (fsclmap.find(reftemp.Data()) != fsclmap.end()) scl = fsclmap[reftemp.Data()];
               else std::cout << "ERROR! Physvol's scale " << reftemp << " not found!" << std::endl;
            }

            subchild = gdml->GetNext(subchild);
         }
         
         //ADD PHYSVOL TO GEOMETRY
         fVolID = fVolID + 1;

         TGeoHMatrix *transform = new TGeoHMatrix();         

         if (pos!=0) transform->SetTranslation(pos->GetTranslation());
         if (rot!=0) transform->SetRotation(rot->GetRotationMatrix());

         if (scl!=0) { // Scaling must be added to the rotation matrix!

            Double_t scale3x3[9];
            memset(scale3x3,0,9*sizeof(Double_t));
            const Double_t *diagonal = scl->GetScale();

            scale3x3[0] = diagonal[0];
            scale3x3[4] = diagonal[1];
            scale3x3[8] = diagonal[2];
         
            TGeoRotation scaleMatrix;
            scaleMatrix.SetMatrix(scale3x3);
            transform->Multiply(&scaleMatrix);
         }

// BEGIN: reflectedSolid. Remove lines between if reflectedSolid will be removed from GDML!!!

         if(freflvolmap.find(volref.Data()) != freflvolmap.end()) { 
            // if the volume is a reflected volume the matrix needs to be CHANGED
            TGDMLRefl* temprefl = freflsolidmap[freflvolmap[volref.Data()]];
            transform->Multiply(temprefl->GetMatrix());
         }

         if(yesrefl == 1) { 
            // reflection is done per solid so that we cancel it if exists in mother volume!!!
            TGeoRotation prot;
            prot.SetMatrix(parentrot);
            transform->MultiplyLeft(&prot);
         }

// END: reflectedSolid

         vol->AddNode(lv,fVolID,transform);
      }
      else if((strcmp(gdml->GetNodeName(child), "divisionvol")) == 0){
   
         TString divVolref = "";
         Int_t axis = 0;
         TString number = "";
         TString width = "";
         TString offset = "";
         TString lunit = "mm";
         
         attr = gdml->GetFirstAttr(child);
         
         while (attr!=0) {
    
            tempattr = gdml->GetAttrName(attr);
            tempattr.ToLower();
       
            if(tempattr == "axis") { 
               axis = SetAxis(gdml->GetAttrValue(attr));
            }
            else if (tempattr == "number") { 
               number = gdml->GetAttrValue(attr);
            }
            else if (tempattr == "width") {
               width = gdml->GetAttrValue(attr);
            }
            else if (tempattr == "offset") {
               offset = gdml->GetAttrValue(attr);
            }
            else if (tempattr == "unit") {
               lunit = gdml->GetAttrValue(attr);
            }
       
            attr = gdml->GetNextAttr(attr);
       
         }
         
         subchild = gdml->GetChild(child);
         
         while (subchild!=0){
            tempattr = gdml->GetNodeName(subchild);
            tempattr.ToLower();
       
            if(tempattr == "volumeref"){ 
               reftemp = gdml->GetAttr(subchild, "ref");
               if((strcmp(fCurrentFile,fStartFile)) != 0){
                  reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
               }
               divVolref = reftemp;
            }
       
            subchild = gdml->GetNext(subchild);             
         } 
    
    
         TString numberline = "";
         TString widthline = "";
         TString offsetline = "";
         TString retunit;
         
         retunit = GetScale(lunit);
         
         numberline = TString::Format("%s", number.Data());
         widthline = TString::Format("%s*%s", width.Data(), retunit.Data());
         offsetline = TString::Format("%s*%s", offset.Data(), retunit.Data());
 
         fVolID = fVolID + 1;
         Double_t xlo, xhi;
         vol->GetShape()->GetAxisRange(axis, xlo, xhi);
         Int_t ndiv = (Int_t)Evaluate(numberline);
         Double_t start = xlo + (Double_t)Evaluate(offsetline);
         Double_t step = (Double_t)Evaluate(widthline);
         Int_t numed = 0;
         TGeoVolume *old = fvolmap[NameShort(reftemp)];
         if (old) {
            // We need to recreate the content of the divided volume
            old = fvolmap[NameShort(reftemp)];
            // medium id
            numed = old->GetMedium()->GetId();
         }   
         TGeoVolume *divvol = vol->Divide(NameShort(reftemp), axis, ndiv, start, step, numed);
         if (old && old->GetNdaughters()) {
            divvol->ReplayCreation(old);
         }
         fvolmap[NameShort(reftemp)] = divvol;

      }//end of Division else if
      
      child = gdml->GetNext(child);
   }
   
   return child;
   
}

//______________________________________________________
XMLNodePointer_t TGDMLParse::BooSolid(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr, int num)
{ 
   //In the solid section of the GDML file, boolean solids can be 
   //declared. when the subtraction, intersection or union   keyword 
   //is found, this function is called, and the values (rotation and 
   //translation) of the solid are converted into type TGeoCompositeShape
   //and stored in fsolmap map using the name as its key.
   //
   //1 = SUBTRACTION
   //2 = INTERSECTION
   //3 = UNION
   
   TString reftemp = "";
   TString tempattr = "";
   XMLNodePointer_t child = gdml->GetChild(node);

   TGeoShape* first = 0;
   TGeoShape* second = 0;

   TGeoTranslation* firstPos = new TGeoTranslation(0,0,0);
   TGeoTranslation* secondPos = new TGeoTranslation(0,0,0);

   TGeoRotation* firstRot = new TGeoRotation();
   TGeoRotation* secondRot = new TGeoRotation();

   firstRot->RotateZ(0);
   firstRot->RotateY(0);
   firstRot->RotateX(0);

   secondRot->RotateZ(0);
   secondRot->RotateY(0);
   secondRot->RotateX(0);

   TString name = gdml->GetAttr(node, "name");

   if((strcmp(fCurrentFile,fStartFile)) != 0)
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   
   while (child!=0){
      tempattr = gdml->GetNodeName(child);
      tempattr.ToLower();
      
      if(tempattr == "first"){
         reftemp = gdml->GetAttr(child, "ref");
         if((strcmp(fCurrentFile,fStartFile)) != 0){
            reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
         }
         if(fsolmap.find(reftemp.Data()) != fsolmap.end()){ 
            first = fsolmap[reftemp.Data()];
         }
      }
      else if(tempattr == "second") {
         reftemp = gdml->GetAttr(child, "ref");
         if((strcmp(fCurrentFile,fStartFile)) != 0){
            reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
         }
         if(fsolmap.find(reftemp.Data()) != fsolmap.end()){   
           second = fsolmap[reftemp.Data()];
         }
      }
      else if(tempattr == "position"){
         attr = gdml->GetFirstAttr(child);
         PosProcess(gdml, child, attr);
         reftemp = gdml->GetAttr(child, "name");
         if((strcmp(fCurrentFile,fStartFile)) != 0){
            reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
         }
         secondPos = fposmap[reftemp.Data()];
      }
      else if(tempattr == "positionref"){
         reftemp = gdml->GetAttr(child, "ref");
         if((strcmp(fCurrentFile,fStartFile)) != 0){
            reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
         }
         if(fposmap.find(reftemp.Data()) != fposmap.end()){ 
            secondPos = fposmap[reftemp.Data()];
         }
      }
      else if(tempattr == "rotation"){
         attr = gdml->GetFirstAttr(child);
         RotProcess(gdml, child, attr);
         reftemp = gdml->GetAttr(child, "name");
         if((strcmp(fCurrentFile,fStartFile)) != 0){
            reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
         }
         secondRot = frotmap[reftemp.Data()];
      }
      else if(tempattr == "rotationref"){
         reftemp = gdml->GetAttr(child, "ref");
         if((strcmp(fCurrentFile,fStartFile)) != 0){
            reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
         }
         if(frotmap.find(reftemp.Data()) != frotmap.end()){ 
            secondRot = frotmap[reftemp.Data()];
         }
      } 
      else if(tempattr == "firstposition"){
         attr = gdml->GetFirstAttr(child);
         PosProcess(gdml, child, attr);
         reftemp = gdml->GetAttr(child, "name");
         if((strcmp(fCurrentFile,fStartFile)) != 0){
            reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
         }
         firstPos = fposmap[reftemp.Data()];
      }
      else if(tempattr == "firstpositionref"){
         reftemp = gdml->GetAttr(child, "ref");
         if((strcmp(fCurrentFile,fStartFile)) != 0){
            reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
         }
         if(fposmap.find(reftemp.Data()) != fposmap.end()){ 
            firstPos = fposmap[reftemp.Data()];
         }
      }
      else if(tempattr == "firstrotation"){
         attr = gdml->GetFirstAttr(child);
         RotProcess(gdml, child, attr);
         reftemp = gdml->GetAttr(child, "name");
         if((strcmp(fCurrentFile,fStartFile)) != 0){
            reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
         }
         firstRot = frotmap[reftemp.Data()];
      }
      else if(tempattr == "firstrotationref"){
         reftemp = gdml->GetAttr(child, "ref");
         if((strcmp(fCurrentFile,fStartFile)) != 0){
            reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
         }
         if(frotmap.find(reftemp.Data()) != frotmap.end()){ 
            firstRot = frotmap[reftemp.Data()];
         }
      }       
      child = gdml->GetNext(child);
   }

   TGeoMatrix* firstMatrix = new TGeoCombiTrans(*firstPos,firstRot->Inverse());
   TGeoMatrix* secondMatrix = new TGeoCombiTrans(*secondPos,secondRot->Inverse());

   TGeoCompositeShape* boolean = 0;
   if (!first || !second) {
      Fatal("BooSolid", "Incomplete solid %s, missing shape components", name.Data());
      return child;
   }   
   switch (num) {
   case 1: boolean = new TGeoCompositeShape(NameShort(name),new TGeoSubtraction(first,second,firstMatrix,secondMatrix)); break;      // SUBTRACTION
   case 2: boolean = new TGeoCompositeShape(NameShort(name),new TGeoIntersection(first,second,firstMatrix,secondMatrix)); break;     // INTERSECTION 
   case 3: boolean = new TGeoCompositeShape(NameShort(name),new TGeoUnion(first,second,firstMatrix,secondMatrix)); break;            // UNION
   default:
    break;
   }

   fsolmap[name.Data()] = boolean;

   return child;
}

//________________________________________________________
XMLNodePointer_t TGDMLParse::AssProcess(TXMLEngine* gdml, XMLNodePointer_t node)
{   
   //In the structure section of the GDML file, assembly volumes can be 
   //declared. when the assembly keyword is found, this function is called, 
   //and the name is converted into type TGeoVolumeAssembly and
   //stored in fvolmap map using the name as its key. Some assembly volumes 
   //reference to other physical volumes to contain inside that assembly, 
   //declaring positions and rotations within that volume. When each 'physvol' 
   //is declared, a matrix for its rotation and translation is built and the 
   //'physvol node' is added to the original assembly using TGeoVolume->AddNode.

   TString name = gdml->GetAttr(node, "name");
   TString reftemp = "";

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   XMLAttrPointer_t attr;
   XMLNodePointer_t subchild;
   XMLNodePointer_t child = gdml->GetChild(node);
   TString tempattr = "";
   TGeoVolume* lv = 0;
   TGeoTranslation* pos = 0;
   TGeoRotation* rot = 0;
   TGeoCombiTrans* matr;
   
   TGeoVolumeAssembly* assem = new TGeoVolumeAssembly(NameShort(name));

   
   //PHYSVOL - run through child nodes of VOLUME again..
   
//   child = gdml->GetChild(node);
   
   while (child!=0) {
      if((strcmp(gdml->GetNodeName(child), "physvol")) == 0){
         subchild = gdml->GetChild(child);
         pos = new TGeoTranslation(0,0,0);
         rot = new TGeoRotation();
         
         while (subchild!=0){
            tempattr = gdml->GetNodeName(subchild);
            tempattr.ToLower();
    
            if(tempattr == "volumeref"){
               reftemp = gdml->GetAttr(subchild, "ref");
               if((strcmp(fCurrentFile,fStartFile)) != 0){
                  reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
               }
               lv = fvolmap[reftemp.Data()];
            }       
            else if(tempattr == "positionref"){
               reftemp = gdml->GetAttr(subchild, "ref");
               if((strcmp(fCurrentFile,fStartFile)) != 0){
                  reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
               }
               if(fposmap.find(reftemp.Data()) != fposmap.end()){ 
                  pos = fposmap[reftemp.Data()];
               }
            }
            else if(tempattr == "position"){
               attr = gdml->GetFirstAttr(subchild);
               PosProcess(gdml, subchild, attr);
               reftemp = gdml->GetAttr(subchild, "name");
               if((strcmp(fCurrentFile,fStartFile)) != 0){
                  reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
               }
               pos = fposmap[reftemp.Data()];
            }
            else if(tempattr == "rotationref"){
               reftemp = gdml->GetAttr(subchild, "ref");
               if((strcmp(fCurrentFile,fStartFile)) != 0){
                  reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
               }
               if(frotmap.find(reftemp.Data()) != frotmap.end()){ 
                  rot = frotmap[reftemp.Data()];
               }
            }
            else if(tempattr == "rotation"){
               attr = gdml->GetFirstAttr(subchild);
               RotProcess(gdml, subchild, attr);
               reftemp = gdml->GetAttr(subchild, "name");
               if((strcmp(fCurrentFile,fStartFile)) != 0){
                  reftemp = TString::Format("%s_%s", reftemp.Data(), fCurrentFile);
               }
               rot = frotmap[reftemp.Data()];
            }
       
            subchild = gdml->GetNext(subchild);
         }
         
         //ADD PHYSVOL TO GEOMETRY
         fVolID = fVolID + 1;
         matr = new TGeoCombiTrans(*pos, *rot);
         assem->AddNode(lv, fVolID, matr);
         
      }
      child = gdml->GetNext(child);
   }

   fvolmap[name.Data()] = assem;
   return child;
}

//________________________________________________________
XMLNodePointer_t   TGDMLParse::TopProcess(TXMLEngine* gdml, XMLNodePointer_t node)
{   
   //In the setup section of the GDML file, the top volume need to be 
   //declared. when the setup keyword is found, this function is called, 
   //and the top volume ref is taken and 'world' is set

   const char* name = gdml->GetAttr(node, "name");
   gGeoManager->SetName(name);
   XMLNodePointer_t child = gdml->GetChild(node);
 
   while(child != 0){
      
      if((strcmp(gdml->GetNodeName(child), "world") == 0)){
         const char* reftemp; 
         reftemp = gdml->GetAttr(child, "ref");
         if((strcmp(fCurrentFile,fStartFile)) != 0){
            reftemp = TString::Format("%s_%s", reftemp, fCurrentFile);
         }
         fWorld = fvolmap[reftemp];
         fWorldName = reftemp;
      } 
      child = gdml->GetNext(child);
   }   
   return node;
}

//___________________________________________________________________
XMLNodePointer_t TGDMLParse::Box(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, a box may be declared. 
   //when the box keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoBBox and stored in fsolmap map using the name 
   //as its key.
   
   TString lunit = "mm"; 
   TString xpos = "0"; 
   TString ypos = "0"; 
   TString zpos = "0"; 
   TString name = "";
   TString tempattr; 

   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "x") { 
         xpos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "y"){
         ypos = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "z"){
         zpos = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit"){
         lunit = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }
   
   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString xline = "";
   TString yline = "";
   TString zline = "";
   TString retunit;
   
   retunit = GetScale(lunit);
   
   xline = TString::Format("%s*%s", xpos.Data(), retunit.Data());
   yline = TString::Format("%s*%s", ypos.Data(), retunit.Data());
   zline = TString::Format("%s*%s", zpos.Data(), retunit.Data());

   
   TGeoBBox* box = new TGeoBBox(NameShort(name),Evaluate(xline)/2,
                        Evaluate(yline)/2,
                        Evaluate(zline)/2);
   
   fsolmap[name.Data()] = box;
   
   return node;
   
}

//___________________________________________________________________
XMLNodePointer_t TGDMLParse::Ellipsoid(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, an ellipsoid may be declared. 
   //Unfortunately, the ellipsoid is not supported under ROOT so,
   //when the ellipsoid keyword is found, this function is called
   //to convert it to a simple box with similar dimensions, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoBBox and stored in fsolmap map using the name 
   //as its key.
   
   TString lunit = "mm"; 
   TString ax = "0"; 
   TString by = "0"; 
   TString cz = "0";
   TString zcut1 = "0"; 
   TString zcut2 = "0";
   TString name = "";
   TString tempattr; 

   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name"){ 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "ax"){ 
         ax = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "by"){
         by = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "cz"){
         cz = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "zcut1"){ 
         zcut1 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "zcut2"){
         zcut2 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "lunit"){
         lunit = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }
   
   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString axline = "";
   TString byline = "";
   TString czline = "";
   TString zcut1line = "";
   TString zcut2line = "";
   TString retunit;
   
   retunit = GetScale(lunit);
   
   axline = TString::Format("%s*%s", ax.Data(), retunit.Data());
   byline = TString::Format("%s*%s", by.Data(), retunit.Data());
   czline = TString::Format("%s*%s", cz.Data(), retunit.Data());
   Double_t radius = Evaluate(czline);
   Double_t dx = Evaluate(axline);
   Double_t dy = Evaluate(byline);
   Double_t sx = dx/radius;
   Double_t sy = dy/radius;
   Double_t sz = 1.;
   zcut1line = TString::Format("%s*%s", zcut1.Data(), retunit.Data());
   zcut2line = TString::Format("%s*%s", zcut2.Data(), retunit.Data());
   Double_t z1 = Evaluate(zcut1line);
   Double_t z2 = Evaluate(zcut2line);

   TGeoSphere *sph = new TGeoSphere(0,radius);
   TGeoScale *scl = new TGeoScale("",sx,sy,sz);
   TGeoScaledShape *shape = new TGeoScaledShape(NameShort(name), sph, scl);
   if(z1==0.0 && z2==0.0)
   {
      fsolmap[name.Data()] = shape;
   }
   else
   {
      Double_t origin[3] = {0.,0.,0.};
      origin[2] = 0.5*(z1+z2);
      Double_t dz = 0.5*(z2-z1);
      TGeoBBox *pCutBox = new TGeoBBox("cutBox", dx, dy, dz, origin);
      TGeoBoolNode *pBoolNode = new TGeoIntersection(shape,pCutBox,0,0);
      TGeoCompositeShape *cs = new TGeoCompositeShape(NameShort(name), pBoolNode);
      fsolmap[name.Data()] = cs;
   }
   
   return node;
   
}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Paraboloid(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, a Paraboloid may be declared. 
   //when the paraboloid keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoParaboloid and stored in fsolmap map using the name 
   //as its key.

   TString lunit = "mm"; 
   TString rlopos = "0"; 
   TString rhipos = "0"; 
   TString dzpos = "0"; 
   TString name = "";
   TString tempattr; 

   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rlo") { 
         rlopos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rhi"){
         rhipos = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "dz"){
         dzpos = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit"){
         lunit = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString rloline = "";
   TString rhiline = "";
   TString dzline = "";
   TString retunit;
   
   retunit = GetScale(lunit);
   
   rloline = TString::Format("%s*%s", rlopos.Data(), retunit.Data());
   rhiline = TString::Format("%s*%s", rhipos.Data(), retunit.Data());
   dzline = TString::Format("%s*%s", dzpos.Data(), retunit.Data());
   
   TGeoParaboloid* paraboloid = new TGeoParaboloid(NameShort(name),Evaluate(rloline),
                              Evaluate(rhiline),
                              Evaluate(dzline));
   
   fsolmap[name.Data()] = paraboloid;
   
   return node;
   
}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Arb8(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, an Arb8 may be declared. 
   //when the arb8 keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoArb8 and stored in fsolmap map using the name 
   //as its key.

   TString lunit = "mm"; 
   TString v1xpos = "0";
   TString v1ypos = "0";
   TString v2xpos = "0";
   TString v2ypos = "0";
   TString v3xpos = "0";
   TString v3ypos = "0";
   TString v4xpos = "0";
   TString v4ypos = "0";
   TString v5xpos = "0";
   TString v5ypos = "0";
   TString v6xpos = "0";
   TString v6ypos = "0";
   TString v7xpos = "0";
   TString v7ypos = "0";
   TString v8xpos = "0";
   TString v8ypos = "0"; 
   TString dzpos = "0"; 
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v1x") { 
         v1xpos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v1y") {
         v1ypos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v2x") { 
         v2xpos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v2y") {
         v2ypos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v3x") { 
         v3xpos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v3y") {
         v3ypos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v4x") { 
         v4xpos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v4y") {
         v4ypos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v5x") { 
         v5xpos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v5y") {
         v5ypos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v6x") { 
         v6xpos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v6y") {
         v6ypos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v7x") { 
         v7xpos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v7y") {
         v7ypos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v8x") { 
         v8xpos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "v8y") {
         v8ypos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "dz") { 
         dzpos = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit") {
         lunit = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString v1xline = ""; 
   TString v1yline = "";
   TString v2xline = "";
   TString v2yline   = "";
   TString v3xline = "";
   TString v3yline = "";
   TString v4xline = "";
   TString v4yline   = "";
   TString v5xline = "";
   TString v5yline = "";
   TString v6xline = "";
   TString v6yline   = "";
   TString v7xline = "";
   TString v7yline = "";
   TString v8xline = "";
   TString v8yline   = "";
   TString dzline = "";
   
   TString retunit;
   
   retunit = GetScale(lunit);
   
   v1xline = TString::Format("%s*%s", v1xpos.Data(), retunit.Data());
   v1yline = TString::Format("%s*%s", v1ypos.Data(), retunit.Data());
   v2xline = TString::Format("%s*%s", v2xpos.Data(), retunit.Data());
   v2yline = TString::Format("%s*%s", v2ypos.Data(), retunit.Data());
   v3xline = TString::Format("%s*%s", v3xpos.Data(), retunit.Data());
   v3yline = TString::Format("%s*%s", v3ypos.Data(), retunit.Data());
   v4xline = TString::Format("%s*%s", v4xpos.Data(), retunit.Data());
   v4yline = TString::Format("%s*%s", v4ypos.Data(), retunit.Data());
   v5xline = TString::Format("%s*%s", v5xpos.Data(), retunit.Data());
   v5yline = TString::Format("%s*%s", v5ypos.Data(), retunit.Data());
   v6xline = TString::Format("%s*%s", v6xpos.Data(), retunit.Data());
   v6yline = TString::Format("%s*%s", v6ypos.Data(), retunit.Data());
   v7xline = TString::Format("%s*%s", v7xpos.Data(), retunit.Data());
   v7yline = TString::Format("%s*%s", v7ypos.Data(), retunit.Data());
   v8xline = TString::Format("%s*%s", v8xpos.Data(), retunit.Data());
   v8yline = TString::Format("%s*%s", v8ypos.Data(), retunit.Data());
   dzline  = TString::Format("%s*%s", dzpos.Data(),  retunit.Data());

   
   TGeoArb8* arb8 = new TGeoArb8(NameShort(name), Evaluate(dzline));

   arb8->SetVertex(0, Evaluate(v1xline),Evaluate(v1yline));
   arb8->SetVertex(1, Evaluate(v2xline),Evaluate(v2yline));
   arb8->SetVertex(2, Evaluate(v3xline),Evaluate(v3yline));
   arb8->SetVertex(3, Evaluate(v4xline),Evaluate(v4yline));
   arb8->SetVertex(4, Evaluate(v5xline),Evaluate(v5yline));
   arb8->SetVertex(5, Evaluate(v6xline),Evaluate(v6yline));
   arb8->SetVertex(6, Evaluate(v7xline),Evaluate(v7yline));
   arb8->SetVertex(7, Evaluate(v8xline),Evaluate(v8yline));                   

   fsolmap[name.Data()] = arb8;
   
   return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Tube(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, a Tube may be declared. 
   //when the tube keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoTubeSeg and stored in fsolmap map using the name 
   //as its key.

   TString lunit = "mm"; 
   TString aunit = "rad";
   TString rmin = "0"; 
   TString rmax = "0"; 
   TString z = "0";
   TString startphi = "0";
   TString deltaphi = "0";
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rmin") { 
         rmin = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rmax") {
         rmax = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "z") {
         z = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit") {
         lunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "aunit") {
         aunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "startphi") {
         startphi = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "deltaphi") {
         deltaphi = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }
   
   TString rminline = "";
   TString rmaxline= "";
   TString zline = "";
   TString startphiline = "";
   TString deltaphiline = "";
   
   TString retlunit; 
   TString retaunit;

   retlunit = GetScale(lunit);
   retaunit = GetScale(aunit);

   rminline = TString::Format("%s*%s", rmin.Data(), retlunit.Data());
   rmaxline = TString::Format("%s*%s", rmax.Data(), retlunit.Data());
   zline = TString::Format("%s*%s", z.Data(), retlunit.Data());
   startphiline = TString::Format("%s*%s", startphi.Data(), retaunit.Data());
   deltaphiline = TString::Format("(%s*%s) + %s", deltaphi.Data(), retaunit.Data(), startphiline.Data());

   TGeoTubeSeg* tube = new TGeoTubeSeg(NameShort(name),Evaluate(rminline),
                           Evaluate(rmaxline),
                           Evaluate(zline)/2, 
                           Evaluate(startphiline),
                           Evaluate(deltaphiline));

   fsolmap[name.Data()] = tube;
   
   return node;
   
}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::CutTube(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, a Cut Tube may be declared. 
   //when the cutTube keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoCtub and stored in fsolmap map using the name 
   //as its key.

   TString lunit = "mm"; 
   TString aunit = "rad";
   TString rmin = "0"; 
   TString rmax = "0"; 
   TString z = "0";
   TString startphi = "0";
   TString deltaphi = "0";
   TString lowX = "0";
   TString lowY = "0";
   TString lowZ = "0";
   TString highX = "0";
   TString highY = "0";
   TString highZ = "0";
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rmin") { 
         rmin = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rmax"){
         rmax = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "z") {
         z = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit") {
         lunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "aunit") {
         aunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "startphi") {
         startphi = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "deltaphi") {
         deltaphi = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lowx") {
         lowX = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lowy") {
         lowY = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lowz") {
         lowZ = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "highx") {
         highX = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "highy") {
         highY = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "highz") {
         highZ = gdml->GetAttrValue(attr);
      }

      attr = gdml->GetNextAttr(attr);   
   }

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString rminline = "";
   TString rmaxline = "";
   TString zline = "";
   TString startphiline = "";
   TString deltaphiline = "";
   TString lowXline = "";
   TString lowYline = "";
   TString lowZline = "";
   TString highXline = "";
   TString highYline = "";
   TString highZline = "";

   TString retlunit; 
   TString retaunit;
   
   retlunit = GetScale(lunit);
   retaunit = GetScale(aunit);
   
   rminline = TString::Format("%s*%s", rmin.Data(), retlunit.Data());
   rmaxline = TString::Format("%s*%s", rmax.Data(), retlunit.Data());
   zline = TString::Format("%s*%s", z.Data(), retlunit.Data());
   startphiline = TString::Format("%s*%s", startphi.Data(), retaunit.Data());
   deltaphiline = TString::Format("(%s*%s) + %s", deltaphi.Data(), retaunit.Data(), startphiline.Data());
   lowXline = TString::Format("%s*%s", lowX.Data(), retlunit.Data());
   lowYline = TString::Format("%s*%s", lowY.Data(), retlunit.Data());
   lowZline = TString::Format("%s*%s", lowZ.Data(), retlunit.Data());
   highXline = TString::Format("%s*%s", highX.Data(), retlunit.Data());
   highYline = TString::Format("%s*%s", highY.Data(), retlunit.Data());
   highZline = TString::Format("%s*%s", highZ.Data(), retlunit.Data());

   
   TGeoCtub* cuttube = new TGeoCtub(NameShort(name),Evaluate(rminline),
                      Evaluate(rmaxline),
                      Evaluate(zline)/2, 
                      Evaluate(startphiline),
                      Evaluate(deltaphiline),
                      Evaluate(lowXline),
                      Evaluate(lowYline),
                      Evaluate(lowZline),
                      Evaluate(highXline),
                      Evaluate(highYline),
                      Evaluate(highZline));
   

   fsolmap[name.Data()] = cuttube;
   
   return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Cone(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, a cone may be declared. 
   //when the cone keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoConSeg and stored in fsolmap map using the name 
   //as its key.

   TString lunit = "mm"; 
   TString aunit = "rad";
   TString rmin1 = "0"; 
   TString rmax1 = "0"; 
   TString rmin2 = "0"; 
   TString rmax2 = "0"; 
   TString z = "0";
   TString startphi = "0";
   TString deltaphi = "0";
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rmin1") { 
         rmin1 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rmax1"){
         rmax1 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rmin2") {
         rmin2 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rmax2"){
         rmax2 = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "z"){
         z = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit"){
         lunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "aunit"){
         aunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "startphi"){
         startphi = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "deltaphi"){
         deltaphi = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString rmin1line = "";
   TString rmax1line = "";
   TString rmin2line = "";
   TString rmax2line = "";
   TString zline = "";
   TString startphiline = "";
   TString deltaphiline = "";
   TString retlunit; 
   TString retaunit;
   
   retlunit = GetScale(lunit);
   retaunit = GetScale(aunit);
   
   rmin1line = TString::Format("%s*%s", rmin1.Data(), retlunit.Data());
   rmax1line = TString::Format("%s*%s", rmax1.Data(), retlunit.Data());
   rmin2line = TString::Format("%s*%s", rmin2.Data(), retlunit.Data());
   rmax2line = TString::Format("%s*%s", rmax2.Data(), retlunit.Data());
   zline = TString::Format("%s*%s", z.Data(), retlunit.Data());
   startphiline = TString::Format("%s*%s", startphi.Data(), retaunit.Data());
   deltaphiline = TString::Format("%s*%s", deltaphi.Data(), retaunit.Data());
   Double_t sphi = Evaluate(startphiline);
   Double_t ephi = sphi + Evaluate(deltaphiline);

      
   TGeoConeSeg* cone = new TGeoConeSeg(NameShort(name),Evaluate(zline)/2,
                           Evaluate(rmin1line),
                           Evaluate(rmax1line),
                           Evaluate(rmin2line),
                           Evaluate(rmax2line),
                           sphi, ephi);
                   
   fsolmap[name.Data()] = cone;
   
   return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Trap(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, a Trap may be declared. 
   //when the trap keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoTrap and stored in fsolmap map using the name 
   //as its key.

   TString lunit = "mm"; 
   TString aunit = "rad";
   TString x1 = "0"; 
   TString x2 = "0"; 
   TString x3 = "0"; 
   TString x4 = "0"; 
   TString y1 = "0"; 
   TString y2 = "0";
   TString z = "0";
   TString phi = "0";
   TString theta = "0";
   TString alpha1 = "0"; 
   TString alpha2 = "0"; 
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "x1") {
         x1 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "x2") {
         x2 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "x3") { 
         x3 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "x4") {
         x4 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "y1") { 
         y1 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "y2") {
         y2 = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "z") {
         z = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit") {
         lunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "aunit"){
         aunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "phi"){
         phi = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "theta"){
         theta = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "alpha1") { 
         alpha1 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "alpha2"){
         alpha2 = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString x1line = ""; 
   TString x2line = ""; 
   TString x3line    = ""; 
   TString x4line = ""; 
   TString y1line = ""; 
   TString y2line = ""; 
   TString zline = ""; 
   TString philine = ""; 
   TString thetaline = ""; 
   TString alpha1line = ""; 
   TString alpha2line = ""; 
   TString retlunit; 
   TString retaunit;
   
   retlunit = GetScale(lunit);
   retaunit = GetScale(aunit);
   
   x1line = TString::Format("%s*%s", x1.Data(), retlunit.Data());
   x2line = TString::Format("%s*%s", x2.Data(), retlunit.Data());
   x3line = TString::Format("%s*%s", x3.Data(), retlunit.Data());
   x4line = TString::Format("%s*%s", x4.Data(), retlunit.Data());
   y1line = TString::Format("%s*%s", y1.Data(), retlunit.Data());
   y2line = TString::Format("%s*%s", y2.Data(), retlunit.Data());
   zline = TString::Format("%s*%s", z.Data(), retlunit.Data());
   philine = TString::Format("%s*%s", phi.Data(), retaunit.Data());
   thetaline = TString::Format("%s*%s", theta.Data(), retaunit.Data());
   alpha1line = TString::Format("%s*%s", alpha1.Data(), retaunit.Data());
   alpha2line = TString::Format("%s*%s", alpha2.Data(), retaunit.Data());
   
   TGeoTrap* trap = new TGeoTrap(NameShort(name),Evaluate(zline)/2,
                  Evaluate(thetaline),
                  Evaluate(philine),
                  Evaluate(y1line)/2,
                  Evaluate(x1line)/2,
                  Evaluate(x2line)/2,
                  Evaluate(alpha1line),
                  Evaluate(y2line)/2,
                  Evaluate(x3line)/2,
                  Evaluate(x4line)/2,
                  Evaluate(alpha2line));
   
   fsolmap[name.Data()] = trap;
   
   return node;
   
}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Trd(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, a Trd may be declared. 
   //when the trd keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoTrd2 and stored in fsolmap map using the name 
   //as its key.

   TString lunit = "mm"; 
   TString x1 = "0"; 
   TString x2 = "0";   
   TString y1 = "0"; 
   TString y2 = "0";
   TString z = "0";
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "x1") { 
         x1 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "x2"){
         x2 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "y1") { 
         y1 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "y2"){
         y2 = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "z"){
         z = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit"){
         lunit = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }
   
   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString x1line = "";
   TString x2line = "";
   TString y1line = "";
   TString y2line = "";
   TString zline = "";
   TString retlunit; 
   
   retlunit = GetScale(lunit);
   
   x1line = TString::Format("%s*%s", x1.Data(), retlunit.Data());
   x2line = TString::Format("%s*%s", x2.Data(), retlunit.Data());
   y1line = TString::Format("%s*%s", y1.Data(), retlunit.Data());
   y2line = TString::Format("%s*%s", y2.Data(), retlunit.Data());
   zline = TString::Format("%s*%s", z.Data(), retlunit.Data());
   
   TGeoTrd2* trd = new TGeoTrd2(NameShort(name),
                        Evaluate(x1line)/2,
                        Evaluate(x2line)/2,
                        Evaluate(y1line)/2,
                        Evaluate(y2line)/2,
                        Evaluate(zline)/2);
                   
   fsolmap[name.Data()] = trd;
   
   return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Polycone(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, a Polycone may be declared. 
   //when the polycone keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoPCon and stored in fsolmap map using the name 
   //as its key. Polycone has Zplanes, planes along the z axis specifying 
   //the rmin, rmax dimenstions at that point along z.

   TString lunit = "mm"; 
   TString aunit = "rad";
   TString rmin = "0"; 
   TString rmax = "0"; 
   TString z = "0";
   TString startphi = "0";
   TString deltaphi = "0";
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      } else if (tempattr == "lunit") {
         lunit = gdml->GetAttrValue(attr);
      } else if (tempattr == "aunit") {
         aunit = gdml->GetAttrValue(attr);
      } else if (tempattr == "startphi") {
         startphi = gdml->GetAttrValue(attr);
      } else if (tempattr == "deltaphi") {
         deltaphi = gdml->GetAttrValue(attr);
      }
      attr = gdml->GetNextAttr(attr);   
   } 
   
   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString retlunit; 
   TString retaunit;
   
   retlunit = GetScale(lunit);
   retaunit = GetScale(aunit);
   
   //START TO LOOK THRU CHILD (ZPLANE) NODES... 

   XMLNodePointer_t child = gdml->GetChild(node);
   int numplanes = 0;

   while (child!=0){
      numplanes = numplanes + 1;
      child = gdml->GetNext(child);
   }
   
   int cols;
   int i;
   cols = 3;
   double ** table = new double*[numplanes];
   for(i = 0; i < numplanes; i++) {
      table[i] = new double[cols];
   }
   
   child = gdml->GetChild(node);
   int planeno = 0;
   
   while (child!=0) {
      if(strcmp(gdml->GetNodeName(child), "zplane")==0) {
         //removed original dec
         TString rminline = "";
         TString rmaxline = "";
         TString zline = "";
         
         attr = gdml->GetFirstAttr(child);
         
         while (attr!=0) {
            tempattr = gdml->GetAttrName(attr);
            tempattr.ToLower();
       
            if(tempattr == "rmin") { 
               rmin = gdml->GetAttrValue(attr);
               rminline = TString::Format("%s*%s", rmin.Data(), retlunit.Data());
               table[planeno][0] = Evaluate(rminline);
            } else if(tempattr == "rmax") {
               rmax = gdml->GetAttrValue(attr);
               rmaxline = TString::Format("%s*%s", rmax.Data(), retlunit.Data());
               table[planeno][1] = Evaluate(rmaxline);
            } else if (tempattr == "z") {
               z = gdml->GetAttrValue(attr);
               zline = TString::Format("%s*%s", z.Data(), retlunit.Data());
               table[planeno][2] = Evaluate(zline);
            }
            attr = gdml->GetNextAttr(attr);
         }
      }
      planeno = planeno + 1;
      child = gdml->GetNext(child);
   }
   
   TString startphiline = "";
   TString deltaphiline = "";
   
   startphiline = TString::Format("%s*%s", startphi.Data(), retaunit.Data());
   deltaphiline = TString::Format("%s*%s", deltaphi.Data(), retaunit.Data());

   TGeoPcon* poly = new TGeoPcon(NameShort(name), 
                  Evaluate(startphiline), 
                  Evaluate(deltaphiline), 
                  numplanes);
   Int_t zno = 0;
   
   for (int j = 0; j < numplanes; j++) {
      poly->DefineSection(zno, table[j][2], table[j][0], table[j][1]);
      zno = zno + 1;
   }
   
   fsolmap[name.Data()] = poly;
   for(i = 0; i < numplanes; i++) {
      delete [] table[i];
   }
   delete [] table;
   
   return node;
}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Polyhedra(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, a Polyhedra may be declared. 
   //when the polyhedra keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoPgon and stored in fsolmap map using the name 
   //as its key. Polycone has Zplanes, planes along the z axis specifying 
   //the rmin, rmax dimenstions at that point along z.

   TString lunit = "mm"; 
   TString aunit = "rad";
   TString rmin = "0"; 
   TString rmax = "0"; 
   TString z = "0";
   TString startphi = "0";
   TString deltaphi = "0";
   TString numsides = "1";
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit"){
         lunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "aunit"){
         aunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "startphi"){
         startphi = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "deltaphi"){
         deltaphi = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "numsides"){
         numsides = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   } 
   
   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString retlunit; 
   TString retaunit;
   
   retlunit = GetScale(lunit);
   retaunit = GetScale(aunit);
   
   //START TO LOOK THRU CHILD (ZPLANE) NODES... 

   XMLNodePointer_t child = gdml->GetChild(node);
   int numplanes = 0;

   while (child!=0){
      numplanes = numplanes + 1;
      child = gdml->GetNext(child);
   }
   
   int cols;
   int i;
   cols = 3;
   double ** table = new double*[numplanes];
   for(i = 0; i < numplanes; i++){
      table[i] = new double[cols];
   }
   
   child = gdml->GetChild(node);
   int planeno = 0;
   
   while (child!=0) {
      if (strcmp(gdml->GetNodeName(child), "zplane")==0){
    
         TString rminline = "";
         TString rmaxline = "";
         TString zline = "";
         attr = gdml->GetFirstAttr(child);
    
         while (attr!=0){
            tempattr = gdml->GetAttrName(attr);
            tempattr.ToLower();
       
            if(tempattr == "rmin") { 
               rmin = gdml->GetAttrValue(attr);
               rminline = TString::Format("%s*%s", rmin.Data(), retlunit.Data());
               table[planeno][0] = Evaluate(rminline);
            }
            else if(tempattr == "rmax"){
               rmax = gdml->GetAttrValue(attr);
               rmaxline = TString::Format("%s*%s", rmax.Data(), retlunit.Data());
               table[planeno][1] = Evaluate(rmaxline);
            }
            else if (tempattr == "z"){
               z = gdml->GetAttrValue(attr);
               zline = TString::Format("%s*%s", z.Data(), retlunit.Data());
               table[planeno][2] = Evaluate(zline);
            }
       
            attr = gdml->GetNextAttr(attr);
         }
      }
      planeno = planeno + 1;
      child = gdml->GetNext(child);
   }
   
   TString startphiline = "";
   TString deltaphiline = "";
   TString numsidesline = "";

   startphiline = TString::Format("%s*%s", startphi.Data(), retaunit.Data());
   deltaphiline = TString::Format("%s*%s", deltaphi.Data(), retaunit.Data());
   numsidesline = TString::Format("%s", numsides.Data());

   TGeoPgon* polyg = new TGeoPgon(NameShort(name), 
                  Evaluate(startphiline), 
                  Evaluate(deltaphiline), 
                  (int)Evaluate(numsidesline),
                  numplanes);
   Int_t zno = 0;

   for (int j = 0; j < numplanes; j++){
      polyg->DefineSection(zno, table[j][2], table[j][0], table[j][1]);
      zno = zno + 1;
   }
   
   fsolmap[name.Data()] = polyg;
   for(i = 0; i < numplanes; i++){
      delete [] table[i];
   }
   delete [] table;
   
   return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Sphere(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, a Sphere may be declared. 
   //when the sphere keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoSphere and stored in fsolmap map using the name 
   //as its key.

   TString lunit = "mm"; 
   TString aunit = "rad";
   TString rmin = "0"; 
   TString rmax = "0"; 
   TString startphi = "0";
   TString deltaphi = "0";
   TString starttheta = "0";
   TString deltatheta = "0";
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rmin") { 
         rmin = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rmax") {
         rmax = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit"){
         lunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "aunit"){
         aunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "startphi"){
         startphi = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "deltaphi"){
         deltaphi = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "starttheta"){
         starttheta = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "deltatheta"){
         deltatheta = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString rminline = "";
   TString rmaxline = "";
   TString startphiline = "";
   TString deltaphiline = "";
   TString startthetaline = "";
   TString deltathetaline = "";
   TString retlunit; 
   TString retaunit;
   
   retlunit = GetScale(lunit);
   retaunit = GetScale(aunit);
   
   rminline = TString::Format("%s*%s", rmin.Data(), retlunit.Data());
   rmaxline = TString::Format("%s*%s", rmax.Data(), retlunit.Data());
   startphiline = TString::Format("%s*%s", startphi.Data(), retaunit.Data());
   deltaphiline = TString::Format("(%s*%s) + %s", deltaphi.Data(), retaunit.Data(), startphiline.Data());
   startthetaline = TString::Format("%s*%s", starttheta.Data(), retaunit.Data());
   deltathetaline = TString::Format("(%s*%s) + %s", deltatheta.Data(), retaunit.Data(), startthetaline.Data()); 

   TGeoSphere* sphere = new TGeoSphere(NameShort(name),
                           Evaluate(rminline),
                           Evaluate(rmaxline),
                           Evaluate(startthetaline),
                           Evaluate(deltathetaline),
                           Evaluate(startphiline),
                           Evaluate(deltaphiline));
                   
   fsolmap[name.Data()] = sphere;
   
   return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Torus(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, a Torus may be declared. 
   //when the torus keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoTorus and stored in fsolmap map using the name 
   //as its key.
   
   TString lunit = "mm"; 
   TString aunit = "rad";
   TString rmin = "0"; 
   TString rmax = "0"; 
   TString rtor = "0";
   TString startphi = "0";
   TString deltaphi = "0";
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rmin") { 
         rmin = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rmax"){
         rmax = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "rtor"){
         rtor = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit"){
         lunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "aunit"){
         aunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "startphi"){
         startphi = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "deltaphi"){
         deltaphi = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString rminline = "";
   TString rmaxline = "";
   TString rtorline = "";
   TString startphiline = "";
   TString deltaphiline = "";
   TString retlunit; 
   TString retaunit;
   
   retlunit = GetScale(lunit);
   retaunit = GetScale(aunit);
   
   rminline = TString::Format("%s*%s", rmin.Data(), retlunit.Data());
   rmaxline = TString::Format("%s*%s", rmax.Data(), retlunit.Data());
   rtorline = TString::Format("%s*%s", rtor.Data(), retlunit.Data());
   startphiline = TString::Format("%s*%s", startphi.Data(), retaunit.Data());
   deltaphiline = TString::Format("%s*%s", deltaphi.Data(), retaunit.Data());

      
   TGeoTorus* torus = new TGeoTorus(NameShort(name),Evaluate(rtorline),
                           Evaluate(rminline),
                           Evaluate(rmaxline),
                           Evaluate(startphiline),
                           Evaluate(deltaphiline));
                   
   fsolmap[name.Data()] = torus;
   
   return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Hype(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{
   //In the solids section of the GDML file, a Hype may be declared. 
   //when the hype keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoHype and stored in fsolmap map using the name 
   //as its key.

   TString lunit = "mm"; 
   TString aunit = "rad";
   TString rmin = "0"; 
   TString rmax = "0"; 
   TString z = "0";
   TString inst = "0";
   TString outst = "0";
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rmin") { 
         rmin = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rmax"){
         rmax = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "z"){
         z = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit"){
         lunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "aunit"){
         aunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "inst"){
         inst = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "outst"){
         outst = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString rminline = "";
   TString rmaxline = "";
   TString zline = "";
   TString instline = "";
   TString outstline = "";
   TString retlunit; 
   TString retaunit;
   
   retlunit = GetScale(lunit);
   retaunit = GetScale(aunit);
   
   rminline = TString::Format("%s*%s", rmin.Data(), retlunit.Data());
   rmaxline = TString::Format("%s*%s", rmax.Data(), retlunit.Data());
   zline = TString::Format("%s*%s", z.Data(), retlunit.Data());
   instline = TString::Format("%s*%s", inst.Data(), retaunit.Data());
   outstline = TString::Format("%s*%s", outst.Data(), retaunit.Data());

      
   TGeoHype* hype = new TGeoHype(NameShort(name),
                  Evaluate(rminline),
                  Evaluate(instline),
                  Evaluate(rmaxline),
                  Evaluate(outstline),
                  Evaluate(zline)/2);
                   
   fsolmap[name.Data()] = hype;
   
   return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Para(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{
   //In the solids section of the GDML file, a Para may be declared. 
   //when the para keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoPara and stored in fsolmap map using the name 
   //as its key.

   TString lunit = "mm"; 
   TString aunit = "rad";
   TString x = "0"; 
   TString y = "0"; 
   TString z = "0";
   TString phi = "0";
   TString theta = "0";
   TString alpha = "0";
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "x") {
         x = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "y"){
         y = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "z"){
         z = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit"){
         lunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "aunit"){
         aunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "phi"){
         phi = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "theta"){
         theta = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "alpha"){
         alpha = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString xline = "";
   TString yline = "";
   TString zline = "";
   TString philine = "";
   TString alphaline = "";
   TString thetaline = "";
   TString retlunit = ""; 
   TString retaunit = "";
   
   retlunit = GetScale(lunit);
   retaunit = GetScale(aunit);
   
   xline = TString::Format("%s*%s", x.Data(), retlunit.Data());
   yline = TString::Format("%s*%s", y.Data(), retlunit.Data());
   zline = TString::Format("%s*%s", z.Data(), retlunit.Data());
   philine = TString::Format("%s*%s", phi.Data(), retaunit.Data());
   alphaline = TString::Format("%s*%s", alpha.Data(), retaunit.Data());
   thetaline = TString::Format("%s*%s", theta.Data(), retaunit.Data());


   TGeoPara* para = new TGeoPara(NameShort(name),
                  Evaluate(xline)/2,
                  Evaluate(yline)/2,
                  Evaluate(zline)/2,
                  Evaluate(alphaline),
                  Evaluate(thetaline),
                  Evaluate(philine));
                   
   fsolmap[name.Data()] = para;
   
   return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::TwistTrap(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, a TwistTrap may be declared. 
   //when the twistedtrap keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoGTra and stored in fsolmap map using the name 
   //as its key.

   TString lunit = "mm"; 
   TString aunit = "rad";
   TString x1 = "0"; 
   TString x2 = "0"; 
   TString x3 = "0"; 
   TString x4 = "0"; 
   TString y1 = "0"; 
   TString y2 = "0";
   TString z = "0";
   TString phi = "0";
   TString theta = "0";
   TString alpha1 = "0"; 
   TString alpha2 = "0"; 
   TString twist = "0";
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "x1") { 
         x1 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "x2"){
         x2 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "x3") { 
         x3 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "x4"){
         x4 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "y1") { 
         y1 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "y2"){
         y2 = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "z"){
         z = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit"){
         lunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "aunit"){
         aunit = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "phi"){
         phi = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "theta") {
         theta = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "alpha1")   { 
         alpha1 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "alpha2"){
         alpha2 = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "phitwist") {
         twist = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }
 
   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString x1line = "";
   TString x2line = "";
   TString x3line = "";
   TString x4line = "";
   TString y1line = "";
   TString y2line = "";
   TString zline = "";
   TString philine = "";
   TString thetaline = "";
   TString alpha1line = "";
   TString alpha2line = "";
   TString twistline = "";
   TString retlunit; 
   TString retaunit;
   
   retlunit = GetScale(lunit);
   retaunit = GetScale(aunit);
   
   x1line = TString::Format("%s*%s", x1.Data(), retlunit.Data());
   x2line = TString::Format("%s*%s", x2.Data(), retlunit.Data());
   x3line = TString::Format("%s*%s", x3.Data(), retlunit.Data());
   x4line = TString::Format("%s*%s", x4.Data(), retlunit.Data());
   y1line = TString::Format("%s*%s", y1.Data(), retlunit.Data());
   y2line = TString::Format("%s*%s", y2.Data(), retlunit.Data());
   zline  = TString::Format("%s*%s", z.Data(), retlunit.Data());
   philine = TString::Format("%s*%s", phi.Data(), retaunit.Data());
   thetaline = TString::Format("%s*%s", theta.Data(), retaunit.Data());
   alpha1line = TString::Format("%s*%s", alpha1.Data(), retaunit.Data());
   alpha2line = TString::Format("%s*%s", alpha2.Data(), retaunit.Data());
   twistline = TString::Format("%s*%s", twist.Data(), retaunit.Data());

      
   TGeoGtra* twtrap = new TGeoGtra(NameShort(name),Evaluate(zline)/2,
                  Evaluate(thetaline),
                  Evaluate(philine),
                  Evaluate(twistline),
                  Evaluate(y1line)/2,
                  Evaluate(x1line)/2,
                  Evaluate(x2line)/2,
                  Evaluate(alpha1line),
                  Evaluate(y2line)/2,
                  Evaluate(x3line)/2,
                  Evaluate(x4line)/2,
                  Evaluate(alpha2line));
                   
   fsolmap[name.Data()] = twtrap;
   
   return node;

}


//___________________________________________________________________
XMLNodePointer_t TGDMLParse::ElTube(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, a ElTube may be declared. 
   //when the eltube keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoEltu and stored in fsolmap map using the name 
   //as its key.

   TString lunit = "mm"; 
   TString xpos = "0"; 
   TString ypos = "0"; 
   TString zpos = "0"; 
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "dx") { 
         xpos = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "dy") {
         ypos = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "dz"){
         zpos = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit"){
         lunit = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString xline = "";
   TString yline = "";
   TString zline = "";
   TString retunit;
   
   retunit = GetScale(lunit);
   
   xline = TString::Format("%s*%s", xpos.Data(), retunit.Data());
   yline = TString::Format("%s*%s", ypos.Data(), retunit.Data());
   zline = TString::Format("%s*%s", zpos.Data(), retunit.Data());

   TGeoEltu* eltu = new TGeoEltu(NameShort(name),Evaluate(xline),
                        Evaluate(yline),
                        Evaluate(zline));

   fsolmap[name.Data()] = eltu;
   
   return node;

}
//___________________________________________________________________
XMLNodePointer_t TGDMLParse::Orb(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, an Orb may be declared. 
   //when the orb keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoSphere and stored in fsolmap map using the name 
   //as its key.

   TString lunit = "mm"; 
   TString r = "0"; 
   TString name = "";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "r") { 
         r = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit"){
         lunit = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   }

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString rline = "";
   TString retunit;
   
   retunit = GetScale(lunit);
   
   rline = TString::Format("%s*%s", r.Data(), retunit.Data());
   
   TGeoSphere* orb = new TGeoSphere(NameShort(name), 0, Evaluate(rline), 0, 180, 0, 360);

   fsolmap[name.Data()] = orb;
   
   return node;

}


//_______________________________________________________
XMLNodePointer_t TGDMLParse::Xtru(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{   
   //In the solids section of the GDML file, an Xtru may be declared. 
   //when the xtru keyword is found, this function is called, and the 
   //dimensions required are taken and stored, these are then bound and
   //converted to type TGeoXtru and stored in fsolmap map using the name 
   //as its key. The xtru has child nodes of either 'twoDimVertex'or 
   //'section'.   These two nodes define the real structure of the shape.
   //The twoDimVertex's define the x,y sizes of a vertice. The section links
   //the vertice to a position within the xtru. 

   TString lunit = "mm"; 
//   TString aunit = "rad";
   TString x = "0"; 
   TString y = "0"; 
   TString zorder = "0";
   TString zpos = "0";
   TString xoff = "0";
   TString yoff = "0";
   TString scale = "0";
   TString name = "";
   TString tempattr; 

   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "lunit"){
         lunit = gdml->GetAttrValue(attr);
      }
      
      attr = gdml->GetNextAttr(attr);   
   } 

   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }

   TString retlunit; 
   
   retlunit = GetScale(lunit);
   
   //START TO LOOK THRU CHILD NODES... 

   XMLNodePointer_t child = gdml->GetChild(node);
   int nosects = 0;
   int noverts = 0;

   while (child!=0){
      tempattr = gdml->GetNodeName(child);
      
      if(tempattr == "twoDimVertex"){
         noverts = noverts + 1;
      }
      else if(tempattr == "section"){ 
         nosects = nosects + 1;
      }
      
      child = gdml->GetNext(child);
   }
   
   //Build the dynamic arrays..
   int cols;
   int i;
   double *vertx = new double[noverts];
   double *verty = new double[noverts];
   cols = 5;
   double ** section = new double*[nosects];
   for(i = 0; i < nosects; i++){
      section[i] = new double[cols];
   }
   
   child = gdml->GetChild(node);
   int sect = 0;
   int vert = 0;

   while (child!=0) {
      if(strcmp(gdml->GetNodeName(child), "twoDimVertex")==0){
         TString xline = ""; 
         TString yline = "";
         
         attr = gdml->GetFirstAttr(child);
         
         while (attr!=0){
            tempattr = gdml->GetAttrName(attr);
    
            if(tempattr == "x") { 
               x = gdml->GetAttrValue(attr);
               xline = TString::Format("%s*%s", x.Data(), retlunit.Data());
               vertx[vert] = Evaluate(xline);
            }
            else if(tempattr == "y"){
               y = gdml->GetAttrValue(attr);
               yline = TString::Format("%s*%s", y.Data(), retlunit.Data());
               verty[vert] = Evaluate(yline);
            }
       
            attr = gdml->GetNextAttr(attr);
         }

         vert = vert + 1;
      }
      
      else if(strcmp(gdml->GetNodeName(child), "section") == 0){

         TString zposline = "";
         TString xoffline = "";
         TString yoffline = "";
         
         attr = gdml->GetFirstAttr(child);
         
         while (attr!=0){
            tempattr = gdml->GetAttrName(attr);
    
            if(tempattr == "zOrder") { 
               zorder = gdml->GetAttrValue(attr);
               section[sect][0] = Evaluate(zorder);
            }
            else if(tempattr == "zPosition"){
               zpos = gdml->GetAttrValue(attr);
               zposline = TString::Format("%s*%s", zpos.Data(), retlunit.Data());
               section[sect][1] = Evaluate(zposline);
            }
            else if (tempattr == "xOffset"){
               xoff = gdml->GetAttrValue(attr);
               xoffline = TString::Format("%s*%s", xoff.Data(), retlunit.Data());
               section[sect][2] = Evaluate(xoffline);
            }
            else if (tempattr == "yOffset"){
               yoff = gdml->GetAttrValue(attr);
               yoffline = TString::Format("%s*%s", yoff.Data(), retlunit.Data());
               section[sect][3] = Evaluate(yoffline);
            }
            else if (tempattr == "scalingFactor"){
               scale = gdml->GetAttrValue(attr);
               section[sect][4] = Evaluate(scale);
            }
       
            attr = gdml->GetNextAttr(attr);
         }
		
         sect = sect + 1; 
      }      
      child = gdml->GetNext(child);
   }
   
   TGeoXtru* xtru = new TGeoXtru(nosects);
   xtru->SetName(NameShort(name));
   xtru->DefinePolygon(vert, vertx, verty);
   
   for (int j = 0; j < sect; j++){
      xtru->DefineSection((int)section[j][0], section[j][1], section[j][2], section[j][3], section[j][4]);
   }
  
   fsolmap[name.Data()] = xtru;
   delete [] vertx;
   delete [] verty;
   for(i = 0; i < nosects; i++){
      delete [] section[i];
   }
   delete [] section;
   return node;
}

//____________________________________________________________
XMLNodePointer_t TGDMLParse::Reflection(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr)
{
   //In the solids section of the GDML file, a Reflected Solid may be 
   //declared when the ReflectedSolid keyword is found, this function 
   //is called. The rotation, position and scale for the reflection are 
   //applied to a matrix that is then stored in the class object 
   //TGDMLRefl.   This is then stored in the map freflsolidmap, with 
   //the reflection name as a reference. also the name of the solid to 
   //be reflected is stored in a map called freflectmap with the reflection 
   //name as a reference.

   std::cout << "WARNING! The reflectedSolid is obsolete! Use scale transformation instead!" << std::endl;

   TString sx = "0"; 
   TString sy = "0"; 
   TString sz = "0"; 
   TString rx = "0"; 
   TString ry = "0"; 
   TString rz = "0"; 
   TString dx = "0"; 
   TString dy = "0"; 
   TString dz = "0"; 
   TString name = "0";
   TString solid = "0";
   TString tempattr; 
   
   while (attr!=0) {
      
      tempattr = gdml->GetAttrName(attr);
      tempattr.ToLower();
      
      if(tempattr == "name") { 
         name = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "sx") { 
         sx = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "sy"){
         sy = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "sz"){
         sz = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "rx") { 
         rx = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "ry"){
         ry = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "rz"){
         rz = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "dx") { 
         dx = gdml->GetAttrValue(attr);
      }
      else if(tempattr == "dy"){
         dy = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "dz"){
         dz = gdml->GetAttrValue(attr);
      }
      else if (tempattr == "solid"){
         solid = gdml->GetAttrValue(attr);
      }
      attr = gdml->GetNextAttr(attr);   
   }
   
   if((strcmp(fCurrentFile,fStartFile)) != 0){
      name = TString::Format("%s_%s", name.Data(), fCurrentFile);
   }
   if((strcmp(fCurrentFile,fStartFile)) != 0){
      solid = TString::Format("%s_%s", solid.Data(), fCurrentFile);
   }

   TGeoRotation* rot = new TGeoRotation();
   rot->RotateZ(-(Evaluate(rz)));
   rot->RotateY(-(Evaluate(ry)));
   rot->RotateX(-(Evaluate(rx)));

   if(atoi(sx) == -1) {
      rot->ReflectX(kTRUE);
   }
   if(atoi(sy) == -1) {
      rot->ReflectY(kTRUE);
   }
   if(atoi(sz) == -1) {
      rot->ReflectZ(kTRUE);
   }

   TGeoCombiTrans* relf_matx = new TGeoCombiTrans(Evaluate(dx), Evaluate(dy), Evaluate(dz), rot);

   TGDMLRefl* reflsol = new TGDMLRefl(NameShort(name), solid, relf_matx);
   freflsolidmap[name.Data()] = reflsol;
   freflectmap[name.Data()] = solid;

   return node;
}



//===================================================================

ClassImp(TGDMLRefl)

/******************************************************************
____________________________________________________________

TGDMLRefl Class

------------------------------------------------------------

This class is a helper class for TGDMLParse.   It assists in the 
reflection process.   This process takes a previously defined solid 
and can reflect the matrix of it. This class stores the name of the 
reflected solid, along with the name of the solid that is being 
reflected, and finally the reflected solid's matrix.   This is then 
recalled when the volume is used in the structure part of the gdml 
file.

******************************************************************/

//___________________________________________________________________
TGDMLRefl::TGDMLRefl(const char* name, const char* solid, TGeoMatrix* matrix)
{   
   //this constructor method stores the values brought in as params.

   fNameS = name;
   fSolid = solid;
   fMatrix = matrix; 
}

//_________________________________________________________________
TGeoMatrix* TGDMLRefl::GetMatrix()
{
   //this accessor method returns the matrix.

   return fMatrix;
}
