/* @(#)root/netx:$Name:  $:$Id: TGDMLParse.cxx,v 1.1 2006/11/17 17:40:02 brun Exp $ */
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

#ifndef ROOT_TGeoMatrix
#include "TGeoMatrix.h"
#endif

#ifndef ROOT_TXMLEngine
#include "TXMLEngine.h"
#endif

#ifndef ROOT_TGeoVolume
#include "TGeoVolume.h"
#endif

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
#include "TGeoVolume.h"
#include "TROOT.h"
#include "TMath.h"
#include "TFormula.h"
#include "TGeoMaterial.h"
#include "TGeoBoolNode.h"
#include "TGeoMedium.h"
#include "TGeoElement.h"
#include "TGeoShape.h"
#include "TGeoCompositeShape.h"
#include "TGDMLParse.h"

ClassImp(TGDMLParse)

//_________________________________________________________________
TGeoVolume* TGDMLParse::GDMLReadFile(const char* filename){

  //creates the new instance of the XMLEngine called 'gdml', using the filename 
  //then parses the file and creates the DOM tree. Then passes the DOM to the 
  //next funtion to translate it. 

   // First create engine
   TXMLEngine* gdml = new TXMLEngine;
   
   // Now try to parse xml file
   XMLDocPointer_t gdmldoc = gdml->ParseFile(filename);
   if (gdmldoc==0) {
      delete gdml;
      return 0;  
   }

   // take access to main node   
   XMLNodePointer_t mainnode = gdml->DocGetRootElement(gdmldoc);
   
   // display recursively all nodes and subnodes
     ParseGDML(gdml, mainnode, 1);


   // Release memory before exit
   gdml->FreeDoc(gdmldoc);
   delete gdml;
   
   return fWorld;

}

//________________________________________________________________
void TGDMLParse::ParseGDML(TXMLEngine* gdml, XMLNodePointer_t node, Int_t level) {

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
  const char* rotastr = "rotation";
  const char* elemstr = "element";
  const char* isotstr = "isotope";
  const char* matestr = "material";
  const char* volustr = "volume";
  const char* assestr = "assembly";
  const char* twtrstr = "twistTrap";
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
  
  if ((strcmp(name, posistr)) == 0){ 
    node = PosProcess(gdml, node, attr);
  }
  else if ((strcmp(name, rotastr)) == 0){ 
    node = RotProcess(gdml, node, attr);
  } 
  else if ((strcmp(name, setustr)) == 0){ 
    node = TopProcess(gdml, node);
  } 
  else if ((strcmp(name, consstr)) == 0){ 
    node = ConProcess(gdml, node, attr);
  }
  else if (((strcmp(name, "atom")) == 0) && ((strcmp(parent, elemstr)) == 0)){ 
    node = EleProcess(gdml, node, parentn);
  }
  else if ((strcmp(name, isotstr)) == 0){ 
    std::cout << "Isotope NOT supported by ROOT.." << std::endl;
  }
  else if ((strcmp(name, matestr)) == 0){ 
    if(gdml->HasAttr(node, "Z")){
      int z = 1;
      node = MatProcess(gdml, node, attr, z);
    }
    else{
      int z = 0;
      node = MatProcess(gdml, node, attr, z);
    }
  }
  
  else if ((strcmp(name, volustr)) == 0){ 
    node = VolProcess(gdml, node);
  }
  else if ((strcmp(name, bboxstr)) == 0){ 
    node = Box(gdml, node, attr);
  }
  else if ((strcmp(name, cutTstr)) == 0){ 
    node = CutTube(gdml, node, attr);
  }
  else if ((strcmp(name, arb8str)) == 0){ 
    node = Arb8(gdml, node, attr);
  }
  else if ((strcmp(name, tubestr)) == 0){ 
    node = Tube(gdml, node, attr);
  }
  else if ((strcmp(name, conestr)) == 0){ 
    node = Cone(gdml, node, attr);
  }
  else if ((strcmp(name, polystr)) == 0){ 
    node = Polycone(gdml, node, attr);
  }
  else if ((strcmp(name, trapstr)) == 0){ 
    node = Trap(gdml, node, attr);
  }
  else if ((strcmp(name, trdstr)) == 0){ 
    node = Trd(gdml, node, attr);
  }
  else if ((strcmp(name, sphestr)) == 0){ 
    node = Sphere(gdml, node, attr);
  } 
  else if ((strcmp(name, xtrustr)) == 0){ 
    node = Xtru(gdml, node, attr);
  }
  else if ((strcmp(name, twtrstr)) == 0){ 
    node = TwistTrap(gdml, node, attr);
  }
  else if ((strcmp(name, hypestr)) == 0){ 
    node = Hype(gdml, node, attr);
  }
  else if ((strcmp(name, orbstr)) == 0){ 
    node = Orb(gdml, node, attr);
  }
  else if ((strcmp(name, parastr)) == 0){ 
    node = Para(gdml, node, attr);
  }
  else if ((strcmp(name, torustr)) == 0){ 
    node = Torus(gdml, node, attr);
  }
  else if ((strcmp(name, eltustr)) == 0){ 
    node = ElTube(gdml, node, attr);
  }
  else if ((strcmp(name, hedrstr)) == 0){ 
    node = Polyhedra(gdml, node, attr);
  }
  else if ((strcmp(name, parbstr)) == 0){ 
    node = Paraboloid(gdml, node, attr); 
  }
  else if ((strcmp(name, subtstr)) == 0){ 
    node = BooSolid(gdml, node, attr, 1);
  }
  else if ((strcmp(name, intestr)) == 0){ 
    node = BooSolid(gdml, node, attr, 2);
  }
  else if ((strcmp(name, uniostr)) == 0){ 
    node = BooSolid(gdml, node, attr, 3);
  }
  else if ((strcmp(name, reflstr)) == 0){ 
    node = Reflection(gdml, node, attr);
  }
  else if ((strcmp(name, assestr)) == 0){ 
    node = AssProcess(gdml, node);
  }
  
  // Check for Child node - if present call this funct. recursively until no more
  
  XMLNodePointer_t child = gdml->GetChild(node);
  while (child!=0) {
    ParseGDML(gdml, child, level+2); 
    child = gdml->GetNext(child);
  }
}

//___________________________________________________________
double TGDMLParse::Evaluate(const char* evalline){

  //takes in a string containing a mathematical expression and
  //returns the result of the expression as a double.

  TFormula form;
  
  double result = 0;
  form.Compile(evalline);
  result = form.Eval(0);
  return result;
}

//____________________________________________________________
Int_t TGDMLParse::SetAxis(const char* axisString){

  //When using the 'divide' process in the geometry this function
  //sets the variable 'axis' depending on what is specified.

  Int_t axis = 0;

  if((strcmp(axisString, "kXAxis")) == 0){
    axis = 1;
  }
  else if((strcmp(axisString, "kYAxis")) == 0){
    axis = 2;
  }
  else if((strcmp(axisString, "kZAxis")) == 0){
    axis = 3;
  }
  else if((strcmp(axisString, "kRho")) == 0){
    axis = 1;
  }
  else if((strcmp(axisString, "kPhi")) == 0){
    axis = 2;
  }
  
  return axis;

}

//____________________________________________________________
const char* TGDMLParse::NameShort(const char* name){

  //this function looks thru a string for the chars '0x' next to
  //each other, when it finds this, it calls another function to strip
  //the hex address.  It does this recursively until the end of the 
  //string is reached, returning a string without any hex addresses.

  int len = strlen(name);
  int offset = 0;
  const char* newname = name;
  
  while(offset != len){
    if((name[offset] == '0') && (name[offset+1] == 'x')){
      newname = NameShortB(newname);
    }
    offset = offset + 1;
  }
  
  return newname;
}

//_____________________________________________________________
const char* TGDMLParse::NameShortB(const char* name){ 

  //this function is passed a string, and removes the first hex address
  //it finds.  This function is called recursively by NameShort to
  //fully strip a string of all hex addresses within it.

  char* shortname = NULL;
  const char* retname = NULL;  
  int char_offset = 0; /* 8 hex + 0x */
  int len = strlen(name);

  while (shortname == NULL && char_offset != len){
    if((name[char_offset] == '0') && (name[(char_offset+1)] == 'x')){
      
      shortname = new char[len];
      memcpy(shortname,name,char_offset);
      shortname[char_offset]='\0';
      
      const char *temp = &name[(char_offset + 10)];
      shortname = strcat(shortname, temp);
      retname = shortname;
    }
    else{
      retname = name;
    }          
    char_offset = char_offset + 1;
  }
  
  if(shortname == NULL){
    retname = name;
  }
  
  return retname;  
}

//____________________________________________________________
const char* TGDMLParse::FindConst(const char* retval){

  //If a value is given within the GDML file, this function checks
  //the constant map set up when all constants are declared.  If a matching
  //constant name is found, the corresponding value is returned, if not,
  //the original is returned.

  if(fconmap.find(retval) != fconmap.end()){
    retval = fconmap[retval];
  }
  return retval;
}

//________________________________________________________
XMLNodePointer_t TGDMLParse::ConProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){

  //In the define section of the GDML file, constants can be declared.
  //when the constant keyword is found, this function is called, and the
  //name and value of the constant is stored in fconmap map using the 
  //constant name as its key.

  const char* name = "";
  const char* value = "";
  const char* tempattr; 
  
  while (attr!=0) {
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    if((strcmp(tempattr, "value")) == 0) { 
	  value = gdml->GetAttrValue(attr);
    }    
    attr = gdml->GetNextAttr(attr);  
  }
  
  fconmap[name] = value;
  
  return node;
  
}
//__________________________________________________________
const char* TGDMLParse::GetScale(const char* unit){

  //Throughout the GDML file, a unit can de specified.  Whether it be
  //angular or linear, values can be used as well as abbreviations such as
  // 'mm' or 'deg'. This function is passed the specified unit and if it is 
  //found, replaces it with the appropriate value.
  
  const char* retunit = "";
  
  if(strcmp(unit, "mm") == 0){
    retunit = "1.0";
  }
  else if(strcmp(unit, "milimeter") == 0){
    retunit = "1.0";
  }
  else if(strcmp(unit, "m") == 0){
    retunit = "1000.0";
  }
  else if(strcmp(unit, "meter") == 0){
    retunit = "1000.0";
  }
  else if(strcmp(unit, "km") == 0){
    retunit = "1000000.0";
  }
  else if(strcmp(unit, "kilometer") == 0){
    retunit = "1000000.0";
  }
  else if(strcmp(unit, "rad") == 0){
    retunit = Form("%f", TMath::RadToDeg());
  }
  else if(strcmp(unit, "radian") == 0){
    retunit = Form("%f", TMath::RadToDeg());
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
    retunit = Form("%f", TMath::Na());
  }
  else{
    retunit = "0";
  }
  return retunit;
  
}

//____________________________________________________________
XMLNodePointer_t TGDMLParse::PosProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){

  //In the define section of the GDML file, positions can be declared.
  //when the position keyword is found, this function is called, and the
  //name and values of the position are converted into type TGeoPosition 
  //and stored in fposmap map using the name as its key. This function 
  //can also be called when declaring solids.
  
  const char* lunit = "mm"; 
  const char* xpos = "0"; 
  const char* ypos = "0"; 
  const char* zpos = "0"; 
  const char* name = "0";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "x")) == 0) { 
      xpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "y") == 0){
      ypos = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "z") == 0){
      zpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    
    attr = gdml->GetNextAttr(attr);  
  }

  const char* xline = "";
  const char* yline = "";
  const char* zline = "";
  const char* retunit;
  
  retunit = GetScale(lunit);
  
  xline = Form("%s*%s", xpos, retunit);
  yline = Form("%s*%s", ypos, retunit);
  zline = Form("%s*%s", zpos, retunit);
  
  TGeoTranslation* pos = new TGeoTranslation(Evaluate(xline),
					     Evaluate(yline),
					     Evaluate(zline));
  
  fposmap[name] = pos;
  
  return node;
  
}

//___________________________________________________________________
XMLNodePointer_t TGDMLParse::RotProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the define section of the GDML file, rotations can be declared.
  //when the rotation keyword is found, this function is called, and the
  //name and values of the rotation are converted into type TGeoRotation 
  //and stored in frotmap map using the name as its key. This function 
  //can also be called when declaring solids.

  const char* aunit = "deg"; 
  const char* xpos = "0"; 
  const char* ypos = "0"; 
  const char* zpos = "0"; 
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "x")) == 0) { 
      xpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "y") == 0){
      ypos = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "z") == 0){
      zpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "aunit") == 0){
      aunit = gdml->GetAttrValue(attr);
    }
    
    attr = gdml->GetNextAttr(attr);  
  }

  const char* xline = "";
  const char* yline = "";
  const char* zline = "";
  const char* retunit;
  
  retunit = GetScale(aunit);
  
  xline = Form("%s*%s", xpos, retunit);
  yline = Form("%s*%s", ypos, retunit);
  zline = Form("%s*%s", zpos, retunit);
  
  TGeoRotation* rot = new TGeoRotation();
  
  rot->RotateZ(-(Evaluate(zline)));
  rot->RotateY(-(Evaluate(yline)));
  rot->RotateX(-(Evaluate(xline)));
  
  frotmap[name] = rot;
  
  return node;
  
}

//___________________________________________________________
XMLNodePointer_t TGDMLParse::EleProcess(TXMLEngine* gdml, XMLNodePointer_t node,  XMLNodePointer_t parentn){
  
  //In the materials section of the GDML file, elements can be declared.
  //when the element keyword is found, this function is called, and the
  //name and values of the element are converted into type TGeoElement and
  //stored in felemap map using the name as its key.

  const char* Z = "0";
  const char* name = "";
  const char* formula = "";
  const char* atom = "0";
  const char* tempattr;
  
  //obtain attributes for the element
  
  XMLAttrPointer_t attr = gdml->GetFirstAttr(parentn);
  
  while (attr!=0){   
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "Z")) == 0) { 
      Z = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "formula") == 0) {
      formula = gdml->GetAttrValue(attr);
    }
    
    attr = gdml->GetNextAttr(attr);  
  }
  
  //get the atom value for the element
  
  attr = gdml->GetFirstAttr(node);
  
  while (attr!=0){    
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "value")) == 0) { 
      atom = FindConst(gdml->GetAttrValue(attr));
    }
    
    attr = gdml->GetNextAttr(attr);  
  }
  
  
  Int_t z2 = (Int_t)Evaluate(Z);
  Double_t atom2 = Evaluate(atom);
  
  TGeoElement* ele = new TGeoElement(NameShort(name), formula, z2 , atom2);
  
  felemap[name] = ele;
  
  return node;
}

//_________________________________________________________________________
XMLNodePointer_t TGDMLParse::MatProcess(TXMLEngine* gdml, XMLNodePointer_t node,  XMLAttrPointer_t attr, int z){
  
  //In the materials section of the GDML file, materials can be declared.
  //when the material keyword is found, this function is called, and the
  //name and values of the material are converted into type TGeoMaterial 
  //and stored in fmatmap map using the name as its key. Mixtures can also
  // be declared, and they are converted to TGeoMixture and stored in
  //fmixmap.  These mixtures and materials are then all converted into one
  //common type - TGeoMedium.  The map fmedmap is then built up of all the 
  //mixtures and materials.

 //!Map to hold fractions while being processed
  typedef FracMap::iterator fractions;
  FracMap fracmap;
  
  static int medid = 0;
  XMLNodePointer_t child = gdml->GetChild(node);
  const char* tempattr = "";
  int ncompo = 0, mixflag = 2;
  double density = 0;
  const char* name = "";
  TGeoMixture* mix = 0; 
  TGeoMaterial* mat = 0;
  const char* tempconst = "";
  
  if (z == 1){
    double a = 0;
    double d = 0;
    
    while (child!=0) {
      attr = gdml->GetFirstAttr(child);
      
      if((strcmp(gdml->GetNodeName(child), "atom")) == 0){
	while (attr!=0){  
	  tempattr = gdml->GetAttrName(attr);
	  
	  if((strcmp(tempattr, "value")) == 0) { 
	    a = Evaluate(FindConst(gdml->GetAttrValue(attr)));
	  }	  
	  attr = gdml->GetNextAttr(attr);
	}		
      }
      
      if((strcmp(gdml->GetNodeName(child), "D")) == 0){
	while (attr!=0){  
	  tempattr = gdml->GetAttrName(attr);
	  
	  if((strcmp(tempattr, "value")) == 0) { 
	    d = Evaluate(FindConst(gdml->GetAttrValue(attr)));  
	  }	  
	  attr = gdml->GetNextAttr(attr);
	} 
      }      
      child = gdml->GetNext(child);
    }
    
    //still in the is Z else...but not in the while..
    
    name = gdml->GetAttr(node, "name");
    
    //CHECK FOR CONSTANTS   
    tempconst = FindConst(gdml->GetAttr(node, "Z"));
   
    mat = new TGeoMaterial(NameShort(name), a, Evaluate(tempconst), d);    
    mixflag = 0;
    TGeoElement* mat_ele = new TGeoElement(NameShort(name), "", atoi(tempconst), a);
    felemap[name] = mat_ele;
  }
  
  else if (z == 0){
    while (child!=0) {
      attr = gdml->GetFirstAttr(child);
      
      if((strcmp(gdml->GetNodeName(child), "fraction")) == 0){
	double n = 0;
	const char* ref = ""; 
	ncompo = ncompo + 1;
	
	while (attr!=0){  
	  tempattr = gdml->GetAttrName(attr);
	  
	  if((strcmp(tempattr, "n")) == 0) { 
	    n = Evaluate(FindConst(gdml->GetAttrValue(attr)));
	  }
	  else if((strcmp(tempattr, "ref")) == 0) { 
	    ref = FindConst(gdml->GetAttrValue(attr));
	  }
	  
	  attr = gdml->GetNextAttr(attr);
	}
	
	fracmap[ref] = n; 
      }
      
      else if((strcmp(gdml->GetNodeName(child), "composite")) == 0){
	double n = 0;
	const char* ref = ""; 
	ncompo = ncompo + 1;
	
	while (attr!=0){  
	  tempattr = gdml->GetAttrName(attr);
	  
	  if((strcmp(tempattr, "n")) == 0) { 
	    n = Evaluate(FindConst(gdml->GetAttrValue(attr)));
	  }
	  else if((strcmp(tempattr, "ref")) == 0) { 
	    ref = FindConst(gdml->GetAttrValue(attr));
	  }
	  
	  attr = gdml->GetNextAttr(attr);
	}
	
	fracmap[ref] = n;
      }
      
      else if((strcmp(gdml->GetNodeName(child), "D")) == 0){
	while (attr!=0){      
	  tempattr = gdml->GetAttrName(attr);
	  
	  if((strcmp(tempattr, "value")) == 0) { 
	    density = Evaluate(FindConst(gdml->GetAttrValue(attr)));
	  }
	  
	  attr = gdml->GetNextAttr(attr);
	}
      }
      
      child = gdml->GetNext(child);	  
    }
    
    //still in the not Z else...but not in the while..
       
    name = gdml->GetAttr(node, "name");
    mix = new TGeoMixture(NameShort(name), ncompo, density);
    mixflag = 1;
    int i = 0;
    
    for(fractions f = fracmap.begin(); f != fracmap.end(); f++){
      if(felemap.find(f->first) != felemap.end()){
	mix->DefineElement(i, felemap[f->first], f->second);
      }
      else{
	// mix->DefineElement(i, fmixmap[f->first], f->second); BUG IN PYTHON???
      }
      
      i = i + 1;
      
    }
    
  }//end of not Z else
  
  medid = medid + 1;
  
  TGeoMedium* med = 0;
  
  if(mixflag == 1){
    fmixmap[name] = mix;
    med = new TGeoMedium(NameShort(name), medid, mix);
  }
  else if (mixflag == 0){
    fmatmap[name] = mat;
    med = new TGeoMedium(NameShort(name), medid, mat);
  }
  
  fmedmap[name] = med;
  
  return child;
}

//____________________________________________________________
XMLNodePointer_t TGDMLParse::VolProcess(TXMLEngine* gdml, XMLNodePointer_t node){
  
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
  const char* name;
  const char* solidname = "";
  const char* tempattr = "";
  TGeoShape* solid = 0;
  TGeoMedium* medium = 0;
  TGeoVolume* vol = 0; 
  TGeoVolume* lv = 0;
  TGeoShape* reflex = 0;
  TGeoTranslation* pos = new TGeoTranslation(0,0,0);
  TGeoRotation* rot = new TGeoRotation();
  TGeoMatrix* matr;
  TGeoMatrix* tempmatr = 0;
  const Double_t* parentrot = 0;
  int yesrefl = 0;
  
  
  while (child!=0) {
    if((strcmp(gdml->GetNodeName(child), "solidref")) == 0){
      if(fsolmap.find(gdml->GetAttr(child, "ref")) != fsolmap.end()){ 
	solid = fsolmap[gdml->GetAttr(child, "ref")];
      }
      else if(freflectmap.find(gdml->GetAttr(child, "ref")) != freflectmap.end()){
	solidname = gdml->GetAttr(child, "ref");
	reflex = fsolmap[freflectmap[gdml->GetAttr(child, "ref")]];
      } 
      else{
	printf("Solid: %s, Not Yet Defined!\n", gdml->GetAttr(child, "ref"));
      }
    }
    
    if((strcmp(gdml->GetNodeName(child), "materialref")) == 0){ 
      if(fmedmap.find(gdml->GetAttr(child, "ref")) != fmedmap.end()){ 
	medium = fmedmap[gdml->GetAttr(child, "ref")];
      } 
      else{
	printf("Medium: %s, Not Yet Defined!\n", gdml->GetAttr(child, "ref"));
      }
    }
    
    child = gdml->GetNext(child);
  }
  
  name = gdml->GetAttr(node, "name");
  
  if(reflex == 0){
    vol = new TGeoVolume(NameShort(name), solid, medium);
  }
  else{
    vol = new TGeoVolume(NameShort(name), reflex, medium);
    freflvolmap[name] = solidname;
    TGDMLRefl* parentrefl = freflsolidmap[solidname];
    parentrot = parentrefl->GetMatrix()->GetRotationMatrix();
    yesrefl = 1;
  }
  
  fvolmap[name] = vol;
  
  //PHYSVOL - run through child nodes of VOLUME again..
  
  child = gdml->GetChild(node);
  
  while (child!=0) {
    
    if((strcmp(gdml->GetNodeName(child), "physvol")) == 0){
      
      const char* volref = "";
      subchild = gdml->GetChild(child);
      pos = new TGeoTranslation(0,0,0);
      rot = new TGeoRotation();
      
      while (subchild!=0){
	tempattr = gdml->GetNodeName(subchild);
	
	if((strcmp(tempattr, "volumeref")) == 0){
	  lv = fvolmap[gdml->GetAttr(subchild, "ref")];
	  volref = gdml->GetAttr(subchild, "ref");
	}
	
	else if((strcmp(tempattr, "positionref")) == 0){
	  if(fposmap.find(gdml->GetAttr(subchild, "ref")) != fposmap.end()){ 
	    pos = fposmap[gdml->GetAttr(subchild, "ref")];
	  }
	}
	else if((strcmp(tempattr, "position")) == 0){
	  attr = gdml->GetFirstAttr(subchild);
	  PosProcess(gdml, subchild, attr);
	  pos = fposmap[gdml->GetAttr(subchild, "name")];
	}
	else if((strcmp(tempattr, "rotationref")) == 0){
	  if(frotmap.find(gdml->GetAttr(subchild, "ref")) != frotmap.end()){ 
	    rot = frotmap[gdml->GetAttr(subchild, "ref")];
	  }
	}
	else if((strcmp(tempattr, "rotation")) == 0){
	  attr = gdml->GetFirstAttr(subchild);
	  RotProcess(gdml, subchild, attr);
	  rot = frotmap[gdml->GetAttr(subchild, "name")];
	}
	
	subchild = gdml->GetNext(subchild);
      }
      
      //ADD PHYSVOL TO GEOMETRY
      fVolID = fVolID + 1;

      //if the volume is a reflected volume the matrix needs to be CHANGED
      if(freflvolmap.find(volref) != freflvolmap.end()){ 
	
	TGDMLRefl* temprefl = freflsolidmap[freflvolmap[volref]];
	tempmatr = new TGeoCombiTrans(*pos, *rot);
	
	const Double_t* A_pos = temprefl->GetMatrix()->GetTranslation();
	const Double_t* B_pos = tempmatr->GetTranslation();
	const Double_t* A_rot = temprefl->GetMatrix()->GetRotationMatrix();
	const Double_t* B_rot = tempmatr->GetRotationMatrix();
	double C_rot[8]; for (int i = 0; i < 9; i++){C_rot[i] = 0;}
	int a = 0, c = 0;
	
	//matrix multiplication
	for(a=0; a<9; a+=3){
	  for(c=a; c<(a+3); c++){
	    int b = (c-a);
	    C_rot[c] = (A_rot[a] * B_rot[b]) + (A_rot[(a+1)] * B_rot[(b+3)]) + (A_rot[(a+2)] * B_rot[(b+6)]);
	  } 
	}
	
	//new matrix values being set
	rot = new TGeoRotation();
	rot->SetMatrix(C_rot);
	pos = new TGeoTranslation();
	pos->SetDx((A_pos[0] + B_pos[0]));
	pos->SetDy((A_pos[1] + B_pos[1]));
	pos->SetDz((A_pos[2] + B_pos[2]));
	
      }
      
      //reflection from mother volume being carried forward
      if(yesrefl == 1){
	const Double_t* tempPos = pos->GetTranslation();    
	pos->SetDx(tempPos[0]*parentrot[0]);
	pos->SetDy(tempPos[1]*parentrot[4]);
	pos->SetDz(tempPos[2]*parentrot[8]);
      }
      
      matr = new TGeoCombiTrans(*pos, *rot);
      
      vol->AddNode(lv, fVolID, matr);
      
    }
    
    else if((strcmp(gdml->GetNodeName(child), "divisionvol")) == 0){
      
      const char* divVolref = "";
      Int_t axis = 0;
      const char* number = "";
      const char* width = "";
      const char* offset = "";
      const char* lunit = "mm";
      
      attr = gdml->GetFirstAttr(child);
      
      while (attr!=0) {
	
	tempattr = gdml->GetAttrName(attr);
	
	if((strcmp(tempattr, "axis")) == 0) { 
	  axis = SetAxis(gdml->GetAttrValue(attr));
	}
	else if((strcmp(tempattr, "number")) == 0) { 
	  number = FindConst(gdml->GetAttrValue(attr));
	}
	else if(strcmp(tempattr, "width") == 0){
	  width = FindConst(gdml->GetAttrValue(attr));
	}
	else if (strcmp(tempattr, "offset") == 0){
	  offset = FindConst(gdml->GetAttrValue(attr));
	}
	else if (strcmp(tempattr, "unit") == 0){
	  lunit = gdml->GetAttrValue(attr);
	}
	
	attr = gdml->GetNextAttr(attr);
	
      }
      
      subchild = gdml->GetChild(child);
      
      while (subchild!=0){
	tempattr = gdml->GetNodeName(subchild);
	
	if((strcmp(tempattr, "volumeref")) == 0){ 
	  divVolref = gdml->GetAttr(subchild, "ref");
	}
	
	subchild = gdml->GetNext(subchild);	      
      } 


      const char* numberline = "";
      const char* widthline = "";
      const char* offsetline = "";
      const char* retunit;
      
      retunit = GetScale(lunit);
      
      numberline = Form("%s*%s", number, retunit);
      widthline = Form("%s*%s", width, retunit);
      offsetline = Form("%s*%s", offset, retunit);
 
      
      fVolID = fVolID + 1;
      
      vol->Divide(NameShort(name), axis, (Int_t)Evaluate(numberline), (Double_t)Evaluate(offsetline), (Double_t)Evaluate(widthline));
      
    }//end of Division else if
    
    child = gdml->GetNext(child);
  }
  
  return child;
  
}

//______________________________________________________
XMLNodePointer_t TGDMLParse::BooSolid(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr, int num){
 
  //In the solid section of the GDML file, boolean solids can be 
  //declared. when the subtraction, intersection or union  keyword 
  //is found, this function is called, and the values (rotation and 
  //translation) of the solid are converted into type TGeoCompositeShape
  //and stored in fsolmap map using the name as its key.

  //1 = SUBTRACTION
  //2 = INTERSECTION
  //3 = UNION
  
  const char* tempattr = "";
  XMLNodePointer_t child = gdml->GetChild(node);
  TGeoCompositeShape* boolean = 0;
  TGeoShape* first = 0;
  TGeoShape* second = 0;
  TGeoTranslation* pos = new TGeoTranslation(0,0,0);
  
  TGeoRotation* rot = new TGeoRotation();
  rot->RotateZ(0);
  rot->RotateY(0);
  rot->RotateX(0);
  
  const char* name = gdml->GetAttr(node, "name");
  
  while (child!=0){
    tempattr = gdml->GetNodeName(child);
    
    if((strcmp(tempattr, "first")) == 0){
      if(fsolmap.find(gdml->GetAttr(child, "ref")) != fsolmap.end()){ 
	first = fsolmap[gdml->GetAttr(child, "ref")];
      }
    }
    else if((strcmp(tempattr, "second")) == 0) {
      if(fsolmap.find(gdml->GetAttr(child, "ref")) != fsolmap.end()){  
	second = fsolmap[gdml->GetAttr(child, "ref")];
      }
    }
    else if((strcmp(tempattr, "position")) == 0){
      attr = gdml->GetFirstAttr(child);
      PosProcess(gdml, child, attr);
      pos = fposmap[gdml->GetAttr(child, "name")];
    }
    else if((strcmp(tempattr, "positionref")) == 0){
      if(fposmap.find(gdml->GetAttr(child, "ref")) != fposmap.end()){ 
	pos = fposmap[gdml->GetAttr(child, "ref")];
      }
    }
    else if((strcmp(tempattr, "rotation")) == 0){
      attr = gdml->GetFirstAttr(child);
      RotProcess(gdml, child, attr);
      rot = frotmap[gdml->GetAttr(child, "name")];
    }
    else if((strcmp(tempattr, "rotationref")) == 0){
      if(frotmap.find(gdml->GetAttr(child, "ref")) != frotmap.end()){ 
	rot = frotmap[gdml->GetAttr(child, "ref")];
      }
    }    
    child = gdml->GetNext(child);
  }
  
  if(num == 1){
    TGeoRotation* myrot = new TGeoRotation(rot->Inverse());
    TGeoMatrix* matr = new TGeoCombiTrans(*pos, *myrot);
    TGeoMatrix* mat2 = new TGeoCombiTrans();
    TGeoSubtraction* sub = new TGeoSubtraction(first, second, mat2, matr);
    boolean = new TGeoCompositeShape(NameShort(name), sub);
  }
  else if(num == 2){
    TGeoRotation* myrot = new TGeoRotation(rot->Inverse());
    TGeoTranslation* mypos = new TGeoTranslation(*pos);
    TGeoMatrix* matr = new TGeoCombiTrans(*mypos, *myrot);
    TGeoMatrix* mat2 = new TGeoCombiTrans();
    TGeoIntersection* inter = new TGeoIntersection(first, second, mat2, matr);
    boolean = new TGeoCompositeShape(NameShort(name), inter);  
  }
  else if(num == 3){
    TGeoRotation* myrot = new TGeoRotation(rot->Inverse());
    TGeoMatrix* matr = new TGeoCombiTrans(*pos, *myrot);
    TGeoMatrix* mat2 = new TGeoCombiTrans();
    TGeoUnion* un = new TGeoUnion(first, second, mat2, matr);
    boolean = new TGeoCompositeShape(NameShort(name), un);
  }
  
  fsolmap[name] = boolean;
  
  return node;
}

//________________________________________________________
XMLNodePointer_t TGDMLParse::AssProcess(TXMLEngine* gdml, XMLNodePointer_t node){
  
  //In the structure section of the GDML file, assembly volumes can be 
  //declared. when the assembly keyword is found, this function is called, 
  //and the name is converted into type TGeoVolumeAssembly and
  //stored in fvolmap map using the name as its key. Some assembly volumes 
  //reference to other physical volumes to contain inside that assembly, 
  //declaring positions and rotations within that volume. When each 'physvol' 
  //is declared, a matrix for its rotation and translation is built and the 
  //'physvol node' is added to the original assembly using TGeoVolume->AddNode.

  const char* name = gdml->GetAttr(node, "name");
  XMLAttrPointer_t attr;
  XMLNodePointer_t subchild;
  XMLNodePointer_t child = gdml->GetChild(node);
  const char* tempattr = "";
  TGeoVolume* lv = 0;
  TGeoTranslation* pos = new TGeoTranslation(0,0,0);
  TGeoRotation* rot = new TGeoRotation();
  TGeoCombiTrans* matr;
  
  TGeoVolumeAssembly* assem = new TGeoVolumeAssembly(NameShort(name));
  fvolmap[name] = assem;
  
  //PHYSVOL - run through child nodes of VOLUME again..
  
  child = gdml->GetChild(node);
  
  while (child!=0) {
    if((strcmp(gdml->GetNodeName(child), "physvol")) == 0){
      subchild = gdml->GetChild(child);
      pos = new TGeoTranslation(0,0,0);
      rot = new TGeoRotation();
      
      while (subchild!=0){
	tempattr = gdml->GetNodeName(subchild);
	
	if((strcmp(tempattr, "volumeref")) == 0){
	  lv = fvolmap[gdml->GetAttr(subchild, "ref")];
	}     
	else if((strcmp(tempattr, "positionref")) == 0){
	  if(fposmap.find(gdml->GetAttr(subchild, "ref")) != fposmap.end()){ 
	    pos = fposmap[gdml->GetAttr(subchild, "ref")];
	  }
	}
	else if((strcmp(tempattr, "position")) == 0){
	  attr = gdml->GetFirstAttr(subchild);
	  PosProcess(gdml, subchild, attr);
	  pos = fposmap[gdml->GetAttr(subchild, "name")];
	}
	else if((strcmp(tempattr, "rotationref")) == 0){
	  if(frotmap.find(gdml->GetAttr(subchild, "ref")) != frotmap.end()){ 
	    rot = frotmap[gdml->GetAttr(subchild, "ref")];
	  }
	}
	else if((strcmp(tempattr, "rotation")) == 0){
	  attr = gdml->GetFirstAttr(subchild);
	  RotProcess(gdml, subchild, attr);
	  rot = frotmap[gdml->GetAttr(subchild, "name")];
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
  return child;
}

//________________________________________________________
XMLNodePointer_t  TGDMLParse::TopProcess(TXMLEngine* gdml, XMLNodePointer_t node){
  
  //In the setup section of the GDML file, the top volume need to be 
  //declared. when the setup keyword is found, this function is called, 
  //and the top volume ref is taken and 'world' is set

  XMLNodePointer_t child = gdml->GetChild(node);
 
  while(child != 0){
    
    if((strcmp(gdml->GetNodeName(child), "world") == 0)){
      fWorld = fvolmap[gdml->GetAttr(child, "ref")];
    } 
    child = gdml->GetNext(child);
  }  
  return node;
}

//___________________________________________________________________
XMLNodePointer_t TGDMLParse::Box(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, a box may be declared. 
  //when the box keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoBBox and stored in fsolmap map using the name 
  //as its key.
  
  const char* lunit = "mm"; 
  const char* xpos = "0"; 
  const char* ypos = "0"; 
  const char* zpos = "0"; 
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "x")) == 0) { 
      xpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "y") == 0){
      ypos = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "z") == 0){
      zpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    
    attr = gdml->GetNextAttr(attr);  
  }
  
  const char* xline = "";
  const char* yline = "";
  const char* zline = "";
  const char* retunit;
  
  retunit = GetScale(lunit);
  
  xline = Form("%s*%s", xpos, retunit);
  yline = Form("%s*%s", ypos, retunit);
  zline = Form("%s*%s", zpos, retunit);

  
  TGeoBBox* box = new TGeoBBox(NameShort(name),Evaluate(xline)/2,
			       Evaluate(yline)/2,
			       Evaluate(zline)/2);
  
  fsolmap[name] = box;
  
  return node;
  
}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Paraboloid(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, a Paraboloid may be declared. 
  //when the paraboloid keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoParaboloid and stored in fsolmap map using the name 
  //as its key.

  const char* lunit = "mm"; 
  const char* rlopos = "0"; 
  const char* rhipos = "0"; 
  const char* dzpos = "0"; 
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "rlo")) == 0) { 
      rlopos = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "rhi") == 0){
      rhipos = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "dz") == 0){
      dzpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    
    attr = gdml->GetNextAttr(attr);  
  }


  const char* rloline = "";
  const char* rhiline = "";
  const char* dzline = "";
  const char* retunit;
  
  retunit = GetScale(lunit);
  
  rloline = Form("%s*%s", rlopos, retunit);
  rhiline = Form("%s*%s", rhipos, retunit);
  dzline = Form("%s*%s", dzpos, retunit);
  
  TGeoParaboloid* paraboloid = new TGeoParaboloid(NameShort(name),Evaluate(rloline),
						  Evaluate(rhiline),
						  Evaluate(dzline));
  
  fsolmap[name] = paraboloid;
  
  return node;
  
}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Arb8(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, an Arb8 may be declared. 
  //when the arb8 keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoArb8 and stored in fsolmap map using the name 
  //as its key.

  const char* lunit = "mm"; 
  const char* v1xpos = "0";
  const char* v1ypos = "0";
  const char* v2xpos = "0";
  const char* v2ypos = "0";
  const char* v3xpos = "0";
  const char* v3ypos = "0";
  const char* v4xpos = "0";
  const char* v4ypos = "0";
  const char* v5xpos = "0";
  const char* v5ypos = "0";
  const char* v6xpos = "0";
  const char* v6ypos = "0";
  const char* v7xpos = "0";
  const char* v7ypos = "0";
  const char* v8xpos = "0";
  const char* v8ypos = "0"; 
  const char* dzpos = "0"; 
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "v1x")) == 0) { 
      v1xpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "v1y") == 0){
      v1ypos = FindConst(gdml->GetAttrValue(attr));
    }
    else if((strcmp(tempattr, "v2x")) == 0) { 
      v2xpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "v2y") == 0){
      v2ypos = FindConst(gdml->GetAttrValue(attr));
    }
    else if((strcmp(tempattr, "v3x")) == 0) { 
      v3xpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "v3y") == 0){
      v3ypos = FindConst(gdml->GetAttrValue(attr));
    }
    else if((strcmp(tempattr, "v4x")) == 0) { 
      v4xpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "v4y") == 0){
      v4ypos = FindConst(gdml->GetAttrValue(attr));
    }
    else if((strcmp(tempattr, "v5x")) == 0) { 
      v5xpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "v5y") == 0){
      v5ypos = FindConst(gdml->GetAttrValue(attr));
    }
    else if((strcmp(tempattr, "v6x")) == 0) { 
      v6xpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "v6y") == 0){
      v6ypos = FindConst(gdml->GetAttrValue(attr));
    }
    else if((strcmp(tempattr, "v7x")) == 0) { 
      v7xpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "v7y") == 0){
	v7ypos = FindConst(gdml->GetAttrValue(attr));
    }
    else if((strcmp(tempattr, "v8x")) == 0) { 
      v8xpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "v8y") == 0){
      v8ypos = FindConst(gdml->GetAttrValue(attr));
    }
    else if((strcmp(tempattr, "dz")) == 0) { 
      dzpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    
    attr = gdml->GetNextAttr(attr);  
  }

  const char* v1xline = ""; 
  const char* v1yline = "";
  const char* v2xline = "";
  const char* v2yline  = "";
  const char* v3xline = "";
  const char* v3yline = "";
  const char* v4xline = "";
  const char* v4yline  = "";
  const char* v5xline = "";
  const char* v5yline = "";
  const char* v6xline = "";
  const char* v6yline  = "";
  const char* v7xline = "";
  const char* v7yline = "";
  const char* v8xline = "";
  const char* v8yline  = "";
  const char* dzline = "";
  
  const char* retunit;
  
  retunit = GetScale(lunit);
  
  v1xline = Form("%s*%s", v1xpos, retunit);
  v1yline = Form("%s*%s", v1ypos, retunit);
  v2xline = Form("%s*%s", v2xpos, retunit);
  v2yline = Form("%s*%s", v2ypos, retunit);
  v3xline = Form("%s*%s", v3xpos, retunit);
  v3yline = Form("%s*%s", v3ypos, retunit);
  v4xline = Form("%s*%s", v4xpos, retunit);
  v4yline = Form("%s*%s", v4ypos, retunit);
  v5xline = Form("%s*%s", v5xpos, retunit);
  v5yline = Form("%s*%s", v5ypos, retunit);
  v6xline = Form("%s*%s", v6xpos, retunit);
  v6yline = Form("%s*%s", v6ypos, retunit);
  v7xline = Form("%s*%s", v7xpos, retunit);
  v7yline = Form("%s*%s", v7ypos, retunit);
  v8xline = Form("%s*%s", v8xpos, retunit);
  v8yline = Form("%s*%s", v8ypos, retunit);
  dzline = Form("%s*%s", dzpos , retunit);

  
  TGeoArb8* arb8 = new TGeoArb8(NameShort(name), Evaluate(dzline));

  arb8->SetVertex(0, Evaluate(v1xline),Evaluate(v1yline));
  arb8->SetVertex(1, Evaluate(v2xline),Evaluate(v2yline));
  arb8->SetVertex(2, Evaluate(v3xline),Evaluate(v3yline));
  arb8->SetVertex(3, Evaluate(v4xline),Evaluate(v4yline));
  arb8->SetVertex(4, Evaluate(v5xline),Evaluate(v5yline));
  arb8->SetVertex(5, Evaluate(v6xline),Evaluate(v6yline));
  arb8->SetVertex(6, Evaluate(v7xline),Evaluate(v7yline));
  arb8->SetVertex(7, Evaluate(v8xline),Evaluate(v8yline));			    

  fsolmap[name] = arb8;
  
  return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Tube(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, a Tube may be declared. 
  //when the tube keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoTubeSeg and stored in fsolmap map using the name 
  //as its key.

  const char* lunit = "mm"; 
  const char* aunit = "deg";
  const char* rmin = "0"; 
  const char* rmax = "0"; 
  const char* z = "0";
  const char* startphi = "0";
  const char* deltaphi = "0";
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "rmin")) == 0) { 
      rmin = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "rmax") == 0){
      rmax = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "z") == 0){
      z = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "aunit") == 0){
      aunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "startphi") == 0){
      startphi = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "deltaphi") == 0){
      deltaphi = FindConst(gdml->GetAttrValue(attr));
    }
    
    attr = gdml->GetNextAttr(attr);  
  }

  
  const char* rminline = "";
  const char* rmaxline= "";
  const char* zline = "";
  const char* startphiline = "";
  const char* deltaphiline = "";
  
  const char* retlunit; 
  const char* retaunit;

  retlunit = GetScale(lunit);
  retaunit = GetScale(aunit);

  rminline = Form("%s*%s", rmin, retlunit);
  rmaxline = Form("%s*%s", rmax, retlunit);
  zline = Form("%s*%s", z, retlunit);
  startphiline = Form("%s*%s", startphi, retaunit);
  deltaphiline = Form("(%s*%s) + %s", deltaphi, retaunit, startphiline);

  TGeoTubeSeg* tube = new TGeoTubeSeg(NameShort(name),Evaluate(rminline),
				      Evaluate(rmaxline),
				      Evaluate(zline)/2, 
				      Evaluate(startphiline),
				      Evaluate(deltaphiline));

  fsolmap[name] = tube;
  
  return node;
  
}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::CutTube(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, a Cut Tube may be declared. 
  //when the cutTube keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoCtub and stored in fsolmap map using the name 
  //as its key.

  const char* lunit = "mm"; 
  const char* aunit = "deg";
  const char* rmin = "0"; 
  const char* rmax = "0"; 
  const char* z = "0";
  const char* startphi = "0";
  const char* deltaphi = "0";
  const char* lowX = "0";
  const char* lowY = "0";
  const char* lowZ = "0";
  const char* highX = "0";
  const char* highY = "0";
  const char* highZ = "0";
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
	name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "rmin")) == 0) { 
      rmin = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "rmax") == 0){
      rmax = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "z") == 0){
      z = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "aunit") == 0){
      aunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "startphi") == 0){
      startphi = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "deltaphi") == 0){
      deltaphi = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lowX") == 0){
      lowX = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lowY") == 0){
      lowY = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lowZ") == 0){
      lowZ = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "highX") == 0){
      highX = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "highY") == 0){
      highY = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "highZ") == 0){
      highZ = FindConst(gdml->GetAttrValue(attr));
    }

    attr = gdml->GetNextAttr(attr);  
  }


  const char* rminline = "";
  const char* rmaxline = "";
  const char* zline = "";
  const char* startphiline = "";
  const char* deltaphiline = "";
  const char* lowXline = "";
  const char* lowYline = "";
  const char* lowZline = "";
  const char* highXline = "";
  const char* highYline = "";
  const char* highZline = "";

  const char* retlunit; 
  const char* retaunit;
  
  retlunit = GetScale(lunit);
  retaunit = GetScale(aunit);
  
  rminline = Form("%s*%s", rmin, retlunit);
  rmaxline = Form("%s*%s", rmax, retlunit);
  zline = Form("%s*%s", z, retlunit);
  startphiline = Form("%s*%s", startphi, retaunit);
  deltaphiline = Form("(%s*%s) + %s", deltaphi, retaunit, startphiline);
  lowXline = Form("%s*%s", lowX, retlunit);
  lowYline = Form("%s*%s", lowY, retlunit);
  lowZline = Form("%s*%s", lowZ, retlunit);
  highXline = Form("%s*%s", highX, retlunit);
  highYline = Form("%s*%s", highY, retlunit);
  highZline = Form("%s*%s", highZ, retlunit);

  
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
  

  fsolmap[name] = cuttube;
  
  return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Cone(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, a cone may be declared. 
  //when the cone keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoConSeg and stored in fsolmap map using the name 
  //as its key.

  const char* lunit = "mm"; 
  const char* aunit = "deg";
  const char* rmin1 = "0"; 
  const char* rmax1 = "0"; 
  const char* rmin2 = "0"; 
  const char* rmax2 = "0"; 
  const char* z = "0";
  const char* startphi = "0";
  const char* deltaphi = "0";
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "rmin1")) == 0) { 
      rmin1 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "rmax1") == 0){
      rmax1 = FindConst(gdml->GetAttrValue(attr));
    }
    else if((strcmp(tempattr, "rmin2")) == 0) {
      rmin2 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "rmax2") == 0){
      rmax2 = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "z") == 0){
      z = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "aunit") == 0){
      aunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "startphi") == 0){
      startphi = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "deltaphi") == 0){
      deltaphi = FindConst(gdml->GetAttrValue(attr));
    }
    
    attr = gdml->GetNextAttr(attr);  
  }


  const char* rmin1line = "";
  const char* rmax1line = "";
  const char* rmin2line = "";
  const char* rmax2line = "";
  const char* zline = "";
  const char* startphiline = "";
  const char* deltaphiline = "";
  const char* retlunit; 
  const char* retaunit;
  
  retlunit = GetScale(lunit);
  retaunit = GetScale(aunit);
  
  rmin1line = Form("%s*%s", rmin1, retlunit);
  rmax1line = Form("%s*%s", rmax1, retlunit);
  rmin2line = Form("%s*%s", rmin2, retlunit);
  rmax2line = Form("%s*%s", rmax2, retlunit);
  zline = Form("%s*%s", z, retlunit);
  startphiline = Form("%s*%s", startphi, retaunit);
  deltaphiline = Form("(%s*%s) + %s", deltaphi, retaunit, startphiline);

    
  TGeoConeSeg* cone = new TGeoConeSeg(NameShort(name),Evaluate(zline)/2,
				      Evaluate(rmin1line),
				      Evaluate(rmax1line),
				      Evaluate(rmin2line),
				      Evaluate(rmax2line),
				      Evaluate(startphiline),
				      Evaluate(deltaphiline));
			    
  fsolmap[name] = cone;
  
  return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Trap(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, a Trap may be declared. 
  //when the trap keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoTrap and stored in fsolmap map using the name 
  //as its key.

  const char* lunit = "mm"; 
  const char* aunit = "deg";
  const char* x1 = "0"; 
  const char* x2 = "0"; 
  const char* x3 = "0"; 
  const char* x4 = "0"; 
  const char* y1 = "0"; 
  const char* y2 = "0";
  const char* z = "0";
  const char* phi = "0";
  const char* theta = "0";
  const char* alpha1 = "0"; 
  const char* alpha2 = "0"; 
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if(strcmp(tempattr, "name") == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if(strcmp(tempattr, "x1") == 0) {
      x1 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "x2") == 0) {
      x2 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "x3") == 0) { 
      x3 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "x4") == 0) {
      x4 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "y1") == 0) { 
      y1 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "y2") == 0) {
      y2 = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "z") == 0) {
      z = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lunit") == 0) {
      lunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "aunit") == 0){
      aunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "phi") == 0){
      phi = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "theta") == 0){
      theta = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "alpha1") == 0) { 
      alpha1 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "alpha2") == 0){
      alpha2 = FindConst(gdml->GetAttrValue(attr));
    }
    
    attr = gdml->GetNextAttr(attr);  
  }

  const char* x1line = ""; 
  const char* x2line = ""; 
  const char*x3line   = ""; 
  const char*x4line = ""; 
  const char*y1line = ""; 
  const char*y2line = ""; 
  const char*zline = ""; 
  const char*philine = ""; 
  const char*thetaline = ""; 
  const char*alpha1line = ""; 
  const char*alpha2line = ""; 
  const char* retlunit; 
  const char* retaunit;
  
  retlunit = GetScale(lunit);
  retaunit = GetScale(aunit);
  
  x1line = Form("%s*%s", x1, retlunit);
  x2line = Form("%s*%s", x2, retlunit);
  x3line = Form("%s*%s", x3, retlunit);
  x4line = Form("%s*%s", x4, retlunit);
  y1line = Form("%s*%s", y1, retlunit);
  y2line = Form("%s*%s", y2, retlunit);
  zline = Form("%s*%s", z, retlunit);
  philine = Form("%s*%s", phi, retaunit);
  thetaline = Form("%s*%s", theta, retaunit);
  alpha1line = Form("%s*%s", alpha1, retaunit);
  alpha2line = Form("%s*%s", alpha2, retaunit);
  
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
  
  fsolmap[name] = trap;
  
  return node;
  
}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Trd(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, a Trd may be declared. 
  //when the trd keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoTrd2 and stored in fsolmap map using the name 
  //as its key.

  const char* lunit = "mm"; 
  const char* x1 = "0"; 
  const char* x2 = "0";  
  const char* y1 = "0"; 
  const char* y2 = "0";
  const char* z = "0";
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if(strcmp(tempattr, "name") == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if(strcmp(tempattr, "x1") == 0) { 
      x1 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "x2") == 0){
      x2 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "y1") == 0) { 
      y1 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "y2") == 0){
      y2 = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "z") == 0){
      z = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    
    attr = gdml->GetNextAttr(attr);  
  }
  
  const char* x1line = "";
  const char* x2line = "";
  const char* y1line = "";
  const char* y2line = "";
  const char* zline = "";
  const char* retlunit; 
  
  retlunit = GetScale(lunit);
  
  x1line = Form("%s*%s", x1, retlunit);
  x2line = Form("%s*%s", x2, retlunit);
  y1line = Form("%s*%s", y1, retlunit);
  y2line = Form("%s*%s", y2, retlunit);
  zline = Form("%s*%s", z, retlunit);
  
  TGeoTrd2* trd = new TGeoTrd2(NameShort(name),
			       Evaluate(x1line)/2,
			       Evaluate(x2line)/2,
			       Evaluate(y1line)/2,
			       Evaluate(y2line)/2,
			       Evaluate(zline)/2);
			    
  fsolmap[name] = trd;
  
  return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Polycone(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, a Polycone may be declared. 
  //when the polycone keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoPCon and stored in fsolmap map using the name 
  //as its key. Polycone has Zplanes, planes along the z axis specifying 
  //the rmin, rmax dimenstions at that point along z.

  const char* lunit = "mm"; 
  const char* aunit = "deg";
  const char* rmin = "0"; 
  const char* rmax = "0"; 
  const char* z = "0";
  const char* startphi = "0";
  const char* deltaphi = "0";
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "aunit") == 0){
      aunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "startphi") == 0){
      startphi = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "deltaphi") == 0){
      deltaphi = FindConst(gdml->GetAttrValue(attr));
    }
    
    attr = gdml->GetNextAttr(attr);  
  } 
  
  const char* retlunit; 
  const char* retaunit;
  
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
    if((strcmp(gdml->GetNodeName(child), "zplane")) == 0){
      //removed original dec
     const char* rminline = "";
     const char* rmaxline = "";
     const char* zline = "";
      
      attr = gdml->GetFirstAttr(child);
      
      while (attr!=0){
	tempattr = gdml->GetAttrName(attr);
	
	if((strcmp(tempattr, "rmin")) == 0) { 
	  rmin = FindConst(gdml->GetAttrValue(attr));
	  rminline = Form("%s*%s", rmin, retlunit);
	  table[planeno][0] = Evaluate(rminline);
	}
	else if(strcmp(tempattr, "rmax") == 0){
	  rmax = FindConst(gdml->GetAttrValue(attr));
	  rmaxline = Form("%s*%s", rmax, retlunit);
	  table[planeno][1] = Evaluate(rmaxline);
	}
	else if (strcmp(tempattr, "z") == 0){
	  z = FindConst(gdml->GetAttrValue(attr));
	  zline = Form("%s*%s", z, retlunit);
	  table[planeno][2] = Evaluate(zline);
	}
	
	attr = gdml->GetNextAttr(attr);
      }
    }
    planeno = planeno + 1;
    child = gdml->GetNext(child);
  }

  const char* startphiline = "";
  const char* deltaphiline = "";
  
  startphiline = Form("%s*%s", startphi, retaunit);
  deltaphiline = Form("%s*%s", deltaphi, retaunit);

  TGeoPcon* poly = new TGeoPcon(NameShort(name), 
				Evaluate(startphiline), 
				Evaluate(deltaphiline), 
				numplanes);
  Int_t zno = 0;
  
  for (int j = 0; j < numplanes; j++){
    poly->DefineSection(zno, table[j][2], table[j][0], table[j][1]);
    zno = zno + 1;
  }
  
  fsolmap[name] = poly;
  
  return node;
  
}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Polyhedra(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, a Polyhedra may be declared. 
  //when the polyhedra keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoPgon and stored in fsolmap map using the name 
  //as its key. Polycone has Zplanes, planes along the z axis specifying 
  //the rmin, rmax dimenstions at that point along z.

  const char* lunit = "mm"; 
  const char* aunit = "deg";
  const char* rmin = "0"; 
  const char* rmax = "0"; 
  const char* z = "0";
  const char* startphi = "0";
  const char* deltaphi = "0";
  const char* numsides = "1";
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "aunit") == 0){
      aunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "startphi") == 0){
      startphi = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "deltaphi") == 0){
      deltaphi = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "numsides") == 0){
      numsides  = FindConst(gdml->GetAttrValue(attr));
    }
    
    attr = gdml->GetNextAttr(attr);  
  } 
  
  const char* retlunit; 
  const char* retaunit;
  
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
    if((strcmp(gdml->GetNodeName(child), "zplane")) == 0){
	
	const char* rminline = "";
	const char* rmaxline = "";
	const char* zline = "";

	attr = gdml->GetFirstAttr(child);
	
	while (attr!=0){
	  tempattr = gdml->GetAttrName(attr);
	  
	  if((strcmp(tempattr, "rmin")) == 0) { 
	    rmin = FindConst(gdml->GetAttrValue(attr));
	    rminline = Form("%s*%s", rmin, retlunit);
	    table[planeno][0] = Evaluate(rminline);
	  }
	  else if(strcmp(tempattr, "rmax") == 0){
	    rmax = FindConst(gdml->GetAttrValue(attr));
	    rmaxline = Form("%s*%s", rmax, retlunit);
	    table[planeno][1] = Evaluate(rmaxline);
	  }
	  else if (strcmp(tempattr, "z") == 0){
	    z = FindConst(gdml->GetAttrValue(attr));
	    zline = Form("%s*%s", z, retlunit);
	    table[planeno][2] = Evaluate(zline);
	  }
	  
	  attr = gdml->GetNextAttr(attr);
	}
      }
    planeno = planeno + 1;
    child = gdml->GetNext(child);
  }

  
  const char* startphiline = "";
  const char* deltaphiline = "";
  const char* numsidesline = "";

  startphiline = Form("%s*%s", startphi, retaunit);
  deltaphiline = Form("%s*%s", deltaphi, retaunit);
  numsidesline = Form("%s", numsides);

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
  
  fsolmap[name] = polyg;
  
  return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Sphere(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, a Sphere may be declared. 
  //when the sphere keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoSphere and stored in fsolmap map using the name 
  //as its key.

  const char* lunit = "mm"; 
  const char* aunit = "deg";
  const char* rmin = "0"; 
  const char* rmax = "0"; 
  const char* startphi = "0";
  const char* deltaphi = "0";
  const char* starttheta = "0";
  const char* deltatheta = "0";
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "rmin")) == 0) { 
      rmin = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "rmax") == 0) {
      rmax = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "aunit") == 0){
      aunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "startphi") == 0){
      startphi = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "deltaphi") == 0){
      deltaphi = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "starttheta") == 0){
      starttheta = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "deltatheta") == 0){
      deltatheta = FindConst(gdml->GetAttrValue(attr));
    }
    
    attr = gdml->GetNextAttr(attr);  
  }

  const char* rminline = "";
  const char* rmaxline = "";
  const char* startphiline = "";
  const char* deltaphiline = "";
  const char* startthetaline = "";
  const char* deltathetaline = "";
  const char* retlunit; 
  const char* retaunit;
  
  retlunit = GetScale(lunit);
  retaunit = GetScale(aunit);
  
  rminline = Form("%s*%s", rmin, retlunit);
  rmaxline = Form("%s*%s", rmax, retlunit);
  startphiline = Form("%s*%s", startphi, retaunit);
  deltaphiline = Form("(%s*%s) + %s", deltaphi, retaunit, startphiline);
  startthetaline = Form("%s*%s", starttheta, retaunit);
  deltathetaline = Form("(%s*%s) + %s", deltatheta, retaunit, startthetaline); 

  TGeoSphere* sphere = new TGeoSphere(NameShort(name),
				      Evaluate(rminline),
				      Evaluate(rmaxline),
				      Evaluate(startthetaline),
				      Evaluate(deltathetaline),
				      Evaluate(startphiline),
				      Evaluate(deltaphiline));
			    
  fsolmap[name] = sphere;
  
  return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Torus(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, a Torus may be declared. 
  //when the torus keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoTorus and stored in fsolmap map using the name 
  //as its key.
  
  const char* lunit = "mm"; 
  const char* aunit = "deg";
  const char* rmin = "0"; 
  const char* rmax = "0"; 
  const char* rtor = "0";
  const char* startphi = "0";
  const char* deltaphi = "0";
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "rmin")) == 0) { 
      rmin = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "rmax") == 0){
      rmax = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "rtor") == 0){
      rtor = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "aunit") == 0){
      aunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "startphi") == 0){
      startphi = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "deltaphi") == 0){
      deltaphi = FindConst(gdml->GetAttrValue(attr));
    }
    
    attr = gdml->GetNextAttr(attr);  
  }


  const char* rminline = "";
  const char* rmaxline = "";
  const char* rtorline = "";
  const char* startphiline = "";
  const char* deltaphiline = "";
  const char* retlunit; 
  const char* retaunit;
  
  retlunit = GetScale(lunit);
  retaunit = GetScale(aunit);
  
  rminline = Form("%s*%s", rmin, retlunit);
  rmaxline = Form("%s*%s", rmax, retlunit);
  rtorline = Form("%s*%s", rtor, retlunit);
  startphiline = Form("%s*%s", startphi, retaunit);
  deltaphiline = Form("%s*%s", deltaphi, retaunit);

    
  TGeoTorus* torus = new TGeoTorus(NameShort(name),Evaluate(rtorline),
				      Evaluate(rminline),
				      Evaluate(rmaxline),
				      Evaluate(startphiline),
				      Evaluate(deltaphiline));
			    
  fsolmap[name] = torus;
  
  return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Hype(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){

  //In the solids section of the GDML file, a Hype may be declared. 
  //when the hype keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoHype and stored in fsolmap map using the name 
  //as its key.

  const char* lunit = "mm"; 
  const char* aunit = "deg";
  const char* rmin = "0"; 
  const char* rmax = "0"; 
  const char* z = "0";
  const char* inst = "0";
  const char* outst = "0";
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "rmin")) == 0) { 
      rmin = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "rmax") == 0){
      rmax = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "z") == 0){
      z = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "aunit") == 0){
      aunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "inst") == 0){
      inst = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "outst") == 0){
      outst = FindConst(gdml->GetAttrValue(attr));
    }
    
    attr = gdml->GetNextAttr(attr);  
  }

  const char* rminline = "";
  const char* rmaxline = "";
  const char* zline = "";
  const char* instline = "";
  const char* outstline = "";
  const char* retlunit; 
  const char* retaunit;
  
  retlunit = GetScale(lunit);
  retaunit = GetScale(aunit);
  
  rminline = Form("%s*%s", rmin, retlunit);
  rmaxline = Form("%s*%s", rmax, retlunit);
  zline = Form("%s*%s", z, retlunit);
  instline = Form("%s*%s", inst, retaunit);
  outstline = Form("%s*%s", outst, retaunit);

    
  TGeoHype* hype = new TGeoHype(NameShort(name),
				Evaluate(rminline),
				Evaluate(instline),
				Evaluate(rmaxline),
				Evaluate(outstline),
				Evaluate(zline)/2);
			    
  fsolmap[name] = hype;
  
  return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::Para(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){

  //In the solids section of the GDML file, a Para may be declared. 
  //when the para keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoPara and stored in fsolmap map using the name 
  //as its key.

  const char* lunit = "mm"; 
  const char* aunit = "deg";
  const char* x = "0"; 
  const char* y = "0"; 
  const char* z = "0";
  const char* phi = "0";
  const char* theta = "0";
  const char* alpha = "0";
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "x")) == 0) {
      x = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "y") == 0){
      y = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "z") == 0){
      z = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "aunit") == 0){
      aunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "phi") == 0){
      phi = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "theta") == 0){
      theta = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "alpha") == 0){
      alpha = FindConst(gdml->GetAttrValue(attr));
    }
    
    attr = gdml->GetNextAttr(attr);  
  }


  const char* xline = "";
  const char* yline = "";
  const char* zline = "";
  const char* philine = "";
  const char* alphaline = "";
  const char* thetaline = "";
  const char* retlunit; 
  const char* retaunit;
  
  retlunit = GetScale(lunit);
  retaunit = GetScale(aunit);
  
  xline = Form("%s*%s", x, retlunit);
  yline = Form("%s*%s", y, retlunit);
  zline = Form("%s*%s", z, retlunit);
  philine = Form("%s*%s", phi, retaunit);
  alphaline = Form("%s*%s", alpha, retaunit);
  thetaline = Form("%s*%s", theta, retaunit);


  TGeoPara* para = new TGeoPara(NameShort(name),
				Evaluate(x),
				Evaluate(y),
				Evaluate(z),
				Evaluate(alphaline),
				Evaluate(thetaline),
				Evaluate(philine));
			    
  fsolmap[name] = para;
  
  return node;

}

//_______________________________________________________
XMLNodePointer_t TGDMLParse::TwistTrap(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, a TwistTrap may be declared. 
  //when the twistedtrap keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoGTra and stored in fsolmap map using the name 
  //as its key.

  const char* lunit = "mm"; 
  const char* aunit = "deg";
  const char* x1 = "0"; 
  const char* x2 = "0"; 
  const char* x3 = "0"; 
  const char* x4 = "0"; 
  const char* y1 = "0"; 
  const char* y2 = "0";
  const char* z = "0";
  const char* phi = "0";
  const char* theta = "0";
  const char* alpha1 = "0"; 
  const char* alpha2 = "0"; 
  const char* twist = "0";
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if(strcmp(tempattr, "name") == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if(strcmp(tempattr, "x1") == 0) { 
      x1 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "x2") == 0){
      x2 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "x3") == 0) { 
      x3 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "x4") == 0){
      x4 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "y1") == 0) { 
      y1 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "y2") == 0){
      y2 = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "z") == 0){
      z = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "aunit") == 0){
      aunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "phi") == 0){
      phi = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "theta") == 0) {
      theta = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "alpha1") == 0)  { 
      alpha1 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "alpha2") == 0){
      alpha2 = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "twist") == 0) {
      twist = FindConst(gdml->GetAttrValue(attr));
    }
    
    attr = gdml->GetNextAttr(attr);  
  }
 

  const char* x1line = "";
  const char* x2line = "";
  const char* x3line = "";
  const char* x4line = "";
  const char* y1line = "";
  const char* y2line = "";
  const char* zline = "";
  const char* philine = "";
  const char* thetaline = "";
  const char* alpha1line = "";
  const char* alpha2line = "";
  const char* twistline = "";
  const char* retlunit; 
  const char* retaunit;
  
  retlunit = GetScale(lunit);
  retaunit = GetScale(aunit);
  
  x1line = Form("%s*%s", x1, retlunit);
  x2line = Form("%s*%s", x2, retlunit);
  x3line = Form("%s*%s", x3, retlunit);
  x4line = Form("%s*%s", x4, retlunit);
  y1line = Form("%s*%s", y1, retlunit);
  y2line = Form("%s*%s", y2, retlunit);
  zline = Form("%s*%s", z, retlunit);
  philine = Form("%s*%s", phi, retaunit);
  thetaline = Form("%s*%s", theta, retaunit);
  alpha1line = Form("%s*%s", alpha1, retaunit);
  alpha2line = Form("%s*%s", alpha2, retaunit);
  twistline = Form("%s*%s", twist, retaunit);

    
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
			    
  fsolmap[name] = twtrap;
  
  return node;

}


//___________________________________________________________________
XMLNodePointer_t TGDMLParse::ElTube(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, a ElTube may be declared. 
  //when the eltube keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoEltu and stored in fsolmap map using the name 
  //as its key.

  const char* lunit = "mm"; 
  const char* xpos = "0"; 
  const char* ypos = "0"; 
  const char* zpos = "0"; 
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "dx")) == 0) { 
      xpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "dy") == 0){
      ypos = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "dz") == 0){
      zpos = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    
    attr = gdml->GetNextAttr(attr);  
  }

  const char* xline = "";
  const char* yline = "";
  const char* zline = "";
  const char* retunit;
  
  retunit = GetScale(lunit);
  
  xline = Form("%s*%s", xpos, retunit);
  yline = Form("%s*%s", ypos, retunit);
  zline = Form("%s*%s", zpos, retunit);

  TGeoEltu* eltu = new TGeoEltu(NameShort(name),Evaluate(xline),
			       Evaluate(yline),
			       Evaluate(zline));

  fsolmap[name] = eltu;
  
  return node;

}
//___________________________________________________________________
XMLNodePointer_t TGDMLParse::Orb(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, an Orb may be declared. 
  //when the orb keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoSphere and stored in fsolmap map using the name 
  //as its key.

  const char* aunit = "deg"; 
  const char* r = "0"; 
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "r")) == 0) { 
      r = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "aunit") == 0){
      aunit = gdml->GetAttrValue(attr);
    }
    
    attr = gdml->GetNextAttr(attr);  
  }


  const char* rline = "";
  const char* retunit;
  
  retunit = GetScale(aunit);
  
  rline = Form("%s*%s", r, retunit);

  
  TGeoSphere* orb = new TGeoSphere(NameShort(name), 0, Evaluate(rline), 0, 180, 0, 360);

  fsolmap[name] = orb;
  
  return node;

}


//_______________________________________________________
XMLNodePointer_t TGDMLParse::Xtru(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){
  
  //In the solids section of the GDML file, an Xtru may be declared. 
  //when the xtru keyword is found, this function is called, and the 
  //dimensions required are taken and stored, these are then bound and
  //converted to type TGeoXtru and stored in fsolmap map using the name 
  //as its key. The xtru has child nodes of either 'twoDimVertex'or 
  //'section'.  These two nodes define the real structure of the shape.
  //The twoDimVertex's define the x,y sizes of a vertice. The section links
  //the vertice to a position within the xtru. 

  const char* lunit = "mm"; 
  const char* aunit = "deg";
  const char* x = "0"; 
  const char* y = "0"; 
  const char* zorder = "0";
  const char* zpos = "0";
  const char* xoff = "0";
  const char* yoff = "0";
  const char* scale = "0";
  const char* name = "";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "lunit") == 0){
      lunit = gdml->GetAttrValue(attr);
    }
    else if (strcmp(tempattr, "aunit") == 0){
      aunit = gdml->GetAttrValue(attr);
    }
    
    attr = gdml->GetNextAttr(attr);  
  } 
  
  const char* retlunit; 
  const char* retaunit;
  
  retlunit = GetScale(lunit);
  retaunit = GetScale(aunit);
  
  //START TO LOOK THRU CHILD NODES... 

  XMLNodePointer_t child = gdml->GetChild(node);
  int nosects = 0;
  int noverts = 0;

  while (child!=0){
    tempattr = gdml->GetNodeName(child);
    
    if((strcmp(tempattr, "twoDimVertex")) == 0){ 
      noverts = noverts + 1;
    }
    else if((strcmp(tempattr, "section")) == 0){ 
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
    if((strcmp(gdml->GetNodeName(child), "twoDimVertex")) == 0){
      const char* xline = ""; 
      const char* yline = "";
      
      attr = gdml->GetFirstAttr(child);
      
      while (attr!=0){
	tempattr = gdml->GetAttrName(attr);
	
	if((strcmp(tempattr, "x")) == 0) { 
	  x = FindConst(gdml->GetAttrValue(attr));
	  xline = Form("%s*%s", x, retlunit);
	  vertx[vert] = Evaluate(xline);
	}
	else if(strcmp(tempattr, "y") == 0){
	  y = FindConst(gdml->GetAttrValue(attr));
	  yline = Form("%s*%s", y, retlunit);
	  verty[vert] = Evaluate(yline);
	}
	
	attr = gdml->GetNextAttr(attr);
      }
      
      vert = vert + 1;
      
    }
    
    else if((strcmp(gdml->GetNodeName(child), "section")) == 0){

      const char* zorderline = "";
      const char* zposline = "";
      const char* xoffline = "";
      const char* yoffline = "";
      const char* scaleline = "";
      
      attr = gdml->GetFirstAttr(child);
      
      while (attr!=0){
	tempattr = gdml->GetAttrName(attr);
	
	if((strcmp(tempattr, "zOrder")) == 0) { 
	  zorder = FindConst(gdml->GetAttrValue(attr));
	  zorderline = Form("%s*%s", zorder, retlunit);
	  section[sect][0] = Evaluate(zorderline);
	}
	else if(strcmp(tempattr, "zPosition") == 0){
	  zpos = FindConst(gdml->GetAttrValue(attr));
	  zposline = Form("%s*%s", zpos, retlunit);
	  section[sect][1] = Evaluate(zposline);
	}
	else if (strcmp(tempattr, "xOffset") == 0){
	  xoff = FindConst(gdml->GetAttrValue(attr));
	  xoffline = Form("%s*%s", xoff, retlunit);
	  section[sect][2] = Evaluate(xoffline);
	}
	else if (strcmp(tempattr, "yOffset") == 0){
	  yoff = FindConst(gdml->GetAttrValue(attr));
	  yoffline = Form("%s*%s", yoff, retlunit);
	  section[sect][3] = Evaluate(yoffline);
	}
	else if (strcmp(tempattr, "scalingFactor") == 0){
	  scale = FindConst(gdml->GetAttrValue(attr));
	  scaleline = Form("%s*%s", scale, retlunit);
	  section[sect][4] = Evaluate(scaleline);
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
  
  fsolmap[name] = xtru;
  
  return node;

}

//____________________________________________________________
XMLNodePointer_t TGDMLParse::Reflection(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr){

  //In the solids section of the GDML file, a Reflected Solid may be 
  //declared when the ReflectedSolid keyword is found, this function 
  //is called. The rotation, position and scale for the reflection are 
  //applied to a matrix that is then stored in the class object 
  //TGDMLRefl.  This is then stored in the map freflsolidmap, with 
  //the reflection name as a reference. also the name of the solid to 
  //be reflected is stored in a map called freflectmap with the reflection 
  //name as a reference.

  const char* sx = "0"; 
  const char* sy = "0"; 
  const char* sz = "0"; 
  const char* rx = "0"; 
  const char* ry = "0"; 
  const char* rz = "0"; 
  const char* dx = "0"; 
  const char* dy = "0"; 
  const char* dz = "0"; 
  const char* name = "0";
  const char* solid = "0";
  const char* tempattr; 
  
  while (attr!=0) {
    
    tempattr = gdml->GetAttrName(attr);
    
    if((strcmp(tempattr, "name")) == 0) { 
      name = gdml->GetAttrValue(attr);
    }
    else if((strcmp(tempattr, "sx")) == 0) { 
      sx = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "sy") == 0){
      sy = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "sz") == 0){
      sz = FindConst(gdml->GetAttrValue(attr));
    }
    else if((strcmp(tempattr, "rx")) == 0) { 
      rx = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "ry") == 0){
      ry = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "rz") == 0){
      rz = FindConst(gdml->GetAttrValue(attr));
    }
    else if((strcmp(tempattr, "dx")) == 0) { 
      dx = FindConst(gdml->GetAttrValue(attr));
    }
    else if(strcmp(tempattr, "dy") == 0){
      dy = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "dz") == 0){
      dz = FindConst(gdml->GetAttrValue(attr));
    }
    else if (strcmp(tempattr, "solid") == 0){
      solid = gdml->GetAttrValue(attr);
    }
    attr = gdml->GetNextAttr(attr);  
  }
  
  TGeoRotation* rot = new TGeoRotation();
  rot->RotateZ(-(Evaluate(rz)));
  rot->RotateY(-(Evaluate(ry)));
  rot->RotateX(-(Evaluate(rx)));

  rot->ReflectZ(kTRUE);

  TGeoCombiTrans* relf_matx = new TGeoCombiTrans(Evaluate(dx), Evaluate(dy), Evaluate(dz), rot);
  TGDMLRefl* reflsol = new TGDMLRefl(NameShort(name), solid, relf_matx);
  freflsolidmap[name] = reflsol;
  freflectmap[name] = solid;

  return node;
  
}



//===================================================================

ClassImp(TGDMLRefl)

/******************************************************************
____________________________________________________________

TGDMLRefl Class

------------------------------------------------------------

This class is a helper class for TGDMLParse.  It assists in the 
reflection process.  This process takes a previously defined solid 
and can reflect the matrix of it. This class stores the name of the 
reflected solid, along with the name of the solid that is being 
reflected, and finally the reflected solid's matrix.  This is then 
recalled when the volume is used in the structure part of the gdml 
file.

******************************************************************/

//___________________________________________________________________
TGDMLRefl::TGDMLRefl(const char* name, const char* solid, TGeoMatrix* matrix){
  
  //this constructor method stores the values brought in as params.

  fNameS = name;
  fSolid = solid;
  fMatrix = matrix; 
}

//_________________________________________________________________
TGeoMatrix* TGDMLRefl::GetMatrix(){

  //this accessor method returns the matrix.

  return fMatrix;
}
