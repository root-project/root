/* @(#)root/netx:$Name:  $:$Id: TGDMLParse.h,v 1.1 2006/11/17 17:40:02 brun Exp $ */
// Authors: Ben Lloyd 09/11/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGDMLParse
#define ROOT_TGDMLParse

#ifndef ROOT_TGeoMatrix
#include "TGeoMatrix.h"
#endif

#ifndef ROOT_TXMLEngine
#include "TXMLEngine.h"
#endif

#ifndef ROOT_TGeoVolume
#include "TGeoVolume.h"
#endif

#include <map>
#include <iostream>

/*************************************************************************
 * TGDMLRefl - helper class for the import of GDML to ROOT.              * 
 *************************************************************************/

class TGDMLRefl : public TObject {
  
public:


   TGDMLRefl() {              //constructor
    
      fNameS = ""; 
      fSolid = "";
      fMatrix = 0;
      
   }            
   virtual ~TGDMLRefl() {}    //destructor 
   TGDMLRefl(const char* name, const char* solid, TGeoMatrix* matrix);
   TGeoMatrix* GetMatrix();  
   
private:
   
   const char*     fNameS;      //!reflected solid name
   const char*     fSolid;      //!solid name being reflected
   TGeoMatrix     *fMatrix;     //!matrix of reflected solid

   
   ClassDef(TGDMLRefl, 0)     //helper class used for the storage of reflected solids
    
};
     
/*************************************************************************
 * TGDMLParse - base class for the import of GDML to ROOT.               * 
 *************************************************************************/

class TGDMLParse : public TObject {

public:
  
   TGeoVolume* fWorld; //top volume of geometry
   int fVolID;   //volume ID, incremented as assigned.
   
   TGDMLParse() {              //constructor
      fVolID = 0;
   }            
   virtual ~TGDMLParse() {}    //destructor
   static TGeoVolume* StartGDML(const char* filename){
     
      TGDMLParse* fParser = new TGDMLParse;
      TGeoVolume* fWorld = fParser->GDMLReadFile(filename);
      
      return fWorld;
      
   }
   
   TGeoVolume*       GDMLReadFile(const char* filename = "test.gdml");
    
private:
    
   void              ParseGDML(TXMLEngine* gdml, XMLNodePointer_t node, Int_t level) ;
   const char*             GetScale(const char* unit);
   const char*       FindConst(const char* retval);
   double            Evaluate(const char* evalline);
   const char*       NameShort(const char* name);
   const char*       NameShortB(const char* name);
    
   //'define' section
   XMLNodePointer_t  ConProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  PosProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  RotProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
    
   //'materials' section
   XMLNodePointer_t  EleProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLNodePointer_t parentn);
   XMLNodePointer_t  MatProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr, int z);
    
   //'solids' section
   XMLNodePointer_t  BooSolid(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr, int num);
   XMLNodePointer_t  Box(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Paraboloid(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Arb8(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Tube(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  CutTube(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Cone(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Trap(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Trd(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Polycone(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Polyhedra(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Sphere(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Torus(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr); 
   XMLNodePointer_t  Hype(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr); 
   XMLNodePointer_t  Para(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  TwistTrap(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  ElTube(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Orb(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Xtru(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Reflection(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
    
   //'structure' section
   XMLNodePointer_t  VolProcess(TXMLEngine* gdml, XMLNodePointer_t node);
   XMLNodePointer_t  AssProcess(TXMLEngine* gdml, XMLNodePointer_t node);
   Int_t             SetAxis(const char* axisString); //Set Axis for Division
    
   //'setup' section
   XMLNodePointer_t  TopProcess(TXMLEngine* gdml, XMLNodePointer_t node);
    
    
   typedef std::map<std::string, TGeoTranslation*> PosMap;
   typedef std::map<std::string, TGeoRotation*> RotMap;
   typedef std::map<std::string, TGeoElement*> EleMap;
   typedef std::map<std::string, TGeoMaterial*> MatMap;
   typedef std::map<std::string, TGeoMedium*> MedMap;
   typedef std::map<std::string, TGeoMixture*> MixMap;
   typedef std::map<std::string, const char*> ConMap;
   typedef std::map<std::string, TGeoShape*> SolMap;
   typedef std::map<std::string, TGeoVolume*> VolMap;
   typedef std::map<std::string, std::string> ReflectionsMap;
   typedef std::map<std::string, TGDMLRefl*> ReflSolidMap;
   typedef std::map<std::string, std::string> ReflVolMap; 
   typedef std::map<std::string, double> FracMap;
    
   PosMap fposmap;                //!Map containing position names and the TGeoTranslation for it
   RotMap frotmap;                //!Map containing rotation names and the TGeoRotation for it
   EleMap felemap;                //!Map containing element names and the TGeoElement for it
   MatMap fmatmap;                //!Map containing material names and the TGeoMaterial for it
   MedMap fmedmap;                //!Map containing medium names and the TGeoMedium for it
   MixMap fmixmap;                //!Map containing mixture names and the TGeoMixture for it
   ConMap fconmap;                //!Map containing constant names and the constant's value
   SolMap fsolmap;                //!Map containing solid names and the TGeoShape for it
   VolMap fvolmap;                //!Map containing volume names and the TGeoVolume for it
   ReflectionsMap freflectmap;    //!Map containing reflection names and the Solid name ir references to
   ReflSolidMap freflsolidmap;    //!Map containing reflection names and the TGDMLRefl for it - containing refl matrix
   ReflVolMap freflvolmap;        //!Map containing reflected volume names and the solid ref for it
    
   ClassDef(TGDMLParse, 1)    //imports GDML using DOM and binds it to ROOT
      
};
      
#endif
      
      
