/* @(#)root/gdml:$Id$ */
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

#include "TFormula.h"

#include <map>
#include <vector>
#include <iostream>

/*************************************************************************
 * TGDMLRefl - helper class for the import of GDML to ROOT.              *
 *************************************************************************/

class TGDMLRefl : public TObject {
public:

   TGDMLRefl() {

      fNameS = "";
      fSolid = "";
      fMatrix = 0;
   }

   virtual ~TGDMLRefl() {}

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

class TGDMLBaseTGDMMapHelper : public std::map<std::string, const void *> {
};

//map's [] operator returns reference.
//to avoid ugly UB casts like static_cast<SomeType * &>(voidPtrLValue)
//I have this helper class.
template<typename T>

class TGDMAssignmentHelper {
private:
   TGDMLBaseTGDMMapHelper::iterator fPosInMap;

public:
   TGDMAssignmentHelper(TGDMLBaseTGDMMapHelper &baseMap, const std::string &key) {
      baseMap[key];//if we do not have this key-value pair before, insert it now (with zero for pointer).
      //find iterator for this key now :)
      fPosInMap = baseMap.find(key);
   }

   operator T * ()const {
      return (T*)fPosInMap->second;//const_cast<T*>(static_cast<const T *>(fPosInMap->second));
   }

   TGDMAssignmentHelper & operator = (const T * ptr) {
      fPosInMap->second = ptr;
      return *this;
   }
};

template<class T>
class TGDMMapHelper : public TGDMLBaseTGDMMapHelper {
public:
   TGDMAssignmentHelper<T> operator [](const std::string &key) {
      return TGDMAssignmentHelper<T>(*this, key);
   }
};

class TGDMLParse : public TObject {
public:

   const char* fWorldName; //top volume of geometry name
   TGeoVolume* fWorld; //top volume of geometry
   int fVolID;   //volume ID, incremented as assigned.
   int fFILENO; //Holds which level of file the parser is at
   TXMLEngine* fFileEngine[20]; //array of dom object pointers
   const char* fStartFile; //name of originating file
   const char* fCurrentFile; //current file name being parsed

   TGDMLParse() { //constructor

      fVolID = 0;
      fFILENO = 0;
   }

   virtual ~TGDMLParse() { //destructor

      for (size_t i = 0; i < fformvec.size(); i++)
         if (fformvec[i] != NULL) delete fformvec[i];
   }

   static TGeoVolume* StartGDML(const char* filename) {
      TGDMLParse* parser = new TGDMLParse;
      TGeoVolume* world = parser->GDMLReadFile(filename);
      return world;
   }

   TGeoVolume*       GDMLReadFile(const char* filename = "test.gdml");

private:

   const char*       ParseGDML(TXMLEngine* gdml, XMLNodePointer_t node) ;
   TString           GetScale(const char* unit);
   double            Evaluate(const char* evalline);
   const char*       NameShort(const char* name);

   //'define' section
   XMLNodePointer_t  ConProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  PosProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  RotProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  SclProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);

   //'materials' section
   XMLNodePointer_t  IsoProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLNodePointer_t parentn);
   XMLNodePointer_t  EleProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLNodePointer_t parentn, Bool_t hasIsotopes);
   XMLNodePointer_t  MatProcess(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr, int z);

   //'solids' section
   XMLNodePointer_t  BooSolid(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr, int num);
   XMLNodePointer_t  Box(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Paraboloid(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Arb8(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Tube(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  CutTube(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  Cone(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
   XMLNodePointer_t  ElCone(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr);
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
   XMLNodePointer_t  Ellipsoid(TXMLEngine* gdml, XMLNodePointer_t node, XMLAttrPointer_t attr); //not really implemented: just approximation to a TGeoBBox

   //'structure' section
   XMLNodePointer_t  VolProcess(TXMLEngine* gdml, XMLNodePointer_t node);
   XMLNodePointer_t  AssProcess(TXMLEngine* gdml, XMLNodePointer_t node);
   Int_t             SetAxis(const char* axisString); //Set Axis for Division

   //'setup' section
   XMLNodePointer_t  TopProcess(TXMLEngine* gdml, XMLNodePointer_t node);

   typedef TGDMMapHelper<TGeoTranslation> PosMap;
   typedef TGDMMapHelper<TGeoRotation> RotMap;
   typedef TGDMMapHelper<TGeoScale> SclMap;
   typedef TGDMMapHelper<TGeoElement> EleMap;
   typedef TGDMMapHelper<TGeoIsotope> IsoMap;
   typedef TGDMMapHelper<TGeoMaterial> MatMap;
   typedef TGDMMapHelper<TGeoMedium> MedMap;
   typedef TGDMMapHelper<TGeoMixture> MixMap;

   typedef TGDMMapHelper<TGeoShape> SolMap;
   typedef TGDMMapHelper<TGeoVolume> VolMap;
   typedef TGDMMapHelper<TGDMLRefl> ReflSolidMap;
   typedef TGDMMapHelper<const char> FileMap;
   typedef std::map<std::string, std::string> ReflectionsMap;
   typedef std::map<std::string, std::string> ReflVolMap;
   typedef std::map<std::string, double> FracMap;
   typedef std::vector<TFormula*> FormVec;

   PosMap fposmap;                //!Map containing position names and the TGeoTranslation for it
   RotMap frotmap;                //!Map containing rotation names and the TGeoRotation for it
   SclMap fsclmap;                //!Map containing scale names and the TGeoScale for it
   IsoMap fisomap;                //!Map containing isotope names and the TGeoIsotope for it
   EleMap felemap;                //!Map containing element names and the TGeoElement for it
   MatMap fmatmap;                //!Map containing material names and the TGeoMaterial for it
   MedMap fmedmap;                //!Map containing medium names and the TGeoMedium for it
   MixMap fmixmap;                //!Map containing mixture names and the TGeoMixture for it
   SolMap fsolmap;                //!Map containing solid names and the TGeoShape for it
   VolMap fvolmap;                //!Map containing volume names and the TGeoVolume for it
   ReflectionsMap freflectmap;    //!Map containing reflection names and the Solid name ir references to
   ReflSolidMap freflsolidmap;    //!Map containing reflection names and the TGDMLRefl for it - containing refl matrix
   ReflVolMap freflvolmap;        //!Map containing reflected volume names and the solid ref for it
   FileMap ffilemap;              //!Map containing files parsed during entire parsing, with their world volume name
   FormVec fformvec;              //!Vector containing constant functions for GDML constant definitions

   ClassDef(TGDMLParse, 0)    //imports GDML using DOM and binds it to ROOT
};

#endif
