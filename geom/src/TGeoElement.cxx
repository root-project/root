// @(#)root/geom:$Name:  $:$Id: TGeoElement.cxx,v 1.7 2006/05/23 04:47:37 brun Exp $
// Author: Andrei Gheata   17/06/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// Full description with examples and pictures
//
//
//
//
#include"TObjArray.h"
#include "TGeoManager.h"
#include"TGeoElement.h"

// statics and globals

ClassImp(TGeoElement)

//______________________________________________________________________________
TGeoElement::TGeoElement()
{
// Default constructor
   SetDefined(kFALSE);
   SetUsed(kFALSE);
   fZ = 0;
   fA = 0.0;
}

//______________________________________________________________________________
TGeoElement::TGeoElement(const char *name, const char *title, Int_t z, Double_t a)
            :TNamed(name, title)
{
// Constructor
   SetDefined(kFALSE);
   SetUsed(kFALSE);
   fZ = z;
   fA = a;
}

ClassImp(TGeoElementTable)

//______________________________________________________________________________
TGeoElementTable *TGeoElement::GetElementTable() const
{
// Returns pointer to the table.
   return gGeoManager->GetElementTable();
}

//______________________________________________________________________________
TGeoElementTable::TGeoElementTable()
{
// default constructor
   fNelements = 0;
   fList      = 0;
}

//______________________________________________________________________________
TGeoElementTable::TGeoElementTable(Int_t /*nelements*/)
{
// constructor
   fNelements = 0;
   fList = new TObjArray(128);
   BuildDefaultElements();
}

//______________________________________________________________________________
TGeoElementTable::TGeoElementTable(const TGeoElementTable& get) :
  TObject(get),
  fNelements(get.fNelements),
  fList(get.fList)
{ 
   //copy constructor
}

//______________________________________________________________________________
TGeoElementTable& TGeoElementTable::operator=(const TGeoElementTable& get) 
{
   //equal operator
   if(this!=&get) {
      TObject::operator=(get);
      fNelements=get.fNelements;
      fList=get.fList;
   } 
   return *this;
}

//______________________________________________________________________________
TGeoElementTable::~TGeoElementTable()
{
// destructor
   if (fList) {
      fList->Delete();
      delete fList;
   }
}

//______________________________________________________________________________
void TGeoElementTable::AddElement(const char *name, const char *title, Int_t z, Double_t a)
{
// Add an element to the table.
   fList->AddAt(new TGeoElement(name,title,z,a), fNelements++);
}

//______________________________________________________________________________
void TGeoElementTable::BuildDefaultElements()
{
// Creates the default element table
   AddElement("VACUUM","VACUUM"   ,0, 0.0);
   AddElement("H"   ,"HYDROGEN"   ,1, 1.00794);
   AddElement("HE"  ,"HELIUM"     ,2, 4.002602);
   AddElement("LI"  ,"LITHIUM"    ,3, 6.941);
   AddElement("BE"  ,"BERYLLIUM"  ,4, 9.01218);
   AddElement("B"   ,"BORON"      ,5, 10.811);
   AddElement("C"   ,"CARBON"     ,6 ,12.0107);
   AddElement("N"   ,"NITROGEN"   ,7 ,14.00674);
   AddElement("O"   ,"OXYGEN"     ,8 ,15.9994);
   AddElement("F"   ,"FLUORINE"   ,9 ,18.9984032);
   AddElement("NE"  ,"NEON"       ,10 ,20.1797);
   AddElement("NA"  ,"SODIUM"     ,11 ,22.989770);
   AddElement("MG"  ,"MAGNESIUM"  ,12 ,24.3050);
   AddElement("AL"  ,"ALUMINIUM"  ,13 ,26.981538);
   AddElement("SI"  ,"SILICON"    ,14 ,28.0855);
   AddElement("P"   ,"PHOSPHORUS" ,15 ,30.973761);
   AddElement("S"   ,"SULFUR"     ,16 ,32.066);
   AddElement("CL"  ,"CHLORINE"   ,17 ,35.4527);
   AddElement("AR"  ,"ARGON"      ,18 ,39.948);
   AddElement("K"   ,"POTASSIUM"  ,19 ,39.0983);
   AddElement("CA"  ,"CALCIUM"    ,20 ,40.078);
   AddElement("SC"  ,"SCANDIUM"   ,21 ,44.955910);
   AddElement("TI"  ,"TITANIUM"   ,22 ,47.867);
   AddElement("V"   ,"VANADIUM"   ,23 ,50.9415);
   AddElement("CR"  ,"CHROMIUM"   ,24 ,51.9961);
   AddElement("MN"  ,"MANGANESE"  ,25 ,54.938049);
   AddElement("FE"  ,"IRON"       ,26 ,55.845);
   AddElement("CO"  ,"COBALT"     ,27 ,58.933200);
   AddElement("NI"  ,"NICKEL"     ,28 ,58.6934);
   AddElement("CU"  ,"COPPER"     ,29 ,63.546);
   AddElement("ZN"  ,"ZINC"       ,30 ,65.39);
   AddElement("GA"  ,"GALLIUM"    ,31 ,69.723);
   AddElement("GE"  ,"GERMANIUM"  ,32 ,72.61);
   AddElement("AS"  ,"ARSENIC"    ,33 ,74.92160);
   AddElement("SE"  ,"SELENIUM"   ,34 ,78.96);
   AddElement("BR"  ,"BROMINE"    ,35 ,79.904);
   AddElement("KR"  ,"KRYPTON"    ,36 ,83.80);
   AddElement("RB"  ,"RUBIDIUM"   ,37 ,85.4678);
   AddElement("SR"  ,"STRONTIUM"  ,38 ,87.62);
   AddElement("Y"   ,"YTTRIUM"    ,39 ,88.90585);
   AddElement("ZR"  ,"ZIRCONIUM"  ,40 ,91.224);
   AddElement("NB"  ,"NIOBIUM"    ,41 ,92.90638);
   AddElement("MO"  ,"MOLYBDENUM" ,42 ,95.94);
   AddElement("TC"  ,"TECHNETIUM" ,43 ,98.0);
   AddElement("RU"  ,"RUTHENIUM"  ,44 ,101.07);
   AddElement("RH"  ,"RHODIUM"    ,45 ,102.90550);
   AddElement("PD"  ,"PALLADIUM"  ,46 ,106.42);
   AddElement("AG"  ,"SILVER"     ,47 ,107.8682);
   AddElement("CD"  ,"CADMIUM"    ,48 ,112.411);
   AddElement("IN"  ,"INDIUM"     ,49 ,114.818);
   AddElement("SN"  ,"TIN"        ,50 ,118.710);
   AddElement("SB"  ,"ANTIMONY"   ,51 ,121.760);
   AddElement("TE"  ,"TELLURIUM"  ,52 ,127.60);
   AddElement("I"   ,"IODINE"     ,53 ,126.90447);
   AddElement("XE"  ,"XENON"      ,54 ,131.29);
   AddElement("CS"  ,"CESIUM"     ,55 ,132.90545);
   AddElement("BA"  ,"BARIUM"     ,56 ,137.327);
   AddElement("LA"  ,"LANTHANUM"  ,57 ,138.9055);
   AddElement("CE"  ,"CERIUM"     ,58 ,140.116);
   AddElement("PR"  ,"PRASEODYMIUM" ,59 ,140.90765);
   AddElement("ND"  ,"NEODYMIUM"  ,60 ,144.24);
   AddElement("PM"  ,"PROMETHIUM" ,61 ,145.0);
   AddElement("SM"  ,"SAMARIUM"   ,62 ,150.36);
   AddElement("EU"  ,"EUROPIUM"   ,63 ,151.964);
   AddElement("GD"  ,"GADOLINIUM" ,64 ,157.25);
   AddElement("TB"  ,"TERBIUM"    ,65 ,158.92534);
   AddElement("DY"  ,"DYSPROSIUM" ,66 ,162.50);
   AddElement("HO"  ,"HOLMIUM"    ,67 ,164.93032);
   AddElement("ER"  ,"ERBIUM"     ,68 ,167.26);
   AddElement("TM"  ,"THULIUM"    ,69 ,168.93421);
   AddElement("YB"  ,"YTTERBIUM"  ,70 ,173.04);
   AddElement("LU"  ,"LUTETIUM"   ,71 ,174.967);
   AddElement("HF"  ,"HAFNIUM"    ,72 ,178.49);
   AddElement("TA"  ,"TANTALUM"   ,73 ,180.9479);
   AddElement("W"   ,"TUNGSTEN"   ,74 ,183.84);
   AddElement("RE"  ,"RHENIUM"    ,75 ,186.207);
   AddElement("OS"  ,"OSMIUM"     ,76 ,190.23);
   AddElement("IR"  ,"IRIDIUM"    ,77 ,192.217);
   AddElement("PT"  ,"PLATINUM"   ,78 ,195.078);
   AddElement("AU"  ,"GOLD"       ,79 ,196.96655);
   AddElement("HG"  ,"MERCURY"    ,80 ,200.59);
   AddElement("TL"  ,"THALLIUM"   ,81 ,204.3833);
   AddElement("PB"  ,"LEAD"       ,82 ,207.2);
   AddElement("BI"  ,"BISMUTH"    ,83 ,208.98038);
   AddElement("PO"  ,"POLONIUM"   ,84 ,209.0);
   AddElement("AT"  ,"ASTATINE"   ,85 ,210.0);
   AddElement("RN"  ,"RADON"      ,86 ,222.0);
   AddElement("FR"  ,"FRANCIUM"   ,87 ,223.0);
   AddElement("RA"  ,"RADIUM"     ,88 ,226.0);
   AddElement("AC"  ,"ACTINIUM"   ,89 ,227.0);
   AddElement("TH"  ,"THORIUM"    ,90 ,232.0381);
   AddElement("PA"  ,"PROTACTINIUM" ,91 ,231.03588);
   AddElement("U"   ,"URANIUM"    ,92 ,238.0289);
   AddElement("NP"  ,"NEPTUNIUM"  ,93 ,237.0);
   AddElement("PU"  ,"PLUTONIUM"  ,94 ,244.0);
   AddElement("AM"  ,"AMERICIUM"  ,95 ,243.0);
   AddElement("CM"  ,"CURIUM"     ,96 ,247.0);
   AddElement("BK"  ,"BERKELIUM"  ,97 ,247.0);
   AddElement("CF"  ,"CALIFORNIUM",98 ,251.0);
   AddElement("ES"  ,"EINSTEINIUM",99 ,252.0);
   AddElement("FM"  ,"FERMIUM"    ,100 ,257.0);
   AddElement("MD"  ,"MENDELEVIUM",101 ,258.0);
   AddElement("NO"  ,"NOBELIUM"   ,102 ,259.0);
   AddElement("LR"  ,"LAWRENCIUM" ,103 ,262.0);
   AddElement("RF"  ,"RUTHERFORDIUM" ,104,261.0);
   AddElement("DB"  ,"DUBNIUM" ,105 ,262.0);
   AddElement("SG"  ,"SEABORGIUM" ,106 ,263.0);
   AddElement("BH"  ,"BOHRIUM"    ,107 ,262.0);
   AddElement("HS"  ,"HASSIUM"    ,108 ,265.0);
   AddElement("MT"  ,"MEITNERIUM" ,109 ,266.0);
   AddElement("UUN" ,"UNUNNILIUM" ,110 ,269.0);
   AddElement("UUU" ,"UNUNUNIUM"  ,111 ,272.0);
   AddElement("UUB" ,"UNUNBIUM"   ,112 ,277.0);
}

//______________________________________________________________________________
TGeoElement *TGeoElementTable::FindElement(const char *name)
{
// Search an element by symbol or full name
   TString s(name);
   s.ToUpper();
   TGeoElement *elem;
   elem = (TGeoElement*)fList->FindObject(s.Data());
   if (elem) return elem;
   // Search by full name
   TIter next(fList);
   while ((elem=(TGeoElement*)next())) {
      if (s == elem->GetTitle()) return elem;
   }
   return 0;
}      
