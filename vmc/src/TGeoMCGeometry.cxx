// @(#)root/mc:$Name:  $:$Id: TGeoMCGeometry.cxx,v 1.3 2003/09/23 14:03:15 brun Exp $
// Authors: ... 25/06/2002

//______________________________________________________________________________
//
// Implementation of the TVirtualMCGeometry interface
// for building TGeo geometry.
//______________________________________________________________________________

   
#include "TGeoMCGeometry.h"
#include "TGeoManager.h" 
#include "TGeoVolume.h" 

ClassImp(TGeoMCGeometry)

TGeoMCGeometry* TGeoMCGeometry::fgInstance=0;

//_____________________________________________________________________________
TGeoMCGeometry::TGeoMCGeometry(const char *name, const char *title,
                               Bool_t g3CompatibleVolumeNames) 
  : TVirtualMCGeometry(name, title),
    fG3CompatibleVolumeNames(g3CompatibleVolumeNames)
{
  //
  // Standard constructor
  //
}

//_____________________________________________________________________________
TGeoMCGeometry::TGeoMCGeometry()
  : TVirtualMCGeometry(),
    fG3CompatibleVolumeNames(kFALSE)
{    
  //
  // Default constructor
  //
}

//_____________________________________________________________________________
TGeoMCGeometry::~TGeoMCGeometry() 
{
  //
  // Destructor
  //
  fgInstance=0;
}

//
// private methods
//


//_____________________________________________________________________________
TGeoMCGeometry::TGeoMCGeometry(const TGeoMCGeometry &geom)
  : TVirtualMCGeometry(geom)
{
  //
  // Copy constructor
  //
}

//_____________________________________________________________________________
Double_t* TGeoMCGeometry::CreateDoubleArray(Float_t* array, Int_t size) const
{
// Converts Float_t* array to Double_t*,
// !! The new array has to be deleted by user.
// ---

  Double_t* doubleArray;
  if (size>0) {
    doubleArray = new Double_t[size]; 
    for (Int_t i=0; i<size; i++) doubleArray[i] = array[i];
  }
  else {
    //doubleArray = 0; 
    doubleArray = new Double_t[1]; 
  }  
  return doubleArray;
}

//______________________________________________________________________________
void TGeoMCGeometry::Vname(const char *name, char *vname) const
{
  //
  //  convert name to upper case. Make vname at least 4 chars
  //
  if (fG3CompatibleVolumeNames) {
    Int_t l = strlen(name);
    Int_t i;
    l = l < 4 ? l : 4;
    for (i=0;i<l;i++) vname[i] = toupper(name[i]);
    for (i=l;i<4;i++) vname[i] = ' ';
    vname[4] = 0;
  }
  else {
    Int_t l = strlen(name);
    for (Int_t i=0;i<l;i++) vname[i] = name[i];
    vname[l] = 0;
  }
}
 
//
// public methods
//

//_____________________________________________________________________________
void TGeoMCGeometry::Material(Int_t& kmat, const char* name, Double_t a, Double_t z,
		       Double_t dens, Double_t radl, Double_t absl, Float_t* buf,
		       Int_t nwbuf)
{
  //
  // Defines a Material
  // 
  //  kmat               number assigned to the material
  //  name               material name
  //  a                  atomic mass in au
  //  z                  atomic number
  //  dens               density in g/cm3
  //  absl               absorbtion length in cm
  //                     if >=0 it is ignored and the program 
  //                     calculates it, if <0. -absl is taken
  //  radl               radiation length in cm
  //                     if >=0 it is ignored and the program 
  //                     calculates it, if <0. -radl is taken
  //  buf                pointer to an array of user words
  //  nbuf               number of user words
  //
  
  Double_t* dbuf = CreateDoubleArray(buf, nwbuf);  
  Material(kmat, name, a, z, dens, radl, absl, dbuf, nwbuf);
  delete [] dbuf;
}  

//_____________________________________________________________________________
void TGeoMCGeometry::Material(Int_t& kmat, const char* name, Double_t a, Double_t z,
		       Double_t dens, Double_t radl, Double_t absl, Double_t* /*buf*/,
		       Int_t /*nwbuf*/)
{
  //
  // Defines a Material
  // 
  //  kmat               number assigned to the material
  //  name               material name
  //  a                  atomic mass in au
  //  z                  atomic number
  //  dens               density in g/cm3
  //  absl               absorbtion length in cm
  //                     if >=0 it is ignored and the program 
  //                     calculates it, if <0. -absl is taken
  //  radl               radiation length in cm
  //                     if >=0 it is ignored and the program 
  //                     calculates it, if <0. -radl is taken
  //  buf                pointer to an array of user words
  //  nbuf               number of user words
  //

  gGeoManager->Material(name, a, z, dens, kmat, radl, absl);
}

//_____________________________________________________________________________
void TGeoMCGeometry::Mixture(Int_t& kmat, const char* name, Float_t* a, Float_t* z, 
		      Double_t dens, Int_t nlmat, Float_t* wmat)
{
  //
  // Defines mixture OR COMPOUND IMAT as composed by 
  // THE BASIC NLMAT materials defined by arrays A,Z and WMAT
  // 
  // If NLMAT > 0 then wmat contains the proportion by
  // weights of each basic material in the mixture. 
  // 
  // If nlmat < 0 then WMAT contains the number of atoms 
  // of a given kind into the molecule of the COMPOUND
  // In this case, WMAT in output is changed to relative
  // weigths.
  //
  
  Double_t* da = CreateDoubleArray(a, TMath::Abs(nlmat));  
  Double_t* dz = CreateDoubleArray(z, TMath::Abs(nlmat));  
  Double_t* dwmat = CreateDoubleArray(wmat, TMath::Abs(nlmat));  

  Mixture(kmat, name, da, dz, dens, nlmat, dwmat);
  for (Int_t i=0; i<nlmat; i++) {
    a[i] = da[i]; z[i] = dz[i]; wmat[i] = dwmat[i];
  }  

  delete [] da;
  delete [] dz;
  delete [] dwmat;
}

//_____________________________________________________________________________
void TGeoMCGeometry::Mixture(Int_t& kmat, const char* name, Double_t* a, Double_t* z, 
		      Double_t dens, Int_t nlmat, Double_t* wmat)
{
  //
  // Defines mixture OR COMPOUND IMAT as composed by 
  // THE BASIC NLMAT materials defined by arrays A,Z and WMAT
  // 
  // If NLMAT > 0 then wmat contains the proportion by
  // weights of each basic material in the mixture. 
  // 
  // If nlmat < 0 then WMAT contains the number of atoms 
  // of a given kind into the molecule of the COMPOUND
  // In this case, WMAT in output is changed to relative
  // weigths.
  //

  if (nlmat < 0) {
     nlmat = - nlmat;
     Double_t amol = 0;
     Int_t i;
     for (i=0;i<nlmat;i++) {
        amol += a[i]*wmat[i];
     }
     for (i=0;i<nlmat;i++) {
        wmat[i] *= a[i]/amol;
     }
  }
  gGeoManager->Mixture(name, a, z, dens, nlmat, wmat, kmat);
}

//_____________________________________________________________________________
void TGeoMCGeometry::Medium(Int_t& kmed, const char* name, Int_t nmat, Int_t isvol,
		     Int_t ifield, Double_t fieldm, Double_t tmaxfd,
		     Double_t stemax, Double_t deemax, Double_t epsil,
		     Double_t stmin, Float_t* ubuf, Int_t nbuf)
{
  //
  //  kmed      tracking medium number assigned
  //  name      tracking medium name
  //  nmat      material number
  //  isvol     sensitive volume flag
  //  ifield    magnetic field
  //  fieldm    max. field value (kilogauss)
  //  tmaxfd    max. angle due to field (deg/step)
  //  stemax    max. step allowed
  //  deemax    max. fraction of energy lost in a step
  //  epsil     tracking precision (cm)
  //  stmin     min. step due to continuous processes (cm)
  //
  //  ifield = 0 if no magnetic field; ifield = -1 if user decision in guswim;
  //  ifield = 1 if tracking performed with g3rkuta; ifield = 2 if tracking
  //  performed with g3helix; ifield = 3 if tracking performed with g3helx3.
  //  

  //printf("Creating mediuma: %s, numed=%d, nmat=%d\n",name,kmed,nmat);
  Double_t* dubuf = CreateDoubleArray(ubuf, nbuf);  
  Medium(kmed, name, nmat, isvol, ifield, fieldm, tmaxfd, stemax, deemax, epsil,
         stmin, dubuf, nbuf);
  delete [] dubuf;	 
}

//_____________________________________________________________________________
void TGeoMCGeometry::Medium(Int_t& kmed, const char* name, Int_t nmat, Int_t isvol,
		     Int_t ifield, Double_t fieldm, Double_t tmaxfd,
		     Double_t stemax, Double_t deemax, Double_t epsil,
		     Double_t stmin, Double_t* /*ubuf*/, Int_t /*nbuf*/)
{
  //
  //  kmed      tracking medium number assigned
  //  name      tracking medium name
  //  nmat      material number
  //  isvol     sensitive volume flag
  //  ifield    magnetic field
  //  fieldm    max. field value (kilogauss)
  //  tmaxfd    max. angle due to field (deg/step)
  //  stemax    max. step allowed
  //  deemax    max. fraction of energy lost in a step
  //  epsil     tracking precision (cm)
  //  stmin     min. step due to continuos processes (cm)
  //
  //  ifield = 0 if no magnetic field; ifield = -1 if user decision in guswim;
  //  ifield = 1 if tracking performed with g3rkuta; ifield = 2 if tracking
  //  performed with g3helix; ifield = 3 if tracking performed with g3helx3.
  //  

  gGeoManager->Medium(name,kmed,nmat, isvol, ifield, fieldm, tmaxfd, stemax,deemax, epsil, stmin);
}

//_____________________________________________________________________________
void TGeoMCGeometry::Matrix(Int_t& krot, Double_t thex, Double_t phix, Double_t they,
		     Double_t phiy, Double_t thez, Double_t phiz)
{
  //
  //  krot     rotation matrix number assigned
  //  theta1   polar angle for axis i
  //  phi1     azimuthal angle for axis i
  //  theta2   polar angle for axis ii
  //  phi2     azimuthal angle for axis ii
  //  theta3   polar angle for axis iii
  //  phi3     azimuthal angle for axis iii
  //
  //  it defines the rotation matrix number irot.
  //  

  krot = gGeoManager->GetListOfMatrices()->GetEntriesFast();
  gGeoManager->Matrix(krot, thex, phix, they, phiy, thez, phiz);  
}

//_____________________________________________________________________________
Int_t TGeoMCGeometry::Gsvolu(const char *name, const char *shape, Int_t nmed,  
		      Float_t *upar, Int_t npar) 
{ 
  //
  //  NAME   Volume name
  //  SHAPE  Volume type
  //  NUMED  Tracking medium number
  //  NPAR   Number of shape parameters
  //  UPAR   Vector containing shape parameters
  //
  //  It creates a new volume in the JVOLUM data structure.
  //  

  Double_t* dupar = CreateDoubleArray(upar, npar);
  Int_t id = Gsvolu(name, shape, nmed, dupar, npar);
  delete [] dupar;  
  return id;
} 

//_____________________________________________________________________________
Int_t TGeoMCGeometry::Gsvolu(const char *name, const char *shape, Int_t nmed,  
		      Double_t *upar, Int_t npar) 
{ 
  //
  //  NAME   Volume name
  //  SHAPE  Volume type
  //  NUMED  Tracking medium number
  //  NPAR   Number of shape parameters
  //  UPAR   Vector containing shape parameters
  //
  //  It creates a new volume in the JVOLUM data structure.
  //  

  char vname[5];
  Vname(name,vname);
  char vshape[5];
  Vname(shape,vshape);

  TGeoVolume* vol = gGeoManager->Volume(vname, vshape, nmed, upar, npar); 
  return vol->GetNumber();
} 
 
//_____________________________________________________________________________
void  TGeoMCGeometry::Gsdvn(const char *name, const char *mother, Int_t ndiv,
		     Int_t iaxis) 
{ 
  //
  // Create a new volume by dividing an existing one
  // 
  //  NAME   Volume name
  //  MOTHER Mother volume name
  //  NDIV   Number of divisions
  //  IAXIS  Axis value
  //
  //  X,Y,Z of CAXIS will be translated to 1,2,3 for IAXIS.
  //  It divides a previously defined volume.
  //  
  char vname[5];
  Vname(name,vname);
  char vmother[5];
  Vname(mother,vmother);
 
  gGeoManager->Division(vname, vmother, iaxis, ndiv, 0, 0, 0, "n");
} 
 
//_____________________________________________________________________________
void  TGeoMCGeometry::Gsdvn2(const char *name, const char *mother, Int_t ndiv,
		      Int_t iaxis, Double_t c0i, Int_t numed) 
{ 
  //
  // Create a new volume by dividing an existing one
  // 
  // Divides mother into ndiv divisions called name
  // along axis iaxis starting at coordinate value c0.
  // the new volume created will be medium number numed.
  //
  char vname[5];
  Vname(name,vname);
  char vmother[5];
  Vname(mother,vmother);
  
  gGeoManager->Division(vname, vmother, iaxis, ndiv, c0i, 0, numed, "nx");
} 
//_____________________________________________________________________________
void  TGeoMCGeometry::Gsdvt(const char *name, const char *mother, Double_t step,
		     Int_t iaxis, Int_t numed, Int_t /*ndvmx*/) 
{ 
  //
  // Create a new volume by dividing an existing one
  // 
  //       Divides MOTHER into divisions called NAME along
  //       axis IAXIS in steps of STEP. If not exactly divisible 
  //       will make as many as possible and will centre them 
  //       with respect to the mother. Divisions will have medium 
  //       number NUMED. If NUMED is 0, NUMED of MOTHER is taken.
  //       NDVMX is the expected maximum number of divisions
  //          (If 0, no protection tests are performed) 
  //
  char vname[5];
  Vname(name,vname);
  char vmother[5];
  Vname(mother,vmother);
  
  gGeoManager->Division(vname, vmother, iaxis, 0, 0, step, numed, "s");
} 

//_____________________________________________________________________________
void  TGeoMCGeometry::Gsdvt2(const char *name, const char *mother, Double_t step,
		      Int_t iaxis, Double_t c0, Int_t numed, Int_t /*ndvmx*/) 
{ 
  //
  // Create a new volume by dividing an existing one
  //                                                                    
  //           Divides MOTHER into divisions called NAME along          
  //            axis IAXIS starting at coordinate value C0 with step    
  //            size STEP.                                              
  //           The new volume created will have medium number NUMED.    
  //           If NUMED is 0, NUMED of mother is taken.                 
  //           NDVMX is the expected maximum number of divisions        
  //             (If 0, no protection tests are performed)              
  //
  char vname[5];
  Vname(name,vname);
  char vmother[5];
  Vname(mother,vmother);
  
  gGeoManager->Division(vname, vmother, iaxis, 0, c0, step, numed, "sx");
} 

//_____________________________________________________________________________
void  TGeoMCGeometry::Gsord(const char * /*name*/, Int_t /*iax*/) 
{ 
  //
  //    Flags volume CHNAME whose contents will have to be ordered 
  //    along axis IAX, by setting the search flag to -IAX
  //           IAX = 1    X axis 
  //           IAX = 2    Y axis 
  //           IAX = 3    Z axis 
  //           IAX = 4    Rxy (static ordering only  -> GTMEDI)
  //           IAX = 14   Rxy (also dynamic ordering -> GTNEXT)
  //           IAX = 5    Rxyz (static ordering only -> GTMEDI)
  //           IAX = 15   Rxyz (also dynamic ordering -> GTNEXT)
  //           IAX = 6    PHI   (PHI=0 => X axis)
  //           IAX = 7    THETA (THETA=0 => Z axis)
  //

  // TBC - keep this function
  // nothing to be done for TGeo  //xx
} 
 
//_____________________________________________________________________________
void  TGeoMCGeometry::Gspos(const char *name, Int_t nr, const char *mother, Double_t x,
		     Double_t y, Double_t z, Int_t irot, const char *konly) 
{ 
  //
  // Position a volume into an existing one
  //
  //  NAME   Volume name
  //  NUMBER Copy number of the volume
  //  MOTHER Mother volume name
  //  X      X coord. of the volume in mother ref. sys.
  //  Y      Y coord. of the volume in mother ref. sys.
  //  Z      Z coord. of the volume in mother ref. sys.
  //  IROT   Rotation matrix number w.r.t. mother ref. sys.
  //  ONLY   ONLY/MANY flag
  //
  //  It positions a previously defined volume in the mother.
  //  
    
  TString only = konly;
  only.ToLower();
  Bool_t isOnly = kFALSE;
  if (only.Contains("only")) isOnly = kTRUE;
  char vname[5];
  Vname(name,vname);
  char vmother[5];
  Vname(mother,vmother);
  
  Double_t *upar=0;
  gGeoManager->Node(vname, nr, vmother, x, y, z, irot, isOnly, upar);
} 
 
//_____________________________________________________________________________
void  TGeoMCGeometry::Gsposp(const char *name, Int_t nr, const char *mother,  
		      Double_t x, Double_t y, Double_t z, Int_t irot,
		      const char *konly, Float_t *upar, Int_t np ) 
{ 
  //
  //      Place a copy of generic volume NAME with user number
  //      NR inside MOTHER, with its parameters UPAR(1..NP)
  //

  Double_t* dupar = CreateDoubleArray(upar, np);
  Gsposp(name, nr, mother, x, y, z, irot, konly, dupar, np); 
  delete [] dupar;
} 
 
//_____________________________________________________________________________
void  TGeoMCGeometry::Gsposp(const char *name, Int_t nr, const char *mother,  
		      Double_t x, Double_t y, Double_t z, Int_t irot,
		      const char *konly, Double_t *upar, Int_t np ) 
{ 
  //
  //      Place a copy of generic volume NAME with user number
  //      NR inside MOTHER, with its parameters UPAR(1..NP)
  //

  TString only = konly;
  only.ToLower();
  Bool_t isOnly = kFALSE;
  if (only.Contains("only")) isOnly = kTRUE;
  char vname[5];
  Vname(name,vname);
  char vmother[5];
  Vname(mother,vmother);

  gGeoManager->Node(vname,nr,vmother, x,y,z,irot,isOnly,upar,np);
} 
 
//_____________________________________________________________________________
Int_t TGeoMCGeometry::VolId(const Text_t *name) const
{
  //
  // Return the unique numeric identifier for volume name
  //

  Int_t uid = gGeoManager->GetUID(name);
  if (uid<0) {
     printf("VolId: Volume %s not found\n",name);
     return 0;
  }
  return uid;
}

//_____________________________________________________________________________
const char* TGeoMCGeometry::VolName(Int_t id) const
{
  //
  // Return the volume name given the volume identifier
  //

  TGeoVolume *volume = gGeoManager->GetVolume(id);
  if (!volume) {
     Error("VolName","volume with id=%d does not exist",id);
     return "NULL";
  }
  return volume->GetName();
}

//_____________________________________________________________________________
Int_t TGeoMCGeometry::NofVolumes() const 
{
  //
  // Return total number of volumes in the geometry
  //

  return gGeoManager->GetListOfUVolumes()->GetEntriesFast()-1;
}

//_____________________________________________________________________________
Int_t TGeoMCGeometry::VolId2Mate(Int_t id) const 
{
  //
  // Return material number for a given volume id
  //

  TGeoVolume *volume = gGeoManager->GetVolume(id);
  if (!volume) {
     Error("VolId2Mate","volume with id=%d does not exist",id);
     return 0;
  }
  TGeoMedium *med = volume->GetMedium();
  if (!med) return 0;
  return med->GetId();
}

