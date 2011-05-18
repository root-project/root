// @(#)root/vmc:$Id$
// Authors: Alice collaboration 25/06/2002

/*************************************************************************
 * Copyright (C) 2006, Rene Brun and Fons Rademakers.                    *
 * Copyright (C) 2002, ALICE Experiment at CERN.                         *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualMCGeometry
#define ROOT_TVirtualMCGeometry

//
// Class TVirtualMCGeometry
// -------------------------
// Interface to Monte Carlo geometry construction
// (separated from VirtualMC)

#include "TNamed.h"

class TGeoHMatrix;
class TArrayD;
class TString;

class TVirtualMCGeometry : public TNamed {

public:
   // Standard constructor
   TVirtualMCGeometry(const char *name, const char *title);

   // Default constructor
   TVirtualMCGeometry();

   // Destructor
   virtual ~TVirtualMCGeometry();

   //
   // detector composition
   // ------------------------------------------------
   //

   // Define a material
   // kmat   number assigned to the material
   // name   material name
   // a      atomic mass in au
   // z      atomic number
   // dens   density in g/cm3
   // absl   absorption length in cm;
   //               if >=0 it is ignored and the program
   //               calculates it, if <0. -absl is taken
   // radl   radiation length in cm
   //               if >=0 it is ignored and the program
   //               calculates it, if <0. -radl is taken
   // buf    pointer to an array of user words
   // nwbuf  number of user words
   virtual void  Material(Int_t& kmat, const char* name, Double_t a,
                     Double_t z, Double_t dens, Double_t radl, Double_t absl,
                     Float_t* buf, Int_t nwbuf) = 0;

   // The same as previous but in double precision
   virtual void  Material(Int_t& kmat, const char* name, Double_t a,
                     Double_t z, Double_t dens, Double_t radl, Double_t absl,
                     Double_t* buf, Int_t nwbuf) = 0;

   // Define mixture or compound
   // with a number kmat composed by the basic nlmat materials defined
   // by arrays a, z and wmat
   //
   // If nlmat > 0 then wmat contains the proportion by
   // weights of each basic material in the mixture.
   //
   // If nlmat < 0 then wmat contains the number of atoms
   // of a given kind into the molecule of the compound.
   // In this case, wmat in output is changed to relative
   // weights.
   virtual void  Mixture(Int_t& kmat, const char *name, Float_t *a,
                     Float_t *z, Double_t dens, Int_t nlmat, Float_t *wmat) = 0;

   // The same as previous but in double precision
   virtual void  Mixture(Int_t& kmat, const char *name, Double_t *a,
                     Double_t *z, Double_t dens, Int_t nlmat, Double_t *wmat) = 0;

   // Define a medium.
   // kmed      tracking medium number assigned
   // name      tracking medium name
   // nmat      material number
   // isvol     sensitive volume flag
   // ifield    magnetic field:
   //                  - ifield = 0 if no magnetic field;
   //                  - ifield = -1 if user decision in guswim;
   //                  - ifield = 1 if tracking performed with g3rkuta;
   //                  - ifield = 2 if tracking
   // fieldm    max. field value (kilogauss)
   // tmaxfd    max. angle due to field (deg/step)
   // stemax    max. step allowed
   // deemax    max. fraction of energy lost in a step
   // epsil     tracking precision (cm)
   // stmin     min. step due to continuous processes (cm)
   // ubuf      pointer to an array of user words
   // nbuf      number of user words
   //  performed with g3helix; ifield = 3 if tracking performed with g3helx3.
   virtual void  Medium(Int_t& kmed, const char *name, Int_t nmat,
                     Int_t isvol, Int_t ifield, Double_t fieldm, Double_t tmaxfd,
                     Double_t stemax, Double_t deemax, Double_t epsil,
                     Double_t stmin, Float_t* ubuf, Int_t nbuf) = 0;

   // The same as previous but in double precision
   virtual void  Medium(Int_t& kmed, const char *name, Int_t nmat,
                     Int_t isvol, Int_t ifield, Double_t fieldm, Double_t tmaxfd,
                     Double_t stemax, Double_t deemax, Double_t epsil,
                     Double_t stmin, Double_t* ubuf, Int_t nbuf) = 0;

   // Define a rotation matrix
   // krot     rotation matrix number assigned
   // thetaX   polar angle for axis X
   // phiX     azimuthal angle for axis X
   // thetaY   polar angle for axis Y
   // phiY     azimuthal angle for axis Y
   // thetaZ   polar angle for axis Z
   // phiZ     azimuthal angle for axis Z
   virtual void  Matrix(Int_t& krot, Double_t thetaX, Double_t phiX,
                     Double_t thetaY, Double_t phiY, Double_t thetaZ,
                     Double_t phiZ) = 0;

   //
   // functions from GGEOM
   // ------------------------------------------------
   //

   // Create a new volume
   // name   Volume name
   // shape  Volume type
   // nmed   Tracking medium number
   // np     Number of shape parameters
   // upar   Vector containing shape parameters
   virtual Int_t  Gsvolu(const char *name, const char *shape, Int_t nmed,
                          Float_t *upar, Int_t np) = 0;

   // The same as previous but in double precision
   virtual Int_t  Gsvolu(const char *name, const char *shape, Int_t nmed,
                          Double_t *upar, Int_t np) = 0;

   // Create a new volume by dividing an existing one.
   // It divides a previously defined volume
   // name   Volume name
   // mother Mother volume name
   // ndiv   Number of divisions
   // iaxis  Axis value:
   //               X,Y,Z of CAXIS will be translated to 1,2,3 for IAXIS.
   virtual void  Gsdvn(const char *name, const char *mother, Int_t ndiv,
                         Int_t iaxis) = 0;

   // Create a new volume by dividing an existing one.
   // Divide mother into ndiv divisions called name
   // along axis iaxis starting at coordinate value c0i.
   // The new volume created will be medium number numed.
   virtual void  Gsdvn2(const char *name, const char *mother, Int_t ndiv,
                         Int_t iaxis, Double_t c0i, Int_t numed) = 0;

   // Create a new volume by dividing an existing one
   // Divide mother into divisions called name along
   // axis iaxis in steps of step. If not exactly divisible
   // will make as many as possible and will center them
   // with respect to the mother. Divisions will have medium
   // number numed. If numed is 0, numed of mother is taken.
   // ndvmx is the expected maximum number of divisions
   // (If 0, no protection tests are performed in Geant3)
   virtual void  Gsdvt(const char *name, const char *mother, Double_t step,
                         Int_t iaxis, Int_t numed, Int_t ndvmx) = 0;

   // Create a new volume by dividing an existing one
   // Divides mother into divisions called name along
   // axis iaxis starting at coordinate value c0 with step
   // size step.
   // The new volume created will have medium number numed.
   // If numed is 0, numed of mother is taken.
   // ndvmx is the expected maximum number of divisions
   // (If 0, no protection tests are performed in Geant3)
   virtual void  Gsdvt2(const char *name, const char *mother, Double_t step,
                         Int_t iaxis, Double_t c0, Int_t numed, Int_t ndvmx) = 0;

   // Flag volume name whose contents will have to be ordered
   // along axis iax, by setting the search flag to -iax
   // (Geant3 only)
   virtual void  Gsord(const char *name, Int_t iax) = 0;

   // Position a volume into an existing one.
   // It positions a previously defined volume in the mother.
   //   name   Volume name
   //   nr     Copy number of the volume
   //   mother Mother volume name
   //   x      X coord. of the volume in mother ref. sys.
   //   y      Y coord. of the volume in mother ref. sys.
   //   z      Z coord. of the volume in mother ref. sys.
   //   irot   Rotation matrix number w.r.t. mother ref. sys.
   //   konly  ONLY/MANY flag
   virtual void  Gspos(const char *name, Int_t nr, const char *mother,
                         Double_t x, Double_t y, Double_t z, Int_t irot,
                         const char *konly="ONLY") = 0;

   // Place a copy of generic volume name with user number
   //  nr inside mother, with its parameters upar(1..np)
   virtual void  Gsposp(const char *name, Int_t nr, const char *mother,
                         Double_t x, Double_t y, Double_t z, Int_t irot,
                         const char *konly, Float_t *upar, Int_t np) = 0;

   // The same as previous but in double precision
   virtual void  Gsposp(const char *name, Int_t nr, const char *mother,
                         Double_t x, Double_t y, Double_t z, Int_t irot,
                         const char *konly, Double_t *upar, Int_t np) = 0;

   // Helper function for resolving MANY.
   // Specify the ONLY volume that overlaps with the
   // specified MANY and has to be substracted.
   // (Geant4 only)
   virtual void  Gsbool(const char* onlyVolName, const char* manyVolName) = 0;

   //
   // functions for access to geometry
   // ------------------------------------------------
   //

   // Return the transformation matrix between the volume specified by
   // the path volumePath and the top or master volume.
   virtual Bool_t GetTransformation(const TString& volumePath,
                         TGeoHMatrix& matrix) = 0;

   // Return the name of the shape (shapeType)  and its parameters par
   // for the volume specified by the path volumePath .
   virtual Bool_t GetShape(const TString& volumePath,
                         TString& shapeType, TArrayD& par) = 0;

   // Return the material parameters for the volume specified by
   // the volumeName.
   virtual Bool_t GetMaterial(const TString& volumeName,
                               TString& name, Int_t& imat,
                               Double_t& a, Double_t& z, Double_t& density,
                               Double_t& radl, Double_t& inter, TArrayD& par) = 0;

   // Return the medium parameters for the volume specified by the
   // volumeName.
   virtual Bool_t GetMedium(const TString& volumeName,
                             TString& name, Int_t& imed,
                             Int_t& nmat, Int_t& isvol, Int_t& ifield,
                             Double_t& fieldm, Double_t& tmaxfd, Double_t& stemax,
                             Double_t& deemax, Double_t& epsil, Double_t& stmin,
                             TArrayD& par) = 0;

   // functions for drawing
   //virtual void  DrawOneSpec(const char* name) = 0;
   //virtual void  Gsatt(const char* name, const char* att, Int_t val) = 0;
   //virtual void  Gdraw(const char*,Double_t theta = 30, Double_t phi = 30,
   //                    Double_t psi = 0, Double_t u0 = 10, Double_t v0 = 10,
   //                    Double_t ul = 0.01, Double_t vl = 0.01) = 0;

   // Euclid
   // virtual void  WriteEuclid(const char*, const char*, Int_t, Int_t) = 0;

   //
   // get methods
   // ------------------------------------------------
   //


   // Return the unique numeric identifier for volume name volName
   virtual Int_t VolId(const char* volName) const = 0;

   // Return the volume name for a given volume identifier id
   virtual const char* VolName(Int_t id) const = 0;

   // Return the unique numeric identifier for medium name mediumName
   virtual Int_t MediumId(const char* mediumName) const = 0;

   // Return total number of volumes in the geometry
   virtual Int_t NofVolumes() const = 0;

   // Return number of daughters of the volume specified by volName
   virtual Int_t NofVolDaughters(const char* volName) const = 0;

   // Return the name of i-th daughter of the volume specified by volName
   virtual const char*  VolDaughterName(const char* volName, Int_t i) const = 0;

   // Return the copyNo of i-th daughter of the volume specified by volName
   virtual Int_t        VolDaughterCopyNo(const char* volName, Int_t i) const = 0;

   // Return material number for a given volume id
   virtual Int_t VolId2Mate(Int_t id) const = 0;

protected:
   TVirtualMCGeometry(const TVirtualMCGeometry& /*rhs*/);
   TVirtualMCGeometry & operator=(const TVirtualMCGeometry& /*rhs*/);

   ClassDef(TVirtualMCGeometry,1)  //Interface to Monte Carlo geometry construction
};

#endif //ROOT_TVirtualMCGeometry

