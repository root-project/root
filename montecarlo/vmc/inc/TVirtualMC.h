// @(#)root/vmc:$Name:  $:$Id$
// Authors: Ivana Hrivnacova, Rene Brun, Federico Carminati 13/04/2002

/*************************************************************************
 * Copyright (C) 2006, Rene Brun and Fons Rademakers.                    *
 * Copyright (C) 2002, ALICE Experiment at CERN.                         *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualMC
#define ROOT_TVirtualMC

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                                                                           //
//   Abstract Monte Carlo interface                                          //
//                                                                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "TMCProcess.h"
#include "TMCParticleType.h"
#include "TMCOptical.h"
#include "TMCtls.h"
#include "TVirtualMCApplication.h"
#include "TVirtualMCStack.h"
#include "TMCManagerStack.h"
#include "TVirtualMCDecayer.h"
#include "TVirtualMagField.h"
#include "TRandom.h"
#include "TString.h"

class TLorentzVector;
class TGeoHMatrix;
class TArrayI;
class TArrayD;
class TVirtualMCSensitiveDetector;

class TVirtualMC : public TNamed {

   // To have access to private methods
   friend class TMCManager;

public:
   /// Standard constructor
   ///
   /// isRootGeometrySupported = True if implementation of TVirtualMC
   ///        supports geometry defined with TGeo
   TVirtualMC(const char *name, const char *title, Bool_t isRootGeometrySupported = kFALSE);

   /// Default constructor
   TVirtualMC();

   /// Destructor
   virtual ~TVirtualMC();

   /// Static access method
   static TVirtualMC *GetMC();

   //
   // ------------------------------------------------
   // methods for building/management of geometry
   // ------------------------------------------------
   //

   /// Info about supporting geometry defined via Root
   virtual Bool_t IsRootGeometrySupported() const = 0;

   //
   // functions from GCONS
   // ------------------------------------------------
   //

   /// Define a material
   /// - kmat   number assigned to the material
   /// - name   material name
   /// - a      atomic mass in au
   /// - z      atomic number
   /// - dens   density in g/cm3
   /// - absl   absorption length in cm;
   ///               if >=0 it is ignored and the program
   ///               calculates it, if <0. -absl is taken
   /// - radl   radiation length in cm
   ///               if >=0 it is ignored and the program
   ///               calculates it, if <0. -radl is taken
   /// - buf    pointer to an array of user words
   /// - nwbuf  number of user words
   virtual void Material(Int_t &kmat, const char *name, Double_t a, Double_t z, Double_t dens, Double_t radl,
                         Double_t absl, Float_t *buf, Int_t nwbuf) = 0;

   /// The same as previous but in double precision
   virtual void Material(Int_t &kmat, const char *name, Double_t a, Double_t z, Double_t dens, Double_t radl,
                         Double_t absl, Double_t *buf, Int_t nwbuf) = 0;

   /// Define a mixture or a compound
   /// with a number kmat composed by the basic nlmat materials defined
   /// by arrays a, z and wmat
   ///
   /// If nlmat > 0 then wmat contains the proportion by
   /// weights of each basic material in the mixture.
   ///
   /// If nlmat < 0 then wmat contains the number of atoms
   /// of a given kind into the molecule of the compound.
   /// In this case, wmat in output is changed to relative
   /// weights.
   virtual void
   Mixture(Int_t &kmat, const char *name, Float_t *a, Float_t *z, Double_t dens, Int_t nlmat, Float_t *wmat) = 0;

   /// The same as previous but in double precision
   virtual void
   Mixture(Int_t &kmat, const char *name, Double_t *a, Double_t *z, Double_t dens, Int_t nlmat, Double_t *wmat) = 0;

   /// Define a medium.
   /// - kmed      tracking medium number assigned
   /// - name      tracking medium name
   /// - nmat      material number
   /// - isvol     sensitive volume flag
   /// - ifield    magnetic field:
   ///                  - ifield = 0 if no magnetic field;
   ///                  - ifield = -1 if user decision in guswim;
   ///                  - ifield = 1 if tracking performed with g3rkuta;
   ///                  - ifield = 2 if tracking performed with g3helix;
   ///                  - ifield = 3 if tracking performed with g3helx3.
   /// - fieldm    max. field value (kilogauss)
   /// - tmaxfd    max. angle due to field (deg/step)
   /// - stemax    max. step allowed
   /// - deemax    max. fraction of energy lost in a step
   /// - epsil     tracking precision (cm)
   /// - stmin     min. step due to continuous processes (cm)
   /// - ubuf      pointer to an array of user words
   /// - nbuf      number of user words
   virtual void Medium(Int_t &kmed, const char *name, Int_t nmat, Int_t isvol, Int_t ifield, Double_t fieldm,
                       Double_t tmaxfd, Double_t stemax, Double_t deemax, Double_t epsil, Double_t stmin, Float_t *ubuf,
                       Int_t nbuf) = 0;

   /// The same as previous but in double precision
   virtual void Medium(Int_t &kmed, const char *name, Int_t nmat, Int_t isvol, Int_t ifield, Double_t fieldm,
                       Double_t tmaxfd, Double_t stemax, Double_t deemax, Double_t epsil, Double_t stmin,
                       Double_t *ubuf, Int_t nbuf) = 0;

   /// Define a rotation matrix
   /// - krot     rotation matrix number assigned
   /// - thetaX   polar angle for axis X
   /// - phiX     azimuthal angle for axis X
   /// - thetaY   polar angle for axis Y
   /// - phiY     azimuthal angle for axis Y
   /// - thetaZ   polar angle for axis Z
   /// - phiZ     azimuthal angle for axis Z
   virtual void Matrix(Int_t &krot, Double_t thetaX, Double_t phiX, Double_t thetaY, Double_t phiY, Double_t thetaZ,
                       Double_t phiZ) = 0;

   /// Change the value of cut or mechanism param
   /// to a new value parval for tracking medium itmed.
   /// In Geant3, the  data  structure JTMED contains the standard tracking
   /// parameters (CUTS and flags to control the physics processes)  which
   /// are used  by default for all tracking media.
   /// It is possible to redefine individually with this function any of these
   /// parameters for a given tracking medium.
   /// - itmed   tracking medium number
   /// - param   is a character string (variable name)
   /// - parval  must be given as a floating point.
   virtual void Gstpar(Int_t itmed, const char *param, Double_t parval) = 0;

   //
   // functions from GGEOM
   // ------------------------------------------------
   //

   /// Create a new volume
   /// - name   Volume name
   /// - shape  Volume type
   /// - nmed   Tracking medium number
   /// - np     Number of shape parameters
   /// - upar   Vector containing shape parameters
   virtual Int_t Gsvolu(const char *name, const char *shape, Int_t nmed, Float_t *upar, Int_t np) = 0;

   /// The same as previous but in double precision
   virtual Int_t Gsvolu(const char *name, const char *shape, Int_t nmed, Double_t *upar, Int_t np) = 0;

   /// Create a new volume by dividing an existing one.
   /// It divides a previously defined volume
   /// - name   Volume name
   /// - mother Mother volume name
   /// - ndiv   Number of divisions
   /// - iaxis  Axis value:
   ///               X,Y,Z of CAXIS will be translated to 1,2,3 for IAXIS.
   virtual void Gsdvn(const char *name, const char *mother, Int_t ndiv, Int_t iaxis) = 0;

   /// Create a new volume by dividing an existing one.
   /// Divide mother into ndiv divisions called name
   /// along axis iaxis starting at coordinate value c0i.
   /// The new volume created will be medium number numed.
   virtual void Gsdvn2(const char *name, const char *mother, Int_t ndiv, Int_t iaxis, Double_t c0i, Int_t numed) = 0;

   /// Create a new volume by dividing an existing one
   /// Divide mother into divisions called name along
   /// axis iaxis in steps of step. If not exactly divisible
   /// will make as many as possible and will center them
   /// with respect to the mother. Divisions will have medium
   /// number numed. If numed is 0, numed of mother is taken.
   /// ndvmx is the expected maximum number of divisions
   /// (If 0, no protection tests are performed in Geant3)
   virtual void Gsdvt(const char *name, const char *mother, Double_t step, Int_t iaxis, Int_t numed, Int_t ndvmx) = 0;

   /// Create a new volume by dividing an existing one
   /// Divides mother into divisions called name along
   /// axis iaxis starting at coordinate value c0 with step
   /// size step.
   /// The new volume created will have medium number numed.
   /// If numed is 0, numed of mother is taken.
   /// ndvmx is the expected maximum number of divisions
   /// (If 0, no protection tests are performed in Geant3)
   virtual void
   Gsdvt2(const char *name, const char *mother, Double_t step, Int_t iaxis, Double_t c0, Int_t numed, Int_t ndvmx) = 0;

   /// Flag volume name whose contents will have to be ordered
   /// along axis iax, by setting the search flag to -iax
   /// (Geant3 only)
   virtual void Gsord(const char *name, Int_t iax) = 0;

   /// Position a volume into an existing one.
   /// It positions a previously defined volume in the mother.
   /// - name   Volume name
   /// - nr     Copy number of the volume
   /// - mother Mother volume name
   /// - x      X coord. of the volume in mother ref. sys.
   /// - y      Y coord. of the volume in mother ref. sys.
   /// - z      Z coord. of the volume in mother ref. sys.
   /// - irot   Rotation matrix number w.r.t. mother ref. sys.
   /// - konly  ONLY/MANY flag
   virtual void Gspos(const char *name, Int_t nr, const char *mother, Double_t x, Double_t y, Double_t z, Int_t irot,
                      const char *konly = "ONLY") = 0;

   /// Place a copy of generic volume name with user number
   ///  nr inside mother, with its parameters upar(1..np)
   virtual void Gsposp(const char *name, Int_t nr, const char *mother, Double_t x, Double_t y, Double_t z, Int_t irot,
                       const char *konly, Float_t *upar, Int_t np) = 0;

   /// The same as previous but in double precision
   virtual void Gsposp(const char *name, Int_t nr, const char *mother, Double_t x, Double_t y, Double_t z, Int_t irot,
                       const char *konly, Double_t *upar, Int_t np) = 0;

   /// Helper function for resolving MANY.
   /// Specify the ONLY volume that overlaps with the
   /// specified MANY and has to be substracted.
   /// (Geant4 only)
   virtual void Gsbool(const char *onlyVolName, const char *manyVolName) = 0;

   /// Define the tables for UV photon tracking in medium itmed.
   /// Please note that it is the user's responsibility to
   /// provide all the coefficients:
   /// - itmed       Tracking medium number
   /// - npckov      Number of bins of each table
   /// - ppckov      Value of photon momentum (in GeV)
   /// - absco       Absorption coefficients
   ///                     - dielectric: absorption length in cm
   ///                     - metals    : absorption fraction (0<=x<=1)
   /// - effic       Detection efficiency for UV photons
   /// - rindex      Refraction index (if=0 metal)
   virtual void
   SetCerenkov(Int_t itmed, Int_t npckov, Float_t *ppckov, Float_t *absco, Float_t *effic, Float_t *rindex) = 0;

   /// The same as previous but in double precision
   virtual void
   SetCerenkov(Int_t itmed, Int_t npckov, Double_t *ppckov, Double_t *absco, Double_t *effic, Double_t *rindex) = 0;

   //
   // functions for definition of surfaces
   // and material properties for optical physics
   // ------------------------------------------------
   //

   /// Define the optical surface
   /// - name           surface name
   /// - model          selection of model (see #EMCOpSurfaceModel values)
   /// - surfaceType    surface type (see #EMCOpSurfaceType values)
   /// - surfaceFinish  surface quality (see #EMCOpSurfaceType values)
   /// - sigmaAlpha     an unified model surface parameter
   /// (Geant4 only)
   virtual void DefineOpSurface(const char *name, EMCOpSurfaceModel model, EMCOpSurfaceType surfaceType,
                                EMCOpSurfaceFinish surfaceFinish, Double_t sigmaAlpha) = 0;

   /// Define the optical surface border
   /// - name        border surface name
   /// - vol1Name    first volume name
   /// - vol1CopyNo  first volume copy number
   /// - vol2Name    second volume name
   /// - vol2CopyNo  second volume copy number
   /// - opSurfaceName  name of optical surface which this border belongs to
   /// (Geant4 only)
   virtual void SetBorderSurface(const char *name, const char *vol1Name, int vol1CopyNo, const char *vol2Name,
                                 int vol2CopyNo, const char *opSurfaceName) = 0;

   /// Define the optical skin surface
   /// - name        skin surface name
   /// - volName     volume name
   /// - opSurfaceName  name of optical surface which this border belongs to
   /// (Geant4 only)
   virtual void SetSkinSurface(const char *name, const char *volName, const char *opSurfaceName) = 0;

   /// Define material property via a table of values
   /// - itmed         tracking medium id
   /// - propertyName  property name
   /// - np            number of bins of the table
   /// - pp            value of photon momentum (in GeV)
   /// - values        property values
   /// (Geant4 only)
   virtual void
   SetMaterialProperty(Int_t itmed, const char *propertyName, Int_t np, Double_t *pp, Double_t *values) = 0;

   /// Define material property via a value
   /// - itmed         tracking medium id
   /// - propertyName  property name
   /// - value         property value
   /// (Geant4 only)
   virtual void SetMaterialProperty(Int_t itmed, const char *propertyName, Double_t value) = 0;

   /// Define optical surface property via a table of values
   /// - surfaceName   optical surface name
   /// - propertyName  property name
   /// - np            number of bins of the table
   /// - pp            value of photon momentum (in GeV)
   /// - values        property values
   /// (Geant4 only)
   virtual void
   SetMaterialProperty(const char *surfaceName, const char *propertyName, Int_t np, Double_t *pp, Double_t *values) = 0;

   //
   // functions for access to geometry
   // ------------------------------------------------
   //

   /// Return the transformation matrix between the volume specified by
   /// the path volumePath and the top or master volume.
   virtual Bool_t GetTransformation(const TString &volumePath, TGeoHMatrix &matrix) = 0;

   /// Return the name of the shape (shapeType)  and its parameters par
   /// for the volume specified by the path volumePath .
   virtual Bool_t GetShape(const TString &volumePath, TString &shapeType, TArrayD &par) = 0;

   /// Return the material parameters for the material specified by
   /// the material Id
   virtual Bool_t GetMaterial(Int_t imat, TString &name, Double_t &a, Double_t &z, Double_t &density, Double_t &radl,
                              Double_t &inter, TArrayD &par) = 0;

   /// Return the material parameters for the volume specified by
   /// the volumeName.
   virtual Bool_t GetMaterial(const TString &volumeName, TString &name, Int_t &imat, Double_t &a, Double_t &z,
                              Double_t &density, Double_t &radl, Double_t &inter, TArrayD &par) = 0;

   /// Return the medium parameters for the volume specified by the
   /// volumeName.
   virtual Bool_t GetMedium(const TString &volumeName, TString &name, Int_t &imed, Int_t &nmat, Int_t &isvol,
                            Int_t &ifield, Double_t &fieldm, Double_t &tmaxfd, Double_t &stemax, Double_t &deemax,
                            Double_t &epsil, Double_t &stmin, TArrayD &par) = 0;

   /// Write out the geometry of the detector in EUCLID file format
   /// - filnam  file name - will be with the extension .euc                 *
   /// - topvol  volume name of the starting node
   /// - number  copy number of topvol (relevant for gsposp)
   /// - nlevel  number of  levels in the tree structure
   ///                to be written out, starting from topvol
   /// (Geant3 only)
   /// Deprecated
   virtual void WriteEuclid(const char *filnam, const char *topvol, Int_t number, Int_t nlevel) = 0;

   /// Set geometry from Root (built via TGeo)
   virtual void SetRootGeometry() = 0;

   /// Activate the parameters defined in tracking media
   /// (DEEMAX, STMIN, STEMAX), which are, be default, ignored.
   /// In Geant4 case, only STEMAX is taken into account.
   /// In FLUKA, all tracking media parameters are ignored.
   virtual void SetUserParameters(Bool_t isUserParameters) = 0;

   //
   // get methods
   // ------------------------------------------------
   //

   /// Return the unique numeric identifier for volume name volName
   virtual Int_t VolId(const char *volName) const = 0;

   /// Return the volume name for a given volume identifier id
   virtual const char *VolName(Int_t id) const = 0;

   /// Return the unique numeric identifier for medium name mediumName
   virtual Int_t MediumId(const char *mediumName) const = 0;

   /// Return total number of volumes in the geometry
   virtual Int_t NofVolumes() const = 0;

   /// Return material number for a given volume id
   virtual Int_t VolId2Mate(Int_t id) const = 0;

   /// Return number of daughters of the volume specified by volName
   virtual Int_t NofVolDaughters(const char *volName) const = 0;

   /// Return the name of i-th daughter of the volume specified by volName
   virtual const char *VolDaughterName(const char *volName, Int_t i) const = 0;

   /// Return the copyNo of i-th daughter of the volume specified by volName
   virtual Int_t VolDaughterCopyNo(const char *volName, Int_t i) const = 0;

   //
   // ------------------------------------------------
   // methods for sensitive detectors
   // ------------------------------------------------
   //

   /// Set a sensitive detector to a volume
   /// - volName - the volume name
   /// - sd - the user sensitive detector
   virtual void SetSensitiveDetector(const TString &volName, TVirtualMCSensitiveDetector *sd);

   /// Get a sensitive detector of a volume
   /// - volName - the volume name
   virtual TVirtualMCSensitiveDetector *GetSensitiveDetector(const TString &volName) const;

   /// The scoring option:
   /// if true, scoring is performed only via user defined sensitive detectors and
   /// MCApplication::Stepping is not called
   virtual void SetExclusiveSDScoring(Bool_t exclusiveSDScoring);

   //
   // ------------------------------------------------
   // methods for physics management
   // ------------------------------------------------
   //

   //
   // set methods
   // ------------------------------------------------
   //

   /// Set transport cuts for particles
   virtual Bool_t SetCut(const char *cutName, Double_t cutValue) = 0;

   /// Set process control
   virtual Bool_t SetProcess(const char *flagName, Int_t flagValue) = 0;

   /// Set a user defined particle
   /// Function is ignored if particle with specified pdg
   /// already exists and error report is printed.
   /// - pdg           PDG encoding
   /// - name          particle name
   /// - mcType        VMC Particle type
   /// - mass          mass [GeV]
   /// - charge        charge [eplus]
   /// - lifetime      time of life [s]
   /// - pType         particle type as in Geant4
   /// - width         width [GeV]
   /// - iSpin         spin
   /// - iParity       parity
   /// - iConjugation  conjugation
   /// - iIsospin      isospin
   /// - iIsospinZ     isospin - #rd component
   /// - gParity       gParity
   /// - lepton        lepton number
   /// - baryon        baryon number
   /// - stable        stability
   /// - shortlived    is shorlived?
   /// - subType       particle subType as in Geant4
   /// - antiEncoding  anti encoding
   /// - magMoment     magnetic moment
   /// - excitation    excitation energy [GeV]
   virtual Bool_t DefineParticle(Int_t pdg, const char *name, TMCParticleType mcType, Double_t mass, Double_t charge,
                                 Double_t lifetime) = 0;

   /// Set a user defined particle
   /// Function is ignored if particle with specified pdg
   /// already exists and error report is printed.
   /// - pdg           PDG encoding
   /// - name          particle name
   /// - mcType        VMC Particle type
   /// - mass          mass [GeV]
   /// - charge        charge [eplus]
   /// - lifetime      time of life [s]
   /// - pType         particle type as in Geant4
   /// - width         width [GeV]
   /// - iSpin         spin
   /// - iParity       parity
   /// - iConjugation  conjugation
   /// - iIsospin      isospin
   /// - iIsospinZ     isospin - #rd component
   /// - gParity       gParity
   /// - lepton        lepton number
   /// - baryon        baryon number
   /// - stable        stability
   /// - shortlived    is shorlived?
   /// - subType       particle subType as in Geant4
   /// - antiEncoding  anti encoding
   /// - magMoment     magnetic moment
   /// - excitation    excitation energy [GeV]
   virtual Bool_t DefineParticle(Int_t pdg, const char *name, TMCParticleType mcType, Double_t mass, Double_t charge,
                                 Double_t lifetime, const TString &pType, Double_t width, Int_t iSpin, Int_t iParity,
                                 Int_t iConjugation, Int_t iIsospin, Int_t iIsospinZ, Int_t gParity, Int_t lepton,
                                 Int_t baryon, Bool_t stable, Bool_t shortlived = kFALSE, const TString &subType = "",
                                 Int_t antiEncoding = 0, Double_t magMoment = 0.0, Double_t excitation = 0.0) = 0;

   /// Set a user defined ion.
   /// - name          ion name
   /// - Z             atomic number
   /// - A             atomic mass
   /// - Q             charge [eplus}
   /// - excitation    excitation energy [GeV]
   /// - mass          mass  [GeV] (if not specified by user, approximative
   ///                 mass is calculated)
   virtual Bool_t DefineIon(const char *name, Int_t Z, Int_t A, Int_t Q, Double_t excEnergy, Double_t mass = 0.) = 0;

   /// Set a user phase space decay for a particle
   /// -  pdg           particle PDG encoding
   /// -  bratios       the array with branching ratios (in %)
   /// -  mode[6][3]    the array with daughters particles PDG codes  for each
   ///                 decay channel
   virtual Bool_t SetDecayMode(Int_t pdg, Float_t bratio[6], Int_t mode[6][3]) = 0;

   /// Calculate X-sections
   /// (Geant3 only)
   /// Deprecated
   virtual Double_t Xsec(char *, Double_t, Int_t, Int_t) = 0;

   //
   // particle table usage
   // ------------------------------------------------
   //

   /// Return MC specific code from a PDG and pseudo ENDF code (pdg)
   virtual Int_t IdFromPDG(Int_t pdg) const = 0;

   /// Return PDG code and pseudo ENDF code from MC specific code (id)
   virtual Int_t PDGFromId(Int_t id) const = 0;

   //
   // get methods
   // ------------------------------------------------
   //

   /// Return name of the particle specified by pdg.
   virtual TString ParticleName(Int_t pdg) const = 0;

   /// Return mass of the particle specified by pdg.
   virtual Double_t ParticleMass(Int_t pdg) const = 0;

   /// Return charge (in e units) of the particle specified by pdg.
   virtual Double_t ParticleCharge(Int_t pdg) const = 0;

   /// Return life time of the particle specified by pdg.
   virtual Double_t ParticleLifeTime(Int_t pdg) const = 0;

   /// Return VMC type of the particle specified by pdg.
   virtual TMCParticleType ParticleMCType(Int_t pdg) const = 0;
   //
   // ------------------------------------------------
   // methods for step management
   // ------------------------------------------------
   //

   //
   // action methods
   // ------------------------------------------------
   //

   /// Stop the transport of the current particle and skip to the next
   virtual void StopTrack() = 0;

   /// Stop simulation of the current event and skip to the next
   virtual void StopEvent() = 0;

   /// Stop simulation of the current event and set the abort run flag to true
   virtual void StopRun() = 0;

   //
   // set methods
   // ------------------------------------------------
   //

   /// Set the maximum step allowed till the particle is in the current medium
   virtual void SetMaxStep(Double_t) = 0;

   /// Set the maximum number of steps till the particle is in the current medium
   virtual void SetMaxNStep(Int_t) = 0;

   /// Force the decays of particles to be done with Pythia
   /// and not with the Geant routines.
   virtual void SetUserDecay(Int_t pdg) = 0;

   /// Force the decay time of the current particle
   virtual void ForceDecayTime(Float_t) = 0;

   //
   // tracking volume(s)
   // ------------------------------------------------
   //

   /// Return the current volume ID and copy number
   virtual Int_t CurrentVolID(Int_t &copyNo) const = 0;

   /// Return the current volume off upward in the geometrical tree
   /// ID and copy number
   virtual Int_t CurrentVolOffID(Int_t off, Int_t &copyNo) const = 0;

   /// Return the current volume name
   virtual const char *CurrentVolName() const = 0;

   /// Return the current volume off upward in the geometrical tree
   /// name and copy number'
   /// if name=0 no name is returned
   virtual const char *CurrentVolOffName(Int_t off) const = 0;

   /// Return the path in geometry tree for the current volume
   virtual const char *CurrentVolPath() = 0;

   /// If track is on a geometry boundary, fill the normal vector of the crossing
   /// volume surface and return true, return false otherwise
   virtual Bool_t CurrentBoundaryNormal(Double_t &x, Double_t &y, Double_t &z) const = 0;

   /// Return the parameters of the current material during transport
   virtual Int_t CurrentMaterial(Float_t &a, Float_t &z, Float_t &dens, Float_t &radl, Float_t &absl) const = 0;

   //// Return the number of the current medium
   virtual Int_t CurrentMedium() const = 0;
   // new function (to replace GetMedium() const)

   /// Return the number of the current event
   virtual Int_t CurrentEvent() const = 0;

   /// Computes coordinates xd in daughter reference system
   /// from known coordinates xm in mother reference system.
   /// - xm    coordinates in mother reference system (input)
   /// - xd    coordinates in daughter reference system (output)
   /// - iflag
   ///   - IFLAG = 1  convert coordinates
   ///   - IFLAG = 2  convert direction cosines
   virtual void Gmtod(Float_t *xm, Float_t *xd, Int_t iflag) = 0;

   /// The same as previous but in double precision
   virtual void Gmtod(Double_t *xm, Double_t *xd, Int_t iflag) = 0;

   /// Computes coordinates xm in mother reference system
   /// from known coordinates xd in daughter reference system.
   /// - xd    coordinates in daughter reference system (input)
   /// - xm    coordinates in mother reference system (output)
   /// - iflag
   ///   - IFLAG = 1  convert coordinates
   ///   - IFLAG = 2  convert direction cosines
   virtual void Gdtom(Float_t *xd, Float_t *xm, Int_t iflag) = 0;

   /// The same as previous but in double precision
   virtual void Gdtom(Double_t *xd, Double_t *xm, Int_t iflag) = 0;

   /// Return the maximum step length in the current medium
   virtual Double_t MaxStep() const = 0;

   /// Return the maximum number of steps allowed in the current medium
   virtual Int_t GetMaxNStep() const = 0;

   //
   // get methods
   // tracking particle
   // dynamic properties
   // ------------------------------------------------
   //

   /// Return the current position in the master reference frame of the
   /// track being transported
   virtual void TrackPosition(TLorentzVector &position) const = 0;

   /// Only return spatial coordinates (as double)
   virtual void TrackPosition(Double_t &x, Double_t &y, Double_t &z) const = 0;

   /// Only return spatial coordinates (as float)
   virtual void TrackPosition(Float_t &x, Float_t &y, Float_t &z) const = 0;

   /// Return the direction and the momentum (GeV/c) of the track
   /// currently being transported
   virtual void TrackMomentum(TLorentzVector &momentum) const = 0;

   /// Return the direction and the momentum (GeV/c) of the track
   /// currently being transported (as double)
   virtual void TrackMomentum(Double_t &px, Double_t &py, Double_t &pz, Double_t &etot) const = 0;

   /// Return the direction and the momentum (GeV/c) of the track
   /// currently being transported (as float)
   virtual void TrackMomentum(Float_t &px, Float_t &py, Float_t &pz, Float_t &etot) const = 0;

   /// Return the length in centimeters of the current step (in cm)
   virtual Double_t TrackStep() const = 0;

   /// Return the length of the current track from its origin (in cm)
   virtual Double_t TrackLength() const = 0;

   /// Return the current time of flight of the track being transported
   virtual Double_t TrackTime() const = 0;

   /// Return the energy lost in the current step
   virtual Double_t Edep() const = 0;

   /// Return the non-ionising energy lost (NIEL) in the current step
   virtual Double_t NIELEdep() const;

   /// Return the current step number
   virtual Int_t StepNumber() const;

   /// Get the current weight
   virtual Double_t TrackWeight() const;

   /// Get the current polarization
   virtual void TrackPolarization(Double_t &polX, Double_t &polY, Double_t &polZ) const;

   /// Get the current polarization
   virtual void TrackPolarization(TVector3 &pol) const;

   //
   // get methods
   // tracking particle
   // static properties
   // ------------------------------------------------
   //

   /// Return the PDG of the particle transported
   virtual Int_t TrackPid() const = 0;

   /// Return the charge of the track currently transported
   virtual Double_t TrackCharge() const = 0;

   /// Return the mass of the track currently transported
   virtual Double_t TrackMass() const = 0;

   /// Return the total energy of the current track
   virtual Double_t Etot() const = 0;

   //
   // get methods - track status
   // ------------------------------------------------
   //

   /// Return true when the track performs the first step
   virtual Bool_t IsNewTrack() const = 0;

   /// Return true if the track is not at the boundary of the current volume
   virtual Bool_t IsTrackInside() const = 0;

   /// Return true if this is the first step of the track in the current volume
   virtual Bool_t IsTrackEntering() const = 0;

   /// Return true if this is the last step of the track in the current volume
   virtual Bool_t IsTrackExiting() const = 0;

   /// Return true if the track is out of the setup
   virtual Bool_t IsTrackOut() const = 0;

   /// Return true if the current particle has disappeared
   /// either because it decayed or because it underwent
   /// an inelastic collision
   virtual Bool_t IsTrackDisappeared() const = 0;

   /// Return true if the track energy has fallen below the threshold
   virtual Bool_t IsTrackStop() const = 0;

   /// Return true if the current particle is alive and will continue to be
   /// transported
   virtual Bool_t IsTrackAlive() const = 0;

   //
   // get methods - secondaries
   // ------------------------------------------------
   //

   /// Return the number of secondary particles generated in the current step
   virtual Int_t NSecondaries() const = 0;

   /// Return the parameters of the secondary track number isec produced
   /// in the current step
   virtual void GetSecondary(Int_t isec, Int_t &particleId, TLorentzVector &position, TLorentzVector &momentum) = 0;

   /// Return the VMC code of the process that has produced the secondary
   /// particles in the current step
   virtual TMCProcess ProdProcess(Int_t isec) const = 0;

   /// Return the array of the VMC code of the processes active in the current
   /// step
   virtual Int_t StepProcesses(TArrayI &proc) const = 0;

   /// Return the information about the transport order needed by the stack
   virtual Bool_t SecondariesAreOrdered() const = 0;

   //
   // ------------------------------------------------
   // Control methods
   // ------------------------------------------------
   //

   /// Initialize MC
   virtual void Init() = 0;

   /// Initialize MC physics
   virtual void BuildPhysics() = 0;

   /// Process one event
   virtual void ProcessEvent(Int_t eventId);

   /// Process one event (backward-compatibility)
   virtual void ProcessEvent();

   /// Process one  run and return true if run has finished successfully,
   /// return false in other cases (run aborted by user)
   virtual Bool_t ProcessRun(Int_t nevent) = 0;

   /// Additional cleanup after a run can be done here (optional)
   virtual void TerminateRun() {}

   /// Set switches for lego transport
   virtual void InitLego() = 0;

   /// (In)Activate collecting TGeo tracks
   virtual void SetCollectTracks(Bool_t collectTracks) = 0;

   /// Return the info if collecting tracks is activated
   virtual Bool_t IsCollectTracks() const = 0;

   /// Return the info if multi-threading is supported/activated
   virtual Bool_t IsMT() const { return kFALSE; }

   //
   // ------------------------------------------------
   // Set methods
   // ------------------------------------------------
   //

   /// Set the particle stack
   virtual void SetStack(TVirtualMCStack *stack);

   /// Set the external decayer
   virtual void SetExternalDecayer(TVirtualMCDecayer *decayer);

   /// Set the random number generator
   virtual void SetRandom(TRandom *random);

   /// Set the magnetic field
   virtual void SetMagField(TVirtualMagField *field);

   //
   // ------------------------------------------------
   // Get methods
   // ------------------------------------------------
   //

   /// Return the particle stack
   TVirtualMCStack *GetStack() const { return fStack; }

   /// Return the particle stack managed by the TMCManager (if any)
   TMCManagerStack *GetManagerStack() const { return fManagerStack; }

   /// Return the external decayer
   TVirtualMCDecayer *GetDecayer() const { return fDecayer; }

   /// Return the random number generator
   TRandom *GetRandom() const { return fRandom; }

   /// Return the magnetic field
   TVirtualMagField *GetMagField() const { return fMagField; }

   /// Return the VMC's ID
   Int_t GetId() const { return fId; }

   /// Check whether external geometry construction should be used
   Bool_t UseExternalGeometryConstruction() const { return fUseExternalGeometryConstruction; }

   /// Check whether external particle generation should be used
   Bool_t UseExternalParticleGeneration() const { return fUseExternalParticleGeneration; }

private:
   /// Set the VMC id
   void SetId(UInt_t id);

   /// Set container holding additional information for transported TParticles
   void SetManagerStack(TMCManagerStack *stack);

   /// Disables internal dispatch to TVirtualMCApplication::ConstructGeometry()
   /// and hence rely on geometry construction being trigeered from outside.
   void SetExternalGeometryConstruction(Bool_t value = kTRUE);

   /// Disables internal dispatch to TVirtualMCApplication::GeneratePrimaries()
   /// and tells the engine to not make any implicit assumptions on whether it's
   /// a primary or a secondary. The track could have even been transported by
   /// another engine to the current point.
   void SetExternalParticleGeneration(Bool_t value = kTRUE);

   /// An interruptible event can be paused and resumed at any time. It must not
   /// call TVirtualMCApplication::BeginEvent() and ::FinishEvent()
   /// Further, when tracks are popped from the TVirtualMCStack it must be
   /// checked whether these are new tracks or whether they have been
   /// transported up to their current point.
   virtual void ProcessEvent(Int_t eventId, Bool_t isInterruptible);

   /// That triggers stopping the transport of the current track without dispatching
   /// to common routines like TVirtualMCApplication::PostTrack() etc.
   virtual void InterruptTrack();

   // Private, no copying.
   TVirtualMC(const TVirtualMC &mc);
   TVirtualMC &operator=(const TVirtualMC &);

protected:
   TVirtualMCApplication *fApplication; //!< User MC application

private:
#if !defined(__CINT__)
   static TMCThreadLocal TVirtualMC *fgMC; ///< Static TVirtualMC pointer
#else
   static TVirtualMC *fgMC; ///< Static TVirtualMC pointer
#endif

private:
   Int_t fId;                               //!< Unique identification of this VMC
                                            // (don't use TObject::SetUniqueId since this
                                            // is used to uniquely identify TObjects in
                                            // in general)
                                            // An ID is given by the running TVirtualMCApp
                                            // and not by the user.
   TVirtualMCStack *fStack;                 //!< Particles stack
   TMCManagerStack *fManagerStack;          //!< Stack handled by the TMCManager
   TVirtualMCDecayer *fDecayer;             //!< External decayer
   TRandom *fRandom;                        //!< Random number generator
   TVirtualMagField *fMagField;             //!< Magnetic field
   Bool_t fUseExternalGeometryConstruction; //!< Don't attempt to
                                            // call
                                            // TVirtualMCApplication
                                            // hooks related to geometry
   // construction
   Bool_t fUseExternalParticleGeneration;

   ClassDef(TVirtualMC, 1) // Interface to Monte Carlo
};

// inline functions (with temorary implementation)

inline void TVirtualMC::SetSensitiveDetector(const TString & /*volName*/, TVirtualMCSensitiveDetector * /*sd*/)
{
   /// Set a sensitive detector to a volume
   /// - volName - the volume name
   /// - sd - the user sensitive detector

   Warning("SetSensitiveDetector(...)", "New function - not yet implemented.");
}

inline TVirtualMCSensitiveDetector *TVirtualMC::GetSensitiveDetector(const TString & /*volName*/) const
{
   /// Get a sensitive detector of a volume
   /// - volName - the volume name

   Warning("GetSensitiveDetector()", "New function - not yet implemented.");

   return 0;
}

inline void TVirtualMC::SetExclusiveSDScoring(Bool_t /*exclusiveSDScoring*/)
{
   /// The scoring option:
   /// if true, scoring is performed only via user defined sensitive detectors and
   /// MCApplication::Stepping is not called

   Warning("SetExclusiveSDScoring(...)", "New function - not yet implemented.");
}

inline Double_t TVirtualMC::NIELEdep() const
{
   /// Return the non-ionising energy lost (NIEL) in the current step

   Warning("NIELEdep()", "New function - not yet implemented.");

   return 0.;
}

#define gMC (TVirtualMC::GetMC())

#endif // ROOT_TVirtualMC
