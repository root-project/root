// @(#)root/vmc:$Name:  $:$Id: TVirtualMC.h,v 1.15 2005/11/18 21:20:15 brun Exp $
// Authors: Ivana Hrivnacova, Rene Brun, Federico Carminati 13/04/2002

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
#include "TVirtualMCApplication.h"
#include "TVirtualMCStack.h"
#include "TVirtualMCDecayer.h"
#include "TRandom.h"
#include "TString.h"
#include "TError.h"

class TLorentzVector;
class TGeoHMatrix;
class TArrayI;
class TArrayD;

class TVirtualMC : public TNamed {

  public:
    TVirtualMC(const char *name, const char *title,
               Bool_t isRootGeometrySupported = kFALSE);
    TVirtualMC();
    virtual ~TVirtualMC();
    TVirtualMC(const TVirtualMC &mc) : TNamed(mc) {}

    // static access method
    static TVirtualMC* GetMC() { return fgMC; }

    //
    // methods for building/management of geometry
    // ------------------------------------------------
    //

    // info about supporting geometry defined via Root
    virtual Bool_t IsRootGeometrySupported() const { return kFALSE; }
                      // make this function =0 with next release

    // functions from GCONS
    virtual void  Gfmate(Int_t imat, char *name, Float_t &a, Float_t &z,
                           Float_t &dens, Float_t &radl, Float_t &absl,
                         Float_t* ubuf, Int_t& nbuf) = 0;
    virtual void  Gfmate(Int_t imat, char *name, Double_t &a, Double_t &z,
                           Double_t &dens, Double_t &radl, Double_t &absl,
                         Double_t* ubuf, Int_t& nbuf) = 0;
    virtual void  Gckmat(Int_t imed, char* name) = 0;

    // detector composition
    virtual void  Material(Int_t& kmat, const char* name, Double_t a,
                     Double_t z, Double_t dens, Double_t radl, Double_t absl,
                     Float_t* buf, Int_t nwbuf) = 0;
    virtual void  Material(Int_t& kmat, const char* name, Double_t a,
                     Double_t z, Double_t dens, Double_t radl, Double_t absl,
                     Double_t* buf, Int_t nwbuf) = 0;
    virtual void  Mixture(Int_t& kmat, const char *name, Float_t *a,
                     Float_t *z, Double_t dens, Int_t nlmat, Float_t *wmat) = 0;
    virtual void  Mixture(Int_t& kmat, const char *name, Double_t *a,
                     Double_t *z, Double_t dens, Int_t nlmat, Double_t *wmat) = 0;
    virtual void  Medium(Int_t& kmed, const char *name, Int_t nmat,
                     Int_t isvol, Int_t ifield, Double_t fieldm, Double_t tmaxfd,
                     Double_t stemax, Double_t deemax, Double_t epsil,
                     Double_t stmin, Float_t* ubuf, Int_t nbuf) = 0;
    virtual void  Medium(Int_t& kmed, const char *name, Int_t nmat,
                     Int_t isvol, Int_t ifield, Double_t fieldm, Double_t tmaxfd,
                     Double_t stemax, Double_t deemax, Double_t epsil,
                     Double_t stmin, Double_t* ubuf, Int_t nbuf) = 0;
    virtual void  Matrix(Int_t& krot, Double_t thetaX, Double_t phiX,
                     Double_t thetaY, Double_t phiY, Double_t thetaZ,
                     Double_t phiZ) = 0;
    virtual void  Gstpar(Int_t itmed, const char *param, Double_t parval) = 0;

    // functions from GGEOM
    virtual Int_t  Gsvolu(const char *name, const char *shape, Int_t nmed,
                          Float_t *upar, Int_t np) = 0;
    virtual Int_t  Gsvolu(const char *name, const char *shape, Int_t nmed,
                          Double_t *upar, Int_t np) = 0;
    virtual void  Gsdvn(const char *name, const char *mother, Int_t ndiv,
                         Int_t iaxis) = 0;
    virtual void  Gsdvn2(const char *name, const char *mother, Int_t ndiv,
                         Int_t iaxis, Double_t c0i, Int_t numed) = 0;
    virtual void  Gsdvt(const char *name, const char *mother, Double_t step,
                         Int_t iaxis, Int_t numed, Int_t ndvmx) = 0;
    virtual void  Gsdvt2(const char *name, const char *mother, Double_t step,
                         Int_t iaxis, Double_t c0, Int_t numed, Int_t ndvmx) = 0;
    virtual void  Gsord(const char *name, Int_t iax) = 0;
    virtual void  Gspos(const char *name, Int_t nr, const char *mother,
                         Double_t x, Double_t y, Double_t z, Int_t irot,
                         const char *konly="ONLY") = 0;
    virtual void  Gsposp(const char *name, Int_t nr, const char *mother,
                         Double_t x, Double_t y, Double_t z, Int_t irot,
                         const char *konly, Float_t *upar, Int_t np) = 0;
    virtual void  Gsposp(const char *name, Int_t nr, const char *mother,
                         Double_t x, Double_t y, Double_t z, Int_t irot,
                         const char *konly, Double_t *upar, Int_t np) = 0;
    virtual void  Gsbool(const char* onlyVolName, const char* manyVolName) = 0;

    virtual void  SetCerenkov(Int_t itmed, Int_t npckov, Float_t *ppckov,
                               Float_t *absco, Float_t *effic, Float_t *rindex) = 0;
    virtual void  SetCerenkov(Int_t itmed, Int_t npckov, Double_t *ppckov,
                               Double_t *absco, Double_t *effic, Double_t *rindex) = 0;

    // functions for definition of surfaces
    // and material properties for optical physics
    virtual void  DefineOpSurface(const char* name,
                         EMCOpSurfaceModel model,
                         EMCOpSurfaceType surfaceType,
                         EMCOpSurfaceFinish surfaceFinish,
                         Double_t sigmaAlpha);
    virtual void  SetBorderSurface(const char* name,
                         const char* vol1Name, int vol1CopyNo,
                         const char* vol2Name, int vol2CopyNo,
                         const char* opSurfaceName);
    virtual void  SetSkinSurface(const char* name,
                         const char* volName,
                         const char* opSurfaceName);
    virtual void  SetMaterialProperty(
                         Int_t itmed, const char* propertyName,
                         Int_t np, Double_t* pp, Double_t* values);
    virtual void  SetMaterialProperty(
                         Int_t itmed, const char* propertyName,
                         Double_t value);
    virtual void  SetMaterialProperty(
                         const char* surfaceName, const char* propertyName,
                         Int_t np, Double_t* pp, Double_t* values);

    // functions for access to geometry
    //
    // Return the Transformation matrix between the volume specified by
    // the path volumePath and the top or master volume.
    virtual Bool_t GetTransformation(const TString& volumePath,
                         TGeoHMatrix& matrix);

    // Return the name of the shape and its parameters for the volume
    // specified by the volume name.
    virtual Bool_t GetShape(const TString& volumePath,
                         TString& shapeType, TArrayD& par);

    // Returns the material parameters for the volume specified by
    // the volume name.
    virtual Bool_t GetMaterial(const TString& volumeName,
                               TString& name, Int_t& imat,
                               Double_t& a, Double_t& z, Double_t& density,
                               Double_t& radl, Double_t& inter, TArrayD& par);

    // Returns the medium parameters for the volume specified by the
    // volume name.
    virtual Bool_t GetMedium(const TString& volumeName,
                             TString& name, Int_t& imed,
                             Int_t& nmat, Int_t& isvol, Int_t& ifield,
                             Double_t& fieldm, Double_t& tmaxfd, Double_t& stemax,
                             Double_t& deemax, Double_t& epsil, Double_t& stmin,
                             TArrayD& par);

    // functions for drawing
    // to be removed with complete move to TGeo
    virtual void  DrawOneSpec(const char* name) = 0;
    virtual void  Gsatt(const char* name, const char* att, Int_t val) = 0;
    virtual void  Gdraw(const char*,Double_t theta = 30, Double_t phi = 30,
                        Double_t psi = 0, Double_t u0 = 10, Double_t v0 = 10,
                        Double_t ul = 0.01, Double_t vl = 0.01) = 0;

    // Euclid
    virtual void  WriteEuclid(const char*, const char*, Int_t, Int_t) = 0;

    // set geometry from Root (built via TGeo)
    virtual void  SetRootGeometry() = 0;

    // get methods
    virtual Int_t VolId(const Text_t* volName) const = 0;
    virtual const char* VolName(Int_t id) const = 0;
    virtual Int_t NofVolumes() const = 0;
    virtual Int_t VolId2Mate(Int_t id) const = 0;
    virtual Int_t NofVolDaughters(const char* volName) const = 0;
    virtual const char*  VolDaughterName(const char* volName, Int_t i) const = 0;
    virtual Int_t        VolDaughterCopyNo(const char* volName, Int_t i) const = 0;

    //
    // methods for physics management
    // ------------------------------------------------
    //

    // set methods
    virtual Bool_t   SetCut(const char* cutName, Double_t cutValue) = 0;
    virtual Bool_t   SetProcess(const char* flagName, Int_t flagValue) = 0;
    virtual Bool_t   DefineParticle(Int_t pdg, const char* name,
                        TMCParticleType pType,
                        Double_t mass, Double_t charge, Double_t lifetime) = 0;
    virtual Bool_t   DefineIon(const char* name, Int_t Z, Int_t A,
                        Int_t Q, Double_t excEnergy, Double_t mass = 0.) = 0;
    virtual Double_t Xsec(char*, Double_t, Int_t, Int_t) = 0;

        // particle table usage
    virtual Int_t   IdFromPDG(Int_t id) const =0;
    virtual Int_t   PDGFromId(Int_t pdg) const =0;

        // get methods
    virtual TString   ParticleName(Int_t pdg) const = 0;
    virtual Double_t  ParticleMass(Int_t pdg) const = 0;
    virtual Double_t  ParticleCharge(Int_t pdg) const = 0;
    virtual Double_t  ParticleLifeTime(Int_t pdg) const = 0;
    virtual TMCParticleType ParticleMCType(Int_t pdg) const = 0;

    //
    // methods for step management
    // ------------------------------------------------
    //

    // action methods
    virtual void StopTrack() = 0;
    virtual void StopEvent() = 0;
    virtual void StopRun() = 0;

    // set methods
    virtual void SetMaxStep(Double_t) = 0;
    virtual void SetMaxNStep(Int_t) = 0;
    virtual void SetUserDecay(Int_t) = 0;
    virtual void ForceDecayTime(Float_t) = 0;

    // get methods
    // tracking volume(s)
    virtual Int_t    CurrentVolID(Int_t& copyNo) const =0;
    virtual Int_t    CurrentVolOffID(Int_t off, Int_t& copyNo) const =0;
    virtual const char* CurrentVolName() const =0;
    virtual const char* CurrentVolOffName(Int_t off) const =0;
    virtual const char* CurrentVolPath() = 0;
    virtual Int_t    CurrentMaterial(Float_t &a, Float_t &z,
                       Float_t &dens, Float_t &radl, Float_t &absl) const =0;
    virtual Int_t    CurrentMedium() const;
                         // new function (to replace GetMedium() const)
    virtual Int_t    CurrentEvent() const =0;
    virtual void     Gmtod(Float_t* xm, Float_t* xd, Int_t iflag) = 0;
    virtual void     Gmtod(Double_t* xm, Double_t* xd, Int_t iflag) = 0;
    virtual void     Gdtom(Float_t* xd, Float_t* xm, Int_t iflag)= 0 ;
    virtual void     Gdtom(Double_t* xd, Double_t* xm, Int_t iflag)= 0 ;
    virtual Double_t MaxStep() const =0;
    virtual Int_t    GetMaxNStep() const = 0;
    virtual Int_t    GetMedium() const = 0;
                         // Replaced with CurrentMedium(), to be removed

        // tracking particle
        // dynamic properties
    virtual void     TrackPosition(TLorentzVector& position) const =0;
    virtual void     TrackPosition(Double_t &x, Double_t &y, Double_t &z) const =0;
    virtual void     TrackMomentum(TLorentzVector& momentum) const =0;
    virtual void     TrackMomentum(Double_t &px, Double_t &py, Double_t &pz, Double_t &etot) const =0;
    virtual Double_t TrackStep() const =0;
    virtual Double_t TrackLength() const =0;
    virtual Double_t TrackTime() const =0;
    virtual Double_t Edep() const =0;
        // static properties
    virtual Int_t    TrackPid() const =0;
    virtual Double_t TrackCharge() const =0;
    virtual Double_t TrackMass() const =0;
    virtual Double_t Etot() const =0;

        // track status
    virtual Bool_t   IsNewTrack() const =0;
    virtual Bool_t   IsTrackInside() const =0;
    virtual Bool_t   IsTrackEntering() const =0;
    virtual Bool_t   IsTrackExiting() const =0;
    virtual Bool_t   IsTrackOut() const =0;
    virtual Bool_t   IsTrackDisappeared() const =0;
    virtual Bool_t   IsTrackStop() const =0;
    virtual Bool_t   IsTrackAlive() const=0;

        // secondaries
    virtual Int_t    NSecondaries() const=0;
    virtual void     GetSecondary(Int_t isec, Int_t& particleId,
                        TLorentzVector& position, TLorentzVector& momentum) =0;
    virtual TMCProcess ProdProcess(Int_t isec) const =0;
    virtual Int_t    StepProcesses(TArrayI &proc) const = 0;
    // Information about the transport order needed by the stack
    virtual Bool_t   SecondariesAreOrdered() const = 0;

    //
    // Geant3 specific methods
    // !!! to be removed with move to TGeo
    //
    virtual void Gdopt(const char*,const char*) = 0;
    virtual void SetClipBox(const char*,Double_t=-9999,Double_t=0, Double_t=-9999,
                            Double_t=0,Double_t=-9999,Double_t=0) = 0;
    virtual void DefaultRange() = 0;
    virtual void Gdhead(Int_t, const char*, Double_t=0) = 0;
    virtual void Gdman(Double_t, Double_t, const char*) = 0;

    //
    // control methods
    // ------------------------------------------------
    //

    virtual void Init() = 0;
    virtual void BuildPhysics() = 0;
    virtual void ProcessEvent() = 0;
    virtual Bool_t ProcessRun(Int_t nevent) = 0;
    virtual void InitLego() = 0;

    //
    // Set methods
    //
    virtual void SetStack(TVirtualMCStack* stack);
    virtual void SetExternalDecayer(TVirtualMCDecayer* decayer);
    virtual void SetRandom(TRandom* random);

    //
    // Get methods
    //
    virtual TVirtualMCStack*   GetStack() const   { return fStack; }
    virtual TVirtualMCDecayer* GetDecayer() const { return fDecayer; }
    virtual TRandom*           GetRandom() const  { return fRandom; }


  protected:
    TVirtualMCApplication* fApplication; //! User MC application

  private:
    TVirtualMC & operator=(const TVirtualMC &) {return (*this);}

    static TVirtualMC*  fgMC; // Monte Carlo singleton instance

    TVirtualMCStack*    fStack;   //! Particles stack
    TVirtualMCDecayer*  fDecayer; //! External decayer
    TRandom*            fRandom;  //! Random number generator

  ClassDef(TVirtualMC,1)  //Interface to Monte Carlo
};

// new functions

inline void  TVirtualMC::DefineOpSurface(const char* /*name*/,
                EMCOpSurfaceModel /*model*/, EMCOpSurfaceType /*surfaceType*/,
                EMCOpSurfaceFinish /*surfaceFinish*/, Double_t /*sigmaAlpha*/) {

   Warning("DefineOpSurface", "New function - not yet implemented.");
}

inline void  TVirtualMC::SetBorderSurface(const char* /*name*/,
                const char* /*vol1Name*/, int /*vol1CopyNo*/,
                const char* /*vol2Name*/, int /*vol2CopyNo*/,
                const char* /*opSurfaceName*/) {
   Warning("SetBorderSurface", "New function - not yet implemented.");
}

inline void  TVirtualMC::SetSkinSurface(const char* /*name*/,
                const char* /*volName*/,
                const char* /*opSurfaceName*/) {
   Warning("SetSkinSurface", "New function - not yet implemented.");
}

inline void  TVirtualMC::SetMaterialProperty(
                Int_t /*itmed*/, const char* /*propertyName*/,
                Int_t /*np*/, Double_t* /*pp*/, Double_t* /*values*/) {
   Warning("SetMaterialProperty", "New function - not yet implemented.");
}

inline void  TVirtualMC::SetMaterialProperty(
                Int_t /*itmed*/, const char* /*propertyName*/,
                Double_t /*value*/) {
   Warning("SetMaterialProperty", "New function - not yet implemented.");
}

inline void  TVirtualMC::SetMaterialProperty(
                const char* /*surfaceName*/, const char* /*propertyName*/,
                Int_t /*np*/, Double_t* /*pp*/, Double_t* /*values*/) {
   Warning("SetMaterialProperty", "New function - not yet implemented.");
}

inline Bool_t TVirtualMC::GetTransformation(const TString& /*volumePath*/,
                 TGeoHMatrix& /*matrix*/) {
   Warning("GetTransformation", "New function - not yet implemented.");
   return kFALSE;
}

inline Bool_t TVirtualMC::GetShape(const TString& /*volumeName*/,
                 TString& /*shapeType*/, TArrayD& /*par*/) {
   Warning("GetShape", "New function - not yet implemented.");
   return kFALSE;
}

inline Bool_t TVirtualMC::GetMaterial(const TString& /*volumeName*/,
                 TString& /*name*/, Int_t& /*imat*/,
                 Double_t& /*a*/, Double_t& /*z*/, Double_t& /*density*/,
                 Double_t& /*radl*/, Double_t& /*inter*/, TArrayD& /*par*/) {
   Warning("GetMaterial", "New function - not yet implemented.");
   return kFALSE;
}

inline Bool_t TVirtualMC::GetMedium(const TString& /*volumeName*/,
                 TString& /*name*/, Int_t& /*imed*/,
                 Int_t& /*nmat*/, Int_t& /*isvol*/, Int_t& /*ifield*/,
                 Double_t& /*fieldm*/, Double_t& /*tmaxfd*/, Double_t& /*stemax*/,
                 Double_t& /*deemax*/, Double_t& /*epsil*/, Double_t& /*stmin*/,
                 TArrayD& /*par*/) {
   Warning("GetMedium", "New function - not yet implemented.");
   return kFALSE;
}

inline Int_t TVirtualMC::CurrentMedium() const {
   Warning("CurrentMedium", "New function - not yet implemented.");
   return 0;
}


R__EXTERN TVirtualMC *gMC;

#endif //ROOT_TVirtualMC

