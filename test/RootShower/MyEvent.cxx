// Author: Bertrand Bellenot   22/08/02

/*************************************************************************
 * Copyright (C) 1995-2002, Bertrand Bellenot.                           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see the LICENSE file.                         *
 *************************************************************************/

#include <stdlib.h>

#include <TROOT.h>
#include <TPolyLine3D.h>
#include <TRandom.h>
#include <TParticle.h>
#include <TDecayChannel.h>
#include "MyParticle.h"
#include "TVector3.h"
#include "MyEvent.h"
#include "RootShower.h"

//______________________________________________________________________________
//
// MyEvent class implementation
//______________________________________________________________________________

ClassImp(EventHeader)
ClassImp(MyEvent)

TObjArray *MyEvent::fgTracks = 0;
TClonesArray *MyEvent::fgParticles = 0;

//______________________________________________________________________________
MyEvent::MyEvent()
{
    // Create an Event object.
    // When the constructor is invoked for the first time, the 
    // class static variables fgParticles and fgTracks is 0 and 
    // the TClonesArray fgParticles is created.
    if (!fgTracks) fgTracks = new TObjArray(1);
    fTracks = fgTracks;
    fNtrack = 0;
    if (!fgParticles) fgParticles = new TClonesArray("MyParticle", 100000);
    fgParticles->BypassStreamer(kFALSE);
    fParticles = fgParticles;
    fNparticles = 0;
}

//______________________________________________________________________________
MyEvent::~MyEvent()
{
    // Destructor
    Clear();
}

//______________________________________________________________________________
void MyEvent::Init(Int_t id, Int_t first_particle, Double_t E_0, Double_t B_0,
                   Int_t mat, Double_t dimx, Double_t dimy, Double_t dimz)
{
    // Initialize event ...
    // creates detector and set initial values

    Char_t  strtmp[80];
    Int_t i;
    fId = id;
    fB = B_0;

    // generate array of energies threshold used
    // to give a track color related to the particle
    // energy
    for(i=0;i<10;i++)
        E_thresh[i] = E_0/(2<<i);

    Clear();
    Reset();

    if (!fgTracks) fgTracks = new TObjArray(1);
    fTracks = fgTracks;
    fNtrack = 0;
    if (!fgParticles) fgParticles = new TClonesArray("MyParticle", 100000);
    fParticles = fgParticles;
    fNparticles = 0;

    fTotalParticles = 0;
    fLast = 0;
    fAliveParticles = 1;

    CreateDetector(mat, dimx, dimy, dimz);
    TVector3 location(0.0,fDetector.GetMinY(),0.0);
    TVector3 momentum(0.0,E_0,0.0);

    AddParticle(0,first_particle, location, momentum);
    GetParticle(0)->GenerateTimeOfDecay();

    AddTrack(0, fDetector.GetMinY(), 0, gColIndex);

    gTmpLTI = gEventListTree->AddItem(gBaseLTI,
            GetParticle(0)->GetName());
    sprintf(strtmp,"%1.2e GeV",GetParticle(0)->Energy());
    gEventListTree->SetToolTipItem(gTmpLTI, strtmp);
    gLTI[0] = gTmpLTI;

}

//______________________________________________________________________________
void MyEvent::CreateDetector(Int_t mat, Double_t dimx, Double_t dimy, Double_t dimz)
{
    // create detector with given material and dimensions
    fDetector.Init(mat, dimx, dimy, dimz);
}

//______________________________________________________________________________
void MyEvent::Clear(Option_t *option)
{
    // Clear tracks and particles arrays
    fTracks->Clear(option);
    fParticles->Clear(option);
}

//______________________________________________________________________________
void MyEvent::Reset(Option_t *option)
{
    // Static function to reset all static objects for this event
    delete fgTracks; fgTracks = 0;
    delete fgParticles; fgParticles = 0;
}

//______________________________________________________________________________
void MyEvent::SetHeader(Int_t i, Int_t run, Long_t date, Int_t primary, Double_t energy)
{
    // set event header with event identification and startup parameters
    fNtrack = 0;
    fEvtHdr.Set(i, run, date, primary, energy);
}

//______________________________________________________________________________
TPolyLine3D *MyEvent::AddTrack(const TVector3 &pos, Int_t color)
{
    // Add a new track to the list of tracks for this event.
    TPolyLine3D *poly;
    fTracks->Add(new TPolyLine3D());
    fNtrack = fTracks->GetLast();
    poly = (TPolyLine3D *)fTracks->At(fNtrack);
    poly->SetPoint(0, pos.x(), pos.y(), pos.z());
    poly->SetLineColor(color);
    return poly;
}

//______________________________________________________________________________
TPolyLine3D *MyEvent::AddTrack(Double_t x, Double_t y, Double_t z, Int_t col)
{
    // Add a new track to the list of tracks for this event.
    TPolyLine3D *poly;
    fTracks->Add(new TPolyLine3D());
    fNtrack = fTracks->GetLast();
    poly = (TPolyLine3D *)fTracks->At(fNtrack);
    poly->SetPoint(0, x, y, z);
    poly->SetLineColor(col);
    return poly;
}

//______________________________________________________________________________
MyParticle *MyEvent::AddParticle(Int_t id, Int_t pdg_code, const TVector3 &pos,const TVector3 &mom)
{
    // Add a new particle to the list of particles for this event.
    // To avoid calling the very time consuming operator new for each track,
    // the standard but not well know C++ operator "new with placement"
    // is called. If particle[i] is 0, a new MyParticle object will be created
    // otherwise the previous particle[i] will be overwritten.
    TClonesArray &parts = *fParticles;
    MyParticle *part = new(parts[fNparticles++]) MyParticle(id,pdg_code,CREATED,UNDEFINE,pos,mom);
    //Save reference to last Track in the collection of Tracks
    fLastParticle = part;
    return part;
}

//______________________________________________________________________________
Int_t MyEvent::dE_dX(Int_t id)
{
    // Compute de/dx for particle "id" into detector material
    // for more infos, please refer to the particle data booklet
    // from which the formulas has been extracted

    Double_t gamma,abs_beta,abs_p,abs_loss,dX;

    // if particle's energy is equal to its mass, it is at rest, 
    // so set its status as dead
    if(GetParticle(id)->Energy() <= GetParticle(id)->GetMass()) return(DEAD);
    else {
        // absolute value of momentum
        abs_p = GetParticle(id)->GetvMoment().Mag();
        if(abs_p <= 0) {
            // if absolute value of momentum is less or equal to zero,
            // set it to the particle's mass (minimum allowed value for momentum)
            GetParticle(id)->SetMomentum(0.0, 0.0, 0.0,GetParticle(id)->GetMass());
        }
        else {
            // Compute energy loss in detector's material
            // cf Bethe Bloch formula
            TVector3 p_0(GetParticle(id)->GetvMoment() * (1 / abs_p));
            abs_beta = abs_p / GetParticle(id)->Energy();
            dX = fDetector.GetdT() * C * abs_beta;
            abs_beta *= abs_beta;
            if(abs_beta < .9999999999) gamma = 1/sqrt(1-abs_beta);
            else gamma = MAX_GAMMA;
            abs_loss = (fDetector.GetPreconst() * dX / abs_beta) *
                       (log(2 * GetParticle(id)->GetMass() * gamma * gamma * abs_beta /
                        fDetector.GetI()) - abs_beta);
            if(abs_loss < 0) abs_loss = -abs_loss;
            if(abs_loss >= (GetParticle(id)->Energy() - GetParticle(id)->GetMass())) {
                // if energy loss leave less energy to the particle than 
                // its mass, set its momentum equal to its mass 
                // (minimum allowed value for momentum)
                GetParticle(id)->SetMomentum(0.0, 0.0, 0.0, GetParticle(id)->GetMass());
            }
            else {
                // else decrease its energy by calculated energy loss
                GetParticle(id)->SetMoment(GetParticle(id)->GetvMoment(),
                    GetParticle(id)->Energy() - abs_loss);
                abs_p = sqrt((GetParticle(id)->Energy() * GetParticle(id)->Energy()) -
                             (GetParticle(id)->GetMass() * GetParticle(id)->GetMass()));
                GetParticle(id)->SetMoment(p_0 * abs_p);
                // Add calculated energy loss at total particle's energy loss
                GetParticle(id)->AddELoss(abs_loss);
            }
        }
        if(GetParticle(id)->Energy() > GetParticle(id)->GetMass()) return(ALIVE);
        else return(DEAD);
    }
}

//______________________________________________________________________________
Int_t MyEvent::Bremsstrahlung(Int_t id)
{
    // compute bremsstrahlung for particle "id"
    Double_t  ratio;
    Int_t     d_num1,d_num2;
    Char_t    strtmp[80];
    MyParticle *part;

    // find two ids for children particles
    if((FindFreeId(&d_num1) != DEAD) && (FindFreeId(&d_num2) != DEAD)) {
        // compute the particle's energy ratio...
        ratio = (GetParticle(id)->Energy() - GetParticle(id)->GetMass()) /
                (2 * GetParticle(id)->GetvMoment().Mag());
        // create first child if fact, electron continues with less energy 
        // and in a different direction. To that end the electron is added 
        // to its own list of children, because otherwise it would vanish.
        part = AddParticle(d_num1, GetParticle(id)->GetPdgCode(), GetParticle(id)->GetvLocation(),
                    GetParticle(id)->GetvMoment() * ratio);
        part->SetFirstMother(id);
        // as its first child is in fact the same particle, 
        // keep the same decay time
        part->SetTimeOfDecay(GetParticle(id)->GetTimeOfDecay());
        GetParticle(id)->SetChild(0, d_num1);
        // add a track related to this particle
        AddTrack(GetParticle(id)->GetvLocation(),Particle_color(id));

        // add a particle related list tree item to the event list tree
        gTmpLTI = gEventListTree->AddItem(gLTI[id], part->GetName());
        sprintf(strtmp,"%1.2e GeV",part->Energy());
        gEventListTree->SetToolTipItem(gTmpLTI, strtmp);
        gLTI[d_num1] = gTmpLTI;

        // create second child
        part = AddParticle(d_num2,PHOTON, GetParticle(id)->GetvLocation(),
                    GetParticle(id)->GetvMoment() * ratio);
        part->SetFirstMother(id);
        // generate time of decay (not used in this case, as it is a photon,
        // but to keep the same philosophy in every case...
        part->GenerateTimeOfDecay();
        GetParticle(id)->SetChild(1, d_num2);
        
        // add a track related to this particle
        AddTrack(GetParticle(id)->GetvLocation(),Particle_color(id));

        // add a particle related list tree item to the event list tree
        gTmpLTI = gEventListTree->AddItem(gLTI[id],part->GetName());
        sprintf(strtmp,"%1.2e GeV",part->Energy());
        gEventListTree->SetToolTipItem(gTmpLTI, strtmp);
        gLTI[d_num2] = gTmpLTI;

        // increment number of children by the two created particles
        GetParticle(id)->SetNChildren(2);

        return(ALIVE);
    }
    else return(DEAD);
}

//______________________________________________________________________________
Int_t MyEvent::Pair_production(Int_t id)
{
    // compute the pair production for particle "id"
    Double_t ratio = 1.0;
    Int_t    d_num1,d_num2;
    Char_t   strtmp[80];
    MyParticle *part;

    // find two ids for children particles
    if((FindFreeId(&d_num1) != DEAD) && (FindFreeId(&d_num2) != DEAD)) {
        // compute energy ratio for particles creation
        ratio = sqrt((GetParticle(id)->Energy() * GetParticle(id)->Energy())/4.0)
                          / GetParticle(id)->GetvMoment().Mag();
        // create first child
        part = AddParticle(d_num1, POSITRON, GetParticle(id)->GetvLocation(),
                    GetParticle(id)->GetvMoment() * ratio);
        part->SetFirstMother(id);
        // generate time of decay (not used in this case, as it is an electron
        // or a positron, but to keep the same philosophy in every case...
        part->GenerateTimeOfDecay();
        GetParticle(id)->SetChild(0, d_num1);
        // add a track related to this particle
        AddTrack(GetParticle(id)->GetvLocation(),Particle_color(id));

        // add a particle related list tree item to the event list tree
        gTmpLTI = gEventListTree->AddItem(gLTI[id], part->GetName());
        sprintf(strtmp,"%1.2e GeV",part->Energy());
        gEventListTree->SetToolTipItem(gTmpLTI, strtmp);
        gLTI[d_num1] = gTmpLTI;

        // create second child
        part = AddParticle(d_num2, ELECTRON, GetParticle(id)->GetvLocation(),
                    GetParticle(id)->GetvMoment() * ratio);
        part->SetFirstMother(id);
        // generate time of decay (not used in this case, as it is an electron
        // or a positron, but to keep the same philosophy in every case...
        part->GenerateTimeOfDecay();
        GetParticle(id)->SetChild(1, d_num2);
        // add a track related to this particle
        AddTrack(GetParticle(id)->GetvLocation(),Particle_color(id));

        // add a particle related list tree item to the event list tree
        gTmpLTI = gEventListTree->AddItem(gLTI[id], part->GetName());
        sprintf(strtmp,"%1.2e GeV",part->Energy());
        gEventListTree->SetToolTipItem(gTmpLTI, strtmp);
        gLTI[d_num2] = gTmpLTI;

        // increment number of children by the two created particles
        GetParticle(id)->SetNChildren(2);

        return(ALIVE);
    }
    else return(DEAD);

}

//______________________________________________________________________________
Int_t MyEvent::Action(Int_t id)
{
    // main event's action
    Int_t  nchild;
    if(GetParticle(id)->GetDecayType() == UNDEFINE)
        Define_decay(id);
    if(GetParticle(id)->GetPdgCode() == PHOTON){
        // compute the step delta x to be covered by the particle
        TVector3 delta_x(GetParticle(id)->GetvMoment() * (C * fDetector.GetdT() / GetParticle(id)->Energy()));
        // check if moved too far (out of detector's bouds) 
        if(Move(id, delta_x) == DEAD)
            // set its status as dead 
            DeleteParticle(id);
        else {
            // if distance covered is greater than particle's decay length,
            // apply pair production and check if particle is dead. If not,
            // increment total alive particles by the two created children,
            // then set the particle status as dead
            if(GetParticle(id)->GetPassed() >= GetParticle(id)->GetDecayLength()) {
                if(Pair_production(id) == DEAD) return(DEAD);
                else {
                    fAliveParticles += 2;
                    DeleteParticle(id);
                }
            }
        }
    }
    else if((GetParticle(id)->GetPdgCode() == NEUTRINO_E) ||
            (GetParticle(id)->GetPdgCode() == NEUTRINO_TAU) ||
            (GetParticle(id)->GetPdgCode() == NEUTRINO_MUON) ||
            (GetParticle(id)->GetPdgCode() == ANTINEUTRINO_E) ||
            (GetParticle(id)->GetPdgCode() == ANTINEUTRINO_TAU) ||
            (GetParticle(id)->GetPdgCode() == ANTINEUTRINO_MUON) ) {
            // if current particle is a neutrino ( or antineutrino )
            // set its status as dead ( estimate its probability of 
            // interaction as null )
            DeleteParticle(id);
    }
    else { // particle is not a photon or neutrino
        // if current particle is charged, apply magnetic field influence
        if(GetParticle(id)->GetPDG()->Charge() != 0)
            ScatterAngle(id);
        if((fB != 0) && (GetParticle(id)->GetPDG()->Charge() != 0)) Magnetic_field(id);
        // compute the step delta x to be covered by the particle
        TVector3 delta_x(GetParticle(id)->GetvMoment() * (C * fDetector.GetdT() / GetParticle(id)->Energy()));
        // check if moved too far (out of detector's bouds) 
        if(Move(id, delta_x) == DEAD) {
            // set its status as dead 
            DeleteParticle(id);
        }
        else {
            // check energy loss, and if too much energy loss ( particle at rest )
            // set its status as dead 
            if(dE_dX(id) == DEAD) DeleteParticle(id);
            else {
                // if at end of particle's life time, decay it
                if(CheckDecayTime(id) == 1) {
                    // if no child found
                    if((nchild = Decay(id)) == -1) return(DEAD);
                    else {
                        // else increment total alive particles by amount
                        // of particle's children
                        fAliveParticles += nchild;
                        DeleteParticle(id);
                    }
                }
                // if not at end of particle's life time, check if distance 
                // covered is greater than particle's decay length, apply 
                // defined decay type and check if particle is dead. If not,
                // increment total alive particles by the two created children,
                // then set the particle status as dead
                else if(GetParticle(id)->GetPassed() >= GetParticle(id)->GetDecayLength()) {
                    switch(GetParticle(id)->GetDecayType()) {
                        case BREMS:
                            if(Bremsstrahlung(id) == DEAD) return(DEAD);
                            else {
                                fAliveParticles += 2;
                                DeleteParticle(id);
                            }
                            break;
                        case CONVERSION:
                            if(Pair_production(id) == DEAD) return(DEAD);
                            else {
                                fAliveParticles += 2;
                                DeleteParticle(id);
    		                }
                            break;
                    }
                }
            }
	    }
    }
    return(ALIVE);
}

//______________________________________________________________________________
void MyEvent::Magnetic_field(Int_t id)
{
    // apply magnetic field ...
    Double_t abs_p;

    TVector3 e_B(1.0, 0.0, 0.0);

    TVector3 beta(GetParticle(id)->GetvMoment() * (1.0e-03 / GetParticle(id)->Energy()));
    TVector3 tmp_p(beta.Cross(e_B));
    TVector3 delta_p(tmp_p * (fB * C * fDetector.GetdT()));
    abs_p = GetParticle(id)->GetvMoment().Mag();
    if(GetParticle(id)->GetPDG()->Charge() < 0)
        GetParticle(id)->SetMoment(GetParticle(id)->GetvMoment() - delta_p);
    else if(GetParticle(id)->GetPDG()->Charge() > 0)
        GetParticle(id)->SetMoment(GetParticle(id)->GetvMoment() + delta_p);
    Double_t module = GetParticle(id)->GetvMoment().Mag();
    GetParticle(id)->SetMoment(GetParticle(id)->GetvMoment() * (abs_p / module));
}

//______________________________________________________________________________
Int_t MyEvent::Move(Int_t id, const TVector3 &dist)
{
    // Move particle "id" by step dist, update the distance covered
    // then check if out of detector's bounds
    
    GetParticle(id)->SetLocation(GetParticle(id)->GetvLocation() + dist);
    GetParticle(id)->SetPassed(GetParticle(id)->GetPassed() + dist.Mag());

    if((GetParticle(id)->GetvLocation().x() > fDetector.GetMaxX()) ||
       (GetParticle(id)->GetvLocation().x() < fDetector.GetMinX()) ||
       (GetParticle(id)->GetvLocation().y() > fDetector.GetMaxY()) ||
       (GetParticle(id)->GetvLocation().y() < fDetector.GetMinY()) ||
       (GetParticle(id)->GetvLocation().z() > fDetector.GetMaxZ()) ||
       (GetParticle(id)->GetvLocation().z() < fDetector.GetMinZ())) {
        return(DEAD);
    }
    // If not out of bounds, set related Track's next point
    else {
        GetTrack(id)->SetNextPoint(GetParticle(id)->GetvLocation().x(),
            GetParticle(id)->GetvLocation().y(),GetParticle(id)->GetvLocation().z());
        return(ALIVE);
    }
}

//______________________________________________________________________________
void MyEvent::Define_decay(Int_t id)
{
    // Define decay type for particle "id", then check decay length for it

    Double_t idecay_length = -1.;
    Double_t iactual_length;
    Int_t    idecay_type = CONVERSION;

    if ( (GetParticle(id)->GetPdgCode() == ELECTRON) ||
         (GetParticle(id)->GetPdgCode() == POSITRON)) {
        // check if bremsstrahlung is allowed
        if( (iactual_length = Brems_prob(id)) > 0.) {
            if( (idecay_length == -1) || (iactual_length < idecay_length) ) {
                idecay_length = iactual_length;
                idecay_type = BREMS;
            }
        }
    }
    else if(GetParticle(id)->GetPdgCode() == PHOTON) {
        // check if pair production is allowed
        if( (iactual_length = Pair_prob(id)) > 0. ) {
            if( (idecay_length == -1) ||
                (iactual_length < idecay_length) ) {
                idecay_length = iactual_length;
                idecay_type = CONVERSION;
            }
        }
    }
    if( idecay_length > 0) {
        GetParticle(id)->SetDecayType(idecay_type);
        GetParticle(id)->SetDecayLength(idecay_length);
    }
    else {
        GetParticle(id)->SetDecayType(STABLE);
        GetParticle(id)->SetDecayLength(0.0);
    }
}

//______________________________________________________________________________
Double_t MyEvent::Pair_prob(Int_t id)
{
    // check if pair production is allowed and generate
    // a random decay length related to detector's material
    // radiation length (X0)
    Double_t p;

    if(GetParticle(id)->Energy() > 2.0 * m_e) {
        p = gRandom->Uniform(0.,1.0);
        return ((-9.)*fDetector.GetX0()*log(p)/7.);
    }
    return (-1.);
}

//______________________________________________________________________________
Double_t MyEvent::Brems_prob(Int_t id)
{
    // Check if bremsstrahlung is allowed and generate
    // a random decay length related to detector's material
    // radiation length (X0)
    Double_t p, retval;

    if(GetParticle(id)->Energy() > GetParticle(id)->GetMass()) {
        p = gRandom->Uniform(0.,1.0);
        retval = (-fDetector.GetX0())*log(p);
        return (retval);
    }
    else return (-1.);
}

//______________________________________________________________________________
void MyEvent::DeleteParticle(Int_t id)
{
    // in the case of the particle has been created and died
    // before to can take a step, there is only one point on
    // its track...then set second point before marking its
    // status as dead.
    // it should not append, but who knows...
    if(GetTrack(id)->GetN() == 2) {
        Float_t *pts = GetTrack(id)->GetP();
        // check if track's second point is not set
        if(isnan(pts[4])) {
            GetTrack(id)->SetPoint(1,pts[0], pts[1], pts[2]);
        }
    }
    // Add this particle's energy loss at the total 
    // energy loss into the detector
    fDetector.AddELoss(GetParticle(id)->GetELoss());
    // Mark the particle's status as dead and decrement
    // the total alive particles
    GetParticle(id)->SetStatus(DEAD);
    fAliveParticles --;
}

//______________________________________________________________________________
Int_t MyEvent::FindFreeId(Int_t *FreeId)
{
    // give next available particle's id
    fTotalParticles++;
    *FreeId = fTotalParticles;
    if(fTotalParticles > fLast) fLast = fTotalParticles;
    return(ALIVE);
}

//______________________________________________________________________________
Int_t MyEvent::Particle_color(Int_t id)
{
    // return color index related to particle's energy
    Int_t i;
    for(i=0;i<10;i++)
        if(GetParticle(id)->Energy() > E_thresh[i]) break;
    if(i > 9) i = 9;
    return(gColIndex + i);
}

//______________________________________________________________________________
TVector3 MyEvent::FindOrtho(const TVector3 &vec)
{
    Double_t abs_b;
    // Find what is the orthogonal direction of vector vec 
    // (used by scatter angle calculation)
    if(vec.Mag() == 0) return(vec);
    else {
        TVector3 n_0(1.0, 0.0, 0.0);
        TVector3 u_0(0.0, 1.0, 0.0);
        TVector3 b(vec.Cross(n_0));
        if((abs_b = b.Mag()) == 0) {
            TVector3 b(vec.Cross(u_0)); 
            abs_b = b.Mag();
        }
        if(abs_b > 1) b = b * (1.0/abs_b);
        return(b);
    }
}

//______________________________________________________________________________
void MyEvent::ScatterAngle(Int_t id)
{
    // compute scatter angle into the detector's material
    // for the current particle
    // for more infos, please refer to the particle data booklet
    // from which the formulas has been extracted - chapter 23.3
    // Multiple scattering through small angles formula 23.9
    Double_t alpha,beta;
    Double_t abs_p,p1,p2,r_2;
    Double_t fact1,fact2;

    do {
        p1 = gRandom->Uniform(-1.,1.);
        p2 = gRandom->Uniform(-1.,1.);
        r_2 = (p1 * p1) + (p2 * p2);
    } while(r_2 > 1.);
    abs_p = GetParticle(id)->GetvMoment().Mag();
    alpha = sqrt((-2.)*log(r_2)/r_2) * fDetector.GetTheta0() / abs_p;
    beta  = gRandom->Uniform(0.,2.* TMath::Pi());
    alpha *= p1;
    TVector3 x_0(FindOrtho(GetParticle(id)->GetvMoment()));
    TVector3 p_0(GetParticle(id)->GetvMoment() * (1./abs_p));
    TVector3 y_0(x_0.Cross(p_0));
    fact1 = sin(alpha);
    fact2 = fact1*cos(beta);
    fact1 *= sin(beta);
    TVector3 vtmp1(x_0 * fact1);
    TVector3 vtmp2(y_0 * fact2);
    TVector3 vtmp3(vtmp2 + p_0);

    GetParticle(id)->SetMoment(vtmp1 + vtmp3);
    GetParticle(id)->SetMoment(GetParticle(id)->GetvMoment() *
        (abs_p/ GetParticle(id)->GetvMoment().Mag()));
}

//______________________________________________________________________________
Int_t MyEvent::CheckDecayTime(Int_t id)
{
    if ( (GetParticle(id)->GetPdgCode() == PHOTON) ||
         (GetParticle(id)->GetPdgCode() == ELECTRON) ||
         (GetParticle(id)->GetPdgCode() == POSITRON ))
         return 0;
    Double_t timeofdecay = GetParticle(id)->GetTimeOfDecay();
    if(timeofdecay == 0.0)  return 0;
//    if(timeofdecay == 0.0)  timeofdecay = gRandom->Uniform(1.0e-9, 1.0e-6);
    Double_t distToDecay = timeofdecay * 0.996 * C;
    // check if actual particle life is greater than particle life time
    if (GetParticle(id)->GetPassed() >= distToDecay) {
        return 1;
    }
    return 0;
}

//______________________________________________________________________________
Int_t MyEvent::Decay(Int_t id)
{
    Char_t   strtmp[80];
    Double_t ratio;
    Int_t    d_num[5];
    Int_t    n_daughters;
    Int_t    ptype[5];
    Double_t mass[5];
    Int_t    i, index;
    Double_t sumBR = 0.0;
    MyParticle *Particle[5];
    MyParticle *part;

    // compute total branching ratio
    for(i=0;i<GetParticle(id)->GetPDG()->NDecayChannels();i++) {
        sumBR += GetParticle(id)->GetPDG()->DecayChannel(i)->BranchingRatio();
    }
    // choose random decay in respect to the branching ratio
    float r = gRandom->Uniform(sumBR);
    index = 0;
    while ((r -= GetParticle(id)->GetPDG()->DecayChannel(index)->BranchingRatio()) > 0
        && index < GetParticle(id)->GetPDG()->NDecayChannels()) index++;

    // set number of daughters
    n_daughters = GetParticle(id)->GetPDG()->DecayChannel(index)->NDaughters();
    for(i=0;i<n_daughters;i++) {
        // create temporary child particle to obtain its mass
        ptype[i] = GetParticle(id)->GetPDG()->DecayChannel(index)->DaughterPdgCode(i);
        Particle[i] = new MyParticle(0,ptype[i], CREATED, UNDEFINE,
            GetParticle(id)->GetvLocation(), GetParticle(id)->GetvMoment());
        mass[i] = Particle[i]->GetMass();
        delete Particle[i];
    }

    // find ids for children
    for(i=0;i<n_daughters;i++) {
        if(FindFreeId(&d_num[i]) == DEAD) return -1;
    }
    // total children mass
    Double_t total_mass = 0.0;
    for(i=0;i<n_daughters;i++) {
        total_mass += mass[i];
    }

    // compute energy ratio
    ratio = sqrt(((GetParticle(id)->Energy() * GetParticle(id)->Energy())/(2*(n_daughters+1)))
        - (total_mass * total_mass)) / GetParticle(id)->GetvMoment().Mag();

    for(i=0;i<n_daughters;i++) {

        // create child
        part = AddParticle(d_num[i], ptype[i], GetParticle(id)->GetvLocation(),
                    GetParticle(id)->GetvMoment() * ratio);
        part->SetFirstMother(id);
        // generate time of decay (may be useful in this case)
        part->GenerateTimeOfDecay();
        GetParticle(id)->SetChild(i, d_num[i]);
        // add a track related to this child
        AddTrack(GetParticle(id)->GetvLocation(),Particle_color(id));

        // add a child related list tree item to the event list tree
        gTmpLTI = gEventListTree->AddItem(gLTI[id],
            part->GetName());
        sprintf(strtmp,"%1.2e GeV",part->Energy());
        gEventListTree->SetToolTipItem(gTmpLTI, strtmp);
        gLTI[d_num[i]] = gTmpLTI;

    }
    // increment number of children by the number of created particles
    GetParticle(id)->SetNChildren(n_daughters);
    return(n_daughters);

}

