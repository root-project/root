// @(#)root/eve:$Id: text_test.C 26717 2008-12-07 22:07:55Z matevz $
// Author: Alja Mrak-Tadel

// Makes some tracks with three different magnetic field types.

void track(Int_t bCase = 2)
{
  gSystem->IgnoreSignal(kSigSegmentationViolation, true);
  TEveManager::Create();

  TEveTrackPropagator* prop = new TEveTrackPropagator();
  prop->SetRnrDaughters(kTRUE);
  gEve->AddElement(prop);

  TEveTrack *track = 0;
  switch (bCase)
  {
    case 0:
    {
      // B = 0 no difference btween signed and charge particles
      prop->SetMagField(0.);
      prop->SetName("ZeroB");
      track = make_track(prop, 0);
      track = make_track(prop, 1);
      break;
    }

    case 1:
    {
      // constant B field, const angle
      prop->SetMagFieldObj(new TEveMagFieldConst(0., 0., -3.8));
      prop->SetName("ConstB");
      prop->SetMaxR(123);
      prop->SetMaxZ(300);
      track = make_track(prop, 0);
      track = make_track(prop, -1);
      break;
    }
    case 2:
    {
      // variable B field, sign change at  R = 200 cm
      prop->SetMagFieldObj(new TEveMagFieldDuo(200, -4.4, 2));
      prop->SetName("DuoB");
      prop->SetMaxR(600);
      prop->SetMaxZ(700);
      prop->SetMinAng(1.f);
      track = make_track(prop, 0);
      track = make_track(prop, -1);
      break;
    }
  };
  //  make_projected(track);
  gEve->Redraw3D(1);
}

//______________________________________________________________________________
TEveTrack* make_track(TEveTrackPropagator* prop, Int_t sign)
{
  // Make track with given propagator.
  // Add to math-marks to test fit.

  TEveRecTrack *rc = new TEveRecTrack();
  rc->fV.Set(0.028558, -0.000918, 3.691919);
  rc->fP.Set(0.767095, -2.400006, -0.313103);
  rc->fSign = sign;
  TEveTrack* track = new TEveTrack(rc, prop);
  track->SetName(Form("Charge %d", sign));

  // daughter 0
  TEvePathMark* pm1 = new TEvePathMark(TEvePathMark::kDaughter);
  pm1->fV.Set(1.479084, -4.370661, 3.119761);
  track->AddPathMark(*pm1);
  // daughter 1
  TEvePathMark* pm2 = new TEvePathMark(TEvePathMark::kDaughter);
  pm2->fV.Set(57.72345, -89.77011, -9.783746);
  track->AddPathMark(*pm2);

  // color by sign
  if (sign)
  {
    track->SetLineColor(kRed-3);
    track->SetLineWidth(2);
  }
  else
  {
    track->SetLineColor(kOrange);
  }

  track->MakeTrack();
  gEve->AddElement(track, prop);

  return track;
}

//______________________________________________________________________________
void make_projected(TEveTrack *track)
{
   // Make non-linear projected view.

  TEveViewer* v1 = gEve->SpawnNewViewer("2D Viewer");
  TEveScene*  s1 = gEve->SpawnNewScene("Projected Event");
  v1->AddScene(s1);
  TGLViewer* v = v1->GetGLViewer();
  v->SetCurrentCamera(TGLViewer::kCameraOrthoXOY);
  v->SetGuideState(TGLUtil::kAxesOrigin, kTRUE, kFALSE, 0);

  TEveProjectionManager* mng = new TEveProjectionManager();
  mng->SetProjection(TEveProjection::kPT_RhoZ);
  gEve->AddElement(mng, s1);
  gEve->AddToListTree(mng, kTRUE);
  mng->ImportElements(track);
}
