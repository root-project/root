/// \file
/// \ingroup tutorial_eve
///
/// \macro_code
///
/// \author Matevz Tadel

void camera_restore(const char* fname, int mode = 0)
{
  TEveManager::Create();
  writeCurrentCamera(fname);
}

void writeCurrentCamera(const char* fname)
{
  TGLCamera& c = gEve->GetDefaultGLViewer()->CurrentCamera();
  TFile* f = TFile::Open(fname, "RECREATE");
  c.Write();
  f->Close();
}


void readCurrentCamera(const char* fname)
{
  TGLCamera& c = gEve->GetDefaultGLViewer()->CurrentCamera();
  TFile* f = TFile::Open(fname, "READ");
  if (!f)
    return;

  if (f->GetKey(c.ClassName())) {
    f->GetKey(c.ClassName())->Read(&c);
    c.IncTimeStamp();
    gEve->GetDefaultGLViewer()->RequestDraw();
  }
}
