/// \file
/// \ingroup tutorial_legacy
/// A set of classes deriving from TTask.
/// See macro tasks.C to see an example of use
/// The Exec function of each class prints one line when it is called.
///
/// \macro_code
///
/// \author Rene Brun

#include "TTask.h"

class MyRun : public TTask {

public:
   MyRun() {;}
   MyRun(const char *name, const char *title);
   ~MyRun() override {;}
   void Exec(Option_t *option="") override;

   ClassDefOverride(MyRun,1)   // Run Reconstruction task
};

class MyEvent : public TTask {

public:
   MyEvent() {;}
   MyEvent(const char *name, const char *title);
   ~MyEvent() override {;}
   void Exec(Option_t *option="") override;

   ClassDefOverride(MyEvent,1)   // Event Reconstruction task
};

class MyGeomInit : public TTask {

public:
   MyGeomInit() {;}
   MyGeomInit(const char *name, const char *title);
   ~MyGeomInit() override {;}
   void Exec(Option_t *option="") override;

   ClassDefOverride(MyGeomInit,1)   // Geometry initialisation task
};

class MyMaterialInit : public TTask {

public:
   MyMaterialInit() {;}
   MyMaterialInit(const char *name, const char *title);
   ~MyMaterialInit() override {;}
   void Exec(Option_t *option="") override;

   ClassDefOverride(MyMaterialInit,1)   // Materials initialisation task
};

class MyTracker : public TTask {

public:
   MyTracker() {;}
   MyTracker(const char *name, const char *title);
   ~MyTracker() override {;}
   void Exec(Option_t *option="") override;

   ClassDefOverride(MyTracker,1)   // Main Reconstruction task
};

class MyRecTPC : public TTask {

public:
   MyRecTPC() {;}
   MyRecTPC(const char *name, const char *title);
   ~MyRecTPC() override {;}
   void Exec(Option_t *option="") override;

   ClassDefOverride(MyRecTPC,1)   // TPC Reconstruction
};


class MyRecITS : public TTask {

public:
   MyRecITS() {;}
   MyRecITS(const char *name, const char *title);
   ~MyRecITS() override {;}
   void Exec(Option_t *option="") override;

   ClassDefOverride(MyRecITS,1)   // ITS Reconstruction
};


class MyRecMUON : public TTask {

public:
   MyRecMUON() {;}
   MyRecMUON(const char *name, const char *title);
   ~MyRecMUON() override {;}
   void Exec(Option_t *option="") override;

   ClassDefOverride(MyRecMUON,1)   // MUON Reconstruction
};


class MyRecPHOS : public TTask {

public:
   MyRecPHOS() {;}
   MyRecPHOS(const char *name, const char *title);
   ~MyRecPHOS() override {;}
   void Exec(Option_t *option="") override;

   ClassDefOverride(MyRecPHOS,1)   // PHOS Reconstruction
};


class MyRecRICH : public TTask {

public:
   MyRecRICH() {;}
   MyRecRICH(const char *name, const char *title);
   ~MyRecRICH() override {;}
   void Exec(Option_t *option="") override;

   ClassDefOverride(MyRecRICH,1)   // RICH Reconstruction
};


class MyRecTRD : public TTask {

public:
   MyRecTRD() {;}
   MyRecTRD(const char *name, const char *title);
   ~MyRecTRD() override {;}
   void Exec(Option_t *option="") override;

   ClassDefOverride(MyRecTRD,1)   // TRD Reconstruction
};


class MyRecGlobal : public TTask {

public:
   MyRecGlobal() {;}
   MyRecGlobal(const char *name, const char *title);
   ~MyRecGlobal() override {;}
   void Exec(Option_t *option="") override;

   ClassDefOverride(MyRecGlobal,1)   // Global Reconstruction
};


////////////////////////////////////////////////////////////////////////////////

MyRun::MyRun(const char *name, const char *title)
      :TTask(name,title)
{
}

void MyRun::Exec(Option_t * /*option*/)
{
   printf("MyRun executing\n");
}

////////////////////////////////////////////////////////////////////////////////

MyEvent::MyEvent(const char *name, const char *title)
      :TTask(name,title)
{
}

void MyEvent::Exec(Option_t * /*option*/)
{
   printf("MyEvent executing\n");
}

////////////////////////////////////////////////////////////////////////////////

MyGeomInit::MyGeomInit(const char *name, const char *title)
      :TTask(name,title)
{
}

void MyGeomInit::Exec(Option_t * /*option*/)
{
   printf("MyGeomInit executing\n");
}

////////////////////////////////////////////////////////////////////////////////

MyMaterialInit::MyMaterialInit(const char *name, const char *title)
      :TTask(name,title)
{
}

void MyMaterialInit::Exec(Option_t * /*option*/)
{
   printf("MyMaterialInit executing\n");
}

////////////////////////////////////////////////////////////////////////////////

MyTracker::MyTracker(const char *name, const char *title)
      :TTask(name,title)
{
}

void MyTracker::Exec(Option_t * /*option*/)
{
   printf("MyTracker executing\n");
}

////////////////////////////////////////////////////////////////////////////////

MyRecTPC::MyRecTPC(const char *name, const char *title)
      :TTask(name,title)
{
}

void MyRecTPC::Exec(Option_t * /*option*/)
{
   printf("MyRecTPC executing\n");
}

////////////////////////////////////////////////////////////////////////////////

MyRecITS::MyRecITS(const char *name, const char *title)
      :TTask(name,title)
{
}

void MyRecITS::Exec(Option_t * /*option*/)
{
   printf("MyRecITS executing\n");
}

////////////////////////////////////////////////////////////////////////////////

MyRecMUON::MyRecMUON(const char *name, const char *title)
      :TTask(name,title)
{
}

void MyRecMUON::Exec(Option_t * /*option*/)
{
   printf("MyRecMUON executing\n");
}

////////////////////////////////////////////////////////////////////////////////

MyRecPHOS::MyRecPHOS(const char *name, const char *title)
      :TTask(name,title)
{
}

void MyRecPHOS::Exec(Option_t * /*option*/)
{
   printf("MyRecPHOS executing\n");
}

////////////////////////////////////////////////////////////////////////////////

MyRecRICH::MyRecRICH(const char *name, const char *title)
      :TTask(name,title)
{
}

void MyRecRICH::Exec(Option_t * /*option*/)
{
   printf("MyRecRICH executing\n");
}

////////////////////////////////////////////////////////////////////////////////

MyRecTRD::MyRecTRD(const char *name, const char *title)
      :TTask(name,title)
{
}

void MyRecTRD::Exec(Option_t * /*option*/)
{
   printf("MyRecTRD executing\n");
}

////////////////////////////////////////////////////////////////////////////////

MyRecGlobal::MyRecGlobal(const char *name, const char *title)
      :TTask(name,title)
{
}

void MyRecGlobal::Exec(Option_t * /*option*/)
{
   printf("MyRecGlobal executing\n");
}
