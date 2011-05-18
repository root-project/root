
///////////////////////////////////////////////////////////////////
//  ROOT implementation of the X11 xclock.
//
//  To run this example do the following:
//  $ root
//  root [0] gSystem.Load("Aclock")
//  root [1] Aclock a
//  <enjoy>
//  root [2] .q
//
//  Other ROOT fun examples: Tetris, Hello ...
///////////////////////////////////////////////////////////////////

#ifndef ACLOCK_H
#define ACLOCK_H

#include <TTimer.h>
#include <TCanvas.h>
#include <TPolyLine.h>
#include <TDatime.h>
#include <TPoints.h>
#include <TMath.h>
#include <TList.h>

class TPolygon : public TPolyLine {

protected:
   TPad  *fPad;

public:
   TPolygon(Int_t n, Float_t *x, Float_t *y);
   virtual ~TPolygon() { fPad->GetListOfPrimitives()->Remove(this); }

   virtual void Paint(Option_t *option="");

   TPad  *GetPad() { return fPad; }
};



class ClockPoints : public TPoints {

public:
   ClockPoints(Coord_t x=0, Coord_t y=0) : TPoints(x,y) { }
   ~ClockPoints() { }

   void SetXY(Coord_t x, Coord_t y) { SetX(x); SetY(y); }

   void Rotate(Float_t clock_angle)  // Rotates the coordinate system a clock_angle degrees clockwise
   {
      const float deg2rad = .017453292519943295769F;  // pi/180

      Float_t rX, rY;
      Float_t angle = clock_angle*deg2rad;     // clock_angle to angle in radians

      rX = GetX()*TMath::Cos(angle)+GetY()*TMath::Sin(angle);
      rY = GetY()*TMath::Cos(angle)-GetX()*TMath::Sin(angle);
      SetXY(rX,rY);
   }

   void Scale(Float_t factorX, Float_t factorY) { SetX(GetX()*factorX); SetY(GetY()*factorY); }
   void Shift(Coord_t x, Coord_t y) { SetX(GetX()+x); SetY(GetY()+y); }
};



class ClockHand : public TPolygon {

protected:
   UInt_t   fPrevTimeValue;    // used during updating
   Float_t *fX0;         // initial shape of clock hand corresponds to 00:00:00
   Float_t *fY0;         // initial shape of clock hand corresponds to 00:00:00

   static TDatime *fgTime;           // current date/time

   void  Move(Float_t angle);        // rotate initial shape to angle
   virtual UInt_t   GetTimeValue() { return GetMinute(); } // could be overloaded
   virtual Float_t  GetHandAngle() { return 0; }           // must be overloaded

public:
   ClockHand(Int_t n, Float_t *x, Float_t *y);
   virtual ~ClockHand() { }

   UInt_t GetTime()    { fgTime->Set(); return fgTime->GetTime(); }
   UInt_t GetHour()    { return  GetTime()/10000; }
   UInt_t GetMinute()  { return (GetTime()%10000)/100; }
   UInt_t GetSecond()  { return (GetTime()%100); }

   void   Update();

   Bool_t IsModified() { return (fPrevTimeValue != GetTimeValue()); }
};



class MinuteHand : public ClockHand {

private:
   static Float_t fgMinuteHandX[];
   static Float_t fgMinuteHandY[];

public:
   MinuteHand(Int_t n=3, Float_t *x=fgMinuteHandX, Float_t *y=fgMinuteHandY)
      : ClockHand(n,x,y) { }
   ~MinuteHand() { }

   Float_t GetHandAngle() { return 6.*(GetMinute()+ GetSecond()/60.); }
};



class HourHand : public ClockHand {

private:
   static Float_t fgHourHandX[];
   static Float_t fgHourHandY[];

public:
   HourHand(Int_t n=3, Float_t *x=fgHourHandX, Float_t *y=fgHourHandY)
      : ClockHand(n,x,y) { }
   ~HourHand() { }

   Float_t GetHandAngle() { return 30.*(GetHour()%12 + GetMinute()/60.); }
};



class SecondHand : public ClockHand {

private:
   static Float_t fgSecondHandX[];
   static Float_t fgSecondHandY[];

protected:
   UInt_t GetTimeValue() { return GetSecond(); }   // used to update every second

public:
   SecondHand(Int_t n=4, Float_t *x=fgSecondHandX, Float_t *y=fgSecondHandY)
      : ClockHand(n,x,y) { }
   ~SecondHand() { }

   Float_t GetHandAngle() { return  6.*GetSecond(); }
};



class Aclock : public TTimer {

private:
   TPad       *fPad;            // pad where this clock is drawn
   MinuteHand *fMinuteHand;     // minute hand
   HourHand   *fHourHand;       // hour hand
   SecondHand *fSecondHand;     // second hand

public:
   Aclock(Int_t csize=100);
   virtual ~Aclock();

   virtual Bool_t Notify();
   void   Paint(Option_t *option);
   void   Animate();

   ClassDef(Aclock,0)  // analog clock = xclock
};

#endif   // ACLOCK
