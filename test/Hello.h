
///////////////////////////////////////////////////////////////////
//  Animated Text with cool wave effect.
//
//  ROOT implementation of the hello world example borrowed
//  from the Qt hello world example.
//
//  To run this example do the following:
//  $ root
//  root [0] gSystem.Load("Hello")
//  root [1] Hello h
//  <enjoy>
//  root [2] .q
//
//  Other ROOT fun examples: Tetris, Aclock ...
///////////////////////////////////////////////////////////////////

#ifndef HELLO_H
#define HELLO_H

#include <TTimer.h>
#include <TCanvas.h>
#include <TText.h>

class TList;

class TChar : public TText {

public:
   TChar(char ch='\0',Coord_t x=0, Coord_t y=0);
   ~TChar() override { }

   char GetChar() {
      return GetTitle()[0];
   }

   virtual Float_t GetWidth();
};


class Hello : public TTimer {

private:
   TList  *fList;     // list of characters
   UInt_t  fI;        // "infinit"  counter
   TPad   *fPad;      // pad where this text is drawn

public:
   Hello(const char *text = "Hello, World!");
   ~Hello() override;

   Bool_t  Notify() override;
   void    ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
   Int_t   DistancetoPrimitive(Int_t, Int_t) override { return 0; }

   Float_t GetWidth();
   void    Paint(Option_t* option="") override;
   void    Print(Option_t * = "") const override;
   void    ls(Option_t * = "") const override;
   TList  *GetList() { return fList; }

   ClassDefOverride(Hello,0)   // animated text with cool wave effect
};

#endif
