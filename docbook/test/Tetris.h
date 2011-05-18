
///////////////////////////////////////////////////////////////////
//  ROOT implementation of the simple Tetris game
//  Layout and some hints were taken from Qt /examples/tetris
//
//  To run this game do the following:
//  $ root
//  root [0] gSystem.Load("Tetris")
//  root [1] Tetris t
//  <play game>
//  root [2] .q
//
//  Other ROOT fun examples: Hello, Aclock ...
///////////////////////////////////////////////////////////////////

#ifndef TETRIS_H
#define TETRIS_H

#include <TTimer.h>
#include <TCanvas.h>
#include <TGFrame.h>
#include <TRandom.h>
#include <TButton.h>
#include <TWbox.h>
#include <TText.h>
#include <TSystem.h>

class Tetris;
class TetrisBoard;

///////////////////////////////////////////////////////////////////
//  TetrisBox - the brick of the game
///////////////////////////////////////////////////////////////////
class TetrisBox : public TWbox {

private:
   Int_t    fX;     // X position(column)  in the fPad
   Int_t    fY;     // Y position(line)  in the fPad
   UInt_t   fType;  // if type==0 box is invisible (hide/show state)
   TPad    *fPad;   // pad  where box is in

public:
   TetrisBox(Int_t x=0, Int_t y=0, UInt_t type=0, TPad *pad=(TPad*)TVirtualPad::Pad());
   virtual ~TetrisBox() { }

   Int_t   GetX()                  { return fX; }
   Int_t   GetY()                  { return fY; }
   UInt_t  GetType()               { return fType; }

   void    SetX(Int_t x);
   void    SetY(Int_t y);
   void    SetXY(Int_t x, Int_t y) { SetX(x); SetY(y); }

   void    SetType(UInt_t type)    { fType=type; }
   Bool_t  IsHidden()              { return (fType==0); }
   void    Hide()                  { SetType(0); }
   void    Show()                  { SetType(1); }

   virtual void MoveOneLineDown()  { SetY(GetY()-1); }
   virtual void MoveRight()        { SetX(GetX()+1); }
   virtual void MoveLeft()         { SetX(GetX()-1); }

   void    Erase();
   void    Paint(Option_t *option="");
   void    ExecuteEvent(Int_t, Int_t, Int_t)  { return; }  // disable any actions on it
};



///////////////////////////////////////////////////////////////////
//  TetrisPiece -  tetris piece is set of up to 4 TetrisBoxes
///////////////////////////////////////////////////////////////////
class TetrisPiece {

protected:
   TetrisBox  *fBoxes[4];  // TetrisPiece  skeleton shape (up to 10 different shapes available)
   UInt_t      fType;      // type of piece
   Int_t       fY;         // current Y position (line) of the piece
   Int_t       fX;         // current X position (column) of the piece

private:
   void    Initialize(UInt_t type, TPad *pad);
   UInt_t  GetRandomType() { return (UInt_t)(gRandom->Rndm()*9)+1; }  // random 1..10
   void    HideSomeBoxes(UInt_t type);

protected:
   void    SetXx(int index,int value)      { fBoxes[index]->SetX(value+fX); }
   void    SetYy(int index,int value)      { fBoxes[index]->SetY(fY+value); }
   void    SetXxYy(int index,int x,int y)  { fBoxes[index]->SetX(x+fX);  fBoxes[index]->SetY(fY+y); }

   Int_t   GetXx(int index)                { return fBoxes[index]->GetX()-fX; }
   Int_t   GetYy(int index)                { return fBoxes[index]->GetY()-fY; }
   void    GetXxYy(int index,int &x,int&y) { x = fBoxes[index]->GetX()-fX;
                                             y = fBoxes[index]->GetY()-fY; }
public:
   TetrisPiece(TPad *pad = (TPad*)TVirtualPad::Pad())  { Initialize(GetRandomType()%11, pad); }
   TetrisPiece(UInt_t type, TPad *pad=(TPad*)TVirtualPad::Pad())  { Initialize(type%11, pad); }
   virtual ~TetrisPiece();

   void    SetX(int index,int value)      { fBoxes[index]->SetX(value); }
   void    SetY(int index,int value)      { fBoxes[index]->SetY(value); }
   void    SetXY(int index,int x,int y)   { fBoxes[index]->SetX(x);  fBoxes[index]->SetY(y); }

   UInt_t  GetType()                      { return fType; }

   virtual void SetType(UInt_t type);
   virtual void SetRandomType()           { SetType(GetRandomType()); }

   Int_t   GetX(int index)                { return fBoxes[index]->GetX(); }
   Int_t   GetY(int index)                { return fBoxes[index]->GetY(); }
   void    GetXY(int index,int &x,int&y)  { x = fBoxes[index]->GetX();
                                            y = fBoxes[index]->GetY(); }
   Int_t   GetX()                         { return fX; }
   Int_t   GetY()                         { return fY; }
   TetrisBox *GetTetrisBox(Int_t i)       { return fBoxes[i]; }

   void    SetX(Int_t x);
   void    SetY(Int_t y);
   void    SetXY(Int_t x,Int_t y);

   virtual void Hide();
   virtual void Show();

   virtual Bool_t RotateLeft();
   virtual Bool_t RotateRight();
};


typedef TetrisBox* TetrisBoxPtr;

///////////////////////////////////////////////////////////////////
//  TetrisBoard - the game board
///////////////////////////////////////////////////////////////////
class TetrisBoard : public TPad {

friend class Tetris;

private:
   Int_t   fHeight;     // board height
   Int_t   fWidth;      // board width
   Bool_t  fIsDropped;  // kTRUE when piece stopped

   TetrisBoxPtr *fBoard;   //! 2d array of pointers to Tetrisboxes,
                           // if pointer is 0 - the cell is empty
   Int_t  fFilledLines;    // number of non empty lines in pad

   void   AllAboveLinesDown(Int_t line)    // assume that line is empty
            { for (int i = line; i < fFilledLines; i++) MoveOneLineDown(i); }

   void   MoveOneLineDown(Int_t line);
   void   RemoveFullLines();
   void   RemoveLine(int line);
   Bool_t IsLineFull(int line);
   Bool_t IsLineEmpty(int line);
   void   GluePiece(TetrisPiece *piece);

   void   Clear(Option_t *option = "");
   void   Hide();
   void   Show();
   void   Print(const char *option = "") const;
   void   Print(const char *, Option_t *) { }  // removes "hiding" warning
   Bool_t IsEmptyLine(int line);
   Bool_t IsFullLine(Int_t line);

   TetrisBoxPtr &Board(Int_t x, Int_t y)    { return  fBoard[fWidth*y + x]; }

public:
   TetrisBoard(Float_t xlow, Float_t ylow, Float_t xup, Float_t yup);
   virtual ~TetrisBoard() { }

   Int_t  GetHeight()                     { return fHeight; }
   Int_t  GetWidth()                      { return fWidth; }
   Bool_t IsEmpty(Int_t x, Int_t y)       { return !Board(x,y); }
   void   SetDropped(Bool_t flag=kTRUE)   { fIsDropped=flag; }

   virtual void PaintModified();
   void   PieceDropped(TetrisPiece *piece, Int_t height);
   void   ExecuteEvent(Int_t, Int_t, Int_t)  { return; }  // disable any actions on it
};


///////////////////////////////////////////////////////////////////
//  CurrentPiece =  TetrisPiece + TTimer = live TetrisPiece
///////////////////////////////////////////////////////////////////
class CurrentPiece : public TetrisPiece, public TTimer {

private:
   TetrisBoard  *fBoard;          // tetris board

protected:
   Bool_t  CanPosition();
   Bool_t  CanMoveTo(int xPosition, int line);
   void    MoveTo(int xPosition,int line);
   void    Erase();

public:
   CurrentPiece(UInt_t type, TetrisBoard* board);
   ~CurrentPiece() { }

   Bool_t  MoveLeft(Int_t steps = 1);
   Bool_t  MoveRight(Int_t steps = 1);
   Bool_t  RotateLeft();
   Bool_t  RotateRight();
   Bool_t  DropDown();
   Bool_t  OneLineDown();
   Bool_t  Notify();
   void    SetSpeed();
   void    Paint(Option_t *option="");
   void    ExecuteEvent(Int_t, Int_t, Int_t)  { return; }  // disable any actions on it
};



///////////////////////////////////////////////////////////////////
//  NextPiecePad
//  used to show next piece.
///////////////////////////////////////////////////////////////////
class NextPiecePad : public TPad {

private:
   TetrisPiece *fPiece;   // next piece

public:
   NextPiecePad(Float_t xlow, Float_t ylow, Float_t xup, Float_t yup);
   ~NextPiecePad() { }

   void   NewPiece() { fPiece->SetRandomType(); fPiece->Show(); Modified(kTRUE); }
   void   Hide()     { fPiece->Hide(); Modified(kTRUE); }
   void   Show()     { fPiece->Show(); Modified(kTRUE); }

   TetrisPiece  *GetPiece() { return fPiece; }
   void ExecuteEvent(Int_t, Int_t, Int_t)  { return; }  // disable any actions on it
};



///////////////////////////////////////////////////////////////////
//  QuitButton
//  ExecuteEvent mehtod is overloaded.
///////////////////////////////////////////////////////////////////
class QuitButton : public TButton {

public:
   QuitButton(Float_t xlow, Float_t ylow, Float_t xup, Float_t yup);
   ~QuitButton() { }

   void ExecuteEvent(Int_t event, Int_t px, Int_t py);
};


///////////////////////////////////////////////////////////////////
//  PauseButton - push button
//  ExecuteEvent mehtod used to pause the game
///////////////////////////////////////////////////////////////////
class PauseButton : public TButton {

private:
   Bool_t fPressed;     // button press state

public:
   PauseButton(Float_t xlow, Float_t ylow, Float_t xup, Float_t yup);
   ~PauseButton() { }

   void SetPressed(Bool_t state) {
      fPressed = state;
      state ? SetBorderMode(-1) : SetBorderMode(1);
      Modified(kTRUE);
      Update();
   }

   Bool_t   IsPressed()               { return fPressed; }
   void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
};


///////////////////////////////////////////////////////////////////
//  NewGameButton
//  ExecuteEvent mehtod used to start new game
///////////////////////////////////////////////////////////////////
class NewGameButton : public TButton {

private:
   Bool_t   fPressed;   // button press state

public:
   NewGameButton(Float_t xlow, Float_t ylow, Float_t xup, Float_t yup);
   ~NewGameButton() { }

   void SetPressed(Bool_t state) {
      fPressed = state;
      state ? SetBorderMode(-1) : SetBorderMode(1);
      Modified(kTRUE);
      Update();
   }

   Bool_t   IsPressed()               { return fPressed; }
   void ExecuteEvent(Int_t event, Int_t px, Int_t py);
};


///////////////////////////////////////////////////////////////////
//  InfoPad - used to display digital info
///////////////////////////////////////////////////////////////////
class InfoPad : public TPad, public TAttText {

protected:
   UInt_t   fValue;   // value  to be displayed

public:
   InfoPad(const char *title="",Float_t xlow=0, Float_t ylow=0, Float_t xup=0, Float_t yup=0);
   virtual ~InfoPad() { }

   UInt_t  GetValue()                  { return fValue; }
   void    SetValue(Int_t value)       { fValue = value; Modified(kTRUE); }
   void    Reset(Option_t * = "")      { SetValue(0); }
   virtual void AddValue(Int_t addValue=1) { fValue = fValue+addValue; Modified(kTRUE); }

   virtual void PaintModified();
   void ExecuteEvent(Int_t, Int_t, Int_t)  { return; }  // disable any actions on it
};



///////////////////////////////////////////////////////////////////
//  KeyHandler = virtual frame
//  used to catch and handle key events in Tetris canvas
///////////////////////////////////////////////////////////////////
class KeyHandler : public TGFrame {

public:
   KeyHandler();
   ~KeyHandler();

   Bool_t HandleKey(Event_t *event);    // handler of the key events
};


///////////////////////////////////////////////////////////////////
//  UpdateLevelTimer used to periodically update game level
///////////////////////////////////////////////////////////////////
class UpdateLevelTimer : public TTimer {

public:
   UpdateLevelTimer(ULong_t time);
   ~UpdateLevelTimer() { }

   Bool_t Notify();
};


//////////////////////////////////////////////////////////////////
//  Tetris = Game manager
//////////////////////////////////////////////////////////////////
class Tetris : public TCanvas {

friend class KeyHandler;
friend class TetrisBoard;
friend class UpdateLevelTimer;

private:
   CurrentPiece     *fCurrentPiece;       // live tetris piece
   TetrisBoard      *fBoard;              // pad were everything is going on
   NextPiecePad     *fNextPiece;          // pad which show next piece
   InfoPad          *fLinesRemoved;       // number of removed lines
   InfoPad          *fLevel;              // game level
   InfoPad          *fScore;              // game score
   NewGameButton    *fNewGame;            // clicking on button initiates new game
   QuitButton       *fQuit;               // clicking on button makes game over
   PauseButton      *fPause;              // pause/continue button
   KeyHandler       *fKeyHandler;         // handler for arrow keys

   Int_t             fPiecesDropped;      // number of pieces dropped
   UpdateLevelTimer *fUpdateLevelTimer;   // periodically updates game level

protected:
   void   SetFixedSize();
   void   CreateNewPiece();
   void   UpdatePiecesDropped()             { fPiecesDropped++; }
   void   UpdateLinesRemoved()              { fLinesRemoved->AddValue(1); }
   void   UpdateScore(Int_t add_value)      { fScore->AddValue(add_value); }
   void   UpdateLevel()                     { if (GetLevel()<10) fLevel->AddValue(1); }
   void   PrintHelpInfo();

   virtual  void  MoveLeft();
   virtual  void  MoveRight();
   virtual  void  DropDown();
   virtual  void  RotateRight();
   virtual  void  RotateLeft();

public:
   Tetris();
   virtual ~Tetris() { delete fKeyHandler; }

   Int_t  GetLevel()           { return fLevel->GetValue(); }
   Int_t  GetLinesRemoved()    { return fLinesRemoved->GetValue(); }
   Int_t  GetPiecesDropped()   { return fPiecesDropped; }
   Int_t  GetScore()           { return fScore->GetValue(); }

   Bool_t IsGameOn()           { return fNewGame->IsPressed(); }
   Bool_t IsPaused()           { return fPause->IsPressed(); }
   Bool_t IsWaiting()          { return kFALSE; }

   void   SetLevel(int level);
   void   Quit();
   void   Pause();
   void   Continue();
   void   NewGame();
   void   StopGame();

   ClassDef(Tetris,0)  // ROOT implementation of the Tetris game
};

#endif   // TETRIS_H
