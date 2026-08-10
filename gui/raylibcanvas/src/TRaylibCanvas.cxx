// Author: Sergey Linev, GSI  06/08/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TRaylibCanvas.h"
#include "TRaylibPadPainter.h"

#include "TSystem.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TClass.h"
#include "TError.h"
#include "TTimer.h"

#include <raylib.h>
#include <raymath.h>
#include <iostream>
#include <mutex>

#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

using namespace ROOT::Experimental;

// ─── Shared window state ──────────────────────────────────────────────

std::atomic<bool> TRaylibCanvas::sWindowReady{false};
std::atomic<int> TRaylibCanvas::sActiveCanvasCount{0};

static bool sRaylibInitialized = false;
static int sWindowWidth = 0;
static int sWindowHeight = 0;

RenderTexture2D persistentCanvas; // = LoadRenderTexture(canvasWidth, canvasHeight);

const float menuBarHeight = 32.0f;
const float statusBarHeight = 24.0f;

// UI Interaction Tracking variables
bool fileDropdownOpen = false;
int fileMenuSelection = 0; // Keeps track of active dropdown choices
const char* statusMessage = "System Status: Ready";

class TRaylibEventsTimer : public TTimer {

public:
   TRaylibEventsTimer(Long_t milliSec, Bool_t mode) : TTimer(milliSec, mode) {}

   /// used to send control messages to clients
   void Timeout() override
   {
      if (!::IsWindowReady())
         return;

      // Process input events
      ::PollInputEvents();

      TCanvas *canv = gPad ? gPad->GetCanvas() : nullptr;

      if (::WindowShouldClose() && canv) {
         ::CloseWindow();
         return;
      }

      if (::IsWindowResized() && canv) {
         sWindowWidth = GetScreenWidth();
         sWindowHeight = GetScreenHeight();
         canv->Resize();
         UnloadRenderTexture(persistentCanvas);
         persistentCanvas = LoadRenderTexture(sWindowWidth, sWindowHeight - menuBarHeight - statusBarHeight);
         canv->ModifiedUpdate();
      }

      ::BeginDrawing();
      ::ClearBackground(RAYWHITE);

      Rectangle sourceRect = { 0, 0, (float)persistentCanvas.texture.width, -(float)persistentCanvas.texture.height };
      Vector2 targetPos = { 0, menuBarHeight }; // Place below our top menu layout border line
      DrawTextureRec(persistentCanvas.texture, sourceRect, targetPos, WHITE);

      // Draw top layout border frame
      GuiPanel((Rectangle){ 0, 0, (float)sWindowWidth, menuBarHeight }, "");

      // Draw bottom status layout bar
      // GuiStatusBar is a native control built for standard message layouts
      GuiStatusBar((Rectangle){ 0, (float)sWindowHeight - statusBarHeight, (float)sWindowWidth, statusBarHeight }, statusMessage);

      // RENDER LAYER C: Interactive UI Components
      // Top Left Menu Text/Label Button
      //if (GuiLabelButton((Rectangle){ 10, 4, 60, 24 }, "#01# File")) {
      //   fileDropdownOpen = true; // Toggle dropdown view state
      //}

      // Secondary Menu Bar buttons can go here horizontally
      if (GuiLabelButton((Rectangle){ 80, 4, 60, 24 }, "#11# Edit")) {
         statusMessage = "Status: Edit Menu Activated";
      }
      if (GuiLabelButton((Rectangle){ 150, 4, 60, 24 }, "#195# Help")) {
         statusMessage = "Status: Showing Help Manual";
      }

      const char* dropdownOptions = "File;New;Open;Save;Exit";
      // GuiDropdownBox tracks focus selections and updates internal states
      if (GuiDropdownBox( (Rectangle){ 10, 4, 90, 20 }, dropdownOptions, &fileMenuSelection, fileDropdownOpen)) {
         fileDropdownOpen = !fileDropdownOpen; // Hide dropdown overlay on action complete
         switch (fileMenuSelection) {
            case 0: statusMessage = "Action: Creating New Project..."; break;
            case 1: statusMessage = "Action: Loading Project Files..."; break;
            case 2: statusMessage = "Action: Changes Saved Successfully!"; break;
            case 3: statusMessage = "Action: Exit ROOT!"; break;
         }
         fileMenuSelection = 0;
      }

      ::EndDrawing();

   }
};

static TRaylibEventsTimer *sTimer = nullptr;

/*
static std::mutex sRaylibInitMutex;

// Timer callback for render loop
static void raylib_render_timer_callback(void *)
{
   // Process input events
   PollInputEvents();

   // Check if any canvas needs redrawing
   if (sRaylibInitialized && IsWindowReady()) {
      BeginDrawing();
      ClearBackground(RAYWHITE);

      // Paint all modified canvases
      auto *list = gROOT->GetListOfCanvases();
      if (list) {
         TIter next(list);
         TCanvas *canv;
         while ((canv = static_cast<TCanvas *>(next()))) {
            if (canv->IsModified() && canv->GetCanvasImp()) {
               canv->Paint();
               canv->ResetModified();
            }
         }
      }

      EndDrawing();
   }

   // Check window close
   if (WindowShouldClose()) {
      if (gApplication)
         gApplication->Terminate();
   }
}

*/

// ─── Initialize raylib window ─────────────────────────────────────────


void TRaylibCanvas::EnsureRaylibInitialized(int width, int height)
{
   // std::lock_guard<std::mutex> lock(sRaylibInitMutex);
   if (!sRaylibInitialized) {
      SetTraceLogLevel(LOG_WARNING);
      SetConfigFlags(FLAG_WINDOW_RESIZABLE | FLAG_MSAA_4X_HINT);
      ::InitWindow(width > 0 ? width : 1200, height > 0 ? height : 800, "ROOT Canvas");
      SetTargetFPS(30);
      // SetRenderHint(RENDER_HINT_ANTIALIASED_LINES, true);
      sWindowWidth = GetScreenWidth();
      sWindowHeight = GetScreenHeight();
      sRaylibInitialized = true;
      sWindowReady.store(::IsWindowReady());

      persistentCanvas = LoadRenderTexture(sWindowWidth, sWindowHeight - menuBarHeight - statusBarHeight);

      // enable timer for the raylib events processing
      sTimer = new TRaylibEventsTimer(10, kTRUE);
      sTimer->TurnOn();
   }
}


// ─── Constructor / Destructor ──────────────────────────────────────────

TRaylibCanvas::TRaylibCanvas(TCanvas *c, const char *name, Int_t x, Int_t y,
                             UInt_t width, UInt_t height)
   : TCanvasImp(c, name, x, y, width, height),
     fWindowWidth(width), fWindowHeight(height),
     fPosX(x), fPosY(y)
{
//   sActiveCanvasCount.fetch_add(1);
}

TRaylibCanvas::~TRaylibCanvas()
{
   /*
   sActiveCanvasCount.fetch_sub(1);

   // If last canvas, optionally close raylib window
   if (sActiveCanvasCount.load() <= 0 && sRaylibInitialized) {
      CloseWindow();
      sRaylibInitialized = false;
      sWindowReady.store(false);
   }
   */
}

// ─── Window Lifecycle ─────────────────────────────────────────────────

Int_t TRaylibCanvas::InitWindow()
{
   // raylib uses shared window — nothing to initialize per-canvas
   return 0;
}

void TRaylibCanvas::Close()
{
    // not implemented yet

}

void TRaylibCanvas::Show()
{
   if (!IsWindowReady() && Canvas())
      ::InitWindow(Canvas()->GetWw(), Canvas()->GetWh(), Canvas()->GetTitle());
}

// ─── Geometry ─────────────────────────────────────────────────────────

UInt_t TRaylibCanvas::GetWindowGeometry(Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
{
   if (sRaylibInitialized && IsWindowReady()) {
      sWindowWidth = GetScreenWidth();
      sWindowHeight = GetScreenHeight();

      Vector2 pos = GetWindowPosition();
      x = (int)pos.x;
      y = (int)pos.y;
      w = (UInt_t)sWindowWidth;
      h = (UInt_t)(sWindowHeight - menuBarHeight - statusBarHeight);
   } else {
      x = fPosX;
      y = fPosY;
      w = (UInt_t)fWindowWidth;
      h = (UInt_t)fWindowHeight;
   }
   return 0;
}

void TRaylibCanvas::GetCanvasGeometry(Int_t /*wid*/, UInt_t &w, UInt_t &h)
{
   if (sRaylibInitialized && IsWindowReady()) {
      sWindowWidth = GetScreenWidth();
      sWindowHeight = GetScreenHeight();
      w = (UInt_t)sWindowWidth;
      h = (UInt_t)sWindowHeight - menuBarHeight - statusBarHeight;
   } else {
      w = (UInt_t)fWindowWidth;
      h = (UInt_t)fWindowHeight;
   }
}

void TRaylibCanvas::ResizeCanvasWindow(Int_t /*wid*/)
{
   // No-op in raylib immediate mode
}

void TRaylibCanvas::UpdateDisplay(Int_t /*mode*/, Bool_t /*sleep*/)
{
}

// ─── Force Update ─────────────────────────────────────────────────────

void TRaylibCanvas::ForceUpdate()
{
   if (Canvas())
      Canvas()->Modified();
}


// ─── Position & Size ──────────────────────────────────────────────────

void TRaylibCanvas::SetWindowPosition(Int_t x, Int_t y)
{
   fPosX = x;
   fPosY = y;
   if (sRaylibInitialized && IsWindowReady()) {
      ::SetWindowPosition(x, y);
   }
}

void TRaylibCanvas::SetWindowSize(UInt_t w, UInt_t h)
{
   fWindowWidth = (int)w;
   fWindowHeight = (int)h;
   if (sRaylibInitialized && IsWindowReady()) {
      ::SetWindowSize((int)w, (int)h);
      sWindowWidth = GetScreenWidth();
      sWindowHeight = GetScreenHeight();
   }
}

void TRaylibCanvas::SetWindowTitle(const char *newTitle)
{
   fWindowTitle = newTitle ? newTitle : "";
   if (sRaylibInitialized && IsWindowReady()) {
      ::SetWindowTitle(newTitle ? newTitle : "");
   }
}

void TRaylibCanvas::SetCanvasSize(UInt_t cw, UInt_t ch)
{
   fWindowWidth = (int)cw;
   fWindowHeight = (int)ch;
}

void TRaylibCanvas::Iconify()
{
   if (sRaylibInitialized && IsWindowReady()) {
      ::MinimizeWindow();
   }
}

void TRaylibCanvas::RaiseWindow()
{
   if (sRaylibInitialized && IsWindowReady()) {
      if (::IsWindowMinimized())
         ::RestoreWindow();
   }
}

// ─── Perform Update (called from TCanvas::Update) ─────────────────────


Bool_t TRaylibCanvas::PerformUpdate(Bool_t /*async*/)
{
   if (!Canvas() || !Canvas()->IsModified() || !IsWindowReady())
      return kFALSE;

   // One can make painting directly

   BeginTextureMode(persistentCanvas);
   ClearBackground(RAYWHITE);

   // actually now perform paint
   Canvas()->Paint();

   EndTextureMode();

   // empty for now
   return kTRUE;
}

// ─── Create Pad Painter ───────────────────────────────────────────────

TVirtualPadPainter *TRaylibCanvas::CreatePadPainter()
{
   return new TRaylibPadPainter();
}

// ─── Static Factory: NewCanvas (plugin entry point) ───────────────────

TCanvasImp *TRaylibCanvas::NewCanvas(TCanvas *c, const char *name, Int_t x, Int_t y,
                                     UInt_t width, UInt_t height)
{
   // Ensure raylib is initialized
   EnsureRaylibInitialized(width, height);

   auto *imp = new TRaylibCanvas(c, name, x, y, width, height);

   if (x >= 0 && y >= 0)
      imp->SetWindowPosition(x, y);
   imp->SetWindowSize(width, height);

   TString title = c->GetTitle();
   imp->SetWindowTitle(title.Data());

   // Set internal dimensions
   c->Resize();

   return imp;
}

