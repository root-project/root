// Author: Rene Brun   02/09/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTask
#define ROOT_TTask


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTask                                                                //
//                                                                      //
// Base class for recursive execution of tasks.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TBrowser;
   
class TTask : public TNamed {

public:

  TTask();
  TTask(const char* name, const char *title);
  TTask(const TTask &task);
  virtual ~TTask();

  // Main functions
          void  Abort();  // *MENU*
          void  Add(TTask *task) {fTasks->Add(task);}
  virtual void  Browse(TBrowser *b);
          void  CleanTasks();
  virtual void  Clear(Option_t *option="");
  virtual void  Continue(); // *MENU*
  virtual void  Execute(Option_t *option);
          void  ExecuteTask(Option_t *option="0");  // *MENU*
          void  ExecuteTasks(Option_t *option);
          Int_t GetBreakin() {return fBreakin;}
          Int_t GetBreakout() {return fBreakout;}
         Bool_t IsFolder() const { return kTRUE; }
  virtual void  ls(Option_t *option="*");  // *MENU*
          void  SetBreakin(Int_t breakin=1)   {fBreakin = breakin;} // *TOGGLE*
          void  SetBreakout(Int_t breakout=1) {fBreakout=breakout;} // *TOGGLE*
  TList        *GetListOfTasks() {return fTasks;}
 
  // Data members
protected:      
  
  TList        *fTasks;        //List of Tasks
  static TTask *fBeginTask;    //pointer to task initiator
  static TTask *fBreakPoint;   //pointer to current break point
  TString       fOption;       //Option specified in ExecuteTask
  Int_t         fBreakin;      //=1 if a break point set at task extry
  Int_t         fBreakout;     //=1 if a break point set at task exit
  Bool_t        fHasExecuted;  //True if task has executed
    
  ClassDef(TTask,1)  //Base class for tasks
};
#endif
