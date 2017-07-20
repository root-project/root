// Copyright (c) 2013 The Chromium Embedded Framework Authors. All rights
// reserved. Use of this source code is governed by a BSD-style license that
// can be found in the LICENSE file.

#include "include/cef_app.h"

#include <unistd.h>

#include "include/base/cef_logging.h"


class ROOTV8Handler : public CefV8Handler {
public:
   ROOTV8Handler() {}

   virtual bool Execute(const CefString &name, CefRefPtr<CefV8Value> object, const CefV8ValueList &arguments,
                        CefRefPtr<CefV8Value> &retval, CefString &exception) OVERRIDE
   {
      if (name == "ROOT_BATCH_FUNC") {
         // Return my string value.
         retval = CefV8Value::CreateString("My Value!");
         printf("CALLING ROOT_BATCH_FUNC\n");
         return true;
      }

      // Function does not exist.
      return false;
   }

   // Provide the reference counting implementation for this class.
   IMPLEMENT_REFCOUNTING(ROOTV8Handler);
};




// Implement application-level callbacks for the browser process.
class MyRendererProcessApp : public CefApp, public CefRenderProcessHandler {
protected:
public:
   MyRendererProcessApp() {}
   virtual ~MyRendererProcessApp() {}

   // CefApp methods:
   virtual CefRefPtr<CefRenderProcessHandler> GetRenderProcessHandler() OVERRIDE { return this; }

   // CefRenderProcessHandler methods
   virtual void OnContextCreated(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
                                 CefRefPtr<CefV8Context> context) OVERRIDE
   {
      printf("MyRendererProcessApp::OnContextCreated\n");

      // Retrieve the context's window object.
      CefRefPtr<CefV8Value> object = context->GetGlobal();

      // Create a new V8 string value. See the "Basic JS Types" section below.
      CefRefPtr<CefV8Value> str = CefV8Value::CreateString("My Value!");

      // Add the string to the window object as "window.myval". See the "JS Objects" section below.
      object->SetValue("ROOT_BATCH_FLAG", str, V8_PROPERTY_ATTRIBUTE_NONE);

      printf("ADD BATCH FALG\n");

      CefRefPtr<CefV8Handler> handler = new ROOTV8Handler;
      CefRefPtr<CefV8Value> func = CefV8Value::CreateFunction("ROOT_BATCH_FUNC", handler);

      // Add the string to the window object as "window.myval". See the "JS Objects" section below.
      object->SetValue("ROOT_BATCH_FUNC", func, V8_PROPERTY_ATTRIBUTE_NONE);

      printf("ADD BATCH FUNC\n");
   }

private:
   // Include the default reference counting implementation.
   IMPLEMENT_REFCOUNTING(MyRendererProcessApp);
   DISALLOW_COPY_AND_ASSIGN(MyRendererProcessApp);
};



// Entry point function for all processes.
int main(int argc, char *argv[])
{
   // Provide CEF with command-line arguments.
   CefMainArgs main_args(argc, argv);

   printf("Starting CEF_MAIN\n");

   CefRefPtr<CefApp> app = new MyRendererProcessApp();

   // CEF applications have multiple sub-processes (render, plugin, GPU, etc)
   // that share the same executable. This function checks the command-line and,
   // if this is a sub-process, executes the appropriate logic.
   int exit_code = CefExecuteProcess(main_args, app, NULL);
   if (exit_code >= 0) {
      // The sub-process has completed so return here.
      return exit_code;
   }

   printf("do nothing\n");

   return 0;
}
