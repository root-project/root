// @(#)root/gl:$Id$
// Author:  Thomas Keck  13/02/2015

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLOculusViewer.h"
#include "TGLIncludes.h"
#include "TGLStopwatch.h"
#include "TGLRnrCtx.h"
#include "TGLSelectBuffer.h"
#include "TGLLightSet.h"
#include "TGLManipSet.h"
#include "TGLCameraOverlay.h"
#include "TGLAutoRotator.h"

#include "TGLScenePad.h"
#include "TGLLogicalShape.h"
#include "TGLPhysicalShape.h"
#include "TGLObject.h"
#include "TGLStopwatch.h"
#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"

#include "TGLOutput.h"

#include "TROOT.h"
#include "TVirtualMutex.h"

#include "TVirtualPad.h" // Remove when pad removed - use signal
#include "TVirtualX.h"

#include "TMath.h"
#include "TColor.h"
#include "TError.h"
#include "TClass.h"
#include "TROOT.h"
#include "TEnv.h"

// For event type translation ExecuteEvent
#include "Buttons.h"
#include "GuiTypes.h"

#include "TVirtualGL.h"

#include "TGLWidget.h"
#include "TGLFBO.h"
#include "TGLViewerEditor.h"
#include "TGedEditor.h"
#include "TGLPShapeObj.h"

#include "KeySymbols.h"
#include "TContextMenu.h"
#include "TImage.h"

#include "../../LibOVR/Include/OVR.h"
#include "../../LibOVR/Src/OVR_CAPI_GL.h"

#include <stdexcept>
#include <iostream>

ClassImp(TGLSAOculusViewer);
ClassImp(TGLEmbeddedOculusViewer);

// OculusInterface takes care of all interaction with the oculus rift SDK
class OculusInterface {

  public:
    OculusInterface(TGLCamera *_fCamera) : fCamera(_fCamera) {

      // Initialize Oculus rift SDK
      ovr_Initialize();

      // Create head-mounted-display instance, and check if its present
      hmd = ovrHmd_Create(0);
      if (not hmd) {
        throw std::runtime_error("Couldn't find oculus device");
      } else {
        std::cout << "Found oculus rift: " 
                  << hmd->ProductName << " "
                  << hmd->Manufacturer << " "
                  << hmd->DisplayDeviceName << std::endl;
      }

      // Setup head and position tracking, by stating which tracking capabilities are supported (second argument)
      // and which are required (third argument: 0 means none are required at the moment)
      if( not ovrHmd_ConfigureTracking(hmd, ovrTrackingCap_Orientation |
                                       ovrTrackingCap_MagYawCorrection |
                                       ovrTrackingCap_Position, 0) ) {
        throw std::runtime_error("Required tracking capabilities are not supported");
      }
      

      // The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.
      FramebufferName = 0;
      glGenFramebuffers(1, &FramebufferName);
      glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

      // The texture we're going to render to
      glGenTextures(1, &renderedTexture);
        
      // "Bind" the newly created texture : all future texture functions will modify this texture
      glBindTexture(GL_TEXTURE_2D, renderedTexture);
         
      // Give an empty image to OpenGL ( the last "0" )
      // Texture size is arbitrary but large enough to hold the viewport areas for both eyes
      glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, 4096, 2048, 0,GL_RGB, GL_UNSIGNED_BYTE, 0);
          
      // Poor filtering. Needed !
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

      // The depth buffer
      GLuint depthrenderbuffer;
      glGenRenderbuffers(1, &depthrenderbuffer);
      glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
      glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 4096, 2048);
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

      // Set "renderedTexture" as our colour attachement #0
      glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);

      // Set the list of draw buffers.
      GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
      glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers
      
      // Always check that our framebuffer is ok
      if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        throw std::runtime_error("OpenGL framebuffer is incomplete");
      }
      
      eyeTextures[0].OGL.Header.API = ovrRenderAPI_OpenGL;
      eyeTextures[0].OGL.Header.TextureSize = OVR::Sizei(4096, 2048);
      eyeTextures[0].OGL.TexId = renderedTexture; 
      eyeTextures[1] = eyeTextures[0];

    }

    ~OculusInterface() {

      // Destroy head-mounted-display instance
      ovrHmd_Destroy(hmd);

      // Shutdown SDK
      ovr_Shutdown();
    }

    void update(ovrEyeType eye) {

      int ieye = static_cast<int>(eye);
      
      TGLRect &vp = fCamera->RefViewport();
      
      // Fix perspective, which is fucked up by the TGLPerspectiveCamera in PreRender
      glMatrixMode(GL_PROJECTION);
      glLoadIdentity();
      gluPerspective(reinterpret_cast<TGLPerspectiveCamera*>(fCamera)->GetFOV(), vp.Aspect(), 1.0, 1000.0);
      
      // Load Modelview and reset previous camera position from PreRender by loading identity
      glMatrixMode(GL_MODELVIEW);
      glLoadIdentity();

      // Render to our framebuffer
      glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
      glViewport(ieye*vp.Width(), 0, vp.Width(), vp.Height());
      
      eyeTextures[ieye].OGL.Header.RenderViewport.Pos.x = ieye*vp.Width();
      eyeTextures[ieye].OGL.Header.RenderViewport.Pos.y = 0;
      eyeTextures[ieye].OGL.Header.RenderViewport.Size = OVR::Sizei(vp.Width(), vp.Height());

      // Now look at point given by head position
      headPose[ieye] = ovrHmd_GetHmdPosePerEye(hmd, eye); 
      OVR::Posef pose = headPose[ieye]; 
      OVR::Vector3f up = pose.Rotation.Rotate(OVR::Vector3f(0.0f, 1.0f, 0.0f));
      OVR::Vector3f center = pose.Rotation.Rotate(OVR::Vector3f(0.0f, 0.0f, -1.0f));
      OVR::Vector3f pos = pose.Translation;
      gluLookAt(pos[0]*200, pos[1]*200, pos[2]*200,
                center[0]*200, center[1]*200, center[2]*200,
                up[0], up[1], up[2]);

    }

    void begin() {
      frameTiming = ovrHmd_BeginFrame(hmd, 0);

      ovrHSWDisplayState hswDisplayState;
      ovrHmd_GetHSWDisplayState(hmd, &hswDisplayState);
      if (hswDisplayState.Displayed) {
        ovrHmd_DismissHSWDisplay(hmd);
      }
    }
    
    void end() {

      static ovrGLConfig cfg;
      TGLRect &vp = fCamera->RefViewport();
      cfg.OGL.Header.API            = ovrRenderAPI_OpenGL;
      cfg.OGL.Header.BackBufferSize = OVR::Sizei(vp.Width(), vp.Height());
      cfg.OGL.Header.Multisample    = 0;
      cfg.OGL.Disp                  = NULL;

      unsigned distortionCaps = ovrDistortionCap_Chromatic;
      distortionCaps |= ovrDistortionCap_Vignette;
      // distortionCaps |= ovrDistortionCap_SRGB;
      // distortionCaps |= ovrDistortionCap_Overdrive;
      // distortionCaps |= ovrDistortionCap_TimeWarp;
      // distortionCaps |= ovrDistortionCap_ProfileNoTimewarpSpinWaits;
      // distortionCaps |= ovrDistortionCap_HqDistortion;
      // distortionCaps |= ovrDistortionCap_LinuxDevFullscreen;

      ovrFovPort eyeFov[2];
      eyeFov[0] = hmd->DefaultEyeFov[0];
      eyeFov[1] = hmd->DefaultEyeFov[1];

      ovrEyeRenderDesc EyeRenderDesc[2];
      if( not ovrHmd_ConfigureRendering(hmd, &cfg.Config, distortionCaps, eyeFov, EyeRenderDesc) ) {
        throw std::runtime_error("Configure OVR Rendering failed");
      }

      ovrHmd_EndFrame(hmd, headPose, reinterpret_cast<ovrTexture*>(eyeTextures));
    }

  private:
    GLuint FramebufferName;
    GLuint renderedTexture;
    ovrHmd hmd;
    ovrFrameTiming frameTiming;
    ovrPosef headPose[2];
    ovrGLTexture eyeTextures[2];
    TGLCamera *fCamera;

};

void TGLEmbeddedOculusViewer::DoDrawStereo(Bool_t swap_buffers)
{
   // Create static instance of oculus interface, which takes care of everything
   // e.g. setup new window, head-tracking and interaction with oculus rift SDK
   static OculusInterface oi(fCamera);
   
   MakeCurrent();

   if (!fIsPrinting) PreDraw();
   PreRender();

   oi.begin();
 
   fRnrCtx->StartStopwatch();

   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   oi.update(ovrEye_Left);
   Render();
   glPopMatrix();

   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   oi.update(ovrEye_Right);
   Render();
   glPopMatrix();

   fRnrCtx->StopStopwatch();
   
   oi.end();

   PostRender();
   PostDraw();

   if (swap_buffers)
   {
      SwapBuffers();
   }

   // Redraw every 10 ms
   fRedrawTimer->RequestDraw(10, fLOD);

}

void TGLSAOculusViewer::DoDrawStereo(Bool_t swap_buffers)
{
   // Create static instance of oculus interface, which takes care of everything
   // e.g. setup new window, head-tracking and interaction with oculus rift SDK
   static OculusInterface oi(fCamera);
   
   MakeCurrent();

   if (!fIsPrinting) PreDraw();
   PreRender();

   oi.begin();
 
   fRnrCtx->StartStopwatch();

   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   oi.update(ovrEye_Left);
   Render();
   glPopMatrix();

   glMatrixMode(GL_MODELVIEW);
   glPushMatrix();
   oi.update(ovrEye_Right);
   Render();
   glPopMatrix();

   fRnrCtx->StopStopwatch();
   
   oi.end();

   PostRender();
   PostDraw();

   if (swap_buffers)
   {
      SwapBuffers();
   }

   // Redraw every 10 ms
   fRedrawTimer->RequestDraw(10, fLOD);

}

