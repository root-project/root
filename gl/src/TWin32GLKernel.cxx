// @(#)root/gl:$Name$:$Id$
// Author: Valery Fine(fine@vxcern.cern.ch)   05/03/97

#include "TWin32GLKernel.h"


#ifndef ROOT_TGWin32
#include "TGWin32.h"
#endif

#ifndef ROOT_TGWin32Command
#include "TGWin32Command.h"
#endif

#include "TWin32GLViewerImp.h"

// Force creation of TWin32GLKernel when shared library will be loaded.
static TWin32GLKernel gGLKernelCreator;


//______________________________________________________________________________
//*-*-*-*-*-*-*-*-*-*-*-* T G L K e r n e l class *-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================
//*-*
//*-*   TWin32GLKernel class defines the interface for OpenGL command and utilities
//*-*   Those are defined with GL/gl and GL/glu include directories
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

//*-* Macros to call the Callback methods via Window thread:
//*-*
#define CallMethodThread(_function,_p1,_p2,_p3)    \
  else                                             \
  {                                                \
      TWin32SendWaitClass code(this,(UInt_t)k##_function,(UInt_t)(_p1),(UInt_t)(_p2),(UInt_t)(_p3)); \
      ExecWindowThread(&code);                     \
      code.Wait();                                 \
  }
#define  ReturnMethodThread(_type,_function,_p1,_p2) \
  else                                               \
  {                                                  \
      _type _local;                                  \
      TWin32SendWaitClass code(this,(UInt_t)k##_function,(UInt_t)(_p1),(UInt_t)(_p2),(UInt_t)(&_local)); \
      ExecWindowThread(&code);                       \
      code.Wait();                                   \
      return _local;                                 \
  }


#define CallWindowMethod3(_function,_p1,_p2,_p3)     \
  if ( ((TGWin32 *)gVirtualX)->IsCmdThread())             \
  {TGLKernel::_function(_p1,_p2,_p3);}               \
    CallMethodThread(_function,_p1,_p2,_p3)

//*-*
#define CallWindowMethod2(_function,_p1,_p2)         \
  if ( ((TGWin32 *)gVirtualX)->IsCmdThread())             \
  {TGLKernel::_function(_p1,_p2);}                   \
    CallMethodThread(_function,_p1,_p2,0)
//*-*
#define CallWindowMethod1(_function,_p1)             \
  if ( ((TGWin32 *)gVirtualX)->IsCmdThread())             \
  {TGLKernel::_function(_p1);}                       \
    CallMethodThread(_function,_p1,0,0)
//*-*
#define CallWindowMethod(_function)                  \
  if ( ((TGWin32 *)gVirtualX)->IsCmdThread())             \
  {TGLKernel::_function();}                          \
    CallMethodThread(_function,0,0,0)

//*-*
#define ReturnWindowMethod2(_type,_function,_p1,_p2) \
  if ( ((TGWin32 *)gVirtualX)->IsCmdThread())             \
  {return TGLKernel::_function(_p1,_p2);}            \
    ReturnMethodThread(_type,_function,_p1,_p2)
//*-*
#define ReturnWindowMethod1(_type,_function,_p1)     \
  if ( ((TGWin32 *)gVirtualX)->IsCmdThread())             \
  {return TGLKernel::_function(_p1);}                \
    ReturnMethodThread(_type,_function,_p1,0)
//*-*
#define ReturnWindowMethod(_type,_function)          \
  if ( ((TGWin32 *)gVirtualX)->IsCmdThread())             \
  {return TGLKernel::_function();}                   \
    ReturnMethodThread(_type,_function,0,0)

//______________________________________________________________________________
TWin32GLKernel::TWin32GLKernel() { gVirtualGL = this;}

//______________________________________________________________________________
TWin32GLKernel::~TWin32GLKernel() { gVirtualGL = 0; }


//______________________________________________________________________________
void TWin32GLKernel::ClearColor(Int_t color)
{
    CallWindowMethod1(ClearColor,color);
}
//______________________________________________________________________________
void TWin32GLKernel::ClearGLColor(Float_t red, Float_t green, Float_t blue, Float_t alpha)
{
    Float_t colors[4] = {red,green,blue,alpha};
    CallWindowMethod1(ClearGLColor,colors);
}

//______________________________________________________________________________
void TWin32GLKernel::ClearGL(UInt_t bufbits ) {
    CallWindowMethod1(ClearGL,bufbits);
}

//______________________________________________________________________________
TGLViewerImp *TWin32GLKernel::CreateGLViewerImp(TPadOpenGLView *c, const char *title, UInt_t width, UInt_t height)
{
//*-* Create the OpenGL viewer imp for TGuiFactory object
    return new TWin32GLViewerImp(c,title,width,height);
}
//______________________________________________________________________________
void TWin32GLKernel::DisableGL(EG3D2GLmode mode)
{
    CallWindowMethod1(DisableGL,mode);
}
//______________________________________________________________________________
void TWin32GLKernel::EnableGL(EG3D2GLmode mode)
{
    CallWindowMethod1(EnableGL,mode);
}

//______________________________________________________________________________
void TWin32GLKernel::FlushGL(){ CallWindowMethod(FlushGL);}

//______________________________________________________________________________
void TWin32GLKernel::FrontGLFace(EG3D2GLmode faceflag){
        CallWindowMethod1(FrontGLFace,faceflag);
}
//______________________________________________________________________________
void TWin32GLKernel::NewGLList(UInt_t ilist, EG3D2GLmode mode)
{
    CallWindowMethod2(NewGLList,ilist,mode);
}

//______________________________________________________________________________
void TWin32GLKernel::NewGLModelView(Int_t ilist)
{
    CallWindowMethod1(NewGLModelView,ilist);
}

//______________________________________________________________________________
void TWin32GLKernel::GetGL(EG3D2GLmode mode, Bool_t *params)
{
    CallWindowMethod3(GetGL,mode,params,kBoolType);
}

//______________________________________________________________________________
void TWin32GLKernel::GetGL(EG3D2GLmode mode, Double_t *params)
{
    CallWindowMethod3(GetGL,mode,params,kBoolType);
}

//______________________________________________________________________________
void TWin32GLKernel::GetGL(EG3D2GLmode mode, Float_t *params)
{
    CallWindowMethod3(GetGL,mode,params,kBoolType);
}

//______________________________________________________________________________
void TWin32GLKernel::GetGL(EG3D2GLmode mode, Int_t *params)
{
    CallWindowMethod3(GetGL,mode,params,kBoolType);
}

//______________________________________________________________________________
Int_t TWin32GLKernel::GetGLError()
{
    ReturnWindowMethod(Int_t,GetGLError);
}

//______________________________________________________________________________
void TWin32GLKernel::EndGLList() {CallWindowMethod(EndGLList);}

//______________________________________________________________________________
void TWin32GLKernel::AddRotation(Double_t *rotmatrix,Double_t *extraangles)
{
    CallWindowMethod2(AddRotation,rotmatrix,extraangles);
}

//______________________________________________________________________________
void TWin32GLKernel::BeginGLCmd(EG3D2GLmode mode)
{
    CallWindowMethod1(BeginGLCmd,mode);
}

//______________________________________________________________________________
void TWin32GLKernel::EndGLCmd() {CallWindowMethod(EndGLCmd); }

//______________________________________________________________________________
void TWin32GLKernel::PaintGLPoints(Int_t n, Float_t *p, Option_t *option)
{
    CallWindowMethod3(PaintGLPoints,n,p,option);
}

//______________________________________________________________________________
void TWin32GLKernel::PaintGLPointsObject(const TPoints3DABC *points, Option_t *option)
{  CallWindowMethod2(PaintGLPointsObject,points,option); }

//______________________________________________________________________________
void TWin32GLKernel::PushGLMatrix() {CallWindowMethod(PushGLMatrix);}

//______________________________________________________________________________
void TWin32GLKernel::PolygonGLMode(EG3D2GLmode face , EG3D2GLmode mode)
{  CallWindowMethod2(PolygonGLMode,face,mode); }

//______________________________________________________________________________
void TWin32GLKernel::PopGLMatrix() {CallWindowMethod(PopGLMatrix);}

//______________________________________________________________________________
void TWin32GLKernel::RotateGL(Double_t angle, Double_t x,Double_t y,Double_t z)
{
//*-* The RotateGL function computes a matrix that performs a counterclockwise
//*-* rotation of angle degrees about the vector from the origin through
//*-* the point (x, y, z).
    Double_t axyz[] = {angle,x,y,z};
    CallWindowMethod2(RotateGL,axyz,1);
}

//______________________________________________________________________________
void TWin32GLKernel::RotateGL(Double_t Theta, Double_t Phi, Double_t Psi)
{
//*-* Double_t Theta   - polar angle for the axis x`
//*-* Double_t Phi     - azimutal angle for the axis x`
//*-* Double_t Psi     - azimutal angle for the axis y`
    Double_t angles[] = {Theta,Phi,Psi};
    CallWindowMethod2(RotateGL,angles,0);
}
//______________________________________________________________________________
void TWin32GLKernel::TranslateGL(Double_t x,Double_t y,Double_t z)
{
    Double_t xyz[3] = {x,y,z};
    CallWindowMethod1(TranslateGL,xyz);
}

//______________________________________________________________________________
void TWin32GLKernel::MultGLMatrix(Double_t *mat)
{
    CallWindowMethod1(MultGLMatrix,mat);
}

//______________________________________________________________________________
void TWin32GLKernel::SetGLColor(Float_t *rgb)
{
    CallWindowMethod1(SetGLColor,rgb);
}

//______________________________________________________________________________
void TWin32GLKernel::SetGLVertex(Float_t *vertex)
{
    CallWindowMethod1(SetGLVertex,vertex);
}
//______________________________________________________________________________
void TWin32GLKernel::SetGLColorIndex(Int_t color)
{
    CallWindowMethod1(SetGLColorIndex,color);
}

//______________________________________________________________________________
void TWin32GLKernel::SetGLLineWidth(Float_t width)
{
    CallWindowMethod1(SetGLLineWidth,width);
}

//______________________________________________________________________________
void TWin32GLKernel::SetGLPointSize(Float_t size)
{
    CallWindowMethod1(SetGLPointSize,size);
}
//______________________________________________________________________________
void TWin32GLKernel::SetStack(Double_t *matrix)
{
    CallWindowMethod1(SetStack,matrix);
}
//______________________________________________________________________________
void TWin32GLKernel::SetRootLight(Bool_t flag)
{
    CallWindowMethod1(SetRootLight,flag);
}

//______________________________________________________________________________
void TWin32GLKernel::ShadeGLModel(EG3D2GLmode mode)
{
    CallWindowMethod1(ShadeGLModel,mode);
}

//______________________________________________________________________________
void TWin32GLKernel::DeleteGLLists(Int_t ilist, Int_t range)
{
     CallWindowMethod2(DeleteGLLists,ilist,range);
}

//______________________________________________________________________________
Int_t TWin32GLKernel::CreateGLLists(Int_t range)
{
    ReturnWindowMethod1(Int_t,CreateGLLists,range);
}

//______________________________________________________________________________
void TWin32GLKernel::RunGLList(Int_t list)
{
    CallWindowMethod1(RunGLList,list);
}

//______________________________________________________________________________
void TWin32GLKernel::NewProjectionView(Double_t min[],Double_t max[],Bool_t perspective)
{
    CallWindowMethod3(NewProjectionView,min,max,perspective);
}

//______________________________________________________________________________
void TWin32GLKernel::NewModelView(Double_t *angles,Double_t *delta )
{
    CallWindowMethod2(NewModelView,angles,delta);
}
//______________________________________________________________________________
void TWin32GLKernel::PaintPolyLine(Int_t n, Float_t *p, Option_t *option)
{
    CallWindowMethod3(PaintPolyLine,n,p,option);
}

//______________________________________________________________________________
void TWin32GLKernel::PaintBrik(Float_t vertex[24])
{
    CallWindowMethod1(PaintBrik,vertex);
}

//______________________________________________________________________________
void TWin32GLKernel::PaintCone(Float_t *vertex,Int_t ndiv,Int_t nstacks)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-* Input:
//*-* -----
//*-*  vertex  - array of the 3d coordinates
//*-*  ndiv    - number of division
//*-*            < 0 means the shape is segmented
//*-*            > 0 means the shape is closed
//*-*  nstacks -  number of stack sections
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    CallWindowMethod3(PaintCone,vertex,ndiv,nstacks);
}

//______________________________________________________________________________
void TWin32GLKernel::SetLineAttr(Color_t color, Int_t width)
{
    CallWindowMethod2(SetLineAttr,color,width);
}

//______________________________________________________________________________
void TWin32GLKernel::UpdateMatrix(Double_t *translate, Double_t *rotate, Bool_t isreflection)
{
    CallWindowMethod3(UpdateMatrix,translate,rotate,isreflection);
}
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*   Callback methods:
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//______________________________________________________________________________
void TWin32GLKernel::ExecThreadCB(TWin32SendClass *command)
{
    EGLCallbackCmd cmd = (EGLCallbackCmd)(command->GetData(0));
    Bool_t debug = kFALSE;
    char *listcmd[] = {
                 "NewGLModelView(ilist)"
                ,"EndGLList()"
                ,"BeginGLCmd(EG3D2GLmode mode)"
                ,"Int_t GetGLError()"
                ,"EndGLCmd()"
                ,"PushGLMatrix()"
                ,"PopGLMatrix()"
                ,"TranslateGL(Double_t *xyz)"
                ,"MultGLMatrix(Double_t *mat)"
                ,"SetGLColorIndex(Int_t color)"
                ,"SetGLLineWidth(Float_t width)"
                ,"SetGLVertex(Float_t *vertex)"
                ,"DeleteGLLists(Int_t ilist, Int_t range)"
                ,"Int_t CreateGLLists(Int_t range)"
                ,"RunGLList(Int_t list)"
                ,"NewProjectionView(Double_t min[],Double_t max[])"
                ,"PaintPolyLine(Int_t n, Float_t *p, Option_t *option)"
                ,"PaintBrik(Float_t vertex[24])"
                ,"ClearGLColor(Float_t red, Float_t green, Float_t blue, Float_t alpha)"
                ,"ClearGL(UInt_t bufbits )"
                ,"FlusGL()"
                ,"NewGLList(UInt_t ilist, EG3D2GLmode mode)"
                ,"SetGLColor(Float_t *rgb)"
                ,"PaintCone(Float_t vertex,Int_t ndiv,Int_t nstacks)"
                ,"DisableGL()"
                ,"EnableGL()"
                ,"RotateGL()"
                ,"FrontGLFace()"
                ,"PaintGLPoints(Int_t n, Float_t *p, Option_t *option)"
                ,"SetGLPointSize(Float_t width)"
                ,"ClearColor(Int_t color)"
                ,"NewModelView(Double_t *angles,Double_t *delta )"
                ,"PolygonGLMode(EG3D2GLmode face , EG3D2GLmode mode)"
                ,"GetGL()"
                ,"ShadeGLModel(mode)"
                ,"SetRootLight(Bool_t flag=kTRUE);"
                ,"SetLineAttr(Color_t color, Int_t width)"
                ,"UpdateMatrix(Double_t *translate, Double_t *rotate, Bool_t isreflection)"
                                ,"AddRotation(Double_t *rotmatrix,Double_t *extraangles)"
                                ,"SetStack(Double_t *matrix)"
                ,"PaintGLPointsObject(TPoints3DABC *points, Option_t *option)"
    };

    if (gDebug) printf("TWin32GLKernel: commamd %d: %s",cmd,listcmd[cmd]);
    switch (cmd)
    {
    case kNewGLModelView:
        {
          Int_t ilist = (Int_t)(command->GetData(1));
          if (gDebug) printf(" %d ", ilist);
          TGLKernel::NewGLModelView(ilist);
          break;
        }
    case kEndGLList:
        TGLKernel::EndGLList();
        break;

    case kBeginGLCmd:
        {
            EG3D2GLmode mode = (EG3D2GLmode)(command->GetData(1));
            if (gDebug) printf(" %d ", mode);
            TGLKernel::BeginGLCmd(mode);
            break;
        }
    case kGetGLError:
        {
           Int_t *ierr = (Int_t *)(command->GetData(3));
           *ierr = TGLKernel::GetGLError();
           if (gDebug) printf(" Return %d ", *ierr);
           break;
        }
    case kEndGLCmd:
        TGLKernel::EndGLCmd();
        break;

    case kPushGLMatrix:
        TGLKernel::PushGLMatrix();
        break;

    case kPopGLMatrix:
        TGLKernel::PopGLMatrix();
        break;

    case kTranslateGL:
        {
            Double_t *xyz = (Double_t *)(command->GetData(1));
            TGLKernel::TranslateGL(xyz);
            for (int i=0;i<3;i++) if (gDebug) printf(" %6.3f ", xyz[i]);
            break;
        }
    case kMultGLMatrix:
        {
            Double_t *mat = (Double_t *)(command->GetData(1));
            TGLKernel::MultGLMatrix(mat);
            break;
        }
    case kSetGLColorIndex:
        {
            Int_t color = (Int_t)(command->GetData(1));
            if (gDebug) printf(" %d ", color);
            TGLKernel::SetGLColorIndex(color);
            break;
        }
    case kSetGLLineWidth:
        {
            Float_t width = (Float_t)(command->GetData(1));
            if (gDebug) printf(" %f ",width);
            TGLKernel::SetGLLineWidth(width);
            break;
        }
    case kSetGLVertex:
        {
            Float_t *vertex = (Float_t *)(command->GetData(1));
            TGLKernel::SetGLVertex(vertex);
            break;
        }
    case kDeleteGLLists:
        {
            Int_t ilist = (Int_t)(command->GetData(1));
            Int_t range = (Int_t)(command->GetData(2));
            if (gDebug) printf(" %d %d ", ilist, range);
            TGLKernel::DeleteGLLists(ilist,range);
            break;
        }
    case kCreateGLLists:
        {
            Int_t range = (Int_t)(command->GetData(1));
            Int_t *r    = (Int_t *)(command->GetData(3));
            *r = TGLKernel::CreateGLLists(range);
            if (gDebug) printf(" %d, return %d ", range, *r);
            break;
        }
    case kRunGLList:
        {
            Int_t list = (Int_t)(command->GetData(1));
            if (gDebug) printf(" %d ", list);
            TGLKernel::RunGLList(list);
            break;
        }
    case kNewProjectionView:
        {
            Double_t *min = (Double_t *)(command->GetData(1));
            Double_t *max = (Double_t *)(command->GetData(1));
            Bool_t    flag= (Bool_t)(command->GetData(2));
            TGLKernel::NewProjectionView(min,max,flag);
            if (gDebug) printf(" \n");
            break;
        }
    case kPaintPolyLine:
        {
            Int_t n          = (Int_t)(command->GetData(1));
            Float_t *p       = (Float_t *)(command->GetData(2));
            Option_t *option = (Option_t *)(command->GetData(3));
            if (gDebug) printf(" %d %f %s", n,*p, option);
            TGLKernel::PaintPolyLine(n, p, option);
            break;
        }
    case kPaintBrik:
        {
            Float_t *vertex  = (Float_t *)(command->GetData(1));
            TGLKernel::PaintBrik(vertex);
            break;
        }
    case kClearGLColor:
        {
            Float_t *colors  = (Float_t *)(command->GetData(1));
            for (int i=0;i<4;i++) if (gDebug) printf(" %4.2f ", colors[i]);
            TGLKernel::ClearGLColor(colors);
            break;
        }
    case kClearGL:
        {
            UInt_t bits  = (UInt_t)(command->GetData(1));
            TGLKernel::ClearGL(bits);
            break;
        }

    case kFlushGL:
            TGLKernel::FlushGL();
            break;

    case kNewGLList:
        {
            UInt_t ilist      = (UInt_t)(command->GetData(1));
            EG3D2GLmode mode  = (EG3D2GLmode)(command->GetData(2));
            TGLKernel::NewGLList(ilist,mode);
            break;
        }
    case kSetGLColor:
        {
            Float_t *rgb      = (Float_t *)(command->GetData(1));
            TGLKernel::SetGLColor(rgb);
            break;
        }

    case kPaintCone:
        {
            Float_t *vertex  = (Float_t *)(command->GetData(1));
            Int_t ndiv       = (Int_t)(command->GetData(2));
            Int_t nstacks    = (Int_t)(command->GetData(3));
            if (gDebug) printf(" %d %d", ndiv, nstacks);
            TGLKernel::PaintCone(vertex,ndiv,nstacks);
            break;
        }
    case kDisableGL:
        {
            EG3D2GLmode mode  = (EG3D2GLmode)(command->GetData(1));
            TGLKernel::DisableGL(mode);
            break;
        }
    case kEnableGL:
        {
            EG3D2GLmode mode  = (EG3D2GLmode)(command->GetData(1));
            TGLKernel::EnableGL(mode);
            break;
        }
    case kRotateGL:
        {
            Double_t *array = (Double_t *)(command->GetData(1));
            Int_t mode = (Int_t)(command->GetData(2));
            TGLKernel::RotateGL(array,mode);
            break;
        }
    case kFrontGLFace:
        {
            EG3D2GLmode faceflag  = (EG3D2GLmode)(command->GetData(1));
            TGLKernel::FrontGLFace(faceflag);
            break;
        }
    case kPaintGLPoints:
        {
            Int_t n          = (Int_t)(command->GetData(1));
            Float_t *p       = (Float_t *)(command->GetData(2));
            Option_t *option = (Option_t *)(command->GetData(3));
            if (gDebug) printf(" %d %f %s", n,*p, option);
            TGLKernel::PaintGLPoints(n, p, option);
            break;
        }
    case kSetGLPointSize:
        {
            Float_t size = (Float_t)(command->GetData(1));
            if (gDebug) printf(" %f ",size);
            TGLKernel::SetGLPointSize(size);
            break;
        }
   case kClearColor:
        {
            Int_t color = (Int_t)(command->GetData(1));
            if (gDebug) printf(" %d ", color);
            TGLKernel::ClearColor(color);
            break;
        }
    case kNewModelView:
        {
            Double_t *angles = (Double_t *)(command->GetData(1));
            Double_t *delta = (Double_t *)(command->GetData(2));
            TGLKernel::NewModelView(angles,delta);
            if (gDebug) printf(" \n");
            break;
        }
    case kPolygonGLMode:
        {
            EG3D2GLmode face  = (EG3D2GLmode)(command->GetData(1));
            EG3D2GLmode mode  = (EG3D2GLmode)(command->GetData(1));
            TGLKernel::PolygonGLMode(face,mode);
            break;
        }
    case kGetGL:
        {
            EG3D2GLmode mode  = (EG3D2GLmode)(command->GetData(1));
            EGLTypes type  = (EGLTypes)(command->GetData(3));
            switch (type) {
            case kBoolType:
                {
                    Bool_t *params = (Bool_t *)(command->GetData(2));
                    TGLKernel::GetGL(mode,params);
                }
                break;
            case kDoubleType:
                {
                    Double_t *params = (Double_t *)(command->GetData(2));
                    TGLKernel::GetGL(mode,params);
                }
                break;
            case kFloatType:
                {
                    Float_t *params = (Float_t *)(command->GetData(2));
                    TGLKernel::GetGL(mode,params);
                }
                break;
            case kIntegerType:
                {
                    Float_t *params = (Float_t *)(command->GetData(2));
                    TGLKernel::GetGL(mode,params);
                }
            case kShadeGLModel:
                {
                    EG3D2GLmode mode  = (EG3D2GLmode)(command->GetData(1));
                    TGLKernel::ShadeGLModel(mode);
                }
                break;
            case kSetRootLight:
                {
                    Bool_t mode  = (Bool_t)(command->GetData(1));
                    TGLKernel::SetRootLight(mode);
                }
                break;
            default:
                break;
            };
            break;
        }
    case kSetLineAttr:
        {
            Color_t color  = (Color_t)(command->GetData(1));
            Int_t width  = (Int_t)(command->GetData(2));
            TGLKernel::SetLineAttr(color,width);
        }
        break;
    case kUpdateMatrix:
        {
            Double_t *translate = (Double_t *)(command->GetData(1));
            Double_t *rotate    = (Double_t *)(command->GetData(2));
            Bool_t isreflection = (Bool_t)(command->GetData(3));
            TGLKernel::UpdateMatrix(translate,rotate,isreflection);
        }
        break;
    case kAddRotation:
        {
            Double_t *rotmatrix = (Double_t *)(command->GetData(1));
            Double_t *exangles    = (Double_t *)(command->GetData(2));
            TGLKernel::AddRotation(rotmatrix,exangles);
        }
        break;
    case kSetStack:
        {
            Double_t *matrix = (Double_t *)(command->GetData(1));
            TGLKernel::SetStack(matrix);
        }
        break;
    case kPaintGLPointsObject:
        {
            const TPoints3DABC *points = (TPoints3DABC *)(command->GetData(1));
            Option_t *option     = (Option_t *)(command->GetData(2));
            TGLKernel::PaintGLPointsObject(points,option);
        }
        break;
     default:
        break;
    }
    if (gDebug) printf(" \n");
    if (LOWORD(command->GetCOP()) == kSendWaitClass)
        ((TWin32SendWaitClass *)command)->Release();
    else
        delete command;
}



