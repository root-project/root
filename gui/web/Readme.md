## How to use TWebCanvas prototype

1. Checkout **webgui** branch and compile ROOT with http support 

    [shell] cmake -Dhttp=ON
    
2. Enable web gui factory in .rootrc file

     Gui.Factory:                web
     
3. Checkout latest JSROOT dev version  
  
     [shell] git clone https://github.com/root-project/jsroot.git       
     [shell] cd jsroot
     [shell] git checkout dev
     
4. Set JSROOTSYS shell variable like

     [shell] export JSROOTSYS=/home/user/git/jsroot
     
5. Run ROOT macro with TCanvas, where TCanvas::Update is regularly called.
   One could use gui/web/demo/hsimple.C as example      
     
6. To change http server port number do:

     [shell] export WEBGUI_PORT=8877


## Compilation with CEF support (https://bitbucket.org/chromiumembedded/cef)     

Sources will be taken from canvaspainter/v7
For the momemnt not used here


## Compilation with QT5 WebEngine support

Sources will be taken from canvaspainter/v7
For the momemnt not used here
