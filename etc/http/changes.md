# JSROOT changelog


## Changes in 3.9
1. Support non-equidistant bins for TH1/TH2 objects.
2. Display entries count from histo.fEntries member, only when not set use computed value  
3. Support italic and bold text when used with MathJax
4. Improve TF1 drawing - support exp function in TFormula, fix errors with logx scale, enable zoom-in, (re)calculate function points when zooming
5. Support several columns in TLegend
6. Introduce context menus for x/y axis, add some items similar to native ROOT menus
7. Introduce context menu for TPaveStats, let switch single elements in the box   
8. Enable usage of all context menus on touch devices 
9. Implement JSROOT.Math.Prob function, provides probability value in stat box 
10. Introduce context menu for color palette (z axis)
11. Implement col0 and col0z draw option for TH2 histograms, similar to ROOT6


## Changes in 3.8
1. Let use HTML element pointer in JSROOT.draw function like:
       JSROOT.draw(document.getElementsByTagName("div")[0], obj, "hist");
   Normally unique identifier was used before, which is not required any longer.
   Of course, old functionality with element identifier will work as well. 
2. TreePlayer can also be used for trees, which not yet read from the file.
   Requires appropriate changes in TRootSniffer class. 
3. Fix error in I/O with members like:   `Double_t *fArr; //[fN]`  
4. Introduce JSROOT.OpenFile function. It loads I/O functionality automatically,
   therefore can be used directly after loading JSRootCore.js script
5. Same is done with JSROOT.draw function. It is defined in the JSRootCore.js
   and can be used directly. Makes usage of JSROOT easier    
6. Introduce JSRootPainter.more.js script, where painters for auxiliary classes
   will be implemented.
7. Implement painter for TEllipse, TLine, TArrow classes     
8. Fix several problems with markers drawing; implement plus, asterisk, mult symbols. 
9. Implement custom layout, which allows to configure user-defined layout for displayed objects
10. Fix errors with scaling of axis labels.     
11. Support also Y axis with custom labels like: http://jsroot.gsi.de/dev/?nobrowser&file=../files/atlas.root&item=LEDShapeHeightCorr_Gain0;1&opt=col   


## Changes in 3.7
1. Support of X axis with custom labels like: http://jsroot.gsi.de/dev/index.htm?nobrowser&json=../files/hist_xlabels.json
2. Extend functionality of JSROOT.addDrawFunc() function. One could register type-specific
   `make_request` and `after_request` functions; `icon`, `prereq`, `script`, `monitor` properties.
   This let add more custom elements to the generic gui, implemented with JSROOT.HierarchyPainter   
3. Provide full support of require.js. One could load now JSRootCore.js script like:

      <script type="text/javascript" src="require.js" data-main="scripts/JSRootCore.js"></script>
      
   After this several modules are defined and can be used with syntax like:
   
      require(['JSRootPainter'], function(jsroot) { /*any user code*/});
      
   Also inside JSROOT require.js used to load all dependencies. 


## Changes in 3.6
1. Try to provide workaround for websites where require.js already loaded.
   This makes problem by direct loading of jquery and jquery-ui 
2. Provide workaround for older version of jquery-ui 
3. Prompt for input of command arguments
4. After command execution one could automatically reload hierarchy (_hreload property) or
   update view of displayed object (_update_item property)    
5. Use HiearchyPainter for implementing draw.htm. This let us handle
   all different kinds of extra attributes in central place 
6. Fix problem in tabs layout - new tab should be add to direct child
7. When drawing several tabs, activate frame before drawing - only then
   real frame size will be set
8. Fix problem with GetBBox - it only can be used for visible elements in mozilla.    
9. Support drawing of fit parameters in stat box, use (as far as possible) stat and
   fit format for statistic display 
10. Implement 'g' formatting kind for stat box output - one need to checks 
    significant digits when producing output.  
11. Support new draw options for TGraph: 'C', 'B1', '0', '2', '3', '4', '[]'
12. Primary support for STL containers in IO part. Allows to read ROOT6 TF1.
13. Full support of TGraphBentErrors
14. Support objects drawing from JSON files in default user interface, including
    monitoring. One could open file from link like: https://root.cern.ch/js/dev/?json=demo/canvas_tf1.json 
15. Introduce JSROOT.FFormat function to convert numeric values into string according
    format like 6.4g or 5.7e. Used for statistic display.


## Changes in 3.5
1. Fix error in vertical text alignment
2. Many improvements in TPaletteAxis drawing - draw label, avoid too large ticks.
3. Fix error with col drawing - bin with maximum value got wrong color  
4. Test for existing jquery.js, jquery-ui.js and d3.js libraries, reuse when provided 
5. Fix several I/O problems; now one could read files, produced in Geant4
6. Implement 'e2' drawing option for TH1 class, 
   use by default 'e' option when TH1 has non-empty fSumw2
7. Reuse statistic from histogram itself, when no axis selection done 
8. Support log/lin z scale for color drawing
9. Implement interactive z-scale selection on TPaletteAxis 
10. Allow to redraw item with other draw options (before one should clear drawings)
11. Several improvements in THttpServer user interface - repair hierarchy reload,
    hide unsupported context menu entries, status line update 


## Changes in 3.4
1. Support usage of minimized versions of .js and .css files.
   Minimized scripts used by default on web servers.
2. Implement JSROOT.extend instead of jQuery.extend, reduce
   usage of jquery.js in core JSROOT classes
3. Implement main graphics without jquery at all,
   such mode used in `nobrowser` mode.
4. Provide optional latex drawing with MathJax SVG.
   TMathText always drawn with MathJax,
   other classes require `mathjax` option in URL
5. Improve drawing of different text classes, correctly handle
   their alignment and scaling, special handling for IE
6. Fix error with time axes - time offset was not correctly interpreted


## Changes in 3.3
1. Use d3.time.scale for display of time scales
2. Within JSRootCore.js script URL one could specify JSROOT
   functionality to be loaded: '2d', '3d', 'io', 'load', 'onload'.
   Old method with JSROOT.AssertPrerequisites will also work.
3. With THttpServer JSROOT now provides simple control functionality.
   One could publish commands and execute them from the browser
4. One could open several ROOT files simultaneously
5. Add 'simple' layout - drawing uses full space on the right side
6. Allow to open ROOT files in online session (via url parameter)
7. One could monitor simultaneously objects from server and root files
8. Implement 'autocol' draw option  - when superimposing histograms,
   their line colors will be automatically assigned
9. Implement 'nostat' draw option - disabled stat drawing
10. Using '_same_' identifier in item name, one can easily draw or superimpose
    similar items from different files. Could be used in URL like:
      `...&files=[file1.root,file2.root]&items=[file1.root/hpx, file2.root/_same_]`
      `...&files=[file1.root,file2.root]&item=file1.root/hpx+file2.root/_same_`
    Main limitation - file names should have similar length.
11. When 'autozoom' specified in draw options, histogram zoomed into
    non-empty content. Same command available via context menu.
12. Item of 'Text' kind can be created. It is displayed as
    plain text in the browser. If property 'mathjax' specified,
    MathJax.js library will be loaded and used for rendering.
    See httpcontrol.C macro for example.
13. When using foreignObject, provide workaround for absolute positioning
    problem in Chrome/Safari, see <http://bit.ly/1wjqCQ9>


## Changes in 3.2
1. Support JSON objects embedding in html pages, produced by THttpServer
2. For small resize of canvas use autoscale functionality of SVG. Only when
   relative changes too large, redraw complete canvas again.
3. Use touch-punch.min.js to process touch events with jquery-ui
4. Even when several TH1/TGraph/TF1 objects with fill attribute overlap each other,
   one able to get tooltip for underlying objects
5. Use jquery-ui menu for context menu
6. From context menu one could select several options for drawing
7. Provide user interface for executing TTree::Draw on THttpServer
8. 3D graphic (three.js) works only with IE11


## Changes in 3.1
1. Correctly show tooltips in case of overlapped objects
2. Implement JSROOT.Create() method to create supported
   in JavaScript ROOT classes like TH1 or TGraph
3. Fix problem with JSROOT.draw in HTML element with zero width (display:none)
4. Provide possibility to load user scripts with JSROOT.BuildSimpleGUI
   and JSROOT.AssertPrerequisites, also with main index.htm
5. Support of TCutG drawing
6. Implement hierarchy display (former dtree) with jQuery
7. Fix several problems in drawing optimization
8. Implement dragging objects from hierarchy browser into existing canvas
   to superimpose several objects
9. Implement col2 and col3 draw options, using html5 canvas
10. Support 'p' and 'p0' draw options for TH1 class


## Development of version 3.0

### November 2014
1. Better font size and position in pave stats
2. Resize/move of element only inside correspondent pad
3. Adjust of frame size when Y-axis exceed pad limits
4. Correct values in tooltip for THStack
5. Exclude drawing of markers from TGraph outside visible range
6. Drawing of canvas without TFrame object
7. Many other small bug fixes and improvements, thanks to Maximilian Dietrich

### October 2014
1.  Add "shortcut icon"
2.  Add demo of online THttpServer - shell script copies data from
    running httpserver.C macro on Apache webserver
3.  Evaluate 'monitoring' parameter for online server like:
      <http://localhost:8080/?monitoring=1000>
    Parameter defines how often displayed objects should be updated.
4.  Implement 'opt' and 'opts' URL parameters for main page.
5.  Show progress with scripts loading in the browser window
6.  When one appends "+" to the filename, its content read completely with first I/O operation.
7.  Implement JS custom streamer for TCanvas, restore aspect ratio when drawing
8.  Major redesign of drawing classes. Resize and update of TCanvas are implemented.
    All major draw functions working with HTML element id as first argument.
9.  Extract 3D drawings into separate JSRoot3DPainter.js script
10. Use newest three.min.js (r68) for 3D drawings, solves problem with Firefox.
11. Introduce generic list of draw functions for all supported classes.
12. Add possibility to 'expand' normal objects in the hierarchy browser.
    For instance, this gives access to single elements of canvas,
    when whole canvas cannot be drawn.
13. Correct usage of colors map, provided with TCanvas.
14. Introduce JSROOT.redraw() function which is capable to create or update object drawing.
15. In main index.htm page browser can be disabled (nobrowser parameter) and
    page can be used to display only specified items from the file
16. Add support of TPolyMarker3D in binary I/O

### September 2014
1. First try to handle resize of the browser,
   for the moment works only with collapsible layout
2. Also first try to interactively move separation line between
   browser and drawing field.
3. Small fix of minor ticks drawing on the axis
4. Introduce display class for MDI drawing. Provide two implementations -
   'collapsible' for old kind and 'tabs' for new kinds.
5. Adjust size of color palette drawing when labels would take more place as provided.
6. Add correct filling of statistic for TProfile,
   fix small problem with underflow/overflow bins.
7. Provide way to select display kind ('collapsible', 'tabs') in the simple GUI.
8. Implement 'grid' display, one could specify any number of devision like
   'grid 3x3' or 'grid 4x2'.
9. MDI display object created at the moment when first draw is performed.
10. Introduce painter class for TCanvas, support resize and update of canvas drawing
11. Resize almost works for all layouts and all objects kinds.
12. Implement JSROOT.GetUrlOption to extract options from document URL.
13. Provide example fileitem.htm how read and display item from ROOT file.
14. In default index.htm page one could specify 'file', 'layout',
    'item' and 'items' parameters like:
      <http://root.cern.ch/js/3.0/index.htm?file=../files/hsimple.root&layout=grid3x2&item=hpx;1>
15. Support direct reading of objects from sub-sub-directories.
16. Introduce demo.htm, which demonstrates online usage of JSROOT.
17. One could use demo.htm directly with THttpServer providing address like:
     <http://localhost:8080/jsrootsys/demo/demo.htm?addr=../../Files/job1.root/hpx/root.json.gz&layout=3x3>
18. Also for online server process url options like 'item', 'items', 'layout'
19. Possibility to generate URL, which reproduces opened page with layout and drawn items

### August 2014
1. All communication between server and browser done with JSON format.
2. Fix small error in dtree.js - one should always set
   last sibling (_ls) property while tree can be dynamically changed.
3. In JSRootCore.js provide central function, which handles different kinds
   of XMLHttpRequest.  Use only async requests, also when getting file header.
4. Fully reorganize data management in file/tree/directory/collection hierarchical
   display. Now complete description collected in HPainter class and decoupled from
   visualization, performed with dTree.js.
5. Remove all global variables from the code.
6. Automatic scripts/style loading handled via JSROOT.loadScript() function.
   One can specify arbitrary scripts list, which asynchronously loaded by browser.
7. Method to build simple GUI changed and more simplified :). The example in index.htm.
   While loadScript and AssertPrerequisites functions moved to JSROOT, one
   can easily build many different kinds of GUIs, reusing provided JSRootCore.js functions.
8. In example.htm also use AssertPrerequisites to load necessary scripts.
   This helps to keep code up-to-date even by big changes in JavaScript code.
9. Provide monitoring of online THttpServer with similar interface as for ROOT files.
10. Fix several errors in TKey Streamer, use member names as in ROOT itself.
11. Keep the only version identifier JSROOT.version for JS code
12. One can specify in JSROOT.AssertPrerequisites functionality which is required.
    One could specify '2d', 'io' (default) or '3d'.
13. Use new AssertPrerequisites functionality to load only required functionality.
14. When displaying single element, one could specify draw options and monitor property like:
        <http://localhost:8080/Files/job1.root/hpxpy/draw.htm?opt=col&monitor=2000>
     Such link is best possibility to integrate display into different HTML pages,
     using `<iframe/>` tag like:
        `<iframe src="http://localhost:8080/Files/job1.root/hpx/draw.htm"` 
          `style="width: 800px; height:600px"></iframe>`
15. Remove 'JSROOTIO.' prefix from _typename. Now real class name is used.
16. Use in all scripts JSROOT as central 'namespace'
17. Introduce context menu in 3D, use it for switch between 2D/3D modes
18. Use own code to generate hierarchical structure in HTML, replace dtree.js which is
    extremely slow for complex hierarchies. Dramatically improve performance for
    structures with large (~1000) number of items.
19. Deliver to the server title of the objects, display it as hint in the browser.
20. Better handling of special characters in the hierarchies - allows to display
    symbols like ' or " in the file structure.

### July 2014
1. Migration to d3.v3.js and jQuery v2.1.1
2. Fix errors in filling of histogram statbox
3. Possibility of move and resize of statbox, title, color palete
4. Remove many (not all) global variables
5. Example with direct usage of JSRootIO graphics
6. Example of inserting ROOT graphics from THttpServer into `<iframe></iframe>`

### May 2014
1. This JSRootIO code together with THttpServer class included
   in ROOT repository

### March 2014
1. Introduce JSROOT.TBuffer class, which plays similar role
   as TBuffer in native ROOT I/O. Simplifies I/O logic,
   reduce duplication of code in many places, fix errors.
   Main advantage - one could try to keep code synchronous with C++.
2. Avoid objects cloning when object referenced several times.
3. Treat special cases (collection, arrays) in one place.
   This is major advantage, while any new classes need to be implemented only once.
4. Object representation, produced by JSRootIO is similar to
   objects, produced by TBufferJSON class. By this one can exchange
   I/O engine and use same JavaSctript graphic for display.
5. More clear functions to display different elements of the file.
   In the future functions should be fully separated from I/O part
   and organized in similar way as online part.
6. Eliminate usage of gFile pointer in the I/O part.
7. Provide TBufferJSON::JsonWriteMember method. It allows to stream any
   selected data member of the class. Supported are:
   basic data types, arrays of basic data types, TString, TArray classes.
   Also any object as data member can be streamed.
8. TRootSniffer do not creates sublevels for base classes
