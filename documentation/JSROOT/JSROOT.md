# JavaScript ROOT

The JSROOT project intends to implement ROOT graphics for web browsers.
Reading of binary ROOT files is supported.
It is the successor of the JSRootIO project.


## Installing JSROOT

In most practical cases it is not necessary to install JSROOT - it can be used directly from project web sites <https://root.cern/js/> and <http://jsroot.gsi.de/>.

When required, there are following alternatives to install JSROOT on other web servers:

   - download and unpack [provided](https://github.com/linev/jsroot/releases) packages (recommended)
   - use [bower](https://bower.io/) package manager and invoke `bower install jsroot`
   - clone master branch from [repository](https://github.com/linev/jsroot/)

One could use JSROOT directly from local file system. If source code was unpacked/checked-out in `/home/user/jsroot/` subfolder, one could just open it in browser with <file:///home/user/jsroot/index.htm> address.


## Drawing objects in JSROOT

[The main page](https://root.cern/js/latest/) of the JSROOT project provides the possibility to interactively open ROOT files and draw objects like histogram or canvas.

To automate files loading and objects drawing, one can provide number of URL parameters in address string like:

- file - name of the file, which will be automatically open with page loading
- files - array of file names for loading
- localfile - automatically activate dialog for selecting local ROOT files
- json - name of JSON file with stored ROOT object like histogram or canvas
- item - item name to be displayed
- opt - drawing option for the item
- items - array of items name
- opts - array of drawing options for the items
- layout - can be 'simple', 'flex', 'collapsible', 'tabs', 'gridNxM', 'horizNMK', 'vertNMK'
- browser - layout of the browser 'float', 'fix', 'no', 'off'
- nobrowser - do not display file browser (same as browser=no)
- float - display floating browser (same as browser=float)
- status - configure status line 'no' (default), 'off' (completely disable), 'size'
- load - name of extra JavaScript to load
- optimize - drawing optimization 0:off, 1:only large histograms (default), 2:always
- paltte - id of default color palette, 51..121 - new ROOT6 palette  (default 57)
- interactive - enable/disable interactive functions 0-disable all, 1-enable all
- noselect - hide file-selection part in the browser (only when file name is specified)
- mathjax - use MathJax for latex output
- style - name of TStyle object to define global JSROOT style
- toolbar - show canvas tool buttons 'off', 'on' and 'popup'

For instance:

- <https://root.cern/js/latest/?file=../files/hsimple.root&item=hpx;1>
- <https://root.cern/js/latest/?file=../files/hsimple.root&nobrowser&item=hpxpy;1&opt=colz>
- <https://root.cern/js/latest/?file=../files/hsimple.root&noselect&layout=grid2x2&item=hprof;1>

Following layouts are supported:

  - simple - available space used for single object (default)
  - [collapsible](https://root.cern/js/latest/api.htm#url_syntax_collapsible_layout) - fix-sized canvas group behind each other and can be collapsed individually
  - [flex](https://root.cern/js/latest/api.htm#url_syntax_flexible_layout) - creates as many frames as necessary, each can be individually moved/enlarged
  - [tabs](https://root.cern/js/latest/api.htm#url_syntax_tabs_layout) - tabs widget
  - [gridNxM](https://root.cern/js/latest/api.htm#url_syntax_grid_layout) - fixed-size grid with NxM frames
  - gridiNxM - grid with NxM frames with possibility to adjust frames sizes
  - vertN - N frames sorted in vertical direction (like gridi1xN)
  - horizN - N frames sorted in horizontal direction (like gridiNx1)
  - [vert121](https://root.cern//js/latest/api.htm#url_syntax_veritcal_layout) - 3 frames sorted in vertical direction, second frame divided on two sub-frames
  - [horiz32_12](https://root.cern//js/latest/api.htm#url_syntax_horizontal_layout) - 2 horizontal frames with 3 and 2 subframes, and 1/3 and 2/3 as relative size

When specifying `files`, `items` or `opts` parameters, array of strings could be provided  like `files=['file1.root','file2.root']`.  One could skip quotes when specifying elements names `items=[file1.root/hpx,file2.root/hpy]` or `opts=['',colz]`.

Many examples of URL string usage can be found on [JSROOT examples](https://root.cern/js/latest/api.htm) page.

One can very easy integrate JSROOT graphic into arbitrary HTML pages using a __iframe__ tag:

    <iframe width="700" height="400"
            src="https://root.cern/js/latest/?nobrowser&file=../files/hsimple.root&item=hpxpy&opt=colz">
    </iframe>


## Supported ROOT classes by JSROOT

List of supported classes and draw options:

- TH1 : [hist](https://root.cern/js/latest/examples.htm#th1),
[p](https://root.cern/js/latest/examples.htm#th1_p),
[p0](https://root.cern/js/latest/examples.htm#th1_p0),
[*](https://root.cern/js/latest/examples.htm#th1_star),
[l](https://root.cern/js/latest/examples.htm#th1_l),
[lf2](https://root.cern/js/latest/examples.htm#th1_lf2),
[e](https://root.cern/js/latest/examples.htm#th1_e),
[e0](https://root.cern/js/latest/examples.htm#th1_e0),
[e1](https://root.cern/js/latest/examples.htm#th1_e1),
[e1x0](https://root.cern/js/latest/examples.htm#th1_e1x0),
[e3](https://root.cern/js/latest/examples.htm#th1_e3),
[e4](https://root.cern/js/latest/examples.htm#th1_e4),
[lego](https://root.cern/js/latest/examples.htm#th1_lego),
[text](https://root.cern/js/latest/examples.htm#th1_text),
[X+Y+](https://root.cern/js/latest/examples.htm#th1_x+y+)
- TH2 : [scat](https://root.cern/js/latest/examples.htm#th2),
[col](https://root.cern/js/latest/examples.htm#th2_col),
[colz](https://root.cern/js/latest/examples.htm#th2_colz),
[box](https://root.cern/js/latest/examples.htm#th2_box),
[box1](https://root.cern/js/latest/examples.htm#th2_box1),
[text](https://root.cern/js/latest/examples.htm#th2_text),
[lego](https://root.cern/js/latest/examples.htm#th2_lego),
[arr](https://root.cern/js/latest/examples.htm#th2_arr),
[cont](https://root.cern/js/latest/examples.htm#th2_cont),
[cont1](https://root.cern/js/latest/examples.htm#th2_cont1),
[cont2](https://root.cern/js/latest/examples.htm#th2_cont2),
[cont3](https://root.cern/js/latest/examples.htm#th2_cont3),
[cont4](https://root.cern/js/latest/examples.htm#th2_cont4),
[surf](https://root.cern/js/latest/examples.htm#th2_surf),
[surf1](https://root.cern/js/latest/examples.htm#th2_surf1),
[surf2](https://root.cern/js/latest/examples.htm#th2_surf2),
[surf3](https://root.cern/js/latest/examples.htm#th2_surf3),
[surf4](https://root.cern/js/latest/examples.htm#th2_surf4),
[surf6](https://root.cern/js/latest/examples.htm#th2_surf6),
[surf7](https://root.cern/js/latest/examples.htm#th2_surf7),
[lego](https://root.cern/js/latest/examples.htm#th2_lego),
[lego0](https://root.cern/js/latest/examples.htm#th2_lego0),
[lego1](https://root.cern/js/latest/examples.htm#th2_lego1),
[lego2](https://root.cern/js/latest/examples.htm#th2_lego2),
[lego3](https://root.cern/js/latest/examples.htm#th2_lego3),
[lego4](https://root.cern/js/latest/examples.htm#th2_lego4)
- TH2Poly : [col](https://root.cern/js/latest/examples.htm#th2poly_honeycomb),
[lego](https://root.cern/js/latest/examples.htm#th2poly_lego),
[europe](https://root.cern/js/latest/examples.htm#th2poly_europe),
[usa](https://root.cern/js/latest/examples.htm#th2poly_usa)
- TH3 : [scat](https://root.cern/js/latest/examples.htm#th3),
[box](https://root.cern/js/latest/examples.htm#th3_box),
[box1](https://root.cern/js/latest/examples.htm#th3_box1)
- TProfile : [dflt](https://root.cern/js/latest/examples.htm#tprofile),
[e](https://root.cern/js/latest/examples.htm#tprofile_e),
[e1](https://root.cern/js/latest/examples.htm#tprofile_e1),
[pe2](https://root.cern/js/latest/examples.htm#tprofile_pe2),
[hist](https://root.cern/js/latest/examples.htm#tprofile_hist),
[text](https://root.cern/js/latest/examples.htm#tprofile_text),
[texte](https://root.cern/js/latest/examples.htm#tprofile_texte)
- TProfile2D : [example](https://root.cern/js/latest/examples.htm#misc_profile2d)
- THStack : [example](https://root.cern/js/latest/examples.htm#thstack)
- TF1 : [example](https://root.cern/js/latest/examples.htm#tf1_canv)
- TF2 : [example](https://root.cern/js/latest/examples.htm#misc_tf2)
- TGraph : [dflt](https://root.cern/js/latest/examples.htm#tgraph),
[L](https://root.cern/js/latest/examples.htm#tgraph_l),
[P](https://root.cern/js/latest/examples.htm#tgraph_p),
[*](https://root.cern/js/latest/examples.htm#tgraph_star),
[B](https://root.cern/js/latest/examples.htm#tgraph_b)
- TGraphErrors : [dflt](https://root.cern/js/latest/examples.htm#tgrapherrors),
[l](https://root.cern/js/latest/examples.htm#tgrapherrors_l),
[lx](https://root.cern/js/latest/examples.htm#tgrapherrors_lx),
[z](https://root.cern/js/latest/examples.htm#tgrapherrors_z),
[>](https://root.cern/js/latest/examples.htm#tgrapherrors_>),
[|>](https://root.cern/js/latest/examples.htm#tgrapherrors_|>),
[||](https://root.cern/js/latest/examples.htm#tgrapherrors_||),
[[]](https://root.cern/js/latest/examples.htm#tgrapherrors_[]),
[0](https://root.cern/js/latest/examples.htm#tgrapherrors_0),
[2](https://root.cern/js/latest/examples.htm#tgrapherrors_2),
[3](https://root.cern/js/latest/examples.htm#tgrapherrors_3),
[4](https://root.cern/js/latest/examples.htm#tgrapherrors_4),
[5](https://root.cern/js/latest/examples.htm#tgrapherrors_5),
- TGraphAsymmErrors : [dflt](https://root.cern/js/latest/examples.htm#tgraphasymmerrors),
[z](https://root.cern/js/latest/examples.htm#tgraphasymmerrors_z) and other from TGraphErrors
- TMultiGraph : [example](https://root.cern/js/latest/examples.htm#tmultigraph_c3), [exclusion](https://root.cern/js/latest/examples.htm#tmultigraph_exclusion)
- TGraph2D : [example](https://root.cern/js/latest/examples.htm#tgraph2d)
- TLatex : [example](https://root.cern/js/latest/examples.htm#tlatex_latex)
- TMathText : [example](https://root.cern/js/latest/examples.htm#tlatex_math)
- TCanvas : [example](https://root.cern/js/latest/examples.htm#tcanvas_roofit)
- TLegend : [example](https://root.cern/js/latest/examples.htm#tcanvas_legend)
- TTree : [single-branch draw](https://root.cern/js/latest/examples.htm#ttree_draw)
- TPolyLine : [dflt](https://root.cern/js/latest/examples.htm#misc_polyline)
- TEllipse : [dflt](https://root.cern/js/latest/examples.htm#misc_ellipse)
- TArrow : [dflt](https://root.cern/js/latest/examples.htm#misc_arrow)
- TPolyMarker3D: [dflt](https://root.cern/js/latest/examples.htm#misc_3dmark)

More examples of supported classes can be found on: <https://root.cern/js/latest/examples.htm>


## Superimposing draw objects

In the URL string one could use "+" sign to specify objects superposition:

   - [item=hpx+hprof](https://root.cern/js/latest/?file=../files/hsimple.root&item=hpx+hprof)

Analogue, one could specify individual draw options for superimposed objects

   - [item=hpx+hprof&opt=logy+hist](https://root.cern/js/latest/?file=../files/hsimple.root&item=hpx+hprof&opt=logy+hist)

Here "logy" option will be used for "hpx1" item and "hist" option for "hprof;1" item.

While draw option can include "+" sign itself, for superposition one could specify arrays of items and draw options like:

   - [item=[hpx;1,hprof;1]&opt=[logy,hist]](https://root.cern/js/latest/?file=../files/hsimple.root&item=[hpx;1,hprof;1]&opt=[logy,hist])


## TTree draw

JSROOT provides possibility to display TTree data, using [TTree::Draw](https://root.cern/doc/master/classTTree.html) syntax:

   - [opt=px](https://root.cern/js/latest/?file=../files/hsimple.root&item=ntuple;1&opt=px)
   - [opt=px:py](https://root.cern/js/latest/?file=../files/hsimple.root&item=ntuple;1&opt=px:py)
   - [opt=px:py:pz](https://root.cern/js/latest/?file=../files/hsimple.root&item=ntuple;1&opt=px:py:pz)

Histogram ranges and binning defined after reading first 1000 entries from the tree.
Like in ROOT, one could configure histogram binning and range directly:

   - [opt=px:py>>h(50,-5,5,50,-5,5)](https://root.cern/js/latest/?file=../files/hsimple.root&item=ntuple&opt=px:py>>h%2850,-5,5,50,-5,5%29)

For any integer value one can accumulate histogram with value bits distribution, specifying as output ">>bits(16)" or ">>bits":

   - [opt=event.fTracks.fBits>>bits](https://root.cern/js/latest/?file=https://root.cern/files/Event100000.root&item=T;2&opt=event.fTracks.fBits>>bits)

There is special handling of TBits objects:

   - [opt=event.fTriggerBits](https://root.cern/js/latest/?file=https://root.cern/files/event/event_0.root&item=EventTree&opt=event.fTriggerBits)

It is allowed to use different expressions with branch values:

   - [opt=px+py:px-py](https://root.cern/js/latest/?file=../files/hsimple.root&item=ntuple&opt=px+py:px-py)

Such expression can include arithmetical operations and all methods, provided in JavaScript [Math](http://www.w3schools.com/jsref/jsref_obj_math.asp) class:

   - [opt=Math.abs(px+py)](https://root.cern/js/latest/?file=../files/hsimple.root&item=ntuple&opt=Math.abs%28px+py%29)

In the expression one could use "Entry$" and "Entries$" variables.

One also could specify cut condition, separating it with "::" from the rest draw expression like:

   - [opt=px:py::pz>5](https://root.cern/js/latest/?file=../files/hsimple.root&item=ntuple&opt=px:py::pz>5)

Contrary to the normal ROOT, JSROOT allows to use "(expr?res1:res2)" operator (placed into brackets):

   - [opt=px:py::(pz>5?2:1)](https://root.cern/js/latest/?file=../files/hsimple.root&item=ntuple&opt=px:py::%28pz>5?2:1%29)

It is possible to "dump" content of any branch (by default - first 10 entries):

   - [item=ntuple/px&opt=dump](https://root.cern/js/latest/?file=../files/hsimple.root&item=ntuple/px&opt=dump)

Or one could dump values produced with draw expression (also first 10 entries by default):

   - [opt=px:py::pz>>dump](https://root.cern/js/latest/?file=../files/hsimple.root&item=ntuple&opt=px:py::pz>>dump)

Working with array indexes is supported. By default, all elements in array are used for the drawing.
One could specify index for any array dimension (-1 means last element in the array). For instance, dump last element from `event.fTracks` array:

   - [opt=event.fTracks[-1].fBits>>dump](https://root.cern/js/latest/?file=https://root.cern/files/event/event_0.root&item=EventTree&opt=event.fTracks[-1].fBits>>dump)

For any array or collection kind one could extract its size with expression:

   - [opt=event.fTracks.@size](https://root.cern/js/latest/?file=https://root.cern/files/event/event_0.root&item=EventTree&opt=event.fTracks.@size;num:3000)

At the end of expression one can add several parameters with the syntax:

    <draw_expession>;par1name:par1value;par2name:par2value

Following parameters are supported:
  - "first" - id of the first entry to process
  - "entries" - number of entries to process
  - "monitor" - periodically show intermediate draw results (interval in milliseconds)
  - "maxrange" - maximal number of ranges in single HTTP request
  - "accum" - number of accumulated values before creating histogram
  - "htype" - last letter in histogram type like "I", "F", "D", "S", "L", "C"
  - "hbins" - number of bins on each histogram axis
  - "drawopt" - drawing option for produced histogram

Example - [opt=event.fTracks[].fTriggerBits;entries:1000;first:200;maxrange:25](https://root.cern/js/latest/?file=https://root.cern/files/event/event_0.root&item=EventTree&opt=event.fTracks[].fTriggerBits;entries:1000;first:200;maxrange:25)


## Geometry viewer

JSROOT implements display of TGeo objects like:

- [file=rootgeom.root&item=simple1](https://root.cern/js/latest/?file=../files/geom/rootgeom.root&item=simple1)
- [file=building.root&item=geom&opt=z](https://root.cern/js/latest/?nobrowser&file=../files/geom/building.root&item=geom;1&opt=z)

Following classes are supported by geometry viewer:
  - TGeoVolume
  - TGeoNode
  - TGeoManager (master volume will be displayed)
  - TEveGeoShapeExtract (used in EVE)

Following draw options could be specified (separated by semicolon or ';'):
   - axis  - draw axis coordinates
   - z   - set z axis direction up (normally y axis is up and x looks in user direction)
   - clipx/clipy/clipz - enable correspondent clipping panel
   - clip or clipxyz - enable all three clipping panels
   - ssao - enable Smooth Lighting Shader (or Screen Space Ambient Occlusion)
   - wire - instead of filled surfaces only wireframe will be drawn
   - more  - show 2 times more volumes as usual (normally ~2000 volumes or ~100000 elementary faces are shown)
   - more3 - show 3 times more volumes as usual
   - all - try to display all geometry volumes (may lead to browser hanging)
   - highlight - force highlighting of selected volume, normally activated for moderate-size geometries
   - macro:name.C - invoke ROOT configuration macro
   - dflt_colors - set default volumes colors as TGeoManager::DefaultColors() does
   - transpXY - set global transparency value (XY is number between 1 and 99)
   - rotate - enable automatic rotation of the geometry


It is possible to display only part of geometry model. For instance, one could select sub-item like:

- [file=rootgeom.root&item=simple1/TOP/REPLICA_1](https://root.cern/js/latest/?file=../files/geom/rootgeom.root&item=simple1/TOP/REPLICA_1)

Or one can use simple selection syntax (work only with first-level volumes):

- [item=simple1&opt=-bar1-bar2](https://root.cern/js/latest/?file=../files/geom/rootgeom.root&item=simple1;1&opt=-bar1-bar2)

Syntax uses '+' sign to enable visibility flag of specified volume and '-' sign to disable visibility.
One could use wildcard symbol like '+TUBE1*'.

Another way to configure visibility flags is usage of ROOT macros, which typically looks like:

     {
      TGeoManager::Import("http://root.cern/files/alice2.root");
      gGeoManager->DefaultColors();
      //   gGeoManager->SetVisLevel(4);
      gGeoManager->GetVolume("HALL")->InvisibleAll();
      gGeoManager->GetVolume("ZDCC")->InvisibleAll();
      gGeoManager->GetVolume("ZDCA")->InvisibleAll();
      ...
      gGeoManager->GetVolume("ALIC")->Draw("ogl");
      new TBrowser;
    }

Example of such macro can be found in root tutorials.

From provided macro only following calls will be executed in JSROOT:

   * `gGeoManager->DefaultColors()`
   * `gGeoManager->GetVolume("HALL")->InvisibleAll()`
   * `gGeoManager->GetVolume("HALL")->SetTransparency(30)`
   * `gGeoManager->GetVolume("HALL")->SetLineColor(5)`
   * `gGeoManager->GetVolume("ALIC")->Draw("ogl")`

All other will be ignored.


Example of major LHC detectors:
 * ALICE: [full](https://root.cern/js/latest/?file=https://root.cern/files/alice2.root&item=Geometry;1&opt=macro:https://root.cern/js/files/geomAlice.C)
 * ATLAS: [full](https://root.cern/js/latest/?file=https://root.cern/files/atlas.root&item=atlas;1&opt=clipxyz), [cryo](https://root.cern/js/latest/?file=https://root.cern/files/atlas.root&item=atlas;1&opt=macro:https://root.cern/files/atlas_cryo.C), [sctt](https://root.cern/js/latest/?file=https://root.cern/files/atlas.root&item=atlas;1&opt=macro:https://root.cern/files/atlas_sctt.C)
 * CMS: [cmse](https://root.cern/js/latest/?file=https://root.cern/files/cms.root&item=cms;1&opt=macro:https://root.cern/files/cms_cmse.C;clipxyz), [calo](https://root.cern/js/latest/?file=https://root.cern/files/cms.root&item=cms;1&opt=macro:https://root.cern/files/cms_calo.C;clipxyz)
 * LHCb: [full](https://root.cern/js/latest/?file=https://root.cern/files/lhcbfull.root&item=Geometry;1&opt=all;dflt_colors)

Other detectors examples:
 * HADES: [full](https://root.cern/js/latest/?file=https://root.cern/files/hades2.root&item=CBMGeom;1&opt=all;dflt_colors), [preselected](https://root.cern/js/latest/?json=../files/geom/hades.json.gz)
 * BABAR: [full](https://root.cern/js/latest/?file=https://root.cern/files/babar.root&item=babar;1&opt=macro:https://root.cern/files/babar_all.C), [emca](https://root.cern/js/latest/?file=https://root.cern/files/babar.root&item=babar;1&opt=macro:https://root.cern/files/babar_emca.C)
 * STAR: [full](https://root.cern/js/latest/?file=https://root.cern/files/star.root&item=star;1&opt=macro:https://root.cern/files/star_all.C;clipxyz), [svtt](https://root.cern/js/latest/?file=https://root.cern/files/star.root&item=star;1&opt=macro:https://root.cern/files/star_svtt.C)
 * D0: [full](https://root.cern/js/latest/?file=https://root.cern/files/d0.root&item=d0;1&opt=clipxyz)
 * NA47: [full](https://root.cern/js/latest/?file=https://root.cern/files/na47.root&item=na47;1&opt=dflt_colors)
 * BRAHMS: [full](https://root.cern/js/latest/?file=https://root.cern/files/brahms.root&item=brahms;1&opt=dflt_colors)
 * SLD: [full](https://root.cern/js/latest/?file=https://root.cern/files/sld.root&item=sld;1&opt=dflt_colors;clipxyz)

Together with geometry one could display tracks (TEveTrack) and hits (TEvePointSet) objects.
Either one do it interactively by drag and drop, or superimpose drawing with `+` sign like:

- [item=simple_alice.json.gz+tracks_hits.root/tracks+tracks_hits.root/hits](https://root.cern/js/latest/?nobrowser&json=../files/geom/simple_alice.json.gz&file=../files/geom/tracks_hits.root&item=simple_alice.json.gz+tracks_hits.root/tracks+tracks_hits.root/hits)



## Reading ROOT files from other servers

In principle, one could open any ROOT file placed in the web, providing the full URL to it like:

- <https://jsroot.gsi.de/latest/?file=https://root.cern/js/files/hsimple.root&item=hpx>

But one should be aware of [Same-origin policy](https://en.wikipedia.org/wiki/Same-origin_policy), when the browser blocks requests to files from domains other than current web page.
To enable CORS on Apache web server, hosting ROOT files, one should add following lines to `.htaccess` file:

    <IfModule mod_headers.c>
      <FilesMatch "\.root">
         Header set Access-Control-Allow-Origin "*"
         Header set Access-Control-Allow-Headers "range"
         Header set Access-Control-Expose-Headers "content-range,content-length,accept-ranges"
         Header set Access-Control-Allow-Methods "HEAD,GET"
      </FilesMatch>
    </IfModule>

More details about configuring of CORS headers can be found [here](https://developer.mozilla.org/en/http_access_control).

Alternative - enable CORS requests in the browser. It can be easily done with [CORS Everywhere plugin](https://addons.mozilla.org/de/firefox/addon/cors-everywhere/) for the Firefox browser or [Allow CORS plugin](https://chrome.google.com/webstore/detail/allow-control-allow-origi/nlfbmbojpeacfghkpbjhddihlkkiljbi?hl=en) for the Chrome browser.


Next solution - install JSROOT on the server hosting ROOT files. In such configuration JSROOT does not issue CORS requests, therefore server and browsers can be used with their default settings. A simplified variant of such solution - copy only the top index.htm file from JSROOT package and specify the full path to JSRootCore.js script like:

    ...
    <script type="text/javascript" src="https://root.cern/js/latest/scripts/JSRootCore.js?gui"></script>
    ...

In the `<div>` element with "simpleGUI" id one can specify many custom parameters, which are allowed in the URL string:

    ...
     <div id="simpleGUI" path="files/path" files="userfile1.root;subdir/usefile2.root">
       loading scripts ...
     </div>
    ...


## Reading local ROOT files

JSROOT can read files from local file system using HTML5 FileReader functionality.
Main limitation here - user should interactively select files for reading.
There is button __"..."__ on the main JSROOT page, which starts file selection dialog.
If valid ROOT file is selected, JSROOT will be able to normally read content of such file.

One could try to invoke such dialog with "localfile" parameter in URL string:

   - <https://root.cern/js/latest/?localfile>

It could happen, that due to security limitations automatic popup will be blocked.

For debuging purposes one can install JSROOT on local file system and let read ROOT files from the same location. Like:

   - <file:///home/user/jsroot/index.htm?file=hsimple.root&item=hpx>

But this works only with Firefox.



## JSROOT with THttpServer

THttpServer provides http access to objects from running ROOT application.
JSROOT is used to implement the user interface in the web browsers.

The layout of the main page coming from THttpServer is similar to the file I/O one.
One could browse existing items and display them. A snapshot of running
server can be seen on the [demo page](https://root.cern/js/latest/httpserver.C/).

One could also specify similar URL parameters to configure the displayed items and drawing options.

It is also possible to display one single item from the THttpServer server like:

<https://root.cern/js/latest/httpserver.C/Files/job1.root/hpxpy/draw.htm?opt=colz>


##  Data monitoring with JSROOT

### Monitoring with http server

The best possibility to organize the monitoring of data from a running application
is to use THttpServer. In such case the client can always access the latest
changes and request only the items currently displayed in the browser.
To enable monitoring, one should activate the appropriate checkbox or
provide __monitoring__ parameter in the URL string like:

<https://root.cern/js/latest/httpserver.C/Files/job1.root/hprof/draw.htm?monitoring=1000>

The parameter value is the update interval in milliseconds.


### JSON file-based monitoring

Solid file-based monitoring (without integration of THttpServer into application) can be
implemented in JSON format. There is the [TBufferJSON](https://root.cern/doc/master/classTBufferJSON.html) class,
which is capable to convert any (beside TTree) ROOT object into JSON. Any ROOT application can use such class to
create JSON files for selected objects and write such files in a directory,
which can be accessed via web server. Then one can use JSROOT to read such files and display objects in a web browser.

There is a demonstration page showing such functionality: <https://root.cern/js/latest/demo/demo.htm>.
This demo page reads in cycle 20 json files and displays them.

If one has a web server which already provides such JSON file, one could specify the URL to this file like:

<https://root.cern/js/latest/demo/demo.htm?addr=../httpserver.C/Canvases/c1/root.json.gz>

Here the same problem with [Cross-Origin Request](https://developer.mozilla.org/en/http_access_control) can appear. If the web server configuration cannot be changed, just copy JSROOT to the web server itself.


### Binary file-based monitoring (not recommended)

Theoretically, one could use binary ROOT files to implement monitoring.
With such approach, a ROOT-based application creates and regularly updates content of a ROOT file, which can be accessed via normal web server. From the browser side, JSROOT could regularly read the specified objects and update their drawings. But such solution has three major caveats.

First of all, one need to store the data of all objects, which only potentially could be displayed in the browser. In case of 10 objects it does not matter, but for 1000 or 100000 objects this will be a major performance penalty. With such big amount of data one will never achieve higher update rate.

The second problem is I/O. To read the first object from the ROOT file, one need to perform several (about 5) file-reading operations via http protocol.
There is no http file locking mechanism (at least not for standard web servers),
therefore there is no guarantee that the file content is not changed/replaced between consequent read operations. Therefore, one should expect frequent I/O failures while trying to monitor data from ROOT binary files. There is a workaround for the problem - one could load the file completely and exclude many partial I/O operations by this. To achieve this with JSROOT, one should add "+" sign at the end of the file name. Of course, it only could work for small files.

If somebody still wants to use monitoring of data from ROOT files, could try link like:

- <https://root.cern/js/latest/?nobrowser&file=../files/hsimple.root+&item=hpx;1&monitoring=2000>

In this particular case, the histogram is not changing.


## JSROOT API

JSROOT can be used in arbitrary HTML pages and disaplay data, produced without ROOT-based applications.

Many different examples of JSROOT API usage can be found on [JSROOT API examples](https://root.cern/js/latest/api.htm) page.


### Scripts loading

Before JSROOT can be used, all appropriate scripts should be loaded.
HTML pages where JSROOT is used should include the JSRootCore.js script.
The `<head>` section of the HTML page should have the following line:

    <script type="text/javascript" src="https://root.cern/js/latest/scripts/JSRootCore.js?2d"></script>

Here, the default location of JSROOT is specified. One could have a local copy on the file system or on a private web server. When JSROOT is used with THttpServer, the address looks like:

    <script type="text/javascript" src="http://your_root_server:8080/jsrootsys/scripts/JSRootCore.js?2d"></script>

In URL string with JSRootCore.js script one can specify which JSROOT functionality should be loaded:

    + '2d' normal drawing for objects like TH1/TCanvas/TGraph
    + 'more2d' more classes for 2D drawing like TH2/TF1/TEllipse
    + '3d' 3D drawing for 2D/3D histograms
    + 'geo' 3D drawing of TGeo classes
    + 'io' binary file I/O
    + 'tree' TTree functionality
    + 'math' advanced mathemathical functions
    + 'mathjax' loads MathJax.js and use it for latex output
    + 'gui' default gui for offline/online applications
    + 'load' name of user script(s) to load
    + 'onload' name of function to call when scripts loading completed

For instance, to load functionality with normal 2D graphics and binary ROOT files support, one should specify:

    <script type="text/javascript" src="https://root.cern/js/latest/scripts/JSRootCore.min.js?2d&io"></script>

One could use minified version of all scripts (as shown in example) - this reduce page loading time significantly.

When JSROOT installed with bower package manager, one could re-use basic libraries like `d3.js` or `three.js` from bower itself. For that one should add `bower` into URL:

    <script type="text/javascript" src="vendor/jsroot/scripts/JSRootCore.js?bower&2d&io"></script>

Bower support will be automatically enabled when script path conatin __"bower_components/jsroot/"__ string.



### Use of JSON

It is strongly recommended to use JSON when communicating with ROOT application.
THttpServer provides a JSON representation for every registered object with an url address like:

    http://your_root_server:8080/Canvases/c1/root.json

Such JSON representation generated using the [TBufferJSON](https://root.cern/doc/master/classTBufferJSON.html) class. One could create JSON file for any ROOT object directly, just writing in the code:

    ...
    obj->SaveAs("file.json");
    ...

To access data from a remote web server, it is recommended to use the [XMLHttpRequest](http://en.wikipedia.org/wiki/XMLHttpRequest) class. JSROOT provides a special method to create such object and properly handle it in different browsers.
For receiving JSON from a server one could use following code:

    var req = JSROOT.NewHttpRequest("http://your_root_server:8080/Canvases/c1/root.json", 'object', userCallback);
    req.send(null);

In the callback function one gets JavaScript object (or null in case of failure)

If JSON string was obtained by different method, it should be parsed with:

    var obj = JSROOT.parse(json_string);



### Objects drawing

After an object has been created, one can directly draw it. If HTML page has `<div>` element:

    ...
    <div id="drawing"></div>
    ...

One could use the JSROOT.draw function:

    JSROOT.draw("drawing", obj, "colz");

The first argument is the id of the HTML div element, where drawing will be performed. The second argument is the object to draw and the third one is the drawing option.

Here is complete [running example](https://root.cern/js/latest/api.htm#custom_html_read_json) ans [source code](https://github.com/linev/jsroot/blob/master/demo/read_json.htm):

    var filename = "https://root.cern/js/files/th2ul.json.gz";
    JSROOT.NewHttpRequest(filename, 'object', function(obj) {
       JSROOT.draw("drawing", obj, "lego");
    }).send();

One is also able to update the drawing with a new version of the object:

    // after some interval request object again
    JSROOT.redraw("drawing", obj2, "colz");

The JSROOT.redraw() function will call JSROOT.draw if the drawing was not performed before.

In the case when changing of HTML layout leads to resize of element with JSROOT drawing,
one should call JSROOT.resize() to let JSROOT adjust drawing size. One should do:

    JSROOT.resize("drawing");

 As second argument one could specify exact size for draw elements like:

    JSROOT.resize("drawing", { width: 500, height: 200 } );

To correctly cleanup JSROOT drawings from HTML element, one should call:

    JSROOT.cleanup("drawing");


### File API

JSROOT defines the JSROOT.TFile class, which can be used to access binary ROOT files.
One should always remember that all I/O operations are asynchronous in JSROOT.
Therefore, callback functions are used to react when the I/O operation completed.
For example, reading an object from a file and displaying it will look like:

    var filename = "https://root.cern/js/files/hsimple.root";
    JSROOT.OpenFile(filename, function(file) {
       file.ReadObject("hpxpy;1", function(obj) {
          JSROOT.draw("drawing", obj, "colz");
       });
    });

Here is [running example](https://root.cern/js/latest/api.htm#custom_html_read_root_file) and [source code](https://github.com/linev/jsroot/blob/master/demo/read_file.htm)



### TTree API

Simple TTree::Draw operation can be performed with following code:


    var filename = "https://root.cern/js/files/hsimple.root";
    JSROOT.OpenFile(filename, function(file) {
       file.ReadObject("ntuple;1", function(obj) {
          JSROOT.draw("drawing", obj, "px:py::pz>5");
       });
    });

To get access to selected branches, one should use TSelector class:


    var filename = "https://root.cern/js/files/hsimple.root";
    JSROOT.OpenFile(filename, function(file) {
       file.ReadObject("ntuple;1", function(tree) {

          var selector = new JSROOT.TSelector();

          selector.AddBranch("px");
          selector.AddBranch("py");

          var cnt = 0, sumpx = 0, sumpy = 0;

          selector.Begin = function() {
             // function called before reading of TTree starts
          }

          selector.Process = function() {
             // function called for every entry
             sumpx += this.tgtobj.px;
             sumpy += this.tgtobj.py;
             cnt++;
          }

          selector.Terminate = function(res) {
             if (!res || (cnt===0)) return;
             var meanpx = sumpx/cnt, meanpy = sumpy/cnt;
             console.log('Results', meanpx, meanpy);
          }

          tree.Process(selector);

       });
    });

Here is [running example](https://root.cern/js/latest/api.htm#ttree_tselector) and [source code](https://github.com/linev/jsroot/blob/master/demo/read_tree.htm)

This examples shows how read TTree from binary file and create JSROOT.TSelector object.
Logically it is similar to original TSelector class - for every read entry TSelector::Process() method is called.
Selected branches can be accessed from **tgtobj** data member. At the end of tree reading TSelector::Terminate() method
will be called.

As second parameter of tree.Process() function one could provide object with arguments

    var args = { numentries: 1000, firstentry: 500 };
    tree.Process(selector, args);

