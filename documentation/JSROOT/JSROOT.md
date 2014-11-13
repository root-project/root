# JavaScript ROOT

JSROOT project intends to implement ROOT graphics for web browsers.
Also reading of binary ROOT files is supported.
It is successor of JSRootIO project.


## Installing JSROOT

Actual version of JSROOT can be found in ROOT repository, etc/http/ subfolder.
There all necessary files are located. Just copy them on web server or 
use directly from the file system.
Latest version of JSROOT also can be found online on 
<http://web-docs.gsi.de/~linev/js/>. 
 

## Reading ROOT files in JSROOT

[Main page](http://root.cern.ch/js/3.0/) of the JSROOT project provides 
possibility interactively open ROOT files and draw objects like histogram or canvas.

Following parameters could be specified in the URL string:

- file - name of the file, which will be automatically open with page loading
- item - item name to display 
- opt - draw option for the item 
- items - array of objects to display like ['hpx;1', 'hpxpy;1']
- opts - array of options ['any', 'colz']
- layout - can be 'collapsible', 'tabs' or 'gridNxM' where N and M are integer values
- nobrowser - do not display file browser

Examples: 

- [http://root.cern.ch/js/3.0/index.htm?file=../files/hsimple.root&item=hpx;1](http://root.cern.ch/js/3.0/index.htm?file=../files/hsimple.root&item=hpx;1)
- [http://root.cern.ch/js/3.0/index.htm?file=../files/hsimple.root&nobrowser&item=hpxpy;1&opt=colz](http://root.cern.ch/js/3.0/index.htm?file=../files/hsimple.root&nobrowser&item=hpxpy;1&opt=colz)
- [http://root.cern.ch/js/3.0/index.htm?file=../files/hsimple.root&layout=grid2x2&items=["hpx;1", "hpxpy;1"]&opts=["", "colz"]](http://root.cern.ch/js/3.0/index.htm?file=../files/hsimple.root&layout=grid2x2&items=["hpx;1", "hpxpy;1"]&opts=["", "colz"])

One can very easy integrate JSROOT graphic into other HTML pages using __iframe__ tag:  

<iframe width="600" height="500" src="http://root.cern.ch/js/3.0/index.htm?nobrowser&file=../files/hsimple.root&item=hpxpy;1&opt=colz">
</iframe>

In principle, one could open any ROOT file placed in the web, providing full URL to it like:

[http://root.cern.ch/js/3.0/?file=http://root.cern.ch/js/files/hsimple.root&item=hpx](http://root.cern.ch/js/3.0/?file=http://root.cern.ch/js/files/hsimple.root&item=hpx)

But one should be aware of [Cross-Origin Request blocking](https://developer.mozilla.org/en/http_access_control),
when web-server blocks requests coming from other domains. 

There are two solutions. Either one configures accordingly web-server or 
one copies JSROOT files to the same location where data files are.
In second case one could use server with default settings.  
 
In simple case one could copy only top index.htm file on the server and specify full path 
to JSRootCore.js script like:

    ...
    <script type="text/javascript" src="http://root.cern.ch/js/3.0/scripts/JSRootCore.js"></script>
    ...  

In such case one also could specify custom files list:

    ...
     <div id="simpleGUI" files="userfile1.root;subdir/usefile2.root">
       loading scripts ...
     </div>
    ...



## JSROOT with THttpServer

THttpServer provides http access to objects from running ROOT application.
JSROOT is used to implement user interface in the web browsers. 

Page layout of main page from THttpServer is similar to file I/O.
One could browse existing items and display them. Snapshot of running
server can be seen on the [demo page](http://root.cern.ch/js/3.0/demo/).

One also could specify similar URL parameters to configure displayed items and draw options.   

It is also possible to display single item from the THttpSerever server like

[http://root.cern.ch/js/3.0/demo/Files/job1.root/hpxpy/draw.htm?opt=colz](http://root.cern.ch/js/3.0/demo/Files/job1.root/hpxpy/draw.htm?opt=colz)


##  Data monitoring with JSROOT 

### Monitoring with http server

Best possibility to organize monitoring of data from running application
is to provide THttpServer. In such case client can always access latest 
changes and request only items really displayed in the browser. To enable monitoring,
one should activate appropriate checkbox or provide __monitoring__ parameter in the URL 
string like: 

[http://root.cern.ch/js/3.0/demo/Files/job1.root/hprof/draw.htm?monitoring=1000](http://root.cern.ch/js/3.0/demo/Files/job1.root/hprof/draw.htm?monitoring=1000)

Parameter value is update interval in milliseconds.  


### JSON file-based monitoring

Solid file-based monitoring (without integration of THttpServer into application) can be 
implemented with JSON format. There is TBufferJSON class, which capable to convert potentially 
any ROOT object (beside TTree) into JSON. Any ROOT application can use such class to create
JSON files for selected objects and write such files in the directory, which can be accessed
via web server. Then one can use JSROOT to read such files and display objects in the web browser.
           
There is demonstration page which shows such functionality:

[http://root.cern.ch/js/3.0/demo/demo.htm](http://root.cern.ch/js/3.0/demo/demo.htm)

<iframe width="500" height="300" src="http://root.cern.ch/js/3.0/demo/demo.htm">
</iframe>

This demo page reads in the cycle 20 json files and displays them.

If one have web server which already provides such JSON file, one could specify URL to this file like:
  
[http://root.cern.ch/js/3.0/demo/demo.htm?addr=Canvases/c1/root.json.gz](http://root.cern.ch/js/3.0/demo/demo.htm?addr=Canvases/c1/root.json.gz)

Here same problem with [Cross-Origin Request](https://developer.mozilla.org/en/http_access_control) can appear.
If webserver configuration cannot be changed, just copy JSROOT to the web server.


### Binary file-based monitoring (not recommended)

Theoretically one could use binary ROOT files to implement monitoring. 
With such approach ROOT-based application creates and regularly updates content of ROOT file, 
which can be accessed via normal web server. From the browser side JSROOT 
could regularly read specified objects and update drawings. But such solution has three major caveats.

First of all, one need to store data of all objects, which only potentially could be displayed in
the browser. In case of 10 objects it does not matter, for 1000 or 100000 objects this will be
major performance penalty. With such big amount of data one will never achieve frequent update rate. 

Second problem is I/O. To read first object from the ROOT file,
one need to perform several (about 7) file-reading operations via http protocol. 
There is no http file locking mechanism (at least not for standard web servers), 
therefore there is no guarantee that file content is not changed/replaced 
between consequent read operations. Therefore one should expect frequent I/O failures while
trying to monitor data from ROOT binary files. There is workaround for the problem - one could
load file completely and exclude many partial I/O operations by this. To achieve this
with JSROOT, one should add "+" sign at the end of filename.   

Third problem is limitations of ROOT I/O in JavaScript. Although it tries to fully
repeat binary I/O with streamerinfos evaluation, JavaScript ROOT I/O will never have 
100% functionality of native ROOT. Especially custom streamers are the problem for JavaScript -  
one need to implement them once again and keep synchronous with ROOT itself. And ROOT is full of
custom streamers! Therefore it is just nice feature and it is grate that one can read binary
files from the browser, but one never should rely that such I/O works for all cases.
Let say, major classes like TH1 or TGraph or TCanvas will be supported, but one will never
see full support of TTree or RooWorkspace in JavaScript.

If somebody still want to test such functionality, try monitoring parameter like:

[http://root.cern.ch/js/3.0/index.htm?nobrowser&file=../files/hsimple.root+&item=hpx;1&monitoring=2000](http://root.cern.ch/js/3.0/index.htm?nobrowser&file=../files/hsimple.root+&item=hpx;1&monitoring=2000)

In this particular case histogram is not changing.


## Stand-alone usage of JSROOT

Even without server-side application JSROOT provides nice ROOT-like graphics,
which could be used in arbitrary HTML pages. 
There is [example page](http://root.cern.ch/js/3.0/demo/example.htm),
where 2-D histograms is artificially generated and displayed.
Details about JSROOT API one can find in the next chapters. 


## JSROOT API

JSROOT consists from several libraries (.js files). They all provided in ROOT
repository and available in 'etc/http/scripts/' subfolder. 
Only central classes and functions will be documented here.
  
### Scripts loading

Before JSROOT can be used, all appropriate scripts should be loaded.
Any HTML pages, where JSROOT is used, should include JSRootCore.js script.
In the <head> sections of HTML page one should have line:

    <script type="text/javascript" src="http://root.cern.ch/js/3.0/scripts/JSRootCore.js"></script>  

Here default location of JSROOT is specified, one could have local copy on file system or private
web server. When JSROOT used with THttpServer, address looks like:

    <script type="text/javascript" src="http://your_root_server:8080/jsrootsys/scripts/JSRootCore.js"></script>  

Then one should call JSROOT.AssertPrerequisites(kind,callback,debug) methods, which accepts following arguments:

- kind - functionality to load, can be combination of:
    + '2d' normal drawings for 1D/2D objects
    + '3d' 3D drawings for 2D/3D histograms 
    + 'io' binary file I/O
- callback - call back function which is called when all necessary scripts are loaded
- debug - id of HTML element where debug information will be shown while scripts loading


JSROOT.AssertPrerequisites should be called before any other JSROOT functions can be used.
At the best one call it with `onload` handler like:

    <body onload="JSROOT.AssertPrerequisites('2d', userInitFunction, 'drawing')">
       <div id="drawing">loading...</div>
    </body>
     
Internally JSROOT.loadScript(urllist, callback, debug) method is used. It can be useful
when some other scripts should be loaded as well. __urllist__ is string with scripts names, 
separated by ';' symbol. If script name started with __$$$__ (triple dollar sign), 
script will be loaded from location relative to main JSROOT directory. 
This location detected automatically when JSRootCore.js script is loaded.  
    

### Use of JSON

It is strongly recommended to use JSON when communicating with ROOT application.
THttpServer  provides JSON representation for every registered object with url address like: 

    http://your_root_server:8080/Canvases/c1/root.json
 
One also can generate JSON representation, using [TBufferJSON](http://root.cern.ch/root/html/TBufferJSON.html) class.

To access data from remote web server, it is recommended to use 
[XMLHttpRequest](http://en.wikipedia.org/wiki/XMLHttpRequest) class. 
JSROOT provides special method to create such class and properly handle it in different browsers. 
For receiving JSON from server one could use following code:

    var req = JSROOT.NewHttpRequest("http://your_root_server:8080/Canvases/c1/root.json", 'object', userCallback);
    req.send(null);
    
In callback function one gets JavaScript object (or null in case of failure)


### Objects drawing

After object created, one can directly draw it. If somewhere in HTML pages there is element:
    
    ...
    <div id="drawing"></div>
    ...
 
One could use JSROOT.draw function:

    JSROOT.draw("drawing", obj, "colz");
    
First argument is id of the HTML element where drawing will be done.
Second argument is object to draw and third argument is draw options.
One also able to update drawing with new version of object:

    // after some interval request object again
    JSROOT.redraw("drawing", obj2, "colz");
    
JSROOT.redraw function will call JSROOT.draw if drawing was not performed before.      


### File API

JSROOT defines JSROOT.TFile class, which can be used to access binary ROOT files.

    var filename = "http://root.cern.ch/js/files/hsimple.root";
    var f = new JSROOT.TFile(filename, fileReadyCallback);

    
One should always remember that all I/O operations asynchronous in JSROOT.
Therefore callback functions used to react when I/O operation completed.
For the example read object from the file and display it will look like:

    new JSROOT.TFile(filename, function(file) {
       file.ReadObject("hpxpy;1", function(obj) {
          JSROOT.draw("drawing", obj, "colz");
       });
    });


   
## Links collection

### Loading ROOT files

* [http://root.cern.ch/js/3.0/?file=../files/hsimple.root](http://root.cern.ch/js/3.0/?file=../files/hsimple.root)
* [http://root.cern.ch/js/3.0/?file=../files/hsimple.root&item=hpxpy;1&opt=colz](http://root.cern.ch/js/3.0/?file=../files/hsimple.root&item=hpxpy;1&opt=colz)
* [http://root.cern.ch/js/3.0/?nobrowser&file=../files/hsimple.root&item=hprof;1](http://root.cern.ch/js/3.0/?nobrowser&file=../files/hsimple.root&item=hprof;1)
* [http://root.cern.ch/js/3.0/index.htm?file=../files/hsimple.root&amp;layout=grid2x2&amp;items=["hpx;1","hpxpy;1"]&amp;opts=["","colz"]](http://root.cern.ch/js/3.0/index.htm?file=../files/hsimple.root&layout=grid2x2&items=["hpx;1","hpxpy;1"]&opts=["","colz"])

### Snapshot from running THttpServer

* [http://root.cern.ch/js/3.0/demo/](http://root.cern.ch/js/3.0/demo/)
* [http://root.cern.ch/js/3.0/demo/?item=Files/job1.root/hpxpy&opt=colz&monitoring=1000](http://root.cern.ch/js/3.0/demo/?item=Files/job1.root/hpxpy&opt=colz&monitoring=1000)
* [http://root.cern.ch/js/3.0/demo/Files/job1.root/hpxpy/draw.htm?opt=colz](http://root.cern.ch/js/3.0/demo/Files/job1.root/hpxpy/draw.htm?opt=colz)
* [http://root.cern.ch/js/3.0/demo/Canvases/c1/fPrimitives/?item=hpx](http://root.cern.ch/js/3.0/demo/Canvases/c1/fPrimitives/?item=hpx)

### Standalone JSROOT usage

* [http://root.cern.ch/js/3.0/demo/example.htm](http://root.cern.ch/js/3.0/demo/example.htm)

### Demonstration of online JSROOT usage

* [http://root.cern.ch/js/3.0/demo/demo.htm](http://root.cern.ch/js/3.0/demo/demo.htm)
* [http://root.cern.ch/js/3.0/demo/demo.htm?layout=grid3x2](http://root.cern.ch/js/3.0/demo/demo.htm?layout=grid3x2)

### All supported draw options

* [Only options list](http://root.cern.ch/js/3.0/demo/drawoptions.htm?minimal)
* [List and drawing with default options](http://root.cern.ch/js/3.0/demo/drawoptions.htm?default&w=800&h=600)
* [All drawings at once (many large canvases)](http://root.cern.ch/js/3.0/demo/drawoptions.htm?all&w=1400&h=800)
   
   