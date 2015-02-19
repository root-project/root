# HTTP server in ROOT 

The idea of THttpServer is to provide remote http access to running ROOT application and enable HTML/JavaScript user interface. Any registered object can be requested and displayed in the web browser. There are many benefits of such approach:

   * standard http interface to ROOT application  
   * no any temporary ROOT files to access data
   * user interface running in all browsers

## Starting the HTTP server

To start the http server, at any time, create an instance of the **`THttpServer`** class like: 

    serv = new THttpServer("http:8080");

This will start a civetweb-based http server on the port 8080. Then one should be able to open the address "http://localhost:8080" in any modern browser (IE, Firefox, Chrome, Opera) and browse objects created in application. By default, the server can access files, canvases, and histograms via the gROOT pointer. All those objects can be displayed with JSROOT graphics.

There is a [snapshot (frozen copy)](https://root.cern.ch/js/3.2/demo/) of such server, running in httpserver.C macro from ROOT tutorial.

<iframe width="800" height="500" src="https://root.cern.ch/js/3.2/demo/?layout=grid2x2&item=Canvases/c1">
</iframe>


## Registering objects

At any time, one could register other objects with the command:

    TGraph* gr = new TGraph(10);
    gr->SetName("gr1");
    serv->Register("graphs/subfolder", gr);

One should specify sub-folder name, where objects will be registered.
If sub-folder name does not starts with slash `/`, than top-name folder `/Objects/` will be prepended.
At any time one unregister objects:
 
    serv->Unregister(gr);

THttpServer does not take ownership over registered objects - they should be deleted by user.

If the objects content is changing in the application, one could enable monitoring flag in the browser - then objects view will be regularly updated.


## Command interface

THttpServer class provide simple interface to invoke command from web browser.
One just register command like:

    serv->RegisterCommand("/DoSomething","SomeFunction()");
     
Element with name `DoSomething` will appear in the web browser and can be clicked.
It will result in `gROOT->ProcessLineSync("SomeFunction()")` call. When registering command,
one could specify icon name which will be displayed with the command.  

    serv->RegisterCommand("/DoSomething","SomeFunction()", "/rootsys/icons/ed_execute.png");
    
In example usage of images from `$ROOTSYS/icons` directory is shown. One could prepend `button;`
string to the icon name to let browser show command as extra button. In last case one could 
hide command element from elements list:

    serv->Hide("/DoSomething");


  
## Configuring user access

By default, the http server is open for anonymous access. One could restrict the access to the server for authenticated users only. First of all, one should create a password file, using the **htdigest** utility.  

    [shell] htdigest -c .htdigest domain_name user_name

It is recommended not to use special symbols in domain or user names. Several users can be add to the ".htdigetst" file. When starting the server, the following arguments should be specified:

    root [0] new THttpServer("http:8080?auth_file=.htdigest&auth_domain=domain_name");

After that, the web browser will automatically request to input a name/password for the domain "domain_name"


## Using FastCGI interface

FastCGI is a protocol for interfacing interactive programs with a web server like 
Apache, lighttpd, Microsoft ISS and many others.

When starting THttpServer, one could specify:

    serv = new THttpServer("fastcgi:9000");

An example of configuration file for lighttpd server is:

    server.modules += ( "mod_fastcgi" )
    fastcgi.server = (
       "/remote_scripts/" =>
         (( "host" => "192.168.1.11",
            "port" => 9000,
            "check-local" => "disable",
            "docroot" => "/"
         ))
    )

In this case, to access a running ROOT application, one should open the following address in the web browser:

    http://lighttpd_hostname/remote_scripts/root.cgi/

In fact, the FastCGI interface can run in parallel to http server. One can just call: 

    serv = new THttpServer("http:8080");
    serv->CreateEngine("fastcgi:9000");

One could specify a debug parameter to be able to adjust the FastCGI configuration on the web server:

    serv->CreateEngine("fastcgi:9000?debug=1");
 
All user access will be ruled by the web server - for the moment one cannot restrict with fastcgi engine.


## Integration with existing applications

In many practical cases no change of existing code is required. Opened files (and all objects inside), existing canvas and histograms are automatically scanned by the server and will be available to the users. If necessary, any object can be registered directly to the server with a **`THttpServer::Register()`** call.

Central point of integration - when and how THttpServer get access to data from a running application. By default it is done during the gSystem->ProcessEvents() call - THttpServer uses a synchronous timer which is activated every 100 ms. Such approach works perfectly when running macros in an interactive ROOT session.

If an application runs in compiled code and does not contain gSystem->ProcessEvents() calls, two method are available. 

### Asynchronous timer

The first method is to configure an asynchronous timer for the server, like for example:

    serv->SetTimer(100, kFALSE);

Then, the timer will be activated even without any gSystem->ProcessEvents() method call. The main advantage of such method is that the application code can be used as it is. The disadvantage - there is no control when the communication between the server and the application is performed. It could happen just in-between of **`TH1::Fill()`** calls and an histogram object may be incomplete.


### Explicit call of THttpServer::ProcessRequests() method

The second method is preferable - one just inserts in the application regular calls of the THttpServer::ProcessRequests() method, like:

    serv->ProcessRequests();

In such case, one can fully disable the timer of the server:

    serv->SetTimer(0, kTRUE);




## Data access from command shell

The big advantage of the http protocol is that it is not only supported in web browsers, but also in many other applications. One could use http requests to directly access ROOT objects and data members from any kind of scripts.

If one starts a server and register an object like for example:

    root [1]  serv = new THttpServer("http:8080");
    root [2]  TNamed* n1 = new TNamed("obj", "title");
    root [3]  serv->Register("subfolder", n1);

One could request a JSON representation of such object with the command:

    [shell] wget http://localhost:8080/Objects/subfolder/obj/root.json

Then, its representation will look like:

    {
       "_typename" : "TNamed",
       "fUniqueID" : 0,
       "fBits" : 50331656,
       "fName" : "obj",
       "fTitle" : "title"
    }

One could also access the class members of an object like:

    [shell] wget http://localhost:8080/Objects/subfolder/obj/fTitle/root.json
  
The result will be: "title".

Or one could call object method:

    [shell] wget http://localhost:8080/Objects/subfolder/obj/exe.json?method=GetTitle
  
Result also will be: "title".

If the access to the server is restricted with htdigest, it is recommended to use the **curl** program since only curl correctly implements such authentication method. The command will look like:

    [shell] curl --user "accout:password" http://localhost:8080/Objects/subfolder/obj/fTitle/root.json --digest -o title.json

The following requests can be performed:

  - `root.bin`  - binary data produced by object streaming with TBufferFile
  - `root.json` - ROOT JSON representation for object and objects members
  - `root.xml`  - ROOT XML representation
  - `root.png`  - PNG image
  - `root.gif`  - GIF image
  - `root.jpeg` - JPEG image
  - `exe.json`  - method execution in the object

All data will be automatically zipped if '.gz' extension is appended. Like:

    wget http://localhost:8080/Files/hsimple.root/hpx/root.bin.gz


### Access to object data

Request root.json implemented with [TBufferJSON class](https://root.cern.ch/root/html/TBufferJSON.html)
and be used to access objects (or objects members). It generates such JSON representation,
which could be directly used in JSROOT for drawing.

For the root.json request, one could specify the 'compact' parameter, which will reduce the number of spaces and new lines without lost of data. This parameter can have values from '0' - no compression, till '3' - no spaces at all.

Comparison of different request methods with TH1 object shown in the table:

+-------------------------+------------+
| Request                 |    Size    |
+-------------------------+------------+
| root.bin                | 1658 bytes |
+-------------------------+------------+
| root.bin.gz             |  782 bytes |
+-------------------------+------------+
| root.json               | 7555 bytes |
+-------------------------+------------+
| root.json?compact=3     | 5381 bytes |
+-------------------------+------------+
| root.json.gz?compact=3  | 1207 bytes |
+-------------------------+------------+

One should take into account that json always includes names of the data fields which are not present in the binary representation. Even then, the size difference is negligible.  


### Generating images out of objects

Three kinds of requests (root.png, root.gif, root.jpeg) could be used for creating images like:  

    wget "http://localhost:8080/Files/hsimple.root/hpx/root.png?w=500&h=500&opt=lego1" -O lego1.png

For all such requests one could specify following parameters:

   - `h` - image height
   - `w` - image width
   - `opt` - draw options

Receiving such requests, THttpServer creates TCanvas and draw specified object with provided draw option. 


### Methods execution 

By default THttpServer starts in monitoring (read-only) mode and therefore forbid any methods execution.
One could specify read-write mode when server is started:
 
    serv = new THttpServer("http:8080;rw");

Or one could disable read-only mode with the call:

    serv->SetReadOnly(kFALSE);
    
'exe.json' accepts following parameters:

   - `method` - name of method to execute
   - `prototype` - method prototype (see [TClass::GetMethodWithPrototype](https://root.cern.ch/root/html/TClass.html#TClass:GetMethodWithPrototype) for details)
   - `compact` - compact parameter, used to compress return value
   - `_ret_object_` - name of the object which should be returned as result of method execution (used in TTree::Draw call)  

Example of retrieving object title:

    [shell] wget 'http://localhost:8080/Objects/subfolder/obj/exe.json?method=GetTitle' -O title.txt

Example of TTree::Draw method execution:
   
    [shell] wget 'http://localhost:8080/Files/job1.root/ntuple/exe.json?method=Draw&prototype="Option_t*"&opt="px:py>>h1"&_ret_object_=h1' -O exe.json

To get debug information about command execution, one could submit 'exe.txt' request with same arguments.


### Commands execution

If command registered to the server:

    serv->RegisterCommand("/Folder/Start", "DoSomthing()");
    
It can be invoked with cmd.json request like:

    [shell] wget 'http://localhost:8080/Folder/Start/cmd.json' -O title.txt

If command fails, `false` will be returned, otherwise result of gROOT->ProcessLineSync() execution
    