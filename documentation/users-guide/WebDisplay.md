# Web-based display

In this chapter discussed how web-based display can be created in the ROOT.

## Basics

Idea of web displays is implementation of user interface, which can run remotely
or locally in the web-browsers, fully decoupled from the application code.
For the communication between application and browser websockets are used.
On the server side ROOT application runs THttpServer instance which serves one or
several clients. Client is any web browser


## Creating web-window

**`ROOT::RWebWindow`** class is represent window instance, which can be displayed in the browser

```{.cpp}

auto win = ROOT::RWebWindow::Create();

// set HTML page which is showed when window displayed
win->SetDefaultPage("file:page.html"); // set

// allow unlimited user connections to the window (default only 1)
ROOT::RWebWindowsManager::SetSingleConnMode(false);
win->SetConnLimit(0);

// configure predefined geometry
win->SetGeometry(300, 300);

```

## Communication

To receive data from the browser, data callback should be assigned.
Callback is invoked when data received from the client or when connection is established.
Normally lambda function is used:

```{.cpp}

win->SetDataCallBack([](unsigned connid, const std::string &msg) {

   printf("Msg:%s from connection:%u\n", msg.c_str(), connid);

});

```

Here **connid** is unique identifier, which assign to each connection when it is established.
The connection identifier should be used when sending message to the client:

```{.cpp}

unsigned connid = win->GetConnectionId(); // first connection
std::string msg = "Hello, world";
win->Send(connid, msg);

```

## Client code

The minimal HTML/JavaScript code, which establish connection with the server, looks like:

```{html}
<!DOCTYPE HTML>
<html>
   <head>
      <meta charset="utf-8">
      <title>RWebWindow Example</title>
      <!--jsroot_importmap-->
   </head>
   <body>
      <div id="main"></div>
   </body>
   <script type="module">
      import { connectWebWindow } from 'jsroot/webwindow';
      connectWebWindow({
         receiver: {
            onWebsocketOpened(handle) {
                console.log('Connected');
                handle.send('Init msg from client');
            },
            onWebsocketMsg(handle, msg) {
                console.log('Get message ' + msg);
                document.getElementById('main').innerHTML = msg;
            },
            onWebsocketClosed(handle) {
               // when connection closed, close panel as well
               window?.close();
            }
         }
      });
   </script>
</html>

```

Here `jsroot/webwindow` module is loaded to let establish connection with ROOT application.
It includes all necessary initialization and authentication of websocket connection with the server.
Beside this part there is no limitations which HTML/JS framework should be used to organize layout and code on the client side.


## Display window

To configure web display one uses `--web=<kind>` argument when starting ROOT.
Typical values are:

- "chrome": select Google Chrome browser for interactive web display
- "firefox": select Mozilla Firefox browser for interactive web display
- "edge": select Microsoft Edge browser for interactive web display
- "qt6": uses QWebEngine from Qt6, no real http server started (requires `qt6web` component build for ROOT)
- "cef": uses Chromium Embeded Framework, no real http server started (requires `cefweb` component build for ROOT)
- "default": system default web browser, invoked with `xdg-open` on Linux, `start` on Mac or `open` on Windows
- "off": turns off the web display and comes back to normal graphics in  interactive mode.

Alternatively one can call `gROOT->SetWebDisplay("<kind>")` to specify display kind.
Same argument can be provided directly to the `RWebWindow::ShowWindow()`.

With the method `win->GetUrl()` one obtains URL string, which can be typed in the browser address string directly.


## Use window in multi-threads application

It it highly recommended to always use same `RWebWindow` instance from the same thread.
Means if one plans to use many threads running many web window,
one should create, communicate and destroy window from that thread.
In such situation one avoids need of extra locks for handling window functionality.

When `RWebWindow` object is created, it store thread `id`. That `id` always checked when communication with the client is performed.
One can change this thread `id` by `RWebWindow::AssignThreadId()` method, but it should be done in the beginning before communication
with client is started. Communication callbacks from the RWebWindow invoked always in context of thread `id`.

To let handle window communication with the client, one should regularly call `RWebWindow::Run()` method.
It runs send/receive operations and invoke user call-backs. As argument, one provides time in seconds.
Alternatively, one can invoke `RWebWindow::Sync()` which only handle queued operations and does not block thread for the long time.

There are correspondent `RWebWindow::WaitFor()` and `RWebWindow::WaitForTimed()` methods which allow not simply run window functionality
for the long time, but wait until function returns non-zero value.


## Run web-canvas in multiple threads

`TWebCanvas` in web implementation for `TCanvas` class. It is allowed to create, modify and update canvases
from different threads. But as with `RWebWindow` class, it is highly recommended to work with canvas always from the same thread.
Also objects which are drawn and modified in the canvas, should be modified from this thread.

`TWebCanvas` is automatically created for `TCanvas` when root started with `--web` argument or `gROOT->SetWebDisplay()` in invoked.
One also can use `TWebCanvas::CreateWebCanvas()` method which creates canvas with pre-configured web implementation.

Of the main importance is correct usage of `TCanvas::Update()` method. If canvas runs in special thread,
one should regularly invokde `canv->Upated()` method. This not only updates drawing of the canvas on the client side,
but also perform all necessary communication and callbacks caused by client interaction.

If there are many canvases run from the same thread, one probably can enable async mode of the `TWebCanvas` by
calling `TWebCanvas::SetAsyncMode(kTRUE)`. This let invoke updates of many canvases without blocking caller thread -
drawing will be performed asynchronousely in in the browser.


## Use of i/o threads of `civetweb` servers

`THttpServer` class of ROOT uses `civetweb` for implementing communication over http protocol.
It creates and run several threads to reply on http requests and to run websockets.
Normally recevived packets queued and processed afterwards in the thread where `RWebWindow` created and run.
All this involves mutexes locking and condition wait - which costs extra time.

DANGEROUS! To achive maximal responsiveness and minimal latency one can enable direct usage of `civetweb` threads by
calling `RWebWindow::UseServerThreads()`. In this case callbacks directly in context of `civetweb` threads at the moment when
data received from websocket. One can immediately send new message to websocket - and this will be the fastest way to communicate
with the client. But one should care about proper locking of any data which can be accessed from other application threads.

Example of such threads usage is `tutorials/visualisation/webgui/ping` example.


## Embed widgets for connection sharing

Each websocket connection consume resources of `civetweb` server - which uses one thread per connection.
If many connection are required, one can increase number of threads by specifying `WebGui.HttpThrds` parameter in rootrc file.

But it is possible to share same connection between several widgets.
One such example can be found in `tutorials/visualisation/webgui/bootstrap` demo, where canvas embed into larger bootstrap widget.
In general case following steps are performed:

- create channel on the client side and send id to the server
```javascript
    const new_conn = handle.createChannel();
    new_conn.setReceiver({ onWebsocketMsg(h, msg) { console.log(msg) } });
    handle.send('channel:' + new_conn.getChannelId());
```
- on the server one extracts channel id and embed new RWebWindow instance to main:
```cpp
   if (msg.compare(0, 8, "channel:") == 0) {
      int chid = std::stoi(msg.substr(8));
      printf("Get channel request %d\n", chid);
      auto new_window = RWebWindow::Create();
      new_window->SetDataCallBack(...);
      RWebWindow::ShowWebWindow(new_window, { main_window, connid, chid });
   }
```

From this point `new_window` instance will be able to use `new_conn` created on the client side.
This is exactly the way how `TWebCanvas` embed in bootstrap example or
many different widget kinds embed in `RBrowser`.
At some point probably one may need to increase default credits numbers for the connection by setting
`gEnv->SetValue("WebGui.ConnCredits", 100);` before starting main widget.


## Use ROOT web widgets on the remote nodes

It is advised to use the `rootssh` script with built-in port forwarding and run
the user interface on the local host with the default web browser. Like:

    [localhost] rootssh username@remotenode

As with regular `ssh`, one can specify command which should be run on remote node:

    [localhost] rootssh username@remotenode "root --web -e 'new TBrowser'"

Script automatically creates tunnel and configures several shell variables in remote session. These are:

- `ROOT_WEBGUI_SOCKET` - unix socket which will be used by ROOT `THttpServer` in webgui
- `ROOT_LISTENER_SOCKET` - unix socket which gets messages from ROOT when new web widgets are started

When on remote node in ROOT session new web widget is created, default web browser will be started on the local node and display created widget.

It is highly recommended to use `rootssh` script on public nodes like `lxplus`. Unix sockets, which are created on
the remote session, are configured with `0700` file mode - means only user allowed to access them.

One can provide `--port number` parameter to configure port number on local node and `--browser <name>` to specify
web browser executable to display web widgets. Like:

    [localhost] rootssh --port 8877 --browser chromium username@remotenode

Also any kind of normal `ssh` arguments can be specified:

    [localhost] rootssh -Y -E file.log username@remotenode

On remote node root session should be started with `root --web` argument to advise ROOT use web widgets. Like:

    [remotehost] root --web hsimple.C

[`rootssh` script](https://raw.githubusercontent.com/root-project/root/master/config/rootssh) can be download and used independently from ROOT installation - it is only required that supported ROOT version installed on remote node.


