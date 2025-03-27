# Demo of `bootstrap` usage with `RWebWindow`

Source of demo placed in repository:  https://github.com/linev/bootstrap-example

It is modified [Start Bootstrap - Freelancer](https://github.com/StartBootstrap/startbootstrap-freelancer) example
with connection to ROOT application via RWebWindow functionality.

Example also shows possiblity to embed `TWebCanvas` in such widget and provide regular `TCanvas::Update()` from ROOT application.

## Run example

    root --web=chrome tutorials/visualisation/webgui/bootstrap/webwindow.cxx


## Modify example

First checkout repository

    git clone https://github.com/linev/bootstrap-example.git


Then modify sources and rebuild

    cd bootstrap-example
    npm install
    npm run build

Finally run example again

    root --web=chrome webwindow.cxx


## Use of custom HTML/JS/CSS files with RWebWindow

When creating RWebWindow, one can specify location on main HTML file created by framework such as bootstrap.

    window->SetDefaultPage("file:" + fdir + "dist/index.html");

Typically all realted files (css, scripts, images) also located in same directory as html file (`dist/` in this case). RWebWindow automatically provides access to files in directory where main page is located and configured with `SetDefaultPage("file:" + ...) ` call.


## Loading JSROOT functionality

To be able use `import { something } from 'jsroot';` statements one should configure correspondent importmap.
Most simple way is to add `<!--jsroot_importmap-->` comment into HTML file into head section. Then `THttpServer`
automatically will insert required importmap entry. To put such comment in bootstrap-created HTML file one can simply add to `index.pug` file:

    //jsroot_importmap


## Connect with RWebWindow instance of ROOT

This done with following JS code:

```javascript
   import { connectWebWindow } from 'jsroot/webwindow';

   connectWebWindow({
      receiver: {
          onWebsocketOpened(handle) {},
          onWebsocketMsg(handle, msg, offset) {},
          onWebsocketClosed(handle) {}
      }
   });
```

This code placed in `webwindow.js` file and directly included in main HTML page.
All necessary credentials to connect with ROOT applicatoin will be provided in URL string.


## Communication with RWebWindow

In `receiver` object one specifies `onWebsocketOpened`, `onWebsocketMsg`, `onWebsocketClosed` methods which are
called when correspondent event produced by websocket connection. Example shows how binary and string data can be transferred from the ROOT applition and displayed in custom HTML element.

To send some data to the server, one should call `handle.send('Some string message');`. Only strings are supported.
On the ROOT side one assigns `window->SetDataCallBack(ProcessData);` function which is invoked when message received from the client.


## Embeding TWebCanvas

`TWebCanvas` is web-based implementation for `TCanvas`. It can be used to embed full-functional TCanvas in any web widgets - reusing existing RWebWindow connection. Following step should be done:

1. Create channel inside existing connection on the client side:

```javascript
    const conn = handle.createChannel();
```

2. Create JSROOT `TCanvasPainter` and configure to use it in with such communication channel:

```javascript
    import { TCanvasPainter } from 'jsroot';

    const dom = document.body.querySelector('#rootapp_canvas');

    const painter = new TCanvasPainter(dom, null);

    painter.online_canvas = true; // indicates that canvas gets data from running server
    painter.embed_canvas = true;  // use to indicate that canvas ui should not close complete window when closing
    painter.use_openui = false; // use by default ui5 widgets

    painter.useWebsocket(conn);
```

3. Communicate channel id to the ROOT application:

```javascript
    handle.send('channel:' + conn.getChannelId());
```

4. On server create web `TCanvas` and use received channel id to show it

```cpp
    int chid = std::stoi(arg.substr(8));
    // create and configure canvas
    auto canvas = TWebCanvas::CreateWebCanvas("name", "title");
    // access web implementation
    auto web_canv = static_cast<TWebCanvas *>(canvas->GetCanvasImp());
    // add web canvas to main connection
    web_canv->ShowWebWindow({ window, connid, chid });
```

After this canvas can be regulary modified and updated from the ROOT application.
Any changes in the canvas will be reflected on the client side.
From client ROOT functionality can be accessed via context menu.

