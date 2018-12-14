sap.ui.define(['sap/ui/core/mvc/Controller',
               'sap/ui/model/json/JSONModel',
               'sap/ui/layout/Splitter',
               'sap/ui/layout/SplitterLayoutData',
               "sap/ui/core/ResizeHandler",
               "eve/GeomDraw"
],function(Controller, JSONModel, Splitter, SplitterLayoutData, ResizeHandler, GeomDraw) {
   "use strict";

   return Controller.extend("eve.Geom", {
      onInit: function () {
         
         console.log('GEOM CONTROLLER INIT');
         
         this.websocket = this.getView().getViewData().conn_handle;
         
         this.websocket.SetReceiver(this);
         this.websocket.Connect();
         
         this.data = {
               Nodes: [
                {
                  title: "1",
                  childs: [ { title: "1.1" } , { title: "1.2" },  { title: "1.3" } ]
                },
                {
                   title: "2",
                   childs: [ { title: "2.1" } , { title: "2.2" },  { title: "2.3" } ]
                 },
                 {
                    title: "3",
                    childs: [ { title: "3.1" } , { title: "3.2" },  { title: "3.3" } ]
                 }
              ]
         };
         
         this.descr = null; // description object from 
         
         
         this.model = new JSONModel(this.data);
         this.getView().setModel(this.model);
         
         // PART 2: instantiate Control and place it onto the page

         var myControl = new GeomDraw({color:"#f00"});
         //myControl.placeAt("content");

         // ok, add another instance...:
         //new my.ColorBox({color:"green"}).placeAt("content");
         
         
         this.getView().byId("mainSplitter").addContentArea(myControl);
      },
      
      /** Called when data comes via the websocket */
      OnWebsocketMsg: function(handle, msg, offset) {

         if (typeof msg != "string") {
            // console.log('ArrayBuffer size ',
            // msg.byteLength, 'offset', offset);
            return;
         }

         console.log("msg len=", msg.length, " txt:", msg.substr(0,70), "...");

         // this if first message
         if (!this.descr) {
            this.descr = JSROOT.parse(msg);
            this.buildTree();
            
         } else {
         }
      },
      
      buildNode: function(indx) {
         var node = this.tree_nodes[indx];
         if (node) return node;
         
         node = { title: "any" };
         this.tree_nodes[indx] = node;
         return node;
      },
      
      buildTree: function() {
         if (!this.descr || !this.descr.fDesc) return;
         
         this.clones = new JSROOT.GEO.ClonedNodes(null, this.descr.fDesc);
        
         this.tree_nodes = [];
         
         this.data.Nodes = [ this.buildNode(0) ];
         
         console.log('data', this.data.Nodes);
         
         this.model.refresh();
         
      },

      OnWebsocketClosed: function() {
         // when connection closed, close panel as well
         console.log('CLOSE WINDOW WHEN CONNECTION CLOSED');
         
         if (window) window.close();
      },
      
      
      
      onAfterRendering: function(){
      },

      onToolsMenuAction : function (oEvent) {

         var item = oEvent.getParameter("item");

         switch (item.getText()) {
            case "GED Editor": this.getView().byId("Summary").getController().toggleEditor(); break;
         }
      },
      
      showHelp : function(oEvent) {
         alert("User support: root-webgui@cern.ch");
      },
      
      showUserURL : function(oEvent) {
         sap.m.URLHelper.redirect("https://github.com/alja/jsroot/blob/dev/eve7.md", true);
      }
   });
});
