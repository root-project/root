sap.ui.define(['sap/ui/core/mvc/Controller',
               'sap/ui/layout/Splitter',
               'sap/ui/layout/SplitterLayoutData'
],function(Controller, Splitter, SplitterLayoutData) {
   "use strict";

   return Controller.extend("eve.Main", {
      onInit: function () {
         
         console.log('MAIN CONTROLLER INIT');

         this.handle = this.getView().getViewData().conn_handle;
         this.handle.SetReceiver(this);
         this.handle.Connect();

         this.mgr = new JSROOT.EVE.EveManager();

         // method to found summary controller by ID and set manager to it
         var elem = this.byId("Summary");
         var ctrl = sap.ui.getCore().byId(elem.getId()).getController();
         ctrl.SetMgr(this.mgr);
         
         this.mgr.RegisterUpdate(this, "onManagerUpdate");
         
         
         var toolbar = this.byId("otb1");
         var btn = new sap.m.Button({
            // text: "ButtonNew",
            icon: "sap-icon://step",
            tooltip: "Get new event",
            press: this.newEvent.bind(this)
          });
         toolbar.insertContent(btn, 0);
         
      },
      
      newBtnHandler: function(evt) {
         this.newEvent();
      }, 

      getHandle: function () {
         return this.handle;
      },

      OnWebsocketMsg: function(handle, msg, offset) {

         if (typeof msg != "string") {
            // console.log('ArrayBuffer size ',
            // msg.byteLength, 'offset', offset);
            this.mgr.UpdateBinary(msg, offset);

            return;
         }

         console.log("msg len=", msg.length, " txt:", msg.substr(0,50), "...");
         
         if (msg == "$$nullbinary$$")
            return;

         var resp = JSON.parse(msg);

         if (resp && resp[0] && resp[0].content == "REveManager::DestroyElementsOf") {

            this.mgr.DestroyElements(resp);

            // this.getView().byId("Summary").getController().UpdateMgr(this.mgr);

         } else if (resp && resp[0] && resp[0].content == "REveScene::StreamElements") {

            
            this.mgr.Update(resp);
            // console.log('element',
            // this.getView().byId("Summary").getController());

            // this.getView().byId("Summary").getController().UpdateMgr(this.mgr);

            var viewers = this.mgr.FindViewers();

            // first check number of views to create
            var total_count = 0;
            for (var n=0;n<viewers.length;++n) {
               if (viewers[n].$view_created || viewers[n].$view_staged) continue;
               viewers[n].$view_staged = true; // mark view which will be created in this loop
               if (viewers[n].fRnrSelf) total_count++;
            }

            if (total_count == 0) return;

            console.log("FOUND viewers", viewers.length, "not yet exists", total_count);

            var main = this, vv = null, count = 0, sv = this.getView().byId("MainAreaSplitter");

            for (var n=0;n<viewers.length;++n) {
               var elem = viewers[n];
               var viewid = "EveViewer" + elem.fElementId;
               if (elem.$view_created || !viewers[n].fRnrSelf) continue;

               // create missing view
               elem.$view_created = true;
               delete elem.$view_staged;
               
               console.log("Creating view", viewid);
               
               count++;

               var oLd = undefined;
               if ((count == 1) && (total_count>1))
                  oLd = new SplitterLayoutData({resizable: true, size: "50%"});

               var vtype = "eve.GL";
               if (elem.fName === "Table") vtype = "eve.EveTable"; // AMT temorary solution

               var view = new JSROOT.sap.ui.xmlview({
                  id: viewid,
                  viewName: vtype,
                  viewData: { mgr: main.mgr, elementid: elem.fElementId, kind: (count==1) ? "3D" : "2D" },
                  layoutData: oLd
               });

               if (count == 1) {
                  sv.addContentArea(view);
                  continue;
               }

               if (!vv) {
                  vv = new Splitter("SecondaryViewSplitter", { orientation : "Vertical" });
                  sv.addContentArea(vv);
               }

               vv.addContentArea(view);
            }
         } else if (resp && resp.header && resp.header.content == "ElementsRepresentaionChanges") {
            this.mgr.SceneChanged(resp);
         }
      },
      /*
       * processWaitingMsg: function() { for ( var i = 0; i <
       * msgToWait.length; ++i ) {
       * this.OnWebsocketMsg(handleToWait, msgToWait[i]); }
       * handleToWait = 0; msgToWait = []; },
       */
      event: function() {
         // this._event = lst;

      },

      setMainVerticalSplitterHeight: function(){
         var mainViewHeight = document.body.clientHeight;
         var mainToolbarHeight = 49;
         var height = mainViewHeight - mainToolbarHeight;
         var splitter =  this.getView().byId("MainAreaSplitter");
         if (splitter) {
            // console.log("set splitter height >>> " , height);
            splitter.setHeight(height + "px");
         }
      },

      onAfterRendering: function(){
         var me = this;
         setTimeout(
            function(){
               $(window).on("resize", function(){
                  me.setMainVerticalSplitterHeight();
               });
               me.setMainVerticalSplitterHeight();
            }, 100);
      },

      newEvent: function() {

         console.log("NEW event ", this.mgr.childs[0].childs);
         var top = this.mgr.childs[0].childs;
         for (var i = 0; i < top.length; i++) {
            if (top[i]._typename === "EventManager") {
               console.log("calling event manager on server");
               var obj = { "mir" : "NextEvent()", "fElementId" : top[i].fElementId, "class" : top[i]._typename };
               this.handle.Send(JSON.stringify(obj));
            }
         }

      },

      onToolsMenuAction : function (oEvent) {

         var item = oEvent.getParameter("item");

         switch (item.getText()) {
            case "GED Editor": this.getView().byId("Summary").getController().toggleEditor(); break;
          //  case "Event statusbar": this.toggleShowStatus(); break;
          //  case "Toolbar": this.toggleToolBar(); break;
          //  case "Tooltip info": this.toggleToolTip(); break;
         }
      },
      
      onManagerUpdate: function() {
          this.configureToolBar();
      },

      configureToolBar: function() {
//         var top = this.mgr.childs[0].childs;
//         for (var i = 0; i < top.length; i++) {
//            if (top[i]._typename === "EventManager") {
//               console.log("toolbar id",  this.byId("newEvent"));
//               this.byId("newEvent").setVisible(true);
//            }
//         }s
      },

      showHelp : function(oEvent) {
         alert("User support: root-webgui@cern.ch");
      }
   });
});
