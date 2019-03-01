sap.ui.define(['sap/ui/core/mvc/Controller',
               'sap/ui/layout/Splitter',
               'sap/ui/layout/SplitterLayoutData',
               'rootui5/eve7/lib/EveManager'
], function(Controller, Splitter, SplitterLayoutData, EveManager) {

   "use strict";

   return Controller.extend("rootui5.eve7.controller.Main", {
      onInit: function () {

         console.log('MAIN CONTROLLER INIT');

         this.mgr = new EveManager();

         this.mgr.UseConnection(this.getView().getViewData().conn_handle);

         // method to found summary controller by ID and set manager to it
         var elem = this.byId("Summary");
         var ctrl = sap.ui.getCore().byId(elem.getId()).getController();
         ctrl.SetMgr(this.mgr);

         this.mgr.RegisterUpdate(this, "onManagerUpdate");
      },

      getHandle: function () {
         return this.handle;
      },

      UpdateCommandsButtons: function(cmds) {
          if (!cmds || this.commands) return;

          var toolbar = this.byId("otb1");

          this.commands = cmds;
          for (var k=cmds.length-1;k>=0;--k) {
             var btn = new sap.m.Button({
                // text: "ButtonNew",
                icon: cmds[k].icon,
                tooltip: cmds[k].name,
                press: this.mgr.executeCommand.bind(this.mgr, cmds[k])
              });
             toolbar.insertContent(btn, 0);
          }
      },

      updateViewers: function() {
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
            console.log("ELEMENT", elem.fName);
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

            var vtype = "rootui5.eve7.view.GL";
            if (elem.fName === "Table") vtype = "rootui5.eve7.view.EveTable"; // AMT temporary solution

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
      },

      onManagerUpdate: function() {
         console.log("manager updated");
         this.UpdateCommandsButtons(this.mgr.commands);
         this.updateViewers();
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
               $(window).on("resize", function() {
                  me.setMainVerticalSplitterHeight();
               });
               me.setMainVerticalSplitterHeight();
            }, 100);
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

      showHelp : function(oEvent) {
         alert("User support: root-webgui@cern.ch");
      },
      showUserURL : function(oEvent) {
         sap.m.URLHelper.redirect("https://github.com/alja/jsroot/blob/dev/eve7.md", true);
      }
   });
});
