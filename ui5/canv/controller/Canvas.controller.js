sap.ui.define([
   'jquery.sap.global',
   'sap/ui/core/mvc/Controller',
   'sap/ui/core/Component',
   'sap/ui/model/json/JSONModel',
   'sap/ui/core/mvc/XMLView',
   'sap/ui/core/Fragment',
   'sap/m/MessageToast',
   'sap/m/Dialog',
   'sap/m/List',
   'sap/m/InputListItem',
   'sap/m/Input',
   'sap/m/Button',
   'sap/m/Label',
   'sap/ui/layout/Splitter',
   'sap/ui/layout/SplitterLayoutData',
   'sap/ui/unified/Menu',
   'sap/ui/unified/MenuItem'
], function (jQuery, Controller, Component, JSONModel, XMLView, Fragment, MessageToast, Dialog, List, InputListItem, Input, Button, Label, Splitter, SplitterLayoutData, Menu, MenuItem) {
   "use strict";

   var CController = Controller.extend("rootui5.canv.controller.Canvas", {
      onInit : function() {
         this._Page = this.getView().byId("CanvasMainPage");

         var id = this.getView().getId();
         console.log("Initialization CANVAS id = " + id);

         this.bottomVisible = false;

         var model = new JSONModel({ GedIcon: "", StatusIcon: "", ToolbarIcon: "", TooltipIcon: "sap-icon://accept",
                                     StatusLbl1:"", StatusLbl2:"", StatusLbl3:"", StatusLbl4:"" });
         this.getView().setModel(model);
         
         var vd = this.getView().getViewData();
         var cp = vd ? vd.canvas_painter : null;

         if (!cp) cp = Component.getOwnerComponentFor(this.getView()).getComponentData().canvas_painter;

         if (cp) {

            this.getView().byId("MainPanel").getController().setPainter(cp);

            cp.executeObjectMethod = this.executeObjectMethod.bind(this);

            // overwriting method of canvas with standalone handling of GED
            cp.ActivateGed = this.openuiActivateGed.bind(this);
            cp.RemoveGed = this.cleanupIfGed.bind(this);
            cp.HasGed = this.isGedEditor.bind(this);

            cp.HasEventStatus = this.isStatusShown.bind(this);
            cp.ActivateStatusBar = this.toggleShowStatus.bind(this);
            cp.ShowCanvasStatus = this.showCanvasStatus.bind(this); // used only for UI5, otherwise global func
            cp.ShowMessage = this.showMessage.bind(this);
            cp.ShowSection = this.showSection.bind(this);

            cp.ShowUI5ProjectionArea = this.showProjectionArea.bind(this);
            cp.DrawInUI5ProjectionArea = this.drawInProjectionArea.bind(this);

            cp.ShowUI5Panel = this.showPanelInLeftArea.bind(this);
         }

         // this.toggleGedEditor();
      },

      executeObjectMethod: function(painter, method, menu_obj_id) {

         if (method.fArgs!==undefined) {
            this.showMethodsDialog(painter, method, menu_obj_id);
            return true;
         }

         if (method.fName == "Inspect") {
            painter.ShowInspector();
            return true;
         }

         if (method.fName == "FitPanel") {
            this.showLeftArea("FitPanel");
            return true;
         }

         if (method.fName == "Editor") {
            this.openuiActivateGed(painter);
            return true;
         }

         return false; // not processed

      },

      openuiActivateGed: function(painter, kind, mode) {
         // function used to activate GED in full canvas

         this.showGeEditor(true);

         var canvp = this.getCanvasPainter();

         canvp.SelectObjectPainter(painter);

         if (typeof canvp.ProcessChanges == 'function')
            canvp.ProcessChanges("sbits", canvp);
      },

      getCanvasPainter : function(also_without_websocket) {
         var elem = this.getView().byId("MainPanel");

         var p = elem ? elem.getController().getPainter() : null;

         return (p && (p._websocket || also_without_websocket)) ? p : null;
      },

      closeMethodDialog : function(painter, method, menu_obj_id) {

         var args = "";

         if (method) {
            var cont = this.methodDialog.getContent();

            var items = cont[0].getItems();

            if (method.fArgs.length !== items.length)
               alert('Mismatch between method description' + method.fArgs.length + ' and args list in dialog ' + items.length);

            // console.log('ITEMS', method.fArgs.length, items.length);

            for (var k=0;k<method.fArgs.length;++k) {
               var arg = method.fArgs[k];
               var value = items[k].getContent()[0].getValue();

               if (value==="") value = arg.fDefault;

               if ((arg.fTitle=="Option_t*") || (arg.fTitle=="const char*")) {
                  // check quotes,
                  // TODO: need to make more precise checking of escape characters
                  if (!value) value = '""';
                  if (value[0]!='"') value = '"' + value;
                  if (value[value.length-1] != '"') value += '"';
               }

               args += (k>0 ? "," : "") + value;
            }
         }

         this.methodDialog.close();
         this.methodDialog.destroy();

         if (painter && method && args) {

            if (painter.ExecuteMenuCommand(method, args)) return;
            var exec = method.fExec;
            if (args) exec = exec.substr(0,exec.length-1) + args + ')';
            // invoked only when user press Ok button
            console.log('execute method for object ' + menu_obj_id + ' exec= ' + exec);

            var canvp = this.getCanvasPainter();

            if (canvp)
               canvp.SendWebsocket('OBJEXEC:' + menu_obj_id + ":" + exec);
         }
      },

      showMethodsDialog : function(painter, method, menu_obj_id) {

         // TODO: deliver class name together with menu items
         method.fClassName = painter.GetClassName();
         if ((menu_obj_id.indexOf("#x")>0) || (menu_obj_id.indexOf("#y")>0) || (menu_obj_id.indexOf("#z")>0)) method.fClassName = "TAxis";

         var items = [];

         for (var n=0;n<method.fArgs.length;++n) {
            var arg = method.fArgs[n];
            arg.fValue = arg.fDefault;
            if (arg.fValue == '\"\"') arg.fValue = "";
            var item = new InputListItem({
               label: arg.fName + " (" +arg.fTitle + ")",
               content: new Input({ placeholder: arg.fName, value: arg.fValue })
            });
            items.push(item);
         }

         this.methodDialog = new Dialog({
            title: method.fClassName + '::' + method.fName,
            content: new List({
                items: items
             }),
             beginButton: new Button({
               text: 'Cancel',
               press: this.closeMethodDialog.bind(this)
             }),
             endButton: new Button({
               text: 'Ok',
               press: this.closeMethodDialog.bind(this, painter, method, menu_obj_id)
             })
         });

         // this.getView().getModel().setProperty("/Method", method);
         //to get access to the global model
         // this.getView().addDependent(this.methodDialog);

         this.methodDialog.addStyleClass("sapUiSizeCompact");

         this.methodDialog.open();
      },

      onFileMenuAction : function (oEvent) {
         //var oItem = oEvent.getParameter("item"),
         //    sItemPath = "";
         //while (oItem instanceof sap.m.MenuItem) {
         //   sItemPath = oItem.getText() + " > " + sItemPath;
         //   oItem = oItem.getParent();
         //}
         //sItemPath = sItemPath.substr(0, sItemPath.lastIndexOf(" > "));

         var p = this.getCanvasPainter();
         if (!p) return;

         var name = oEvent.getParameter("item").getText();

         switch (name) {
            case "Close canvas": p.OnWebsocketClosed(); p.CloseWebsocket(true); break;
            case "Interrupt": p.SendWebsocket("INTERRUPT"); break;
            case "Quit ROOT": p.SendWebsocket("QUIT"); break;
            case "Canvas.png":
            case "Canvas.jpeg":
            case "Canvas.svg":
               p.SaveCanvasAsFile(name);
               break;
            case "Canvas.root":
            case "Canvas.pdf":
            case "Canvas.ps":
            case "Canvas.C":
               p.SendSaveCommand(name);
               break;
         }

         MessageToast.show("Action triggered on item: " + name);
      },

      onCloseCanvasPress : function() {
         var p = this.getCanvasPainter();
         if (p) {
            p.OnWebsocketClosed();
            p.CloseWebsocket(true);
         }
      },

      onInterruptPress : function() {
         var p = this.getCanvasPainter();
         if (p) p.SendWebsocket("INTERRUPT");
      },

      onQuitRootPress : function() {
         var p = this.getCanvasPainter();
         if (p) p.SendWebsocket("QUIT");
      },

      onReloadPress : function() {
         var p = this.getCanvasPainter();
         if (p) p.SendWebsocket("RELOAD");
      },

      isGedEditor : function() {
         return this.getView().getModel().getProperty("/LeftArea") == "Ged";
      },

      showGeEditor : function(new_state) {
         this.showLeftArea(new_state ? "Ged" : "");
      },

      cleanupIfGed: function() {
         var ged = this.getLeftController("Ged"),
             p = this.getCanvasPainter();
         if (p) p.RegisterForPadEvents(null);
         if (ged) ged.cleanupGed();
         if (p && p.ProcessChanges) p.ProcessChanges("sbits", p);
      },

      getLeftController: function(name) {
         if (this.getView().getModel().getProperty("/LeftArea") != name) return null;
         var split = this.getView().byId("MainAreaSplitter");
         return split ? split.getContentAreas()[0].getController() : null;
      },

      toggleGedEditor : function() {
         this.showGeEditor(!this.isGedEditor());
      },

      showPanelInLeftArea: function(panel_name, panel_handle, call_back) {

         var split = this.getView().byId("MainAreaSplitter");
         var curr = this.getView().getModel().getProperty("/LeftArea");
         if (!split || (curr === panel_name)) return JSROOT.CallBack(call_back, false);

         // first need to remove existing
         if (curr) {
            this.cleanupIfGed();
            split.removeContentArea(split.getContentAreas()[0]);
         }

         this.getView().getModel().setProperty("/LeftArea", panel_name);
         this.getView().getModel().setProperty("/GedIcon", (panel_name=="Ged") ? "sap-icon://accept" : "");

         if (!panel_handle || !panel_name) return JSROOT.CallBack(call_back, false);

         var oLd = new SplitterLayoutData({
            resizable : true,
            size      : "250px"
         });

         var viewName = panel_name;
         if (viewName.indexOf(".") < 0) viewName = "rootui5.canv.view." + panel_name;

         XMLView.create({
            viewName: viewName,
            viewData: { handle: panel_handle, masterPanel: this },
            layoutData: oLd,
            height: (panel_name == "Panel") ? "100%" : undefined
         }).then(function(oView) {
            split.insertContentArea(oView, 0);
            JSROOT.CallBack(call_back, true);
         });

      },

      // TODO: sync with showPanelInLeftArea, it is more or less same
      showLeftArea: function(panel_name, call_back) {
         var split = this.getView().byId("MainAreaSplitter");
         var curr = this.getView().getModel().getProperty("/LeftArea");
         if (!split || (curr === panel_name))
            return JSROOT.CallBack(call_back, null);

         // first need to remove existing
         if (curr) {
            this.cleanupIfGed();
            split.removeContentArea(split.getContentAreas()[0]);
         }

         this.getView().getModel().setProperty("/LeftArea", panel_name);
         this.getView().getModel().setProperty("/GedIcon", (panel_name=="Ged") ? "sap-icon://accept" : "");

         if (!panel_name) return JSROOT.CallBack(call_back, null);

         var oLd = new SplitterLayoutData({
            resizable: true,
            size: "250px"
         });

         var canvp = this.getCanvasPainter();

         var viewName = "rootui5.canv.view." + panel_name;
         if (panel_name == "FitPanel") viewName = "rootui5.fitpanel.view.FitPanel";

         XMLView.create({
            viewName: viewName,
            viewData: { masterPanel: this },
            layoutData: oLd,
            height: (panel_name == "Panel") ? "100%" : undefined
         }).then(function(oView) {

            split.insertContentArea(oView, 0);

            if (panel_name === "Ged") {
               var ged = oView.getController();
               if (canvp && ged && (typeof canvp.RegisterForPadEvents == "function")) {
                  canvp.RegisterForPadEvents(ged.padEventsReceiver.bind(ged));
                  canvp.SelectObjectPainter(canvp);
               }
            }

            JSROOT.CallBack(call_back, oView.getController());
         });
      },

      getBottomController : function(can, call_back) {
         if (!this.bottomVisible) return null;
         var split = this.getView().byId("MainAreaSplitter");
         var cont = split.getContentAreas();
         var vsplit = cont[cont.length-1];
         var vcont = vsplit.getContentAreas();
         var bottom = vcont[vcont.length-1];
         return bottom ? bottom.getController() : null;
      },

      drawInProjectionArea : function(can, opt, call_back) {
         var ctrl = this.getBottomController();
         if (!ctrl) ctrl = this.getLeftController("Panel");

         if (ctrl && ctrl.drawObject)
            ctrl.drawObject(can, opt, call_back);
         else
            JSROOT.CallBack(call_back, null);
      },

      showProjectionArea : function(kind, call_back) {
         this.showBottomArea((kind == "X"), function(bottom) {
            this.showLeftArea(kind == "Y" ? "Panel" : "", function(left) {

               var ctrl = bottom || left;

               if (!ctrl || ctrl.getView().getDomRef())
                  return JSROOT.CallBack(call_back, !!ctrl);

               if (ctrl.rendering_perfromed) console.log('Rendering already done');

               // FIXME: one should have much easier way to get callback when rendering done
               ctrl.after_render_callback = call_back;
            });
         }.bind(this));
      },

      showBottomArea : function(is_on, call_back) {

         if (this.bottomVisible == is_on)
            return JSROOT.CallBack(call_back, this.getBottomController());

         var split = this.getView().byId("MainAreaSplitter");

         if (!split) return JSROOT.CallBack(call_back, null);

         var cont = split.getContentAreas();

         this.bottomVisible = !this.bottomVisible;

         if (this.bottomVisible == false) {
            // vertical splitter exists - toggle it

            var vsplit = cont[cont.length-1];
            var main = vsplit.removeContentArea(0);
            vsplit.destroyContentAreas();
            split.removeContentArea(vsplit);
            split.addContentArea(main);
            return JSROOT.CallBack(call_back, null);
         }

         // remove panel with normal drawing
         split.removeContentArea(cont[cont.length-1]);
         var vsplit = new Splitter({orientation: "Vertical"});

         split.addContentArea(vsplit);

         vsplit.addContentArea(cont[cont.length-1]);

         var oLd = new SplitterLayoutData({
            resizable : true,
            size      : "200px"
         });

         XMLView.create({
            viewName : "rootui5.canv.view.Panel",
            layoutData: oLd,
            height: "100%"
         }).then(function(oView) {
            vsplit.addContentArea(oView);
            JSROOT.CallBack(call_back, oView.getController());
         });

      },

      showCanvasStatus : function (text1,text2,text3,text4) {
         var model = this.getView().getModel();
         model.setProperty("/StatusLbl1", text1);
         model.setProperty("/StatusLbl2", text2);
         model.setProperty("/StatusLbl3", text3);
         model.setProperty("/StatusLbl4", text4);
      },

      isStatusShown : function() {
         return this._Page.getShowFooter();
      },

      toggleShowStatus : function(new_state) {
         if (new_state === undefined) new_state = !this.isStatusShown();

         this._Page.setShowFooter(new_state);
         this.getView().getModel().setProperty("/StatusIcon", new_state ? "sap-icon://accept" : "");

         var canvp = this.getCanvasPainter();
         if (canvp) canvp.ProcessChanges("sbits", canvp);
      },

      toggleToolBar : function(new_state) {
         if (new_state === undefined) new_state = !this.getView().getModel().getProperty("/ToolbarIcon");

         this._Page.setShowSubHeader(new_state);

         this.getView().getModel().setProperty("/ToolbarIcon", new_state ? "sap-icon://accept" : "");
      },

      toggleToolTip : function(new_state) {
         if (new_state === undefined) new_state = !this.getView().getModel().getProperty("/TooltipIcon");

         this.getView().getModel().setProperty("/TooltipIcon", new_state ? "sap-icon://accept" : "");

         var p = this.getCanvasPainter(true);
         if (p) p.SetTooltipAllowed(new_state);
      },

      setShowMenu: function(new_state) {
         this._Page.setShowHeader(new_state);
      },

      onViewMenuAction: function (oEvent) {

         var item = oEvent.getParameter("item");

         switch (item.getText()) {
            case "Editor": this.toggleGedEditor(); break;
            case "Event statusbar": this.toggleShowStatus(); break;
            case "Toolbar": this.toggleToolBar(); break;
            case "Tooltip info": this.toggleToolTip(); break;
         }
      },

      onToolsMenuAction : function(oEvent) {
         var item = oEvent.getParameter("item"),
             name = item.getText();

         if (name != "Fit panel") return;

         var curr = this.getView().getModel().getProperty("/LeftArea");

         this.showLeftArea(curr == "FitPanel" ? "" : "FitPanel");
      },

      showMessage : function(msg) {
         MessageToast.show(msg);
      },

      showSection : function(that, on) {
         // this function call when section state changed from server side
         switch(that) {
            case "Menu": this.setShowMenu(on); break;
            case "StatusBar": this.toggleShowStatus(on); break;
            case "Editor": this.showGeEditor(on); break;
            case "ToolBar": this.toggleToolBar(on); break;
            case "ToolTips": this.toggleToolTip(on); break;
         }
      }
   });

   return CController;

});
