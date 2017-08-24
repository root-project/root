sap.ui.define([
   'jquery.sap.global',
	'sap/ui/core/mvc/Controller',
   'sap/ui/model/json/JSONModel',
   'sap/m/MessageToast',
   'sap/m/Dialog',
   'sap/m/List',
   'sap/m/InputListItem',
   'sap/m/Input',
   'sap/m/Button',
   'sap/m/Label',
   'sap/ui/layout/SplitterLayoutData',
   'sap/ui/unified/Menu',
   'sap/ui/unified/MenuItem'

], function (jQuery, Controller, JSONModel, MessageToast, Dialog, List, InputListItem, Input, Button, Label, SplitterLayoutData, Menu, MenuItem) {
	"use strict";

	var CController = Controller.extend("sap.ui.jsroot.controller.Canvas", {
		onInit : function() {
		   this._Page = this.getView().byId("CanvasMainPage");

         var model = new JSONModel({ GedIcon: "", StatusIcon: "",
                                     StatusLbl1:"", StatusLbl2:"", StatusLbl3:"", StatusLbl4:"" });
         this.getView().setModel(model);

		   // this.toggleGedEditor();
		},

		getCanvasPainter : function(also_without_websocket) {
         var elem = this.getView().byId("jsroot_canvas");

         var p = elem ? elem.getController().canvas_painter : null;

         return (p && (p._websocket || also_without_websocket)) ? p : null;
		},

		closeMethodDialog : function(method, call_back) {

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
         JSROOT.CallBack(call_back, args);
		},

		showMethodsDialog : function(method, call_back) {
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
//              items: {
//                 path: '/Method/fArgs',
//                 template: new InputListItem({
//                    label: "{fName} ({fTitle})",
//                    content: new Input({placeholder: "{fName}", value: "{fValue}" })
//                 })
//              }
             }),
             beginButton: new Button({
               text: 'Cancel',
               press: this.closeMethodDialog.bind(this, null, null)
             }),
             endButton: new Button({
               text: 'Ok',
               press: this.closeMethodDialog.bind(this, method, call_back)
             })
         });

         // this.getView().getModel().setProperty("/Method", method);
         //to get access to the global model
         // this.getView().addDependent(this.methodDialog);

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

		showGeEditor : function(new_state) {
         this.showLeftArea(new_state ? "Ged" : "");
		},

		getGed : function() {
		   if (this.getView().getModel().getProperty("/LeftArea") != "Ged") return null;
		   var split = this.getView().byId("MainAreaSplitter");
         return split ? split.getContentAreas()[0].getController() : null;
		},

		toggleGedEditor : function() {
		   var new_state = this.getView().getModel().getProperty("/LeftArea") != "Ged";
		   this.showGeEditor(new_state);
		   var p = this.getCanvasPainter();
		   if (new_state && p) p.SelectObjectPainter(p);

		},

		showLeftArea : function(panel_name) {
         var split = this.getView().byId("MainAreaSplitter");
         if (!split) return;

		   var curr = this.getView().getModel().getProperty("/LeftArea");
		   if (curr == panel_name) return;

         // first need to remove existing
         if (curr) split.removeContentArea(split.getContentAreas()[0]);

         this.getView().getModel().setProperty("/LeftArea", panel_name);
         this.getView().getModel().setProperty("/GedIcon", (panel_name=="Ged") ? "sap-icon://accept" : "");

         if (!panel_name) return;

         var oLd = new SplitterLayoutData({
            resizable : true,
            size      : "250px",
            maxSize   : "500px"
         });

         var oContent = sap.ui.xmlview({
            viewName : "sap.ui.jsroot.view." + panel_name,
            layoutData: oLd
         });

         split.insertContentArea(oContent, 0);

         return oContent.getController(); // return controller of new panel
		},

	   ShowCanvasStatus : function (text1,text2,text3,text4) {
	      var model = this.getView().getModel();
	      model.setProperty("/StatusLbl1", text1);
	      model.setProperty("/StatusLbl2", text2);
	      model.setProperty("/StatusLbl3", text3);
	      model.setProperty("/StatusLbl4", text4);
      },

		isStatusShown : function() {
		   return this._Page.getShowFooter();
		},

		toggleShowStatus : function() {
		   var new_state = !this.isStatusShown();

         this._Page.setShowFooter(new_state);
         this.getView().getModel().setProperty("/StatusIcon", new_state ? "sap-icon://accept" : "");
		},

		onViewMenuAction : function (oEvent) {

         var item = oEvent.getParameter("item"),
             name = item.getText();

         if (name=="Editor") return this.toggleGedEditor();
         if (name=="Event statusbar") return this.toggleShowStatus();


         var p = this.getCanvasPainter(true);
         if (!p) return;

         var new_state = !item.getIcon();

         switch (name) {
            case "Toolbar":
               this._Page.setShowSubHeader(new_state)
               break;
            case "Tooltip info":
               p.SetTooltipAllowed(new_state);
               break;
            default: return;
         }
         item.setIcon(new_state ? "sap-icon://accept" : "");

         // MessageToast.show("Action triggered on item: " + name);
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
		}

	});


	return CController;

});
