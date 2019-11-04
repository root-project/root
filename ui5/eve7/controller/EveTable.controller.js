sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/model/json/JSONModel',
   'sap/ui/commons/CheckBox',
   'sap/ui/commons/Menu',
   'sap/ui/commons/MenuItem',
   'sap/ui/core/Item',
   'sap/ui/table/Column',
   'sap/m/Input',
   'sap/m/Button',
   "sap/ui/core/ResizeHandler",
   "sap/ui/layout/VerticalLayout",
   "sap/ui/layout/HorizontalLayout"
], function (Controller, JSONModel, CheckBox, Menu, MenuItem, coreItem, Column,
             mInput, mButton, ResizeHandler, VerticalLayout, HorizontalLayout) {

   "use strict";

   return Controller.extend("rootui5.eve7.controller.EveTable", {

      onInit : function() {
         var data = this.getView().getViewData();
         // console.log("VIEW DATA", data);

         var id = this.getView().getId();
         console.log("eve.GL.onInit id = ", id );

         this._load_scripts = true;
         this._render_html = false;

         this.mgr = data.mgr;
         this.elementid = data.elementid;
         this.kind = data.kind;

         var element = this.mgr.GetElement(this.elementid);
         // loop over scene and add dependency
         for (var k=0;k<element.childs.length;++k) {
            var scene = element.childs[k];
            this.mgr.RegisterSceneReceiver(scene.fSceneId, this);
            console.log("on init ....table controller");
            this.onSceneCreate();
         }
      },

      locateEveTable: function()
      {
         this.eveTable = 0;
         var element = this.mgr.GetElement(this.elementid);
         var sceneInfo = element.childs[0];
         var scene = this.mgr.GetElement(sceneInfo.fSceneId);
         // console.log(">>>table scene", scene);
         if (scene.childs[0]._typename == "ROOT::Experimental::REveTableViewInfo") {
            // presume table view manger is first child of table scene
            this.viewInfo = scene.childs[0];
         }

         console.log("table viewinfo", this.viewInfo  );
         this.collection = this.mgr.GetElement(this.viewInfo.fDisplayedCollection);
         // loop over products
         for (var i = 1; i < scene.childs.length; ++i) {
            var product = scene.childs[i];
            if (product.childs && product.childs.length && product.childs[0].fCollectionId == this.viewInfo.fDisplayedCollection) {
               console.log("table found  ",product.childs[0] );
               this.eveTable =  product.childs[0];
               break;
            }
         }

      },

      buildTableBody: function(doBind)
      {
         var oTable = this.getView().byId("table");

         // row definition
         var rowData = this.eveTable.body;
         for (var i = 0; i < this.collection.childs.length; i++)
         {
            rowData[i].Name =  this.collection.childs[i].fName;
            rowData[i].Filtered =  this.collection.childs[i].fFiltered === true ? "--" : "*";
         }

         if (doBind) {
            // column definition
            var columnData = [];

            columnData.push({columnName:"Name"});
            columnData.push({columnName:"Filtered"});

            console.log("buildTableBody",this.eveTable );
            var eveColumns = this.eveTable.childs;
            for (var i = 0; i < eveColumns.length; i++)
            {
               var cname = eveColumns[i].fName;
               columnData.push({columnName : cname});
            }


            // table model
            var oModel = new JSONModel();
            oModel.setData({
               rows: rowData,
               columns: columnData
            });
            oTable.setModel(oModel);

            /*
            // bind rows and columns
            oTable.bindColumns("/columns", function(sId, oContext) {
            var columnName = oContext.getObject().columnName;
            var oColumn = new Column({
            label: columnName,
            template: columnName,
            sortProperty: columnName,
            showFilterMenuEntry: true
            });
            return oColumn;
            });
            */

            oTable.bindAggregation("columns", "/columns", function(sId, oContext) {
               return new sap.ui.table.Column(sId, {
	          label: "{columnName}",
	          template: new sap.m.Text().bindText(oContext.getProperty("columnName")),
                  showFilterMenuEntry: true,
                  width: "100px"
               });
            });
            oTable.bindRows("/rows");
         }
         else
         {
            var model = oTable.getModel();
            var data = model.getData();
            model.setData({"rows":rowData, "columns":data.columns});
         }
      },

      buildTableHeader: function()
      {
         var oModel = new JSONModel();
         var collection = this.mgr.GetElement(this.eveTable.fCollectionId);
         var clist = this.mgr.GetElement(collection.fMotherId);
         // console.log("collection list ", clist);

	 var mData = {
	    "itemx": [
	    ]};

         for (var i = 0; i < clist.childs.length; i++)
         {
            mData.itemx.push({"text" :clist.childs[i].fName, "key": clist.childs[i].fName, "elementId":clist.childs[i].fElementId });
         }
         oModel.setData(mData);
         this.getView().setModel(oModel, "collections");

         var combo = this.getView().byId("ccombo");
         combo.setSelectedKey(collection.fName);
         combo.data("controller", this);
      },

      onLoadScripts: function() {
         this._load_scripts = true;
         this.checkScenes();
      },

      // function called from GuiPanelController
      onExit : function() {
         if (this.mgr) this.mgr.Unregister(this);
      },

      onSceneCreate: function(element, id) {
         console.log("EveTable onSceneChanged", id);
         this.locateEveTable();
         this.buildTableHeader();
         this.buildTableBody(true);
      },

      UpdateMgr : function(mgr) {
         var elem = mgr.map[this.elementid];
         var scene = mgr.map[ elem.fMotherId];
         this.mgr = mgr;
      },

      onAfterRendering: function() {
         this._render_html = true;

         // this.getView().$().css("overflow", "hidden");

         // this.getView().$().parent().css("overflow", "hidden");

         // only when rendering completed - register for modify events
         var element = this.mgr.GetElement(this.elementid);

         this.checkScenes();
      },

      checkScenes: function() {
      },

      toggleTableEdit: function() {

         var header = this.getView().byId("header");
         if (!this.editor) {
            this.editor = new VerticalLayout("tableEdit", {"width":"100%"});

            header.addContent(this.editor);

            // expression row
            {
               var collection = this.mgr.GetElement(this.eveTable.fCollectionId);
               var oModel = new JSONModel();
               oModel.setData(collection.publicFunction);
               // oModel.setData(aData);
               this.getView().setModel(oModel);

               var exprIn = new mInput("expression", {
                                         width:"98%",
                                         type : sap.m.InputType.Text,
                                         placeholder: "Expression",
                                         showSuggestion: true
                                       });
               exprIn.setModel(oModel);
               exprIn.bindAggregation("suggestionItems", "/", new coreItem({text: "{name}"}));
               exprIn.setFilterFunction(function(sTerm, oItem) {
                  // A case-insensitive 'string contains' style filter
                  // console.log("filter sterm", sTerm);
                  var base = sTerm;
                  var n = base.lastIndexOf("i.");
                  // console.log("last index ", n);
                  if (n>=0) n+=2;
                  var txt = base.substring(n,this.getFocusInfo().cursorPos );
                  // console.log("suggest filter ", txt);
                  // console.log("focus 1", this.getFocusInfo());

                  return oItem.getText().match(new RegExp(txt, "i"));
               });
               this.editor.addContent(exprIn);
            }

            // title & prec
            {
               var hl = new HorizontalLayout();
               var titleIn = new mInput("title", {placeholder:"Title", tooltip:"title"});
               titleIn.setWidth("98%");
               hl.addContent(titleIn);
               //this.editor.addContent(titleIn);

               var precIn = new mInput("precision", {placeholder:"Precision", type: sap.m.InputType.Number, constraints: {minimum:"0", maximum:"9"}});
               // precIn.bindProperty("value", "abc>precision");
               precIn.setWidth("100px");

               hl.addContent(precIn);

               this.editor.addContent(hl);

            }

            //  button actions
            {
               var ll = new HorizontalLayout();
               var addBut = new mButton("AddCol", {text:"Add", press: this.addColumn});
               addBut.data("controller", this);
               ll.addContent(addBut);
               this.editor.visible = true;
               this.editor.addContent(ll);
            }
            return;
         }

         if (this.editor.visible) {
            var x = header.getContent().pop();
            header.removeContent(x);
            this.editor.visible = false;
         }
         else {
            header.addContent(this.editor);
            this.editor.visible = true;
         }
      },

      addColumn: function(event) {
         // console.log("add column s", event.getSource(), this);
         // console.log("add column p", this.data("controller"));
         var pthis = this.data("controller");
         var ws = pthis.editor.getContent();

         var expr = ws[0].getProperty("value");
         if (!expr) {
            alert("need a new column expression");
         }
         var hl =  ws[1].getContent();
         var title = hl[0].getProperty("value");
         if (!title) {
            title = expr;
         }

         var mir = "AddNewColumnToCurrentCollection( \"" + expr + "\", \"" + title + "\" )";

         // console.log("table element id ", pthis.eveTable.fElementId);

         var obj = {"mir" : mir, "fElementId" :pthis.viewInfo.fElementId, "class" : pthis.eveTable._typename};
         // console.log("MIR obj ", obj);
         pthis.mgr.handle.Send(JSON.stringify(obj));
      },

      collectionChanged: function(oEvent) {
         // console.log("collectionChanged ", oEvent.getSource());
         var model = oEvent.oSource.getSelectedItem().getBindingContext("collections");
         var path = model.getPath();
         var entry = model.getProperty(path);
         var coll = entry.elementId;
         var mng =  this.viewInfo;
         var mir = {"elementid" : mng.fElementId, "elementclass":mng._typename};
         mir.func = "SetDisplayedCollection(" + coll + ")";

         this.mgr.executeCommand(mir);
      },

      sceneElementChange : function(msg)
      {
        // AMT for the moment always recreate table in endChanges
        // var el = this.mgr.GetElement(msg.fElementId);
        // this[msg.tag](el);
      },

      endChanges : function(oEvent) {
         console.log("table controller endChanges ",this.eveTable );
         this.locateEveTable();
         this.buildTableBody(false);
      },

      elementRemoved: function(elId) {
         var el = this.mgr.GetElement(elId);
         // console.log("EveTable element removed ", el);
      }
   });
});
