sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/model/json/JSONModel',
   'sap/ui/commons/CheckBox',
   'sap/ui/commons/Menu',
   'sap/ui/commons/MenuItem',
   'sap/ui/table/Column',
   "sap/ui/core/ResizeHandler"
], function (Controller, JSONModel, ResizeHandler) {

   "use strict";

   return Controller.extend("eve.EveTable", {

      onInit : function() {

         var data = this.getView().getViewData();
         // console.log("VIEW DATA", data);

         var id = this.getView().getId();
         console.log("eve.GL.onInit id = ", id );


         this._load_scripts = true;
         this._render_html = false;

         // console.log("TABLE VIEW CREATED");

         this.mgr = data.mgr;
         this.elementid = data.elementid;
         this.kind = data.kind;

         var element = this.mgr.GetElement(this.elementid);
         // loop over scene and add dependency
         for (var k=0;k<element.childs.length;++k) {
            var scene = element.childs[k];
            this.mgr.RegisterSceneReceiver(scene.fSceneId, this);
            this.setEveData();
         }

         this.mgr.RegisterUpdate(this, "setEveData");
      },

      findTable: function(holder) {
         // presume table view manger is first child of table scene
         var mng = holder.childs[0];
         this.collectionMng = mng;
         this.collection = this.mgr.GetElement(mng.fDisplayedCollection);
         for (var i = 1; i < holder.childs.length; i++ )
         {
            var product = holder.childs[i];
            if (product.childs.length)
               return product.childs[0];
         }

      },

      setEveData: function() {
         var element = this.mgr.GetElement(this.elementid);
         console.log("table ", element);
         for (var k=0;k<element.childs.length;++k) {
            var sceneInfo = element.childs[k];
            var scene = this.mgr.GetElement(sceneInfo.fSceneId);
            this.tableEveElement = this.findTable(scene)
            this.setupTable(this.tableEveElement);
         }
      },

      setupTable: function(eveData) {
         var oTable = this.getView().byId("table");
         console.log(oTable);

         var columnData = [];
         columnData.push({columnName:"Name"});
         columnData.push({columnName:"Rnr"});

         for (var i = 0; i < eveData.childs.length; i++)
         {
            columnData.push({columnName:eveData.childs[i].fName});
         }

         var rowData = eveData.body;

         var collection = this.mgr.GetElement(eveData.fCollectionId);
         var pass = 0;

         for (var i = 0; i < collection.childs.length; i++)
         {
            rowData[i].Name =  collection.childs[i].fName;
            rowData[i].Rnr =  collection.childs[i].fFiltered === true ? "" : "*";
            if ( !collection.childs[i].fFiltered) pass++;
         }

         // console.log("collection ",collection );
         // console.log("rowData ", rowData );
         // console.log("columnData", columnData);

         var oModel = new sap.ui.model.json.JSONModel();
         oModel.setData({
            rows: rowData,
            columns: columnData
         });
         oTable.setModel(oModel);

         var uuu= this;

         oTable.bindColumns("/columns", function(sId, oContext) {
            var columnName = oContext.getObject().columnName;
            var oColumn = new sap.ui.table.Column({

               label: columnName,
               template: columnName,
               sortProperty: columnName,
               showFilterMenuEntry: true
            });

            return oColumn;
         });

         oTable.bindRows("/rows");


         //______________________________________________________________________________
         // get list of collections

         var oModel = new sap.ui.model.json.JSONModel();
         var clist = this.mgr.GetElement(this.collection.fMotherId);
         console.log("collection list ", clist);

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
         combo.setSelectedKey("XYTracks");
         combo.data("controller", this);

      },

      onLoadScripts: function() {
         this._load_scripts = true;
         this.checkScenes();
      },

      acm: function()
      {
         alert("Custom Menu");
      },

      // function called from GuiPanelController
      onExit : function() {
         if (this.mgr) this.mgr.Unregister(this);
      },

      onSceneCreate: function(element, id) {
         console.log("EveTable onSceneChanged", id);
         this.setEveData();
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
            this.editor = new sap.ui.layout.VerticalLayout("tableEdit", {"width":"100%"});

            header.addContent(this.editor);
            // this.editor.bindElement("abc>/gedcol");
            // expression row
            {

               var collection = this.mgr.GetElement(this.tableEveElement.fCollectionId);
               var oModel = new sap.ui.model.json.JSONModel();
               oModel.setData(collection.publicFunction);
               // oModel.setData(aData);
               console.log("XXX suggest ", oModel);
               this.getView().setModel(oModel);

               var exprIn = new sap.m.Input("expression", { width:"98%",
                                                            type : sap.m.InputType.Text,
                                                            placeholder:"Expression",
                                                            showSuggestion: true
                                                          }
                                           );
               exprIn.setModel(oModel);
               exprIn.bindAggregation("suggestionItems", "/", new sap.ui.core.Item({text: "{name}"}));
               exprIn.setFilterFunction(function(sTerm, oItem) {
             // A case-insensitive 'string contains' style filter
                  console.log("filter sterm", sTerm);
                  var base = sTerm;
                  var n = base.lastIndexOf("i.");
                  console.log("last index ", n);
                  if (n>=0) n+=2;
                  var txt = base.substring(n,this.getFocusInfo().cursorPos );
                  console.log("suggest filter ", txt);
                  console.log("focus 1", this.getFocusInfo());

             return oItem.getText().match(new RegExp(txt, "i"));
          });



               this.editor.addContent(exprIn);
            }
            // title & prec
            {
               var hl = new sap.ui.layout.HorizontalLayout();
               var titleIn = new sap.m.Input("title", {placeholder:"Title", tooltip:"title"});
               titleIn.setWidth("98%");
               hl.addContent(titleIn);
               //this.editor.addContent(titleIn);

               var precIn = new sap.m.Input("precision", {placeholder:"Precision", type:sap.m.InputType.Number, constraints:{minimum:"0", maximum:"9"}});
               // precIn.bindProperty("value", "abc>precision");
               precIn.setWidth("100px");

               hl.addContent(precIn);

               this.editor.addContent(hl);

            }
            //  button actions
            {
               var ll = new sap.ui.layout.HorizontalLayout();
               //   var precision = new sap.m.Input("precsion", {placeholder:"Precision"});
               //   ll.addContent(precision);
               var addBut = new sap.m.Button("AddCol", {text:"Add", press: this.addColumn});
               addBut.data("controller", this);
               ll.addContent(addBut);
               //   ll.addContent(new sap.m.Button("ModifyCol", {text:"Modify", press:"modifyColumn"}));
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
         console.log("add column s", event.getSource(), this);
         console.log("add column p", this.data("controller"));
         var pthis = this.data("controller");
         var ws = pthis.editor.getContent();

         /*
           console.log("properties ", ws[0].getProperty("value"));
           var title = pthis.editor.byId("expression");
           console.log("title ", title.getParameters.value());
         */

         var expr = ws[0].getProperty("value");
         if (!expr) {
            alert("need a new column expression");
         }
         var hl =  ws[1].getContent();
         var title = hl[0].getProperty("value");
         if (!title) {
            title = expr;
         }

         var mir = "AddNewColumn( \"" + expr + "\", \"" + title + "\" )";

         console.log("table element id ", pthis.tableEveElement.fElementId);

         var obj = {"mir" : mir, "fElementId" : pthis.tableEveElement.fElementId, "class" : pthis.tableEveElement._typename};
         console.log("MIR obj ", obj);
         pthis.mgr.handle.Send(JSON.stringify(obj));


      },
/*
     replaceElement: function(el) {
           console.log("REPLACE TABLE !!! ");
         this.setupTable( this.tableEveElement);
      },


      elementAdded : function(el) {
           this.setEveData();
      },
      elementRemoved: function() {
      },


      beginChanges : function() {
      },
*/
      collectionChanged: function(oEvent) {
         console.log("collectionChanged ", oEvent.getSource());
         console.log("xxx ", this);
       //  var pthis = this.data("controller");
         var model = oEvent.oSource.getSelectedItem().getBindingContext("collections");
         var path = model.getPath();
         var entry = model.getProperty(path);
         var coll = entry.elementId;
         var mng = this.collectionMng;
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
         console.log("table controller endchanges ",this.tableEveElement );
         this.setEveData();
      },


      elementRemoved: function(elId) {

      var el = this.mgr.GetElement(elId);
         console.log("EveTable element remobedf ", el);

      }
   });
});
