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
   "sap/m/FormattedText",
   "sap/ui/core/ResizeHandler",
   "sap/ui/layout/VerticalLayout",
   "sap/ui/layout/HorizontalLayout",
   "sap/m/MessageBox"
], function (Controller, JSONModel, CheckBox, Menu, MenuItem, coreItem, Column,
             mInput, mButton, FormattedText, ResizeHandler, VerticalLayout, HorizontalLayout) {

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
         this.eveViewerId = data.eveViewerId;
         this.kind = data.kind;

<<<<<<< HEAD
         var rh = this.mgr.handle.GetUserArgs("TableRowHeight");
         if (rh && (rh>0))
            this.getView().byId("table").setRowHeight(rh);

=======
>>>>>>> 6360691bd8... Use two level sort and shade entries in cell table that do not pass filter.
         this.bindTableColumns = true;
         var element = this.mgr.GetElement(this.eveViewerId);
         // loop over scene and add dependency
         for (var k=0;k<element.childs.length;++k) {
            var scene = element.childs[k];
            this.mgr.RegisterSceneReceiver(scene.fSceneId, this);
            this.onSceneCreate();
         }
      },
      sortTable: function(e) {
	   var col = e.mParameters.column;
	   var colId = col.getId();

	   var col = e.mParameters.column;
	   var bDescending = (e.mParameters.sortOrder ==  sap.ui.core.SortOrder.Descending);
	   var sv = bDescending;

	   var oSorter0 = new sap.ui.model.Sorter({
	       path: "Filtered",
	       descending: false
	   });

	   var oSorter1 = new sap.ui.model.Sorter({
	       path: col.mProperties.sortProperty,
	       descending: sv
	   });

	   oSorter1.fnCompare = function(value1, value2) {
	       value2 = parseFloat(value2);
	       value1 = parseFloat(value1);
	       if (value1 < value2) return -1;
	       if (value1 == value2) return 0;
	       if (value1 > value2) return 1;
	   };

           var oTable = this.getView().byId("table");
	   var oItemsBinding = oTable.getBinding("rows");
	   oItemsBinding.sort([oSorter0, oSorter1]);
           col.setSorted(true);

           if (sv)
               col.setSortOrder(sap.ui.table.SortOrder.Descending);
           else
               col.setSortOrder(sap.ui.table.SortOrder.Ascending);

	   e.preventDefault();
      },

      locateEveTable: function()
      {
         this.eveTable = 0;
         var element = this.mgr.GetElement(this.eveViewerId);
         var sceneInfo = element.childs[0];
         var scene = this.mgr.GetElement(sceneInfo.fSceneId);
         // console.log(">>>table scene", scene);
         if (scene.childs[0]._typename == "ROOT::Experimental::REveTableViewInfo") {
            // presume table view manger is first child of table scene
            this.viewInfo = scene.childs[0];
         }

         // console.log("table viewinfo", this.viewInfo  );
         this.collection = this.mgr.GetElement(this.viewInfo.fDisplayedCollection);
         // loop over products
         for (var i = 1; i < scene.childs.length; ++i) {
            var product = scene.childs[i];
            if (product.childs && product.childs.length && product.childs[0].fCollectionId == this.viewInfo.fDisplayedCollection) {
               // console.log("table found  ",product.childs[0] );
               this.eveTable =  product.childs[0];
               break;
            }
         }

      },
      getCellText : function(value, filtered) {
	 return "<span class='" + (filtered ? "eveTableCellUnfiltered" : "eveTableCellFiltered") + "'>" + value + "</span>"
      },
      buildTableBody: function()
      {
         var oTable = this.getView().byId("table");

         // row definition
         var rowData = this.eveTable.body;

         // parse to float -- AMT in future data should be streamed as floats
         for (var r=0; r < rowData.length; r++) {
            var xr = rowData[r];
            for (var xri = 0; xri < xr.length; xri++) {
               var nv = parseFloat(xr[i]);
               if (nv != NaN) {
                  rowData[r][ri] = nv;
               }
            }
         }

         for (var i = 0; i < this.collection.childs.length; i++)
         {
            rowData[i].Name =  this.collection.childs[i].fName;
            rowData[i].Filtered =  this.collection.childs[i].fFiltered === true ? 1 : 0;
         }

         if (this.bindTableColumns) {
            // column definition
            var columnData = [];

            columnData.push({columnName:"Name"});
            columnData.push({columnName:"Filtered"});

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

	     var pthis = this;
             oTable.bindAggregation("columns", "/columns", function(sId, oContext) {
                 return new sap.ui.table.Column(sId, {
	             label: "{columnName}",
	             sortProperty: "{columnName}",
		     template: new FormattedText({
		         htmlText: {
			     parts: [
			         {path: oContext.getProperty("columnName")},
			         {path: "Filtered"}
			     ],
			     formatter: pthis.getCellText
		         }
		     }),
                     showFilterMenuEntry: true,
                     width: "100px"
               });
            });

	     // bind the Table items to the data collection
	     var oBinding = oTable.bindRows({
		 path : "/rows",
		 sorter: [
		     new sap.ui.model.Sorter({
			 path: 'Filtered',
			 descending: false
		     })
		 ]

	     });
            this.bindTableColumns = false;
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
            mData.itemx.push({"text" :clist.childs[i].fName, "key": clist.childs[i].fName, "collectionEveId":clist.childs[i].fElementId });
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
         var elem = mgr.map[this.eveViewerId];
         var scene = mgr.map[ elem.fMotherId];
         this.mgr = mgr;
      },

      onAfterRendering: function() {
         this._render_html = true;
         this.checkScenes();
      },

      checkScenes: function() {
      },

      toggleTableEdit: function() {
         var header = sap.ui.getCore().byId("EveViewer21--header");
         if (!this.editor) {
            this.editor = new VerticalLayout("tableEdit", {"width":"100%"});

            header.addContent(this.editor);

            // expression row
            {
               var collection = this.mgr.GetElement(this.eveTable.fCollectionId);
               var exprIn = new sap.m.Input("inputExp", {
    		  placeholder: "Start expression with \"i.\" to access object",
    		  showValueHelp: true,
    		  showTableSuggestionValueHelp: false,
    		  width: "100%",
    		  maxSuggestionWidth: "500px",
    		  showSuggestion: true,
    		  valueHelpRequest: function (oEvent) {
                     MessageBox.alert("Write any valid expression.\n Using \"i.\" convetion to access an object in collection. Below is an example:\ni.GetPdgCode() + 2");
    		  },
    		  suggestionItemSelected: function(oEvent){
    		     var oItem = oEvent.getParameter("selectedRow");
    		     console.log("sap.m.Input id with suggestion: selected item text is ------ " , oItem.getCells());
                     // fill in title if empty
                     var it = sap.ui.getCore().byId("titleEx");
                     if ((it.getValue() && it.getValue().length) == false) {
                        var v =  oItem.getCells()[0].getText().substring(2);
                        var t = v.split("(");
                        it.setValue(t[0]);
                     }
                     //fill in precision if empty
                     var ip = sap.ui.getCore().byId("precisionEx");
                     if ((ip.getValue() && ip.getValue().length) == false) {
                        var p = oItem.getCells()[1].getText();
                        if (p.startsWith("Int") || p.startsWith("int"))
                           ip.setValue("0");
                        else
                           ip.setValue("3");
                     }
    		  },
    		  suggestionColumns : [
    		     new sap.m.Column({
    			styleClass : "f",
    			hAlign : "Begin",
    			header : new sap.m.Label({
                           text : "Funcname"
    			})
    		     }),
    		     new sap.m.Column({
    			hAlign : "Center",
    			styleClass : "r",
    			popinDisplay : "Inline",
    			header : new sap.m.Label({
    			   text : "Return"
    			}),
    			minScreenWidth : "Tablet",
    			demandPopin : true
    		     }),

    		     new sap.m.Column({
    			hAlign : "End",
    			styleClass : "c",
    			width : "30%",
    			popinDisplay : "Inline",
    			header : new sap.m.Label({
                           text : "Class"
    			}),
    			minScreenWidth : "400px",
    			demandPopin : true
    		     })
    		  ]
    	       }).addStyleClass("inputRight");


               exprIn.setModel(oModel);

    	       var oTableItemTemplate = new sap.m.ColumnListItem({
    		  type : "Active",
    		  vAlign : "Middle",
    		  cells : [
    		     new sap.m.Label({
    			text : "{f}"
    		     }),
                     new sap.m.Label({
    			text: "{r}"
    		     }),
                     new sap.m.Label({
    			text : "{c}"
    		     })
    		  ]
    	       });
    	       var oModel = new sap.ui.model.json.JSONModel();
               var oSuggestionData = collection.fPublicFunctions;
               console.log("suggested data ",oSuggestionData );
    	       oModel.setData(oSuggestionData);
    	       exprIn.setModel(oModel);
    	       exprIn.bindAggregation("suggestionRows", "/", oTableItemTemplate);

               this.editor.addContent(exprIn);
            }

            // title & prec
            {
               var hl = new HorizontalLayout( {"width":"100%"});
               var titleIn = new mInput("titleEx", {placeholder:"Title", tooltip:"column title"});
             titleIn.setWidth("100%");
               hl.addContent(titleIn);

               var precIn = new mInput("precisionEx", {placeholder:"Precision", type: sap.m.InputType.Number, constraints: {minimum:"0", maximum:"9"}});
               precIn.setWidth("100%");
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

         pthis.mgr.SendMIR("AddNewColumnToCurrentCollection( \"" + expr + "\", \"" + title + "\" )",
                           pthis.viewInfo.fElementId, pthis.viewInfo._typename);

         // reset values
         sap.ui.getCore().byId("titleEx").setValue("");
         sap.ui.getCore().byId("precisionEx").setValue("");
      },

      collectionChanged: function(oEvent) {
         // console.log("collectionChanged ", oEvent.getSource());
         var model = oEvent.oSource.getSelectedItem().getBindingContext("collections");
         var path = model.getPath();
         var entry = model.getProperty(path);
         var coll = entry.collectionEveId;
         var mng =  this.viewInfo;

<<<<<<< HEAD
         this.mgr.SendMIR("SetDisplayedCollection(" + coll + ")", mng.fElementId, mng._typename);
=======
         var mir = "SetDisplayedCollection(" + coll + ")";
         var obj = {"mir" : mir, "fElementId" : mng.fElementId, "class":mng._typename };
         this.mgr.SendMIR(obj);
>>>>>>> 6360691bd8... Use two level sort and shade entries in cell table that do not pass filter.
      },

      sceneElementChange : function(el)
      {
         // console.log("table sceneElementChange", el);
         if (el._typename == "ROOT::Experimental::REveTableViewInfo") {
            this.bindTableColumns = true;
         }
      },

      endChanges : function(oEvent) {
         // console.log("table controller endChanges ",this.eveTable );
         this.locateEveTable();
         this.buildTableBody();
      },

      elementRemoved: function(elId) {
         var el = this.mgr.GetElement(elId);
         // console.log("EveTable element removed ", el);
      }
   });
});
