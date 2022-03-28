sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/core/UIComponent',
   'sap/ui/model/json/JSONModel',
   'sap/ui/model/Sorter',
   'sap/m/Column',
   'sap/m/ColumnListItem',
   'sap/m/Input',
   'sap/m/Label',
   'sap/m/Button',
   "sap/m/FormattedText",
   "sap/ui/layout/VerticalLayout",
   "sap/ui/layout/HorizontalLayout",
   "sap/ui/table/Column",
   "sap/m/MessageBox"
], function (Controller, UIComponent, JSONModel, Sorter,
   mColumn, mColumnListItem, mInput, mLabel, mButton,
   FormattedText, VerticalLayout, HorizontalLayout, tableColumn, MessageBox) {

   "use strict";

   return Controller.extend("rootui5.eve7.controller.EveTable", {

      onInit: function () {
         let viewData = this.getView().getViewData();
         if (viewData) {
            this.setupManagerAndViewType(viewData.eveViewerId, viewData.mgr);
         }
         else {
             UIComponent.getRouterFor(this).getRoute("Table").attachPatternMatched(this.onTableObjectMatched, this);
         }
      },
      onTableObjectMatched: function (oEvent) {
         let args = oEvent.getParameter("arguments");
         this.setupManagerAndViewType(EVE.$eve7tmp.eveViewerId, EVE.$eve7tmp.mgr);
         delete EVE.$eve7tmp;

         this.checkViewReady();
      },
      setupManagerAndViewType: function(eveViewerId, mgr)
      {
         this._load_scripts = true;
         this._render_html = false;

         this.mgr = mgr;
         this.eveViewerId = eveViewerId;

         this.table = this.getView().byId("table");


         let rh = this.mgr.handle.getUserArgs("TableRowHeight");
         if (rh && (rh > 0))
            this.getView().byId("table").setRowHeight(rh);

         this.bindTableColumns = true;
         let element = this.mgr.GetElement(this.eveViewerId);
         // loop over scene and add dependency
         for (let k = 0; k < element.childs.length; ++k) {
            let scene = element.childs[k];
            this.mgr.RegisterSceneReceiver(scene.fSceneId, this)
         };
         this.onSceneCreate();

         // attach to changes in 'Collection' scene
         let sceneList = this.mgr.childs[0].childs[2].childs;
         for (let i = 0; i < sceneList.length; ++i) {
            if (sceneList[i].fName == "Collections")
               this.mgr.RegisterSceneReceiver(sceneList[i].fElementId, this);
         }

         let table = this.table;
         let pthis = this;
         table.attachRowSelectionChange(function (d) {
            if (pthis.mgr.busyProcessingChanges)
               return;

            let idx = d.getParameter("rowIndex");
            let oData = table.getContextByIndex(idx);
            if (oData) {
               let ui = oData.getPath().substring(6);
               // console.log("idx =", idx, "path idx = ", ui);

               let itemList = pthis.collection.childs[0];
               let secIdcs = [ui];
               let fcall = "ProcessSelectionStr(" + pthis.mgr.global_selection_id + ", " + pthis.multiSelect + ", true";
               fcall += ", \" " + secIdcs.join(", ") + " \"";
               fcall += ")";
               pthis.mgr.SendMIR(fcall, itemList.fElementId, itemList._typename);
            }
            else {
               // console.log("attachRowSelectionChange no path ", oData);
            }
         });


         if (table._enableLegacyMultiSelection) {
            table._enableLegacyMultiSelection();
         }
         else {
            console.error("Ctrl modifier not supported !")
         }

         // listen to Ctrl key events
         this.multiSelect = false;
         window.addEventListener('keydown', function(event) { if (event.ctrlKey) pthis.multiSelect = true; });
         window.addEventListener('keyup',   function(event) { if (event.key == 'Control') pthis.multiSelect = false; });
      },

      sortTable: function (e) {
         // let colId = col.getId();
         let col = e.mParameters.column;
         let bDescending = (e.mParameters.sortOrder == sap.ui.core.SortOrder.Descending);
         let sv = bDescending;

         let oSorter0 = new Sorter({
            path: "Filtered",
            descending: true
         });

         let oSorter1 = new Sorter({
            path: col.mProperties.sortProperty,
            descending: sv
         });

         if (col.mProperties.sortProperty === "Name") {
            let off = this.collection.fName.length;
            oSorter1.fnCompare = function (value1In, value2In) {
               let value1 = value1In.substring(off);
               let value2 = value2In.substring(off);
               value2 = parseInt(value2);
               value1 = parseInt(value1);
               if (value1 < value2) return -1;
               if (value1 == value2) return 0;
               if (value1 > value2) return 1;
            };
         }
         else {
            oSorter1.fnCompare = function (value1, value2) {
               value2 = parseFloat(value2);
               value1 = parseFloat(value1);
               if (value1 < value2) return -1;
               if (value1 == value2) return 0;
               if (value1 > value2) return 1;
            };
         }
         let oTable = this.getView().byId("table");
         let oItemsBinding = oTable.getBinding("rows");
         // Do one-level sort on Filtered entry
         if (col.mProperties.sortProperty === "Filtered")
            oItemsBinding.sort([oSorter1]);
         else
            oItemsBinding.sort([oSorter0, oSorter1]);

         // show indicators in column header
         col.setSorted(true);
         if (sv)
            col.setSortOrder(sap.ui.table.SortOrder.Descending);
         else
            col.setSortOrder(sap.ui.table.SortOrder.Ascending);

         e.preventDefault();
         this.updateSortMap();
      },

      updateSortMap() {
         // update sorted/unsorted idx map
         let oTable = this.getView().byId("table");
         let nr = oTable.getModel().oData.rows.length;
         if (!oTable.sortMap)
            oTable.sortMap = new Map();

         for (let r = 0; r < nr; ++r) {
            let oData = oTable.getContextByIndex(r);
            let unsortedIdx = oData.sPath.substring(6);
            oTable.sortMap[unsortedIdx] = r;
         }
      },

      locateEveTable: function () {
         this.eveTable = 0;
         let element = this.mgr.GetElement(this.eveViewerId);
         let sceneInfo = element.childs[0];
         let scene = this.mgr.GetElement(sceneInfo.fSceneId);
         // console.log(">>>table scene", scene);
         if (scene.childs[0]._typename == "ROOT::Experimental::REveTableViewInfo") {
            // presume table view manger is first child of table scene
            this.viewInfo = scene.childs[0];
         }

         this.collection = this.mgr.GetElement(this.viewInfo.fDisplayedCollection);
         // loop over products
         for (let i = 1; i < scene.childs.length; ++i) {
            let product = scene.childs[i];
            if (product.childs && product.childs.length && product.childs[0].fCollectionId == this.viewInfo.fDisplayedCollection) {
               // console.log("table found  ",product.childs[0] );
               this.eveTable = product.childs[0];
               break;
            }
         }
      },

      getCellText: function (value, filtered) {
         return "<span class='" + (filtered ? "eveTableCellFiltered" : "eveTableCellUnfiltered") + "'>" + value + "</span>"
      },

      buildTableBody: function () {
         if (!(this.eveTable && this.eveTable.body))
            return;

         let oTable = this.getView().byId("table");

         // row definition
         let rowData = this.eveTable.body;

         // parse to float -- AMT in future data should be streamed as floats
         for (let r = 0; r < rowData.length; r++) {
            let xr = rowData[r];
            for (let xri = 0; xri < xr.length; xri++) {
               let nv = parseFloat(xr[i]);
               if (!isNaN(nv)) {
                  rowData[r][ri] = nv;
               }
            }
         }

         let itemList = this.collection.childs[0].items;
         for (let i = 0; i < itemList.length; i++) {
            rowData[i].Name = this.collection.fName + " " + i;
            rowData[i].Filtered = itemList[i].fFiltered === true ? 0 : 1;
         }

         if (this.bindTableColumns) {
            // column definition
            let columnData = [];

            columnData.push({ columnName: "Name" });
            columnData.push({ columnName: "Filtered" });

            let eveColumns = this.eveTable.childs;
            for (let i = 0; i < eveColumns.length; i++) {
               let cname = eveColumns[i].fName;
               columnData.push({ columnName: cname });
            }

            // table model
            let oModel = new JSONModel();
            oModel.setData({
               rows: rowData,
               columns: columnData
            });
            oTable.setModel(oModel);

            let pthis = this;
            oTable.bindAggregation("columns", "/columns", function (sId, oContext) {
               return new tableColumn(sId, {
                  label: "{columnName}",
                  sortProperty: "{columnName}",
                  template: new FormattedText({
                     htmlText: {
                        parts: [
                           { path: oContext.getProperty("columnName") },
                           { path: "Filtered" }
                        ],
                        formatter: pthis.getCellText
                     }
                  }),
                  showFilterMenuEntry: true,
               });
            });

            // bind the Table items to the data collection
            let oBinding = oTable.bindRows({
               path: "/rows",
               sorter: [
                  new Sorter({
                     path: 'Filtered',
                     descending: true
                  })
               ]

            });

            if (sap.ui.getCore().byId("inputExp")) {
               let ent = sap.ui.getCore().byId("inputExp");
               let sm = ent.getModel();
               sm.setData(this.eveTable.fPublicFunctions);
            }

            this.bindTableColumns = false;
         }
         else {
            let model = oTable.getModel();
            let data = model.getData();
            model.setData({ "rows": rowData, "columns": data.columns });
         }
         this.updateSortMap();
      },

      buildTableHeader: function () {
         let oModel = new JSONModel();
         let collection = this.mgr.GetElement(this.eveTable.fCollectionId);
         let clist = this.mgr.GetElement(collection.fMotherId);

         let mData = {
            "itemx": [
            ]
         };

         for (let i = 0; i < clist.childs.length; i++) {
            mData.itemx.push({ "text": clist.childs[i].fName, "key": clist.childs[i].fName, "collectionEveId": clist.childs[i].fElementId });
         }
         oModel.setData(mData);
         this.getView().setModel(oModel, "collections");

         let combo = this.getView().byId("ccombo");
         combo.setSelectedKey(collection.fName);
         combo.data("controller", this);
      },

      onLoadScripts: function () {
         this._load_scripts = true;
         this.checkScenes();
      },

      // function called from GuiPanelController
      onExit: function () {
         if (this.mgr) this.mgr.Unregister(this);
      },

      onSceneCreate: function (element, id) {
         this.locateEveTable();
         this.buildTableHeader();
         this.buildTableBody(true);
         this.UpdateSelection(false);
      },

      UpdateMgr: function (mgr) {
         let elem = mgr.map[this.eveViewerId];
         let scene = mgr.map[elem.fMotherId];
         this.mgr = mgr;
      },

      onAfterRendering: function () {
         this._render_html = true;
         this.checkScenes();
      },

      checkScenes: function () {
      },

      toggleTableEdit: function () {
         let header = this.getView().byId("header");
         if (!this.editor) {
            this.editor = new VerticalLayout("tableEdit", { "width": "100%" });

            header.addContent(this.editor);

            // expression row
            {
               let collection = this.mgr.GetElement(this.eveTable.fCollectionId);
               let exprIn = new mInput("inputExp", {
                  placeholder: "Start expression with \"i.\" to access object",
                  showValueHelp: true,
                  showTableSuggestionValueHelp: false,
                  width: "100%",
                  maxSuggestionWidth: "500px",
                  showSuggestion: true,
                  valueHelpRequest: function (oEvent) {
                     MessageBox.alert("Write any valid expression.\n Using \"i.\" convetion to access an object in collection. Below is an example:\ni.GetPdgCode() + 2");
                  },
                  suggestionItemSelected: function (oEvent) {
                     let oItem = oEvent.getParameter("selectedRow");
                     // fill in title if empty
                     let it = sap.ui.getCore().byId("titleEx");
                     if ((it.getValue() && it.getValue().length) == false) {
                        let v = oItem.getCells()[0].getText().substring(2);
                        let t = v.split("(");
                        it.setValue(t[0]);
                     }
                     //fill in precision if empty
                     let ip = sap.ui.getCore().byId("precisionEx");
                     if ((ip.getValue() && ip.getValue().length) == false) {
                        let p = oItem.getCells()[1].getText();
                        if (p.startsWith("Int") || p.startsWith("int"))
                           ip.setValue("0");
                        else
                           ip.setValue("3");
                     }
                  },
                  suggestionColumns: [
                     new mColumn({
                        styleClass: "f",
                        hAlign: "Begin",
                        header: new mLabel({
                           text: "Funcname"
                        })
                     }),
                     new mColumn({
                        hAlign: "Center",
                        styleClass: "r",
                        popinDisplay: "Inline",
                        header: new mLabel({
                           text: "Return"
                        }),
                        minScreenWidth: "Tablet",
                        demandPopin: true
                     }),

                     new mColumn({
                        hAlign: "End",
                        styleClass: "c",
                        width: "30%",
                        popinDisplay: "Inline",
                        header: new mLabel({
                           text: "Class"
                        }),
                        minScreenWidth: "400px",
                        demandPopin: true
                     })
                  ]
               }).addStyleClass("inputRight");


               exprIn.setModel(oModel);

               let oTableItemTemplate = new mColumnListItem({
                  type: "Active",
                  vAlign: "Middle",
                  cells: [
                     new mLabel({
                        text: "{f}"
                     }),
                     new mLabel({
                        text: "{r}"
                     }),
                     new mLabel({
                        text: "{c}"
                     })
                  ]
               });
               let oModel = new JSONModel();
               let oSuggestionData = this.eveTable.fPublicFunctions;
               oModel.setData(oSuggestionData);
               exprIn.setModel(oModel);
               exprIn.bindAggregation("suggestionRows", "/", oTableItemTemplate);

               this.editor.addContent(exprIn);
            }

            // title & prec
            {
               let hl = new HorizontalLayout({ "width": "100%" });
               let titleIn = new mInput("titleEx", { placeholder: "Title", tooltip: "column title" });
               titleIn.setWidth("100%");
               hl.addContent(titleIn);

               let precIn = new mInput("precisionEx", { placeholder: "Precision", type: sap.m.InputType.Number, constraints: { minimum: "0", maximum: "9" } });
               precIn.setWidth("100%");
               hl.addContent(precIn);

               this.editor.addContent(hl);
            }

            //  button actions
            {
               let ll = new HorizontalLayout();
               let addBut = new mButton("AddCol", { text: "Add", press: this.addColumn });
               addBut.data("controller", this);
               ll.addContent(addBut);
               this.editor.visible = true;
               this.editor.addContent(ll);
            }
            return;
         }

         if (this.editor.visible) {
            let x = header.getContent().pop();
            header.removeContent(x);
            this.editor.visible = false;
         }
         else {
            header.addContent(this.editor);
            this.editor.visible = true;
         }
      },

      addColumn: function (event) {
         let pthis = this.data("controller");
         let ws = pthis.editor.getContent();

         let expr = ws[0].getProperty("value");
         if (!expr) {
            alert("need a new column expression");
         }
         let hl = ws[1].getContent();
         let title = hl[0].getProperty("value");
         if (!title) {
            title = expr;
         }

         pthis.mgr.SendMIR("AddNewColumnToCurrentCollection( \"" + expr + "\", \"" + title + "\" )",
            pthis.viewInfo.fElementId, pthis.viewInfo._typename);

         // reset values
         sap.ui.getCore().byId("titleEx").setValue("");
         sap.ui.getCore().byId("precisionEx").setValue("");
      },

      collectionChanged: function (oEvent) {
         // console.log("collectionChanged ", oEvent.getSource());
         let model = oEvent.oSource.getSelectedItem().getBindingContext("collections");
         let path = model.getPath();
         let entry = model.getProperty(path);
         let coll = entry.collectionEveId;
         let mng = this.viewInfo;

         this.mgr.SendMIR("SetDisplayedCollection(" + coll + ")", mng.fElementId, mng._typename);
      },

      sceneElementChange: function (el) {
         this.refreshTable = true;
         if (el._typename == "ROOT::Experimental::REveTableViewInfo") {
            this.bindTableColumns = true;
         }
      },

      endChanges: function (oEvent) {
         if (this.refreshTable) {
            this.locateEveTable();
            this.buildTableHeader();
            this.buildTableBody();
            this.UpdateSelection(false);
            this.refreshTable = false;
         }
      },

      elementRemoved: function (elId) {
         let el = this.mgr.GetElement(elId);
      },

      SelectElement: function (selection_obj, element_id, sec_idcs)
      {
         if (this.mgr.global_selection_id == selection_obj.fElementId
            && element_id == this.collection.childs[0].fElementId)
            this.UpdateSelection(true);
      },

      UnselectElement: function (selection_obj, element_id)
      {
         if (this.mgr.global_selection_id == selection_obj.fElementId
            && element_id == this.collection.childs[0].fElementId)
            this.UpdateSelection(true);
      },

      UpdateSelection: function(curList)
      {
         this.table.clearSelection();

         // select rows
         let sel = this.mgr.GetElement(this.mgr.global_selection_id);
         let sel_list = curList ? sel.sel_list : sel.prev_sel_list;
         for (let i = 0; i < sel_list.length; ++i) {
            let element_id = sel_list[i].primary;
            if (element_id == this.collection.childs[0].fElementId) {
               let sec_idcs = sel_list[i].sec_idcs;
               for (let i = 0; i < sec_idcs.length; i++) {
                  let si = sec_idcs[i];
                  let ui = si;
                  if (this.table.sortMap) ui = this.table.sortMap[si];
                  this.table.addSelectionInterval(ui, ui);
               }
            }
         }
      }
   });
});
