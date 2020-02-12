sap.ui.define([
   'sap/ui/core/mvc/Controller',
   "sap/ui/model/json/JSONModel",
   "sap/ui/core/mvc/XMLView",
   "sap/m/CustomTreeItem",
   "sap/m/FlexBox",
   "sap/m/CheckBox",
   "sap/m/Text",
   "sap/m/Button",
   "sap/ui/layout/SplitterLayoutData"
], function(Controller, JSONModel, XMLView, CustomTreeItem,
            FlexBox, mCheckBox, mText, mButton, SplitterLayoutData) {

   "use strict";

   var EveSummaryCustomItem = CustomTreeItem.extend('rootui5.eve7.lib.EveSummaryCustomItem', {
      renderer: {},

      metadata: {
         properties: {
            elementId: 'string',
            background: 'string',
            mainColor: 'string',
            showCheckbox: 'boolean',
            showRnrChildren: 'boolean'
         }
      },

      onAfterRendering: function()
      {
         var btn = this.getContent()[0].getItems()[1];
         btn.$().css('background-color', this.getMainColor());
      }

   });

   return Controller.extend("rootui5.eve7.controller.Summary", {

      onInit: function () {

         var data = [{ fName: "Event" }];

         this.summaryElements = {}; // object with all elements, used for fast access to elements by id

         var oTree = this.getView().byId("tree");
         this.expandLevel = 2;

         var oModel = new JSONModel();
         oModel.setData([]);
         oModel.setSizeLimit(10000);
         oModel.setDefaultBindingMode("OneWay");
         this.getView().setModel(oModel, "treeModel");

         var oItemTemplate = new EveSummaryCustomItem({
            content: [
                new FlexBox({
                   width: "100%",
                   alignItems: "Start",
                   justifyContent: "SpaceBetween",
                   items: [
                     new FlexBox({
                        alignItems: "Start",
                        items: [
                           new mCheckBox({ visible: "{treeModel>fShowCheckbox}", selected: "{treeModel>fSelected}", select: this.clickItemSelected.bind(this) }),
                           new mText({text:" {treeModel>fName}", tooltip: "{treeModel>fTitle}" , renderWhitespace: true, wrapping: false })
                         ]
                      }),
                      new mButton({ id: "detailBtn", visible: "{treeModel>fShowButton}", icon: "sap-icon://edit", type: "Transparent", tooltip: "Actiavte GED", press: this.pressGedButton.bind(this) })
                    ]
                })
            ],

            elementId: "{treeModel>fElementId}",
            mainColor: "{treeModel>fMainColor}",
            showCheckbox: "{treeModel>fShowCheckbox}",
            showRnrChildren: "{treeModel>fShowRnrChildren}"
         });

         oItemTemplate.addStyleClass("eveSummaryItem");
         oItemTemplate.attachBrowserEvent("mouseenter", this.onMouseEnter, this);
         oItemTemplate.attachBrowserEvent("mouseleave", this.onMouseLeave, this);
         oTree.bindItems("treeModel>/", oItemTemplate);

         this.template = oItemTemplate;

         var make_col_obj = function(stem) {
            return { name: stem, member: "f" + stem, srv: "Set" + stem + "RGB", _type: "Color" };
         };

         var make_main_col_obj = function(label, use_main_setter) {
            return { name: label, member: "fMainColor", srv: "Set" + (use_main_setter ? "MainColor" : label) + "RGB", _type: "Color" };
         };

         /** Used in creating items and configuring GED */
         this.oGuiClassDef = {
            "REveElement" : [
               { name : "RnrSelf",     _type : "Bool" },
               { name : "RnrChildren", _type : "Bool" },
               make_main_col_obj("Color", true),
               { name : "Destroy",  member : "fElementId", srv : "Destroy",  _type : "Action" },
            ],
            "REveElementList" : [ { sub: ["REveElement"] }, ],
            "REveSelection"   : [ make_col_obj("VisibleEdgeColor"), make_col_obj("HiddenEdgeColor"), ],
            "REveGeoShape"    : [ { sub: ["REveElement"] } ],
            "REveCompound"    : [ { sub: ["REveElement"] } ],
            "REvePointSet" : [
               { sub: ["REveElement" ] },
               { name : "MarkerSize", _type : "Number" }
            ],
            "REveJetCone" : [
               { name : "RnrSelf", _type : "Bool" },
               make_main_col_obj("ConeColor", true),
               { name : "NDiv",    _type : "Number" }
            ],
            "REveDataCollection" : [
               { name : "FilterExpr",  _type : "String",   quote : 1 },
               { name : "CollectionVisible",  member :"fRnrSelf",  _type : "Bool" },
               make_main_col_obj("CollectionColor")
            ],
            "REveDataItem" : [
               make_main_col_obj("ItemColor"),
               { name : "ItemRnrSelf",   member : "fRnrSelf",  _type : "Bool" },
               { name : "Filtered",   _type : "Bool" }
            ],
            "REveTrack" : [
               { name : "RnrSelf",   _type : "Bool" },
               make_main_col_obj("LineColor", true),
               { name : "LineWidth", _type : "Number" },
               { name : "Destroy",  member : "fElementId",  srv : "Destroy", _type : "Action" }
            ],
         };

         this.rebuild = false;
      },

      clickItemSelected: function(oEvent) {
         var item = oEvent.getSource().getParent().getParent().getParent();

         var selected = oEvent.getSource().getSelected();

         var elem = this.mgr.GetElement(item.getElementId());

         if (item.getShowRnrChildren())
             this.mgr.SendMIR("SetRnrChildren(" + selected + ")", elem.fElementId, elem._typename);
         else
             this.mgr.SendMIR("SetRnrSelf(" + selected + ")", elem.fElementId, elem._typename);
      },

      pressGedButton: function(oEvent) {
         var item = oEvent.getSource().getParent().getParent();

         // var path = item.getBindingContext("treeModel").getPath(),
         //    ttt = item.getBindingContext("treeModel").getProperty(path);

         this.showGedEditor(item.getElementId());
      },

      SetMgr: function(mgr) {
         this.mgr = mgr;
         this.mgr.RegisterController(this);
         this.selected = {}; // container of selected objects
      },

      OnEveManagerInit: function() {
         var model = this.getView().getModel("treeModel");
         model.setData(this.createSummaryModel());
         model.refresh();

         var oTree = this.getView().byId("tree");
         oTree.expandToLevel(this.expandLevel);

         // hide editor
         if (this.ged)
            this.ged.getController().closeGedEditor();

         var scenes = this.mgr.getSceneElements();
         for (var i = 0; i < scenes.length; ++i) {
            this.mgr.RegisterSceneReceiver(scenes[i].fElementId, this);
         }
      },

      onToggleOpenState: function(oEvent) {
      },

      processHighlight: function(kind, evid, force) {

         if (!force) {
            if (this._trigger_timer)
               clearTimeout(this._trigger_timer);

            this._trigger_timer = setTimeout(this.processHighlight.bind(this,kind,evid,true), 200);
            return;
         }

         delete this._trigger_timer;

         var objid = 0;

         if (kind != "leave") {
            var tree = this.getView().byId("tree"),
                items = tree.getItems(true), item = null;
            for (var n = 0; n < items.length; ++n)
               if (items[n].getId() == evid) {
                  item = items[n]; break;
               }

            if (item) objid = item.getElementId();
         }

         // FIXME: provide more generic code which should
         this.mgr.SendMIR("NewElementPicked(" + objid + ",false,false)",
                          this.mgr.global_highlight_id, "ROOT::Experimental::REveSelection");
      },

      onMouseEnter: function(oEvent) {
         this.processHighlight("enter", oEvent.target.id);
      },

      onMouseLeave: function(oEvent) {
         this.processHighlight("leave");
      },

      GetSelectionColor: function(selection_obj) {
         return selection_obj.fName == "Global Highlight" ? "rgb(230, 230, 230)" : "rgb(66, 124, 172)";
      },

      FindTreeItemForEveElement:function(element_id) {
         var items = this.getView().byId("tree").getItems();
         for (var n = 0; n<items.length;++n)
            if (items[n].getElementId() == element_id)
               return items[n];
         return null;
      },

      SelectElement: function(selection_obj, element_id, sec_idcs) {
         var item = this.FindTreeItemForEveElement(element_id);
         if (item) {
            var color = this.GetSelectionColor(selection_obj);
            item.$().css("background-color", color);
         }
      },

      UnselectElement: function (selection_obj, element_id) {
         var item = this.FindTreeItemForEveElement(element_id);
         if (item) {
            var color = this.GetSelectionColor(selection_obj);
            var cc = item.$().css("background-color");
            if (cc == color)
               item.$().css("background-color", "");
         }
      },

      showGedEditor: function(elementId) {

         var sumSplitter = this.byId("sumSplitter");

         if (!this.ged) {
            var pthis = this;

            XMLView.create({
               viewName: "rootui5.eve7.view.Ged",
               viewData: { summaryCtrl : this },
               layoutData: new SplitterLayoutData("sld", {size : "30%"}),
               height: "100%"
            }).then(function(oView) {
               pthis.ged = oView;

               pthis.ged.getController().showGedEditor(sumSplitter, elementId);

            });
         } else {
            this.ged.getController().showGedEditor(sumSplitter, elementId);
         }
      },

      canEdit: function(elem) {
         var t = elem._typename.substring(20);
         var ledit = this.oGuiClassDef;
         if (ledit.hasOwnProperty(t))
            return true;
         return false;
      },

      anyVisible: function(arr) {
         if (!arr) return false;
         for (var k=0;k<arr.length;++k) {
            if (arr[k].fName) return true;
         }
         return false;
      },

      /** Set summary element attributes from original element */
      setElementsAttributes: function(newelem, elem) {
         newelem.fShowCheckbox = false;
         newelem.fShowRnrChildren = false;
         newelem.fElementId = elem.fElementId;
         newelem.fShowButton = false;
         newelem.fMainColor = "";

         if (this.canEdit(elem)) {
            newelem.fShowButton = true;

            if (!elem.childs) {
               newelem.fShowCheckbox = true;
               newelem.fSelected = elem.fRnrSelf;
            }  else if (elem.fRnrChildren !== undefined) {
               newelem.fShowCheckbox = true;
               newelem.fShowRnrChildren = true;
               newelem.fSelected = elem.fRnrChildren;
            }

            if (elem.fMainColor) {
               newelem.fMainColor = JSROOT.Painter.root_colors[elem.fMainColor];
            }
         }
      },

      createSummaryModel: function(tgt, src, path) {
         if (tgt === undefined) {
            tgt = [];
            src = this.mgr.childs;
            this.summaryElements = {};
            path = "/";
            // console.log('original model', src);
         }
         for (var n=0;n<src.length;++n) {
            var elem = src[n];

            var newelem = { fName: elem.fName, fTitle: elem.fTitle || elem.fName, id: elem.fElementId, fHighlight: "None", fBackground: "", fMainColor: "", fSelected: false };

            this.setElementsAttributes(newelem, elem);

            newelem.path = path + n;
            newelem.masterid = elem.fMasterId || elem.fElementId;

            tgt.push(newelem);

            this.summaryElements[newelem.id] = newelem;

            if ((elem.childs !== undefined) && this.anyVisible(elem.childs))
               newelem.childs = this.createSummaryModel([], elem.childs, newelem.path + "/childs/");
         }

         return tgt;
      },

      beginChanges: function() {
        // this.rebuild=false;
      },

      elementsRemoved: function(ids) {
         this.rebuild = true;
      },

      sceneElementChange: function(msg) {

         if (this.ged)
            this.ged.getController().updateGED(msg.fElementId);

         var newelem = this.summaryElements[msg.fElementId];

         var elem = this.mgr.GetElement(msg.fElementId);

         if (newelem && elem)
            this.setElementsAttributes(newelem, elem);

         // console.log('SUMMURY: detect changed', elem.id, elem.path);

         if (msg.rnr_self_changed)
            elem.fSelected = msg.fRnrSelf;

         this.any_changed = true;

      },

      endChanges: function() {
         if (this.rebuild) {
            var oTree = this.getView().byId("tree");
            oTree.unbindItems();

            var model = this.getView().getModel("treeModel");
            model.setData(this.createSummaryModel());
            model.refresh();

            this.getView().setModel(model, "treeModel");
            oTree.bindItems("treeModel>/", this.template);
            oTree.setModel(model, "treeModel");

            oTree.expandToLevel(this.expandLevel);

            if (this.ged)
               this.ged.getController().closeGedEditor();

            this.rebuild = false;
         } else if (this.any_changed) {
            var model = this.getView().getModel("treeModel");
            model.refresh();

            this.any_changed = false;
         }
      },

      /** Invoked via EveManager when specified element should be focused */
      BrowseElement: function(elid) {
         var summaryElement = this.summaryElements[elid];
         if (!summaryElement) return;

         var oTree = this.getView().byId("tree"),
             element_path = summaryElement.path,
             bestindx = 1, bestlen = 0;

         while (bestindx >= 0) {

            bestindx = -1;

            var items = oTree.getItems();

            for (var k = 0; k < items.length; ++k) {
               var item = items[k],
                   model = item.getBindingContext("treeModel"),
                   path = model.getPath();

               if (element_path == path) {
                  var dom = item.$()[0];
                  if (dom) dom.scrollIntoView();
                  return;
               }

               if ((element_path.substr(0, path.length) == path) && (path.length > bestlen)) {
                  bestindx = k;
                  bestlen = path.length;
               }
            }

            if (bestindx >= 0) oTree.expand(bestindx);
         }

      }
   });
});
