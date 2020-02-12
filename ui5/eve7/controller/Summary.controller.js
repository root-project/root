sap.ui.define([
   'sap/ui/core/mvc/Controller',
   "sap/ui/model/json/JSONModel",
   "sap/ui/core/mvc/XMLView",
   "sap/m/StandardTreeItem",
   "sap/m/CustomTreeItem",
   "sap/m/CheckBox",
   "sap/m/Text",
   "sap/ui/layout/SplitterLayoutData"
], function(Controller, JSONModel, XMLView,
            StandardTreeItem, CustomTreeItem,
            mCheckBox, mText, SplitterLayoutData) {

   "use strict";

   var EveSummaryTreeItem = StandardTreeItem.extend('rootui5.eve7.lib.EveSummaryTreeItem', {
      // when default value not specified - openui tries to load custom
      renderer: {},

      metadata: {
         properties: {
            background: 'string',
            mainColor: 'string',
            showCheckbox: 'boolean',
            showRnrChildren: 'boolean'
         }
      },

      onAfterRendering: function() {
         if (!this.getShowCheckbox()) {
            // TODO: find better way to select check box
            var chkbox = this.$().children().first().next();
            if (chkbox.attr("role") == "checkbox")
               chkbox.css("display","none");
         } else {
            this.$().children().last().css("background-color", this.getMainColor());
         }

         // this.$().css("background-color", this.getBackground());
      }

   });

   return Controller.extend("rootui5.eve7.controller.Summary", {

      onInit: function () {

         var data = [{ fName: "Event" }];

         this.summaryElements = {}; // object with all elements, used for fast access to elements by id

         var oTree = this.getView().byId("tree");
         oTree.setMode(sap.m.ListMode.MultiSelect);
         oTree.setIncludeItemInSelection(true);
         this.expandLevel = 2;

         var oModel = new JSONModel();
         oModel.setData([]);
         oModel.setSizeLimit(10000);
         oModel.setDefaultBindingMode("OneWay");
         this.getView().setModel(oModel, "treeModel");

         var oItemTemplate = new EveSummaryTreeItem({
            title: "{treeModel>fName}",
            visible: "{treeModel>fVisible}",
            type: "{treeModel>fType}",
            highlight: "{treeModel>fHighlight}",
            background: "{treeModel>fBackground}",
            tooltip: "{treeModel>fTitle}",
            selected: "{treeModel>fSelected}",
            mainColor: "{treeModel>fMainColor}",
            showCheckbox: "{treeModel>fShowCheckbox}",
            showRnrChildren: "{treeModel>fShowRnrChildren}"
         });

/*
         var oItemTemplate = new CustomTreeItem({
            content: [
                 new mCheckBox({ selected: "{treeModel>fVisible}", select: this.clickItemSelected.bind(this) }),
                 new mText({text:" {treeModel>fName}", tooltip: "{treeModel>fTitle}" , renderWhitespace: true, wrapping: false })
            ]
         });
*/
         oItemTemplate.addStyleClass("eveSummaryItem");
         oItemTemplate.attachDetailPress({}, this.onDetailPress, this);
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

      },

      onSelectionChange: function(oEvent) {

         var items = oEvent.getParameters().listItems;
         if (!items) return;

         for (var k = 0; k < items.length; ++k) {
            var item = items[k];
            if (!item) continue;

            if (!item.getShowCheckbox()) {
               // workaround, to suppress checkboxes from standard item
               item.setSelected(false);
               var chkbox = item.$().children().first().next();
               if (chkbox.attr("role") == "checkbox")
                  chkbox.css("display","none");
            } else {

               var  path = item.getBindingContext("treeModel").getPath(),
                    ttt = item.getBindingContext("treeModel").getProperty(path);

               var elem = this.mgr.GetElement(ttt.id);

               if (item.getShowRnrChildren())
                  this.mgr.SendMIR("SetRnrChildren(" + item.getSelected() + ")", elem.fElementId, elem._typename);
               else
                  this.mgr.SendMIR("SetRnrSelf(" + item.getSelected() + ")", elem.fElementId, elem._typename);
            }
         }

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

      addNodesToTreeItemModel: function(el, model) {
         // console.log("FILL el ", el.fName)
         model.fName = el.fName;
         model.guid = el.guid;
         if (el.arr) {
            model.arr = new Array(el.arr.length);
            for (var n=0; n< el.arr.length; ++n) {
               model.arr[n]= { fName: "unset"};
               this.addNodesToTreeItemModel(el.arr[n], model.arr[n]);
            }
         }

         /*
           for (var n=0; n< lst.arr.length; ++n)
           {
           var el = lst.arr[n];
           var node = {
           "fName" : el.fName,
           "guid" : el.guid
           };

           model.arr.push(node);
           if (el.arr) {
           node.arr = [];
           this.addNodesToTreeItemModel(el, node);
           }
           }
    */
      },

      addNodesToCustomModel:function(lst, model) {/*
                      for ((var n=0; n< lst.arr.length; ++n))
                      {
                      var el = lst.arr[n];
                      var node = {fName : el.fName , guid : el.guid};
                      model.push(node);
                      if (el.arr) {
                      node.arr = [];
                      addNodesToTreeItemModel(el, node);
                      }
                      }
                    */
      },


      /** When item pressed - not handled now */
      onItemPressed: function(oEvent) {
         var listItem = oEvent.getParameter("listItem");
         //     model = listItem.getBindingContext("treeModel"),
         //     path =  model.getPath(),
         //     ttt = model.getProperty(path);


         // workaround, to suppress checkboxes from standard item
         if (!listItem.getShowCheckbox()) {
            listItem.setSelected(false);
            var chkbox = listItem.$().children().first().next();
            if (chkbox.attr("role") == "checkbox")
               chkbox.css("display","none");
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
                items = tree.getItems(true),         item = null;
            for (var n = 0; n < items.length; ++n)
               if (items[n].getId() == evid) {
                  item = items[n]; break;
               }

            if (item) {
               var path = item.getBindingContext("treeModel").getPath();
               var ttt = item.getBindingContext("treeModel").getProperty(path);
               objid = ttt.id;
            }
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
         for (var n = 0; n<items.length;++n) {
            var item = items[n],
                ctxt = item.getBindingContext("treeModel"),
                path = ctxt.getPath(),
                ttt = item.getBindingContext("treeModel").getProperty(path);

            if (ttt.id == element_id)
               return item;
         }
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

      /** When edit button pressed */
      onDetailPress: function(oEvent) {
         var item = oEvent.getSource(),
             path = item.getBindingContext("treeModel").getPath(),
             ttt = item.getBindingContext("treeModel").getProperty(path);

         this.showGedEditor(path, ttt);
      },

      showGedEditor: function(path, newelem) {

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

               pthis.ged.getController().showGedEditor(sumSplitter, path, newelem.id);

            });
         } else {
            this.ged.getController().showGedEditor(sumSplitter, path, newelem.id);
         }
      },

/*
      changeNumPoints:function() {
         var myJSON = "changeNumPoints(" +  this.editorElement.guid + ", "  + this.editorElement.fN +  ")";
         this.mgr.handle.Send(myJSON);
      },

      printEvent: function(event) {
         var propertyPath = event.getSource().getBinding("value").getPath();
         // console.log("property path ", propertyPath);
         var bindingContext = event.getSource().getBindingContext("event");

         var path =  bindingContext.getPath(propertyPath);
         var object =  bindingContext.getObject(propertyPath);
         // console.log("obj ",object );

         this.changeNumPoints();
      },

      changeRnrSelf: function(event) {
         var myJSON = "changeRnrSelf(" +  this.editorElement.guid + ", "  + event.getParameters().selected +  ")";
         this.mgr.handle.Send(myJSON);
      },

      changeRnrChld: function(event) {
         console.log("change Rnr ", event, " source ", event.getSource());
      },
*/

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

         if (this.canEdit(elem)) {
            newelem.fType = "DetailAndActive";
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
         } else {
            newelem.fType = "Active";
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
