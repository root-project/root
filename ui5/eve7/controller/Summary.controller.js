sap.ui.define([
   'sap/ui/core/mvc/Controller',
   "sap/ui/model/json/JSONModel",
   "sap/m/Button",
   "sap/m/ColorPalettePopover",
   "sap/m/StandardTreeItem",
   "sap/m/Input",
   "sap/m/CheckBox",
   "sap/m/Panel",
   "sap/m/Text",
   "sap/ui/layout/SplitterLayoutData",
   "sap/ui/layout/VerticalLayout",
   "sap/ui/layout/HorizontalLayout"
], function(Controller, JSONModel, Button, ColorPalettePopover, StandardTreeItem, 
            mInput, mCheckBox, mPanel, mText, 
            SplitterLayoutData, VerticalLayout, HorizontalLayout) {

   "use strict";
   
   var UI5PopupColors = {
         aliceblue: 'f0f8ff',
         antiquewhite: 'faebd7',
         aqua: '00ffff',
         aquamarine: '7fffd4',
         azure: 'f0ffff',
         beige: 'f5f5dc',
         bisque: 'ffe4c4',
         black: '000000',
         blanchedalmond: 'ffebcd',
         blue: '0000ff',
         blueviolet: '8a2be2',
         brown: 'a52a2a',
         burlywood: 'deb887',
         cadetblue: '5f9ea0',
         chartreuse: '7fff00',
         chocolate: 'd2691e',
         coral: 'ff7f50',
         cornflowerblue: '6495ed',
         cornsilk: 'fff8dc',
         crimson: 'dc143c',
         cyan: '00ffff',
         darkblue: '00008b',
         darkcyan: '008b8b',
         darkgoldenrod: 'b8860b',
         darkgray: 'a9a9a9',
         darkgrey: 'a9a9a9',
         darkgreen: '006400',
         darkkhaki: 'bdb76b',
         darkmagenta: '8b008b',
         darkolivegreen: '556b2f',
         darkorange: 'ff8c00',
         darkorchid: '9932cc',
         darkred: '8b0000',
         darksalmon: 'e9967a',
         darkseagreen: '8fbc8f',
         darkslateblue: '483d8b',
         darkslategray: '2f4f4f',
         darkslategrey: '2f4f4f',
         darkturquoise: '00ced1',
         darkviolet: '9400d3',
         deeppink: 'ff1493',
         deepskyblue: '00bfff',
         dimgray: '696969',
         dimgrey: '696969',
         dodgerblue: '1e90ff',
         firebrick: 'b22222',
         floralwhite: 'fffaf0',
         forestgreen: '228b22',
         fuchsia: 'ff00ff',
         gainsboro: 'dcdcdc',
         ghostwhite: 'f8f8ff',
         gold: 'ffd700',
         goldenrod: 'daa520',
         gray: '808080',
         grey: '808080',
         green: '008000',
         greenyellow: 'adff2f',
         honeydew: 'f0fff0',
         hotpink: 'ff69b4',
         indianred: 'cd5c5c',
         indigo: '4b0082',
         ivory: 'fffff0',
         khaki: 'f0e68c',
         lavender: 'e6e6fa',
         lavenderblush: 'fff0f5',
         lawngreen: '7cfc00',
         lemonchiffon: 'fffacd',
         lightblue: 'add8e6',
         lightcoral: 'f08080',
         lightcyan: 'e0ffff',
         lightgoldenrodyellow: 'fafad2',
         lightgray: 'd3d3d3',
         lightgrey: 'd3d3d3',
         lightgreen: '90ee90',
         lightpink: 'ffb6c1',
         lightsalmon: 'ffa07a',
         lightseagreen: '20b2aa',
         lightskyblue: '87cefa',
         lightslategray: '778899',
         lightslategrey: '778899',
         lightsteelblue: 'b0c4de',
         lightyellow: 'ffffe0',
         lime: '00ff00',
         limegreen: '32cd32',
         linen: 'faf0e6',
         magenta: 'ff00ff',
         maroon: '800000',
         mediumaquamarine: '66cdaa',
         mediumblue: '0000cd',
         mediumorchid: 'ba55d3',
         mediumpurple: '9370db',
         mediumseagreen: '3cb371',
         mediumslateblue: '7b68ee',
         mediumspringgreen: '00fa9a',
         mediumturquoise: '48d1cc',
         mediumvioletred: 'c71585',
         midnightblue: '191970',
         mintcream: 'f5fffa',
         mistyrose: 'ffe4e1',
         moccasin: 'ffe4b5',
         navajowhite: 'ffdead',
         navy: '000080',
         oldlace: 'fdf5e6',
         olive: '808000',
         olivedrab: '6b8e23',
         orange: 'ffa500',
         orangered: 'ff4500',
         orchid: 'da70d6',
         palegoldenrod: 'eee8aa',
         palegreen: '98fb98',
         paleturquoise: 'afeeee',
         palevioletred: 'db7093',
         papayawhip: 'ffefd5',
         peachpuff: 'ffdab9',
         peru: 'cd853f',
         pink: 'ffc0cb',
         plum: 'dda0dd',
         powderblue: 'b0e0e6',
         purple: '800080',
         red: 'ff0000',
         rosybrown: 'bc8f8f',
         royalblue: '4169e1',
         saddlebrown: '8b4513',
         salmon: 'fa8072',
         sandybrown: 'f4a460',
         seagreen: '2e8b57',
         seashell: 'fff5ee',
         sienna: 'a0522d',
         silver: 'c0c0c0',
         skyblue: '87ceeb',
         slateblue: '6a5acd',
         slategray: '708090',
         slategrey: '708090',
         snow: 'fffafa',
         springgreen: '00ff7f',
         steelblue: '4682b4',
         tan: 'd2b48c',
         teal: '008080',
         thistle: 'd8bfd8',
         tomato: 'ff6347',
         turquoise: '40e0d0',
         violet: 'ee82ee',
         wheat: 'f5deb3',
         white: 'ffffff',
         whitesmoke: 'f5f5f5',
         yellow: 'ffff00',
         yellowgreen: '9acd32',
         transparent: '00000000'
   };// colorButton colors
   
   
   // TODO: move to separate file
   var EVEColorButton = Button.extend("rootui5.eve7.controller.EVEColorButton", {
      // when default value not specified - openui tries to load custom
      renderer: {}, // ButtonRenderer.render,

      metadata: {
         properties: {
            background: 'string'
         }
      },

      onAfterRendering: function() {
         this.$().children().css("background-color", this.getBackground());
      }

   });

   var EveSummaryTreeItem = StandardTreeItem.extend('rootui5.eve7.lib.EveSummaryTreeItem', {
      // when default value not specified - openui tries to load custom
      renderer: {},

      metadata: {
         properties: {
            background: 'string'
         }
      },

      onAfterRendering: function() {
         this.$().css("background-color", this.getBackground());
      }

   });

   return Controller.extend("rootui5.eve7.controller.Summary", {

      onInit: function () {

         console.log('Summary CONTROLLER INIT');

         var data = [{ fName: "Event" }];

         var oTree = this.getView().byId("tree");
         oTree.setMode(sap.m.ListMode.Single);
         oTree.setIncludeItemInSelection(true);

         if (false) {
            var oModel = new JSONModel();
            oModel.setData([]);
            oModel.setSizeLimit(10000);
            this.getView().setModel(oModel, "treeModel");

         } else {
            var oModel = new JSONModel();
            oModel.setData([]);
            oModel.setSizeLimit(10000);
            this.getView().setModel(oModel, "treeModel");

            var oItemTemplate = new EveSummaryTreeItem({
               title: "{treeModel>fName}",
               visible: "{treeModel>fVisible}",
               type: "{treeModel>fType}",
               highlight: "{treeModel>fHighlight}",
               background: "{treeModel>fBackground}"
            });
            oItemTemplate.attachDetailPress({}, this.onDetailPress, this);
            oItemTemplate.attachBrowserEvent("mouseenter", this.onMouseEnter, this);
            oItemTemplate.attachBrowserEvent("mouseleave", this.onMouseLeave, this);
            /*
              var oDataTemplate = new sap.ui.core.CustomData({
              key:"eveElement"
              });
              oDataTemplate.bindProperty("value", "answer");
            */
            oTree.bindItems("treeModel>/", oItemTemplate);
         }

         this.oModelGED = new JSONModel({ "widgetlist" : [] });
         this.getView().setModel(this.oModelGED, "ged");
         
         this.oGuiClassDef = {
            "REveElement" : [{
               name : "RnrSelf",
               _type   : "Bool"
            }, {
               name : "RnrChildren",
               _type   : "Bool"
            }, {
               name : "MainColor",
               member: "fMainColor",
               srv  : "SetMainColorRGB",
               _type   : "Color"
            }, {
               name : "Destroy ",
               member: "fElementId",
               srv  : "Destroy",
               _type   : "Action"
            }],
            "REveElementList" : [ {sub: ["REveElement"]}],
            "REveGeoShape" : [ {sub: ["REveElement"]}, ],
            "REveCompound" : [ {sub: ["REveElement"]}],
            "REvePointSet" : [
            {
               sub: ["REveElement" ]
            }, {
               name : "MarkerSize",
               _type   : "Number"
            }],
             "REveJetCone" : [
            {
               name : "RnrSelf",
               _type   : "Bool"
            }, {
               name : "ConeColor",
               member: "fMainColor",
               srv  : "SetMainColorRGB",
                _type   : "Color"
            }, {
               name : "NDiv",
               _type   : "Number"
            }],
            "REveDataCollection" : [{
               name : "FilterExpr",
                _type   : "String",
                quote : 1
            },{
                name : "CollectionVisible",
                member:"fRnrSelf",
               _type   : "Bool"
            }, {
               name : "Collection Color",
               member: "fMainColor",
               srv  : "SetCollectionColorRGB",
               _type   : "Color"
           }],
           "REveDataItem" : [{
               name : "ItemColor",
               member: "fMainColor",
               srv: "SetItemColorRGB",
               _type   : "Color"
           },{
               name : "ItemRnrSelf",
               member: "fRnrSelf",
               _type   : "Bool"
           },{
               name : "Filtered",
               _type   : "Bool"
           }],
           "REveTrack" : [
            {
               name : "RnrSelf",
               _type   : "Bool"
            }, {
               name : "LineColor",
               member: "fMainColor",
               srv  : "SetMainColorRGB",
               _type   : "Color"
            }, {
               name : "LineWidth",
               _type   : "Number"
            }, {
               name : "Destroy ",
               member: "fElementId",
               srv  : "Destroy",
               _type   : "Action"
            }],
           "REveDataGeoShape" : [{
            }]
         };

      },

      SetMgr: function(mgr) {
         this.mgr = mgr;

         this.mgr.RegisterUpdate(this, "UpdateMgr");
         this.mgr.RegisterElementUpdate(this, "updateGED");

         this.selected = {}; // container of selected objects
         // process scene-specific events
         this.mgr.addSceneHandler(this);
      },

      UpdateMgr: function(mgr) {

         console.log('UPDATE MGR', (new Date).toTimeString());
         var model = this.getView().getModel("treeModel");
         model.setData(this.createSummaryModel());
         model.refresh();

         var oTree = this.getView().byId("tree");
         oTree.expandToLevel(2);

         // hide editor
         if (this.ged) {
            var gedFrame =  this.gedVert;
            gedFrame.unbindElement();
            gedFrame.destroyContent();
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

      event: function(lst) {
         this._event = lst;
         // console.log("summary event lst \n", lst);

         var oTreeData = {fName: "unset"}

         oTreeData.arr = [];
         this.addNodesToTreeItemModel(lst, oTreeData);
         // console.log("event model ", { "top" : oTreeData});

         this.model.setData({ fName: "Top", arr: oTreeData }); // ??? is this necessary

         this.model.refresh(true);
         this.tree.expandToLevel(2);
         this.getView().setModel(this.model, "treeModel");

         this.oProductModel = new JSONModel();
         this.oProductModel.setData([this._event]);
         this.getView().setModel(this.oProductModel, "event");
      },

      makeDataForGED: function (element) {
         // remove ROOT::Experimental::
         var shtype = element._typename.substring(20);
         var cgd = this.oGuiClassDef[shtype];
         var arrw = [];
         var modelw = [];

         this.maxLabelLength = 0;
         var off = 0;

         // sub editors
         var subEds= [];
         if (cgd[0].sub) {
            off = 1;
            var sarr = cgd[0].sub;
            for (var i = 0; i< sarr.length; ++i) {
               var x = this.oGuiClassDef[sarr[i]];
               for (var j=0; j < x.length; j++)
               {
                  arrw.push(x[j]);
               }
            }
         }

         for (var i = off; i < cgd.length; ++i)
         {
            arrw.push(cgd[i]);
         }

          for (var i=0; i< arrw.length; ++i) {
            var parName = arrw[i].name;

            if (!arrw[i].member) {
               arrw[i].member = "f" + parName;
            }

            if (!arrw[i].srv) {
               arrw[i].srv = "Set" + parName;
            }

            var v  = element[arrw[i].member];
            if (arrw[i]._type == "Color") {
               v = JSROOT.Painter.root_colors[v];
            }
            var labeledInput = {
               value: v,
               name: arrw[i].name,
               data: arrw[i]
            };
            
            modelw.push({ value: v, name: arrw[i].name, data: arrw[i]});

            if (this.maxLabelLength < arrw[i].name.length) this.maxLabelLength = arrw[i].name.length;
          }
          
          this.oModelGED.setData({ "widgetlist": modelw });
      },

      /** Selection of element in the other editors */
      setElementSelected: function(mstrid, col, indx, from_interactive) {
         if (!from_interactive)
            this.selected[mstrid] = { id: mstrid, col: col, indx: indx };

         var model = this.getView().getModel("treeModel");

         this.iterateTreeModel(model.getData(), function(elem) {
            if (elem.masterid == mstrid) elem.fHighlight = (col && mstrid) ? "Information" : "None";
         });

         model.refresh();
      },

      iterateTreeModel: function(data, func) {
         if (!data) return;

         for (var k=0;k<data.length;++k) {
            func(data[k]);
            if (data[k].childs)
               this.iterateTreeModel(data[k].childs, func);
         }
      },

      setElementHighlighted: function(mstrid, col, indx) {
         var model = this.getView().getModel("treeModel");

         this.iterateTreeModel(model.getData(), function(elem) {
            if (elem.masterid == mstrid) elem.fBackground = (col && mstrid) ? "yellow" : "";
         });

         model.refresh();

         /*
         var items = this.getView().byId("tree").getItems();

         for (var n = 0; n<items.length;++n) {
            var item = items[n],
                ctxt = item.getBindingContext("treeModel"),
                path = ctxt.getPath(),
                ttt = item.getBindingContext("treeModel").getProperty(path);

            var h_col = (col && mstrid && (mstrid == ttt.masterid)) ? "yellow" : "";
            if (!h_col && ttt.sel_color) h_col = ttt.sel_color;

            item.$().css("background-color", h_col);
         }
         */
      },

      onItemPressed: function(oEvent) {
         var model = oEvent.getParameter("listItem").getBindingContext("treeModel"),
             path =  model.getPath(),
             ttt = model.getProperty(path);
         
         console.log("Summary::onItemPressed ", this.mgr.GetElement(ttt.id));
         if (!ttt || (ttt.childs !== undefined) || !ttt.masterid) return;

         var sel_color = ttt.fHighlight == "None" ? "blue" : "";

         this.setElementSelected(ttt.masterid, sel_color, undefined);

         this.mgr.invokeInOtherScenes(this, "setElementSelected", ttt.masterid, sel_color, undefined);

         // var obj = this.mgr.GetElement(ttt.id);
      },


      onToggleOpenState: function(oEvent) {
      },

      onMouseEnter: function(oEvent) {
         var items = this.getView().byId("tree").getItems(), item = null;
         for (var n = 0; n < items.length; ++n)
            if (items[n].getId() == oEvent.target.id) {
               item = items[n]; break;
            }

         // var item = this.getView().byId(oEvent.target.id).getControl();

         if (!item) return;

         var path = item.getBindingContext("treeModel").getPath();

         var ttt = item.getBindingContext("treeModel").getProperty(path);

         var masterid = this.mgr.GetMasterId(ttt.id);

         this.mgr.invokeInOtherScenes(this, "setElementHighlighted", masterid, "cyan");
      },

      onMouseLeave: function(oEvent) {
         // actual call will be performed 100ms later and can be overwritten

         var items = this.getView().byId("tree").getItems(), item = null;
         for (var n = 0; n < items.length; ++n)
            if (items[n].getId() == oEvent.target.id) {
               item = items[n]; break;
            }

         // var item = this.getView().byId(oEvent.target.id).getControl();

         if (!item) return;

         var path = item.getBindingContext("treeModel").getPath();

         var ttt = item.getBindingContext("treeModel").getProperty(path);

         var masterid = this.mgr.GetMasterId(ttt.id);

         this.mgr.invokeInOtherScenes(this, "setElementHighlighted", masterid, null);
      },

      toggleEditor: function() {
         var pp = this.byId("sumSplitter");
         if (!this.ged) {
            var panel = new mPanel("productDetailsPanel", { height: "100%", width: "97%" });
            panel.setHeaderText("ElementGED");
            panel.addStyleClass("sapUiSizeCompact");

            panel.setLayoutData(new SplitterLayoutData("sld", {size : "30%"}));
            pp.addContentArea(panel);

            var vert = new VerticalLayout("GED",  {});
            vert.addStyleClass("sapUiSizeCompact");
            vert.addStyleClass("eveTreeItem");
            vert.addStyleClass("sapUiNoMarginTop");
            vert.addStyleClass("sapUiNoMarginBottom");

            panel.addContent(vert);
            this.ged = panel;
            this.gedVert = vert;
            this.ged.visible = true;
         } else if (this.ged.visible) {
            pp.removeContentArea(this.ged);
            this.ged.visible = false;
         } else {
            pp.addContentArea(this.ged);
            this.ged.visible = true;
         }
      },

      onDetailPress: function(oEvent) {
         // when edit button pressed
         var item = oEvent.getSource(),
             path = item.getBindingContext("treeModel").getPath(),
             ttt = item.getBindingContext("treeModel").getProperty(path);

         if (!ttt) return;

         if (!this.ged || !this.ged.visible) {
            this.toggleEditor();
         } else if (this.ged.editorItemPath == path) {
            this.toggleEditor(); // hide editor when clicked several times
            return;
         }

         this.ged.editorItemPath = path;

         this.editorElement = this.mgr.GetElement(ttt.id);

         var oProductDetailPanel = this.ged;
        // var oProductDetailPanel = this.byId("productDetailsPanel");
         var title = this.editorElement.fName + " (" +  this.editorElement._typename.substring(20) + " )" ;
         oProductDetailPanel.setHeaderText(title);

         //var oProductDetailPanel = this.byId("productDetailsPanel");
         // console.log("event path ", eventPath);
         var eventPath = item.getBindingContext("treeModel").getPath();
         oProductDetailPanel.bindElement({ path: eventPath, model: "event" });

         var gedFrame =  this.gedVert;
         gedFrame.unbindElement();
         gedFrame.destroyContent();

         this.makeDataForGED(this.editorElement);

         // console.log("going to bind >>> ", this.getView().getModel("ged"));
         gedFrame.bindAggregation("content", "ged>/widgetlist",  this.gedFactory.bind(this) );

      },

      gedFactory: function(sId, oContext) {
         var base = "/widgetlist/";
         var path = oContext.getPath();
         var idx = path.substring(base.length);
         var customData =  oContext.oModel.oData["widgetlist"][idx].data;
         var controller = this;
         var widget = null;

         switch (customData._type) {

         case "Number":
            widget = new mInput(sId, {
               value: { path: "ged>value" },
               change: this.sendMethodInvocationRequest.bind(this, "Number")
            });
            widget.setType(sap.m.InputType.Number);
            break;

         case "String":
            widget = new mInput(sId, {
               value: { path: "ged>value" },
               change: this.sendMethodInvocationRequest.bind(this, "String")

            });
            widget.setType(sap.m.InputType.String);
            widget.setWidth("250px"); // AMT this should be handled differently
            break;
         case "Bool":
            widget = new mCheckBox(sId, {
               selected: { path: "ged>value" },
               select: this.sendMethodInvocationRequest.bind(this, "Bool")
            });
            break;

         case "Color":
            var colVal = oContext.oModel.oData["widgetlist"][idx].value;
            // var model = this.getView().getModel("colors");
            //   model["mainColor"] = colVal;
            //  console.log("col value ", colVal, JSROOT.Painter.root_colors[colVal]);
            widget = new EVEColorButton(sId, {
               icon: "sap-icon://palette",
               background: colVal,

               press: function () {
                  var colButton = this;
                  var oCPPop = new ColorPalettePopover( {
                      defaultColor: "cyan",
                       colors: ['gold','darkorange', 'indianred','rgb(102,51,0)', 'cyan',// 'magenta'
                                'blue', 'lime', 'gray','slategray','rgb(204, 198, 170)',
                                'white', 'black','red' , 'rgb(102,154,51)', 'rgb(200, 0, 200)'],
                       colorSelect: function(event) {
                          colButton.setBackground(event.getParameters().value);
                          controller.handleColorSelect(event);
                       }
                   });

                   oCPPop.openBy(colButton);
                   oCPPop.data("myData", customData);
                 }
            });
            break;

         case "Action":
            widget = new Button(sId, {
               //text: "Action",
               icon: "sap-icon://accept",
               press: this.sendMethodInvocationRequest.bind(this, "Action")
            });
            break;
         }

         if (widget) widget.data("myData", customData);

         var label = new mText(sId + "label", { text: { path: "ged>name" } });
         label.setWidth(this.maxLabelLength +"ex");
         label.addStyleClass("sapUiTinyMargin");
         
         return new HorizontalLayout({
            content : [label, widget]
         });
      },

      handleColorSelect: function(event) {
          var val = event.getParameters().value;
          var myData = event.getSource().data("myData");

         var rgb,
             regex = /rgb\((\d+)\,\s?(\d+)\,\s?(\d+)\)/,
             found = val.match(regex);
         if (found) {
            console.log("match color ", found);
            /*
            rgb.r = found[1];
            rgb.g = found[2];
            rgb.b = found[3];
            */
            rgb = { r: found[1], g: found[2], b: found[3] };
         } else {
            var hex = UI5PopupColors[val];

            // Expand shorthand form (e.g. "03F") to full form (e.g. "0033FF")
            var shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;

            hex = hex.replace(shorthandRegex, function(m, r, g, b) {
               return r + r + g + g + b + b;
            });

            rgb = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);

            rgb = rgb ? { r: parseInt(rgb[1], 16), g: parseInt(rgb[2], 16), b: parseInt(rgb[3], 16) } : null;
         }


         var mir =  myData.srv + "((UChar_t)" + rgb.r + ", (UChar_t)" + rgb.g +  ", (UChar_t)" + rgb.b + ")";
         var obj = { "mir": mir, "fElementId": this.editorElement.fElementId, "class": this.editorElement._typename };
         this.mgr.handle.Send(JSON.stringify(obj));
      },

      sendMethodInvocationRequest: function(kind, event) {
         var value = "";
         switch (kind) {
         case "Bool":
            value = event.getSource().getSelected();
            break;

         case "Action":
            value = "";
            break;
         default:
            value =  event.getParameter("value");
         }

         console.log("on change !!!!!!", event.getSource().data("myData"));

         if (event.getSource().data("myData").quote !== undefined ) {
              value = "\"" + value + " \"";
         }
         var mir =  event.getSource().data("myData").srv + "( " + value + " )";

         console.log("=====> ", mir);
         var obj = {"mir" : mir, "fElementId" : this.editorElement.fElementId, "class" : this.editorElement._typename};

         this.mgr.handle.Send(JSON.stringify(obj));
      },

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

      updateGED : function (elementId) {
         if (!this.editorElement) return;
         if (this.editorElement.fElementId == elementId) {
            var gedFrame =  this.gedVert;
            gedFrame.unbindElement();
            gedFrame.destroyContent();
            this.makeDataForGED(this.editorElement);
            gedFrame.bindAggregation("content", "ged>/widgetlist", this.gedFactory.bind(this));
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

      createSummaryModel: function(tgt, src) {
         if (tgt === undefined) {
            tgt = [];
            src = this.mgr.childs;
            // console.log('original model', src);
         }
         for (var n=0;n<src.length;++n) {
            var elem = src[n];

            var newelem = { fName: elem.fName, id: elem.fElementId, fHighlight: "None", fBackground: "" };

            if (this.canEdit(elem))
               newelem.fType = "DetailAndActive";
            else
               newelem.fType = "Active";

            newelem.masterid = elem.fMasterId || elem.fElementId;

            tgt.push(newelem);
            if ((elem.childs !== undefined) && this.anyVisible(elem.childs))
               newelem.childs = this.createSummaryModel([], elem.childs);
         }

         return tgt;
      },

      endChanges: function(rebuild) {
         if (rebuild) updateManger(this.mgr);
      }

   });
});
