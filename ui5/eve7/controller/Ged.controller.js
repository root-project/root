sap.ui.define([
   "sap/ui/core/mvc/Controller",
   "sap/ui/model/json/JSONModel",
   "sap/m/Button",
   "sap/m/Input",
   "sap/m/CheckBox",
   "sap/m/Text",
   "sap/m/ColorPalettePopover",
   "sap/ui/layout/HorizontalLayout"
], function (Controller, JSONModel, Button, mInput, mCheckBox, mText, ColorPalettePopover, HorizontalLayout) {
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

   var GedController = Controller.extend("rootui5.eve7.controller.Ged", {

      onInit : function() {
         this.oModel = new JSONModel({ title: "GED title", "widgetlist" : [] });
         this.getView().setModel(this.oModel, "ged");

         this.ged_visible = false;
         this.ged_id = -1;
      },

      onExit : function() {
      },

      setManager: function(mgr) {
         this.mgr = mgr;
      },

      isGedVisible : function() {
         return this.ged_visible;
      },

      closeGedEditor: function() {
         if (this.ged_visible) {
            var prnt = this.getView().getParent();
            if (prnt) prnt.removeContentArea(this.getView());
            this.ged_visible = false;
         }

         this.ged_id = -1;
         this.editorElement = null;
      },

      showGedEditor: function(sumSplitter, elementId) {

         if (this.ged_visible && (elementId == this.ged_id))
            return this.closeGedEditor();

         var editorElement = this.mgr ? this.mgr.GetElement(elementId) : null;
         if (!editorElement)
            return this.closeGedEditor();

         if (!this.ged_visible)
            sumSplitter.addContentArea(this.getView());

         this.ged_id = elementId;
         this.ged_visible = true;

         this.editorElement = editorElement;

         // removing ROOT::Experimental:: from class name
         var title = this.editorElement.fName + " (" +  this.editorElement._typename.substring(20) + " )" ;
         this.oModel.setProperty("/title", title);

         var gedFrame =  this.getView().byId("GED");

         gedFrame.unbindElement();
         gedFrame.destroyContent();

         this.makeDataForGED(this.editorElement);

         // console.log("going to bind >>> ", this.getView().getModel("ged"));
         gedFrame.bindAggregation("content", "ged>/widgetlist",  this.gedFactory.bind(this) );
      },

      makeDataForGED: function (element) {

         var cgd = GedController.canEditClass(element._typename);
         if (!cgd)
            return this.oModel.setProperty("/widgetlist", []);

         var arrw = [], modelw = [], off = 0, subEds = [];
         this.maxLabelLength = 0;

         // sub editors
         if (cgd[0].sub) {
            off = 1;
            var sarr = cgd[0].sub;
            for (var i = 0; i< sarr.length; ++i) {
               var x = GedController.canEditClass(sarr[i]);
               if (x)
                  for (var j=0; j < x.length; j++)
                     arrw.push(x[j]);
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

         this.oModel.setProperty("/widgetlist", modelw);
      },

      /** Method used to create custom items for GED */
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
               change: controller.sendMethodInvocationRequest.bind(controller, "Number")
            });
            widget.setType(sap.m.InputType.Number);
            break;

         case "String":
            widget = new mInput(sId, {
               value: { path: "ged>value" },
               change: controller.sendMethodInvocationRequest.bind(controller, "String")

            });
            widget.setType(sap.m.InputType.String);
            widget.setWidth("250px"); // AMT this should be handled differently
            break;
         case "Bool":
            widget = new mCheckBox(sId, {
               selected: { path: "ged>value" },
               select: controller.sendMethodInvocationRequest.bind(controller, "Bool")
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
               press: controller.sendMethodInvocationRequest.bind(controller, "Action")
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

      sendMethodInvocationRequest: function(kind, event) {
         if (!this.editorElement || !this.mgr)
            return;

         var value = "";
         switch (kind) {
            case "Bool": value = event.getSource().getSelected(); break;
            case "Action": value = ""; break;
            default: value =  event.getParameter("value");
         }

         var myData = event.getSource().data("myData");

         if (myData.quote !== undefined)
              value = "\"" + value + " \"";

         this.mgr.SendMIR(myData.srv + "( " + value + " )", this.editorElement.fElementId, this.editorElement._typename );
      },

      handleColorSelect: function(event) {
         var val = event.getParameters().value;
         var myData = event.getSource().data("myData");

        var rgb, regex = /rgb\((\d+)\,\s?(\d+)\,\s?(\d+)\)/,
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
        if (this.mgr)
           this.mgr.SendMIR(mir, this.editorElement.fElementId, this.editorElement._typename);
     },

      updateGED: function(elementId) {
         if (this.ged_visible && this.editorElement && (this.editorElement.fElementId == elementId)) {
            var gedFrame =  this.getView().byId("GED");
            gedFrame.unbindElement();
            gedFrame.destroyContent();
            this.makeDataForGED(this.editorElement);
            gedFrame.bindAggregation("content", "ged>/widgetlist", this.gedFactory.bind(this));
         }
      }

   });

   function make_col_obj(stem) {
      return { name: stem, member: "f" + stem, srv: "Set" + stem + "RGB", _type: "Color" };
   }

   function make_main_col_obj(label, use_main_setter) {
      return { name: label, member: "fMainColor", srv: "Set" + (use_main_setter ? "MainColor" : label) + "RGB", _type: "Color" };
   };

   /** Used in creating items and configuring GED */
   GedController.oGuiClassDef = {
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
         { sub: ["REveElement"] },
         { name : "FilterExpr",  _type : "String",   quote : 1 }
      ],
      "REveDataItem" : [
         make_main_col_obj("ItemColor"),
         { name : "RnrSelf",   member : "fRnrSelf",  _type : "Bool" },
         { name : "Filtered",   _type : "Bool" }
      ],
      "REveTrack" : [
         { name : "RnrSelf",   _type : "Bool" },
         make_main_col_obj("LineColor", true),
         { name : "LineWidth", _type : "Number" },
         { name : "Destroy",  member : "fElementId",  srv : "Destroy", _type : "Action" }
      ],
      "REveStraightLineSet" : [{ sub: ["REveElement" ] }]
   };

   GedController.canEditClass = function(typename) {
      // suppress ROOT::Exeperimental:: prefix
      var t = typename || "";
      if (t.indexOf("ROOT::Experimental::")==0) t = t.substring(20);
      return this.oGuiClassDef[t];
   };

   /** Return method to toggle rendering self */
   GedController.GetRnrSelfMethod = function(typename) {
      var desc = this.canEditClass(typename);
      if (desc)
         for (var k=0;k<desc.length;++k)
            if ((desc[k].member == "fRnrSelf") && desc[k].name)
               return "Set" + desc[k].name;

      return "SetRnrSelf";
   }


   return GedController;

});
