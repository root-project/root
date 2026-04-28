/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN  04/2026
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RooONNXFunc_h
#define RooFit_RooONNXFunc_h

#include <RooAbsReal.h>
#include <RooListProxy.h>

#include <any>

class RooONNXFunc : public RooAbsReal {
public:
   RooONNXFunc() = default;

   RooONNXFunc(const char *name, const char *title, const std::vector<RooArgList> &inputTensors,
                   const std::string &onnxFile, const std::vector<std::string> &inputNames = {},
                   const std::vector<std::vector<int>> &inputShapes = {});

   RooONNXFunc(const RooONNXFunc &other, const char *newName = nullptr);

   TObject *clone(const char *newName) const override { return new RooONNXFunc(*this, newName); }

   std::size_t nInputTensors() const { return _inputTensors.size(); }
   RooArgList const &inputTensorList(int iTensor) const { return *(_inputTensors[iTensor]); }

   std::string funcName() const { return _funcName; }
   std::string outerWrapperName() const { return "TMVA_SOFIE_" + funcName() + "::roo_outer_wrapper"; }

protected:
   double evaluate() const override;

private:
   /// Build transient runtime backend on first use.
   void initialize();

   /// Gather current RooFit inputs into a contiguous feature buffer.
   void fillInputBuffer() const;

   struct RuntimeCache;

   std::vector<std::unique_ptr<RooListProxy>> _inputTensors; ///< Inputs mapping to flattened input tensors.
   std::vector<std::uint8_t> _onnxBytes;                     ///< Persisted ONNX model bytes.
   std::shared_ptr<RuntimeCache> _runtime;                   ///<! Transient runtime information.
   mutable std::vector<float> _inputBuffer;                  ///<!
   std::string _funcName;                                    ///<!

   ClassDefOverride(RooONNXFunc, 1)
};

namespace RooFit::Detail {

struct AnyWithVoidPtr {
   std::any any;
   void *ptr = nullptr;

   template <class T>
   void emplace()
   {
      any = std::make_any<T>();
      ptr = std::any_cast<T>(&any);
   }

   void emplace(std::string const &typeName);
};

template <class Session_t, class... Inputs>
void doInferWithSessionVoidPtr(void *session, float *out, Inputs const *...inputs)
{
   doInfer(*reinterpret_cast<Session_t *>(session), inputs..., out);
}

} // namespace RooFit::Detail

#endif
