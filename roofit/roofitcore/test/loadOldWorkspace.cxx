#include <TFile.h>
#include <RooWorkspace.h>
#include <RooAddPdf.h>
#include <RooDataSet.h>

#include "gtest/gtest.h"

class LoadOldWorkspace : public testing::TestWithParam<std::string> {};

TEST_P(LoadOldWorkspace, DifferentVersions)
{
   TFile file(GetParam().c_str());
   ASSERT_TRUE(file.IsOpen());

   RooWorkspace *w = nullptr;
   file.GetObject("w", w);
   ASSERT_NE(w, nullptr);

   RooAddPdf *model = dynamic_cast<RooAddPdf *>(w->pdf("model"));
   ASSERT_NE(model, nullptr);
   EXPECT_STREQ(model->GetName(), "model");
   EXPECT_STREQ(model->GetTitle(), "g1+g2+a");

   RooDataSet *data = dynamic_cast<RooDataSet *>(w->data("modelData"));
   ASSERT_NE(data, nullptr);

   std::unique_ptr<RooArgSet> observables(model->getObservables(*data));
   std::unique_ptr<RooArgSet> parameters(model->getParameters(*data));

   *observables = *data->get(0);
   EXPECT_NEAR(model->getVal(*observables), 0.393976, 1.E-6);

   *observables = *data->get(1);
   EXPECT_NEAR(model->getVal(*observables), 0.344877, 1.E-6);
}

INSTANTIATE_TEST_SUITE_P(ROOT6, LoadOldWorkspace,
                         testing::Values("rf502_workspace_v6.14.root", "rf502_workspace_v6.04.root"));

INSTANTIATE_TEST_SUITE_P(DISABLED_ROOT5, LoadOldWorkspace, testing::Values("rf502_workspace_v5.34.root"));
