## \file
## \ingroup tutorial_ml
## \notebook -nodraw
## This tutorial demonstrates training a classifier model directly
## from remote ROOT data without intermediate preparation steps,
## using XGBoost and the RDataLoader interface.
##
## \macro_code
## \macro_output
##
## \date July 2026
## \author Silia Taider

import ROOT

variables = ["Muon_pt_1", "Muon_pt_2", "Electron_pt_1", "Electron_pt_2"]


def filter_events(df):
    """Reduce initial dataset to only events which shall be used for training"""
    return df.Filter("nElectron>=2 && nMuon>=2", "At least two electrons and two muons")


def define_variables(df):
    """Define the variables which shall be used for training"""
    return (
        df
        .Define("Muon_pt_1", "Muon_pt[0]")
        .Define("Muon_pt_2", "Muon_pt[1]")
        .Define("Electron_pt_1", "Electron_pt[0]")
        .Define("Electron_pt_2", "Electron_pt[1]")
    )


def prepare_rdf(filename, label_value):
    """Load, filter, define variables, and add label column"""
    filepath = "root://eospublic.cern.ch//eos/root-eos/cms_opendata_2012_nanoaod/" + filename
    df = ROOT.RDataFrame("Events", filepath)
    df = filter_events(df)
    df = define_variables(df)
    df = df.Define("label", f"{label_value}.0")
    return df


def load_data():
    """Load signal and background data"""
    rdf_sig = prepare_rdf("SMHiggsToZZTo4L.root", 1)
    rdf_bkg = prepare_rdf("ZZTo2e2mu.root", 0)

    # Compute class-balancing weights
    num_sig = rdf_sig.Count().GetValue()
    num_bkg = rdf_bkg.Count().GetValue()
    num_all = num_sig + num_bkg

    rdf_sig = rdf_sig.Define("weight", f"{num_all}.0/{num_sig}.0")
    rdf_bkg = rdf_bkg.Define("weight", f"{num_all}.0/{num_bkg}.0")

    loader = ROOT.Experimental.ML.RDataLoader(
        [rdf_sig, rdf_bkg],
        columns=variables + ["label", "weight"],
        target="label",
        weights="weight",
        batch_size=num_all,  # Load all data in one batch
        drop_remainder=False,
        set_seed=42,
    )

    # Split into training and testing sets
    train, test = loader.train_test_split(test_size=0.5)

    # train.as_numpy() and test.as_numpy() return generators of batches.
    # Since batch_size=num_all, each split contains exactly one batch;
    # next(iter(...)) materializes it into in-memory numpy arrays.
    X_train, y_train, w_train = next(iter(train.as_numpy()))
    X_test, y_test, w_test = next(iter(test.as_numpy()))

    # Flatten target and weights from (n,1) to (n,) as expected by XGBoost
    return (X_train, y_train.ravel(), w_train.ravel(), X_test, y_test.ravel(), w_test.ravel())


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    X_train, y_train, w_train, X_test, y_test, w_test = load_data()

    print(f"Training events: {X_train.shape[0]}")
    print(f"Testing events:  {X_test.shape[0]}")

    bdt = XGBClassifier(max_depth=3, n_estimators=500)
    bdt.fit(X_train, y_train, sample_weight=w_train)

    # Evaluate on test set
    y_proba = bdt.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)

    print(f"Training done. ROC AUC: {auc:.4f}")
