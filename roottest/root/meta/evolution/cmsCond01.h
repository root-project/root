#include <string>
#include <vector>

enum L1GtObject
{
    Mu,
    NoIsoEG,
    IsoEG,
    CenJet,
    ForJet,
    TauJet,
    ETM,
    ETT,
    HTT,
    HTM,
    JetCounts,
    HfBitCounts,
    HfRingEtSums,
    TechTrig,
    Castor,
    BPTX,
    GtExternal,
    ObjNull
};

enum L1GtConditionType {
    TypeNull,
    Type1s,
    Type2s,
    Type2wsc,
    Type2cor,
    Type3s,
    Type4s,
    TypeETM,
    TypeETT,
    TypeHTT,
    TypeHTM,
    TypeJetCounts,
    TypeCastor,
    TypeHfBitCounts,
    TypeHfRingEtSums,
    TypeBptx,
    TypeExternal
};

enum L1GtConditionCategory {
    CondNull,
    CondMuon,
    CondCalo,
    CondEnergySum,
    CondJetCounts,
    CondCorrelation,
    CondCastor,
    CondHfBitCounts,
    CondHfRingEtSums,
    CondBptx,
    CondExternal
};

struct L1GtCondition {

    /// the name of the condition
    std::string m_condName;

    /// the category of the condition
    L1GtConditionCategory m_condCategory;

    /// the type of the condition (1s, etc)
    L1GtConditionType m_condType;

    /// the trigger object type(s)
    std::vector<L1GtObject> m_objectType;

    /// the operator used for the condition (>=, =): true for >=
    bool m_condGEq;

    /// condition is located on condition chip m_condChipNr
    int m_condChipNr;
};

