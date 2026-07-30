"""
Post-Mortem / ICA-DVA Evidence Map for the 75 Attribution Benchmark Scenarios
(Reviewer R3.3: separate independently-validated subset from rule-labeled set)

Each scenario's condition-based label is classified by the strength of
independent experimental evidence in the published literature:

  strong    - post-mortem or operando study at matching conditions confirms
              the labeled dominant mechanism
  moderate  - ICA/DVA or diagnostic-mode literature supports the label
  weak      - mechanistically plausible but no direct experimental match at
              these conditions (label rests on the condition-based rule)
  contested - published mode analysis at matching conditions suggests a
              different or mixed dominant mechanism

Model accuracy is then reported separately per tier, so accuracy on the
'strong' subset is an independent (non-circular) validation number.

NOTE FOR AUTHORS: citation page numbers should be double-checked against the
final reference list before inclusion in the manuscript.
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.models.pinn_causal_attribution import PINNCausalAttributionModel
from test_unified_validation import (
    get_nasa_scenarios, get_panasonic_scenarios, get_nature_scenarios,
    get_randomized_scenarios, get_hust_scenarios, make_context, BASE_FEATURES,
)

MECHANISM_MAP = {
    "SEI Layer Growth": 0,
    "Lithium Plating": 1,
    "Active Material Loss": 2,
    "Electrolyte Decomposition": 3,
    "Collector Corrosion": 4,
}
IDX_TO_NAME = {v: k for k, v in MECHANISM_MAP.items()}

REFERENCES = {
    "waldmann2014": "Waldmann et al., 'Temperature dependent ageing mechanisms in Lithium-ion batteries - A Post-Mortem study', J. Power Sources 262 (2014) 129-135",
    "petzl2015": "Petzl, Kasper, Danzer, 'Lithium plating in a commercial lithium-ion battery - A low-temperature aging study', J. Power Sources 275 (2015) 799-807",
    "ansean2017": "Ansean et al., 'Operando lithium plating quantification and early detection of a commercial LiFePO4 cell cycled under dynamic driving schedule', J. Power Sources 356 (2017) 36-46",
    "keil2016": "Keil et al., 'Calendar Aging of Lithium-Ion Batteries I. Impact of the Graphite Anode on Capacity Fade', J. Electrochem. Soc. 163 (2016) A1872-A1880",
    "ecker2014": "Ecker et al., 'Calendar and cycle life study of Li(NiMnCo)O2-based 18650 lithium-ion batteries', J. Power Sources 248 (2014) 839-851",
    "bodenes2013": "Bodenes et al., 'Lithium secondary batteries working at very high temperature: Capacity fade and understanding of aging mechanisms', J. Power Sources 236 (2013) 265-275",
    "klett2014": "Klett et al., 'Non-uniform aging of cycled commercial LiFePO4//graphite cylindrical cells revealed by post-mortem analysis', J. Power Sources 257 (2014) 126-137",
    "birkl2017": "Birkl et al., 'Degradation diagnostics for lithium ion cells', J. Power Sources 341 (2017) 373-386",
    "severson2019": "Severson et al., 'Data-driven prediction of battery cycle life before capacity degradation', Nature Energy 4 (2019) 383-391",
    "dubarry2012": "Dubarry, Truchot, Liaw, 'Synthesize battery degradation modes via a diagnostic and prognostic model', J. Power Sources 219 (2012) 204-216",
    "han2014": "Han et al., 'A comparative study of commercial lithium ion battery cycle life in electrical vehicle: Aging mechanism identification', J. Power Sources 251 (2014) 38-54",
    "braithwaite1999": "Braithwaite et al., 'Corrosion of lithium-ion battery current collectors', J. Electrochem. Soc. 146 (1999) 448-456",
    "maleki2006": "Maleki, Howard, 'Effects of overdischarge on performance and thermal stability of a Li-ion cell', J. Power Sources 160 (2006) 1395-1402",
}

# Tier assignments by scenario name (must cover all 75 exactly once)
TIERS = {
    # --- STRONG: cold-temperature cycling -> lithium plating.
    # Post-mortem: Waldmann 2014 (plating dominant below ~25C, worsens with
    # decreasing T), Petzl 2015 (low-T plating, post-mortem confirmed);
    # operando: Ansean 2017.
    "NASA 4°C 1.5C charge":        ("strong", ["waldmann2014", "petzl2015"]),
    "NASA 4°C 0.5C charge":        ("strong", ["waldmann2014", "petzl2015"]),
    "NASA 4°C high discharge":     ("strong", ["waldmann2014", "petzl2015"]),
    "NASA 8°C threshold":          ("strong", ["waldmann2014", "petzl2015"]),
    "Panasonic -20°C US06 regen":  ("strong", ["waldmann2014", "petzl2015", "ansean2017"]),
    "Panasonic -20°C UDDS gentle": ("strong", ["waldmann2014", "petzl2015", "ansean2017"]),
    "Panasonic -20°C no regen":    ("strong", ["waldmann2014", "petzl2015", "ansean2017"]),
    "Panasonic -10°C UDDS":        ("strong", ["waldmann2014", "petzl2015", "ansean2017"]),
    "Panasonic -10°C HWFET":       ("strong", ["waldmann2014", "petzl2015", "ansean2017"]),
    "Panasonic -10°C aggressive":  ("strong", ["waldmann2014", "petzl2015", "ansean2017"]),
    "Panasonic 0°C HWFET":         ("strong", ["waldmann2014", "petzl2015", "ansean2017"]),
    "Panasonic 5°C LA92":          ("strong", ["waldmann2014", "petzl2015", "ansean2017"]),
    "Panasonic 10°C threshold":    ("strong", ["waldmann2014", "petzl2015"]),

    # --- STRONG: calendar aging / storage -> SEI growth.
    # Keil 2016 (DVA, SOC dependence), Ecker 2014 (NMC calendar/cycle life),
    # Bodenes 2013 (very high T, post-mortem).
    "NASA 4°C storage":     ("strong", ["keil2016", "ecker2014"]),
    "NASA 24°C storage 80%": ("strong", ["keil2016", "ecker2014"]),
    "NASA 43°C storage":    ("strong", ["keil2016", "ecker2014", "bodenes2013"]),
    "Panasonic 25°C parked": ("strong", ["keil2016", "ecker2014"]),
    "Nature 30°C rest":     ("strong", ["keil2016", "ecker2014"]),
    "40°C storage 90%":     ("strong", ["keil2016", "ecker2014", "bodenes2013"]),
    "40°C storage 50%":     ("strong", ["keil2016", "ecker2014", "bodenes2013"]),
    "25°C storage 80%":     ("strong", ["keil2016", "ecker2014"]),
    "50°C storage":         ("strong", ["keil2016", "bodenes2013"]),

    # --- MODERATE: hot cycling -> SEI-dominated aging.
    # Waldmann 2014 attributes high-T cycling fade to SEI growth plus cathode
    # degradation (mixed but SEI-led).
    "NASA 43°C 1C cycling":  ("moderate", ["waldmann2014", "ecker2014"]),
    "NASA 43°C 0.3C gentle": ("moderate", ["waldmann2014", "ecker2014"]),
    "NASA 50°C extreme":     ("moderate", ["waldmann2014", "bodenes2013"]),
    "40°C 0.5C cycling":     ("moderate", ["waldmann2014", "ecker2014"]),
    "40°C 1C cycling":       ("moderate", ["waldmann2014", "ecker2014"]),
    "45°C 0.5C hot":         ("moderate", ["waldmann2014", "bodenes2013"]),

    # --- MODERATE: gentle room-temperature cycling -> SEI/LLI-dominated.
    # Han 2014, Dubarry 2012 (ICA mode analysis), Ecker 2014.
    "NASA 24°C 0.5C gentle": ("moderate", ["han2014", "dubarry2012"]),
    "Panasonic 25°C UDDS":   ("moderate", ["han2014", "dubarry2012"]),
    "Panasonic 25°C HWFET":  ("moderate", ["han2014", "dubarry2012"]),
    "Nature 0.5C/0.5C":      ("moderate", ["severson2019", "dubarry2012"]),
    "Nature 1C/1C":          ("moderate", ["severson2019", "dubarry2012"]),
    "25°C 0.5C gentle":      ("moderate", ["han2014", "dubarry2012"]),
    "HUST 0.3C very gentle": ("moderate", ["klett2014", "dubarry2012"]),
    "HUST 0.5C gentle":      ("moderate", ["klett2014", "dubarry2012"]),
    "HUST 0.7C moderate":    ("moderate", ["klett2014", "dubarry2012"]),

    # --- MODERATE: high-rate cycling -> active material loss.
    # Klett 2014 (LFP post-mortem, LAM at elevated rates), Birkl 2017,
    # Han 2014 (ICA LAM identification).
    "NASA 24°C 2C aggressive": ("moderate", ["klett2014", "birkl2017", "han2014"]),
    "Panasonic 25°C US06":     ("moderate", ["klett2014", "birkl2017"]),
    "Panasonic 25°C LA92":     ("moderate", ["klett2014", "birkl2017"]),
    "25°C 2C aggressive":      ("moderate", ["klett2014", "birkl2017"]),
    "25°C 3C extreme":         ("moderate", ["klett2014", "birkl2017"]),
    "HUST 2C high":            ("moderate", ["klett2014", "birkl2017"]),
    "HUST 2.5C very high":     ("moderate", ["klett2014", "birkl2017"]),
    "HUST 3C extreme":         ("moderate", ["klett2014", "birkl2017"]),
    "HUST 2C/2C symmetric":    ("moderate", ["klett2014", "birkl2017"]),
    "HUST 2C charge":          ("moderate", ["klett2014", "birkl2017"]),
    "HUST 3C charge":          ("moderate", ["klett2014", "birkl2017"]),
    "Nature 30°C pulsed":      ("moderate", ["klett2014", "birkl2017"]),

    # --- WEAK: labels resting on the condition-based rule only.
    # Room-temperature ~1-1.5C AM-vs-SEI splits are rule-arbitrary: nearly
    # identical conditions receive different labels (e.g. 'HUST 1C standard'
    # -> AM Loss vs 'HUST 30°C 1C warm' -> SEI).
    "NASA 24°C 1C cycling":   ("weak", ["han2014"]),
    "NASA 43°C high discharge": ("weak", ["waldmann2014"]),
    "Panasonic 12°C above":   ("weak", ["waldmann2014"]),
    "HUST 1C standard":       ("weak", ["klett2014"]),
    "HUST 0.5C/1C asymm":     ("weak", ["klett2014"]),
    "HUST 1C/0.5C asymm":     ("weak", ["klett2014"]),
    "HUST 1.5C elevated":     ("weak", ["klett2014"]),
    "HUST 30°C 1C warm":      ("weak", ["klett2014"]),
    "HUST 20°C 1C cool":      ("weak", ["klett2014"]),
    "20°C 1C room":           ("weak", ["han2014"]),
    "30°C 1C baseline":       ("weak", ["han2014"]),
    "35°C 1.5C mixed":        ("weak", ["han2014"]),
    "40°C 2C discharge":      ("weak", ["klett2014"]),

    # --- WEAK: low-SOC storage -> collector corrosion.
    # Cu dissolution established for over-discharge (Braithwaite 1999,
    # Maleki 2006) but no direct post-mortem for storage at 10-20% SOC.
    "NASA 24°C storage 20%": ("weak", ["braithwaite1999", "maleki2006"]),
    "25°C storage 10%":      ("weak", ["braithwaite1999", "maleki2006"]),

    # --- CONTESTED: fast charging of MATR LFP cells labeled AM Loss.
    # Severson 2019 / follow-up mode analyses attribute fade in these cells
    # primarily to LLI (SEI growth + plating at the anode), not LAM.
    "Nature 0.5C/4C":    ("contested", ["severson2019", "dubarry2012"]),
    "Nature 1C/4C":      ("contested", ["severson2019", "dubarry2012"]),
    "Nature 2C/4C":      ("contested", ["severson2019", "dubarry2012"]),
    "Nature 4C/4C":      ("contested", ["severson2019", "dubarry2012"]),
    "Nature 6C/4C":      ("contested", ["severson2019", "dubarry2012"]),
    "Nature 8C/4C":      ("contested", ["severson2019", "dubarry2012"]),
    "Nature 4C/2C":      ("contested", ["severson2019", "dubarry2012"]),
    "Nature 2C/1C":      ("contested", ["severson2019", "dubarry2012"]),
    "Nature 25°C 4C/4C": ("contested", ["severson2019", "dubarry2012"]),
    "Nature 35°C 4C/4C": ("contested", ["severson2019", "dubarry2012"]),
    "Nature 40°C 4C/4C": ("contested", ["severson2019", "dubarry2012"]),
}


def load_scenarios():
    scenarios = []
    for getter, ds in [
        (get_nasa_scenarios, "NASA"),
        (get_panasonic_scenarios, "Panasonic"),
        (get_nature_scenarios, "Nature"),
        (get_randomized_scenarios, "Randomized"),
        (get_hust_scenarios, "HUST"),
    ]:
        for s in getter():
            scenarios.append({**s, 'dataset': ds})
    return scenarios


@torch.no_grad()
def predict(model, scenario):
    context = make_context(
        scenario['temp'], scenario['charge'], scenario['discharge'],
        scenario.get('soc', 0.5), scenario.get('mode', 'cycling'))
    features = torch.FloatTensor(BASE_FEATURES).unsqueeze(0)
    context_t = torch.FloatTensor(context).unsqueeze(0)
    output = model(features, context_t)
    return int(output['logits'].argmax(dim=-1).item())


def main():
    scenarios = load_scenarios()
    assert len(scenarios) == 75, f"expected 75 scenarios, got {len(scenarios)}"

    names = [s['name'] for s in scenarios]
    missing = [n for n in names if n not in TIERS]
    extra = [n for n in TIERS if n not in names]
    assert not missing, f"scenarios without tier assignment: {missing}"
    assert not extra, f"tier assignments without scenario: {extra}"

    model = PINNCausalAttributionModel(feature_dim=9, context_dim=6)
    model.load_state_dict(torch.load(
        "reports/pinn_causal/pinn_causal_retrained.pt",
        map_location='cpu', weights_only=True))
    model.eval()

    rows = []
    tier_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for s in scenarios:
        tier, refs = TIERS[s['name']]
        pred_idx = predict(model, s)
        expected_idx = MECHANISM_MAP[s['expected']]
        ok = pred_idx == expected_idx
        tier_stats[tier]['total'] += 1
        tier_stats[tier]['correct'] += int(ok)
        rows.append({
            'dataset': s['dataset'],
            'name': s['name'],
            'temp_C': s['temp'],
            'charge_C': s['charge'],
            'discharge_C': s['discharge'],
            'soc': s.get('soc'),
            'mode': s.get('mode', 'cycling'),
            'label': s['expected'],
            'label_rationale': s.get('rationale', ''),
            'evidence_tier': tier,
            'evidence_refs': refs,
            'model_prediction': IDX_TO_NAME[pred_idx],
            'correct': ok,
        })

    print("=" * 72)
    print("ATTRIBUTION ACCURACY BY EVIDENCE TIER")
    print("=" * 72)
    for tier in ['strong', 'moderate', 'weak', 'contested']:
        st = tier_stats[tier]
        acc = st['correct'] / st['total'] * 100 if st['total'] else 0
        print(f"  {tier:10}: {st['correct']:2}/{st['total']:2}  ({acc:5.1f}%)")
    total_c = sum(v['correct'] for v in tier_stats.values())
    total_n = sum(v['total'] for v in tier_stats.values())
    print(f"  {'overall':10}: {total_c:2}/{total_n:2}  ({total_c/total_n*100:5.1f}%)")

    indep = tier_stats['strong']
    print(f"\nIndependently validated (post-mortem/operando) subset: "
          f"{indep['correct']}/{indep['total']} = "
          f"{indep['correct']/indep['total']*100:.1f}%")

    out = {
        'description': __doc__.strip(),
        'references': REFERENCES,
        'tier_accuracy': {t: dict(v) for t, v in tier_stats.items()},
        'scenarios': rows,
    }
    out_path = Path('reports/causal_attribution/postmortem_evidence_map.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
