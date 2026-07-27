import numpy as np
from eval_core import (make_three_way_split, repeated_group_folds, assert_no_leakage,
                       aggregate_fruit, select_vote_threshold, evaluate_test,
                       results_to_frame, wilson_ci)

rng = np.random.default_rng(0)

# --- build 111 synthetic fruits with day x dose strata ---
days = [4,5,7,10]; doses = [0,50,100,500]
fruits=[]; fid=0
for _ in range(7):                       # ~7 per (day,dose) cell -> ~112 fruits
    for d in days:
        for k in doses:
            label = 0 if k==0 else 1
            fruits.append((f"F{fid}", label, f"d{d}_k{k}"))
            fid+=1
fruit_ids=[f[0] for f in fruits]; labels=[f[1] for f in fruits]; strata=[f[2] for f in fruits]
print("total fruits:", len(fruit_ids), "infested:", sum(labels))

# --- 1. three-way split, no leakage ---
tr,va,te = make_three_way_split(fruit_ids, labels, seed=42)
assert_no_leakage(("train",tr),("val",va),("test",te))
print(f"split  train={len(tr)} val={len(va)} test={len(te)}  (no leakage OK)")

# --- 2. repeated group CV, no leakage per fold ---
nfold=0
for rep,trf,tef in repeated_group_folds(fruit_ids, labels, n_splits=5, n_repeats=3, seed=1):
    assert_no_leakage(("train",trf),("test",tef)); nfold+=1
print(f"repeated CV folds generated: {nfold} (expected 15), all leakage-free")

# --- 3. simulate a model that votes 'infested' too eagerly (the paper's artifact) ---
lab = dict(zip(fruit_ids, labels))
def fake_slice_probs(y):
    # infested fruits: high; control fruits: still leans high (eager voter)
    base = 0.75 if y==1 else 0.55
    return np.clip(rng.normal(base, 0.15, size=72), 0, 1)

def scores_for(ids):
    return [aggregate_fruit(f, lab[f], fake_slice_probs(lab[f])) for f in ids]

val_scores  = scores_for(va)
test_scores = scores_for(te)

t,_ = select_vote_threshold(val_scores, objective="f1")
print(f"vote threshold selected ON VALIDATION only: {t}")

res = evaluate_test(test_scores, t)
print("test confusion:", res["confusion"])
for m in ["AUC","accuracy","precision","recall","F1"]:
    print(f"  {m:9s}: {res[m]}")

# --- 4. Wilson sanity: recall 24/24 must NOT report [1,1] ---
p,lo,hi = wilson_ci(24,24)
print(f"\nWilson CI for 24/24 recall: point={p:.3f} CI=[{lo:.3f},{hi:.3f}]  <-- not a naked 1.00")
assert lo < 1.0, "Wilson lower bound should be < 1"

df = results_to_frame({"DemoNet": res})
print("\n", df.to_string(index=False))
print("\nALL CORE CHECKS PASSED")
