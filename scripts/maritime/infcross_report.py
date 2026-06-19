"""Ask-2 dual report from the per-experiment Infiray cross-domain grids.

Lifted verbatim from the embedded heredoc in the old run_infcross.sh.

  Variant A (maritime-config): the study's maritime val-best config, evaluated on Infiray val+test.
  Variant B (infiray-val-select): best config picked on Infiray val, reported on Infiray test.

Usage (as crossdomain.sh invokes it):
  python infcross_report.py <grids_dir> <tables_dir>
"""
import csv, glob, os, sys
grids, res = sys.argv[1], sys.argv[2]

allrows = []
for f in sorted(glob.glob(os.path.join(grids, "infcross_*_full.csv"))):
    with open(f) as fh:
        allrows += list(csv.DictReader(fh))
if not allrows:
    print("no infcross_*_full.csv found"); sys.exit(0)

# Variant B: best config on Infiray val per (experiment, variant).
bestB = {}
for r in allrows:
    k = (r["experiment"], r["variant"]); va = float(r["val_auroc"])
    if k not in bestB or va > bestB[k][0]:
        bestB[k] = (va, r)

# Variant A: each detector's maritime val-best config (from best_per_experiment.csv),
# looked up in the Infiray grid. 'dataset' column there == our experiment code.
mar = {}
with open(os.path.join(res, "best_per_experiment.csv")) as fh:
    for r in csv.DictReader(fh):
        mar[r["dataset"]] = r
def match(row, m):
    return (row["layer"] == m["layer"] and row["agg"] == m["agg"]
            and abs(float(row["ev"]) - float(m["ev"])) < 1e-9
            and row["score"] == m["score"] and int(row["drop_k"]) == int(m["drop_k"]))
rowsA = []
for (exp, var) in sorted(bestB):
    m = mar.get(exp)
    if not m:
        continue
    hit = [r for r in allrows if r["experiment"] == exp and r["variant"] == var and match(r, m)]
    if hit:
        rowsA.append(hit[0])
    else:
        print(f"  [warn] no Infiray-grid row for maritime config of {exp}/{var} "
              f"({m['layer']} {m['agg']} EV{m['ev']} {m['score']} dk{m['drop_k']})")

def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

rowsB = [bestB[k][1] for k in sorted(bestB)]
write(os.path.join(res, "infcross_valselect.csv"), rowsB)
if rowsA:
    write(os.path.join(res, "infcross_maritimecfg.csv"), rowsA)

print("\n=== Variant A — maritime val-best config → Infiray val & test ===")
for r in rowsA:
    print(f"  {r['experiment']:<10} {r['variant']:<6} {r['layer']:<4} {r['agg']:<6} "
          f"EV{r['ev']:<5} {r['score']:<14} dk{r['drop_k']:<3} "
          f"infiray_val={r['val_auroc']} infiray_test={r['test_auroc']}")
print("\n=== Variant B — best config on Infiray val → Infiray test ===")
for r in rowsB:
    print(f"  {r['experiment']:<10} {r['variant']:<6} {r['layer']:<4} {r['agg']:<6} "
          f"EV{r['ev']:<5} {r['score']:<14} dk{r['drop_k']:<3} "
          f"val={r['val_auroc']} test={r['test_auroc']}")
print(f"\nwrote infcross_maritimecfg.csv ({len(rowsA)}) + infcross_valselect.csv ({len(rowsB)})")
