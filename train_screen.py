import pickle, csv, os, statistics as st

SP = os.path.dirname(os.path.abspath(__file__))
data, _ = pickle.load(open(f"{SP}/data.pkl", "rb"))
a15 = data["a15_baseline"]          # July public_2601 run, 1 rollout/scene

rows = list(csv.DictReader(open(f"{SP}/curated.csv")))
train = {r["scene_id"] for r in rows if r["test_suite_id"] == "nurec_curated_train"}

have = sorted(train & set(a15))
print(f"curated_train: {len(train)} scenes; A1.5 July results for {len(have)} "
      f"({100*len(have)/len(train):.0f}%); {len(train)-len(have)} unseen (26.04 only)")

def bad(r):   return r["collision_any"] > 0 or r["offroad"] > 0
def kind(r):
    k = ("C" if r["collision_any"] > 0 else "") + ("O" if r["offroad"] > 0 else "")
    return k or "-"

inc = [s for s in have if bad(a15[s])]
clean = [s for s in have if not bad(a15[s])]
wl = [s for s in clean if a15[s]["wrong_lane"] > 0]
print(f"  incident (collision or offroad): {len(inc)} ({100*len(inc)/len(have):.1f}%)")
print(f"  clean: {len(clean)}; of those wrong_lane: {len(wl)}")

# rank the incident scenes worst-first: low progress, large deviation
inc.sort(key=lambda s: (a15[s]["progress"], -a15[s]["dist_to_gt_location"]))
print("\nWorst 20 train scenes (incident + lowest progress):")
for s in inc[:20]:
    r = a15[s]
    print(f"  {s}  progress={r['progress']:.2f} dist_to_gt_loc={r['dist_to_gt_location']:6.1f}m "
          f"[{kind(r)}] wrong_lane={r['wrong_lane']:.0f}")

out = "/mnt/efs/users/rod/repos/alpasim/train_hard_candidates.csv"
with open(out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["scene_id", "bucket", "progress", "dist_to_gt_location",
                "collision_any", "offroad", "wrong_lane", "plan_deviation"])
    for s in inc:
        r = a15[s]; w.writerow([s, "incident", f"{r['progress']:.4f}",
            f"{r['dist_to_gt_location']:.2f}", int(r["collision_any"]),
            int(r["offroad"]), int(r["wrong_lane"]), f"{r['plan_deviation']:.4f}"])
    for s in sorted(wl, key=lambda s: -a15[s]["dist_to_gt_location"]):
        r = a15[s]; w.writerow([s, "wrong_lane_only", f"{r['progress']:.4f}",
            f"{r['dist_to_gt_location']:.2f}", 0, 0, 1, f"{r['plan_deviation']:.4f}"])
print(f"\nwrote {out}  ({len(inc)} incident + {len(wl)} wrong_lane_only rows)")
