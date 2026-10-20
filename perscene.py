import re, ast, csv, sys, os

RUNS = {'a15_baseline':'fulleval.log','a15_nocot':'fulleval_nocot.log','r1':'r1eval.log'}
sess = re.compile(r"Session COMPLETED: uuid=([0-9a-f-]+) scene=([A-Za-z0-9-]+) simulated ([\d.]+) sim seconds in ([\d.]+) wall")
agg  = re.compile(r"Aggregated metrics for ([0-9a-f-]+): (\{.*\})")
fail = re.compile(r"Rollout FAILED: job=\w+ scene=([A-Za-z0-9-]+) uuid=([0-9a-f-]+) error=(.*)")

data = {}   # run -> scene -> dict
fails = {}
for run, fn in RUNS.items():
    u2s, u2w, m = {}, {}, {}
    fails[run] = []
    for line in open(fn, errors='replace'):
        if 'Session COMPLETED' in line:
            g = sess.search(line)
            if g: u2s[g.group(1)] = g.group(2); u2w[g.group(1)] = (float(g.group(3)), float(g.group(4)))
        elif 'Aggregated metrics for' in line:
            g = agg.search(line)
            if g: m[g.group(1)] = {k: float(v) for k, v in ast.literal_eval(g.group(2)).items()}
        elif 'Rollout FAILED' in line:
            g = fail.search(line)
            if g: fails[run].append((g.group(1), g.group(3)[:90]))
    d = {}
    for uid, met in m.items():
        if uid in u2s:
            met = dict(met); met['sim_s'], met['wall_s'] = u2w[uid]
            d[u2s[uid]] = met
    data[run] = d
    print(f"{run}: {len(d)} scenes joined, {len(fails[run])} failed rollouts")

allscenes = sorted(set().union(*[set(d) for d in data.values()]))
print(f"union of scenes: {len(allscenes)}; in all three: {len(set(data['a15_baseline']) & set(data['a15_nocot']) & set(data['r1']))}")

SP = os.path.dirname(os.path.abspath(__file__))
cols = ['collision_front','collision_rear','collision_lateral','collision_any','offroad','wrong_lane',
        'progress','progress_rel','dist_to_gt_trajectory','dist_to_gt_location','dist_traveled_m',
        'gt_dist_traveled_m','plan_deviation','min_distance_to_obstacle_m','sim_s','wall_s']
out = os.path.join(SP,'per_scene_metrics.csv')
with open(out,'w',newline='') as fh:
    w = csv.writer(fh); w.writerow(['scene','run']+cols)
    for s in allscenes:
        for run in RUNS:
            if s in data[run]:
                r = data[run][s]; w.writerow([s,run]+[r.get(c,'') for c in cols])
print("wrote", out)
import pickle; pickle.dump((data,fails), open(os.path.join(SP,'data.pkl'),'wb'))
