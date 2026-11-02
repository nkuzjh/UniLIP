#!/usr/bin/env python3
"""Summarize complete gradient diagnostic JSON reports using only the stdlib."""
import argparse
import hashlib
import json
import math
from pathlib import Path

TASKS = ("gen", "loc", "perception", "aux_loc")
PAIRS = ("gen-loc", "perception-gen", "perception-loc", "aux_loc-gen", "aux_loc-loc", "perception-aux_loc")
SCALARS = ([f"components.{t}.{n}" for t in TASKS for n in ("raw_norm", "weighted_norm")]
           + ["pairwise_weighted_cosine." + a + "." + b for a,b in
              (("gen","loc"),("perception","gen"),("perception","loc"),("aux_loc","gen"),("aux_loc","loc"),("perception","aux_loc"))]
           + ["combined." + n for n in ("aux_over_baseline","aux_signed_contribution_to_total","actual_aux_increment_vs_component_cosine")]
           + [f"combined.primary_directions.{t}.{n}" for t in ("gen","loc") for n in
              ("raw_descent_first_order_total","raw_descent_first_order_aux_increment")]
           + ["linearity.relative_difference", "components.linearity_residual.weighted_norm",
              "combined.actual_aux_increment_norm", "combined.actual_aux_increment_vs_component_cosine",
              "combined.actual_aux_increment_vs_component_difference_norm"])
DERIVED_MODE_SCALARS = {"linearity.relative_difference", "components.linearity_residual.weighted_norm",
                        "combined.actual_aux_increment_norm", "combined.actual_aux_increment_vs_component_cosine",
                        "combined.actual_aux_increment_vs_component_difference_norm"}

def get(obj, path):
    for part in path.split('.'):
        if not isinstance(obj, dict): return None
        obj = obj.get(part)
    return obj

def has_path(obj, path):
    for part in path.split('.'):
        if not isinstance(obj, dict) or part not in obj: return False
        obj = obj[part]
    return True

def fmt(v):
    if v is None: return "—"
    if isinstance(v, float): return f"{v:.6g}"
    return str(v)

def stats(vals):
    vals = [v for v in vals if isinstance(v,(int,float)) and math.isfinite(v)]
    if not vals: return None
    return {"median": sorted(vals)[len(vals)//2] if len(vals)%2 else (sorted(vals)[len(vals)//2-1]+sorted(vals)[len(vals)//2])/2,
            "min": min(vals), "max": max(vals), "count": len(vals)}

def measurement_mode(result, report):
    metadata = report.get("metadata") if isinstance(report.get("metadata"), dict) else {}
    return result.get("measurement_mode", report.get("measurement_mode", metadata.get("measurement_mode", "full")))

def measurement_scope(result, mode):
    scope = result.get("measurement_scope")
    if isinstance(scope, dict): return scope
    if mode == "component_sum":
        return {"components":"direct_vjp_every_microbatch", "main_total":"component_sum_of_mean_weighted_vjps",
                "other":"zero_by_verified_weights", "direct_reconstruction":"first_microbatch_only"}
    return {"components":"direct_vjp_every_microbatch", "main_total":"direct_vjp_every_microbatch",
            "other":"direct_vjp_every_microbatch", "direct_reconstruction":"not applicable"}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-root', required=True); ap.add_argument('--output-json', required=True); ap.add_argument('--output-md', required=True)
    args=ap.parse_args(); root=Path(args.input_root)
    files=sorted(p for p in root.rglob('*.json') if p.name != 'manifest.json') if root.exists() else []
    rows=[]; warnings=[]; stages={"screen","confirm","temporal"}
    for path in files:
        rel=path.relative_to(root)
        folder_stage=rel.parts[0] if len(rel.parts)>1 else None
        if folder_stage not in stages:
            # Ignore controller/status JSON and reports outside known stage folders.
            continue
        try: d=json.loads(path.read_text())
        except Exception as e: warnings.append(f"{path}: JSON读取失败: {e}"); continue
        if not isinstance(d,dict) or not isinstance(d.get('results'),dict) or not isinstance(d.get('metadata'),dict):
            warnings.append(f"{path}: 未识别schema，保留为warning"); continue
        internal_stage=d.get('stage')
        valid_stage=(internal_stage == 'screen') if folder_stage == 'screen' else (internal_stage == 'effective')
        if not valid_stage:
            warnings.append(f"{path}: stage mismatch folder={folder_stage} report={internal_stage!r}; retained")
        stage=folder_stage
        for seed,res in d['results'].items():
            m=d['metadata']
            mode=measurement_mode(res,d)
            for field in SCALARS:
                if mode == 'component_sum' and field in DERIVED_MODE_SCALARS:
                    continue
                if not has_path(res,field):
                    warnings.append(f"{rel}: missing field {field} (seed={seed}); retained as pending")
            for check_name in ('linearity','repeated_total_vjp','total_backward_check','aux_freeze'):
                check=res.get(check_name)
                if isinstance(check,dict) and check.get('passed') is False:
                    warnings.append(f"{path}: failed {check_name} check (seed={seed}); retained")
            first_direct=res.get('first_micro_direct_checks')
            if mode == 'component_sum':
                if not isinstance(first_direct,dict):
                    warnings.append(f"{path}: missing first_micro_direct_checks (seed={seed}); pending")
                else:
                    for direct_name in ('main_vs_components','total_vs_components'):
                        direct_check=first_direct.get(direct_name)
                        if not isinstance(direct_check,dict):
                            warnings.append(f"{path}: missing first_micro_direct_checks.{direct_name} (seed={seed}); pending")
                        elif direct_check.get('passed') is False:
                            warnings.append(f"{path}: failed first_micro_direct_checks.{direct_name} (seed={seed})")
            small_meta={k:m[k] for k in ('dtype','precision_reference','deterministic_probe','experiment','checkpoint_step','loss_weights','training_runtime','train_microbatch','train_gradient_accumulation_steps','local_step') if k in m}
            rows.append({"file":str(rel),"stage":stage,"experiment":m.get('experiment','unknown'),
              "checkpoint":m.get('checkpoint',d.get('checkpoint_dir','unknown')), "step":m.get('checkpoint_step'),
              "offset":d.get('source_offset'),"noise_seed":seed,"source_count":len(d.get('source_rows') or []),
              "source_rows":d.get('source_rows'),"coordinate_names":res.get('coordinate',{}).get('names',d.get('coordinate_names')),
              "coordinate_numel":res.get('coordinate',{}).get('numel',d.get('coordinate_numel')),
              "result":res,"metadata":small_meta,"weights_seen":res.get('weights_seen',{}),"microbatch":d.get('microbatch_size',m.get('train_microbatch')),
              "gas":d.get('accumulations',m.get('train_gradient_accumulation_steps')),
              "measurement_mode":mode,"measurement_scope":measurement_scope(res,mode),
              "first_micro_direct_checks":first_direct})
    # Alignment hashes use only map/file_frame/row_id, in listed order.
    align={}
    for r in rows:
        key=(r['stage'],r['offset']); sig=[{k:x.get(k) for k in ('map','file_frame','row_id')} for x in (r['source_rows'] or [])]
        name_sig=r['coordinate_names']; align.setdefault(key,[]).append((hashlib.sha256(json.dumps(sig,sort_keys=True,separators=(',',':')).encode()).hexdigest(),name_sig,r['file']))
    align_checks=[]
    for (stage,off), entries in align.items():
        # Multiple seed rows from same report collapse.
        entries=list({e[2]:e for e in entries}.values())
        if len(entries)>1:
            ok=len({e[0] for e in entries})==1 and len({json.dumps(e[1],sort_keys=True) for e in entries})==1
            align_checks.append({"stage":stage,"offset":off,"experiments":sorted({next((x['experiment'] for x in rows if x['file']==e[2]),'unknown') for e in entries}),"status":"passed" if ok else "FAILED","files":[e[2] for e in entries]})
            if not ok: warnings.append(f"source alignment mismatch: {stage} offset={off}")
    groups={}
    for r in rows: groups.setdefault((r['stage'],r['experiment'],r['checkpoint'],r['measurement_mode']),[]).append(r)
    summary=[]
    for key,rs in sorted(groups.items()):
        metrics={}
        mode=key[3]
        for field in SCALARS:
            metrics[field]=({"status":"not_measured_derived_mode","median":None,"min":None,"max":None,"count":0}
                            if mode == 'component_sum' and field in DERIVED_MODE_SCALARS
                            else stats([get(x['result'],field) for x in rs]))
        summary.append({"stage":key[0],"experiment":key[1],"checkpoint":key[2],"measurement_mode":mode,
          "measurement_scope":rs[0]['measurement_scope'],"rows":len(rs),"metrics":metrics,
          "direction_counts":{f"{t}.{n}":{"positive_local_rise":sum(1 for r in rs if (v:=get(r['result'],f"combined.primary_directions.{t}.{n}")) is not None and v>0),"negative":sum(1 for r in rs if (v:=get(r['result'],f"combined.primary_directions.{t}.{n}")) is not None and v<0)} for t in ('gen','loc') for n in ('raw_descent_first_order_total','raw_descent_first_order_aux_increment')},
          "checks":[{"file":r['file'],"seed":r['noise_seed'],"measurement_mode":r['measurement_mode'],
                     "linearity":r['result'].get('linearity'),"first_micro_direct_checks":r['first_micro_direct_checks'],
                     "repeated_total_vjp":r['result'].get('repeated_total_vjp'),"total_backward_check":r['result'].get('total_backward_check'),"aux_freeze":r['result'].get('aux_freeze')} for r in rs]})
    out={"input_root":str(root),"files_scanned":len(files),"complete_result_rows":len(rows),"pending_stages":sorted(stages-{r['stage'] for r in rows}),"summaries":summary,"rows":rows,"source_alignment":align_checks,"warnings":warnings}
    Path(args.output_json).parent.mkdir(parents=True,exist_ok=True); Path(args.output_json).write_text(json.dumps(out,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    md=["# Gradient diagnostics summary","","Scalar aggregates are median [min, max] across available report windows/seeds within stage, experiment, checkpoint, and measurement mode. Screen and confirm remain separate. Cosines are report-level values summarized across rows, not recomputed over concatenated vectors. Missing required fields are pending; present nulls are undefined/inactive (—), except direct metrics intentionally not measured in component_sum mode.",""]
    md += ["## Configuration","","| Stage | Experiment | Checkpoint step | Measurement mode | Components scope | Main/total scope | Coordinate numel | Microbatch / GAS | Source count | Weights seen (actual alpha) | Precision reference | Deterministic probe |","|---|---|---:|---|---|---|---:|---:|---:|---|---|---|"]
    config_rows={}
    for r in rows: config_rows.setdefault(r['file'],r)
    for r in config_rows.values():
        md.append(f"| {r['stage']} | {r['experiment']} | {fmt(r['step'])} | {r['measurement_mode']} | `{r['measurement_scope'].get('components')}` | `{r['measurement_scope'].get('main_total')}` | {fmt(r['coordinate_numel'])} | {fmt(r['microbatch'])} / {fmt(r['gas'])} | {r['source_count']} | `{json.dumps(r['weights_seen'],sort_keys=True)}` | {fmt(r['metadata'].get('precision_reference'))} | {fmt(r['metadata'].get('deterministic_probe'))} |")
    for s in sorted(stages):
        md += ["",f"## {s}",""]
        ss=[x for x in summary if x['stage']==s]
        if not ss: md.append("Pending: no complete JSON reports found."); continue
        for g in ss:
            md += [f"### {g['experiment']} — {g['checkpoint']} ({g['measurement_mode']})","",f"Rows (windows/seeds): {g['rows']}","", "| Metric | Median [min, max] | n |","|---|---:|---:|"]
            for k,v in g['metrics'].items():
                if isinstance(v,dict) and v.get('status') == 'not_measured_derived_mode': md.append(f"| `{k}` | not measured (derived mode) | 0 |")
                elif v: md.append(f"| `{k}` | {fmt(v['median'])} [{fmt(v['min'])}, {fmt(v['max'])}] | {v['count']} |")
                elif any(has_path(row['result'],k) for row in rows
                         if row['stage']==g['stage'] and row['experiment']==g['experiment']
                         and row['checkpoint']==g['checkpoint'] and row['measurement_mode']==g['measurement_mode']):
                    md.append(f"| `{k}` | — (undefined/inactive) | 0 |")
                else:
                    md.append(f"| `{k}` | pending | 0 |")
            if g['measurement_mode'] == 'component_sum':
                md.append("")
                md.append("Full-window direct-linearity and actual auxiliary-increment metrics: not measured (derived mode).")
            md += ["", "Direction counts (positive raw descent score = local rise):"]
            md += [f"- {k}: >0 {v['positive_local_rise']}, <0 {v['negative']}" for k,v in g['direction_counts'].items()]
            md += ["", "Check summary:"]
            for c in g['checks']:
                linearity_label = "not measured (derived mode)" if c['measurement_mode'] == 'component_sum' else fmt((c['linearity'] or {}).get('passed'))
                direct=c.get('first_micro_direct_checks') or {}
                direct_label="not required" if c['measurement_mode'] != 'component_sum' else ",".join(
                    f"{name}={fmt((direct.get(name) or {}).get('passed'))}" for name in ('main_vs_components','total_vs_components'))
                md.append(f"- {c['file']} seed={c['seed']}: mode={c['measurement_mode']}, full_window_linearity={linearity_label}, first_micro_direct={direct_label}, repeated_total_vjp={fmt((c['repeated_total_vjp'] or {}).get('passed'))}, total_backward_check={fmt((c['total_backward_check'] or {}).get('passed'))}, aux_freeze={fmt((c['aux_freeze'] or {}).get('status'))}/{fmt((c['aux_freeze'] or {}).get('passed'))}")
    md += ["", "## Raw window / seed rows", "", "| Stage | Experiment | Checkpoint | Offset | Noise seed | Source count | " + " | ".join(SCALARS) + " |", "|---|---|---|---:|---:|---:|"+"---:|"*len(SCALARS)]
    for r in rows:
        vals=[("not measured (derived mode)" if r['measurement_mode']=='component_sum' and f in DERIVED_MODE_SCALARS else "pending" if not has_path(r['result'],f) else "— (undefined/inactive)" if get(r['result'],f) is None else fmt(get(r['result'],f))) for f in SCALARS]
        md.append("| "+" | ".join([r['stage'],r['experiment'],str(r['checkpoint']),fmt(r['offset']),str(r['noise_seed']),str(r['source_count']),*vals])+" |")
    md += ["", "## Source alignment review", ""]
    md += [f"- {x['status']}: {x['stage']} offset={x['offset']} experiment={', '.join(x['experiments'])} ({len(x['files'])} reports)" for x in align_checks] or ["No same-stage/offset multi-report comparisons available."]
    md += ["", "## Warnings", ""] + ([f"- {w}" for w in warnings] or ["None."])
    Path(args.output_md).parent.mkdir(parents=True,exist_ok=True); Path(args.output_md).write_text('\n'.join(md)+'\n')
    print(f"scanned={len(files)} rows={len(rows)} warnings={len(warnings)} json={args.output_json} md={args.output_md}")
if __name__=='__main__': main()
