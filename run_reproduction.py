"""Run the public aggregate Chapter 4 calculations without private records or API calls."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'enb'))
from src.data_sources import load_config
from src.model import prototype_inputs,enumerate_pathways,scenario_registry,evaluate_path,Pathway,reconcile_ledger
from src.exact_optimisation import _evaluate_group,_select_preferred,pareto_mask

def inputs():
    c=load_config()
    e={'grid_factors':pd.read_csv(ROOT/'enb/data/electricity_carbon_factor_series.csv').set_index('year'),
       'electricity_price_gbp_kwh':c['energy_prices_gbp_per_kwh']['central']['electricity'],
       'gas_price_gbp_kwh':c['energy_prices_gbp_per_kwh']['central']['gas'],
       'gas_factor_kgco2e_kwh_gross_cv':c['carbon']['natural_gas_kgco2e_per_kwh_gross_cv']}
    return prototype_inputs(pd.read_csv(ROOT/'enb/data/prototype_energy_baselines.csv'),e)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--all-scenarios',action='store_true');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    start=time.time();paths=enumerate_pathways()
    scenarios=scenario_registry() if args.all_scenarios else scenario_registry()[:1]
    if args.smoke:paths=[Pathway(()),Pathway(((0,1),)),Pathway(((0,2),)),Pathway(((0,3),)),Pathway(((5,1),(10,2),(15,3)))]
    rows=[];counts=[]
    for prototype,data in inputs().items():
        for scenario in scenarios:
            frame,rejected=_evaluate_group(prototype,data,scenario,paths)
            frame['is_exact_pareto']=pareto_mask(frame.discounted_lcc_gbp.to_numpy(),frame.total_included_wlc_kgco2e.to_numpy())
            selected=_select_preferred(frame);rows.extend(selected)
            frame.to_csv(out/f"{prototype}_{scenario['scenario']}_pathways.csv",index=False)
            counts.append({'prototype':prototype,'scenario':scenario['scenario'],'candidate_paths':len(paths),'feasible':len(frame),'infeasible':rejected,'pareto':int(frame.is_exact_pareto.sum())})
            # Annual ledger for the balanced preferred sequence.
            chosen=next(x for x in selected if x['preference']=='balanced')
            path=next(p for p in paths if p.pathway_id==chosen['pathway_id'])
            summary,ledger=evaluate_path(prototype,data,path,scenario,return_ledger=True)
            assert max(abs(v) for v in reconcile_ledger(ledger,summary).values()) < 1e-6
            pd.DataFrame(ledger).to_csv(out/f"{prototype}_{scenario['scenario']}_annual_ledger.csv",index=False)
            print(prototype,scenario['scenario'],len(frame),'feasible',flush=True)
    pd.DataFrame(rows).to_csv(out/'preferred_pathways.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'path_counts.csv',index=False)
    records=pd.read_csv(ROOT/'llm/human_review_decisions.csv')
    llm=pd.read_csv(ROOT/'llm/human_review_aggregate.csv')
    assert len(records)==612 and records.review_id.nunique()==612
    for row in llm.itertuples():
        subset=records[(records.condition==row.condition)&(records.task_type==row.task_type)]
        assert len(subset)==row.reviewed_items
        for label in ['accept','accept_with_edit','reject']:
            assert int((subset.human_decision==label).sum())==getattr(row,label+'_count')
    assert llm.reviewed_items.sum()==612
    assert (llm.accept_count+llm.accept_with_edit_count+llm.reject_count).equals(llm.reviewed_items)
    llm.groupby('condition')[['reviewed_items','accept_count','accept_with_edit_count','reject_count']].sum().to_csv(out/'llm_review_totals.csv')
    report={'mode':'smoke' if args.smoke else ('all_scenarios' if args.all_scenarios else 'central'), 'candidates_per_prototype_scenario':len(paths),'scenario_count':len(scenarios),'prototype_count':3,'elapsed_seconds':round(time.time()-start,2),'llm_review_records':612,'live_llm_calls':0}
    (out/'run_summary.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report))

if __name__=='__main__':main()
