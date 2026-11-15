import unittest
import numpy as np
from run_reproduction import inputs,enumerate_pathways,Pathway,evaluate_path,scenario_registry,pareto_mask

class ReproductionTests(unittest.TestCase):
    def test_enumeration_is_complete_and_monotone(self):
        paths=enumerate_pathways()
        self.assertEqual(len(paths),5456)
        self.assertEqual(len({p.pathway_id for p in paths}),5456)
        for p in paths:
            self.assertTrue(all(a[0]<b[0] and a[1]<b[1] for a,b in zip(p.actions,p.actions[1:])))

    def test_ledger_and_no_action(self):
        for prototype,data in inputs().items():
            for path in [Pathway(()),Pathway(((0,3),)),Pathway(((5,1),(10,2),(15,3)))]:
                summary,ledger=evaluate_path(prototype,data,path,scenario_registry()[0],return_ledger=True)
                self.assertTrue(summary['feasible'])
                self.assertTrue(np.isfinite(summary['discounted_lcc_gbp']))
                self.assertGreater(summary['total_included_wlc_kgco2e'],0)
                self.assertEqual(len(ledger),30)

    def test_pareto_tradeoffs_and_dominance(self):
        np.testing.assert_array_equal(pareto_mask(np.array([1.,2.,3.,4.]),np.array([4.,2.,3.,1.])),[True,True,False,True])

    def test_llm_record_aggregation(self):
        import pandas as pd
        from run_reproduction import ROOT
        records=pd.read_csv(ROOT/'llm/human_review_decisions.csv')
        aggregate=pd.read_csv(ROOT/'llm/human_review_aggregate.csv')
        self.assertEqual(len(records),612)
        self.assertEqual(records.review_id.nunique(),612)
        for row in aggregate.itertuples():
            subset=records[(records.condition==row.condition)&(records.task_type==row.task_type)]
            self.assertEqual(len(subset),row.reviewed_items)
            for label in ['accept','accept_with_edit','reject']:
                self.assertEqual(int((subset.human_decision==label).sum()),getattr(row,label+'_count'))

if __name__=='__main__':unittest.main()

