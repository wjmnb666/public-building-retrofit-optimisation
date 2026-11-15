"""Compare newly generated public results with the retained publication snapshots."""
from pathlib import Path
import pandas as pd
import numpy as np
R=Path(__file__).resolve().parent
a=pd.read_csv(R/'outputs/preferred_pathways.csv')
b=pd.read_csv(R/'reference/enb/exact_preferred_paths.csv')
keys=['prototype','scenario','preference']
m=a.merge(b,on=keys,suffixes=('_new','_reference'),validate='one_to_one')
assert len(m)==len(a)==len(b)==153,'Run all 17 scenarios before this comparison'
assert (m.pathway_id_new==m.pathway_id_reference).all()
for c in ['discounted_lcc_gbp','total_included_wlc_kgco2e']:
    np.testing.assert_allclose(m[c+'_new'],m[c+'_reference'],rtol=0,atol=1e-6)
for name in ['bcnn_lstm_revision_metrics.csv','dqn_selected_sequences.csv']:
    a=pd.read_csv(R/'cais/tables'/name);b=pd.read_csv(R/'reference/cais'/name)
    assert a.shape==b.shape
    for c in a.select_dtypes('number').columns:
        np.testing.assert_allclose(a[c],b[c],rtol=0,atol=1e-9)
print('153 preferred paths and both CAIS result snapshots agree within the declared tolerances.')
