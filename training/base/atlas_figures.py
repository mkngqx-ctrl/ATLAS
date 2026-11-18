"""Re-export benchmark figures with ATLAS display names; preserve saved IDs/data."""
from pathlib import Path
import argparse, ast, hashlib, json, re
import numpy as np
import pandas as pd

METHOD_LABELS={'iid':'IID','stratified':'Stratified','sobol':'Sobol',
    'antithetic':'Antithetic','importance':'Importance',
    'weighted_branching':'WeightedBranching','conditional_mc':'ConditionalMC'}
METHODS=tuple(METHOD_LABELS)

def display_name(model,method=None):
    model=str(model)
    if model.startswith('Glazier-'):
        base='ATLAS-'+model[len('Glazier-'):]
        return base+('-'+METHOD_LABELS.get(str(method),str(method)) if method is not None else '')
    return model+(f' / {method}' if method is not None else '')

def rename_text(text):
    return re.sub(r'Glazier-(Transformer|GRU|RNN)(?:\s*/\s*([a-z_]+))?',
                  lambda m:display_name('Glazier-'+m.group(1),m.group(2)),text)

def export_figures(kind,evaluation,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    evaluation=Path(evaluation).resolve();destination=Path(destination).resolve()
    if destination==evaluation or evaluation in destination.parents:
        raise ValueError('Place exports outside the archived evaluation directory')
    destination.mkdir(parents=True,exist_ok=True)
    bundle=Path(__file__).parent/('Glazier_Weibull_HardRule24' if kind=='hard24' else 'Glazier_CompetingRisks_3Covariates')
    workflow=bundle/('hard_rule_workflow.py' if kind=='hard24' else 'cr_workflow.py')
    tree=ast.parse(workflow.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='plot')
    code=ast.unparse(node).replace("dest = out / 'plots'",'dest = destination')
    assert 'dest = destination' in code
    summary=pd.read_csv(evaluation/'summary.csv')
    globals_=dict(Path=Path,pd=pd,np=np,destination=destination,METHODS=METHODS,
        DIRECT=[m for m in summary.model.unique() if not m.startswith('Glazier-') and m!='Oracle sampler'],
        HISTORY=('DynamicDeepHit','SurvTransformer','SurvTransformer-Discrete'),
        TARGETS=('Survival','CIF_Death_A','CIF_Death_B'),
        STATES=['Healthy','Sick_A','Sick_B','Sick_AB','Sicker_A','Sicker_B','Sicker_AB','Death_A','Death_B','Censored'])
    exec(compile(code,str(workflow),'exec'),globals_)
    actual_save=Figure.savefig;actual_show=plt.show;records=[]
    def save(fig,filename,*args,**kwargs):
        fig.canvas.draw()
        for axis in fig.axes:
            for direction in ('x','y'):
                labels=getattr(axis,f'get_{direction}ticklabels')()
                if any('Glazier-' in t.get_text() for t in labels):
                    ticks=getattr(axis,f'get_{direction}ticks')()
                    getattr(axis,f'set_{direction}ticks')(ticks,[rename_text(t.get_text()) for t in labels])
        texts=fig.findobj(match=Text)
        for item in texts: item.set_text(rename_text(item.get_text()))
        if any('Glazier' in item.get_text() for item in texts):
            raise AssertionError('Unconverted figure label')
        filename=destination/Path(filename).name
        kwargs=dict(kwargs);kwargs['dpi']=220;kwargs['bbox_inches']='tight'
        actual_save(fig,filename,*args,**kwargs)
        actual_save(fig,filename.with_suffix('.svg'),bbox_inches='tight')
        records.append(dict(png=filename.name,svg=filename.with_suffix('.svg').name,
                            visible_labels=sorted(set(t.get_text() for t in texts if t.get_text()))))
    try:
        Figure.savefig=save;plt.show=lambda:None
        globals_['plot'](evaluation)
    finally:
        Figure.savefig=actual_save;plt.show=actual_show;plt.close('all')
    for filename in ('summary.csv','replicate_metrics.csv'):
        table=pd.read_csv(evaluation/filename)
        table['display_name']=[display_name(m,s) for m,s in zip(table.model,table.method)]
        table.to_csv(destination/('ATLAS_'+filename),index=False)
    manifest=dict(experiment=kind,source_evaluation=str(evaluation),
        source_summary_sha256=hashlib.sha256((evaluation/'summary.csv').read_bytes()).hexdigest(),
        naming='ATLAS-architecture-sampler; original model and method keys retained',figures=records)
    (destination/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'Exported {len(records)} figures as PNG and SVG to {destination}')
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',choices=['hard24','cr3'],required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    a=parser.parse_args();export_figures(a.experiment,a.evaluation,a.destination)
