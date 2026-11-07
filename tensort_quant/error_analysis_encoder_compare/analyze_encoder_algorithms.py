#!/usr/bin/env python3
"""Compare encoder-only MinMax, Percentile and SmoothQuant with Polygraphy."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ANALYSIS=HERE.parent/'error_analysis_Conv_MatMul_compare'
sys.path.insert(0,str(ANALYSIS))
from analyze_quant_error import onnx, run_onnxrt, tensor_metrics, qdq_audit, write_csv, make_plots  # noqa

ROOT=HERE.parent.parent
FP=ROOT/'python_infer/output/rfdetr-xlarge.onnx'
INPUTS=ROOT/'tensort_quant/polygraphy_analysis/.inputs.json'
MODELS={
 'MinMax': ROOT/'tensort_quant/artifacts_xlarge_encoder_minmax/rfdetr-xlarge-encoder-minmax-int8.onnx',
 'Percentile': ROOT/'tensort_quant/artifacts_xlarge_encoder_percentile/rfdetr-xlarge-encoder-percentile-int8.onnx',
 'SmoothQuant': ROOT/'tensort_quant/artifacts_xlarge_smoothquant_256/rfdetr-xlarge-smoothquant-256.onnx',
}

def specs(model):
 out=[]
 for n in model.graph.node:
  if n.op_type!='MatMul' or '/backbone/backbone.0/encoder/encoder/encoder/layer.' not in n.name: continue
  if not any(x in n.name for x in ['/attention/attention/query/MatMul','/attention/attention/key/MatMul','/attention/attention/value/MatMul','/attention/output/dense/MatMul','/mlp/fc1/MatMul','/mlp/fc2/MatMul']): continue
  out.append({'tensor':n.output[0],'node_name':n.name,'op_type':n.op_type})
 return out

def stats(x):
 x=np.asarray(x,dtype=np.float32); a=np.abs(x); f=a[np.isfinite(a)]
 return {'mean_abs':float(np.mean(a)),'std':float(np.std(x)),'min':float(np.min(x)),'max':float(np.max(x)),'p99':float(np.percentile(f,99)),'p99_9':float(np.percentile(f,99.9)),'p99_99':float(np.percentile(f,99.99)),'nonfinite':int(x.size-f.size)}

def main():
 out=HERE; out.mkdir(exist_ok=True)
 fp_model=onnx.load(str(FP),load_external_data=True); ss=specs(fp_model); names=[x['tensor'] for x in ss]; by={x['tensor']:x for x in ss}
 # Polygraphy/ONNX Runtime uses the same serialized input for every graph.
 fp=run_onnxrt(fp_model,names,INPUTS,'fp32_encoder')
 reports={}
 for label,path in MODELS.items():
  if not path.exists(): reports[label]={'status':'missing','model':str(path)}; continue
  m=onnx.load(str(path),load_external_data=True); model_outputs={o for node in m.graph.node for o in node.output}; common=[n for n in names if n in model_outputs]
  got=run_onnxrt(m,common,INPUTS,label)
  rows=[]
  for order,n in enumerate(common):
   met=tensor_metrics(fp[n],got[n]); rows.append({'order':order,'layer':int(next(x for x in by[n]['node_name'].split('/layer.')[1].split('/')[0:1])[0:]),'node_name':by[n]['node_name'],'tensor':n,**met,'fp_stats':stats(fp[n]),'candidate_stats':stats(got[n])})
  # Flatten CSV-friendly metrics and distributions.
  flat=[]
  for r in rows:
   z={k:v for k,v in r.items() if k not in ('fp_stats','candidate_stats')}; z.update({'fp_'+k:v for k,v in r['fp_stats'].items()}); z.update({'candidate_'+k:v for k,v in r['candidate_stats'].items()}); flat.append(z)
  write_csv(out/f'{label.lower()}_layerwise_distributions.csv',flat)
  audit=qdq_audit(m)
  enc_scales=[r for r in audit['scale_rows'] if r['op_type']=='QuantizeLinear' and '/backbone/backbone.0/encoder/' in r['tensor'] and r.get('scale_min') is not None]
  scale_values=[r['scale_min'] for r in enc_scales if r['scale_min']>0]
  scale_summary={'count':len(scale_values),'min':float(np.min(scale_values)) if scale_values else None,'median':float(np.median(scale_values)) if scale_values else None,'max':float(np.max(scale_values)) if scale_values else None,'p95':float(np.percentile(scale_values,95)) if scale_values else None}
  reports[label]={'status':'ok','model':str(path),'tensor_count':len(rows),'rows':rows,'qdq_audit':{k:v for k,v in audit.items() if k!='scale_rows'},'encoder_activation_scale_summary':scale_summary}
 (out/'encoder_comparison.json').write_text(json.dumps({'scope':'DINOv2 encoder 12 layers x Q/K/V/output/FFN-fc1/fc2 (72 MatMul)','fp32_model':str(FP),'inputs':str(INPUTS),'algorithms':reports},indent=2,ensure_ascii=False),encoding='utf-8')
 # concise summary table
 summary=[]
 for label,d in reports.items():
  if d.get('status')!='ok': continue
  rows=d['rows']; rel=np.array([r['relative_l1'] for r in rows]); cos=np.array([r['cosine_similarity'] for r in rows]);
  summary.append({'algorithm':label,'tensor_count':len(rows),'mean_relative_l1':float(rel.mean()),'p95_relative_l1':float(np.percentile(rel,95)),'max_relative_l1':float(rel.max()),'first_rel_l1_ge_0.5':next((r['order'] for r in rows if r['relative_l1']>=.5),None),'mean_cosine':float(cos.mean()),'min_cosine':float(cos.min())})
 (out/'encoder_summary.csv').write_text('algorithm,tensor_count,mean_relative_l1,p95_relative_l1,max_relative_l1,first_rel_l1_ge_0.5,mean_cosine,min_cosine\n'+'\n'.join(','.join(str(x.get(k,'')) for k in summary[0]) for x in []),encoding='utf-8')
 with (out/'encoder_summary.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
 try:
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  colors={'MinMax':'#4575b4','Percentile':'#d73027','SmoothQuant':'#1a9850'}
  fig,ax=plt.subplots(figsize=(13,6))
  for label,d in reports.items():
   if d.get('status')=='ok': ax.plot([r['order'] for r in d['rows']],[r['relative_l1'] for r in d['rows']],marker='.',label=label,color=colors[label])
  ax.axhline(.5,color='gray',ls='--',lw=1); ax.set_yscale('symlog',linthresh=.02); ax.set_xlabel('encoder MatMul order (12 layers x 6)'); ax.set_ylabel('relative L1'); ax.set_title('Encoder layerwise error by calibration algorithm'); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(out/'layerwise_relative_l1.png',dpi=160); plt.close(fig)
  fig,ax=plt.subplots(figsize=(10,6))
  vals=[]; labs=[]
  for label,d in reports.items():
   if d.get('status')=='ok': vals.append([r['relative_l1'] for r in d['rows']]); labs.append(label)
  ax.boxplot(vals,tick_labels=labs,showfliers=False); ax.set_ylabel('relative L1'); ax.set_title('Distribution of encoder errors'); ax.grid(axis='y',alpha=.2); fig.tight_layout(); fig.savefig(out/'error_distribution_boxplot.png',dpi=160); plt.close(fig)
  fig,ax=plt.subplots(figsize=(10,6))
  for label,d in reports.items():
   if d.get('status')=='ok': ax.plot([r['order'] for r in d['rows']],[r['candidate_p99_99'] if 'candidate_p99_99' in r else r['candidate_stats']['p99_99'] for r in d['rows']],marker='.',label=label,color=colors[label])
  ax.set_yscale('log'); ax.set_xlabel('encoder MatMul order'); ax.set_ylabel('candidate |activation| p99.99'); ax.set_title('Activation-tail distribution after quantization'); ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(out/'activation_tail_p99_99.png',dpi=160); plt.close(fig)
 except ImportError:
  pass
 print(json.dumps(summary,indent=2))
main()
