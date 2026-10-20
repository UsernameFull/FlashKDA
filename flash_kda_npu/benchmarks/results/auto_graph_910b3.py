import json,os
import torch,torch_npu
import flash_kda
from flash_kda_npu.ops import clear_auto_graph_cache
from flash_kda_npu.benchmarks.bench_fwd import bench_fn
from pathlib import Path
rows=[]
for t in (256,8192):
 torch.manual_seed(42);shape=(1,t,96,128)
 q,k,v,g=[torch.randn(shape,device='npu',dtype=torch.bfloat16) for _ in range(4)]
 beta=torch.randn(shape[:-1],device='npu',dtype=torch.bfloat16)
 a,bias=torch.rand(96,device='npu'),torch.rand(96,128,device='npu')
 initial=torch.randn((1,96,128,128),device='npu');final=torch.empty_like(initial);out=torch.empty_like(q)
 def run():flash_kda.fwd(q,k,v,g,beta,128**-.5,out,a,bias,-5.,initial_state=initial,final_state=final)
 ref=None
 for mode in ('0','1','0','1'):
  os.environ['KDA_AUTO_GRAPH']=mode
  run();torch.npu.synchronize()
  if ref is None:ref=out.clone(),final.clone()
  assert torch.equal(out,ref[0]) and torch.equal(final,ref[1])
  timing=bench_fn(run,30,200,3)
  row=dict(T=t,auto_graph=mode,bitwise_equal=True,**timing);rows.append(row);print(json.dumps(row),flush=True)
  Path('/tmp/kda_auto_bench.json').write_text(json.dumps(rows,indent=2)+'\n')
 clear_auto_graph_cache()
 del q,k,v,g,beta,a,bias,initial,final,out,ref
 torch.npu.empty_cache()
