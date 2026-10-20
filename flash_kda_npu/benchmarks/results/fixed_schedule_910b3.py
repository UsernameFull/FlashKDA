import json, os, statistics
from pathlib import Path
import torch
import torch_npu
import flash_kda
from flash_kda_npu.benchmarks.bench_fwd import bench_fn
from flash_kda_npu.ops.ascendc import get_last_profile

from flash_kda_npu.ops.ascendc import _persistent_fixed_blocks
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--json',type=Path,required=True)
args=parser.parse_args()
rows=[]
for h,t,old in ((96,8192,32),(128,8192,32),(96,256,32)):
    torch.manual_seed(42)
    shape=(1,t,h,128)
    q,k,v,g=[torch.randn(shape,device='npu',dtype=torch.bfloat16) for _ in range(4)]
    beta=torch.randn(shape[:-1],device='npu',dtype=torch.bfloat16)
    a,bias=torch.rand(h,device='npu'),torch.rand(h,128,device='npu')
    initial=torch.randn((1,h,128,128),device='npu')
    final=torch.empty_like(initial)
    out=torch.empty_like(q)
    def run():
        flash_kda.fwd(q,k,v,g,beta,128**-.5,out,a,bias,-5.,initial_state=initial,final_state=final)
    ref=None
    for round_id, configs in enumerate(((old,0),(0,old))):
        for override in configs:
            os.environ['KDA_PERSIST_LOOP_BLOCKS']=str(override)
            os.environ['KDA_PROFILE']='0'
            run(); torch.npu.synchronize()
            assert bool(torch.isfinite(out).all()) and bool(torch.isfinite(final).all())
            if ref is None: ref=out.clone(),final.clone()
            assert torch.equal(out,ref[0]) and torch.equal(final,ref[1])
            timing=bench_fn(run,30,200,3)
            os.environ['KDA_PROFILE']='1'
            ps=[]
            for _ in range(7):
                run(); ps.append(get_last_profile()['k2_ms'])
            row=dict(H=h,T=t,round=round_id,override=override,
                     blocks=get_last_profile()['persistent_blocks'],
                     bitwise_equal=True,k2_host_ms=statistics.median(ps),**timing)
            rows.append(row); print(json.dumps(row),flush=True)
            args.json.write_text(json.dumps(rows,indent=2)+'\n')
    del q,k,v,g,beta,a,bias,initial,final,out,ref
    torch.npu.empty_cache()
