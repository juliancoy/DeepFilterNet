import argparse,hashlib,json,time,datetime,pathlib,os
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from df.enhance import init_df,enhance

parser=argparse.ArgumentParser(description='Enhance speech using CUDA DeepFilterNet with overlapping sections.')
parser.add_argument('input',type=pathlib.Path)
parser.add_argument('output',type=pathlib.Path)
args=parser.parse_args()
source=args.input
target=args.output
partial=target.with_name(target.stem+'.processing.wav')
evidence=target.with_suffix('.run.jsonl')
def record(**items):
    entry={'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),**items}
    with evidence.open('a') as f:f.write(json.dumps(entry)+'\n')
    print(json.dumps(entry),flush=True)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
assert torch.cuda.is_available(),'CUDA is required; refusing CPU fallback'
assert target.suffix.lower()=='.wav','Output must be WAV'
torch.set_num_threads(4)
model,state,*_=init_df(model_base_dir='DeepFilterNet3',post_filter=False,log_file=None)
assert next(model.parameters()).is_cuda
assert not target.exists() and not partial.exists()
info=sf.info(source)
rate=info.samplerate
assert state.sr()==48000
from math import gcd
factor=gcd(rate,state.sr())
up,down=state.sr()//factor,rate//factor
record(event='start',command=' '.join(__import__('sys').argv),gpu=torch.cuda.get_device_name(0),device=str(next(model.parameters()).device),torch=torch.__version__,script_sha256=sha(pathlib.Path(__file__)),source_sha256=sha(source),frames=info.frames,channels=info.channels,sample_rate=rate,model='DeepFilterNet3',segment_seconds=30,overlap_seconds=1)
start_time=time.monotonic()
window=rate*30;overlap=rate;hop=window-overlap
fade=np.linspace(0,1,overlap,dtype=np.float32)[:,None]
pending=None;written=0;clipped=0
with sf.SoundFile(source) as src,sf.SoundFile(partial,'x',samplerate=rate,channels=info.channels,subtype='PCM_16',format='WAV') as out:
    for start in range(0,info.frames,hop):
        src.seek(start)
        audio=src.read(min(window,info.frames-start),dtype='float32',always_2d=True)
        model_audio=resample_poly(audio,up,down,axis=0)
        clean=enhance(model,state,torch.from_numpy(np.ascontiguousarray(model_audio.T)),pad=True).numpy().T
        clean=resample_poly(clean,down,up,axis=0)[:len(audio)]
        assert len(clean)==len(audio) and np.isfinite(clean).all()
        if pending is not None:
            n=min(len(pending),len(clean));clean[:n]=pending[:n]*(1-fade[:n])+clean[:n]*fade[:n]
        last=start+len(audio)>=info.frames
        emit=clean if last else clean[:-overlap]
        clipped+=int(np.count_nonzero(np.abs(emit)>1))
        out.write(emit);written+=len(emit)
        pending=clean[-overlap:].copy()
        if start//hop%10==0 or last:
            out.flush()
            elapsed=time.monotonic()-start_time
            record(event='progress',percent=round(100*written/info.frames,2),audio_minutes=round(written/rate/60,2),elapsed_seconds=round(elapsed,1),speed=round(written/rate/elapsed,2),gpu_peak_memory_bytes=torch.cuda.max_memory_allocated())
        if last:break
assert written==info.frames,(written,info.frames)
check=sf.info(partial)
assert (check.frames,check.samplerate,check.channels)==(info.frames,rate,info.channels)
partial.rename(target)
record(event='validation',command='per-segment finite checks; exact final frame/rate/channel assertions; SHA256',result='PASS',output=str(target),output_sha256=sha(target),frames=check.frames,clipped_samples=clipped,elapsed_seconds=round(time.monotonic()-start_time,1),script_sha256=sha(pathlib.Path(__file__)))
