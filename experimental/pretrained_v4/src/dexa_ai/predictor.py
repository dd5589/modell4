from __future__ import annotations
import os, json, time, base64, io
from pathlib import Path
import numpy as np, torch
from .dicomio import read_dicom
from .models import build_model
from .preprocess import tensor_from_array
from .labels import TARGET_TO_DESCRIPTION

FINAL_REGION_NAME = {
    'spine': 'Поясничный отдел позвоночника',
    'hip': 'Проксимальный отдел бедра',
    'unknown': 'Неизвестно',
}
FINAL_VIOLATION_NAME = {
    'spine_placement': 'Некорректная укладка',
    'spine_axis': 'Не выравнена ось позвоночника',
    'spine_artifact': 'Присутствуют посторонние предметы',
    'hip_rotation': 'Некорректная укладка',
    'hip_roi': 'Некорректная область интереса',
}

class EnsemblePredictor:
    def __init__(self, weights_dir, device=None, thresholds_file=None):
        self.root=Path(weights_dir)
        self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        cfg=json.load(open(thresholds_file or self.root/'ensemble.json',encoding='utf8')) if os.path.exists(thresholds_file or self.root/'ensemble.json') else {}
        self.model_names=cfg.get('models',['fastcnn','dilatedcnn','depthwisecnn'])
        self.thresholds=cfg.get('thresholds',{'spine':{},'hip':{}})
        self.targets={'spine':['quality','placement','axis','artifact'],'hip':['quality','rotation','roi']}
        self.region_weights=cfg.get('region_weights',{'spine':{'deep':0.5,'hog':0.5},'hip':{'deep':1.0,'hog':0.0}})
        advanced_cfg_path=self.root/'advanced_ensemble.json'
        self.advanced_weights={}
        if advanced_cfg_path.exists():
            with open(advanced_cfg_path,encoding='utf8') as f: self.advanced_weights=json.load(f).get('target_weights',{})
        self.models={}
        for region in self.targets:
            self.models[region]=[]
            for name in self.model_names:
                p=self.root/'weights'/region/f'{name}.pt'
                if not p.exists(): raise FileNotFoundError(f'Model weight not found: {p}')
                ck=torch.load(p,map_location='cpu')
                m=build_model(name,len(self.targets[region])); m.load_state_dict(ck['state_dict']); m.to(self.device).eval(); self.models[region].append(m)
        import pickle
        self.hog_models={}
        for region in self.targets:
            self.hog_models[region]={}
            for t in self.targets[region]:
                hp=self.root/'weights'/'hog'/region/f'{t}.pkl'
                if hp.exists():
                    with open(hp,'rb') as f: self.hog_models[region][t]=pickle.load(f)
        # Task-specific expert heads (geometry / classical CV). They are optional
        # so an older artifact bundle remains loadable.
        self.expert_models={}
        self.expert_cfg={}
        expert_dir=self.root/'experts'
        cfg_path=expert_dir/'expert_config.json'
        if cfg_path.exists():
            with open(cfg_path,encoding='utf8') as f: self.expert_cfg=json.load(f)
            for target in ('spine_placement','spine_artifact','hip_rotation','hip_roi'):
                ep=expert_dir/f'{target}.pkl'
                if ep.exists():
                    with open(ep,'rb') as f: self.expert_models[target]=pickle.load(f)
        # Additional compact-feature LightGBM bagger. These members are optional so
        # the original frozen bundle remains backward compatible.
        self.lgbm_models={}
        self.lgbm_thresholds={}
        lgbm_dir=self.root/'lgbm_bag'
        if lgbm_dir.exists():
            try:
                import lightgbm as lgb
                for region, targets in self.targets.items():
                    self.lgbm_models[region]={}
                    for target in targets:
                        prefix=f'{region}_{target}'
                        model_files=sorted(lgbm_dir.glob(prefix+'_fold*.txt'))
                        full=lgbm_dir/(prefix+'_full.txt')
                        boosters=[]
                        for mp in model_files + ([full] if full.exists() else []):
                            try: boosters.append(lgb.Booster(model_file=str(mp)))
                            except Exception: pass
                        if boosters: self.lgbm_models[region][target]=boosters
                        mp=lgbm_dir/(prefix+'_metrics.json')
                        if mp.exists():
                            with open(mp,encoding='utf8') as f: self.lgbm_thresholds[region,target]=float(json.load(f).get('threshold',0.5))
            except Exception:
                self.lgbm_models={}
        # Optional v0.4 pretrained heterogeneous vision ensemble. It stays disabled
        # until H200 training + same-holdout validation produces a justified blend.
        self.pretrained_models={}
        self.pretrained_cfg={}
        pre_cfg_path=self.root/'pretrained_ensemble.json'
        if pre_cfg_path.exists():
            try:
                with open(pre_cfg_path,encoding='utf8') as f: self.pretrained_cfg=json.load(f)
                if self.pretrained_cfg.get('enabled',False):
                    from .pretrained_models import build_pretrained
                    for region in self.targets:
                        self.pretrained_models[region]=[]
                        for bk in self.pretrained_cfg.get('backbones',[]):
                            wp=self.root/'pretrained'/region/f'{bk}.pt'
                            if not wp.exists(): continue
                            ck=torch.load(wp,map_location='cpu',weights_only=False)
                            m=build_pretrained(bk,len(self.targets[region]),pretrained=False)
                            m.load_state_dict(ck['state_dict']); m.to(self.device).eval(); self.pretrained_models[region].append(m)
            except Exception:
                self.pretrained_models={}
    def region(self,d):
        if 295<=d.cols<=305: return 'spine'
        if 220<=d.cols<=294: return 'hip'
        name=(d.metadata or {}).get('SeriesDescription','').lower()
        if 'spine' in name or 'позвон' in name: return 'spine'
        if 'hip' in name or 'бедр' in name: return 'hip'
        return 'unknown'
    def predict_one(self,path,explain=False):
        t=time.perf_counter(); d=read_dicom(path); region=self.region(d)
        result={'path_to_study':str(path),'study_uid':d.study_uid,'image_uid':d.sop_uid,'anatomical_region':FINAL_REGION_NAME[region]}
        if region=='unknown':
            result.update({'quality_class':None,'violation_type':'','processing_status':'Failure','time_of_processing':time.perf_counter()-t,'quality_prob':None,'error_message':'Unsupported/unknown anatomical projection'})
            return result
        x=tensor_from_array(d.array,train=False).unsqueeze(0).to(self.device)
        deep_ps=[]
        for m in self.models[region]:
            with torch.no_grad(): deep_ps.append(torch.sigmoid(m(x)).cpu().numpy()[0])
        deep_p=np.mean(deep_ps,axis=0); targets=self.targets[region]; th=self.thresholds.get(region,{})
        rw=self.region_weights.get(region,{'deep':1.0,'hog':0.0}); p=deep_p.copy()
        hw=float(rw.get('hog',0.0)); dw=float(rw.get('deep',1.0));
        if hw>0 and len(self.hog_models.get(region,{}))==len(targets):
            from .hog_member import hog_features
            h=hog_features([tensor_from_array(d.array,train=False)])
            hp=np.array([self.hog_models[region][t].predict_proba(h)[:,1][0] for t in targets],dtype=float)
            p=(dw*deep_p+hw*hp)/max(dw+hw,1e-8)
        # Blend rare-label expert heads into the deep ensemble. Axis uses the explicit
        # 5° geometric criterion from the specification; other experts are learned.
        expert_probs={}
        for target in targets[1:]:
            key=f'{region}_{target}'
            if key in self.expert_cfg.get('target_weights',{}):
                try:
                    from .experts import expert_probability
                    ep=expert_probability(key,d.array,self.expert_models,self.expert_cfg)
                    expert_probs[target]=float(ep)
                    w=self.expert_cfg['target_weights'][key]
                    dw,ew=float(w.get('deep',1.0)),float(w.get('expert',0.0))
                    i=targets.index(target)
                    p[i]=(dw*p[i]+ew*ep)/max(dw+ew,1e-8)
                except Exception:
                    pass

        # Target-specific weights chosen from study-level validation evidence.
        # The deep/HOG/expert members remain primary where their evidence is stronger.
        default_lgbm_weight={
            'spine': {'quality':0.30,'placement':0.25,'axis':0.20,'artifact':0.0},
            'hip': {'quality':0.25,'rotation':0.25,'roi':0.55},
        }.get(region,{})
        lgbm_weight={}
        for target in targets:
            aw=self.advanced_weights.get(f'{region}_{target}',{})
            lgbm_weight[target]=float(aw.get('lgbm', default_lgbm_weight.get(target,0.0)))
        # v0.4 pretrained ensemble: average independent ImageNet-initialized
        # representations. Blend weights are target-specific and frozen in JSON.
        if self.pretrained_models.get(region):
            try:
                from .pretrained_models import imagenet_normalize
                pp=[]
                with torch.no_grad():
                    px=imagenet_normalize(tensor_from_array(d.array,train=False).unsqueeze(0)).to(self.device)
                    for m in self.pretrained_models[region]:
                        pp.append(torch.sigmoid(m(px)).cpu().numpy()[0])
                        # horizontal-flip TTA is safe because laterality has no impact
                        # on the organizer's anatomical_region vocabulary.
                        fx=torch.flip(px,[3])
                        pp.append(torch.sigmoid(m(fx)).cpu().numpy()[0])
                pretrained_p=np.mean(np.asarray(pp),axis=0)
                for i,target in enumerate(targets):
                    w=float(self.pretrained_cfg.get('target_weights',{}).get(f'{region}_{target}',0.0))
                    p[i]=(1.0-w)*p[i]+w*float(pretrained_p[i])
            except Exception:
                pretrained_p=None
        else:
            pretrained_p=None

        if region in self.lgbm_models:
            from .lgbm_features import small_image_features
            lx=small_image_features(d.array).reshape(1,-1)
            lgbm_p={}
            for target,boosters in self.lgbm_models[region].items():
                try:
                    vals=[float(b.predict(lx)[0]) for b in boosters]
                    lgbm_p[target]=float(np.mean(vals))
                except Exception:
                    continue
            for target,lp in lgbm_p.items():
                w=float(lgbm_weight.get(target,0.0)); i=targets.index(target)
                p[i]=(1.0-w)*p[i]+w*lp
        qthr=float(th.get('quality',[0.5])[0] if isinstance(th.get('quality'),list) else th.get('quality',0.5))
        violations=[]; probs={}
        for i,target in enumerate(targets):
            probs[target]=float(p[i])
        for i,target in enumerate(targets[1:],start=1):
            tv=th.get(target,[0.5])[0] if isinstance(th.get(target),list) else th.get(target,0.5)
            if p[i]>=float(tv):
                name=FINAL_VIOLATION_NAME[f'{region}_{target}']
                if name not in violations:
                    violations.append(name)
        # quality_prob is the probability that at least one quality violation is present.
        violation_probs=[probs[t] for t in targets[1:]]
        any_violation_prob=float(1.0 - np.prod([1.0-v for v in violation_probs])) if violation_probs else 0.0
        quality_prob=float(np.clip(max(float(probs['quality']), any_violation_prob),0.0,1.0))
        # Keep the binary quality flag logically consistent with the detected taxonomy.
        quality=int((float(probs['quality'])>=qthr) or bool(violations))
        disagreements=[target for target in targets[1:] if target in expert_probs and abs(expert_probs[target]-float(deep_p[targets.index(target)]))>=0.35]
        near_threshold=[]
        for target in targets[1:]:
            raw_t = th.get(target, 0.5)
            tv = raw_t[0] if isinstance(raw_t, list) else raw_t
            if abs(float(p[targets.index(target)]) - float(tv)) <= 0.10:
                near_threshold.append(target)
        result.update({
            'quality_class':quality,
            'violation_type':'; '.join(violations),
            'processing_status':'Success',
            'time_of_processing':time.perf_counter()-t,
            'quality_prob':quality_prob,
            'probabilities':probs,
            'expert_probabilities':expert_probs,
            'review_required':bool(disagreements or near_threshold),
            'ensemble_members': {
                'deep_cnn': len(self.models.get(region,[])),
                'hog': len(self.hog_models.get(region,{})),
                'experts': sorted(self.expert_models),
                'lgbm_bag': {t: len(v) for t,v in self.lgbm_models.get(region,{}).items()},
                'pretrained_backbones': [getattr(m,'name','unknown') for m in self.pretrained_models.get(region,[])],
            },
        })
        if region=='spine':
            from .experts import centerline_angle_deg
            result['axis_angle_deg']=float(centerline_angle_deg(d.array))
        if explain:
            result['saliency_base64']=self.saliency(d.array,region)
        return result
    def saliency(self,arr,region):
        # Lightweight input-gradient explanation. Returned as PNG data URI for API/UI.
        x=tensor_from_array(arr,train=False).unsqueeze(0).to(self.device); x.requires_grad_(True)
        logits=[]
        for m in self.models[region]: logits.append(m(x)[0,0])
        score=torch.stack(logits).mean(); score.backward(); g=x.grad.detach().abs()[0,0].cpu().numpy()
        g=(g-g.min())/(g.max()-g.min()+1e-8)
        from PIL import Image
        import matplotlib.cm as cm
        # Avoid hard-coded app colors; colormap is only for an optional heatmap asset.
        rgb=(cm.inferno(g)[...,:3]*255).astype('uint8')
        bio=io.BytesIO(); Image.fromarray(rgb).save(bio,format='PNG')
        return 'data:image/png;base64,'+base64.b64encode(bio.getvalue()).decode('ascii')
