import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from dexa_ai.pretrained_models import available_backbones, build_pretrained, imagenet_normalize

def test_pretrained_architectures_construct_without_download():
    for name in available_backbones():
        m=build_pretrained(name, 4, pretrained=False)
        x=torch.rand(2,1,224,224)
        y=m(imagenet_normalize(x))
        assert y.shape==(2,4)
