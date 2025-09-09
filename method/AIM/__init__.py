from method.api.base import SynthRegistry
from .native import AIMSynthesizer

def _factory() -> AIMSynthesizer:
    return AIMSynthesizer()

SynthRegistry.register(AIMSynthesizer.method_id(), _factory)
