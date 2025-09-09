from method.api.base import SynthRegistry
from .native import PrivSynSynthesizer

def _factory() -> PrivSynSynthesizer:
    return PrivSynSynthesizer()

SynthRegistry.register(PrivSynSynthesizer.method_id(), _factory)
