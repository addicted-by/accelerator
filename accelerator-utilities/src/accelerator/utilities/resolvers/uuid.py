import omegaconf
import uuid

omegaconf.OmegaConf.register_new_resolver('uuid4', lambda n=-1: uuid.uuid4().hex[:n], use_cache=True)