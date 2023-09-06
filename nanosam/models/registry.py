MODELS = {}


def register_model(name: str):
    def _register_model(fn):
        MODELS[name] = fn
        return fn
    return _register_model


def create_model(name: str):
    return MODELS[name]()


def list_models():
    return sorted(MODELS.keys())