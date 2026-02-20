from services.model_registry import ModelRegistry
reg = ModelRegistry()
versions = reg.get_history()
for v in versions:
    print(v)
