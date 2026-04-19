__all__ = [
    "DFlashDraftModel",
    "extract_context_feature",
    "load_and_process_dataset",
    "sample",
]

# Personal note: __getattr__ enables lazy imports so heavy dependencies
# (torch, transformers, etc.) are only loaded when actually needed.
def __getattr__(name):
    if name == "load_and_process_dataset":
        from .benchmark import load_and_process_dataset

        return load_and_process_dataset

    if name in {"DFlashDraftModel", "extract_context_feature", "sample"}:
        from .model import DFlashDraftModel, extract_context_feature, sample

        return {
            "DFlashDraftModel": DFlashDraftModel,
            "extract_context_feature": extract_context_feature,
            "sample": sample,
        }[name]

    # Provide a friendlier error message listing valid names
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}. "
        f"Available attributes: {__all__}"
    )
