import os


def save_model(model, out_dir: str) -> str:
    if model.__class__.__module__.startswith("xgboost"):
        path = os.path.join(out_dir, "xgb.json")
        model.save_model(path)
    else:
        import joblib
        path = os.path.join(out_dir, "model.joblib")
        joblib.dump(model, path)
    return os.path.basename(path)


def load_model(model_dir: str, filename: str):
    path = os.path.join(model_dir, filename)
    if filename.endswith(".json"):
        from xgboost import XGBClassifier
        m = XGBClassifier()
        m.load_model(path)
        return m
    import joblib
    return joblib.load(path)
