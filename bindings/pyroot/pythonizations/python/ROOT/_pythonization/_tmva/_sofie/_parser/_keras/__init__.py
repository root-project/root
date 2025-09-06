def get_keras_version() -> str:
    
    import keras
    
    return keras.__version__

keras_version = get_keras_version()