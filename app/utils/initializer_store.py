class InitializerStore:
    _instance = None
    initializer_dict = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(InitializerStore, cls).__new__(cls, *args, **kwargs)
        return cls._instance

