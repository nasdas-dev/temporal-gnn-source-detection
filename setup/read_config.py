class Config:
    def __init__(self, data):
        # create a recursive config object so that we can access parameters via dot notation, e.g. cfg.nwk.name
        for k, v in data.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)
    def __repr__(self):
        if hasattr(self, "nwk"):
            n = self.nwk
            return f"Network {n.name} from {n.original_start} to {n.original_end} has {n.time_steps} time steps with granularity {n.time_granularity}"
        else:
            return ""