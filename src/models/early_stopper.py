class EarlyStopCounter:
    def __init__(self,
                 patience: int = 30,
                 delta: float = 1e-3,
                 mode: str = "max",
                 threshold: float = None,
                 verbose: bool = True,
                 ) -> None:

        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.threshold = threshold if threshold else 0.0
        self.verbose = verbose

        self.epoch = None
        self.counter = None # improvement counter
        self.best_epoch = None
        self.best_score = None
        self.found_best = False

        self.history = {"epoch": [],
                        "counter": [],
                        "found_best": [],
                        "best_epoch": [],
                        "best_score": [],
                        }

        self.trigger_stop = False
        self.is_active = False
        # self.best_params = None

    def get_params_shallow(self) -> dict[str,...]:
        params = {kw: getattr(self, kw) for kw in ["patience", "delta", "mode", "threshold"]}
        return params

    def state_dict(self) -> dict:
        state = {'history': self.history, # must have
                 'is_active': self.is_active,
                 }

        # do we want to store the trigger? What if we want to override an already triggered stopper to continue training?
        # state['trigger_stop'] = self.trigger_stop

        return state

    def load_state_dict(self, state_dict: dict) -> None:

        self.history = state_dict['history']

        for kw, value in state_dict['history'].items():
            if len(value):
                setattr(self, kw, value[-1])

        if 'is_active' in state_dict:
            self.is_active = state_dict['is_active']
        else: # fallback in case is_active was not stored
            self.is_active = (len(self.history['counter']) > 0)


    def _update_history(self):
        for key in ["epoch", "counter", "found_best", "best_epoch", "best_score"]:
            self.history[key].append(getattr(self, key))

    def _check_improvement(self, score: float):
        if self.mode == 'max':
            return score > (self.best_score + self.delta)
        else:
            return score < (self.best_score - self.delta)

    def _check_threshold(self, score: float):
        if self.mode == 'max':
            return score >= self.threshold
        else:
            return score <= self.threshold

    def activate(self,
                 current_epoch: int,
                 reference_score: float,
                 ) -> None:

        if not self._check_threshold(reference_score):
            return

        else:
            self.epoch = current_epoch
            self.counter = 0
            self.found_best = True
            self.best_epoch = current_epoch
            self.best_score = reference_score

            self.is_active = True
            print(f"Stopping activated (Reference score: {self.best_score:.4f})")

    def step(self,
             current_epoch: int,
             current_score: float,
             ) -> None:

        if not self.is_active:
            self.activate(current_epoch, current_score)

        else:
            # check = self._check_improvement(current_score)

            self.found_best = self._check_improvement(current_score)
            self.epoch = current_epoch

            if self.found_best:

                if self.verbose:
                    delta = abs(current_score - self.best_score)
                    print(f"Score improved by {delta:.6f} ({self.best_score:.6f} -> {current_score:.6f})")

                self.counter = 0
                self.best_score = current_score
                self.best_epoch = current_epoch

            else:
                self.counter+=1

                if self.verbose:
                    print(f"Early stopping counter: {self.counter}/{self.patience} (Current best: {self.best_score:.6f})")

                if self.counter >= self.patience:
                    print(f"Early stopping triggered at epoch {current_epoch}.")
                    self.trigger_stop = True

        self._update_history()

    def _reset(self) -> None:
        self.__init__(patience=self.patience,
                      delta=self.delta,
                      mode=self.mode,
                      threshold=self.threshold,
                      verbose=self.verbose)


# Public alias: the rest of the codebase (workflow_v3, tracking, scripts) refers
# to this class as `EarlyStopper`.
EarlyStopper = EarlyStopCounter