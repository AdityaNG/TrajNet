import torch


class BaseModel(torch.nn.Module):
    def load_net(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        if path is None or not path:
            # No model to load
            return
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            print("Loading optimizer state dict")
            parameters = parameters["model"]

        # validate all keys in state_dict are present in self.state_dict()
        for k in parameters:
            if k not in self.state_dict():
                raise Exception(
                    "Loading: {self.__class__.__name__} state_dict does not \
                        contain key {k} when loading from {path}"
                )

        incompatible_keys = self.load_state_dict(
            parameters,
            strict=True,
        )
        print("incompatible_keys", incompatible_keys)

        del parameters
        torch.cuda.empty_cache()

    def get_device(
        self,
    ):
        return next(self.parameters()).device
