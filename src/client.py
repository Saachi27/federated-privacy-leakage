"""A federated client: owns a private data shard and trains locally."""

import copy

import torch
import torch.nn as nn


class Client:
    def __init__(self, client_id, dataloader, device):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.num_samples = len(dataloader.dataset)
        # DP accounting state, accumulated across rounds this client participates in.
        self.dp_steps = 0
        self.dp_sample_rate = None
        self.dp_noise = None

    def local_train(self, model, global_state, local_epochs, lr, momentum, dp=None):
        """Train a fresh copy of the model from `global_state`."""
        local_model = copy.deepcopy(model).to(self.device)
        local_model.load_state_dict(global_state)
        local_model.train()

        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr, momentum=momentum)
        criterion = nn.CrossEntropyLoss()
        loader = self.dataloader

        privacy_engine = None
        if dp is not None:
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine()
            local_model, optimizer, loader = privacy_engine.make_private(
                module=local_model, optimizer=optimizer, data_loader=loader,
                noise_multiplier=dp["noise_multiplier"],
                max_grad_norm=dp["max_grad_norm"],
            )

        running_loss, n_batches = 0.0, 0
        for _ in range(local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(local_model(x), y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n_batches += 1

        if privacy_engine is not None:
            # Accumulate this round's steps for (epsilon, delta) accounting.
            self.dp_steps += n_batches
            self.dp_noise = dp["noise_multiplier"]
            # Opacus' Poisson sampler uses exactly this rate (see make_private).
            self.dp_sample_rate = 1 / len(self.dataloader)

        # Opacus wraps the model in a GradSampleModule; unwrap for clean keys.
        trained = getattr(local_model, "_module", local_model)
        state = {k: v.detach().cpu() for k, v in trained.state_dict().items()}
        avg_loss = running_loss / max(n_batches, 1)
        return state, self.num_samples, avg_loss
