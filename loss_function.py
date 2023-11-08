import torch

class DistanceLoss(torch.nn.Module):
    def __init__(self, distance_type, d1=0.0, d2=0.0, dt=0.0, learnable=False):
        super(DistanceLoss, self).__init__()
        self.distance_type = distance_type
        self.learnable = learnable
        self.learnable_d1 = learnable and distance_type != "temporal"
        self.learnable_d2 = learnable and "2level" in distance_type
        self.learnable_dt = learnable and "temporal" in distance_type
        self.d1 = torch.nn.Parameter(torch.DoubleTensor([-2.0])) if self.learnable_d1 else d1
        self.d2 = torch.nn.Parameter(torch.DoubleTensor([-2.0])) if self.learnable_d2 else d2
        self.dt = torch.nn.Parameter(torch.DoubleTensor([-2.0])) if self.learnable_dt else dt

    def get_distance_vals(self):
        d1 = torch.nn.functional.softplus(self.d1, beta=2.0) if self.learnable_d1 else self.d1
        d2 = torch.nn.functional.softplus(self.d2, beta=2.0) if self.learnable_d2 else self.d2
        dt = torch.nn.functional.softplus(self.dt, beta=2.0) if self.learnable_dt else self.dt
        return d1, d2, dt

    def get_distance_vals_4display(self):
        dvals = self.get_distance_vals()
        return [
            dvals[0].item() if self.learnable_d1 else dvals[0],
            dvals[1].item() if self.learnable_d2 else dvals[1],
            dvals[2].item() if self.learnable_dt else dvals[2],
        ]

    def get_labels(self, distances):
        distances = distances.clone()
        conditions = [(distances == 0), (distances == 1), (distances == 2), (distances == 3)]
        distance_vals = self.get_distance_vals()
        distances[conditions[0]] = 1.0
        distances[conditions[1]] = distance_vals[0]
        distances[conditions[2]] = distance_vals[1]
        distances[conditions[3]] = 0.0
        if self.learnable:
            distances = distances/distances.sum(dim=1, keepdims=True)
        return distances

    def get_labels_temporal(self, distances):
        distances = distances.clone()
        conditions = [(distances == 1), (distances < 1), (distances == 0)]
        distance_vals = self.get_distance_vals()
        distances[conditions[0]] = 1.0
        distances[conditions[1]] = distances[conditions[1]] * distance_vals[2]
        distances[conditions[2]] = 0.0
        if self.learnable:
            distances = distances/distances.sum(dim=1, keepdims=True)
        return distances

    def forward(self, model_outputs, _, distance):
        softmax_outputs = torch.nn.functional.log_softmax(model_outputs, dim=1)
        if self.distance_type == "temporal":
            distance_embedded_labels = self.get_labels_temporal(distance)
        elif self.distance_type == "none":
            distance_embedded_labels = distance
        elif self.distance_type in ["temporal_verb", "temporal_object", "temporal_both", "temporal_object_2level", "temporal_verb_2level"]:
            del_1 = self.get_labels_temporal(distance[:, :, 0])
            del_2 = self.get_labels(distance[:, :, 1])
            ce_1 = -torch.sum(del_1 * softmax_outputs, dim=1)
            ce_2 = -torch.sum(del_2 * softmax_outputs, dim=1)
            return torch.mean(ce_1) + torch.mean(ce_2)
        else:
            distance_embedded_labels = self.get_labels(distance)
        ce = -torch.sum(distance_embedded_labels * softmax_outputs, dim=1)
        return torch.mean(ce)
