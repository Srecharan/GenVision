class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_average_metrics(metrics_list):
    """Compute average metrics across batches"""
    avg_metrics = {}
    keys = metrics_list[0].keys()
    
    for key in keys:
        values = [metrics[key].detach().cpu().numpy() for metrics in metrics_list]
        avg_metrics[key] = sum(values) / len(values)
    
    return avg_metrics