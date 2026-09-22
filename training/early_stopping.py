class EarlyStopping:
    """验证损失连续多轮没有明显改善时停止训练。"""

    def __init__(self, patience, min_delta):
        if patience < 1:
            raise ValueError("Early Stopping的patience必须大于0。")
        if min_delta < 0:
            raise ValueError("Early Stopping的min_delta不能小于0。")
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait_count = 0

    def update(self, val_loss):
        improved = val_loss < self.best_loss - self.min_delta
        if improved:
            self.best_loss = val_loss
            self.wait_count = 0
        else:
            self.wait_count += 1
        should_stop = self.wait_count >= self.patience
        return improved, should_stop
