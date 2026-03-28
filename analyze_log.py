import re

log_content = open('log3.log', 'r').read()

# Metrics
recall_pattern = r'recall@20 : ([\d\.]+)'
mrr_pattern = r'mrr@20 : ([\d\.]+)'
ndcg_pattern = r'ndcg@20 : ([\d\.]+)'
hit_pattern = r'hit@20 : ([\d\.]+)'
precision_pattern = r'precision@20 : ([\d\.]+)'

# Losses
train_loss_pattern = r'train loss: ([\d\.]+)'
mf_loss_pattern = r'mf_loss:([\d\.]+)'
ssl_loss_pattern = r'ssl_loss_list:([\d\.]+)'

# Times
train_time_pattern = r'training \[time: ([\d\.]+)s'
eval_time_pattern = r'evaluating \[time: ([\d\.]+)s'

recalls = [float(x) for x in re.findall(recall_pattern, log_content)]
mrrs = [float(x) for x in re.findall(mrr_pattern, log_content)]
ndcgs = [float(x) for x in re.findall(ndcg_pattern, log_content)]
hits = [float(x) for x in re.findall(hit_pattern, log_content)]
precisions = [float(x) for x in re.findall(precision_pattern, log_content)]

train_losses = [float(x) for x in re.findall(train_loss_pattern, log_content)]
mf_losses = [float(x) for x in re.findall(mf_loss_pattern, log_content)]
ssl_losses = [float(x) for x in re.findall(ssl_loss_pattern, log_content)]
train_times = [float(x) for x in re.findall(train_time_pattern, log_content)]
eval_times = [float(x) for x in re.findall(eval_time_pattern, log_content)]

print("=== Model Performance Analysis ===")
print(f"Total epochs evaluated: {len(recalls)}")
if len(recalls) > 0:
    print(f"Best Recall@20: {max(recalls):.4f} at epoch {recalls.index(max(recalls))}")
    print(f"Best NDCG@20: {max(ndcgs):.4f} at epoch {ndcgs.index(max(ndcgs))}")
    print(f"Best MRR@20: {max(mrrs):.4f} at epoch {mrrs.index(max(mrrs))}")
    print(f"Best Hit@20: {max(hits):.4f} at epoch {hits.index(max(hits))}")
    print(f"Best Precision@20: {max(precisions):.4f} at epoch {precisions.index(max(precisions))}")
    
    print("\n=== Overfitting & Convergence Analysis ===")
    best_epoch = recalls.index(max(recalls))
    print(f"Train loss at best epoch ({best_epoch}): {train_losses[best_epoch] if best_epoch < len(train_losses) else 'N/A'}")
    print(f"Train loss at last epoch: {train_losses[-1] if train_losses else 'N/A'}")
    if best_epoch < len(train_losses) and len(train_losses) > 0:
        print(f"Train loss dropped by {train_losses[best_epoch] - train_losses[-1]:.4f} after best epoch, but evaluation metrics got worse (Overfitting).")
        
    print(f"MF loss at best epoch ({best_epoch}): {mf_losses[best_epoch] if best_epoch < len(mf_losses) else 'N/A'}")
    print(f"MF loss at last epoch: {mf_losses[-1] if mf_losses else 'N/A'}")

    print(f"SSL loss at best epoch ({best_epoch}): {ssl_losses[best_epoch] if best_epoch < len(ssl_losses) else 'N/A'}")
    print(f"SSL loss at last epoch: {ssl_losses[-1] if ssl_losses else 'N/A'}")

print("\n=== Efficiency Analysis ===")
if train_times:
    print(f"Average train time per epoch: {sum(train_times)/len(train_times):.2f}s")
    print(f"Average evaluation time: {sum(eval_times)/len(eval_times):.2f}s")

