import re

def analyze_log(filename):
    try:
        with open(filename, 'r') as f:
            log_content = f.read()
    except:
        return None
        
    recall_pattern = r'recall@20 : ([\d\.]+)'
    ndcg_pattern = r'ndcg@20 : ([\d\.]+)'
    mf_loss_pattern = r'mf_loss:([\d\.]+)'
    ssl_loss_pattern = r'ssl_loss_list:([\d\.]+)'

    recalls = [float(x) for x in re.findall(recall_pattern, log_content)]
    ndcgs = [float(x) for x in re.findall(ndcg_pattern, log_content)]
    mf_losses = [float(x) for x in re.findall(mf_loss_pattern, log_content)]
    ssl_losses = [float(x) for x in re.findall(ssl_loss_pattern, log_content)]

    if not recalls:
        return None
        
    max_recall = max(recalls)
    max_epoch = recalls.index(max_recall)
    
    return {
        'total_epochs': len(recalls),
        'first_recall': recalls[0],
        'last_recall': recalls[-1],
        'max_recall': max_recall,
        'max_ndcg': max(ndcgs),
        'best_epoch': max_epoch,
        'last_5_mf': mf_losses[-5:] if len(mf_losses)>=5 else mf_losses,
        'last_5_ssl': ssl_losses[-5:] if len(ssl_losses)>=5 else ssl_losses,
        'recalls': recalls
    }

old_log = analyze_log('log.log')
new_log = analyze_log('log1.log')

if old_log:
    print("--- OLD LOG (log.log) ---")
    print(f"Total Epochs: {old_log['total_epochs']}")
    print(f"Max Recall@20: {old_log['max_recall']:.4f} at epoch {old_log['best_epoch']}")
    print(f"Initial -> Final Recall: {old_log['first_recall']:.4f} -> {old_log['last_recall']:.4f}")

if new_log:
    print("\n--- NEW LOG (log1.log) ---")
    print(f"Total Epochs: {new_log['total_epochs']}")
    print(f"Max Recall@20: {new_log['max_recall']:.4f} at epoch {new_log['best_epoch']}")
    print(f"Initial -> Final Recall: {new_log['first_recall']:.4f} -> {new_log['last_recall']:.4f}")
    if len(new_log['recalls']) > 30:
        print(f"Recall change over the last 30 epochs: {new_log['recalls'][-1] - new_log['recalls'][-30]:.4f}")
    print(f"Final MF Losses: {new_log['last_5_mf']}")
    print(f"Final SSL Losses: {new_log['last_5_ssl']}")

if old_log and new_log:
    diff = new_log['max_recall'] - old_log['max_recall']
    print(f"\nDifference in Max Recall: {diff:+.4f}")
