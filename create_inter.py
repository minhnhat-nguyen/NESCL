import os

os.makedirs('dataset/ml-1m_recbole', exist_ok=True)
with open('dataset/ml-1m/ratings.dat', 'r') as f_in, open('dataset/ml-1m_recbole/ml-1m_recbole.inter', 'w') as f_out:
    f_out.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
    for line in f_in:
        parts = line.strip().split('::')
        if len(parts) == 4:
            f_out.write(f"{parts[0]}\t{parts[1]}\t{parts[2]}\t{parts[3]}\n")

print("ml-1m_recbole.inter created successfully.")
