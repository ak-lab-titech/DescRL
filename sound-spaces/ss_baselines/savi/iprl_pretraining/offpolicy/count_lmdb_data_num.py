import lmdb


def count_entries_in_lmdb(env_path):
    env = lmdb.open(env_path, readonly=True, lock=False)
    
    with env.begin() as txn:
        cursor = txn.cursor()
        entry_count = sum(1 for _ in cursor)
    env.close()
    
    return entry_count


if __name__=="__main__":
    env_path = '/home/4/ud02274/navigation/myss/sound-spaces/data/lmdb_dataset/iprl_pretrain/train-last-step'
    entry_count = count_entries_in_lmdb(env_path)
    print(f'Total number of entries in the LMDB: {entry_count}')
