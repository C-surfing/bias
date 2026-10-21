"""Immutable dataset split and independent LLMRank-style evaluation."""
from collections import Counter,defaultdict
import math
from pathlib import Path
import random
import zipfile

def load_dataset(directory, dataset):
    directory = Path(directory)
    events = defaultdict(list)
    with zipfile.ZipFile(directory / f"{dataset}.inter.zip") as z:
        filenames = [n for n in z.namelist() if n.endswith(".inter")]
        if len(filenames) == 1:
            # ML-1M: timestamped raw interactions.
            lines = z.read(filenames[0]).decode("utf-8").splitlines()
            columns = lines[0].split("\t")
            ui, ii, ti = [next(j for j,k in enumerate(columns) if k.startswith(prefix))
                          for prefix in ("user_id:", "item_id:", "timestamp:")]
            for line in lines[1:]:
                fields = line.split("\t")
                events[fields[ui]].append((float(fields[ti]), fields[ii]))
        elif {f"{dataset}.{part}.inter" for part in ("train","valid","test")} <= set(filenames):
            # Games: official pre-split RecBole sequential files. The valid
            # prefix is the training history; targets are from valid/test.
            def split_rows(part):
                rows = {}
                raw = z.read(f"{dataset}.{part}.inter").decode("utf-8").splitlines()
                for line in raw[1:]:
                    fields = line.split("\t")
                    rows[fields[0]] = (fields[1].split(), fields[2])
                return rows
            val, test = split_rows("valid"), split_rows("test")
            for user in sorted(val.keys() & test.keys()):
                history, validation_target = val[user]
                _, test_target = test[user]
                items = [*history, validation_target, test_target]
                events[user] = [(float(idx), item) for idx,item in enumerate(items)]
        else:
            raise ValueError("unsupported interaction archive: expected raw or official train/valid/test .inter")
    titles = {}
    with (directory / f"{dataset}.item").open(encoding="utf-8") as f:
        headers = f.readline().rstrip("\n").split("\t")
        i = next(j for j,k in enumerate(headers) if k.startswith("item_id:"))
        t = next(j for j,k in enumerate(headers) if "title:" in k)
        for line in f:
            fields = line.rstrip("\n").split("\t")
            titles[fields[i]] = fields[t]
    for history in events.values():
        history.sort()
    train = {}
    positives = {}
    known = {}
    popularity = Counter()
    for user, seq in events.items():
        items = [i for _,i in seq if i in titles]
        if len(items) < 5:
            continue
        train[user] = items[:-2]
        positives[user] = {"validation": items[-2], "test": items[-1]}
        known[user] = set(items)
        popularity.update(items[:-2])
    return train, positives, known, titles, popularity


def cases_for(split, users, train, positives, known, titles, seed, n_candidates=20, retrieval="random"):
    rng = random.Random(seed)
    catalog = sorted(titles)
    if retrieval not in ("random","bm25"):
        raise ValueError("Amazon retrieval supports random or bm25 only")
    from .retrieval import TitleBM25
    retriever=TitleBM25(titles) if retrieval=="bm25" else None
    cases = []
    for user in users:
        gt = positives[user][split]
        if gt not in titles:
            continue
        if retriever:
            import hashlib
            user_seed=int.from_bytes(hashlib.sha256(
                f"{seed}:{user}:{split}".encode()).digest()[:8],"big")
            negatives=retriever.select(train[user],known[user],titles,user_seed,
                                        n_candidates-1)
        else:
            pool = [i for i in catalog if i not in known[user]]
            if len(pool) < n_candidates-1:
                continue
            negatives=rng.sample(pool,n_candidates-1)
        cand = negatives + [gt]
        rng.shuffle(cand)
        cases.append({"user":user, "history":train[user][-50:], "target":gt, "candidates":cand})
    return cases


def user_utility(rank, target, k):
    if rank is None or target not in rank[:k]:
        return 0.0, 0.0
    return 1.0, 1 / math.log2(rank.index(target)+2)


_CATALOG_CACHE = {}

def _catalog_groups(popularity, catalog):
    # Stable training popularity and catalog are frozen within a run. Group
    # membership does not depend on bootstrap resampling of users.
    key=(id(popularity),id(catalog))
    row=_CATALOG_CACHE.get(key)
    if row is not None and row[0] is popularity and row[1] is catalog:
        return row[2],row[3],row[4]
    items=sorted(catalog,key=lambda item:(popularity.get(item,0),item))
    long_tail=set(items[:math.ceil(0.8*len(items))])
    groups={item:min(4,index*5//len(items)) for index,item in enumerate(items)}
    if len(_CATALOG_CACHE)>8:
        _CATALOG_CACHE.clear()
    _CATALOG_CACHE[key]=(popularity,catalog,long_tail,groups,len(items))
    return long_tail,groups,len(items)

def summarize(rows, popularity, catalog, k):
    n = len(rows)
    if not n:
        raise ValueError("empty sample")
    valid = [r for r in rows if r["ranking"] is not None]
    long_tail,groups,catalog_size=_catalog_groups(popularity,catalog)
    hit = ndcg = 0.0
    head = long = total = 0
    group_positives = Counter()
    group_hits = Counter()
    rec_counts = Counter()
    for r in rows:
        h,d = user_utility(r["ranking"],r["target"],k)
        hit += h
        ndcg += d
        group = groups.get(r["target"])
        if group is not None:
            group_positives[group] += 1
            if h:
                group_hits[group] += 1
        if r["ranking"] is None:
            continue
        for item in r["ranking"][:k]:
            total += 1
            head += popularity.get(item,0)
            long += item in long_tail
            rec_counts[item] += 1
    tpr = [group_hits[g]/group_positives[g] if group_positives[g] else None for g in range(5)]
    good = [t for t in tpr if t is not None]
    reo = None
    if len(good)==5 and sum(good)>0:
        mean = sum(good)/len(good)
        reo = math.sqrt(sum((v-mean)**2 for v in good)/len(good)) / mean
    return {
        "users": n, "valid_rankings": len(valid), "invalid_rankings": n-len(valid),
        f"HR@{k}": hit/n, f"NDCG@{k}":ndcg/n,
        f"ARP@{k}": head/total if total else None,
        f"APLT@{k}":long/total if total else None,
        f"PopREO@{k}":reo, "popreo_group_positive_counts":[group_positives[g] for g in range(5)],
        "popreo_group_recall":tpr,
        f"coverage@{k}":len(rec_counts)/catalog_size,
        "bias_metric_denominator_valid_rankings":len(valid),
    }


def cpb(a,b):
    # Equation (23) diverges when empirical HR/NDCG is exactly one.
    # Small-sample boundaries are undefined, not a finite effect of 27.63.
    if a is None or b is None or a >= 1 or b >= 1:
        return None
    return -math.log(1-a) + math.log(1-b)
